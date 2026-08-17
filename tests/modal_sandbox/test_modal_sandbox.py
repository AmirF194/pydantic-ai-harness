"""Tests for the public Modal sandbox capability API."""

from __future__ import annotations

import inspect
from collections.abc import AsyncGenerator, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass, replace
from typing import Any, Protocol, TypeGuard, runtime_checkable

import pytest
from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.exceptions import ModelRetry, UserError
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.sandboxes import LocalSandbox, Sandbox, SandboxBackend, SandboxCommand, SandboxRef
from pydantic_ai.toolsets import AbstractToolset
from pydantic_ai.usage import RunUsage
from typing_extensions import Never

from pydantic_ai_harness.code_mode import CodeMode
from pydantic_ai_harness.modal_sandbox import (
    ModalSandbox,
    ModalSandboxBackend,
    ModalSandboxError,
    ModalSandboxTerminalError,
    ModalSandboxUnavailableError,
)

from .fake_modal import FakeModal, FileInfo


@runtime_checkable
class _ModalSandboxTools(Protocol):  # pragma: no cover - structural typing only
    async def run_command(
        self, ctx: RunContext[None], command: str, *, timeout_seconds: float | None = None
    ) -> str: ...

    async def read_file(
        self, ctx: RunContext[None], path: str, *, offset: int | None = None, limit: int | None = None
    ) -> str: ...

    async def write_file(self, ctx: RunContext[None], path: str, content: str) -> str: ...

    async def list_directory(self, ctx: RunContext[None], path: str = '.') -> str: ...


class _TimingOutSandbox:
    """A non-Modal backend whose commands always report a bare deadline kill."""

    provider = 'stub'
    sandbox_id = 'stub-1'

    async def run(
        self,
        command: SandboxCommand,
        *,
        shell: bool = False,
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> Never:
        raise TimeoutError('deadline reached')

    async def working_dir(self) -> str:
        return '/'  # pragma: no cover - the timeout fires before any path is resolved


@dataclass
class _Tools:
    """The capability's tools bound to a run context, the way an agent run calls them."""

    tools: _ModalSandboxTools
    ctx: RunContext[None]

    async def run_command(self, command: str, *, timeout_seconds: float | None = None) -> str:
        return await self.tools.run_command(self.ctx, command, timeout_seconds=timeout_seconds)

    async def read_file(self, path: str, *, offset: int | None = None, limit: int | None = None) -> str:
        return await self.tools.read_file(self.ctx, path, offset=offset, limit=limit)

    async def write_file(self, path: str, content: str) -> str:
        return await self.tools.write_file(self.ctx, path, content)

    async def list_directory(self, path: str = '.') -> str:
        return await self.tools.list_directory(self.ctx, path)


def _is_abstract_toolset(value: object) -> TypeGuard[AbstractToolset[None]]:
    return isinstance(value, AbstractToolset)


def _run_context() -> RunContext[None]:
    return RunContext[None](
        deps=None,
        model=TestModel(),
        usage=RunUsage(),
        prompt=None,
        messages=[],
        run_step=0,
    )


def _tool_surface(capability: ModalSandbox[None]) -> _ModalSandboxTools:
    toolset = capability.get_toolset()
    if not _is_abstract_toolset(toolset) or not isinstance(toolset, _ModalSandboxTools):  # pragma: no cover
        raise AssertionError('ModalSandbox must return a toolset with its public tools')
    return toolset


@asynccontextmanager
async def _tools(**settings: Any) -> AsyncGenerator[_Tools]:
    """Drive the capability's sandbox lifecycle the way a run does, then hand back its tools."""
    settings.setdefault('default_command_timeout', 30.0)
    capability = ModalSandbox[None](**settings)
    ctx = _run_context()
    ref = await capability.create_sandbox(ctx)

    async def resolve(reference: SandboxRef) -> SandboxBackend:
        backend = await capability.get_sandbox(ctx, reference)
        assert backend is not None
        return backend

    try:
        yield _Tools(_tool_surface(capability), replace(ctx, sandbox=Sandbox.from_ref(ref, resolve)))
    finally:
        await capability.destroy_sandbox(ctx, ref)


class TestRunCommand:
    async def test_labels_stdout(self, fake_modal: FakeModal) -> None:
        fake_modal.responder = lambda argv, timeout: ('hello\n', '', 0)
        async with _tools() as ts:
            result = await ts.run_command('echo hello')
        assert result == '[stdout]\nhello'
        # The protocol takes argv, so a shell command is wrapped explicitly.
        assert fake_modal.sandboxes[0].exec_calls[-1].argv == ['/bin/sh', '-c', 'echo hello']

    async def test_combines_stdout_stderr_and_exit_code(self, fake_modal: FakeModal) -> None:
        fake_modal.responder = lambda argv, timeout: ('out\n', 'err\n', 2)
        async with _tools() as ts:
            result = await ts.run_command('false')
        assert result == '[stdout]\nout\n[stderr]\nerr\n[exit code: 2]'

    async def test_no_output(self, fake_modal: FakeModal) -> None:
        fake_modal.responder = lambda argv, timeout: ('', '', 0)
        async with _tools() as ts:
            assert await ts.run_command('true') == '(no output)'

    async def test_per_call_timeout_passed(self, fake_modal: FakeModal) -> None:
        async with _tools() as ts:
            await ts.run_command('echo', timeout_seconds=12.0)
        assert fake_modal.sandboxes[0].exec_calls[-1].timeout == 12

    async def test_omitted_timeout_falls_back_to_default_never_unbounded(self, fake_modal: FakeModal) -> None:
        # Modal cannot kill a running command, so every command must carry a finite
        # deadline; an omitted timeout_seconds means default_command_timeout, not None.
        async with _tools() as ts:
            await ts.run_command('echo hi')
        assert fake_modal.sandboxes[0].exec_calls[-1].timeout == 30

    async def test_timeout_clamped_to_sandbox_timeout(self, fake_modal: FakeModal) -> None:
        # Modal cannot kill a running command, so a model-supplied timeout is capped. With no
        # explicit ceiling it falls back to the sandbox lifetime.
        async with _tools(sandbox_timeout=120) as ts:
            await ts.run_command('echo', timeout_seconds=9999)
        assert fake_modal.sandboxes[0].exec_calls[-1].timeout == 120

    async def test_max_command_timeout_overrides_ceiling(self, fake_modal: FakeModal) -> None:
        # An explicit ceiling lets an attached sandbox allow longer or shorter single
        # commands than the sandbox lifetime.
        async with _tools(sandbox_timeout=120, max_command_timeout=50) as ts:
            await ts.run_command('echo', timeout_seconds=9999)
        assert fake_modal.sandboxes[0].exec_calls[-1].timeout == 50

    async def test_fractional_timeout_rounds_up(self, fake_modal: FakeModal) -> None:
        async with _tools() as ts:
            await ts.run_command('echo', timeout_seconds=0.5)
        # A sub-second timeout must not floor to 0, which Modal treats as "no timeout".
        assert fake_modal.sandboxes[0].exec_calls[-1].timeout == 1

    @pytest.mark.parametrize('bad_timeout', [0, -5.0])
    async def test_non_positive_timeout_rejected(self, fake_modal: FakeModal, bad_timeout: float) -> None:
        # A 0 or negative request is a model mistake; reject it rather than let the backend
        # floor it to a surprise 1-second deadline.
        async with _tools() as ts:
            with pytest.raises(ModelRetry, match='timeout_seconds must be greater than 0'):
                await ts.run_command('echo', timeout_seconds=bad_timeout)
        assert fake_modal.sandboxes[0].exec_calls == []

    async def test_output_truncated(self, fake_modal: FakeModal) -> None:
        fake_modal.responder = lambda argv, timeout: ('x' * 1000, '', 0)
        async with _tools(max_output_bytes=100) as ts:
            result = await ts.run_command('big')
        assert 'output truncated' in result

    async def test_output_is_bounded_before_it_reaches_the_model(self, fake_modal: FakeModal) -> None:
        # The sandbox protocol delivers a command's whole output, so the cap is a
        # presentation one: the tail survives, where the exit status lives.
        fake_modal.responder = lambda argv, timeout: ('A' * 500 + 'END', '', 0)
        async with _tools(max_output_bytes=50) as ts:
            result = await ts.run_command('flood')
        assert 'output truncated to the last 50B' in result
        assert result.endswith('END')

    async def test_output_line_cap_keeps_last_lines(self, fake_modal: FakeModal) -> None:
        fake_modal.responder = lambda argv, timeout: ('first\nsecond\nthird', '', 0)
        async with _tools(max_output_lines=1) as ts:
            result = await ts.run_command('many-lines')
        assert 'truncated to the last 1 lines' in result
        assert result.endswith('third')

    async def test_multibyte_output_truncation_drops_partial_character(self, fake_modal: FakeModal) -> None:
        fake_modal.responder = lambda argv, timeout: ('é' * 20, '', 0)
        async with _tools(max_output_bytes=5) as ts:
            result = await ts.run_command('unicode')
        assert 'output truncated to the last 5B' in result

    async def test_stream_labels_survive_truncation(self, fake_modal: FakeModal) -> None:
        # Each stream is truncated separately with its label attached afterwards, so a
        # flood on one stream cannot erase the labels or crowd out the other stream.
        fake_modal.responder = lambda argv, timeout: ('ok\n', 'E' * 500, 1)
        async with _tools(max_output_bytes=50) as ts:
            result = await ts.run_command('noisy')
        assert result.startswith('[stdout]\nok\n[stderr]\n')
        assert 'output truncated to the last 50B' in result
        assert result.endswith('[exit code: 1]')

    async def test_timeout_keeps_the_output_that_preceded_it(self, fake_modal: FakeModal) -> None:
        # Modal's -1 deadline sentinel becomes a legible note, and the output produced before
        # the kill -- which the protocol's raise-on-timeout cannot return -- still reaches the
        # model.
        fake_modal.responder = lambda argv, timeout: ('partial\n', '', -1)
        async with _tools() as ts:
            result = await ts.run_command('sleep 99', timeout_seconds=5)
        assert result == '[stdout]\npartial\n[timed out after 5s]'

    async def test_a_backend_without_partial_output_still_reports_the_timeout(self) -> None:
        # The tools work against whatever sandbox the run has, so a backend that reports a
        # deadline kill with no output alongside it still gets a legible note.
        ctx = replace(_run_context(), sandbox=Sandbox(_TimingOutSandbox()))
        tools = _tool_surface(ModalSandbox[None]())
        assert await tools.run_command(ctx, 'sleep 99', timeout_seconds=5) == '[timed out after 5s]'

    async def test_exec_failure_raises_model_retry(self, fake_modal: FakeModal) -> None:
        async with _tools() as ts:
            fake_modal.exec_error = fake_modal.error_type('transient blip')
            with pytest.raises(ModelRetry, match='Command could not run in the sandbox: transient blip'):
                await ts.run_command('echo hi')

    @pytest.mark.parametrize(
        ('exc_property', 'match'),
        [
            ('unavailable_type', 'no longer running'),
            ('sandbox_terminated_type', 'no longer running'),
            ('sandbox_timeout_type', 'no longer running'),
            ('auth_type', 'Modal rejected the credentials'),
        ],
    )
    async def test_terminal_failure_ends_the_run(self, fake_modal: FakeModal, exc_property: str, match: str) -> None:
        # A terminal failure must not become a ModelRetry (which would loop the model
        # against a sandbox that is never coming back, or against dead credentials). It
        # propagates. All three Modal spellings of "the sandbox is gone" classify the same
        # way; sandbox expiry (SandboxTimeoutError) is the one an owned run outliving its
        # lifetime actually produces, and rejected credentials are the non-sandbox case.
        exc_type: type[Exception] = getattr(fake_modal, exc_property)
        async with _tools() as ts:
            fake_modal.exec_error = exc_type('terminal failure')
            with pytest.raises(ModalSandboxTerminalError, match=match):
                await ts.run_command('echo hi')

    async def test_dead_sandbox_conflict_on_first_exec_is_terminal(self, fake_modal: FakeModal) -> None:
        # A first exec on a dead sandbox surfaces as Modal's ambiguous ConflictError; the
        # poll disambiguates so the model is not sent into a retry loop against a corpse.
        async with _tools() as ts:
            fake_modal.exec_error = fake_modal.conflict_type('Sandbox already finished')
            fake_modal.sandboxes[0].poll_result = 0
            with pytest.raises(ModalSandboxUnavailableError, match='no longer running'):
                await ts.run_command('echo hi')

    async def test_transient_conflict_on_exec_is_retried(self, fake_modal: FakeModal) -> None:
        async with _tools() as ts:
            fake_modal.exec_error = fake_modal.conflict_type('aborted')
            with pytest.raises(ModelRetry, match='aborted'):
                await ts.run_command('echo hi')

    async def test_non_modal_stream_error_is_retried(self, fake_modal: FakeModal) -> None:
        # Raw transport failures (grpclib errors, a ValueError on an empty message) are not
        # modal.exception.Error; they must still become a retryable sandbox error rather
        # than abort the agent run.
        fake_modal.stdout_error = ValueError('Received empty message')
        async with _tools() as ts:
            with pytest.raises(ModelRetry, match='ValueError: Received empty message'):
                await ts.run_command('echo hi')

    async def test_tool_without_a_sandbox_explains_how_to_attach(self) -> None:
        # Outside a run there is no sandbox, and the framework default says so rather than
        # reaching for the host.
        tools = _tool_surface(ModalSandbox[None]())
        with pytest.raises(UserError, match='No sandbox is attached'):
            await tools.run_command(_run_context(), 'echo hi')

    async def test_the_tools_run_against_any_sandbox_backend(self) -> None:
        # The toolset reads `ctx.sandbox`, so a run that supplied its own sandbox gets these
        # tools over that one instead of a Modal container.
        import sniffio

        if sniffio.current_async_library() != 'asyncio':
            pytest.skip('LocalSandbox spawns asyncio subprocesses')
        async with LocalSandbox() as local:
            ctx = replace(_run_context(), sandbox=Sandbox(local))
            tools = _tool_surface(ModalSandbox[None]())
            assert await tools.run_command(ctx, 'echo hi') == '[stdout]\nhi'


class TestReadFile:
    async def test_returns_contents(self, fake_modal: FakeModal) -> None:
        async with _tools() as ts:
            # The single trailing newline is dropped with its phantom empty line, so the read
            # returns the real line content rather than a body counted as two lines.
            fake_modal.sandboxes[0].files['/etc/hosts'] = b'file body\n'
            assert await ts.read_file('/etc/hosts') == 'file body'

    async def test_at_size_limit_is_not_truncated(self, fake_modal: FakeModal) -> None:
        async with _tools(max_output_bytes=100) as ts:
            fake_modal.sandboxes[0].files['/f'] = b'x' * 100
            assert await ts.read_file('/f') == 'x' * 100

    async def test_over_size_limit_pages_from_head(self, fake_modal: FakeModal) -> None:
        async with _tools(max_output_bytes=20) as ts:
            fake_modal.sandboxes[0].files['/big'] = b'\n'.join(b'line%02d' % i for i in range(10))
            result = await ts.read_file('/big')
        # File reads keep the head and tell the model how to page the rest.
        assert result.startswith('line00')
        assert 'Use offset=' in result

    async def test_offset_and_limit(self, fake_modal: FakeModal) -> None:
        async with _tools() as ts:
            fake_modal.sandboxes[0].files['/f'] = b'a\nb\nc\nd\ne'
            result = await ts.read_file('/f', offset=2, limit=2)
        assert result.startswith('b\nc')
        assert 'more lines in file. Use offset=4 to continue.' in result

    async def test_binary_file_raises_model_retry(self, fake_modal: FakeModal) -> None:
        async with _tools() as ts:
            fake_modal.sandboxes[0].files['/img.png'] = b'\xff\xfe\x00\x01'
            with pytest.raises(ModelRetry, match='not valid UTF-8'):
                await ts.read_file('/img.png')

    async def test_missing_file_raises_model_retry(self, fake_modal: FakeModal) -> None:
        # The backend translates Modal's missing-file exception into the builtin
        # `FileNotFoundError` the protocol promises; the tool turns that into a retry.
        async with _tools() as ts:
            with pytest.raises(ModelRetry, match="Could not read '/nope': No such file"):
                await ts.read_file('/nope')

    async def test_filesystem_error_raises_model_retry(self, fake_modal: FakeModal) -> None:
        async with _tools() as ts:
            fake_modal.sandboxes[0].fs_error = fake_modal.filesystem_error_type('Permission denied: /root/x')
            with pytest.raises(ModelRetry, match="Could not read '/root/x': Permission denied"):
                await ts.read_file('/root/x')

    async def test_refuses_file_over_read_limit(self, fake_modal: FakeModal) -> None:
        async with _tools(max_read_bytes=1000) as ts:
            # Report a large size without allocating the bytes; the guard fires before the read.
            fake_modal.sandboxes[0].stat_sizes['/big.log'] = 5000
            with pytest.raises(ModelRetry, match='over the 1000B read limit'):
                await ts.read_file('/big.log')

    async def test_read_limit_formats_megabytes(self, fake_modal: FakeModal) -> None:
        async with _tools(max_read_bytes=1000) as ts:
            fake_modal.sandboxes[0].stat_sizes['/big.log'] = 3 * 1024 * 1024
            with pytest.raises(ModelRetry, match='File is 3.0MB'):
                await ts.read_file('/big.log')

    async def test_line_cap_is_configurable(self, fake_modal: FakeModal) -> None:
        # Bytes are well under budget, so only the line cap can fire: proves it is plumbed
        # through, not silently fixed at the helper default.
        async with _tools(max_output_lines=3, max_output_bytes=50_000) as ts:
            fake_modal.sandboxes[0].files['/many'] = b'\n'.join(b'L%d' % i for i in range(20))
            result = await ts.read_file('/many')
        assert 'Showing lines 1-3 of 20' in result
        assert 'Use offset=4 to continue.' in result

    async def test_refuses_file_that_grew_after_stat(self, fake_modal: FakeModal) -> None:
        async with _tools(max_read_bytes=1000) as ts:
            # stat reports under the limit, but the read returns more (the file grew between
            # the two round-trips). The post-read guard refuses before the decode/window.
            fake_modal.sandboxes[0].stat_sizes['/grows'] = 10
            fake_modal.sandboxes[0].files['/grows'] = b'x' * 5000
            with pytest.raises(ModelRetry, match='over the 1000B read limit'):
                await ts.read_file('/grows')

    async def test_a_directory_has_no_size_to_guard(self, fake_modal: FakeModal) -> None:
        # The protocol reports no size for a directory, so the guard cannot fire and the read
        # itself is what refuses.
        async with _tools() as ts:
            fake_modal.sandboxes[0].directories.add('/pkg')
            with pytest.raises(ModelRetry, match="Could not read '/pkg'"):
                await ts.read_file('/pkg')

    @pytest.mark.parametrize(
        ('kwargs', 'message'),
        [
            ({'offset': 0}, 'offset must be >= 1'),
            ({'limit': 0}, 'limit must be >= 1'),
            ({'offset': 99}, 'beyond end of file'),
        ],
    )
    async def test_invalid_window_is_a_model_retry(
        self, fake_modal: FakeModal, kwargs: dict[str, int], message: str
    ) -> None:
        async with _tools() as ts:
            fake_modal.sandboxes[0].files['/f'] = b'one\ntwo'
            with pytest.raises(ModelRetry, match=message):
                await ts.read_file('/f', **kwargs)

    async def test_oversized_first_line_is_omitted(self, fake_modal: FakeModal) -> None:
        async with _tools(max_output_bytes=10) as ts:
            fake_modal.sandboxes[0].files['/f'] = b'x' * 200
            result = await ts.read_file('/f')
        assert result == '[Line 1 is 200B, exceeds the 10B limit and was omitted.]'

    async def test_oversized_first_line_points_to_next_line(self, fake_modal: FakeModal) -> None:
        async with _tools(max_output_bytes=10) as ts:
            fake_modal.sandboxes[0].files['/f'] = b'x' * 200 + b'\nnext'
            result = await ts.read_file('/f')
        assert 'Use offset=2 to continue.' in result

    async def test_byte_cap_returns_continuation_offset(self, fake_modal: FakeModal) -> None:
        async with _tools(max_output_bytes=12) as ts:
            fake_modal.sandboxes[0].files['/f'] = b'line00\nline01\nline02'
            result = await ts.read_file('/f')
        assert '(12B limit). Use offset=' in result

    async def test_terminal_error_ends_the_run(self, fake_modal: FakeModal) -> None:
        # A missing sandbox during a read is terminal, not a retryable "could not read".
        async with _tools() as ts:
            fake_modal.sandboxes[0].fs_error = fake_modal.unavailable_type('sandbox not found')
            fake_modal.sandboxes[0].poll_result = 0
            with pytest.raises(ModalSandboxUnavailableError):
                await ts.read_file('/x')

    async def test_wrapped_not_found_is_recoverable_while_sandbox_runs(self, fake_modal: FakeModal) -> None:
        async with _tools() as ts:
            fake_modal.sandboxes[0].fs_error = fake_modal.unavailable_type('filesystem request failed')
            with pytest.raises(ModelRetry, match='filesystem request failed'):
                await ts.read_file('/x')

    async def test_wrapped_auth_error_is_terminal(self, fake_modal: FakeModal) -> None:
        async with _tools() as ts:
            fake_modal.sandboxes[0].fs_error = fake_modal.filesystem_error_type('filesystem request failed')
            fake_modal.sandboxes[0].poll_error = fake_modal.auth_type('unauthenticated')
            with pytest.raises(ModalSandboxError, match='Modal rejected the credentials'):
                await ts.read_file('/x')


class TestWriteFile:
    async def test_writes_file_content(self, fake_modal: FakeModal) -> None:
        async with _tools() as ts:
            result = await ts.write_file('/tmp/pkg/a.py', 'print(1)\n')
        assert result == "Wrote 9 bytes to '/tmp/pkg/a.py'."
        assert fake_modal.sandboxes[0].files['/tmp/pkg/a.py'] == b'print(1)\n'

    async def test_unencodable_content_raises_model_retry(self, fake_modal: FakeModal) -> None:
        # A provider that pre-parses tool args to a dict can hand the tool an unpaired
        # surrogate; that is a model mistake, so it must retry, not abort the run.
        async with _tools() as ts:
            with pytest.raises(ModelRetry, match='cannot be encoded as UTF-8'):
                await ts.write_file('/tmp/x', '\ud800')

    async def test_error_raises_model_retry(self, fake_modal: FakeModal) -> None:
        async with _tools() as ts:
            fake_modal.sandboxes[0].fs_error = fake_modal.filesystem_error_type('Permission denied: /root/x')
            with pytest.raises(ModelRetry, match="Could not write '/root/x': Permission denied"):
                await ts.write_file('/root/x', 'data')

    async def test_terminal_error_ends_the_run(self, fake_modal: FakeModal) -> None:
        async with _tools() as ts:
            fake_modal.sandboxes[0].fs_error = fake_modal.unavailable_type('sandbox not found')
            fake_modal.sandboxes[0].poll_result = 0
            with pytest.raises(ModalSandboxUnavailableError):
                await ts.write_file('/x', 'data')


class TestListDirectory:
    async def test_lists_with_trailing_slash_on_dirs(self, fake_modal: FakeModal) -> None:
        async with _tools() as ts:
            fake_modal.sandboxes[0].listing = [FileInfo('b', True), FileInfo('a', False)]
            assert await ts.list_directory('/tmp') == 'a\nb/'
        assert fake_modal.sandboxes[0].list_paths == ['/tmp']

    async def test_directories_keep_plain_name_order(self, fake_modal: FakeModal) -> None:
        # Sorting happens before the '/' suffix is added; '/' sorts after '-' and '.', so
        # sorting the suffixed strings would misplace directories relative to `ls` order.
        async with _tools() as ts:
            fake_modal.sandboxes[0].listing = [
                FileInfo('a.txt', False),
                FileInfo('a', True),
                FileInfo('a-b', False),
            ]
            assert await ts.list_directory('/tmp') == 'a/\na-b\na.txt'

    async def test_oversized_entry_name_is_marked_not_blank(self, fake_modal: FakeModal) -> None:
        # A single entry name wider than the byte cap cannot be shown head-first; the model
        # gets an explicit marker instead of an empty body under a "truncated" note.
        async with _tools(max_output_bytes=10) as ts:
            fake_modal.sandboxes[0].listing = [FileInfo('a-very-long-entry-name', False)]
            result = await ts.list_directory('/tmp')
        assert result == '[... first line exceeds the 10B limit, output omitted ...]'

    async def test_head_truncation_appends_marker_after_the_listing(self, fake_modal: FakeModal) -> None:
        # Directory listings truncate head-first: the kept entries come first and the
        # marker says "first", appended after the body (the opposite arrangement to
        # command output).
        async with _tools(max_output_lines=2) as ts:
            fake_modal.sandboxes[0].listing = [
                FileInfo('a', False),
                FileInfo('b', False),
                FileInfo('c', False),
            ]
            result = await ts.list_directory('/tmp')
        assert result == 'a\nb\n[... output truncated to the first 2 lines ...]'

    async def test_default_path_resolves_to_cwd(self, fake_modal: FakeModal) -> None:
        fake_modal.responder = lambda argv, timeout: ('/work\n', '', 0)
        async with _tools() as ts:
            assert await ts.list_directory() == '(empty)'
        # The default '.' is resolved against the sandbox working directory, discovered
        # with a single `pwd`.
        assert fake_modal.sandboxes[0].list_paths == ['/work']
        assert [call.argv for call in fake_modal.sandboxes[0].exec_calls] == [['pwd']]

    async def test_relative_path_resolved_against_cwd(self, fake_modal: FakeModal) -> None:
        fake_modal.responder = lambda argv, timeout: ('/work\n', '', 0)
        async with _tools() as ts:
            fake_modal.sandboxes[0].listing = [FileInfo('a.py', False)]
            assert await ts.list_directory('src') == 'a.py'
        assert fake_modal.sandboxes[0].list_paths == ['/work/src']

    async def test_parent_segments_are_normalized_away(self, fake_modal: FakeModal) -> None:
        # The facade normalizes a resolved path textually. That is a spelling convenience,
        # not confinement: isolation is the sandbox's job, not the path helper's.
        fake_modal.responder = lambda argv, timeout: ('/work\n', '', 0)
        async with _tools() as ts:
            await ts.list_directory('link/../secret')
        assert fake_modal.sandboxes[0].list_paths == ['/work/secret']

    async def test_error_raises_model_retry(self, fake_modal: FakeModal) -> None:
        async with _tools() as ts:
            fake_modal.sandboxes[0].fs_error = fake_modal.filesystem_error_type('Not a directory: /etc/hosts')
            with pytest.raises(ModelRetry, match="Could not list '/etc/hosts': Not a directory"):
                await ts.list_directory('/etc/hosts')

    async def test_terminal_error_ends_the_run(self, fake_modal: FakeModal) -> None:
        async with _tools() as ts:
            fake_modal.sandboxes[0].fs_error = fake_modal.unavailable_type('sandbox not found')
            fake_modal.sandboxes[0].poll_result = 0
            with pytest.raises(ModalSandboxUnavailableError):
                await ts.list_directory('/x')


class TestSandboxLifecycle:
    async def test_create_provisions_from_config_and_destroy_terminates(self, fake_modal: FakeModal) -> None:
        capability = ModalSandbox[None](image='ubuntu:22.04', app_name='my-app', sandbox_timeout=120, workdir='/work')
        ctx = _run_context()
        ref = await capability.create_sandbox(ctx)
        assert ref == SandboxRef(provider='modal', sandbox_id='sb-owned')
        assert fake_modal.app_lookups[-1]['name'] == 'my-app'
        assert fake_modal.image_tags[-1] == 'ubuntu:22.04'
        assert fake_modal.create_kwargs[-1]['timeout'] == 120
        assert fake_modal.create_kwargs[-1]['workdir'] == '/work'

        await capability.destroy_sandbox(ctx, ref)
        assert fake_modal.sandboxes[0].terminated is True
        assert fake_modal.sandboxes[0].detached is True

    async def test_env_passed_to_owned_sandbox(self, fake_modal: FakeModal) -> None:
        await ModalSandbox[None](env={'FOO': 'bar'}).create_sandbox(_run_context())
        assert fake_modal.create_kwargs[-1]['env'] == {'FOO': 'bar'}

    async def test_get_sandbox_reuses_the_handle_this_run_created(self, fake_modal: FakeModal) -> None:
        # `get_sandbox` receives only a serializable ref, so without the run's own handle
        # every run would look its sandbox up again just to use it.
        capability = ModalSandbox[None]()
        ctx = _run_context()
        ref = await capability.create_sandbox(ctx)
        backend = await capability.get_sandbox(ctx, ref)
        assert isinstance(backend, ModalSandboxBackend)
        assert backend.sandbox is fake_modal.sandboxes[0]
        assert fake_modal.attach_ids == []

    async def test_get_sandbox_connects_when_the_handle_is_elsewhere(self, fake_modal: FakeModal) -> None:
        # A run that resumed in another process holds no handle, so the ref is connected to.
        capability = ModalSandbox[None]()
        backend = await capability.get_sandbox(_run_context(), SandboxRef(provider='modal', sandbox_id='sb-other'))
        assert isinstance(backend, ModalSandboxBackend)
        assert fake_modal.attach_ids == ['sb-other']

    async def test_other_providers_refs_are_declined(self, fake_modal: FakeModal) -> None:
        capability = ModalSandbox[None]()
        assert await capability.get_sandbox(_run_context(), SandboxRef(provider='e2b', sandbox_id='x')) is None

    async def test_destroy_without_a_handle_is_a_no_op(self, fake_modal: FakeModal) -> None:
        # Modal's `sandbox_timeout` is the backstop for a run whose teardown happens where
        # the handle does not live.
        capability = ModalSandbox[None]()
        await capability.destroy_sandbox(_run_context(), SandboxRef(provider='modal', sandbox_id='sb-gone'))
        assert fake_modal.sandboxes == []

    async def test_attached_sandbox_is_used_but_not_created_or_terminated(self, fake_modal: FakeModal) -> None:
        capability = ModalSandbox[None](sandbox_id='sb-keep')
        ctx = _run_context()
        ref = await capability.create_sandbox(ctx)
        assert ref == SandboxRef(provider='modal', sandbox_id='sb-keep')
        # Naming an existing sandbox provisions nothing.
        assert fake_modal.sandboxes == []

        await capability.get_sandbox(ctx, ref)
        await capability.destroy_sandbox(ctx, ref)
        assert fake_modal.attach_ids == ['sb-keep']
        assert fake_modal.sandboxes[0].terminated is False
        assert fake_modal.sandboxes[0].detached is True

    async def test_attaching_to_a_dead_sandbox_fails(self, fake_modal: FakeModal) -> None:
        # Connecting must fail rather than silently provision a replacement.
        fake_modal.attach_poll_result = 0
        capability = ModalSandbox[None](sandbox_id='sb-gone')
        ctx = _run_context()
        ref = await capability.create_sandbox(ctx)
        with pytest.raises(ModalSandboxUnavailableError):
            await capability.get_sandbox(ctx, ref)

    async def test_two_runs_do_not_share_one_handle(self, fake_modal: FakeModal) -> None:
        # Runs sharing an attached sandbox each connect for themselves, so one run's
        # teardown cannot detach the other's connection.
        capability = ModalSandbox[None](sandbox_id='sb-keep')
        first = replace(_run_context(), run_id='run-1')
        second = replace(_run_context(), run_id='run-2')
        ref = await capability.create_sandbox(first)
        first_backend = await capability.get_sandbox(first, ref)
        second_backend = await capability.get_sandbox(second, ref)
        assert first_backend is not second_backend

        await capability.destroy_sandbox(first, ref)
        assert fake_modal.sandboxes[0].detached is True
        assert fake_modal.sandboxes[1].detached is False


class TestCapability:
    def test_defaults(self) -> None:
        cap = ModalSandbox()
        assert cap.image == 'python:3.12-slim'
        assert cap.sandbox_id is None
        assert cap.app_name == 'pydantic-ai-harness'
        assert cap.sandbox_timeout == 300
        assert cap.default_command_timeout == 60.0

    def test_get_toolset(self) -> None:
        assert isinstance(ModalSandbox().get_toolset(), AbstractToolset)

    def test_serialization_name(self) -> None:
        assert ModalSandbox.get_serialization_name() == 'ModalSandbox'

    def test_configuration_is_keyword_only(self) -> None:
        parameters = list(inspect.signature(ModalSandbox).parameters.values())
        assert parameters
        assert all(parameter.kind is inspect.Parameter.KEYWORD_ONLY for parameter in parameters)

    @pytest.mark.parametrize(
        ('name', 'value'),
        [
            ('sandbox_timeout', 0),
            ('max_output_bytes', -1),
            ('max_output_lines', 0),
            ('max_read_bytes', -1),
            ('default_command_timeout', 0),
            ('default_command_timeout', float('nan')),
            ('default_command_timeout', float('inf')),
            ('max_command_timeout', 0),
            # The agent-spec path does not type-check dataclass fields, so a bad YAML value
            # must fail at construction, not deep in the agent build.
            ('instructions', 123),
        ],
    )
    def test_rejects_invalid_limits(self, name: str, value: object) -> None:
        with pytest.raises(ValueError, match=name):
            ModalSandbox(**{name: value})  # type: ignore[arg-type]

    def test_attach_with_only_defaults_is_allowed(self) -> None:
        cap = ModalSandbox(sandbox_id='sb-keep')
        assert cap.sandbox_id == 'sb-keep'

    @pytest.mark.parametrize(
        ('kwargs', 'expected'),
        [
            ({'image': 'ubuntu:22.04'}, 'image'),
            ({'app_name': 'other'}, 'app_name'),
            ({'create_app_if_missing': False}, 'create_app_if_missing'),
            ({'sandbox_timeout': 600}, 'sandbox_timeout'),
            ({'workdir': '/work'}, 'workdir'),
            ({'env': {'A': 'b'}}, 'env'),
        ],
    )
    def test_attach_rejects_owned_only_settings(self, kwargs: dict[str, object], expected: str) -> None:
        with pytest.raises(ValueError, match=f'{expected} only apply when creating a sandbox'):
            ModalSandbox(sandbox_id='sb-keep', **kwargs)  # type: ignore[arg-type]

    def test_attach_error_lists_every_conflicting_setting(self) -> None:
        with pytest.raises(ValueError, match='image, sandbox_timeout only apply'):
            ModalSandbox(sandbox_id='sb-keep', image='ubuntu:22.04', sandbox_timeout=600)

    def test_attach_rejecting_sandbox_timeout_points_at_max_command_timeout(self) -> None:
        # The reuse-mode redirect: a rejected sandbox_timeout names the setting that works.
        with pytest.raises(ValueError, match='set `max_command_timeout`'):
            ModalSandbox(sandbox_id='sb-keep', sandbox_timeout=600)

    def test_attach_rejecting_other_settings_omits_the_ceiling_hint(self) -> None:
        with pytest.raises(ValueError, match='workdir only apply') as exc:
            ModalSandbox(sandbox_id='sb-keep', workdir='/work')
        assert 'max_command_timeout' not in str(exc.value)

    def test_instructions_enabled_by_default(self) -> None:
        instructions = ModalSandbox().get_instructions()
        assert instructions is not None
        assert 'Modal sandbox' in instructions
        assert 'run_command' in instructions

    def test_instructions_can_be_disabled(self) -> None:
        assert ModalSandbox(instructions='').get_instructions() is None

    def test_owned_instructions_say_reset_between_runs(self) -> None:
        instructions = ModalSandbox().get_instructions()
        assert instructions is not None
        assert 'reset between' in instructions

    def test_attached_instructions_say_persists(self) -> None:
        instructions = ModalSandbox(sandbox_id='sb-keep').get_instructions()
        assert instructions is not None
        assert 'persists across runs' in instructions
        assert 'reset between' not in instructions

    def test_instructions_can_be_overridden(self) -> None:
        assert ModalSandbox(instructions='Use the modal_run_command tool.').get_instructions() == (
            'Use the modal_run_command tool.'
        )

    def test_instructions_state_the_real_timeouts(self) -> None:
        # The default text carries the applied deadline and ceiling, not placeholders, so
        # the model learns the numbers that `run_command` will actually enforce.
        instructions = ModalSandbox(default_command_timeout=45.5, sandbox_timeout=120).get_instructions()
        assert instructions is not None
        assert 'times out after 46s' in instructions  # ceil(45.5), the deadline Modal gets
        assert 'up to 120s' in instructions

    def test_instructions_ceiling_uses_max_command_timeout(self) -> None:
        instructions = ModalSandbox(sandbox_id='sb-keep', max_command_timeout=900).get_instructions()
        assert instructions is not None
        assert 'up to 900s' in instructions

    def test_instructions_clamp_default_to_the_ceiling(self) -> None:
        # A default above the enforceable ceiling must not be advertised: the model would
        # be told a deadline that enforcement then cuts short.
        instructions = ModalSandbox(default_command_timeout=400.0).get_instructions()
        assert instructions is not None
        assert 'times out after 300s' in instructions
        assert 'up to 300s' in instructions

    def test_owned_rejects_ceiling_above_sandbox_timeout(self) -> None:
        # In owned mode a command cannot outlive the sandbox, so a higher ceiling is a
        # dead value; in attach mode it is the documented escape hatch.
        with pytest.raises(ValueError, match='cannot exceed sandbox_timeout'):
            ModalSandbox(sandbox_timeout=300, max_command_timeout=600)

    def test_attach_allows_ceiling_above_default_sandbox_timeout(self) -> None:
        assert ModalSandbox(sandbox_id='sb-keep', max_command_timeout=600).max_command_timeout == 600

    def test_exported_from_capability_submodule(self) -> None:
        import pydantic_ai_harness
        import pydantic_ai_harness.modal_sandbox as modal_sandbox
        from pydantic_ai_harness.modal_sandbox import ModalSandbox as Exported

        assert Exported is ModalSandbox
        assert 'ModalSandboxToolset' not in modal_sandbox.__all__
        assert 'ModalSandboxBackend' in modal_sandbox.__all__
        assert pydantic_ai_harness.ModalSandbox is ModalSandbox

    @pytest.mark.anyio(backends=['asyncio'])
    async def test_agent_integration(self, fake_modal: FakeModal) -> None:
        import sniffio

        if sniffio.current_async_library() != 'asyncio':  # pragma: no cover
            pytest.skip('Agent.run() requires asyncio')
        model = TestModel(custom_output_text='done', call_tools=[])
        agent: Agent[None, str] = Agent(model, capabilities=[ModalSandbox()])
        result = await agent.run('set up the project')
        assert result.output == 'done'
        assert fake_modal.sandboxes[0].terminated is True

    @pytest.mark.anyio(backends=['asyncio'])
    async def test_agent_can_call_run_command(self, fake_modal: FakeModal) -> None:
        import sniffio

        if sniffio.current_async_library() != 'asyncio':  # pragma: no cover
            pytest.skip('Agent.run() requires asyncio')
        fake_modal.responder = lambda argv, timeout: ('hello\n', '', 0)

        def call_then_finish(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            if not any(isinstance(part, ToolReturnPart) for message in messages for part in message.parts):
                return ModelResponse(
                    parts=[ToolCallPart('run_command', {'command': 'echo hello'}, tool_call_id='run-1')]
                )
            return ModelResponse(parts=[TextPart('done')])

        agent: Agent[None, str] = Agent(FunctionModel(call_then_finish), capabilities=[ModalSandbox()])
        result = await agent.run('run a command')

        assert result.output == 'done'
        tool_returns = [
            part.content
            for message in result.all_messages()
            for part in message.parts
            if isinstance(part, ToolReturnPart) and part.tool_name == 'run_command'
        ]
        assert tool_returns == ['[stdout]\nhello']
        assert fake_modal.sandboxes[0].terminated is True

    @pytest.mark.anyio(backends=['asyncio'])
    async def test_agent_context_does_not_create_an_unused_base_sandbox(self, fake_modal: FakeModal) -> None:
        import sniffio

        if sniffio.current_async_library() != 'asyncio':  # pragma: no cover
            pytest.skip('Agent.run() requires asyncio')
        model = TestModel(custom_output_text='done', call_tools=[])
        agent: Agent[None, str] = Agent(model, capabilities=[ModalSandbox()])

        async with agent:
            assert fake_modal.sandboxes == []
            assert (await agent.run('first')).output == 'done'
            assert len(fake_modal.sandboxes) == 1
            assert fake_modal.sandboxes[0].terminated is True
            assert (await agent.run('second')).output == 'done'

        assert len(fake_modal.sandboxes) == 2
        assert all(sandbox.terminated for sandbox in fake_modal.sandboxes)

    @pytest.mark.anyio(backends=['asyncio'])
    async def test_agent_run_failing_terminally_still_tears_down(self, fake_modal: FakeModal) -> None:
        import sniffio

        if sniffio.current_async_library() != 'asyncio':  # pragma: no cover
            pytest.skip('Agent.run() requires asyncio')

        # A run that dies on a terminal sandbox error must still request termination on
        # its way out; the error exit path gets the same teardown as a clean one.
        def gone(argv: list[str], timeout: int | None) -> tuple[str, str, int]:
            raise fake_modal.sandbox_terminated_type('sandbox terminated')

        fake_modal.responder = gone

        def call_tool(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return ModelResponse(parts=[ToolCallPart('run_command', {'command': 'echo hi'}, tool_call_id='run-1')])

        agent: Agent[None, str] = Agent(FunctionModel(call_tool), capabilities=[ModalSandbox()])
        with pytest.raises(ModalSandboxUnavailableError):
            await agent.run('run a command')
        assert fake_modal.sandboxes[0].terminated is True
        assert fake_modal.sandboxes[0].detached is True


async def _tools_offered_to_model(*, modal_first: bool) -> dict[str, str | None]:
    """Run an agent with ModalSandbox and CodeMode and return the tools the model was offered."""
    offered: dict[str, str | None] = {}

    def capture(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        offered.update({tool.name: tool.description for tool in info.function_tools})
        return ModelResponse(parts=[TextPart('done')])

    sandbox = ModalSandbox[object]()
    code_mode = CodeMode[object]()
    capabilities: list[AbstractCapability[object]] = [sandbox, code_mode] if modal_first else [code_mode, sandbox]
    agent: Agent[None, str] = Agent(FunctionModel(capture), capabilities=capabilities)
    await agent.run('go')
    return offered


class TestCodeModeInterop:
    """`run_command` takes a command line, so CodeMode leaves it native.

    Folding it into `run_code` would make the model write a Monty script whose argument is a
    shell script quoted as a Python string, running the outer script on the host and the inner
    one in the microVM. The file tools carry no command line, so they stay sandboxed like any
    other tool.
    """

    @pytest.mark.anyio(backends=['asyncio'])
    @pytest.mark.parametrize('modal_first', [True, False], ids=['modal-first', 'code-mode-first'])
    async def test_run_command_stays_native(self, fake_modal: FakeModal, modal_first: bool) -> None:
        import sniffio

        if sniffio.current_async_library() != 'asyncio':  # pragma: no cover
            pytest.skip('Agent.run() requires asyncio')
        tools = await _tools_offered_to_model(modal_first=modal_first)

        assert 'run_command' in tools
        run_code_description = tools['run_code']
        assert run_code_description is not None
        assert 'async def run_command' not in run_code_description

    @pytest.mark.anyio(backends=['asyncio'])
    @pytest.mark.parametrize('modal_first', [True, False], ids=['modal-first', 'code-mode-first'])
    async def test_file_tools_are_still_sandboxed(self, fake_modal: FakeModal, modal_first: bool) -> None:
        import sniffio

        if sniffio.current_async_library() != 'asyncio':  # pragma: no cover
            pytest.skip('Agent.run() requires asyncio')
        tools = await _tools_offered_to_model(modal_first=modal_first)

        run_code_description = tools['run_code']
        assert run_code_description is not None
        for name in ('read_file', 'write_file', 'list_directory'):
            assert name not in tools
            assert f'async def {name}' in run_code_description
