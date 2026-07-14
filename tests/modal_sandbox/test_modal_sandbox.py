"""Tests for the public Modal sandbox capability API."""

from __future__ import annotations

import inspect
from collections.abc import AsyncGenerator, Mapping
from contextlib import asynccontextmanager
from typing import Protocol, TypeGuard, runtime_checkable

import pytest
from pydantic_ai import Agent, RunContext
from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.toolsets import AbstractToolset
from pydantic_ai.usage import RunUsage

from pydantic_ai_harness.modal_sandbox import (
    ModalSandbox,
    ModalSandboxError,
    ModalSandboxSession,
    ModalSandboxUnavailableError,
)

from .fake_modal import FakeModal, FileInfo


@runtime_checkable
class _ModalSandboxTools(Protocol):  # pragma: no cover - structural typing only
    async def run_command(self, command: str, *, timeout_seconds: float | None = None) -> str: ...

    async def read_file(self, path: str, *, offset: int = 1, limit: int | None = None) -> str: ...

    async def write_file(self, path: str, content: str) -> str: ...

    async def list_directory(self, path: str = '.') -> str: ...


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


@asynccontextmanager
async def _toolset(
    *,
    sandbox_id: str | None = None,
    max_output_bytes: int = 50_000,
    max_output_lines: int = 2000,
    image: str = 'python:3.12-slim',
    app_name: str = 'pydantic-ai-harness',
    create_app_if_missing: bool = True,
    sandbox_timeout: int = 300,
    workdir: str | None = None,
    max_command_timeout: int | None = None,
    max_read_bytes: int = 5 * 1024 * 1024,
    env: Mapping[str, str] | None = None,
    session: ModalSandboxSession | None = None,
) -> AsyncGenerator[_ModalSandboxTools]:
    toolset = ModalSandbox[None](
        image=image,
        sandbox_id=sandbox_id,
        app_name=app_name,
        create_app_if_missing=create_app_if_missing,
        sandbox_timeout=sandbox_timeout,
        workdir=workdir,
        default_command_timeout=30.0,
        max_command_timeout=max_command_timeout,
        max_output_bytes=max_output_bytes,
        max_output_lines=max_output_lines,
        max_read_bytes=max_read_bytes,
        env=env,
        session=session,
    ).get_toolset()
    if not _is_abstract_toolset(toolset):  # pragma: no cover - capability contract
        raise AssertionError('ModalSandbox must return an AbstractToolset')
    run_toolset = await toolset.for_run(_run_context())
    if not isinstance(run_toolset, _ModalSandboxTools):  # pragma: no cover - capability contract
        raise AssertionError('ModalSandbox toolset is missing its public tools')
    async with run_toolset:
        yield run_toolset


class TestRunCommand:
    async def test_labels_stdout(self, fake_modal: FakeModal) -> None:
        fake_modal.responder = lambda argv, timeout: ('hello\n', '', 0)
        async with _toolset() as ts:
            result = await ts.run_command('echo hello')
        assert result == '[stdout]\nhello\n'
        assert fake_modal.sandboxes[0].exec_calls[-1].argv == ['sh', '-c', 'echo hello']

    async def test_combines_stdout_stderr_and_exit_code(self, fake_modal: FakeModal) -> None:
        fake_modal.responder = lambda argv, timeout: ('out\n', 'err\n', 2)
        async with _toolset() as ts:
            result = await ts.run_command('false')
        assert result == '[stdout]\nout\n\n[stderr]\nerr\n\n[exit code: 2]'

    async def test_no_output(self, fake_modal: FakeModal) -> None:
        fake_modal.responder = lambda argv, timeout: ('', '', 0)
        async with _toolset() as ts:
            assert await ts.run_command('true') == '(no output)'

    async def test_per_call_timeout_passed(self, fake_modal: FakeModal) -> None:
        fake_modal.responder = lambda argv, timeout: (str(timeout), '', 0)
        async with _toolset() as ts:
            await ts.run_command('echo', timeout_seconds=12.0)
        assert fake_modal.sandboxes[0].exec_calls[-1].timeout == 12

    async def test_timeout_clamped_to_sandbox_timeout(self, fake_modal: FakeModal) -> None:
        # Modal cannot kill a running command, so a model-supplied timeout is capped. With no
        # explicit ceiling it falls back to the sandbox lifetime.
        fake_modal.responder = lambda argv, timeout: (str(timeout), '', 0)
        async with _toolset(sandbox_timeout=120) as ts:
            await ts.run_command('echo', timeout_seconds=9999)
        assert fake_modal.sandboxes[0].exec_calls[-1].timeout == 120

    async def test_max_command_timeout_overrides_ceiling(self, fake_modal: FakeModal) -> None:
        # An explicit ceiling lets an attached/injected sandbox allow longer or shorter
        # single commands than the sandbox lifetime.
        fake_modal.responder = lambda argv, timeout: (str(timeout), '', 0)
        async with _toolset(sandbox_timeout=120, max_command_timeout=50) as ts:
            await ts.run_command('echo', timeout_seconds=9999)
        assert fake_modal.sandboxes[0].exec_calls[-1].timeout == 50

    async def test_fractional_timeout_rounds_up(self, fake_modal: FakeModal) -> None:
        fake_modal.responder = lambda argv, timeout: (str(timeout), '', 0)
        async with _toolset() as ts:
            await ts.run_command('echo', timeout_seconds=0.5)
        # A sub-second timeout must not floor to 0, which Modal treats as "no timeout".
        assert fake_modal.sandboxes[0].exec_calls[-1].timeout == 1

    @pytest.mark.parametrize('bad_timeout', [0, -5.0])
    async def test_non_positive_timeout_rejected(self, fake_modal: FakeModal, bad_timeout: float) -> None:
        # A 0 or negative request is a model mistake; reject it rather than let the session
        # floor it to a surprise 1-second deadline.
        fake_modal.responder = lambda argv, timeout: ('', '', 0)
        async with _toolset() as ts:
            with pytest.raises(ModelRetry, match='timeout_seconds must be greater than 0'):
                await ts.run_command('echo', timeout_seconds=bad_timeout)
        assert fake_modal.sandboxes[0].exec_calls == []

    async def test_output_truncated(self, fake_modal: FakeModal) -> None:
        fake_modal.responder = lambda argv, timeout: ('x' * 1000, '', 0)
        async with _toolset(max_output_bytes=100) as ts:
            result = await ts.run_command('big')
        assert 'output truncated' in result

    async def test_timeout_is_reported(self, fake_modal: FakeModal) -> None:
        # Modal's -1 timeout sentinel becomes a legible note, not a bare exit code.
        fake_modal.responder = lambda argv, timeout: ('partial\n', '', -1)
        async with _toolset() as ts:
            result = await ts.run_command('sleep 99', timeout_seconds=5)
        assert result == '[stdout]\npartial\n\n[timed out after 5s]'

    async def test_exec_failure_raises_model_retry(self, fake_modal: FakeModal) -> None:
        def boom(argv: list[str], timeout: int | None) -> tuple[str, str, int]:
            raise fake_modal.error_type('transient blip')

        fake_modal.responder = boom
        async with _toolset() as ts:
            with pytest.raises(ModelRetry, match='Command could not run in the sandbox: transient blip'):
                await ts.run_command('echo hi')

    async def test_terminal_failure_ends_the_run(self, fake_modal: FakeModal) -> None:
        # A dead sandbox is terminal: run_command must not turn it into a ModelRetry (which
        # would loop the model against a sandbox that is never coming back). It propagates.
        def gone(argv: list[str], timeout: int | None) -> tuple[str, str, int]:
            raise fake_modal.sandbox_terminated_type('sandbox terminated')

        fake_modal.responder = gone
        async with _toolset() as ts:
            with pytest.raises(ModalSandboxUnavailableError, match='no longer running'):
                await ts.run_command('echo hi')

    async def test_output_is_bounded_end_to_end(self, fake_modal: FakeModal) -> None:
        # A command that floods stdout is capped before it reaches the model, and the cut is
        # marked. One-char chunks drive the session's bounded reader.
        fake_modal.output_chunk_size = 1
        fake_modal.responder = lambda argv, timeout: ('A' * 500 + 'END', '', 0)
        async with _toolset(max_output_bytes=50) as ts:
            result = await ts.run_command('flood')
        assert 'output truncated' in result
        assert result.endswith('END')  # the tail, where the exit status lives, survives

    async def test_output_is_bounded_when_one_chunk_exceeds_limit(self, fake_modal: FakeModal) -> None:
        fake_modal.responder = lambda argv, timeout: ('A' * 500 + 'END', '', 0)
        async with _toolset(max_output_bytes=50) as ts:
            result = await ts.run_command('flood')
        assert 'output truncated' in result
        assert result.endswith('END')

    async def test_output_line_cap_keeps_last_lines(self, fake_modal: FakeModal) -> None:
        fake_modal.responder = lambda argv, timeout: ('first\nsecond\nthird', '', 0)
        async with _toolset(max_output_lines=1) as ts:
            result = await ts.run_command('many-lines')
        assert 'truncated to the last 1 lines' in result
        assert result.endswith('third')

    async def test_multibyte_output_truncation_drops_partial_character(self, fake_modal: FakeModal) -> None:
        fake_modal.responder = lambda argv, timeout: ('é' * 20, '', 0)
        async with _toolset(max_output_bytes=5) as ts:
            result = await ts.run_command('unicode')
        assert 'output truncated to the last 5B' in result


class TestReadFile:
    async def test_returns_contents(self, fake_modal: FakeModal) -> None:
        async with _toolset() as ts:
            # The single trailing newline is dropped with its phantom empty line, so the read
            # returns the real line content rather than a body counted as two lines.
            fake_modal.sandboxes[0].files['/etc/hosts'] = b'file body\n'
            assert await ts.read_file('/etc/hosts') == 'file body'

    async def test_at_size_limit_is_not_truncated(self, fake_modal: FakeModal) -> None:
        async with _toolset(max_output_bytes=100) as ts:
            fake_modal.sandboxes[0].files['/f'] = b'x' * 100
            assert await ts.read_file('/f') == 'x' * 100

    async def test_over_size_limit_pages_from_head(self, fake_modal: FakeModal) -> None:
        async with _toolset(max_output_bytes=20) as ts:
            fake_modal.sandboxes[0].files['/big'] = b'\n'.join(b'line%02d' % i for i in range(10))
            result = await ts.read_file('/big')
        # File reads keep the head and tell the model how to page the rest.
        assert result.startswith('line00')
        assert 'Use offset=' in result

    async def test_offset_and_limit(self, fake_modal: FakeModal) -> None:
        async with _toolset() as ts:
            fake_modal.sandboxes[0].files['/f'] = b'a\nb\nc\nd\ne'
            result = await ts.read_file('/f', offset=2, limit=2)
        assert result.startswith('b\nc')
        assert 'more lines in file. Use offset=4 to continue.' in result

    async def test_binary_file_raises_model_retry(self, fake_modal: FakeModal) -> None:
        async with _toolset() as ts:
            fake_modal.sandboxes[0].files['/img.png'] = b'\xff\xfe\x00\x01'
            with pytest.raises(ModelRetry, match='not valid UTF-8'):
                await ts.read_file('/img.png')

    async def test_error_raises_model_retry(self, fake_modal: FakeModal) -> None:
        async with _toolset() as ts:
            fake_modal.sandboxes[0].fs_error = fake_modal.filesystem_error_type('No such file: /nope')
            with pytest.raises(ModelRetry, match="Could not read '/nope': No such file"):
                await ts.read_file('/nope')

    async def test_refuses_file_over_read_limit(self, fake_modal: FakeModal) -> None:
        async with _toolset(max_read_bytes=1000) as ts:
            # Report a large size without allocating the bytes; the guard fires before read_bytes.
            fake_modal.sandboxes[0].stat_sizes['/big.log'] = 5000
            with pytest.raises(ModelRetry, match='over the 1000B read limit'):
                await ts.read_file('/big.log')

    async def test_read_limit_formats_megabytes(self, fake_modal: FakeModal) -> None:
        async with _toolset(max_read_bytes=1000) as ts:
            fake_modal.sandboxes[0].stat_sizes['/big.log'] = 3 * 1024 * 1024
            with pytest.raises(ModelRetry, match='File is 3.0MB'):
                await ts.read_file('/big.log')

    async def test_line_cap_is_configurable(self, fake_modal: FakeModal) -> None:
        # Bytes are well under budget, so only the line cap can fire: proves it is plumbed
        # through, not silently fixed at the helper default.
        async with _toolset(max_output_lines=3, max_output_bytes=50_000) as ts:
            fake_modal.sandboxes[0].files['/many'] = b'\n'.join(b'L%d' % i for i in range(20))
            result = await ts.read_file('/many')
        assert 'Showing lines 1-3 of 20' in result
        assert 'Use offset=4 to continue.' in result

    async def test_refuses_file_that_grew_after_stat(self, fake_modal: FakeModal) -> None:
        async with _toolset(max_read_bytes=1000) as ts:
            # stat reports under the limit, but the read returns more (the file grew between
            # the two round-trips). The post-read guard refuses before the decode/window.
            fake_modal.sandboxes[0].stat_sizes['/grows'] = 10
            fake_modal.sandboxes[0].files['/grows'] = b'x' * 5000
            with pytest.raises(ModelRetry, match='over the 1000B read limit'):
                await ts.read_file('/grows')

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
        async with _toolset() as ts:
            fake_modal.sandboxes[0].files['/f'] = b'one\ntwo'
            with pytest.raises(ModelRetry, match=message):
                await ts.read_file('/f', **kwargs)

    async def test_oversized_first_line_is_omitted(self, fake_modal: FakeModal) -> None:
        async with _toolset(max_output_bytes=10) as ts:
            fake_modal.sandboxes[0].files['/f'] = b'x' * 200
            result = await ts.read_file('/f')
        assert result == '[Line 1 is 200B, exceeds the 10B limit and was omitted.]'

    async def test_oversized_first_line_points_to_next_line(self, fake_modal: FakeModal) -> None:
        async with _toolset(max_output_bytes=10) as ts:
            fake_modal.sandboxes[0].files['/f'] = b'x' * 200 + b'\nnext'
            result = await ts.read_file('/f')
        assert 'Use offset=2 to continue.' in result

    async def test_byte_cap_returns_continuation_offset(self, fake_modal: FakeModal) -> None:
        async with _toolset(max_output_bytes=12) as ts:
            fake_modal.sandboxes[0].files['/f'] = b'line00\nline01\nline02'
            result = await ts.read_file('/f')
        assert '(12B limit). Use offset=' in result

    async def test_terminal_error_ends_the_run(self, fake_modal: FakeModal) -> None:
        # A missing sandbox during a read is terminal, not a retryable "could not read".
        async with _toolset() as ts:
            fake_modal.sandboxes[0].fs_error = fake_modal.unavailable_type('sandbox not found')
            fake_modal.sandboxes[0].poll_result = 0
            with pytest.raises(ModalSandboxUnavailableError):
                await ts.read_file('/x')

    async def test_wrapped_not_found_is_recoverable_while_sandbox_runs(self, fake_modal: FakeModal) -> None:
        async with _toolset() as ts:
            fake_modal.sandboxes[0].fs_error = fake_modal.unavailable_type('filesystem request failed')
            with pytest.raises(ModelRetry, match='filesystem request failed'):
                await ts.read_file('/x')

    async def test_wrapped_auth_error_is_terminal(self, fake_modal: FakeModal) -> None:
        async with _toolset() as ts:
            fake_modal.sandboxes[0].fs_error = fake_modal.filesystem_error_type('filesystem request failed')
            fake_modal.sandboxes[0].poll_error = fake_modal.auth_type('unauthenticated')
            with pytest.raises(ModalSandboxError, match='Modal rejected the credentials'):
                await ts.read_file('/x')


class TestWriteFile:
    async def test_writes_and_creates_parents(self, fake_modal: FakeModal) -> None:
        async with _toolset() as ts:
            result = await ts.write_file('/tmp/pkg/a.py', 'print(1)\n')
        assert result == "Wrote 9 bytes to '/tmp/pkg/a.py'."
        sandbox = fake_modal.sandboxes[0]
        assert sandbox.files['/tmp/pkg/a.py'] == b'print(1)\n'
        assert sandbox.made_dirs == ['/tmp/pkg']

    async def test_error_raises_model_retry(self, fake_modal: FakeModal) -> None:
        async with _toolset() as ts:
            fake_modal.sandboxes[0].fs_error = fake_modal.filesystem_error_type('Permission denied: /root/x')
            with pytest.raises(ModelRetry, match="Could not write '/root/x': Permission denied"):
                await ts.write_file('/root/x', 'data')

    async def test_terminal_error_ends_the_run(self, fake_modal: FakeModal) -> None:
        async with _toolset() as ts:
            fake_modal.sandboxes[0].fs_error = fake_modal.unavailable_type('sandbox not found')
            fake_modal.sandboxes[0].poll_result = 0
            with pytest.raises(ModalSandboxUnavailableError):
                await ts.write_file('/x', 'data')


class TestListDirectory:
    async def test_lists_with_trailing_slash_on_dirs(self, fake_modal: FakeModal) -> None:
        async with _toolset() as ts:
            fake_modal.sandboxes[0].listing = [FileInfo('b', True), FileInfo('a', False)]
            assert await ts.list_directory('/tmp') == 'a\nb/'
        assert fake_modal.sandboxes[0].list_paths == ['/tmp']

    async def test_default_path_resolves_to_cwd(self, fake_modal: FakeModal) -> None:
        fake_modal.responder = lambda argv, timeout: ('/work\n', '', 0)
        async with _toolset() as ts:
            assert await ts.list_directory() == '(empty)'
        # The default '.' is resolved against the sandbox working directory.
        assert fake_modal.sandboxes[0].list_paths == ['/work']

    async def test_relative_path_resolved_against_cwd(self, fake_modal: FakeModal) -> None:
        fake_modal.responder = lambda argv, timeout: ('/work\n', '', 0)
        async with _toolset() as ts:
            fake_modal.sandboxes[0].listing = [FileInfo('a.py', False)]
            assert await ts.list_directory('src') == 'a.py'
        assert fake_modal.sandboxes[0].list_paths == ['/work/src']

    async def test_relative_path_preserves_parent_segments(self, fake_modal: FakeModal) -> None:
        fake_modal.responder = lambda argv, timeout: ('/work\n', '', 0)
        async with _toolset() as ts:
            await ts.list_directory('link/../secret')
        assert fake_modal.sandboxes[0].list_paths == ['/work/link/../secret']

    async def test_error_raises_model_retry(self, fake_modal: FakeModal) -> None:
        async with _toolset() as ts:
            fake_modal.sandboxes[0].fs_error = fake_modal.filesystem_error_type('Not a directory: /etc/hosts')
            with pytest.raises(ModelRetry, match="Could not list '/etc/hosts': Not a directory"):
                await ts.list_directory('/etc/hosts')

    async def test_terminal_error_ends_the_run(self, fake_modal: FakeModal) -> None:
        async with _toolset() as ts:
            fake_modal.sandboxes[0].fs_error = fake_modal.unavailable_type('sandbox not found')
            fake_modal.sandboxes[0].poll_result = 0
            with pytest.raises(ModalSandboxUnavailableError):
                await ts.list_directory('/x')


class TestToolsetLifecycle:
    async def test_enter_creates_sandbox_from_config(self, fake_modal: FakeModal) -> None:
        async with _toolset(image='ubuntu:22.04', app_name='my-app', sandbox_timeout=120, workdir='/work'):
            pass
        assert fake_modal.app_lookups[-1]['name'] == 'my-app'
        assert fake_modal.image_tags[-1] == 'ubuntu:22.04'
        assert fake_modal.create_kwargs[-1]['timeout'] == 120
        assert fake_modal.create_kwargs[-1]['workdir'] == '/work'

    async def test_env_passed_to_owned_sandbox(self, fake_modal: FakeModal) -> None:
        async with _toolset(env={'FOO': 'bar'}):
            pass
        assert fake_modal.create_kwargs[-1]['env'] == {'FOO': 'bar'}

    async def test_for_run_carries_config_to_a_fresh_instance(self, fake_modal: FakeModal) -> None:
        original = ModalSandbox[None](image='ubuntu:22.04', app_name='my-app', sandbox_timeout=99).get_toolset()
        assert _is_abstract_toolset(original)
        fresh = await original.for_run(_run_context())
        assert fresh is not original
        async with fresh:
            pass
        assert fake_modal.image_tags[-1] == 'ubuntu:22.04'
        assert fake_modal.app_lookups[-1]['name'] == 'my-app'
        assert fake_modal.create_kwargs[-1]['timeout'] == 99

    async def test_agent_level_toolset_enter_does_not_create_a_sandbox(self, fake_modal: FakeModal) -> None:
        toolset = ModalSandbox[None]().get_toolset()
        assert _is_abstract_toolset(toolset)
        async with toolset:
            pass
        assert fake_modal.sandboxes == []

    async def test_attached_sandbox_not_terminated(self, fake_modal: FakeModal) -> None:
        async with _toolset(sandbox_id='sb-keep') as ts:
            await ts.run_command('echo hi')
        assert fake_modal.attach_ids == ['sb-keep']
        assert fake_modal.sandboxes[0].terminated is False
        assert fake_modal.sandboxes[0].detached is True


class TestInjectedSession:
    async def test_uses_caller_session_without_opening_or_terminating(self, fake_modal: FakeModal) -> None:
        fake_modal.responder = lambda argv, timeout: ('hi\n', '', 0)
        async with ModalSandboxSession() as session:
            # The caller opened exactly one sandbox.
            assert len(fake_modal.sandboxes) == 1
            async with _toolset(session=session) as ts:
                assert await ts.run_command('echo hi') == '[stdout]\nhi\n'
            # The run reused the caller's sandbox (no new one) and left it running.
            assert len(fake_modal.sandboxes) == 1
            assert fake_modal.sandboxes[0].terminated is False
        # Closing the caller-owned session terminates its sandbox.
        assert fake_modal.sandboxes[0].terminated is True

    async def test_unopened_session_fails_at_run_start(self, fake_modal: FakeModal) -> None:
        # A session the caller never entered must fail clearly when the run starts.
        session = ModalSandboxSession()
        with pytest.raises(ModalSandboxError, match='injected session is not open'):
            async with _toolset(session=session):
                pass  # pragma: no cover

    async def test_for_run_carries_the_session(self, fake_modal: FakeModal) -> None:
        async with ModalSandboxSession() as session:
            original = ModalSandbox[None](session=session).get_toolset()
            assert _is_abstract_toolset(original)
            fresh = await original.for_run(_run_context())
            async with fresh:
                assert isinstance(fresh, _ModalSandboxTools)
                await fresh.run_command('echo hi')
            # The per-run clone reused the injected session rather than opening its own.
            assert len(fake_modal.sandboxes) == 1
            assert fake_modal.sandboxes[0].terminated is False


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

    async def test_session_with_only_defaults_is_allowed(self, fake_modal: FakeModal) -> None:
        async with ModalSandboxSession() as session:
            cap = ModalSandbox(session=session)
            assert cap.session is session

    async def test_session_rejects_sandbox_id(self, fake_modal: FakeModal) -> None:
        async with ModalSandboxSession() as session:
            with pytest.raises(ValueError, match='sandbox_id cannot be combined with `session`'):
                ModalSandbox(session=session, sandbox_id='sb-keep')

    async def test_session_rejects_owned_settings(self, fake_modal: FakeModal) -> None:
        async with ModalSandboxSession() as session:
            with pytest.raises(ValueError, match='image cannot be combined with `session`'):
                ModalSandbox(session=session, image='ubuntu:22.04')

    async def test_session_rejects_env(self, fake_modal: FakeModal) -> None:
        async with ModalSandboxSession() as session:
            with pytest.raises(ValueError, match='env cannot be combined with `session`'):
                ModalSandbox(session=session, env={'A': 'b'})

    async def test_session_rejecting_sandbox_timeout_points_at_max_command_timeout(self, fake_modal: FakeModal) -> None:
        async with ModalSandboxSession() as session:
            with pytest.raises(ValueError, match='set `max_command_timeout`'):
                ModalSandbox(session=session, sandbox_timeout=600)

    async def test_injected_session_instructions_say_persists(self, fake_modal: FakeModal) -> None:
        async with ModalSandboxSession() as session:
            instructions = ModalSandbox(session=session).get_instructions()
            assert instructions is not None
            assert 'persists across sessions' in instructions

    def test_instructions_enabled_by_default(self) -> None:
        instructions = ModalSandbox().get_instructions()
        assert instructions is not None
        assert 'Modal sandbox' in instructions
        assert 'run_command' in instructions

    def test_instructions_can_be_disabled(self) -> None:
        assert ModalSandbox(include_instructions=False).get_instructions() is None

    def test_owned_instructions_say_reset_between_sessions(self) -> None:
        instructions = ModalSandbox().get_instructions()
        assert instructions is not None
        assert 'reset between' in instructions

    def test_attached_instructions_say_persists(self) -> None:
        instructions = ModalSandbox(sandbox_id='sb-keep').get_instructions()
        assert instructions is not None
        assert 'persists across sessions' in instructions
        assert 'reset between' not in instructions

    def test_exported_from_capability_submodule(self) -> None:
        import pydantic_ai_harness
        import pydantic_ai_harness.modal_sandbox as modal_sandbox
        from pydantic_ai_harness.modal_sandbox import ModalSandbox as Exported

        assert Exported is ModalSandbox
        assert 'ModalSandboxToolset' not in modal_sandbox.__all__
        assert 'ModalSandboxExecResult' in modal_sandbox.__all__
        # Capabilities with optional dependencies are imported from their submodule, not the package root.
        assert 'ModalSandbox' not in pydantic_ai_harness.__all__

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
        assert tool_returns == ['[stdout]\nhello\n']
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
