"""Tests for the ModalSandbox capability and ModalSandboxToolset."""

from __future__ import annotations

import pytest
from pydantic_ai import Agent
from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.models.test import TestModel

from pydantic_ai_harness.experimental.modal_sandbox import (
    ModalSandbox,
    ModalSandboxError,
    ModalSandboxSession,
    ModalSandboxToolset,
)

from .fake_modal import FakeModal, FileInfo


def _toolset(
    *,
    sandbox_id: str | None = None,
    max_output_chars: int = 50_000,
    image: str = 'python:3.12-slim',
    app_name: str = 'pydantic-ai-harness',
    create_app_if_missing: bool = True,
    sandbox_timeout: int = 300,
    workdir: str | None = None,
    session: ModalSandboxSession | None = None,
) -> ModalSandboxToolset[None]:
    return ModalSandboxToolset[None](
        image=image,
        sandbox_id=sandbox_id,
        app_name=app_name,
        create_app_if_missing=create_app_if_missing,
        sandbox_timeout=sandbox_timeout,
        workdir=workdir,
        default_timeout=30.0,
        max_output_chars=max_output_chars,
        session=session,
    )


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

    async def test_fractional_timeout_rounds_up(self, fake_modal: FakeModal) -> None:
        fake_modal.responder = lambda argv, timeout: (str(timeout), '', 0)
        async with _toolset() as ts:
            await ts.run_command('echo', timeout_seconds=0.5)
        # A sub-second timeout must not floor to 0, which Modal treats as "no timeout".
        assert fake_modal.sandboxes[0].exec_calls[-1].timeout == 1

    async def test_output_truncated(self, fake_modal: FakeModal) -> None:
        fake_modal.responder = lambda argv, timeout: ('x' * 1000, '', 0)
        async with _toolset(max_output_chars=100) as ts:
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
            raise fake_modal.error_type('sandbox gone')

        fake_modal.responder = boom
        async with _toolset() as ts:
            with pytest.raises(ModelRetry, match='Command could not run in the sandbox: sandbox gone'):
                await ts.run_command('echo hi')


class TestReadFile:
    async def test_returns_contents(self, fake_modal: FakeModal) -> None:
        async with _toolset() as ts:
            fake_modal.sandboxes[0].files['/etc/hosts'] = b'file body\n'
            assert await ts.read_file('/etc/hosts') == 'file body\n'

    async def test_at_size_limit_is_not_truncated(self, fake_modal: FakeModal) -> None:
        async with _toolset(max_output_chars=100) as ts:
            fake_modal.sandboxes[0].files['/f'] = b'x' * 100
            assert await ts.read_file('/f') == 'x' * 100

    async def test_over_size_limit_pages_from_head(self, fake_modal: FakeModal) -> None:
        async with _toolset(max_output_chars=20) as ts:
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


class TestWriteFile:
    async def test_writes_and_creates_parents(self, fake_modal: FakeModal) -> None:
        async with _toolset() as ts:
            result = await ts.write_file('/tmp/pkg/a.py', 'print(1)\n')
        assert result == "Wrote 9 characters to '/tmp/pkg/a.py'."
        sandbox = fake_modal.sandboxes[0]
        assert sandbox.files['/tmp/pkg/a.py'] == b'print(1)\n'
        assert sandbox.made_dirs == ['/tmp/pkg']

    async def test_error_raises_model_retry(self, fake_modal: FakeModal) -> None:
        async with _toolset() as ts:
            fake_modal.sandboxes[0].fs_error = fake_modal.filesystem_error_type('Permission denied: /root/x')
            with pytest.raises(ModelRetry, match="Could not write '/root/x': Permission denied"):
                await ts.write_file('/root/x', 'data')


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

    async def test_error_raises_model_retry(self, fake_modal: FakeModal) -> None:
        async with _toolset() as ts:
            fake_modal.sandboxes[0].fs_error = fake_modal.filesystem_error_type('Not a directory: /etc/hosts')
            with pytest.raises(ModelRetry, match="Could not list '/etc/hosts': Not a directory"):
                await ts.list_directory('/etc/hosts')


class TestToolsetLifecycle:
    async def test_enter_creates_sandbox_from_config(self, fake_modal: FakeModal) -> None:
        async with _toolset(image='ubuntu:22.04', app_name='my-app', sandbox_timeout=120, workdir='/work'):
            pass
        assert fake_modal.app_lookups[-1]['name'] == 'my-app'
        assert fake_modal.image_tags[-1] == 'ubuntu:22.04'
        assert fake_modal.create_kwargs[-1]['timeout'] == 120
        assert fake_modal.create_kwargs[-1]['workdir'] == '/work'

    async def test_for_run_carries_config_to_a_fresh_instance(self, fake_modal: FakeModal) -> None:
        original = _toolset(image='ubuntu:22.04', app_name='my-app', sandbox_timeout=99)
        fresh = await original.for_run(None)  # type: ignore[arg-type]
        assert isinstance(fresh, ModalSandboxToolset)
        assert fresh is not original
        async with fresh:
            pass
        assert fake_modal.image_tags[-1] == 'ubuntu:22.04'
        assert fake_modal.app_lookups[-1]['name'] == 'my-app'
        assert fake_modal.create_kwargs[-1]['timeout'] == 99

    async def test_aexit_without_session_is_safe(self) -> None:
        ts = _toolset()
        await ts.__aexit__(None, None, None)
        assert ts._session is None

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
            fresh = await _toolset(session=session).for_run(None)  # type: ignore[arg-type]
            assert isinstance(fresh, ModalSandboxToolset)
            async with fresh:
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
        assert cap.default_timeout == 60.0

    def test_get_toolset(self) -> None:
        assert isinstance(ModalSandbox().get_toolset(), ModalSandboxToolset)

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
        ],
    )
    def test_attach_rejects_owned_only_settings(self, kwargs: dict[str, object], expected: str) -> None:
        with pytest.raises(ValueError, match=f'{expected} only apply when creating a sandbox'):
            ModalSandbox(sandbox_id='sb-keep', **kwargs)  # type: ignore[arg-type]

    def test_attach_error_lists_every_conflicting_setting(self) -> None:
        with pytest.raises(ValueError, match='image, sandbox_timeout only apply'):
            ModalSandbox(sandbox_id='sb-keep', image='ubuntu:22.04', sandbox_timeout=600)

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

    def test_exported_from_experimental_namespace(self) -> None:
        import pydantic_ai_harness
        from pydantic_ai_harness.experimental.modal_sandbox import ModalSandbox as Exported

        assert Exported is ModalSandbox
        # Experimental capabilities are reached via the experimental namespace, not the package root.
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
