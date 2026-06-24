"""Tests for the ModalSandbox capability and ModalSandboxToolset."""

from __future__ import annotations

import pytest
from pydantic_ai import Agent
from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.models.test import TestModel

from pydantic_ai_harness.experimental.modal_sandbox import ModalSandbox, ModalSandboxToolset

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

    async def test_output_truncated(self, fake_modal: FakeModal) -> None:
        fake_modal.responder = lambda argv, timeout: ('x' * 1000, '', 0)
        async with _toolset(max_output_chars=100) as ts:
            result = await ts.run_command('big')
        assert 'output truncated' in result


class TestReadFile:
    async def test_returns_contents(self, fake_modal: FakeModal) -> None:
        async with _toolset() as ts:
            fake_modal.sandboxes[0].files['/etc/hosts'] = 'file body\n'
            assert await ts.read_file('/etc/hosts') == 'file body\n'

    async def test_at_size_limit_is_not_truncated(self, fake_modal: FakeModal) -> None:
        async with _toolset(max_output_chars=100) as ts:
            fake_modal.sandboxes[0].files['/f'] = 'x' * 100
            assert await ts.read_file('/f') == 'x' * 100

    async def test_over_size_limit_keeps_tail(self, fake_modal: FakeModal) -> None:
        async with _toolset(max_output_chars=100) as ts:
            fake_modal.sandboxes[0].files['/big'] = 'HEAD' + 'T' * 100
            result = await ts.read_file('/big')
        assert result.endswith('T' * 100)
        assert 'HEAD' not in result
        assert 'output truncated' in result

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
        assert sandbox.files['/tmp/pkg/a.py'] == 'print(1)\n'
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

    def test_instructions_enabled_by_default(self) -> None:
        instructions = ModalSandbox().get_instructions()
        assert instructions is not None
        assert 'Modal sandbox' in instructions
        assert 'run_command' in instructions

    def test_instructions_can_be_disabled(self) -> None:
        assert ModalSandbox(include_instructions=False).get_instructions() is None

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
