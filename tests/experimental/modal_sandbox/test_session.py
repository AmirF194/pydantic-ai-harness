"""Tests for ModalSandboxSession."""

from __future__ import annotations

import builtins
import sys

import anyio
import pytest

from pydantic_ai_harness.experimental.modal_sandbox import ModalSandboxError, ModalSandboxSession
from pydantic_ai_harness.experimental.modal_sandbox import _session as session_module

from .fake_modal import FakeModal, FileInfo, _AioCallable


class _HangingCall(_AioCallable):
    """A teardown RPC that never returns, to prove the teardown deadline bounds it."""

    def __init__(self) -> None:
        super().__init__(lambda: None)

    async def aio(self, *args: object, **kwargs: object) -> None:
        await anyio.sleep_forever()


class TestOwnedLifecycle:
    async def test_creates_from_config_then_terminates(self, fake_modal: FakeModal) -> None:
        async with ModalSandboxSession(
            image='ubuntu:22.04',
            app_name='my-app',
            create_app_if_missing=False,
            sandbox_timeout=120,
            workdir='/work',
        ) as session:
            assert session.sandbox_id == 'sb-owned'
        # The sandbox is created from the configured app, image, timeout, and workdir.
        assert fake_modal.app_lookups[-1] == {'name': 'my-app', 'create_if_missing': False}
        assert fake_modal.image_tags[-1] == 'ubuntu:22.04'
        create_kwargs = fake_modal.create_kwargs[-1]
        assert create_kwargs['timeout'] == 120
        assert create_kwargs['workdir'] == '/work'
        # An owned sandbox is terminated and the client detached on exit.
        assert fake_modal.sandboxes[0].terminated is True
        assert fake_modal.sandboxes[0].detached is True

    async def test_default_app_and_image(self, fake_modal: FakeModal) -> None:
        async with ModalSandboxSession():
            pass
        assert fake_modal.app_lookups[-1] == {'name': 'pydantic-ai-harness', 'create_if_missing': True}
        assert fake_modal.image_tags[-1] == 'python:3.12-slim'

    async def test_sandbox_id_none_before_enter(self, fake_modal: FakeModal) -> None:
        session = ModalSandboxSession()
        assert session.sandbox_id is None

    async def test_exit_without_enter_is_safe(self) -> None:
        await ModalSandboxSession().__aexit__(None, None, None)

    async def test_detaches_even_when_terminate_fails(self, fake_modal: FakeModal) -> None:
        session = ModalSandboxSession()
        await session.__aenter__()
        fake_modal.sandboxes[0].terminate_error = RuntimeError('terminate boom')
        with pytest.raises(RuntimeError, match='terminate boom'):
            await session.__aexit__(None, None, None)
        # The client is detached even though terminate raised, so the attachment is not leaked.
        assert fake_modal.sandboxes[0].detached is True

    async def test_teardown_bounded_when_terminate_hangs(
        self, fake_modal: FakeModal, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # If Modal's control plane stalls, terminate must not hang the caller forever: the
        # shielded teardown gives each RPC a deadline, and detach still runs after it fires.
        monkeypatch.setattr(session_module, '_TEARDOWN_TIMEOUT', 0.05)
        session = ModalSandboxSession()
        await session.__aenter__()
        fake_modal.sandboxes[0].terminate = _HangingCall()
        with anyio.fail_after(5):
            await session.__aexit__(None, None, None)
        assert fake_modal.sandboxes[0].detached is True


class TestAttachLifecycle:
    async def test_attaches_detaches_but_does_not_terminate(self, fake_modal: FakeModal) -> None:
        async with ModalSandboxSession(sandbox_id='sb-existing') as session:
            assert session.sandbox_id == 'sb-existing'
        assert fake_modal.attach_ids == ['sb-existing']
        # An attached sandbox keeps running (no terminate) but the client is detached.
        assert fake_modal.sandboxes[0].terminated is False
        assert fake_modal.sandboxes[0].detached is True


class TestErrors:
    async def test_missing_modal_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        real_import = builtins.__import__

        def fake_import(name: str, *args: object, **kwargs: object) -> object:
            if name == 'modal':
                raise ImportError('No module named modal')
            return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

        monkeypatch.delitem(sys.modules, 'modal', raising=False)
        monkeypatch.setattr(builtins, '__import__', fake_import)
        with pytest.raises(ModalSandboxError, match='modal.*package is required'):
            async with ModalSandboxSession():
                pass  # pragma: no cover

    async def test_modal_error_wrapped(self, fake_modal: FakeModal) -> None:
        fake_modal.create_error = fake_modal.error_type('boom')
        with pytest.raises(ModalSandboxError, match='Could not start Modal sandbox: boom'):
            async with ModalSandboxSession():
                pass  # pragma: no cover

    async def test_exec_without_session_raises(self) -> None:
        session = ModalSandboxSession()
        with pytest.raises(ModalSandboxError, match='sandbox is not running'):
            await session.exec(['echo', 'hi'])


class TestExec:
    async def test_returns_stdout_stderr_nonzero_code(self, fake_modal: FakeModal) -> None:
        fake_modal.responder = lambda argv, timeout: ('out', 'err', 7)
        async with ModalSandboxSession() as session:
            result = await session.exec(['whatever'], timeout=5)
            assert (result.stdout, result.stderr, result.returncode) == ('out', 'err', 7)
            call = fake_modal.sandboxes[0].exec_calls[-1]
            assert call.argv == ['whatever']
            assert call.timeout == 5

    async def test_zero_exit_code(self, fake_modal: FakeModal) -> None:
        fake_modal.responder = lambda argv, timeout: ('done\n', '', 0)
        async with ModalSandboxSession() as session:
            result = await session.exec(['echo', 'done'])
            assert (result.stdout, result.stderr, result.returncode) == ('done\n', '', 0)
            assert result.timed_out is False

    async def test_timeout_sentinel_sets_timed_out(self, fake_modal: FakeModal) -> None:
        # Modal returns -1 when it kills a command at its timeout.
        fake_modal.responder = lambda argv, timeout: ('partial\n', '', -1)
        async with ModalSandboxSession() as session:
            result = await session.exec(['sleep', '99'], timeout=1)
            assert result.timed_out is True
            assert result.returncode == -1

    async def test_exec_error_wrapped(self, fake_modal: FakeModal) -> None:
        def boom(argv: list[str], timeout: int | None) -> tuple[str, str, int]:
            raise fake_modal.error_type('exec boom')

        fake_modal.responder = boom
        async with ModalSandboxSession() as session:
            with pytest.raises(ModalSandboxError, match='Command could not run in the sandbox: exec boom'):
                await session.exec(['whatever'])


class TestFilesystem:
    async def test_write_then_read_round_trips(self, fake_modal: FakeModal) -> None:
        async with ModalSandboxSession() as session:
            await session.write_bytes('/work/app/main.py', b'print(1)\n')
            assert await session.read_bytes('/work/app/main.py') == b'print(1)\n'
        sandbox = fake_modal.sandboxes[0]
        # Parent directories are created before the write.
        assert sandbox.made_dirs == ['/work/app']

    async def test_write_at_root_skips_make_directory(self, fake_modal: FakeModal) -> None:
        async with ModalSandboxSession() as session:
            await session.write_bytes('/file.txt', b'data')
        # The parent is the filesystem root, so no directory is created.
        assert fake_modal.sandboxes[0].made_dirs == []
        assert '/file.txt' in fake_modal.sandboxes[0].files

    async def test_list_files_normalizes_to_name_is_dir(self, fake_modal: FakeModal) -> None:
        async with ModalSandboxSession() as session:
            fake_modal.sandboxes[0].listing = [FileInfo('a.py', False), FileInfo('sub', True)]
            assert await session.list_files('/work') == [('a.py', False), ('sub', True)]
            assert fake_modal.sandboxes[0].list_paths == ['/work']

    async def test_read_error_wrapped(self, fake_modal: FakeModal) -> None:
        async with ModalSandboxSession() as session:
            fake_modal.sandboxes[0].fs_error = fake_modal.filesystem_error_type('No such file: /x')
            with pytest.raises(ModalSandboxError, match='No such file: /x'):
                await session.read_bytes('/x')

    async def test_write_error_wrapped(self, fake_modal: FakeModal) -> None:
        async with ModalSandboxSession() as session:
            fake_modal.sandboxes[0].fs_error = fake_modal.filesystem_error_type('Permission denied: /x')
            with pytest.raises(ModalSandboxError, match='Permission denied: /x'):
                await session.write_bytes('/x', b'data')

    async def test_list_error_wrapped(self, fake_modal: FakeModal) -> None:
        async with ModalSandboxSession() as session:
            fake_modal.sandboxes[0].fs_error = fake_modal.filesystem_error_type('Not a directory: /x')
            with pytest.raises(ModalSandboxError, match='Not a directory: /x'):
                await session.list_files('/x')

    async def test_file_size_returns_size_without_reading(self, fake_modal: FakeModal) -> None:
        async with ModalSandboxSession() as session:
            fake_modal.sandboxes[0].files['/f'] = b'hello'
            assert await session.file_size('/f') == 5

    async def test_file_size_error_wrapped(self, fake_modal: FakeModal) -> None:
        async with ModalSandboxSession() as session:
            fake_modal.sandboxes[0].fs_error = fake_modal.filesystem_error_type('No such file: /x')
            with pytest.raises(ModalSandboxError, match='No such file: /x'):
                await session.file_size('/x')

    async def test_filesystem_without_session_raises(self) -> None:
        session = ModalSandboxSession()
        with pytest.raises(ModalSandboxError, match='sandbox is not running'):
            await session.read_bytes('/x')

    async def test_filesystem_wraps_plain_modal_error(self, fake_modal: FakeModal) -> None:
        # A non-filesystem Modal error (e.g. a dropped connection) must still come back as a
        # ModalSandboxError, not leak the raw modal exception to the caller.
        async with ModalSandboxSession() as session:
            fake_modal.sandboxes[0].fs_error = fake_modal.error_type('connection lost')
            with pytest.raises(ModalSandboxError, match='connection lost'):
                await session.read_bytes('/x')


class TestPathResolution:
    async def test_relative_path_joined_with_pwd(self, fake_modal: FakeModal) -> None:
        fake_modal.responder = lambda argv, timeout: ('/work\n', '', 0)
        async with ModalSandboxSession() as session:
            await session.write_bytes('pkg/main.py', b'x')
        sandbox = fake_modal.sandboxes[0]
        assert '/work/pkg/main.py' in sandbox.files
        assert sandbox.made_dirs == ['/work/pkg']

    async def test_absolute_path_skips_pwd(self, fake_modal: FakeModal) -> None:
        async with ModalSandboxSession() as session:
            await session.write_bytes('/abs/file.txt', b'x')
        # No `pwd` lookup is needed for an absolute path.
        assert fake_modal.sandboxes[0].exec_calls == []

    async def test_cwd_queried_once_and_cached(self, fake_modal: FakeModal) -> None:
        fake_modal.responder = lambda argv, timeout: ('/work\n', '', 0)
        async with ModalSandboxSession() as session:
            fake_modal.sandboxes[0].files['/work/a.txt'] = b'body'
            await session.read_bytes('a.txt')
            await session.list_files('sub')
        pwd_calls = [c for c in fake_modal.sandboxes[0].exec_calls if c.argv == ['sh', '-c', 'pwd']]
        assert len(pwd_calls) == 1
        # The internal pwd probe carries a finite deadline so it cannot orphan on cancel.
        assert pwd_calls[0].timeout is not None and pwd_calls[0].timeout > 0

    async def test_blank_pwd_falls_back_to_root(self, fake_modal: FakeModal) -> None:
        fake_modal.responder = lambda argv, timeout: ('', '', 0)
        async with ModalSandboxSession() as session:
            await session.write_bytes('file.txt', b'x')
        assert '/file.txt' in fake_modal.sandboxes[0].files

    async def test_absolute_path_normalized(self, fake_modal: FakeModal) -> None:
        # An absolute path with `..` is normalized before hitting Modal's filesystem API.
        async with ModalSandboxSession() as session:
            await session.write_bytes('/work/../data/f.txt', b'x')
        assert '/data/f.txt' in fake_modal.sandboxes[0].files
        assert fake_modal.sandboxes[0].made_dirs == ['/data']

    async def test_double_slash_absolute_parent_skips_make_directory(self, fake_modal: FakeModal) -> None:
        # POSIX normpath preserves a leading '//', so its parent is '//' (still root). The
        # guard must skip make_directory for it rather than try to create a root alias.
        async with ModalSandboxSession() as session:
            await session.write_bytes('//file.txt', b'x')
        assert fake_modal.sandboxes[0].made_dirs == []
        assert '//file.txt' in fake_modal.sandboxes[0].files

    async def test_failed_pwd_probe_not_cached(self, fake_modal: FakeModal) -> None:
        # A timed-out/failed pwd probe must not cache a bogus cwd: it raises and the next
        # call re-probes rather than silently resolving every relative path against '/'.
        codes = iter([-1, 0])
        fake_modal.responder = lambda argv, timeout: ('', '', next(codes))
        async with ModalSandboxSession() as session:
            with pytest.raises(ModalSandboxError, match='Could not determine the sandbox working directory'):
                await session.read_bytes('rel.txt')
            fake_modal.sandboxes[0].files['/rel.txt'] = b'ok'
            assert await session.read_bytes('rel.txt') == b'ok'

    async def test_cwd_not_carried_across_reentry(self, fake_modal: FakeModal) -> None:
        # A reused session must re-query pwd for the new sandbox rather than reuse the
        # cwd cached during the first entry.
        responses = iter(['/first\n', '/second\n'])
        fake_modal.responder = lambda argv, timeout: (next(responses), '', 0)
        session = ModalSandboxSession()
        async with session:
            await session.write_bytes('a.txt', b'x')
        async with session:
            await session.write_bytes('b.txt', b'y')
        assert '/first/a.txt' in fake_modal.sandboxes[0].files
        assert '/second/b.txt' in fake_modal.sandboxes[1].files
