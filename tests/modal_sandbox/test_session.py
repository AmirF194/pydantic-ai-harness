"""Tests for ModalSandboxSession."""

from __future__ import annotations

import builtins
import sys

import anyio
import pytest

from pydantic_ai_harness.modal_sandbox import (
    ModalSandboxError,
    ModalSandboxSession,
    ModalSandboxTerminalError,
    ModalSandboxUnavailableError,
)

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
        assert await fake_modal.sandboxes[0].poll.aio() == 0

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

    async def test_terminating_an_already_gone_sandbox_is_not_an_error(self, fake_modal: FakeModal) -> None:
        # An owned run that outlived its sandbox_timeout self-terminates; the teardown terminate
        # then hits "already gone". That is success, not a failure to raise -- a raise here would
        # mask the terminal error the tool already surfaced.
        session = ModalSandboxSession()
        await session.__aenter__()
        fake_modal.sandboxes[0].terminate_error = fake_modal.sandbox_terminated_type('already terminated')
        await session.__aexit__(None, None, None)
        assert fake_modal.sandboxes[0].detached is True

    async def test_teardown_bounded_when_terminate_hangs(
        self, fake_modal: FakeModal, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # If Modal's control plane stalls, terminate must not hang the caller forever: the
        # shielded teardown gives each RPC a deadline, and detach still runs after it fires.
        monkeypatch.setattr('pydantic_ai_harness.modal_sandbox._session._TEARDOWN_TIMEOUT', 0.05)
        session = ModalSandboxSession()
        await session.__aenter__()
        fake_modal.sandboxes[0].terminate = _HangingCall()
        with anyio.fail_after(5):
            await session.__aexit__(None, None, None)
        assert fake_modal.sandboxes[0].detached is True

    async def test_cancel_during_enter_terminates_created_sandbox(self, fake_modal: FakeModal) -> None:
        # A run cancelled while the sandbox is being created must not orphan it: creation is
        # shielded so the handle survives, then the cancellation tears the sandbox down here
        # instead of leaving it for `sandbox_timeout` to reap.
        session = ModalSandboxSession()
        with anyio.CancelScope() as scope:
            scope.cancel()
            await session.__aenter__()
        # The scope absorbed the cancellation; the created sandbox was terminated and detached,
        # and the session holds no handle.
        assert fake_modal.sandboxes[0].terminated is True
        assert fake_modal.sandboxes[0].detached is True
        assert session.sandbox_id is None


class TestAttachLifecycle:
    async def test_attaches_detaches_but_does_not_terminate(self, fake_modal: FakeModal) -> None:
        async with ModalSandboxSession(sandbox_id='sb-existing') as session:
            assert session.sandbox_id == 'sb-existing'
        assert fake_modal.attach_ids == ['sb-existing']
        # An attached sandbox keeps running (no terminate) but the client is detached.
        assert fake_modal.sandboxes[0].terminated is False
        assert fake_modal.sandboxes[0].detached is True

    async def test_attach_to_terminated_sandbox_fails_at_enter(self, fake_modal: FakeModal) -> None:
        fake_modal.attach_poll_result = 0
        with pytest.raises(ModalSandboxUnavailableError, match='does not exist or has terminated'):
            async with ModalSandboxSession(sandbox_id='sb-finished'):
                pass  # pragma: no cover
        assert fake_modal.attach_ids == ['sb-finished']


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

    async def test_create_auth_error_is_terminal(self, fake_modal: FakeModal) -> None:
        # Rejected credentials cannot be fixed by retrying, so surface a terminal error with
        # an actionable message rather than a generic create failure.
        fake_modal.create_error = fake_modal.auth_type('bad token')
        with pytest.raises(ModalSandboxTerminalError, match='Modal rejected the credentials'):
            async with ModalSandboxSession():
                pass  # pragma: no cover

    async def test_attach_to_missing_sandbox_is_unavailable(self, fake_modal: FakeModal) -> None:
        # Attaching to an id that does not exist (or has terminated) is terminal: there is no
        # sandbox to talk to, so retrying cannot help.
        fake_modal.attach_error = fake_modal.unavailable_type('no such sandbox')
        with pytest.raises(ModalSandboxUnavailableError, match='does not exist or has terminated'):
            async with ModalSandboxSession(sandbox_id='sb-missing'):
                pass  # pragma: no cover

    async def test_owned_create_not_found_is_a_generic_start_failure(self, fake_modal: FakeModal) -> None:
        # A NotFound during creation can refer to app or image configuration, not a sandbox
        # that was already usable, so do not misclassify it as terminal sandbox expiry.
        fake_modal.create_error = fake_modal.unavailable_type('vanished mid-create')
        with pytest.raises(ModalSandboxError, match='Could not start Modal sandbox: vanished mid-create') as exc:
            async with ModalSandboxSession():
                pass  # pragma: no cover
        assert not isinstance(exc.value, ModalSandboxTerminalError)

    @pytest.mark.parametrize('value', [0, -1, True])
    def test_invalid_sandbox_timeout_rejected(self, value: int) -> None:
        with pytest.raises(ValueError, match='sandbox_timeout must be a positive integer'):
            ModalSandboxSession(sandbox_timeout=value)

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
    def test_attach_rejects_owned_configuration(self, kwargs: dict[str, object], expected: str) -> None:
        with pytest.raises(ValueError, match=f'{expected} only apply when creating a sandbox'):
            ModalSandboxSession(sandbox_id='sb-existing', **kwargs)  # type: ignore[arg-type]

    async def test_create_timeout_does_not_hang(self, fake_modal: FakeModal, monkeypatch: pytest.MonkeyPatch) -> None:
        # A wedged control plane must not make enter uncancellable: the bounded, shielded
        # create gives up after its deadline and fails instead of hanging forever.
        monkeypatch.setattr('pydantic_ai_harness.modal_sandbox._session._CREATE_TIMEOUT', 0.05)
        fake_modal.module.Sandbox.create = _HangingCall()  # type: ignore[attr-defined]
        with anyio.fail_after(5):
            with pytest.raises(ModalSandboxError, match='did not complete within'):
                async with ModalSandboxSession():
                    pass  # pragma: no cover


class TestExec:
    async def test_returns_stdout_stderr_nonzero_code(self, fake_modal: FakeModal) -> None:
        fake_modal.responder = lambda argv, timeout: ('out', 'err', 7)
        async with ModalSandboxSession() as session:
            result = await session.exec(['whatever'], timeout=5)
            assert (result.stdout, result.stderr, result.returncode) == ('out', 'err', 7)
            call = fake_modal.sandboxes[0].exec_calls[-1]
            assert call.argv == ['whatever']
            assert call.timeout == 5
            assert call.text is False

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

    async def test_fractional_timeout_rounded_to_whole_seconds(self, fake_modal: FakeModal) -> None:
        # The session owns Modal's whole-second quantization: a sub-second deadline rounds up
        # to 1 (Modal treats 0 as "no timeout") and the applied value rides back on the result.
        fake_modal.responder = lambda argv, timeout: ('', '', 0)
        async with ModalSandboxSession() as session:
            result = await session.exec(['x'], timeout=0.2)
            assert fake_modal.sandboxes[0].exec_calls[-1].timeout == 1
            assert result.applied_timeout == 1

    async def test_timeout_none_stays_unbounded(self, fake_modal: FakeModal) -> None:
        fake_modal.responder = lambda argv, timeout: ('', '', 0)
        async with ModalSandboxSession() as session:
            result = await session.exec(['x'])
            assert fake_modal.sandboxes[0].exec_calls[-1].timeout is None
            assert result.applied_timeout is None

    @pytest.mark.parametrize('timeout', [0, -1, float('nan'), float('inf')])
    async def test_invalid_timeout_rejected(self, fake_modal: FakeModal, timeout: float) -> None:
        async with ModalSandboxSession() as session:
            with pytest.raises(ValueError, match='timeout must be a positive finite number'):
                await session.exec(['x'], timeout=timeout)

    @pytest.mark.parametrize('max_output_bytes', [0, -1, True])
    async def test_invalid_output_limit_rejected(self, fake_modal: FakeModal, max_output_bytes: int) -> None:
        async with ModalSandboxSession() as session:
            with pytest.raises(ValueError, match='max_output_bytes must be a positive integer'):
                await session.exec(['x'], max_output_bytes=max_output_bytes)

    async def test_exec_error_wrapped(self, fake_modal: FakeModal) -> None:
        def boom(argv: list[str], timeout: int | None) -> tuple[str, str, int]:
            raise fake_modal.error_type('exec boom')

        fake_modal.responder = boom
        async with ModalSandboxSession() as session:
            with pytest.raises(ModalSandboxError, match='Command could not run in the sandbox: exec boom'):
                await session.exec(['whatever'])

    async def test_sentinel_without_deadline_is_not_a_timeout(self, fake_modal: FakeModal) -> None:
        # -1 is only Modal's timeout sentinel when we set a deadline. With no deadline, a -1
        # from some other cause must not be mislabelled as a timeout.
        fake_modal.responder = lambda argv, timeout: ('', '', -1)
        async with ModalSandboxSession() as session:
            result = await session.exec(['x'])
            assert result.returncode == -1
            assert result.timed_out is False

    async def test_bounded_output_keeps_the_tail(self, fake_modal: FakeModal) -> None:
        # A flood of output must not balloon client memory: with a cap only the last bytes are
        # retained. One-char chunks make the drop loop run per character.
        fake_modal.output_chunk_size = 1
        fake_modal.responder = lambda argv, timeout: ('0123456789', 'ABCDEFGHIJ', 0)
        async with ModalSandboxSession() as session:
            result = await session.exec(['big'], timeout=5, max_output_bytes=4)
            # The end of each stream survives -- that is where errors and exit status sit.
            assert result.stdout == '6789'
            assert result.stderr == 'GHIJ'

    async def test_bounded_output_keeps_exact_tail_from_one_large_chunk(self, fake_modal: FakeModal) -> None:
        fake_modal.responder = lambda argv, timeout: ('0123456789', 'ABCDEFGHIJ', 0)
        async with ModalSandboxSession() as session:
            result = await session.exec(['big'], timeout=5, max_output_bytes=4)
        assert result.stdout == '6789'
        assert result.stderr == 'GHIJ'

    async def test_bounded_output_under_cap_is_whole(self, fake_modal: FakeModal) -> None:
        fake_modal.output_chunk_size = 1
        fake_modal.responder = lambda argv, timeout: ('short', '', 0)
        async with ModalSandboxSession() as session:
            result = await session.exec(['x'], max_output_bytes=100)
            assert result.stdout == 'short'

    async def test_invalid_utf8_output_uses_replacement_characters(self, fake_modal: FakeModal) -> None:
        fake_modal.responder = lambda argv, timeout: (b'\xff\xfeok', b'', 0)
        async with ModalSandboxSession() as session:
            result = await session.exec(['binary'])
            assert result.stdout == '\ufffd\ufffdok'

    async def test_incomplete_utf8_tail_uses_replacement_character(self, fake_modal: FakeModal) -> None:
        fake_modal.responder = lambda argv, timeout: (b'ok\xe2\x82', b'', 0)
        async with ModalSandboxSession() as session:
            result = await session.exec(['partial'])
            assert result.stdout == 'ok\ufffd'

    async def test_bounded_output_decodes_split_utf8_chunks(self, fake_modal: FakeModal) -> None:
        fake_modal.output_chunk_size = 1
        fake_modal.responder = lambda argv, timeout: (b'\xe2\x82\xacOK', b'', 0)
        async with ModalSandboxSession() as session:
            result = await session.exec(['unicode'], max_output_bytes=16)
            assert result.stdout == '\u20acOK'

    async def test_exec_on_terminated_sandbox_is_unavailable(self, fake_modal: FakeModal) -> None:
        # The sandbox died (e.g. hit its lifetime): exec against it is terminal, so the run can
        # end with an actionable message instead of retrying against a dead sandbox.
        def gone(argv: list[str], timeout: int | None) -> tuple[str, str, int]:
            raise fake_modal.sandbox_terminated_type('sandbox terminated')

        fake_modal.responder = gone
        async with ModalSandboxSession(sandbox_timeout=120) as session:
            with pytest.raises(ModalSandboxUnavailableError, match='no longer running'):
                await session.exec(['whatever'])

    async def test_exec_auth_error_is_terminal(self, fake_modal: FakeModal) -> None:
        def denied(argv: list[str], timeout: int | None) -> tuple[str, str, int]:
            raise fake_modal.auth_type('token expired')

        fake_modal.responder = denied
        async with ModalSandboxSession() as session:
            with pytest.raises(ModalSandboxTerminalError, match='Modal rejected the credentials'):
                await session.exec(['whatever'])

    @pytest.mark.parametrize('failure_point', ['stdout', 'stderr', 'wait'])
    @pytest.mark.parametrize('max_output_bytes', [None, 100], ids=['unbounded', 'bounded'])
    async def test_process_stream_error_is_wrapped(
        self, fake_modal: FakeModal, failure_point: str, max_output_bytes: int | None
    ) -> None:
        setattr(fake_modal, f'{failure_point}_error', fake_modal.error_type(f'{failure_point} failed'))
        async with ModalSandboxSession() as session:
            with pytest.raises(ModalSandboxError, match=f'Command could not run.*{failure_point} failed'):
                await session.exec(['whatever'], max_output_bytes=max_output_bytes)


class TestFilesystem:
    async def test_write_then_read_round_trips(self, fake_modal: FakeModal) -> None:
        async with ModalSandboxSession() as session:
            await session.write_bytes('/work/app/main.py', b'print(1)\n')
            assert await session.read_bytes('/work/app/main.py') == b'print(1)\n'
        sandbox = fake_modal.sandboxes[0]
        # Parent directories are created before the write.
        assert sandbox.made_dirs == ['/work/app']

    async def test_existing_parent_directory_is_write_success(self, fake_modal: FakeModal) -> None:
        async with ModalSandboxSession() as session:
            fake_modal.sandboxes[0].make_directory_error = fake_modal.path_already_exists_type('already exists')
            await session.write_bytes('/work/app/main.py', b'print(1)\n')
            assert await session.read_bytes('/work/app/main.py') == b'print(1)\n'

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

    async def test_missing_file_is_recoverable_not_terminal(self, fake_modal: FakeModal) -> None:
        # A missing *file* is the model's mistake to fix (a retry), not a dead sandbox: it must
        # stay a plain ModalSandboxError so the toolset retries rather than ending the run.
        async with ModalSandboxSession() as session:
            fake_modal.sandboxes[0].fs_error = fake_modal.file_not_found_type('No such file: /x')
            with pytest.raises(ModalSandboxError) as exc:
                await session.read_bytes('/x')
            assert not isinstance(exc.value, ModalSandboxTerminalError)

    async def test_missing_sandbox_during_read_is_terminal(self, fake_modal: FakeModal) -> None:
        # A missing *sandbox* (not a missing file) is terminal: the whole sandbox is gone.
        async with ModalSandboxSession() as session:
            fake_modal.sandboxes[0].fs_error = fake_modal.unavailable_type('sandbox not found')
            fake_modal.sandboxes[0].poll_result = 0
            with pytest.raises(ModalSandboxUnavailableError, match='no longer running'):
                await session.read_bytes('/x')

    async def test_wrapped_auth_failure_during_read_is_terminal(self, fake_modal: FakeModal) -> None:
        async with ModalSandboxSession() as session:
            fake_modal.sandboxes[0].fs_error = fake_modal.filesystem_error_type('filesystem failed')
            fake_modal.sandboxes[0].poll_error = fake_modal.auth_type('bad token')
            with pytest.raises(ModalSandboxTerminalError, match='Modal rejected the credentials'):
                await session.read_bytes('/x')

    async def test_direct_auth_failure_during_read_is_terminal(self, fake_modal: FakeModal) -> None:
        async with ModalSandboxSession() as session:
            fake_modal.sandboxes[0].fs_error = fake_modal.auth_type('bad token')
            with pytest.raises(ModalSandboxTerminalError, match='Modal rejected the credentials'):
                await session.read_bytes('/x')

    async def test_poll_unavailable_after_filesystem_error_is_terminal(self, fake_modal: FakeModal) -> None:
        async with ModalSandboxSession() as session:
            fake_modal.sandboxes[0].fs_error = fake_modal.filesystem_error_type('filesystem failed')
            fake_modal.sandboxes[0].poll_error = fake_modal.unavailable_type('sandbox gone')
            with pytest.raises(ModalSandboxUnavailableError, match='no longer running'):
                await session.read_bytes('/x')

    async def test_poll_failure_preserves_original_filesystem_error(self, fake_modal: FakeModal) -> None:
        async with ModalSandboxSession() as session:
            fake_modal.sandboxes[0].fs_error = fake_modal.filesystem_error_type('filesystem failed')
            fake_modal.sandboxes[0].poll_error = fake_modal.error_type('poll failed')
            with pytest.raises(ModalSandboxError, match='filesystem failed'):
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

    async def test_absolute_path_preserves_parent_segments(self, fake_modal: FakeModal) -> None:
        # Do not normalize before the remote filesystem resolves symlinks: /work/.. can
        # differ from / when /work itself is a symlink.
        async with ModalSandboxSession() as session:
            await session.write_bytes('/work/../data/f.txt', b'x')
        assert '/work/../data/f.txt' in fake_modal.sandboxes[0].files
        assert fake_modal.sandboxes[0].made_dirs == ['/work/../data']

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

    async def test_concurrent_relative_paths_probe_pwd_once(self, fake_modal: FakeModal) -> None:
        # A batch of concurrent tool calls resolving relative paths must share one `pwd` probe,
        # not fire one each: the probe is single-flighted behind a lock.
        fake_modal.responder = lambda argv, timeout: ('/work\n', '', 0)
        async with ModalSandboxSession() as session:
            async with anyio.create_task_group() as tg:
                tg.start_soon(session.write_bytes, 'a.txt', b'x')
                tg.start_soon(session.write_bytes, 'b.txt', b'y')
        pwd_calls = [c for c in fake_modal.sandboxes[0].exec_calls if c.argv == ['sh', '-c', 'pwd']]
        assert len(pwd_calls) == 1
        assert '/work/a.txt' in fake_modal.sandboxes[0].files
        assert '/work/b.txt' in fake_modal.sandboxes[0].files

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
