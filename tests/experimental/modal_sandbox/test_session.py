"""Tests for ModalSandboxSession."""

from __future__ import annotations

import builtins
import sys

import pytest

from pydantic_ai_harness.experimental.modal_sandbox import ModalSandboxError, ModalSandboxSession

from .fake_modal import FakeModal


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
            assert await session.exec(['whatever'], timeout=5) == ('out', 'err', 7)
            call = fake_modal.sandboxes[0].exec_calls[-1]
            assert call.argv == ['whatever']
            assert call.timeout == 5

    async def test_zero_exit_code(self, fake_modal: FakeModal) -> None:
        fake_modal.responder = lambda argv, timeout: ('done\n', '', 0)
        async with ModalSandboxSession() as session:
            assert await session.exec(['echo', 'done']) == ('done\n', '', 0)
