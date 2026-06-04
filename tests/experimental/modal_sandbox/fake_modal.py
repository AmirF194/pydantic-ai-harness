"""A controllable fake `modal` SDK for ModalSandbox tests.

Tests never reach real Modal: a fake `modal` module is injected into `sys.modules`
(via the `fake_modal` fixture in `conftest.py`), so the lazy `import modal` inside
the session returns it. The fake records calls and lets each test decide what
`exec` returns.
"""

from __future__ import annotations

import types
from collections.abc import Callable
from dataclasses import dataclass

# A responder maps (argv, timeout) to (stdout, stderr, exit_code).
Responder = Callable[[list[str], 'int | None'], 'tuple[str, str, int]']


def _echo_responder(argv: list[str], timeout: int | None) -> tuple[str, str, int]:
    return (' '.join(argv) + '\n', '', 0)


@dataclass
class ExecCall:
    argv: list[str]
    timeout: int | None


class _FakeStream:
    def __init__(self, data: str) -> None:
        self._data = data

    def read(self) -> str:
        return self._data


class _FakeProcess:
    def __init__(self, stdout: str, stderr: str, returncode: int) -> None:
        self.stdout = _FakeStream(stdout)
        self.stderr = _FakeStream(stderr)
        self._returncode = returncode
        self.returncode: int | None = None

    def wait(self) -> int:
        self.returncode = self._returncode
        return self._returncode


class FakeModalError(Exception):
    """Stand-in for `modal.exception.Error`."""


class FakeSandbox:
    def __init__(self, control: FakeModal, object_id: str) -> None:
        self._control = control
        self.object_id = object_id
        self.exec_calls: list[ExecCall] = []
        self.terminated = False
        self.detached = False

    def exec(self, *args: str, timeout: int | None = None, **kwargs: object) -> _FakeProcess:
        argv = list(args)
        self.exec_calls.append(ExecCall(argv=argv, timeout=timeout))
        stdout, stderr, code = self._control.responder(argv, timeout)
        return _FakeProcess(stdout, stderr, code)

    def terminate(self) -> None:
        self.terminated = True

    def detach(self) -> None:
        self.detached = True


class FakeModal:
    """Control surface for the injected fake `modal` module."""

    def __init__(self) -> None:
        self.responder: Responder = _echo_responder
        self.sandboxes: list[FakeSandbox] = []
        self.create_kwargs: list[dict[str, object]] = []
        self.app_lookups: list[dict[str, object]] = []
        self.image_tags: list[str] = []
        self.attach_ids: list[str] = []
        self.create_error: Exception | None = None
        self.module = self._build_module()

    @property
    def error_type(self) -> type[Exception]:
        return FakeModalError

    def _build_module(self) -> types.ModuleType:
        control = self
        module = types.ModuleType('modal')

        class App:
            @staticmethod
            def lookup(name: str, *, create_if_missing: bool = False) -> object:
                control.app_lookups.append({'name': name, 'create_if_missing': create_if_missing})
                return object()

        class Image:
            @staticmethod
            def from_registry(tag: str, **kwargs: object) -> object:
                control.image_tags.append(tag)
                return object()

        class Sandbox:
            @staticmethod
            def create(*args: object, **kwargs: object) -> FakeSandbox:
                if control.create_error is not None:
                    raise control.create_error
                control.create_kwargs.append(kwargs)
                sandbox = FakeSandbox(control, 'sb-owned')
                control.sandboxes.append(sandbox)
                return sandbox

            @staticmethod
            def from_id(sandbox_id: str) -> FakeSandbox:
                control.attach_ids.append(sandbox_id)
                sandbox = FakeSandbox(control, sandbox_id)
                control.sandboxes.append(sandbox)
                return sandbox

        module.App = App  # type: ignore[attr-defined]
        module.Image = Image  # type: ignore[attr-defined]
        module.Sandbox = Sandbox  # type: ignore[attr-defined]
        module.exception = types.SimpleNamespace(Error=FakeModalError)  # type: ignore[attr-defined]
        return module
