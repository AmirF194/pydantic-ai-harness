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
from typing import Any

# A responder maps (argv, timeout) to (stdout, stderr, exit_code).
Responder = Callable[[list[str], 'int | None'], 'tuple[str, str, int]']


def _echo_responder(argv: list[str], timeout: int | None) -> tuple[str, str, int]:
    return (' '.join(argv) + '\n', '', 0)


@dataclass
class ExecCall:
    argv: list[str]
    timeout: int | None


class _AioCallable:
    """Mimics a synchronicity-wrapped Modal method: callable, plus an `.aio` async twin.

    The session only ever calls `.aio`, but exposing both mirrors the real SDK shape.
    """

    def __init__(self, fn: Callable[..., Any]) -> None:
        self._fn = fn

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._fn(*args, **kwargs)

    async def aio(self, *args: Any, **kwargs: Any) -> Any:
        return self._fn(*args, **kwargs)


class _FakeStream:
    def __init__(self, data: str) -> None:
        self._data = data
        self.read = _AioCallable(lambda: self._data)


class _FakeProcess:
    def __init__(self, stdout: str, stderr: str, returncode: int) -> None:
        self.stdout = _FakeStream(stdout)
        self.stderr = _FakeStream(stderr)
        self._returncode = returncode
        self.returncode: int | None = None
        self.wait = _AioCallable(self._wait)

    def _wait(self) -> int:
        self.returncode = self._returncode
        return self._returncode


class FakeModalError(Exception):
    """Stand-in for `modal.exception.Error`."""


class FakeSandboxFilesystemError(FakeModalError):
    """Stand-in for `modal.exception.SandboxFilesystemError`."""


@dataclass
class FileInfo:
    """Minimal stand-in for `modal.sandbox_fs.FileInfo`."""

    name: str
    _is_dir: bool

    def is_dir(self) -> bool:
        return self._is_dir


class _FakeFilesystem:
    """Mirrors `sandbox.filesystem`: an in-memory store the tests can drive and inspect."""

    def __init__(self, sandbox: FakeSandbox) -> None:
        self._sandbox = sandbox
        self.read_bytes = _AioCallable(self._read_bytes)
        self.write_bytes = _AioCallable(self._write_bytes)
        self.make_directory = _AioCallable(self._make_directory)
        self.list_files = _AioCallable(self._list_files)

    def _read_bytes(self, remote_path: str) -> bytes:
        if self._sandbox.fs_error is not None:
            raise self._sandbox.fs_error
        return self._sandbox.files[remote_path]

    def _write_bytes(self, data: bytes, remote_path: str) -> None:
        if self._sandbox.fs_error is not None:
            raise self._sandbox.fs_error
        self._sandbox.files[remote_path] = data

    def _make_directory(self, remote_path: str, *, create_parents: bool = True) -> None:
        if self._sandbox.fs_error is not None:
            raise self._sandbox.fs_error
        self._sandbox.made_dirs.append(remote_path)

    def _list_files(self, remote_path: str) -> list[FileInfo]:
        if self._sandbox.fs_error is not None:
            raise self._sandbox.fs_error
        self._sandbox.list_paths.append(remote_path)
        return self._sandbox.listing


class FakeSandbox:
    def __init__(self, control: FakeModal, object_id: str) -> None:
        self._control = control
        self.object_id = object_id
        self.exec_calls: list[ExecCall] = []
        self.terminated = False
        self.detached = False
        self.terminate_error: Exception | None = None
        self.exec = _AioCallable(self._exec)
        self.terminate = _AioCallable(self._terminate)
        self.detach = _AioCallable(self._detach)
        # Filesystem state the tests read and write.
        self.files: dict[str, bytes] = {}
        self.made_dirs: list[str] = []
        self.list_paths: list[str] = []
        self.listing: list[FileInfo] = []
        self.fs_error: Exception | None = None
        self._filesystem = _FakeFilesystem(self)

    @property
    def filesystem(self) -> _FakeFilesystem:
        return self._filesystem

    def _exec(self, *args: str, timeout: int | None = None, **kwargs: object) -> _FakeProcess:
        argv = list(args)
        self.exec_calls.append(ExecCall(argv=argv, timeout=timeout))
        stdout, stderr, code = self._control.responder(argv, timeout)
        return _FakeProcess(stdout, stderr, code)

    def _terminate(self) -> None:
        if self.terminate_error is not None:
            raise self.terminate_error
        self.terminated = True

    def _detach(self) -> None:
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

    @property
    def filesystem_error_type(self) -> type[Exception]:
        return FakeSandboxFilesystemError

    def _build_module(self) -> types.ModuleType:
        control = self
        module = types.ModuleType('modal')

        def app_lookup(name: str, *, create_if_missing: bool = False) -> object:
            control.app_lookups.append({'name': name, 'create_if_missing': create_if_missing})
            return object()

        def image_from_registry(tag: str, **kwargs: object) -> object:
            control.image_tags.append(tag)
            return object()

        def sandbox_create(*args: object, **kwargs: object) -> FakeSandbox:
            if control.create_error is not None:
                raise control.create_error
            control.create_kwargs.append(kwargs)
            sandbox = FakeSandbox(control, 'sb-owned')
            control.sandboxes.append(sandbox)
            return sandbox

        def sandbox_from_id(sandbox_id: str) -> FakeSandbox:
            control.attach_ids.append(sandbox_id)
            sandbox = FakeSandbox(control, sandbox_id)
            control.sandboxes.append(sandbox)
            return sandbox

        class App:
            lookup = _AioCallable(app_lookup)

        class Image:
            from_registry = staticmethod(image_from_registry)

        class Sandbox:
            create = _AioCallable(sandbox_create)
            from_id = _AioCallable(sandbox_from_id)

        module.App = App  # type: ignore[attr-defined]
        module.Image = Image  # type: ignore[attr-defined]
        module.Sandbox = Sandbox  # type: ignore[attr-defined]
        module.exception = types.SimpleNamespace(  # type: ignore[attr-defined]
            Error=FakeModalError, SandboxFilesystemError=FakeSandboxFilesystemError
        )
        return module
