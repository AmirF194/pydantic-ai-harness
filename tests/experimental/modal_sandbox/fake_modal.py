"""A controllable fake `modal` SDK for ModalSandbox tests.

Tests never reach real Modal: a fake `modal` module is injected into `sys.modules`
(via the `fake_modal` fixture in `conftest.py`), so the lazy `import modal` inside
the session returns it. The fake records calls and lets each test decide what
`exec` returns.
"""

from __future__ import annotations

import posixpath
import types
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import anyio.lowlevel

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

    def __call__(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover - capability only uses `.aio`
        # Modal's callables work sync or async; we mirror both for fidelity, but the
        # capability drives the async `.aio` path exclusively, so this never runs in tests.
        return self._fn(*args, **kwargs)

    async def aio(self, *args: Any, **kwargs: Any) -> Any:
        # A real Modal `.aio` call suspends (it awaits gRPC); yield here so a concurrent
        # batch of tool calls actually interleaves in tests -- otherwise the sync fake would
        # run each call start-to-finish and hide races like a duplicated `pwd` probe.
        await anyio.lowlevel.checkpoint()
        return self._fn(*args, **kwargs)


class _FakeStream:
    """Mimics a Modal exec stdio stream: readable whole via `.read.aio()`, or iterable.

    The session reads unbounded output with `.read.aio()` and bounded output by iterating
    chunks. `chunk_size` controls how the iterable path splits the data, so a test can drive
    the bounded reader's drop logic with more than one chunk. `None` yields the data as a
    single chunk (the realistic "one message" case, and what most tests want).
    """

    def __init__(self, data: str, chunk_size: int | None) -> None:
        self._data = data
        self._chunk_size = chunk_size
        self.read = _AioCallable(lambda: self._data)
        self._pending: list[str] = []
        self._pos = 0

    def __aiter__(self) -> _FakeStream:
        if self._chunk_size is None:
            self._pending = [self._data] if self._data else []
        else:
            self._pending = [self._data[i : i + self._chunk_size] for i in range(0, len(self._data), self._chunk_size)]
        self._pos = 0
        return self

    async def __anext__(self) -> str:
        if self._pos >= len(self._pending):
            raise StopAsyncIteration
        piece = self._pending[self._pos]
        self._pos += 1
        return piece


class _FakeProcess:
    def __init__(self, stdout: str, stderr: str, returncode: int, chunk_size: int | None) -> None:
        self.stdout = _FakeStream(stdout, chunk_size)
        self.stderr = _FakeStream(stderr, chunk_size)
        self._returncode = returncode
        self.returncode: int | None = None
        self.wait = _AioCallable(self._wait)

    def _wait(self) -> int:
        self.returncode = self._returncode
        return self._returncode


class FakeModalError(Exception):
    """Stand-in for `modal.exception.Error`."""


class FakeNotFoundError(FakeModalError):
    """Stand-in for `modal.exception.NotFoundError` (the sandbox itself is missing/gone)."""


class FakeAuthError(FakeModalError):
    """Stand-in for `modal.exception.AuthError`."""


class FakeSandboxTerminatedError(FakeModalError):
    """Stand-in for `modal.exception.SandboxTerminatedError`."""


class FakeSandboxTimeoutError(FakeModalError):
    """Stand-in for `modal.exception.SandboxTimeoutError`."""


class FakeSandboxFilesystemError(FakeModalError):
    """Stand-in for `modal.exception.SandboxFilesystemError`."""


class FakeSandboxFilesystemNotFoundError(FakeSandboxFilesystemError):
    """Stand-in for `modal.exception.SandboxFilesystemNotFoundError` (a missing file, recoverable)."""


@dataclass
class FileInfo:
    """Minimal stand-in for `modal.sandbox_fs.FileInfo`."""

    name: str
    _is_dir: bool
    size: int = 0

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
        self.stat = _AioCallable(self._stat)

    def _read_bytes(self, remote_path: str) -> bytes:
        self._check(remote_path)
        return self._sandbox.files[remote_path]

    def _stat(self, remote_path: str) -> FileInfo:
        self._check(remote_path)
        # Size comes from the stored bytes, or an override the test set for this path.
        size = self._sandbox.stat_sizes.get(remote_path, len(self._sandbox.files.get(remote_path, b'')))
        return FileInfo(remote_path, False, size=size)

    def _write_bytes(self, data: bytes, remote_path: str) -> None:
        self._check(remote_path)
        self._sandbox.files[remote_path] = data

    def _make_directory(self, remote_path: str, *, create_parents: bool = True) -> None:
        self._check(remote_path)
        self._sandbox.made_dirs.append(remote_path)

    def _list_files(self, remote_path: str) -> list[FileInfo]:
        self._check(remote_path)
        self._sandbox.list_paths.append(remote_path)
        return self._sandbox.listing

    def _check(self, remote_path: str) -> None:
        # Real Modal's filesystem API only accepts absolute paths; assert it here so a
        # regression that let a relative path bypass `_resolve` fails in the fake the way it
        # would in prod, instead of silently keying the in-memory store on a relative path.
        assert posixpath.isabs(remote_path), f'Modal filesystem requires an absolute path, got {remote_path!r}'
        if self._sandbox.fs_error is not None:
            raise self._sandbox.fs_error


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
        # Lets a test report a large size for a path without allocating the bytes.
        self.stat_sizes: dict[str, int] = {}
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
        return _FakeProcess(stdout, stderr, code, self._control.output_chunk_size)

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
        self.attach_error: Exception | None = None
        # How the fake splits exec output when the bounded reader iterates it; None yields the
        # whole output as one chunk. A test bounding output sets a small size to force drops.
        self.output_chunk_size: int | None = None
        self.module = self._build_module()

    @property
    def error_type(self) -> type[Exception]:
        return FakeModalError

    @property
    def filesystem_error_type(self) -> type[Exception]:
        return FakeSandboxFilesystemError

    @property
    def unavailable_type(self) -> type[Exception]:
        """A missing/terminated *sandbox* (Modal `NotFoundError`) -- terminal, not retried."""
        return FakeNotFoundError

    @property
    def auth_type(self) -> type[Exception]:
        return FakeAuthError

    @property
    def sandbox_terminated_type(self) -> type[Exception]:
        return FakeSandboxTerminatedError

    @property
    def file_not_found_type(self) -> type[Exception]:
        """A missing *file* (Modal `SandboxFilesystemNotFoundError`) -- recoverable, retried."""
        return FakeSandboxFilesystemNotFoundError

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
            if control.attach_error is not None:
                raise control.attach_error
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
            Error=FakeModalError,
            NotFoundError=FakeNotFoundError,
            AuthError=FakeAuthError,
            SandboxTerminatedError=FakeSandboxTerminatedError,
            SandboxTimeoutError=FakeSandboxTimeoutError,
            SandboxFilesystemError=FakeSandboxFilesystemError,
            SandboxFilesystemNotFoundError=FakeSandboxFilesystemNotFoundError,
        )
        return module
