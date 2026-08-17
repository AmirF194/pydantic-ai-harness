"""A Modal sandbox behind Pydantic AI's [`SandboxBackend`][pydantic_ai.sandboxes.SandboxBackend] protocol.

This is the mechanism layer: every Modal-specific operation (create, connect, exec, file
access, working-directory discovery, teardown) lives here, behind the protocol the rest of
Pydantic AI already speaks. The capability in `_capability.py` owns the lifecycle and the
toolset in `_toolset.py` only ever sees `ctx.sandbox`.
"""

from __future__ import annotations

import asyncio
import functools
import itertools
import math
import posixpath
import time
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import anyio
import anyio.lowlevel
from typing_extensions import Self

if TYPE_CHECKING:
    import modal
    import modal.container_process
    import modal.io_streams
    from pydantic_ai.sandboxes import (
        SandboxBackend,
        SandboxCommand,
        SandboxFilesystem,
        SandboxProcess,
        SupportsFilesystem,
        SupportsStart,
        SupportsStream,
    )

__all__ = (
    'ModalSandboxAuthError',
    'ModalSandboxBackend',
    'ModalSandboxCommandTimeoutError',
    'ModalSandboxError',
    'ModalSandboxTerminalError',
    'ModalSandboxUnavailableError',
)

# Defaults shared by `ModalSandboxBackend.create` and the `ModalSandbox` capability (which
# imports them), so the two cannot drift: a setting is "left at its default" iff it equals
# the constant here.
DEFAULT_IMAGE = 'python:3.12-slim'
DEFAULT_APP_NAME = 'pydantic-ai-harness'
DEFAULT_SANDBOX_TIMEOUT = 300

PROVIDER = 'modal'
"""The `provider` half of a `SandboxRef` this backend answers to."""

_Stream = Literal['stdout', 'stderr']

_MISSING_MODAL = (
    'The \'modal\' package is required for ModalSandbox. Install it with `uv add "pydantic-ai-harness[modal]"`.'
)

_AUTH_MESSAGE = 'Modal rejected the credentials. Set MODAL_TOKEN_ID / MODAL_TOKEN_SECRET or run `modal token new`.'

# Bound the sandbox-create RPCs (app lookup + create) so a wedged control plane cannot make
# creation uncancellable. Creation is shielded so a normal cancellation cannot orphan a
# just-created sandbox (see `create`), but a shield with no deadline would hang forever if
# the RPC never returns. Generous, since its only job is to break a true hang: a cold start is
# well under this. If it fires after Modal already provisioned the sandbox, that sandbox is
# reaped server-side by its own `sandbox_timeout` -- the same backstop as any create leak.
_CREATE_TIMEOUT = 120

# Teardown runs shielded from cancellation, so an unreachable Modal control plane could
# otherwise hang the caller forever. Bound each teardown RPC so a stalled terminate/detach
# gives up rather than wedging the process; an owned sandbox is still reaped server-side by
# its own `sandbox_timeout`.
_TEARDOWN_TIMEOUT = 30

# Modal exposes no per-command kill, so every command carries a deadline -- including the
# internal `pwd` probe behind `working_dir()`.
_INTERNAL_EXEC_TIMEOUT = 10

# What Modal's client reports when its own copy of a command's deadline ends the wait.
_CLIENT_DEADLINE_EXIT = -1
# The exit status of a process killed by SIGKILL (128 + 9): what the server side of the same
# deadline looks like when its kill lands before the client's deadline fires.
_SIGKILL_EXIT = 137


class ModalSandboxError(RuntimeError):
    """Base class for failures reported by the Modal sandbox integration.

    The toolset turns direct instances into `ModelRetry`. Terminal subclasses
    propagate because retrying cannot restore a missing sandbox or credentials.
    """


class ModalSandboxTerminalError(ModalSandboxError):
    """A sandbox failure that retrying cannot fix, so the run should end, not loop.

    The toolset lets this propagate out of the tool (ending the run) instead of
    turning it into a `ModelRetry`: re-issuing the command would hit the same wall.
    Raised as `ModalSandboxUnavailableError` for a sandbox that no longer exists
    and `ModalSandboxAuthError` for rejected credentials.
    """


class ModalSandboxUnavailableError(ModalSandboxTerminalError):
    """The sandbox no longer exists: terminated, or expired at its `sandbox_timeout`.

    Every later command against it would fail the same way, so it is terminal. For an
    owned sandbox this is what a run outliving the sandbox lifetime looks like; raise
    `sandbox_timeout` (or shorten the work) if runs legitimately need longer.
    """


class ModalSandboxAuthError(ModalSandboxTerminalError):
    """Modal rejected the credentials, so no sandbox operation can succeed.

    Fixing this is an operator action (configure credentials), not something a
    retry or a new run can do, which is why it is terminal.
    """


class ModalSandboxCommandTimeoutError(TimeoutError):
    """A command hit the deadline it was started with and Modal killed it.

    Derives from the builtin `TimeoutError` because that is what the sandbox protocol
    promises for an expired `timeout=`. It carries the output the command produced before
    the kill, which the protocol's result-or-raise shape has nowhere else to put.
    """

    def __init__(self, message: str, *, stdout: str, stderr: str, timeout: int) -> None:
        super().__init__(message)
        self.stdout = stdout
        """Standard output produced before the deadline kill."""
        self.stderr = stderr
        """Standard error produced before the deadline kill."""
        self.timeout = timeout
        """The whole-second deadline Modal actually enforced."""


@dataclass(frozen=True)
class _ModalResult:
    exit_code: int
    stdout: str
    stderr: str


@dataclass(frozen=True)
class _ModalOutputChunk:
    stream: _Stream
    data: str


@dataclass(frozen=True, kw_only=True)
class _ModalFileEntry:
    """`SandboxFileEntry` carrier for Modal's `FileInfo`.

    Declared here rather than reused: Pydantic AI's own `FileEntry` carrier is internal to
    `pydantic_ai.sandboxes.protocol`, so a third-party backend brings its own.
    """

    name: str
    path: str
    is_dir: bool
    size: int | None


def _unavailable_sandbox_exc_types() -> tuple[type[BaseException], ...]:
    """Modal exception types that mean the sandbox itself no longer exists -- a terminal condition.

    A missing *file* is a different, recoverable error (translated to the builtin
    `FileNotFoundError`); these are the ones that say the whole sandbox is unusable.
    """
    import modal

    return (
        modal.exception.NotFoundError,
        modal.exception.SandboxTerminatedError,
        modal.exception.SandboxTimeoutError,
    )


def _command_argv(command: SandboxCommand, shell: bool) -> Sequence[str]:
    if shell:
        if not isinstance(command, str):
            raise TypeError('an argv sequence cannot be combined with shell=True; pass a single command string')
        # Modal executes argv and never a shell string, so shell interpretation is requested
        # explicitly. `/bin/sh` rather than bash: it is the one shell every sandbox image carries.
        return ['/bin/sh', '-c', command]
    if isinstance(command, str):
        raise TypeError('a string command requires shell=True; pass an argv sequence otherwise')
    if not command:
        raise TypeError('a command needs at least the program to run; the argv sequence is empty')
    return command


class _ModalProcess:
    """A command running inside a Modal sandbox, as returned by `ModalSandboxBackend.start`."""

    def __init__(
        self,
        process: modal.container_process.ContainerProcess[bytes],
        *,
        classify: Callable[[Exception, str], Awaitable[ModalSandboxError]],
        deadline: int | None,
        started: float,
    ) -> None:
        self._process = process
        self._classify = classify
        self._deadline = deadline
        self._started = started
        self._streaming = False
        self._lock = anyio.Lock()
        self._outcome: _ModalResult | Exception | None = None

    @property
    def pid(self) -> int | None:
        # Modal identifies a command by an exec id of its own and never reports the container's
        # OS process id, so there is no honest number to give here.
        return None

    def stream(self) -> AsyncGenerator[_ModalOutputChunk]:
        """Iterate over the command's output as Modal produces it.

        Chunks from the two streams are interleaved in arrival order and decoded per chunk, so
        a multi-byte character split across two transport chunks decodes to replacement
        characters here (`wait()` decodes the whole stream and does not have that seam).
        Modal's readers have a single consumer, so a second `stream()` call raises. Nothing
        here is load-bearing for `wait()`, which asks Modal for the output in full.

        Merging two readers needs concurrency inside an async generator, which asyncio
        primitives express safely and anyio's task groups do not, so this is the one part of
        the backend that requires asyncio specifically. Modal's SDK requires it anyway.
        """
        if self._streaming:
            raise ModalSandboxError(
                "Modal's output streams have a single consumer: `stream()` can only be iterated "
                'once per started command.'
            )
        self._streaming = True
        return self._stream()

    async def _stream(self) -> AsyncGenerator[_ModalOutputChunk]:
        iterators: dict[_Stream, AsyncIterator[bytes]] = {
            'stdout': aiter(self._process.stdout),
            'stderr': aiter(self._process.stderr),
        }
        arrival = itertools.count()

        async def read_one(name: _Stream) -> tuple[int, _Stream, bytes | None]:
            chunk = await anext(iterators[name], None)
            # Stamped where the chunk lands rather than where it is collected: a wake-up that
            # finds both streams ready hands them back as an unordered set, and this is what
            # keeps the merge in the order Modal produced the output.
            return next(arrival), name, chunk

        pending = {asyncio.ensure_future(read_one(name)) for name in iterators}
        try:
            while pending:
                done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
                try:
                    chunks = sorted(task.result() for task in done)
                except Exception as error:
                    # Same taxonomy as `wait()`: a read that fails mid-stream may be a dead
                    # sandbox rather than the transport blip it looks like.
                    raise await self._classify(error, 'Could not read the command output') from error
                for _, name, chunk in chunks:
                    if chunk is None:  # that stream reached EOF and is not re-armed
                        continue
                    pending.add(asyncio.ensure_future(read_one(name)))
                    yield _ModalOutputChunk(stream=name, data=chunk.decode('utf-8', errors='replace'))
        finally:
            # Reaped, not just cancelled: an abandoned read would otherwise keep retrying
            # against the worker long after the consumer walked away.
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)

    async def wait(self) -> _ModalResult:
        """Wait for the command and return its result, the same one on every call."""
        # The timeout verdict below can only be reached once, so the first call's verdict is
        # the command's verdict: caching it is what makes repeated and concurrent waits agree.
        async with self._lock:
            if self._outcome is None:
                try:
                    self._outcome = await self._settle()
                except Exception as error:
                    self._outcome = error
        if isinstance(self._outcome, Exception):
            raise self._outcome
        return self._outcome

    async def _settle(self) -> _ModalResult:
        output: dict[_Stream, str] = {}
        exit_code: int | None = None
        elapsed = 0.0
        failure: Exception | None = None

        # The readers catch Exception, not just modal's Error: stream iteration can surface
        # raw transport failures (grpclib stream errors, a ValueError on an empty message)
        # that are not modal.exception.Error, and an unmapped exception here would abort the
        # whole agent run instead of becoming a typed sandbox error.
        async def read(name: _Stream, reader: modal.io_streams.StreamReader[bytes]) -> None:
            nonlocal failure
            try:
                # Every `read()` opens a stream of its own at offset zero and returns the
                # command's whole output, so what `stream()` consumed changes nothing here:
                # the output is accumulated exactly once however the two are interleaved.
                output[name] = (await reader.read.aio()).decode('utf-8', errors='replace')
            except Exception as error:
                failure = error
                task_group.cancel_scope.cancel()

        async def collect_exit_code() -> None:
            nonlocal exit_code, elapsed, failure
            try:
                exit_code = await self._process.wait.aio()
                # Stamped where the exit code lands rather than where the last of the output
                # does: a command that exited long before its output finished streaming must
                # not be read as one the deadline killed.
                elapsed = time.monotonic() - self._started
            except Exception as error:
                failure = error
                task_group.cancel_scope.cancel()

        async with anyio.create_task_group() as task_group:
            task_group.start_soon(read, 'stdout', self._process.stdout)
            task_group.start_soon(read, 'stderr', self._process.stderr)
            task_group.start_soon(collect_exit_code)

        if failure is not None:
            raise await self._classify(
                failure, 'Could not read the command result (the command may still run until its deadline)'
            ) from failure
        if exit_code is None:  # pragma: no cover - the task group sets all three together
            raise ModalSandboxError('Modal command result was incomplete.')
        if self._timed_out(exit_code, elapsed):
            assert self._deadline is not None
            raise ModalSandboxCommandTimeoutError(
                f'Command timed out after {self._deadline} seconds and was killed.',
                stdout=output['stdout'],
                stderr=output['stderr'],
                timeout=self._deadline,
            )
        return _ModalResult(exit_code=exit_code, stdout=output['stdout'], stderr=output['stderr'])

    def _timed_out(self, exit_code: int, elapsed: float) -> bool:
        if self._deadline is None:
            return False
        if exit_code == _CLIENT_DEADLINE_EXIT:
            return True
        # A command can exit 137 on its own account (an OOM kill, a `kill -9` it asked for), so
        # that exit only means "the deadline killed it" once the whole window has elapsed. The
        # window is measured from before the exec call, which makes it a superset of the one
        # Modal's own timer runs -- the platform starts counting when the command starts, inside
        # that round trip -- so a deadline kill always lands inside it and an earlier exit does not.
        return exit_code == _SIGKILL_EXIT and elapsed >= self._deadline

    async def kill(self) -> None:
        raise NotImplementedError(
            'Modal exposes no way to kill an individual command; start it with `timeout=` so the '
            'platform kills it at the deadline, or terminate the whole sandbox.'
        )


class _ModalFilesystem:
    """Modal's sandbox filesystem API behind the `SandboxFilesystem` protocol."""

    def __init__(self, sandbox: modal.Sandbox, classify: Callable[[Exception], Awaitable[ModalSandboxError]]) -> None:
        self._sandbox = sandbox
        self._classify = classify

    @asynccontextmanager
    async def _translated(self, path: str) -> AsyncGenerator[None]:
        """Map Modal's filesystem exceptions onto the ones the protocol promises.

        A missing path is the builtin `FileNotFoundError` every backend raises. Everything
        else may be masking a dead sandbox or rejected credentials, so it goes through the
        poll-based classification.
        """
        import modal

        try:
            yield
        except modal.exception.SandboxFilesystemNotFoundError as e:
            raise FileNotFoundError(f'No such file or directory in the Modal sandbox: {path!r}') from e
        except modal.exception.Error as e:
            raise await self._classify(e) from e

    async def read_bytes(self, path: str) -> bytes:
        async with self._translated(path):
            return await self._sandbox.filesystem.read_bytes.aio(path)

    async def write_bytes(self, path: str, data: bytes) -> None:
        # Modal takes the data first, creates missing parents, and replaces existing contents.
        async with self._translated(path):
            await self._sandbox.filesystem.write_bytes.aio(data, path)

    async def stat(self, path: str) -> _ModalFileEntry:
        async with self._translated(path):
            return _file_entry(await self._sandbox.filesystem.stat.aio(path), path)

    async def list_dir(self, path: str) -> Sequence[_ModalFileEntry]:
        async with self._translated(path):
            entries = await self._sandbox.filesystem.list_files.aio(path)
        # `name` is the entry's base name, so joining it onto the directory we asked about
        # gives the absolute path the protocol promises.
        return [_file_entry(entry, posixpath.join(path, entry.name)) for entry in entries]

    async def make_dir(self, path: str) -> None:
        # Modal's default `create_parents=True` is `mkdir -p`: missing parents are created and
        # an existing directory is not an error.
        async with self._translated(path):
            await self._sandbox.filesystem.make_directory.aio(path)

    async def remove(self, path: str) -> None:
        # `recursive=True` is what lets this remove a non-empty directory; on a file it changes
        # nothing, so one call covers both halves of the protocol's `remove`.
        async with self._translated(path):
            await self._sandbox.filesystem.remove.aio(path, recursive=True)

    async def exists(self, path: str) -> bool:
        import modal

        # Modal has no existence check of its own, and `stat` is the cheapest question that
        # answers one. Modal splits "there is nothing at that path" in two: a missing
        # component, and a non-leaf component that is a file rather than a directory
        # (`/tmp/notes.txt/deeper`). Both are answers here, which is why this calls Modal
        # directly rather than through `stat`, whose translation only knows the first.
        try:
            await self._sandbox.filesystem.stat.aio(path)
        except (
            modal.exception.SandboxFilesystemNotFoundError,
            modal.exception.SandboxFilesystemNotADirectoryError,
        ):
            return False
        except modal.exception.Error as e:
            raise await self._classify(e) from e
        return True


def _file_entry(entry: modal.types.FileInfo, path: str) -> _ModalFileEntry:
    is_dir = entry.is_dir()
    # A directory's reported size is an implementation detail of the underlying filesystem
    # rather than a content length, so report none for it, like the built-in backends.
    return _ModalFileEntry(name=entry.name, path=path, is_dir=is_dir, size=None if is_dir else entry.size)


class ModalSandboxBackend:
    """A [Modal](https://modal.com) sandbox as a Pydantic AI [`SandboxBackend`][pydantic_ai.sandboxes.SandboxBackend].

    Commands and file operations run inside a Modal container, so the host is never exposed.
    Build one with [`create`][pydantic_ai_harness.modal_sandbox.ModalSandboxBackend.create] or
    attach to an existing environment with
    [`connect`][pydantic_ai_harness.modal_sandbox.ModalSandboxBackend.connect]; the
    `ModalSandbox` capability does both for you.

    Both process opt-ins are implemented: background commands (`SupportsStart`) and live output
    (`SupportsStream`, for one consumer per command, which is all Modal's readers hand out).
    What Modal has no API for is killing a single command, so `kill()` raises
    `NotImplementedError` and `timeout=` -- which Modal enforces itself -- is how a command is
    bounded. For the same reason, cancelling a `run()` stops the wait but not the command: it
    runs on until its deadline, or until the sandbox is terminated. Modal takes whole seconds,
    so a fractional `timeout=` rounds up to the deadline actually applied.

    Deliberately no base class: it conforms to the protocol structurally, like any third-party
    backend would.

    Args:
        sandbox: A live `modal.Sandbox`. Whoever created it owns terminating it.
        working_dir: The sandbox's working directory when it is already known (`create` passes
            its own `workdir`); otherwise it is discovered with `pwd` on first use.
        sandbox_timeout: The lifetime an owned sandbox was created with, used to explain an
            expired sandbox. `None` for a sandbox this process did not create.
    """

    provider = PROVIDER

    def __init__(
        self,
        sandbox: modal.Sandbox,
        *,
        working_dir: str | None = None,
        sandbox_timeout: int | None = None,
    ) -> None:
        self.sandbox = sandbox
        """The underlying `modal.Sandbox`, for provider-specific functionality."""
        self.fs = _ModalFilesystem(sandbox, self._ambiguous_error)
        self._working_dir = working_dir
        self._sandbox_timeout = sandbox_timeout

    @property
    def sandbox_id(self) -> str:
        return self.sandbox.object_id

    @classmethod
    async def create(
        cls,
        *,
        image: str = DEFAULT_IMAGE,
        app_name: str = DEFAULT_APP_NAME,
        create_app_if_missing: bool = True,
        sandbox_timeout: int = DEFAULT_SANDBOX_TIMEOUT,
        workdir: str | None = None,
        env: Mapping[str, str] | None = None,
    ) -> Self:
        """Provision a fresh Modal sandbox. The caller owns terminating it with `close`.

        Args:
            image: Registry tag the sandbox runs (e.g. `python:3.12-slim`).
            app_name: Modal app the sandbox is created under.
            create_app_if_missing: Create the Modal app when it does not exist yet.
            sandbox_timeout: How long Modal keeps the sandbox alive, in seconds.
            workdir: Absolute directory commands start in; Modal's default when `None`.
            env: Environment variables set for the whole sandbox.
        """
        try:
            import modal
        except ImportError as e:
            raise ModalSandboxError(_MISSING_MODAL) from e

        sandbox: modal.Sandbox | None = None
        try:
            # Shield creation so a cancellation arriving mid-create cannot drop the sandbox
            # handle before we return it. Without this, a sandbox created server-side would be
            # orphaned (reaped only by its own `sandbox_timeout`) because the caller would have
            # no handle to terminate. The inner deadline bounds the shielded RPC so a wedged
            # control plane cannot make this uncancellable. The shield holds for anyio-scope
            # cancellation; a raw `asyncio.Task.cancel()` can still interrupt it, in which case
            # the server-side `sandbox_timeout` is the backstop.
            with anyio.CancelScope(shield=True):
                with anyio.move_on_after(_CREATE_TIMEOUT):
                    app = await modal.App.lookup.aio(app_name, create_if_missing=create_app_if_missing)
                    # `from_registry` builds the image spec locally (no network), so it has no
                    # `.aio` variant. Its typing uses an untyped `**kwargs`, so pyright flags it.
                    built = modal.Image.from_registry(image)  # pyright: ignore[reportUnknownMemberType]
                    # Modal types env values as `str | None` (None unsets); widen ours to match,
                    # since dict is invariant in its value type.
                    variables: dict[str, str | None] | None = dict(env) if env is not None else None
                    # `create.aio` is typed with a partially-`Any` coroutine return.
                    sandbox = await modal.Sandbox.create.aio(  # pyright: ignore[reportUnknownMemberType]
                        app=app, image=built, timeout=sandbox_timeout, workdir=workdir, env=variables
                    )
        except modal.exception.AuthError as e:
            raise ModalSandboxAuthError(_AUTH_MESSAGE) from e
        except modal.exception.Error as e:
            raise ModalSandboxError(f'Could not start Modal sandbox: {e}') from e
        if sandbox is None:
            # The deadline fired: the create RPC never returned. Fail here rather than hand back
            # no sandbox. Anything Modal provisioned before the hang is reaped by its own
            # `sandbox_timeout`, the same backstop as a create leak.
            raise ModalSandboxError(
                f'Modal sandbox creation did not complete within {_CREATE_TIMEOUT}s; '
                'the Modal control plane may be unreachable.'
            )
        backend = cls(sandbox, working_dir=workdir, sandbox_timeout=sandbox_timeout)
        try:
            # If the caller was cancelled during the shielded create, this raises; tear the
            # just-created sandbox down here rather than leaving it for `sandbox_timeout`.
            await anyio.lowlevel.checkpoint()
        except BaseException:
            await backend.close(terminate=True)
            raise
        return backend

    @classmethod
    async def connect(cls, sandbox_id: str) -> Self:
        """Attach to a Modal sandbox that already exists, without taking over its lifecycle.

        Modal hands back a handle for a sandbox it still knows about even after that sandbox
        has terminated, so this polls: a `SandboxRef` must not resolve to a dead environment.
        """
        try:
            import modal
        except ImportError as e:
            raise ModalSandboxError(_MISSING_MODAL) from e

        try:
            sandbox = await modal.Sandbox.from_id.aio(sandbox_id)
            finished = await sandbox.poll.aio()
        except modal.exception.AuthError as e:
            raise ModalSandboxAuthError(_AUTH_MESSAGE) from e
        except _unavailable_sandbox_exc_types() as e:
            raise ModalSandboxUnavailableError(_attached_gone_message(sandbox_id)) from e
        if finished is not None:
            raise ModalSandboxUnavailableError(_attached_gone_message(sandbox_id))
        return cls(sandbox)

    async def close(self, *, terminate: bool) -> None:
        """Release this handle, terminating the sandbox with it when we own its lifetime.

        Runs shielded from cancellation, since a run that is being torn down must still get its
        termination request out, with each RPC bounded so a stalled control plane cannot wedge
        the caller.
        """
        with anyio.CancelScope(shield=True):
            try:
                if terminate:
                    # Bound each RPC independently so a stalled terminate still lets detach run;
                    # a single shared deadline would cancel the detach the moment terminate hung.
                    with anyio.move_on_after(_TEARDOWN_TIMEOUT):
                        try:
                            await self.sandbox.terminate.aio(wait=True)
                        except Exception:
                            # Termination is best-effort. A sandbox that no longer exists is
                            # success, not an error. Any other failure must not replace the
                            # exception unwinding through the caller, and the server-side
                            # `sandbox_timeout` reaps the sandbox regardless.
                            pass
            finally:
                with anyio.move_on_after(_TEARDOWN_TIMEOUT):
                    try:
                        await self.sandbox.detach.aio()  # pyright: ignore[reportUnknownMemberType]
                    except Exception:
                        # Best-effort like terminate: a failed local detach must not replace the
                        # exception unwinding through the caller.
                        pass

    @functools.cached_property
    def _working_dir_lock(self) -> anyio.Lock:
        # `anyio.Lock` binds to the event loop on which it is first used.
        return anyio.Lock()

    async def working_dir(self) -> str:
        """The sandbox's default working directory (absolute POSIX path)."""
        # Modal exposes no API for a running sandbox's working directory -- it is the image's
        # unless `create(workdir=...)` overrode it -- so ask the environment itself. It cannot
        # change, so one answer serves the sandbox's whole life. The lock single-flights the
        # probe: without it a batch of concurrent tool calls would each run their own `pwd`.
        if self._working_dir is None:
            async with self._working_dir_lock:
                if self._working_dir is None:
                    result = await self.run(['pwd'], timeout=_INTERNAL_EXEC_TIMEOUT)
                    printed = result.stdout.strip()
                    # Only an absolute path is an answer. Caching whatever else the environment
                    # printed would hand every later `resolve()` a working directory that is not
                    # one, mis-resolving relative paths with no error.
                    if result.exit_code != 0 or not posixpath.isabs(printed):
                        raise ModalSandboxError(
                            f'Could not determine the working directory of Modal sandbox {self.sandbox_id!r}: '
                            f'`pwd` exited {result.exit_code} and printed {result.stdout!r}. Use absolute paths.'
                        )
                    self._working_dir = printed
        return self._working_dir

    async def run(
        self,
        command: SandboxCommand,
        *,
        shell: bool = False,
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> _ModalResult:
        """Execute a command and wait for it to complete.

        Modal has no per-command kill, so a cancelled `run()` stops the wait but leaves the
        command running until its `timeout` deadline. Pass a finite `timeout` (the toolset
        always does) so an abandoned command cannot run on indefinitely.
        """
        process = await self.start(command, shell=shell, cwd=cwd, env=env, timeout=timeout)
        return await process.wait()

    async def start(
        self,
        command: SandboxCommand,
        *,
        shell: bool = False,
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> _ModalProcess:
        """Start a command without waiting, returning a handle to the running process."""
        import modal

        argv = _command_argv(command, shell)
        if timeout is not None and (not math.isfinite(timeout) or timeout <= 0):
            raise ValueError(f'timeout must be a positive finite number or None, got {timeout!r}.')
        # Modal takes whole seconds and reads a missing deadline as "run until the sandbox
        # dies", so a sub-second deadline rounds up rather than silently becoming unbounded.
        deadline = None if timeout is None else max(1, math.ceil(timeout))
        # Stamped before the call, so the window a `137` is dated against is a superset of the
        # one Modal's own timer runs: the platform starts counting when the command starts,
        # which happens inside this round trip.
        started = time.monotonic()
        variables: dict[str, str | None] | None = dict(env) if env is not None else None
        try:
            # Modal's text mode decodes strictly, so read bytes and decode with replacement:
            # a command printing invalid UTF-8 must not abort the run.
            process = await self.sandbox.exec.aio(*argv, timeout=deadline, workdir=cwd, env=variables, text=False)
        except modal.exception.Error as e:
            raise await self._exec_error(e, 'Command could not run in the sandbox') from e
        return _ModalProcess(process, classify=self._exec_error, deadline=deadline, started=started)

    def _unavailable_message(self) -> str:
        if self._sandbox_timeout is None:
            return _attached_gone_message(self.sandbox_id)
        return (
            'The Modal sandbox is no longer running (it may have reached its '
            f'sandbox_timeout of {self._sandbox_timeout}s, or been terminated). '
            'Start a new run, or raise sandbox_timeout for longer work.'
        )

    async def _exec_error(self, e: Exception, context: str) -> ModalSandboxError:
        """Map an exception raised while running a command.

        A `ConflictError` is ambiguous (first exec on a dead sandbox, or a transient abort), so
        it is classified by polling. A terminated or missing sandbox and rejected credentials
        are terminal. Everything else -- another Modal error or a non-Modal transport failure --
        stays a recoverable `ModalSandboxError`. `context` distinguishes "the command never
        started" from "the result could not be read", so the model is warned when the command
        may still be running.
        """
        import modal

        if isinstance(e, modal.exception.ConflictError):
            return await self._ambiguous_error(e)
        if isinstance(e, modal.exception.AuthError):
            return ModalSandboxAuthError(_AUTH_MESSAGE)
        if isinstance(e, _unavailable_sandbox_exc_types()):
            return ModalSandboxUnavailableError(self._unavailable_message())
        if isinstance(e, modal.exception.Error):
            return ModalSandboxError(f'{context}: {e}')
        return ModalSandboxError(f'{context}: {type(e).__name__}: {e}')

    async def _ambiguous_error(self, e: Exception) -> ModalSandboxError:
        """Classify a Modal error that may mask sandbox death by polling the sandbox.

        Two Modal layers report ambiguously: the filesystem wraps authentication failures as
        `SandboxFilesystemError` and transient control-plane failures as `NotFoundError`, and a
        first exec on a dead sandbox raises `ConflictError` (also used for transient aborts).
        Polling only after an error recovers the distinction without adding a round trip to
        successful operations.
        """
        import modal

        if isinstance(e, modal.exception.AuthError):
            return ModalSandboxAuthError(_AUTH_MESSAGE)
        try:
            finished = await self.sandbox.poll.aio()
        except modal.exception.AuthError:
            return ModalSandboxAuthError(_AUTH_MESSAGE)
        except _unavailable_sandbox_exc_types():
            return ModalSandboxUnavailableError(self._unavailable_message())
        except Exception:
            # The classifying poll can itself fail, including with a raw transport error;
            # fall back to the original error rather than letting the probe abort the run.
            return ModalSandboxError(str(e))
        if finished is not None:
            return ModalSandboxUnavailableError(self._unavailable_message())
        return ModalSandboxError(str(e))


def _attached_gone_message(sandbox_id: str) -> str:
    return (
        f'The Modal sandbox {sandbox_id!r} is no longer running '
        '(it does not exist, was terminated, or expired at its configured lifetime). '
        'Attach to a live sandbox, or create a new one.'
    )


if TYPE_CHECKING:
    # Pins full structural conformance -- signatures included -- which `isinstance` cannot
    # check. `__new__` rather than a call, because neither SDK object can be constructed
    # without a live sandbox behind it; this block never runs.
    _sandbox = modal.Sandbox.__new__(modal.Sandbox)
    _container_process = modal.container_process.ContainerProcess[bytes].__new__(
        modal.container_process.ContainerProcess[bytes]
    )
    _backend = ModalSandboxBackend(_sandbox)
    _process = _ModalProcess(_container_process, classify=_backend._exec_error, deadline=None, started=0.0)  # pyright: ignore[reportPrivateUsage]
    _backend_conforms: SandboxBackend = _backend
    _filesystem_backend_conforms: SupportsFilesystem = _backend
    _start_conforms: SupportsStart = _backend
    _filesystem_conforms: SandboxFilesystem = _backend.fs
    _process_conforms: SandboxProcess = _process
    _stream_conforms: SupportsStream = _process
