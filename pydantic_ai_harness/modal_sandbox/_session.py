"""Lifecycle management for a Modal sandbox."""

from __future__ import annotations

import codecs
import math
import posixpath
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import anyio
import anyio.lowlevel
from typing_extensions import Self

if TYPE_CHECKING:
    import modal
    import modal.io_streams


class ModalSandboxError(RuntimeError):
    """Base class for failures reported by the Modal sandbox integration.

    The toolset turns direct instances into `ModelRetry`. Terminal subclasses
    propagate because retrying cannot restore a missing sandbox or credentials.
    """


class ModalSandboxTerminalError(ModalSandboxError):
    """A sandbox failure that retrying cannot fix, so the run should end, not loop.

    The toolset lets this propagate out of the tool (ending the run) instead of
    turning it into a `ModelRetry`: re-issuing the command would hit the same wall.
    Covers a sandbox that no longer exists and rejected credentials.
    """


class ModalSandboxUnavailableError(ModalSandboxTerminalError):
    """The sandbox no longer exists: terminated, or expired at its `sandbox_timeout`.

    Every later command against it would fail the same way, so it is terminal. In
    owned mode this is what a run outliving the sandbox lifetime looks like; raise
    `sandbox_timeout` (or shorten the work) if runs legitimately need longer.
    """


@dataclass(frozen=True)
class ModalSandboxExecResult:
    """The outcome of running a command in the sandbox."""

    stdout: str
    stderr: str
    returncode: int
    timed_out: bool = False
    """True when a deadline was applied and Modal reported its `-1` timeout sentinel."""
    applied_timeout: int | None = None
    """The whole-second deadline Modal enforced for this command, or None if unbounded.

    This is the quantized value actually sent to Modal, not the (possibly fractional)
    timeout the caller requested, so the caller can report the exact deadline.
    """


_MISSING_MODAL = (
    'The \'modal\' package is required for ModalSandbox. Install it with `uv add "pydantic-ai-harness[modal]"`.'
)

_AUTH_MESSAGE = 'Modal rejected the credentials. Set MODAL_TOKEN_ID / MODAL_TOKEN_SECRET or run `modal token new`.'

# Bound the sandbox-create RPCs (app lookup + create) so a wedged control plane cannot make
# the enter uncancellable. Creation is shielded so a normal cancellation cannot orphan a
# just-created sandbox (see `__aenter__`), but a shield with no deadline would hang forever if
# the RPC never returns. Generous, since its only job is to break a true hang: a cold start is
# well under this. If it fires after Modal already provisioned the sandbox, that sandbox is
# reaped server-side by its own `sandbox_timeout` -- the same backstop as any create leak.
_CREATE_TIMEOUT = 120


def _unavailable_sandbox_exc_types() -> tuple[type[BaseException], ...]:
    """Modal exception types that mean the sandbox itself no longer exists -- a terminal condition.

    A missing *file* is a different, recoverable error (`SandboxFilesystemNotFoundError`);
    these are the ones that say the whole sandbox is unusable.
    """
    import modal

    return (
        modal.exception.NotFoundError,
        modal.exception.SandboxTerminatedError,
        modal.exception.SandboxTimeoutError,
    )


# Modal does not currently expose a per-exec kill: a command is reaped by its own
# server-side timeout (or by the whole sandbox being terminated). So every command we run
# carries a deadline, even
# internal ones like the `pwd` used for path resolution, so a cancelled or abandoned run
# cannot leave a command billing indefinitely. This bounds that internal probe.
_INTERNAL_EXEC_TIMEOUT = 10

# Teardown runs shielded from cancellation, so an unreachable Modal control plane could
# otherwise hang the caller forever on exit. Bound each teardown RPC so a stalled
# terminate/detach gives up rather than wedging the process; the owned sandbox is still
# reaped server-side by its own `sandbox_timeout`.
_TEARDOWN_TIMEOUT = 30


# This is the mechanism layer: every Modal-specific operation (create/attach,
# exec, file access, path resolution, lifecycle) is contained here, behind a small
# byte-oriented method surface that the toolset depends on. Keeping it isolated from
# the presentation in `_toolset.py` is what lets the sandbox internals change without
# touching the tools or the capability.
class ModalSandboxSession:
    """Async context manager that owns or attaches to a Modal sandbox.

    In *owned* mode (the default) it creates a fresh sandbox from `image` on
    enter. On exit it requests termination and waits for a bounded period;
    `sandbox_timeout` is the server-side cleanup backstop. In *attach* mode
    (`sandbox_id` set) it looks
    up an existing sandbox and leaves it running on exit, so a sandbox you manage
    elsewhere can be reused across runs.

    Modal's SDK is asyncio-native, so this session drives its `.aio` coroutine API
    directly and requires an asyncio event loop. It authenticates from the standard
    `MODAL_TOKEN_ID` / `MODAL_TOKEN_SECRET` environment variables.

    ```python
    from pydantic_ai_harness.modal_sandbox import ModalSandboxSession

    async with ModalSandboxSession(image='python:3.12-slim') as session:
        result = await session.exec(['echo', 'hello'])
    ```
    """

    def __init__(
        self,
        *,
        image: str = 'python:3.12-slim',
        sandbox_id: str | None = None,
        app_name: str = 'pydantic-ai-harness',
        create_app_if_missing: bool = True,
        sandbox_timeout: int = 300,
        workdir: str | None = None,
        env: Mapping[str, str] | None = None,
    ) -> None:
        if type(sandbox_timeout) is not int or sandbox_timeout <= 0:
            raise ValueError(f'sandbox_timeout must be a positive integer, got {sandbox_timeout!r}.')
        if sandbox_id is not None:
            conflicts = [
                name
                for name, value, default in (
                    ('image', image, 'python:3.12-slim'),
                    ('app_name', app_name, 'pydantic-ai-harness'),
                    ('create_app_if_missing', create_app_if_missing, True),
                    ('sandbox_timeout', sandbox_timeout, 300),
                    ('workdir', workdir, None),
                    ('env', env, None),
                )
                if value != default
            ]
            if conflicts:
                raise ValueError(
                    f'{", ".join(conflicts)} only apply when creating a sandbox, but `sandbox_id` attaches '
                    'to an existing one. Remove them, or drop `sandbox_id` to create a sandbox.'
                )
        self._image = image
        self._sandbox_id = sandbox_id
        self._app_name = app_name
        self._create_app_if_missing = create_app_if_missing
        self._sandbox_timeout = sandbox_timeout
        self._workdir = workdir
        self._env = dict(env) if env is not None else None
        self._sandbox: modal.Sandbox | None = None
        self._cwd: str | None = None
        # Serializes the one-time `pwd` probe so a batch of concurrent tool calls resolving
        # relative paths fires a single probe, not one per call (see `_resolve`).
        self._cwd_lock = anyio.Lock()

    @property
    def sandbox_id(self) -> str | None:
        """The id of the running sandbox, or None when it is not running."""
        if self._sandbox is None:
            return None
        return self._sandbox.object_id

    async def __aenter__(self) -> Self:
        """Create or attach to the sandbox."""
        # Clear any cwd cached from a prior entry: a reused session must resolve relative
        # paths against the new sandbox's tree, not the previous one's.
        self._cwd = None
        try:
            import modal
        except ImportError as e:
            raise ModalSandboxError(_MISSING_MODAL) from e
        try:
            # Shield creation so a cancellation arriving mid-create cannot drop the sandbox
            # handle before we store it. Without this, an owned sandbox created server-side
            # would be orphaned (reaped only by its own `sandbox_timeout`) because `__aexit__`
            # would see no handle to terminate. The cold-start wait is brief, and we honor the
            # cancellation at the checkpoint just below. The inner deadline bounds the shielded
            # RPC so a wedged control plane cannot make this uncancellable (see `_CREATE_TIMEOUT`).
            with anyio.CancelScope(shield=True):
                with anyio.move_on_after(_CREATE_TIMEOUT):
                    self._sandbox = await self._open_sandbox()
        except modal.exception.Error as e:
            raise self._open_error(e) from e
        if self._sandbox is None:
            # The deadline fired: the create RPC never returned. Fail here rather than proceed
            # with no sandbox. Any sandbox Modal provisioned before the hang is reaped by its
            # own `sandbox_timeout`, the same backstop as a create leak.
            raise ModalSandboxError(
                f'Modal sandbox creation did not complete within {_CREATE_TIMEOUT}s; '
                'the Modal control plane may be unreachable.'
            )
        try:
            # If the run was cancelled during the shielded create, this raises; tear the
            # just-created sandbox down here rather than leaving it for `sandbox_timeout`.
            await anyio.lowlevel.checkpoint()
        except BaseException:
            await self.__aexit__(None, None, None)
            raise
        return self

    async def _open_sandbox(self) -> modal.Sandbox:
        """Create an owned sandbox or attach to an existing one."""
        import modal

        if self._sandbox_id is not None:
            sandbox = await modal.Sandbox.from_id.aio(self._sandbox_id)
            if await sandbox.poll.aio() is not None:
                raise ModalSandboxUnavailableError(
                    f'Could not attach to Modal sandbox {self._sandbox_id!r}: it does not exist or has terminated.'
                )
            return sandbox
        app = await modal.App.lookup.aio(self._app_name, create_if_missing=self._create_app_if_missing)
        # `from_registry` builds the image spec locally (no network), so it has no `.aio` variant.
        # Its typing uses an untyped `**kwargs`, so pyright flags the access.
        image = modal.Image.from_registry(self._image)  # pyright: ignore[reportUnknownMemberType]
        # Modal types env values as `str | None` (None unsets); widen our `dict[str, str]` to
        # match, since dict is invariant in its value type.
        env: dict[str, str | None] | None = (
            {key: value for key, value in self._env.items()} if self._env is not None else None
        )
        # `create.aio` is typed with a partially-`Any` coroutine return, so pyright flags the call.
        return await modal.Sandbox.create.aio(  # pyright: ignore[reportUnknownMemberType]
            app=app, image=image, timeout=self._sandbox_timeout, workdir=self._workdir, env=env
        )

    async def __aexit__(self, *args: object) -> None:
        """Request termination when owned, then attempt to detach within bounded waits."""
        sandbox = self._sandbox
        self._sandbox = None
        self._cwd = None
        if sandbox is None:
            return
        owned = self._sandbox_id is None
        # Shield cleanup so cancellation does not interrupt the termination request. Stop a
        # sandbox we created; an attached one keeps running. Attempt detach in `finally` --
        # Modal's recommended cleanup -- even if terminating the owned sandbox fails.
        with anyio.CancelScope(shield=True):
            try:
                if owned:
                    # Bound each RPC independently so a stalled terminate still lets detach run;
                    # a single shared deadline would cancel the detach the moment terminate hung.
                    with anyio.move_on_after(_TEARDOWN_TIMEOUT):
                        try:
                            await sandbox.terminate.aio(wait=True)
                        except _unavailable_sandbox_exc_types():
                            # Terminating a sandbox that no longer exists is success, not an error: an
                            # owned run that outlived its `sandbox_timeout` self-terminates, and a
                            # raise here would mask the ModalSandboxUnavailableError the tool already saw.
                            pass
            finally:
                with anyio.move_on_after(_TEARDOWN_TIMEOUT):
                    await sandbox.detach.aio()  # pyright: ignore[reportUnknownMemberType]

    def _require_sandbox(self) -> modal.Sandbox:
        sandbox = self._sandbox
        if sandbox is None:
            raise ModalSandboxError('The sandbox is not running; use the session as an async context manager.')
        return sandbox

    def _unavailable_message(self) -> str:
        return (
            'The Modal sandbox is no longer running (it may have reached its '
            f'sandbox_timeout of {self._sandbox_timeout}s, or been terminated). '
            'Start a new run, or raise sandbox_timeout for longer work.'
        )

    def _open_error(self, e: modal.exception.Error) -> ModalSandboxError:
        """Map a Modal error raised while creating or attaching to a sandbox.

        Rejected credentials and a missing/terminated sandbox are terminal (the toolset
        never reaches this -- open errors abort the run before tools run -- but the typed
        error still tells a direct session caller what went wrong); anything else is a
        plain create failure.
        """
        import modal

        if isinstance(e, modal.exception.AuthError):
            return ModalSandboxTerminalError(_AUTH_MESSAGE)
        if self._sandbox_id is not None and isinstance(e, _unavailable_sandbox_exc_types()):
            return ModalSandboxUnavailableError(
                f'Could not attach to Modal sandbox {self._sandbox_id!r}: it does not exist or has terminated.'
            )
        return ModalSandboxError(f'Could not start Modal sandbox: {e}')

    def _use_error(self, e: modal.exception.Error, context: str) -> ModalSandboxError:
        """Map a Modal error raised while *using* the sandbox to a ModalSandbox error.

        A terminated or missing sandbox and rejected credentials are terminal -- retrying cannot
        help, so the toolset ends the run instead of prompting the model to try again.
        Everything else (a bad path, a transient sandbox-side failure) stays a recoverable
        `ModalSandboxError`. `context` prefixes the provider message with the operation
        that failed.
        """
        import modal

        if isinstance(e, modal.exception.AuthError):
            return ModalSandboxTerminalError(_AUTH_MESSAGE)
        if isinstance(e, _unavailable_sandbox_exc_types()):
            return ModalSandboxUnavailableError(self._unavailable_message())
        return ModalSandboxError(f'{context}: {e}')

    async def _filesystem_error(self, e: modal.exception.Error) -> ModalSandboxError:
        """Classify filesystem wrapper errors by checking the sandbox itself.

        Modal's filesystem layer can wrap authentication failures as
        `SandboxFilesystemError` and transient control-plane failures as
        `NotFoundError`. Polling only after an error recovers the distinction without
        adding a round trip to successful operations.
        """
        import modal

        if isinstance(e, modal.exception.AuthError):
            return ModalSandboxTerminalError(_AUTH_MESSAGE)
        sandbox = self._require_sandbox()
        try:
            returncode = await sandbox.poll.aio()
        except modal.exception.AuthError:
            return ModalSandboxTerminalError(_AUTH_MESSAGE)
        except _unavailable_sandbox_exc_types():
            return ModalSandboxUnavailableError(self._unavailable_message())
        except modal.exception.Error:
            return ModalSandboxError(str(e))
        if returncode is not None:
            return ModalSandboxUnavailableError(self._unavailable_message())
        return ModalSandboxError(str(e))

    async def _resolve(self, path: str) -> str:
        """Resolve a possibly-relative path against the sandbox working directory.

        Modal's filesystem API only accepts absolute paths, while `run_command` runs
        in the sandbox working directory. Relative paths are joined with that directory
        -- queried once with `pwd` and cached -- so the file tools and shell commands
        share one view of the tree.
        """
        if posixpath.isabs(path):
            return path
        if self._cwd is None:
            # Single-flight the probe: a batch of concurrent tool calls resolving relative
            # paths all find `_cwd` unset, so without the lock each would run its own `pwd`.
            # The re-check inside the lock lets the losers use the winner's cached result.
            async with self._cwd_lock:
                if self._cwd is None:
                    result = await self.exec(['sh', '-c', 'pwd'], timeout=_INTERNAL_EXEC_TIMEOUT)
                    # Only cache a successful probe. A timeout (returncode -1) or error returns
                    # empty stdout; caching '/' from it would silently mis-resolve every later
                    # relative path with no retry. Leave `_cwd` unset and fail this call so the
                    # next one probes again.
                    if result.returncode != 0:
                        raise ModalSandboxError(
                            'Could not determine the sandbox working directory to resolve a relative '
                            f'path ({path!r}); use an absolute path or retry.'
                        )
                    self._cwd = result.stdout.strip() or '/'
        if path in ('', '.'):
            return self._cwd
        return posixpath.join(self._cwd, path)

    async def exec(
        self, argv: Sequence[str], *, timeout: float | None = None, max_output_bytes: int | None = None
    ) -> ModalSandboxExecResult:
        """Run an argument vector in the sandbox (without a shell) and return its result.

        Modal does not currently expose a per-exec kill, so cancelling this coroutine stops
        us waiting for the command but does not stop the command: it keeps running until its
        `timeout` deadline (or until the sandbox itself is terminated). Pass a finite
        `timeout` so a cancelled or abandoned command cannot run on indefinitely;
        `timeout=None` leaves it unbounded, which is why the toolset always sets one.

        Args:
            argv: The command and its arguments.
            timeout: Per-command deadline in seconds, enforced server-side by Modal. None
                means no deadline (the command can outlive a cancellation).
            max_output_bytes: Cap on how much of each stream is retained in client memory.
                A command can print far more than the caller will ever show the model, so
                with this set only the last `max_output_bytes` bytes of each stream are kept
                exactly. A retained byte suffix can begin inside a multi-byte character and
                is decoded with replacement. None reads each stream in full -- fine for the
                small outputs of a direct session caller, but the toolset always sets it.
        """
        sandbox = self._require_sandbox()
        import modal

        if timeout is not None and (not math.isfinite(timeout) or timeout <= 0):
            raise ValueError(f'timeout must be a positive finite number or None, got {timeout!r}.')
        if max_output_bytes is not None and (type(max_output_bytes) is not int or max_output_bytes <= 0):
            raise ValueError(f'max_output_bytes must be a positive integer or None, got {max_output_bytes!r}.')

        # Modal takes whole-second timeouts and treats 0 as "no timeout", so round a finite
        # request up and floor it at 1. Owning this here keeps the Modal quantization in the
        # mechanism layer: any caller passing a fractional or sub-second deadline still gets a
        # finite, Modal-legal one. The applied value rides back on ModalSandboxExecResult so the caller
        # can report the exact deadline without re-deriving it.
        deadline = None if timeout is None else max(1, math.ceil(timeout))
        # Drain both streams and wait concurrently. They share the same server-side deadline,
        # so reading one to completion before starting the other can lose buffered output once
        # that deadline expires. Modal's text mode decodes strictly, so read bytes and decode here.
        try:
            process = await sandbox.exec.aio(*argv, timeout=deadline, text=False)
        except modal.exception.Error as e:
            raise self._use_error(e, 'Command could not run in the sandbox') from e

        stdout: str | None = None
        stderr: str | None = None
        returncode: int | None = None
        command_error: modal.exception.Error | None = None

        async def read_stdout() -> None:
            nonlocal stdout, command_error
            try:
                stdout = await self._read_stream(process.stdout, max_output_bytes)
            except modal.exception.Error as e:
                command_error = e
                task_group.cancel_scope.cancel()

        async def read_stderr() -> None:
            nonlocal stderr, command_error
            try:
                stderr = await self._read_stream(process.stderr, max_output_bytes)
            except modal.exception.Error as e:
                command_error = e
                task_group.cancel_scope.cancel()

        async def wait_for_exit() -> None:
            nonlocal returncode, command_error
            try:
                returncode = await process.wait.aio()
            except modal.exception.Error as e:
                command_error = e
                task_group.cancel_scope.cancel()

        async with anyio.create_task_group() as task_group:
            task_group.start_soon(read_stdout)
            task_group.start_soon(read_stderr)
            task_group.start_soon(wait_for_exit)
        if command_error is not None:
            raise self._use_error(command_error, 'Command could not run in the sandbox') from command_error
        if stdout is None or stderr is None or returncode is None:  # pragma: no cover - task group completion contract
            raise ModalSandboxError('Modal command result was incomplete.')
        # Modal returns `-1` when it kills a command at its timeout (real exits are 0-255,
        # signals are 128+n). Only read it as a timeout when we actually set a deadline, so an
        # unbounded command reporting -1 for some other reason is not mislabelled "timed out".
        return ModalSandboxExecResult(
            stdout=stdout,
            stderr=stderr,
            returncode=returncode,
            timed_out=deadline is not None and returncode == -1,
            applied_timeout=deadline,
        )

    @staticmethod
    def _decode_stream_chunks(chunks: Iterable[bytes]) -> str:
        decoder = codecs.getincrementaldecoder('utf-8')(errors='replace')
        parts = [decoder.decode(chunk, final=False) for chunk in chunks]
        parts.append(decoder.decode(b'', final=True))
        return ''.join(parts)

    @staticmethod
    async def _read_stream(stream: modal.io_streams.StreamReader[bytes], max_output_bytes: int | None) -> str:
        """Drain a Modal exec stream, optionally retaining only its last `max_output_bytes`.

        Unbounded (`max_output_bytes is None`) reads the whole stream in one call. Bounded
        keeps the most recent bytes under the exact cap, including when one transport chunk
        is larger than the cap. The newest output -- where a command's error and exit status
        sit -- survives. Retained bytes are decoded as UTF-8 with replacement after selection.
        """
        if max_output_bytes is None:
            return ModalSandboxSession._decode_stream_chunks([await stream.read.aio()])
        retained = bytearray()
        async for chunk in stream:
            if len(chunk) >= max_output_bytes:
                retained = bytearray(chunk[-max_output_bytes:])
                continue
            retained.extend(chunk)
            excess = len(retained) - max_output_bytes
            if excess > 0:
                del retained[:excess]
        return ModalSandboxSession._decode_stream_chunks([bytes(retained)])

    async def file_size(self, path: str) -> int:
        """Return a file's size in bytes via Modal's filesystem API, without reading it.

        Lets a caller check size before reading the whole file. A relative `path` is resolved
        against the sandbox working directory (see `_resolve`).

        Raises:
            ModalSandboxError: if the file cannot be stat-ed (missing, a directory, ...).
        """
        sandbox = self._require_sandbox()
        import modal

        target = await self._resolve(path)
        # Catch `Error`, not just `SandboxFilesystemError`: a transient connection or auth
        # failure raises a plain Modal `Error` here too, and it must surface as a
        # ModalSandboxError (a retryable tool error) rather than leak raw to the agent loop.
        try:
            info = await sandbox.filesystem.stat.aio(target)
        except modal.exception.Error as e:
            raise await self._filesystem_error(e) from e
        return info.size

    async def read_bytes(self, path: str) -> bytes:
        """Read a file's raw bytes from the sandbox via Modal's filesystem API.

        The session deals in bytes so each tool layer can decode (or not) as it needs;
        text handling lives above the session, not here. A relative `path` is resolved
        against the sandbox working directory (see `_resolve`).

        Raises:
            ModalSandboxError: if the file cannot be read (missing, a directory, ...).
        """
        sandbox = self._require_sandbox()
        import modal

        target = await self._resolve(path)
        try:
            return await sandbox.filesystem.read_bytes.aio(target)
        except modal.exception.Error as e:
            raise await self._filesystem_error(e) from e

    async def write_bytes(self, path: str, data: bytes) -> None:
        """Write raw bytes to a file in the sandbox, creating parent directories.

        A relative `path` is resolved against the sandbox working directory (see
        `_resolve`). Unlike shelling out, Modal's filesystem API streams the content,
        so the size is not bounded by the argument-length limit of a command.

        Raises:
            ModalSandboxError: if the file cannot be written (bad path, permissions, ...).
        """
        sandbox = self._require_sandbox()
        import modal

        target = await self._resolve(path)
        # `target` is always absolute, so its parent is at least the root, which always
        # exists -- skip make_directory for it. Test for "no path component" rather than
        # `== '/'`: POSIX `normpath` preserves a leading '//' as a distinct root spelling,
        # so a parent of '//' is still the root and must be skipped too.
        parent = posixpath.dirname(target)
        try:
            if parent.strip('/'):
                try:
                    await sandbox.filesystem.make_directory.aio(parent, create_parents=True)
                except modal.exception.SandboxFilesystemPathAlreadyExistsError:
                    pass
            await sandbox.filesystem.write_bytes.aio(data, target)
        except modal.exception.Error as e:
            raise await self._filesystem_error(e) from e

    async def list_files(self, path: str) -> list[tuple[str, bool]]:
        """List a sandbox directory as `(name, is_dir)` pairs.

        A relative `path` is resolved against the sandbox working directory (see
        `_resolve`). The Modal-native `FileInfo` entries are normalized to plain tuples
        here so the provider type does not leak past the session.

        Raises:
            ModalSandboxError: if the directory cannot be listed.
        """
        sandbox = self._require_sandbox()
        import modal

        target = await self._resolve(path)
        try:
            entries = await sandbox.filesystem.list_files.aio(target)
        except modal.exception.Error as e:
            raise await self._filesystem_error(e) from e
        return [(entry.name, entry.is_dir()) for entry in entries]
