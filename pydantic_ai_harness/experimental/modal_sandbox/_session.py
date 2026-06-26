"""Lifecycle management for a Modal sandbox."""

from __future__ import annotations

import posixpath
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import anyio
from typing_extensions import Self

if TYPE_CHECKING:
    import modal


class ModalSandboxError(RuntimeError):
    """Raised when a Modal sandbox cannot be created, attached to, or used."""


@dataclass(frozen=True)
class ExecResult:
    """The outcome of running a command in the sandbox."""

    stdout: str
    stderr: str
    returncode: int
    timed_out: bool = False
    """True if Modal killed the command at its timeout (its `-1` exit sentinel)."""


_MISSING_MODAL = (
    "The 'modal' package is required for the ModalSandbox capability. "
    'Install it with `pip install "pydantic-ai-harness[modal]"`.'
)

# Modal does not currently expose a per-exec kill: a command is reaped by its own
# server-side timeout (or by the whole sandbox being terminated). So every command we run
# carries a deadline, even
# internal ones like the `pwd` used for path resolution, so a cancelled or abandoned run
# cannot leave a command billing indefinitely. This bounds that internal probe.
_INTERNAL_EXEC_TIMEOUT = 10


# This is the mechanism layer: every Modal-specific operation (create/attach,
# exec, file access, path resolution, lifecycle) is contained here, behind a small
# byte-oriented method surface that the toolset depends on. Keeping it isolated from
# the presentation in `_toolset.py` is what lets the sandbox internals change without
# touching the tools or the capability.
class ModalSandboxSession:
    """Async context manager that owns or attaches to a Modal sandbox.

    In *owned* mode (the default) it creates a fresh sandbox from `image` on
    enter and terminates it on exit. In *attach* mode (`sandbox_id` set) it looks
    up an existing sandbox and leaves it running on exit, so a sandbox you manage
    elsewhere can be reused across runs.

    Modal's SDK is asyncio-native, so this session drives its `.aio` coroutine API
    directly and requires an asyncio event loop. It authenticates from the standard
    `MODAL_TOKEN_ID` / `MODAL_TOKEN_SECRET` environment variables.

    ```python
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
        env: dict[str, str] | None = None,
    ) -> None:
        self._image = image
        self._sandbox_id = sandbox_id
        self._app_name = app_name
        self._create_app_if_missing = create_app_if_missing
        self._sandbox_timeout = sandbox_timeout
        self._workdir = workdir
        self._env = env
        self._sandbox: modal.Sandbox | None = None
        self._cwd: str | None = None

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
            self._sandbox = await self._open_sandbox()
        except modal.exception.Error as e:
            raise ModalSandboxError(f'Could not start Modal sandbox: {e}') from e
        return self

    async def _open_sandbox(self) -> modal.Sandbox:
        """Create an owned sandbox or attach to an existing one."""
        import modal

        if self._sandbox_id is not None:
            return await modal.Sandbox.from_id.aio(self._sandbox_id)
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

    async def __aexit__(self, *args: Any) -> None:
        """Release the sandbox: terminate it when owned, and always detach the client."""
        sandbox = self._sandbox
        self._sandbox = None
        self._cwd = None
        if sandbox is None:
            return
        owned = self._sandbox_id is None
        # Shield cleanup so a cancellation mid-run still tears the sandbox down. Stop a
        # sandbox we created; an attached one keeps running. Detach in `finally` so the
        # local client connection is always released -- Modal's recommended cleanup --
        # even if terminating the owned sandbox fails.
        with anyio.CancelScope(shield=True):
            try:
                if owned:
                    await sandbox.terminate.aio()
            finally:
                await sandbox.detach.aio()  # pyright: ignore[reportUnknownMemberType]

    def _require_sandbox(self) -> modal.Sandbox:
        sandbox = self._sandbox
        if sandbox is None:
            raise ModalSandboxError('The sandbox is not running; use the session as an async context manager.')
        return sandbox

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
            result = await self.exec(['sh', '-c', 'pwd'], timeout=_INTERNAL_EXEC_TIMEOUT)
            self._cwd = result.stdout.strip() or '/'
        return posixpath.normpath(posixpath.join(self._cwd, path))

    async def exec(self, argv: list[str], *, timeout: int | None = None) -> ExecResult:
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
        """
        sandbox = self._require_sandbox()
        import modal

        # Modal buffers exec output server-side and streams it over its own connection,
        # so draining stdout then stderr before waiting cannot deadlock the way OS pipes
        # would. `text=True` (Modal's default) makes the streams yield str.
        try:
            process = await sandbox.exec.aio(*argv, timeout=timeout)
            stdout = await process.stdout.read.aio()
            stderr = await process.stderr.read.aio()
            returncode = await process.wait.aio()
        except modal.exception.Error as e:
            raise ModalSandboxError(f'Command could not run in the sandbox: {e}') from e
        # Modal returns `-1` when it kills a command at its timeout (real exits are 0-255,
        # signals are 128+n), so `-1` flags a timeout rather than a command exit status.
        return ExecResult(stdout=stdout, stderr=stderr, returncode=returncode, timed_out=returncode == -1)

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
        try:
            info = await sandbox.filesystem.stat.aio(target)
        except modal.exception.SandboxFilesystemError as e:
            raise ModalSandboxError(str(e)) from e
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
        except modal.exception.SandboxFilesystemError as e:
            raise ModalSandboxError(str(e)) from e

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
        # `target` is always absolute, so its parent is at least '/'; only skip the
        # filesystem root, which always exists.
        parent = posixpath.dirname(target)
        try:
            if parent != '/':
                await sandbox.filesystem.make_directory.aio(parent, create_parents=True)
            await sandbox.filesystem.write_bytes.aio(data, target)
        except modal.exception.SandboxFilesystemError as e:
            raise ModalSandboxError(str(e)) from e

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
        except modal.exception.SandboxFilesystemError as e:
            raise ModalSandboxError(str(e)) from e
        return [(entry.name, entry.is_dir()) for entry in entries]
