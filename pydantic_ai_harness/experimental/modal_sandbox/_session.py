"""Lifecycle management for a Modal sandbox."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import anyio
from anyio import to_thread
from typing_extensions import Self

if TYPE_CHECKING:
    import modal


class ModalSandboxError(RuntimeError):
    """Raised when a Modal sandbox cannot be created, attached to, or used."""


_MISSING_MODAL = (
    "The 'modal' package is required for the ModalSandbox capability. "
    'Install it with `pip install "pydantic-ai-harness[modal]"`.'
)


class ModalSandboxSession:
    """Async context manager that owns or attaches to a Modal sandbox.

    In *owned* mode (the default) it creates a fresh sandbox from `image` on
    enter and terminates it on exit. In *attach* mode (`sandbox_id` set) it looks
    up an existing sandbox and leaves it running on exit, so a sandbox you manage
    elsewhere can be reused across runs.

    Modal's blocking API is driven through worker threads, so the session works
    under any async backend and authenticates from the standard `MODAL_TOKEN_ID`
    / `MODAL_TOKEN_SECRET` environment variables.

    ```python
    async with ModalSandboxSession(image='python:3.12-slim') as session:
        stdout, stderr, code = await session.exec(['echo', 'hello'])
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
    ) -> None:
        self._image = image
        self._sandbox_id = sandbox_id
        self._app_name = app_name
        self._create_app_if_missing = create_app_if_missing
        self._sandbox_timeout = sandbox_timeout
        self._workdir = workdir
        self._sandbox: modal.Sandbox | None = None

    @property
    def sandbox_id(self) -> str | None:
        """The id of the running sandbox, or None when it is not running."""
        if self._sandbox is None:
            return None
        return self._sandbox.object_id

    async def __aenter__(self) -> Self:
        """Create or attach to the sandbox."""
        try:
            import modal
        except ImportError as e:
            raise ModalSandboxError(_MISSING_MODAL) from e
        try:
            self._sandbox = await to_thread.run_sync(self._open_sandbox)
        except modal.exception.Error as e:
            raise ModalSandboxError(f'Could not start Modal sandbox: {e}') from e
        return self

    def _open_sandbox(self) -> modal.Sandbox:
        """Create an owned sandbox or attach to an existing one (runs in a worker thread)."""
        import modal

        if self._sandbox_id is not None:
            return modal.Sandbox.from_id(self._sandbox_id)
        app = modal.App.lookup(self._app_name, create_if_missing=self._create_app_if_missing)
        # modal's `from_registry` is typed with an untyped `**kwargs`, so pyright flags the access.
        image = modal.Image.from_registry(self._image)  # pyright: ignore[reportUnknownMemberType]
        return modal.Sandbox.create(app=app, image=image, timeout=self._sandbox_timeout, workdir=self._workdir)

    async def __aexit__(self, *args: Any) -> None:
        """Release the sandbox: terminate it when owned, and always detach the client."""
        sandbox = self._sandbox
        self._sandbox = None
        if sandbox is None:
            return
        owned = self._sandbox_id is None

        def close() -> None:
            # Stop a sandbox we created; an attached one keeps running. Always detach to
            # release the local client connection — Modal's recommended cleanup.
            if owned:
                sandbox.terminate()
            sandbox.detach()

        with anyio.CancelScope(shield=True):
            await to_thread.run_sync(close)

    async def exec(self, argv: list[str], *, timeout: int | None = None) -> tuple[str, str, int]:
        """Run an argument vector in the sandbox and return (stdout, stderr, exit code).

        Args:
            argv: The command and its arguments (executed without a shell).
            timeout: Optional per-command timeout in seconds, enforced by Modal.
        """
        sandbox = self._sandbox
        if sandbox is None:
            raise ModalSandboxError('The sandbox is not running; use the session as an async context manager.')

        def run() -> tuple[str, str, int]:
            # Modal buffers exec output server-side and streams it over its own connection,
            # so draining stdout then stderr before waiting cannot deadlock the way OS pipes
            # would. `text=True` (Modal's default) makes the streams yield str.
            process = sandbox.exec(*argv, timeout=timeout)
            stdout = process.stdout.read()
            stderr = process.stderr.read()
            process.wait()
            return stdout, stderr, process.returncode or 0

        return await to_thread.run_sync(run)
