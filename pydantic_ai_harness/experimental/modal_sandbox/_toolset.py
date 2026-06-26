"""Modal sandbox toolset — gives agents a cloud sandbox to work in."""

from __future__ import annotations

import math
from typing import Annotated, Any

from pydantic import Field
from pydantic_ai import RunContext
from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.tools import AgentDepsT
from pydantic_ai.toolsets import AbstractToolset, FunctionToolset
from typing_extensions import Self

from pydantic_ai_harness._tool_output import render_file_window, truncate_output
from pydantic_ai_harness.experimental.modal_sandbox._session import ModalSandboxError, ModalSandboxSession


class ModalSandboxToolset(FunctionToolset[AgentDepsT]):
    """Gives an agent a Modal sandbox to run commands and manage files in.

    Holds the sandbox configuration and, for each run, opens a `ModalSandboxSession`
    (creating a fresh sandbox, or attaching to `sandbox_id`) that the tools execute
    against. The sandbox is torn down when the run ends if this toolset owns it.
    """

    def __init__(
        self,
        *,
        image: str,
        sandbox_id: str | None,
        app_name: str,
        create_app_if_missing: bool,
        sandbox_timeout: int,
        workdir: str | None,
        default_command_timeout: float,
        max_output_chars: int,
        session: ModalSandboxSession | None = None,
    ) -> None:
        super().__init__()
        self._image = image
        self._sandbox_id = sandbox_id
        self._app_name = app_name
        self._create_app_if_missing = create_app_if_missing
        self._sandbox_timeout = sandbox_timeout
        self._workdir = workdir
        self._default_command_timeout = default_command_timeout
        self._max_output_chars = max_output_chars
        # A caller-owned session to reuse instead of opening one per run; when set, this
        # toolset uses it but never opens or closes it.
        self._external_session = session
        self._session: ModalSandboxSession | None = None

        self.add_function(self.run_command, name='run_command')
        self.add_function(self.read_file, name='read_file')
        self.add_function(self.write_file, name='write_file')
        self.add_function(self.list_directory, name='list_directory')

    async def for_run(self, ctx: RunContext[AgentDepsT]) -> AbstractToolset[AgentDepsT]:
        """Return a fresh instance per run so each run gets its own sandbox session.

        `get_toolset` builds one shared instance at agent construction; this toolset
        opens a per-run sandbox in `__aenter__`, so each run needs its own instance
        whose `__aexit__` tears that sandbox down.
        """
        return ModalSandboxToolset[AgentDepsT](
            image=self._image,
            sandbox_id=self._sandbox_id,
            app_name=self._app_name,
            create_app_if_missing=self._create_app_if_missing,
            sandbox_timeout=self._sandbox_timeout,
            workdir=self._workdir,
            default_command_timeout=self._default_command_timeout,
            max_output_chars=self._max_output_chars,
            session=self._external_session,
        )

    async def __aenter__(self) -> Self:
        """Make a sandbox session available before tools run.

        With a caller-owned `session`, use it as-is (the caller opened it). Otherwise open a
        per-run session here so each run gets its own sandbox.
        """
        if self._external_session is not None:
            # The caller owns this session and must open it before the run; check here so an
            # unopened session fails at run start with a clear message, not mid-tool-call.
            if self._external_session.sandbox_id is None:
                raise ModalSandboxError(
                    'The injected session is not open. Enter it with `async with session:` before running the agent.'
                )
            self._session = self._external_session
            return self
        session = ModalSandboxSession(
            image=self._image,
            sandbox_id=self._sandbox_id,
            app_name=self._app_name,
            create_app_if_missing=self._create_app_if_missing,
            sandbox_timeout=self._sandbox_timeout,
            workdir=self._workdir,
        )
        await session.__aenter__()
        self._session = session
        return self

    async def __aexit__(self, *args: Any) -> None:
        """Close the per-run session; leave a caller-owned session for its owner to close."""
        session = self._session
        self._session = None
        if session is not None and self._external_session is None:
            await session.__aexit__(*args)

    def _require_session(self) -> ModalSandboxSession:
        if self._session is None:  # pragma: no cover - tools only run inside the toolset context
            raise RuntimeError('The Modal sandbox session is not open.')
        return self._session

    def _command_timeout(self, timeout_seconds: float | None) -> int:
        # Modal takes whole-second timeouts; round fractional values up so a sub-second
        # request is not floored to 0 (which Modal treats as "no timeout").
        timeout = timeout_seconds if timeout_seconds is not None else self._default_command_timeout
        return max(1, math.ceil(timeout))

    async def run_command(self, command: str, *, timeout_seconds: float | None = None) -> str:
        """Run a shell command in the sandbox and return its output.

        The command runs through `sh -c`, so pipes, redirection, `&&`, and globs
        work. A non-zero exit is reported, not raised, so you can react to it.

        Args:
            command: The shell command to run.
            timeout_seconds: Maximum seconds to wait (default: the configured timeout).

        Returns:
            Labelled stdout/stderr output, with an exit code on non-zero exit.
        """
        session = self._require_session()
        timeout = self._command_timeout(timeout_seconds)
        # Surface a sandbox-side failure as a retryable tool error, matching the file tools,
        # rather than letting it abort the whole run.
        try:
            result = await session.exec(['sh', '-c', command], timeout=timeout)
        except ModalSandboxError as e:
            raise ModelRetry(str(e))
        parts: list[str] = []
        if result.stdout:
            parts.append(f'[stdout]\n{result.stdout}')
        if result.stderr:
            parts.append(f'[stderr]\n{result.stderr}')
        body = '\n'.join(parts) if parts else '(no output)'
        # Command output: keep the tail, where errors and the exit status live.
        output = truncate_output(body, max_bytes=self._max_output_chars, direction='tail')
        if result.timed_out:
            return f'{output}\n[timed out after {timeout}s]'
        if result.returncode:
            return f'{output}\n[exit code: {result.returncode}]'
        return output

    async def read_file(
        self,
        path: str,
        *,
        offset: Annotated[int | None, Field(description='Line number to start reading from (1-indexed)')] = None,
        limit: Annotated[int | None, Field(description='Maximum number of lines to read')] = None,
    ) -> str:
        """Read a text file from the sandbox and return its contents.

        Large files are truncated to a safety cap; the result ends with the next
        `offset` to use to page through the rest.

        Args:
            path: Path to the file inside the sandbox. Relative paths are resolved
                against the working directory used by `run_command`.
            offset: Line number to start reading from (1-indexed).
            limit: Maximum number of lines to read.
        """
        session = self._require_session()
        try:
            data = await session.read_bytes(path)
        except ModalSandboxError as e:
            raise ModelRetry(f'Could not read {path!r}: {e}')
        return render_file_window(data, offset=offset, limit=limit, max_bytes=self._max_output_chars)

    async def write_file(self, path: str, content: str) -> str:
        """Write text to a file in the sandbox, creating parent directories.

        Args:
            path: Path to the file inside the sandbox. Relative paths are resolved
                against the working directory used by `run_command`.
            content: The text to write.
        """
        session = self._require_session()
        try:
            await session.write_bytes(path, content.encode('utf-8'))
        except ModalSandboxError as e:
            raise ModelRetry(f'Could not write {path!r}: {e}')
        return f'Wrote {len(content)} characters to {path!r}.'

    async def list_directory(self, path: str = '.') -> str:
        """List the entries in a sandbox directory (directories shown with a trailing `/`).

        Args:
            path: Directory to list. Relative paths (including the default `.`) are
                resolved against the working directory used by `run_command`.
        """
        session = self._require_session()
        try:
            entries = await session.list_files(path)
        except ModalSandboxError as e:
            raise ModelRetry(f'Could not list {path!r}: {e}')
        if not entries:
            return '(empty)'
        names = sorted(f'{name}/' if is_dir else name for name, is_dir in entries)
        # Directory listing is sorted, so keep the head if it overflows the cap.
        return truncate_output('\n'.join(names), max_bytes=self._max_output_chars, direction='head')
