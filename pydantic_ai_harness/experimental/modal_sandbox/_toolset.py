"""Modal sandbox toolset — gives agents a cloud sandbox to work in."""

from __future__ import annotations

import base64
from typing import Any

from pydantic_ai import RunContext
from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.tools import AgentDepsT
from pydantic_ai.toolsets import AbstractToolset, FunctionToolset
from typing_extensions import Self

from pydantic_ai_harness.experimental.modal_sandbox._session import ModalSandboxSession

# Write `$2` (base64) to `$1`, creating parent directories. Passing the content as an
# argument (base64-encoded) avoids stdin handling and keeps it free of shell metacharacters.
_WRITE_SCRIPT = 'mkdir -p "$(dirname "$1")" && printf %s "$2" | base64 -d > "$1"'


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
        default_timeout: float,
        max_output_chars: int,
    ) -> None:
        super().__init__()
        self._image = image
        self._sandbox_id = sandbox_id
        self._app_name = app_name
        self._create_app_if_missing = create_app_if_missing
        self._sandbox_timeout = sandbox_timeout
        self._workdir = workdir
        self._default_timeout = default_timeout
        self._max_output_chars = max_output_chars
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
            default_timeout=self._default_timeout,
            max_output_chars=self._max_output_chars,
        )

    async def __aenter__(self) -> Self:
        """Open the sandbox session before tools run."""
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
        """Close the sandbox session, terminating an owned sandbox."""
        session = self._session
        self._session = None
        if session is not None:
            await session.__aexit__(*args)

    def _require_session(self) -> ModalSandboxSession:
        if self._session is None:  # pragma: no cover - tools only run inside the toolset context
            raise RuntimeError('The Modal sandbox session is not open.')
        return self._session

    def _command_timeout(self, timeout_seconds: float | None) -> int:
        return int(timeout_seconds if timeout_seconds is not None else self._default_timeout)

    def _truncate(self, text: str) -> str:
        """Truncate output to the configured cap, keeping the tail.

        Errors and the `[stderr]` section land at the end, so the head is dropped
        and the final `max_output_chars` are kept.
        """
        if len(text) <= self._max_output_chars:
            return text
        marker = f'[... output truncated, showing last {self._max_output_chars} chars]\n'
        return marker + text[-self._max_output_chars :]

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
        stdout, stderr, exit_code = await session.exec(
            ['sh', '-c', command], timeout=self._command_timeout(timeout_seconds)
        )
        parts: list[str] = []
        if stdout:
            parts.append(f'[stdout]\n{stdout}')
        if stderr:
            parts.append(f'[stderr]\n{stderr}')
        output = self._truncate('\n'.join(parts) if parts else '(no output)')
        if exit_code:
            return f'{output}\n[exit code: {exit_code}]'
        return output

    async def read_file(self, path: str) -> str:
        """Read a text file from the sandbox and return its contents.

        Args:
            path: Path to the file inside the sandbox.
        """
        session = self._require_session()
        stdout, stderr, exit_code = await session.exec(['cat', '--', path], timeout=self._command_timeout(None))
        if exit_code:
            raise ModelRetry(f'Could not read {path!r}: {stderr.strip() or f"exit code {exit_code}"}')
        return self._truncate(stdout)

    async def write_file(self, path: str, content: str) -> str:
        """Write text to a file in the sandbox, creating parent directories.

        Args:
            path: Path to the file inside the sandbox.
            content: The text to write.
        """
        session = self._require_session()
        encoded = base64.b64encode(content.encode('utf-8')).decode('ascii')
        _, stderr, exit_code = await session.exec(
            ['sh', '-c', _WRITE_SCRIPT, 'sh', path, encoded], timeout=self._command_timeout(None)
        )
        if exit_code:
            raise ModelRetry(f'Could not write {path!r}: {stderr.strip() or f"exit code {exit_code}"}')
        return f'Wrote {len(content)} characters to {path!r}.'

    async def list_directory(self, path: str = '.') -> str:
        """List the entries in a sandbox directory (directories shown with a trailing `/`).

        Args:
            path: Directory to list (default: the working directory).
        """
        session = self._require_session()
        stdout, stderr, exit_code = await session.exec(['ls', '-1Ap', '--', path], timeout=self._command_timeout(None))
        if exit_code:
            raise ModelRetry(f'Could not list {path!r}: {stderr.strip() or f"exit code {exit_code}"}')
        return self._truncate(stdout) if stdout else '(empty)'
