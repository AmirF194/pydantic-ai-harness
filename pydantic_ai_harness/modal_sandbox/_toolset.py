"""Modal sandbox toolset: shell and file tools over the run's sandbox."""

from __future__ import annotations

import math
from typing import Annotated

from pydantic import Field
from pydantic_ai import RunContext
from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.tools import AgentDepsT
from pydantic_ai.toolsets import FunctionToolset

from pydantic_ai_harness.modal_sandbox._backend import (
    ModalSandboxCommandTimeoutError,
    ModalSandboxError,
    ModalSandboxTerminalError,
)
from pydantic_ai_harness.modal_sandbox._tool_output import guard_read_size, render_file_window, truncate_output


class ModalSandboxToolset(FunctionToolset[AgentDepsT]):
    """Shell and file tools that operate on the run's [`sandbox`][pydantic_ai.tools.RunContext.sandbox].

    The toolset holds presentation and timeout policy only. The sandbox itself is supplied by
    the `ModalSandbox` capability (or by whatever else the run attached), read from
    `ctx.sandbox` on each call, so nothing here owns a connection or a lifetime.
    """

    def __init__(
        self,
        *,
        sandbox_timeout: int,
        default_command_timeout: float,
        max_command_timeout: int | None,
        max_output_bytes: int,
        max_output_lines: int,
        max_read_bytes: int,
    ) -> None:
        super().__init__()
        self._sandbox_timeout = sandbox_timeout
        self._default_command_timeout = default_command_timeout
        self._max_command_timeout = max_command_timeout
        self._max_output_bytes = max_output_bytes
        self._max_output_lines = max_output_lines
        self._max_read_bytes = max_read_bytes

        self.add_function(
            self.run_command,
            name='run_command',
            metadata={'code_arg_name': 'command', 'code_arg_language': 'shell'},
        )
        self.add_function(self.read_file, name='read_file')
        self.add_function(self.write_file, name='write_file')
        self.add_function(self.list_directory, name='list_directory')

    def _truncate_stream(self, text: str) -> str:
        return truncate_output(
            text,
            max_lines=self._max_output_lines,
            max_bytes=self._max_output_bytes,
            direction='tail',
        )

    def _render(self, stdout: str, stderr: str, note: str | None) -> str:
        # Truncate each stream separately and attach its label afterwards, so the
        # `[stdout]` / `[stderr]` markers always survive truncation and a large stderr
        # cannot crowd stdout out of a shared budget. Tail direction: errors and the
        # exit status live at the end.
        parts: list[str] = []
        if stdout:
            parts.append(f'[stdout]\n{self._truncate_stream(stdout)}')
        if stderr:
            parts.append(f'[stderr]\n{self._truncate_stream(stderr)}')
        output = '\n'.join(parts) if parts else '(no output)'
        return f'{output}\n{note}' if note else output

    def _command_timeout(self, timeout_seconds: float | None) -> int:
        if timeout_seconds is not None and (not math.isfinite(timeout_seconds) or timeout_seconds <= 0):
            # Reject rather than let the backend floor it to 1s: a 0 or negative request is a
            # model mistake, and a surprise "[timed out after 1s]" hides that from the model.
            raise ModelRetry(f'timeout_seconds must be greater than 0, got {timeout_seconds}.')
        requested = timeout_seconds if timeout_seconds is not None else self._default_command_timeout
        # Clamp to a hard ceiling. Modal cannot kill a running command, so a cancelled one
        # runs until its deadline; the ceiling bounds that worst case. It defaults to the
        # sandbox lifetime, beyond which an owned command cannot run anyway. Round up before
        # clamping so the whole-second Modal deadline cannot exceed the configured ceiling.
        ceiling = self._max_command_timeout if self._max_command_timeout is not None else self._sandbox_timeout
        return min(max(1, math.ceil(requested)), ceiling)

    async def run_command(  # noqa: D417
        self, ctx: RunContext[AgentDepsT], command: str, *, timeout_seconds: float | None = None
    ) -> str:
        """Run a shell command in the sandbox and return its output.

        The command runs through `sh -c`, so pipes, redirection, `&&`, and globs
        work. A non-zero exit is reported, not raised, so you can react to it.

        Args:
            command: The shell command to run.
            timeout_seconds: Maximum seconds to wait (default: the configured timeout).

        Returns:
            Labelled stdout/stderr output, with an exit code on non-zero exit.
        """
        deadline = self._command_timeout(timeout_seconds)
        try:
            result = await ctx.sandbox.run(command, shell=True, timeout=deadline)
        except ModalSandboxCommandTimeoutError as e:
            # Modal reports what the command printed before the deadline kill; the protocol's
            # raise-on-timeout has nowhere to return it, so it rides on the exception.
            return self._render(e.stdout, e.stderr, f'[timed out after {e.timeout}s]')
        except TimeoutError:
            # Another backend's deadline kill, which reports no output alongside it.
            return f'[timed out after {deadline}s]'
        # Surface a recoverable sandbox-side failure as a retryable tool error, matching the
        # file tools. A terminal failure (the sandbox is gone, or credentials were rejected)
        # propagates instead: retrying the command cannot fix it, so end the run cleanly.
        except ModalSandboxTerminalError:
            raise
        except ModalSandboxError as e:
            raise ModelRetry(str(e))
        note = f'[exit code: {result.exit_code}]' if result.exit_code else None
        return self._render(result.stdout, result.stderr, note)

    async def read_file(  # noqa: D417
        self,
        ctx: RunContext[AgentDepsT],
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
        sandbox = ctx.sandbox
        target = await sandbox.resolve(path)
        try:
            # Check size first: the read pulls the whole file into memory before windowing,
            # so refuse an oversized file rather than transfer and decode all of it for a slice.
            entry = await sandbox.fs.stat(target)
            if entry.size is not None:
                guard_read_size(entry.size, max_bytes=self._max_read_bytes)
            data = await sandbox.fs.read_bytes(target)
        except ModalSandboxTerminalError:
            raise
        except (FileNotFoundError, ModalSandboxError) as e:
            raise ModelRetry(f'Could not read {path!r}: {e}')
        # Re-check against the bytes actually returned. The stat and the read are separate
        # round-trips, so the file could have grown past the limit in between. The transfer
        # already happened, but refusing here still avoids the large UTF-8 decode and
        # windowing that would otherwise follow.
        guard_read_size(len(data), max_bytes=self._max_read_bytes)
        return render_file_window(
            data, offset=offset, limit=limit, max_lines=self._max_output_lines, max_bytes=self._max_output_bytes
        )

    async def write_file(self, ctx: RunContext[AgentDepsT], path: str, content: str) -> str:  # noqa: D417
        """Write text to a file in the sandbox, creating parent directories.

        Args:
            path: Path to the file inside the sandbox. Relative paths are resolved
                against the working directory used by `run_command`.
            content: The text to write.
        """
        sandbox = ctx.sandbox
        try:
            data = content.encode('utf-8')
        except UnicodeEncodeError:
            # Reachable when a provider's pre-parsed tool arguments carry an unpaired
            # surrogate; a model mistake, so retry rather than abort the run.
            raise ModelRetry('content contains characters that cannot be encoded as UTF-8 (unpaired surrogates).')
        target = await sandbox.resolve(path)
        try:
            await sandbox.fs.write_bytes(target, data)
        except ModalSandboxTerminalError:
            raise
        except (FileNotFoundError, ModalSandboxError) as e:
            raise ModelRetry(f'Could not write {path!r}: {e}')
        return f'Wrote {len(data)} bytes to {path!r}.'

    async def list_directory(self, ctx: RunContext[AgentDepsT], path: str = '.') -> str:  # noqa: D417
        """List the entries in a sandbox directory (directories shown with a trailing `/`).

        Args:
            path: Directory to list. Relative paths (including the default `.`) are
                resolved against the working directory used by `run_command`.
        """
        sandbox = ctx.sandbox
        target = await sandbox.resolve(path)
        try:
            entries = await sandbox.fs.list_dir(target)
        except ModalSandboxTerminalError:
            raise
        except (FileNotFoundError, ModalSandboxError) as e:
            raise ModelRetry(f'Could not list {path!r}: {e}')
        if not entries:
            return '(empty)'
        # Sort by name before adding the `/` suffix so directories keep plain name order
        # ('/' sorts after '-' and '.', which would misplace suffixed names).
        names = [f'{entry.name}/' if entry.is_dir else entry.name for entry in sorted(entries, key=lambda e: e.name)]
        # Directory listing is sorted, so keep the head if it overflows the cap.
        return truncate_output(
            '\n'.join(names), max_lines=self._max_output_lines, max_bytes=self._max_output_bytes, direction='head'
        )
