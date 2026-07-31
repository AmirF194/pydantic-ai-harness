"""Shell toolset -- gives agents the ability to run commands via `ctx.sandbox`."""

from __future__ import annotations

import fnmatch
import functools
import os
import re
import shlex
import uuid
from collections.abc import Awaitable, Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Concatenate, ParamSpec

from pydantic_ai import RunContext
from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.sandboxes import Sandbox, SandboxProcess
from pydantic_ai.tools import AgentDepsT
from pydantic_ai.toolsets import AbstractToolset, FunctionToolset, ToolsetTool

from pydantic_ai_harness._output import truncate_tail

_P = ParamSpec('_P')

_CWD_MARKER = '__HARNESS_PWD_MARKER__'


def _recoverable(
    fn: Callable[Concatenate[ShellToolset, _P], Awaitable[str]],
) -> Callable[Concatenate[ShellToolset, _P], Awaitable[str]]:
    @functools.wraps(fn)
    async def wrapper(self: ShellToolset, *args: _P.args, **kwargs: _P.kwargs) -> str:
        try:
            return await fn(self, *args, **kwargs)
        except PermissionError as e:
            raise ModelRetry(str(e)) from e
        except NotImplementedError as e:
            raise ModelRetry(str(e)) from e

    return wrapper


def _is_interactive_command(command: str) -> bool:
    interactive_patterns = [
        r'^(vi|vim|nano|emacs|less|more|top|htop|man)\b',
        r'^sudo\s',
        r'^passwd\b',
        r'^ssh\b',
        r'^telnet\b',
        r'^ftp\b',
    ]
    return any(re.match(p, command.strip()) for p in interactive_patterns)


class ShellToolset(FunctionToolset[AgentDepsT]):
    """Shell command execution over `ctx.sandbox`.

    Supports synchronous execution (`run_command`) and background processes
    (`start_command`/`check_command`/`stop_command`). Output is truncated to fit model
    context and labelled with stdout/stderr/exit code.

    Background commands require a sandbox backend that implements
    [`SupportsStart`][pydantic_ai.sandboxes.SupportsStart]. The framework-default
    `LocalSandbox` does not; use `ModalSandbox` or another backend for background work.
    """

    def __init__(
        self,
        *,
        cwd: str | Path = '.',
        allowed_commands: Sequence[str],
        denied_commands: Sequence[str],
        denied_operators: Sequence[str],
        default_timeout: float,
        max_output_chars: int,
        persist_cwd: bool,
        allow_interactive: bool,
        env: Mapping[str, str] | None = None,
        denied_env_patterns: Sequence[str] = (),
        sandbox: Sandbox | None = None,
    ) -> None:
        super().__init__()
        self._initial_cwd = str(cwd)
        self._cwd = self._initial_cwd
        self._allowed_commands = list(allowed_commands)
        self._denied_commands = list(denied_commands)
        self._denied_operators = list(denied_operators)
        self._default_timeout = default_timeout
        self._max_output_chars = max_output_chars
        self._persist_cwd = persist_cwd
        self._allow_interactive = allow_interactive
        self._env = dict(env) if env is not None else None
        self._denied_env_patterns = list(denied_env_patterns)
        self._sandbox = sandbox
        self._background: dict[str, SandboxProcess] = {}

        if self._allowed_commands and self._denied_commands:
            raise ValueError('Specify allowed_commands or denied_commands, not both.')
        if max_output_chars <= 0:
            raise ValueError('max_output_chars must be a positive integer.')

        self.add_function(self.run_command, name='run_command')
        self.add_function(self.start_command, name='start_command')
        self.add_function(self.check_command, name='check_command')
        self.add_function(self.stop_command, name='stop_command')

    async def for_run(self, ctx: RunContext[AgentDepsT]) -> AbstractToolset[AgentDepsT]:
        return ShellToolset(
            cwd=self._initial_cwd,
            allowed_commands=self._allowed_commands,
            denied_commands=self._denied_commands,
            denied_operators=self._denied_operators,
            default_timeout=self._default_timeout,
            max_output_chars=self._max_output_chars,
            persist_cwd=self._persist_cwd,
            allow_interactive=self._allow_interactive,
            env=self._env,
            denied_env_patterns=self._denied_env_patterns,
            sandbox=ctx.sandbox,
        )

    @property
    def sandbox(self) -> Sandbox:
        if self._sandbox is None:
            raise RuntimeError(
                'ShellToolset has no sandbox; construct it with sandbox=... or use it inside an agent run.'
            )
        return self._sandbox

    async def call_tool(
        self,
        name: str,
        tool_args: dict[str, Any],
        ctx: RunContext[AgentDepsT],
        tool: ToolsetTool[AgentDepsT],
    ) -> Any:
        result = await super().call_tool(name, tool_args, ctx, tool)
        if not isinstance(result, str):
            return result
        return truncate_tail(result, self._max_output_chars)

    def _resolve_env(self) -> dict[str, str] | None:
        if self._env is None and not self._denied_env_patterns:
            return None
        base = dict(self._env) if self._env is not None else dict(os.environ)
        if not self._denied_env_patterns:
            return base
        return {
            name: value
            for name, value in base.items()
            if not any(fnmatch.fnmatchcase(name, pattern) for pattern in self._denied_env_patterns)
        }

    async def __aexit__(self, *args: Any) -> None:
        for proc in self._background.values():
            try:
                await proc.kill()
            except (NotImplementedError, ProcessLookupError, OSError):
                pass
        self._background.clear()

    def _first_denied_operator(self, command: str) -> str | None:
        return next((op for op in self._denied_operators if op in command), None)

    def _check_command(self, command: str) -> None:
        if not self._allow_interactive and _is_interactive_command(command):
            raise PermissionError(f'Interactive commands are not allowed. Command: {command!r}')

        matched_op = self._first_denied_operator(command)
        if matched_op:
            raise PermissionError(f'Shell operator {matched_op!r} is not allowed.')

        try:
            tokens = shlex.split(command)
        except ValueError:
            return
        if not tokens:
            return
        executable = tokens[0]

        if self._denied_commands and executable in self._denied_commands:
            raise PermissionError(f'Command {executable!r} is denied.')
        if self._allowed_commands and executable not in self._allowed_commands:
            raise PermissionError(f'Command {executable!r} is not in the allowed list.')

    async def _resolved_cwd(self) -> str:
        return await self.sandbox.resolve(self._cwd)

    def _wrap_for_cwd_capture(self, command: str) -> str:
        return f'{command}\n__harness_ec=$?\necho {_CWD_MARKER}\npwd\nexit $__harness_ec'

    def _wrap_for_strict_env(self, command: str) -> str:
        """Enforce env-strip via `env -i` so host-env-merging backends can't leak.

        `LocalSandbox` overlays the host environment on top of any explicit `env`
        passed to `run()`, which would leak vars we intentionally stripped. Wrap
        the command with `env -i` when the toolset was configured with `env=` or
        `denied_env_patterns=`; otherwise fall back to backend-default env
        inheritance.
        """
        resolved = self._resolve_env()
        if resolved is None:
            return command
        env_args = ' '.join(f'{name}={shlex.quote(value)}' for name, value in resolved.items())
        return f'env -i {env_args} sh -c {shlex.quote(command)}'

    def _extract_captured_cwd(self, stdout: str) -> tuple[str, str | None]:
        # The marker is echoed on its own line; user output may or may not end
        # with a trailing newline, so match the trailing separator only.
        marker = f'{_CWD_MARKER}\n'
        if marker not in stdout:
            return stdout, None
        prefix, tail = stdout.rsplit(marker, 1)
        if prefix.endswith('\n'):
            prefix = prefix[:-1]
        recorded = tail.strip().splitlines()
        if not recorded:
            return prefix, None
        return prefix, recorded[-1]

    @_recoverable
    async def run_command(self, command: str, *, timeout_seconds: float | None = None) -> str:
        """Execute a shell command and return its output.

        Args:
            command: The shell command to run.
            timeout_seconds: Maximum seconds to wait (default: 30).

        Returns:
            Labeled stdout/stderr output with exit code on non-zero exit.
        """
        self._check_command(command)
        timeout = timeout_seconds if timeout_seconds is not None else self._default_timeout

        actual_command = self._wrap_for_cwd_capture(command) if self._persist_cwd else command
        actual_command = self._wrap_for_strict_env(actual_command)
        try:
            result = await self.sandbox.run(
                actual_command,
                shell=True,
                cwd=await self._resolved_cwd(),
                timeout=timeout,
            )
        except TimeoutError:
            return f'[Command timed out after {timeout}s]'

        stdout = result.stdout
        recorded_cwd: str | None = None
        if self._persist_cwd:
            stdout, recorded_cwd = self._extract_captured_cwd(stdout)

        parts: list[str] = []
        if stdout:
            parts.append(f'[stdout]\n{stdout}')
        if result.stderr:
            parts.append(f'[stderr]\n{result.stderr}')
        output = '\n'.join(parts) if parts else '(no output)'

        if recorded_cwd is not None and result.exit_code == 0:
            self._cwd = recorded_cwd

        if result.exit_code != 0:
            output = f'{output}\n[exit code: {result.exit_code}]'
        return output

    @_recoverable
    async def start_command(self, command: str) -> str:
        """Start a long-running command in the background (e.g. a server or watcher).

        Callers MUST call `stop_command(command_id)` when done to terminate the process.

        Args:
            command: The shell command to run in the background.

        Returns:
            A message containing the unique command ID for later check/stop calls.
        """
        self._check_command(command)
        command_id = uuid.uuid4().hex[:12]
        proc = await self.sandbox.start(
            self._wrap_for_strict_env(command),
            shell=True,
            cwd=await self._resolved_cwd(),
        )
        self._background[command_id] = proc
        return f'Started background command: {command!r}\nID: {command_id}'

    async def check_command(self, command_id: str) -> str:
        """Check the status and recent output of a background command.

        Args:
            command_id: The ID returned by start_command.

        Returns:
            Status and recent output of the background command.
        """
        proc = self._background.get(command_id)
        if proc is None:
            return f'[Error: unknown command ID {command_id!r}]'

        pid = proc.pid
        alive = pid is not None and await self._pid_alive(pid)
        if alive:
            return '(no output yet)\n[status: running]'

        result = await proc.wait()
        output_sections: list[str] = []
        if result.stdout:
            output_sections.append(f'[stdout]\n{result.stdout}')
        if result.stderr:
            output_sections.append(f'[stderr]\n{result.stderr}')
        parts = ['\n'.join(output_sections) if output_sections else '(no output)', '[status: finished]']
        parts.append(f'[exit code: {result.exit_code}]')
        return '\n'.join(parts)

    async def stop_command(self, command_id: str) -> str:
        """Stop a background command and return its final output.

        Args:
            command_id: The ID returned by start_command.

        Returns:
            Final output and exit status of the stopped command.
        """
        proc = self._background.pop(command_id, None)
        if proc is None:
            return f'[Error: unknown command ID {command_id!r}]'

        try:
            await proc.kill()
        except (NotImplementedError, ProcessLookupError, OSError):
            pass
        result = await proc.wait()

        output_sections: list[str] = []
        if result.stdout:
            output_sections.append(f'[stdout]\n{result.stdout}')
        if result.stderr:
            output_sections.append(f'[stderr]\n{result.stderr}')
        parts = ['\n'.join(output_sections) if output_sections else '(no output)', '[stopped]']
        parts.append(f'[exit code: {result.exit_code}]')
        return '\n'.join(parts)

    async def _pid_alive(self, pid: int) -> bool:
        result = await self.sandbox.run(f'kill -0 {pid}', shell=True)
        return result.exit_code == 0
