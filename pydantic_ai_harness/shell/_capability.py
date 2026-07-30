"""Shell capability that provides command execution for agents."""

from __future__ import annotations

from collections.abc import Mapping, Sequence, Set
from dataclasses import dataclass
from pathlib import Path
from typing import overload

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.tools import AgentDepsT

from pydantic_ai_harness.shell._command_policy import normalize_command_policy
from pydantic_ai_harness.shell._toolset import ShellToolset

LLM_API_KEY_ENV_PATTERNS: tuple[str, ...] = (
    'ANTHROPIC_*',
    'GATEWAY_*',
    'GEMINI_*',
    'GOOGLE_*',
    'OPENAI_*',
    'OPENROUTER_*',
    'PYDANTIC_AI_GATEWAY_API_KEY',
)
"""Glob patterns for common LLM provider credentials, for `denied_env_patterns`.

Pass these when an agent runs untrusted commands that must not read the host's
LLM API keys. Covers provider prefixes only -- not other host secrets, and the
prefixes are coarse (`GOOGLE_*` also strips `GOOGLE_APPLICATION_CREDENTIALS`),
so treat it as a starting point. Not a default: stripping env silently would
break agents that rely on inherited credentials, so opt in explicitly.
"""


@dataclass(init=False)
class Shell(AbstractCapability[AgentDepsT]):
    """Shell command execution for agents.

    Commands execute in a subprocess rooted at `cwd`. Use `allowed_commands`
    or `denied_commands` to control what the agent can invoke.
    """

    cwd: str | Path
    """Working directory for command execution."""

    allowed_commands: tuple[str, ...] | None
    """Command names allowed in allowlist mode, or `None` in denylist mode."""

    denied_commands: tuple[str, ...] | None
    """Command names rejected in denylist mode, or `None` in allowlist mode.

    When neither command control is supplied, this contains the built-in list
    of destructive commands. Pass an empty collection to disable command-name
    filtering.
    """

    denied_operators: tuple[str, ...]
    """Shell operators that are blocked (e.g. '>', '>>', '|' for restrictive mode)."""

    default_timeout: float
    """Default timeout in seconds for command execution."""

    max_output_chars: int
    """Maximum characters of output returned to the model."""

    persist_cwd: bool
    """If True, track cd commands and adjust the working directory for subsequent calls."""

    allow_interactive: bool
    """If True, allow interactive commands (vi, nano, ssh, etc.). Blocked by default."""

    env: Mapping[str, str] | None
    """Explicit environment for spawned subprocesses, replacing inheritance.

    When `None` (default) the subprocess inherits the parent environment. Set
    this to a fixed mapping to start subprocesses with exactly these variables
    and nothing else -- a hard boundary that keeps host secrets (LLM API keys,
    tokens) out of commands the agent runs.
    """

    denied_env_patterns: tuple[str, ...]
    """Glob patterns for environment variable names to strip before spawning.

    Follows the `denied_*` naming convention but matches by glob (`fnmatch`,
    e.g. `OPENAI_*`), since env secrets cluster by prefix -- unlike
    `denied_commands`, which matches executable names exactly. Names matching
    any pattern are removed from the base environment; applied on top of `env`
    when both are set, so patterns filter an explicit `env` too. See
    `LLM_API_KEY_ENV_PATTERNS` for a ready-made provider-credential denylist.
    """

    @overload
    def __init__(  # pragma: no cover - overload is enforced by static type checking
        self,
        cwd: str | Path = '.',
        *,
        allowed_commands: Sequence[str] | Set[str],
        denied_commands: None = None,
        denied_operators: Sequence[str] = (),
        default_timeout: float = 30.0,
        max_output_chars: int = 50_000,
        persist_cwd: bool = False,
        allow_interactive: bool = False,
        env: Mapping[str, str] | None = None,
        denied_env_patterns: Sequence[str] = (),
        id: str | None = None,
        description: str | None = None,
        defer_loading: bool = False,
    ) -> None: ...

    @overload
    def __init__(  # pragma: no cover - overload is enforced by static type checking
        self,
        cwd: str | Path = '.',
        *,
        allowed_commands: None = None,
        denied_commands: Sequence[str] | Set[str] | None = None,
        denied_operators: Sequence[str] = (),
        default_timeout: float = 30.0,
        max_output_chars: int = 50_000,
        persist_cwd: bool = False,
        allow_interactive: bool = False,
        env: Mapping[str, str] | None = None,
        denied_env_patterns: Sequence[str] = (),
        id: str | None = None,
        description: str | None = None,
        defer_loading: bool = False,
    ) -> None: ...

    def __init__(
        self,
        cwd: str | Path = '.',
        *,
        allowed_commands: Sequence[str] | Set[str] | None = None,
        denied_commands: Sequence[str] | Set[str] | None = None,
        denied_operators: Sequence[str] = (),
        default_timeout: float = 30.0,
        max_output_chars: int = 50_000,
        persist_cwd: bool = False,
        allow_interactive: bool = False,
        env: Mapping[str, str] | None = None,
        denied_env_patterns: Sequence[str] = (),
        id: str | None = None,
        description: str | None = None,
        defer_loading: bool = False,
    ) -> None:
        """Configure shell command execution."""
        policy = normalize_command_policy(allowed_commands, denied_commands)

        self.cwd = cwd
        self.allowed_commands = policy.commands if policy.mode == 'allow' else None
        self.denied_commands = policy.commands if policy.mode == 'deny' else None
        self.denied_operators = tuple(denied_operators)
        self.default_timeout = default_timeout
        self.max_output_chars = max_output_chars
        self.persist_cwd = persist_cwd
        self.allow_interactive = allow_interactive
        self.env = dict(env) if env is not None else None
        self.denied_env_patterns = tuple(denied_env_patterns)
        self.id = id
        self.description = description
        self.defer_loading = defer_loading

    def get_toolset(self) -> ShellToolset[AgentDepsT]:
        """Build and return the shell toolset."""
        policy = normalize_command_policy(self.allowed_commands, self.denied_commands)
        if policy.mode == 'allow':
            return ShellToolset[AgentDepsT](
                cwd=Path(self.cwd),
                allowed_commands=policy.commands,
                denied_operators=self.denied_operators,
                default_timeout=self.default_timeout,
                max_output_chars=self.max_output_chars,
                persist_cwd=self.persist_cwd,
                allow_interactive=self.allow_interactive,
                env=self.env,
                denied_env_patterns=self.denied_env_patterns,
            )

        return ShellToolset[AgentDepsT](
            cwd=Path(self.cwd),
            denied_commands=policy.commands,
            denied_operators=self.denied_operators,
            default_timeout=self.default_timeout,
            max_output_chars=self.max_output_chars,
            persist_cwd=self.persist_cwd,
            allow_interactive=self.allow_interactive,
            env=self.env,
            denied_env_patterns=self.denied_env_patterns,
        )
