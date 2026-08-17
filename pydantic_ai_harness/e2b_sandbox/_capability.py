"""E2B sandbox capability that gives agents an isolated cloud computer."""

from __future__ import annotations

import math
import posixpath
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from pydantic_ai import RunContext
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.exceptions import UserError
from pydantic_ai.tools import AgentDepsT
from pydantic_ai.toolsets import AgentToolset
from typing_extensions import Self

from pydantic_ai_harness.e2b_sandbox._session import (
    DEFAULT_SANDBOX_TIMEOUT,
    DEFAULT_WORKDIR,
    E2BSandboxSession,
)
from pydantic_ai_harness.e2b_sandbox._tool_output import DEFAULT_MAX_BYTES, DEFAULT_MAX_LINES
from pydantic_ai_harness.e2b_sandbox._toolset import E2BSandboxToolset

if TYPE_CHECKING:
    from pydantic_ai.agent.abstract import AbstractAgent

_DEFAULT_MAX_READ_BYTES = 5 * 1024 * 1024

_OWNED_INSTRUCTIONS = (
    'You have an E2B sandbox: an isolated, ephemeral cloud computer. Use `run_command` to run '
    'Bash commands, and `read_file` / `write_file` / `list_directory` to manage files. '
    'Commands run through Bash, so pipes and redirection work. A command times out after '
    '{default_timeout}s unless you pass `timeout_seconds` (up to {max_timeout}s). The sandbox '
    'is reset between runs, so persist anything important elsewhere.'
)

_REUSED_INSTRUCTIONS = (
    'You have an E2B sandbox: an isolated cloud computer. Use `run_command` to run Bash commands, '
    'and `read_file` / `write_file` / `list_directory` to manage files. Commands run through Bash, '
    'so pipes and redirection work. A command times out after {default_timeout}s unless you pass '
    '`timeout_seconds` (up to {max_timeout}s). This sandbox persists across runs, so files from '
    'earlier runs can still be present.'
)


def _durability_engines(capabilities: Iterable[AbstractCapability[Any]]) -> set[str]:
    """Name the durable execution engines among `capabilities` (e.g. Temporal, DBOS).

    Durability capabilities are the `innermost` ordering tier by definition (see
    `AbstractCapability.get_ordering`), so the public ordering declaration identifies
    them without importing any engine.
    """
    engines: set[str] = set()
    for capability in capabilities:
        ordering = capability.get_ordering()
        if ordering is not None and ordering.position == 'innermost':
            engines.add(type(capability).__name__.removesuffix('Durability'))
    return engines


def _reject_durability(engines: set[str]) -> None:
    if engines:
        names = ', '.join(sorted(engines))
        raise UserError(
            f'E2BSandbox cannot be combined with durable execution capabilities ({names}). '
            'Its run-scoped E2B client and sandbox lifecycle cannot safely cross or replay '
            'activity, step, or task boundaries.'
        )


@dataclass(kw_only=True)
class E2BSandbox(AbstractCapability[AgentDepsT]):
    """Access to an isolated cloud sandbox powered by [E2B](https://e2b.dev).

    Each agent run gets a fresh sandbox by default. Set `sandbox_id` to attach to a
    sandbox managed elsewhere, or pass an open `E2BSandboxSession` whose lifetime you
    manage. Command and file operations emit content-free OpenTelemetry spans through
    the active Pydantic AI tracer, so they appear in Logfire when the agent is instrumented.

    Requires the `e2b` extra (`uv add "pydantic-ai-harness[e2b]"`) and an
    `E2B_API_KEY`. Durable execution capabilities are rejected during agent
    construction and at run start because E2B's run-scoped client and sandbox
    lifecycle cannot safely cross or replay their activity, step, or task boundaries.
    """

    template: str | None = None
    """E2B template name or id for newly created sandboxes. None uses E2B's default."""

    sandbox_id: str | None = None
    """Attach to an existing sandbox by id instead of creating one. Attached sandboxes are not killed.

    Use this to reuse a sandbox created elsewhere (e.g. via the E2B dashboard or SDK). The
    settings that only apply when creating a sandbox (`template`, `sandbox_timeout`, `env`,
    `metadata`, `allow_internet_access`) cannot be combined with `sandbox_id`. Unlike
    creation settings, `workdir` still applies: E2B sets it per command, not at creation.
    """

    session: E2BSandboxSession | None = None
    """Use a caller-owned, already-open session without opening or closing it."""

    sandbox_timeout: int = DEFAULT_SANDBOX_TIMEOUT
    """Maximum lifetime in seconds for an owned sandbox."""

    workdir: str = DEFAULT_WORKDIR
    """Absolute working directory for commands and relative file paths."""

    env: Mapping[str, str] | None = None
    """Environment variables applied when creating an owned sandbox."""

    metadata: Mapping[str, str] | None = None
    """E2B metadata applied when creating an owned sandbox."""

    allow_internet_access: bool = True
    """Whether an owned sandbox may access the internet."""

    default_command_timeout: float = 60.0
    """Default maximum runtime in seconds for one command."""

    max_command_timeout: int | None = None
    """Hard ceiling in seconds for any single `run_command`, including a model-supplied
    `timeout_seconds`. None falls back to `sandbox_timeout`.

    An owned command cannot outlive `sandbox_timeout` anyway, so the default ceiling is
    exact for owned sandboxes. For an attached or injected sandbox the fallback is pinned
    to the default (300s), because the capability does not know the real lifetime of a
    sandbox it did not create. So every command there is capped at 300s unless you set
    `max_command_timeout` to the value the sandbox actually allows.
    """

    max_output_bytes: int = DEFAULT_MAX_BYTES
    """Maximum payload retained per command stream or file read, measured in UTF-8 bytes.

    For commands the cap applies to stdout and stderr separately, both inside the sandbox
    (each capture file retains at most this many bytes) and in the tool output, so a large
    stderr cannot crowd out stdout. Labels, truncation notes, continuation offsets,
    timeouts, and exit codes add a small amount beyond this payload limit. Whichever of
    `max_output_bytes` and `max_output_lines` is reached first wins.
    """

    max_output_lines: int = DEFAULT_MAX_LINES
    """Maximum payload lines retained per command stream or file read, alongside `max_output_bytes`.

    A second cap so many short lines cannot pile up under the byte budget. Whichever cap is
    reached first wins. Labels and truncation or status notes can add lines beyond this
    payload limit.
    """

    max_read_bytes: int = _DEFAULT_MAX_READ_BYTES
    """Largest file `read_file` will read whole; larger files are refused with a hint to use shell tools.

    The tool checks E2B file metadata first and then streams at most this many bytes plus
    one into client memory, so a file that grows after the metadata check still cannot
    create an unbounded client buffer.
    """

    instructions: str | None = None
    """Model instructions. None uses mode-aware defaults; an empty string disables them."""

    def __post_init__(self) -> None:
        """Validate limits and reject settings ignored by the selected lifecycle mode."""
        self._validate_configuration()
        if self.env is not None:
            self.env = dict(self.env)
        if self.metadata is not None:
            self.metadata = dict(self.metadata)

        if self.session is not None:
            conflicts = self._non_default_owned_settings()
            if self.sandbox_id is not None:
                conflicts.append('sandbox_id')
            if conflicts:
                raise ValueError(
                    f'{", ".join(conflicts)} cannot be combined with `session`, which already owns '
                    'the sandbox and its configuration.' + self._command_ceiling_hint(conflicts)
                )
            return

        if self.sandbox_id is not None:
            conflicts = self._non_default_creation_settings()
            if conflicts:
                raise ValueError(
                    f'{", ".join(conflicts)} only apply when creating a sandbox, but `sandbox_id` attaches '
                    'to an existing one. Remove them, or drop `sandbox_id` to create a sandbox.'
                    + self._command_ceiling_hint(conflicts)
                )
            return

        # Owned mode: a command cannot outlive the sandbox, so a ceiling above the sandbox
        # lifetime is a dead value. In attach/injected modes a higher ceiling is the
        # documented escape hatch for sandboxes whose real lifetime exceeds the pinned
        # default, so no check there.
        ceiling = self.max_command_timeout
        if ceiling is not None and ceiling > self.sandbox_timeout:
            raise ValueError(
                f'max_command_timeout ({ceiling}) cannot exceed sandbox_timeout '
                f'({self.sandbox_timeout}) for an owned sandbox: the sandbox is reaped '
                'before such a command could finish. Raise sandbox_timeout instead.'
            )

    def for_agent(self, agent: AbstractAgent[AgentDepsT, Any]) -> Self:
        """Reject durability wrappers that cannot preserve the E2B session lifecycle."""
        leaves: list[AbstractCapability[AgentDepsT]] = []
        agent.root_capability.apply(leaves.append)
        _reject_durability(_durability_engines(leaves))
        return self

    def _validate_runtime_capabilities(
        self, ctx: RunContext[AgentDepsT], capabilities: Sequence[AbstractCapability[AgentDepsT]]
    ) -> None:
        """Apply the durability rejection to capabilities added for a single run.

        This private core hook is the existing seam for vetoing per-run capability
        additions (pydantic-ai#5477); the coupling is isolated to this one override.
        """
        _reject_durability(_durability_engines(capabilities))

    def _validate_configuration(self) -> None:
        for name, value in (
            ('sandbox_timeout', self.sandbox_timeout),
            ('max_output_bytes', self.max_output_bytes),
            ('max_output_lines', self.max_output_lines),
            ('max_read_bytes', self.max_read_bytes),
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(f'{name} must be a positive integer, got {value!r}.')

        timeout = self.default_command_timeout
        if type(timeout) is bool or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError(f'default_command_timeout must be a positive finite number, got {timeout!r}.')

        ceiling = self.max_command_timeout
        if ceiling is not None and (type(ceiling) is not int or ceiling <= 0):
            raise ValueError(f'max_command_timeout must be a positive integer or None, got {ceiling!r}.')
        if not self.workdir or not posixpath.isabs(self.workdir):
            raise ValueError(f'workdir must be an absolute sandbox path, got {self.workdir!r}.')
        if type(self.allow_internet_access) is not bool:
            raise ValueError(f'allow_internet_access must be a boolean, got {self.allow_internet_access!r}.')
        if self.instructions is not None and type(self.instructions) is not str:
            raise ValueError(f'instructions must be a string or None, got {self.instructions!r}.')

    def _command_ceiling_hint(self, rejected: list[str]) -> str:
        """Redirect a rejected `sandbox_timeout` to the setting that works in reuse modes.

        `sandbox_timeout` is the natural-but-wrong reach for "let commands run longer" on a
        reused sandbox (it only sizes an owned sandbox's lifetime). The per-command ceiling
        there is `max_command_timeout`, so point the user at it instead of just rejecting.
        """
        if 'sandbox_timeout' not in rejected:
            return ''
        return ' To raise the per-command timeout ceiling on a reused sandbox, set `max_command_timeout`.'

    def _non_default_creation_settings(self) -> list[str]:
        return [
            name
            for name, value, default in (
                ('template', self.template, None),
                ('sandbox_timeout', self.sandbox_timeout, DEFAULT_SANDBOX_TIMEOUT),
                ('env', self.env, None),
                ('metadata', self.metadata, None),
                ('allow_internet_access', self.allow_internet_access, True),
            )
            if value != default
        ]

    def _non_default_owned_settings(self) -> list[str]:
        return [
            *self._non_default_creation_settings(),
            *(['workdir'] if self.workdir != DEFAULT_WORKDIR else []),
        ]

    def get_instructions(self) -> str | None:
        """Describe the configured sandbox lifecycle and deadlines to the model."""
        if self.instructions is not None:
            return self.instructions or None
        reused = self.sandbox_id is not None or self.session is not None
        template = _REUSED_INSTRUCTIONS if reused else _OWNED_INSTRUCTIONS
        # In reuse modes validation pins `sandbox_timeout` to its default, so this is the
        # documented 300s ceiling there and the exact sandbox lifetime when owned.
        ceiling = self.max_command_timeout if self.max_command_timeout is not None else self.sandbox_timeout
        default_timeout = min(max(1, math.ceil(self.default_command_timeout)), ceiling)
        return template.format(default_timeout=default_timeout, max_timeout=ceiling)

    def get_toolset(self) -> AgentToolset[AgentDepsT]:
        """Build the model-facing E2B toolset."""
        return E2BSandboxToolset[AgentDepsT](
            template=self.template,
            sandbox_id=self.sandbox_id,
            sandbox_timeout=self.sandbox_timeout,
            workdir=self.workdir,
            default_command_timeout=self.default_command_timeout,
            max_command_timeout=self.max_command_timeout,
            max_output_bytes=self.max_output_bytes,
            max_output_lines=self.max_output_lines,
            max_read_bytes=self.max_read_bytes,
            env=self.env,
            metadata=self.metadata,
            allow_internet_access=self.allow_internet_access,
            session=self.session,
        )
