"""Modal sandbox capability that gives agents a cloud sandbox to work in."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.sandboxes import SandboxBackend, SandboxRef
from pydantic_ai.tools import AgentDepsT, RunContext
from pydantic_ai.toolsets import AgentToolset

from pydantic_ai_harness.modal_sandbox._backend import (
    DEFAULT_APP_NAME as _DEFAULT_APP_NAME,
)
from pydantic_ai_harness.modal_sandbox._backend import (
    DEFAULT_IMAGE as _DEFAULT_IMAGE,
)
from pydantic_ai_harness.modal_sandbox._backend import (
    DEFAULT_SANDBOX_TIMEOUT as _DEFAULT_SANDBOX_TIMEOUT,
)
from pydantic_ai_harness.modal_sandbox._backend import (
    PROVIDER,
    ModalSandboxBackend,
)
from pydantic_ai_harness.modal_sandbox._tool_output import DEFAULT_MAX_BYTES, DEFAULT_MAX_LINES
from pydantic_ai_harness.modal_sandbox._toolset import ModalSandboxToolset

# read_file pulls the whole file into memory before windowing it, so cap how large a file
# it will read; bigger files should be sliced with a shell command instead.
_DEFAULT_MAX_READ_BYTES = 5 * 1024 * 1024

# Default instruction templates; `get_instructions` fills in the per-instance timeout
# numbers so the model learns the real deadline and ceiling, not a placeholder.
_OWNED_INSTRUCTIONS = (
    'You have a Modal sandbox: an isolated, ephemeral cloud container. Use `run_command` to run '
    'shell commands in it, and `read_file` / `write_file` / `list_directory` to manage files. '
    'Commands run through `sh`, so pipes and redirection work. A command times out after '
    '{default_timeout}s unless you pass `timeout_seconds` (up to {max_timeout}s). The sandbox '
    'is reset between runs, so persist anything important outside it.'
)

_ATTACHED_INSTRUCTIONS = (
    'You have a Modal sandbox: an isolated cloud container. Use `run_command` to run shell '
    'commands in it, and `read_file` / `write_file` / `list_directory` to manage files. '
    'Commands run through `sh`, so pipes and redirection work. A command times out after '
    '{default_timeout}s unless you pass `timeout_seconds` (up to {max_timeout}s). This sandbox '
    'persists across runs, so files from earlier runs can still be present.'
)


@dataclass(kw_only=True)
class ModalSandbox(AbstractCapability[AgentDepsT]):
    """Access to an isolated cloud sandbox powered by [Modal](https://modal.com).

    Supplies the run's [`sandbox`][pydantic_ai.tools.RunContext.sandbox] and gives the agent
    tools to run commands and manage files inside it, a place to execute untrusted or
    model-generated code without touching the host. By default each run gets a fresh sandbox
    created from `image`, terminated when the run ends; `sandbox_timeout` is the server-side
    cleanup backstop. Set `sandbox_id` to attach to a sandbox you manage elsewhere, which the
    capability uses but never terminates.

    Because the capability supplies the sandbox rather than owning a private connection, the
    tools work against whatever sandbox the run ends up with: a `sandbox=` run argument (or a
    sandbox from an earlier capability in the list) wins over this one, and the tools use that
    instead.

    Requires the `modal` extra (`uv add "pydantic-ai-harness[modal]"`) and Modal
    credentials, configured as for the Modal CLI: run `modal token new` once, or set
    `MODAL_TOKEN_ID` / `MODAL_TOKEN_SECRET` in the environment.

    ```python
    from pydantic_ai import Agent
    from pydantic_ai_harness.modal_sandbox import ModalSandbox

    agent = Agent('anthropic:claude-sonnet-4-6', capabilities=[ModalSandbox()])
    result = agent.run_sync('Write a Python script that prints the first 10 primes and run it.')
    print(result.output)
    ```
    """

    image: str = _DEFAULT_IMAGE
    """Container image for owned sandboxes, as a registry tag (e.g. `python:3.12-slim`)."""

    sandbox_id: str | None = None
    """Attach to an existing sandbox by id instead of creating one. Attached sandboxes are not terminated.

    Use this to reuse a sandbox created elsewhere (e.g. via the Modal CLI) across runs. The
    settings that only apply when creating a sandbox (`image`, `app_name`,
    `create_app_if_missing`, `sandbox_timeout`, `workdir`, `env`) cannot be combined with it.
    Like any shared sandbox, it is not concurrency-safe across overlapping runs.
    """

    app_name: str = _DEFAULT_APP_NAME
    """Modal app the owned sandbox is created under."""

    create_app_if_missing: bool = True
    """If True, create the Modal app when it does not already exist."""

    sandbox_timeout: int = _DEFAULT_SANDBOX_TIMEOUT
    """Maximum lifetime in seconds of an owned sandbox before Modal shuts it down.

    This bounds the whole sandbox; `default_command_timeout` bounds a single command.
    """

    workdir: str | None = None
    """Working directory for commands inside an owned sandbox (Modal's default when None)."""

    env: Mapping[str, str] | None = None
    """Environment variables to set in an owned sandbox.

    Owned sandboxes only. To inject secrets or env into an attached sandbox, set them when you
    create that sandbox yourself (e.g. with `modal.Secret`).
    """

    default_command_timeout: float = 60.0
    """Default timeout in seconds for one `run_command`, used when the model omits one.

    This bounds a single command; `sandbox_timeout` bounds the whole sandbox's lifetime.
    Modal enforces whole-second deadlines, so fractional values are rounded up (0.5
    behaves as 1).
    """

    max_command_timeout: int | None = None
    """Hard ceiling in seconds for any single `run_command`, including a model-supplied
    `timeout_seconds`. None falls back to `sandbox_timeout`.

    Modal has no per-command kill, so a cancelled command keeps running until its deadline;
    this caps how long that worst case can be. An owned command cannot outlive
    `sandbox_timeout` anyway, so the default ceiling is exact for owned sandboxes.

    For an attached sandbox the fallback is still `sandbox_timeout`, which is pinned to its
    default (300s) there because the capability does not know the real lifetime of a sandbox
    it did not create. So every command there is capped at 300s unless you set
    `max_command_timeout` to the value the sandbox actually allows.
    """

    max_output_bytes: int = DEFAULT_MAX_BYTES
    """Maximum payload retained per command stream or file read, measured in UTF-8 bytes.

    For commands the cap applies to stdout and stderr separately, so a large stderr cannot
    crowd out stdout. Labels, truncation notes, continuation offsets, timeouts, and exit codes
    add a small amount beyond this payload limit. Whichever of `max_output_bytes` and
    `max_output_lines` is reached first wins. This bounds what reaches the model, not what
    crosses the wire: the sandbox protocol delivers a command's whole output.
    """

    max_output_lines: int = DEFAULT_MAX_LINES
    """Maximum payload lines retained per command stream or file read, alongside `max_output_bytes`.

    A second cap so many short lines cannot pile up under the byte budget. Whichever cap is
    reached first wins. Labels and truncation or status notes can add lines beyond this
    payload limit. Both caps proxy a context budget; a future token-based cap would be additive.
    """

    max_read_bytes: int = _DEFAULT_MAX_READ_BYTES
    """Largest file `read_file` will read whole; larger files are refused with a hint to use shell tools.

    The tool checks metadata before reading and checks the returned byte count again, but a
    file that grows between those operations can briefly exceed this value in client memory
    before it is rejected.
    """

    instructions: str | None = None
    """Instructions telling the model how to use the sandbox, added to the system prompt.

    Leave as `None` for a default that matches the mode (fresh sandbox per run, or a
    reused one that can carry files from earlier runs) and states the command timeout
    and its ceiling. Set `''` to add no instructions, or pass your own text -- e.g. when
    wrapping with `PrefixTools`, so the tool names in the text match the prefixed ones.
    """

    _backends: dict[tuple[str | None, str], ModalSandboxBackend] = field(
        default_factory=dict[tuple[str | None, str], ModalSandboxBackend], init=False, repr=False, compare=False
    )
    """Live handles keyed by `(run_id, sandbox_id)`.

    `get_sandbox` receives only a serializable `SandboxRef`, so without this every run would
    look the sandbox up again to use it and a third time to tear it down. Keying by run rather
    than by sandbox alone keeps two runs sharing one attached sandbox from detaching each
    other's connection.
    """

    def __post_init__(self) -> None:
        """Reject settings that the chosen mode would ignore, so a dead value can't mislead.

        There are two modes: owned (the default) and attach (`sandbox_id`). Attach reuses an
        existing sandbox, so the owned-only creation settings have no effect there. Rather than
        ignore a conflicting value, fail at construction with the names to remove.
        """
        self._validate_configuration()
        if self.env is not None:
            self.env = dict(self.env)

        if self.sandbox_id is None:
            # Owned mode: a command cannot outlive the sandbox, so a ceiling above the
            # sandbox lifetime is a dead value -- reject it like the other mode conflicts.
            # In attach mode a higher ceiling is the documented escape hatch for sandboxes
            # whose real lifetime exceeds the pinned default, so no check there.
            ceiling = self.max_command_timeout
            if ceiling is not None and ceiling > self.sandbox_timeout:
                raise ValueError(
                    f'max_command_timeout ({ceiling}) cannot exceed sandbox_timeout '
                    f'({self.sandbox_timeout}) for an owned sandbox: the sandbox is reaped '
                    'before such a command could finish. Raise sandbox_timeout instead.'
                )
            return
        ignored = self._non_default_owned_settings()
        if ignored:
            raise ValueError(
                f'{", ".join(ignored)} only apply when creating a sandbox, but `sandbox_id` attaches '
                'to an existing one. Remove them, or drop `sandbox_id` to create a sandbox.'
                + self._command_ceiling_hint(ignored)
            )

    def _validate_configuration(self) -> None:
        """Reject limits that Modal cannot enforce with the documented semantics."""
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

        # Validated like the numeric fields because the agent-spec path does not type-check
        # custom-capability dataclass fields; a bad YAML value should fail here, not deep
        # in the agent build.
        if self.instructions is not None and type(self.instructions) is not str:
            raise ValueError(f'instructions must be a string or None, got {self.instructions!r}.')

    def _command_ceiling_hint(self, rejected: list[str]) -> str:
        """Redirect a rejected `sandbox_timeout` to the setting that works in attach mode.

        `sandbox_timeout` is the natural-but-wrong reach for "let commands run longer" on a
        reused sandbox (it only sizes an owned sandbox's lifetime). The per-command ceiling
        there is `max_command_timeout`, so point the user at it instead of just rejecting.
        """
        if 'sandbox_timeout' not in rejected:
            return ''
        return ' To raise the per-command timeout ceiling on a reused sandbox, set `max_command_timeout`.'

    def _non_default_owned_settings(self) -> list[str]:
        """Names of the owned-sandbox creation settings left at a non-default value."""
        return [
            name
            for name, value, default in (
                ('image', self.image, _DEFAULT_IMAGE),
                ('app_name', self.app_name, _DEFAULT_APP_NAME),
                ('create_app_if_missing', self.create_app_if_missing, True),
                ('sandbox_timeout', self.sandbox_timeout, _DEFAULT_SANDBOX_TIMEOUT),
                ('workdir', self.workdir, None),
                ('env', self.env, None),
            )
            if value != default
        ]

    def get_instructions(self) -> str | None:
        """Explain the sandbox to the model, unless overridden or disabled via `instructions`."""
        if self.instructions is not None:
            return self.instructions or None
        # An attached sandbox can carry files from earlier runs; only a per-run owned sandbox
        # starts clean each time.
        template = _ATTACHED_INSTRUCTIONS if self.sandbox_id is not None else _OWNED_INSTRUCTIONS
        # Report the deadline the toolset will actually apply (quantized and clamped, see
        # `ModalSandboxToolset._command_timeout`), so the numbers cannot contradict behavior.
        ceiling = self.max_command_timeout if self.max_command_timeout is not None else self.sandbox_timeout
        default_timeout = min(max(1, math.ceil(self.default_command_timeout)), ceiling)
        return template.format(default_timeout=default_timeout, max_timeout=ceiling)

    async def create_sandbox(self, ctx: RunContext[AgentDepsT]) -> SandboxRef:
        """Provision this run's Modal sandbox, or name the one to attach to.

        Modal has no create-or-reuse key for sandboxes, so an owned sandbox is created on every
        call. Under a durable engine that retries this unit after a crash, the sandbox from the
        abandoned attempt is left to its own `sandbox_timeout`.
        """
        if self.sandbox_id is not None:
            # Attach mode contributes an identity without provisioning anything; `get_sandbox`
            # connects to it and `destroy_sandbox` leaves it running.
            return SandboxRef(provider=PROVIDER, sandbox_id=self.sandbox_id)
        backend = await ModalSandboxBackend.create(
            image=self.image,
            app_name=self.app_name,
            create_app_if_missing=self.create_app_if_missing,
            sandbox_timeout=self.sandbox_timeout,
            workdir=self.workdir,
            env=self.env,
        )
        self._backends[(ctx.run_id, backend.sandbox_id)] = backend
        return SandboxRef(provider=PROVIDER, sandbox_id=backend.sandbox_id)

    async def get_sandbox(self, ctx: RunContext[AgentDepsT], ref: SandboxRef) -> SandboxBackend | None:
        """Connect to a Modal sandbox by reference, reusing this run's handle when it has one."""
        if ref.provider != PROVIDER:
            return None
        key = (ctx.run_id, ref.sandbox_id)
        backend = self._backends.get(key)
        if backend is None:
            backend = await ModalSandboxBackend.connect(ref.sandbox_id)
            self._backends[key] = backend
        return backend

    async def destroy_sandbox(self, ctx: RunContext[AgentDepsT], ref: SandboxRef) -> None:
        """Terminate an owned sandbox and release this run's handle to it.

        An attached sandbox is only disconnected from: its owner decides when it goes away. A
        handle this process never held (a run that resumed elsewhere) leaves Modal's
        `sandbox_timeout` as the backstop.
        """
        backend = self._backends.pop((ctx.run_id, ref.sandbox_id), None)
        if backend is not None:
            await backend.close(terminate=self.sandbox_id is None)

    def get_toolset(self) -> AgentToolset[AgentDepsT]:
        """Build and return the Modal sandbox toolset."""
        return ModalSandboxToolset[AgentDepsT](
            sandbox_timeout=self.sandbox_timeout,
            default_command_timeout=self.default_command_timeout,
            max_command_timeout=self.max_command_timeout,
            max_output_bytes=self.max_output_bytes,
            max_output_lines=self.max_output_lines,
            max_read_bytes=self.max_read_bytes,
        )
