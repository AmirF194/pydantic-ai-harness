"""Modal sandbox capability that gives agents a cloud sandbox to work in."""

from __future__ import annotations

from dataclasses import dataclass

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.tools import AgentDepsT
from pydantic_ai.toolsets import AgentToolset

from pydantic_ai_harness._tool_output import DEFAULT_MAX_LINES
from pydantic_ai_harness.experimental.modal_sandbox._session import ModalSandboxSession
from pydantic_ai_harness.experimental.modal_sandbox._toolset import ModalSandboxToolset

# Defaults shared by the field declarations and the validation below, so the two cannot
# drift: a setting is "left at its default" iff it equals the constant here.
_DEFAULT_IMAGE = 'python:3.12-slim'
_DEFAULT_APP_NAME = 'pydantic-ai-harness'
_DEFAULT_SANDBOX_TIMEOUT = 300
# read_file pulls the whole file into memory before windowing it, so cap how large a file
# it will read; bigger files should be sliced with a shell command instead.
_DEFAULT_MAX_READ_BYTES = 5 * 1024 * 1024

_OWNED_INSTRUCTIONS = (
    'You have a Modal sandbox: an isolated, ephemeral cloud container. Use `run_command` to run '
    'shell commands in it, and `read_file` / `write_file` / `list_directory` to manage files. '
    'Commands run through `sh`, so pipes and redirection work. The sandbox is reset between '
    'sessions, so persist anything important outside it.'
)

_ATTACHED_INSTRUCTIONS = (
    'You have a Modal sandbox: an isolated cloud container. Use `run_command` to run shell '
    'commands in it, and `read_file` / `write_file` / `list_directory` to manage files. '
    'Commands run through `sh`, so pipes and redirection work. This sandbox persists across '
    'sessions, so files from earlier runs can still be present.'
)


@dataclass
class ModalSandbox(AbstractCapability[AgentDepsT]):
    """Access to an isolated cloud sandbox powered by [Modal](https://modal.com).

    Gives the agent tools to run commands and manage files inside a Modal sandbox,
    a place to execute untrusted or model-generated code without touching the host.
    By default each run gets a fresh sandbox created from `image` and torn down when
    the run ends. To keep one sandbox across runs, either set `sandbox_id` to attach
    to a sandbox you manage elsewhere, or pass a `session` you own (an open
    `ModalSandboxSession`) so you control its lifetime and can read its `sandbox_id`.
    The capability never opens or terminates a `session` you pass.

    Requires the `modal` extra (`pip install "pydantic-ai-harness[modal]"`) and
    Modal credentials in the environment (`MODAL_TOKEN_ID` / `MODAL_TOKEN_SECRET`).

    ```python
    from pydantic_ai import Agent
    from pydantic_ai_harness.experimental.modal_sandbox import ModalSandbox

    agent = Agent('anthropic:claude-sonnet-4-6', capabilities=[ModalSandbox()])
    result = agent.run_sync('Write a Python script that prints the first 10 primes and run it.')
    print(result.output)
    ```
    """

    image: str = _DEFAULT_IMAGE
    """Container image for owned sandboxes, as a registry tag (e.g. `python:3.12-slim`)."""

    sandbox_id: str | None = None
    """Attach to an existing sandbox by id instead of creating one. Attached sandboxes are not terminated.

    Use this to reuse a sandbox created elsewhere (e.g. via the Modal CLI). The settings
    that only apply when creating a sandbox (`image`, `app_name`, `create_app_if_missing`,
    `sandbox_timeout`, `workdir`) cannot be combined with `sandbox_id`.
    """

    session: ModalSandboxSession | None = None
    """Use a sandbox session you own and keep open across runs, instead of a per-run one.

    Pass an already-entered `ModalSandboxSession` to reuse one sandbox across runs while
    controlling its lifetime yourself: the capability uses it but never opens or terminates
    it. Cannot be combined with `sandbox_id` or the owned-sandbox creation settings (the
    session already owns those). Like `sandbox_id`, a shared session is not concurrency-safe
    across overlapping runs.
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

    env: dict[str, str] | None = None
    """Environment variables to set in an owned sandbox.

    Owned sandboxes only. To inject secrets or env into an attached or injected sandbox,
    set them when you create that sandbox yourself (e.g. with `modal.Secret`).
    """

    default_command_timeout: float = 60.0
    """Default timeout in seconds for one `run_command`, used when the model omits one.

    This bounds a single command; `sandbox_timeout` bounds the whole sandbox's lifetime.
    """

    max_command_timeout: float | None = None
    """Hard ceiling in seconds for any single `run_command`, including a model-supplied
    `timeout_seconds`. None falls back to `sandbox_timeout`.

    Modal has no per-command kill, so a cancelled command keeps running until its deadline;
    this caps how long that worst case can be. An owned command cannot outlive
    `sandbox_timeout` anyway, so the default ceiling is exact for owned sandboxes.

    For an attached or injected sandbox the fallback is still `sandbox_timeout`, which is
    pinned to its default (300s) in those modes because the capability does not know the
    real lifetime of a sandbox it did not create. So every command there is capped at 300s
    unless you set `max_command_timeout` to the value the sandbox actually allows.
    """

    max_output_bytes: int = 50 * 1024
    """Maximum output returned to the model, measured in UTF-8 bytes.

    Caps what `run_command` and `read_file` hand back. Bytes (not characters) to size the
    model's context precisely and to match `max_read_bytes`. This caps the returned text,
    not the sandbox-side capture: Modal buffers a command's full output before we read it,
    so `max_output_bytes` bounds context while `max_read_bytes` bounds a file read's
    transfer. Whichever of `max_output_bytes` and `max_output_lines` is reached first wins.
    """

    max_output_lines: int = DEFAULT_MAX_LINES
    """Maximum lines of output returned to the model, alongside `max_output_bytes`.

    A second cap so many short lines cannot pile up under the byte budget. Whichever cap is
    reached first wins. Both caps proxy a context budget; a future token-based cap would be
    additive.
    """

    max_read_bytes: int = _DEFAULT_MAX_READ_BYTES
    """Largest file `read_file` will read whole; larger files are refused with a hint to use shell tools.

    `read_file` loads the whole file before windowing it, so this caps the memory and
    transfer a single read can cost.
    """

    include_instructions: bool = True
    """If True, add instructions telling the model how to use the sandbox."""

    def __post_init__(self) -> None:
        """Reject settings that the chosen mode would ignore, so a dead value can't mislead.

        There are three modes: owned (the default), attach (`sandbox_id`), and injected
        (`session`). Attach and injected both reuse an existing sandbox, so the owned-only
        creation settings have no effect there; `session` also subsumes `sandbox_id`. Rather
        than ignore a conflicting value, fail at construction with the names to remove.
        """
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
        if self.sandbox_id is None:
            return
        ignored = self._non_default_owned_settings()
        if ignored:
            raise ValueError(
                f'{", ".join(ignored)} only apply when creating a sandbox, but `sandbox_id` attaches '
                'to an existing one. Remove them, or drop `sandbox_id` to create a sandbox.'
                + self._command_ceiling_hint(ignored)
            )

    def _command_ceiling_hint(self, rejected: list[str]) -> str:
        """Redirect a rejected `sandbox_timeout` to the setting that works in reuse modes.

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
        """Explain the sandbox to the model, unless disabled."""
        if not self.include_instructions:
            return None
        # A reused sandbox (attach or injected session) can carry files from earlier runs;
        # only a per-run owned sandbox starts clean each time.
        reused = self.sandbox_id is not None or self.session is not None
        return _ATTACHED_INSTRUCTIONS if reused else _OWNED_INSTRUCTIONS

    def get_toolset(self) -> AgentToolset[AgentDepsT]:
        """Build and return the Modal sandbox toolset."""
        return ModalSandboxToolset[AgentDepsT](
            image=self.image,
            sandbox_id=self.sandbox_id,
            app_name=self.app_name,
            create_app_if_missing=self.create_app_if_missing,
            sandbox_timeout=self.sandbox_timeout,
            workdir=self.workdir,
            default_command_timeout=self.default_command_timeout,
            max_command_timeout=self.max_command_timeout,
            max_output_bytes=self.max_output_bytes,
            max_output_lines=self.max_output_lines,
            max_read_bytes=self.max_read_bytes,
            env=self.env,
            session=self.session,
        )
