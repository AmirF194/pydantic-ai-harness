"""Modal sandbox capability that gives agents a cloud sandbox to work in."""

from __future__ import annotations

from dataclasses import dataclass

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.tools import AgentDepsT
from pydantic_ai.toolsets import AgentToolset

from pydantic_ai_harness.experimental.modal_sandbox._toolset import ModalSandboxToolset

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

    Gives the agent tools to run commands and manage files inside a Modal sandbox —
    a safe place to execute untrusted or model-generated code without touching the
    host. By default each run gets a fresh sandbox created from `image` and torn
    down when the run ends; set `sandbox_id` to attach to a sandbox you manage
    yourself instead.

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

    image: str = 'python:3.12-slim'
    """Container image for owned sandboxes, as a registry tag (e.g. `python:3.12-slim`)."""

    sandbox_id: str | None = None
    """Attach to an existing sandbox by id instead of creating one. Attached sandboxes are not terminated.

    Attach is the only way to reuse a sandbox across runs today; keeping an owned sandbox
    warm across runs (outer-scope reuse) is not implemented yet.
    """

    app_name: str = 'pydantic-ai-harness'
    """Modal app the owned sandbox is created under."""

    create_app_if_missing: bool = True
    """If True, create the Modal app when it does not already exist."""

    sandbox_timeout: int = 300
    """Maximum lifetime in seconds of an owned sandbox before Modal shuts it down."""

    workdir: str | None = None
    """Working directory for commands inside an owned sandbox (Modal's default when None)."""

    default_timeout: float = 60.0
    """Default per-command timeout in seconds."""

    max_output_chars: int = 50_000
    """Maximum characters of output returned to the model."""

    include_instructions: bool = True
    """If True, add instructions telling the model how to use the sandbox."""

    def get_instructions(self) -> str | None:
        """Explain the sandbox to the model, unless disabled."""
        if not self.include_instructions:
            return None
        return _ATTACHED_INSTRUCTIONS if self.sandbox_id is not None else _OWNED_INSTRUCTIONS

    def get_toolset(self) -> AgentToolset[AgentDepsT]:
        """Build and return the Modal sandbox toolset."""
        return ModalSandboxToolset[AgentDepsT](
            image=self.image,
            sandbox_id=self.sandbox_id,
            app_name=self.app_name,
            create_app_if_missing=self.create_app_if_missing,
            sandbox_timeout=self.sandbox_timeout,
            workdir=self.workdir,
            default_timeout=self.default_timeout,
            max_output_chars=self.max_output_chars,
        )
