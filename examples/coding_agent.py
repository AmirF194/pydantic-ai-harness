"""Start a complete coding agent in the current repository."""

import os
from pathlib import Path

from pydantic_ai import Agent
from pydantic_ai.models import Model

from pydantic_ai_harness.coder import Coder

DEFAULT_MODEL = os.environ.get('PYDANTIC_AI_MODEL', 'anthropic:claude-fable-5')


def build_agent(model: Model | str = DEFAULT_MODEL, workspace: Path | None = None) -> Agent:
    """Build a coding agent for `workspace`."""
    return Agent(model, capabilities=[Coder(workspace or Path.cwd())])


if __name__ == '__main__':
    build_agent().to_cli_sync()
