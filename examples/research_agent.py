"""Start a complete web research agent."""

import os

from pydantic_ai import Agent
from pydantic_ai.models import Model

from pydantic_ai_harness.researcher import Researcher

DEFAULT_MODEL = os.environ.get('PYDANTIC_AI_MODEL', 'openai:gpt-5.6-sol')


def build_agent(model: Model | str = DEFAULT_MODEL) -> Agent:
    """Build a web research agent."""
    return Agent(model, capabilities=[Researcher()])


if __name__ == '__main__':
    build_agent().to_cli_sync()
