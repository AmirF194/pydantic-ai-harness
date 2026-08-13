"""Build a research agent from the blocks packaged as `Researcher`.

Run the packaged equivalent without assembling the blocks:

    uvx --with 'pydantic-ai-harness[codemode]' clai -a pydantic_ai_harness.researcher:researcher_agent
"""

import os

from pydantic_ai import Agent
from pydantic_ai.capabilities import Capability, WebSearch
from pydantic_ai.models import Model

from pydantic_ai_harness.code_mode import CodeMode
from pydantic_ai_harness.tool_output_limits import ToolOutputLimits

DEFAULT_MODEL = os.environ.get('PYDANTIC_AI_MODEL', 'openai:gpt-5.6-sol')

# Keep these written-out blocks in sync with the defaults in `pydantic_ai_harness.researcher`.
INSTRUCTIONS = """\
Search broadly before drawing conclusions.
Read the sources that support each important claim.
Prefer primary and authoritative sources.
Cite every factual claim with a direct source link.
Distinguish sourced facts from your own inference.
"""


def build_agent(model: Model | str = DEFAULT_MODEL) -> Agent:
    """Build a web research agent."""
    return Agent(
        model,
        capabilities=[
            Capability(instructions=INSTRUCTIONS),  # Set evidence and citation expectations.
            CodeMode(),  # Let the model orchestrate research tools in code.
            WebSearch(native=False, local=True),  # Provide portable local web search.
            ToolOutputLimits(),  # Bound large search responses.
        ],
    )


def main() -> None:
    """Start an interactive research session."""
    build_agent().to_cli_sync()


if __name__ == '__main__':
    main()
