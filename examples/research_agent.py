"""A research agent that produces a cited report from web sources.

`CodeMode` lets the model batch many searches and fetches into one sandboxed
round-trip, and `OverflowingToolOutput` keeps oversized pages out of the context
window. The report comes back as a typed `ResearchReport` with per-claim
citations.

    uv run examples/research_agent.py "What is post-quantum TLS and who has deployed it?"

Requires the `code-mode` extra and pydantic-ai's `duckduckgo` extra for the
local web-search fallback.
"""

import os
import sys

from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.capabilities import WebSearch
from pydantic_ai.models import Model

from pydantic_ai_harness import CodeMode
from pydantic_ai_harness.overflowing_tool_output import OverflowingToolOutput

DEFAULT_MODEL = os.environ.get('PYDANTIC_AI_MODEL', 'anthropic:claude-opus-4-7')


class Citation(BaseModel):
    """A source backing one or more claims in the report."""

    url: str
    title: str
    supports: str = Field(description='The claim in the report this source supports.')


class ResearchReport(BaseModel):
    """A short, sourced answer to the research question."""

    answer: str = Field(description='Three to six paragraphs answering the question.')
    key_facts: list[str] = Field(description='The load-bearing facts, one sentence each.')
    citations: list[Citation]
    confidence: float = Field(ge=0, le=1, description='How well the sources agree, 0-1.')


INSTRUCTIONS = """\
You are a research agent. For each question:

1. Search broadly first -- multiple query phrasings in one `run_code` call -- then
   read the most promising sources.
2. Prefer primary sources; when two sources disagree, say so and lower confidence.
3. Every key fact must trace to a citation. No citation, no claim.
4. Stop when additional sources stop changing the answer.
"""


def build_agent(model: Model | str = DEFAULT_MODEL) -> Agent[None, ResearchReport]:
    """Build the research agent."""
    return Agent(
        model,
        capabilities=[
            # One run_code call replaces N tool calls; searches run inside the sandbox.
            CodeMode(),
            # native=False routes through the local search fallback so CodeMode can
            # batch searches; on providers with server-side search, drop it to use theirs.
            WebSearch(native=False, local='duckduckgo'),
            # Big pages get spilled to disk with a preview; the model reads more on demand.
            OverflowingToolOutput(),
        ],
        instructions=INSTRUCTIONS,
        output_type=ResearchReport,
    )


def main() -> None:
    """Research the question given on the command line."""
    question = ' '.join(sys.argv[1:]) or 'What is post-quantum TLS and who has deployed it?'
    report = build_agent().run_sync(question).output
    print(report.answer)
    print(f'\nconfidence: {report.confidence:.0%}')
    for c in report.citations:
        print(f'- {c.title} <{c.url}>')


if __name__ == '__main__':
    main()
