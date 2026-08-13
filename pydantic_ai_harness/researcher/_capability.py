"""Complete research-agent harness assembled from regular capabilities."""

from pydantic_ai.capabilities import AbstractCapability, Capability, CombinedCapability, WebSearch
from pydantic_ai.tools import AgentDepsT

from pydantic_ai_harness.code_mode import CodeMode
from pydantic_ai_harness.tool_output_limits import ToolOutputLimits

DEFAULT_RESEARCHER_INSTRUCTIONS = """\
Search broadly before drawing conclusions.
Read the sources that support each important claim.
Prefer primary and authoritative sources.
Cite every factual claim with a direct source link.
Distinguish sourced facts from your own inference.
"""
"""Default instructions for `Researcher`."""


class Researcher(CombinedCapability[AgentDepsT]):
    """A complete research-agent harness built as a regular combined capability.

    See the class definition and [Researcher docs](https://pydantic.dev/docs/ai/harness/researcher/) for the exact composition.

    It comes with concise default instructions. Pass `instructions=` to
    replace them, or `instructions=None` to run with no default instructions.
    """

    def __init__(self, *, instructions: str | None = DEFAULT_RESEARCHER_INSTRUCTIONS) -> None:
        capabilities: list[AbstractCapability[AgentDepsT]] = []
        if instructions is not None:
            capabilities.append(Capability[AgentDepsT](instructions=instructions))
        capabilities.extend(
            [
                CodeMode[AgentDepsT](),
                WebSearch[AgentDepsT](local=True),
                ToolOutputLimits[AgentDepsT](),
            ]
        )
        super().__init__(capabilities)
