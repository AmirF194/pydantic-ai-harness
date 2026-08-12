"""Complete research-agent preset assembled from harness capabilities."""

from pydantic_ai.capabilities import Capability, CombinedCapability, WebSearch
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
    """A complete research-agent preset.

    This is literally `Capability`, `CodeMode`, `WebSearch(native=False)`, and
    `ToolOutputLimits` combined.
    """

    def __init__(self, *, instructions: str | None = None) -> None:
        super().__init__(
            [
                Capability[AgentDepsT](
                    instructions=DEFAULT_RESEARCHER_INSTRUCTIONS if instructions is None else instructions
                ),
                CodeMode[AgentDepsT](),
                WebSearch[AgentDepsT](native=False, local=True),
                ToolOutputLimits[AgentDepsT](),
            ]
        )
