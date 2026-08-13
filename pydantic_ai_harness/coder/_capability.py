"""Complete coding-agent harness assembled from regular capabilities."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from pydantic_ai import Agent
from pydantic_ai.capabilities import AbstractCapability, Capability, CombinedCapability
from pydantic_ai.tools import AgentDepsT

from pydantic_ai_harness.compaction import ClearToolResults, WarnNearLimits
from pydantic_ai_harness.filesystem import FileSystem
from pydantic_ai_harness.planning import Planning
from pydantic_ai_harness.repo_context import RepoContext
from pydantic_ai_harness.shell import Shell
from pydantic_ai_harness.subagents import SubAgent, SubAgents
from pydantic_ai_harness.tool_output_limits import ToolOutputLimits

DEFAULT_ALLOWED_COMMANDS: tuple[str, ...] = (
    'git',
    'rg',
    'grep',
    'find',
    'ls',
    'cat',
    'sed',
    'head',
    'tail',
    'python',
    'uv',
    'pytest',
    'ruff',
    'make',
)
"""Commands available to `Coder` unless an explicit allowlist is supplied."""

DEFAULT_CODER_INSTRUCTIONS = """\
Work step by step and keep the task's goal in view.
Read relevant files and repository instructions before editing.
Make the smallest coherent change that solves the task.
Keep the plan current during multi-step work.
Run the relevant tests and checks after editing.
Report what changed, verification results, and any remaining risks.
"""
"""Default instructions for `Coder`."""

_READ_TOOLS = frozenset({'read_file', 'list_directory', 'search_files', 'find_files', 'file_info'})


def _explorer(workspace: str | Path) -> SubAgent[AgentDepsT]:
    files = FileSystem[AgentDepsT](workspace).get_toolset().filtered(lambda ctx, tool: tool.name in _READ_TOOLS)
    agent = Agent[AgentDepsT](  # pyright: ignore[reportCallIssue, reportArgumentType]
        name='explorer',
        description='Explore the codebase and answer questions without modifying anything',
        instructions='Explore the codebase using read-only tools. Answer with concrete paths and evidence.',
        toolsets=[files],
    )
    return SubAgent(agent)


class Coder(CombinedCapability[AgentDepsT]):
    """A complete coding-agent harness.

    This is literally `Capability`, `FileSystem`, `Shell`, `RepoContext`,
    `Planning`, optional `SubAgents`, `ClearToolResults`, `WarnNearLimits`, and
    `ToolOutputLimits` combined.

    `Coder` is a regular combined capability. Use its component capabilities
    directly when you need a different composition.

    It comes with concise default instructions. Pass `instructions=` to
    replace them, or `instructions=None` to run with no default instructions.
    """

    def __init__(
        self,
        workspace: str | Path = '.',
        *,
        allowed_commands: Sequence[str] | None = None,
        subagents: Sequence[SubAgent[AgentDepsT]] | None = None,
        instructions: str | None = DEFAULT_CODER_INSTRUCTIONS,
    ) -> None:
        delegates = [_explorer(workspace)] if subagents is None else subagents
        capabilities: list[AbstractCapability[AgentDepsT]] = []
        if instructions is not None:
            capabilities.append(Capability[AgentDepsT](instructions=instructions))
        capabilities.extend(
            [
                FileSystem[AgentDepsT](workspace),
                Shell[AgentDepsT](
                    cwd=workspace,
                    allowed_commands=DEFAULT_ALLOWED_COMMANDS if allowed_commands is None else allowed_commands,
                ),
                RepoContext[AgentDepsT](workspace_dir=Path(workspace)),
                Planning[AgentDepsT](),
            ]
        )
        if delegates:
            capabilities.append(SubAgents[AgentDepsT](agents=delegates, agent_folders=None))
        capabilities.extend(
            [
                ClearToolResults[AgentDepsT](max_fraction=0.7),
                WarnNearLimits[AgentDepsT](max_context_fraction=0.9),
                ToolOutputLimits[AgentDepsT](),
            ]
        )
        super().__init__(capabilities)
