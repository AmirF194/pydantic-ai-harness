"""The `Coder` preset expanded into its component capabilities.

Run it from any repository root:

    uv run examples/coding_agent.py

Set `PYDANTIC_AI_MODEL` to change the model (defaults to Claude).
"""

import os
from pathlib import Path

from pydantic_ai import Agent
from pydantic_ai.capabilities import Capability
from pydantic_ai.models import Model

from pydantic_ai_harness import FileSystem, Shell
from pydantic_ai_harness.coder import DEFAULT_ALLOWED_COMMANDS, DEFAULT_CODER_INSTRUCTIONS
from pydantic_ai_harness.compaction import ClearToolResults, WarnNearLimits
from pydantic_ai_harness.planning import Planning
from pydantic_ai_harness.repo_context import RepoContext
from pydantic_ai_harness.subagents import SubAgent, SubAgents
from pydantic_ai_harness.tool_output_limits import ToolOutputLimits

DEFAULT_MODEL = os.environ.get('PYDANTIC_AI_MODEL', 'anthropic:claude-fable-5')


def build_agent(model: Model | str = DEFAULT_MODEL, workspace: Path | None = None) -> Agent:
    """Build the coding agent for `workspace` (defaults to the current directory)."""
    workspace = workspace or Path.cwd()
    read_tools = {'read_file', 'list_directory', 'search_files', 'find_files', 'file_info'}
    read_only = FileSystem(workspace).get_toolset().filtered(lambda ctx, tool: tool.name in read_tools)
    explorer = SubAgent(
        Agent(
            name='explorer',
            description='Explore the codebase and answer questions without modifying anything',
            instructions='Explore the codebase using read-only tools and answer with concrete evidence.',
            toolsets=[read_only],
        )
    )
    return Agent(
        model,
        capabilities=[
            Capability(instructions=DEFAULT_CODER_INSTRUCTIONS),
            FileSystem(root_dir=workspace),
            Shell(cwd=workspace, allowed_commands=DEFAULT_ALLOWED_COMMANDS),
            RepoContext(workspace_dir=workspace),
            Planning(),
            SubAgents(agents=[explorer], agent_folders=None),
            ClearToolResults(max_fraction=0.7),
            WarnNearLimits(max_context_fraction=0.9),
            ToolOutputLimits(),
        ],
    )


def main() -> None:
    """Start an interactive session in the current repository."""
    build_agent().to_cli_sync()


if __name__ == '__main__':
    main()
