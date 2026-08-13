"""Build a coding agent from the blocks packaged as `Coder`.

Run the packaged equivalent without assembling the blocks:

    uvx --with pydantic-ai-harness clai -a pydantic_ai_harness.coder:coder_agent
"""

import os
from pathlib import Path

from pydantic_ai import Agent
from pydantic_ai.capabilities import Capability
from pydantic_ai.models import Model

from pydantic_ai_harness import FileSystem, Shell
from pydantic_ai_harness.compaction import ClearToolResults, WarnNearLimits
from pydantic_ai_harness.planning import Planning
from pydantic_ai_harness.repo_context import RepoContext
from pydantic_ai_harness.subagents import SubAgent, SubAgents
from pydantic_ai_harness.tool_output_limits import ToolOutputLimits

DEFAULT_MODEL = os.environ.get('PYDANTIC_AI_MODEL', 'anthropic:claude-fable-5')

# Keep these written-out blocks in sync with the defaults in `pydantic_ai_harness.coder`.
ALLOWED_COMMANDS = (
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
INSTRUCTIONS = """\
Work step by step and keep the task's goal in view.
Read relevant files and repository instructions before editing.
Make the smallest coherent change that solves the task.
Keep the plan current during multi-step work.
Run the relevant tests and checks after editing.
Report what changed, verification results, and any remaining risks.
"""


def build_agent(model: Model | str = DEFAULT_MODEL, workspace: Path | None = None) -> Agent:
    """Build the coding agent for `workspace` (defaults to the current directory)."""
    workspace = workspace or Path.cwd()
    explorer = SubAgent(
        Agent(
            name='explorer',
            description='Explore the codebase and answer questions without modifying anything',
            instructions='Answer with concrete paths and evidence.',
            capabilities=[FileSystem(workspace, read_only=True), RepoContext(workspace_dir=workspace)],
        )
    )
    return Agent(
        model,
        capabilities=[
            Capability(instructions=INSTRUCTIONS),  # Set a focused working method.
            FileSystem(root_dir=workspace),  # Provide workspace-scoped file operations.
            Shell(cwd=workspace, allowed_commands=ALLOWED_COMMANDS),  # Run common development commands.
            RepoContext(workspace_dir=workspace),  # Add repository metadata to the context.
            Planning(),  # Track progress through multi-step work.
            SubAgents(agents=[explorer], agent_folders=None),  # Delegate read-only exploration.
            ClearToolResults(max_fraction=0.7),  # Reclaim context from old tool results.
            WarnNearLimits(max_context_fraction=0.9),  # Warn before the context window fills.
            ToolOutputLimits(),  # Bound large tool responses.
        ],
    )


def main() -> None:
    """Start an interactive session in the current repository."""
    build_agent().to_cli_sync()


if __name__ == '__main__':
    main()
