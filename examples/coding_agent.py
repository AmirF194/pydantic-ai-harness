"""Build a coding agent from the blocks packaged as `Coder`.

Run the packaged equivalent without assembling the blocks:

    uvx --with pydantic-ai-harness clai -a pydantic_ai_harness.coder:coder_agent
"""

import os
from pathlib import Path

from pydantic_ai import Agent
from pydantic_ai.models import Model

from pydantic_ai_harness import (
    ClearToolResults,
    FileSystem,
    Planning,
    RepoContext,
    Shell,
    SubAgent,
    SubAgents,
    ToolOutputLimits,
    WarnNearLimits,
)

DEFAULT_MODEL = os.environ.get('PYDANTIC_AI_MODEL', 'anthropic:claude-fable-5')

# Keep this written-out allowlist in sync with the default in `pydantic_ai_harness.coder`.
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
