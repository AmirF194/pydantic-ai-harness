"""An interactive coding agent for the current repository.

Six capabilities turn a bare model into a working coding agent; each choice is
explained where it's made in `build_agent`. Run it from any repository root:

    uv run examples/coding_agent.py

Set `PYDANTIC_AI_MODEL` to change the model (defaults to Claude).
"""

import os
from pathlib import Path

from pydantic_ai import Agent
from pydantic_ai.models import Model

from pydantic_ai_harness import FileSystem, Shell
from pydantic_ai_harness.compaction import LimitWarner, SlidingWindow
from pydantic_ai_harness.context import RepoContext
from pydantic_ai_harness.planning import Planning

DEFAULT_MODEL = os.environ.get('PYDANTIC_AI_MODEL', 'anthropic:claude-opus-4-7')

INSTRUCTIONS = """\
You are a careful coding agent working in the user's repository.

- Before changing anything non-trivial, make a plan and keep it updated as you work.
- Read code before editing it. Prefer small, targeted edits over rewrites.
- After editing, verify: run the relevant tests or linters through the shell.
- Match the conventions of the surrounding code and any repo instruction files.
- If a task is ambiguous, state your assumption and proceed; don't stall.
"""


def build_agent(model: Model | str = DEFAULT_MODEL, workspace: Path | None = None) -> Agent:
    """Build the coding agent for `workspace` (defaults to the current directory)."""
    workspace = workspace or Path.cwd()
    return Agent(
        model,
        capabilities=[
            # Reads and edits are confined to the workspace; `.git`, `.env`, and
            # key/secret files are read-only by default.
            FileSystem(root_dir=workspace),
            # Anything not on this list is rejected before it runs.
            Shell(
                cwd=workspace,
                denied_commands=[],
                allowed_commands=['git', 'rg', 'ls', 'python', 'uv', 'pytest', 'ruff', 'make'],
                default_timeout=120.0,
            ),
            # Auto-loads AGENTS.md / CLAUDE.md so the agent follows repo conventions.
            RepoContext(workspace_dir=workspace),
            # A durable plan keeps multi-step tasks from drifting.
            Planning(),
            # Long sessions: trim old tool output, warn the model as limits approach.
            SlidingWindow(max_tokens=150_000, keep_messages=40),
            LimitWarner(max_context_tokens=180_000),
        ],
        instructions=INSTRUCTIONS,
    )


def main() -> None:
    """Start an interactive session in the current repository."""
    build_agent().to_cli_sync()


if __name__ == '__main__':
    main()
