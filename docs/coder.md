---
title: Coder
description: A complete Pydantic AI coding-agent harness assembled from transparent capabilities.
---

# Coder

`Coder` gives a Pydantic AI agent a complete, opinionated stack for working in a local codebase.
It is a regular [combined capability](https://pydantic.dev/docs/ai/capabilities/custom/#composition-and-middleware-semantics) made from the [capabilities](https://pydantic.dev/docs/ai/capabilities/overview/) below, so you can use it as-is or take it apart.

> The API may change between releases. Where practical, breaking changes ship with a deprecation warning.

## Usage

Hand the agent a task directly:

```python
from pydantic_ai import Agent
from pydantic_ai_harness import Coder

agent = Agent('anthropic:claude-fable-5', capabilities=[Coder('.')])

result = agent.run_sync('Find out why tests/test_parser.py fails and fix the bug it caught.')
print(result.output)
```

The same agent works with every Pydantic AI interface: [`agent.to_cli_sync()`](https://pydantic.dev/docs/ai/cli/#using-the-cli-from-python) starts an interactive chat in your terminal, and [`agent.to_web()`](https://pydantic.dev/docs/ai/web/) serves a browser chat UI.

Or skip the file entirely and run the exported [`coder_agent`](#api-reference) with [Pydantic AI's CLI](https://pydantic.dev/docs/ai/cli/#custom-agents):

```bash
uvx --with pydantic-ai-harness clai -a pydantic_ai_harness.coder:coder_agent
```

`clai` supplies its default model when `-m` is omitted; pass `-m provider:model` to select or override it.

## What's inside

It is literally these capabilities combined, in this order:

- Concise default coding instructions — see [Instructions](#instructions) below
- [`FileSystem`](filesystem.md) — sandboxed read, write, edit, and search tools rooted at the workspace
- [`Shell`](shell.md) — allowlisted commands rooted at the workspace (`DEFAULT_ALLOWED_COMMANDS` is the default allowlist)
- [`RepoContext`](repo-context.md) — repository instructions and structure
- [`Planning`](planning.md) — a plan the agent creates and keeps current during multi-step work
- [`SubAgents`](subagents.md) — delegation, with a read-only `explorer` sub-agent by default
- [`ClearToolResults`](compaction.md) — clears stale tool results at 70% of the model context window
- [`WarnNearLimits`](compaction.md) — warns the agent at 90% of the model context window
- [`ToolOutputLimits`](tool-output-limits.md) — bounds how much context any single tool result can consume

Pass `subagents=[]` to disable delegation, or supply your own `SubAgent` entries.

### Instructions

`Coder` comes with short default coding instructions (`DEFAULT_CODER_INSTRUCTIONS`, written out in full in the [blown-out equivalent](#blown-out-equivalent) below). Pass `instructions='...'` to replace them with your own, or `instructions=None` to get only the abilities, with no default instructions at all.

### Not included by default

Other capabilities pair well with `Coder` — add them alongside it in `capabilities`:

- [Skills](skills.md) — reusable procedure documents the agent loads on demand
- [Memory](memory.md) — persistent memory across conversations
- [Conversation Search](conversation-search.md) — let the agent search earlier sessions
- [Guardrails](guardrails.md) — validate what the agent does before and after it acts
- [Dynamic Workflow](dynamic-workflow.md) — let the agent author multi-step workflows; best activated on demand

To remove or replace one of the built-in components instead, start from the blown-out form below and adjust the list.

## Blown-out equivalent

<!-- Keep this in sync with pydantic_ai_harness/coder — it intentionally shows the complete picture. -->

```python
from pathlib import Path

from pydantic_ai import Agent
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

instructions = """\
Work step by step and keep the task's goal in view.
Read relevant files and repository instructions before editing.
Make the smallest coherent change that solves the task.
Keep the plan current during multi-step work.
Run the relevant tests and checks after editing.
Report what changed, verification results, and any remaining risks.
"""

allowed_commands = [
    'git', 'rg', 'grep', 'find', 'ls', 'cat', 'sed', 'head', 'tail',
    'python', 'uv', 'pytest', 'ruff', 'make',
]

explorer = SubAgent(
    Agent(
        name='explorer',
        description='Explore the codebase and answer questions without modifying anything',
        instructions='Answer with concrete paths and evidence.',
        capabilities=[FileSystem('.', read_only=True), RepoContext(workspace_dir=Path('.'))],
    )
)

agent = Agent(
    'anthropic:claude-fable-5',
    instructions=instructions,
    capabilities=[
        FileSystem('.'),
        Shell(cwd='.', allowed_commands=allowed_commands),
        RepoContext(workspace_dir=Path('.')),
        Planning(),
        SubAgents(agents=[explorer], agent_folders=None),  # don't auto-load agent definitions from `agents/` folders
        ClearToolResults(max_fraction=0.7),
        WarnNearLimits(max_context_fraction=0.9),
        ToolOutputLimits(),
    ],
)
```

See the [source](https://github.com/pydantic/pydantic-ai-harness/tree/main/pydantic_ai_harness/coder/).

## API reference

::: pydantic_ai_harness.coder.Coder

::: pydantic_ai_harness.coder.coder_agent
