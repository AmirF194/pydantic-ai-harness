# Coder

`Coder` gives a Pydantic AI agent a complete, opinionated stack for working in a local codebase.
It is a regular combined capability made from the capabilities below, so you can use it as-is or take it apart.

```python
from pydantic_ai import Agent
from pydantic_ai_harness import Coder

agent = Agent('anthropic:claude-fable-5', capabilities=[Coder('.')])

result = agent.run_sync('Find out why tests/test_parser.py fails and fix the bug it caught.')
print(result.output)
```

The same agent works with every Pydantic AI interface: `agent.to_cli_sync()` starts an interactive chat in your terminal, and `agent.to_web()` serves a browser chat UI.

Or skip the file entirely and run the exported `coder_agent` with Pydantic AI's CLI:

```bash
uvx --with pydantic-ai-harness clai -a pydantic_ai_harness.coder:coder_agent
```

`clai` supplies its default model when `-m` is omitted; pass `-m provider:model` to select or override it.

It is literally these capabilities combined, in this order:

- `FileSystem` — sandboxed read, write, edit, and search tools rooted at the workspace
- `Shell` — allowlisted commands rooted at the workspace (`DEFAULT_ALLOWED_COMMANDS` is the default allowlist)
- `RepoContext` — repository instructions and structure
- `Planning` — a plan the agent creates and keeps current during multi-step work
- `SubAgents` — delegation, with a read-only `explorer` sub-agent by default
- `ClearToolResults` — clears stale tool results at 70% of the model context window
- `WarnNearLimits` — warns the agent at 90% of the model context window
- `ToolOutputLimits` — bounds how much context any single tool result can consume

Pass `subagents=[]` to disable delegation, or supply your own `SubAgent` entries.

`Coder` ships with **no default instructions**: modern models don't need procedural coaching, and each composed capability already contributes its own tool guidance. Pass `instructions='...'` to add your own — identity, tone, or house rules.

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

See the [source](https://github.com/pydantic/pydantic-ai-harness/tree/main/pydantic_ai_harness/coder/). The API may change between releases.
