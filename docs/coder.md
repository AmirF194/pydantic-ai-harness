---
title: Coder
description: A complete Pydantic AI coding-agent harness assembled from transparent capabilities.
---

# Coder

`Coder` gives a Pydantic AI agent a complete, opinionated stack for working in a local codebase.
It is a regular combined capability made from the capabilities below, so you can use it as-is or take it apart.

```python
from pydantic_ai import Agent
from pydantic_ai_harness.coder import Coder

agent = Agent('anthropic:claude-fable-5', capabilities=[Coder('.')])
agent.to_cli_sync()
```

To run the exported model-less agent with Pydantic AI's CLI:

```bash
uvx --with pydantic-ai-harness clai -a pydantic_ai_harness.coder:coder_agent
```

`clai` supplies its default model when `-m` is omitted; pass `-m provider:model` to select or override it.

It is literally these capabilities combined, in this order:

- core `Capability` with concise coding instructions
- [`FileSystem`](filesystem.md) rooted at the workspace
- [`Shell`](shell.md) rooted at the workspace with an allowlist
- [`RepoContext`](repo-context.md) for repository instructions and structure
- [`Planning`](planning.md)
- [`SubAgents`](subagents.md) with a read-only `explorer` by default
- [`ClearToolResults`](compaction.md) at 70% of the model context window
- [`WarnNearLimits`](compaction.md) at 90% of the model context window
- [`ToolOutputLimits`](tool-output-limits.md)

Pass `subagents=[]` to disable delegation, or supply your own `SubAgent` entries. `DEFAULT_ALLOWED_COMMANDS` contains the default shell allowlist.

## Blown-out equivalent

```python
from pathlib import Path

from pydantic_ai import Agent
from pydantic_ai.capabilities import Capability
from pydantic_ai_harness.coder import DEFAULT_ALLOWED_COMMANDS, DEFAULT_CODER_INSTRUCTIONS
from pydantic_ai_harness.compaction import ClearToolResults, WarnNearLimits
from pydantic_ai_harness.filesystem import FileSystem
from pydantic_ai_harness.planning import Planning
from pydantic_ai_harness.repo_context import RepoContext
from pydantic_ai_harness.shell import Shell
from pydantic_ai_harness.subagents import SubAgent, SubAgents
from pydantic_ai_harness.tool_output_limits import ToolOutputLimits

read_tools = {'read_file', 'list_directory', 'search_files', 'find_files', 'file_info'}
read_only = FileSystem('.').get_toolset().filtered(lambda ctx, tool: tool.name in read_tools)
explorer = SubAgent(Agent(name='explorer', toolsets=[read_only]))

agent = Agent(
    'anthropic:claude-fable-5',
    capabilities=[
        Capability(instructions=DEFAULT_CODER_INSTRUCTIONS),
        FileSystem('.'),
        Shell(cwd='.', allowed_commands=DEFAULT_ALLOWED_COMMANDS),
        RepoContext(workspace_dir=Path('.')),
        Planning(),
        SubAgents(agents=[explorer], agent_folders=None),
        ClearToolResults(max_fraction=0.7),
        WarnNearLimits(max_context_fraction=0.9),
        ToolOutputLimits(),
    ],
)
```

See the [source](https://github.com/pydantic/pydantic-ai-harness/tree/main/pydantic_ai_harness/coder/). The API may change between releases.

## API reference

::: pydantic_ai_harness.coder.Coder
