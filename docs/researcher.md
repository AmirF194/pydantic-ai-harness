---
title: Researcher
description: A complete Pydantic AI web-research harness with Code Mode and bounded tool output.
---

# Researcher

`Researcher` gives a Pydantic AI agent a compact stack for broad web research with source-backed answers.
It is a regular combined capability made from the capabilities below, so you can use it as-is or take it apart.

Install the Code Mode dependencies, which also provide the local DuckDuckGo fallback:

```bash
uv add "pydantic-ai-harness[codemode]"
```

```python
from pydantic_ai import Agent
from pydantic_ai_harness.researcher import Researcher

agent = Agent('openai:gpt-5.6-sol', capabilities=[Researcher()])
```

To run the exported model-less agent with Pydantic AI's CLI:

```bash
uvx --with 'pydantic-ai-harness[codemode]' clai -a pydantic_ai_harness.researcher:researcher_agent
```

The `codemode` extra supplies Code Mode and local search. `clai` supplies its default model when `-m` is omitted; pass `-m provider:model` to select or override it.

It is literally core `Capability` with research instructions, [`CodeMode`](code-mode.md), core `WebSearch(native=False, local=True)`, and [`ToolOutputLimits`](tool-output-limits.md) combined. Core places `CodeMode` outermost when it normalizes capability ordering.

## Blown-out equivalent

```python
from pydantic_ai import Agent
from pydantic_ai.capabilities import Capability, WebSearch
from pydantic_ai_harness.code_mode import CodeMode
from pydantic_ai_harness.researcher import DEFAULT_RESEARCHER_INSTRUCTIONS
from pydantic_ai_harness.tool_output_limits import ToolOutputLimits

agent = Agent(
    'openai:gpt-5.6-sol',
    capabilities=[
        Capability(instructions=DEFAULT_RESEARCHER_INSTRUCTIONS),
        CodeMode(),
        WebSearch(native=False, local=True),
        ToolOutputLimits(),
    ],
)
```

See the [source](https://github.com/pydantic/pydantic-ai-harness/tree/main/pydantic_ai_harness/researcher/). The API may change between releases.

## API reference

::: pydantic_ai_harness.researcher.Researcher
