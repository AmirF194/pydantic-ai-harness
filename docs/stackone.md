---
title: StackOne
description: Give a Pydantic AI agent actions on the user's SaaS accounts (HRIS, ATS, CRM, and more) via the StackOne integration platform.
---

# StackOne

Actions on the user's SaaS accounts (HRIS, ATS, CRM, and more) for Pydantic AI agents, via [StackOne](https://www.stackone.com) -- an integration platform exposing actions across 400+ providers through linked accounts and a single MCP endpoint. The capability handles the StackOne-specific parts: API-key auth, account scoping, action filtering, tool modes, and usage instructions.

[Source](https://github.com/pydantic/pydantic-ai-harness/tree/main/pydantic_ai_harness/stackone/)

## Installation

```bash
pip install "pydantic-ai-harness[stackone]"
```

Set `STACKONE_API_KEY` (or pass `api_key=`). Keys and linked accounts are managed at <https://app.stackone.com>.

## Usage

```python {test="skip"}
from pydantic_ai import Agent
from pydantic_ai_harness.stackone import StackOne

agent = Agent(
    'openai:gpt-5.2',
    capabilities=[
        StackOne(account_id='45320', actions=['*_list_*']),
    ],
)
result = agent.run_sync('List the first 5 employees')
print(result.output)
```

A linked account is one authenticated connection to one provider.

## Tool modes

| `tool_mode` | Tools registered | Best for |
|---|---|---|
| `search_execute` | Two server-side meta-tools: search the catalog, execute an action by id | Any catalog size; constant prompt footprint |
| `individual` | One tool per enabled action, each with its own schema | Filtered or moderate catalogs; per-tool validation, filtering, approval |

The default (`tool_mode=None`) resolves to `search_execute`, or `individual` when `actions` are given. `actions` are case-insensitive `fnmatch` globs over full tool names, which follow `{connector}_{action}_{entity}` (for example `['*_list_*', 'bamboohr_get_employee']`); they only apply to individually registered tools. Provider catalogs can be large enough in `individual` mode to exceed model context windows, so constrain `individual` mode with `actions`. In `search_execute` mode individual action names never reach the agent, so `actions` cannot apply; explicitly requesting `search_execute` alongside `actions` raises an error.

## Agent spec (YAML/JSON)

The capability works with Pydantic AI's [agent spec](/ai/core-concepts/agent-spec/) format for defining agents in YAML or JSON. Keep the API key in the `STACKONE_API_KEY` environment variable rather than in the file:

```yaml
# agent.yaml
model: openai:gpt-5.2
capabilities:
  - StackOne:
      account_id: '45320'
      actions: ['*_list_*']
```

```python {test="skip"}
from pydantic_ai import Agent
from pydantic_ai_harness.stackone import StackOne

agent = Agent.from_file('agent.yaml', custom_capability_types=[StackOne])
```

Pass `custom_capability_types` so the spec loader knows how to instantiate `StackOne`.

The lower-level `StackOneToolset` is public for use with [`Agent(toolsets=[...])`](/ai/tools-toolsets/toolsets/) and core toolset combinators.

This capability's API may change while it stabilizes; breaking changes ship deprecation warnings where practical.

## API reference

::: pydantic_ai_harness.stackone.StackOne

::: pydantic_ai_harness.stackone.StackOneToolset
