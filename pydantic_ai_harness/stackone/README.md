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

A linked account is one authenticated connection to one provider. `actions` globs (case-insensitive `fnmatch` over the `{connector}_{action}_{entity}` tool names) control which actions the agent sees. The lower-level `StackOneToolset` is also public for `Agent(toolsets=[...])` and combinators like `approval_required()`; it carries no instructions of its own.

## Tool modes

| `tool_mode` | Tools registered | Best for |
|---|---|---|
| `search_execute` | Two server-side meta-tools: search the catalog, execute an action by id | Any catalog size; constant prompt footprint |
| `individual` | One tool per enabled action, each with its own schema | Filtered or moderate catalogs; per-tool validation, filtering, approval |

The default (`tool_mode=None`) resolves to `search_execute`, or `individual` when `actions` globs are given, since the globs only apply to individually registered tools. Provider catalogs can be large enough in `individual` mode to exceed model context windows, so constrain `individual` mode with `actions`, or use `defer_loading=True` (with a stable `id`) for Pydantic AI's [deferred tool loading](https://ai.pydantic.dev/deferred-tools/). In `search_execute` mode individual action names never reach the agent, so `actions` globs cannot apply (explicitly combining them warns), and combining `defer_loading` with `search_execute` stacks one discovery hop on another, which also warns.

## How it composes

- `code_mode=True` tags every tool with `code_mode=True` metadata for [`CodeMode`](../code_mode/README.md)`(tools={'code_mode': True})`.
- Wrap write actions (`*_create_*`, `*_update_*`, `*_delete_*`) with [tool approval](https://ai.pydantic.dev/deferred-tools/#human-in-the-loop-tool-approval).
- All configuration is agent-spec expressible; keep the API key in the environment, not the spec file:

```yaml
capabilities:
  - type: StackOne
    account_id: '45320'
    actions: ['*_list_*']
```

The API may change while the capability stabilizes; breaking changes ship deprecation warnings where practical.
