# Researcher

`Researcher` gives a Pydantic AI agent a compact stack for broad web research with source-backed answers.
It is a regular combined capability made from the capabilities below, so you can use it as-is or take it apart.

Install the local search and fetch fallbacks (DuckDuckGo search, page-to-Markdown fetching):

```bash
uv add "pydantic-ai-harness[researcher]"
```

```python
from pydantic_ai import Agent
from pydantic_ai_harness import Researcher

agent = Agent('openai:gpt-5.6-sol', capabilities=[Researcher()])

result = agent.run_sync('What changed in the last three major releases of Django?')
print(result.output)
```

The same agent works with every Pydantic AI interface: `agent.to_cli_sync()` for terminal chat, `agent.to_web()` for a browser chat UI.

Or skip the file entirely and run the exported `researcher_agent` with Pydantic AI's CLI:

```bash
uvx --with 'pydantic-ai-harness[researcher]' clai -a pydantic_ai_harness.researcher:researcher_agent
```

`clai` supplies its default model when `-m` is omitted; pass `-m provider:model` to select or override it.

It is literally these capabilities combined, in this order:

- Concise default research instructions (`DEFAULT_RESEARCHER_INSTRUCTIONS`) — pass `instructions='...'` to replace them, or `instructions=None` to disable
- Core `WebSearch(local=True)` — the provider's native web search when the model supports it, with a local DuckDuckGo fallback when it doesn't
- Core `WebFetch(local=True)` — read the pages behind the results, native where supported with a local fallback, so claims can be checked against their sources
- `ToolOutputLimits` — bounds how much context any single tool result can consume

## Blown-out equivalent

<!-- Keep this in sync with pydantic_ai_harness/researcher — it intentionally shows the complete picture. -->

```python
from pydantic_ai import Agent
from pydantic_ai.capabilities import WebFetch, WebSearch
from pydantic_ai_harness import ToolOutputLimits

instructions = """\
Search broadly before drawing conclusions.
Read the sources that support each important claim.
Prefer primary and authoritative sources.
Cite every factual claim with a direct source link.
Distinguish sourced facts from your own inference.
"""

agent = Agent(
    'openai:gpt-5.6-sol',
    instructions=instructions,
    capabilities=[
        WebSearch(local=True),  # native provider search, DuckDuckGo fallback on models without it
        WebFetch(local=True),  # read the pages behind the results, native or local
        ToolOutputLimits(),
    ],
)
```

See the [source](https://github.com/pydantic/pydantic-ai-harness/tree/main/pydantic_ai_harness/researcher/). The API may change between releases.
