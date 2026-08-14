---
title: Researcher
description: A complete Pydantic AI web-research harness with source-backed answers.
---

# Researcher

`Researcher` gives a Pydantic AI agent a compact stack for broad web research with source-backed answers.
It is a regular [combined capability](https://pydantic.dev/docs/ai/capabilities/custom/#composition-and-middleware-semantics) made from the [capabilities](https://pydantic.dev/docs/ai/capabilities/overview/) below, so you can use it as-is or take it apart.

> The API may change between releases. Where practical, breaking changes ship with a deprecation warning.

## Usage

Install the local search and fetch fallbacks (DuckDuckGo search, page-to-Markdown fetching):

```bash
uv add "pydantic-ai-harness[researcher]"
```

Then ask it a question:

```python
from pydantic_ai import Agent
from pydantic_ai_harness import Researcher

agent = Agent('openai:gpt-5.6-sol', capabilities=[Researcher()])

result = agent.run_sync('What changed in the last three major releases of Django?')
print(result.output)
```

The same agent works with every Pydantic AI interface: [`agent.to_cli_sync()`](https://pydantic.dev/docs/ai/cli/#custom-agents) for terminal chat, [`agent.to_web()`](https://pydantic.dev/docs/ai/web/) for a browser chat UI.

Or skip the file entirely and run the exported [`researcher_agent`](#api-reference) with [Pydantic AI's CLI](https://pydantic.dev/docs/ai/cli/#custom-agents):

```bash
uvx --with 'pydantic-ai-harness[researcher]' clai -a pydantic_ai_harness.researcher:researcher_agent
```

`clai` supplies its default model when `-m` is omitted; pass `-m provider:model` to select or override it.

## What's inside

It is literally these capabilities combined, in this order:

- Concise default research instructions — see [Instructions](#instructions) below
- Core [`WebSearch(local=True)`](https://pydantic.dev/docs/ai/capabilities/web-search/) — the provider's native web search when the model supports it, with a local DuckDuckGo fallback when it doesn't
- Core [`WebFetch(local=True)`](https://pydantic.dev/docs/ai/capabilities/web-fetch/) — read the pages behind the results, native where supported with a local fallback, so claims can be checked against their sources
- [`ToolOutputLimits`](tool-output-limits.md) — bounds how much context any single tool result can consume

### Instructions

`Researcher` comes with short default research instructions (`DEFAULT_RESEARCHER_INSTRUCTIONS`, written out in full in the [blown-out equivalent](#blown-out-equivalent) below). Pass `instructions='...'` to replace them with your own, or `instructions=None` to get only the abilities, with no default instructions at all.

### Making it more powerful

- **Research in a specific format**: give the agent a typed [`output_type`](https://pydantic.dev/docs/ai/output/) — a Pydantic model of findings, each with its source link — and the researcher returns structured data instead of prose.
- **Higher-quality search**: swap in [`Exa Search`](exa-search.md) as the search backend.
- **Fan out**: add [`Dynamic Workflow`](dynamic-workflow.md) so the agent can spawn typed researcher sub-agents in parallel and combine their structured results.

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

See the [source](https://github.com/pydantic/pydantic-ai-harness/tree/main/pydantic_ai_harness/researcher/).

## API reference

::: pydantic_ai_harness.researcher.Researcher

::: pydantic_ai_harness.researcher.researcher_agent
