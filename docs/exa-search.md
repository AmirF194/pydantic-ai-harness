---
title: Exa Search
description: Give a Pydantic AI agent web research tools backed by the Exa search API -- search with page text, full-page retrieval, and opt-in deep search, with page text capped per result to bound tool output.
---

# Exa Search

`ExaSearch` gives an agent web research tools backed by the
[Exa](https://exa.ai) search API: search that returns page text alongside each
hit, full-page retrieval for digging into a specific URL, and opt-in deep
search that synthesizes a cited answer in one call. Page text is capped per
result to bound tool output.

[Source](https://github.com/pydantic/pydantic-ai-harness/tree/main/pydantic_ai_harness/exa/)

## The problem

Search tools that return only titles and snippets force a second round of
fetching before the agent can judge a source. Wiring a search API together with
a page fetcher, capping page text so tool output doesn't overwhelm the model's
context, and prompting the agent to research methodically is boilerplate every
research agent reinvents.

`ExaSearch` bundles that plumbing into a single
[capability](/ai/core-concepts/capabilities/): the research tools, a per-result
page-text cap, and short research guidance in the system prompt.

## Usage

Install the `exa` extra and set the `EXA_API_KEY` environment variable (create
a key at <https://dashboard.exa.ai>):

```bash
uv add "pydantic-ai-harness[exa]"
```

Then pass `ExaSearch` to an `Agent` via the `capabilities` parameter:

```python
from pydantic_ai import Agent
from pydantic_ai_harness.exa import ExaSearch

agent = Agent('anthropic:claude-sonnet-4-6', capabilities=[ExaSearch()])

result = agent.run_sync('What changed in the latest stable Python release?')
print(result.output)
```

## Tools

`ExaSearch` contributes these tools to the agent:

| Tool | Purpose |
|---|---|
| `web_search` | Search the web and return the top `num_results` pages, each with title, URL, and page text. |
| `get_page` | Retrieve the text contents of one specific URL. |
| `deep_search` | Run Exa's multi-step deep search and return a synthesized, cited answer. Opt-in via `include_deep_search=True`. |

Page text is capped at `max_text_chars` characters per result: the cap is sent
to Exa as the contents limit and re-enforced when tool output is formatted, so
text stays bounded even with a custom client. The result count is bounded the
same way (`num_results` is requested from Exa and re-applied to the response).
When text is cut, the **head** is kept (a page's lead carries the substance)
and a `[... page text truncated at N characters]` marker is appended.

A URL or question that returns no content surfaces to the model as a
[`ModelRetry`](/ai/tools-toolsets/tools-advanced/#tool-retries) rather than a
hard error: the run continues and the model can correct the URL or rephrase.

## Deep search

`deep_search` calls Exa search with `type='deep'` and a plain-text output
schema: Exa expands the question into multiple queries, searches, and returns
an answer grounded in citations -- all in **one tool call**, with the cited
sources listed under the answer. Each call invests more time and search depth
than `web_search` (Exa's research-grade mode), and the model decides when to
invoke tools, so the tool is off by default -- enable it explicitly:

```python
from pydantic_ai_harness.exa import ExaSearch

ExaSearch(include_deep_search=True)
```

When enabled, the capability's instructions tell the model to treat it as an
escalation from `web_search`, not a replacement.

## Instructions

`ExaSearch` contributes short research guidance to the system prompt: search
wide with `web_search` first, read the most promising pages in full with
`get_page` before drawing conclusions, prefer primary sources, and cite the
URLs relied on. With `include_deep_search=True`, the guidance also covers when
to escalate to `deep_search`.

## Configuration

Every field of `ExaSearch` with its default:

```python
from pydantic_ai_harness.exa import ExaSearch

ExaSearch(
    num_results=5,              # results per web_search call
    max_text_chars=10_000,      # page-text cap per result, in characters
    include_deep_search=False,  # also expose the deep_search tool
    client=None,                # ExaClient -- None builds exa_py.AsyncExa from EXA_API_KEY
)
```

## Custom client

The default client is `exa_py.AsyncExa`, configured from the `EXA_API_KEY`
environment variable; when the variable is missing, construction fails with a
setup hint. Pass any object satisfying the `ExaClient` protocol -- the subset of
`AsyncExa` the toolset calls -- to configure authentication or the base URL
explicitly, or to substitute a fake in tests:

```python
from exa_py import AsyncExa

from pydantic_ai_harness.exa import ExaSearch

ExaSearch(client=AsyncExa(api_key='...'))
```

The API may change between releases while the capability settles; breaking
changes ship deprecation warnings where practical.

## ExaSearch vs Exa's MCP server

Exa also ships an official hosted MCP server at `https://mcp.exa.ai/mcp`
([exa-labs/exa-mcp-server](https://github.com/exa-labs/exa-mcp-server)) with
their full tool catalog: `web_search_exa`, `get_code_context_exa`,
`crawling_exa`, `company_research_exa`, `linkedin_search_exa`, and a
`deep_researcher_start` / `deep_researcher_check` pair.

`ExaSearch` is the curated, typed path: bounded output, a retry-on-empty
contract, bundled research instructions, and a client seam that makes it
testable offline. The MCP server is how you get Exa's full catalog with zero
wrapper code, via Pydantic AI core's
[MCP capability](/ai/core-concepts/capabilities/). Where the MCP deep
researcher is a start/check tool pair the model must poll across calls,
`deep_search` returns the answer in a single call. The two compose in one
`capabilities` list, and the tool names don't collide (the MCP tools are
`*_exa`-suffixed):

```python
from pydantic_ai import Agent
from pydantic_ai.capabilities import MCP
from pydantic_ai_harness.exa import ExaSearch

agent = Agent('anthropic:claude-sonnet-4-6', capabilities=[ExaSearch(), MCP('https://mcp.exa.ai/mcp')])
```

## Further reading

- [Pydantic AI capabilities](/ai/core-concepts/capabilities/)
- [Toolsets](/ai/tools-toolsets/toolsets/)
- [Exa API documentation](https://docs.exa.ai)

## API reference

::: pydantic_ai_harness.exa.ExaSearch
