# Exa Search

Give an agent web research tools backed by the [Exa](https://exa.ai) search
API: search that returns the most relevant excerpts from each hit, full-page
retrieval for digging into a specific URL, and opt-in deep search that
synthesizes a cited answer in one call.

[Source](https://github.com/pydantic/pydantic-ai-harness/tree/main/pydantic_ai_harness/exa/)

## Installation

```bash
uv add "pydantic-ai-harness[exa]"
```

Set the `EXA_API_KEY` environment variable (create a key at
<https://dashboard.exa.ai>), or pass a configured client (see
[Custom client](#custom-client)).

## The problem

Search tools that return only titles and snippets force a second round of
fetching before the agent can judge a source, while search tools that return
full page text flood the context with pages the agent will discard. Wiring a
search API together with a page fetcher, budgeting what each tool returns, and
prompting the agent to research methodically is boilerplate every research
agent reinvents.

## The solution

`ExaSearch` bundles the tools with output budgeting and short research
guidance in the system prompt: survey cheaply with excerpts, then read the
pages that matter in full.

```python
from pydantic_ai import Agent
from pydantic_ai_harness.exa import ExaSearch

agent = Agent('anthropic:claude-sonnet-4-6', capabilities=[ExaSearch()])

result = agent.run_sync('What changed in the latest stable Python release?')
print(result.output)
```

## Tools

| Tool | Purpose |
|---|---|
| `web_search` | Search the web and return the top `num_results` pages, each with title, URL, and its most relevant excerpts. |
| `get_page` | Retrieve the full text of one specific URL -- a promising `web_search` hit, or a URL the user provided. |
| `deep_search` | Run Exa's multi-step deep search and return a synthesized, cited answer. Opt-in via `include_deep_search=True`. |

`web_search` returns short excerpts (Exa highlights) rather than full page
text, following [Exa's own guidance for agents](https://exa.ai/docs/reference/search-api-guide-for-coding-agents),
so surveying several sources stays cheap; the agent reads a chosen page with
`get_page`.

`get_page` text is capped at `max_text_chars` characters, keeping the **head**
(a page's lead carries the substance). One character of headroom above the cap
is requested from Exa, so when a page exceeds the cap the output ends with a
`[... page text truncated at N characters]` marker; at the API ceiling of
10,000 characters no headroom exists, so the marker cannot appear there. The
result count is bounded the same way: `num_results` is requested from Exa and
re-applied to the response.

A URL or question that returns no content, a rate limit, or a transient API or
network failure surfaces to the model as a `ModelRetry` (the model can correct
the URL, rephrase, or try again) rather than aborting the run. Authentication
failures (401/403) are configuration errors and propagate.

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
escalation from `web_search`, not a replacement. The synthesized answer is
returned in full (it is Exa-generated and inherently bounded); `max_text_chars`
only applies to `get_page`.

## Instructions

`ExaSearch` contributes short research guidance to the system prompt: search
wide with `web_search` first, read the most promising pages in full with
`get_page` before drawing conclusions, prefer primary sources, and cite the
URLs relied on. With `include_deep_search=True`, the guidance also covers when
to escalate to `deep_search`. Set `guidance` to replace the default text, or
to `''` to contribute no instructions at all.

## Configuration

Every field of `ExaSearch` with its default:

```python
from pydantic_ai_harness.exa import ExaSearch

ExaSearch(
    num_results=5,              # results per web_search call (1 to 100)
    max_text_chars=10_000,      # get_page text cap, in characters (1 to 10,000)
    include_deep_search=False,  # also expose the deep_search tool
    include_domains=[],         # only search these domains (allowlist)
    exclude_domains=[],         # never search these domains (denylist)
    guidance=None,              # None = default instructions, '' = none, str = custom
    client=None,                # ExaClient -- None builds exa_py.AsyncExa from EXA_API_KEY
)
```

`include_domains` and `exclude_domains` apply to `web_search` and
`deep_search`, and are mutually exclusive -- set one, not both. Out-of-range
limits and setting both domain lists raise a `ValueError` at construction.

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

## ExaSearch vs core `WebSearch`

Pydantic AI core ships a provider-adaptive
[`WebSearch`](https://ai.pydantic.dev/capabilities/#provider-adaptive-tools)
capability: on models with a native search tool it uses the provider's own
search, executed server-side; elsewhere it falls back to a local DuckDuckGo
tool. Reach for it when you want search that follows the model.

Reach for `ExaSearch` when you want the same search behavior on every model:
one vendor, excerpts with every hit, explicit page retrieval, domain filters,
and opt-in deep search.

One caveat when combining them: on Anthropic models the provider-native search
tool is also named `web_search` on the wire, so
`capabilities=[WebSearch(), ExaSearch()]` puts two tools with the same name in
the request. Use one search capability per agent on native-search models, or
force the local fallback with `WebSearch(native=False)` (its DuckDuckGo tool is
named `duckduckgo_search`, which does not collide).

## ExaSearch vs Exa's MCP server

Exa also ships an official hosted MCP server at `https://mcp.exa.ai/mcp`
([exa-labs/exa-mcp-server](https://github.com/exa-labs/exa-mcp-server)). By
default it exposes `web_search_exa` and `web_fetch_exa`; the full catalog adds
`web_search_advanced_exa` and an agent-run set (`agent_create_run`,
`agent_wait_for_run`, `agent_get_run_output`, `agent_cancel_run`).

`ExaSearch` is the curated, typed path: bounded output, a retry-on-empty
contract, bundled research instructions, and a client seam that makes it
testable offline. The MCP server is how you get Exa's full catalog with zero
wrapper code, via Pydantic AI core's MCP capability. Their agent runs are
create-then-poll (`agent_create_run` returns an ID immediately;
`agent_wait_for_run` polls it), where `deep_search` returns the answer in a
single call. The two compose in one `capabilities` list, and none of the MCP
tool names collide with `web_search`, `get_page`, or `deep_search`:

```python
from pydantic_ai import Agent
from pydantic_ai.capabilities import MCP
from pydantic_ai_harness.exa import ExaSearch

agent = Agent('anthropic:claude-sonnet-4-6', capabilities=[ExaSearch(), MCP('https://mcp.exa.ai/mcp')])
```

## Agent spec (YAML/JSON)

`ExaSearch` works with Pydantic AI's
[agent spec](https://ai.pydantic.dev/agent-spec/):

```yaml
# agent.yaml
model: anthropic:claude-sonnet-4-6
capabilities:
  - ExaSearch:
      num_results: 3
      include_deep_search: true
```

```python
from pydantic_ai import Agent
from pydantic_ai_harness.exa import ExaSearch

agent = Agent.from_file('agent.yaml', custom_capability_types=[ExaSearch])
```

Pass `custom_capability_types` so the spec loader knows how to instantiate
`ExaSearch`. The `client` field is not spec-serializable; spec-loaded instances
always build the default client from `EXA_API_KEY`.

## Further reading

- [Pydantic AI capabilities](https://ai.pydantic.dev/capabilities/)
- [Toolsets](https://ai.pydantic.dev/toolsets/)
- [Exa API documentation](https://docs.exa.ai)
