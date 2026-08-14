# Pydantic AI Harness

[![CI](https://github.com/pydantic/pydantic-ai-harness/actions/workflows/main.yml/badge.svg?event=push)](https://github.com/pydantic/pydantic-ai-harness/actions/workflows/main.yml?query=branch%3Amain)
[![PyPI](https://img.shields.io/pypi/v/pydantic-ai-harness.svg)](https://pypi.python.org/pypi/pydantic-ai-harness)
[![versions](https://img.shields.io/pypi/pyversions/pydantic-ai-harness.svg)](https://github.com/pydantic/pydantic-ai-harness)
[![license](https://img.shields.io/github/license/pydantic/pydantic-ai-harness.svg)](https://github.com/pydantic/pydantic-ai-harness/blob/main/LICENSE)

**Everything your [Pydantic AI](https://ai.pydantic.dev/) agent can do, one capability at a time.**

---

A harness is everything around the model that turns it into an agent: tools, context management, memory, delegation, safety. [Pydantic AI](https://ai.pydantic.dev/) ships the typed agent loop, [every model](https://ai.pydantic.dev/models/), and the [capability](https://ai.pydantic.dev/capabilities/) primitive; **Pydantic AI Harness** is its official capability library — 30+ building blocks, and complete harnesses assembled from them, maintained by the Pydantic AI team. Snap on one capability, compose your own stack from the blocks, or start from a complete coding agent and take it apart later — it's the same primitive all the way down.

## Quick start

```bash
uv add pydantic-ai-harness 'pydantic-ai-slim[anthropic,cli]'
```

```python
from pydantic_ai import Agent
from pydantic_ai_harness import Coder

agent = Agent('anthropic:claude-fable-5', capabilities=[Coder()])
agent.to_cli_sync()
```

Or run the exported agent directly: `uvx --with pydantic-ai-harness clai -a pydantic_ai_harness.coder:coder_agent`.

That's a complete coding agent in your terminal: workspace-rooted file access, allowlisted shell, repo orientation, planning, and context management that survives long sessions.

Every model works — swap the string for `'openai:gpt-5.6-sol'`, `'google:gemini-3-pro-preview'`, or [any other provider](https://ai.pydantic.dev/models/). Need more? Add capabilities to the list:

```python
from pydantic_ai import Agent
from pydantic_ai_harness import Coder, Memory
from pydantic_ai_harness.memory import FileStore

agent = Agent(
    'anthropic:claude-fable-5',
    capabilities=[Coder(), Memory(FileStore('.agent-memory'))],  # remembers across sessions
)
```

And it's an ordinary Pydantic AI agent: run it headless with `agent.run_sync(...)`, serve it as a [web chat UI](https://ai.pydantic.dev/web/), attach it to [your own frontend](https://ai.pydantic.dev/ui/), or hand it [structured output types](https://ai.pydantic.dev/output/).

## No magic: it's capabilities all the way down

`Coder` is not a framework inside the framework — it's a [`CombinedCapability`](https://ai.pydantic.dev/capabilities/custom/) bundling the same blocks you can use directly. The main blocks are spelled out below; the [Coder README](pydantic_ai_harness/coder/) includes the read-only explorer and exact equivalent:

```python
from pathlib import Path

from pydantic_ai import Agent
from pydantic_ai_harness import (
    DEFAULT_ALLOWED_COMMANDS,
    ClearToolResults,
    FileSystem,
    LLM_API_KEY_ENV_PATTERNS,
    Planning,
    RepoContext,
    Shell,
    SubAgent,
    SubAgents,
    ToolOutputLimits,
    WarnNearLimits,
)

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
        FileSystem('.'),  # read/write/edit/search, path-traversal safe
        Shell(
            cwd='.',
            allowed_commands=DEFAULT_ALLOWED_COMMANDS,
            denied_env_patterns=LLM_API_KEY_ENV_PATTERNS,
        ),
        RepoContext(workspace_dir=Path('.')),  # loads AGENTS.md/CLAUDE.md + repo structure
        Planning(),  # structured task plans the model maintains
        SubAgents(agents=[explorer], agent_folders=None),
        ClearToolResults(max_fraction=0.7),  # clears old tool results near the limit
        WarnNearLimits(max_context_fraction=0.9),  # warns the model before it hits limits
        ToolOutputLimits(),  # bounds oversized tool results
    ],
)
```

Start from the harness and remove what you don't want, or start from the blocks and build up — both are first-class. Constructor arguments (working directory, command allowlist, window sizes) thread through to the underlying capabilities.

## Capabilities

One index, both packages: every capability is a self-contained unit you drop into `capabilities=[...]`, and they all compose. The **Ships in** column says where each one lives — *core* comes with [`pydantic-ai`](https://github.com/pydantic/pydantic-ai) itself, *Harness* with this package. Grouped by what they give your agent:

### Harnesses

These harnesses are regular combined capabilities: one import gives you a working stack, and you can take either apart into the blocks below.

| Harness | Ships in | What it provides |
|---|---|---|
| [Coder](pydantic_ai_harness/coder/) | Harness | A complete coding-agent stack: files, shell, repo context, planning, a read-only explorer sub-agent, and context controls |
| [Researcher](pydantic_ai_harness/researcher/) | Harness | A complete web-research stack: web search and page fetching — native or local — with bounded tool output |

### Tools & environments

What the agent can touch — real systems, local or isolated.

| Capability | Ships in | What it does |
|---|---|---|
| [MCP](https://ai.pydantic.dev/capabilities/mcp/) | Core | Connect any MCP server's tools — local by default, provider-native connectors opt-in |
| [Shell](pydantic_ai_harness/shell/) | Harness | Command execution with allowlists, denylists, timeouts, and credential-stripping |
| [FileSystem](pydantic_ai_harness/filesystem/) | Harness | Read, write, edit, search files under a root — path-traversal and symlink safe, secrets read-only |
| [Modal Sandbox](pydantic_ai_harness/modal_sandbox/) | Harness | Commands and files in an isolated [Modal](https://modal.com) cloud sandbox |
| [LocalStack](pydantic_ai_harness/localstack/) | Harness | An emulated AWS environment with AWS CLI tools |
| [Browser Use](pydantic_ai_harness/browser_use/) | Harness | Hand web tasks to an autonomous [browser-use](https://github.com/browser-use/browser-use) agent driving a real browser |
| [StackOne](pydantic_ai_harness/stackone/) | Harness | Act on linked SaaS accounts (HRIS, ATS, CRM, …) via [StackOne](https://www.stackone.com) |
| [Macroscope](pydantic_ai_harness/macroscope/) | Harness | Run a local [Macroscope](https://docs.macroscope.com/cli) code review and hand the findings to the agent |

### Web & research

| Capability | Ships in | What it does |
|---|---|---|
| [Web Search](https://ai.pydantic.dev/capabilities/web-search/) | Core | Provider-native search where available, local DuckDuckGo fallback everywhere |
| [Web Fetch](https://ai.pydantic.dev/capabilities/web-fetch/) | Core | Fetch and read URLs — native or local |
| [X Search](https://ai.pydantic.dev/capabilities/x-search/) | Core | Search X — native on xAI, subagent fallback elsewhere |
| [Exa Search](pydantic_ai_harness/exa/) | Harness | Web research via [Exa](https://exa.ai): excerpted search, full-page reads, opt-in cited deep search |
| [Exa Agent](pydantic_ai_harness/exa/) | Harness | Delegate open-ended research to the Exa Agent API |

### Context efficiency

How the agent spends its context window — the difference between an agent that degrades over a long run and one that doesn't, and the difference between paying for tokens N times or once.

| Capability | Ships in | What it does |
|---|---|---|
| [Code Mode](pydantic_ai_harness/code_mode/) | Harness | The model writes one Python script that calls many tools inside a [Monty](https://github.com/pydantic/monty) sandbox — one round-trip instead of N, and intermediate results never enter the context window. The answer to tool-call token bloat |
| [Tool Search](https://ai.pydantic.dev/capabilities/tool-search/) | Core | Load tool definitions on demand instead of carrying hundreds in every prompt |
| [Compaction](pydantic_ai_harness/compaction/) | Harness | Tool-result clearing, sliding-window trimming, LLM summarization, tiered strategies — all window-relative, with live usage reporting |
| [Tool Output Limits](pydantic_ai_harness/tool_output_limits/) | Harness | Truncate, spill to a queryable file, or summarize oversized tool returns at the source |
| [Media](pydantic_ai_harness/media/) | Harness | Offload large binary/text parts to content-addressed stores (disk, SQLite, S3, Mongo) so they don't ride along in history |
| [Warn On Cache Busts](pydantic_ai_harness/warn_on_cache_busts/) | Harness | Detect prompt-cache prefix collapses between requests, from the provider's own numbers |

### Knowledge & memory

What the agent knows and remembers — loaded when relevant instead of carried in every prompt.

| Capability | Ships in | What it does |
|---|---|---|
| [Memory](pydantic_ai_harness/memory/) | Harness | A persistent, namespaced notebook: bounded prompt injection, on-demand search; in-memory/file/Postgres stores |
| [Conversation Search](pydantic_ai_harness/conversation_search/) | Harness | BM25 search over stored history — including turns compaction dropped |
| [Step Persistence](pydantic_ai_harness/step_persistence/) | Harness | Save, restore, resume (`continue_run`), and fork (`fork_run`) runs; file/SQLite/Mongo backends |
| [Skills](pydantic_ai_harness/skills/) | Harness | Load [Agent Skill](https://ai.pydantic.dev/capabilities/on-demand/) (`SKILL.md`) instructions on demand |
| [Repo Context](pydantic_ai_harness/repo_context/) | Harness | Start runs oriented: `AGENTS.md`/`CLAUDE.md` + repository structure |
| [Pydantic AI Docs](pydantic_ai_harness/pydantic_ai_docs/) | Harness | On-demand Pydantic AI documentation lookup |

### Delegation & planning

| Capability | Ships in | What it does |
|---|---|---|
| [Subagents](pydantic_ai_harness/subagents/) | Harness | Delegate self-contained tasks to named child agents |
| [Dynamic Workflow](pydantic_ai_harness/dynamic_workflow/) | Harness | The model orchestrates sub-agents from one Python script — fan-out, chain, vote in a single tool call, with hard `max_agent_calls` budgets |
| [Planning](pydantic_ai_harness/planning/) | Harness | Model-owned task plans with a cache-safe live reminder |
| [Advisor](pydantic_ai_harness/advisor/) | Harness | Let an executor consult a stronger model mid-run |

### Steering & safety

How you shape and bound what the agent does — before, during, and after the model acts.

| Capability | Ships in | What it does |
|---|---|---|
| [Guardrails](pydantic_ai_harness/guardrails/) | Harness | Validate/block/redact user input, tool calls, tool results, and output — including secret masking and parallel async guards |
| [Spend Limits](pydantic_ai_harness/spend/) | Harness | Cross-window USD/token budgets and per-response cost tracking, per model and per tenant |
| [System Reminders](pydantic_ai_harness/system_reminders/) | Harness | Cache-safe re-injection of guidance mid-run to counter instruction fade |
| [Managed Prompt](pydantic_ai_harness/logfire/) | Harness | Back instructions with a [Logfire](https://pydantic.dev/logfire)-managed prompt — version and roll out without redeploying |
| [Tool approval](https://ai.pydantic.dev/deferred-tools#human-in-the-loop-tool-approval) | Core | Flag tool calls that need human approval before they run |
| [Thinking](https://ai.pydantic.dev/capabilities/thinking/) | Core | Provider-adaptive extended thinking at configurable effort |

### Self-extension

| Capability | Ships in | What it does |
|---|---|---|
| [Capability Creation](pydantic_ai_harness/capability_creation/) | Harness | The agent writes, validates, and persists *new capabilities* during a run, loaded on the next run — self-extension with typed, inspectable units instead of arbitrary code |

And the agent plugs into any interface: [ACP](pydantic_ai_harness/experimental/acp/) *(experimental, Harness)* serves it to editors like Zed over the [Agent Client Protocol](https://agentclientprotocol.com), and core ships the [web chat UI](https://ai.pydantic.dev/web/), [CLI](https://ai.pydantic.dev/cli/), [frontend adapters](https://ai.pydantic.dev/ui/) (AG-UI, Vercel AI), and [realtime voice](https://ai.pydantic.dev/realtime/).

This index deliberately spans both packages — there's one capability system, and this repo is where new capabilities start before they [graduate into core](#when-do-you-need-the-harness). Community packages (by [vstorm-co](https://github.com/vstorm-co), [DougTrajano](https://github.com/DougTrajano/pydantic-ai-skills), and others) extend it further — see [third-party capabilities](https://ai.pydantic.dev/capabilities/third-party/).

## Composing from blocks

A research agent from regular capabilities -- this is literally [`Researcher`](pydantic_ai_harness/researcher/)'s composition, minus its short default instructions:

```python
from pydantic_ai import Agent
from pydantic_ai.capabilities import WebFetch, WebSearch
from pydantic_ai_harness import SubAgent, SubAgents, ToolOutputLimits

sub_researcher = SubAgent(
    Agent(
        name='researcher',
        description='Research a focused sub-question on the web and report back with findings and source links',
        capabilities=[WebSearch(local=True), WebFetch(local=True), ToolOutputLimits()],
    )
)

agent = Agent(
    'anthropic:claude-fable-5',
    capabilities=[
        WebSearch(local=True),  # native provider search, DuckDuckGo fallback elsewhere
        WebFetch(local=True),  # read the pages behind the results, native or local
        SubAgents(agents=[sub_researcher], agent_folders=None),
        ToolOutputLimits(),  # fetched pages don't flood the context
    ],
)

result = agent.run_sync('What changed in the top three Python agent frameworks this month? Cite sources.')
print(result.output)
#> ...
```

Everything is observable: `logfire.instrument_pydantic_ai()` gives you [a full trace of every run](https://logfire-us.pydantic.dev/public-trace/84bcf123-2106-49da-9f6f-5c26395339bb?spanId=7650806a0785b946) — every model call and tool call, with token and cost tracking. It's standard OpenTelemetry, so any OTLP backend works; [Logfire](https://pydantic.dev/logfire) is the easiest way to see it during development.

## When do you need the Harness?

Pydantic AI alone gives you the typed agent loop, structured output, and the fundamentals every agent needs — [web search](https://ai.pydantic.dev/capabilities/web-search/), [thinking](https://ai.pydantic.dev/capabilities/thinking/), [MCP](https://ai.pydantic.dev/capabilities/mcp/). Reach for the Harness when your agent should *do* more: touch files, run code, browse, remember, delegate, or stay coherent through hours-long runs. Capabilities start here, move fast under [0.x versioning](#version-policy), and graduate into core once they stabilize — so this repo is where the frontier of the harness lives.

## Installation

```bash
uv add pydantic-ai-harness
```

Some capabilities need an extra for their optional dependencies:

```bash
uv add "pydantic-ai-harness[codemode]"          # Code Mode (adds the Monty sandbox)
uv add "pydantic-ai-harness[dynamic-workflow]"  # Dynamic Workflow (adds the Monty sandbox)
uv add "pydantic-ai-harness[researcher]"        # Researcher (local search + page-fetch fallbacks)
uv add "pydantic-ai-harness[modal]"             # Modal Sandbox (adds the Modal SDK)
uv add "pydantic-ai-harness[logfire]"           # Managed Prompt (Logfire-managed prompts)
uv add "pydantic-ai-harness[exa]"               # Exa Search + Exa Agent (web research via the Exa API)
uv add "pydantic-ai-harness[skills]"            # Skills (loads SKILL.md frontmatter)
uv add "pydantic-ai-harness[browser-use]"       # Browser Use (autonomous web tasks; Python 3.11+)
uv add "pydantic-ai-harness[stackone]"          # StackOne (actions on linked business applications)
uv add "pydantic-ai-harness[acp]"               # ACP (serve an agent to editors)
uv add "pydantic-ai-harness[mongodb]"           # MongoDB backends for step persistence + media
```

The `code-mode` extra is also supported as an alias. Requires Python 3.10+ and `pydantic-ai-slim>=2.18.0`.

## Build your own

[Capabilities](https://ai.pydantic.dev/capabilities/#building-custom-capabilities) are the primary extension point for Pydantic AI, and every capability in this repo doubles as a worked example. Publishing a standalone package? Use the `pydantic-ai-<name>` naming convention — see [Publishing capability packages](https://ai.pydantic.dev/extensibility/#publishing-capability-packages).

## Contributing

We welcome capability contributions:

1. **Start with an issue.** [Open a capability request](https://github.com/pydantic/pydantic-ai-harness/issues/new?template=capability-request.yml) so we can discuss approach and priority before code is written.
2. **Then open a PR** and link the issue. We review based on community interest — upvotes on both count.
3. **Don't chase green CI.** Get the approach working and let us know; we may push to your branch or follow up, and you'll be credited as the original author. (See the [Pydantic AI contributing guide](https://github.com/pydantic/pydantic-ai/blob/main/CONTRIBUTING.md).)

> **Note**: PRs that modify `pyproject.toml` or `uv.lock` from non-team members are auto-closed by CI to prevent supply chain risk. If you need a new dependency, [open an issue](https://github.com/pydantic/pydantic-ai-harness/issues/new).

### Development

```bash
make install   # install dependencies
make format    # ruff format
make lint      # ruff check
make typecheck # pyright strict
make test      # pytest
make testcov   # pytest with 100% branch coverage
```

## Version policy

Pydantic AI Harness uses **0.x versioning** to signal that APIs are still stabilizing. During 0.x:

- **Minor releases** (0.1 -> 0.2) may include breaking changes — renamed parameters, changed defaults, restructured APIs — always with deprecation warnings where practical.
- **Patch releases** (0.1.0 -> 0.1.1) will not intentionally break existing behavior.
- **All breaking changes** are documented in release notes with migration guidance.

This is why the Harness is a separate package from [Pydantic AI](https://github.com/pydantic/pydantic-ai), which has a [stricter version policy](https://ai.pydantic.dev/version-policy/): capabilities iterate here at the speed the field moves, and graduate into core as they stabilize.

## Part of the Pydantic Stack

Everything you need to ship production-grade AI agents:

- [Pydantic AI](https://pydantic.dev/pydantic-ai?utm_source=github&utm_medium=readme&utm_campaign=pydantic-ai-harness) — the type-safe agent framework
- [Pydantic Logfire](https://pydantic.dev/logfire?utm_source=github&utm_medium=readme&utm_campaign=pydantic-ai-harness) — AI-first, full-stack observability
- [Logfire AI Gateway](https://pydantic.dev/ai-gateway?utm_source=github&utm_medium=readme&utm_campaign=pydantic-ai-harness) — unified LLM proxy
- [Pydantic Evals](https://ai.pydantic.dev/evals/) — evaluate any Python function, agents included
- [genai-prices](https://github.com/pydantic/genai-prices) — model pricing data, kept current

## License

MIT — see [LICENSE](LICENSE).
