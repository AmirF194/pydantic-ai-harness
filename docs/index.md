---
title: Pydantic AI Harness
description: The official capability library for Pydantic AI -- 30+ building blocks and complete harnesses assembled from them, from a coding agent to your own custom stack.
---

# Pydantic AI Harness

**Everything your [Pydantic AI](/ai/) agent can do, one capability at a time.**

A harness is everything around the model that turns it into an agent: tools, context management, memory, delegation, safety. [Pydantic AI](/ai/) ships the typed agent loop, [every model](/ai/models/overview/), and the [capability](/ai/capabilities/overview/) primitive; **Pydantic AI Harness** is its official capability library — 30+ building blocks, and complete harnesses assembled from them, maintained by the Pydantic AI team. Snap on one capability, compose your own stack from the blocks, or start from a complete coding agent and take it apart later — it's the same primitive all the way down.

## Quick start

```bash
uv add pydantic-ai-harness
```

```python
from pydantic_ai import Agent
from pydantic_ai_harness import Coder

agent = Agent('anthropic:claude-fable-5', capabilities=[Coder()])
agent.to_cli_sync()
```

Or run the exported agent directly: `uvx --with pydantic-ai-harness clai -a pydantic_ai_harness.coder:coder_agent`.

That's a complete coding agent in your terminal: workspace-rooted file access, allowlisted shell, repo orientation, planning, and context management that survives long sessions.

Every model works — swap the string for `'openai:gpt-5.6-sol'`, `'google:gemini-3-pro-preview'`, or [any other provider](/ai/models/overview/). Need more? Add capabilities to the list:

```python
from pydantic_ai import Agent
from pydantic_ai_harness import Coder, Memory

agent = Agent(
    'anthropic:claude-fable-5',
    capabilities=[Coder(), Memory()],  # remembers across sessions
)
```

And it's an ordinary Pydantic AI agent: run it headless with `agent.run_sync(...)`, serve it as a [web chat UI](/ai/web/), attach it to [your own frontend](/ai/ui/overview/), or hand it [structured output types](/ai/core-concepts/output/).

## No magic: it's capabilities all the way down

`Coder` is not a framework inside the framework — it's a [`CombinedCapability`](/ai/capabilities/custom/) bundling the same blocks you can use directly. The main blocks are spelled out below; the [Coder page](coder.md) includes the read-only explorer and exact equivalent:

```python
from pathlib import Path

from pydantic_ai import Agent
from pydantic_ai_harness import (
    DEFAULT_ALLOWED_COMMANDS,
    ClearToolResults,
    FileSystem,
    Planning,
    RepoContext,
    Shell,
    ToolOutputLimits,
    WarnNearLimits,
)

agent = Agent(
    'anthropic:claude-fable-5',
    capabilities=[
        FileSystem('.'),  # read/write/edit/search, path-traversal safe
        Shell(cwd='.', allowed_commands=DEFAULT_ALLOWED_COMMANDS),  # allowlisted command execution
        RepoContext(workspace_dir=Path('.')),  # loads AGENTS.md/CLAUDE.md + repo structure
        Planning(),  # structured task plans the model maintains
        ClearToolResults(max_fraction=0.7),  # clears old tool results near the limit
        WarnNearLimits(max_context_fraction=0.9),  # warns the model before it hits limits
        ToolOutputLimits(),  # bounds oversized tool results
    ],
)
```

Start from the harness and remove what you don't want, or start from the blocks and build up — both are first-class. Constructor arguments (working directory, command allowlist, window sizes) thread through to the underlying capabilities.

## Capabilities

One index, both packages: every capability is a self-contained unit you drop into `capabilities=[...]`, and they all compose. The **Ships in** column says where each one lives — *core* comes with [`pydantic-ai`](/ai/) itself, *Harness* with this package. Grouped by what they give your agent:

### Harnesses

These harnesses are regular combined capabilities: one import gives you a working stack, and you can take either apart into the blocks below.

| Harness | Ships in | What it provides |
|---|---|---|
| [Coder](coder.md) | Harness | A complete coding-agent stack: files, shell, repo context, planning, a read-only explorer sub-agent, and context controls |
| [Researcher](researcher.md) | Harness | A complete web-research stack: web search and page fetching — native or local — with bounded tool output |

### Tools & environments

What the agent can touch — real systems, local or isolated.

| Capability | Ships in | What it does |
|---|---|---|
| [MCP](/ai/capabilities/mcp/) | Core | Connect any MCP server's tools — local by default, provider-native connectors opt-in |
| [Shell](shell.md) | Harness | Command execution with allowlists, denylists, timeouts, and credential-stripping |
| [FileSystem](filesystem.md) | Harness | Read, write, edit, search files under a root — path-traversal and symlink safe, secrets read-only |
| [Modal Sandbox](modal-sandbox.md) | Harness | Commands and files in an isolated [Modal](https://modal.com) cloud sandbox |
| [LocalStack](localstack.md) | Harness | An emulated AWS environment with AWS CLI tools |
| [Browser Use](browser-use.md) | Harness | Hand web tasks to an autonomous [browser-use](https://github.com/browser-use/browser-use) agent driving a real browser |
| [StackOne](stackone.md) | Harness | Act on linked SaaS accounts (HRIS, ATS, CRM, …) via [StackOne](https://www.stackone.com) |
| [Macroscope](macroscope.md) | Harness | Run a local [Macroscope](https://docs.macroscope.com/cli) code review and hand the findings to the agent |

### Web & research

| Capability | Ships in | What it does |
|---|---|---|
| [Web Search](/ai/capabilities/web-search/) | Core | Provider-native search where available, local DuckDuckGo fallback everywhere |
| [Web Fetch](/ai/capabilities/web-fetch/) | Core | Fetch and read URLs — native or local |
| [X Search](/ai/capabilities/x-search/) | Core | Search X — native on xAI, subagent fallback elsewhere |
| [Exa Search](exa-search.md) | Harness | Web research via [Exa](https://exa.ai): excerpted search, full-page reads, opt-in cited deep search |
| [Exa Agent](exa-search.md) | Harness | Delegate open-ended research to the Exa Agent API |

### Context efficiency

How the agent spends its context window — the difference between an agent that degrades over a long run and one that doesn't, and the difference between paying for tokens N times or once.

| Capability | Ships in | What it does |
|---|---|---|
| [Code Mode](code-mode.md) | Harness | The model writes one Python script that calls many tools inside a [Monty](https://github.com/pydantic/monty) sandbox — one round-trip instead of N, and intermediate results never enter the context window. The answer to tool-call token bloat |
| [Tool Search](/ai/capabilities/tool-search/) | Core | Load tool definitions on demand instead of carrying hundreds in every prompt |
| [Compaction](compaction.md) | Harness | Tool-result clearing, sliding-window trimming, LLM summarization, tiered strategies — all window-relative, with live usage reporting |
| [Tool Output Limits](tool-output-limits.md) | Harness | Truncate, spill to a queryable file, or summarize oversized tool returns at the source |
| [Media](media.md) | Harness | Offload large binary/text parts to content-addressed stores (disk, SQLite, S3, Mongo) so they don't ride along in history |
| [Warn On Cache Busts](warn-on-cache-busts.md) | Harness | Detect prompt-cache prefix collapses between requests, from the provider's own numbers |

### Knowledge & memory

What the agent knows and remembers — loaded when relevant instead of carried in every prompt.

| Capability | Ships in | What it does |
|---|---|---|
| [Memory](memory.md) | Harness | A persistent, namespaced notebook: bounded prompt injection, on-demand search; in-memory/file/Postgres stores |
| [Conversation Search](conversation-search.md) | Harness | BM25 search over stored history — including turns compaction dropped |
| [Step Persistence](step-persistence.md) | Harness | Save, restore, resume (`continue_run`), and fork (`fork_run`) runs; file/SQLite/Mongo backends |
| [Skills](skills.md) | Harness | Load [Agent Skill](/ai/capabilities/on-demand/) (`SKILL.md`) instructions on demand |
| [Repo Context](repo-context.md) | Harness | Start runs oriented: `AGENTS.md`/`CLAUDE.md` + repository structure |
| [Pydantic AI Docs](pydantic-ai-docs.md) | Harness | On-demand Pydantic AI documentation lookup |

### Delegation & planning

| Capability | Ships in | What it does |
|---|---|---|
| [Subagents](subagents.md) | Harness | Delegate self-contained tasks to named child agents |
| [Dynamic Workflow](dynamic-workflow.md) | Harness | The model orchestrates sub-agents from one Python script — fan-out, chain, vote in a single tool call, with hard `max_agent_calls` budgets |
| [Planning](planning.md) | Harness | Model-owned task plans with a cache-safe live reminder |
| [Advisor](advisor.md) | Harness | Let an executor consult a stronger model mid-run |

### Steering & safety

How you shape and bound what the agent does — before, during, and after the model acts.

| Capability | Ships in | What it does |
|---|---|---|
| [Guardrails](guardrails.md) | Harness | Validate/block/redact user input, tool calls, tool results, and output — including secret masking and parallel async guards |
| [Spend Limits](spend.md) | Harness | Cross-window USD/token budgets and per-response cost tracking, per model and per tenant |
| [System Reminders](system-reminders.md) | Harness | Cache-safe re-injection of guidance mid-run to counter instruction fade |
| [Managed Prompt](managed-prompt.md) | Harness | Back instructions with a [Logfire](https://pydantic.dev/logfire)-managed prompt — version and roll out without redeploying |
| [Tool approval](/ai/tools-toolsets/deferred-tools/#human-in-the-loop-tool-approval) | Core | Flag tool calls that need human approval before they run |
| [Thinking](/ai/capabilities/thinking/) | Core | Provider-adaptive extended thinking at configurable effort |

### Self-extension

| Capability | Ships in | What it does |
|---|---|---|
| [Capability Creation](capability-creation.md) | Harness | The agent writes, validates, and persists *new capabilities* during a run, loaded on the next run — self-extension with typed, inspectable units instead of arbitrary code |

And the agent plugs into any interface: [ACP](acp.md) *(experimental, Harness)* serves it to editors like Zed over the [Agent Client Protocol](https://agentclientprotocol.com), and core ships the [web chat UI](/ai/web/), [CLI](/ai/cli/), [frontend adapters](/ai/ui/overview/) (AG-UI, Vercel AI), and [realtime voice](/ai/realtime/overview/).

This index deliberately spans both packages — there's one capability system, and this repo is where new capabilities start before they [graduate into core](#when-do-you-need-the-harness). Community packages (by [vstorm-co](https://github.com/vstorm-co), [DougTrajano](https://github.com/DougTrajano/pydantic-ai-skills), and others) extend it further — see [third-party capabilities](/ai/capabilities/third-party/).

## Composing from blocks

A research agent, from three capabilities — this is literally [`Researcher`](researcher.md)'s composition, minus its short default instructions:

```python
from pydantic_ai import Agent
from pydantic_ai.capabilities import WebFetch, WebSearch
from pydantic_ai_harness import ToolOutputLimits

agent = Agent(
    'anthropic:claude-fable-5',
    capabilities=[
        WebSearch(local=True),  # native provider search, DuckDuckGo fallback elsewhere
        WebFetch(local=True),  # read the pages behind the results, native or local
        ToolOutputLimits(),  # fetched pages don't flood the context
    ],
)

result = agent.run_sync('What changed in the top three Python agent frameworks this month? Cite sources.')
print(result.output)
#> ...
```

## When do you need the Harness?

Pydantic AI alone gives you the typed agent loop, structured output, and the fundamentals every agent needs — [web search](/ai/capabilities/web-search/), [thinking](/ai/capabilities/thinking/), [MCP](/ai/capabilities/mcp/). Reach for the Harness when your agent should *do* more: touch files, run code, browse, remember, delegate, or stay coherent through hours-long runs.

**What goes where:** Pydantic AI core ships capabilities that require model or framework support (provider-native tools like [image generation](/ai/capabilities/image-generation/), provider APIs like [compaction](/ai/capabilities/compaction/), deep loop integration like [tool search](/ai/capabilities/tool-search/)) and the ones fundamental to every agent. Everything else lives here — and this is where new capabilities *start*. Many follow a "fall up" pattern: a local implementation that works with every model first, provider-native support auto-switched in when available. As a capability stabilizes and proves broadly essential, it can graduate into core — [code mode](code-mode.md) is an early candidate.

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

## Observability

Everything the harness does is observable: `logfire.instrument_pydantic_ai()` gives you [a full trace of every run](https://logfire-us.pydantic.dev/public-trace/84bcf123-2106-49da-9f6f-5c26395339bb?spanId=7650806a0785b946) — every model call and tool call, with token and cost tracking. It's standard OpenTelemetry, so any OTLP backend works; [Logfire](https://pydantic.dev/logfire) is the easiest way to see it during development.

## Build your own

[Capabilities](/ai/capabilities/custom/) are the primary extension point for Pydantic AI, and every capability in this library doubles as a worked example. Publishing a standalone package? Use the `pydantic-ai-<name>` naming convention — see [Publishing capability packages](/ai/guides/extensibility/#publishing-capability-packages).

## Version policy

Pydantic AI Harness uses **0.x versioning** to signal that APIs are still stabilizing. During 0.x, minor releases (0.1 -> 0.2) may include breaking changes — renamed parameters, changed defaults, restructured APIs, always with deprecation warnings where practical — while patch releases will not intentionally break existing behavior. All breaking changes are documented in release notes with migration guidance. This is why the Harness is a separate package from [Pydantic AI](https://github.com/pydantic/pydantic-ai), which has a [stricter version policy](/ai/project/version-policy/): capabilities iterate here at the speed the field moves, and graduate into core as they stabilize.

## Pydantic AI references

- [Capabilities](/ai/capabilities/overview/) — what capabilities are, built-in capabilities, building your own
- [Hooks](/ai/core-concepts/hooks/) — lifecycle hooks reference, ordering, error handling
- [Extensibility](/ai/guides/extensibility/) — publishing packages, third-party ecosystem
- [Toolsets](/ai/tools-toolsets/toolsets/) — building tools for capabilities
- [API reference](/ai/api/pydantic-ai/capabilities/) — full API docs
