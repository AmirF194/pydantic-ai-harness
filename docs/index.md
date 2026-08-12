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
from pydantic_ai_harness import CodingHarness

agent = Agent('anthropic:claude-opus-4-7', capabilities=[CodingHarness()])
agent.to_cli_sync()
```

That's a complete coding agent in your terminal: sandboxed file access, allowlisted shell, repo orientation, planning, and context management that survives long sessions.

Every model works — swap the string for `'openai:gpt-5.5'`, `'google:gemini-3-pro'`, or [any other provider](/ai/models/overview/). Need more? Add capabilities to the list:

```python
from pydantic_ai_harness import CodingHarness, Memory

agent = Agent(
    'anthropic:claude-opus-4-7',
    capabilities=[CodingHarness(), Memory()],  # remembers across sessions
)
```

And it's an ordinary Pydantic AI agent: run it headless with `agent.run_sync(...)`, serve it as a [web chat UI](/ai/web/), attach it to [your own frontend](/ai/ui/overview/), or hand it [structured output types](/ai/core-concepts/output/).

## No magic: it's capabilities all the way down

`CodingHarness` is not a framework inside the framework — it's a [`CombinedCapability`](/ai/capabilities/custom/) bundling the same blocks you can use directly. This is its actual definition, spelled out:

```python
from pydantic_ai import Agent
from pydantic_ai_harness import FileSystem, Planning, RepoContext, Shell
from pydantic_ai_harness.compaction import SlidingWindowCompaction, WarnNearLimits

agent = Agent(
    'anthropic:claude-opus-4-7',
    capabilities=[
        FileSystem('.'),  # read/write/edit/search, path-traversal safe
        Shell(),  # allowlisted command execution
        RepoContext(),  # loads AGENTS.md/CLAUDE.md + repo structure
        Planning(),  # structured task plans the model maintains
        SlidingWindowCompaction(),  # stays within the context window
        WarnNearLimits(),  # warns the model before it hits limits
    ],
)
```

Start from the harness and remove what you don't want, or start from the blocks and build up — both are first-class. Constructor arguments (working directory, command allowlist, window sizes) thread through to the underlying capabilities.

## Capabilities

Each capability is a self-contained unit you drop into `capabilities=[...]`. They compose with each other and with Pydantic AI's [built-in capabilities](/ai/capabilities/overview/). Grouped by what they give your agent:

### Execution

Let the agent act on real systems — locally or in an isolated cloud sandbox.

| Capability | What it does |
|---|---|
| [Code Mode](code-mode.md) | Wraps all tools into one sandboxed `run_code` tool ([Monty](https://github.com/pydantic/monty)) — the model writes Python that calls N tools in one round-trip |
| [Shell](shell.md) | Command execution with allowlists, denylists, timeouts, and credential-stripping |
| [FileSystem](filesystem.md) | Read, write, edit, search files under a root — path-traversal and symlink safe, secrets read-only |
| [Modal Sandbox](modal-sandbox.md) | Commands and files in an isolated [Modal](https://modal.com) cloud sandbox |
| [LocalStack](localstack.md) | An emulated AWS environment with AWS CLI tools |

### Web & research

| Capability | What it does |
|---|---|
| [Exa Search](exa-search.md) | Web research via [Exa](https://exa.ai): excerpted search, full-page reads, opt-in cited deep search |
| [Exa Agent](exa-search.md) | Delegate open-ended research to the Exa Agent API |
| [Browser Use](browser-use.md) | Hand web tasks to an autonomous [browser-use](https://github.com/browser-use/browser-use) agent driving a real browser |

### Context

Control what the model sees — the difference between an agent that degrades over a long run and one that doesn't.

| Capability | What it does |
|---|---|
| [Compaction](compaction.md) | Sliding-window trimming, LLM summarization, tiered strategies, window-relative thresholds, live usage reporting |
| [Tool Output Limits](tool-output-limits.md) | Truncate, spill to a queryable file, or summarize oversized tool returns at the source |
| [System Reminders](system-reminders.md) | Cache-safe re-injection of guidance mid-run to counter instruction fade |
| [Repo Context](repo-context.md) | Start runs oriented: `AGENTS.md`/`CLAUDE.md` + repository structure |
| [Skills](skills.md) | Load [Agent Skill](/ai/capabilities/on-demand/) (`SKILL.md`) instructions on demand |
| [Pydantic AI Docs](pydantic-ai-docs.md) | On-demand Pydantic AI documentation lookup |
| [Warn On Cache Busts](warn-on-cache-busts.md) | Detect prompt-cache prefix collapses between requests, from the provider's own numbers |

### Memory & persistence

| Capability | What it does |
|---|---|
| [Memory](memory.md) | A persistent, namespaced notebook: bounded prompt injection, on-demand search; in-memory/file/Postgres stores |
| [Step Persistence](step-persistence.md) | Save, restore, resume (`continue_run`), and fork (`fork_run`) runs; file/SQLite/Mongo backends |
| [Conversation Search](conversation-search.md) | BM25 search over stored history — including turns compaction dropped |
| [Media](media.md) | Offload large binary/text parts to content-addressed stores (disk, SQLite, S3, Mongo) |

### Delegation & planning

| Capability | What it does |
|---|---|
| [Subagents](subagents.md) | Delegate self-contained tasks to named child agents |
| [Dynamic Workflow](dynamic-workflow.md) | The model orchestrates sub-agents from one Python script — fan-out, chain, vote in a single tool call, with hard `max_agent_calls` budgets |
| [Planning](planning.md) | Model-owned task plans with a cache-safe live reminder |
| [Advisor](advisor.md) | Let an executor consult a stronger model mid-run |

### Self-improvement

| Capability | What it does |
|---|---|
| [Capability Creation](capability-creation.md) | The agent writes, validates, and persists *new capabilities* during a run, loaded on the next run — self-extension with typed, inspectable units instead of arbitrary code |

### Safety & spend

| Capability | What it does |
|---|---|
| [Guardrails](guardrails.md) | Validate/block/redact user input, tool calls, tool results, and output — including secret masking and parallel async guards |
| [Spend Limits](spend.md) | Cross-window USD/token budgets and per-response cost tracking, per model and per tenant |

### Integrations

| Capability | What it does |
|---|---|
| [ACP](acp.md) *(experimental)* | Serve your agent to editors (Zed and friends) over the [Agent Client Protocol](https://agentclientprotocol.com) |
| [Managed Prompt](managed-prompt.md) | Back instructions with a [Logfire](https://pydantic.dev/logfire)-managed prompt — version and roll out without redeploying |
| [StackOne](stackone.md) | Act on linked SaaS accounts (HRIS, ATS, CRM, …) via [StackOne](https://www.stackone.com) |
| [Macroscope](macroscope.md) | Run a local [Macroscope](https://docs.macroscope.com/cli) code review and hand the findings to the agent |

**In the works:** [verification loop](https://github.com/pydantic/pydantic-ai-harness/pull/355) · [task tracking](https://github.com/pydantic/pydantic-ai-harness/pull/404) · [stuck-loop detection](https://github.com/pydantic/pydantic-ai-harness/pull/336) · [tool error recovery](https://github.com/pydantic/pydantic-ai-harness/pull/171) · [adaptive reasoning](https://github.com/pydantic/pydantic-ai-harness/pull/174) — vote on the PRs to help us prioritize, or [request a capability](https://github.com/pydantic/pydantic-ai-harness/issues/new?template=capability-request.yml).

Community capability packages (by [vstorm-co](https://github.com/vstorm-co), [DougTrajano](https://github.com/DougTrajano/pydantic-ai-skills), and others) extend this list further — see [third-party capabilities](/ai/capabilities/third-party/).

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

Everything the harness does is observable: `logfire.instrument_pydantic_ai()` gives you [a full trace of every run](https://logfire-us.pydantic.dev/public-trace/84bcf123-2106-49da-9f6f-5c26395339bb?spanId=7650806a0785b946) — each `run_code` span fans out into the tool calls the model made inside the sandbox. It's standard OpenTelemetry, so any OTLP backend works; [Logfire](https://pydantic.dev/logfire) is the easiest way to see it during development.

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
