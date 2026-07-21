---
title: Examples
description: Six complete agents, each a single runnable file built from harness capabilities, with the reasoning behind every capability choice commented next to it.
---

# Examples

The [`examples/`](https://github.com/pydantic/pydantic-ai-harness/tree/main/examples)
directory contains six complete agents, each a single runnable file built from
harness capabilities. They are meant to be read as much as run: the reasoning
behind every capability choice is a comment next to it.

| Example | What it does | Built with |
|---|---|---|
| `coding_agent.py` | Drives an interactive coding session in the current repo | `FileSystem`, `Shell`, `RepoContext`, `Planning`, `SlidingWindow`, `LimitWarner` |
| `code_review_agent.py` | Reviews a git diff with specialist sub-reviewers and returns a typed report | `SubAgents` with `shared_capabilities`, read-only `FileSystem`, `Shell` |
| `atlas.py` | Builds and maintains a knowledge map of a repo, refreshing only what changed | `FileSystem`, `Shell`, `Planning`, `SlidingWindow`, plus a no-op gate in plain code |
| `research_agent.py` | Researches a question on the web and cites every claim | `CodeMode`, `WebSearch`, `OverflowingToolOutput` |
| `data_analysis_agent.py` | Analyzes a dataset by computing in a sandbox instead of guessing numbers | `CodeMode` with a read-only dataset mount, `OverflowingToolOutput` |
| `support_agent.py` | Triages support messages with guardrails and per-customer memory | `InputGuard`, `OutputGuard`, `Memory` with `FileStore` |

Run one from the repo root:

```bash
make install
export ANTHROPIC_API_KEY=...   # or set PYDANTIC_AI_MODEL to any provider:model
uv run examples/coding_agent.py
```

Every example exposes a `build_agent()` factory you can import and embed in your
own code, and a `main()` that runs a small demo. See
[`examples/README.md`](https://github.com/pydantic/pydantic-ai-harness/blob/main/examples/README.md)
for per-example details.
