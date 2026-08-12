---
title: Examples
description: Seven complete agents, each a runnable file built from harness presets or capabilities.
---

# Examples

The [`examples/`](https://github.com/pydantic/pydantic-ai-harness/tree/main/examples)
directory contains seven complete agents, each a runnable file built from
harness capabilities. They are meant to be read as much as run: the reasoning
behind every capability choice is a comment next to it.

| Example | What it does | Built with |
|---|---|---|
| `coding_agent.py` | Drives an interactive coding session in the current repo | `Coder` |
| `coding_agent_from_blocks.py` | Shows the coding preset expanded into component capabilities | `FileSystem`, `Shell`, `RepoContext`, `Planning`, delegation, compaction, output limits |
| `code_review_agent.py` | Reviews a git diff with specialist sub-reviewers and returns a typed report | `SubAgents`, a `filtered` read-only `FileSystem` toolset, a read-only `git_history` tool |
| `atlas.py` | Builds and maintains a knowledge map of a repo, refreshing only what changed | `FileSystem` toolsets split by scope (`filtered` read, `prefixed` write), `Planning`, `SlidingWindowCompaction`, a read-only `git_history` tool |
| `research_agent.py` | Researches a question on the web and cites every claim | `Researcher` |
| `data_analysis_agent.py` | Analyzes a dataset by computing in a sandbox instead of guessing numbers | `CodeMode` with a read-only dataset mount, `ToolOutputLimits` |
| `support_agent.py` | Triages support messages with guardrails and per-customer memory | `InputGuardrail`, `OutputGuardrail`, `Memory` with `FileStore` |

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
