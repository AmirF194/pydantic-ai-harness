---
title: Examples
description: Two runnable agents shown as complete harnesses and component capabilities.
---

# Examples

The [`examples/`](https://github.com/pydantic/pydantic-ai-harness/tree/main/examples)
directory contains two runnable harnesses. The coding harness is also shown
expanded into its component capabilities. They are meant to be read as much as
run: the reasoning behind every capability choice is a comment next to it.

| Example | What it does | Built with |
|---|---|---|
| `coding_agent.py` | Drives an interactive coding session in the current repo | `Coder` |
| `coding_agent_from_blocks.py` | Shows the coding harness expanded into component capabilities | `FileSystem`, `Shell`, `RepoContext`, `Planning`, delegation, compaction, output limits |
| `research_agent.py` | Researches a question on the web and cites every claim | `Researcher` |

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
