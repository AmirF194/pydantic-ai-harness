# Examples

Two runnable harnesses, with the coding harness also expanded into its component
capabilities. They are meant to be read as much as run: the reasoning behind
every capability choice is a comment next to it.

Setup, from the repo root:

```bash
make install                      # or: uv sync --all-extras
export ANTHROPIC_API_KEY=...      # or set PYDANTIC_AI_MODEL to any provider:model
uv run examples/coding_agent.py
```

| Example | What it does | Built with |
|---|---|---|
| [`coding_agent.py`](coding_agent.py) | Drives an interactive coding session in the current repo | `Coder` |
| [`coding_agent_from_blocks.py`](coding_agent_from_blocks.py) | Shows the coding harness expanded into component capabilities | `FileSystem`, `Shell`, `RepoContext`, `Planning`, delegation, compaction, output limits |
| [`research_agent.py`](research_agent.py) | Researches a question on the web and cites every claim | `Researcher` |

Every example exposes a `build_agent()` factory (imported by the test suite, and
handy for embedding the agent in your own code) and a `main()` that runs a small
demo. Set `PYDANTIC_AI_MODEL` (e.g. `openai:gpt-5.6-sol`) to switch provider.
