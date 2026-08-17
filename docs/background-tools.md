---
title: Background Tools
description: Run selected tools as fire-and-forget asyncio tasks -- the agent gets an immediate acknowledgment, keeps working, and receives the result as a follow-up message when the task completes.
---

# Background Tools

`BackgroundTools` runs selected tools as fire-and-forget asyncio tasks, so the agent can keep
working while they finish. The model gets an immediate acknowledgment string in place of the
tool result, and the real result (or error) arrives later as a follow-up message it can act on.

[Source](https://github.com/pydantic/pydantic-ai-harness/tree/main/pydantic_ai_harness/background_tools/)

> While Pydantic AI Harness is on 0.x releases, the API may change between minor releases; when it does, deprecation warnings and release-note migration guidance tell you (or your agent) exactly how to upgrade. See the [version policy](index.md#version-policy).

## The problem

Some tools take seconds to minutes -- deep research, big aggregations, sub-agent delegation. With normal tool calls the agent is blocked: it makes the call, waits, then plans its next step. Over a long task the conversation effectively serializes.

## The solution

`BackgroundTools` spawns the matching tool calls as `asyncio.Task`s. The agent receives an immediate acknowledgment string and continues planning. When the task finishes, its result is enqueued as a follow-up message via [`RunContext.enqueue`](https://ai.pydantic.dev/api/tools/#pydantic_ai.tools.RunContext.enqueue); Pydantic AI's [pending message queue](https://ai.pydantic.dev/message-history/#injecting-messages-mid-run) delivers it on the next model request, or redirects the agent into a fresh request instead of ending, so the model sees the result and can use it.

## Usage

```python
from pydantic_ai import Agent
from pydantic_ai_harness import BackgroundTools

agent = Agent('openai:gpt-5', capabilities=[BackgroundTools()])

@agent.tool_plain(metadata={'background': True})
async def slow_research(query: str) -> str:
    """Research a topic thoroughly. Runs in the background."""
    return await do_expensive_research(query)
```

By default any tool with `metadata={'background': True}` runs in the background. The agent's instructions are augmented automatically so the model knows it shouldn't block waiting for the result.

## Selecting which tools run in the background

`BackgroundTools(tools=...)` accepts the standard [`ToolSelector`](https://ai.pydantic.dev/api/tools/#pydantic_ai.tools.ToolSelector):

```python
# By metadata key (default)
BackgroundTools()                                 # tools with metadata={'background': True}
BackgroundTools(tools={'background': True})       # explicit form
BackgroundTools(tools={'kind': 'research'})       # custom metadata key

# By name
BackgroundTools(tools=['slow_research', 'deep_dig'])

# By predicate
BackgroundTools(tools=lambda ctx, td: td.name.startswith('research_'))
```

### Marking tools in bulk

Combine with [`SetToolMetadata`](https://ai.pydantic.dev/capabilities/set-tool-metadata/) or `FunctionToolset.with_metadata(...)` to mark several tools as background without touching individual definitions:

```python
from pydantic_ai import Agent
from pydantic_ai.capabilities import MCP, SetToolMetadata
from pydantic_ai_harness import BackgroundTools

agent = Agent('openai:gpt-5', capabilities=[
    MCP('https://research.example/mcp/'),
    SetToolMetadata(tools=['deep_research', 'crawl_site'], background=True),
    BackgroundTools(),
])
```

## Result delivery

Results are enqueued as `'asap'` priority messages on Pydantic AI's pending message queue. The next time the agent makes a model request, the result is delivered alongside it; if the agent would otherwise terminate before another request, the queue redirects it through a fresh `ModelRequest` so the model still receives the result.

The message format is a plain string (wrapped in a `UserPromptPart` by `enqueue`):

- On success: `Background tool 'X' (task <id>) completed.\nResult: <return value>`
- On failure: `Background tool 'X' (task <id>) failed: <error message>`

The model sees the task ID alongside the result so it can correlate against the ack string it received earlier.

## Lifecycle and cancellation

- Each agent run gets fresh task state, so concurrent runs do not share tasks
- The run does not finish while a background task is still live: a would-be final answer is held until the task delivers its follow-up, so total run duration still includes every background task
- A run that pauses for [deferred tools](https://ai.pydantic.dev/deferred-tools/) (human-in-the-loop approval, external execution) is never held behind background work; remaining background tasks are cancelled when the run pauses
- When the run ends for any other reason with tasks still live (caller cancellation, a usage limit, an unexpected error), those tasks are cancelled and their results are dropped
- `asyncio.CancelledError` from a cancelled task does not produce a result message; it propagates as a normal task cancellation

## Limitations

- **Streaming**: `run_stream()` completes on the model's final response and does not take the extra model turn that delivers late results, so background results are only guaranteed with `agent.run()` or a driven `agent.iter()` loop.
- **Durable execution (Temporal / DBOS)**: spawning asyncio tasks inside a durable workflow is not replay-safe and this capability is untested there; don't combine them yet.

## API

```python {test="skip"}
BackgroundTools(
    tools: ToolSelector = {'background': True},
)
```

## Agent spec (YAML/JSON)

```yaml
# agent.yaml
model: openai:gpt-5
capabilities:
  - BackgroundTools: {}
```

```python
from pydantic_ai import Agent
from pydantic_ai_harness import BackgroundTools

agent = Agent.from_file('agent.yaml', custom_capability_types=[BackgroundTools])
```

## Further reading

- [Pydantic AI message history -- injecting messages mid-run](https://ai.pydantic.dev/message-history/#injecting-messages-mid-run) -- the underlying primitive
- [Pydantic AI capabilities](https://ai.pydantic.dev/capabilities/)
