# Background Tools

Run selected tools concurrently with the current agent run. The model receives an immediate acknowledgment, keeps working, and, if the run remains active, receives the result as a follow-up message.

[Source](https://github.com/pydantic/pydantic-ai-harness/tree/main/pydantic_ai_harness/background_tools/)

```python
import asyncio

from pydantic_ai import Agent
from pydantic_ai_harness import BackgroundTools

agent = Agent('openai:gpt-5.6-sol', capabilities=[BackgroundTools()])

@agent.tool_plain(metadata={'background': True})
async def slow_research(query: str) -> str:
    """Research a topic thoroughly. Runs in the background."""
    await asyncio.sleep(60)  # stand-in for a long-running job
    return f'Research findings for {query!r}'
```

By default any tool with `metadata={'background': True}` runs in the background. The agent's instructions are augmented automatically so the model knows it shouldn't block waiting for the result.

> While Pydantic AI Harness is on 0.x releases, the API may change between minor releases; when it does, deprecation warnings and release-note migration guidance tell you (or your agent) exactly how to upgrade. See the [version policy](https://github.com/pydantic/pydantic-ai-harness#version-policy).

## Selecting which tools run in the background

`BackgroundTools(tools=...)` accepts the standard [`ToolSelector`](https://ai.pydantic.dev/api/tools/#pydantic_ai.tools.ToolSelector):

```python
from pydantic_ai_harness import BackgroundTools

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
from pydantic_ai import Agent, FunctionToolset
from pydantic_ai_harness import BackgroundTools

async def deep_research(query: str) -> str:
    return f'Research findings for {query!r}'

async def crawl_site(url: str) -> str:
    return f'Crawled {url}'

research_tools = FunctionToolset([deep_research, crawl_site]).with_metadata(background=True)
agent = Agent(
    'openai:gpt-5.6-sol',
    toolsets=[research_tools],
    capabilities=[BackgroundTools()],
)
```

## Result delivery

If the run remains active, a finished background tool produces a follow-up message:

- On success: `Background tool 'X' (task <id>) completed.\nResult: <return value>`
- On failure: `Background tool 'X' (task <id>) failed: <error>`

The task ID matches the acknowledgment. The follow-up is a text user message, not another tool
return. Retries and deferred calls are reported as text failures. Expected tool errors include their
message; unexpected exceptions include only their type.

## Execution behavior

Normal completion waits for background tasks and delivers their follow-ups. Concurrent runs track
their tasks separately. If a run pauses for [deferred tools](https://ai.pydantic.dev/deferred-tools/)
or ends through cancellation, a usage limit, or an error, live tasks are cancelled and their results
are dropped. Async tools get up to one second for cooperative cancellation cleanup.

> [!WARNING]
> Cancellation cannot stop a synchronous tool's worker thread. The function may continue with the
> same dependencies and shared `RunContext` state after the run ends, although its result is
> discarded. Use a cancellation-cooperative async tool when work must stop with the run.
> An async tool that suppresses cancellation may also outlive the run.
>
> A synchronous background tool runs concurrently with the agent. Make mutable dependencies and
> other shared state it uses thread-safe.
>
> A synchronous background tool must not call `ctx.enqueue()`: its worker thread can race the
> pending-message drain and lose the message. Async background tools do not have this cross-thread
> race, but delivery still requires the run to continue.

## Limitations

- **Streaming**: `run_stream()` completes on the model's final response and does not take the extra model turn that delivers late results, so background results are only guaranteed with `agent.run()` or a driven `agent.iter()` loop.
- **Result hooks**: Background tool results do not pass through [Guardrails](https://pydantic.dev/docs/ai/harness/guardrails/) result guards, `ToolOutputLimits`, or other result hooks. Screen and bound the result inside the tool instead.
- **Durable execution**: in-process tasks cannot survive workflow replay or worker restart, so this composition is rejected at agent construction.

## API

```python {test="skip"}
BackgroundTools(
    tools: ToolSelector = {'background': True},
)
```

## Agent spec (YAML/JSON)

```yaml
# agent.yaml
model: openai:gpt-5.6-sol
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
