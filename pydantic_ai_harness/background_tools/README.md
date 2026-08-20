# Background Tools

Run selected tools as in-process fire-and-forget asyncio tasks, so the agent can keep working while they finish.

[Source](https://github.com/pydantic/pydantic-ai-harness/tree/main/pydantic_ai_harness/background_tools/)

> While Pydantic AI Harness is on 0.x releases, the API may change between minor releases; when it does, deprecation warnings and release-note migration guidance tell you (or your agent) exactly how to upgrade. See the [version policy](https://github.com/pydantic/pydantic-ai-harness#version-policy).

## The problem

Some tools take seconds to minutes -- deep research, big aggregations, sub-agent delegation. With normal tool calls the agent is blocked: it makes the call, waits, then plans its next step. Over a long task the conversation effectively serializes.

## The solution

`BackgroundTools` runs the matching tool calls as background tasks. The agent receives an immediate acknowledgment and keeps working. When the work finishes while the run remains active, the result (or error) comes back into the conversation as a follow-up message: the model sees it on its next request, and an agent that was about to finish takes one more turn so the result still arrives.

## Usage

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

When a background tool finishes, the model receives a follow-up message:

- On success: `Background tool 'X' (task <id>) completed.\nResult: <return value>`
- On failure: `Background tool 'X' (task <id>) failed: <error message>`

The task ID matches the one in the acknowledgment, so the model can tell which call each result belongs to.

The follow-up is a text-only user message, not a second tool return. Return values are rendered as
text, and control-flow results such as retries or deferred calls cannot preserve their control flow
through it; they are reported as text failures instead.

## Lifecycle and cancellation

- Each agent run gets fresh task state, so concurrent runs do not share tasks
- The run does not finish while a background task is still live: a would-be final answer is held until the task delivers its follow-up, so total run duration still includes every background task
- A run that pauses for [deferred tools](https://ai.pydantic.dev/deferred-tools/) (human-in-the-loop approval, external execution) is never held behind background work; remaining background tasks are cancelled, so their results are not delivered
- If the run ends for any other reason with tasks still live (caller cancellation, a usage limit, an unexpected error), those tasks are cancelled and their results are dropped; delivery is conditional on the run continuing
- Cancellation is cooperative for synchronous tools: work already running in an executor cannot be interrupted and may continue after the run ends

## Limitations

- **Streaming**: `run_stream()` completes on the model's final response and does not take the extra model turn that delivers late results, so background results are only guaranteed with `agent.run()` or a driven `agent.iter()` loop.
- **Result guardrails**: Background tool results do not pass through [Guardrails](https://pydantic.dev/docs/ai/harness/guardrails/) result guards. Treat background tools as incompatible with result guards unless the tool screens sensitive output itself.
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
