# Background Tools

Run selected tools as fire-and-forget asyncio tasks, so the agent can keep working while they finish.

[Source](https://github.com/pydantic/pydantic-ai-harness/tree/main/pydantic_ai_harness/background_tools/)

> While Pydantic AI Harness is on 0.x releases, the API may change between minor releases; when it does, deprecation warnings and release-note migration guidance tell you (or your agent) exactly how to upgrade. See the [version policy](https://github.com/pydantic/pydantic-ai-harness#version-policy).

## The problem

Some tools take seconds to minutes -- deep research, big aggregations, sub-agent delegation. With normal tool calls the agent is blocked: it makes the call, waits, then plans its next step. Over a long task the conversation effectively serializes.

## The solution

`BackgroundTools` spawns the matching tool calls as `asyncio.Task`s. The agent receives an immediate acknowledgment string and continues planning. When the task finishes, its result is enqueued as a follow-up message via [`RunContext.enqueue`][pydantic_ai.tools.RunContext.enqueue]; Pydantic AI's pending message queue redirects the agent into a fresh `ModelRequest` instead of ending, so the model sees the result and can use it.

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

`BackgroundTools(tools=...)` accepts the standard [`ToolSelector`][pydantic_ai.tools.ToolSelector]:

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

### Marking a whole MCP server or toolset

Combine with [`SetToolMetadata`][pydantic_ai.capabilities.SetToolMetadata] or `FunctionToolset.with_metadata(...)` to mark every tool from a source as background, without touching individual definitions:

```python
from pydantic_ai import Agent
from pydantic_ai.capabilities import MCP, SetToolMetadata
from pydantic_ai_harness import BackgroundTools

agent = Agent('openai:gpt-5', capabilities=[
    MCP('https://research.example/mcp/'),
    SetToolMetadata(tools=lambda ctx, td: td.name.startswith('mcp_'), background=True),
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

- Each agent run gets fresh task state via the capability's `for_run` hook -- concurrent runs do not share tasks
- If the surrounding agent run is cancelled (e.g. via `asyncio.wait_for` timeout), all live background tasks are cancelled in the capability's `wrap_run` cleanup
- `asyncio.CancelledError` from a cancelled task does not produce a result message; it propagates as a normal task cancellation

## Limitations

- **Temporal / DBOS**: tools run inside durable activities and don't share state with the surrounding workflow. Tool-side `ctx.enqueue` calls do not currently propagate back, so background results from durable tools are lost. If you need this, file an issue.

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
