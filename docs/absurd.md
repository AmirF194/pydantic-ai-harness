---
title: Absurd Durability
description: Checkpoint Pydantic AI agent I/O in Absurd steps for crash recovery.
---

# Absurd Durability

[Source](https://github.com/pydantic/pydantic-ai-harness/tree/main/pydantic_ai_harness/absurd/)

`AbsurdDurability` checkpoints Pydantic AI model requests, function-tool calls, and MCP
operations in [Absurd](https://github.com/earendil-works/absurd) steps. A worker can resume an
agent run after a crash from the last completed step.

## Installation

```bash
uv add "pydantic-ai-harness[absurd]"
```

The `absurd` extra installs `absurd-sdk`. The SDK stores checkpoints in PostgreSQL. Create the
Absurd schema and queue using the setup instructions in the [Absurd repository](https://github.com/earendil-works/absurd).

## Usage

```python {test="skip"}
from absurd_sdk import AsyncAbsurd, AsyncTaskContext, JsonValue
from pydantic_ai import Agent
from pydantic_ai_harness import AbsurdDurability

absurd = AsyncAbsurd('postgresql://localhost/absurd', queue_name='agents')
agent = Agent('openai:gpt-5.6-sol', name='analyst', capabilities=[AbsurdDurability()])


@absurd.register_task(name='analyse')
async def analyse(params: JsonValue, ctx: AsyncTaskContext) -> JsonValue:
    assert isinstance(params, dict)
    result = await agent.run(params['prompt'])
    return {'output': result.output}
```

Call `agent.run()` from an async Absurd task handler. A synchronous task context raises a
`UserError`. Outside an Absurd task, the capability does not checkpoint work.

## Checkpoint names

The agent name prefixes every step. Function and MCP toolset IDs are included when present,
but IDs are not required. The resulting names match the standalone Absurd integration:

- `{name}__model.request` and `{name}__model.request_stream`
- `{name}__function_toolset[__{id}].call_tool:{tool_name}`
- `{name}__mcp_server[__{id}].get_tools`, `.get_instructions`, and `.call_tool`
- `{name}__event_stream_handler` for tool events delivered to an event stream handler

Absurd disambiguates repeated step names by encounter order (`name`, `name#2`, ...). Keep the
agent name, toolset IDs, and tool names stable while tasks are in flight. Checkpointed results
must be JSON-serializable.

The wrapped MCP toolset is not entered by this capability. `MCPToolset` manages its own sessions,
or the application can manage the toolset lifecycle externally. If an externally entered session
already has instructions, they are read directly from that session rather than through a
`.get_instructions` checkpoint; discovery and tool calls remain checkpointed.

Steps are at-least-once: a crash after an operation and before its checkpoint can run the
operation again, so checkpointed side effects should be idempotent. Set Absurd's `claim_timeout`
longer than the longest model, tool, MCP, or event-handler operation. The step does not heartbeat
while its callback is running. Sibling model-request hooks run again when a task replays; avoid
composing stateful side effects into those hooks. Concurrent runs sharing a task context and
checkpoint namespace are rejected; await one run before starting another.

## Parallel execution

`parallel_execution_mode` accepts `'sequential'` or `'parallel_ordered_events'`. Plain
`'parallel'` is not supported because completion-order event delivery can change Absurd's
encounter-order step names on replay.

```python {test="skip"}
capability = AbsurdDurability(parallel_execution_mode='parallel_ordered_events')
```

## Compatibility

The step names and raw function/MCP tool-result payloads are compatible with
`pydantic-ai-absurd`. Streamed checkpoints written by its deprecated `AbsurdAgent` are also
readable here. A task started with one integration can therefore resume under the other when the
agent and toolset names are unchanged.

## API

::: pydantic_ai_harness.absurd.AbsurdDurability
