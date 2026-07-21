---
title: Absurd Durability
description: Make a Pydantic AI agent run durable on Absurd, a Postgres-based durable-execution engine -- model requests, MCP calls, and function tool calls are checkpointed into steps so a crashed worker resumes mid-run.
---

# Absurd Durability

`AbsurdDurability` makes an agent run durable on [Absurd](https://github.com/earendil-works/absurd), a
Postgres-based durable-execution engine by Armin Ronacher (Python SDK `absurd-sdk`). Attach the
capability and call `agent.run()` inside an Absurd task handler: every model request, MCP call, and
function tool call is checkpointed into an Absurd step (`ctx.step(...)`), so if a worker crashes
part-way through a run it resumes from the last completed step instead of restarting -- no tokens
are re-spent, and a tool side effect runs once. Outside a task the capability is transparent and the
run is a normal, non-durable agent run.

[Source](https://github.com/pydantic/pydantic-ai-harness/tree/main/pydantic_ai_harness/absurd/)

## Installation

```bash
uv add "pydantic-ai-harness[absurd]"
```

Absurd stores its state in Postgres. Once per database, install the Absurd schema and create a
queue. The schema SQL and the queue helpers ship with the upstream project; see the
[Absurd repository](https://github.com/earendil-works/absurd) for the schema file and setup steps.

```python {test="skip"}
from absurd_sdk import AsyncAbsurd

absurd = AsyncAbsurd('postgresql://localhost/absurd', queue_name='agents')
await absurd.create_queue()
```

## Quick start

Construct the agent with the capability, register a task handler that runs it, then split the work
across a producer that spawns tasks and a worker that executes them. The agent needs a `name`; it
prefixes every checkpoint step.

```python {test="skip"}
from absurd_sdk import AsyncAbsurd, AsyncTaskContext, JsonValue
from pydantic_ai import Agent
from pydantic_ai_harness.absurd import AbsurdDurability

absurd = AsyncAbsurd('postgresql://localhost/absurd', queue_name='agents')
agent = Agent('openai:gpt-5', name='analyst', capabilities=[AbsurdDurability()])


@absurd.register_task(name='analyse')
async def analyse(params: JsonValue, ctx: AsyncTaskContext) -> JsonValue:
    assert isinstance(params, dict)
    result = await agent.run(params['prompt'])
    return {'output': result.output}


# Producer: enqueue a task.
await absurd.spawn('analyse', {'prompt': 'Summarize the Q3 report.'})

# Worker: claim and run tasks (in its own process). `start_worker` polls continuously.
await absurd.start_worker()
```

The task handler runs inside an `AsyncTaskContext`, which is how the capability knows to checkpoint.
Call `agent.run()` (async) from an async handler; a synchronous `TaskContext` raises a `UserError`
because an agent run cannot be awaited from one.

## What gets checkpointed, and what replay means

Each of these is wrapped in its own `ctx.step(...)`, so its result is recorded once and served from
the checkpoint on replay rather than being recomputed:

- **model requests** -- `{name}__model.request`, `.request_stream`, and `.cancel_suspended_response`
  (the request's model id is appended as a `.{model_id}` suffix when it differs from the agent's
  default);
- **function tool calls** -- `{name}__function_toolset__{id}.call_tool:{tool_name}`;
- **MCP I/O** -- `{name}__mcp_server__{id}.get_tools`, `.get_instructions`, and `.call_tool`;
- **event-stream handler calls** -- `{name}__event_stream_handler`, when an `event_stream_handler`
  is set.

Replay means: after a crash, Absurd re-runs the task handler from the top. Plain Python in the
handler body runs again, but each checkpointed step returns its stored result instead of re-issuing
the model request or re-calling the tool. A `ModelRetry`, `ApprovalRequired`, or `CallDeferred`
raised by a tool crosses the checkpoint as a serialized value, so on replay the same outcome is
reproduced without re-running the tool.

## Constraints

- The agent needs a `name` (or pass `name=` to `AbsurdDurability`); it prefixes every step.
- Leaf toolsets that execute their own tools (function toolsets, MCP servers) need a unique `id`,
  which identifies their steps within the task.
- A checkpointed tool's return value is stored in Postgres as JSON, so it must be JSON-serializable.
- The executing toolsets are fixed when the agent is constructed. Passing an executing toolset
  per-run via `run(toolsets=...)` inside a task raises a `UserError`, because a runtime toolset has
  no registered steps and would re-run its side effects on recovery. Non-executing toolsets such as
  `ExternalToolset` are allowed at runtime.
- Streaming inside a task is a replay, not a live wire: the model stream is consumed and captured
  inside the step, and the run-side stream replays the captured events.
- An `event_stream_handler` runs live inside the model-request step (its call is itself
  checkpointed), so its side effects run once.
- Do not use `run_sync` inside a task handler. The handler is async; use `await agent.run(...)`.

## Per-tool opt-out

A function tool can opt out of checkpointing with `metadata={'absurd': False}`, which runs it inline
inside the task rather than in a step. Reach for this for a cheap, side-effect-free tool where a
checkpoint would add more overhead than it saves.

```python {test="skip"}
from pydantic_ai import Agent
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai_harness.absurd import AbsurdDurability

tools = FunctionToolset(id='math')


@tools.tool_plain(metadata={'absurd': False})
def add(a: int, b: int) -> int:
    return a + b


agent = Agent('openai:gpt-5', name='calc', toolsets=[tools], capabilities=[AbsurdDurability()])
```

MCP tools cannot opt out: they perform I/O and so are always checkpointed. Setting
`metadata={'absurd': False}` on an MCP tool raises a `UserError`.

## Checkpoint format compatibility

The step names and the checkpoint payload shapes are byte-compatible with the `pydantic-ai-absurd`
package (the standalone Absurd integration by Marcelo Trylesinski), so a run started under one
package can resume under the other. Treat the step names and payload shapes as a stable persistence
format.

## Relation to Step Persistence

`AbsurdDurability` and the [Step Persistence](step-persistence.md) capability solve different
problems and compose. Absurd gives exactly-once crash-resume *within* a single run: a worker that
dies mid-run picks up from the last completed step. Step Persistence records step events and
continuation snapshots *across* runs, so a run can be resumed, forked, or replayed as a separate
invocation later. Use Absurd for durability against crashes during a run, and Step Persistence to
persist and resume runs as first-class records.

## Further reading

- [Pydantic AI capabilities](/ai/capabilities/overview/)
- [Absurd](https://github.com/earendil-works/absurd)

## API reference

::: pydantic_ai_harness.absurd.AbsurdDurability
