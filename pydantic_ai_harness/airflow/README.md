# Airflow Durability

> [!NOTE]
> Import `AirflowDurability` and the storage backends from the submodule -- there is no top-level
> `pydantic_ai_harness` re-export:
>
> ```python
> from pydantic_ai_harness.airflow import AirflowDurability, InMemoryDurableStorage
> ```
>
> The API may change between releases. Where practical, breaking changes ship with a deprecation warning.

`AirflowDurability` makes an agent recover across an Apache Airflow task retry by memoizing its I/O.
Every model request, function tool call, MCP call, and dynamic-toolset resolution is stored under a
positional step key plus a fingerprint of the request that produced it. If the agent run fails, the
Airflow task retry re-runs it from the top: each completed operation is served from the store without
re-executing, so finished work is not repeated and tokens are not re-spent, while the operation that
failed (and everything after it) runs live.

It is the capability-shaped reference version of Airflow's
[`apache-airflow-providers-common-ai`](https://github.com/apache/airflow/tree/main/providers/common/ai)
durable integration, which memoizes the same operations but is attached through an operator rather
than a user-attached capability. The point of this version is to show the Airflow team the capability
shape so they can decide whether to adopt it.

[Source](https://github.com/pydantic/pydantic-ai-harness/tree/main/pydantic_ai_harness/airflow/)

## How it works

The durable unit is the whole agent run, which maps to one Airflow task. There is no
suspend-and-resume: recovery is a full re-run against a store that outlived the failed attempt.

1. **Positional step keys.** A per-run counter numbers each operation in the order it is reached
   (`model_step_0`, `tool_step_1`, `model_step_2`, ...). The counter advances even on a store hit, so
   a retry that replays earlier steps still reaches later ones at the same positions.
2. **Fingerprint-verified replay.** Each entry stores a fingerprint of its request (model identity,
   message history, content settings, and request parameters for a model call; tool name, arguments,
   and `tool_call_id` for a tool call). On a positional hit the current request's fingerprint is
   compared to the stored one; a mismatch is a miss and the step re-runs live. Because a fresh model
   response carries new `tool_call_id` values, a model divergence cascades to invalidate the tool
   steps recorded under the old conversation.
3. **Pluggable storage.** The store is a small `DurableStorage` Protocol. An in-memory backend and a
   JSON-file backend ship here for offline use; a backend writing to Airflow's ObjectStorage or task
   state store would live in the provider and satisfy the same Protocol.

## Quick start

```python {test="skip"}
from pydantic_ai import Agent

from pydantic_ai_harness.airflow import AirflowDurability, InMemoryDurableStorage

storage = InMemoryDurableStorage()
agent = Agent('openai:gpt-5', name='analyst', capabilities=[AirflowDurability(storage=storage)])


@agent.tool_plain
def get_weather(city: str) -> str:
    return f'It is sunny in {city}.'


result = agent.run_sync('What is the weather in Mexico City?')
```

In an Airflow deployment, construct the agent with a provider-backed storage, run it inside a task,
and let Airflow's retry re-run the task on failure. Pass `durable=False` to make the capability
transparent (a normal, non-memoized run).

## What it improves over the provider it models

- **Structured tool results round-trip.** `ToolReturn` and `BinaryContent` are serialized through
  Pydantic, so they are memoized. The provider's raw `json.dumps` skips them.
- **Control-flow signals round-trip.** `ModelRetry`, `ApprovalRequired`, and `CallDeferred` cross the
  store as values and re-raise on replay, so approval and deferred-tool flows survive a retry.
- **MCP tool listings, instructions, and dynamic-toolset resolution are memoized.** The provider
  re-runs those live on every attempt.
- **A non-serializable result is skipped with a warning**, not a crash.

## Honest limitations

This is reference-quality, not a drop-in production engine:

- **No suspend/resume.** Recovery is a full re-run from the top, not a resume from a checkpoint. A
  deferred tool (`ApprovalRequired`/`CallDeferred`) ends the run with a `DeferredToolRequests`
  output, memoized like any other result; resolving it is a fresh run the caller drives, exactly as
  without durability. There is no run-suspension to resume.
- **`in_durable_context` is a construction-time flag** (`durable=`), not a runtime check. Unlike an
  out-of-process engine, there is no Airflow-task context to detect at run time, so attaching the
  capability is what makes the run memoized. Running it outside an Airflow task still memoizes.
- **Positional keying is order-sensitive.** Changing the number or order of operations (adding or
  removing a tool, an opt-out, an `event_stream_handler`) invalidates in-flight stores. Tool calls
  run sequentially inside a memoized run so their positions are deterministic.
- **A step is stored after it runs.** An interruption between a tool's side effect and its store
  write re-runs the tool on retry, so keep tool side effects idempotent.
- **`ctx.enqueue()` is not available** inside a memoized tool call or event handler, because a retry
  serves the stored result without re-running the code and would drop the enqueued messages.

## Per-tool configuration

Memoization takes no per-operation options, so the only tool metadata is an opt-out:
`metadata={'airflow': False}` runs a function tool inline (un-memoized) on every attempt. MCP tools
cannot opt out, because they perform I/O that must not re-run on a retry.

## Dependencies

None beyond `pydantic-ai`. The reference works with the bundled in-memory or JSON-file storage
backends and needs no `apache-airflow` install; a real Airflow storage backend would live in the
provider.

## Further reading

- [Airflow `common.ai` durable provider](https://github.com/apache/airflow/tree/main/providers/common/ai)
- [Pydantic AI durable execution](https://pydantic.dev/docs/ai/durable_execution/overview/)
- [Airflow Durability source code](https://github.com/pydantic/pydantic-ai-harness/tree/main/pydantic_ai_harness/airflow/)
- [Pydantic AI Harness version policy](https://github.com/pydantic/pydantic-ai-harness#version-policy)
