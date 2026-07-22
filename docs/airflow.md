---
title: Airflow Durability
description: Memoize a Pydantic AI agent's model and tool I/O into a pluggable store so an Apache Airflow task retry replays completed work instead of repeating it.
---

# Airflow Durability

`AirflowDurability` makes an agent recover across an Apache Airflow task retry by memoizing its I/O.
Every model request, function tool call, MCP call, and dynamic-toolset resolution is stored under a
positional step key plus a fingerprint of the request that produced it. If the agent run fails, the
Airflow task retry re-runs it from the top: each completed operation is served from the store without
re-executing, so finished work is not repeated and tokens are not re-spent, while the operation that
failed and everything after it runs live.

It is the capability-shaped reference version of Airflow's `common.ai` durable provider, which
memoizes the same operations but is attached through an operator rather than a user-attached
capability.

[Source](https://github.com/pydantic/pydantic-ai-harness/tree/main/pydantic_ai_harness/airflow/)

> The API may change between releases. Where practical, breaking changes ship with a deprecation warning.

## How it works

The durable unit is the whole agent run, which maps to one Airflow task. There is no
suspend-and-resume: recovery is a full re-run against a store that outlived the failed attempt.

1. **Positional step keys.** A per-run counter numbers each operation in the order it is reached
   (`model_step_0`, `tool_step_1`, `model_step_2`, ...). The counter advances even on a store hit, so
   a retry that replays earlier steps still reaches later ones at the same positions.
2. **Fingerprint-verified replay.** Each entry stores a fingerprint of its request. On a positional
   hit the current request's fingerprint is compared to the stored one; a mismatch is a miss and the
   step re-runs live. A fresh model response carries new `tool_call_id` values, and `tool_call_id` is
   part of the tool fingerprint, so a model divergence cascades to invalidate the tool steps recorded
   under the old conversation.
3. **Pluggable storage.** The store is a small `DurableStorage` Protocol. An in-memory backend and a
   JSON-file backend ship for offline use; a backend writing to Airflow's ObjectStorage or task state
   store would live in the provider and satisfy the same Protocol.

## Quick start

```python {test="skip"}
from pydantic_ai import Agent

from pydantic_ai_harness.airflow import AirflowDurability, InMemoryDurableStorage

storage = InMemoryDurableStorage()
agent = Agent('openai:gpt-5', name='analyst', capabilities=[AirflowDurability(storage=storage)])


@agent.tool_plain
def get_weather(city: str) -> str:
    return f'It is sunny in {city}.'
```

In an Airflow deployment, construct the agent with a provider-backed storage, run it inside a task,
and let Airflow's retry re-run the task on failure. Pass `durable=False` to make the capability
transparent (a normal, non-memoized run).

## Improvements over the provider it models

- Structured tool results (`ToolReturn`, `BinaryContent`) round-trip through the store; the provider's
  raw `json.dumps` skips them.
- Control-flow signals (`ModelRetry`, `ApprovalRequired`, `CallDeferred`) cross the store as values
  and re-raise on replay.
- MCP tool listings, instructions, and dynamic-toolset resolution are memoized; the provider re-runs
  those live.
- A non-serializable result is skipped with a warning rather than crashing the run.

## Limitations

This is reference-quality, not a drop-in production engine:

- **No suspend/resume.** Recovery is a full re-run from the top. A deferred tool ends the run with a
  `DeferredToolRequests` output, memoized like any other result; resolving it is a fresh run the
  caller drives, exactly as without durability.
- **The active flag is set at construction** (`durable=`), not detected at run time. Unlike an
  out-of-process engine, there is no Airflow-task context to detect, so attaching the capability is
  what makes the run memoized.
- **Positional keying is order-sensitive.** Changing the number or order of operations invalidates
  in-flight stores. Tool calls run sequentially inside a memoized run so their positions are
  deterministic.
- **A step is stored after it runs**, so keep tool side effects idempotent: an interruption between a
  tool's side effect and its store write re-runs the tool on retry.
- **`ctx.enqueue()` is not available** inside a memoized tool call or event handler, because a retry
  serves the stored result without re-running the code.

## Per-tool configuration

Memoization takes no per-operation options. `metadata={'airflow': False}` runs a function tool inline
(un-memoized) on every attempt; MCP tools cannot opt out, because they perform I/O that must not
re-run on a retry.

## API reference

::: pydantic_ai_harness.airflow.AirflowDurability

::: pydantic_ai_harness.airflow.DurableStorage

::: pydantic_ai_harness.airflow.InMemoryDurableStorage

::: pydantic_ai_harness.airflow.JSONFileDurableStorage
