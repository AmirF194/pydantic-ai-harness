"""Tests for model checkpointing and run integration in `AbsurdDurability`.

The cases map to `pydantic-ai-absurd/tests/test_model.py`,
`pydantic-ai-absurd/tests/test_agent.py`, and `pydantic-ai-absurd/tests/test_durability.py`. The
ordinary checkpoint/replay cases use the real PostgreSQL-backed task context; fake contexts are
reserved for deterministic validation and edge cases.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterable, AsyncIterator

import pytest

pytest.importorskip('absurd_sdk')

from absurd_sdk import AsyncAbsurd, JsonValue
from pydantic_ai import Agent
from pydantic_ai.capabilities import ResolveModelId
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import (
    AgentStreamEvent,
    FunctionToolCallEvent,
    ModelMessage,
    ModelResponse,
    PartDeltaEvent,
    PartStartEvent,
    TextPart,
)
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel
from pydantic_ai.tools import RunContext
from pydantic_ai.toolsets import FunctionToolset

from pydantic_ai_harness.absurd import AbsurdDurability

from ._helpers import _text_model
from ._postgres import reenter_running_task, running_task_context
from .conftest import FakeAsyncTaskContext, FakeSyncTaskContext, absurd_task_context

pytestmark = pytest.mark.anyio


class TestTransparency:
    async def test_run_outside_task_is_transparent(self) -> None:
        counter = {'calls': 0}
        agent = Agent(_text_model(counter), name='a', capabilities=[AbsurdDurability()])
        result = await agent.run('hi')
        assert result.output == 'ok'
        assert counter['calls'] == 1


class TestConcurrentRuns:
    async def test_overlapping_runs_in_same_context_and_namespace_are_rejected(self) -> None:
        started = asyncio.Event()
        release = asyncio.Event()

        async def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            started.set()
            await release.wait()
            return ModelResponse(parts=[TextPart(content='ok')])

        agent = Agent(FunctionModel(model_fn), name='same', capabilities=[AbsurdDurability()])
        ctx = FakeAsyncTaskContext()
        with absurd_task_context(ctx):
            first = asyncio.create_task(agent.run('first'))
            await started.wait()
            try:
                with pytest.raises(UserError, match='Concurrent Absurd agent runs'):
                    await agent.run('second')
            finally:
                release.set()
                await first

    async def test_same_namespace_is_allowed_in_distinct_task_contexts(self) -> None:
        started = asyncio.Event()
        release = asyncio.Event()
        started_count = 0

        async def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            nonlocal started_count
            started_count += 1
            if started_count == 2:
                started.set()
            await release.wait()
            return ModelResponse(parts=[TextPart(content='ok')])

        agent = Agent(FunctionModel(model_fn), name='same', capabilities=[AbsurdDurability()])

        async def run_in_context(task_ctx: FakeAsyncTaskContext) -> str:
            with absurd_task_context(task_ctx):
                return (await agent.run('run')).output

        first = asyncio.create_task(run_in_context(FakeAsyncTaskContext()))
        second = asyncio.create_task(run_in_context(FakeAsyncTaskContext()))
        await started.wait()
        release.set()
        assert await asyncio.gather(first, second) == ['ok', 'ok']

    async def test_distinct_names_are_allowed_in_same_task_context(self) -> None:
        started = asyncio.Event()
        release = asyncio.Event()
        started_count = 0

        async def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            nonlocal started_count
            started_count += 1
            if started_count == 2:
                started.set()
            await release.wait()
            return ModelResponse(parts=[TextPart(content='ok')])

        agent_a = Agent(FunctionModel(model_fn), name='a', capabilities=[AbsurdDurability()])
        agent_b = Agent(FunctionModel(model_fn), name='b', capabilities=[AbsurdDurability()])
        ctx = FakeAsyncTaskContext()
        with absurd_task_context(ctx):
            first = asyncio.create_task(agent_a.run('run'))
            second = asyncio.create_task(agent_b.run('run'))
            await started.wait()
            release.set()
            assert [result.output for result in await asyncio.gather(first, second)] == ['ok', 'ok']

    async def test_run_namespace_is_released_after_failure(self) -> None:
        attempts = 0

        async def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise RuntimeError('model failed')
            return ModelResponse(parts=[TextPart(content='ok')])

        agent = Agent(FunctionModel(model_fn), name='recover', capabilities=[AbsurdDurability()])
        ctx = FakeAsyncTaskContext()
        with absurd_task_context(ctx):
            with pytest.raises(RuntimeError, match='model failed'):
                await agent.run('first')
            result = await agent.run('second')

        assert result.output == 'ok'


class TestModelRequestCheckpoint:
    async def test_request_checkpointed_and_replay_serves_cache(self, absurd_client: AsyncAbsurd) -> None:
        counter = {'calls': 0}
        agent = Agent(_text_model(counter), name='a', capabilities=[AbsurdDurability()])

        async with running_task_context(absurd_client, 'model', max_attempts=2) as ctx:
            first = await agent.run('hi')
            task_id = ctx.task_id

        async with reenter_running_task(absurd_client, task_id):
            second = await agent.run('hi')

        assert counter['calls'] == 1
        assert first.output == second.output == 'ok'


class TestStreaming:
    async def test_stream_checkpointed_and_replayed(self, absurd_client: AsyncAbsurd) -> None:
        counter = {'calls': 0}
        agent = Agent(_text_model(counter), name='stream', capabilities=[AbsurdDurability()])

        async with running_task_context(absurd_client, 'stream', max_attempts=2) as ctx:
            async with agent.run_stream('hi') as result:
                first_out = await result.get_output()
            task_id = ctx.task_id
        assert first_out == 'ok'

        async with reenter_running_task(absurd_client, task_id):
            async with agent.run_stream('hi') as result:
                replay_out = await result.get_output()

        assert replay_out == 'ok'
        assert counter['calls'] == 1

    async def test_handler_sees_live_events_and_stream_replays_equal(self, absurd_client: AsyncAbsurd) -> None:
        live_events: list[AgentStreamEvent] = []

        async def handler(run_ctx: RunContext[object], stream: AsyncIterable[AgentStreamEvent]) -> None:
            async for event in stream:
                live_events.append(event)

        counter = {'calls': 0}
        agent = Agent(
            _text_model(counter), name='stream', capabilities=[AbsurdDurability(event_stream_handler=handler)]
        )

        async with running_task_context(absurd_client, 'stream-events', max_attempts=2) as ctx:
            async with agent.run_stream_events('hi') as stream:
                first_events = [event async for event in stream]
            task_id = ctx.task_id

        # The model-stream events were delivered live to the handler inside the request_stream step.
        assert any(isinstance(event, (PartStartEvent, PartDeltaEvent)) for event in live_events)
        live_event_count = len(live_events)

        async with reenter_running_task(absurd_client, task_id):
            async with agent.run_stream_events('hi') as stream:
                replay_events = [event async for event in stream]

        assert counter['calls'] == 1
        assert replay_events == first_events
        assert len(live_events) == live_event_count


class TestModelSelection:
    async def test_registered_model_folds_id_into_step_and_replays(self) -> None:
        primary = {'n': 0}
        cheap = {'n': 0}

        def primary_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            primary['n'] += 1
            return ModelResponse(parts=[TextPart(content='primary')])

        def cheap_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            cheap['n'] += 1
            return ModelResponse(parts=[TextPart(content='cheap')])

        agent = Agent(
            FunctionModel(primary_fn, model_name='primary'),
            name='sw',
            capabilities=[AbsurdDurability(models={'cheap': FunctionModel(cheap_fn, model_name='cheap')})],
        )

        ctx = FakeAsyncTaskContext()
        with absurd_task_context(ctx):
            default_result = await agent.run('hi')
            cheap_result = await agent.run('hi', model='cheap')

        assert default_result.output == 'primary'
        assert cheap_result.output == 'cheap'
        assert 'sw__model.request' in ctx.stored
        assert 'sw__model.request.cheap' in ctx.stored

        replay = ctx.replay()
        with absurd_task_context(replay):
            replayed = await agent.run('hi', model='cheap')

        assert replayed.output == 'cheap'
        assert cheap['n'] == 1
        assert replay.invoked == []

    async def test_model_id_with_hash_is_rejected(self) -> None:
        with pytest.raises(UserError, match='contains'):
            AbsurdDurability(models={'cheap#2': FunctionModel(lambda m, i: ModelResponse(parts=[]), model_name='c')})

    async def test_runtime_model_id_with_hash_is_rejected(self) -> None:
        # A `run(model=...)` string never passes through the `models=` check, so the same collision
        # has to be caught where the id is folded into the step name.
        cheap = FunctionModel(lambda m, i: ModelResponse(parts=[TextPart(content='cheap')]), model_name='cheap')
        agent = Agent(
            _text_model(),
            name='sw',
            capabilities=[
                ResolveModelId(lambda ctx, model_id: cheap if model_id == 'cheap#2' else None),
                AbsurdDurability(),
            ],
        )

        with absurd_task_context(FakeAsyncTaskContext()):
            with pytest.raises(UserError, match='Model id .* contains'):
                await agent.run('hi', model='cheap#2')

    async def test_string_default_model_with_hash_is_allowed(self) -> None:
        # The default model's id is never folded into the step name, so it cannot collide with
        # Absurd's `#` counter suffix and the `#` guard does not apply to it. The suffix is
        # suppressed before the guard runs, so the step keeps the plain unsuffixed name.
        cheap = FunctionModel(lambda m, i: ModelResponse(parts=[TextPart(content='cheap')]), model_name='cheap')
        agent = Agent(
            'cheap#2',
            name='hashdef',
            capabilities=[
                ResolveModelId(lambda ctx, model_id: cheap if model_id == 'cheap#2' else None),
                AbsurdDurability(),
            ],
        )

        ctx = FakeAsyncTaskContext()
        with absurd_task_context(ctx):
            result = await agent.run('hi')

        assert result.output == 'cheap'
        assert list(ctx.stored) == ['hashdef__model.request']

    async def test_string_default_model_gets_unsuffixed_step_name(self) -> None:
        agent = Agent('test', name='strdef', capabilities=[AbsurdDurability()])

        ctx = FakeAsyncTaskContext()
        with absurd_task_context(ctx):
            await agent.run('hi')

        assert 'strdef__model.request' in ctx.stored
        assert not any(name.endswith('.test') for name in ctx.stored)


class TestSyncContext:
    async def test_sync_task_context_raises(self) -> None:
        agent = Agent(_text_model(), name='a', capabilities=[AbsurdDurability()])
        with absurd_task_context(FakeSyncTaskContext()):
            with pytest.raises(UserError, match='requires an async Absurd task context'):
                await agent.run('hi')


class TestEventStreamHandler:
    async def test_handler_events_are_checkpointed(self, absurd_client: AsyncAbsurd) -> None:
        events: list[AgentStreamEvent] = []
        model_calls = {'n': 0}
        greet_calls = {'n': 0}

        async def handler(run_ctx: RunContext[object], stream: AsyncIterable[AgentStreamEvent]) -> None:
            async for event in stream:
                events.append(event)

        async def stream_fn(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | DeltaToolCalls]:
            model_calls['n'] += 1
            if len(messages) == 1:
                yield {0: DeltaToolCall(name='greet', json_args='{}')}
            else:
                yield 'done'

        toolset = FunctionToolset(id='tools')

        @toolset.tool_plain
        def greet() -> str:
            greet_calls['n'] += 1
            return 'hello'

        agent = Agent(
            FunctionModel(stream_function=stream_fn, model_name='fn'),
            name='ev',
            toolsets=[toolset],
            capabilities=[AbsurdDurability(event_stream_handler=handler)],
        )

        async with running_task_context(absurd_client, 'event-handler', max_attempts=2) as ctx:
            result = await agent.run('hi')
            task_id = ctx.task_id

        assert result.output == 'done'
        assert any(isinstance(event, FunctionToolCallEvent) for event in events)
        event_count = len(events)
        assert model_calls['n'] == 2
        assert greet_calls['n'] == 1

        async with reenter_running_task(absurd_client, task_id):
            replayed = await agent.run('hi')

        assert replayed.output == 'done'
        assert len(events) == event_count
        assert model_calls['n'] == 2
        assert greet_calls['n'] == 1


class TestCancelSuspendedResponse:
    async def test_cancel_suspended_response_is_checkpointed(self, absurd_client: AsyncAbsurd) -> None:
        # Drive the cancel through a real run: the model first returns a `'suspended'` response, so
        # the agent re-issues it as a continuation; the continuation request then fails, and the
        # graph tears down the suspended job via `cancel_suspended_response`, which Absurd checkpoints.
        cancelled: list[ModelResponse] = []

        class CancellableModel(FunctionModel):
            async def cancel_suspended_response(self, response: ModelResponse) -> None:
                cancelled.append(response)

        def fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            if not any(m.parts and getattr(m, 'state', None) == 'suspended' for m in messages):
                return ModelResponse(parts=[TextPart(content='partial')], state='suspended')
            raise RuntimeError('continuation failed')

        agent = Agent(CancellableModel(fn, model_name='fn'), name='a', capabilities=[AbsurdDurability()])

        async with running_task_context(absurd_client, 'cancel-suspended', max_attempts=2) as ctx:
            with pytest.raises(RuntimeError, match='continuation failed'):
                await agent.run('hi')
            task_id = ctx.task_id

        assert len(cancelled) == 1
        assert cancelled[0].state == 'suspended'

        async with reenter_running_task(absurd_client, task_id):
            with pytest.raises(RuntimeError, match='continuation failed'):
                await agent.run('hi')

        # The continuation request must re-run because it failed before checkpointing, while the
        # cancellation callback is served from its PostgreSQL checkpoint.
        assert len(cancelled) == 1


class TestCheckpointFormat:
    async def test_hand_written_model_request_payload_replays(self) -> None:
        counter = {'calls': 0}
        agent = Agent(_text_model(counter), name='gold', capabilities=[AbsurdDurability()])

        # A hand-authored checkpoint payload pins the persistence format for a model request.
        payload: JsonValue = {
            'parts': [{'content': 'golden-response', 'part_kind': 'text'}],
            'model_name': 'fn',
            'kind': 'response',
        }
        ctx = FakeAsyncTaskContext(store={'gold__model.request': payload})
        with absurd_task_context(ctx):
            result = await agent.run('hi')

        assert result.output == 'golden-response'
        assert counter['calls'] == 0

    async def test_hand_written_stream_payload_replays(self) -> None:
        counter = {'calls': 0}
        agent = Agent(_text_model(counter), name='gold', capabilities=[AbsurdDurability()])

        # A hand-authored stream checkpoint pins the `{response, events}` payload shape.
        payload: JsonValue = {
            'response': {
                'parts': [{'content': 'golden-stream', 'part_kind': 'text'}],
                'model_name': 'fn',
                'kind': 'response',
            },
            'events': [
                {'index': 0, 'part': {'content': 'golden-stream', 'part_kind': 'text'}, 'event_kind': 'part_start'}
            ],
        }
        ctx = FakeAsyncTaskContext(store={'gold__model.request_stream': payload})
        with absurd_task_context(ctx):
            async with agent.run_stream('hi') as result:
                out = await result.get_output()

        assert out == 'golden-stream'
        assert counter['calls'] == 0
