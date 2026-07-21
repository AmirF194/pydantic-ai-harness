"""Tests for the `AbsurdDurability` capability.

Behavior is driven through `Agent(..., capabilities=[AbsurdDurability()])` inside an in-memory
`FakeAsyncTaskContext` (see `conftest.py`) so there is no Postgres or Docker dependency. The two
production behaviors the capability relies on -- encounter-order step-name disambiguation and a
replay that serves stored checkpoints without re-running `fn` -- are reproduced faithfully by the
fake.
"""

from __future__ import annotations

from collections.abc import AsyncIterable, AsyncIterator
from contextlib import AbstractContextManager

import pytest

pytest.importorskip('absurd_sdk')

from absurd_sdk import JsonValue
from pydantic_ai import Agent
from pydantic_ai.agent import ParallelExecutionMode
from pydantic_ai.exceptions import ModelRetry, UserError
from pydantic_ai.messages import (
    AgentStreamEvent,
    FunctionToolCallEvent,
    ModelMessage,
    ModelResponse,
    PartDeltaEvent,
    PartStartEvent,
    RetryPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models import ModelRequestContext, ModelRequestParameters
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel
from pydantic_ai.tools import RunContext
from pydantic_ai.toolsets import ExternalToolset, FunctionToolset
from pydantic_ai.usage import RunUsage

from pydantic_ai_harness.absurd import AbsurdDurability

from .conftest import FakeAsyncTaskContext, FakeSyncTaskContext, absurd_task_context

pytestmark = pytest.mark.anyio


def _text_model(counter: dict[str, int] | None = None) -> FunctionModel:
    tally = counter if counter is not None else {'calls': 0}

    def fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        tally['calls'] += 1
        return ModelResponse(parts=[TextPart(content='ok')])

    async def stream_fn(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        tally['calls'] += 1
        yield 'ok'

    return FunctionModel(fn, stream_function=stream_fn, model_name='fn')


def _tool_then_done_model(tool_name: str, args: dict[str, JsonValue]) -> FunctionModel:
    def fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        answered = any(
            isinstance(part, (ToolReturnPart, RetryPromptPart)) for message in messages for part in message.parts
        )
        if not answered:
            return ModelResponse(parts=[ToolCallPart(tool_name=tool_name, args=args)])
        return ModelResponse(parts=[TextPart(content='done')])

    return FunctionModel(fn, model_name='fn')


class TestTransparency:
    async def test_run_outside_task_is_transparent(self) -> None:
        counter = {'calls': 0}
        agent = Agent(_text_model(counter), name='a', capabilities=[AbsurdDurability()])
        result = await agent.run('hi')
        assert result.output == 'ok'
        assert counter['calls'] == 1


class TestModelRequestCheckpoint:
    async def test_request_checkpointed_and_replay_serves_cache(self) -> None:
        counter = {'calls': 0}
        agent = Agent(_text_model(counter), name='a', capabilities=[AbsurdDurability()])

        ctx = FakeAsyncTaskContext()
        with absurd_task_context(ctx):
            first = await agent.run('hi')
        assert 'a__model.request' in ctx.stored
        assert ctx.invoked == ['a__model.request']

        replay = ctx.replay()
        with absurd_task_context(replay):
            second = await agent.run('hi')

        assert counter['calls'] == 1
        assert first.output == second.output == 'ok'
        assert replay.invoked == []


class TestStreaming:
    async def test_stream_checkpointed_and_replayed(self) -> None:
        counter = {'calls': 0}
        agent = Agent(_text_model(counter), name='stream', capabilities=[AbsurdDurability()])

        ctx = FakeAsyncTaskContext()
        with absurd_task_context(ctx):
            async with agent.run_stream('hi') as result:
                first_out = await result.get_output()
        assert first_out == 'ok'
        assert 'stream__model.request_stream' in ctx.stored

        replay = ctx.replay()
        with absurd_task_context(replay):
            async with agent.run_stream('hi') as result:
                replay_out = await result.get_output()

        assert replay_out == 'ok'
        assert counter['calls'] == 1
        assert replay.invoked == []

    async def test_handler_sees_live_events_and_stream_replays_equal(self) -> None:
        live_events: list[AgentStreamEvent] = []

        async def handler(run_ctx: RunContext[object], stream: AsyncIterable[AgentStreamEvent]) -> None:
            async for event in stream:
                live_events.append(event)

        counter = {'calls': 0}
        agent = Agent(
            _text_model(counter), name='stream', capabilities=[AbsurdDurability(event_stream_handler=handler)]
        )

        ctx = FakeAsyncTaskContext()
        with absurd_task_context(ctx):
            async with agent.run_stream_events('hi') as stream:
                first_events = [event async for event in stream]

        # The model-stream events were delivered live to the handler inside the request_stream step.
        assert any(isinstance(event, (PartStartEvent, PartDeltaEvent)) for event in live_events)
        assert 'stream__model.request_stream' in ctx.stored

        replay = ctx.replay()
        with absurd_task_context(replay):
            async with agent.run_stream_events('hi') as stream:
                replay_events = [event async for event in stream]

        assert counter['calls'] == 1
        assert replay_events == first_events


class TestFunctionTool:
    async def test_tool_checkpointed_exactly_once_across_replay(self) -> None:
        calls = {'n': 0}
        toolset = FunctionToolset(id='billing')

        @toolset.tool_plain
        def charge_card(amount: int) -> str:
            calls['n'] += 1
            return f'charged {amount}'

        agent = Agent(
            _tool_then_done_model('charge_card', {'amount': 7}),
            name='pay',
            toolsets=[toolset],
            capabilities=[AbsurdDurability()],
        )

        ctx = FakeAsyncTaskContext()
        with absurd_task_context(ctx):
            first = await agent.run('charge it')
        assert 'pay__function_toolset__billing.call_tool:charge_card' in ctx.stored

        replay = ctx.replay()
        with absurd_task_context(replay):
            second = await agent.run('charge it')

        assert calls['n'] == 1
        assert first.output == second.output == 'done'
        assert replay.invoked == []


class TestModelRetry:
    async def test_model_retry_crosses_checkpoint_and_replays(self) -> None:
        calls = {'n': 0}
        toolset = FunctionToolset(id='tools')

        @toolset.tool_plain
        def flaky() -> str:
            calls['n'] += 1
            raise ModelRetry('nope, try again')

        agent = Agent(
            _tool_then_done_model('flaky', {}), name='retry', toolsets=[toolset], capabilities=[AbsurdDurability()]
        )

        ctx = FakeAsyncTaskContext()
        with absurd_task_context(ctx):
            first = await agent.run('go')
        step = 'retry__function_toolset__tools.call_tool:flaky'
        # The raised `ModelRetry` crossed the checkpoint as a serialized value, not an exception.
        assert ctx.stored[step] == {'message': 'nope, try again', 'kind': 'model_retry'}
        assert first.output == 'done'
        assert calls['n'] == 1

        replay = ctx.replay()
        with absurd_task_context(replay):
            second = await agent.run('go')

        # On replay the stored `ModelRetry` is re-raised without re-running the tool.
        assert calls['n'] == 1
        assert second.output == 'done'
        assert replay.invoked == []


class TestPerToolOptOut:
    async def test_metadata_false_runs_inline_uncheckpointed(self) -> None:
        calls = {'n': 0}
        toolset = FunctionToolset(id='tools')

        @toolset.tool_plain(metadata={'absurd': False})
        def ping() -> str:
            calls['n'] += 1
            return 'pong'

        agent = Agent(
            _tool_then_done_model('ping', {}), name='inline', toolsets=[toolset], capabilities=[AbsurdDurability()]
        )

        ctx = FakeAsyncTaskContext()
        with absurd_task_context(ctx):
            result = await agent.run('ping it')

        assert result.output == 'done'
        assert calls['n'] == 1
        assert not any('call_tool:ping' in name for name in ctx.stored)


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

    async def test_string_default_model_gets_unsuffixed_step_name(self) -> None:
        agent = Agent('test', name='strdef', capabilities=[AbsurdDurability()])

        ctx = FakeAsyncTaskContext()
        with absurd_task_context(ctx):
            await agent.run('hi')

        assert 'strdef__model.request' in ctx.stored
        assert not any(name.endswith('.test') for name in ctx.stored)


class TestRuntimeToolsets:
    async def test_runtime_executing_toolset_rejected_inside_task(self) -> None:
        agent = Agent(_text_model(), name='a', capabilities=[AbsurdDurability()])
        late = FunctionToolset(id='late')

        @late.tool_plain
        def echo(value: str) -> str:  # pragma: no cover - rejected before it can run
            return value

        ctx = FakeAsyncTaskContext()
        with absurd_task_context(ctx):
            with pytest.raises(UserError, match=r'cannot be passed to `run\(toolsets=...\)` at runtime'):
                await agent.run('hi', toolsets=[late])

    async def test_non_executing_runtime_toolset_allowed_inside_task(self) -> None:
        agent = Agent(_text_model(), name='a', capabilities=[AbsurdDurability()])
        ctx = FakeAsyncTaskContext()
        with absurd_task_context(ctx):
            result = await agent.run('hi', toolsets=[ExternalToolset(tool_defs=[])])
        assert result.output == 'ok'


class TestNonWrappedLeaf:
    async def test_external_toolset_passes_through_unwrapped(self) -> None:
        external = ExternalToolset(tool_defs=[])
        agent = Agent(_text_model(), name='a', toolsets=[external], capabilities=[AbsurdDurability()])
        assert any(leaf is external for leaf in agent.toolsets)


class TestBindingErrors:
    async def test_unnamed_agent_raises(self) -> None:
        with pytest.raises(UserError, match='unique `name`'):
            Agent(_text_model(), capabilities=[AbsurdDurability()])

    async def test_duplicate_toolset_ids_raise(self) -> None:
        first = FunctionToolset(id='dup')

        @first.tool_plain
        def echo(value: str) -> str:  # pragma: no cover - never invoked, only the wrap check runs
            return value

        second = FunctionToolset(id='dup')

        @second.tool_plain
        def shout(value: str) -> str:  # pragma: no cover - never invoked, only the wrap check runs
            return value.upper()

        with pytest.raises(UserError, match='same `id`'):
            Agent(_text_model(), name='a', toolsets=[first, second], capabilities=[AbsurdDurability()])

    async def test_idless_leaf_function_toolset_raises(self) -> None:
        toolset = FunctionToolset()

        @toolset.tool_plain
        def echo(value: str) -> str:  # pragma: no cover - never invoked, only the wrap check runs
            return value

        with pytest.raises(UserError, match='need to have a unique `id`'):
            Agent(_text_model(), name='a', toolsets=[toolset], capabilities=[AbsurdDurability()])


class TestSyncContext:
    async def test_sync_task_context_raises(self) -> None:
        agent = Agent(_text_model(), name='a', capabilities=[AbsurdDurability()])
        with absurd_task_context(FakeSyncTaskContext()):
            with pytest.raises(UserError, match='requires an async Absurd task context'):
                await agent.run('hi')


class TestParallelExecutionMode:
    async def test_parallel_execution_mode_applied_during_run(self, monkeypatch: pytest.MonkeyPatch) -> None:
        agent = Agent(
            _text_model(), name='a', capabilities=[AbsurdDurability(parallel_execution_mode='parallel_ordered_events')]
        )
        recorded: list[ParallelExecutionMode] = []
        real = agent.parallel_tool_call_execution_mode

        def spy(mode: ParallelExecutionMode = 'parallel') -> AbstractContextManager[None]:
            recorded.append(mode)
            return real(mode)

        monkeypatch.setattr(agent, 'parallel_tool_call_execution_mode', spy)

        ctx = FakeAsyncTaskContext()
        with absurd_task_context(ctx):
            await agent.run('hi')

        assert 'parallel_ordered_events' in recorded


class TestEventStreamHandler:
    async def test_handler_events_are_checkpointed(self) -> None:
        events: list[AgentStreamEvent] = []

        async def handler(run_ctx: RunContext[object], stream: AsyncIterable[AgentStreamEvent]) -> None:
            async for event in stream:
                events.append(event)

        async def stream_fn(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | DeltaToolCalls]:
            if len(messages) == 1:
                yield {0: DeltaToolCall(name='greet', json_args='{}')}
            else:
                yield 'done'

        toolset = FunctionToolset(id='tools')

        @toolset.tool_plain
        def greet() -> str:
            return 'hello'

        agent = Agent(
            FunctionModel(stream_function=stream_fn, model_name='fn'),
            name='ev',
            toolsets=[toolset],
            capabilities=[AbsurdDurability(event_stream_handler=handler)],
        )

        ctx = FakeAsyncTaskContext()
        with absurd_task_context(ctx):
            result = await agent.run('hi')

        assert result.output == 'done'
        assert any(isinstance(event, FunctionToolCallEvent) for event in events)
        assert 'ev__event_stream_handler' in ctx.stored


class TestCancelSuspendedResponse:
    async def test_cancel_suspended_response_is_checkpointed(self) -> None:
        cancelled: list[ModelResponse] = []

        class CancellableModel(FunctionModel):
            async def cancel_suspended_response(self, response: ModelResponse) -> None:
                cancelled.append(response)

        def fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:  # pragma: no cover - never requested
            return ModelResponse(parts=[TextPart(content='ok')])

        model = CancellableModel(fn, model_name='fn')
        agent = Agent(model, name='a', capabilities=[AbsurdDurability()])
        bound = AbsurdDurability.from_agent(agent)
        assert bound is not None

        run_ctx = RunContext[object](deps=None, model=model, usage=RunUsage())
        request_context = ModelRequestContext(
            model=model, messages=[], model_settings=None, model_request_parameters=ModelRequestParameters()
        )
        response = ModelResponse(parts=[TextPart(content='suspended')])

        async def handler(request: ModelRequestContext) -> ModelResponse:
            await request.model.cancel_suspended_response(response)
            return response

        ctx = FakeAsyncTaskContext()
        with absurd_task_context(ctx):
            result = await bound.wrap_model_request(run_ctx, request_context=request_context, handler=handler)

        assert result is response
        assert cancelled == [response]
        assert 'a__model.cancel_suspended_response' in ctx.stored


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
