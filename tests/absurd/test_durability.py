from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterable, AsyncIterator
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Any

import anyio
import pytest
from absurd_sdk import AsyncAbsurd, AsyncTaskContext, JsonValue, TaskContext, _current_task_context
from pydantic import TypeAdapter
from pydantic_ai import Agent, ModelMessage, ModelResponse
from pydantic_ai.agent import ParallelExecutionMode
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.durable_exec._toolset import wrap_tool_call_result
from pydantic_ai.exceptions import ApprovalRequired, CallDeferred, ModelRetry, ToolFailed, UserError
from pydantic_ai.messages import (
    AgentStreamEvent,
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models import Model, ModelRequestContext, ModelRequestParameters, ModelResolutionContext
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, FunctionModel
from pydantic_ai.tools import RunContext
from pydantic_ai.toolsets import DynamicToolset, ExternalToolset, FunctionToolset
from pydantic_ai.usage import RunUsage

from pydantic_ai_harness.absurd import AbsurdDurability
from pydantic_ai_harness.absurd._tool_result import (
    serialize_tool_call_result,
    unwrap_tool_call_checkpoint,
)
from pydantic_ai_harness.absurd._utils import current_async_context

from .conftest import reenter_running_task, running_task_context

pytestmark = pytest.mark.anyio


@dataclass
class _LogicalModelResolver(AbstractCapability[object]):
    target: Model

    async def resolve_model_id(self, ctx: ModelResolutionContext[object], *, model_id: str) -> Model | None:
        if model_id == 'platform-default':
            return self.target
        return None  # pragma: no cover - resolver deliberately declines unrelated model IDs


def _text_model(calls: dict[str, int] | None = None) -> FunctionModel:
    tally = calls if calls is not None else {'calls': 0}

    def fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        tally['calls'] += 1
        return ModelResponse(parts=[TextPart(content='ok')])

    async def stream_fn(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        tally['calls'] += 1
        yield 'ok'

    return FunctionModel(fn, stream_function=stream_fn, model_name='fn')


async def _noop(params: JsonValue, ctx: AsyncTaskContext) -> JsonValue:
    return None  # pragma: no cover - task body is not entered by these agent-run tests


def _tool_model(tool_name: str, args: dict[str, JsonValue]) -> FunctionModel:
    def fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if any(isinstance(part, ToolReturnPart) for message in messages for part in message.parts):
            return ModelResponse(parts=[TextPart(content='done')])
        return ModelResponse(parts=[ToolCallPart(tool_name=tool_name, args=args)])

    return FunctionModel(fn, model_name='fn')


async def test_requires_agent_name() -> None:
    with pytest.raises(UserError, match='unique `name`'):
        Agent(_text_model(), capabilities=[AbsurdDurability()])


def test_construction_dynamic_toolset_is_rejected() -> None:
    def build_toolset(ctx: RunContext[object]) -> FunctionToolset[object] | None:
        return None  # pragma: no cover - construction rejects the dynamic toolset before resolution

    with pytest.raises(UserError, match='DynamicToolset is not supported.*not checkpointed'):
        Agent(
            _text_model(),
            name='dynamic-toolset',
            toolsets=[DynamicToolset(build_toolset, id='dynamic')],
            capabilities=[AbsurdDurability()],
        )


async def test_duplicate_explicit_toolset_ids_are_rejected() -> None:
    first = FunctionToolset[object](id='duplicate')
    second = FunctionToolset[object](id='duplicate')

    with pytest.raises(UserError, match="same `id` 'duplicate'"):
        Agent(
            _text_model(),
            name='duplicate-toolsets',
            toolsets=[first, second],
            capabilities=[AbsurdDurability()],
        )


async def test_invalid_parallel_mode_is_rejected() -> None:
    with pytest.raises(UserError, match='parallel execution mode must be'):
        AbsurdDurability(parallel_execution_mode='parallel')  # pyright: ignore[reportArgumentType]


async def test_run_outside_task_is_transparent() -> None:
    calls = {'calls': 0}
    agent = Agent(_text_model(calls), name='outside', capabilities=[AbsurdDurability()])
    result = await agent.run('hello')
    assert result.output == 'ok'
    assert calls['calls'] == 1


async def test_function_toolset_is_transparent_outside_task() -> None:
    calls = {'calls': 0}
    toolset = FunctionToolset[object]()

    @toolset.tool_plain
    def charge(amount: int) -> str:
        calls['calls'] += 1
        return f'charged {amount}'

    agent = Agent[object, str](
        _tool_model('charge', {'amount': 7}),
        name='outside-function',
        toolsets=[toolset],
        capabilities=[AbsurdDurability()],
    )
    result = await agent.run('charge it')
    assert result.output == 'done'
    assert calls['calls'] == 1


async def test_function_tool_control_flow_is_checkpointed_and_replayed(absurd: AsyncAbsurd) -> None:
    calls = {'calls': 0}
    toolset = FunctionToolset[object](id='billing')

    @toolset.tool_plain
    def flaky() -> str:
        calls['calls'] += 1
        if calls['calls'] == 1:
            raise ModelRetry('try again')
        return 'recovered'

    agent = Agent[object, str](
        _tool_model('flaky', {}),
        name='control-flow',
        toolsets=[toolset],
        capabilities=[AbsurdDurability()],
    )
    absurd.register_task(name='control-flow')(_noop)

    async with running_task_context(absurd, 'control-flow', max_attempts=2) as ctx:
        first = await agent.run('go')
        task_id = ctx.task_id
        checkpoint = ctx._checkpoint_cache['control-flow__function_toolset__billing.call_tool:flaky']
        assert checkpoint == {
            '__pydantic_ai_harness_call_tool_result__': {
                'version': 1,
                'result': {'message': 'try again', 'kind': 'model_retry'},
            }
        }

    async with reenter_running_task(absurd, task_id):
        replay = await agent.run('go')

    assert first.output == replay.output == 'done'
    assert calls['calls'] == 2


@pytest.mark.parametrize(
    ('payload', 'error'),
    [
        ({'__pydantic_ai_harness_call_tool_result__': 'not an object'}, 'Malformed'),
        (
            {'__pydantic_ai_harness_call_tool_result__': {'version': True, 'result': None}},
            'Unsupported',
        ),
        (
            {'__pydantic_ai_harness_call_tool_result__': {'version': 2, 'result': None}},
            'Unsupported',
        ),
    ],
)
def test_tool_result_checkpoint_rejects_malformed_envelopes(payload: JsonValue, error: str) -> None:
    with pytest.raises(UserError, match=error):
        unwrap_tool_call_checkpoint(payload)


def test_tool_result_checkpoint_preserves_raw_legacy_value() -> None:
    raw: JsonValue = {'kind': 'tool_return', 'result': 'legacy'}
    assert unwrap_tool_call_checkpoint(raw) == raw


async def test_successful_tool_result_retains_standalone_raw_shape() -> None:
    scalar_result = await wrap_tool_call_result(_successful_tool_call())
    assert serialize_tool_call_result(scalar_result) == 'ok'

    raw_result = await wrap_tool_call_result(_raw_tool_content_call())
    assert serialize_tool_call_result(raw_result) == {'kind': 'tool-return', 'result': 'raw'}


async def _successful_tool_call() -> str:
    return 'ok'


async def _raw_tool_content_call() -> dict[str, JsonValue]:
    return {'kind': 'tool-return', 'result': 'raw'}


@pytest.mark.parametrize(
    ('value', 'result_kind'),
    [
        ({'__pydantic_ai_harness_call_tool_result__': 'not an envelope'}, 'tool_return'),
        (
            {
                '__pydantic_ai_harness_call_tool_result__': {
                    'version': 1,
                    'result': {'kind': 'model_retry', 'message': 'user data'},
                }
            },
            'tool_return',
        ),
        (
            {'kind': 'tool-return', 'result': 'user data', '__pydantic_ai_harness_call_tool_result__': 'reserved'},
            'tool_content_result',
        ),
    ],
)
async def test_successful_reserved_key_dict_round_trips_exactly(value: JsonValue, result_kind: str) -> None:
    async def return_value() -> JsonValue:
        return value

    wrapped = await wrap_tool_call_result(return_value())
    checkpoint = serialize_tool_call_result(wrapped)
    assert checkpoint == {
        '__pydantic_ai_harness_call_tool_result__': {
            'version': 1,
            'result': {'kind': result_kind, 'result': value},
        }
    }
    checkpoint_json: JsonValue = json.loads(json.dumps(checkpoint))
    assert unwrap_tool_call_checkpoint(checkpoint_json) == value


@pytest.mark.parametrize(
    ('exception', 'expected'),
    [
        (ModelRetry('retry'), {'message': 'retry', 'kind': 'model_retry'}),
        (
            ApprovalRequired(metadata={'request_id': 'approval'}),
            {'metadata': {'request_id': 'approval'}, 'kind': 'approval_required'},
        ),
        (
            CallDeferred(metadata={'request_id': 'deferred'}),
            {'metadata': {'request_id': 'deferred'}, 'kind': 'call_deferred'},
        ),
        (ToolFailed('failed'), {'message': 'failed', 'kind': 'tool_failed'}),
    ],
)
async def test_control_flow_tool_results_round_trip(
    exception: ModelRetry | ApprovalRequired | CallDeferred | ToolFailed, expected: dict[str, JsonValue]
) -> None:
    async def raise_exception() -> None:
        raise exception

    wrapped = await wrap_tool_call_result(raise_exception())
    assert serialize_tool_call_result(wrapped) == {
        '__pydantic_ai_harness_call_tool_result__': {'version': 1, 'result': expected}
    }
    payload: JsonValue = {'__pydantic_ai_harness_call_tool_result__': {'version': 1, 'result': expected}}

    with pytest.raises(type(exception)) as raised:
        unwrap_tool_call_checkpoint(payload)
    if isinstance(exception, ApprovalRequired):
        assert isinstance(raised.value, ApprovalRequired)
        assert raised.value.metadata == exception.metadata
    elif isinstance(exception, CallDeferred):
        assert isinstance(raised.value, CallDeferred)
        assert raised.value.metadata == exception.metadata
    elif isinstance(exception, ModelRetry):
        assert isinstance(raised.value, ModelRetry)
        assert raised.value.message == exception.message
    else:
        assert isinstance(raised.value, ToolFailed)
        assert raised.value.message == exception.message


async def test_same_function_toolset_instance_is_wrapped_once() -> None:
    toolset = FunctionToolset[object](id='shared')
    agent = Agent(_text_model(), name='shared-toolset', toolsets=[toolset, toolset], capabilities=[AbsurdDurability()])
    bound = AbsurdDurability.from_agent(agent)
    assert bound is not None
    assert bound.get_wrapper_toolset(toolset) is bound.get_wrapper_toolset(toolset)


async def test_construction_external_toolset_passes_through() -> None:
    external = ExternalToolset[object](tool_defs=[])
    agent = Agent(_text_model(), name='external-toolset', toolsets=[external], capabilities=[AbsurdDurability()])
    assert any(toolset is external for toolset in agent.toolsets)


async def test_outside_task_leaves_parallel_mode_untouched(monkeypatch: pytest.MonkeyPatch) -> None:
    agent = Agent(
        _text_model(),
        name='outside-parallel',
        capabilities=[AbsurdDurability(parallel_execution_mode='parallel_ordered_events')],
    )
    recorded: list[ParallelExecutionMode] = []
    real = agent.parallel_tool_call_execution_mode

    def spy(
        mode: ParallelExecutionMode = 'parallel',
    ) -> AbstractContextManager[None]:  # pragma: no cover - outside-task run must not call it
        recorded.append(mode)
        return real(mode)

    monkeypatch.setattr(agent, 'parallel_tool_call_execution_mode', spy)
    await agent.run('hello')
    assert recorded == []


def test_sync_task_context_is_rejected() -> None:
    dummy: Any = object.__new__(TaskContext)
    token = _current_task_context.set(dummy)
    try:
        with pytest.raises(UserError, match='async Absurd task context'):
            current_async_context()
    finally:
        _current_task_context.reset(token)


async def test_string_default_model_resolver_replays_suffixless_checkpoint(absurd: AsyncAbsurd) -> None:
    calls = {'calls': 0}
    resolver = _LogicalModelResolver(target=_text_model(calls))
    agent = Agent('platform-default', name='platform-model', capabilities=[resolver, AbsurdDurability()])
    absurd.register_task(name='platform-model')(_noop)

    async with running_task_context(absurd, 'platform-model', max_attempts=2) as ctx:
        first = await agent.run('hello')
        task_id = ctx.task_id
        assert 'platform-model__model.request' in ctx._checkpoint_cache
        assert 'platform-model__model.request.platform-default' not in ctx._checkpoint_cache

    async with reenter_running_task(absurd, task_id):
        replay = await agent.run('hello')

    assert first.output == replay.output == 'ok'
    assert calls['calls'] == 1


async def test_model_request_is_checkpointed_and_replayed(absurd: AsyncAbsurd) -> None:
    calls = {'calls': 0}
    agent = Agent(_text_model(calls), name='model', capabilities=[AbsurdDurability()])
    absurd.register_task(name='model')(_noop)

    async with running_task_context(absurd, 'model', max_attempts=2) as ctx:
        first = await agent.run('hello')
        task_id = ctx.task_id

    async with reenter_running_task(absurd, task_id):
        replay = await agent.run('hello')

    assert first.output == replay.output == 'ok'
    assert calls['calls'] == 1


async def test_cancel_suspended_response_is_checkpointed(absurd: AsyncAbsurd) -> None:
    cancelled: list[ModelResponse] = []

    class CancellableModel(FunctionModel):
        async def cancel_suspended_response(self, response: ModelResponse) -> None:
            cancelled.append(response)

    model = CancellableModel(lambda messages, info: ModelResponse(parts=[TextPart(content='ok')]), model_name='cancel')
    agent = Agent(model, name='cancel', capabilities=[AbsurdDurability()])
    bound = AbsurdDurability.from_agent(agent)
    assert bound is not None
    run_context = RunContext[object](deps=None, model=model, usage=RunUsage())
    request_context = ModelRequestContext(
        model=model, messages=[], model_settings=None, model_request_parameters=ModelRequestParameters()
    )
    response = ModelResponse(parts=[TextPart(content='suspended')])

    async def handler(request: ModelRequestContext) -> ModelResponse:
        await request.model.cancel_suspended_response(response)
        return response

    absurd.register_task(name='cancel')(_noop)
    async with running_task_context(absurd, 'cancel') as task_ctx:
        result = await bound.wrap_model_request(run_context, request_context=request_context, handler=handler)
        task_id = task_ctx.task_id
        assert task_ctx._checkpoint_cache['cancel__model.cancel_suspended_response'] is None

    replay_context = RunContext[object](deps=None, model=model, usage=RunUsage())
    replay_request = ModelRequestContext(
        model=model, messages=[], model_settings=None, model_request_parameters=ModelRequestParameters()
    )
    async with reenter_running_task(absurd, task_id):
        replay = await bound.wrap_model_request(replay_context, request_context=replay_request, handler=handler)

    assert result is response
    assert replay is response
    assert cancelled == [response]


async def test_streamed_response_is_checkpointed_and_replayed(absurd: AsyncAbsurd) -> None:
    calls = {'calls': 0}
    agent = Agent(_text_model(calls), name='stream', capabilities=[AbsurdDurability()])
    absurd.register_task(name='stream')(_noop)

    async with running_task_context(absurd, 'stream', max_attempts=2) as ctx:
        async with agent.run_stream('hello') as result:
            first = await result.get_output()
        task_id = ctx.task_id

    async with reenter_running_task(absurd, task_id):
        async with agent.run_stream('hello') as result:
            replay = await result.get_output()

    assert first == replay == 'ok'
    assert calls['calls'] == 1


async def test_event_stream_handler_is_checkpointed_and_replayed(absurd: AsyncAbsurd) -> None:
    calls = {'calls': 0}
    handled: list[AgentStreamEvent] = []

    async def handler(ctx: RunContext[object], stream: AsyncIterable[AgentStreamEvent]) -> None:
        async for event in stream:
            handled.append(event)

    toolset = FunctionToolset[object](id='event-tools')

    @toolset.tool_plain
    def echo() -> str:
        return 'echoed'

    async def stream_fn(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | dict[int, DeltaToolCall]]:
        calls['calls'] += 1
        if any(isinstance(part, ToolReturnPart) for message in messages for part in message.parts):
            yield 'done'
        else:
            yield {0: DeltaToolCall(name='echo', json_args='{}', tool_call_id='echo-1')}

    agent = Agent(
        FunctionModel(stream_function=stream_fn, model_name='stream-events'),
        name='stream-handler',
        toolsets=[toolset],
        capabilities=[AbsurdDurability(event_stream_handler=handler)],
    )
    absurd.register_task(name='stream-handler')(_noop)

    async with running_task_context(absurd, 'stream-handler', max_attempts=2) as ctx:
        async with agent.run_stream_events('hello') as stream:
            first = [event async for event in stream]
        task_id = ctx.task_id
        assert any(isinstance(event, (FunctionToolCallEvent, FunctionToolResultEvent)) for event in handled)
        assert 'stream-handler__event_stream_handler' in ctx._checkpoint_cache
    handled_count = len(handled)

    async with reenter_running_task(absurd, task_id):
        async with agent.run_stream_events('hello') as stream:
            replay = [event async for event in stream]

    assert [type(event) for event in replay] == [type(event) for event in first]
    assert [event for event in replay if isinstance(event, (FunctionToolCallEvent, FunctionToolResultEvent))]
    assert handled_count > 0
    assert len(handled) == handled_count
    assert calls['calls'] == 2


async def test_legacy_raw_stream_checkpoint_is_replayed(absurd: AsyncAbsurd) -> None:
    calls = {'calls': 0}
    agent = Agent(_text_model(calls), name='legacy-stream', capabilities=[AbsurdDurability()])
    absurd.register_task(name='legacy-stream')(_noop)
    legacy_response: JsonValue = TypeAdapter(ModelResponse).dump_python(
        ModelResponse(parts=[TextPart(content='legacy')]), mode='json'
    )

    async with running_task_context(absurd, 'legacy-stream') as ctx:
        await ctx._persist_checkpoint('legacy-stream__model.request_stream', legacy_response)
        async with agent.run_stream('hello') as result:
            output = await result.get_output()

    assert output == 'legacy'
    assert calls['calls'] == 0


async def test_idless_function_toolset_is_checkpointed_and_replayed(absurd: AsyncAbsurd) -> None:
    calls = {'calls': 0}
    toolset = FunctionToolset[object]()

    @toolset.tool_plain
    def charge(amount: int) -> str:
        calls['calls'] += 1
        return f'charged {amount}'

    agent = Agent[object, str](
        _tool_model('charge', {'amount': 7}),
        name='idless-function',
        toolsets=[toolset],
        capabilities=[AbsurdDurability()],
    )
    absurd.register_task(name='idless-function')(_noop)

    async with running_task_context(absurd, 'idless-function', max_attempts=2) as ctx:
        first = await agent.run('charge it')
        task_id = ctx.task_id
        assert {
            'idless-function__model.request',
            'idless-function__function_toolset.call_tool:charge',
        } <= set(ctx._checkpoint_cache)
        assert ctx._checkpoint_cache['idless-function__function_toolset.call_tool:charge'] == 'charged 7'

    async with reenter_running_task(absurd, task_id):
        replay = await agent.run('charge it')

    assert first.output == replay.output == 'done'
    assert calls['calls'] == 1


async def test_parallel_ordered_events_replay_preserves_scheduling_order(absurd: AsyncAbsurd) -> None:
    toolset = FunctionToolset[object](id='parallel-tools')
    completed: list[str] = []

    @toolset.tool_plain
    async def record(marker: str, delay_ms: int) -> str:
        await asyncio.sleep(delay_ms / 1000)
        completed.append(marker)
        return marker

    model_calls = {'calls': 0}

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        model_calls['calls'] += 1
        if any(isinstance(part, ToolReturnPart) for message in messages for part in message.parts):
            return ModelResponse(parts=[TextPart(content='done')])
        return ModelResponse(
            parts=[
                # The first-scheduled call completes last; checkpoint slots must still follow
                # this model order rather than completion order.
                ToolCallPart(tool_name='record', args={'marker': 'first', 'delay_ms': 40}, tool_call_id='r1'),
                ToolCallPart(tool_name='record', args={'marker': 'second', 'delay_ms': 0}, tool_call_id='r2'),
            ]
        )

    agent = Agent(
        FunctionModel(model_fn, model_name='parallel-model'),
        name='parallel-order',
        toolsets=[toolset],
        capabilities=[AbsurdDurability(parallel_execution_mode='parallel_ordered_events')],
    )
    absurd.register_task(name='parallel-order')(_noop)
    step = 'parallel-order__function_toolset__parallel-tools.call_tool:record'

    async with running_task_context(absurd, 'parallel-order', max_attempts=2) as ctx:
        first = await agent.run('record both')
        task_id = ctx.task_id
        assert ctx._checkpoint_cache[step] == 'first'
        assert ctx._checkpoint_cache[f'{step}#2'] == 'second'
    assert completed == ['second', 'first']

    async with reenter_running_task(absurd, task_id):
        replay = await agent.run('record both')

    assert first.output == replay.output == 'done'
    assert model_calls['calls'] == 2
    assert completed == ['second', 'first']


async def test_explicit_function_toolset_and_model_resolver_replay(absurd: AsyncAbsurd) -> None:
    tool_calls = {'calls': 0}
    model_calls = {'primary': 0, 'cheap': 0}
    toolset = FunctionToolset[object](id='billing')

    @toolset.tool_plain
    def charge(amount: int) -> str:
        tool_calls['calls'] += 1
        return f'charged {amount}'

    def primary(
        messages: list[ModelMessage], info: AgentInfo
    ) -> ModelResponse:  # pragma: no cover - cheap model is selected
        model_calls['primary'] += 1
        return ModelResponse(parts=[TextPart(content='primary')])

    def cheap(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        model_calls['cheap'] += 1
        if not any(isinstance(part, ToolReturnPart) for message in messages for part in message.parts):
            return ModelResponse(parts=[ToolCallPart(tool_name='charge', args={'amount': 3})])
        return ModelResponse(parts=[TextPart(content='cheap')])

    agent = Agent[object, str](
        FunctionModel(primary, model_name='primary'),
        name='explicit-function',
        toolsets=[toolset],
        capabilities=[AbsurdDurability(models={'cheap': FunctionModel(cheap, model_name='cheap')})],
    )
    absurd.register_task(name='explicit-function')(_noop)

    async with running_task_context(absurd, 'explicit-function', max_attempts=2) as ctx:
        first = await agent.run('hello', model='cheap')
        task_id = ctx.task_id
        assert {
            'explicit-function__model.request.cheap',
            'explicit-function__function_toolset__billing.call_tool:charge',
        } <= set(ctx._checkpoint_cache)
        assert ctx._checkpoint_cache['explicit-function__function_toolset__billing.call_tool:charge'] == 'charged 3'

    async with reenter_running_task(absurd, task_id):
        replay = await agent.run('hello', model='cheap')

    assert first.output == replay.output == 'cheap'
    assert model_calls == {'primary': 0, 'cheap': 2}
    assert tool_calls['calls'] == 1


async def test_runtime_function_toolset_is_rejected(absurd: AsyncAbsurd) -> None:
    agent = Agent[object, str](_text_model(), name='runtime', capabilities=[AbsurdDurability()])
    absurd.register_task(name='runtime')(_noop)
    late = FunctionToolset[object](id='late')

    @late.tool_plain
    def echo(value: str) -> str:
        return value  # pragma: no cover - runtime toolset is rejected before invocation

    async with running_task_context(absurd, 'runtime'):
        with pytest.raises(UserError, match=r'cannot be passed to `run\(toolsets=\.\.\.\)` at runtime'):
            await agent.run('hello', toolsets=[late])


async def test_model_id_with_hash_is_rejected(absurd: AsyncAbsurd) -> None:
    agent = Agent(
        _text_model(),
        name='hash-model',
        capabilities=[AbsurdDurability(models={'cheap#2': _text_model()})],
    )
    absurd.register_task(name='hash-model')(_noop)

    async with running_task_context(absurd, 'hash-model'):
        with pytest.raises(UserError, match='Model ID.*contains `#`'):
            await agent.run('hello', model='cheap#2')


async def test_tool_name_with_hash_is_rejected(absurd: AsyncAbsurd) -> None:
    toolset = FunctionToolset[object]()

    @toolset.tool_plain(name='bad#tool')
    def bad_tool() -> str:
        return 'bad'  # pragma: no cover - unsafe tool name is rejected before invocation

    agent = Agent[object, str](
        _tool_model('bad#tool', {}),
        name='hash-tool',
        toolsets=[toolset],
        capabilities=[AbsurdDurability()],
    )
    absurd.register_task(name='hash-tool')(_noop)

    async with running_task_context(absurd, 'hash-tool'):
        with pytest.raises(UserError, match='tool name.*contains `#`'):
            await agent.run('hello')


async def test_same_task_namespace_cannot_overlap(absurd: AsyncAbsurd) -> None:
    started = anyio.Event()
    release = anyio.Event()

    async def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        started.set()
        await release.wait()
        return ModelResponse(parts=[TextPart(content='ok')])

    agent = Agent(FunctionModel(model, model_name='blocking'), name='overlap', capabilities=[AbsurdDurability()])
    absurd.register_task(name='overlap')(_noop)

    async with running_task_context(absurd, 'overlap'):
        with anyio.fail_after(5):
            async with anyio.create_task_group() as task_group:  # pragma: no branch
                task_group.start_soon(agent.run, 'first')
                await started.wait()
                with pytest.raises(UserError, match='Concurrent Absurd agent runs'):
                    await agent.run('second')
                release.set()


async def test_enqueue_is_rejected_inside_checkpointed_tool(absurd: AsyncAbsurd) -> None:
    toolset = FunctionToolset[object]()

    @toolset.tool
    def enqueue(ctx: RunContext[object]) -> str:
        ctx.enqueue('follow-up')
        return 'unreachable'  # pragma: no cover - enqueue raises before this return

    agent = Agent[object, str](
        _tool_model('enqueue', {}),
        name='enqueue-tool',
        toolsets=[toolset],
        capabilities=[AbsurdDurability()],
    )
    absurd.register_task(name='enqueue-tool')(_noop)

    async with running_task_context(absurd, 'enqueue-tool'):
        with pytest.raises(UserError, match=r'`ctx\.enqueue\(\)` is not supported inside a durable step'):
            await agent.run('hello')
