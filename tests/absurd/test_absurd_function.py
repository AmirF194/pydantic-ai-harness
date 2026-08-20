"""Tests for function-tool execution and result serialization in `AbsurdDurability`.

The cases map to `pydantic-ai-absurd/tests/test_function_toolset.py` and
`pydantic-ai-absurd/tests/test_durability.py`. Ordinary tool replay cases use the real
PostgreSQL-backed task context; fake contexts are reserved for deterministic validation and edge
cases.
"""

from __future__ import annotations

import pytest

pytest.importorskip('absurd_sdk')

from absurd_sdk import AsyncAbsurd, JsonValue
from pydantic_ai import Agent
from pydantic_ai.exceptions import ModelRetry, UserError
from pydantic_ai.messages import (
    RetryPromptPart,
    ToolReturnPart,
)
from pydantic_ai.tools import RunContext
from pydantic_ai.toolsets import ExternalToolset, FunctionToolset

from pydantic_ai_harness.absurd import AbsurdDurability

from ._helpers import _text_model, _tool_then_done_model
from ._postgres import reenter_running_task, running_task_context
from .conftest import FakeAsyncTaskContext, absurd_task_context

pytestmark = pytest.mark.anyio


class TestFunctionTool:
    async def test_replay_does_not_rerun_checkpointed_tool(self, absurd_client: AsyncAbsurd) -> None:
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

        async with running_task_context(absurd_client, 'pay', max_attempts=2) as ctx:
            first = await agent.run('charge it')
            task_id = ctx.task_id

        async with reenter_running_task(absurd_client, task_id):
            second = await agent.run('charge it')

        assert calls['n'] == 1
        assert first.output == second.output == 'done'

    async def test_wrapper_shaped_tool_result_round_trips(self) -> None:
        calls = {'n': 0}
        toolset = FunctionToolset(id='tools')
        wrapper_shaped_result: JsonValue = {'kind': 'tool_return', 'result': 'raw value'}

        @toolset.tool_plain
        def return_dict() -> dict[str, str]:
            calls['n'] += 1
            return {'kind': 'tool_return', 'result': 'raw value'}

        agent = Agent(
            _tool_then_done_model('return_dict', {}),
            name='wrapped',
            toolsets=[toolset],
            capabilities=[AbsurdDurability()],
        )

        ctx = FakeAsyncTaskContext()
        with absurd_task_context(ctx):
            first = await agent.run('go')
        replay = ctx.replay()
        with absurd_task_context(replay):
            second = await agent.run('go')

        assert calls['n'] == 1
        for result in (first, second):
            returned = [
                part.content
                for message in result.all_messages()
                for part in message.parts
                if isinstance(part, ToolReturnPart)
            ]
            assert returned == [wrapper_shaped_result]

    async def test_raw_checkpoint_from_the_reference_package_replays(self) -> None:
        # `pydantic-ai-absurd` stores a tool's return value directly rather than the wrapped form,
        # so a run started there and resumed here finds a raw checkpoint under the same step name.
        # The `kind` field is deliberately the same as the old wrapper's discriminator: the
        # capability must preserve this dictionary instead of unwrapping it.
        calls = {'n': 0}
        toolset = FunctionToolset(id='billing')

        @toolset.tool_plain
        def charge_card(amount: int) -> str:  # pragma: no cover - served from the seeded checkpoint
            calls['n'] += 1
            return f'charged {amount}'

        agent = Agent(
            _tool_then_done_model('charge_card', {'amount': 7}),
            name='pay',
            toolsets=[toolset],
            capabilities=[AbsurdDurability()],
        )

        step = 'pay__function_toolset__billing.call_tool:charge_card'
        legacy_result: JsonValue = {'kind': 'tool_return', 'result': 'charged 99'}
        ctx = FakeAsyncTaskContext(store={step: legacy_result})
        with absurd_task_context(ctx):
            result = await agent.run('charge it')

        assert result.output == 'done'
        assert calls['n'] == 0
        assert ctx.invoked == ['pay__model.request', 'pay__model.request#2']
        returned = [part.content for m in result.all_messages() for part in m.parts if isinstance(part, ToolReturnPart)]
        assert returned == [legacy_result]

    async def test_malformed_current_wrapper_is_not_passed_through_raw(self) -> None:
        # A malformed current envelope must still fail validation. Un-enveloped values, including
        # dictionaries carrying a legacy wrapper discriminator, are raw tool results.
        calls = {'n': 0}
        toolset = FunctionToolset(id='billing')

        @toolset.tool_plain
        def charge_card(amount: int) -> str:  # pragma: no cover - the model stops after the retry
            calls['n'] += 1
            return f'charged {amount}'

        agent = Agent(
            _tool_then_done_model('charge_card', {'amount': 7}),
            name='pay',
            toolsets=[toolset],
            capabilities=[AbsurdDurability()],
        )

        step = 'pay__function_toolset__billing.call_tool:charge_card'
        # A real wrapper kind, but missing the `result` the `tool_return` wrapper requires.
        ctx = FakeAsyncTaskContext(
            store={
                step: {
                    '__pydantic_ai_harness_call_tool_result__': {
                        'version': 1,
                        'result': {'kind': 'tool_return'},
                    }
                }
            }
        )
        with absurd_task_context(ctx):
            result = await agent.run('charge it')

        parts = [part for m in result.all_messages() for part in m.parts]
        assert not any(isinstance(part, ToolReturnPart) for part in parts)
        retries = [part for part in parts if isinstance(part, RetryPromptPart)]
        assert len(retries) == 1
        assert 'result' in repr(retries[0].content)

    @pytest.mark.parametrize(
        ('envelope', 'error'),
        [
            ('not an object', 'Malformed'),
            ({'version': True, 'result': {'kind': 'tool_return', 'result': 'x'}}, 'Unsupported'),
            ({'version': 1.0, 'result': {'kind': 'tool_return', 'result': 'x'}}, 'Unsupported'),
            ({'version': 2, 'result': {'kind': 'tool_return', 'result': 'x'}}, 'Unsupported'),
        ],
    )
    async def test_invalid_current_envelope_is_rejected(self, envelope: JsonValue, error: str) -> None:
        toolset = FunctionToolset(id='tools')

        @toolset.tool_plain
        def ping() -> str:  # pragma: no cover - served from the seeded checkpoint
            return 'pong'

        agent = Agent(
            _tool_then_done_model('ping', {}), name='bad', toolsets=[toolset], capabilities=[AbsurdDurability()]
        )
        step = 'bad__function_toolset__tools.call_tool:ping'
        ctx = FakeAsyncTaskContext(store={step: {'__pydantic_ai_harness_call_tool_result__': envelope}})

        with absurd_task_context(ctx):
            with pytest.raises(UserError, match=error):
                await agent.run('go')

    async def test_tool_name_with_hash_is_rejected(self) -> None:
        toolset = FunctionToolset(id='tools')

        @toolset.tool_plain(name='ping#2')
        def ping() -> str:  # pragma: no cover - rejected before the step runs
            return 'pong'

        agent = Agent(
            _tool_then_done_model('ping#2', {}), name='h', toolsets=[toolset], capabilities=[AbsurdDurability()]
        )

        with absurd_task_context(FakeAsyncTaskContext()):
            with pytest.raises(UserError, match='Tool name .* contains'):
                await agent.run('ping it')

    @pytest.mark.parametrize(('operation', 'error_name'), [('enqueue', r'ctx\.enqueue\(\)'), ('cancel', 'cancel')])
    async def test_checkpointed_tool_rejects_run_context_mutation(self, operation: str, error_name: str) -> None:
        toolset = FunctionToolset(id='mutating')

        @toolset.tool
        def mutate_run(ctx: RunContext[object]) -> None:
            # Both branches raise inside the step, so the tool never returns.
            if operation == 'enqueue':
                ctx.enqueue('security instruction')
            else:
                ctx.cancel()

        agent = Agent(
            _tool_then_done_model('mutate_run', {}),
            name='guarded',
            toolsets=[toolset],
            capabilities=[AbsurdDurability()],
        )

        with absurd_task_context(FakeAsyncTaskContext()):
            with pytest.raises(UserError, match=rf'`{error_name}` is not supported inside a durable step'):
                await agent.run('mutate the run')


class TestModelRetry:
    async def test_model_retry_crosses_checkpoint_and_replays(self, absurd_client: AsyncAbsurd) -> None:
        calls = {'n': 0}
        toolset = FunctionToolset(id='tools')

        @toolset.tool_plain
        def flaky() -> str:
            calls['n'] += 1
            raise ModelRetry('nope, try again')

        agent = Agent(
            _tool_then_done_model('flaky', {}), name='retry', toolsets=[toolset], capabilities=[AbsurdDurability()]
        )

        async with running_task_context(absurd_client, 'retry', max_attempts=2) as ctx:
            first = await agent.run('go')
            task_id = ctx.task_id
        assert first.output == 'done'
        assert calls['n'] == 1

        async with reenter_running_task(absurd_client, task_id):
            second = await agent.run('go')

        # On replay the stored `ModelRetry` is re-raised without re-running the tool.
        assert calls['n'] == 1
        assert second.output == 'done'


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


class TestToolStepConfigRejected:
    async def test_non_empty_dict_config_raises(self) -> None:
        toolset = FunctionToolset(id='tools')

        @toolset.tool_plain(metadata={'absurd': {'retries': 3}})
        def ping() -> str:  # pragma: no cover - rejected before it can run
            return 'pong'

        agent = Agent(
            _tool_then_done_model('ping', {}), name='cfg', toolsets=[toolset], capabilities=[AbsurdDurability()]
        )

        ctx = FakeAsyncTaskContext()
        with absurd_task_context(ctx):
            with pytest.raises(UserError, match='take no per-tool options'):
                await agent.run('ping it')

    async def test_empty_dict_config_is_allowed(self) -> None:
        calls = {'n': 0}
        toolset = FunctionToolset(id='tools')

        @toolset.tool_plain(metadata={'absurd': {}})
        def ping() -> str:
            calls['n'] += 1
            return 'pong'

        agent = Agent(
            _tool_then_done_model('ping', {}), name='cfg', toolsets=[toolset], capabilities=[AbsurdDurability()]
        )

        ctx = FakeAsyncTaskContext()
        with absurd_task_context(ctx):
            result = await agent.run('ping it')

        assert result.output == 'done'
        assert calls['n'] == 1
        assert 'cfg__function_toolset__tools.call_tool:ping' in ctx.stored


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
