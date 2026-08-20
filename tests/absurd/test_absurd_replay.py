"""Tests for replay, retry, and parallel execution in `AbsurdDurability`.

The cases map to `pydantic-ai-absurd/tests/test_replay.py` and
`pydantic-ai-absurd/tests/test_durability.py`. They use `FakeAsyncTaskContext` for deterministic
unit coverage; real PostgreSQL replay coverage lives in `test_absurd_postgres.py`.
"""

from __future__ import annotations

import asyncio
from contextlib import AbstractContextManager

import pytest

pytest.importorskip('absurd_sdk')

from pydantic_ai import Agent
from pydantic_ai.agent import ParallelExecutionMode
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.toolsets import FunctionToolset

from pydantic_ai_harness.absurd import AbsurdDurability

from ._helpers import _text_model
from .conftest import FakeAsyncTaskContext, absurd_task_context

pytestmark = pytest.mark.anyio


class TestCrashMidRunRetry:
    async def test_model_step_served_from_checkpoint_while_failed_tool_reruns(self) -> None:
        # The core value prop: the model step completes and is checkpointed, then a tool raises a
        # real (non-`ModelRetry`) error that fails the task. On retry, Absurd replays: the model
        # step is served from its checkpoint (model not called again) while the tool re-runs.
        model_calls = {'n': 0}
        tool_attempts = {'n': 0}
        toolset = FunctionToolset(id='tools')

        @toolset.tool_plain
        def flaky() -> str:
            tool_attempts['n'] += 1
            if tool_attempts['n'] == 1:
                raise RuntimeError('worker died mid-tool')
            return 'recovered'

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            model_calls['n'] += 1
            answered = any(isinstance(p, ToolReturnPart) for m in messages for p in m.parts)
            if answered:
                return ModelResponse(parts=[TextPart(content='done')])
            return ModelResponse(parts=[ToolCallPart(tool_name='flaky', args={})])

        agent = Agent(FunctionModel(model_fn), name='crash', toolsets=[toolset], capabilities=[AbsurdDurability()])

        model_step = 'crash__model.request'
        tool_step = 'crash__function_toolset__tools.call_tool:flaky'

        ctx = FakeAsyncTaskContext()
        with absurd_task_context(ctx):
            with pytest.raises(RuntimeError, match='worker died mid-tool'):
                await agent.run('go')

        # The model step checkpointed before the tool ran; the failed tool step did not.
        assert model_step in ctx.stored
        assert tool_step not in ctx.stored
        assert model_calls['n'] == 1
        assert tool_attempts['n'] == 1

        replay = ctx.replay()
        with absurd_task_context(replay):
            result = await agent.run('go')

        assert result.output == 'done'
        # The first model request was served from its checkpoint (not re-invoked); the tool re-ran
        # and the second model turn is a fresh step.
        assert model_step not in replay.invoked
        assert tool_step in replay.invoked
        assert f'{model_step}#2' in replay.invoked
        assert model_calls['n'] == 2
        assert tool_attempts['n'] == 2


class TestParallelExecutionMode:
    async def test_parallel_mode_is_rejected(self) -> None:
        # `'parallel'` is outside the declared type, but an untyped caller can still pass it.
        with pytest.raises(UserError, match=r"'parallel' is not a supported .*Use 'sequential'"):
            AbsurdDurability(parallel_execution_mode='parallel')  # pyright: ignore[reportArgumentType]

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

    async def test_parallel_execution_mode_untouched_outside_task(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Outside a task the capability is transparent, so it must not override the agent's
        # configured tool-call execution mode.
        agent = Agent(
            _text_model(), name='a', capabilities=[AbsurdDurability(parallel_execution_mode='parallel_ordered_events')]
        )
        recorded: list[ParallelExecutionMode] = []
        real = agent.parallel_tool_call_execution_mode

        def spy(mode: ParallelExecutionMode = 'parallel') -> AbstractContextManager[None]:
            recorded.append(mode)
            return real(mode)

        monkeypatch.setattr(agent, 'parallel_tool_call_execution_mode', spy)

        # Inside a task the override applies; outside one it must not touch the configured mode.
        ctx = FakeAsyncTaskContext()
        with absurd_task_context(ctx):
            await agent.run('hi')
        assert recorded == ['parallel_ordered_events']

        recorded.clear()
        await agent.run('hi')
        assert recorded == []


class TestRepeatedStepNames:
    async def test_two_runs_in_one_task_disambiguate_by_encounter_order(self) -> None:
        # A single task handler that runs the agent twice: the second run's model step reuses the
        # same step name, so Absurd's encounter-order counter records it under a `#2` suffix.
        counter = {'calls': 0}
        agent = Agent(_text_model(counter), name='a', capabilities=[AbsurdDurability()])

        ctx = FakeAsyncTaskContext()
        with absurd_task_context(ctx):
            first = await agent.run('hi')
            second = await agent.run('hi again')

        assert first.output == second.output == 'ok'
        assert 'a__model.request' in ctx.stored
        assert 'a__model.request#2' in ctx.stored

        replay = ctx.replay()
        with absurd_task_context(replay):
            await agent.run('hi')
            await agent.run('hi again')

        assert counter['calls'] == 2
        assert replay.invoked == []

    async def test_same_tool_called_twice_in_one_response(self) -> None:
        calls: list[int] = []
        toolset = FunctionToolset(id='tools')

        @toolset.tool_plain
        def charge(amount: int) -> str:
            calls.append(amount)
            return f'charged {amount}'

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            answered = any(isinstance(p, ToolReturnPart) for m in messages for p in m.parts)
            if answered:
                return ModelResponse(parts=[TextPart(content='done')])
            return ModelResponse(
                parts=[
                    ToolCallPart(tool_name='charge', args={'amount': 1}, tool_call_id='c1'),
                    ToolCallPart(tool_name='charge', args={'amount': 2}, tool_call_id='c2'),
                ]
            )

        agent = Agent(FunctionModel(model_fn), name='pay', toolsets=[toolset], capabilities=[AbsurdDurability()])

        ctx = FakeAsyncTaskContext()
        with absurd_task_context(ctx):
            result = await agent.run('charge both')

        assert result.output == 'done'
        assert calls == [1, 2]
        step = 'pay__function_toolset__tools.call_tool:charge'
        assert step in ctx.stored
        assert f'{step}#2' in ctx.stored


class TestParallelOrderedEventsDeterminism:
    async def test_name_assignment_follows_scheduling_order_not_completion(self) -> None:
        # The adversarial case behind excluding `'parallel'` but keeping `'parallel_ordered_events'`:
        # two concurrent calls of the SAME tool, where the first-scheduled call completes LAST.
        # Absurd assigns the `#1`/`#2` checkpoint slot at `ctx.step(...)` entry, which happens before
        # the tool body runs, so assignment follows call-scheduling order (the model's tool-call
        # order), not completion order. If it followed completion order the slots -- and the cached
        # results served on replay -- would swap.
        toolset = FunctionToolset(id='tools')

        @toolset.tool_plain
        async def record(marker: str, delay_ms: int) -> str:
            await asyncio.sleep(delay_ms / 1000)
            return marker

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            answered = any(isinstance(p, ToolReturnPart) for m in messages for p in m.parts)
            if answered:
                return ModelResponse(parts=[TextPart(content='done')])
            return ModelResponse(
                parts=[
                    # First-scheduled call sleeps longest, so it completes last.
                    ToolCallPart(tool_name='record', args={'marker': 'first', 'delay_ms': 40}, tool_call_id='r1'),
                    ToolCallPart(tool_name='record', args={'marker': 'second', 'delay_ms': 0}, tool_call_id='r2'),
                ]
            )

        agent = Agent(
            FunctionModel(model_fn),
            name='par',
            toolsets=[toolset],
            capabilities=[AbsurdDurability(parallel_execution_mode='parallel_ordered_events')],
        )

        step = 'par__function_toolset__tools.call_tool:record'

        ctx = FakeAsyncTaskContext()
        with absurd_task_context(ctx):
            first = await agent.run('go')

        assert first.output == 'done'
        # `#1` is the first-scheduled call ('first'), even though it completed last.
        assert ctx.stored[step] == {
            '__pydantic_ai_harness_call_tool_result__': {
                'version': 1,
                'result': {'result': 'first', 'kind': 'tool_return'},
            }
        }
        assert ctx.stored[f'{step}#2'] == {
            '__pydantic_ai_harness_call_tool_result__': {
                'version': 1,
                'result': {'result': 'second', 'kind': 'tool_return'},
            }
        }

        # On replay the slots must map the same way: results are served from the checkpoint, not
        # swapped, regardless of which call finishes first.
        replay = ctx.replay()
        with absurd_task_context(replay):
            second = await agent.run('go')

        assert second.output == 'done'
        assert replay.stored[step] == {
            '__pydantic_ai_harness_call_tool_result__': {
                'version': 1,
                'result': {'result': 'first', 'kind': 'tool_return'},
            }
        }
        assert replay.stored[f'{step}#2'] == {
            '__pydantic_ai_harness_call_tool_result__': {
                'version': 1,
                'result': {'result': 'second', 'kind': 'tool_return'},
            }
        }
        assert replay.invoked == []
