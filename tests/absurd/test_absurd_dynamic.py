"""Durable-wrapping tests for a construction-time `DynamicToolset` under `AbsurdDurability`.

A `DynamicToolset` resolves its inner toolset lazily from a user factory. The capability moves that
resolution and the inner tool calls into Absurd steps, so on replay the factory and the tool are not
re-invoked. Behavior is driven through `Agent(..., capabilities=[...])` inside a `FakeAsyncTaskContext`.
"""

from __future__ import annotations

import pytest

pytest.importorskip('absurd_sdk')

from pydantic_ai import Agent
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.tools import RunContext
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai.toolsets._dynamic import DynamicToolset  # pyright: ignore[reportPrivateImportUsage]

from pydantic_ai_harness.absurd import AbsurdDurability

from .conftest import FakeAsyncTaskContext, absurd_task_context

pytestmark = pytest.mark.anyio


def _greet_then_done_model() -> FunctionModel:
    def fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        answered = any(isinstance(p, ToolReturnPart) for m in messages for p in m.parts)
        if answered:
            return ModelResponse(parts=[TextPart(content='done')])
        return ModelResponse(parts=[ToolCallPart(tool_name='greet', args={'name': 'ada'})])

    return FunctionModel(fn, model_name='fn')


def _dynamic_toolset(
    factory_calls: dict[str, int], tool_calls: dict[str, int], *, instructions: str | None = None
) -> DynamicToolset[object]:
    def build(ctx: RunContext[object]) -> FunctionToolset[object]:
        factory_calls['n'] += 1
        inner: FunctionToolset[object] = FunctionToolset(id='inner', instructions=instructions)

        @inner.tool_plain
        def greet(name: str) -> str:
            tool_calls['n'] += 1
            return f'hi {name}'

        return inner

    return DynamicToolset(build, id='dyn')


class TestDynamicToolsetCheckpointing:
    async def test_resolution_and_tool_call_checkpointed(self) -> None:
        factory_calls = {'n': 0}
        tool_calls = {'n': 0}
        agent = Agent(
            _greet_then_done_model(),
            name='d',
            toolsets=[_dynamic_toolset(factory_calls, tool_calls)],
            capabilities=[AbsurdDurability()],
        )

        ctx = FakeAsyncTaskContext()
        with absurd_task_context(ctx):
            result = await agent.run('greet ada')

        assert result.output == 'done'
        assert tool_calls['n'] == 1
        assert 'd__dynamic_toolset__dyn.get_tools' in ctx.stored
        assert 'd__dynamic_toolset__dyn.call_tool:greet' in ctx.stored

        first_factory_calls = factory_calls['n']

        replay = ctx.replay()
        with absurd_task_context(replay):
            second = await agent.run('greet ada')

        assert second.output == 'done'
        # On replay every dynamic step is served from its checkpoint: neither the factory nor the
        # inner tool runs again.
        assert factory_calls['n'] == first_factory_calls
        assert tool_calls['n'] == 1
        assert replay.invoked == []

    async def test_dynamic_toolset_instructions_checkpointed(self) -> None:
        factory_calls = {'n': 0}
        tool_calls = {'n': 0}
        seen_instructions: list[str | None] = []

        def fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            seen_instructions.append(info.instructions)
            return ModelResponse(parts=[TextPart(content='done')])

        agent = Agent(
            FunctionModel(fn, model_name='fn'),
            name='d',
            toolsets=[_dynamic_toolset(factory_calls, tool_calls, instructions='Be terse.')],
            capabilities=[AbsurdDurability()],
        )

        ctx = FakeAsyncTaskContext()
        with absurd_task_context(ctx):
            result = await agent.run('hi')

        assert result.output == 'done'
        # The resolved toolset's instructions are captured in the `get_tools` checkpoint and reach
        # the model.
        payload = ctx.stored['d__dynamic_toolset__dyn.get_tools']
        assert isinstance(payload, dict)
        assert 'Be terse.' in repr(payload['instructions'])
        assert any(instr is not None and 'Be terse.' in instr for instr in seen_instructions)

    async def test_transparent_outside_task(self) -> None:
        factory_calls = {'n': 0}
        tool_calls = {'n': 0}
        agent = Agent(
            _greet_then_done_model(),
            name='d',
            toolsets=[_dynamic_toolset(factory_calls, tool_calls)],
            capabilities=[AbsurdDurability()],
        )

        result = await agent.run('greet ada')

        assert result.output == 'done'
        assert tool_calls['n'] == 1


class TestDynamicToolsetErrors:
    async def test_idless_dynamic_toolset_raises(self) -> None:
        def build(ctx: RunContext[object]) -> FunctionToolset[object]:  # pragma: no cover - never resolved
            return FunctionToolset(id='inner')

        with pytest.raises(UserError, match='need to have a unique `id`'):
            Agent(
                _greet_then_done_model(),
                name='d',
                toolsets=[DynamicToolset(build)],
                capabilities=[AbsurdDurability()],
            )

    async def test_runtime_dynamic_toolset_rejected_inside_task(self) -> None:
        factory_calls = {'n': 0}
        tool_calls = {'n': 0}
        agent = Agent(_greet_then_done_model(), name='d', capabilities=[AbsurdDurability()])

        ctx = FakeAsyncTaskContext()
        with absurd_task_context(ctx):
            with pytest.raises(UserError, match=r'cannot be passed to `run\(toolsets=...\)` at runtime'):
                await agent.run('hi', toolsets=[_dynamic_toolset(factory_calls, tool_calls)])
