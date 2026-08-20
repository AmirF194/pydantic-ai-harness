"""Durable-wrapping tests for a construction-time `DynamicToolset` under `AbsurdDurability`.

A `DynamicToolset` resolves its inner toolset lazily from a user factory. The capability moves that
resolution and the inner tool calls into Absurd steps, so on replay the factory and the tool are not
re-invoked. Ordinary checkpoint tests use a real PostgreSQL-backed Absurd task context; the fake
context remains for checkpoint format and error-path tests below.
"""

from __future__ import annotations

import pytest

pytest.importorskip('absurd_sdk')

from absurd_sdk import AsyncAbsurd
from pydantic_ai import Agent
from pydantic_ai.exceptions import ModelRetry, UserError
from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
    RetryPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.tools import RunContext
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai.toolsets._dynamic import DynamicToolset  # pyright: ignore[reportPrivateImportUsage]

from pydantic_ai_harness.absurd import AbsurdDurability

from ._postgres import reenter_running_task, running_task_context
from .conftest import (
    FakeAsyncTaskContext,
    absurd_task_context,
)

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
    async def test_resolution_and_tool_call_checkpointed(self, absurd_client: AsyncAbsurd) -> None:
        factory_calls = {'n': 0}
        tool_calls = {'n': 0}
        agent = Agent(
            _greet_then_done_model(),
            name='d',
            toolsets=[_dynamic_toolset(factory_calls, tool_calls)],
            capabilities=[AbsurdDurability()],
        )

        async with running_task_context(absurd_client, 'dynamic-resolution', max_attempts=2) as ctx:
            result = await agent.run('greet ada')

        assert result.output == 'done'
        assert tool_calls['n'] == 1
        task_id = ctx.task_id
        first_factory_calls = factory_calls['n']

        async with reenter_running_task(absurd_client, task_id):
            second = await agent.run('greet ada')

        assert second.output == 'done'
        # On replay every dynamic step is served from its checkpoint: neither the factory nor the
        # inner tool runs again.
        assert factory_calls['n'] == first_factory_calls
        assert tool_calls['n'] == 1

    async def test_decorated_toolset_added_after_binding_is_checkpointed(self, absurd_client: AsyncAbsurd) -> None:
        factory_calls = {'n': 0}
        tool_calls = {'n': 0}
        agent = Agent(
            _greet_then_done_model(),
            name='d',
            capabilities=[AbsurdDurability()],
        )

        @agent.toolset(id='decorated')
        def build(ctx: RunContext[object]) -> FunctionToolset[object]:
            factory_calls['n'] += 1
            inner = FunctionToolset[object](id='inner')

            @inner.tool_plain
            def greet(name: str) -> str:
                tool_calls['n'] += 1
                return f'hi {name}'

            return inner

        async with running_task_context(absurd_client, 'decorated-dynamic', max_attempts=2) as ctx:
            result = await agent.run('greet ada')

        assert result.output == 'done'
        assert tool_calls['n'] == 1
        first_factory_calls = factory_calls['n']
        task_id = ctx.task_id

        async with reenter_running_task(absurd_client, task_id):
            replay = await agent.run('greet ada')

        assert replay.output == 'done'
        assert factory_calls['n'] == first_factory_calls
        assert tool_calls['n'] == 1

    async def test_dynamic_toolset_instructions_checkpointed(self, absurd_client: AsyncAbsurd) -> None:
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

        async with running_task_context(absurd_client, 'dynamic-instructions', max_attempts=2) as ctx:
            result = await agent.run('hi')

        assert result.output == 'done'
        # The resolved toolset's instructions are captured in the `get_tools` checkpoint and reach
        # the model.
        task_id = ctx.task_id
        first_factory_calls = factory_calls['n']
        first_instruction_count = len(seen_instructions)
        assert any(instr is not None and 'Be terse.' in instr for instr in seen_instructions)

        async with reenter_running_task(absurd_client, task_id):
            replay = await agent.run('hi')

        assert replay.output == 'done'
        assert factory_calls['n'] == first_factory_calls
        assert len(seen_instructions) == first_instruction_count

    async def test_raw_checkpoint_from_the_reference_package_replays(self) -> None:
        # `pydantic-ai-absurd` stores a tool's return value directly rather than the wrapped form,
        # so a run started there and resumed here finds a raw checkpoint under the same step name.
        factory_calls = {'n': 0}
        tool_calls = {'n': 0}
        agent = Agent(
            _greet_then_done_model(),
            name='d',
            toolsets=[_dynamic_toolset(factory_calls, tool_calls)],
            capabilities=[AbsurdDurability()],
        )

        step = 'd__dynamic_toolset__dyn.call_tool:greet'
        ctx = FakeAsyncTaskContext(store={step: 'hi from the other package'})
        with absurd_task_context(ctx):
            result = await agent.run('greet ada')

        assert result.output == 'done'
        assert tool_calls['n'] == 0
        returned = [part.content for m in result.all_messages() for part in m.parts if isinstance(part, ToolReturnPart)]
        assert returned == ['hi from the other package']

    async def test_transparent_outside_task(self) -> None:
        factory_calls = {'n': 0}
        tool_calls = {'n': 0}
        seen_instructions: list[str | None] = []

        def fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            seen_instructions.append(info.instructions)
            if any(isinstance(p, ToolReturnPart) for m in messages for p in m.parts):
                return ModelResponse(parts=[TextPart(content='done')])
            return ModelResponse(parts=[ToolCallPart(tool_name='greet', args={'name': 'ada'})])

        agent = Agent(
            FunctionModel(fn, model_name='fn'),
            name='d',
            toolsets=[_dynamic_toolset(factory_calls, tool_calls, instructions='Be terse.')],
            capabilities=[AbsurdDurability()],
        )

        result = await agent.run('greet ada')

        assert result.output == 'done'
        assert tool_calls['n'] == 1
        # Off the durable path the resolved toolset's instructions still reach the model.
        assert any(instr is not None and 'Be terse.' in instr for instr in seen_instructions)

        # Nothing is cached across runs: a second run resolves the factory again.
        first_factory_calls = factory_calls['n']
        second = await agent.run('greet ada')

        assert second.output == 'done'
        assert factory_calls['n'] > first_factory_calls
        assert tool_calls['n'] == 2


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

    async def test_tool_name_with_hash_is_rejected(self) -> None:
        def build(ctx: RunContext[object]) -> FunctionToolset[object]:
            inner: FunctionToolset[object] = FunctionToolset(id='inner')

            @inner.tool_plain(name='greet#2')
            def greet() -> str:  # pragma: no cover - rejected before the step runs
                return 'hi'

            return inner

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return ModelResponse(parts=[ToolCallPart(tool_name='greet#2', args={})])

        agent = Agent(
            FunctionModel(model_fn, model_name='fn'),
            name='d',
            toolsets=[DynamicToolset(build, id='dyn')],
            capabilities=[AbsurdDurability()],
        )

        ctx = FakeAsyncTaskContext()
        with absurd_task_context(ctx):
            with pytest.raises(UserError, match='Tool name .* contains'):
                await agent.run('greet')

    async def test_runtime_dynamic_toolset_rejected_inside_task(self) -> None:
        factory_calls = {'n': 0}
        tool_calls = {'n': 0}
        agent = Agent(_greet_then_done_model(), name='d', capabilities=[AbsurdDurability()])

        ctx = FakeAsyncTaskContext()
        with absurd_task_context(ctx):
            with pytest.raises(UserError, match=r'cannot be passed to `run\(toolsets=...\)` at runtime'):
                await agent.run('hi', toolsets=[_dynamic_toolset(factory_calls, tool_calls)])


def _typed_tool_dynamic(tool_calls: list[int], *, metadata: object = None) -> DynamicToolset[object]:
    def build(ctx: RunContext[object]) -> FunctionToolset[object]:
        inner: FunctionToolset[object] = FunctionToolset(id='inner')

        @inner.tool_plain(metadata={'absurd': metadata} if metadata is not None else None)
        def double(x: int) -> str:
            tool_calls.append(x)
            return f'got {x}'

        return inner

    return DynamicToolset(build, id='dyn')


class TestDynamicToolsetArgValidation:
    async def test_invalid_args_become_a_retry_prompt_not_a_task_failure(self) -> None:
        # The restored outer tool carries a pass-through validator, so args must be re-validated
        # inside the step with the re-resolved inner tool's real validator. Invalid args must surface
        # as a `ValidationError` that `ToolManager` turns into a retry prompt, not a raw `TypeError`
        # that fails the whole task.
        tool_calls: list[int] = []
        retries = {'n': 0}

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            if any(isinstance(p, ToolReturnPart) for m in messages for p in m.parts):
                return ModelResponse(parts=[TextPart(content='done')])
            if any(isinstance(p, RetryPromptPart) for m in messages for p in m.parts):
                retries['n'] += 1
                return ModelResponse(parts=[ToolCallPart(tool_name='double', args={'x': 5})])
            return ModelResponse(parts=[ToolCallPart(tool_name='double', args={'x': 'not-a-number'})])

        agent = Agent(
            FunctionModel(model_fn, model_name='fn'),
            name='d',
            toolsets=[_typed_tool_dynamic(tool_calls)],
            capabilities=[AbsurdDurability()],
        )

        ctx = FakeAsyncTaskContext()
        with absurd_task_context(ctx):
            result = await agent.run('double it')

        assert result.output == 'done'
        assert retries['n'] == 1
        # The function only ran with the coerced valid args, never with the invalid ones.
        assert tool_calls == [5]

    async def test_typed_args_are_coerced(self) -> None:
        tool_calls: list[int] = []

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            if any(isinstance(p, ToolReturnPart) for m in messages for p in m.parts):
                return ModelResponse(parts=[TextPart(content='done')])
            # A string the int validator coerces, as the non-durable path would.
            return ModelResponse(parts=[ToolCallPart(tool_name='double', args={'x': '7'})])

        agent = Agent(
            FunctionModel(model_fn, model_name='fn'),
            name='d',
            toolsets=[_typed_tool_dynamic(tool_calls)],
            capabilities=[AbsurdDurability()],
        )

        ctx = FakeAsyncTaskContext()
        with absurd_task_context(ctx):
            result = await agent.run('double it')

        assert result.output == 'done'
        assert tool_calls == [7]

    async def test_model_retry_from_dynamic_tool_round_trips(self, absurd_client: AsyncAbsurd) -> None:
        attempts = {'n': 0}

        def build(ctx: RunContext[object]) -> FunctionToolset[object]:
            inner: FunctionToolset[object] = FunctionToolset(id='inner')

            @inner.tool_plain
            def greet(name: str) -> str:
                attempts['n'] += 1
                raise ModelRetry('nope')

            return inner

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            answered = any(isinstance(p, (ToolReturnPart, RetryPromptPart)) for m in messages for p in m.parts)
            if answered:
                return ModelResponse(parts=[TextPart(content='done')])
            return ModelResponse(parts=[ToolCallPart(tool_name='greet', args={'name': 'ada'})])

        agent = Agent(
            FunctionModel(model_fn, model_name='fn'),
            name='d',
            toolsets=[DynamicToolset(build, id='dyn')],
            capabilities=[AbsurdDurability()],
        )

        async with running_task_context(absurd_client, 'dynamic-retry', max_attempts=2) as ctx:
            first = await agent.run('greet')
        assert first.output == 'done'
        assert attempts['n'] == 1

        async with reenter_running_task(absurd_client, ctx.task_id):
            second = await agent.run('greet')

        # The stored `ModelRetry` is re-raised without re-running the tool.
        assert second.output == 'done'
        assert attempts['n'] == 1


class TestDynamicToolsetPerToolConfig:
    async def test_metadata_false_runs_inline_uncheckpointed(self) -> None:
        tool_calls: list[int] = []
        agent = Agent(
            FunctionModel(
                lambda messages, info: (
                    ModelResponse(parts=[TextPart(content='done')])
                    if any(isinstance(p, ToolReturnPart) for m in messages for p in m.parts)
                    else ModelResponse(parts=[ToolCallPart(tool_name='double', args={'x': 3})])
                ),
                model_name='fn',
            ),
            name='d',
            toolsets=[_typed_tool_dynamic(tool_calls, metadata=False)],
            capabilities=[AbsurdDurability()],
        )

        ctx = FakeAsyncTaskContext()
        with absurd_task_context(ctx):
            result = await agent.run('double it')

        assert result.output == 'done'
        assert tool_calls == [3]
        # Listing is still checkpointed; the opted-out call is not.
        assert 'd__dynamic_toolset__dyn.get_tools' in ctx.stored
        assert not any('call_tool:double' in name for name in ctx.stored)

    async def test_populated_dict_config_raises(self) -> None:
        tool_calls: list[int] = []
        agent = Agent(
            FunctionModel(
                lambda messages, info: ModelResponse(parts=[ToolCallPart(tool_name='double', args={'x': 3})]),
                model_name='fn',
            ),
            name='d',
            toolsets=[_typed_tool_dynamic(tool_calls, metadata={'retries': 3})],
            capabilities=[AbsurdDurability()],
        )

        ctx = FakeAsyncTaskContext()
        with absurd_task_context(ctx):
            with pytest.raises(UserError, match='take no per-tool options'):
                await agent.run('double it')


class TestDynamicCheckpointFormat:
    async def test_get_tools_payload_matches_the_pinned_shape(self) -> None:
        # A core dataclass owns this payload, so pin its serialized shape here: a change to it is a
        # change to the persistence format, which orphans the checkpoints of in-flight tasks.
        factory_calls = {'n': 0}
        tool_calls = {'n': 0}
        agent = Agent(
            FunctionModel(
                lambda messages, info: ModelResponse(parts=[TextPart(content='done')]),
                model_name='fn',
            ),
            name='d',
            toolsets=[_dynamic_toolset(factory_calls, tool_calls, instructions='Be terse.')],
            capabilities=[AbsurdDurability()],
        )

        ctx = FakeAsyncTaskContext()
        with absurd_task_context(ctx):
            result = await agent.run('hi')

        assert result.output == 'done'
        assert ctx.stored['d__dynamic_toolset__dyn.get_tools'] == {
            'tools': {
                'greet': {
                    'tool_def': {
                        'name': 'greet',
                        'description': None,
                        'parameters_json_schema': {
                            'additionalProperties': False,
                            'properties': {'name': {'type': 'string'}},
                            'required': ['name'],
                            'type': 'object',
                        },
                        'return_schema': {'type': 'string'},
                        'include_return_schema': None,
                        'outer_typed_dict_key': None,
                        'strict': None,
                        'sequential': False,
                        'timeout': None,
                        'kind': 'function',
                        'tool_kind': None,
                        'defer_loading': False,
                        'metadata': None,
                        'toolset_id': None,
                        'capability_id': None,
                        'unless_native': None,
                        'with_native': None,
                    },
                    'max_retries': 1,
                }
            },
            'instructions': [{'content': 'Be terse.', 'dynamic': False, 'part_kind': 'instruction'}],
        }
