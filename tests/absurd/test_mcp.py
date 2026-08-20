from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager

import pytest
from absurd_sdk import AsyncAbsurd, AsyncTaskContext, JsonValue
from fastmcp import FastMCP
from pydantic_ai import Agent, ModelMessage, ModelResponse
from pydantic_ai.mcp import MCPToolset
from pydantic_ai.messages import TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import RunContext
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai.usage import RunUsage

from pydantic_ai_harness.absurd import AbsurdDurability
from pydantic_ai_harness.absurd._function_toolset import AbsurdFunctionToolset
from pydantic_ai_harness.absurd._mcp import AbsurdMCPToolset

from .conftest import reenter_running_task, running_task_context

pytestmark = pytest.mark.anyio


def _mcp_model() -> FunctionModel:
    def fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if any(isinstance(part, ToolReturnPart) for message in messages for part in message.parts):
            return ModelResponse(parts=[TextPart(content='done')])
        return ModelResponse(parts=[ToolCallPart(tool_name='add', args={'left': 2, 'right': 3})])

    return FunctionModel(fn, model_name='mcp-model')


async def _noop(params: JsonValue, ctx: AsyncTaskContext) -> JsonValue:
    return None


def _server(calls: dict[str, int], *, instructions: str | None = None) -> FastMCP:
    server = FastMCP('calculator', instructions=instructions)

    @server.tool
    def add(left: int, right: int) -> int:
        calls['calls'] += 1
        return left + right

    return server


def _run_context() -> RunContext[object]:
    return RunContext[object](deps=None, model=TestModel(), usage=RunUsage(), prompt='x', messages=[])


@contextmanager
def _pass_through_scope(ctx: RunContext[object]) -> Generator[RunContext[object], None, None]:
    yield ctx


async def test_wrappers_preserve_identity_and_do_not_manage_lifecycle() -> None:
    function_inner = FunctionToolset[object](id='function')
    function_wrapper = AbsurdFunctionToolset(
        function_inner, step_name_prefix='wrapper', durable_run_context_scope=_pass_through_scope
    )
    mcp_inner = MCPToolset(_server({'calls': 0}), id='mcp')
    mcp_wrapper = AbsurdMCPToolset(mcp_inner, step_name_prefix='wrapper', durable_run_context_scope=_pass_through_scope)

    assert function_wrapper.id == function_inner.id
    assert mcp_wrapper.id == mcp_inner.id
    async with function_wrapper as entered:
        assert entered is function_wrapper
    async with mcp_wrapper as entered:
        assert entered is mcp_wrapper
    assert function_wrapper.visit_and_replace(lambda value: value) is function_wrapper
    assert mcp_wrapper.visit_and_replace(lambda value: value) is mcp_wrapper


async def test_idless_in_process_mcp_toolset_replays(absurd: AsyncAbsurd) -> None:
    calls = {'calls': 0}
    toolset = MCPToolset(_server(calls)).filtered(lambda ctx, tool: tool.name == 'add')
    agent = Agent[object, str](
        _mcp_model(),
        name='idless-mcp',
        toolsets=[toolset],
        capabilities=[AbsurdDurability()],
    )
    absurd.register_task(name='idless-mcp')(_noop)

    async with toolset:
        async with running_task_context(absurd, 'idless-mcp', max_attempts=2) as ctx:
            first = await agent.run('add these')
            task_id = ctx.task_id
            assert {
                'idless-mcp__model.request',
                'idless-mcp__mcp_server.get_tools',
                'idless-mcp__mcp_server.call_tool',
            } <= set(ctx._checkpoint_cache)
            assert ctx._checkpoint_cache['idless-mcp__mcp_server.call_tool'] == 5

    async with toolset:
        async with reenter_running_task(absurd, task_id):
            replay = await agent.run('add these')

    assert first.output == replay.output == 'done'
    assert calls['calls'] == 1


async def test_explicit_mcp_toolset_replays(absurd: AsyncAbsurd) -> None:
    calls = {'calls': 0}
    toolset = MCPToolset(_server(calls), id='calculator')
    agent = Agent[object, str](
        _mcp_model(),
        name='explicit-mcp',
        toolsets=[toolset],
        capabilities=[AbsurdDurability()],
    )
    absurd.register_task(name='explicit-mcp')(_noop)

    async with running_task_context(absurd, 'explicit-mcp', max_attempts=2) as ctx:
        first = await agent.run('add these')
        task_id = ctx.task_id
        assert {
            'explicit-mcp__model.request',
            'explicit-mcp__mcp_server__calculator.get_tools',
            'explicit-mcp__mcp_server__calculator.call_tool',
        } <= set(ctx._checkpoint_cache)
        assert ctx._checkpoint_cache['explicit-mcp__mcp_server__calculator.call_tool'] == 5

    async with reenter_running_task(absurd, task_id):
        replay = await agent.run('add these')

    assert first.output == replay.output == 'done'
    assert calls['calls'] == 1


async def test_mcp_without_tool_cache_checkpoints_each_discovery(absurd: AsyncAbsurd) -> None:
    calls = {'calls': 0}
    toolset = MCPToolset(_server(calls), id='uncached', cache_tools=False)
    agent = Agent[object, str](
        _mcp_model(),
        name='uncached-mcp',
        toolsets=[toolset],
        capabilities=[AbsurdDurability()],
    )
    absurd.register_task(name='uncached-mcp')(_noop)

    async with running_task_context(absurd, 'uncached-mcp', max_attempts=2) as ctx:
        first = await agent.run('add these')
        task_id = ctx.task_id
        assert {
            'uncached-mcp__mcp_server__uncached.get_tools',
            'uncached-mcp__mcp_server__uncached.get_tools#2',
        } <= set(ctx._checkpoint_cache)

    async with reenter_running_task(absurd, task_id):
        replay = await agent.run('add these')

    assert first.output == replay.output == 'done'
    assert calls['calls'] == 1


async def test_mcp_instructions_replay_with_external_lifecycle(absurd: AsyncAbsurd) -> None:
    calls = {'calls': 0}
    toolset = MCPToolset(
        _server(calls, instructions='Use the calculator.'), id='instructions', include_instructions=True
    )
    agent = Agent[object, str](
        _mcp_model(),
        name='mcp-instructions',
        toolsets=[toolset],
        capabilities=[AbsurdDurability()],
    )
    absurd.register_task(name='mcp-instructions')(_noop)

    async with toolset:
        instructions = await toolset.get_instructions(_run_context())
        assert instructions is not None
        assert instructions.content == 'Use the calculator.'
        async with running_task_context(absurd, 'mcp-instructions', max_attempts=2) as ctx:
            first = await agent.run('add these')
            task_id = ctx.task_id
            # The externally entered MCP session supplies static instructions directly. Discovery
            # and tool calls remain checkpointed, but the already-entered lifecycle is not reopened.
            assert 'mcp-instructions__mcp_server__instructions.get_instructions' not in ctx._checkpoint_cache

    async with toolset:
        async with reenter_running_task(absurd, task_id):
            replay = await agent.run('add these')

    assert first.output == replay.output == 'done'
    assert calls['calls'] == 1


async def test_mcp_instructions_are_checkpointed_without_external_lifecycle(absurd: AsyncAbsurd) -> None:
    calls = {'calls': 0}
    toolset = MCPToolset(
        _server(calls, instructions='Use the calculator.'), id='instructions-checkpointed', include_instructions=True
    )
    agent = Agent[object, str](
        _mcp_model(),
        name='mcp-instructions-checkpointed',
        toolsets=[toolset],
        capabilities=[AbsurdDurability()],
    )
    absurd.register_task(name='mcp-instructions-checkpointed')(_noop)
    step = 'mcp-instructions-checkpointed__mcp_server__instructions-checkpointed.get_instructions'

    async with running_task_context(absurd, 'mcp-instructions-checkpointed', max_attempts=2) as ctx:
        first = await agent.run('add these')
        task_id = ctx.task_id
        assert ctx._checkpoint_cache[step] == {
            'content': 'Use the calculator.',
            'dynamic': False,
            'part_kind': 'instruction',
        }

    async with reenter_running_task(absurd, task_id):
        replay = await agent.run('add these')

    assert first.output == replay.output == 'done'
    assert calls['calls'] == 1


async def test_mcp_toolset_is_transparent_outside_task() -> None:
    calls = {'calls': 0}
    agent = Agent[object, str](
        _mcp_model(),
        name='outside-mcp',
        toolsets=[MCPToolset(_server(calls))],
        capabilities=[AbsurdDurability()],
    )
    result = await agent.run('add these')
    assert result.output == 'done'
    assert calls['calls'] == 1
