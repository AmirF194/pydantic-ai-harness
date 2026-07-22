"""MCP and dynamic-toolset memoization tests for `AirflowDurability`.

A lightweight `FakeMCPToolset` stands in for a real server: it is a genuine `MCPToolset` subclass,
so the capability's `isinstance` wrapping and `tool_for_tool_def` rebuild apply, but its wire methods
return in-memory results, keeping the test off Docker and the network.
"""

from __future__ import annotations

import pytest

pytest.importorskip('pydantic_ai.mcp')

from typing import Any

from pydantic_ai import Agent, ToolsetTool
from pydantic_ai.exceptions import UserError
from pydantic_ai.mcp import MCPToolset
from pydantic_ai.messages import (
    InstructionPart,
    ModelMessage,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.tools import RunContext, ToolDefinition
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai.toolsets._dynamic import DynamicToolset  # pyright: ignore[reportPrivateUsage]

from pydantic_ai_harness.airflow import DURABLE_KEY_PREFIX, AirflowDurability, InMemoryDurableStorage

_ADD_SCHEMA = {
    'type': 'object',
    'properties': {'a': {'type': 'integer'}, 'b': {'type': 'integer'}},
    'required': ['a', 'b'],
}


class FakeMCPToolset(MCPToolset[Any]):
    """In-memory `MCPToolset` whose I/O methods return canned results."""

    def __init__(
        self,
        *,
        id: str,
        instructions: str | None = 'Use the calculator.',
        tool_metadata: dict[str, Any] | None = None,
    ) -> None:
        self._id = id
        self.max_retries = None
        self.cache_tools = True
        self.include_instructions = True
        self.include_return_schema = None
        self._instructions_text = instructions
        self._tool_metadata = tool_metadata
        self.tool_calls: list[tuple[str, dict[str, Any]]] = []

    async def __aenter__(self) -> FakeMCPToolset:
        return self

    async def __aexit__(self, *args: object) -> None:
        return None

    async def get_tools(self, ctx: RunContext[Any]) -> dict[str, ToolsetTool[Any]]:
        tool_def = ToolDefinition(
            name='add',
            description='Add two integers.',
            parameters_json_schema=_ADD_SCHEMA,
            metadata=self._tool_metadata,
        )
        return {'add': self.tool_for_tool_def(tool_def)}

    async def get_instructions(self, ctx: RunContext[Any]) -> InstructionPart | None:
        if self._instructions_text is None:
            return None
        return InstructionPart(content=self._instructions_text)

    async def call_tool(
        self, name: str, tool_args: dict[str, Any], ctx: RunContext[Any], tool: ToolsetTool[Any]
    ) -> int:
        self.tool_calls.append((name, dict(tool_args)))
        return int(tool_args['a']) + int(tool_args['b'])


def add_then_done() -> FunctionModel:
    def fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        answered = any(isinstance(part, ToolReturnPart) for message in messages for part in message.parts)
        if answered:
            return ModelResponse(parts=[TextPart(content='summed')])
        return ModelResponse(parts=[ToolCallPart('add', {'a': 2, 'b': 3})])

    return FunctionModel(fn, model_name='fn')


class TestMcpMemoization:
    def test_get_tools_instructions_and_call_tool_are_memoized(self) -> None:
        store = InMemoryDurableStorage()
        server = FakeMCPToolset(id='calc')
        agent = Agent(add_then_done(), name='c', toolsets=[server], capabilities=[AirflowDurability(storage=store)])

        result = agent.run_sync('add 2 and 3')

        assert result.output == 'summed'
        assert server.tool_calls == [('add', {'a': 2, 'b': 3})]
        assert any(k.startswith(f'{DURABLE_KEY_PREFIX}mcp_get_tools_step') for k in store.keys)
        assert any(k.startswith(f'{DURABLE_KEY_PREFIX}mcp_get_instructions_step') for k in store.keys)
        assert any(k.startswith(f'{DURABLE_KEY_PREFIX}mcp_call_tool_step') for k in store.keys)

    def test_retry_does_not_reach_the_server(self) -> None:
        store = InMemoryDurableStorage()

        def make() -> tuple[Agent[Any, Any], FakeMCPToolset]:
            server = FakeMCPToolset(id='calc')
            agent = Agent(add_then_done(), name='c', toolsets=[server], capabilities=[AirflowDurability(storage=store)])
            return agent, server

        first_agent, first_server = make()
        first_agent.run_sync('add 2 and 3')
        assert len(first_server.tool_calls) == 1

        second_agent, second_server = make()
        result = second_agent.run_sync('add 2 and 3')

        assert result.output == 'summed'
        # Fully replayed: the second server was never called.
        assert second_server.tool_calls == []

    def test_mcp_tool_cannot_opt_out_of_memoization(self) -> None:
        store = InMemoryDurableStorage()
        server = FakeMCPToolset(id='calc', tool_metadata={'airflow': False})
        agent = Agent(add_then_done(), name='c', toolsets=[server], capabilities=[AirflowDurability(storage=store)])

        with pytest.raises(UserError, match='cannot run outside a memoized step'):
            agent.run_sync('add 2 and 3')

    def test_instructions_omitted_when_disabled(self) -> None:
        store = InMemoryDurableStorage()
        server = FakeMCPToolset(id='calc', instructions=None)
        agent = Agent(add_then_done(), name='c', toolsets=[server], capabilities=[AirflowDurability(storage=store)])

        agent.run_sync('add 2 and 3')

        assert result_has_no_instructions(server)


def result_has_no_instructions(server: FakeMCPToolset) -> bool:
    return server._instructions_text is None  # pyright: ignore[reportPrivateUsage]


class TestDynamicToolset:
    def test_dynamic_resolution_and_call_are_memoized_and_replay(self) -> None:
        store = InMemoryDurableStorage()
        resolves: list[int] = []

        def factory(ctx: RunContext[Any]) -> FunctionToolset[Any]:
            resolves.append(1)
            inner = FunctionToolset[Any](id='dyn')

            @inner.tool_plain
            def act() -> str:
                return 'dynresult'

            return inner

        def make() -> Agent[Any, Any]:
            dyn = DynamicToolset(factory, id='dyn')
            return Agent(dyn_model(), name='d', toolsets=[dyn], capabilities=[AirflowDurability(storage=store)])

        assert make().run_sync('go').output == 'done'
        assert resolves  # the factory ran at least once on the first run
        assert any(k.startswith(f'{DURABLE_KEY_PREFIX}dynamic_get_tools_step') for k in store.keys)
        assert any(k.startswith(f'{DURABLE_KEY_PREFIX}dynamic_call_tool_step') for k in store.keys)

        resolves.clear()
        assert make().run_sync('go').output == 'done'
        # Replay serves the resolution and the tool call from the store; the factory is not re-run.
        assert resolves == []


def dyn_model() -> FunctionModel:
    def fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        answered = any(isinstance(part, ToolReturnPart) for message in messages for part in message.parts)
        if answered:
            return ModelResponse(parts=[TextPart(content='done')])
        return ModelResponse(parts=[ToolCallPart('act', {})])

    return FunctionModel(fn)
