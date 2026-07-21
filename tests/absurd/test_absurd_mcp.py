"""MCP durable-wrapping tests for `AbsurdDurability`.

The MCP extra (`mcp`/`fastmcp`) may be absent, so the module `importorskip`s it. A lightweight
`FakeMCPToolset` stands in for a real server: it is a genuine `MCPToolset` subclass (so the
capability's `isinstance` wrapping and `tool_for_tool_def` rebuild apply) whose wire methods return
in-memory results, which keeps the test off Docker and the network.
"""

from __future__ import annotations

import pytest

pytest.importorskip('absurd_sdk')
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

from pydantic_ai_harness.absurd import AbsurdDurability

from .conftest import FakeAsyncTaskContext, absurd_task_context

pytestmark = pytest.mark.anyio

_ADD_SCHEMA = {
    'type': 'object',
    'properties': {'a': {'type': 'integer'}, 'b': {'type': 'integer'}},
    'required': ['a', 'b'],
}


class FakeMCPToolset(MCPToolset[object]):
    """In-memory `MCPToolset` whose I/O methods return canned results.

    Bypasses `MCPToolset.__init__` (which would build a real transport) and sets only the attributes
    the durable wrapper and the run touch.
    """

    def __init__(
        self,
        *,
        id: str,
        instructions: str | None = None,
        include_instructions: bool = True,
        tool_metadata: dict[str, object] | None = None,
    ) -> None:
        self._id = id
        self.max_retries = None
        self.cache_tools = True
        self.include_instructions = include_instructions
        self.include_return_schema = None
        self._instructions_text = instructions
        self._tool_metadata = tool_metadata
        self.tool_calls: list[tuple[str, dict[str, Any]]] = []
        self.enter_count = 0

    async def __aenter__(self) -> FakeMCPToolset:
        self.enter_count += 1
        return self

    async def __aexit__(self, *args: object) -> None:
        return None

    async def get_tools(self, ctx: RunContext[object]) -> dict[str, ToolsetTool[object]]:
        tool_def = ToolDefinition(
            name='add',
            description='Add two integers.',
            parameters_json_schema=_ADD_SCHEMA,
            metadata=self._tool_metadata,
        )
        return {'add': self.tool_for_tool_def(tool_def)}

    async def get_instructions(self, ctx: RunContext[object]) -> InstructionPart | None:
        if not self.include_instructions or self._instructions_text is None:
            return None
        return InstructionPart(content=self._instructions_text)

    async def call_tool(
        self, name: str, tool_args: dict[str, Any], ctx: RunContext[object], tool: ToolsetTool[object]
    ) -> int:
        self.tool_calls.append((name, dict(tool_args)))
        return int(tool_args['a']) + int(tool_args['b'])


def _add_then_done_model() -> FunctionModel:
    def fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        answered = any(isinstance(part, ToolReturnPart) for message in messages for part in message.parts)
        if not answered:
            return ModelResponse(parts=[ToolCallPart(tool_name='add', args={'a': 2, 'b': 3})])
        return ModelResponse(parts=[TextPart(content='summed')])

    return FunctionModel(fn, model_name='fn')


class TestMcpCheckpointing:
    async def test_get_tools_get_instructions_and_call_tool_checkpointed(self) -> None:
        server = FakeMCPToolset(id='calc', instructions='Use the calculator.', include_instructions=True)
        agent = Agent(_add_then_done_model(), name='calc', toolsets=[server], capabilities=[AbsurdDurability()])

        ctx = FakeAsyncTaskContext()
        with absurd_task_context(ctx):
            result = await agent.run('add 2 and 3')

        assert result.output == 'summed'
        assert server.tool_calls == [('add', {'a': 2, 'b': 3})]
        assert 'calc__mcp_server__calc.get_tools' in ctx.stored
        assert 'calc__mcp_server__calc.get_instructions' in ctx.stored
        assert 'calc__mcp_server__calc.call_tool' in ctx.stored

    async def test_replay_does_not_rehit_server(self) -> None:
        server = FakeMCPToolset(id='calc', instructions='Use the calculator.', include_instructions=True)
        agent = Agent(_add_then_done_model(), name='calc', toolsets=[server], capabilities=[AbsurdDurability()])

        ctx = FakeAsyncTaskContext()
        with absurd_task_context(ctx):
            first = await agent.run('add 2 and 3')

        replay = ctx.replay()
        with absurd_task_context(replay):
            second = await agent.run('add 2 and 3')

        assert first.output == second.output == 'summed'
        assert server.tool_calls == [('add', {'a': 2, 'b': 3})]
        assert replay.invoked == []


class TestMcpInlineOptOutForbidden:
    async def test_metadata_false_on_mcp_tool_raises(self) -> None:
        server = FakeMCPToolset(id='calc', tool_metadata={'absurd': False})
        agent = Agent(_add_then_done_model(), name='calc', toolsets=[server], capabilities=[AbsurdDurability()])

        ctx = FakeAsyncTaskContext()
        with absurd_task_context(ctx):
            with pytest.raises(UserError, match='MCP tool .* cannot run outside a step'):
                await agent.run('add 2 and 3')
