"""MCP durable-wrapping tests for `AbsurdDurability`.

The MCP extra (`mcp`/`fastmcp`) may be absent, so the module `importorskip`s it. A lightweight
`FakeMCPToolset` stands in for a real server: it is a genuine `MCPToolset` subclass (so the
capability's `isinstance` wrapping and `tool_for_tool_def` rebuild apply) whose wire methods return
in-memory results. The ordinary checkpoint tests still use a real PostgreSQL-backed Absurd task
context; the fake server avoids Docker and network transport setup.
"""

from __future__ import annotations

import pytest

pytest.importorskip('absurd_sdk')
pytest.importorskip('pydantic_ai.mcp')

from typing import Any

from absurd_sdk import AsyncAbsurd
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

from ._postgres import reenter_running_task, running_task_context
from .conftest import (
    FakeAsyncTaskContext,
    absurd_task_context,
)

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
        self.get_tools_calls = 0
        self.get_instructions_calls = 0
        self.enter_count = 0
        self._session_depth = 0
        self.implicit_sessions = 0

    async def __aenter__(self) -> FakeMCPToolset:
        self.enter_count += 1
        self._session_depth += 1
        return self

    async def __aexit__(self, *args: object) -> None:
        self._session_depth -= 1

    async def _require_session(self) -> None:
        """Model a real MCP server: I/O needs an active session. If the wrapper did not enter one,
        the call opens its own implicit session for the duration of the call, as `MCPToolset` does."""
        if self._session_depth == 0:
            self.implicit_sessions += 1
            await self.__aenter__()
            await self.__aexit__(None, None, None)

    async def get_tools(self, ctx: RunContext[object]) -> dict[str, ToolsetTool[object]]:
        await self._require_session()
        self.get_tools_calls += 1
        tool_def = ToolDefinition(
            name='add',
            description='Add two integers.',
            parameters_json_schema=_ADD_SCHEMA,
            metadata=self._tool_metadata,
        )
        return {'add': self.tool_for_tool_def(tool_def, ctx=ctx)}

    async def get_instructions(self, ctx: RunContext[object]) -> InstructionPart | None:
        await self._require_session()
        self.get_instructions_calls += 1
        if not self.include_instructions or self._instructions_text is None:
            return None
        return InstructionPart(content=self._instructions_text)

    async def call_tool(
        self, name: str, tool_args: dict[str, Any], ctx: RunContext[object], tool: ToolsetTool[object]
    ) -> int:
        await self._require_session()
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
    async def test_replay_does_not_repeat_mcp_operations(self, absurd_client: AsyncAbsurd) -> None:
        server = FakeMCPToolset(id='calc', instructions='Use the calculator.', include_instructions=True)
        agent = Agent(_add_then_done_model(), name='calc', toolsets=[server], capabilities=[AbsurdDurability()])

        async with running_task_context(absurd_client, 'mcp-replay', max_attempts=2) as ctx:
            first = await agent.run('add 2 and 3')

        first_tool_calls = server.tool_calls.copy()
        first_get_tools_calls = server.get_tools_calls
        first_get_instructions_calls = server.get_instructions_calls
        assert first_tool_calls == [('add', {'a': 2, 'b': 3})]
        assert first_get_tools_calls > 0
        assert first_get_instructions_calls > 0

        async with reenter_running_task(absurd_client, ctx.task_id):
            second = await agent.run('add 2 and 3')

        assert first.output == second.output == 'summed'
        assert server.tool_calls == first_tool_calls
        assert server.get_tools_calls == first_get_tools_calls
        assert server.get_instructions_calls == first_get_instructions_calls

    async def test_raw_checkpoint_from_the_reference_package_replays(self) -> None:
        # `pydantic-ai-absurd` stores a tool's return value directly rather than the wrapped form,
        # so a run started there and resumed here finds a raw checkpoint under the same step name.
        server = FakeMCPToolset(id='calc')
        agent = Agent(_add_then_done_model(), name='calc', toolsets=[server], capabilities=[AbsurdDurability()])

        step = 'calc__mcp_server__calc.call_tool'
        ctx = FakeAsyncTaskContext(store={step: 99})
        with absurd_task_context(ctx):
            result = await agent.run('add 2 and 3')

        assert result.output == 'summed'
        # The seeded checkpoint answered the call, so the server was never asked to add anything.
        assert server.tool_calls == []
        returned = [part.content for m in result.all_messages() for part in m.parts if isinstance(part, ToolReturnPart)]
        assert returned == [99]


class TestFakeServerModelsImplicitSessions:
    async def test_io_without_an_open_session_opens_an_implicit_one(self) -> None:
        # Proves the fake server models a real `MCPToolset`: I/O with no session open opens a
        # transient implicit one, so the `implicit_sessions == 0` assertions below are meaningful.
        server = FakeMCPToolset(id='calc')
        await server._require_session()
        assert server.implicit_sessions == 1

        async with server:
            await server._require_session()
        assert server.implicit_sessions == 1


class TestMcpSessionLifecycle:
    async def test_wrapper_holds_one_session_no_implicit_per_call(self) -> None:
        # `enter-always`: the durable wrapper enters the server for the run, so `get_tools` and
        # `call_tool` ride one session instead of each opening its own implicit session.
        server = FakeMCPToolset(id='calc', instructions='Use the calculator.', include_instructions=True)
        agent = Agent(_add_then_done_model(), name='calc', toolsets=[server], capabilities=[AbsurdDurability()])

        ctx = FakeAsyncTaskContext()
        with absurd_task_context(ctx):
            result = await agent.run('add 2 and 3')

        assert result.output == 'summed'
        assert server.implicit_sessions == 0
        assert server.enter_count == 1

    async def test_transparent_run_also_reuses_one_session(self) -> None:
        # Outside a task the capability is transparent, but the wrapper still enters the server, so
        # ordinary non-durable runs also avoid a fresh implicit session per call.
        server = FakeMCPToolset(id='calc', instructions='Use the calculator.', include_instructions=True)
        agent = Agent(_add_then_done_model(), name='calc', toolsets=[server], capabilities=[AbsurdDurability()])

        result = await agent.run('add 2 and 3')

        assert result.output == 'summed'
        assert server.implicit_sessions == 0


class TestMcpInlineOptOutForbidden:
    async def test_metadata_false_on_mcp_tool_raises(self) -> None:
        server = FakeMCPToolset(id='calc', tool_metadata={'absurd': False})
        agent = Agent(_add_then_done_model(), name='calc', toolsets=[server], capabilities=[AbsurdDurability()])

        ctx = FakeAsyncTaskContext()
        with absurd_task_context(ctx):
            with pytest.raises(UserError, match='MCP tool .* cannot run outside a step'):
                await agent.run('add 2 and 3')

    async def test_non_empty_dict_config_on_mcp_tool_raises(self) -> None:
        server = FakeMCPToolset(id='calc', tool_metadata={'absurd': {'retries': 3}})
        agent = Agent(_add_then_done_model(), name='calc', toolsets=[server], capabilities=[AbsurdDurability()])

        ctx = FakeAsyncTaskContext()
        with absurd_task_context(ctx):
            with pytest.raises(UserError, match='take no per-tool options'):
                await agent.run('add 2 and 3')
