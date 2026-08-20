from __future__ import annotations

from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager
from typing import Any

from absurd_sdk import JsonValue
from pydantic import TypeAdapter
from pydantic_ai import ToolsetTool, WrapperToolset
from pydantic_ai.durable_exec._toolset import wrap_tool_call_result
from pydantic_ai.mcp import MCPToolset
from pydantic_ai.messages import InstructionPart
from pydantic_ai.tools import AgentDepsT, RunContext, ToolDefinition
from typing_extensions import Self

from ._tool_result import serialize_tool_call_result, unwrap_tool_call_checkpoint
from ._utils import current_async_context

Instructions = str | InstructionPart | Sequence[str | InstructionPart] | None
_DurableRunContextScope = Callable[[RunContext[AgentDepsT]], AbstractContextManager[RunContext[AgentDepsT]]]

_tool_defs_adapter: TypeAdapter[dict[str, ToolDefinition]] = TypeAdapter(dict[str, ToolDefinition])
_instructions_adapter: TypeAdapter[Instructions] = TypeAdapter(Instructions)


def _serialize_tool_defs(defs: dict[str, ToolDefinition]) -> dict[str, JsonValue]:
    dumped = _tool_defs_adapter.dump_python(defs, mode='json')
    assert isinstance(dumped, dict)
    return dumped  # pyright: ignore[reportUnknownVariableType]


def _deserialize_tool_defs(payload: dict[str, JsonValue]) -> dict[str, ToolDefinition]:
    return _tool_defs_adapter.validate_python(payload)


def _serialize_instructions(value: Instructions) -> JsonValue:
    return _instructions_adapter.dump_python(value, mode='json')


def _deserialize_instructions(payload: JsonValue) -> Instructions:
    return _instructions_adapter.validate_python(payload)


class AbsurdMCPToolset(WrapperToolset[AgentDepsT]):
    """Checkpoint MCP discovery and calls in Absurd steps.

    The wrapper does not enter or exit the wrapped toolset. `MCPToolset` opens its own
    short-lived client session for each operation, or an application can manage the
    toolset lifecycle externally.
    """

    def __init__(
        self,
        wrapped: MCPToolset[AgentDepsT],
        *,
        step_name_prefix: str,
        durable_run_context_scope: _DurableRunContextScope[AgentDepsT],
    ) -> None:
        super().__init__(wrapped)
        id_suffix = f'__{wrapped.id}' if wrapped.id else ''
        self._name = f'{step_name_prefix}__mcp_server{id_suffix}'
        self._durable_run_context_scope = durable_run_context_scope

    @property
    def _server(self) -> MCPToolset[AgentDepsT]:
        assert isinstance(self.wrapped, MCPToolset)
        return self.wrapped

    def tool_for_tool_def(self, tool_def: ToolDefinition, *, ctx: RunContext[AgentDepsT]) -> ToolsetTool[AgentDepsT]:
        return self._server.tool_for_tool_def(tool_def, ctx=ctx)

    @property
    def id(self) -> str | None:
        return self.wrapped.id

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *args: Any) -> bool | None:
        return None

    def visit_and_replace(self, visitor: Callable[[Any], Any]) -> Any:
        return self

    async def get_tools(self, ctx: RunContext[AgentDepsT]) -> dict[str, ToolsetTool[AgentDepsT]]:
        async def _list_tools() -> dict[str, JsonValue]:
            with self._durable_run_context_scope(ctx) as step_ctx:
                tools = await super(AbsurdMCPToolset, self).get_tools(step_ctx)
            return _serialize_tool_defs({name: tool.tool_def for name, tool in tools.items()})

        task_ctx = current_async_context()
        if task_ctx is None:
            return await super().get_tools(ctx)
        payload = await task_ctx.step(f'{self._name}.get_tools', _list_tools)
        tool_defs = _deserialize_tool_defs(payload)
        return {name: self.tool_for_tool_def(tool_def, ctx=ctx) for name, tool_def in tool_defs.items()}

    async def get_instructions(self, ctx: RunContext[AgentDepsT]) -> Instructions:
        result = await super().get_instructions(ctx)
        if result is not None:
            return result
        task_ctx = current_async_context()
        if task_ctx is None or not self._server.include_instructions:
            return None

        async def _get_instructions() -> JsonValue:
            with self._durable_run_context_scope(ctx) as step_ctx:
                async with self.wrapped:
                    return _serialize_instructions(await super(AbsurdMCPToolset, self).get_instructions(step_ctx))

        payload = await task_ctx.step(f'{self._name}.get_instructions', _get_instructions)
        return _deserialize_instructions(payload)

    async def call_tool(
        self,
        name: str,
        tool_args: dict[str, Any],
        ctx: RunContext[AgentDepsT],
        tool: ToolsetTool[AgentDepsT],
    ) -> Any:
        task_ctx = current_async_context()
        if task_ctx is None:
            return await self._server.call_tool(name, tool_args, ctx, tool)

        async def call() -> Any:
            with self._durable_run_context_scope(ctx) as step_ctx:
                result = await wrap_tool_call_result(self._server.call_tool(name, tool_args, step_ctx, tool))
                return serialize_tool_call_result(result)

        payload = await task_ctx.step(f'{self._name}.call_tool', call)
        return unwrap_tool_call_checkpoint(payload)
