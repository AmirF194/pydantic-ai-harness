from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from typing import Any, Literal, TypeAlias

from pydantic_ai.durable_exec._operation import DurableOperationId, OperationConfigRole
from pydantic_ai.durable_exec._operation_backend import CallableOperationBackend
from pydantic_ai.durable_exec._operation_names import JournalOperationNamer

from ._context import current_async_task_context

DurableConfig: TypeAlias = Mapping[str, Any]
ToolConfig: TypeAlias = DurableConfig | Literal[False]


class AbsurdOperationConfig:
    def __init__(self, resolve_tool: Callable[[object | None, str], ToolConfig]) -> None:
        self._resolve_tool = resolve_tool

    def base(self, role: OperationConfigRole, operation_id: DurableOperationId) -> DurableConfig:
        del role, operation_id
        return {}

    def for_tool(
        self,
        role: OperationConfigRole,
        operation_id: DurableOperationId,
        tool: object | None,
        tool_name: str,
    ) -> ToolConfig:
        del role, operation_id
        return self._resolve_tool(tool, tool_name)


class AbsurdOperationBackend(CallableOperationBackend[ToolConfig]):
    def __init__(self, *, agent_name: str, default_model_id: str | None, config: AbsurdOperationConfig) -> None:
        super().__init__(
            namer=JournalOperationNamer(agent_name, default_model_id=default_model_id or 'default'), config=config
        )

    async def _execute(
        self,
        *,
        name: str,
        body: Callable[[], Awaitable[object]],
        cache_key: tuple[object, ...],
        config: object,
    ) -> object:
        del cache_key
        assert not config
        task_ctx = current_async_task_context()
        assert task_ctx is not None
        return await task_ctx.step(name, body)
