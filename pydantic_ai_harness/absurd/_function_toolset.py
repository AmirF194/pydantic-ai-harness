from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager
from typing import Any

from pydantic_ai import FunctionToolset, ToolsetTool, WrapperToolset
from pydantic_ai.exceptions import UserError
from pydantic_ai.tools import AgentDepsT, RunContext
from typing_extensions import Self

from ._utils import current_async_context

_DurableRunContextScope = Callable[[RunContext[AgentDepsT]], AbstractContextManager[RunContext[AgentDepsT]]]


class AbsurdFunctionToolset(WrapperToolset[AgentDepsT]):
    """Checkpoint calls made through a `FunctionToolset` in Absurd steps."""

    def __init__(
        self,
        wrapped: FunctionToolset[AgentDepsT],
        *,
        step_name_prefix: str,
        durable_run_context_scope: _DurableRunContextScope[AgentDepsT],
    ) -> None:
        super().__init__(wrapped)
        id_suffix = f'__{wrapped.id}' if wrapped.id else ''
        self._name = f'{step_name_prefix}__function_toolset{id_suffix}'
        self._durable_run_context_scope = durable_run_context_scope

    @property
    def id(self) -> str | None:
        return self.wrapped.id

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *args: Any) -> bool | None:
        return None

    def visit_and_replace(self, visitor: Callable[[Any], Any]) -> Any:
        return self

    async def call_tool(
        self,
        name: str,
        tool_args: dict[str, Any],
        ctx: RunContext[AgentDepsT],
        tool: ToolsetTool[AgentDepsT],
    ) -> Any:
        task_ctx = current_async_context()
        if task_ctx is not None and '#' in name:
            raise UserError(
                f'Function tool name {name!r} contains `#`, which Absurd uses to disambiguate repeated steps.'
            )

        if task_ctx is None:
            return await self.wrapped.call_tool(name, tool_args, ctx, tool)

        async def call() -> Any:
            with self._durable_run_context_scope(ctx) as step_ctx:
                return await self.wrapped.call_tool(name, tool_args, step_ctx, tool)

        return await task_ctx.step(f'{self._name}.call_tool:{name}', call)
