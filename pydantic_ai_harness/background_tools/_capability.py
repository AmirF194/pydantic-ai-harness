"""Background tools capability that spawns selected tools as fire-and-forget tasks."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import ToolCallPart
from pydantic_ai.tools import (
    AgentDepsT,
    RunContext,
    ToolDefinition,
    ToolSelector,
    matches_tool_selector,
)

if TYPE_CHECKING:
    from pydantic_ai import _agent_graph
    from pydantic_ai._instructions import AgentInstructions
    from pydantic_ai.capabilities.abstract import WrapRunHandler, WrapToolExecuteHandler
    from pydantic_ai.result import FinalResult
    from pydantic_ai.run import AgentRunResult
    from pydantic_graph import End


_INSTRUCTIONS = """\
Some tools run in the background: when you call them you'll get an immediate \
acknowledgment, and the real result will be delivered automatically as a follow-up \
message when the task completes. Continue working on other things in the meantime; \
do not block waiting for the result.\
"""


@dataclass
class BackgroundTools(AbstractCapability[AgentDepsT]):
    """Run selected tools as fire-and-forget asyncio tasks.

    When the model calls a tool that matches the selector, the capability spawns the
    tool's handler in an `asyncio.Task` and immediately returns an acknowledgment
    string to the agent. When the task completes, its result (or error) is enqueued
    via [`RunContext.enqueue`][pydantic_ai.tools.RunContext.enqueue] as an `'asap'`
    message — Pydantic AI's pending message queue delivers it on the next model
    request, or redirects the agent to a fresh request if it would otherwise end,
    so the model receives the result and can act on it.

    ```python
    from pydantic_ai import Agent
    from pydantic_ai_harness import BackgroundTools

    # Default: any tool with `metadata={'background': True}` runs in the background.
    agent = Agent('openai:gpt-5', capabilities=[BackgroundTools()])

    @agent.tool_plain(metadata={'background': True})
    async def slow_research(query: str) -> str:
        return await do_expensive_research(query)
    ```

    Combine with [`SetToolMetadata`][pydantic_ai.capabilities.SetToolMetadata] to mark
    every tool from a specific MCP server, or with `FunctionToolset.with_metadata(...)`
    to mark a whole toolset. Or pass a name list / predicate via `tools=...` to ignore
    metadata entirely.
    """

    tools: ToolSelector[AgentDepsT] = field(default_factory=lambda: {'background': True})
    """Which tools should run in the background.

    - `dict[str, Any]` (default `{'background': True}`): tools whose metadata deeply
      includes the given key-value pairs.
    - `'all'`: every tool in the agent's toolset (rarely what you want).
    - `Sequence[str]`: tools with matching names.
    - Callable `(ctx, tool_def) -> bool | Awaitable[bool]`: custom predicate.
    """

    _tasks: dict[str, asyncio.Task[None]] = field(
        default_factory=dict[str, 'asyncio.Task[None]'], init=False, repr=False
    )

    def get_instructions(self) -> AgentInstructions[AgentDepsT] | None:
        return _INSTRUCTIONS

    async def for_run(self, ctx: RunContext[AgentDepsT]) -> BackgroundTools[AgentDepsT]:
        # Fresh per-run state so concurrent runs don't share tasks.
        return BackgroundTools(tools=self.tools)

    async def wrap_tool_execute(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: dict[str, Any],
        handler: WrapToolExecuteHandler,
    ) -> Any:
        if not await matches_tool_selector(self.tools, ctx, tool_def):
            return await handler(args)

        task_id = call.tool_call_id
        tool_name = call.tool_name

        async def _run() -> None:
            try:
                result = await handler(args)
                ctx.enqueue(f"Background tool '{tool_name}' (task {task_id}) completed.\nResult: {result}")
            except Exception as e:
                ctx.enqueue(f"Background tool '{tool_name}' (task {task_id}) failed: {e}")
            finally:
                self._tasks.pop(task_id, None)

        self._tasks[task_id] = asyncio.create_task(_run())
        return (
            f"Tool '{tool_name}' is running in background (task {task_id}). "
            f'You will receive the result automatically when it completes. '
            f'Continue with other work in the meantime.'
        )

    async def after_node_run(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        node: _agent_graph.AgentNode[AgentDepsT, Any],
        result: _agent_graph.AgentNode[AgentDepsT, Any] | End[FinalResult[Any]],
    ) -> _agent_graph.AgentNode[AgentDepsT, Any] | End[FinalResult[Any]]:
        from pydantic_graph import End

        if isinstance(result, End) and self._tasks:
            # Hold `End` until at least one task completes and enqueues its follow-up,
            # so the core pending-message drain capability (always ordered outermost,
            # i.e. after us) redirects the run to deliver it instead of terminating.
            await asyncio.wait(list(self._tasks.values()), return_when=asyncio.FIRST_COMPLETED)
        return result

    async def wrap_run(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        handler: WrapRunHandler,
    ) -> AgentRunResult[Any]:
        try:
            return await handler()
        finally:
            for task in self._tasks.values():
                task.cancel()
            await asyncio.gather(*self._tasks.values(), return_exceptions=True)
