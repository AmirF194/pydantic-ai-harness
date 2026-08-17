"""Background tools capability that spawns selected tools as fire-and-forget tasks."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

import anyio
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import ToolCallPart
from pydantic_ai.tools import (
    AgentDepsT,
    DeferredToolRequests,
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
    message -- Pydantic AI's pending message queue delivers it on the next model
    request, or redirects the agent to a fresh request if it would otherwise end,
    so the model receives the result and can act on it.

    ```python
    import asyncio

    from pydantic_ai import Agent
    from pydantic_ai_harness import BackgroundTools

    # Default: any tool with `metadata={'background': True}` runs in the background.
    agent = Agent('openai:gpt-5', capabilities=[BackgroundTools()])

    @agent.tool_plain(metadata={'background': True})
    async def slow_research(query: str) -> str:
        await asyncio.sleep(60)  # stand-in for a long-running job
        return f'Research findings for {query!r}'
    ```

    Combine with [`SetToolMetadata`][pydantic_ai.capabilities.SetToolMetadata] to mark
    several tools at once, or with `FunctionToolset.with_metadata(...)` to mark a whole
    toolset. Or pass a name list / predicate via `tools=...` to ignore metadata entirely.
    """

    tools: ToolSelector[AgentDepsT] = field(default_factory=lambda: {'background': True})
    """Which tools should run in the background.

    - `dict[str, Any]` (default `{'background': True}`): tools whose metadata deeply
      includes the given key-value pairs.
    - `'all'`: every tool in the agent's toolset (rarely what you want).
    - `Sequence[str]`: tools with matching names.
    - Callable `(ctx, tool_def) -> bool | Awaitable[bool]`: custom predicate.
    """

    _tasks: set[asyncio.Task[None]] = field(default_factory=set[asyncio.Task[None]], init=False, repr=False)

    def get_instructions(self) -> AgentInstructions[AgentDepsT] | None:
        return _INSTRUCTIONS

    async def for_run(self, ctx: RunContext[AgentDepsT]) -> BackgroundTools[AgentDepsT]:
        # `_tasks` is init=False, so `replace` gives the copy a fresh set:
        # concurrent runs don't share tasks.
        return replace(self)

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

        task = asyncio.create_task(_run())
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
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

        if not isinstance(result, End) or isinstance(result.data.output, DeferredToolRequests):
            # A deferred-tools pause must reach the caller immediately; never hold it
            # behind background work.
            return result
        # Hold `End` until a follow-up is queued, so the core pending-message drain
        # capability (always ordered outermost, i.e. after us) redirects the run to
        # deliver it instead of terminating. The loop re-waits if a task finished
        # without enqueueing (e.g. it was cancelled); the queue check skips waiting
        # when a result is already pending delivery.
        while self._tasks and not ctx.pending_messages:
            await asyncio.wait(self._tasks, return_when=asyncio.FIRST_COMPLETED)
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
            # The shield holds the drain open under anyio-scope cancellation of
            # the run. A raw second `task.cancel()` still pierces it and abandons
            # the drain, which is accepted: every task is already cancelled here.
            with anyio.CancelScope(shield=True):
                for task in self._tasks:
                    task.cancel()
                await asyncio.gather(*self._tasks, return_exceptions=True)
