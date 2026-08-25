"""Background tools capability that runs selected tools concurrently."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

import anyio
from anyio.abc import Event, TaskGroup
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.exceptions import ApprovalRequired, CallDeferred, ToolFailedError, ToolRetryError
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
acknowledgment. If the run remains active, the text result will be delivered \
automatically as a follow-up message when the task completes. Continue working on other \
things in the meantime; do not block waiting for the result.\
"""


def _format_background_error(error: Exception) -> str:
    """Format a background-tool error without exposing unexpected exception details."""
    if isinstance(error, (ApprovalRequired, CallDeferred)):
        return f'{type(error).__name__} was raised; background tools cannot defer a running task.'
    if isinstance(error, ToolRetryError):
        content = error.tool_retry.content
        return content if isinstance(content, str) and content else type(error).__name__
    if isinstance(error, ToolFailedError):
        content = error.tool_failed.content
        return content if isinstance(content, str) and content else type(error).__name__
    return type(error).__name__


@dataclass
class BackgroundTools(AbstractCapability[AgentDepsT]):
    """Run selected tools concurrently with the current agent run.

    When the model calls a tool that matches the selector, the capability spawns the
    tool's handler in an `asyncio.Task` and immediately returns an acknowledgment
    string to the agent. When the task completes, its result (or error) is formatted as
    text and enqueued via
    [`RunContext.enqueue`][pydantic_ai.tools.RunContext.enqueue] as an `'asap'` message.
    Pydantic AI's pending message queue delivers it on the next model request, or
    redirects the agent to a fresh request if it would otherwise end, so the model
    receives the result and can act on it while the run remains active.

    ```python
    import asyncio

    from pydantic_ai import Agent
    from pydantic_ai_harness import BackgroundTools

    # Default: any tool with `metadata={'background': True}` runs in the background.
    agent = Agent('openai:gpt-5.6-sol', capabilities=[BackgroundTools()])

    @agent.tool_plain(metadata={'background': True})
    async def slow_research(query: str) -> str:
        await asyncio.sleep(60)  # stand-in for a long-running job
        return f'Research findings for {query!r}'
    ```

    Combine with [`SetToolMetadata`][pydantic_ai.capabilities.SetToolMetadata] to mark
    several tools at once, or with `FunctionToolset.with_metadata(...)` to mark a whole
    toolset. Or pass a name list / predicate via `tools=...` to ignore metadata entirely.

    Warning:
        Run cleanup cancels and drains background tasks before it completes. Async tools
        must propagate cancellation. Python cannot stop a synchronous tool's worker
        thread, so cleanup may wait for the function to return depending on the configured
        executor's cancellation behavior.

        A synchronous background tool runs concurrently with the agent. Make mutable
        dependencies and other shared state it uses thread-safe.

        A synchronous background tool must not call `ctx.enqueue()`: its worker thread
        can race the pending-message drain and lose the message. Async background tools
        do not have this cross-thread race, but delivery still requires the run to
        continue.

    `BackgroundTools` composes with Temporal and Prefect durable execution. With DBOS,
    ordinary function tools are not automatically durable steps, so a background tool
    must delegate durable work to an explicit DBOS step. A tool handler running inside
    a durable activity or task must not call `ctx.enqueue()` because replay restores
    only its return value.
    """

    tools: ToolSelector[AgentDepsT] = field(default_factory=lambda: {'background': True})
    """Which tools should run in the background.

    - `dict[str, Any]` (default `{'background': True}`): tools whose metadata deeply
      includes the given key-value pairs.
    - `'all'`: every tool in the agent's toolset (rarely what you want).
    - `Sequence[str]`: tools with matching names.
    - Callable `(ctx, tool_def) -> bool | Awaitable[bool]`: custom predicate.
    """

    _task_group: TaskGroup | None = field(default=None, init=False, repr=False)
    _completion_event: Event | None = field(default=None, init=False, repr=False)
    _pending_tasks: int = field(default=0, init=False, repr=False)
    _completed: list[str] = field(default_factory=list[str], init=False, repr=False)

    def get_instructions(self) -> AgentInstructions[AgentDepsT] | None:
        return _INSTRUCTIONS

    async def for_run(self, ctx: RunContext[AgentDepsT]) -> BackgroundTools[AgentDepsT]:
        run_capability = replace(self)
        run_capability._completion_event = anyio.Event()
        return run_capability

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
                try:
                    result = await handler(args)
                    message = f"Background tool '{tool_name}' (task {task_id}) completed.\nResult: {result}"
                except Exception as e:
                    error = _format_background_error(e)
                    message = f"Background tool '{tool_name}' (task {task_id}) failed: {error}"
                self._completed.append(message)
            finally:
                self._pending_tasks -= 1
                completion_event = self._completion_event
                assert completion_event is not None
                completion_event.set()

        task_group = self._task_group
        assert task_group is not None
        self._pending_tasks += 1
        task_group.start_soon(_run, name=f'background tool {tool_name} ({task_id})')
        return (
            f"Tool '{tool_name}' is running in background (task {task_id}). "
            f'If this run remains active, you will receive the text result automatically when it completes. '
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

        if isinstance(result, End) and isinstance(result.data.output, DeferredToolRequests):
            # Background results are dropped when the run pauses.
            return result

        # Let the outer drain deliver anything already queued before waiting for
        # another completion.
        while isinstance(result, End) and self._pending_tasks and not self._completed and not ctx.pending_messages:
            completion_event = self._completion_event
            assert completion_event is not None
            await completion_event.wait()

        for message in self._completed:
            ctx.enqueue(message)
        self._completed.clear()
        self._completion_event = anyio.Event()
        return result

    async def wrap_run(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        handler: WrapRunHandler,
    ) -> AgentRunResult[Any]:
        run_result: AgentRunResult[Any] | None = None
        try:
            async with anyio.create_task_group() as task_group:
                self._task_group = task_group
                try:
                    run_result = await handler()
                finally:
                    task_group.cancel_scope.cancel()
        finally:
            self._task_group = None
        assert run_result is not None
        return run_result
