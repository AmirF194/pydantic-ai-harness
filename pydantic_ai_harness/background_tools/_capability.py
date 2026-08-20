"""Background tools capability that runs selected tools concurrently."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

import anyio
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.durable_exec._base import BaseDurabilityCapability  # pyright: ignore[reportPrivateUsage]
from pydantic_ai.exceptions import ApprovalRequired, CallDeferred, ToolFailedError, ToolRetryError, UserError
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
    from pydantic_ai.agent import AbstractAgent
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
    """Format a background-tool error without exposing deferred-call metadata."""
    if isinstance(error, (ApprovalRequired, CallDeferred)):
        return f'{type(error).__name__} was raised; background tools cannot defer a running task.'
    if isinstance(error, ToolRetryError):
        content = error.tool_retry.content
        return content if isinstance(content, str) and content else type(error).__name__
    if isinstance(error, ToolFailedError):
        content = error.tool_failed.content
        return content if isinstance(content, str) and content else type(error).__name__
    return str(error) or type(error).__name__


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

    Warning:
        Tools selected for background execution must not call `ctx.enqueue()`. This
        capability lets the agent continue while a synchronous tool runs in a worker
        thread, so its enqueue can race the pending-message drain and be lost. Return
        the tool's value instead; this capability enqueues the follow-up message.

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

    Durable execution is rejected because in-process tasks cannot survive workflow
    replay or worker restart.
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
    _completed: list[str] = field(default_factory=list[str], init=False, repr=False)

    def get_instructions(self) -> AgentInstructions[AgentDepsT] | None:
        return _INSTRUCTIONS

    def for_agent(self, agent: AbstractAgent[AgentDepsT, object]) -> AbstractCapability[AgentDepsT]:
        siblings: list[AbstractCapability[AgentDepsT]] = []
        agent.root_capability.apply(siblings.append)
        if any(isinstance(sibling, BaseDurabilityCapability) for sibling in siblings):
            raise UserError(
                '`BackgroundTools` does not support durable execution because in-process tasks cannot survive '
                'workflow replay or worker restart.'
            )
        return self

    async def for_run(self, ctx: RunContext[AgentDepsT]) -> BackgroundTools[AgentDepsT]:
        # The runtime fields are init=False, so concurrent runs do not share state.
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
                message = f"Background tool '{tool_name}' (task {task_id}) completed.\nResult: {result}"
            except Exception as e:
                error = _format_background_error(e)
                message = f"Background tool '{tool_name}' (task {task_id}) failed: {error}"
            self._completed.append(message)

        task = asyncio.create_task(_run())
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
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
        while isinstance(result, End) and self._tasks and not self._completed and not ctx.pending_messages:
            await asyncio.wait(self._tasks, return_when=asyncio.FIRST_COMPLETED)

        for message in self._completed:
            ctx.enqueue(message)
        self._completed.clear()
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
