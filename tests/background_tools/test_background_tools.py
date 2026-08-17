"""Tests for the `BackgroundTools` capability."""

from __future__ import annotations

import asyncio

import pytest
from pydantic_ai import Agent, RunContext
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.tools import DeferredToolRequests

from pydantic_ai_harness import BackgroundTools

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


def _ack_seen(messages: list[ModelMessage]) -> bool:
    """True if any tool return in the history is a background-execution ack."""
    return any(
        isinstance(part, ToolReturnPart) and 'running in background' in str(part.content)
        for msg in messages
        if isinstance(msg, ModelRequest)
        for part in msg.parts
    )


def _follow_up_seen(messages: list[ModelMessage], needle: str) -> bool:
    """True if any drained user prompt in the history contains `needle`."""
    return any(
        isinstance(part, UserPromptPart) and isinstance(part.content, str) and needle in part.content
        for msg in messages
        if isinstance(msg, ModelRequest)
        for part in msg.parts
    )


def _model_calling(tool_name: str, args: str = '{}') -> FunctionModel:
    """A model that calls `tool_name` once, idles while the task runs, and answers `done` on the follow-up."""

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if _follow_up_seen(messages, f"Background tool '{tool_name}'"):
            return ModelResponse(parts=[TextPart(content='done')])
        if _ack_seen(messages):
            return ModelResponse(parts=[TextPart(content='waiting')])
        return ModelResponse(parts=[ToolCallPart(tool_name=tool_name, args=args)])

    return FunctionModel(model_fn)


class TestBackgroundTools:
    """The metadata-default selector path: spawn, ack, deliver, error, cancel."""

    async def test_metadata_marked_tool_acks_then_delivers_result_as_follow_up(self) -> None:
        agent = Agent(
            _model_calling('slow_research', args='{"query": "topic"}'),
            capabilities=[BackgroundTools()],
        )

        @agent.tool_plain(metadata={'background': True})
        async def slow_research(query: str) -> str:  # pyright: ignore[reportUnusedFunction]
            await asyncio.sleep(0.01)
            return f'researched {query}'

        result = await agent.run('go')

        assert result.output == 'done'
        assert _ack_seen(result.all_messages())
        assert _follow_up_seen(result.all_messages(), 'completed.\nResult: researched topic')

    async def test_failure_delivered_as_follow_up(self) -> None:
        agent = Agent(_model_calling('broken'), capabilities=[BackgroundTools()])

        @agent.tool_plain(metadata={'background': True})
        async def broken() -> str:  # pyright: ignore[reportUnusedFunction]
            await asyncio.sleep(0.01)
            raise RuntimeError('boom')

        result = await agent.run('go')

        assert result.output == 'done'
        assert _follow_up_seen(result.all_messages(), 'failed: boom')

    async def test_unmarked_tool_runs_normally(self) -> None:
        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            returned = any(
                isinstance(part, ToolReturnPart) and part.content == 'sync result'
                for msg in messages
                if isinstance(msg, ModelRequest)
                for part in msg.parts
            )
            if returned:
                return ModelResponse(parts=[TextPart(content='done')])
            return ModelResponse(parts=[ToolCallPart(tool_name='plain', args='{}')])

        agent = Agent(FunctionModel(model_fn), capabilities=[BackgroundTools()])

        @agent.tool_plain
        def plain() -> str:  # pyright: ignore[reportUnusedFunction]
            return 'sync result'

        result = await agent.run('go')

        assert result.output == 'done'
        assert not _ack_seen(result.all_messages())
        assert not _follow_up_seen(result.all_messages(), 'Background tool')

    async def test_instructions_tell_model_not_to_block(self) -> None:
        seen: list[str | None] = []

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            seen.append(info.instructions)
            return ModelResponse(parts=[TextPart(content='done')])

        agent = Agent(FunctionModel(model_fn), capabilities=[BackgroundTools()])
        await agent.run('go')

        assert 'do not block waiting for the result' in (seen[0] or '')

    async def test_concurrent_runs_do_not_share_tasks(self) -> None:
        release = {'first': asyncio.Event(), 'second': asyncio.Event()}
        started = {'first': asyncio.Event(), 'second': asyncio.Event()}

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            first_part = messages[0].parts[-1] if isinstance(messages[0], ModelRequest) else None
            run_name = first_part.content if isinstance(first_part, UserPromptPart) else ''
            if _follow_up_seen(messages, 'completed'):
                return ModelResponse(parts=[TextPart(content='done')])
            if _ack_seen(messages):
                return ModelResponse(parts=[TextPart(content='waiting')])
            return ModelResponse(parts=[ToolCallPart(tool_name='waiter', args=f'{{"name": "{run_name}"}}')])

        agent = Agent(FunctionModel(model_fn), capabilities=[BackgroundTools()])

        @agent.tool_plain(metadata={'background': True})
        async def waiter(name: str) -> str:  # pyright: ignore[reportUnusedFunction]
            started[name].set()
            await release[name].wait()
            return name

        async def run_and_check(name: str) -> None:
            result = await agent.run(name)
            assert result.output == 'done'
            assert _follow_up_seen(result.all_messages(), f'completed.\nResult: {name}')

        first = asyncio.ensure_future(run_and_check('first'))
        second = asyncio.ensure_future(run_and_check('second'))
        await started['first'].wait()
        await started['second'].wait()

        # Finish the first run completely while the second's task is still live;
        # shared state would let the first run's cleanup cancel the second's task.
        release['first'].set()
        await first
        release['second'].set()
        await second

    async def test_deferred_tool_pause_is_not_held_behind_background_tasks(self) -> None:
        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            if len(messages) == 1:
                return ModelResponse(
                    parts=[
                        ToolCallPart(tool_name='slow', args='{}'),
                        ToolCallPart(tool_name='needs_approval', args='{}'),
                    ]
                )
            # The pause must end the run without another model turn.
            return ModelResponse(parts=[TextPart(content='unreachable')])  # pragma: no cover

        agent = Agent(
            FunctionModel(model_fn),
            output_type=[str, DeferredToolRequests],
            capabilities=[BackgroundTools()],
        )

        @agent.tool_plain(metadata={'background': True})
        async def slow() -> str:  # pyright: ignore[reportUnusedFunction]
            await asyncio.sleep(60)
            return 'never'  # pragma: no cover -- task is cancelled when the run pauses

        @agent.tool_plain(requires_approval=True)
        def needs_approval() -> str:  # pyright: ignore[reportUnusedFunction]
            return 'approved'  # pragma: no cover -- never approved in this test

        result = await asyncio.wait_for(agent.run('go'), timeout=1)

        assert isinstance(result.output, DeferredToolRequests)

    async def test_queued_message_at_end_of_run_is_not_delayed_by_live_tasks(self) -> None:
        release = asyncio.Event()

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            if _follow_up_seen(messages, 'completed'):
                return ModelResponse(parts=[TextPart(content='done')])
            if _follow_up_seen(messages, 'note delivered'):
                # The note reached us without waiting on the live task; let it finish now.
                release.set()
                return ModelResponse(parts=[TextPart(content='waiting')])
            if _ack_seen(messages):
                # Enqueue a message and end the run in the same step, while `slow_bg` is live.
                return ModelResponse(parts=[ToolCallPart(tool_name='note', args='{}'), TextPart(content='all done')])
            return ModelResponse(parts=[ToolCallPart(tool_name='slow_bg', args='{}')])

        agent = Agent(FunctionModel(model_fn), capabilities=[BackgroundTools()])

        @agent.tool_plain(metadata={'background': True})
        async def slow_bg() -> str:  # pyright: ignore[reportUnusedFunction]
            await release.wait()
            return 'bg value'

        @agent.tool
        def note(ctx: RunContext[object]) -> str:  # pyright: ignore[reportUnusedFunction]
            ctx.enqueue('note delivered')
            return 'noted'

        result = await asyncio.wait_for(agent.run('go'), timeout=5)

        assert result.output == 'done'
        assert _follow_up_seen(result.all_messages(), 'completed.\nResult: bg value')

    async def test_run_abort_cancels_live_tasks(self) -> None:
        cancel_seen = asyncio.Event()
        agent = Agent(_model_calling('slow'), capabilities=[BackgroundTools()])

        @agent.tool_plain(metadata={'background': True})
        async def slow() -> str:  # pyright: ignore[reportUnusedFunction]
            try:
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                cancel_seen.set()
                raise
            return 'never'  # pragma: no cover -- task is cancelled before completing

        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(agent.run('go'), timeout=0.5)

        await asyncio.wait_for(cancel_seen.wait(), timeout=1)

    async def test_name_list_selector(self) -> None:
        agent = Agent(_model_calling('by_name'), capabilities=[BackgroundTools(tools=['by_name'])])

        @agent.tool_plain
        async def by_name() -> str:  # pyright: ignore[reportUnusedFunction]
            await asyncio.sleep(0.01)
            return 'value'

        result = await agent.run('go')

        assert result.output == 'done'
        assert _ack_seen(result.all_messages())

    async def test_constructs_from_agent_spec(self) -> None:
        agent = Agent.from_spec(
            {'model': 'test', 'capabilities': [{'BackgroundTools': {}}]},
            custom_capability_types=[BackgroundTools],
        )

        @agent.tool_plain(metadata={'background': True})
        async def slow() -> str:  # pyright: ignore[reportUnusedFunction]
            return 'value'

        result = await agent.run('go')

        assert _ack_seen(result.all_messages())
        assert _follow_up_seen(result.all_messages(), 'completed')

    async def test_custom_metadata_key_selector(self) -> None:
        agent = Agent(_model_calling('custom'), capabilities=[BackgroundTools(tools={'async': True})])

        @agent.tool_plain(metadata={'async': True})
        async def custom() -> str:  # pyright: ignore[reportUnusedFunction]
            await asyncio.sleep(0.01)
            return 'value'

        result = await agent.run('go')

        assert result.output == 'done'
        assert _ack_seen(result.all_messages())
