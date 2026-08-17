"""Tests for the `BackgroundTools` capability."""

from __future__ import annotations

import asyncio

import pytest
from pydantic_ai import Agent
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

    async def test_custom_metadata_key_selector(self) -> None:
        agent = Agent(_model_calling('custom'), capabilities=[BackgroundTools(tools={'async': True})])

        @agent.tool_plain(metadata={'async': True})
        async def custom() -> str:  # pyright: ignore[reportUnusedFunction]
            await asyncio.sleep(0.01)
            return 'value'

        result = await agent.run('go')

        assert result.output == 'done'
        assert _ack_seen(result.all_messages())
