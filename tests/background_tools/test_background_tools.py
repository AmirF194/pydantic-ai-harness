"""Tests for the `BackgroundTools` capability."""

from __future__ import annotations

import asyncio
import re
import threading
from collections.abc import AsyncIterator, Callable

import anyio
import pytest
from pydantic_ai import Agent, CancellationToken, RunCancelled
from pydantic_ai.exceptions import ApprovalRequired, CallDeferred, ModelRetry, ToolFailed
from pydantic_ai.messages import (
    BinaryContent,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturn,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, FunctionModel
from pydantic_ai.tools import DeferredToolRequests, RunContext

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
        isinstance(part, UserPromptPart)
        and any(
            needle in item
            for item in ([part.content] if isinstance(part.content, str) else part.content)
            if isinstance(item, str)
        )
        for msg in messages
        if isinstance(msg, ModelRequest)
        for part in msg.parts
    )


def _model_calling(tool_name: str, args: str = '{}', ack_callback: Callable[[], None] | None = None) -> FunctionModel:
    """A model that calls `tool_name` once, idles while the task runs, and answers `done` on the follow-up."""

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if _follow_up_seen(messages, f"Background tool '{tool_name}'"):
            return ModelResponse(parts=[TextPart(content='done')])
        if _ack_seen(messages):
            if ack_callback is not None:
                ack_callback()
            return ModelResponse(parts=[TextPart(content='waiting')])
        return ModelResponse(parts=[ToolCallPart(tool_name=tool_name, args=args)])

    return FunctionModel(model_fn)


class TestBackgroundTools:
    """The metadata-default selector path: spawn, ack, deliver, error, cancel."""

    async def test_metadata_marked_tool_acks_then_delivers_result_as_follow_up(self) -> None:
        release = asyncio.Event()
        ack_seen = asyncio.Event()

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            if _follow_up_seen(messages, "Background tool 'slow_research'"):
                return ModelResponse(parts=[TextPart(content='done')])
            if _ack_seen(messages):
                ack_seen.set()
                return ModelResponse(parts=[TextPart(content='waiting')])
            return ModelResponse(parts=[ToolCallPart(tool_name='slow_research', args='{"query": "topic"}')])

        agent = Agent(
            FunctionModel(model_fn),
            capabilities=[BackgroundTools()],
        )

        @agent.tool_plain(metadata={'background': True})
        async def slow_research(query: str) -> str:  # pyright: ignore[reportUnusedFunction]
            await release.wait()
            return f'researched {query}'

        run = asyncio.ensure_future(agent.run('go'))
        await asyncio.wait_for(ack_seen.wait(), timeout=5)
        assert not run.done()
        release.set()
        result = await asyncio.wait_for(run, timeout=5)

        assert result.output == 'done'
        parts = [part for message in result.all_messages() for part in message.parts]
        ack = next(part.content for part in parts if isinstance(part, ToolReturnPart))
        assert isinstance(ack, str)
        match = re.search(r'\(task (?P<task_id>[^)]+)\)', ack)
        assert match is not None
        follow_up = next(
            part.content
            for part in parts
            if isinstance(part, UserPromptPart) and isinstance(part.content, str) and 'Background tool' in part.content
        )
        assert follow_up == (
            f"Background tool 'slow_research' (task {match['task_id']}) completed.\nResult: researched topic"
        )

    @pytest.mark.parametrize(
        ('error_factory', 'expected'),
        [
            (lambda: RuntimeError('private backend detail'), 'failed: RuntimeError'),
            (lambda: ModelRetry('use different arguments'), 'failed: use different arguments'),
            (lambda: ToolFailed('service unavailable'), 'failed: service unavailable'),
            (
                lambda: CallDeferred(metadata={'private_job_id': 'secret'}),
                'failed: CallDeferred was raised; background tools cannot defer a running task.',
            ),
            (RuntimeError, 'failed: RuntimeError'),
        ],
        ids=['runtime', 'retry', 'tool-failed', 'deferred', 'empty-error'],
    )
    async def test_control_flow_and_empty_errors_have_readable_follow_ups(
        self, error_factory: Callable[[], Exception], expected: str
    ) -> None:
        release = asyncio.Event()
        agent = Agent(_model_calling('broken', ack_callback=release.set), capabilities=[BackgroundTools()])

        @agent.tool_plain(metadata={'background': True})
        async def broken() -> str:  # pyright: ignore[reportUnusedFunction]
            await release.wait()
            raise error_factory()

        result = await agent.run('go')

        assert result.output == 'done'
        assert _follow_up_seen(result.all_messages(), expected)
        assert not _follow_up_seen(result.all_messages(), 'private_job_id')
        assert not _follow_up_seen(result.all_messages(), 'secret')

    async def test_unexpected_error_is_logged_without_exposing_details_to_model(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        agent = Agent(_model_calling('broken'), capabilities=[BackgroundTools()])

        @agent.tool_plain(metadata={'background': True})
        async def broken() -> str:  # pyright: ignore[reportUnusedFunction]
            raise RuntimeError('private backend detail')

        result = await agent.run('go')

        assert _follow_up_seen(result.all_messages(), 'failed: RuntimeError')
        assert not _follow_up_seen(result.all_messages(), 'private backend detail')
        assert 'Background tool broken failed' in caplog.text
        assert 'private backend detail' in caplog.text

    async def test_run_stream_waits_for_live_task_then_drops_its_result(self) -> None:
        started = asyncio.Event()
        release = asyncio.Event()

        async def stream_model(
            messages: list[ModelMessage], info: AgentInfo
        ) -> AsyncIterator[str | dict[int, DeltaToolCall]]:
            if _ack_seen(messages):
                yield 'final answer'
            else:
                yield {0: DeltaToolCall(name='slow', json_args='{}')}

        agent = Agent(FunctionModel(stream_function=stream_model), capabilities=[BackgroundTools()])

        @agent.tool_plain(metadata={'background': True})
        async def slow() -> str:  # pyright: ignore[reportUnusedFunction]
            started.set()
            await release.wait()
            return 'late result'

        async with agent.run_stream('go') as stream_result:
            output = asyncio.ensure_future(stream_result.get_output())
            await asyncio.wait_for(started.wait(), timeout=1)
            await asyncio.sleep(0)
            assert not output.done()
            release.set()
            assert await asyncio.wait_for(output, timeout=1) == 'final answer'

        assert not _follow_up_seen(stream_result.all_messages(), 'late result')

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

    async def test_selected_sync_tool_acks_then_delivers_result_as_follow_up(self) -> None:
        agent = Agent(_model_calling('sync_bg'), capabilities=[BackgroundTools()])

        @agent.tool_plain(metadata={'background': True})
        def sync_bg() -> str:  # pyright: ignore[reportUnusedFunction]
            return 'sync result'

        result = await agent.run('go')

        assert result.output == 'done'
        assert _ack_seen(result.all_messages())
        assert _follow_up_seen(result.all_messages(), 'completed.\nResult: sync result')

    async def test_background_tool_uses_context_cancel_to_stop_the_run(self) -> None:
        agent = Agent(_model_calling('stop'), capabilities=[BackgroundTools()])

        @agent.tool(metadata={'background': True})
        async def stop(ctx: RunContext[object]) -> str:  # pyright: ignore[reportUnusedFunction]
            ctx.cancel()
            await asyncio.sleep(0)
            return 'discarded'  # pragma: no cover -- cancellation is delivered at the await

        with pytest.raises(RunCancelled):
            await agent.run('go')

    async def test_tool_return_preserves_model_content_without_exposing_metadata(self) -> None:
        image = BinaryContent(data=b'image bytes', media_type='image/png')
        agent = Agent(_model_calling('structured'), capabilities=[BackgroundTools()])

        @agent.tool_plain(metadata={'background': True})
        async def structured() -> ToolReturn[str]:  # pyright: ignore[reportUnusedFunction]
            return ToolReturn(
                return_value='public answer',
                content=['supporting image', image],
                metadata={'api_key': 'secret application metadata'},
                tools=['deferred_tool'],
            )

        result = await agent.run('go')

        follow_up = next(
            part
            for message in result.all_messages()
            if isinstance(message, ModelRequest)
            for part in message.parts
            if isinstance(part, UserPromptPart) and 'Background tool' in str(part.content)
        )
        assert isinstance(follow_up.content, list)
        assert len(follow_up.content) == 3
        assert isinstance(follow_up.content[0], str)
        assert follow_up.content[0].startswith("Background tool 'structured' (task ")
        assert follow_up.content[0].endswith(') completed.\nResult: public answer')
        assert follow_up.content[1:] == ['supporting image', image]
        assert 'secret application metadata' not in str(follow_up.content)
        assert 'deferred_tool' not in str(follow_up.content)

    async def test_structured_return_value_uses_core_model_serialization(self) -> None:
        agent = Agent(_model_calling('structured'), capabilities=[BackgroundTools()])

        @agent.tool_plain(metadata={'background': True})
        async def structured() -> list[int]:  # pyright: ignore[reportUnusedFunction]
            return [1, 2]

        result = await agent.run('go')

        assert _follow_up_seen(result.all_messages(), 'completed.\nResult: [1,2]')

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
        await asyncio.wait_for(started['first'].wait(), timeout=5)
        await asyncio.wait_for(started['second'].wait(), timeout=5)

        # Finish the first run completely while the second's task is still live;
        # shared state would let the first run's cleanup cancel the second's task.
        release['first'].set()
        await asyncio.wait_for(first, timeout=5)
        release['second'].set()
        await asyncio.wait_for(second, timeout=5)

    async def test_deferred_tool_pause_is_not_held_behind_background_tasks(self) -> None:
        cancel_seen = asyncio.Event()

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
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancel_seen.set()
                raise
            return 'never'  # pragma: no cover -- task is cancelled when the run pauses

        @agent.tool_plain(requires_approval=True)
        def needs_approval() -> str:  # pyright: ignore[reportUnusedFunction]
            return 'approved'  # pragma: no cover -- never approved in this test

        result = await asyncio.wait_for(agent.run('go'), timeout=1)

        assert isinstance(result.output, DeferredToolRequests)
        await asyncio.wait_for(cancel_seen.wait(), timeout=1)

    async def test_deferred_tool_pause_wins_over_a_completed_background_result(self) -> None:
        fast_done = asyncio.Event()
        model_calls = 0

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            nonlocal model_calls
            model_calls += 1
            return ModelResponse(
                parts=[ToolCallPart(tool_name='fast', args='{}'), ToolCallPart(tool_name='needs_approval', args='{}')]
            )

        agent = Agent(
            FunctionModel(model_fn),
            output_type=[str, DeferredToolRequests],
            capabilities=[BackgroundTools()],
        )

        @agent.tool_plain(metadata={'background': True})
        async def fast() -> str:  # pyright: ignore[reportUnusedFunction]
            fast_done.set()
            return 'finished'

        @agent.tool_plain
        async def needs_approval() -> str:  # pyright: ignore[reportUnusedFunction]
            await fast_done.wait()
            raise ApprovalRequired

        result = await agent.run('go')

        assert isinstance(result.output, DeferredToolRequests)
        assert result.output.approvals[0].tool_name == 'needs_approval'
        assert fast_done.is_set()
        assert model_calls == 1

    async def test_queued_result_is_not_delayed_by_an_unrelated_live_task(self) -> None:
        release = {'fast_bg': asyncio.Event(), 'slow_bg': asyncio.Event()}

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            if _follow_up_seen(messages, 'fast value') and _follow_up_seen(messages, 'slow value'):
                return ModelResponse(parts=[TextPart(content='done')])
            if _follow_up_seen(messages, 'fast value'):
                # The fast result arrived without waiting on `slow_bg`; let it finish now.
                release['slow_bg'].set()
                return ModelResponse(parts=[TextPart(content='waiting')])
            if _ack_seen(messages):
                release['fast_bg'].set()
                return ModelResponse(parts=[TextPart(content='waiting')])
            return ModelResponse(
                parts=[ToolCallPart(tool_name='fast_bg', args='{}'), ToolCallPart(tool_name='slow_bg', args='{}')]
            )

        agent = Agent(FunctionModel(model_fn), capabilities=[BackgroundTools()])

        @agent.tool_plain(metadata={'background': True})
        async def fast_bg() -> str:  # pyright: ignore[reportUnusedFunction]
            await release['fast_bg'].wait()
            return 'fast value'

        @agent.tool_plain(metadata={'background': True})
        async def slow_bg() -> str:  # pyright: ignore[reportUnusedFunction]
            await release['slow_bg'].wait()
            return 'slow value'

        result = await asyncio.wait_for(agent.run('go'), timeout=5)

        assert result.output == 'done'
        assert _follow_up_seen(result.all_messages(), 'completed.\nResult: fast value')
        assert _follow_up_seen(result.all_messages(), 'completed.\nResult: slow value')

    async def test_failed_background_tool_does_not_cancel_its_sibling(self) -> None:
        release_broken = asyncio.Event()
        release_slow = asyncio.Event()

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            if _follow_up_seen(messages, 'failed: RuntimeError') and _follow_up_seen(messages, 'slow value'):
                return ModelResponse(parts=[TextPart(content='done')])
            if _follow_up_seen(messages, 'failed: RuntimeError'):
                release_slow.set()
                return ModelResponse(parts=[TextPart(content='waiting')])
            if _ack_seen(messages):
                release_broken.set()
                return ModelResponse(parts=[TextPart(content='waiting')])
            return ModelResponse(
                parts=[ToolCallPart(tool_name='broken', args='{}'), ToolCallPart(tool_name='slow', args='{}')]
            )

        agent = Agent(FunctionModel(model_fn), capabilities=[BackgroundTools()])

        @agent.tool_plain(metadata={'background': True})
        async def broken() -> str:  # pyright: ignore[reportUnusedFunction]
            await release_broken.wait()
            raise RuntimeError('private detail')

        @agent.tool_plain(metadata={'background': True})
        async def slow() -> str:  # pyright: ignore[reportUnusedFunction]
            await release_slow.wait()
            return 'slow value'

        result = await asyncio.wait_for(agent.run('go'), timeout=5)

        assert result.output == 'done'
        assert _follow_up_seen(result.all_messages(), 'failed: RuntimeError')
        assert _follow_up_seen(result.all_messages(), 'completed.\nResult: slow value')

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

    async def test_run_abort_waits_for_background_task_cancellation_cleanup(self) -> None:
        started = asyncio.Event()
        cancellation_seen = asyncio.Event()
        release = asyncio.Event()
        finished = asyncio.Event()
        agent = Agent(_model_calling('stubborn'), capabilities=[BackgroundTools()])

        @agent.tool_plain(metadata={'background': True})
        async def stubborn() -> str:  # pyright: ignore[reportUnusedFunction]
            started.set()
            try:
                await asyncio.Event().wait()
                raise AssertionError('unreachable')  # pragma: no cover
            except asyncio.CancelledError:
                cancellation_seen.set()
                with anyio.CancelScope(shield=True):
                    await release.wait()
                finished.set()
                raise

        run = asyncio.ensure_future(agent.run('go'))
        await asyncio.wait_for(started.wait(), timeout=1)
        run.cancel()

        try:
            await asyncio.wait_for(cancellation_seen.wait(), timeout=1)
            await asyncio.sleep(1.1)
            assert not run.done()
        finally:
            release.set()

        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(run, timeout=1)
        assert finished.is_set()

    async def test_run_abort_waits_for_default_sync_worker(self) -> None:
        started = threading.Event()
        release = threading.Event()
        finished = threading.Event()
        agent = Agent(_model_calling('blocking'), capabilities=[BackgroundTools()])

        @agent.tool_plain(metadata={'background': True})
        def blocking() -> str:  # pyright: ignore[reportUnusedFunction]
            started.set()
            assert release.wait(timeout=5)
            finished.set()
            return 'late result'

        run = asyncio.ensure_future(agent.run('go'))
        assert await asyncio.to_thread(started.wait, 1)
        run.cancel()

        try:
            await asyncio.sleep(1.1)
            assert not run.done()
        finally:
            release.set()

        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(run, timeout=1)
        assert finished.is_set()

    async def test_cancellation_token_cancels_live_tasks(self) -> None:
        cancel_seen = asyncio.Event()
        started = asyncio.Event()
        token = CancellationToken()
        agent = Agent(_model_calling('slow'), capabilities=[BackgroundTools()])

        @agent.tool_plain(metadata={'background': True})
        async def slow() -> str:  # pyright: ignore[reportUnusedFunction]
            started.set()
            try:
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                cancel_seen.set()
                raise
            return 'never'  # pragma: no cover -- task is cancelled before completing

        run = asyncio.ensure_future(agent.run('go', cancellation_token=token))
        await asyncio.wait_for(started.wait(), timeout=5)
        token.cancel()

        with pytest.raises(RunCancelled):
            await asyncio.wait_for(run, timeout=5)
        await asyncio.wait_for(cancel_seen.wait(), timeout=1)

    async def test_name_list_selector(self) -> None:
        release = asyncio.Event()
        agent = Agent(
            _model_calling('by_name', ack_callback=release.set), capabilities=[BackgroundTools(tools=['by_name'])]
        )

        @agent.tool_plain
        async def by_name() -> str:  # pyright: ignore[reportUnusedFunction]
            await release.wait()
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
