"""Tests for `CodeMode(eager=True)`: streamed `run_code` statements execute as they close.

Behavioral, through `Agent(..., capabilities=[CodeMode(eager=True)])` with a streaming
`FunctionModel`. The central proof is a stream that refuses to finish until the first
statement's tool call has executed: only mid-stream execution can unblock it.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
from collections.abc import AsyncIterator, Sequence

import pytest
from pydantic_ai import Agent, Tool
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.exceptions import ModelRetry, UserError
from pydantic_ai.messages import (
    AgentStreamEvent,
    ModelMessage,
    ModelResponse,
    PartDeltaEvent,
    PartStartEvent,
    ToolCallPart,
    ToolCallPartDelta,
    ToolReturnPart,
)
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel
from pydantic_ai.tool_manager import ToolManager
from pydantic_ai.toolsets.function import FunctionToolset

from pydantic_ai_harness.code_mode import CodeMode, SpeculativeCodeUpdateEvent
from pydantic_ai_harness.code_mode._toolset import CodeModeToolset  # pyright: ignore[reportPrivateUsage]

from .test_speculation import build_run_context

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    """Eager pumps schedule with `asyncio.ensure_future`; the trio backend does not apply."""
    return 'asyncio'


async def _plain_stream(events: Sequence[AgentStreamEvent]) -> AsyncIterator[AgentStreamEvent]:
    for event in events:
        yield event


def _prior_run_code_calls(messages: list[ModelMessage]) -> int:
    return sum(1 for m in messages if isinstance(m, ModelResponse) for p in m.parts if isinstance(p, ToolCallPart))


def _stream_json_args(code: str, chunk_size: int = 16) -> list[str]:
    args = json.dumps({'code': code})
    return [args[offset : offset + chunk_size] for offset in range(0, len(args), chunk_size)]


def _run_code_return_content(messages: list[ModelMessage]) -> object:
    contents = [
        p.content
        for m in messages
        for p in getattr(m, 'parts', [])
        if isinstance(p, ToolReturnPart) and p.tool_name == 'run_code'
    ]
    assert contents, 'no run_code ToolReturnPart in history'
    return contents[-1]


class TestEagerExecution:
    async def test_eager_and_speculate_are_mutually_exclusive(self):
        with pytest.raises(UserError, match='mutually exclusive'):
            CodeMode[None](eager=True, speculate=['search'])

    async def test_statements_execute_while_the_model_is_still_streaming(self):
        """The stream stalls until the first statement's tool call has run.

        Deadlock-unless-eager: the model refuses to emit the final chunks until `search`
        has executed, which only eager mid-stream feeding can achieve.
        """
        first_call = asyncio.Event()
        calls: list[str] = []

        async def search(query: str) -> str:
            """Return a canned result."""
            calls.append(query)
            first_call.set()
            return f'result:{query}'

        code = 'a = await search(query="alpha")\nb = await search(query="beta")\nprint(a)\nprint(b)\n"ok"'

        async def stream_code(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[DeltaToolCalls | str]:
            if _prior_run_code_calls(messages):
                yield 'done'
                return
            chunks = _stream_json_args(code)
            yield {1: DeltaToolCall(name='run_code')}
            for chunk in chunks[:-1]:
                yield {1: DeltaToolCall(json_args=chunk)}
                await asyncio.sleep(0)
            await asyncio.wait_for(first_call.wait(), timeout=5)
            yield {1: DeltaToolCall(json_args=chunks[-1])}

        capability = CodeMode[None](eager=True)
        agent: Agent[None, str] = Agent(
            FunctionModel(stream_function=stream_code),
            deps_type=type(None),
            capabilities=[capability],
        )
        agent.tool_plain(search)

        result = await agent.run('go')

        assert result.output == 'done'
        assert calls == ['alpha', 'beta']
        content = _run_code_return_content(result.all_messages())
        assert content == {'output': 'result:alpha\nresult:beta\n', 'result': 'ok'}

    async def test_failed_statement_surfaces_and_earlier_state_persists(self):
        """A statement that fails mid-stream stops the pump; the retry sees prior assignments.

        Identical to the non-eager failure contract: assignments before the failing line
        persist, the error arrives as the `run_code` result, and the tool that already ran
        is not re-executed by the retry.
        """
        calls: list[str] = []

        def search(query: str) -> str:
            """Return a canned result."""
            calls.append(query)
            return f'result:{query}'

        bad = 'a = await search(query="alpha")\nboom = 1 // 0\nprint(a)\nprint(boom)\n"x"'
        good = 'print(a)\n"recovered"'

        async def stream_attempts(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[DeltaToolCalls | str]:
            prior = _prior_run_code_calls(messages)
            if prior >= 2:
                yield 'done'
                return
            yield {1: DeltaToolCall(name='run_code')}
            for chunk in _stream_json_args(bad if prior == 0 else good):
                yield {1: DeltaToolCall(json_args=chunk)}
                await asyncio.sleep(0)

        capability = CodeMode[None](eager=True)
        agent: Agent[None, str] = Agent(
            FunctionModel(stream_function=stream_attempts),
            deps_type=type(None),
            capabilities=[capability],
        )
        agent.tool_plain(search)

        result = await agent.run('go')

        assert result.output == 'done'
        assert calls == ['alpha']
        content = _run_code_return_content(result.all_messages())
        assert content == {'output': 'result:alpha\n', 'result': 'recovered'}

    async def test_restart_discards_the_pumped_prefix(self):
        """`restart: true` resets the session; the streamed prefix's state is gone."""
        calls: list[str] = []

        def search(query: str) -> str:
            """Return a canned result."""
            calls.append(query)
            return f'result:{query}'

        code = 'a = await search(query="alpha")\nprint(a)\n"ok"'

        async def stream_restart(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[DeltaToolCalls | str]:
            if _prior_run_code_calls(messages):
                yield 'done'
                return
            args = json.dumps({'code': code, 'restart': True})
            yield {1: DeltaToolCall(name='run_code')}
            for offset in range(0, len(args), 16):
                yield {1: DeltaToolCall(json_args=args[offset : offset + 16])}
                await asyncio.sleep(0)

        capability = CodeMode[None](eager=True)
        agent: Agent[None, str] = Agent(
            FunctionModel(stream_function=stream_restart),
            deps_type=type(None),
            capabilities=[capability],
        )
        agent.tool_plain(search)

        result = await agent.run('go')

        assert result.output == 'done'
        # The pumped prefix ran once, then restart re-ran the full snippet from scratch.
        assert calls == ['alpha', 'alpha']

    async def test_diverged_code_resets_the_session(self):
        """Execution with a different prefix than the pump ran raises a retry and resets."""
        ctx = build_run_context(None)
        ctx.tool_manager = None
        capability = CodeMode[None](eager=True)
        run_capability = await capability.for_run(ctx)
        toolset = run_capability.get_wrapper_toolset(FunctionToolset[None](tools=[]))
        assert isinstance(toolset, CodeModeToolset)
        assert toolset.eager is not None
        tools = await toolset.get_tools(ctx)

        exec_ctx = dataclasses.replace(ctx, tool_call_id='c1', tool_name='run_code')
        exec_ctx.tool_manager = await ToolManager(toolset=toolset).for_run_step(exec_ctx)
        stream_ctx = dataclasses.replace(ctx)
        stream_ctx.tool_manager = exec_ctx.tool_manager
        async with toolset:
            events = [
                PartStartEvent(
                    index=0,
                    part=ToolCallPart(tool_name='run_code', args={'code': 'x = 1\ny = 2\nprint(x)'}, tool_call_id='c1'),
                ),
            ]
            async for _ in run_capability.wrap_run_event_stream(stream_ctx, stream=_plain_stream(events)):
                pass
            with pytest.raises(ModelRetry, match='no longer matches'):
                await toolset.call_tool('run_code', {'code': 'z = 9\nw = 8\nprint(z)'}, exec_ctx, tools['run_code'])
            # The diverged part was consumed; nothing is left to adopt.
            assert toolset.eager.take('cX', 'anything') is None

    async def test_take_tolerates_rewritten_ids_and_rejects_foreign_code(self):
        """A re-keyed execution adopts the part whose executed prefix matches its code."""
        ctx = build_run_context(None)
        ctx.tool_manager = None
        capability = CodeMode[None](eager=True)
        run_capability = await capability.for_run(ctx)
        toolset = run_capability.get_wrapper_toolset(FunctionToolset[None](tools=[]))
        assert isinstance(toolset, CodeModeToolset)
        eager = toolset.eager
        assert eager is not None
        await toolset.get_tools(ctx)

        stream_ctx = dataclasses.replace(ctx, tool_call_id='c1', tool_name='run_code')
        stream_ctx.tool_manager = await ToolManager(toolset=toolset).for_run_step(stream_ctx)
        async with toolset:
            events = [
                PartStartEvent(
                    index=0,
                    part=ToolCallPart(tool_name='run_code', args={'code': 'x = 1\ny = 2\nprint(x)'}, tool_call_id='c1'),
                ),
            ]
            async for _ in run_capability.wrap_run_event_stream(stream_ctx, stream=_plain_stream(events)):
                pass
            taken = eager.take('provider-rewrote-this', 'x = 1\ny = 2\nprint(x)')
            assert taken is not None and taken.tool_call_id == 'c1'
            await eager.drain(taken)

    async def test_observe_edges_and_idle_close(self):
        """Non-run_code parts, unknown ids, dict merges, and broken partial JSON are inert."""
        ctx = build_run_context(None)
        capability = CodeMode[None](eager=True)
        run_capability = await capability.for_run(ctx)
        toolset = run_capability.get_wrapper_toolset(FunctionToolset[None](tools=[]))
        assert isinstance(toolset, CodeModeToolset)
        eager = toolset.eager
        assert eager is not None

        # Other tools are ignored outright.
        assert (
            await eager.observe(
                PartStartEvent(index=0, part=ToolCallPart(tool_name='other', args={}, tool_call_id='o1')), ctx
            )
            == []
        )
        # String args at part start participate; a single statement stays provisional (no feed).
        events = await eager.observe(
            PartStartEvent(
                index=1, part=ToolCallPart(tool_name='run_code', args='{"code": "x = 1"}', tool_call_id='c1')
            ),
            ctx,
        )
        assert [e.closed_statements for e in events if isinstance(e, SpeculativeCodeUpdateEvent)] == [0]
        # A delta for an id the watcher never saw is dropped.
        assert (
            await eager.observe(
                PartDeltaEvent(index=9, delta=ToolCallPartDelta(args_delta='x', tool_call_id='zzz')), ctx
            )
            == []
        )
        # Dict deltas merge without disturbing the code; the newline gate reports and returns.
        second = await eager.observe(
            PartStartEvent(index=2, part=ToolCallPart(tool_name='run_code', args={'code': 'q = 7'}, tool_call_id='c2')),
            ctx,
        )
        assert len(second) == 1
        gated = await eager.observe(
            PartDeltaEvent(index=2, delta=ToolCallPartDelta(args_delta={'restart': False}, tool_call_id='c2')), ctx
        )
        assert [e.closed_statements for e in gated if isinstance(e, SpeculativeCodeUpdateEvent)] == [0]
        # With two live parts, an id-less delta matches nothing.
        assert await eager.observe(PartDeltaEvent(index=3, delta=ToolCallPartDelta(args_delta='x')), ctx) == []
        # Broken partial JSON has no code yet.
        assert (
            await eager.observe(
                PartStartEvent(index=4, part=ToolCallPart(tool_name='run_code', args='{"cod', tool_call_id='c3')), ctx
            )
            == []
        )
        # A `None` args delta changes nothing but still reports the current code.
        untouched = await eager.observe(
            PartDeltaEvent(index=2, delta=ToolCallPartDelta(args_delta=None, tool_call_id='c2')), ctx
        )
        assert [e.closed_statements for e in untouched if isinstance(e, SpeculativeCodeUpdateEvent)] == [0]
        # Clear the empty-prefix parts, then verify a foreign execution matches nothing
        # against a part whose executed prefix is real.
        assert eager.take('c1', 'x = 1') is not None
        assert eager.take('c2', 'q = 7') is not None
        assert eager.take('c3', '') is not None
        await eager.observe(
            PartStartEvent(
                index=5,
                part=ToolCallPart(tool_name='run_code', args={'code': 'm = 1\nn = 2\nprint(m)'}, tool_call_id='c4'),
            ),
            ctx,
        )
        assert eager.take('cZ', 'entirely different program') is None
        taken = eager.take('c4', 'm = 1\nn = 2\nprint(m)')
        assert taken is not None
        await eager.drain(taken)
        # Close skips a part that never grew a pump.
        await eager.observe(
            PartStartEvent(index=6, part=ToolCallPart(tool_name='run_code', args={'code': 'k = 1'}, tool_call_id='c5')),
            ctx,
        )
        await eager.close()

    async def test_inactive_under_temporal_durability(self):
        """Under Temporal, nothing feeds early; execution runs the snippet whole."""

        class TemporalDurability(AbstractCapability[None]):
            in_durable_context = True

        TemporalDurability.__module__ = 'pydantic_ai.durable_exec.temporal'
        calls: list[str] = []

        def search(query: str) -> str:
            """Return a canned result."""
            calls.append(query)
            return f'result:{query}'

        code = 'a = await search(query="alpha")\nprint(a)\n"ok"'

        async def stream_code(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[DeltaToolCalls | str]:
            if _prior_run_code_calls(messages):
                yield 'done'
                return
            yield {1: DeltaToolCall(name='run_code')}
            for chunk in _stream_json_args(code):
                yield {1: DeltaToolCall(json_args=chunk)}
                await asyncio.sleep(0)

        capability = CodeMode[None](eager=True)
        agent: Agent[None, str] = Agent(
            FunctionModel(stream_function=stream_code),
            deps_type=type(None),
            capabilities=[capability, TemporalDurability()],
        )
        agent.tool_plain(search)

        result = await agent.run('go')

        assert result.output == 'done'
        assert calls == ['alpha']

    async def test_on_run_error_closes_eager_state(self):
        """The error hook tears the pump state down and re-raises."""
        ctx = build_run_context(None)
        capability = CodeMode[None](eager=True)
        run_capability = await capability.for_run(ctx)
        assert run_capability is not capability
        with pytest.raises(RuntimeError, match='boom'):
            await run_capability.on_run_error(ctx, error=RuntimeError('boom'))

    async def test_close_cancels_a_blocked_pump(self):
        """Run-end close cancels an in-flight fragment feed without hanging."""
        release = asyncio.Event()
        started = asyncio.Event()

        async def slow(query: str) -> str:
            """Wait until released."""
            started.set()
            await release.wait()
            return query  # pragma: no cover - cancelled before completion

        ctx = build_run_context(None)
        ctx.tool_manager = None
        capability = CodeMode[None](eager=True)
        run_capability = await capability.for_run(ctx)
        toolset = run_capability.get_wrapper_toolset(FunctionToolset[None](tools=[Tool(slow)]))
        assert isinstance(toolset, CodeModeToolset)
        assert toolset.eager is not None
        await toolset.get_tools(ctx)

        stream_ctx = dataclasses.replace(ctx, tool_call_id='c1', tool_name='run_code')
        stream_ctx.tool_manager = await ToolManager(toolset=toolset).for_run_step(stream_ctx)
        async with toolset:
            events = [
                PartStartEvent(
                    index=0,
                    part=ToolCallPart(
                        tool_name='run_code',
                        args={'code': 'a = await slow(query="x")\nb = 1\nprint(b)'},
                        tool_call_id='c1',
                    ),
                ),
            ]
            async for _ in run_capability.wrap_run_event_stream(stream_ctx, stream=_plain_stream(events)):
                pass
            await asyncio.wait_for(started.wait(), timeout=5)
            await toolset.eager.close()
