"""Tests for `CodeMode(speculate=...)`: early launch of sandbox calls from streamed `run_code` deltas.

Behavioral, through `Agent(..., capabilities=[CodeMode(...)])` with a streaming `FunctionModel`:
the model streams the `run_code` arguments in small chunks with an await point between them, so
speculative tasks get scheduled while "generation" is still in flight, and the tools themselves
record when and how often they ran.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field

import pytest
from pydantic_ai import Agent, RunContext, Tool
from pydantic_ai.capabilities import AbstractCapability, HandleDeferredToolCalls
from pydantic_ai.exceptions import ApprovalRequired
from pydantic_ai.messages import (
    AgentStreamEvent,
    ModelMessage,
    ModelResponse,
    PartDeltaEvent,
    PartStartEvent,
    TextPart,
    ToolCallPart,
    ToolCallPartDelta,
    ToolReturn,
)
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import DeferredToolRequests, DeferredToolResults, ToolDenied
from pydantic_ai.toolsets.function import FunctionToolset
from pydantic_ai.usage import RunUsage

from pydantic_ai_harness.code_mode import CodeMode, CodeModeToolset

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    """Run async tests on the asyncio backend (matching upstream pydantic-ai)."""
    return 'asyncio'


@dataclass
class ToolLog:
    """Observations the fake tools record for assertions."""

    calls: list[tuple[str, str]] = field(default_factory=list[tuple[str, str]])
    """(tool name, argument) per execution, in start order."""

    streaming_done: bool = False
    """Set by the model stream after its final chunk; tools read it at start."""

    started_during_stream: int = 0
    """How many tool executions began before the model finished streaming."""


def padded(code: str) -> str:
    """Append filler statements so earlier statements close with plenty of stream still to go.

    Against a real provider the gap between a closed statement and end-of-generation is the
    remaining decode; here the filler plays that role, giving launched tasks event-loop passes
    (one per chunk) to reach the tool body while the stream is still "generating".
    """
    filler = ''.join(f'\npad_{i} = {i}' for i in range(20))
    return f'{code}{filler}\n"ok"'


def build_agent(
    log: ToolLog,
    code: str,
    capability: CodeMode[None],
    chunk_size: int = 16,
    extra_capabilities: Sequence[AbstractCapability[None]] = (),
    raw_args: str | None = None,
) -> Agent[None, str]:
    """Agent whose model streams one `run_code` call for `code` in `chunk_size` pieces."""

    async def stream_code(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[DeltaToolCalls | str]:
        if len(messages) > 1:
            # Two chunks so the text part produces a delta for an index the watcher never registered.
            yield 'do'
            yield 'ne'
            return
        args = raw_args if raw_args is not None else json.dumps({'code': code})
        log.streaming_done = False
        # Name-only first delta: the started part carries no arguments yet.
        yield {1: DeltaToolCall(name='run_code')}
        for offset in range(0, len(args), chunk_size):
            yield {1: DeltaToolCall(json_args=args[offset : offset + chunk_size])}
            # Yield to the event loop so launched speculation tasks actually make progress
            # mid-stream, the way tool latency overlaps decode against a real provider.
            await asyncio.sleep(0)
        log.streaming_done = True

    def call_code(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        """Non-streamed twin of `stream_code`: used when speculation is off and runs stay non-streaming."""
        if len(messages) > 1:
            return ModelResponse(parts=[TextPart('done')])
        log.streaming_done = True
        return ModelResponse(parts=[ToolCallPart(tool_name='run_code', args={'code': code})])

    model = FunctionModel(call_code, stream_function=stream_code)
    agent: Agent[None, str] = Agent(model, deps_type=type(None), capabilities=[capability, *extra_capabilities])

    @agent.tool_plain
    async def search(query: str) -> str:
        """Return a canned result for `query`."""
        if not log.streaming_done:
            log.started_during_stream += 1
        log.calls.append(('search', query))
        await asyncio.sleep(0)
        return f'result:{query}'

    @agent.tool_plain
    async def side_effect(payload: str) -> str:
        """A tool deliberately kept off the allowlist."""
        if not log.streaming_done:  # pragma: no cover - would mean speculation launched it
            log.started_during_stream += 1
        log.calls.append(('side_effect', payload))
        return f'wrote:{payload}'

    @agent.tool_plain
    async def boom(payload: str) -> str:
        """A tool that always fails."""
        log.calls.append(('boom', payload))
        raise RuntimeError('kaboom')

    @agent.tool_plain
    async def approval_gate(value: int) -> str:
        """A tool that requires approval."""
        raise ApprovalRequired()

    @agent.tool_plain
    async def with_metadata(query: str) -> ToolReturn[str]:
        """A tool that returns a `ToolReturn` carrying metadata."""
        log.calls.append(('with_metadata', query))
        return ToolReturn(return_value=f'meta:{query}', metadata={'speculated': True})

    return agent


class TestSpeculation:
    async def test_literal_call_launches_during_stream_and_is_adopted(self):
        """A literal-args call streams past, launches early, and the snippet claims its result."""
        log = ToolLog()
        capability = CodeMode[None](speculate=['search'])
        agent = build_agent(log, padded('a = await search(query="alpha")\nprint(a)'), capability)

        result = await agent.run('go')

        assert result.output == 'done'
        assert log.calls == [('search', 'alpha')]
        assert log.started_during_stream == 1
        assert capability.speculation_stats.launched == 1
        assert capability.speculation_stats.adopted == 1
        assert capability.speculation_stats.evicted == 0

    async def test_disabled_by_default_runs_cold(self):
        """Without `speculate`, nothing launches early and behavior is unchanged."""
        log = ToolLog()
        capability = CodeMode[None]()
        agent = build_agent(log, padded('a = await search(query="alpha")\nprint(a)'), capability)

        result = await agent.run('go')

        assert result.output == 'done'
        assert log.calls == [('search', 'alpha')]
        assert log.started_during_stream == 0
        assert capability.speculation_stats.launched == 0

    async def test_non_allowlisted_tool_never_launches_early(self):
        """Only allowlisted tools speculate; others run cold even with literal arguments."""
        log = ToolLog()
        capability = CodeMode[None](speculate=['search'])
        code = padded('b = await side_effect(payload="x")\na = await search(query="alpha")\nprint(a, b)')
        agent = build_agent(log, code, capability)

        result = await agent.run('go')

        assert result.output == 'done'
        assert sorted(log.calls) == [('search', 'alpha'), ('side_effect', 'x')]
        assert log.started_during_stream == 1
        assert capability.speculation_stats.launched == 1
        assert capability.speculation_stats.adopted == 1

    async def test_variable_arguments_are_not_speculated(self):
        """A call whose argument is a variable has no literal identity to launch early."""
        log = ToolLog()
        capability = CodeMode[None](speculate=['search'])
        agent = build_agent(log, padded('q = "alpha"\na = await search(query=q)\nprint(a)'), capability)

        result = await agent.run('go')

        assert result.output == 'done'
        assert log.calls == [('search', 'alpha')]
        assert log.started_during_stream == 0
        assert capability.speculation_stats.launched == 0

    async def test_unclaimed_branch_launch_is_evicted(self):
        """A literal call in an untaken branch launches, is never claimed, and gets cancelled."""
        log = ToolLog()
        capability = CodeMode[None](speculate=['search'])
        code = padded('if False:\n    a = await search(query="never")\nb = await search(query="alpha")\nprint(b)')
        agent = build_agent(log, code, capability)

        result = await agent.run('go')

        assert result.output == 'done'
        # The wrong-branch launch may or may not reach the tool body before eviction, but the
        # snippet's own dispatch ran exactly once, with the taken branch's argument.
        assert log.calls.count(('search', 'alpha')) == 1
        assert capability.speculation_stats.launched == 2
        assert capability.speculation_stats.adopted == 1
        assert capability.speculation_stats.evicted == 1

    async def test_repeated_identical_calls_launch_and_adopt_per_occurrence(self):
        """N identical dispatches claim N launches; nondeterministic results never collapse."""
        log = ToolLog()
        capability = CodeMode[None](speculate=['search'])
        code = padded('a = await search(query="alpha")\nb = await search(query="alpha")\nprint(a, b)')
        agent = build_agent(log, code, capability)

        result = await agent.run('go')

        assert result.output == 'done'
        assert log.calls == [('search', 'alpha'), ('search', 'alpha')]
        assert capability.speculation_stats.launched == 2
        assert capability.speculation_stats.adopted == 2

    async def test_broken_code_stream_speculates_nothing(self):
        """A snippet that never parses launches nothing and leaves the retry path untouched."""
        log = ToolLog()
        capability = CodeMode[None](speculate=['search'])
        agent = build_agent(log, 'a = await search(query="alpha"\nprint(a)', capability)

        result = await agent.run('go')

        assert result.output == 'done'
        assert capability.speculation_stats.launched == 0


def build_run_context(deps: None) -> RunContext[None]:
    """Build a `RunContext` for invoking the capability's public hooks directly.

    Mirrors the helper in `test_code_mode.py`.
    """
    return RunContext[None](
        deps=deps,
        model=TestModel(),
        usage=RunUsage(),
        prompt=None,
        messages=[],
        run_step=0,
        pending_messages=[],
    )


class _PlainEventStream:
    """An async iterable of events that is not an async generator, so it has no `aclose`."""

    def __init__(self, events: Sequence[AgentStreamEvent]) -> None:
        self._events = list(events)

    def __aiter__(self) -> _PlainEventStream:
        return self

    async def __anext__(self) -> AgentStreamEvent:
        if not self._events:
            raise StopAsyncIteration
        return self._events.pop(0)


class TestSpeculationEdgeCases:
    async def test_inactive_under_temporal_durability(self):
        """Under Temporal, nothing launches early and dispatches simply miss the store."""

        class TemporalDurability(AbstractCapability[None]):
            in_durable_context = True

        TemporalDurability.__module__ = 'pydantic_ai.durable_exec.temporal'
        log = ToolLog()
        capability = CodeMode[None](speculate=['search'])
        code = padded('a = await search(query="alpha")\nprint(a)')
        agent = build_agent(log, code, capability, extra_capabilities=[TemporalDurability()])

        result = await agent.run('go')

        assert result.output == 'done'
        assert log.calls == [('search', 'alpha')]
        assert log.started_during_stream == 0
        assert capability.speculation_stats.launched == 0

    async def test_speculated_tool_error_surfaces_at_claim(self):
        """A failed launch delivers its error where the cold call would have raised it."""
        log = ToolLog()
        capability = CodeMode[None](speculate=['boom'])
        agent = build_agent(log, padded('a = await boom(payload="x")\nprint(a)'), capability)

        result = await agent.run('go')

        assert result.output == 'done'
        assert log.calls == [('boom', 'x')]
        assert capability.speculation_stats.launched == 1
        assert capability.speculation_stats.adopted == 1

    async def test_denied_speculated_call_records_denial(self):
        """A handler denial reached through a launch is recorded and raised like a cold denial."""

        async def deny_all(ctx: RunContext[None], requests: DeferredToolRequests) -> DeferredToolResults:
            return DeferredToolResults(
                approvals={call.tool_call_id: ToolDenied(message='nope') for call in requests.approvals}
            )

        log = ToolLog()
        capability = CodeMode[None](speculate=['approval_gate'])
        code = padded('a = await approval_gate(value=1)\nprint(a)')
        agent = build_agent(log, code, capability, extra_capabilities=[HandleDeferredToolCalls(handler=deny_all)])

        result = await agent.run('go')

        assert result.output == 'done'
        assert capability.speculation_stats.launched == 1
        assert capability.speculation_stats.adopted == 1

    async def test_unhandled_approval_in_speculated_call_becomes_user_error(self):
        """With no handler capability, a launch hitting `ApprovalRequired` mirrors the cold error."""
        log = ToolLog()
        capability = CodeMode[None](speculate=['approval_gate'])
        agent = build_agent(log, padded('a = await approval_gate(value=1)\nprint(a)'), capability)

        result = await agent.run('go')

        assert result.output == 'done'
        assert capability.speculation_stats.launched == 1
        assert capability.speculation_stats.adopted == 1

    async def test_tool_return_metadata_survives_adoption(self):
        """A `ToolReturn`-returning tool keeps its metadata on the adopted nested return part."""
        log = ToolLog()
        capability = CodeMode[None](speculate=['with_metadata'])
        agent = build_agent(log, padded('a = await with_metadata(query="m")\nprint(a)'), capability)

        result = await agent.run('go')

        assert result.output == 'done'
        assert log.calls == [('with_metadata', 'm')]
        assert capability.speculation_stats.adopted == 1

    async def test_malformed_argument_stream_launches_nothing(self):
        """Arguments that never decode as JSON produce no launches and no watcher crash."""
        log = ToolLog()
        capability = CodeMode[None](speculate=['search'])
        agent = build_agent(log, '', capability, raw_args='this is not json at all')

        result = await agent.run('go')

        assert result.output == 'done'
        assert capability.speculation_stats.launched == 0

    async def test_watcher_handles_dict_args_odd_indexes_and_caps_launches(self):
        """Dict-argument deltas, unknown part indexes, plain (non-generator) streams, and the
        per-part launch cap, driven through the capability's public hooks."""

        def search(query: str) -> str:
            """Return a canned result."""
            return f'result:{query}'

        # Launched tasks may be cancelled before any reaches the tool body (that's the eviction
        # contract), so the body's coverage cannot depend on task scheduling; cover it directly.
        assert search(query='direct') == 'result:direct'

        ctx = build_run_context(None)
        capability = CodeMode[None](speculate=['search'])
        run_capability = await capability.for_run(ctx)
        toolset = run_capability.get_wrapper_toolset(FunctionToolset[None](tools=[Tool(search)]))
        assert isinstance(toolset, CodeModeToolset)

        # Before any step stashed dispatch ingredients, streamed parts are watched but nothing
        # can launch. A string-args part start exercises the initial-args accumulation path.
        pre_events: list[AgentStreamEvent] = [
            PartStartEvent(
                index=0, part=ToolCallPart(tool_name='run_code', args='{"code": "y = 1\\n', tool_call_id='c0')
            ),
        ]
        seen = [
            event async for event in run_capability.wrap_run_event_stream(ctx, stream=_PlainEventStream(pre_events))
        ]
        assert seen == pre_events
        assert run_capability.speculation_stats.launched == 0

        await toolset.get_tools(ctx)

        code = (
            'def helper():\n    return search(query="inner")\n'
            'if True:\n    def nested():\n        return search(query="nested")\n'
            'if False:\n    search("positional")\n'
            'if False:\n    search(**{"query": "z"})\n'
            + ''.join(f'x{i} = search(query="q")\n' for i in range(34))
            + 'done = 1\n'
        )
        events: list[AgentStreamEvent] = [
            PartStartEvent(index=0, part=TextPart(content='hi')),
            PartStartEvent(index=1, part=ToolCallPart(tool_name='run_code', args={}, tool_call_id='c1')),
            PartDeltaEvent(index=1, delta=ToolCallPartDelta(args_delta=None, tool_call_id='c1')),
            PartDeltaEvent(index=1, delta=ToolCallPartDelta(args_delta={'code': code})),
            PartDeltaEvent(index=9, delta=ToolCallPartDelta(args_delta='{}')),
        ]

        seen = [event async for event in run_capability.wrap_run_event_stream(ctx, stream=_PlainEventStream(events))]

        assert seen == events
        # 34 identical literal calls stream past; the def bodies (top-level and nested), the
        # positional call, and the double-star call are ineligible, and the per-part cap stops
        # launches at the limit.
        assert run_capability.speculation_stats.launched == 32

        with pytest.raises(RuntimeError, match='synthetic run failure'):
            await run_capability.on_run_error(ctx, error=RuntimeError('synthetic run failure'))
        assert run_capability.speculation_stats.evicted == 32

    async def test_run_error_without_speculation_passes_through(self):
        """`on_run_error` re-raises untouched when speculation was never enabled."""
        capability = CodeMode[None]()
        with pytest.raises(RuntimeError, match='plain failure'):
            await capability.on_run_error(build_run_context(None), error=RuntimeError('plain failure'))
