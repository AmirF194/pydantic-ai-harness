"""Tests for `CodeMode(speculate=...)`: early launch of sandbox calls from streamed `run_code` deltas.

Behavioral, through `Agent(..., capabilities=[CodeMode(...)])` with a streaming `FunctionModel`:
the model streams the `run_code` arguments in small chunks with an await point between them, so
speculative tasks get scheduled while "generation" is still in flight, and the tools themselves
record when and how often they ran.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

import pytest
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel

from pydantic_ai_harness.code_mode import CodeMode

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


def build_agent(log: ToolLog, code: str, capability: CodeMode[None], chunk_size: int = 16) -> Agent[None, str]:
    """Agent whose model streams one `run_code` call for `code` in `chunk_size` pieces."""

    async def stream_code(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[DeltaToolCalls | str]:
        if len(messages) > 1:
            yield 'done'
            return
        args = json.dumps({'code': code})
        log.streaming_done = False
        for offset in range(0, len(args), chunk_size):
            name = 'run_code' if offset == 0 else None
            yield {1: DeltaToolCall(name=name, json_args=args[offset : offset + chunk_size])}
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
    agent: Agent[None, str] = Agent(model, deps_type=type(None), capabilities=[capability])

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
