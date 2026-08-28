"""Tests for the TrajectoryJudge capability."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock

import pytest
from pydantic_ai import Agent
from pydantic_ai.messages import (
    BinaryContent,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    SystemPromptPart,
    TextContent,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models import ModelRequestContext, ModelRequestParameters
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RunUsage

from pydantic_ai_harness.trajectory_judge import AllGood, Steer, TrajectoryJudge, TrajectoryVerdict

pytestmark = pytest.mark.anyio

_WAIT = 5


@pytest.fixture
def anyio_backend() -> str:
    """Run async tests on the asyncio backend (matching upstream pydantic-ai)."""
    return 'asyncio'


def _steer_response(message: str) -> ModelResponse:
    return ModelResponse(parts=[ToolCallPart('final_result_Steer', {'message': message})])


def _all_good_response() -> ModelResponse:
    return ModelResponse(parts=[ToolCallPart('final_result_AllGood', {})])


def _steer_model(message: str = 'refocus', *, seen: list[str] | None = None) -> FunctionModel:
    """A judge model that always steers, optionally recording the prompt it was given."""

    def fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if seen is not None:
            seen.append(_all_user_text(messages))
        return _steer_response(message)

    return FunctionModel(fn)


def _all_good_model() -> FunctionModel:
    def fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return _all_good_response()

    return FunctionModel(fn)


def _all_user_text(messages: list[ModelMessage]) -> str:
    """Every user prompt string in the messages, joined."""
    texts: list[str] = []
    for message in messages:
        if isinstance(message, ModelRequest):
            for part in message.parts:
                if isinstance(part, UserPromptPart) and isinstance(part.content, str):
                    texts.append(part.content)
    return '\n'.join(texts)


def _ctx() -> Any:
    """A minimal RunContext stand-in for direct hook tests."""
    ctx = MagicMock()
    ctx.usage = RunUsage()
    ctx.usage_limits = None
    ctx.enqueue = MagicMock()
    return ctx


def _request_context(messages: list[ModelMessage]) -> ModelRequestContext:
    return ModelRequestContext(
        model=TestModel(),
        messages=messages,
        model_settings=None,
        model_request_parameters=ModelRequestParameters(),
    )


def _hi_request() -> list[ModelMessage]:
    return [ModelRequest(parts=[UserPromptPart('hi')])]


def _text_response(text: str = 'ok') -> ModelResponse:
    return ModelResponse(parts=[TextPart(text)])


class TestConfigValidation:
    def test_requires_model_or_agent(self) -> None:
        with pytest.raises(ValueError, match='Provide a judge `model`'):
            TrajectoryJudge()

    def test_rejects_model_and_agent(self) -> None:
        judge = Agent(_all_good_model(), output_type=[AllGood, Steer])
        with pytest.raises(ValueError, match='not both'):
            TrajectoryJudge(model='test', agent=judge)

    def test_rejects_instructions_with_agent(self) -> None:
        judge = Agent(_all_good_model(), output_type=[AllGood, Steer])
        with pytest.raises(ValueError, match='owns its own'):
            TrajectoryJudge(agent=judge, instructions='be strict')

    def test_rejects_every_below_one(self) -> None:
        with pytest.raises(ValueError, match='every must be >= 1'):
            TrajectoryJudge(model='test', every=0)

    def test_rejects_window_below_one(self) -> None:
        with pytest.raises(ValueError, match='window must be >= 1'):
            TrajectoryJudge(model='test', window=0)

    def test_not_spec_serializable(self) -> None:
        assert TrajectoryJudge.get_serialization_name() is None


class TestJudgeInstructions:
    async def test_review_focus_reaches_the_judge(self) -> None:
        """`instructions` is appended to the built-in judge instructions as the review focus."""
        done = asyncio.Event()
        judge_instructions: list[str | None] = []

        def judge_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            request = messages[0]
            assert isinstance(request, ModelRequest)
            judge_instructions.append(request.instructions)
            return _all_good_response()

        cap = TrajectoryJudge(
            model=FunctionModel(judge_fn),
            instructions='Flag unsupported claims.',
            every=1,
            on_verdict=lambda _: done.set(),
        )
        ctx = _ctx()
        run_cap = await cap.for_run(ctx)
        await run_cap.after_model_request(
            ctx, request_context=_request_context(_hi_request()), response=_text_response()
        )
        await asyncio.wait_for(done.wait(), timeout=_WAIT)

        assert judge_instructions[0] is not None
        assert 'You are a trajectory judge' in judge_instructions[0]
        assert 'Your review focus:\nFlag unsupported claims.' in judge_instructions[0]


class TestSteering:
    async def test_steering_reaches_the_run(self) -> None:
        """A steer verdict is enqueued with attribution and delivered on the next model request."""
        delivered = asyncio.Event()
        verdicts: list[TrajectoryVerdict] = []

        def on_verdict(verdict: TrajectoryVerdict) -> None:
            verdicts.append(verdict)
            delivered.set()

        seen: list[str] = []
        main_requests: list[str] = []

        def main_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            main_requests.append(_all_user_text(messages))
            if len(main_requests) == 1:
                return ModelResponse(parts=[ToolCallPart('work', {})])
            return _text_response('done')

        agent = Agent(
            FunctionModel(main_fn),
            capabilities=[TrajectoryJudge(model=_steer_model(seen=seen), every=1, name='focus', on_verdict=on_verdict)],
        )

        @agent.tool_plain
        async def work() -> str:
            await asyncio.wait_for(delivered.wait(), timeout=_WAIT)
            return 'worked'

        result = await agent.run('do the thing')
        assert result.output == 'done'
        assert verdicts == [Steer(message='refocus')]
        assert "Steering from trajectory judge 'focus': refocus" in main_requests[1]
        # The judge saw the trajectory rendered as a transcript, wrapped in a tag.
        assert '<trajectory>' in seen[0]
        assert 'user: do the thing' in seen[0]
        assert 'assistant called tool work' in seen[0]

    async def test_all_good_injects_nothing(self) -> None:
        delivered = asyncio.Event()
        verdicts: list[TrajectoryVerdict] = []

        def on_verdict(verdict: TrajectoryVerdict) -> None:
            verdicts.append(verdict)
            delivered.set()

        main_requests: list[str] = []

        def main_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            main_requests.append(_all_user_text(messages))
            if len(main_requests) == 1:
                return ModelResponse(parts=[ToolCallPart('work', {})])
            return _text_response('done')

        agent = Agent(
            FunctionModel(main_fn),
            capabilities=[TrajectoryJudge(model=_all_good_model(), every=1, on_verdict=on_verdict)],
        )

        @agent.tool_plain
        async def work() -> str:
            await asyncio.wait_for(delivered.wait(), timeout=_WAIT)
            return 'worked'

        result = await agent.run('do the thing')
        assert result.output == 'done'
        assert verdicts == [AllGood()]
        assert 'Steering from trajectory judge' not in '\n'.join(main_requests)

    async def test_custom_judge_agent_and_name_fallback(self) -> None:
        """A full judge `Agent` is used as-is; attribution falls back to its `name`."""
        delivered = asyncio.Event()

        judge = Agent(
            _steer_model('watch the secrets'),
            name='security-risk',
            instructions='Review for exposed secrets.',
            output_type=[AllGood, Steer],
        )

        main_requests: list[str] = []

        def main_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            main_requests.append(_all_user_text(messages))
            if len(main_requests) == 1:
                return ModelResponse(parts=[ToolCallPart('work', {})])
            return _text_response('done')

        agent = Agent(
            FunctionModel(main_fn),
            capabilities=[TrajectoryJudge(agent=judge, every=1, on_verdict=lambda _: delivered.set())],
        )

        @agent.tool_plain
        async def work() -> str:
            await asyncio.wait_for(delivered.wait(), timeout=_WAIT)
            return 'worked'

        await agent.run('do the thing')
        assert "Steering from trajectory judge 'security-risk': watch the secrets" in main_requests[1]

    async def test_steer_without_on_verdict_uses_default_name(self) -> None:
        """Without a callback the steer path still enqueues, attributed to the default name."""
        cap = TrajectoryJudge(model=_steer_model('back on task'), every=1)
        ctx = _ctx()
        run_cap = await cap.for_run(ctx)
        await run_cap.after_model_request(
            ctx, request_context=_request_context(_hi_request()), response=_text_response()
        )

        async def wait_for_enqueue() -> None:
            while not ctx.enqueue.called:
                await asyncio.sleep(0.01)

        await asyncio.wait_for(wait_for_enqueue(), timeout=_WAIT)
        ctx.enqueue.assert_called_once_with("Steering from trajectory judge 'trajectory-judge': back on task")
        # The judge's own model call was threaded onto the run's usage.
        assert ctx.usage.requests >= 1


class TestCadence:
    async def test_evaluates_on_the_configured_cadence(self) -> None:
        """With `every=2` and three model requests, the judge evaluates exactly once."""
        delivered = asyncio.Event()
        verdicts: list[TrajectoryVerdict] = []

        def on_verdict(verdict: TrajectoryVerdict) -> None:
            verdicts.append(verdict)
            delivered.set()

        main_requests: list[str] = []

        def main_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            main_requests.append(_all_user_text(messages))
            if len(main_requests) < 3:
                return ModelResponse(parts=[ToolCallPart('work', {'step': len(main_requests)})])
            return _text_response('done')

        agent = Agent(
            FunctionModel(main_fn),
            capabilities=[TrajectoryJudge(model=_steer_model('cadence'), every=2, on_verdict=on_verdict)],
        )

        @agent.tool_plain
        async def work(step: int) -> str:
            if step == 2:
                await asyncio.wait_for(delivered.wait(), timeout=_WAIT)
            return 'worked'

        await agent.run('do the thing')
        assert verdicts == [Steer(message='cadence')]
        assert 'Steering from trajectory judge' not in main_requests[1]
        assert "Steering from trajectory judge 'trajectory-judge': cadence" in main_requests[2]

    async def test_in_flight_evaluation_skips_the_tick(self) -> None:
        """A due tick with an evaluation still running is skipped, then resumes after it finishes."""
        gate = asyncio.Event()
        done = asyncio.Event()
        calls = 0

        async def judge_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            nonlocal calls
            calls += 1
            await gate.wait()
            return _all_good_response()

        cap = TrajectoryJudge(model=FunctionModel(judge_fn), every=1, on_verdict=lambda _: done.set())
        ctx = _ctx()
        run_cap = await cap.for_run(ctx)
        request_context = _request_context(_hi_request())

        await run_cap.after_model_request(ctx, request_context=request_context, response=_text_response())
        await asyncio.sleep(0)  # let the evaluation task start
        # Due again, but the first evaluation is still blocked on the gate: skipped.
        await run_cap.after_model_request(ctx, request_context=request_context, response=_text_response())
        gate.set()
        await asyncio.wait_for(done.wait(), timeout=_WAIT)
        assert calls == 1

        # The next tick reaps the finished evaluation and launches a fresh one.
        done.clear()
        await run_cap.after_model_request(ctx, request_context=request_context, response=_text_response())
        await asyncio.wait_for(done.wait(), timeout=_WAIT)
        assert calls == 2

    async def test_fresh_state_per_run(self) -> None:
        """Cadence counting starts over each run: `every=3` never fires across two 2-step runs."""
        verdicts: list[TrajectoryVerdict] = []

        def main_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            if not any(isinstance(part, ToolReturnPart) for message in messages for part in message.parts):
                return ModelResponse(parts=[ToolCallPart('work', {})])
            return _text_response('done')

        agent = Agent(
            FunctionModel(main_fn),
            capabilities=[TrajectoryJudge(model=_steer_model(), every=3, on_verdict=verdicts.append)],
        )

        @agent.tool_plain
        def work() -> str:
            return 'worked'

        await agent.run('first')
        await agent.run('second')
        assert verdicts == []


class TestFailureHandling:
    async def test_judge_failure_surfaces_on_a_later_tick(self) -> None:
        async def judge_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            raise RuntimeError('judge exploded')

        cap = TrajectoryJudge(model=FunctionModel(judge_fn), every=1)
        ctx = _ctx()
        run_cap = await cap.for_run(ctx)
        request_context = _request_context(_hi_request())
        await run_cap.after_model_request(ctx, request_context=request_context, response=_text_response())

        async def tick_until_raise() -> None:
            while True:
                await asyncio.sleep(0.01)
                await run_cap.after_model_request(ctx, request_context=request_context, response=_text_response())

        with pytest.raises(RuntimeError, match='judge exploded'):
            await asyncio.wait_for(tick_until_raise(), timeout=_WAIT)

    async def test_run_end_cancels_an_in_flight_evaluation(self) -> None:
        """A run that ends mid-evaluation completes normally; the evaluation is cancelled."""
        gate = asyncio.Event()
        verdicts: list[TrajectoryVerdict] = []
        main_requests: list[str] = []

        # The gate never opens: the run ends first and the evaluation is cancelled, possibly
        # before the body even starts, so the whole function is excluded from coverage.
        async def judge_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:  # pragma: no cover
            await gate.wait()
            return _all_good_response()

        def main_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            main_requests.append(_all_user_text(messages))
            if len(main_requests) == 1:
                return ModelResponse(parts=[ToolCallPart('work', {})])
            return _text_response('done')

        agent = Agent(
            FunctionModel(main_fn),
            capabilities=[TrajectoryJudge(model=FunctionModel(judge_fn), every=1, on_verdict=verdicts.append)],
        )

        @agent.tool_plain
        def work() -> str:
            return 'worked'

        result = await agent.run('do the thing')
        assert result.output == 'done'
        assert verdicts == []

    async def test_run_failure_discards_the_in_flight_evaluation(self) -> None:
        """The run's own error propagates; the pending evaluation is cancelled, not consulted."""
        gate = asyncio.Event()

        # The gate never opens: the failing run cancels the evaluation, possibly before the
        # body even starts, so the whole function is excluded from coverage.
        async def judge_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:  # pragma: no cover
            await gate.wait()
            return _all_good_response()

        cap = TrajectoryJudge(model=FunctionModel(judge_fn), every=1)
        ctx = _ctx()
        run_cap = await cap.for_run(ctx)

        async def handler() -> Any:
            await run_cap.after_model_request(
                ctx, request_context=_request_context(_hi_request()), response=_text_response()
            )
            raise RuntimeError('run exploded')

        with pytest.raises(RuntimeError, match='run exploded'):
            await run_cap.wrap_run(ctx, handler=handler)

    async def test_run_failure_without_an_evaluation(self) -> None:
        cap = TrajectoryJudge(model=_steer_model(), every=1)
        ctx = _ctx()
        run_cap = await cap.for_run(ctx)

        async def handler() -> Any:
            raise RuntimeError('run exploded')

        with pytest.raises(RuntimeError, match='run exploded'):
            await run_cap.wrap_run(ctx, handler=handler)


class TestTranscript:
    async def test_renders_observable_behavior_only(self) -> None:
        """User, assistant, tool-call, tool-return, and retry parts render; system and thinking don't."""
        seen: list[str] = []
        done = asyncio.Event()
        cap = TrajectoryJudge(model=_steer_model(seen=seen), every=1, on_verdict=lambda _: done.set())
        ctx = _ctx()
        run_cap = await cap.for_run(ctx)

        messages: list[ModelMessage] = [
            ModelRequest(
                parts=[
                    SystemPromptPart('sys-secret'),
                    UserPromptPart('hello'),
                    UserPromptPart(
                        ['part-a', TextContent(content='part-b'), BinaryContent(data=b'\x00', media_type='image/png')]
                    ),
                    UserPromptPart([BinaryContent(data=b'\x00', media_type='image/png')]),
                ]
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(content='thinking-secret'),
                    TextPart(''),
                    TextPart('answer'),
                    ToolCallPart('lookup', {'q': 1}),
                ]
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(tool_name='lookup', content={'k': 'v'}, tool_call_id='c1'),
                    RetryPromptPart(content='bad args', tool_name='lookup', tool_call_id='c2'),
                    RetryPromptPart(content='not done yet'),
                ]
            ),
        ]
        await run_cap.after_model_request(
            ctx, request_context=_request_context(messages), response=_text_response('wrap-up')
        )
        await asyncio.wait_for(done.wait(), timeout=_WAIT)

        transcript = seen[0]
        assert 'user: hello' in transcript
        assert 'user: part-a part-b' in transcript
        assert transcript.count('user:') == 2  # the image-only prompt renders nothing
        assert 'assistant: answer' in transcript
        assert 'assistant called tool lookup with {"q":1}' in transcript
        assert 'tool lookup returned: {"k":"v"}' in transcript
        assert 'retry (lookup):' in transcript
        assert 'bad args' in transcript
        assert 'retry (output):' in transcript
        assert 'not done yet' in transcript
        assert 'assistant: wrap-up' in transcript
        assert 'sys-secret' not in transcript
        assert 'thinking-secret' not in transcript

    async def test_window_clamps_to_the_transcript_tail(self) -> None:
        seen: list[str] = []
        done = asyncio.Event()
        cap = TrajectoryJudge(model=_steer_model(seen=seen), every=1, window=25, on_verdict=lambda _: done.set())
        ctx = _ctx()
        run_cap = await cap.for_run(ctx)

        messages: list[ModelMessage] = [ModelRequest(parts=[UserPromptPart('x' * 2000)])]
        await run_cap.after_model_request(
            ctx, request_context=_request_context(messages), response=_text_response('recent-marker')
        )
        await asyncio.wait_for(done.wait(), timeout=_WAIT)

        transcript = seen[0].split('<trajectory>\n', 1)[1].rsplit('\n</trajectory>', 1)[0]
        assert 'recent-marker' in transcript
        assert len(transcript) <= 25 * 4  # window tokens * ~4 chars per token
