"""Trajectory judge capability: review a live run on a cadence and steer it mid-run."""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, TypeAlias

from pydantic_ai import Agent
from pydantic_ai.capabilities import AbstractCapability, WrapRunHandler
from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
    RetryPromptPart,
    TextContent,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models import KnownModelName, Model
from pydantic_ai.tools import AgentDepsT, RunContext

from pydantic_ai_harness._usage import reserved_usage_limits

if TYPE_CHECKING:
    from pydantic_ai.agent import AgentRunResult
    from pydantic_ai.models import ModelRequestContext


@dataclass
class AllGood:
    """The run is on track. No intervention is needed."""


@dataclass
class Steer:
    """The run needs correction; the message is delivered to the running agent."""

    message: str
    """The corrective guidance to deliver: short, specific, and actionable."""


TrajectoryVerdict: TypeAlias = AllGood | Steer
"""What one evaluation concludes: the run is on track, or it needs the enclosed correction."""

# Same ~4 characters-per-token heuristic as `pydantic_ai_harness.compaction`'s
# `estimate_token_count`. Duplicated here because that helper budgets `ModelMessage`s while
# the window here is clamped on the rendered transcript string; fold the two together if a
# shared text-budget helper ever lands.
_CHARS_PER_TOKEN = 4

_JUDGE_INSTRUCTIONS = (
    "You are a trajectory judge: you review another AI agent's run while it is still in "
    'progress. You are shown the most recent slice of its trajectory: user messages, '
    'assistant messages, tool calls, and tool results. Evaluate the trajectory against '
    'your review focus and deliver exactly one final verdict: all-good when the run is on '
    'track, or steer with a short, specific, actionable message when it needs correction. '
    'Steering interrupts the running agent, so steer only when the correction is worth the '
    'interruption.'
)

_PROMPT_HEADER = "Review the running agent's recent trajectory and deliver your verdict."


@dataclass(kw_only=True)
class TrajectoryJudge(AbstractCapability[AgentDepsT]):
    """Review a live run with a second model on a cadence, and steer it mid-run.

    Long-horizon runs drift: instructions fade, unsupported claims compound, and the agent
    wanders off the goal. A `TrajectoryJudge` evaluates the most recent `window` tokens of
    the run's trajectory every `every` model requests, concurrently with the run, and
    delivers exactly one verdict per evaluation: `AllGood`, or `Steer` with a corrective
    message. Steering is enqueued into the run (`RunContext.enqueue`, `'asap'` priority)
    with attribution to the judge, so the running agent course-corrects while recovery is
    still cheap.

    At most one evaluation per judge is in flight at a time; a cadence tick that finds one
    still running is skipped. An evaluation still in flight when the run ends is cancelled.
    The judge's model and tool usage are threaded onto the run's `usage` and run under its
    `usage_limits` (minus one reserved request), so a judge cannot silently exceed the
    run's budget. An evaluation failure is raised on the run at the next cadence tick or at
    run end; give the judge a fallback model (via `agent`) if you need it to degrade
    instead.

    ```python
    from pydantic_ai import Agent
    from pydantic_ai_harness import TrajectoryJudge

    agent = Agent(
        'anthropic:claude-sonnet-4-6',
        capabilities=[
            TrajectoryJudge(
                model='anthropic:claude-haiku-4-5',
                instructions='Flag claims that lack evidence from files the agent actually read.',
                every=20,
            )
        ],
    )
    ```

    Several judges can watch one run: add one `TrajectoryJudge` per concern to
    `capabilities`. Each schedules and evaluates independently.

    Under durable execution the evaluation runs in orchestration context (it is launched
    from a capability hook), so it is not checkpointed and enqueued steering is not
    persisted across replay. Prefer plain runs for judged work, or treat steering as
    best-effort there.
    """

    model: Model | KnownModelName | str | None = None
    """The model that evaluates the trajectory. Provide this (with optional
    `instructions`) or a full `agent`, not both."""

    instructions: str | None = None
    """The judge's review focus, appended to the built-in judge instructions. Only valid
    together with `model`; a passed `agent` owns its own instructions."""

    agent: Agent[None, TrajectoryVerdict] | None = None
    """A full judge agent, for advanced customization (own instructions, model settings,
    toolsets, fallback models). Its `output_type` must be `[AllGood, Steer]` so the judge
    delivers exactly one final verdict. Mutually exclusive with `model`/`instructions`."""

    every: int = 10
    """Evaluate every N model requests within the run."""

    window: int = 20_000
    """Sliding token window: each evaluation sees at most this many tokens of the most
    recent trajectory (estimated at ~4 characters per token), rendered as a transcript."""

    name: str | None = None
    """Name used to attribute steering messages. Defaults to the judge `agent`'s `name`
    when one is passed, then to `'trajectory-judge'`."""

    on_verdict: Callable[[TrajectoryVerdict], None] | None = None
    """Optional observability callback invoked with each verdict after it is processed
    (after any steering has been enqueued)."""

    _judge: Agent[None, TrajectoryVerdict] = field(init=False, repr=False, compare=False)
    _steps: int = field(default=0, init=False, repr=False, compare=False)
    _task: asyncio.Task[None] | None = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.every < 1:
            raise ValueError(f'every must be >= 1, got {self.every}')
        if self.window < 1:
            raise ValueError(f'window must be >= 1, got {self.window}')
        if self.agent is None:
            if self.model is None:
                raise ValueError('Provide a judge `model` (with optional `instructions`) or a full judge `agent`.')
            instructions = _JUDGE_INSTRUCTIONS
            if self.instructions is not None:
                instructions = f'{_JUDGE_INSTRUCTIONS}\n\nYour review focus:\n{self.instructions}'
            self._judge = Agent(self.model, instructions=instructions, output_type=[AllGood, Steer])
        else:
            if self.model is not None:
                raise ValueError('Provide either a judge `model` or a full judge `agent`, not both.')
            if self.instructions is not None:
                raise ValueError('`instructions` configures the built-in judge; a passed `agent` owns its own.')
            self._judge = self.agent

    async def for_run(self, ctx: RunContext[AgentDepsT]) -> TrajectoryJudge[AgentDepsT]:
        """Return a fresh per-run instance so step counts and in-flight evaluations are not shared.

        `replace` re-runs `__init__` and `__post_init__`, resetting the `init=False` fields:
        `_steps` to `0`, `_task` to `None`, and `_judge` rebuilt from the same config.
        """
        return replace(self)

    async def after_model_request(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        request_context: ModelRequestContext,
        response: ModelResponse,
    ) -> ModelResponse:
        """Count the model request and launch an evaluation when the cadence is due.

        A finished evaluation is reaped first, so a failure surfaces here rather than being
        silently dropped. The evaluation itself runs as a background task: the trajectory is
        rendered synchronously (no race with later mutation), the judge call and any
        steering enqueue happen concurrently with the run, and the steering is delivered
        when the run next drains its pending messages.
        """
        self._collect_finished()
        self._steps += 1
        if self._steps % self.every == 0 and self._task is None:
            prompt = _judge_prompt([*request_context.messages, response], self.window)
            self._task = asyncio.create_task(self._evaluate(ctx, prompt), name=f'trajectory-judge:{self._judge_name()}')
        return response

    async def wrap_run(self, ctx: RunContext[AgentDepsT], *, handler: WrapRunHandler) -> AgentRunResult[Any]:
        """Run the agent, then settle the judge: surface a finished failure, cancel the rest.

        An evaluation still in flight when the run ends is cancelled rather than awaited;
        its steering has nowhere to go. One that already finished with an error is
        re-raised so a judge failure is never silently dropped. When the run itself is
        failing, the evaluation's outcome is discarded entirely so it cannot mask the run's
        own error.
        """
        try:
            result = await handler()
        except BaseException:
            await self._discard_in_flight()
            raise
        self._collect_finished()
        await self._discard_in_flight()
        return result

    async def _evaluate(self, ctx: RunContext[AgentDepsT], prompt: str) -> None:
        """Run the judge once and enqueue attributed steering when it says to steer.

        Provider failures propagate out of this task as-is (`ModelAPIError` subclasses from
        the model layer) and are re-raised on the run by `_collect_finished`.
        """
        result = await self._judge.run(prompt, usage=ctx.usage, usage_limits=reserved_usage_limits(ctx.usage_limits))
        verdict = result.output
        if isinstance(verdict, Steer):
            ctx.enqueue(f'Steering from trajectory judge {self._judge_name()!r}: {verdict.message}')
        if self.on_verdict is not None:
            self.on_verdict(verdict)

    def _collect_finished(self) -> None:
        """Reap a finished evaluation, re-raising its failure on the run."""
        task = self._task
        if task is None or not task.done():
            return
        self._task = None
        task.result()

    async def _discard_in_flight(self) -> None:
        """Cancel and reap the current evaluation without inspecting its outcome.

        The task may already be done, in which case `cancel` is a no-op and awaiting it
        re-raises its failure; both that and the cancellation are discarded deliberately.
        """
        task = self._task
        self._task = None
        if task is None:
            return
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await task

    def _judge_name(self) -> str:
        """The attribution name: `name`, then the judge `agent`'s name, then the default."""
        if self.name is not None:
            return self.name
        if self.agent is not None and self.agent.name is not None:
            return self.agent.name
        return 'trajectory-judge'

    @classmethod
    def get_serialization_name(cls) -> str | None:
        """Not spec-serializable: the capability may hold a live `Agent` and a callback."""
        return None


def _judge_prompt(messages: Sequence[ModelMessage], window: int) -> str:
    """The judge's user prompt: the rendered trajectory, clamped to the window's tail."""
    transcript = _render_transcript(messages)
    max_chars = window * _CHARS_PER_TOKEN
    if len(transcript) > max_chars:
        transcript = transcript[-max_chars:]
    return f'{_PROMPT_HEADER}\n\n<trajectory>\n{transcript}\n</trajectory>'


def _render_transcript(messages: Sequence[ModelMessage]) -> str:
    """Render the trajectory as judge-readable lines.

    System prompts and thinking parts are omitted: the judge evaluates observable behavior
    (what was asked, said, called, and returned), not the agent's configuration or private
    reasoning.
    """
    lines: list[str] = []
    for message in messages:
        for part in message.parts:
            if isinstance(part, UserPromptPart):
                text = _prompt_text(part.content)
                if text:
                    lines.append(f'user: {text}')
            elif isinstance(part, ToolReturnPart):
                lines.append(f'tool {part.tool_name} returned: {part.model_response_str()}')
            elif isinstance(part, RetryPromptPart):
                lines.append(f'retry ({part.tool_name or "output"}): {part.model_response()}')
            elif isinstance(part, TextPart):
                if part.content:
                    lines.append(f'assistant: {part.content}')
            elif isinstance(part, ToolCallPart):
                lines.append(f'assistant called tool {part.tool_name} with {part.args_as_json_str()}')
    return '\n'.join(lines)


def _prompt_text(content: str | Sequence[object]) -> str:
    """The text of a user prompt; non-text content (images, files) is omitted."""
    if isinstance(content, str):
        return content
    texts: list[str] = []
    for item in content:
        if isinstance(item, str):
            texts.append(item)
        elif isinstance(item, TextContent):
            texts.append(item.content)
    return ' '.join(texts)
