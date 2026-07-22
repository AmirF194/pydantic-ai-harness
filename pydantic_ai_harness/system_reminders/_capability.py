"""System reminders capability: re-inject behavioral guidance without busting the cache."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Literal

from pydantic_ai import Agent
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import CachePoint, ModelMessage, ModelRequest, ModelResponse, TextPart, UserPromptPart
from pydantic_ai.models import KnownModelName, Model
from pydantic_ai.tools import AgentDepsT, RunContext

if TYPE_CHECKING:
    from pydantic_ai.capabilities.abstract import WrapModelRequestHandler
    from pydantic_ai.models import ModelRequestContext


@dataclass
class Reminder:
    r"""A static reminder injected on a cadence during an agent run.

    Args:
        content: The reminder text.
        interval: Fire every N model requests within a run. `interval=3` fires on the
            3rd, 6th, 9th, ... request.
        first_after: Request number of the first fire. `None` (the default) fires on the
            first multiple of `interval` (plain modulo). When set, the reminder fires at
            `first_after`, then every `interval` requests after that.
        trigger: Optional predicate over the current `RunContext`. When set, the reminder
            fires only when the trigger returns `True` *and* the cadence condition is met.
        max_fires: Maximum number of times this reminder may fire within a run. `None`
            means no limit.
        tag: When set, wrap the content in an XML tag: `<tag>\ncontent\n</tag>`. Defaults
            to `'system-reminder'` (Claude Code's convention); set `None` to emit the raw
            content.
    """

    content: str
    interval: int = 1
    first_after: int | None = None
    trigger: Callable[[RunContext[Any]], bool] | None = None
    max_fires: int | None = None
    tag: str | None = 'system-reminder'

    def __post_init__(self) -> None:
        if self.interval < 1:
            raise ValueError(f'interval must be >= 1, got {self.interval}')
        if self.first_after is not None and self.first_after < 1:
            raise ValueError(f'first_after must be >= 1, got {self.first_after}')
        if self.max_fires is not None and self.max_fires < 1:
            raise ValueError(f'max_fires must be >= 1, got {self.max_fires}')

    def should_fire(self, count: int) -> bool:
        """Whether the cadence fires on request number `count` (1-based)."""
        base = self.interval if self.first_after is None else self.first_after
        return count >= base and (count - base) % self.interval == 0

    def render_content(self) -> str:
        """Return the content, XML-wrapped when `tag` is set."""
        if self.tag is not None:
            return f'<{self.tag}>\n{self.content}\n</{self.tag}>'
        return self.content


DynamicReminder = Callable[[RunContext[Any]], 'str | None']
"""A callable returning reminder text (or `None` to skip), evaluated every model request.

The callable reads the current `RunContext`, so it subsumes token-budget, post-compaction,
mode-switch, and other conditions without hardcoded detectors: the user writes the condition.
"""

AsyncDynamicReminder = Callable[[RunContext[Any]], 'Awaitable[str | None]']
"""Async variant of `DynamicReminder`."""

ReminderGenerator = AsyncDynamicReminder
"""Alias for an async dynamic reminder: `async (RunContext) -> str | None`.

`LLMReminder` satisfies this shape. `GoalReanchor` is synchronous, so it satisfies
`DynamicReminder` (the sync alias) instead."""


@dataclass
class SystemReminders(AbstractCapability[AgentDepsT]):
    r"""Inject periodic or conditional reminders to counter instruction fade in long sessions.

    Long multi-turn runs suffer instruction fade: after many tool-use turns the model
    progressively ignores start-of-session guidance. `SystemReminders` re-injects targeted
    guidance mid-run, either on a fixed cadence (`Reminder`) or reactively from a callable
    (`dynamic_reminders`).

    Cache safety is the design constraint. Reminders are appended to the *tail* of each
    request as an ephemeral `UserPromptPart` behind a `CachePoint`, inside `wrap_model_request`
    (which runs after core persists the durable history). So reminders reach the model but
    never enter `message_history`: no stale reminders accumulate, and the cached prefix stays
    byte-identical across turns -- only the small reminder falls outside the cache. Injecting
    into the system prompt or a persisted part instead would bust the cache prefix on every
    fire and let reminders pile up.

    ```python
    from pydantic_ai import Agent
    from pydantic_ai_harness.system_reminders import SystemReminders, Reminder

    agent = Agent(
        'anthropic:claude-sonnet-4-6',
        capabilities=[
            SystemReminders(
                reminders=[Reminder('Stay focused on the original request.', interval=5)],
            )
        ],
    )
    ```
    """

    reminders: list[Reminder] = field(default_factory=list[Reminder])
    """Static reminders injected on a cadence."""

    dynamic_reminders: list[DynamicReminder | AsyncDynamicReminder] = field(
        default_factory=list['DynamicReminder | AsyncDynamicReminder']
    )
    """Callables evaluated every model request; return text to inject or `None` to skip."""

    cache_ttl: Literal['5m', '1h'] = '5m'
    """TTL for the cache breakpoint placed before the tail reminder."""

    on_fire: Callable[[str], None] | None = None
    """Optional observability callback invoked with each rendered reminder as it fires."""

    _request_count: int = field(default=0, init=False, repr=False, compare=False)
    _fire_counts: list[int] = field(default_factory=list[int], init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not self.reminders and not self.dynamic_reminders:
            raise ValueError('At least one static or dynamic reminder must be provided.')
        self._fire_counts = [0] * len(self.reminders)

    async def for_run(self, ctx: RunContext[AgentDepsT]) -> SystemReminders[AgentDepsT]:
        """Return a fresh per-run instance with reset counters (config preserved).

        `replace` re-runs `__post_init__`, which zeroes `_request_count` and `_fire_counts`,
        so concurrent runs on the same agent never share fire state.
        """
        return replace(self)

    async def wrap_model_request(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        request_context: ModelRequestContext,
        handler: WrapModelRequestHandler,
    ) -> ModelResponse:
        """Append fired reminders to the request tail behind a cache breakpoint, then call the model.

        Runs after core persists the durable history; the per-request message list mutated
        here is never written back, so the reminder and its `CachePoint` reach the model but
        never enter `ctx.state.message_history`.
        """
        self._request_count += 1
        messages = request_context.messages
        last = messages[-1]
        if isinstance(last, ModelRequest):
            texts = [*self._collect_static(ctx), *await self._collect_dynamic(ctx)]
            if texts:
                reminder = UserPromptPart(content=[CachePoint(ttl=self.cache_ttl), '\n\n'.join(texts)])
                messages[-1] = replace(last, parts=[*last.parts, reminder])
        return await handler(request_context)

    def _collect_static(self, ctx: RunContext[AgentDepsT]) -> list[str]:
        texts: list[str] = []
        for idx, reminder in enumerate(self.reminders):
            if not reminder.should_fire(self._request_count):
                continue
            if reminder.trigger is not None and not reminder.trigger(ctx):
                continue
            if reminder.max_fires is not None and self._fire_counts[idx] >= reminder.max_fires:
                continue
            self._fire_counts[idx] += 1
            rendered = reminder.render_content()
            texts.append(rendered)
            if self.on_fire is not None:
                self.on_fire(rendered)
        return texts

    async def _collect_dynamic(self, ctx: RunContext[AgentDepsT]) -> list[str]:
        texts: list[str] = []
        for dynamic in self.dynamic_reminders:
            result = dynamic(ctx)
            if isinstance(result, Awaitable):
                result = await result
            if result is not None:
                texts.append(result)
                if self.on_fire is not None:
                    self.on_fire(result)
        return texts

    @classmethod
    def get_serialization_name(cls) -> str | None:
        """Not spec-serializable: reminders take arbitrary callables."""
        return None


@dataclass
class GoalReanchor:
    """Zero-cost dynamic reminder that re-states the run's first user request as the anchor.

    No model call and no dependencies: it reads the first user message from `ctx.messages`
    and asks the model to check its next action advances that goal. Falls back to a static
    line when there is no user message yet. Add it to `SystemReminders.dynamic_reminders`.
    """

    fallback: str = 'Stay on task.'

    def __call__(self, ctx: RunContext[Any]) -> str | None:
        goal = _first_user_text(ctx.messages)
        if goal is None:
            return self.fallback
        return (
            f'Your original request was: "{goal}". Check that your next action advances it. '
            'If it is already satisfied, produce the final answer.'
        )


_LLM_INSTRUCTIONS = (
    'You write a short stay-on-task reminder for an AI agent mid-run. Given the original '
    'goal and recent activity, produce at most two sentences that refocus the agent on the '
    'goal. Output only the reminder text.'
)


@dataclass
class LLMReminder:
    """Dynamic reminder whose text a model generates from a compact transcript.

    Opt-in and dependency-free (it uses `pydantic_ai.Agent`). `model` is required and has no
    default -- pass an explicit model. On any error it falls back to `GoalReanchor` text, so a
    failed generation never blocks the run. Add it to `SystemReminders.dynamic_reminders`.
    """

    model: Model | KnownModelName | str
    max_context_messages: int = 10
    instructions: str = _LLM_INSTRUCTIONS
    _agent: Agent[None, str] | None = field(default=None, init=False, repr=False, compare=False)

    async def __call__(self, ctx: RunContext[Any]) -> str | None:
        try:
            agent = self._agent
            if agent is None:
                agent = Agent(self.model, instructions=self.instructions, output_type=str)
                self._agent = agent
            transcript = _build_compact_transcript(ctx.messages, self.max_context_messages)
            result = await agent.run(transcript)
            text = result.output.strip()
            return text or None
        except Exception:
            return GoalReanchor()(ctx)


def _first_user_text(messages: Sequence[ModelMessage]) -> str | None:
    for msg in messages:
        if isinstance(msg, ModelRequest):
            for part in msg.parts:
                if isinstance(part, UserPromptPart):
                    text = _prompt_text(part.content)
                    if text:
                        return text
    return None


def _prompt_text(content: str | Sequence[object]) -> str:
    if isinstance(content, str):
        return content
    return ' '.join(item for item in content if isinstance(item, str))


def _build_compact_transcript(messages: Sequence[ModelMessage], max_messages: int) -> str:
    goal = _first_user_text(messages)
    recent = _recent_texts(messages, max_messages)
    sections: list[str] = []
    if goal is not None:
        sections.append(f'Original goal: {goal}')
    if recent:
        sections.append('Recent activity:\n' + '\n'.join(recent))
    return '\n\n'.join(sections) if sections else 'No activity yet.'


def _recent_texts(messages: Sequence[ModelMessage], max_messages: int) -> list[str]:
    fragments: list[str] = []
    for msg in messages:
        for part in msg.parts:
            if isinstance(part, UserPromptPart):
                text = _prompt_text(part.content)
                if text:
                    fragments.append(f'user: {text}')
            elif isinstance(part, TextPart) and part.content:
                fragments.append(f'assistant: {part.content}')
    return fragments[-max_messages:]
