"""`SlidingWindowCompaction` -- zero-cost trimming of the oldest messages."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from pydantic_ai._run_context import AgentDepsT
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import ModelMessage, ModelRequest
from pydantic_ai.tools import RunContext

from pydantic_ai_harness.compaction._pinning import reinject_pinned
from pydantic_ai_harness.compaction._receipts import (
    ReceiptInfo,
    discover_transcript_handle,
    format_receipt,
    make_receipt_part,
    record_receipt,
)
from pydantic_ai_harness.compaction._shared import (
    compact_with_span,
    estimate_token_count,
    exceeds,
    find_safe_cutoff,
    find_token_cutoff,
    prepend_first_user_message,
)

if TYPE_CHECKING:
    from pydantic_ai.models import ModelRequestContext


@dataclass
class SlidingWindowCompaction(AbstractCapability[AgentDepsT]):
    """Zero-cost sliding-window trimmer.

    When the conversation exceeds a configurable threshold (message count or
    estimated token count), the oldest messages are discarded while preserving
    tool-call / tool-return pairs.  No LLM calls are made.

    Trimming happens in ``before_model_request`` so it is transparent to the
    rest of the agent run.

    Example:
        ```python
        from pydantic_ai import Agent
        from pydantic_ai_harness.compaction import SlidingWindowCompaction

        agent = Agent(
            'openai:gpt-4o',
            capabilities=[SlidingWindowCompaction(max_messages=80, keep_messages=40)],
        )
        ```
    """

    max_messages: int | None = None
    """Trigger trimming when message count reaches this value. ``None`` disables."""

    max_tokens: int | None = None
    """Trigger trimming when estimated token count reaches this value. ``None`` disables."""

    keep_messages: int = 40
    """Number of tail messages to retain after trimming (message-count trigger)."""

    keep_tokens: int | None = None
    """Target token budget after trimming (token-count trigger).

    When ``None``, falls back to ``keep_messages``.
    """

    tokenizer: Callable[[str], int] | None = None
    """Optional tokenizer for accurate token counting.

    A callable that returns the token count for a given string.
    When ``None``, uses a ~4 characters-per-token heuristic.
    """

    preserve_first_user_message: bool = True
    """When ``True``, the first ``ModelRequest`` containing a ``UserPromptPart``
    is always kept after trimming, in addition to system prompts.
    """

    receipts: bool = False
    """When ``True``, prepend a deterministic compaction receipt recording how much history
    was dropped, with a transcript handle when a ``TranscriptStore`` capability is attached.

    Opt-in for now: the receipt text is content, so defaulting it on is deferred to the
    benchmark eval-rig pass.  The mechanism itself is structural.
    """

    def __post_init__(self) -> None:
        if self.max_messages is None and self.max_tokens is None:
            raise ValueError('At least one of max_messages or max_tokens must be set.')
        if self.max_messages is not None and self.max_messages < 1:
            raise ValueError('max_messages must be positive.')
        if self.max_tokens is not None and self.max_tokens < 1:
            raise ValueError('max_tokens must be positive.')
        if self.keep_messages < 0:
            raise ValueError('keep_messages must be non-negative.')
        if self.keep_tokens is not None and self.keep_tokens < 0:
            raise ValueError('keep_tokens must be non-negative.')

    async def compact(
        self,
        messages: list[ModelMessage],
        ctx: RunContext[AgentDepsT],
    ) -> list[ModelMessage]:
        """Drop the oldest messages down to the configured tail."""
        if self.keep_tokens is not None:
            cutoff = find_token_cutoff(messages, self.keep_tokens, self.tokenizer)
        else:
            cutoff = find_safe_cutoff(messages, self.keep_messages)

        if cutoff <= 0:
            return messages

        trimmed = messages[cutoff:]
        if self.preserve_first_user_message:
            trimmed = prepend_first_user_message(messages, cutoff, trimmed)
        trimmed = reinject_pinned(messages, trimmed)
        if self.receipts:
            trimmed = [self._receipt_message(messages[:cutoff], ctx), *trimmed]
        return trimmed

    def _receipt_message(self, dropped: list[ModelMessage], ctx: RunContext[AgentDepsT]) -> ModelRequest:
        """Build (and record for tracing) a receipt for the *dropped* prefix."""
        dropped_tokens = estimate_token_count(dropped, self.tokenizer)
        handle = discover_transcript_handle(ctx)
        record_receipt(
            ReceiptInfo(
                strategy='SlidingWindowCompaction',
                dropped_messages=len(dropped),
                dropped_tokens=dropped_tokens,
                by='the harness',
                handle=handle,
            )
        )
        text = format_receipt(
            dropped_messages=len(dropped),
            dropped_tokens=dropped_tokens,
            by='the harness',
            handle=handle,
            has_summary=False,
        )
        return ModelRequest(parts=[make_receipt_part(text)])

    async def before_model_request(
        self,
        ctx: RunContext[AgentDepsT],
        request_context: ModelRequestContext,
    ) -> ModelRequestContext:
        """Trim the message list if it exceeds the configured threshold."""
        messages: list[ModelMessage] = list(request_context.messages)
        if not exceeds(messages, self.max_messages, self.max_tokens, self.tokenizer):
            return request_context
        request_context.messages = await compact_with_span(
            ctx,
            strategy='SlidingWindowCompaction',
            messages=messages,
            compact=lambda: self.compact(messages, ctx),
            tokenizer=self.tokenizer,
        )
        return request_context
