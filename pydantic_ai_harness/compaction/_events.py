"""Typed lifecycle events emitted by compaction strategies."""

from dataclasses import dataclass

from pydantic_ai.messages import CapabilityEvent

COMPACTION_EVENTS = 'compaction'


@dataclass(kw_only=True)
class BeforeCompactionEvent(CapabilityEvent, namespace=COMPACTION_EVENTS, dispatch='inline'):
    """A cancellable intervention point before one compaction attempt.

    Listeners must answer synchronously because this event uses inline dispatch. Cancelling skips
    only this attempt; the strategy can try again the next time its trigger fires.
    """

    strategy: str
    message_count: int
    estimated_tokens: int | None
    cancelled: bool = False
    cancel_reason: str | None = None

    def cancel(self, reason: str | None = None) -> None:
        """Cancel this compaction attempt, optionally recording *reason*."""
        self.cancelled = True
        self.cancel_reason = reason


@dataclass(kw_only=True)
class CompactionEndEvent(CapabilityEvent, namespace=COMPACTION_EVENTS):
    """A compaction strategy finished rewriting history."""

    strategy: str
    messages_before: int
    messages_after: int
    tokens_before: int | None
    tokens_after: int | None
