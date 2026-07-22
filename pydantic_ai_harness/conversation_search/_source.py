"""History sources: where the search corpus comes from.

The search layer never persists anything itself. It consumes a `HistorySource` --
"enumerate persisted runs, yield each run's durable message record" -- and ships
`SnapshotHistorySource`, an adapter that recovers that record from the snapshots a
`pydantic_ai_harness.step_persistence.StepPersistence` capability already writes.

The adapter exists because today's step-persistence substrate stores per-boundary
full-history snapshots, and compaction strategies that persist their edits (e.g.
`SummarizingCompaction`) carry those edits into later snapshots: the latest snapshot
is then post-compaction, but earlier snapshots of the same run still hold the
originals, so the union of all snapshots (content-hash deduped, compaction artifacts
excluded) recovers the durable record. A substrate that keeps an append-only entry
log (the session-tree direction of pydantic-ai-harness#321) can implement
`HistorySource` directly via replay and replace the adapter without touching the
search layer.
"""

from __future__ import annotations

import hashlib
from typing import Protocol, runtime_checkable

from pydantic_ai.messages import (
    ModelMessage,
    ModelMessagesTypeAdapter,
    ModelRequest,
    SystemPromptPart,
)

from pydantic_ai_harness.step_persistence import ContinuableSnapshot, RunRecord

_SUMMARY_PREFIX = 'Summary of previous conversation:'
"""Prefix a `SummarizingCompaction` writes into the summary artifact it inserts.

Mirrors `pydantic_ai_harness.compaction._summarizing_compaction._SUMMARY_PREFIX`.
Kept as a local literal rather than an import so this capability does not couple to
compaction internals: the corpus holds the originals a summary replaced, never the
derived summary, so snapshots taken after compaction must contribute only what the
earlier snapshots did not already carry.
"""


@runtime_checkable
class HistorySource(Protocol):
    """A source of persisted conversation history for the search corpus.

    This is the seam between the search layer and whatever substrate persists
    history. `SnapshotHistorySource` implements it over step-persistence
    snapshot stores; an event-sourced substrate can implement it via replay.
    """

    async def list_runs(self) -> list[RunRecord]:
        """Return all persisted runs, sorted by `started_at` ascending."""
        ...  # pragma: no cover

    async def run_history(self, *, run_id: str) -> list[ModelMessage]:
        """Return one run's durable message record, in first-occurrence order.

        The record contains the original messages, including any that compaction
        later dropped from the live context, and excludes derived compaction
        artifacts (e.g. summary messages). An unknown `run_id` yields `[]`.
        """
        ...  # pragma: no cover


@runtime_checkable
class SnapshotStore(Protocol):
    """The narrow read surface `SnapshotHistorySource` needs from a snapshot store.

    A structural subset of the step-persistence stores: `InMemoryStepStore`,
    `FileStepStore`, and `SqliteStepStore` all satisfy it. `list_snapshots` is not
    part of the `StepStore` protocol yet -- the shipped stores implement it as a
    plain method; promoting it into the protocol is proposed alongside the
    session-tree evolution (pydantic-ai-harness#321).
    """

    async def list_runs(
        self,
        *,
        parent_run_id: str | None = None,
        conversation_id: str | None = None,
    ) -> list[RunRecord]: ...  # pragma: no cover

    async def list_snapshots(self, *, run_id: str) -> list[ContinuableSnapshot]: ...  # pragma: no cover


def message_hash(message: ModelMessage) -> str:
    """Return a stable content hash of a single message.

    Dedup keys off serialized content, not object identity: consecutive snapshots
    re-serialize the same growing history, and durable executors (Temporal, DBOS)
    re-instantiate messages between steps, so identity-based dedup would re-append.
    The hash is computed over `ModelMessagesTypeAdapter` bytes, so it is stable
    across snapshot round-trips and replay.
    """
    return hashlib.sha256(ModelMessagesTypeAdapter.dump_json([message])).hexdigest()


def is_summary_artifact(message: ModelMessage) -> bool:
    """Return whether a message is a compaction summary artifact (never indexed)."""
    if not isinstance(message, ModelRequest):
        return False
    return any(
        isinstance(part, SystemPromptPart) and part.content.startswith(_SUMMARY_PREFIX) for part in message.parts
    )


class SnapshotHistorySource:
    """Recover each run's durable message record from its persisted snapshots.

    Reads the same store instance a `StepPersistence` capability writes to. Each
    snapshot holds the full live history at one step boundary; compaction edits
    persist forward, so later snapshots may have replaced early originals with a
    summary. Iterating snapshots in write order and keeping the first occurrence of
    every message (by content hash, skipping summary artifacts) yields the originals
    plus everything compaction never touched.
    """

    def __init__(self, store: SnapshotStore) -> None:
        self._store = store

    async def list_runs(self) -> list[RunRecord]:
        """Return all persisted runs, sorted by `started_at` ascending."""
        return await self._store.list_runs()

    async def run_history(self, *, run_id: str) -> list[ModelMessage]:
        """Union one run's snapshots into its durable message record."""
        seen: set[str] = set()
        history: list[ModelMessage] = []
        for snapshot in await self._store.list_snapshots(run_id=run_id):
            for message in snapshot.messages:
                if is_summary_artifact(message):
                    continue
                digest = message_hash(message)
                if digest in seen:
                    continue
                seen.add(digest)
                history.append(message)
        return history
