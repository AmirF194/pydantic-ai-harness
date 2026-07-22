"""The `ConversationArchive` capability: durable full-history archive plus BM25 recall."""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace

from pydantic_ai.agent.abstract import AgentInstructions
from pydantic_ai.capabilities import AbstractCapability, CapabilityOrdering
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models import ModelRequestContext
from pydantic_ai.run import AgentRunResult
from pydantic_ai.tools import AgentDepsT, RunContext
from pydantic_ai.toolsets import AgentToolset

from pydantic_ai_harness.conversation_archive._archive import (
    is_summary_artifact,
    load_archive,
    message_hash,
    write_archive,
)
from pydantic_ai_harness.conversation_archive._toolset import ConversationSearchToolset

_INSTRUCTIONS = (
    'A `search_conversation_history` tool can retrieve exact details from earlier turns that '
    'context compaction has since dropped from the live history. Reach for it when the current '
    'context, or a compaction summary, lacks a detail you need.'
)


@dataclass
class ConversationArchive(AbstractCapability[AgentDepsT]):
    """Keep the full message history and expose a search tool for recall after compaction.

    Compaction capabilities prune the live history to fit the context window, and those
    edits persist forward -- once a prefix is summarized or trimmed, the originals are
    gone from the run's `message_history`. This capability captures every message before
    compaction runs, into an append-only on-disk archive, and exposes a
    `search_conversation_history` tool that BM25-ranks that archive so the model can recall
    what compaction dropped.

    ```python
    from pydantic_ai import Agent
    from pydantic_ai_harness.compaction import SlidingWindow
    from pydantic_ai_harness.conversation_archive import ConversationArchive

    agent = Agent(
        'openai:gpt-5',
        capabilities=[ConversationArchive(path='.pydantic-ai/messages.json'), SlidingWindow(max_messages=40)],
    )
    ```

    Ordering is automatic: `get_ordering` places this in the outermost tier, and the
    built-in compaction capabilities are non-outermost, so the chain runs this
    capability's `before_model_request` before theirs whatever order you list them in.
    Only a custom compaction that also declares itself outermost needs this listed first.
    """

    path: str | os.PathLike[str] = '.pydantic-ai/messages.json'
    """Archive file. Stores the full message history as `ModelMessagesTypeAdapter` JSON."""

    tool_id: str = 'conversation-archive-search'
    """Toolset id for the `search_conversation_history` tool."""

    max_matches: int = 10
    """Maximum number of matching excerpts the search tool returns."""

    context_lines: int = 5
    """Number of surrounding lines shown around each search match."""

    bm25_k1: float = 1.5
    """BM25 term-frequency saturation (Lucene/Elasticsearch default)."""

    bm25_b: float = 0.75
    """BM25 length-normalization (Lucene/Elasticsearch default)."""

    add_instructions: bool = True
    """Emit a short instruction telling the model the recall tool exists."""

    # Dedup keys off serialized content, not object identity (see `message_hash`), so it
    # survives durable replay. Seeded lazily from disk and reset per run by `for_run`.
    _seen: set[str] = field(default_factory=set[str], init=False, repr=False, compare=False)
    _archived: list[ModelMessage] = field(default_factory=list[ModelMessage], init=False, repr=False, compare=False)
    _seeded: bool = field(default=False, init=False, repr=False, compare=False)

    def get_ordering(self) -> CapabilityOrdering:
        """Sit in the outermost tier so archiving runs before compaction prunes history."""
        return CapabilityOrdering(position='outermost')

    async def for_run(self, ctx: RunContext[AgentDepsT]) -> ConversationArchive[AgentDepsT]:
        """Return a fresh instance so concurrent runs don't share dedup state."""
        return replace(self)

    def get_toolset(self) -> AgentToolset[AgentDepsT] | None:
        """Provide the `search_conversation_history` tool over the archive."""
        return ConversationSearchToolset[AgentDepsT](
            self.path,
            tool_id=self.tool_id,
            max_matches=self.max_matches,
            context_lines=self.context_lines,
            bm25_k1=self.bm25_k1,
            bm25_b=self.bm25_b,
        )

    def get_instructions(self) -> AgentInstructions[AgentDepsT] | None:
        """Tell the model the recall tool exists, unless `add_instructions` is false."""
        if not self.add_instructions:
            return None
        return _INSTRUCTIONS

    async def before_model_request(
        self,
        ctx: RunContext[AgentDepsT],
        request_context: ModelRequestContext,
    ) -> ModelRequestContext:
        """Archive the pre-compaction history for this turn without modifying it."""
        self._sync(request_context.messages)
        return request_context

    async def after_run(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        result: AgentRunResult[object],
    ) -> AgentRunResult[object]:
        """Flush the final turn's response, which no `before_model_request` observed."""
        self._sync(result.all_messages())
        return result

    def _sync(self, messages: list[ModelMessage]) -> None:
        if not self._seeded:
            self._archived = load_archive(self.path)
            self._seen = {message_hash(message) for message in self._archived}
            self._seeded = True
        changed = False
        for message in messages:
            if is_summary_artifact(message):
                continue
            digest = message_hash(message)
            if digest in self._seen:
                continue
            self._seen.add(digest)
            self._archived.append(message)
            changed = True
        if changed:
            write_archive(self.path, self._archived)
