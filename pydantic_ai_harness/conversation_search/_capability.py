"""The `ConversationSearch` capability: BM25 recall over persisted step history."""

from __future__ import annotations

from dataclasses import dataclass

from pydantic_ai.agent.abstract import AgentInstructions
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.tools import AgentDepsT
from pydantic_ai.toolsets import AgentToolset

from pydantic_ai_harness.conversation_search._source import HistorySource
from pydantic_ai_harness.conversation_search._toolset import ConversationSearchToolset

_INSTRUCTIONS = (
    'A `search_conversation_history` tool can retrieve exact details from persisted history: '
    'earlier turns that context compaction has since dropped from the live context, and past '
    'runs persisted in the same store. Reach for it when the current context, or a compaction '
    'summary, lacks a detail you need.'
)


@dataclass
class ConversationSearch(AbstractCapability[AgentDepsT]):
    """Search persisted conversation history with a dependency-free BM25 tool.

    This capability persists nothing itself: it reads whatever history a persistence
    capability already stores, through a `HistorySource`. Pair it with
    `StepPersistence` sharing the same store, and the model can recall what
    compaction dropped from the live context as well as anything from past runs:

    ```python
    from pydantic_ai import Agent
    from pydantic_ai_harness.compaction import SlidingWindow
    from pydantic_ai_harness.conversation_search import ConversationSearch, SnapshotHistorySource
    from pydantic_ai_harness.step_persistence import SqliteStepStore, StepPersistence

    store = SqliteStepStore(database='sessions.db')
    agent = Agent(
        'openai:gpt-5',
        capabilities=[
            StepPersistence(store=store),
            ConversationSearch(SnapshotHistorySource(store)),
            SlidingWindow(max_messages=40),
        ],
    )
    ```

    Some compaction strategies persist their edits into the run's durable message
    history (`SummarizingCompaction` replaces summarized prefixes for good; a
    `SlidingWindow` trim only narrows what each request sends). Either way,
    `StepPersistence` snapshots each step boundary before the next compaction runs,
    so the union of a run's snapshots still holds the originals --
    `SnapshotHistorySource` recovers them. No ordering or hook coordination between
    the capabilities is required; the search tool reads the store lazily at call
    time.
    """

    source: HistorySource
    """Where the search corpus comes from. Use `SnapshotHistorySource` over the
    store a `StepPersistence` capability writes to."""

    tool_id: str = 'conversation-search'
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

    def get_toolset(self) -> AgentToolset[AgentDepsT] | None:
        """Provide the `search_conversation_history` tool over the source."""
        return ConversationSearchToolset[AgentDepsT](
            self.source,
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
