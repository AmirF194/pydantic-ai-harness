# Conversation Search

> [!NOTE]
> Import this capability from its submodule -- there is no top-level `pydantic_ai_harness` re-export:
>
> ```python
> from pydantic_ai_harness.conversation_search import ConversationSearch
> ```
>
> The API may change between releases. Where practical, breaking changes ship with a deprecation warning.

Give the model a `search_conversation_history` tool that BM25-ranks the history a `StepPersistence` capability already persists -- earlier turns that compaction dropped from the live context, and past runs in the same store.

[Source](https://github.com/pydantic/pydantic-ai-harness/tree/main/pydantic_ai_harness/conversation_search/)

## The problem

Compaction capabilities (`SlidingWindow`, `SummarizingCompaction`, ...) narrow the live history so it fits the context window. `SummarizingCompaction` persists its edits: once a prefix is replaced by a summary, the originals are gone from the run's `message_history` on the next turn. The model can no longer recall an exact file path, a decision, or a value stated earlier -- only the summary's paraphrase of it. And nothing at all from previous runs is reachable, however well persisted.

## The solution

`ConversationSearch` persists nothing itself. It reads whatever a persistence capability already stores, through a `HistorySource`, and exposes one tool, `search_conversation_history`, that BM25-ranks that history so the model can pull exact details back into context on demand.

The shipped source, `SnapshotHistorySource`, reads the snapshots `StepPersistence` writes: pair the two capabilities on a shared store instance and recall works with no extra write path, no ordering constraints, and no hook coordination.

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

- Ranking is BM25 (the algorithm behind Lucene/Elasticsearch), implemented in pure Python -- no new dependencies. Rare terms and exact matches score higher; multi-word queries score each word independently.
- Results carry provenance (`run: ... | conversation: ...`), and the tool's optional `run_id` argument scopes a search to one run -- so a run referenced elsewhere (for example by a compaction receipt's transcript handle) is directly resolvable.
- The search reads the store lazily at call time, so it always sees everything persisted so far, including earlier steps of the current run.

## How recovery works

`StepPersistence` saves a full-history snapshot at every step boundary. A compaction strategy that persists its edits (like `SummarizingCompaction`) carries those edits into *later* snapshots -- but the earlier snapshots of the same run were taken while the originals were still live. `SnapshotHistorySource` unions each run's snapshots in write order, keeping the first occurrence of every message (by content hash) and skipping derived summary artifacts, which recovers the originals plus everything compaction never touched.

The dedup keys off a content hash of each serialized message, not object identity: consecutive snapshots re-serialize the same growing history, and durable executors (Temporal, DBOS) re-instantiate messages between steps.

`HistorySource` is deliberately substrate-neutral ("enumerate runs, yield each run's durable message record"): a persistence substrate that keeps an append-only entry log can implement it directly by replay, replacing the snapshot-union adapter without touching the search layer.

## Key options

| Option | Default | Purpose |
| --- | --- | --- |
| `source` | (required) | Where the corpus comes from. Use `SnapshotHistorySource(store)` over the store `StepPersistence` writes to. |
| `max_matches` | `10` | Maximum matching excerpts the search tool returns. |
| `context_lines` | `5` | Lines shown around each match (within the match's run). |
| `bm25_k1` | `1.5` | BM25 term-frequency saturation (Lucene default). |
| `bm25_b` | `0.75` | BM25 length normalization (Lucene default). |
| `add_instructions` | `True` | Emit a short note telling the model the recall tool exists. |
| `tool_id` | `conversation-search` | Toolset id for the search tool. |

## Limitations

- Search only reaches what was persisted: history inherited from runs that never ran with `StepPersistence` (for example a long `message_history` passed in from an unpersisted session) cannot be recovered if compaction drops it before the first snapshot.
- The corpus is rebuilt on each tool call by reading every run's snapshots. Snapshot storage is cumulative (each snapshot re-serializes the growing history), so large stores make each search proportionally more expensive. A persistent index (SQLite FTS5, tracked in [#124](https://github.com/pydantic/pydantic-ai-harness/issues/124)) is the scaling path.
- Reading snapshots restores externalized media (large binary payloads) even though the text index never uses it; stores with remote media backends pay that fetch cost per search.

## Further reading

- [Pydantic AI capabilities](https://pydantic.dev/docs/ai/capabilities/overview/)
- [Step Persistence](../step_persistence/) -- the substrate this capability reads
- [Compaction](../compaction/) -- the capabilities whose drops this one recovers from
