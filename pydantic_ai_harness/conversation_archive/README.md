# Conversation Archive

> [!NOTE]
> Import this capability from its submodule -- there is no top-level `pydantic_ai_harness` re-export:
>
> ```python
> from pydantic_ai_harness.conversation_archive import ConversationArchive
> ```
>
> The API may change between releases. Where practical, breaking changes ship with a deprecation warning.

Keep the full message history on disk and give the model a `search_conversation_history` tool to recall what context compaction dropped from the live conversation.

[Source](https://github.com/pydantic/pydantic-ai-harness/tree/main/pydantic_ai_harness/conversation_archive/)

## The problem

Compaction capabilities (`SlidingWindow`, `SummarizingCompaction`, ...) trim the live history so it fits the context window, and those edits persist forward: once a prefix is trimmed or replaced by a summary, the originals are gone from the run's `message_history` on the next turn. The model can no longer recall an exact file path, a decision, or a value stated earlier -- only the summary's paraphrase of it.

## The solution

`ConversationArchive` captures every message into an append-only JSON archive **before** compaction runs, then exposes one tool, `search_conversation_history`, that BM25-ranks that archive so the model can pull exact details back into context on demand.

- The archive is written from `before_model_request`, positioned in the outermost ordering tier so it observes the pre-compaction history for the turn. Compaction runs after it, on the inner layers.
- The derived compaction summary is never archived: any turn whose live history already carries the `Summary of previous conversation:` artifact skips it, so the archive holds the originals the summary replaced.
- The final turn's response, which no `before_model_request` observes, is flushed in `after_run`.
- Ranking is BM25 (the algorithm behind Lucene/Elasticsearch), implemented in pure Python -- no new dependencies. Rare terms and exact matches score higher; multi-word queries score each word independently.

```python
from pydantic_ai import Agent
from pydantic_ai_harness.compaction import SlidingWindow
from pydantic_ai_harness.conversation_archive import ConversationArchive

agent = Agent(
    'openai:gpt-5',
    capabilities=[
        ConversationArchive(path='.pydantic-ai/messages.json'),
        SlidingWindow(max_messages=40),
    ],
)
```

## Composition with compaction

Ordering is handled automatically. `ConversationArchive.get_ordering()` places it in the outermost tier, and the built-in compaction capabilities (`SlidingWindow`, `SummarizingCompaction`, ...) are non-outermost, so the capability chain's topological sort runs the archive's `before_model_request` before theirs regardless of the order you list them. It captures the pre-compaction history whether you write `capabilities=[ConversationArchive(...), SlidingWindow(...)]` or the reverse.

The only caveat is a custom compaction capability that also declares itself `outermost`: within a tier, user list order breaks the tie, so list `ConversationArchive` first in that case.

Dedup keys off a content hash of each serialized message, not object identity. Durable executors (Temporal, DBOS) serialize state between steps, so the same message arrives as a fresh object each turn; a content hash stays stable across re-instantiation and replay, and the archive re-seeds from disk per run.

## Key options

| Option | Default | Purpose |
| --- | --- | --- |
| `path` | `.pydantic-ai/messages.json` | Archive file. Full history as `ModelMessagesTypeAdapter` JSON. |
| `max_matches` | `10` | Maximum matching excerpts the search tool returns. |
| `context_lines` | `5` | Lines shown around each match. |
| `bm25_k1` | `1.5` | BM25 term-frequency saturation (Lucene default). |
| `bm25_b` | `0.75` | BM25 length normalization (Lucene default). |
| `add_instructions` | `True` | Emit a short note telling the model the recall tool exists. |
| `tool_id` | `conversation-archive-search` | Toolset id for the search tool. |

## Limitations

- The archive grows unbounded -- it never prunes. Use a per-session or per-run `path`.
- Concurrent runs writing the same `path` is out of scope: the writer reads-modifies-writes the whole file, so give each session its own path.
- This is an in-session recall archive. Cross-session or semantic search (FTS5 / embeddings over many sessions, tracked in [#124](https://github.com/pydantic/pydantic-ai-harness/issues/124)) is deferred; the archive can serve as the storage substrate a later cross-session index reads.

## Further reading

- [Pydantic AI capabilities](https://pydantic.dev/docs/ai/capabilities/overview/)
- [Compaction](../compaction/) -- the capabilities this one composes with
- [Step Persistence](../step_persistence/) -- step events and resumable snapshots, a different persistence concern
