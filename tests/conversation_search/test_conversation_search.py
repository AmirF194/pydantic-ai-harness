"""Tests for the `ConversationSearch` capability.

End-to-end recall (persist with `StepPersistence`, compact, search) is driven
through `Agent(..., capabilities=[...])`. The search tool's ranking, rendering,
and source edge cases are exercised through the public
`ConversationSearchToolset` and `SnapshotHistorySource` directly, over seeded
`InMemoryStepStore` instances.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic_ai import Agent
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    SystemPromptPart,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import RunContext
from pydantic_ai.usage import RunUsage

from pydantic_ai_harness.compaction import SlidingWindow, SummarizingCompaction
from pydantic_ai_harness.conversation_search import (
    ConversationSearch,
    ConversationSearchToolset,
    HistorySource,
    SnapshotHistorySource,
    SnapshotStore,
)
from pydantic_ai_harness.step_persistence import (
    ContinuableSnapshot,
    FileStepStore,
    InMemoryStepStore,
    RunRecord,
    SqliteStepStore,
    StepPersistence,
)

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


def _run_context() -> RunContext[None]:
    return RunContext[None](
        deps=None,
        model=TestModel(),
        usage=RunUsage(),
        prompt=None,
        messages=[],
        run_step=0,
    )


def _user(content: str) -> ModelRequest:
    return ModelRequest(parts=[UserPromptPart(content=content)])


def _reply(content: str) -> ModelResponse:
    return ModelResponse(parts=[TextPart(content=content)])


async def _seed_run(
    store: InMemoryStepStore,
    run_id: str,
    *snapshots: list[ModelMessage],
    conversation_id: str | None = None,
) -> None:
    """Register a run and save one snapshot per message list, in order."""
    await store.register_run(RunRecord(run_id=run_id, conversation_id=conversation_id))
    for step_index, messages in enumerate(snapshots):
        await store.save_snapshot(ContinuableSnapshot(run_id=run_id, step_index=step_index, messages=messages))


async def _search(
    source: HistorySource,
    query: str,
    *,
    run_id: str | None = None,
    max_matches: int = 10,
    context_lines: int = 5,
) -> str:
    """Invoke the search tool directly against a source."""
    toolset: ConversationSearchToolset[None] = ConversationSearchToolset(
        source,
        tool_id='conversation-search',
        max_matches=max_matches,
        context_lines=context_lines,
        bm25_k1=1.5,
        bm25_b=0.75,
    )
    return await toolset.search_conversation_history(_run_context(), query, run_id=run_id)


class _StubSource:
    """A `HistorySource` that yields messages verbatim, without artifact filtering."""

    def __init__(self, histories: dict[str, list[ModelMessage]]) -> None:
        self._histories = histories

    async def list_runs(self) -> list[RunRecord]:
        return [RunRecord(run_id=run_id) for run_id in self._histories]

    async def run_history(self, *, run_id: str) -> list[ModelMessage]:
        return self._histories[run_id]


class TestSnapshotHistorySource:
    async def test_union_recovers_precompaction_originals(self) -> None:
        store = InMemoryStepStore()
        original = _user('the ZEBRA passphrase is 42')
        reply = _reply('noted')
        summary = ModelRequest(parts=[SystemPromptPart(content='Summary of previous conversation:\n\nolder context')])
        follow_up = _user('what was it again?')
        # Snapshot 1 is pre-compaction; snapshot 2 shows compaction having replaced
        # the original with a summary artifact and the conversation moving on.
        await _seed_run(store, 'r1', [original, reply], [summary, reply, follow_up])

        history = await SnapshotHistorySource(store).run_history(run_id='r1')
        assert history == [original, reply, follow_up]

    async def test_summary_skip_stays_in_sync_with_compaction(self) -> None:
        # Drift guard: the source hand-copies the prefix `SummarizingCompaction` emits. If
        # that prefix ever changes, an artifact built with the real (current) prefix would no
        # longer be recognized and would pollute the corpus. Building the artifact from the
        # compaction module's own constant makes this test fail the moment the two drift
        # apart. The cross-capability import is deliberately test-only; the source keeps a
        # local literal to avoid runtime coupling.
        from pydantic_ai_harness.compaction._summarizing_compaction import _SUMMARY_PREFIX

        store = InMemoryStepStore()
        artifact = ModelRequest(parts=[SystemPromptPart(content=f'{_SUMMARY_PREFIX}older summarized context')])
        original = _user('KEEPME original detail')
        await _seed_run(store, 'r1', [artifact, original])

        history = await SnapshotHistorySource(store).run_history(run_id='r1')
        assert history == [original]

    async def test_unknown_run_yields_empty(self) -> None:
        source = SnapshotHistorySource(InMemoryStepStore())
        assert await source.run_history(run_id='never-ran') == []

    async def test_list_runs_delegates_to_store(self) -> None:
        store = InMemoryStepStore()
        await store.register_run(RunRecord(run_id='r1'))
        await store.register_run(RunRecord(run_id='r2'))
        assert [r.run_id for r in await SnapshotHistorySource(store).list_runs()] == ['r1', 'r2']

    def test_shipped_stores_satisfy_snapshot_store_protocol(self, tmp_path: Path) -> None:
        assert isinstance(InMemoryStepStore(), SnapshotStore)
        assert isinstance(FileStepStore(tmp_path / 'runs'), SnapshotStore)
        assert isinstance(SqliteStepStore(database=tmp_path / 'runs.db'), SnapshotStore)


class TestConversationSearch:
    async def test_cross_run_recall_through_shared_store(self) -> None:
        store = InMemoryStepStore()
        agent = Agent(
            TestModel(custom_output_text='ok'),
            capabilities=[
                StepPersistence(store=store),
                ConversationSearch(SnapshotHistorySource(store)),
                SlidingWindow(max_messages=2),
            ],
        )
        result = await agent.run('remember the ZEBRA passphrase')
        result = await agent.run('now talk about apples', message_history=result.all_messages())
        await agent.run('now talk about oranges', message_history=result.all_messages())

        # A `SlidingWindow` trim narrows what each request sends to the model;
        # the persisted snapshots keep the originals, and search reaches them.
        recall = await _search(SnapshotHistorySource(store), 'ZEBRA passphrase')
        assert 'ZEBRA' in recall
        assert 'persisted messages' in recall

    async def test_recall_survives_persistent_compaction(self) -> None:
        store = InMemoryStepStore()
        agent = Agent(
            TestModel(custom_output_text='reply'),
            capabilities=[
                StepPersistence(store=store),
                ConversationSearch(SnapshotHistorySource(store)),
                SummarizingCompaction(
                    max_messages=2,
                    keep_messages=1,
                    model=TestModel(custom_output_text='SUMMARY TEXT'),
                ),
            ],
        )
        result = await agent.run('the CRIMSON marker is important')
        result = await agent.run('second turn', message_history=result.all_messages())
        result = await agent.run('third turn', message_history=result.all_messages())

        # `SummarizingCompaction` persists its edits: the live history now carries
        # the summary artifact. The corpus recovers the original and excludes the
        # derived summary.
        live = ' '.join(str(part) for message in result.all_messages() for part in message.parts)
        assert 'SUMMARY TEXT' in live
        recall = await _search(SnapshotHistorySource(store), 'CRIMSON marker')
        assert 'CRIMSON' in recall
        assert 'SUMMARY TEXT' not in recall

    async def test_recall_through_agent_tool(self) -> None:
        store = InMemoryStepStore()
        await _seed_run(store, 'past-run', [_user('the ZEBRA passphrase is secret')])
        calls: list[int] = []

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            if not calls:
                calls.append(1)
                assert any(tool.name == 'search_conversation_history' for tool in info.function_tools)
                return ModelResponse(parts=[ToolCallPart('search_conversation_history', {'query': 'ZEBRA'})])
            return ModelResponse(parts=[TextPart('done')])

        agent: Agent[None, str] = Agent(
            FunctionModel(model_fn),
            capabilities=[ConversationSearch(SnapshotHistorySource(store))],
        )
        result = await agent.run('find it')
        returned = next(
            str(part.content)
            for message in result.all_messages()
            for part in getattr(message, 'parts', [])
            if isinstance(part, ToolReturnPart) and part.tool_name == 'search_conversation_history'
        )
        assert 'ZEBRA' in returned

    def test_add_instructions_toggle(self) -> None:
        source = SnapshotHistorySource(InMemoryStepStore())
        assert ConversationSearch(source).get_instructions() is not None
        assert ConversationSearch(source, add_instructions=False).get_instructions() is None


class TestSearchTool:
    async def test_results_carry_run_provenance(self) -> None:
        store = InMemoryStepStore()
        await _seed_run(store, 'r1', [_user('the needle is here')], conversation_id='conv-A')
        rendered = await _search(SnapshotHistorySource(store), 'needle')
        assert 'run: r1' in rendered
        assert 'conversation: conv-A' in rendered

    async def test_run_id_scopes_the_search(self) -> None:
        store = InMemoryStepStore()
        await _seed_run(store, 'r1', [_user('needle in the first run')])
        await _seed_run(store, 'r2', [_user('needle in the second run')])
        source = SnapshotHistorySource(store)

        scoped = await _search(source, 'needle', run_id='r2')
        assert 'second run' in scoped
        assert 'first run' not in scoped

        unscoped = await _search(source, 'needle')
        assert 'first run' in unscoped
        assert 'second run' in unscoped

    async def test_unknown_run_id_reports_missing(self) -> None:
        store = InMemoryStepStore()
        await _seed_run(store, 'r1', [_user('some content')])
        rendered = await _search(SnapshotHistorySource(store), 'anything', run_id='nope')
        assert "No persisted history for run 'nope'" in rendered

    async def test_empty_store_message(self) -> None:
        rendered = await _search(SnapshotHistorySource(InMemoryStepStore()), 'anything')
        assert 'No persisted conversation history to search yet' in rendered

    async def test_run_without_snapshots_is_skipped(self) -> None:
        store = InMemoryStepStore()
        await store.register_run(RunRecord(run_id='registered-only'))
        rendered = await _search(SnapshotHistorySource(store), 'anything')
        assert 'No persisted conversation history to search yet' in rendered

    async def test_context_window_stays_within_run(self) -> None:
        store = InMemoryStepStore()
        await _seed_run(store, 'r1', [_user('unrelated lead-in'), _user('the needle target')])
        await _seed_run(store, 'r2', [_user('OTHERRUN content that must not leak')])
        rendered = await _search(SnapshotHistorySource(store), 'needle', context_lines=5)
        assert 'needle target' in rendered
        assert 'OTHERRUN' not in rendered

    async def test_no_match_returns_message(self) -> None:
        store = InMemoryStepStore()
        await _seed_run(store, 'r1', [_user('alpha beta gamma'), _reply('delta epsilon')])
        # `zzzmissing` appears nowhere -> df 0 -> idf 0 -> every score 0 -> no matches.
        assert 'No matches' in await _search(SnapshotHistorySource(store), 'zzzmissing')

    async def test_empty_query_returns_no_match(self) -> None:
        store = InMemoryStepStore()
        await _seed_run(store, 'r1', [_user('alpha beta')])
        # A query with no word tokens ranks nothing.
        assert 'No matches' in await _search(SnapshotHistorySource(store), '!!!')

    async def test_unicode_terms_are_searchable(self) -> None:
        store = InMemoryStepStore()
        await _seed_run(store, 'r1', [_user('Grüße aus München'), _user('unrelated filler line')])
        rendered = await _search(SnapshotHistorySource(store), 'münchen')
        assert 'München' in rendered

    async def test_rendering_covers_all_part_types(self) -> None:
        # A stub source is used deliberately: it does not filter summary artifacts,
        # exercising the renderer's defensive `[Compaction summary]` collapse for
        # custom sources that pass them through.
        messages: list[ModelMessage] = [
            ModelRequest(
                parts=[
                    SystemPromptPart(content='system guidance ' * 30),
                    UserPromptPart(content='find the ZEBRA passphrase mango'),
                    ToolReturnPart(tool_name='readfile', content='X' * 600, tool_call_id='c1'),
                    RetryPromptPart(content='please retry', tool_name='readfile', tool_call_id='c1'),
                ]
            ),
            ModelResponse(
                parts=[
                    TextPart(content='assistant reply about mango'),
                    ToolCallPart(tool_name='search', args={'q': 'Y' * 300}, tool_call_id='c2'),
                    ThinkingPart(content='internal thought'),
                ]
            ),
            ModelRequest(parts=[SystemPromptPart(content='Summary of previous conversation:\n\nolder kiwi context')]),
        ]
        source = _StubSource({'r1': messages})

        # `mango` appears in two adjacent messages, so the second match's window
        # overlaps the first and is skipped (overlapping-window dedup).
        rendered = await _search(source, 'mango', context_lines=5, max_matches=10)
        assert 'Assistant: assistant reply about mango' in rendered
        assert '[Compaction summary]' in rendered
        assert 'Tool Call [search]' in rendered
        assert 'Tool [readfile]' in rendered
        assert 'Retry [readfile]: please retry' in rendered  # RetryPromptPart is searchable
        assert '...' in rendered  # truncation applied to the long tool return / args

    async def test_user_and_text_parts_truncate_but_stay_searchable(self) -> None:
        long_user = 'pomegranate ' * 60 + 'USERTAIL'
        long_text = 'cranberry ' * 80 + 'TEXTTAIL'
        source = _StubSource(
            {
                'r1': [
                    ModelRequest(parts=[UserPromptPart(content=long_user)]),
                    ModelResponse(parts=[TextPart(content=long_text)]),
                ]
            }
        )
        # Terms past the 500-char display cutoff still match (the index is
        # untruncated), while the displayed excerpt stays truncated.
        rendered = await _search(source, 'USERTAIL')
        excerpts = rendered.split(':\n\n', 1)[1]
        assert 'Found 1 match(es)' in rendered
        assert 'USERTAIL' not in excerpts
        assert '...' in excerpts
        rendered = await _search(source, 'TEXTTAIL')
        excerpts = rendered.split(':\n\n', 1)[1]
        assert 'Found 1 match(es)' in rendered
        assert 'TEXTTAIL' not in excerpts
        assert '...' in excerpts

    async def test_max_matches_and_context_lines_honored(self) -> None:
        store = InMemoryStepStore()
        corpus: list[ModelMessage] = [_user(f'needle term entry number {i} filler') for i in range(20)]
        await _seed_run(store, 'r1', corpus)
        rendered = await _search(SnapshotHistorySource(store), 'needle', max_matches=1, context_lines=0)
        assert rendered.count('[score:') == 1
