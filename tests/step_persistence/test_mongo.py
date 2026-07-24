"""Tests for `MongoStepStore` against an in-memory `mongomock-motor` client.

The store uses only the async collection surface `mongomock-motor` provides
(`insert_one`, `find_one`, `update_one` upsert, `find_one_and_update` with
`$inc`, sorted `find`), so it reaches full coverage without a running mongod.
`_mock_client` is the single type-boundary shim (the fake is not a
`pymongo.AsyncMongoClient`).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from mongomock_motor import AsyncMongoMockClient
from pydantic_ai import Agent
from pydantic_ai.messages import (
    BinaryContent,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.test import TestModel
from pymongo import AsyncMongoClient
from pymongo.errors import DuplicateKeyError

from pydantic_ai_harness.media import MongoMediaStore
from pydantic_ai_harness.step_persistence import (
    ContinuableSnapshot,
    MongoStepStore,
    RunRecord,
    StepEvent,
    StepPersistence,
    ToolEffectRecord,
)

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


def _mock_client() -> AsyncMongoClient[dict[str, object]]:
    return AsyncMongoMockClient()  # pyright: ignore[reportUnknownVariableType, reportReturnType]


class TestMongoStepStoreConstruction:
    def test_requires_exactly_one_of_client_or_db_url(self) -> None:
        with pytest.raises(ValueError, match='exactly one'):
            MongoStepStore(database='t')
        with pytest.raises(ValueError, match='exactly one'):
            MongoStepStore(client=_mock_client(), db_url='mongodb://x', database='t')

    def test_requires_database(self) -> None:
        with pytest.raises(ValueError, match='`database=` is required'):
            MongoStepStore(client=_mock_client())

    def test_auto_media_store_shares_client(self) -> None:
        store = MongoStepStore(client=_mock_client(), database='t')
        assert isinstance(store._media_store, MongoMediaStore)  # pyright: ignore[reportPrivateUsage]

    def test_media_store_none_disables_media(self) -> None:
        store = MongoStepStore(client=_mock_client(), database='t', media_store=None)
        assert store._media_store is None  # pyright: ignore[reportPrivateUsage]

    async def test_db_url_owns_client_and_aclose_closes_it(self) -> None:
        store = MongoStepStore(db_url='mongodb://localhost:59017', database='t', media_store=None)
        await store.aclose()

    async def test_shared_client_aclose_is_noop(self) -> None:
        store = MongoStepStore(client=_mock_client(), database='t', media_store=None)
        await store.aclose()


class TestMongoStepStoreProtocol:
    async def test_register_and_get_run(self) -> None:
        store = MongoStepStore(client=_mock_client(), database='t', media_store=None)
        await store.register_run(RunRecord(run_id='r1', conversation_id='c1', agent_name='agent', metadata={'k': 'v'}))
        fetched = await store.get_run(run_id='r1')
        assert fetched is not None
        assert fetched.conversation_id == 'c1'
        assert fetched.metadata == {'k': 'v'}

    async def test_register_duplicate_run_raises(self) -> None:
        store = MongoStepStore(client=_mock_client(), database='t', media_store=None)
        await store.register_run(RunRecord(run_id='r1'))
        with pytest.raises(DuplicateKeyError):
            await store.register_run(RunRecord(run_id='r1'))

    async def test_list_runs_chronological(self) -> None:
        store = MongoStepStore(client=_mock_client(), database='t', media_store=None)
        base = datetime(2024, 1, 1, tzinfo=timezone.utc)
        await store.register_run(RunRecord(run_id='r3', started_at=base + timedelta(seconds=3)))
        await store.register_run(RunRecord(run_id='r1', started_at=base + timedelta(seconds=1)))
        await store.register_run(RunRecord(run_id='r2', started_at=base + timedelta(seconds=2)))
        records = await store.list_runs()
        assert [r.run_id for r in records] == ['r1', 'r2', 'r3']

    async def test_list_runs_filters(self) -> None:
        store = MongoStepStore(client=_mock_client(), database='t', media_store=None)
        await store.register_run(RunRecord(run_id='r1', conversation_id='a', parent_run_id='p'))
        await store.register_run(RunRecord(run_id='r2', conversation_id='a', parent_run_id='q'))
        await store.register_run(RunRecord(run_id='r3', conversation_id='b', parent_run_id='p'))

        assert {r.run_id for r in await store.list_runs(conversation_id='a')} == {'r1', 'r2'}
        assert {r.run_id for r in await store.list_runs(parent_run_id='p')} == {'r1', 'r3'}
        assert [r.run_id for r in await store.list_runs(parent_run_id='p', conversation_id='a')] == ['r1']

    async def test_append_and_list_events(self) -> None:
        store = MongoStepStore(client=_mock_client(), database='t', media_store=None)
        await store.register_run(RunRecord(run_id='r1'))
        await store.append_event(StepEvent(run_id='r1', kind='run_started', step_index=0))
        await store.append_event(
            StepEvent(run_id='r1', kind='tool_call_started', step_index=1, tool_call_id='t1', tool_name='add')
        )
        events = await store.list_events(run_id='r1')
        assert [e.kind for e in events] == ['run_started', 'tool_call_started']
        assert events[1].tool_call_id == 't1'

    async def test_save_and_load_snapshot(self) -> None:
        store = MongoStepStore(client=_mock_client(), database='t', media_store=None)
        await store.register_run(RunRecord(run_id='r1'))
        messages: list[ModelMessage] = [
            ModelRequest(parts=[UserPromptPart(content='hello')]),
            ModelResponse(parts=[TextPart(content='hi back')]),
        ]
        await store.save_snapshot(ContinuableSnapshot(run_id='r1', step_index=2, messages=messages))
        snap = await store.latest_snapshot(run_id='r1')
        assert snap is not None
        assert snap.step_index == 2
        assert len(snap.messages) == 2

    async def test_latest_snapshot_default_skips_interrupted(self) -> None:
        store = MongoStepStore(client=_mock_client(), database='t', media_store=None)
        msgs: list[ModelMessage] = [ModelRequest(parts=[UserPromptPart(content='a')])]
        await store.save_snapshot(ContinuableSnapshot(run_id='r1', step_index=0, messages=msgs))
        await store.save_snapshot(ContinuableSnapshot(run_id='r1', step_index=1, messages=msgs, state='interrupted'))

        default = await store.latest_snapshot(run_id='r1')
        assert default is not None and default.state == 'complete' and default.step_index == 0
        opted = await store.latest_snapshot(run_id='r1', include_interrupted=True)
        assert opted is not None and opted.state == 'interrupted' and opted.step_index == 1

    async def test_snapshot_seq_monotonic_across_reset_step(self) -> None:
        """A reused run_id whose step_index resets to 0 must not clobber the prior snapshot."""
        store = MongoStepStore(client=_mock_client(), database='t', media_store=None)
        await store.register_run(RunRecord(run_id='r1'))
        msgs: list[ModelMessage] = [ModelRequest(parts=[UserPromptPart(content='a')])]
        await store.save_snapshot(ContinuableSnapshot(run_id='r1', step_index=5, messages=msgs))
        await store.save_snapshot(ContinuableSnapshot(run_id='r1', step_index=0, messages=msgs))
        snap = await store.latest_snapshot(run_id='r1')
        assert snap is not None
        assert snap.step_index == 0  # last write wins, not the highest step_index

    async def test_tool_effect_upsert_and_scope(self) -> None:
        store = MongoStepStore(client=_mock_client(), database='t', media_store=None)
        await store.record_tool_effect(
            ToolEffectRecord(tool_call_id='t1', tool_name='add', run_id='r1', status='started')
        )
        await store.record_tool_effect(
            ToolEffectRecord(tool_call_id='t1', tool_name='add', run_id='r1', status='completed', effect_summary='ok')
        )
        await store.record_tool_effect(
            ToolEffectRecord(tool_call_id='t1', tool_name='add', run_id='r2', status='started')
        )
        effect = await store.get_tool_effect(run_id='r1', tool_call_id='t1')
        assert effect is not None and effect.status == 'completed' and effect.effect_summary == 'ok'
        other = await store.get_tool_effect(run_id='r2', tool_call_id='t1')
        assert other is not None and other.status == 'started'

    async def test_list_unresolved_tool_effects(self) -> None:
        store = MongoStepStore(client=_mock_client(), database='t', media_store=None)
        await store.record_tool_effect(
            ToolEffectRecord(tool_call_id='t1', tool_name='add', run_id='r1', status='started')
        )
        await store.record_tool_effect(
            ToolEffectRecord(tool_call_id='t2', tool_name='mul', run_id='r1', status='completed')
        )
        unresolved = await store.list_unresolved_tool_effects(run_id='r1')
        assert [r.tool_call_id for r in unresolved] == ['t1']

    async def test_missing_lookups_return_none_or_empty(self) -> None:
        store = MongoStepStore(client=_mock_client(), database='t', media_store=None)
        assert await store.get_run(run_id='nope') is None
        assert await store.latest_snapshot(run_id='nope') is None
        assert await store.get_tool_effect(run_id='nope', tool_call_id='x') is None
        assert await store.list_events(run_id='nope') == []
        assert await store.list_unresolved_tool_effects(run_id='nope') == []
        assert await store.list_runs() == []

    async def test_corrupted_snapshot_document_raises(self) -> None:
        """A poked snapshot with a wrong-typed field surfaces as a ValueError."""
        client = _mock_client()
        store = MongoStepStore(client=client, database='t', media_store=None)
        await store.register_run(RunRecord(run_id='r1'))
        msgs: list[ModelMessage] = [ModelRequest(parts=[UserPromptPart(content='x')])]
        await store.save_snapshot(ContinuableSnapshot(run_id='r1', step_index=0, messages=msgs))
        await client['t']['snapshots'].update_one({'run_id': 'r1'}, {'$set': {'timestamp': 123}})
        with pytest.raises(ValueError, match='snapshot document has wrong types'):
            await store.latest_snapshot(run_id='r1')


class TestMongoStepStoreMedia:
    async def test_large_binary_and_text_externalized_and_restored(self) -> None:
        """Both a large binary part and a large text tool-return round-trip through media."""
        client = _mock_client()
        store = MongoStepStore(client=client, database='t', media_threshold_bytes=64 * 1024)
        await store.register_run(RunRecord(run_id='r1'))
        big_binary = b'\xab' * 100_000
        big_text = 'Z' * 100_000
        messages: list[ModelMessage] = [
            ModelRequest(
                parts=[
                    UserPromptPart(content=[BinaryContent(data=big_binary, media_type='image/png')]),
                    ToolReturnPart(tool_name='scrape', content=big_text, tool_call_id='t1'),
                ]
            ),
            ModelResponse(parts=[TextPart(content='done')]),
        ]
        await store.save_snapshot(ContinuableSnapshot(run_id='r1', step_index=0, messages=messages))

        # Two blobs (binary + text) live in the shared media collections.
        assert await client['t']['media'].count_documents({}) == 2

        snap = await store.latest_snapshot(run_id='r1')
        assert snap is not None
        request = snap.messages[0]
        assert isinstance(request, ModelRequest)
        prompt = request.parts[0]
        assert isinstance(prompt, UserPromptPart)
        assert isinstance(prompt.content, list)
        binary = prompt.content[0]
        assert isinstance(binary, BinaryContent)
        assert binary.data == big_binary
        tool_return = request.parts[1]
        assert isinstance(tool_return, ToolReturnPart)
        assert tool_return.content == big_text

    async def test_below_threshold_stays_inline(self) -> None:
        client = _mock_client()
        store = MongoStepStore(client=client, database='t', media_threshold_bytes=64 * 1024)
        await store.register_run(RunRecord(run_id='r1'))
        messages: list[ModelMessage] = [ModelResponse(parts=[TextPart(content='small')])]
        await store.save_snapshot(ContinuableSnapshot(run_id='r1', step_index=0, messages=messages))
        assert await client['t']['media'].count_documents({}) == 0

    async def test_agent_run_round_trips_through_step_persistence(self) -> None:
        """An Agent run with a large BinaryContent prompt persists and restores via Mongo."""
        store = MongoStepStore(client=_mock_client(), database='t')
        agent: Agent[None, str] = Agent(
            TestModel(),
            capabilities=[StepPersistence(store=store, agent_name='vision')],
        )
        big = b'\xab' * 100_000
        result = await agent.run(['classify this image', BinaryContent(data=big, media_type='image/png')])
        assert isinstance(result.output, str)

        runs = await store.list_runs()
        assert len(runs) == 1
        snap = await store.latest_snapshot(run_id=runs[0].run_id)
        assert snap is not None
        request = snap.messages[0]
        assert isinstance(request, ModelRequest)
        prompt = request.parts[0]
        assert isinstance(prompt, UserPromptPart)
        assert isinstance(prompt.content, list)
        binary = next(p for p in prompt.content if isinstance(p, BinaryContent))
        assert binary.data == big


class TestStepPersistenceLazyExport:
    def test_mongo_step_store_lazily_exported(self) -> None:
        import pydantic_ai_harness.step_persistence as sp

        assert sp.MongoStepStore is MongoStepStore

    def test_unknown_attribute_raises(self) -> None:
        import pydantic_ai_harness.step_persistence as sp

        with pytest.raises(AttributeError, match='has no attribute'):
            _ = sp.NoSuchStore  # type: ignore[attr-defined]
