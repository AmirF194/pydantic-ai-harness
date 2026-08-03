"""Tests for public schedule-store backends."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from pydantic_ai.usage import UsageLimits

from pydantic_ai_harness.scheduling import (
    InMemoryScheduleStore,
    IntervalTrigger,
    Schedule,
    ScheduleStore,
    SqliteScheduleStore,
)

pytestmark = pytest.mark.anyio

StoreFactory = Callable[[], ScheduleStore]


@pytest.fixture(params=['memory', 'sqlite'])
def store_factory(request: pytest.FixtureRequest, tmp_path: Path) -> StoreFactory:
    """Build each store backend behind the public protocol."""
    if request.param == 'memory':
        return InMemoryScheduleStore
    return lambda: SqliteScheduleStore(str(tmp_path / 'schedules.db'))


def _schedule(schedule_id: str, name: str = 'job') -> Schedule:
    return Schedule(
        id=schedule_id,
        name=name,
        prompt=f'prompt:{name}',
        trigger=IntervalTrigger(every=timedelta(hours=1)),
        next_run_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )


class TestScheduleStoreCrud:
    async def test_empty_and_unknown(self, store_factory: StoreFactory) -> None:
        store = store_factory()
        assert await store.list() == []
        assert await store.get('missing') is None
        assert await store.remove('missing') is False
        with pytest.raises(ValueError, match='Unknown schedule id'):
            await store.save(_schedule('missing'))

    async def test_add_get_save_remove_and_order(self, store_factory: StoreFactory) -> None:
        store = store_factory()
        first = await store.add(_schedule('first', 'First'))
        await store.add(_schedule('second', 'Second'))
        assert [item.id for item in await store.list()] == ['first', 'second']
        first.name = 'Changed'
        assert (await store.get('first')).name == 'First'  # type: ignore[union-attr]
        await store.save(first)
        assert (await store.get('first')).name == 'Changed'  # type: ignore[union-attr]
        assert await store.remove('first') is True
        assert await store.get('first') is None

    async def test_duplicate_id_rejected(self, store_factory: StoreFactory) -> None:
        store = store_factory()
        await store.add(_schedule('same'))
        with pytest.raises(ValueError, match='already exists'):
            await store.add(_schedule('same'))


class TestInMemoryScheduleStoreCopies:
    async def test_list_and_save_do_not_leak_aliases(self) -> None:
        store = InMemoryScheduleStore()
        original = _schedule('copy')
        await store.add(original)
        original.prompt = 'mutated outside'
        listed = await store.list()
        listed[0].prompt = 'mutated listing'
        stored = await store.get('copy')
        assert stored is not None
        assert stored.prompt == 'prompt:job'


class TestSqliteScheduleStore:
    def test_rejects_memory_and_bad_table(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="does not support ':memory:'"):
            SqliteScheduleStore(':memory:')
        with pytest.raises(ValueError, match='invalid table name'):
            SqliteScheduleStore(str(tmp_path / 'db.sqlite'), table='bad name; drop')

    async def test_roundtrip_across_instances(self, tmp_path: Path) -> None:
        database = str(tmp_path / 'persistent.db')
        expected = _schedule('kept')
        expected.usage_limits = UsageLimits(request_limit=3, total_tokens_limit=10_000)
        await SqliteScheduleStore(database).add(expected)
        loaded = await SqliteScheduleStore(database).get('kept')
        assert loaded is not None
        assert loaded.model_dump() == expected.model_dump()
