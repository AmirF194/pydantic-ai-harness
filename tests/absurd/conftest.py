"""In-memory unit helpers and real PostgreSQL fixtures for Absurd durability tests."""

from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator, Awaitable, Callable, Generator, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TypeVar
from uuid import uuid4

import pytest

pytest.importorskip('absurd_sdk')

import psycopg
from absurd_sdk import (
    AsyncAbsurd,
    AsyncTaskContext,
    JsonValue,
    TaskContext,
    _current_task_context,
)
from psycopg import AsyncConnection
from psycopg.rows import TupleRow
from testcontainers.community.postgres import PostgresContainer

R = TypeVar('R')
AsyncConn = AsyncConnection[TupleRow]

# Copied byte-for-byte from pydantic-ai-absurd v0.8.0 at commit 6ce8e08.
_ABSURD_SQL = (Path(__file__).parent / 'fixtures' / 'absurd.sql').read_text()


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


def _docker_host_env() -> None:  # pragma: no cover - environment-dependent
    """Point Testcontainers at Docker Desktop's user socket on macOS when needed."""
    if 'DOCKER_HOST' in os.environ:
        return
    docker_socket = Path.home() / '.docker' / 'run' / 'docker.sock'
    if docker_socket.exists():
        os.environ['DOCKER_HOST'] = f'unix://{docker_socket}'


def _normalize_dsn(url: str) -> str:
    if url.startswith('postgresql+psycopg2://'):
        return 'postgresql://' + url.split('://', 1)[1]
    return url  # pragma: no cover - Testcontainers currently returns the psycopg2 form


@pytest.fixture(scope='session')
def postgres_container() -> Iterator[PostgresContainer]:
    """Start the real PostgreSQL instance used by Absurd integration tests."""
    _docker_host_env()
    with PostgresContainer('postgres:16-alpine') as container:
        yield container


@pytest.fixture(scope='session')
def absurd_db_dsn(postgres_container: PostgresContainer) -> str:
    """Install Absurd's schema into the disposable PostgreSQL instance."""
    dsn = _normalize_dsn(postgres_container.get_connection_url())
    with psycopg.connect(dsn, autocommit=True) as conn:
        # Psycopg requires `LiteralString` to prevent dynamic SQL injection. This is a trusted,
        # checked-in schema fixture rather than user input.
        conn.execute(_ABSURD_SQL)  # pyright: ignore[reportCallIssue, reportArgumentType]
    return dsn


@pytest.fixture
async def absurd_client(absurd_db_dsn: str) -> AsyncIterator[AsyncAbsurd]:
    """Yield an Absurd client with an isolated queue backed by real PostgreSQL."""
    async with await AsyncConnection.connect(absurd_db_dsn, autocommit=True) as conn:
        client = AsyncAbsurd(conn, queue_name=f'test_{uuid4().hex[:8]}')
        await client.create_queue()
        try:
            yield client
        finally:
            await client.drop_queue()


class FakeAsyncTaskContext(AsyncTaskContext):
    """In-memory stand-in for an Absurd async task context.

    `step` mirrors `AsyncTaskContext.step`: the checkpoint name is disambiguated by
    encounter order (`name`, `name#2`, ...) via the inherited `_get_checkpoint_name`, a stored
    checkpoint is served without calling `fn`, and every stored value is JSON round-tripped just
    as Postgres storage would be, so a non-serializable payload fails here the same way it would
    in production. `invoked` records the checkpoint names whose `fn` actually ran, so a test can
    assert a step was reached exactly once across a replay.
    """

    def __init__(self, *, store: dict[str, JsonValue] | None = None) -> None:
        self.task_id = 'fake-task'
        self._store: dict[str, JsonValue] = {} if store is None else store
        self._step_name_counter: dict[str, int] = {}
        self.invoked: list[str] = []

    @property
    def stored(self) -> dict[str, JsonValue]:
        """The checkpoint store, keyed by disambiguated step name."""
        return self._store

    async def step(self, name: str, fn: Callable[[], Awaitable[R]]) -> R:
        checkpoint_name = self._get_checkpoint_name(name)
        if checkpoint_name in self._store:
            return self._store[checkpoint_name]  # pyright: ignore[reportReturnType]
        self.invoked.append(checkpoint_name)
        stored: JsonValue = json.loads(json.dumps(await fn()))  # pyright: ignore[reportAny]
        self._store[checkpoint_name] = stored
        return stored  # pyright: ignore[reportReturnType]

    def replay(self) -> FakeAsyncTaskContext:
        """A fresh context hydrated from the stored checkpoints, as Absurd does on a retry.

        The encounter counter resets (a new attempt reaches the steps from the top), while the
        stored checkpoints carry over, so `step` serves them without re-running `fn`.
        """
        store: dict[str, JsonValue] = json.loads(json.dumps(self._store))  # pyright: ignore[reportAny]
        return FakeAsyncTaskContext(store=store)


class FakeSyncTaskContext(TaskContext):
    """In-memory stand-in for a synchronous Absurd task context.

    Only used to prove the capability rejects a sync context: an agent run is async and cannot be
    awaited from one.
    """

    def __init__(self) -> None:
        self.task_id = 'fake-sync-task'


@contextmanager
def absurd_task_context(ctx: AsyncTaskContext | TaskContext) -> Generator[None]:
    """Activate `ctx` as the current Absurd task context for the duration of the block."""
    token = _current_task_context.set(ctx)
    try:
        yield
    finally:
        _current_task_context.reset(token)
