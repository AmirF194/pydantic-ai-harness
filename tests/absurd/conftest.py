"""PostgreSQL fixtures shared by the Absurd integration tests."""

from __future__ import annotations

import os
from collections.abc import AsyncGenerator, AsyncIterator, Iterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest

pytest.importorskip('absurd_sdk')

import psycopg
from absurd_sdk import (
    AsyncAbsurd,
    AsyncTaskContext,
    ClaimedTask,
    JsonValue,
    _create_async_task_context,
    _current_task_context,
)
from psycopg import AsyncConnection, sql
from psycopg.rows import TupleRow
from testcontainers.community.postgres import PostgresContainer

FIXTURES = Path(__file__).resolve().parent / 'fixtures'
ABSURD_SQL = (FIXTURES / 'absurd.sql').read_text()
AsyncConn = AsyncConnection[TupleRow]


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
    """Start the disposable PostgreSQL instance used by the integration tests."""
    _docker_host_env()
    container = PostgresContainer('postgres:16-alpine')
    container.start()
    try:
        yield container
    finally:
        container.stop()


@pytest.fixture(scope='session')
def db_dsn(postgres_container: PostgresContainer) -> str:
    """Install Absurd's checked-in schema in the disposable database."""
    dsn = _normalize_dsn(postgres_container.get_connection_url())
    with psycopg.connect(dsn, autocommit=True) as conn:
        # This is a trusted, checked-in schema fixture, not user input.
        conn.execute(ABSURD_SQL)  # pyright: ignore[reportCallIssue, reportArgumentType]
    return dsn


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


@pytest.fixture
async def async_conn(db_dsn: str) -> AsyncIterator[AsyncConn]:
    async with await AsyncConnection.connect(db_dsn, autocommit=True) as conn:
        yield conn


@pytest.fixture
async def absurd(async_conn: AsyncConn) -> AsyncIterator[AsyncAbsurd]:
    """Yield an Absurd client with an isolated queue backed by PostgreSQL."""
    queue = f'test_{uuid4().hex[:8]}'
    client = AsyncAbsurd(async_conn, queue_name=queue)
    await client.create_queue()
    try:
        yield client
    finally:
        await client.drop_queue()


def _absurd_conn(absurd: AsyncAbsurd) -> AsyncConn:
    conn: AsyncConn | None = absurd._conn
    assert conn is not None
    return conn


def _absurd_queue(absurd: AsyncAbsurd) -> str:
    queue: str = absurd._queue_name
    return queue


async def _build_ctx(absurd: AsyncAbsurd, task: ClaimedTask) -> AsyncTaskContext:
    return await _create_async_task_context(
        task['task_id'],
        _absurd_conn(absurd),
        _absurd_queue(absurd),
        task,
        120,
    )


@asynccontextmanager
async def running_task_context(
    absurd: AsyncAbsurd,
    task_name: str,
    params: JsonValue = None,
    *,
    max_attempts: int | None = None,
) -> AsyncGenerator[AsyncTaskContext]:
    """Spawn, claim, and enter an Absurd task context backed by PostgreSQL."""
    spawned = await absurd.spawn(task_name, params, max_attempts=max_attempts)
    claimed = await absurd.claim_tasks(batch_size=1)
    assert claimed, f'no task was claimed for {task_name}'
    task = claimed[0]
    assert task['task_id'] == spawned['task_id']
    ctx = await _build_ctx(absurd, task)
    token = _current_task_context.set(ctx)
    try:
        yield ctx
    finally:
        _current_task_context.reset(token)


@asynccontextmanager
async def reenter_running_task(absurd: AsyncAbsurd, task_id: str) -> AsyncGenerator[AsyncTaskContext]:
    """Reclaim a failed task and enter a fresh context hydrated from PostgreSQL."""
    task = await _force_reclaim(absurd, task_id)
    ctx = await _build_ctx(absurd, task)
    token = _current_task_context.set(ctx)
    try:
        yield ctx
    finally:
        _current_task_context.reset(token)


async def _force_reclaim(absurd: AsyncAbsurd, task_id: str) -> ClaimedTask:
    conn = _absurd_conn(absurd)
    queue = _absurd_queue(absurd)
    run_id = await _current_run_id(conn, queue, task_id)
    await conn.execute(
        'SELECT absurd.fail_run(%s, %s, %s, %s)',
        (queue, run_id, '{"type": "test.FailForReplay"}', None),
    )
    claimed = await absurd.claim_tasks(batch_size=1)
    assert claimed, f'could not re-claim task {task_id} after fail_run'
    task = claimed[0]
    assert task['task_id'] == task_id
    return task


async def _current_run_id(conn: AsyncConn, queue: str, task_id: str) -> str:
    query = sql.SQL('SELECT run_id FROM absurd.{table} WHERE task_id = %s AND state = %s').format(
        table=sql.Identifier(f'r_{queue}')
    )
    cursor = conn.cursor()
    await cursor.execute(query, (task_id, 'running'))
    row: tuple[Any, ...] | None = await cursor.fetchone()
    assert row is not None, f'no running run for task {task_id}'
    return str(row[0])
