"""Real PostgreSQL fixtures and task-context helpers for Absurd tests."""

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

AsyncConn = AsyncConnection[TupleRow]

# Copied byte-for-byte from pydantic-ai-absurd v0.8.0 at commit 6ce8e08.
_ABSURD_SQL = (Path(__file__).parent / 'fixtures' / 'absurd.sql').read_text()


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
    absurd_client: AsyncAbsurd,
    task_name: str,
    params: JsonValue = None,
    *,
    max_attempts: int | None = None,
) -> AsyncGenerator[AsyncTaskContext, None]:
    """Spawn, claim, and enter a task context backed by PostgreSQL."""
    spawned = await absurd_client.spawn(
        task_name, params, max_attempts=max_attempts, queue=_absurd_queue(absurd_client)
    )
    claimed = await absurd_client.claim_tasks(batch_size=1)
    assert claimed, f'no task was claimed for {task_name}'
    task = claimed[0]
    assert task['task_id'] == spawned['task_id']
    ctx = await _build_ctx(absurd_client, task)
    token = _current_task_context.set(ctx)
    try:
        yield ctx
    finally:
        _current_task_context.reset(token)


@asynccontextmanager
async def reenter_running_task(absurd_client: AsyncAbsurd, task_id: str) -> AsyncGenerator[AsyncTaskContext, None]:
    """Reclaim a failed task and enter a fresh context hydrated from PostgreSQL."""
    task = await _force_reclaim(absurd_client, task_id)
    ctx = await _build_ctx(absurd_client, task)
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
