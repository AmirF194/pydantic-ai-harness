"""Tests for the Macroscope capability and toolset."""

from __future__ import annotations

import shlex
import time
from collections.abc import Sequence
from pathlib import Path

import pytest
from pydantic_ai import Agent
from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.messages import ToolReturnPart
from pydantic_ai.models.test import TestModel

from pydantic_ai_harness.macroscope import (
    Macroscope,
    MacroscopeReview,
    MacroscopeToolset,
    parse_macroscope_stream,
)


@pytest.fixture
def anyio_backend() -> str:
    """Pin async tests to asyncio: `Agent.run` schedules work with `asyncio.create_task`."""
    return 'asyncio'


_ISSUE_LINE = (
    'issue_event={"issue_id":"i1","sequence":1,"path":"a.py","line":4,'
    '"severity":"medium","category":"REVIEW_TYPE_CORRECTNESS","body":"only checks completion"}'
)


def _fake_cli(directory: Path, lines: Sequence[str], *, name: str = 'macroscope', sleep: float | None = None) -> str:
    """Write an executable stand-in for `macroscope` that records argv and emits `lines` on stderr."""
    script = directory / name
    body = ['#!/bin/sh', 'printf \'%s\\n\' "$@" > "$0.args"', "printf 'macroscope starting\\n'"]
    if sleep is not None:
        body.append(f'sleep {sleep}')
    body += [f"printf '%s\\n' {shlex.quote(line)} >&2" for line in lines]
    script.write_text('\n'.join(body) + '\n')
    script.chmod(0o755)
    return str(script)


def _recorded_args(command: str) -> list[str]:
    """Return the argv the fake CLI was invoked with (without the leading program name)."""
    return Path(f'{command}.args').read_text().split()


def _toolset(command: str, cwd: Path, *, base: str | None = 'main', timeout: float = 30.0) -> MacroscopeToolset[None]:
    return MacroscopeToolset[None](command=command, cwd=cwd, base=base, timeout=timeout)


class TestParseStream:
    def test_parses_review_issue_and_status(self) -> None:
        review = parse_macroscope_stream(['review_id=rev-1', _ISSUE_LINE, 'issue_status=completed'])
        assert review.review_id == 'rev-1'
        assert review.status == 'completed'
        assert len(review.issues) == 1
        issue = review.issues[0]
        assert (issue.issue_id, issue.path, issue.line, issue.severity) == ('i1', 'a.py', 4, 'medium')

    def test_skips_malformed_and_incomplete_issues(self) -> None:
        review = parse_macroscope_stream(
            [
                'review_id=rev-1',
                'issue_event={not json',
                'issue_event={"issue_id":"x"}',  # missing required fields
                _ISSUE_LINE,
                'issue_status=completed',
            ]
        )
        assert [i.issue_id for i in review.issues] == ['i1']

    def test_markers_tolerate_log_prefixes(self) -> None:
        review = parse_macroscope_stream(['2026-07-10 INFO review_id=rev-9 started', 'ts issue_status=failed now'])
        assert review.review_id == 'rev-9'
        assert review.status == 'failed'

    def test_missing_review_id_and_status_default(self) -> None:
        review = parse_macroscope_stream([_ISSUE_LINE])
        assert review.review_id is None
        assert review.status == 'unknown'
        assert len(review.issues) == 1

    def test_empty_marker_tokens_are_ignored(self) -> None:
        review = parse_macroscope_stream(['review_id=', 'issue_status=', 'unrelated line'])
        assert review.review_id is None
        assert review.status == 'unknown'


class TestRunReview:
    async def test_returns_findings(self, tmp_path: Path) -> None:
        command = _fake_cli(tmp_path, ['review_id=rev-1', _ISSUE_LINE, 'issue_status=completed'])
        review = await _toolset(command, tmp_path).run_macroscope_review()
        assert isinstance(review, MacroscopeReview)
        assert review.review_id == 'rev-1'
        assert review.status == 'completed'
        assert len(review.issues) == 1
        assert _recorded_args(command) == ['codereview', '--base', 'main']

    async def test_clean_review_has_no_issues(self, tmp_path: Path) -> None:
        command = _fake_cli(tmp_path, ['review_id=rev-2', 'issue_status=completed'])
        review = await _toolset(command, tmp_path).run_macroscope_review()
        assert review.issues == []

    async def test_per_call_base_overrides_configured_base(self, tmp_path: Path) -> None:
        command = _fake_cli(tmp_path, ['review_id=rev-3', 'issue_status=completed'])
        await _toolset(command, tmp_path, base='develop').run_macroscope_review(base='release')
        assert _recorded_args(command) == ['codereview', '--base', 'release']

    async def test_missing_binary_raises_model_retry(self, tmp_path: Path) -> None:
        toolset = _toolset('pai-harness-macroscope-absent', tmp_path)
        with pytest.raises(ModelRetry, match='not found'):
            await toolset.run_macroscope_review()

    async def test_no_review_id_raises_model_retry(self, tmp_path: Path) -> None:
        command = _fake_cli(tmp_path, ['issue_status=failed'])
        with pytest.raises(ModelRetry, match='did not start'):
            await _toolset(command, tmp_path).run_macroscope_review()

    async def test_timeout_kills_process_and_raises(self, tmp_path: Path) -> None:
        # sleep(30) far exceeds the 0.2s timeout: a working kill returns promptly, whereas a
        # broken kill would block ~30s in the shielded reap, waiting the process out. The elapsed
        # time is the guard -- it distinguishes "killed" from "waited out" deterministically.
        command = _fake_cli(tmp_path, ['review_id=rev-4', 'issue_status=completed'], sleep=30)
        started = time.monotonic()
        with pytest.raises(ModelRetry, match='timed out'):
            await _toolset(command, tmp_path, timeout=0.2).run_macroscope_review()
        elapsed = time.monotonic() - started
        assert elapsed < 5, f'review took {elapsed:.1f}s -- the process was waited out, not killed'

    async def test_base_omitted_lets_cli_autodetect(self, tmp_path: Path) -> None:
        # With no configured or per-call base, `--base` is dropped so the CLI picks the base itself.
        command = _fake_cli(tmp_path, ['review_id=rev-5', 'issue_status=completed'])
        await _toolset(command, tmp_path, base=None).run_macroscope_review()
        assert _recorded_args(command) == ['codereview']


class TestCapability:
    def test_instructions_toggle(self) -> None:
        assert Macroscope().get_instructions() is not None
        assert Macroscope(include_instructions=False).get_instructions() is None

    async def test_tool_runs_through_agent(self, tmp_path: Path) -> None:
        command = _fake_cli(tmp_path, ['review_id=rev-9', _ISSUE_LINE, 'issue_status=completed'])
        agent = Agent(TestModel(), capabilities=[Macroscope(command=command, cwd=tmp_path, base='main')])
        result = await agent.run('review please')
        returns = [
            part
            for message in result.all_messages()
            for part in message.parts
            if isinstance(part, ToolReturnPart) and part.tool_name == 'run_macroscope_review'
        ]
        assert len(returns) == 1
