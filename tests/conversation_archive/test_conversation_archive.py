"""Tests for the `ConversationArchive` capability.

Write-path, ordering, and dedup behavior is driven through
`Agent(..., capabilities=[...])`. The search tool's ranking, rendering, and
loader edge cases are exercised through the public `ConversationSearchToolset`
directly: driving them through an agent is unreliable because the capability's
own `before_model_request` writes the agent's messages into the same archive the
tool then reads.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
from pydantic_ai import Agent
from pydantic_ai.messages import (
    ModelMessage,
    ModelMessagesTypeAdapter,
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
from pydantic_ai.models import ModelRequestContext, ModelRequestParameters
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import RunContext
from pydantic_ai.usage import RunUsage

from pydantic_ai_harness.compaction import SlidingWindow, SummarizingCompaction
from pydantic_ai_harness.conversation_archive import ConversationArchive, ConversationSearchToolset

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


def _load(path: Path) -> list[ModelMessage]:
    return list(ModelMessagesTypeAdapter.validate_json(path.read_bytes()))


def _write_raw_archive(path: Path, messages: list[ModelMessage]) -> None:
    """Seed an archive file directly via the public type adapter (no harness internals)."""
    path.write_bytes(ModelMessagesTypeAdapter.dump_json(messages))


def _run_context() -> RunContext[None]:
    return RunContext[None](
        deps=None,
        model=TestModel(),
        usage=RunUsage(),
        prompt=None,
        messages=[],
        run_step=0,
    )


async def _search(path: Path, query: str, *, max_matches: int = 10, context_lines: int = 5) -> str:
    """Invoke the search tool directly against the archive at `path`."""
    toolset: ConversationSearchToolset[None] = ConversationSearchToolset(
        str(path),
        tool_id='conversation-archive-search',
        max_matches=max_matches,
        context_lines=context_lines,
        bm25_k1=1.5,
        bm25_b=0.75,
    )
    return await toolset.search_conversation_history(_run_context(), query)


class TestConversationArchive:
    async def test_write_and_recall_survive_compaction(self, tmp_path: Path) -> None:
        path = tmp_path / 'messages.json'
        agent = Agent(
            TestModel(custom_output_text='ok'),
            capabilities=[ConversationArchive(path=str(path)), SlidingWindow(max_messages=2)],
        )
        result = await agent.run('remember the ZEBRA passphrase')
        result = await agent.run('now talk about apples', message_history=result.all_messages())
        result = await agent.run('now talk about oranges', message_history=result.all_messages())

        # The live history was trimmed to 2 messages, but the dropped original survives.
        assert 'ZEBRA' in path.read_text()
        assert len(_load(path)) > 3

        recall = await _search(path, 'ZEBRA passphrase')
        assert 'ZEBRA' in recall
        assert 'archived messages' in recall

    async def test_recall_through_agent_tool(self, tmp_path: Path) -> None:
        path = tmp_path / 'messages.json'
        _write_raw_archive(path, [ModelRequest(parts=[UserPromptPart(content='the ZEBRA passphrase is secret')])])
        calls: list[int] = []

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            if not calls:
                calls.append(1)
                assert any(tool.name == 'search_conversation_history' for tool in info.function_tools)
                return ModelResponse(parts=[ToolCallPart('search_conversation_history', {'query': 'ZEBRA'})])
            return ModelResponse(parts=[TextPart('done')])

        agent: Agent[None, str] = Agent(FunctionModel(model_fn), capabilities=[ConversationArchive(path=str(path))])
        result = await agent.run('find it')
        returned = next(
            str(part.content)
            for message in result.all_messages()
            for part in getattr(message, 'parts', [])
            if isinstance(part, ToolReturnPart) and part.tool_name == 'search_conversation_history'
        )
        assert 'ZEBRA' in returned

    async def test_archive_keeps_originals_not_summary_artifact(self, tmp_path: Path) -> None:
        path = tmp_path / 'messages.json'
        agent = Agent(
            TestModel(custom_output_text='reply'),
            capabilities=[
                ConversationArchive(path=str(path)),
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

        text = path.read_text()
        # Archive-before-compaction: originals kept, derived summary skipped.
        assert 'CRIMSON' in text
        assert 'Summary of previous conversation' not in text
        for message in _load(path):
            if isinstance(message, ModelRequest):
                for part in message.parts:
                    assert not (isinstance(part, SystemPromptPart) and part.content.startswith('Summary'))

    def test_ordering_is_outermost(self) -> None:
        assert ConversationArchive().get_ordering().position == 'outermost'

    async def test_dedup_no_duplicate_carried_messages(self, tmp_path: Path) -> None:
        path = tmp_path / 'messages.json'
        agent = Agent(TestModel(custom_output_text='ok'), capabilities=[ConversationArchive(path=str(path))])
        result = await agent.run('first prompt DISTINCTIVE')
        first_count = len(_load(path))
        await agent.run('second prompt', message_history=result.all_messages())
        second = _load(path)

        digests = [ModelMessagesTypeAdapter.dump_json([m]) for m in second]
        assert len(digests) == len(set(digests))
        assert len(second) > first_count

    async def test_after_run_flushes_final_response(self, tmp_path: Path) -> None:
        path = tmp_path / 'messages.json'
        agent = Agent(
            TestModel(custom_output_text='FINAL ANSWER token'),
            capabilities=[ConversationArchive(path=str(path))],
        )
        await agent.run('a single prompt')
        # The last response is observed by no `before_model_request`; `after_run` flushes it.
        assert 'FINAL ANSWER token' in path.read_text()

    async def test_before_model_request_is_idempotent(self, tmp_path: Path) -> None:
        path = tmp_path / 'messages.json'
        capability: ConversationArchive[None] = ConversationArchive(path=str(path))
        messages: list[ModelMessage] = [ModelRequest(parts=[UserPromptPart(content='hello world')])]
        request_context = ModelRequestContext(
            model=TestModel(),
            messages=messages,
            model_settings=None,
            model_request_parameters=ModelRequestParameters(),
        )
        await capability.before_model_request(_run_context(), request_context)
        mtime = path.stat().st_mtime_ns
        await capability.before_model_request(_run_context(), request_context)
        # Nothing new: no rewrite.
        assert path.stat().st_mtime_ns == mtime
        assert len(_load(path)) == 1

    async def test_for_run_isolates_dedup_state(self, tmp_path: Path) -> None:
        path = tmp_path / 'messages.json'
        capability: ConversationArchive[None] = ConversationArchive(path=str(path))
        fresh = await capability.for_run(_run_context())
        assert fresh is not capability
        assert isinstance(fresh, ConversationArchive)

    def test_add_instructions_toggle(self, tmp_path: Path) -> None:
        path = tmp_path / 'messages.json'
        assert ConversationArchive(path=str(path)).get_instructions() is not None
        assert ConversationArchive(path=str(path), add_instructions=False).get_instructions() is None

    async def test_writes_to_bare_filename(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        agent = Agent(TestModel(custom_output_text='ok'), capabilities=[ConversationArchive(path='messages.json')])
        await agent.run('bare path prompt')
        assert (tmp_path / 'messages.json').exists()

    async def test_no_match_returns_message(self, tmp_path: Path) -> None:
        path = tmp_path / 'messages.json'
        _write_raw_archive(
            path,
            [
                ModelRequest(parts=[UserPromptPart(content='alpha beta gamma')]),
                ModelResponse(parts=[TextPart(content='delta epsilon')]),
            ],
        )
        # `missing` appears nowhere -> df 0 -> idf 0 -> every score 0 -> no matches.
        assert 'No matches' in await _search(path, 'zzzmissing')

    async def test_empty_query_returns_no_match(self, tmp_path: Path) -> None:
        path = tmp_path / 'messages.json'
        _write_raw_archive(path, [ModelRequest(parts=[UserPromptPart(content='alpha beta')])])
        # A query with no alphanumeric tokens ranks nothing.
        assert 'No matches' in await _search(path, '!!!')

    async def test_empty_archive_message(self, tmp_path: Path) -> None:
        path = tmp_path / 'messages.json'
        path.write_bytes(b'')
        assert 'No conversation history archived yet' in await _search(path, 'anything')

    async def test_missing_archive_message(self, tmp_path: Path) -> None:
        path = tmp_path / 'nested' / 'does-not-exist.json'
        assert 'No conversation history archived yet' in await _search(path, 'anything')
        assert not path.exists()

    async def test_corrupt_archive_logs_and_treated_empty(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        path = tmp_path / 'messages.json'
        path.write_bytes(b'{not valid json')
        with caplog.at_level(logging.WARNING):
            rendered = await _search(path, 'anything')
        assert 'No conversation history archived yet' in rendered
        assert any('failed to parse' in record.message for record in caplog.records)

    async def test_unreadable_archive_logs_and_treated_empty(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        # A directory at the archive path exists() but raises OSError on read.
        path = tmp_path / 'messages.json'
        path.mkdir()
        with caplog.at_level(logging.WARNING):
            rendered = await _search(path, 'anything')
        assert 'No conversation history archived yet' in rendered
        assert any('Could not read' in record.message for record in caplog.records)

    async def test_rendering_covers_all_part_types(self, tmp_path: Path) -> None:
        path = tmp_path / 'messages.json'
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
        _write_raw_archive(path, messages)

        # `mango` appears in two adjacent messages, so the second match's window
        # overlaps the first and is skipped (overlapping-window dedup).
        rendered = await _search(path, 'mango', context_lines=5, max_matches=10)
        assert 'Assistant: assistant reply about mango' in rendered
        assert '[Compaction summary]' in rendered
        assert 'Tool Call [search]' in rendered
        assert 'Tool [readfile]' in rendered
        assert '...' in rendered  # truncation applied to the long tool return / args

    async def test_max_matches_and_context_lines_honored(self, tmp_path: Path) -> None:
        path = tmp_path / 'messages.json'
        messages: list[ModelMessage] = [
            ModelRequest(parts=[UserPromptPart(content=f'needle term entry number {i} filler')]) for i in range(20)
        ]
        _write_raw_archive(path, messages)
        rendered = await _search(path, 'needle', max_matches=1, context_lines=0)
        assert rendered.count('[score:') == 1
