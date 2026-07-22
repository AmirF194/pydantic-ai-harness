"""The `search_conversation_history` toolset with dependency-free BM25 ranking.

Ported from VStorm's `pydantic-deepagents`
(`pydantic_deep/features/history_archive/toolset.py`). BM25 tuning is dependency-free
(`math` + `re`) and takes its parameters from the owning capability instead of module
constants, so a caller can tune ranking without editing the port.
"""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass

from pydantic_ai import RunContext
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    SystemPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.tools import AgentDepsT
from pydantic_ai.toolsets import FunctionToolset

from pydantic_ai_harness.conversation_archive._archive import load_archive

SEARCH_HISTORY_DESCRIPTION = """\
Search the full conversation history, including messages that context \
compaction dropped from the live context to save space.

When the conversation is compacted, older messages are replaced by a \
summary in the active context, but the full history stays in an on-disk \
archive. Use this tool to find specific details from earlier in the \
conversation.

Results are ranked by relevance using BM25 -- rare terms and exact matches \
score higher. Multi-word queries search each word independently.

When to use:
- You need to recall exact details from earlier in the conversation
- The conversation summary lacks enough detail for the current task
- You need specific code, file paths, or decisions from before compaction

When NOT to use:
- The information is still in the current conversation context
- You need external or real-time information (use web tools instead)"""

_TOKENIZE_RE = re.compile(r'[a-zA-Z0-9]+')
"""Tokenizer regex: alphanumeric runs, splitting on every other character."""


@dataclass(frozen=True)
class _Bm25Params:
    """BM25 tuning supplied by the owning capability."""

    k1: float
    b: float


def _tokenize(text: str) -> list[str]:
    return [match.group().lower() for match in _TOKENIZE_RE.finditer(text)]


def _compute_idf(term: str, doc_tokens: list[list[str]]) -> float:
    """Inverse document frequency, standard BM25 form `ln((N - df + 0.5)/(df + 0.5) + 1)`."""
    n = len(doc_tokens)
    df = sum(1 for tokens in doc_tokens if term in set(tokens))
    if df == 0:
        return 0.0
    return math.log((n - df + 0.5) / (df + 0.5) + 1.0)


def _bm25_score(
    query_tokens: list[str],
    doc_tokens: list[str],
    idf_scores: dict[str, float],
    avgdl: float,
    params: _Bm25Params,
) -> float:
    dl = len(doc_tokens)
    if dl == 0 or avgdl == 0:  # pragma: no cover - every rendered doc carries its `[index]` token, so dl >= 1
        return 0.0

    tf_map: dict[str, int] = {}
    for token in doc_tokens:
        tf_map[token] = tf_map.get(token, 0) + 1

    score = 0.0
    for qt in query_tokens:
        tf = tf_map.get(qt, 0)
        if tf == 0:
            continue
        idf = idf_scores.get(qt, 0.0)
        numerator = tf * (params.k1 + 1.0)
        denominator = tf + params.k1 * (1.0 - params.b + params.b * dl / avgdl)
        score += idf * numerator / denominator

    return score


def _bm25_rank(query: str, documents: list[str], params: _Bm25Params) -> list[tuple[int, float]]:
    """Rank documents by BM25 relevance, descending; only scores greater than zero."""
    query_tokens = _tokenize(query)
    if not query_tokens or not documents:
        return []

    doc_tokens = [_tokenize(doc) for doc in documents]
    avgdl = sum(len(tokens) for tokens in doc_tokens) / len(doc_tokens)

    unique_query_tokens = list(dict.fromkeys(query_tokens))
    idf_scores = {qt: _compute_idf(qt, doc_tokens) for qt in unique_query_tokens}

    results: list[tuple[int, float]] = []
    for index, tokens in enumerate(doc_tokens):
        score = _bm25_score(unique_query_tokens, tokens, idf_scores, avgdl, params)
        if score > 0:
            results.append((index, score))

    results.sort(key=lambda item: item[1], reverse=True)
    return results


def _format_message(message: ModelMessage, *, truncate: bool) -> str:
    """Render one message to text.

    `truncate=False` keeps content full-length so the BM25 index matches terms past the
    display cutoff; `truncate=True` produces the shorter excerpt shown to the model.
    """
    lines: list[str] = []

    if isinstance(message, ModelRequest):
        for part in message.parts:
            if isinstance(part, UserPromptPart):
                lines.append(f'User: {part.content}')
            elif isinstance(part, SystemPromptPart):
                content = part.content
                if content.startswith('Summary of previous conversation'):
                    lines.append('[Compaction summary]')
                else:
                    if truncate:
                        content = content[:200]
                    lines.append(f'System: {content}')
            elif isinstance(part, ToolReturnPart):
                content = str(part.content)
                if truncate and len(content) > 500:
                    content = content[:500] + '...'
                lines.append(f'Tool [{part.tool_name}]: {content}')
            else:
                # The only remaining `ModelRequestPart` is `RetryPromptPart`
                # (`ToolSearchReturnPart`/`LoadCapabilityReturnPart` subclass `ToolReturnPart`).
                # A retry or validation-error prompt is worth recalling, so index it in full.
                lines.append(f'Retry [{part.tool_name}]: {part.content}')
    else:
        for part in message.parts:
            if isinstance(part, TextPart):
                lines.append(f'Assistant: {part.content}')
            elif isinstance(part, ToolCallPart):
                args = json.dumps(part.args_as_dict(), ensure_ascii=False)
                if truncate and len(args) > 200:
                    args = args[:200] + '...'
                lines.append(f'Tool Call [{part.tool_name}]: {args}')

    return '\n'.join(lines)


def _format_messages(messages: list[ModelMessage], *, truncate: bool) -> list[str]:
    """Render messages to numbered lines, index-aligned with `messages`."""
    lines: list[str] = []
    for index, message in enumerate(messages):
        formatted = _format_message(message, truncate=truncate)
        lines.append(f'[{index}] {formatted}')
    return lines


class ConversationSearchToolset(FunctionToolset[AgentDepsT]):
    """A single `search_conversation_history` tool over an on-disk archive."""

    def __init__(
        self,
        path: str | os.PathLike[str],
        *,
        tool_id: str,
        max_matches: int,
        context_lines: int,
        bm25_k1: float,
        bm25_b: float,
    ) -> None:
        super().__init__(id=tool_id)
        self._path = path
        self._max_matches = max_matches
        self._context_lines = context_lines
        self._params = _Bm25Params(k1=bm25_k1, b=bm25_b)
        self.add_function(
            self.search_conversation_history,
            name='search_conversation_history',
            description=SEARCH_HISTORY_DESCRIPTION,
        )

    async def search_conversation_history(self, ctx: RunContext[AgentDepsT], query: str) -> str:
        """Search the full conversation history using BM25 ranking.

        Args:
            ctx: The active run context.
            query: Text to search for. Multi-word queries search each word
                independently -- rare terms score higher than common ones.
        """
        messages = load_archive(self._path)
        if not messages:
            return (
                'No conversation history archived yet. '
                'History is archived automatically as the conversation progresses.'
            )

        # Rank on the untruncated rendering so terms past the display cutoff stay
        # findable, then show the truncated rendering. Both lists are index-aligned.
        index_lines = _format_messages(messages, truncate=False)
        display_lines = _format_messages(messages, truncate=True)

        ranked = _bm25_rank(query, index_lines, self._params)
        if not ranked:
            return f"No matches for '{query}' in {len(messages)} archived messages."

        results: list[str] = []
        shown_indices: set[int] = set()
        for doc_idx, score in ranked[: self._max_matches]:
            if doc_idx in shown_indices:
                continue
            start = max(0, doc_idx - self._context_lines)
            end = min(len(display_lines), doc_idx + self._context_lines + 1)
            # Record the whole emitted window so a neighbouring match whose context
            # overlaps this one is skipped instead of repeating the same lines.
            shown_indices.update(range(start, end))
            excerpt = '\n'.join(display_lines[start:end])
            results.append(f'[score: {score:.1f}]\n{excerpt}')

        header = f"Found {len(results)} match(es) for '{query}' in {len(messages)} archived messages:\n\n"
        return header + '\n\n---\n\n'.join(results)
