"""Exa toolset -- gives agents web search, page retrieval, and deep search backed by the Exa API."""

from __future__ import annotations

import json
from typing import Literal, Protocol

from pydantic_ai.exceptions import ModelRetry, UserError
from pydantic_ai.tools import AgentDepsT
from pydantic_ai.toolsets import FunctionToolset

try:
    from exa_py import AsyncExa
    from exa_py.api import (
        ContentsOptions,
        DeepOutputSchema,
        Result,
        SearchResponse,
        SearchType,
        TextContentsOptions,
    )
except ImportError as _import_error:  # pragma: no cover
    raise ImportError(
        'exa-py is required for ExaSearch. Install it with: pip install "pydantic-ai-harness[exa]"'
    ) from _import_error


class ExaClient(Protocol):
    """The subset of the `exa_py.AsyncExa` API that `ExaSearchToolset` calls.

    Any object with these two methods can back the toolset. Pass one via
    `ExaSearch.client` to configure authentication or the base URL explicitly,
    or to substitute a fake in tests.

    The parameter types mirror `AsyncExa`'s own signatures (including the
    `ContentsOptions` payload `exa_py` types `search` with), so a real
    `AsyncExa` instance satisfies the protocol as-is.
    """

    async def search(
        self,
        query: str,
        *,
        contents: ContentsOptions | Literal[False],
        num_results: int | None = None,
        type: SearchType | None = None,
        output_schema: DeepOutputSchema | None = None,
    ) -> SearchResponse[Result]:
        """Search the web and return results, optionally with page contents or a synthesized output."""
        ...  # pragma: no cover

    async def get_contents(self, urls: str, *, text: TextContentsOptions) -> SearchResponse[Result]:
        """Retrieve the contents of a specific URL."""
        ...  # pragma: no cover


def _default_client() -> ExaClient:
    """Build an `AsyncExa` client from the `EXA_API_KEY` environment variable."""
    try:
        return AsyncExa()
    except ValueError as error:
        raise UserError(
            'ExaSearch needs an Exa API key: set the EXA_API_KEY environment variable, '
            'or pass a configured client, e.g. ExaSearch(client=AsyncExa(api_key=...)).'
        ) from error


class ExaSearchToolset(FunctionToolset[AgentDepsT]):
    """Gives an agent web research tools backed by the Exa search API.

    `web_search` surveys the web and returns results together with page text,
    and `get_page` retrieves the text of one specific URL. With
    `include_deep_search=True`, `deep_search` runs Exa's multi-step deep search
    and returns a synthesized, cited answer in one call.

    Page text is capped at `max_text_chars` characters per result: the cap is
    sent to Exa as the contents limit and re-enforced when output is formatted,
    so text stays bounded even with a custom `client`. The result count is
    bounded the same way: `num_results` is requested from Exa and re-applied to
    the response.
    """

    def __init__(
        self, *, client: ExaClient | None, num_results: int, max_text_chars: int, include_deep_search: bool
    ) -> None:
        if num_results < 1:
            raise ValueError(f'num_results must be at least 1, got {num_results}')
        if max_text_chars < 1:
            raise ValueError(f'max_text_chars must be at least 1, got {max_text_chars}')
        super().__init__()
        self._client = client if client is not None else _default_client()
        self._num_results = num_results
        self._max_text_chars = max_text_chars
        self.add_function(self.web_search, name='web_search')
        self.add_function(self.get_page, name='get_page')
        if include_deep_search:
            self.add_function(self.deep_search, name='deep_search')

    async def web_search(self, query: str) -> str:
        """Search the web and return matching pages, each with its text content.

        Args:
            query: The search query. Natural-language questions and keyword
                queries both work.

        Returns:
            The matching pages, each with title, URL, and page text.
        """
        response = await self._client.search(
            query,
            contents={'text': {'max_characters': self._max_text_chars}},
            num_results=self._num_results,
        )
        if not response.results:
            return f'No results found for {query!r}.'
        results = response.results[: self._num_results]
        sections = [_format_result(result, self._max_text_chars) for result in results]
        plural = 's' if len(sections) != 1 else ''
        joined = '\n\n---\n\n'.join(sections)
        return f'Found {len(sections)} result{plural} for {query!r}:\n\n{joined}'

    async def get_page(self, url: str) -> str:
        """Retrieve the text contents of a specific URL.

        Args:
            url: The URL of the page to read.

        Returns:
            The page's title, URL, and text content.
        """
        response = await self._client.get_contents(url, text={'max_characters': self._max_text_chars})
        if not response.results or not response.results[0].text:
            raise ModelRetry(f'No content could be retrieved for {url!r}. Check the URL or try another page.')
        return _format_result(response.results[0], self._max_text_chars)

    async def deep_search(self, question: str) -> str:
        """Run Exa's multi-step deep search and return a synthesized answer with its sources.

        A full research pass in a single call; suited to questions that need
        synthesis across many sources rather than a quick survey.

        Args:
            question: The research question to answer.

        Returns:
            The synthesized answer, followed by the sources it drew on.
        """
        response = await self._client.search(
            question,
            contents=False,
            type='deep',
            output_schema={'type': 'text'},
        )
        output = response.output
        if output is None or not output.content:
            raise ModelRetry(
                f'Deep search returned no answer for {question!r}. Rephrase the question, or use web_search.'
            )
        content = output.content
        answer = content if isinstance(content, str) else json.dumps(content)
        sources = {citation.url: citation.title for row in output.grounding for citation in row.citations}
        if not sources:
            sources = {result.url: result.title or '' for result in response.results}
        lines = [_truncate(answer, self._max_text_chars)]
        if sources:
            lines.extend(['', 'Sources:'])
            lines.extend(f'- {title or "(untitled)"}: {url}' for url, title in sources.items())
        return '\n'.join(lines)


def _format_result(result: Result, max_text_chars: int) -> str:
    """Render one result as labelled metadata lines followed by the page text."""
    lines = [f'Title: {result.title or "(untitled)"}', f'URL: {result.url}']
    if result.published_date:
        lines.append(f'Published: {result.published_date}')
    if result.author:
        lines.append(f'Author: {result.author}')
    if result.text:
        lines.extend(['', _truncate(result.text, max_text_chars)])
    return '\n'.join(lines)


def _truncate(text: str, max_chars: int) -> str:
    """Cap page text at `max_chars`, keeping the head.

    Unlike shell output, where errors land at the end, a page's lead carries
    the substance, so the head is kept and the tail dropped.
    """
    if len(text) <= max_chars:
        return text
    return f'{text[:max_chars]}\n[... page text truncated at {max_chars} characters]'
