"""Browser toolset -- gives an agent a real, stateful Chromium browser via async Playwright."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, Protocol
from urllib.parse import urlparse

from pydantic_ai.messages import BinaryContent, ToolReturn
from pydantic_ai.tools import AgentDepsT
from pydantic_ai.toolsets import FunctionToolset

try:
    # Import-time gate (mirrors `pydantic_ai_harness.exa._toolset`): importing the
    # capability fails fast with an install hint when the optional dep is absent.
    from playwright.async_api import async_playwright as async_playwright
except ImportError as _import_error:  # pragma: no cover
    raise ImportError(
        'playwright is required for Browser. Install it with: pip install "pydantic-ai-harness[browser]"\n'
        'Then run: playwright install chromium'
    ) from _import_error

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

DEFAULT_MAX_CONTENT_TOKENS: int = 4000
"""Default token budget for page text injected into the agent context."""

DEFAULT_TIMEOUT_MS: int = 30_000
"""Default Playwright navigation/action timeout in milliseconds."""

_CHARS_PER_TOKEN = 4
"""Characters-per-token estimate used to turn a token budget into a character cap."""


class _Mouse(Protocol):
    """The subset of `playwright.async_api.Mouse` the toolset drives."""

    async def click(self, x: float, y: float) -> None: ...  # pragma: no cover
    async def move(self, x: float, y: float) -> None: ...  # pragma: no cover
    async def wheel(self, delta_x: float, delta_y: float) -> None: ...  # pragma: no cover


class _Page(Protocol):
    """The subset of `playwright.async_api.Page` the toolset drives.

    A structural type rather than the concrete `Page`: a real Playwright page
    satisfies it, and tests supply an in-memory double with the same surface
    without launching Chromium. Parameter types are subsets of the real
    signatures and return types supersets, so a real `Page` is assignable here.
    """

    @property
    def url(self) -> str: ...  # pragma: no cover
    @property
    def mouse(self) -> _Mouse: ...  # pragma: no cover
    async def goto(self, url: str, *, timeout: float | None = None) -> object: ...  # pragma: no cover
    async def wait_for_load_state(self, state: Literal['domcontentloaded']) -> None: ...  # pragma: no cover
    async def title(self) -> str: ...  # pragma: no cover
    async def inner_text(self, selector: str, *, timeout: float | None = None) -> str: ...  # pragma: no cover
    async def click(self, selector: str, *, timeout: float | None = None) -> None: ...  # pragma: no cover
    async def fill(self, selector: str, value: str, *, timeout: float | None = None) -> None: ...  # pragma: no cover
    async def screenshot(self, *, full_page: bool = False) -> bytes: ...  # pragma: no cover
    async def evaluate(self, expression: str) -> object: ...  # pragma: no cover
    async def go_back(self, *, timeout: float | None = None) -> object: ...  # pragma: no cover
    async def go_forward(self, *, timeout: float | None = None) -> object: ...  # pragma: no cover


def check_allowed_domain(url: str, allowed_domains: list[str] | None) -> bool:
    """Return whether `url`'s host is permitted by the allowlist.

    `allowed_domains=None` means every domain is allowed (open egress). A host
    matches when it equals an allowed entry or is a subdomain of one. The port
    is stripped before comparison. A URL that does not parse is rejected.
    """
    if allowed_domains is None:
        return True
    try:
        domain = urlparse(url).netloc.lower().split(':')[0]
    except ValueError:  # pragma: no cover
        return False
    return any(domain == entry.lower() or domain.endswith('.' + entry.lower()) for entry in allowed_domains)


def _truncate(text: str, max_chars: int) -> str:
    """Cap page text at `max_chars`, keeping the head where the substance sits."""
    if len(text) <= max_chars:
        return text
    return f'{text[:max_chars]}\n[... page text truncated at {max_chars} characters]'


@dataclass
class BrowserState:
    """Per-run browser handles shared between `Browser` and `BrowserToolset`.

    `Browser.wrap_run` installs `lazy_launcher`; the first browser-tool call
    triggers it through `ensure_page`, so Chromium starts only when a tool is
    actually used. Each agent run gets a fresh instance (via `Browser.for_run`),
    so concurrent runs never share a page.
    """

    page: _Page | None = None
    """Active page, or `None` before the browser is launched."""

    launch_error: str | None = None
    """Set when a launch attempt failed (e.g. the Chromium binary is missing)."""

    lazy_launcher: Callable[[], Awaitable[None]] | None = field(default=None, init=False, repr=False)
    """Async launcher installed by `Browser.wrap_run`; populates `page` on first use."""

    async def ensure_page(self) -> _Page:
        """Return the active page, launching Chromium lazily on the first call."""
        if self.launch_error is not None:
            raise RuntimeError(self.launch_error)
        if self.page is None:
            if self.lazy_launcher is None:
                raise RuntimeError('Browser is not running: Browser.wrap_run must be active before any browser tool.')
            await self.lazy_launcher()
            if self.page is None:
                raise RuntimeError(self.launch_error or 'Browser failed to launch.')  # pragma: no branch
        return self.page


class BrowserToolset(FunctionToolset[AgentDepsT]):
    """Async Playwright-backed browser tools: navigate, interact, extract, screenshot, run JS.

    The tools read the active page from a shared `BrowserState`, which
    `Browser.wrap_run` populates lazily on the first tool call. Use the toolset
    through `Browser` rather than directly; construct it directly (with a
    `state` whose `page` you set) only to drive tools against a page double.

    Page text is extracted with Playwright itself (`inner_text`) and truncated to
    `max_content_tokens`; no HTML-to-Markdown dependency is pulled in. `screenshot`
    returns a `ToolReturn` carrying `BinaryContent` so vision models see the image
    natively instead of a base64 string bloating the text context.
    """

    def __init__(
        self,
        *,
        state: BrowserState,
        allowed_domains: list[str] | None = None,
        screenshot_on_navigate: bool = False,
        max_content_tokens: int = DEFAULT_MAX_CONTENT_TOKENS,
        timeout_ms: int = DEFAULT_TIMEOUT_MS,
    ) -> None:
        super().__init__(id='browser')
        self._state = state
        self._allowed_domains = allowed_domains
        self._screenshot_on_navigate = screenshot_on_navigate
        self._max_content_tokens = max_content_tokens
        self._timeout_ms = timeout_ms
        self.add_function(self.navigate, name='navigate')
        self.add_function(self.click, name='click')
        self.add_function(self.type_text, name='type_text')
        self.add_function(self.screenshot, name='screenshot')
        self.add_function(self.get_text, name='get_text')
        self.add_function(self.scroll, name='scroll')
        self.add_function(self.go_back, name='go_back')
        self.add_function(self.go_forward, name='go_forward')
        self.add_function(self.execute_js, name='execute_js')

    async def _page_text(self) -> str:
        """Return the current page's visible text, truncated to the token budget."""
        page = await self._state.ensure_page()
        text = await page.inner_text('body')
        return _truncate(text, self._max_content_tokens * _CHARS_PER_TOKEN)

    async def _enforce_allowed_domain(self, page: _Page, action: str) -> str | None:
        """After an action, bounce to `about:blank` if the page left the allowlist.

        Navigation can happen through clicks, `execute_js` setting
        `location.href`, or history moves, so the current URL is re-checked after
        each such action. When it is disallowed the page is moved to `about:blank`
        and an error string is returned, so disallowed content never reaches the
        model. The network-level route guard installed by `Browser.wrap_run` is
        the primary boundary; this is the second layer.
        """
        if check_allowed_domain(page.url, self._allowed_domains):
            return None
        blocked = page.url
        await page.goto('about:blank')
        return f'Error: {action} reached a domain not in allowed_domains: {blocked}'

    async def navigate(self, url: str) -> str | ToolReturn[str]:
        """Navigate to a URL and return the page's title and visible text.

        Args:
            url: Full URL to navigate to (e.g. `https://example.com`).

        Returns:
            The page URL, title, and visible text. When `screenshot_on_navigate`
            is set, a screenshot is attached as image content for vision models.
        """
        if not check_allowed_domain(url, self._allowed_domains):
            return f'Error: domain not in allowed_domains: {url}'
        page = await self._state.ensure_page()
        await page.goto(url, timeout=self._timeout_ms)
        await page.wait_for_load_state('domcontentloaded')
        blocked = await self._enforce_allowed_domain(page, 'navigate')
        if blocked is not None:
            return blocked
        title = await page.title()
        text = await self._page_text()
        result = f'URL: {page.url}\nTitle: {title}\n\n{text}'
        if not self._screenshot_on_navigate:
            return result
        png = await page.screenshot()
        return ToolReturn(result, content=[BinaryContent(data=png, media_type='image/png')])

    async def click(self, selector: str) -> str:
        """Click an element on the current page.

        Args:
            selector: A CSS selector (e.g. `button#submit`) or pixel coordinates
                as `'x,y'` (e.g. `'450,300'`).

        Returns:
            The page's visible text after the click.
        """
        page = await self._state.ensure_page()
        parts = selector.split(',', 1)
        if len(parts) == 2 and all(part.strip().lstrip('-').isdigit() for part in parts):
            await page.mouse.click(int(parts[0]), int(parts[1]))
        else:
            await page.click(selector, timeout=self._timeout_ms)
        await page.wait_for_load_state('domcontentloaded')
        blocked = await self._enforce_allowed_domain(page, 'click')
        if blocked is not None:
            return blocked
        return f"Clicked '{selector}'. URL: {page.url}\n\n{await self._page_text()}"

    async def type_text(self, selector: str, text: str) -> str:
        """Type text into an input field, replacing any existing value.

        Args:
            selector: CSS selector for the target input element.
            text: Text to type into the field.

        Returns:
            The page's visible text after typing.
        """
        page = await self._state.ensure_page()
        await page.fill(selector, text, timeout=self._timeout_ms)
        return f"Typed into '{selector}'.\n\n{await self._page_text()}"

    async def screenshot(self, full_page: bool = False) -> ToolReturn[str]:
        """Capture a screenshot of the current page.

        Args:
            full_page: Capture the full scrollable page when `True`, else the
                current viewport.

        Returns:
            A short note with the page URL, and the PNG as image content so
            vision models can see it.
        """
        page = await self._state.ensure_page()
        png = await page.screenshot(full_page=full_page)
        return ToolReturn(
            f'Screenshot captured. URL: {page.url}',
            content=[BinaryContent(data=png, media_type='image/png')],
        )

    async def get_text(self, selector: str | None = None) -> str:
        """Extract text from the page or a specific element.

        Args:
            selector: CSS selector to read. Omit for the full page's visible text.

        Returns:
            The element's text, or the full page's visible text when no selector
            is given.
        """
        page = await self._state.ensure_page()
        if selector:
            try:
                text = await page.inner_text(selector, timeout=self._timeout_ms)
            except Exception as exc:
                return f"Error getting text from '{selector}': {exc}"
            return _truncate(text, self._max_content_tokens * _CHARS_PER_TOKEN)
        return await self._page_text()

    async def scroll(self, direction: str, x: int | None = None, y: int | None = None) -> str:
        """Scroll the page in a direction.

        Args:
            direction: One of `'up'`, `'down'`, `'left'`, `'right'`.
            x: Optional x coordinate to scroll from (paired with `y`).
            y: Optional y coordinate to scroll from (paired with `x`).

        Returns:
            The page's visible text after scrolling.
        """
        deltas: dict[str, tuple[int, int]] = {'up': (0, -300), 'down': (0, 300), 'left': (-300, 0), 'right': (300, 0)}
        delta = deltas.get(direction.lower())
        if delta is None:
            return f'Error: invalid direction {direction!r}; use up/down/left/right'
        page = await self._state.ensure_page()
        if x is not None and y is not None:
            await page.mouse.move(x, y)
            await page.mouse.wheel(*delta)
        else:
            await page.evaluate(f'window.scrollBy({delta[0]}, {delta[1]})')
        return f'Scrolled {direction}.\n\n{await self._page_text()}'

    async def go_back(self) -> str:
        """Navigate back in the browser history.

        Returns:
            The previous page's visible text.
        """
        page = await self._state.ensure_page()
        await page.go_back(timeout=self._timeout_ms)
        await page.wait_for_load_state('domcontentloaded')
        blocked = await self._enforce_allowed_domain(page, 'go_back')
        if blocked is not None:
            return blocked
        return f'Went back. URL: {page.url}\n\n{await self._page_text()}'

    async def go_forward(self) -> str:
        """Navigate forward in the browser history.

        Returns:
            The next page's visible text.
        """
        page = await self._state.ensure_page()
        await page.go_forward(timeout=self._timeout_ms)
        await page.wait_for_load_state('domcontentloaded')
        blocked = await self._enforce_allowed_domain(page, 'go_forward')
        if blocked is not None:
            return blocked
        return f'Went forward. URL: {page.url}\n\n{await self._page_text()}'

    async def execute_js(self, script: str) -> str:
        """Evaluate a JavaScript expression and return its result.

        Args:
            script: JavaScript expression to evaluate, e.g. `document.title`.

        Returns:
            A string result as-is, objects/arrays as JSON, `null`/`undefined` as
            `'undefined'`, or `JS error: ...` when evaluation raises.
        """
        page = await self._state.ensure_page()
        try:
            result = await page.evaluate(script)
        except Exception as exc:
            return f'JS error: {exc}'
        blocked = await self._enforce_allowed_domain(page, 'execute_js')
        if blocked is not None:
            return blocked
        if result is None:
            return 'undefined'
        if isinstance(result, str):
            return result
        try:
            return json.dumps(result, default=str)
        except TypeError:  # pragma: no cover
            return str(result)
