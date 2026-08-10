"""Playwright toolset -- gives an agent a real, stateful Chromium browser.

External-service assumptions (Playwright SDK + bundled Chromium + `playwright` CLI).
These depend on Playwright internals and Chromium packaging that change on the
vendor's schedule and that the mocked tests do not exercise. Re-verify against the
installed package and the linked sources before changing version, selector, or
teardown handling; bump the date when a fact still holds, update code and date
together when it changed. Pinned in the `[playwright]` extra as
`playwright>=1.61.0` (pyproject.toml). Every fact below was verified against
1.61.0, and the signature-checkable ones re-verified against 1.62.0 on 2026-08-10.

- `page.aria_snapshot(mode='ai')` returns an agent-oriented tree whose nodes carry
  `[ref=eN]` handles. `mode` accepts `Literal['ai', 'default'] | None` through
  1.62.0; the `ref` attributes have shipped since Playwright 1.52. Verified
  2026-08-10.
  Source: <https://playwright.dev/python/docs/aria-snapshots>. Re-check:
  `inspect.signature(playwright.async_api.Page.aria_snapshot)` still offers 'ai'.
- The `aria-ref=eN` handles from that snapshot are resolvable by the `aria-ref=`
  selector engine, so they can be passed straight to `page.click` / `page.fill`.
  Verified 2026-07-24 (engine present in the bundled driver `coreBundle.js`).
  Source: <https://playwright.dev/python/docs/other-locators>. Re-check: pass a
  `snapshot` ref back into `click` against a live page, or grep the installed
  driver bundle for `aria-ref`.
- `browser.new_context(service_workers='block')` disables page service workers;
  the option is `Literal['allow', 'block'] | None` through 1.62.0. Verified
  2026-08-10.
  Source: <https://playwright.dev/python/docs/api/class-browsercontext>
  (`serviceWorkers` option). Re-check: inspect the `service_workers` parameter of
  `Browser.new_context`.
- `TargetClosedError` is not re-exported from `playwright.async_api` through
  1.62.0; it lives at `playwright._impl._errors`. A driver-raised instance only
  carries `.name`, so `isinstance` is the reliable discriminator. Verified
  2026-08-10.
  Source: <https://github.com/microsoft/playwright-python> (async_api `__init__`).
  Re-check: `hasattr(playwright.async_api, 'TargetClosedError')` (expect `False`);
  if it becomes `True`, switch to the public import.
- Missing-binary detection uses `chromium.executable_path` plus an on-disk
  `os.path.exists` check; the install command is `python -m playwright install
  chromium`. Verified 2026-07-24. Source:
  <https://playwright.dev/python/docs/browsers#install-browsers>. Re-check:
  confirm `BrowserType.executable_path` exists and `playwright install chromium`
  still fetches the binary.
- Playwright's default action/navigation timeout is 30000ms; `timeout=0` disables
  the deadline. `DEFAULT_TIMEOUT_MS` mirrors this default and the toolset treats 0
  the same way in `_await_with_timeout`. Verified 2026-07-24
  (`DEFAULT_PLAYWRIGHT_TIMEOUT_IN_MILLISECONDS = 30000` in `_impl/_helper.py`).
  Source: <https://playwright.dev/python/docs/api/class-page#page-set-default-timeout>.
  Re-check: grep the installed package for `DEFAULT_PLAYWRIGHT_TIMEOUT`.
"""

from __future__ import annotations

import asyncio
import ipaddress
import json
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, Protocol, TypeVar
from urllib.parse import urlparse

from pydantic_ai.messages import BinaryContent, ToolReturn
from pydantic_ai.tools import AgentDepsT
from pydantic_ai.toolsets import FunctionToolset

try:
    # Import-time gate (mirrors `pydantic_ai_harness.exa._toolset`): importing the
    # capability fails fast with an install hint when the optional dep is absent.
    # `TargetClosedError` is not re-exported from `playwright.async_api`; the
    # `playwright._impl._errors` module documents its own classes as stable public
    # API, and a driver-raised instance only carries `.name` (so isinstance is the
    # reliable discriminator, not the name attribute).
    from playwright._impl._errors import TargetClosedError as TargetClosedError
    from playwright.async_api import Error as _PlaywrightError
    from playwright.async_api import TimeoutError as PlaywrightTimeoutError
    from playwright.async_api import async_playwright as async_playwright

    PlaywrightError = _PlaywrightError
except ImportError as _import_error:  # pragma: no cover
    raise ImportError(
        'playwright is required for PlaywrightBrowser. '
        'Install it with: pip install "pydantic-ai-harness[playwright]"\n'
        'Then run: playwright install chromium'
    ) from _import_error

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Awaitable, Callable

_T = TypeVar('_T')

DEFAULT_MAX_CONTENT_TOKENS: int = 4000
"""Default token budget for textual tool results injected into the agent context."""

DEFAULT_TIMEOUT_MS: int = 30_000
"""Default Playwright navigation/action timeout in milliseconds."""

_CHARS_PER_TOKEN = 4
"""Characters-per-token estimate used to turn a token budget into a character cap."""

_MAX_SCREENSHOT_BYTES = 5_000_000
"""Largest screenshot PNG returned as image content.

Screenshots bypass the textual token budget, and a full-page capture of a long
page can exceed what model providers accept per image (5 MB is the strictest
mainstream limit). An oversized capture becomes a bounded error string instead
of a `BinaryContent` that would fail the next model request and abort the run.
"""


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
    async def wait_for_load_state(
        self, state: Literal['domcontentloaded'], *, timeout: float | None = None
    ) -> None: ...  # pragma: no cover
    async def title(self) -> str: ...  # pragma: no cover
    async def inner_text(self, selector: str, *, timeout: float | None = None) -> str: ...  # pragma: no cover
    async def click(self, selector: str, *, timeout: float | None = None) -> None: ...  # pragma: no cover
    async def fill(self, selector: str, value: str, *, timeout: float | None = None) -> None: ...  # pragma: no cover
    async def screenshot(
        self, *, full_page: bool = False, timeout: float | None = None
    ) -> bytes: ...  # pragma: no cover
    async def evaluate(self, expression: str) -> object: ...  # pragma: no cover
    async def go_back(self, *, timeout: float | None = None) -> object: ...  # pragma: no cover
    async def go_forward(self, *, timeout: float | None = None) -> object: ...  # pragma: no cover
    async def wait_for_selector(self, selector: str, *, timeout: float | None = None) -> object: ...  # pragma: no cover
    async def aria_snapshot(
        self, *, mode: Literal['ai', 'default'] = 'default', timeout: float | None = None
    ) -> str: ...  # pragma: no cover


def _to_idna(host: str) -> str:
    """Return `host` in its ASCII/IDNA form so Unicode and `xn--` spellings compare equal.

    A host that cannot be IDNA-encoded (over-long or empty labels, IP literals)
    falls back to the input unchanged, so IPv4/IPv6 literals are left alone.
    """
    try:
        return host.encode('idna').decode('ascii')
    except UnicodeError:
        return host


def _url_host(url: str) -> str | None:
    """Extract the host that policy checks run against, or `None` when there is none.

    A URL containing a backslash is treated as hostless: WHATWG parsing (what
    Chromium applies) turns a backslash into `/`, so `urlparse` would report a
    host the browser never connects to. A malformed URL that `urlparse` rejects
    is also treated as hostless, so it fails closed instead of crashing the caller.
    """
    if '\\' in url:
        return None
    try:
        return urlparse(url).hostname
    except ValueError:
        return None


def check_allowed_domain(url: str, allowed_domains: list[str] | None) -> bool:
    """Return whether `url`'s host is permitted by the allowlist.

    `allowed_domains=None` means every domain is allowed (open egress). A host
    matches when it equals an allowed entry or is a subdomain of one. `hostname`
    strips the port and brackets, so bracketed IPv6 literals compare correctly.
    Host and entries are normalized to their IDNA/ASCII form before comparison,
    so a Unicode host and its `xn--` spelling get the same verdict. A URL without
    a host (e.g. `about:blank`, `mailto:`) is rejected.
    """
    host = _url_host(url)
    if host is None:
        return False
    if allowed_domains is None:
        return True
    host = _to_idna(host)
    for entry in allowed_domains:
        domain = _to_idna(entry.strip().lower())
        if domain and (host == domain or host.endswith('.' + domain)):
            return True
    return False


def is_blocked_address(host: str) -> bool:
    """Return whether `host` names an address that is not globally routable.

    Covers private (RFC 1918), loopback, link-local (including the cloud
    metadata endpoint), carrier-grade NAT, reserved, and multicast ranges. Only
    IP literals and the loopback hostnames `localhost` / `*.localhost` are
    detected; a public hostname that resolves to a private address is not (DNS
    resolution happens in Chromium after this check -- resolution-based blocking
    is tracked in https://github.com/pydantic/pydantic-ai-harness/issues/415).
    A trailing dot is stripped so the fully-qualified spelling gets the same
    verdict, and an IPv4-mapped IPv6 literal is classified by its embedded IPv4
    address. The named category flags are checked alongside `is_global` because
    older stdlib versions classify some of these ranges as global.
    """
    host = host.lower().rstrip('.')
    if host == 'localhost' or host.endswith('.localhost'):
        return True
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return False
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    return not ip.is_global or ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast


def blocked_navigation_reason(url: str, allowed_domains: list[str] | None, block_private_addresses: bool) -> str | None:
    """Return why navigating to `url` is denied, or `None` when it is permitted.

    The two policies are orthogonal: `check_allowed_domain` applies the opt-in
    allowlist, and `is_blocked_address` refuses private/link-local/metadata IP
    literals even when the allowlist permits the host or no allowlist is set, so
    open egress still cannot reach `http://169.254.169.254/`. The returned phrase
    names which policy denied the URL, so the model and logs can tell an
    allowlist miss from a private-address block.
    """
    host = _url_host(url)
    if host is None or not check_allowed_domain(url, allowed_domains):
        return 'domain not in allowed_domains'
    if block_private_addresses and is_blocked_address(host):
        return 'blocked private or link-local address'
    return None


def refused_in_every_frame(url: str, block_private_addresses: bool) -> bool:
    """Whether `url` must be refused in any frame, not just the top-level document.

    The private-address block is absolute: `snapshot()` reads the ARIA tree of
    cross-origin child frames, so letting a subframe load
    `http://169.254.169.254/` would hand the model the very response body the
    block exists to withhold. The allowlist stays top-level, where a page's own
    third-party frames (identity providers, payment steps) keep working.
    """
    host = _url_host(url)
    return block_private_addresses and host is not None and is_blocked_address(host)


def _truncate(text: str, max_chars: int) -> str:
    """Cap tool output at `max_chars`, keeping the head where the substance sits."""
    if len(text) <= max_chars:
        return text
    marker = f'\n[... tool output truncated at {max_chars} characters]'
    if len(marker) >= max_chars:
        return text[:max_chars]
    return f'{text[: max_chars - len(marker)]}{marker}'


@dataclass
class PlaywrightBrowserState:
    """Per-run browser handles shared between `PlaywrightBrowser` and `PlaywrightBrowserToolset`.

    `PlaywrightBrowser.wrap_run` installs `lazy_launcher`; the first browser-tool
    call triggers it through `ensure_page`, so Chromium starts only when a tool
    is actually used. Each agent run gets a fresh instance (via
    `PlaywrightBrowser.for_run`), so concurrent runs never share a page.
    """

    page: _Page | None = None
    """Active page, or `None` before the browser is launched."""

    launch_error: str | None = None
    """Set when a launch attempt failed (e.g. the Chromium binary is missing)."""

    lazy_launcher: Callable[[], Awaitable[None]] | None = field(default=None, init=False, repr=False)
    """Async launcher installed by `PlaywrightBrowser.wrap_run`; populates `page` on first use."""

    _launch_lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False, repr=False)
    """Serializes the lazy launch so concurrent first tool calls launch Chromium once."""

    async def ensure_page(self) -> _Page:
        """Return the active page, launching Chromium lazily on the first call.

        Tool calls in one model response run concurrently, so the launch is
        serialized: the first caller runs `lazy_launcher` under the lock and the
        rest observe the populated `page` (or the launch error) instead of
        launching a second Chromium.
        """
        if self.launch_error is not None:
            raise RuntimeError(self.launch_error)
        if self.page is None:
            async with self._launch_lock:
                if self.page is None and self.launch_error is None:
                    if self.lazy_launcher is None:
                        raise RuntimeError(
                            'PlaywrightBrowser is not running: PlaywrightBrowser.wrap_run must be active before any '
                            'browser tool.'
                        )
                    await self.lazy_launcher()
            if self.launch_error is not None:
                raise RuntimeError(self.launch_error)
            if self.page is None:
                raise RuntimeError('Browser failed to launch.')  # pragma: no cover
        return self.page


class PlaywrightBrowserToolset(FunctionToolset[AgentDepsT]):
    """Async Playwright-backed browser tools: navigate, interact, extract, screenshot, run JS.

    The tools read the active page from a shared `PlaywrightBrowserState`, which
    `PlaywrightBrowser.wrap_run` populates lazily on the first tool call. Use the
    toolset through `PlaywrightBrowser` rather than directly; construct it
    directly (with a `state` whose `page` you set) only to drive tools against a
    page double.

    Page text is extracted with Playwright itself (`inner_text`), and every textual
    result is truncated to `max_content_tokens`; no HTML-to-Markdown dependency is
    pulled in. `screenshot` returns a `ToolReturn` carrying `BinaryContent` so vision
    models see the image natively instead of a base64 string bloating the text context.
    """

    def __init__(
        self,
        *,
        state: PlaywrightBrowserState,
        allowed_domains: list[str] | None = None,
        block_private_addresses: bool = True,
        screenshot_on_navigate: bool = False,
        max_content_tokens: int = DEFAULT_MAX_CONTENT_TOKENS,
        timeout_ms: int = DEFAULT_TIMEOUT_MS,
    ) -> None:
        if max_content_tokens < 0:
            raise ValueError('max_content_tokens must be greater than or equal to 0')
        if timeout_ms < 0:
            raise ValueError('timeout_ms must be greater than or equal to 0')
        super().__init__(id='playwright')
        self._state = state
        self._allowed_domains = allowed_domains
        self._block_private_addresses = block_private_addresses
        self._screenshot_on_navigate = screenshot_on_navigate
        self._max_content_tokens = max_content_tokens
        self._timeout_ms = timeout_ms
        self._operation_lock = asyncio.Lock()
        self.add_function(self.navigate, name='navigate')
        self.add_function(self.click, name='click')
        self.add_function(self.type_text, name='type_text')
        self.add_function(self.screenshot, name='screenshot')
        self.add_function(self.get_text, name='get_text')
        self.add_function(self.scroll, name='scroll')
        self.add_function(self.go_back, name='go_back')
        self.add_function(self.go_forward, name='go_forward')
        self.add_function(self.execute_js, name='execute_js')
        self.add_function(self.wait_for, name='wait_for')
        self.add_function(self.snapshot, name='snapshot')

    @asynccontextmanager
    async def _serialize_operation(self) -> AsyncGenerator[None]:
        """Prevent concurrent tools from interleaving reads and writes on the shared page."""
        async with self._operation_lock:
            yield

    async def _page_text(self, timeout_ms: int | None = None) -> str:
        """Return the current page's visible text, truncated to the token budget."""
        page = await self._state.ensure_page()
        text = await page.inner_text('body', timeout=timeout_ms)
        return self._truncate_output(text)

    def _truncate_output(self, text: str) -> str:
        """Apply the configured token budget to a model-facing textual result."""
        return _truncate(text, self._max_content_tokens * _CHARS_PER_TOKEN)

    def _playwright_error(self, action: str, exc: PlaywrightError, timeout_ms: int) -> str:
        """Map a Playwright error to a bounded, model-actionable string.

        A timeout, strict-mode match count, `net::ERR_*` code, or closed target is
        a routine event when a model drives a browser, so it is returned as a tool
        result the model can react to rather than raised to abort the run.
        `timeout_ms` is the deadline the call actually ran under, so a per-call
        override is reported accurately.
        """
        if isinstance(exc, PlaywrightTimeoutError):
            return (
                f'Error: {action} timed out after {timeout_ms}ms. The element may not exist or '
                'the page may be slow; try a different selector, or navigate again.'
            )
        if isinstance(exc, TargetClosedError):
            return (
                f'Error: {action} failed: the browser or page was closed unexpectedly. '
                'Browser tools may be unavailable for the rest of this run.'
            )
        return f'Error: {action} failed: {exc}'

    def _resolve_timeout(self, timeout_ms: int | None) -> int:
        """Return the per-call timeout override in milliseconds, or the capability default."""
        if timeout_ms is None:
            return self._timeout_ms
        return timeout_ms

    async def _await_with_timeout(self, awaitable: Awaitable[_T], timeout_ms: int) -> _T:
        """Bound an operation whose Playwright API has no `timeout` parameter."""
        if timeout_ms == 0:
            return await awaitable
        try:
            return await asyncio.wait_for(awaitable, timeout_ms / 1000)
        except asyncio.TimeoutError as exc:
            # asyncio.wait_for raises asyncio.TimeoutError, which is a distinct class
            # from the builtin TimeoutError on Python 3.10 (aliased only from 3.11).
            raise PlaywrightTimeoutError(f'Timeout {timeout_ms}ms exceeded.') from exc

    def _timeout_error(self, timeout_ms: int | None) -> str | None:
        """Return a bounded error when a per-call timeout override is negative, else `None`."""
        if timeout_ms is not None and timeout_ms < 0:
            return self._truncate_output('Error: timeout_ms must be greater than or equal to 0.')
        return None

    def _oversized_screenshot_error(self, png: bytes) -> str | None:
        """Return a bounded error when a capture exceeds the image size limit, else `None`."""
        if len(png) <= _MAX_SCREENSHOT_BYTES:
            return None
        return (
            f'Error: screenshot is {len(png)} bytes, over the {_MAX_SCREENSHOT_BYTES} byte image limit; '
            'capture the viewport (full_page=False) or scroll and capture sections instead.'
        )

    async def _enforce_navigation_policy(self, page: _Page, action: str) -> str | None:
        """After an action, bounce to `about:blank` if the page left the permitted set.

        Navigation can happen through clicks, `execute_js` setting
        `location.href`, or history moves, so the current URL is re-checked --
        against both the allowlist and the private-address block -- after each
        such action. When it is disallowed the page is moved to `about:blank`
        and an error string is returned, so disallowed content never reaches the
        model. The network-level route guard installed by
        `PlaywrightBrowser.wrap_run` is the primary boundary; this is the second
        layer.
        """
        reason = blocked_navigation_reason(page.url, self._allowed_domains, self._block_private_addresses)
        if reason is None:
            return None
        blocked = page.url
        await page.goto('about:blank')
        return self._truncate_output(f'Error: {action} reached a {reason}: {blocked}')

    async def navigate(self, url: str, timeout_ms: int | None = None) -> str | ToolReturn[str]:
        """Navigate to a URL and return the page's title and visible text.

        Args:
            url: Full URL to navigate to (e.g. `https://example.com`).
            timeout_ms: Override the default action timeout for this call, in milliseconds.

        Returns:
            The page URL, title, and visible text. When `screenshot_on_navigate`
            is set, a screenshot is attached as image content for vision models.
        """
        async with self._serialize_operation():
            if (error := self._timeout_error(timeout_ms)) is not None:
                return error
            reason = blocked_navigation_reason(url, self._allowed_domains, self._block_private_addresses)
            if reason is not None:
                return self._truncate_output(f'Error: {reason}: {url}')
            page = await self._state.ensure_page()
            timeout = self._resolve_timeout(timeout_ms)
            try:
                await page.goto(url, timeout=timeout)
                await page.wait_for_load_state('domcontentloaded', timeout=timeout)
                blocked = await self._enforce_navigation_policy(page, 'navigate')
                if blocked is not None:
                    return blocked
                title = await self._await_with_timeout(page.title(), timeout)
                text = await self._page_text(timeout)
                result = self._truncate_output(f'URL: {page.url}\nTitle: {title}\n\n{text}')
                if not self._screenshot_on_navigate:
                    return result
                png = await page.screenshot(timeout=timeout)
                if (oversized := self._oversized_screenshot_error(png)) is not None:
                    return self._truncate_output(f'{result}\n\n{oversized}')
                return ToolReturn(result, content=[BinaryContent(data=png, media_type='image/png')])
            except PlaywrightError as exc:
                return self._truncate_output(self._playwright_error('navigate', exc, timeout))

    async def click(self, selector: str, timeout_ms: int | None = None) -> str:
        """Click an element on the current page.

        Args:
            selector: A CSS selector (e.g. `button#submit`), an `aria-ref=` handle
                from `snapshot` (the most reliable way to target an element), or
                pixel coordinates as `'x,y'` (e.g. `'450,300'`).
            timeout_ms: Override the default action timeout for this call, in milliseconds.

        Returns:
            The page's visible text after the click.
        """
        async with self._serialize_operation():
            if (error := self._timeout_error(timeout_ms)) is not None:
                return error
            page = await self._state.ensure_page()
            parts = selector.split(',', 1)
            coordinates: tuple[int, int] | None = None
            if len(parts) == 2:
                try:
                    coordinates = (int(parts[0]), int(parts[1]))
                except ValueError:
                    pass
            timeout = self._resolve_timeout(timeout_ms)
            try:
                if coordinates is not None:
                    await page.mouse.click(*coordinates)
                else:
                    await page.click(selector, timeout=timeout)
                await page.wait_for_load_state('domcontentloaded', timeout=timeout)
                blocked = await self._enforce_navigation_policy(page, 'click')
                if blocked is not None:
                    return blocked
                text = await self._page_text(timeout)
                return self._truncate_output(f"Clicked '{selector}'. URL: {page.url}\n\n{text}")
            except PlaywrightError as exc:
                return self._truncate_output(self._playwright_error('click', exc, timeout))

    async def type_text(self, selector: str, text: str, timeout_ms: int | None = None) -> str:
        """Type text into an input field, replacing any existing value.

        Args:
            selector: CSS selector for the target input element, or an `aria-ref=`
                handle from `snapshot`.
            text: Text to type into the field.
            timeout_ms: Override the default action timeout for this call, in milliseconds.

        Returns:
            The page's visible text after typing.
        """
        async with self._serialize_operation():
            if (error := self._timeout_error(timeout_ms)) is not None:
                return error
            page = await self._state.ensure_page()
            timeout = self._resolve_timeout(timeout_ms)
            try:
                await page.fill(selector, text, timeout=timeout)
                return self._truncate_output(f"Typed into '{selector}'.\n\n{await self._page_text(timeout)}")
            except PlaywrightError as exc:
                return self._truncate_output(self._playwright_error('type_text', exc, timeout))

    async def screenshot(self, full_page: bool = False, timeout_ms: int | None = None) -> str | ToolReturn[str]:
        """Capture a screenshot of the current page.

        Args:
            full_page: Capture the full scrollable page when `True`, else the
                current viewport.
            timeout_ms: Override the default action timeout for this call, in milliseconds.

        Returns:
            A short note with the page URL, and the PNG as image content so
            vision models can see it.
        """
        async with self._serialize_operation():
            if (error := self._timeout_error(timeout_ms)) is not None:
                return error
            page = await self._state.ensure_page()
            timeout = self._resolve_timeout(timeout_ms)
            try:
                png = await page.screenshot(full_page=full_page, timeout=timeout)
            except PlaywrightError as exc:
                return self._truncate_output(self._playwright_error('screenshot', exc, timeout))
            if (oversized := self._oversized_screenshot_error(png)) is not None:
                return self._truncate_output(oversized)
            return ToolReturn(
                self._truncate_output(f'Screenshot captured. URL: {page.url}'),
                content=[BinaryContent(data=png, media_type='image/png')],
            )

    async def get_text(self, selector: str | None = None, timeout_ms: int | None = None) -> str:
        """Extract text from the page or a specific element.

        Args:
            selector: CSS selector to read. Omit for the full page's visible text.
            timeout_ms: Override the default action timeout for this call, in milliseconds.

        Returns:
            The element's text, or the full page's visible text when no selector
            is given.
        """
        async with self._serialize_operation():
            if (error := self._timeout_error(timeout_ms)) is not None:
                return error
            page = await self._state.ensure_page()
            timeout = self._resolve_timeout(timeout_ms)
            if selector:
                try:
                    text = await page.inner_text(selector, timeout=timeout)
                except Exception as exc:
                    return self._truncate_output(f"Error getting text from '{selector}': {exc}")
                return self._truncate_output(text)
            try:
                return await self._page_text(timeout)
            except PlaywrightError as exc:
                return self._truncate_output(self._playwright_error('get_text', exc, timeout))

    async def scroll(
        self, direction: str, x: int | None = None, y: int | None = None, timeout_ms: int | None = None
    ) -> str:
        """Scroll the page in a direction.

        Args:
            direction: One of `'up'`, `'down'`, `'left'`, `'right'`.
            x: Optional x coordinate to scroll from (paired with `y`).
            y: Optional y coordinate to scroll from (paired with `x`).
            timeout_ms: Override the default action timeout for this call, in milliseconds.

        Returns:
            The page's visible text after scrolling.
        """
        async with self._serialize_operation():
            if (error := self._timeout_error(timeout_ms)) is not None:
                return error
            deltas: dict[str, tuple[int, int]] = {
                'up': (0, -300),
                'down': (0, 300),
                'left': (-300, 0),
                'right': (300, 0),
            }
            delta = deltas.get(direction.lower())
            if delta is None:
                return self._truncate_output(f'Error: invalid direction {direction!r}; use up/down/left/right')
            page = await self._state.ensure_page()
            timeout = self._resolve_timeout(timeout_ms)
            try:
                if x is not None and y is not None:
                    await page.mouse.move(x, y)
                    await page.mouse.wheel(*delta)
                else:
                    # `evaluate` has no `timeout` parameter and hangs if the page's
                    # main thread is blocked, so it is bounded externally.
                    await self._await_with_timeout(page.evaluate(f'window.scrollBy({delta[0]}, {delta[1]})'), timeout)
                return self._truncate_output(f'Scrolled {direction}.\n\n{await self._page_text(timeout)}')
            except PlaywrightError as exc:
                return self._truncate_output(self._playwright_error('scroll', exc, timeout))

    async def go_back(self, timeout_ms: int | None = None) -> str:
        """Navigate back in the browser history.

        Args:
            timeout_ms: Override the default action timeout for this call, in milliseconds.

        Returns:
            The previous page's visible text.
        """
        async with self._serialize_operation():
            if (error := self._timeout_error(timeout_ms)) is not None:
                return error
            page = await self._state.ensure_page()
            timeout = self._resolve_timeout(timeout_ms)
            try:
                response = await page.go_back(timeout=timeout)
                if response is None:
                    return self._truncate_output('No previous page in browser history.')
                await page.wait_for_load_state('domcontentloaded', timeout=timeout)
                blocked = await self._enforce_navigation_policy(page, 'go_back')
                if blocked is not None:
                    return blocked
                return self._truncate_output(f'Went back. URL: {page.url}\n\n{await self._page_text(timeout)}')
            except PlaywrightError as exc:
                return self._truncate_output(self._playwright_error('go_back', exc, timeout))

    async def go_forward(self, timeout_ms: int | None = None) -> str:
        """Navigate forward in the browser history.

        Args:
            timeout_ms: Override the default action timeout for this call, in milliseconds.

        Returns:
            The next page's visible text.
        """
        async with self._serialize_operation():
            if (error := self._timeout_error(timeout_ms)) is not None:
                return error
            page = await self._state.ensure_page()
            timeout = self._resolve_timeout(timeout_ms)
            try:
                response = await page.go_forward(timeout=timeout)
                if response is None:
                    return self._truncate_output('No next page in browser history.')
                await page.wait_for_load_state('domcontentloaded', timeout=timeout)
                blocked = await self._enforce_navigation_policy(page, 'go_forward')
                if blocked is not None:
                    return blocked
                return self._truncate_output(f'Went forward. URL: {page.url}\n\n{await self._page_text(timeout)}')
            except PlaywrightError as exc:
                return self._truncate_output(self._playwright_error('go_forward', exc, timeout))

    async def execute_js(self, script: str, timeout_ms: int | None = None) -> str:
        """Evaluate a JavaScript expression and return its result.

        Args:
            script: JavaScript expression to evaluate, e.g. `document.title`.
            timeout_ms: Override the default action timeout for this call, in milliseconds.

        Returns:
            A string result as-is, objects/arrays as JSON, `null`/`undefined` as
            `'undefined'`, or `JS error: ...` when evaluation raises.
        """
        async with self._serialize_operation():
            if (error := self._timeout_error(timeout_ms)) is not None:
                return error
            page = await self._state.ensure_page()
            timeout = self._resolve_timeout(timeout_ms)
            try:
                # `evaluate` waits for a returned promise and has no `timeout`
                # parameter, so a never-resolving promise (or a blocked main
                # thread) would hold the operation lock forever without the
                # external deadline.
                result = await self._await_with_timeout(page.evaluate(script), timeout)
            except PlaywrightTimeoutError as exc:
                return self._truncate_output(self._playwright_error('execute_js', exc, timeout))
            except Exception as exc:
                return self._truncate_output(f'JS error: {exc}')
            blocked = await self._enforce_navigation_policy(page, 'execute_js')
            if blocked is not None:
                return blocked
            if result is None:
                return self._truncate_output('undefined')
            if isinstance(result, str):
                return self._truncate_output(result)
            try:
                return self._truncate_output(json.dumps(result, default=str))
            except TypeError:  # pragma: no cover
                return self._truncate_output(str(result))

    async def wait_for(
        self, selector: str | None = None, text: str | None = None, timeout_ms: int | None = None
    ) -> str:
        """Wait for dynamic content to appear, then return the page's visible text.

        Pass exactly one of `selector` or `text`. Use this after an action that
        loads content asynchronously, so a following read sees the settled page.

        Args:
            selector: CSS selector (or an `aria-ref=` handle) to wait for.
            text: Visible text to wait for, matched with Playwright's text engine.
            timeout_ms: Override the default action timeout for this call, in milliseconds.

        Returns:
            A short confirmation followed by the page's visible text, or a bounded
            error string when neither/both arguments are given or the wait times out.
        """
        async with self._serialize_operation():
            if (error := self._timeout_error(timeout_ms)) is not None:
                return error
            invalid = 'Error: wait_for requires exactly one of selector or text.'
            if text is not None:
                if selector is not None:
                    return self._truncate_output(invalid)
                query = f'text={text}'
            elif selector is not None:
                query = selector
            else:
                return self._truncate_output(invalid)
            page = await self._state.ensure_page()
            timeout = self._resolve_timeout(timeout_ms)
            try:
                await page.wait_for_selector(query, timeout=timeout)
                return self._truncate_output(f"Found '{query}'.\n\n{await self._page_text(timeout)}")
            except PlaywrightError as exc:
                return self._truncate_output(self._playwright_error('wait_for', exc, timeout))

    async def snapshot(self, timeout_ms: int | None = None) -> str:
        """Return the page's accessibility tree with `aria-ref` handles for targeting.

        The snapshot is the structured, low-cost way to read the page and obtain
        `aria-ref=eN` handles; pass one back to `click` or `type_text` to target an
        element reliably. It includes iframe content that page-level selectors cannot
        reach. Use `screenshot` only for visual checks (charts, layout).

        Args:
            timeout_ms: Override the default action timeout for this call, in milliseconds.

        Returns:
            The accessibility tree (truncated to the token budget), or a bounded
            error string when the snapshot fails.
        """
        async with self._serialize_operation():
            if (error := self._timeout_error(timeout_ms)) is not None:
                return error
            page = await self._state.ensure_page()
            timeout = self._resolve_timeout(timeout_ms)
            try:
                tree = await page.aria_snapshot(mode='ai', timeout=timeout)
            except PlaywrightError as exc:
                return self._truncate_output(self._playwright_error('snapshot', exc, timeout))
            return self._truncate_output(tree)
