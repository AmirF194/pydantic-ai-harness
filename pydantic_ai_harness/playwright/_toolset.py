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
- `new_context` accepts downloads by default; `accept_downloads=False` refuses
  them, so a page cannot write attachments to the host's temporary storage.
  Verified 2026-08-11 (parameter documented as "Defaults to `true` where all the
  downloads are accepted"). Source:
  <https://playwright.dev/python/docs/api/class-browser#browser-new-context>.
  Re-check: `inspect.getdoc(Browser.new_context)` still states that default.
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
import os
import sys
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Protocol, TypeVar
from urllib.parse import urlparse

from pydantic_ai.messages import BinaryContent, ToolReturn
from pydantic_ai.tools import AgentDepsT
from pydantic_ai.toolsets import FunctionToolset
from typing_extensions import Self

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
    from collections.abc import Awaitable, Callable

    from playwright.async_api import Browser as PlaywrightBrowserHandle
    from playwright.async_api import Page as PlaywrightPage
    from playwright.async_api import Playwright as PlaywrightDriver
    from playwright.async_api import Request as PlaywrightRequest
    from playwright.async_api import Route as PlaywrightRoute
    from playwright.async_api import StorageState

_T = TypeVar('_T')

DEFAULT_MAX_CONTENT_TOKENS: int = 4000
"""Default token budget for textual tool results injected into the agent context."""

DEFAULT_TIMEOUT_MS: int = 30_000
"""Default Playwright navigation/action timeout in milliseconds."""

_CHARS_PER_TOKEN = 4
"""Characters-per-token estimate used to turn a token budget into a character cap."""

_BLANK_PAGE = 'about:blank'
"""The blank page a context starts on, and where a disallowed navigation is sent."""

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
    falls back to the input unchanged, so IPv4/IPv6 literals are left alone. The
    trailing dot of a fully-qualified spelling is dropped first: it names the DNS
    root rather than a label, and `encode('idna')` rejects the empty label it
    produces, which would otherwise deny `example.com.` against an `example.com`
    allowlist entry.
    """
    host = host.rstrip('.')
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

    `about:blank` is permitted: it is the state a context starts in and the target
    this module bounces to, so denying it would refuse every tool call made before
    the first navigation. Every other hostless URL (`file:`, `mailto:`, `data:`)
    stays denied, under a phrase that does not blame an allowlist that may not exist.
    """
    if url == _BLANK_PAGE:
        return None
    host = _url_host(url)
    if host is None:
        return 'URL with no host'
    if not check_allowed_domain(url, allowed_domains):
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


@dataclass(frozen=True)
class NavigationPolicy:
    """What the agent is allowed to reach, and how to say so to the model.

    The two axes are independent and deny wins: an allowlisted private address is
    still refused until `block_private_addresses` is turned off. Holding them
    together keeps enforcement and description from drifting apart -- the failure
    mode being that the instructions promise the model a reach the guards do not
    grant.
    """

    allowed_domains: list[str] | None = None
    block_private_addresses: bool = True

    def blocked_reason(self, url: str) -> str | None:
        """Why navigating to `url` is denied, or `None` when it is permitted."""
        return blocked_navigation_reason(url, self.allowed_domains, self.block_private_addresses)

    def refused_in_every_frame(self, url: str) -> bool:
        """Whether `url` must be refused in any frame and for any resource type."""
        return refused_in_every_frame(url, self.block_private_addresses)

    def enforced(self) -> bool:
        """Whether either axis restricts anything, i.e. whether a route guard is worth installing."""
        return self.allowed_domains is not None or self.block_private_addresses

    def describe(self) -> str:
        """The reach, phrased for the model's instructions."""
        if self.allowed_domains is None:
            domains = 'all'
        elif self.allowed_domains:
            domains = ', '.join(self.allowed_domains)
        else:
            domains = 'none'
        if self.block_private_addresses:
            domains += ' (private/internal addresses blocked)'
        return domains


def _truncate(text: str, max_chars: int) -> str:
    """Cap tool output at `max_chars`, keeping the head where the substance sits."""
    if len(text) <= max_chars:
        return text
    marker = f'\n[... tool output truncated at {max_chars} characters]'
    if len(marker) >= max_chars:
        return text[:max_chars]
    return f'{text[: max_chars - len(marker)]}{marker}'


_CHROMIUM_MISSING_MESSAGE = (
    'Chromium is not installed. Run `playwright install chromium` (on a fresh Linux or CI image use '
    '`playwright install --with-deps chromium` to also install the required system libraries) and restart '
    'the agent to enable browser tools.'
)


async def _auto_install_chromium() -> str | None:
    """Run `playwright install chromium` in this interpreter; `None` on success, else the installer output.

    Only invoked when `auto_install_chromium=True` and the binary is missing. It
    shells out to a subprocess and downloads a browser, so it runs outside the
    mocked test surface. On failure the merged stdout/stderr is returned so the
    launch path can surface why the install failed instead of the generic hint.
    """
    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        '-m',
        'playwright',
        'install',
        'chromium',
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    try:
        stdout, _ = await proc.communicate()
    except asyncio.CancelledError:
        proc.terminate()
        await proc.wait()
        raise
    if proc.returncode == 0:
        return None
    return stdout.decode(errors='replace')


class PlaywrightBrowserSession:
    """One agent run's Chromium: how a page is obtained, guarded, and released.

    Entering the session arms it; nothing starts until `ensure_page` is first
    awaited, so a run that never calls a browser tool never launches a browser.
    Exiting closes whatever was started, in the order that a half-built session
    still tears down cleanly.

    `PlaywrightBrowser` creates one per run (via `for_run`), so concurrent runs
    never share a page. It can also be driven directly:

    ```python
    async with PlaywrightBrowserSession(policy=NavigationPolicy()) as session:
        page = await session.ensure_page()
    ```
    """

    def __init__(
        self,
        *,
        policy: NavigationPolicy | None = None,
        headless: bool = True,
        storage_state: StorageState | None = None,
        cdp_url: str | None = None,
        auto_install_chromium: bool = False,
        launch_timeout_ms: int = DEFAULT_TIMEOUT_MS,
    ) -> None:
        self._policy = policy if policy is not None else NavigationPolicy()
        self._headless = headless
        self._storage_state = storage_state
        self._cdp_url = cdp_url
        self._auto_install_chromium = auto_install_chromium
        self._launch_timeout_ms = launch_timeout_ms
        self.page: _Page | None = None
        """Active page, or `None` before the browser is launched."""
        self.launch_error: str | None = None
        """Set when a launch attempt failed (e.g. the Chromium binary is missing)."""
        self._driver_cm: AbstractAsyncContextManager[PlaywrightDriver] | None = None
        self._driver_entered = False
        self._browser: PlaywrightBrowserHandle | None = None
        self._popup_tasks: set[asyncio.Task[None]] = set()
        self._launch_lock = asyncio.Lock()

    async def ensure_page(self) -> _Page:
        """Return the active page, launching Chromium lazily on the first call.

        Tool calls in one model response run concurrently, so the launch is
        serialized: the first caller runs it under the lock and the rest observe
        the populated `page` (or the launch error) instead of launching a second
        Chromium.
        """
        if self.launch_error is not None:
            raise RuntimeError(self.launch_error)
        if self.page is None:
            async with self._launch_lock:
                if self.page is None and self.launch_error is None:
                    if self._driver_cm is None:
                        raise RuntimeError(
                            'PlaywrightBrowser is not running: PlaywrightBrowser.wrap_run must be active before any '
                            'browser tool.'
                        )
                    await self._launch()
            if self.launch_error is not None:
                raise RuntimeError(self.launch_error)
            if self.page is None:
                raise RuntimeError('Browser failed to launch.')  # pragma: no cover
        return self.page

    async def _launch(self) -> None:
        """Start the driver and Chromium, then open the guarded page."""
        assert self._driver_cm is not None
        pw = await self._driver_cm.__aenter__()
        self._driver_entered = True
        browser = await self._connect(pw)
        if browser is None:
            return
        # Assigned before the context and page are built, so teardown closes Chromium
        # even when a later setup step raises.
        self._browser = browser
        # Service workers can issue requests that context routes never see, so they
        # are blocked to keep all traffic on the routable path. Downloads are refused
        # because no tool exposes them: accepting them only lets a page write to the
        # host's temporary storage for the length of the run.
        context = await browser.new_context(
            storage_state=self._storage_state,
            service_workers='block',
            accept_downloads=False,
        )
        page = await context.new_page()
        if self._policy.enforced():
            await context.route('**/*', self._route_guard)
        page.on('popup', self._on_popup)
        self.page = page

    async def _connect(self, pw: PlaywrightDriver) -> PlaywrightBrowserHandle | None:
        """Attach to or start a browser, or record why none is available.

        Both paths are bounded by `launch_timeout_ms`: a `cdp_url` pointing at an
        unresponsive endpoint would otherwise hold the operation lock with no
        deadline at all. The auto-install download is deliberately left unbounded,
        since a first-run browser fetch legitimately outlasts an action timeout.
        """
        if self._cdp_url is not None:
            return await pw.chromium.connect_over_cdp(self._cdp_url, timeout=self._launch_timeout_ms)
        if os.path.exists(pw.chromium.executable_path):
            return await pw.chromium.launch(headless=self._headless, timeout=self._launch_timeout_ms)
        # Binary genuinely absent: raise a clear install hint, or fetch it when
        # opted in. A launch failure with the binary present (sandbox, missing
        # system libs, no display) is left to surface as its own error rather than
        # being masked as "Chromium is not installed".
        browser = await self._install_and_retry(pw) if self._auto_install_chromium else None
        if browser is None:  # pragma: no branch
            if self.launch_error is None:
                self.launch_error = _CHROMIUM_MISSING_MESSAGE
        return browser

    async def _install_and_retry(self, pw: PlaywrightDriver) -> PlaywrightBrowserHandle | None:
        """Fetch Chromium and relaunch once.

        Returns the browser on success. On install failure it records a launch
        error carrying a bounded tail of the installer output (so the failure is
        diagnosable) and returns `None`.
        """
        install_output = await _auto_install_chromium()
        if install_output is None:
            return await pw.chromium.launch(  # pragma: no cover
                headless=self._headless, timeout=self._launch_timeout_ms
            )
        self.launch_error = f'{_CHROMIUM_MISSING_MESSAGE}\nAuto-install failed:\n{install_output[-300:]}'
        return None

    async def _route_guard(self, route: PlaywrightRoute, request: PlaywrightRequest) -> None:
        """Network-layer egress policy: abort what the configuration refuses, pass the rest.

        The two policies have different reach. The private-address block covers
        every request of every resource type in every frame, because a page can
        read a subresource it fetched itself and hand the body to the model. The
        allowlist covers top-level navigation only, so a permitted page keeps its
        own CDN assets, identity-provider frames, and payment steps.
        """
        if self._policy.refused_in_every_frame(request.url):
            await route.abort()
            return
        permitted = self._policy.blocked_reason(request.url) is None
        if not request.is_navigation_request() or permitted:
            await route.continue_()
            return
        try:
            frame = request.frame
        except PlaywrightError:
            await route.abort()
            return
        if frame == frame.page.main_frame:
            await route.abort()
            return
        await route.continue_()

    async def _handle_popup(self, popup: PlaywrightPage) -> None:
        """Keep the browser single-tab by closing popup pages."""
        await popup.close()

    def _popup_done(self, task: asyncio.Task[None]) -> None:
        """Release a finished popup task and retrieve any close error."""
        self._popup_tasks.discard(task)
        if not task.cancelled():
            task.exception()

    def _on_popup(self, popup: PlaywrightPage) -> None:
        """Schedule popup handling, keeping a strong task reference until it finishes."""
        task = asyncio.create_task(self._handle_popup(popup))
        self._popup_tasks.add(task)
        task.add_done_callback(self._popup_done)

    async def __aenter__(self) -> Self:
        """Arm the session. Cheap by design: no driver and no browser start here."""
        self._driver_cm = async_playwright()
        return self

    async def __aexit__(self, exc_type: type[BaseException] | None, *_: object) -> None:
        """Release everything the session started, without masking the run's own failure.

        Popup handlers are cancelled first so none outlives the browser they act
        on. A `close()` failure still exits the driver, and when the run is
        already unwinding the teardown error is dropped: the exception the caller
        is carrying is the one worth reporting.
        """
        driver_cm = self._driver_cm
        self._driver_cm = None
        if self._popup_tasks:
            for task in self._popup_tasks:
                task.cancel()
            await asyncio.gather(*self._popup_tasks, return_exceptions=True)
            self._popup_tasks.clear()
        if driver_cm is None:  # pragma: no cover
            return
        try:
            browser = self._browser
            if browser is not None:
                self.page = None
                self._browser = None
                try:
                    await browser.close()
                finally:
                    await driver_cm.__aexit__(None, None, None)
            elif self._driver_entered:
                await driver_cm.__aexit__(None, None, None)
        except Exception:
            if exc_type is None:
                raise


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
        session: PlaywrightBrowserSession,
        policy: NavigationPolicy | None = None,
        screenshot_on_navigate: bool = False,
        max_content_tokens: int = DEFAULT_MAX_CONTENT_TOKENS,
        timeout_ms: int = DEFAULT_TIMEOUT_MS,
    ) -> None:
        if max_content_tokens < 0:
            raise ValueError('max_content_tokens must be greater than or equal to 0')
        if timeout_ms < 0:
            raise ValueError('timeout_ms must be greater than or equal to 0')
        super().__init__(id='playwright')
        self._session = session
        self._policy = policy if policy is not None else NavigationPolicy()
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

    async def _page_text(self, page: _Page, timeout_ms: int | None = None) -> str:
        """Return `page`'s visible text, truncated to the token budget.

        The page is passed in rather than re-acquired: every caller already holds
        the one it just acted on, and re-entering `ensure_page` here would be a
        second path into the launch machinery, raising `RuntimeError` where the
        callers only guard against `PlaywrightError`.
        """
        text = await page.inner_text('body', timeout=timeout_ms)
        return self._truncate_output(text)

    def _truncate_output(self, text: str) -> str:
        """Apply the configured token budget to a model-facing textual result."""
        return _truncate(text, self._max_content_tokens * _CHARS_PER_TOKEN)

    def _truncate_output_keeping(self, text: str, note: str) -> str:
        """Bound `text` and `note` together, giving `note` its room first.

        Appending to an already-budgeted string and re-truncating drops the
        appended part, which is the opposite of what a note reporting a dropped
        result should do. A note too large for the budget on its own wins the
        whole budget: it carries why the result is missing.
        """
        budget = self._max_content_tokens * _CHARS_PER_TOKEN
        separator = '\n\n'
        room = budget - len(note) - len(separator)
        if room <= 0:
            return _truncate(note, budget)
        return f'{_truncate(text, room)}{separator}{note}'

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
        """Return a bounded error when a per-call timeout override is not positive, else `None`.

        `0` means "no deadline" to both Playwright and `_await_with_timeout`, so it
        stays available to the developer as the capability default but is refused as
        a per-call override: the model chooses that argument, and an injected page
        could otherwise ask for an unbounded call that never returns.
        """
        if timeout_ms is not None and timeout_ms <= 0:
            return self._truncate_output('Error: timeout_ms must be greater than 0.')
        return None

    def _oversized_screenshot_error(self, png: bytes) -> str | None:
        """Return a bounded error when a capture exceeds the image size limit, else `None`."""
        if len(png) <= _MAX_SCREENSHOT_BYTES:
            return None
        return (
            f'Error: screenshot is {len(png)} bytes, over the {_MAX_SCREENSHOT_BYTES} byte image limit; '
            'capture the viewport (full_page=False) or scroll and capture sections instead.'
        )

    async def _enforce_navigation_policy(self, page: _Page, action: str, timeout: int) -> str | None:
        """After an action, bounce to `about:blank` if the page left the permitted set.

        Navigation can happen through clicks, `execute_js` setting
        `location.href`, or history moves, so the current URL is re-checked --
        against both the allowlist and the private-address block -- after each
        such action. When it is disallowed the page is moved to `about:blank`
        and an error string is returned, so disallowed content never reaches the
        model. The network-level route guard installed by
        `PlaywrightBrowser.wrap_run` is the primary boundary; this is the second
        layer.

        The bounce runs under the caller's resolved `timeout` so a short
        per-call `timeout_ms` is not silently replaced by Playwright's default.
        """
        reason = self._policy.blocked_reason(page.url)
        if reason is None:
            return None
        blocked = page.url
        await page.goto(_BLANK_PAGE, timeout=timeout)
        return self._truncate_output(f'Error: {action} reached a {reason}: {blocked}')

    async def _in_operation(
        self, action: str, timeout_ms: int | None, body: Callable[[_Page, int], Awaitable[_T]]
    ) -> _T | str:
        """Run `body` as one complete operation on the shared page.

        Every tool needs the same five things in the same order: exclusive use of
        the page, a per-call deadline that is either absent or positive, the page
        itself (launching Chromium on the first call), the resolved deadline, and
        a Playwright failure turned into a result the model can read instead of an
        exception that ends the run.

        Acquiring the page is inside the guarded region: starting or attaching to
        a browser can fail the same way an action can (a `cdp_url` whose endpoint
        is gone, most realistically), and the model can act on that only if it
        arrives as a result rather than as an exception that ends the run.

        An argument check that must not launch a browser belongs in the tool,
        before this call -- see `_refuse`.
        """
        async with self._operation_lock:
            if (error := self._timeout_error(timeout_ms)) is not None:
                return error
            timeout = self._resolve_timeout(timeout_ms)
            try:
                page = await self._session.ensure_page()
                return await body(page, timeout)
            except PlaywrightError as exc:
                return self._truncate_output(self._playwright_error(action, exc, timeout))

    async def _refuse(self, timeout_ms: int | None, message: str) -> str:
        """Return a bounded refusal without acquiring a page.

        A rejected argument must not start a browser, so these refusals happen
        before `_in_operation`. The deadline is still validated first, so a call
        that is wrong in both ways reports the same error either way.
        """
        async with self._operation_lock:
            if (error := self._timeout_error(timeout_ms)) is not None:
                return error
            return self._truncate_output(message)

    async def _settle(self, page: _Page, action: str, timeout: int) -> str | None:
        """Let the navigation finish, then re-check where it landed.

        The order is the point: the policy has to see the settled URL, because
        reading it before the load completes checks the page the action started
        from. Returns the bounced error, or `None` when the result is permitted.
        """
        await page.wait_for_load_state('domcontentloaded', timeout=timeout)
        return await self._enforce_navigation_policy(page, action, timeout)

    async def navigate(self, url: str, timeout_ms: int | None = None) -> str | ToolReturn[str]:
        """Navigate to a URL and return the page's title and visible text.

        Args:
            url: Full URL to navigate to (e.g. `https://example.com`).
            timeout_ms: Override the default action timeout for this call, in milliseconds.
                Must be greater than 0.

        Returns:
            The page URL, title, and visible text. When `screenshot_on_navigate`
            is set, a screenshot is attached as image content for vision models.
        """
        reason = self._policy.blocked_reason(url)
        if reason is not None:
            # Refused before `_in_operation`, so a disallowed URL never launches Chromium.
            return await self._refuse(timeout_ms, f'Error: {reason}: {url}')

        async def _navigate(page: _Page, timeout: int) -> str | ToolReturn[str]:
            await page.goto(url, timeout=timeout)
            if (blocked := await self._settle(page, 'navigate', timeout)) is not None:
                return blocked
            title = await self._await_with_timeout(page.title(), timeout)
            text = await self._page_text(page, timeout)
            result = self._truncate_output(f'URL: {page.url}\nTitle: {title}\n\n{text}')
            if not self._screenshot_on_navigate:
                return result
            png = await page.screenshot(timeout=timeout)
            if (oversized := self._oversized_screenshot_error(png)) is not None:
                return self._truncate_output_keeping(result, oversized)
            return ToolReturn(result, content=[BinaryContent(data=png, media_type='image/png')])

        return await self._in_operation('navigate', timeout_ms, _navigate)

    async def click(self, selector: str, timeout_ms: int | None = None) -> str:
        """Click an element on the current page.

        Args:
            selector: A CSS selector (e.g. `button#submit`), an `aria-ref=` handle
                from `snapshot` (the most reliable way to target an element), or
                pixel coordinates as `'x,y'` (e.g. `'450,300'`).
            timeout_ms: Override the default action timeout for this call, in milliseconds.
                Must be greater than 0.

        Returns:
            The page's visible text after the click.
        """
        parts = selector.split(',', 1)
        coordinates: tuple[int, int] | None = None
        if len(parts) == 2:
            try:
                coordinates = (int(parts[0]), int(parts[1]))
            except ValueError:
                pass

        async def _click(page: _Page, timeout: int) -> str:
            if coordinates is not None:
                await page.mouse.click(*coordinates)
            else:
                await page.click(selector, timeout=timeout)
            if (blocked := await self._settle(page, 'click', timeout)) is not None:
                return blocked
            text = await self._page_text(page, timeout)
            return self._truncate_output(f"Clicked '{selector}'. URL: {page.url}\n\n{text}")

        return await self._in_operation('click', timeout_ms, _click)

    async def type_text(self, selector: str, text: str, timeout_ms: int | None = None) -> str:
        """Type text into an input field, replacing any existing value.

        Args:
            selector: CSS selector for the target input element, or an `aria-ref=`
                handle from `snapshot`.
            text: Text to type into the field.
            timeout_ms: Override the default action timeout for this call, in milliseconds.
                Must be greater than 0.

        Returns:
            The page's visible text after typing.
        """

        async def _type_text(page: _Page, timeout: int) -> str:
            await page.fill(selector, text, timeout=timeout)
            return self._truncate_output(f"Typed into '{selector}'.\n\n{await self._page_text(page, timeout)}")

        return await self._in_operation('type_text', timeout_ms, _type_text)

    async def screenshot(self, full_page: bool = False, timeout_ms: int | None = None) -> str | ToolReturn[str]:
        """Capture a screenshot of the current page.

        Args:
            full_page: Capture the full scrollable page when `True`, else the
                current viewport.
            timeout_ms: Override the default action timeout for this call, in milliseconds.
                Must be greater than 0.

        Returns:
            A short note with the page URL, and the PNG as image content so
            vision models can see it.
        """

        async def _screenshot(page: _Page, timeout: int) -> str | ToolReturn[str]:
            png = await page.screenshot(full_page=full_page, timeout=timeout)
            if (oversized := self._oversized_screenshot_error(png)) is not None:
                return self._truncate_output(oversized)
            return ToolReturn(
                self._truncate_output(f'Screenshot captured. URL: {page.url}'),
                content=[BinaryContent(data=png, media_type='image/png')],
            )

        return await self._in_operation('screenshot', timeout_ms, _screenshot)

    async def get_text(self, selector: str | None = None, timeout_ms: int | None = None) -> str:
        """Extract text from the page or a specific element.

        Args:
            selector: CSS selector to read. Omit for the full page's visible text.
            timeout_ms: Override the default action timeout for this call, in milliseconds.
                Must be greater than 0.

        Returns:
            The element's text, or the full page's visible text when no selector
            is given.
        """

        async def _get_text(page: _Page, timeout: int) -> str:
            if not selector:
                return await self._page_text(page, timeout)
            try:
                text = await page.inner_text(selector, timeout=timeout)
            except Exception as exc:
                # Named after the selector rather than the action: which selector failed
                # is the part the model needs to act on.
                return self._truncate_output(f"Error getting text from '{selector}': {exc}")
            return self._truncate_output(text)

        return await self._in_operation('get_text', timeout_ms, _get_text)

    async def scroll(
        self, direction: str, x: int | None = None, y: int | None = None, timeout_ms: int | None = None
    ) -> str:
        """Scroll the page in a direction.

        Args:
            direction: One of `'up'`, `'down'`, `'left'`, `'right'`.
            x: Optional x coordinate to scroll from (paired with `y`).
            y: Optional y coordinate to scroll from (paired with `x`).
            timeout_ms: Override the default action timeout for this call, in milliseconds.
                Must be greater than 0.

        Returns:
            The page's visible text after scrolling.
        """
        deltas: dict[str, tuple[int, int]] = {
            'up': (0, -300),
            'down': (0, 300),
            'left': (-300, 0),
            'right': (300, 0),
        }
        delta = deltas.get(direction.lower())
        if delta is None:
            return await self._refuse(timeout_ms, f'Error: invalid direction {direction!r}; use up/down/left/right')

        async def _scroll(page: _Page, timeout: int) -> str:
            if x is not None and y is not None:
                await page.mouse.move(x, y)
                await page.mouse.wheel(*delta)
            else:
                # `evaluate` has no `timeout` parameter and hangs if the page's
                # main thread is blocked, so it is bounded externally.
                await self._await_with_timeout(page.evaluate(f'window.scrollBy({delta[0]}, {delta[1]})'), timeout)
            return self._truncate_output(f'Scrolled {direction}.\n\n{await self._page_text(page, timeout)}')

        return await self._in_operation('scroll', timeout_ms, _scroll)

    async def go_back(self, timeout_ms: int | None = None) -> str:
        """Navigate back in the browser history.

        Args:
            timeout_ms: Override the default action timeout for this call, in milliseconds.
                Must be greater than 0.

        Returns:
            The previous page's visible text.
        """

        async def _go_back(page: _Page, timeout: int) -> str:
            response = await page.go_back(timeout=timeout)
            if response is None:
                return self._truncate_output('No previous page in browser history.')
            if (blocked := await self._settle(page, 'go_back', timeout)) is not None:
                return blocked
            return self._truncate_output(f'Went back. URL: {page.url}\n\n{await self._page_text(page, timeout)}')

        return await self._in_operation('go_back', timeout_ms, _go_back)

    async def go_forward(self, timeout_ms: int | None = None) -> str:
        """Navigate forward in the browser history.

        Args:
            timeout_ms: Override the default action timeout for this call, in milliseconds.
                Must be greater than 0.

        Returns:
            The next page's visible text.
        """

        async def _go_forward(page: _Page, timeout: int) -> str:
            response = await page.go_forward(timeout=timeout)
            if response is None:
                return self._truncate_output('No next page in browser history.')
            if (blocked := await self._settle(page, 'go_forward', timeout)) is not None:
                return blocked
            return self._truncate_output(f'Went forward. URL: {page.url}\n\n{await self._page_text(page, timeout)}')

        return await self._in_operation('go_forward', timeout_ms, _go_forward)

    async def execute_js(self, script: str, timeout_ms: int | None = None) -> str:
        """Evaluate a JavaScript expression and return its result.

        Args:
            script: JavaScript expression to evaluate, e.g. `document.title`.
            timeout_ms: Override the default action timeout for this call, in milliseconds.
                Must be greater than 0.

        Returns:
            A string result as-is, objects/arrays as JSON, `null`/`undefined` as
            `'undefined'`, or `JS error: ...` when evaluation raises.
        """

        async def _execute_js(page: _Page, timeout: int) -> str:
            try:
                # `evaluate` waits for a returned promise and has no `timeout`
                # parameter, so a never-resolving promise (or a blocked main
                # thread) would hold the operation lock forever without the
                # external deadline.
                result = await self._await_with_timeout(page.evaluate(script), timeout)
            except (PlaywrightTimeoutError, TargetClosedError):
                # `evaluate` raises for script exceptions too, so the browser's own
                # failures are picked out by type and left to the operation's mapper.
                raise
            except Exception as exc:
                return self._truncate_output(f'JS error: {exc}')
            if (blocked := await self._enforce_navigation_policy(page, 'execute_js', timeout)) is not None:
                return blocked
            if result is None:
                return self._truncate_output('undefined')
            if isinstance(result, str):
                return self._truncate_output(result)
            try:
                return self._truncate_output(json.dumps(result, default=str))
            except TypeError:  # pragma: no cover
                return self._truncate_output(str(result))

        return await self._in_operation('execute_js', timeout_ms, _execute_js)

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
                Must be greater than 0.

        Returns:
            A short confirmation followed by the page's visible text, or a bounded
            error string when neither/both arguments are given or the wait times out.
        """
        invalid = 'Error: wait_for requires exactly one of selector or text.'
        if text is not None:
            if selector is not None:
                return await self._refuse(timeout_ms, invalid)
            query = f'text={text}'
        elif selector is not None:
            query = selector
        else:
            return await self._refuse(timeout_ms, invalid)

        async def _wait_for(page: _Page, timeout: int) -> str:
            await page.wait_for_selector(query, timeout=timeout)
            return self._truncate_output(f"Found '{query}'.\n\n{await self._page_text(page, timeout)}")

        return await self._in_operation('wait_for', timeout_ms, _wait_for)

    async def snapshot(self, timeout_ms: int | None = None) -> str:
        """Return the page's accessibility tree with `aria-ref` handles for targeting.

        The snapshot is the structured, low-cost way to read the page and obtain
        `aria-ref=eN` handles; pass one back to `click` or `type_text` to target an
        element reliably. It includes iframe content that page-level selectors cannot
        reach. Use `screenshot` only for visual checks (charts, layout).

        Args:
            timeout_ms: Override the default action timeout for this call, in milliseconds.
                Must be greater than 0.

        Returns:
            The accessibility tree (truncated to the token budget), or a bounded
            error string when the snapshot fails.
        """

        async def _snapshot(page: _Page, timeout: int) -> str:
            return self._truncate_output(await page.aria_snapshot(mode='ai', timeout=timeout))

        return await self._in_operation('snapshot', timeout_ms, _snapshot)
