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
- Playwright's own default action/navigation timeout is a single 30000ms, and
  `timeout=0` disables the deadline (the toolset treats 0 the same way in
  `_await_with_timeout`). The defaults here deliberately split that number in two:
  a missed element is usually a wrong selector and should fail fast, while a page
  load legitimately takes longer. Verified 2026-07-24
  (`DEFAULT_PLAYWRIGHT_TIMEOUT_IN_MILLISECONDS = 30000` in `_impl/_helper.py`).
  Source: <https://playwright.dev/python/docs/api/class-page#page-set-default-timeout>.
  Re-check: grep the installed package for `DEFAULT_PLAYWRIGHT_TIMEOUT`.
- `aria-ref=fNeM` handles from a `mode='ai'` snapshot resolve inside the child
  frame they came from: the ref carries the frame sequence and the driver jumps to
  that frame before matching (`_jumpToAriaRefFrameIfNeeded` in the bundled
  driver). Plain CSS selectors do not cross frames, so a snapshot ref is the only
  handle that reaches embedded content. Verified 2026-08-12 against a real
  Chromium (read and click inside a cross-origin child frame).
  Source: <https://playwright.dev/python/docs/other-locators>. Re-check: grep the
  driver bundle for `_jumpToAriaRefFrameIfNeeded`.
- A request aborted by a context route reaches `requestfailed` with the failure
  text `net::ERR_FAILED`, which is what lets the guard's own entry (carrying the
  reason) stand alone in the event log. Verified 2026-08-12 against a real
  Chromium: a refused navigation produced exactly one recorded event. Source:
  <https://playwright.dev/python/docs/api/class-route#route-abort> (`errorCode`
  defaults to `failed`). Re-check: the private-address scenario in
  `scripts/playwright_smoke.py` asserts the single entry.
- `frame.inner_text('body')` reads a child frame the page-level call cannot see;
  `page.wait_for_selector` matches in the main frame only. Verified 2026-08-12
  against a real Chromium. Source:
  <https://playwright.dev/python/docs/frames>. Re-check: the iframe scenario in
  `scripts/playwright_smoke.py`.
"""

from __future__ import annotations

import asyncio
import ipaddress
import json
import os
import re
import sys
from collections import deque
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Literal, Protocol, TypeVar
from urllib.parse import urlparse

import idna
from opentelemetry import trace
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
    from collections.abc import Awaitable, Callable, Sequence

    from opentelemetry.trace import Span, Tracer
    from playwright.async_api import Browser as PlaywrightBrowserHandle
    from playwright.async_api import Page as PlaywrightPage
    from playwright.async_api import Playwright as PlaywrightDriver
    from playwright.async_api import Request as PlaywrightRequest
    from playwright.async_api import Route as PlaywrightRoute
    from playwright.async_api import StorageState

_T = TypeVar('_T')

DEFAULT_MAX_CONTENT_TOKENS: int = 4000
"""Default token budget for textual tool results injected into the agent context."""

DEFAULT_ACTION_TIMEOUT_MS: int = 5_000
"""Default deadline for element actions (click, type, read, wait), in milliseconds.

Deliberately shorter than the navigation budget: an action that misses is
normally a selector that matches nothing, and a long deadline turns that into a
stall a developer reads as a hung agent rather than a fast, actionable failure.
"""

DEFAULT_NAVIGATION_TIMEOUT_MS: int = 60_000
"""Default deadline for navigation, load settling, and starting or attaching to the browser, in milliseconds."""

_FRAME_TEXT_BUDGET_MS: int = 2_000
"""Total deadline for reading text out of a page's child frames.

One budget for the whole sweep rather than one per frame: a page can carry many
frames, and an unresponsive one must not consume the action's deadline. Whatever
was collected before the budget runs out is kept.
"""

_CHARS_PER_TOKEN = 4
"""Characters-per-token estimate used to turn a token budget into a character cap."""

_PRIVATE_ADDRESS_REASON = 'blocked private or link-local address'
"""Why the private-address block refused a URL, shared by the pre-check and the route guard."""

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


class _Keyboard(Protocol):
    """The subset of `playwright.async_api.Keyboard` the toolset drives."""

    async def press(self, key: str) -> None: ...  # pragma: no cover


class _Frame(Protocol):
    """The subset of `playwright.async_api.Frame` the toolset reads.

    Child frames are read directly because page-level calls do not cross a frame
    boundary: an embedded schedule, checkout step, or chat widget is invisible to
    `page.inner_text` and to `page.wait_for_selector`.
    """

    @property
    def url(self) -> str: ...  # pragma: no cover
    async def inner_text(self, selector: str, *, timeout: float | None = None) -> str: ...  # pragma: no cover
    async def wait_for_selector(self, selector: str, *, timeout: float | None = None) -> object: ...  # pragma: no cover


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
    @property
    def keyboard(self) -> _Keyboard: ...  # pragma: no cover
    @property
    def frames(self) -> Sequence[_Frame]: ...  # pragma: no cover
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
    async def hover(self, selector: str, *, timeout: float | None = None) -> None: ...  # pragma: no cover
    async def press(self, selector: str, key: str, *, timeout: float | None = None) -> None: ...  # pragma: no cover
    async def select_option(
        self, selector: str, value: Sequence[str], *, timeout: float | None = None
    ) -> list[str]: ...  # pragma: no cover


@dataclass(frozen=True)
class _Deadlines:
    """The two deadlines one operation runs under.

    A tool that acts on an element and a tool that loads a page fail for
    different reasons and deserve different budgets: a selector that matches
    nothing should fail fast, while a page load legitimately takes longer. A
    per-call `timeout_ms` collapses both to the value the model asked for.
    """

    action: int
    navigation: int


def _to_idna(host: str) -> str:
    """Return `host` in its ASCII/IDNA form so Unicode and `xn--` spellings compare equal.

    Encoding goes through the `idna` package under UTS46 non-transitional rules,
    which is what Chromium applies. The stdlib `'idna'` codec implements the older
    IDNA-2003 mapping and disagrees on the deviation characters (`ß`, `ς`, ZWJ,
    ZWNJ): it renders `faß.de` as `fass.de` where the browser connects to
    `xn--fa-hia.de`, so an allowlist entry and the request it is meant to permit
    would never match.

    A host that cannot be encoded (over-long or empty labels, IP literals) falls
    back to the input unchanged, so IPv4/IPv6 literals are left alone. The trailing
    dot of a fully-qualified spelling is dropped first: it names the DNS root rather
    than a label, and encoding rejects the empty label it produces, which would
    otherwise deny `example.com.` against an `example.com` allowlist entry.
    """
    host = host.rstrip('.')
    try:
        return idna.encode(host, uts46=True, transitional=False).decode('ascii')
    except (idna.IDNAError, UnicodeError):
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
        return _PRIVATE_ADDRESS_REASON
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


def _without_userinfo(url: str) -> str:
    """Return `url` with any `user:password@` credentials removed.

    Recorded URLs reach an OpenTelemetry backend, where userinfo is replayable
    credential material; the OTel HTTP conventions ask for it to be stripped from
    `url.full`. A regex rather than a parse-and-rebuild so a URL Chromium accepted
    but `urlsplit` rejects is still cleaned rather than raising.
    """
    return re.sub(r'^([a-zA-Z][\w+.\-]*://)[^/?#@]*@', r'\1', url)


def _without_endpoint_credentials(message: str, cdp_url: str) -> str:
    """Replace the CDP endpoint in `message` with a form carrying no credentials.

    A failure to attach reaches the model as a tool result, and Playwright quotes
    the endpoint it tried, call log included. Managed-browser providers routinely
    put a token in that URL's query string or path, so only the scheme, host and
    port survive -- enough to see which endpoint was unreachable, which the driver
    reports separately anyway.
    """
    try:
        parsed = urlparse(cdp_url)
        host = parsed.hostname or ''
        # `.port` parses lazily and raises on a non-numeric port, so it has to be
        # read inside the guard: this runs while an error is already being built.
        if parsed.port is not None:
            host = f'{host}:{parsed.port}'
        scheme = parsed.scheme
    except ValueError:
        return message.replace(cdp_url, '<cdp_url>')
    safe = f'{scheme}://{host}' if scheme else host
    return message.replace(cdp_url, safe)


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


_FALLBACK_TRACER = trace.get_tracer('pydantic-ai-harness.playwright')
"""Tracer used when the session is driven outside an agent run.

Inside a run the session takes the run's own tracer, so browser spans land
wherever the agent's instrumentation settings send its tool calls.
"""

_ABORTED_FAILURE = 'net::ERR_FAILED'
"""Failure text Chromium reports for a request the route guard aborted.

The guard records its own refusal with the reason, so the matching
`requestfailed` event is dropped rather than logged a second time without one.
"""

_EVENT_LOG_LIMIT = 500
"""Browser events retained per run. A ring buffer: a chatty page cannot grow it without bound."""


@dataclass(frozen=True)
class BrowserEvent:
    """Something the browser did that no tool call records.

    Console output, uncaught page errors, responses, requests the egress policy
    refused, and popups the session closed. The `console_messages` and
    `network_requests` tools read this log, and every entry is also added as an
    OpenTelemetry span event on the browser operation that was running, so a
    trace shows what the page did between tool calls.
    """

    kind: Literal['console', 'page_error', 'response', 'request_failed', 'request_blocked', 'popup_closed']
    level: Literal['info', 'warning', 'error']
    message: str
    url: str | None = None
    method: str | None = None
    status: int | None = None

    def describe(self) -> str:
        """Render the event as one model-facing line."""
        parts = [f'[{self.level}] {self.kind}']
        if self.method is not None:
            parts.append(self.method)
        if self.status is not None:
            parts.append(str(self.status))
        if self.url is not None:
            parts.append(self.url)
        if self.message:
            parts.append(self.message)
        return ' '.join(parts)

    def attributes(self) -> dict[str, str | int]:
        """Render the event as OpenTelemetry span-event attributes.

        Names follow the OTel HTTP semantic conventions where one applies, so a
        backend that already understands `url.full` or `http.response.status_code`
        reads these without a custom mapping.
        """
        attributes: dict[str, str | int] = {'browser.event.kind': self.kind, 'browser.event.level': self.level}
        if self.message:
            attributes['browser.event.message'] = self.message
        if self.url is not None:
            attributes['url.full'] = self.url
        if self.method is not None:
            attributes['http.request.method'] = self.method
        if self.status is not None:
            attributes['http.response.status_code'] = self.status
        return attributes


class _ConsoleMessage(Protocol):
    """The subset of `playwright.async_api.ConsoleMessage` the session reads."""

    @property
    def type(self) -> str: ...  # pragma: no cover
    @property
    def text(self) -> str: ...  # pragma: no cover


class _Response(Protocol):
    """The subset of `playwright.async_api.Response` the session reads."""

    @property
    def url(self) -> str: ...  # pragma: no cover
    @property
    def status(self) -> int: ...  # pragma: no cover
    @property
    def request(self) -> _FailedRequest: ...  # pragma: no cover


class _FailedRequest(Protocol):
    """The subset of `playwright.async_api.Request` the session reads."""

    @property
    def url(self) -> str: ...  # pragma: no cover
    @property
    def method(self) -> str: ...  # pragma: no cover
    @property
    def failure(self) -> str | None: ...  # pragma: no cover


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
        launch_timeout_ms: int = DEFAULT_NAVIGATION_TIMEOUT_MS,
    ) -> None:
        self.policy = policy if policy is not None else NavigationPolicy()
        """Egress policy the route guard enforces, and the default for a toolset built on this session."""
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
        self.tracer: Tracer = _FALLBACK_TRACER
        """Tracer browser operations report to. `PlaywrightBrowser.wrap_run` sets the run's own."""
        self.events: deque[BrowserEvent] = deque(maxlen=_EVENT_LOG_LIMIT)
        """What the browser did outside the tool calls, newest last."""
        self.operation_span: Span | None = None
        """Span of the browser operation currently running, when one is.

        Playwright delivers page events from its own receive task, so the
        operation's span cannot be recovered from the OpenTelemetry context there.
        The toolset publishes it here for the length of an operation instead.
        """

    def record(self, event: BrowserEvent) -> None:
        """Log a browser event and attach it to the operation that was running.

        The URL is stripped of credentials here, at the one point every event
        passes through, so neither the tool output nor the exported span carries
        them.
        """
        if event.url is not None:
            event = replace(event, url=_without_userinfo(event.url))
        self.events.append(event)
        span = self.operation_span
        if span is not None and span.is_recording():
            span.add_event(f'browser.{event.kind}', event.attributes())

    def _on_console(self, message: _ConsoleMessage) -> None:
        """Record a console message, keeping the page's own severity."""
        level: Literal['info', 'warning', 'error'] = 'info'
        if message.type == 'error':
            level = 'error'
        elif message.type == 'warning':
            level = 'warning'
        self.record(BrowserEvent(kind='console', level=level, message=f'{message.type}: {message.text}'))

    def _on_page_error(self, error: object) -> None:
        """Record an uncaught exception from page scripts."""
        self.record(BrowserEvent(kind='page_error', level='error', message=str(error)))

    def _on_response(self, response: _Response) -> None:
        """Record a response, so the agent can find the request a page made for its data."""
        self.record(
            BrowserEvent(
                kind='response',
                level='error' if response.status >= 400 else 'info',
                message='',
                url=response.url,
                method=response.request.method,
                status=response.status,
            )
        )

    def _on_request_failed(self, request: _FailedRequest) -> None:
        """Record a request the network layer never completed.

        Requests the egress policy aborts arrive here too, but the guard has
        already recorded those with their reason, so they are dropped to keep one
        entry per refusal.
        """
        failure = request.failure or 'failed'
        if failure == _ABORTED_FAILURE:
            return
        self.record(
            BrowserEvent(kind='request_failed', level='error', message=failure, url=request.url, method=request.method)
        )

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
        if self.policy.enforced():
            await context.route('**/*', self._route_guard)
        page.on('popup', self._on_popup)
        page.on('console', self._on_console)
        page.on('pageerror', self._on_page_error)
        page.on('response', self._on_response)
        page.on('requestfailed', self._on_request_failed)
        self.page = page

    async def _connect(self, pw: PlaywrightDriver) -> PlaywrightBrowserHandle | None:
        """Attach to or start a browser, or record why none is available.

        Both paths are bounded by `launch_timeout_ms`: a `cdp_url` pointing at an
        unresponsive endpoint would otherwise hold the operation lock with no
        deadline at all. The auto-install download is deliberately left unbounded,
        since a first-run browser fetch legitimately outlasts an action timeout.
        """
        if self._cdp_url is not None:
            try:
                return await pw.chromium.connect_over_cdp(self._cdp_url, timeout=self._launch_timeout_ms)
            except PlaywrightError as exc:
                raise PlaywrightError(_without_endpoint_credentials(str(exc), self._cdp_url)) from exc
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
        if self.policy.refused_in_every_frame(request.url):
            await self._abort(route, request.url, _PRIVATE_ADDRESS_REASON)
            return
        reason = self.policy.blocked_reason(request.url)
        if not request.is_navigation_request() or reason is None:
            await route.continue_()
            return
        try:
            frame = request.frame
        except PlaywrightError:
            await self._abort(route, request.url, reason)
            return
        if frame == frame.page.main_frame:
            await self._abort(route, request.url, reason)
            return
        await route.continue_()

    async def _abort(self, route: PlaywrightRoute, url: str, reason: str) -> None:
        """Refuse a request and record why, so a blocked page is diagnosable."""
        self.record(BrowserEvent(kind='request_blocked', level='warning', message=reason, url=url))
        await route.abort()

    async def _handle_popup(self, popup: PlaywrightPage) -> None:
        """Keep the browser single-tab by closing popup pages."""
        self.record(BrowserEvent(kind='popup_closed', level='warning', message='popup closed', url=popup.url))
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
    """Async Playwright-backed browser tools: navigate, interact, extract, screenshot, run JS, inspect.

    The tools read the active page from a shared `PlaywrightBrowserSession`, which
    `PlaywrightBrowser.wrap_run` populates lazily on the first tool call. Use the
    toolset through `PlaywrightBrowser` rather than directly; construct it
    directly (with a `session` whose `page` you set) only to drive tools against a
    page double.

    Page text is extracted with Playwright itself (`inner_text`), across the main
    frame and any child frames, and every textual result is truncated to
    `max_content_tokens`; no HTML-to-Markdown dependency is pulled in. `screenshot`
    returns a `ToolReturn` carrying `BinaryContent` so vision models see the image
    natively instead of a base64 string bloating the text context.

    Each operation runs inside an OpenTelemetry span, and browser events the page
    produced during it (console output, responses, refused requests) are attached
    to that span as events.
    """

    def __init__(
        self,
        *,
        session: PlaywrightBrowserSession,
        policy: NavigationPolicy | None = None,
        screenshot_on_navigate: bool = False,
        max_content_tokens: int = DEFAULT_MAX_CONTENT_TOKENS,
        action_timeout_ms: int = DEFAULT_ACTION_TIMEOUT_MS,
        navigation_timeout_ms: int = DEFAULT_NAVIGATION_TIMEOUT_MS,
    ) -> None:
        if max_content_tokens < 0:
            raise ValueError('max_content_tokens must be greater than or equal to 0')
        if action_timeout_ms < 0:
            raise ValueError('action_timeout_ms must be greater than or equal to 0')
        if navigation_timeout_ms < 0:
            raise ValueError('navigation_timeout_ms must be greater than or equal to 0')
        super().__init__(id='playwright')
        self._session = session
        # Defaults to the session's policy rather than a fresh one: the guard the
        # session installs and the checks these tools run have to agree, and a
        # second default would silently refuse what the session was built to allow.
        self._policy = policy if policy is not None else session.policy
        self._screenshot_on_navigate = screenshot_on_navigate
        self._max_content_tokens = max_content_tokens
        self._action_timeout_ms = action_timeout_ms
        self._navigation_timeout_ms = navigation_timeout_ms
        self._operation_lock = asyncio.Lock()
        self.add_function(self.navigate, name='navigate')
        self.add_function(self.click, name='click')
        self.add_function(self.type_text, name='type_text')
        self.add_function(self.press_key, name='press_key')
        self.add_function(self.select_option, name='select_option')
        self.add_function(self.hover, name='hover')
        self.add_function(self.screenshot, name='screenshot')
        self.add_function(self.get_text, name='get_text')
        self.add_function(self.scroll, name='scroll')
        self.add_function(self.go_back, name='go_back')
        self.add_function(self.go_forward, name='go_forward')
        self.add_function(self.execute_js, name='execute_js')
        self.add_function(self.wait_for, name='wait_for')
        self.add_function(self.snapshot, name='snapshot')
        self.add_function(self.console_messages, name='console_messages')
        self.add_function(self.network_requests, name='network_requests')

    async def _frame_text(self, page: _Page, budget_ms: int) -> list[str]:
        """Return the text of each child frame that has any, newest layout first.

        Page-level reads stop at the frame boundary, so an embedded schedule,
        checkout step, or chat widget is absent from `inner_text` even though the
        model can see it in a `snapshot`. Reading the frames directly closes that
        gap for every tool that returns page text.

        Failures are skipped rather than raised: a frame can detach mid-read, and
        a missing embed should not turn a successful action into an error.
        """
        texts: list[str] = []

        async def sweep() -> None:
            for frame in page.frames[1:]:
                try:
                    text = await frame.inner_text('body', timeout=budget_ms)
                except PlaywrightError:
                    continue
                if text.strip():
                    texts.append(f'[frame {frame.url}]\n{text}')

        try:
            await asyncio.wait_for(sweep(), budget_ms / 1000)
        except asyncio.TimeoutError:
            pass
        return texts

    async def _page_text(self, page: _Page, timeout_ms: int | None = None) -> str:
        """Return the visible text of `page` and its child frames, truncated to the token budget.

        The page is passed in rather than re-acquired: every caller already holds
        the one it just acted on, and re-entering `ensure_page` here would be a
        second path into the launch machinery, raising `RuntimeError` where the
        callers only guard against `PlaywrightError`.
        """
        text = await page.inner_text('body', timeout=timeout_ms)
        frames = await self._frame_text(page, self._frame_budget(timeout_ms))
        return self._truncate_output('\n\n'.join([text, *frames]))

    def _frame_budget(self, timeout_ms: int | None) -> int:
        """Return the deadline for the child-frame sweep.

        The sweep is capped so one unresponsive embed cannot spend the whole
        action budget, and it is capped again by a shorter caller deadline, so a
        tight `timeout_ms` is not overrun by the frames the caller never asked
        about. A caller with no deadline at all still gets the cap.
        """
        if timeout_ms is None or timeout_ms == 0:
            return _FRAME_TEXT_BUDGET_MS
        return min(timeout_ms, _FRAME_TEXT_BUDGET_MS)

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
                'the page may be slow; try a different selector, or navigate again. '
                'Content inside an iframe needs an `aria-ref=` handle from `snapshot`, not a CSS selector.'
            )
        if isinstance(exc, TargetClosedError):
            return (
                f'Error: {action} failed: the browser or page was closed unexpectedly. '
                'Browser tools may be unavailable for the rest of this run.'
            )
        return f'Error: {action} failed: {exc}'

    def _deadlines(self, timeout_ms: int | None) -> _Deadlines:
        """Resolve the deadlines an operation runs under.

        A per-call override replaces both: the model asked one call to take at
        most this long, and a navigation triggered by that call is part of it.
        """
        if timeout_ms is not None:
            return _Deadlines(action=timeout_ms, navigation=timeout_ms)
        return _Deadlines(action=self._action_timeout_ms, navigation=self._navigation_timeout_ms)

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
        stays available to the developer as a capability default but is refused as
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
        self,
        action: str,
        timeout_ms: int | None,
        body: Callable[[_Page, _Deadlines], Awaitable[_T]],
        *,
        governed_by_navigation: bool = False,
    ) -> _T | str:
        """Run `body` as one complete operation on the shared page.

        Every tool needs the same things in the same order: exclusive use of the
        page, a per-call deadline that is either absent or positive, a span the
        page's own events can attach to, the page itself (launching Chromium on
        the first call), the resolved deadlines, and a Playwright failure turned
        into a result the model can read instead of an exception that ends the run.

        Acquiring the page is inside the guarded region: starting or attaching to
        a browser can fail the same way an action can (a `cdp_url` whose endpoint
        is gone, most realistically), and the model can act on that only if it
        arrives as a result rather than as an exception that ends the run.

        `governed_by_navigation` names which deadline a failure is reported
        against, so a navigation timeout is not described with the action budget.

        An argument check that must not launch a browser belongs in the tool,
        before this call -- see `_refuse`.
        """
        async with self._operation_lock:
            if (error := self._timeout_error(timeout_ms)) is not None:
                return error
            deadlines = self._deadlines(timeout_ms)
            reported = deadlines.navigation if governed_by_navigation else deadlines.action
            with self._session.tracer.start_as_current_span(
                f'browser {action}',
                attributes={'browser.action': action, 'browser.timeout_ms': reported},
            ) as span:
                self._session.operation_span = span
                try:
                    page = await self._session.ensure_page()
                    result = await body(page, deadlines)
                except PlaywrightError as exc:
                    span.set_attribute('browser.outcome', 'error')
                    return self._truncate_output(self._playwright_error(action, exc, reported))
                finally:
                    self._session.operation_span = None
                span.set_attribute('browser.outcome', 'ok')
                span.set_attribute('url.full', _without_userinfo(page.url))
                return result

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

    async def _settle(self, page: _Page, action: str, deadlines: _Deadlines) -> str | None:
        """Let the navigation finish, then re-check where it landed.

        The order is the point: the policy has to see the settled URL, because
        reading it before the load completes checks the page the action started
        from. Returns the bounced error, or `None` when the result is permitted.
        """
        await page.wait_for_load_state('domcontentloaded', timeout=deadlines.navigation)
        return await self._enforce_navigation_policy(page, action, deadlines.navigation)

    async def navigate(self, url: str, timeout_ms: int | None = None) -> str | ToolReturn[str]:
        """Navigate to a URL and return the page's title and visible text.

        Args:
            url: Full URL to navigate to (e.g. `https://example.com`).
            timeout_ms: Override the default timeout for this call, in milliseconds.
                Must be greater than 0.

        Returns:
            The page URL, title, and visible text, including the text of any
            embedded frames. When `screenshot_on_navigate` is set, a screenshot is
            attached as image content for vision models.
        """
        reason = self._policy.blocked_reason(url)
        if reason is not None:
            # Refused before `_in_operation`, so a disallowed URL never launches Chromium.
            return await self._refuse(timeout_ms, f'Error: {reason}: {url}')

        async def _navigate(page: _Page, deadlines: _Deadlines) -> str | ToolReturn[str]:
            await page.goto(url, timeout=deadlines.navigation)
            if (blocked := await self._settle(page, 'navigate', deadlines)) is not None:
                return blocked
            title = await self._await_with_timeout(page.title(), deadlines.action)
            text = await self._page_text(page, deadlines.action)
            result = self._truncate_output(f'URL: {page.url}\nTitle: {title}\n\n{text}')
            if not self._screenshot_on_navigate:
                return result
            png = await page.screenshot(timeout=deadlines.action)
            if (oversized := self._oversized_screenshot_error(png)) is not None:
                return self._truncate_output_keeping(result, oversized)
            return ToolReturn(result, content=[BinaryContent(data=png, media_type='image/png')])

        return await self._in_operation('navigate', timeout_ms, _navigate, governed_by_navigation=True)

    async def click(self, selector: str, timeout_ms: int | None = None) -> str:
        """Click an element on the current page.

        Args:
            selector: A CSS selector (e.g. `button#submit`), an `aria-ref=` handle
                from `snapshot` (the most reliable way to target an element, and
                the only one that reaches inside an iframe), or pixel coordinates
                as `'x,y'` (e.g. `'450,300'`).
            timeout_ms: Override the default timeout for this call, in milliseconds.
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

        async def _click(page: _Page, deadlines: _Deadlines) -> str:
            if coordinates is not None:
                await page.mouse.click(*coordinates)
            else:
                await page.click(selector, timeout=deadlines.action)
            if (blocked := await self._settle(page, 'click', deadlines)) is not None:
                return blocked
            text = await self._page_text(page, deadlines.action)
            return self._truncate_output(f"Clicked '{selector}'. URL: {page.url}\n\n{text}")

        return await self._in_operation('click', timeout_ms, _click)

    async def type_text(self, selector: str, text: str, timeout_ms: int | None = None) -> str:
        """Type text into an input field, replacing any existing value.

        This sets the field's value; it does not submit. Use `press_key('Enter')`
        for a form or search box that submits on Enter.

        Args:
            selector: CSS selector for the target input element, or an `aria-ref=`
                handle from `snapshot`.
            text: Text to type into the field.
            timeout_ms: Override the default timeout for this call, in milliseconds.
                Must be greater than 0.

        Returns:
            The page's visible text after typing.
        """

        async def _type_text(page: _Page, deadlines: _Deadlines) -> str:
            await page.fill(selector, text, timeout=deadlines.action)
            return self._truncate_output(f"Typed into '{selector}'.\n\n{await self._page_text(page, deadlines.action)}")

        return await self._in_operation('type_text', timeout_ms, _type_text)

    async def press_key(self, key: str, selector: str | None = None, timeout_ms: int | None = None) -> str:
        """Press a keyboard key, optionally focusing an element first.

        Reaches what typing cannot: submitting a search box with `Enter`, closing
        an overlay with `Escape`, moving between fields with `Tab`.

        Args:
            key: A Playwright key name, e.g. `Enter`, `Escape`, `Tab`, `ArrowDown`,
                `Control+a`.
            selector: Element to focus before pressing. Omit to send the key to
                whatever currently has focus.
            timeout_ms: Override the default timeout for this call, in milliseconds.
                Must be greater than 0.

        Returns:
            The page's visible text after the key press.
        """

        async def _press_key(page: _Page, deadlines: _Deadlines) -> str:
            if selector is None:
                await self._await_with_timeout(page.keyboard.press(key), deadlines.action)
            else:
                await page.press(selector, key, timeout=deadlines.action)
            # A key press is a common way to trigger navigation (Enter in a search
            # box), so the result is settled and re-checked like a click.
            if (blocked := await self._settle(page, 'press_key', deadlines)) is not None:
                return blocked
            return self._truncate_output(f"Pressed '{key}'.\n\n{await self._page_text(page, deadlines.action)}")

        return await self._in_operation('press_key', timeout_ms, _press_key)

    async def select_option(self, selector: str, values: list[str], timeout_ms: int | None = None) -> str:
        """Choose one or more options in a `<select>` dropdown.

        A native dropdown does not open as page content, so clicking it does not
        expose its options.

        Args:
            selector: CSS selector for the `<select>` element, or an `aria-ref=`
                handle from `snapshot`.
            values: Option values (or labels) to select. Pass one value for a
                single-choice dropdown.
            timeout_ms: Override the default timeout for this call, in milliseconds.
                Must be greater than 0.

        Returns:
            The page's visible text after the selection.
        """

        async def _select_option(page: _Page, deadlines: _Deadlines) -> str:
            selected = await page.select_option(selector, values, timeout=deadlines.action)
            if (blocked := await self._settle(page, 'select_option', deadlines)) is not None:
                return blocked
            text = await self._page_text(page, deadlines.action)
            return self._truncate_output(f"Selected {selected} in '{selector}'.\n\n{text}")

        return await self._in_operation('select_option', timeout_ms, _select_option)

    async def hover(self, selector: str, timeout_ms: int | None = None) -> str:
        """Hover over an element, revealing menus and tooltips that appear on hover.

        Args:
            selector: CSS selector for the element, or an `aria-ref=` handle from
                `snapshot`.
            timeout_ms: Override the default timeout for this call, in milliseconds.
                Must be greater than 0.

        Returns:
            The page's visible text after hovering.
        """

        async def _hover(page: _Page, deadlines: _Deadlines) -> str:
            await page.hover(selector, timeout=deadlines.action)
            return self._truncate_output(f"Hovered '{selector}'.\n\n{await self._page_text(page, deadlines.action)}")

        return await self._in_operation('hover', timeout_ms, _hover)

    async def screenshot(self, full_page: bool = False, timeout_ms: int | None = None) -> str | ToolReturn[str]:
        """Capture a screenshot of the current page.

        Args:
            full_page: Capture the full scrollable page when `True`, else the
                current viewport.
            timeout_ms: Override the default timeout for this call, in milliseconds.
                Must be greater than 0.

        Returns:
            A short note with the page URL, and the PNG as image content so
            vision models can see it.
        """

        async def _screenshot(page: _Page, deadlines: _Deadlines) -> str | ToolReturn[str]:
            png = await page.screenshot(full_page=full_page, timeout=deadlines.action)
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
            selector: CSS selector to read, or an `aria-ref=` handle from
                `snapshot`. A CSS selector matches the main frame only; reading
                inside an iframe needs the `aria-ref=` handle. Omit for the whole
                page's visible text, which includes embedded frames.
            timeout_ms: Override the default timeout for this call, in milliseconds.
                Must be greater than 0.

        Returns:
            The element's text, or the full page's visible text when no selector
            is given.
        """

        async def _get_text(page: _Page, deadlines: _Deadlines) -> str:
            if not selector:
                return await self._page_text(page, deadlines.action)
            try:
                text = await page.inner_text(selector, timeout=deadlines.action)
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
            timeout_ms: Override the default timeout for this call, in milliseconds.
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

        async def _scroll(page: _Page, deadlines: _Deadlines) -> str:
            if x is not None and y is not None:
                # `Mouse.move`/`Mouse.wheel` take no `timeout`, so they are bounded
                # externally like `evaluate` below: an unbounded await here would
                # hold the operation lock for the rest of the run.
                await self._await_with_timeout(page.mouse.move(x, y), deadlines.action)
                await self._await_with_timeout(page.mouse.wheel(*delta), deadlines.action)
            else:
                # `evaluate` has no `timeout` parameter and hangs if the page's
                # main thread is blocked, so it is bounded externally.
                await self._await_with_timeout(
                    page.evaluate(f'window.scrollBy({delta[0]}, {delta[1]})'), deadlines.action
                )
            return self._truncate_output(f'Scrolled {direction}.\n\n{await self._page_text(page, deadlines.action)}')

        return await self._in_operation('scroll', timeout_ms, _scroll)

    async def go_back(self, timeout_ms: int | None = None) -> str:
        """Navigate back in the browser history.

        Args:
            timeout_ms: Override the default timeout for this call, in milliseconds.
                Must be greater than 0.

        Returns:
            The previous page's visible text, or a note when there is no history
            entry to go back to.
        """

        async def _go_back(page: _Page, deadlines: _Deadlines) -> str:
            if await page.go_back(timeout=deadlines.navigation) is None:
                return self._truncate_output('No previous page in browser history.')
            if (blocked := await self._settle(page, 'go_back', deadlines)) is not None:
                return blocked
            return self._truncate_output(
                f'Went back. URL: {page.url}\n\n{await self._page_text(page, deadlines.action)}'
            )

        return await self._in_operation('go_back', timeout_ms, _go_back, governed_by_navigation=True)

    async def go_forward(self, timeout_ms: int | None = None) -> str:
        """Navigate forward in the browser history.

        Args:
            timeout_ms: Override the default timeout for this call, in milliseconds.
                Must be greater than 0.

        Returns:
            The next page's visible text, or a note when there is no history entry
            to go forward to.
        """

        async def _go_forward(page: _Page, deadlines: _Deadlines) -> str:
            if await page.go_forward(timeout=deadlines.navigation) is None:
                return self._truncate_output('No next page in browser history.')
            if (blocked := await self._settle(page, 'go_forward', deadlines)) is not None:
                return blocked
            return self._truncate_output(
                f'Went forward. URL: {page.url}\n\n{await self._page_text(page, deadlines.action)}'
            )

        return await self._in_operation('go_forward', timeout_ms, _go_forward, governed_by_navigation=True)

    async def execute_js(self, script: str, timeout_ms: int | None = None) -> str:
        """Evaluate a JavaScript expression and return its result.

        Args:
            script: JavaScript expression to evaluate, e.g. `document.title`.
            timeout_ms: Override the default timeout for this call, in milliseconds.
                Must be greater than 0.

        Returns:
            A string result as-is, objects/arrays as JSON, `null`/`undefined` as
            `'undefined'`, or `JS error: ...` when evaluation raises.
        """

        async def _execute_js(page: _Page, deadlines: _Deadlines) -> str:
            try:
                # `evaluate` waits for a returned promise and has no `timeout`
                # parameter, so a never-resolving promise (or a blocked main
                # thread) would hold the operation lock forever without the
                # external deadline.
                result = await self._await_with_timeout(page.evaluate(script), deadlines.action)
            except (PlaywrightTimeoutError, TargetClosedError):
                # `evaluate` raises for script exceptions too, so the browser's own
                # failures are picked out by type and left to the operation's mapper.
                raise
            except Exception as exc:
                return self._truncate_output(f'JS error: {exc}')
            try:
                blocked = await self._enforce_navigation_policy(page, 'execute_js', deadlines.navigation)
            except PlaywrightError as exc:
                # Kept out of the try above on purpose: that one ends early so a
                # script exception maps to 'JS error', which a failed bounce is not.
                return self._truncate_output(self._playwright_error('execute_js', exc, deadlines.navigation))
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

        return await self._in_operation('execute_js', timeout_ms, _execute_js)

    async def wait_for(
        self, selector: str | None = None, text: str | None = None, timeout_ms: int | None = None
    ) -> str:
        """Wait for dynamic content to appear, then return the page's visible text.

        Pass exactly one of `selector` or `text`. Use this after an action that
        loads content asynchronously, so a following read sees the settled page.
        The wait covers embedded frames as well as the main page.

        Args:
            selector: CSS selector (or an `aria-ref=` handle) to wait for.
            text: Visible text to wait for, matched with Playwright's text engine.
            timeout_ms: Override the default timeout for this call, in milliseconds.
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

        async def _wait_for(page: _Page, deadlines: _Deadlines) -> str:
            await self._wait_in_any_frame(page, query, deadlines.action)
            return self._truncate_output(f"Found '{query}'.\n\n{await self._page_text(page, deadlines.action)}")

        return await self._in_operation('wait_for', timeout_ms, _wait_for)

    async def _wait_in_any_frame(self, page: _Page, query: str, timeout_ms: int) -> None:
        """Wait until `query` matches in the main page or in any child frame.

        `page.wait_for_selector` matches the main frame only, so content arriving
        inside an embed would never satisfy a wait even though the model can see
        that frame in a snapshot. Every frame is watched at once and the first
        match wins; when none matches, the failure of whichever wait finished
        last is raised, which is the timeout the caller is waiting on anyway.
        """
        waits = [page.wait_for_selector(query, timeout=timeout_ms)]
        waits.extend(frame.wait_for_selector(query, timeout=timeout_ms) for frame in page.frames[1:])
        tasks = [asyncio.ensure_future(wait) for wait in waits]
        failure: BaseException = PlaywrightTimeoutError(f'Timeout {timeout_ms}ms exceeded.')
        try:
            while tasks:
                done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                tasks = list(pending)
                # Every finished wait is read, including when an earlier one in the
                # same batch already matched: an exception nobody retrieves surfaces
                # later, out of context, as an unhandled task error.
                matched = False
                for task in done:
                    error = task.exception()
                    if error is None:
                        matched = True
                    else:
                        failure = error
                if matched:
                    return
        finally:
            for task in tasks:
                task.cancel()
        raise failure

    async def snapshot(self, timeout_ms: int | None = None) -> str:
        """Return the page's accessibility tree with `aria-ref` handles for targeting.

        The snapshot is the structured, low-cost way to read the page and obtain
        `aria-ref=eN` handles; pass one back to `click`, `type_text`, `hover` or
        `get_text` to target an element reliably. It includes iframe content that
        CSS selectors cannot reach: refs inside an embed look like `f1e4`, and they
        resolve in the frame they came from. Use `screenshot` only for visual
        checks (charts, layout).

        Args:
            timeout_ms: Override the default timeout for this call, in milliseconds.
                Must be greater than 0.

        Returns:
            The accessibility tree (truncated to the token budget), or a bounded
            error string when the snapshot fails.
        """

        async def _snapshot(page: _Page, deadlines: _Deadlines) -> str:
            return self._truncate_output(await page.aria_snapshot(mode='ai', timeout=deadlines.action))

        return await self._in_operation('snapshot', timeout_ms, _snapshot)

    async def console_messages(self, errors_only: bool = False) -> str:
        """Return the console output and uncaught script errors the page produced.

        Reaches what the page never renders: a failed script, a rejected request
        logged by the site's own code, a framework warning explaining why content
        is missing.

        Args:
            errors_only: Return only errors, dropping logs and warnings.

        Returns:
            One line per message, oldest first, or a note when there are none.
        """
        kinds = ('console', 'page_error')
        events = [event for event in self._session.events if event.kind in kinds]
        if errors_only:
            events = [event for event in events if event.level == 'error']
        return self._render_events(events, 'console messages')

    async def network_requests(self, url_contains: str | None = None) -> str:
        """Return the network requests the page made, with their status.

        Pages that render from an API often expose the data more directly than the
        DOM does: find the request here, then read it with `navigate` or
        `execute_js`. Requests the egress policy refused are listed with the
        reason, so a page that fails to load is diagnosable.

        Args:
            url_contains: Keep only requests whose URL contains this substring.

        Returns:
            One line per request, oldest first, or a note when there are none.
        """
        kinds = ('response', 'request_failed', 'request_blocked')
        events = [event for event in self._session.events if event.kind in kinds]
        if url_contains is not None:
            events = [event for event in events if event.url is not None and url_contains in event.url]
        return self._render_events(events, 'network requests')

    def _render_events(self, events: list[BrowserEvent], label: str) -> str:
        """Render browser events as bounded, model-facing lines."""
        if not events:
            return self._truncate_output(f'No {label} recorded.')
        return self._truncate_output('\n'.join(event.describe() for event in events))
