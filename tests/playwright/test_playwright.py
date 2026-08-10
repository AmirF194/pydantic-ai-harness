"""Tests for the Playwright capability.

The Playwright API surface is mocked throughout: no real Chromium is launched.
An in-memory page double (`_FakePage`) backs the toolset, and a fake Playwright
driver chain backs the `wrap_run` lifecycle, so the suite runs in CI with only
the `playwright` Python package installed (no browser binary).
"""

from __future__ import annotations

import asyncio
import sys
from collections.abc import Awaitable, Callable
from typing import Protocol

import pytest
from playwright._impl._errors import TargetClosedError
from playwright.async_api import Error as PlaywrightError
from playwright.async_api import StorageState
from playwright.async_api import TimeoutError as PlaywrightTimeoutError
from pydantic_ai import Agent, AgentRunResult
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.capabilities.abstract import CapabilityOrdering
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import BinaryContent, ToolReturn
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import RunContext, ToolDefinition
from pydantic_ai.usage import RunUsage

import pydantic_ai_harness.playwright._capability as capability_module
import pydantic_ai_harness.playwright._toolset as toolset_module
from pydantic_ai_harness.playwright import (
    DEFAULT_MAX_CONTENT_TOKENS,
    DEFAULT_TIMEOUT_MS,
    PlaywrightBrowser,
    PlaywrightBrowserState,
    PlaywrightBrowserToolset,
)

pytestmark = pytest.mark.anyio

_STORAGE_STATE: StorageState = {'cookies': [{'name': 'session', 'value': 'abc', 'domain': 'example.com', 'path': '/'}]}

_HISTORY_RESPONSE = object()


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


# --- Doubles for the Playwright API surface ---------------------------------


class _FakeMouse:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int, int]] = []

    async def click(self, x: float, y: float) -> None:
        self.calls.append(('click', int(x), int(y)))

    async def move(self, x: float, y: float) -> None:
        self.calls.append(('move', int(x), int(y)))

    async def wheel(self, delta_x: float, delta_y: float) -> None:
        self.calls.append(('wheel', int(delta_x), int(delta_y)))


class _FakeRoute:
    def __init__(self) -> None:
        self.aborted = False
        self.continued = False

    async def abort(self) -> None:
        self.aborted = True

    async def continue_(self) -> None:
        self.continued = True


class _FakeRequestPage:
    def __init__(self) -> None:
        self.main_frame = _FakeFrame(self)


class _FakeFrame:
    def __init__(self, page: _FakeRequestPage) -> None:
        self.page = page


class _FakeRequest:
    def __init__(
        self,
        url: str,
        *,
        navigation: bool,
        frame: _FakeFrame | None = None,
        frame_error: PlaywrightError | None = None,
    ) -> None:
        self.url = url
        self._navigation = navigation
        self._frame = frame
        self._frame_error = frame_error

    def is_navigation_request(self) -> bool:
        return self._navigation

    @property
    def frame(self) -> _FakeFrame:
        if self._frame_error is not None:
            raise self._frame_error
        assert self._frame is not None
        return self._frame


class _FakeRouteHandler(Protocol):
    def __call__(self, route: _FakeRoute, request: _FakeRequest) -> Awaitable[None]: ...  # pragma: no cover


class _FakeBrowserContext:
    def __init__(
        self, page: _FakePage, *, storage_state: StorageState | None = None, service_workers: str | None = None
    ) -> None:
        self.page = page
        self.storage_state = storage_state
        self.service_workers = service_workers
        self.routes: list[str] = []
        self.route_handler: _FakeRouteHandler | None = None

    async def new_page(self) -> _FakePage:
        return self.page

    async def route(self, url: str, handler: _FakeRouteHandler) -> None:
        self.routes.append(url)
        self.route_handler = handler

    async def dispatch(self, request: _FakeRequest) -> _FakeRoute:
        assert self.route_handler is not None
        route = _FakeRoute()
        await self.route_handler(route, request)
        return route


class _FakePage:
    def __init__(
        self,
        *,
        url: str = 'https://example.com/',
        title: str = 'Example',
        body: str = 'Hello body',
        evaluate_result: object = None,
        evaluate_raises: Exception | None = None,
        selector_raises: bool = False,
        element_text: str | None = None,
        redirect_to: str | None = None,
        screenshot_bytes: bytes = b'PNG-BYTES',
        close_error: Exception | None = None,
        goto_error: PlaywrightError | None = None,
        click_error: PlaywrightError | None = None,
        fill_error: PlaywrightError | None = None,
        inner_text_error: PlaywrightError | None = None,
        screenshot_error: PlaywrightError | None = None,
        go_back_error: PlaywrightError | None = None,
        go_forward_error: PlaywrightError | None = None,
        go_back_result: object | None = _HISTORY_RESPONSE,
        go_forward_result: object | None = _HISTORY_RESPONSE,
        wait_for_error: PlaywrightError | None = None,
        aria_snapshot_tree: str = '- heading "Example" [ref=e1]\n- button "Go" [ref=e2]',
        aria_snapshot_error: PlaywrightError | None = None,
    ) -> None:
        self._url = url
        self._title = title
        self._body = body
        self._evaluate_result = evaluate_result
        self._evaluate_raises = evaluate_raises
        self._selector_raises = selector_raises
        self._element_text = element_text
        self._redirect_to = redirect_to
        self._screenshot_bytes = screenshot_bytes
        self._close_error = close_error
        self._goto_error = goto_error
        self._click_error = click_error
        self._fill_error = fill_error
        self._inner_text_error = inner_text_error
        self._screenshot_error = screenshot_error
        self._go_back_error = go_back_error
        self._go_forward_error = go_forward_error
        self._go_back_result = go_back_result
        self._go_forward_result = go_forward_result
        self._wait_for_error = wait_for_error
        self._aria_snapshot_tree = aria_snapshot_tree
        self._aria_snapshot_error = aria_snapshot_error
        self._context: _FakeBrowserContext | None = None
        self._popup_on_screenshot: _FakePage | None = None
        self.mouse = _FakeMouse()
        self.timeouts: dict[str, float | None] = {}
        self.goto_calls: list[str] = []
        self.load_states: list[str] = []
        self.filled: list[tuple[str, str]] = []
        self.clicked: list[str] = []
        self.popup_events: list[str] = []
        self.popup_handlers: list[Callable[[_FakePage], None]] = []
        self.closed = False

    @property
    def url(self) -> str:
        return self._url

    @property
    def context(self) -> _FakeBrowserContext:
        assert self._context is not None
        return self._context

    async def goto(self, url: str, *, timeout: float | None = None) -> None:
        self.goto_calls.append(url)
        self.timeouts['goto'] = timeout
        if self._goto_error is not None and url != 'about:blank':
            raise self._goto_error
        # A configured redirect lands the page on a different host than requested,
        # modelling a 3xx to a disallowed domain (but never for the bounce itself).
        self._url = self._redirect_to if self._redirect_to is not None and url != 'about:blank' else url

    async def wait_for_load_state(self, state: str, *, timeout: float | None = None) -> None:
        self.load_states.append(state)
        self.timeouts['wait_for_load_state'] = timeout

    async def title(self) -> str:
        return self._title

    async def inner_text(self, selector: str, *, timeout: float | None = None) -> str:
        self.timeouts['inner_text'] = timeout
        if self._inner_text_error is not None:
            raise self._inner_text_error
        if selector == 'body':
            return self._body
        if self._selector_raises:
            raise RuntimeError('element not found')
        return self._element_text if self._element_text is not None else f'text:{selector}'

    async def click(self, selector: str, *, timeout: float | None = None) -> None:
        self.timeouts['click'] = timeout
        if self._click_error is not None:
            raise self._click_error
        self.clicked.append(selector)

    async def fill(self, selector: str, value: str, *, timeout: float | None = None) -> None:
        self.timeouts['fill'] = timeout
        if self._fill_error is not None:
            raise self._fill_error
        self.filled.append((selector, value))

    async def screenshot(self, *, full_page: bool = False, timeout: float | None = None) -> bytes:
        self.timeouts['screenshot'] = timeout
        if self._screenshot_error is not None:
            raise self._screenshot_error
        if self._popup_on_screenshot is not None:
            for handler in self.popup_handlers:
                handler(self._popup_on_screenshot)
            await asyncio.sleep(0)
            await asyncio.sleep(0)
        return self._screenshot_bytes

    async def evaluate(self, expression: str) -> object:
        if self._evaluate_raises is not None:
            raise self._evaluate_raises
        return self._evaluate_result

    async def go_back(self, *, timeout: float | None = None) -> object | None:
        self.timeouts['go_back'] = timeout
        if self._go_back_error is not None:
            raise self._go_back_error
        return self._go_back_result

    async def go_forward(self, *, timeout: float | None = None) -> object | None:
        self.timeouts['go_forward'] = timeout
        if self._go_forward_error is not None:
            raise self._go_forward_error
        return self._go_forward_result

    async def wait_for_selector(self, selector: str, *, timeout: float | None = None) -> object:
        self.timeouts['wait_for_selector'] = timeout
        if self._wait_for_error is not None:
            raise self._wait_for_error
        return None

    async def aria_snapshot(self, *, mode: str = 'default', timeout: float | None = None) -> str:
        self.timeouts['aria_snapshot'] = timeout
        if self._aria_snapshot_error is not None:
            raise self._aria_snapshot_error
        return self._aria_snapshot_tree

    async def close(self) -> None:
        self.closed = True
        if self._close_error is not None:
            raise self._close_error

    def on(self, event: str, handler: Callable[[_FakePage], None]) -> None:
        self.popup_events.append(event)
        self.popup_handlers.append(handler)


class _ControlledNavigationPage(_FakePage):
    def __init__(self) -> None:
        super().__init__()
        self.first_navigation_started = asyncio.Event()
        self.release_first_navigation = asyncio.Event()

    async def goto(self, url: str, *, timeout: float | None = None) -> None:
        self.goto_calls.append(url)
        self._url = url
        if url == 'https://example.com/first':
            self.first_navigation_started.set()
            await self.release_first_navigation.wait()


class _HangingScreenshotPage(_FakePage):
    """A page whose `screenshot` blocks until cancelled, to drive mid-tool teardown."""

    def __init__(self) -> None:
        super().__init__()
        self.screenshot_started = asyncio.Event()

    async def screenshot(self, *, full_page: bool = False, timeout: float | None = None) -> bytes:
        self.screenshot_started.set()
        await asyncio.Event().wait()
        return self._screenshot_bytes  # pragma: no cover -- unreachable; the wait is cancelled


class _HangingTitlePage(_FakePage):
    async def title(self) -> str:
        await asyncio.Event().wait()
        return self._title  # pragma: no cover -- unreachable; the wait is cancelled


class _HangingEvaluatePage(_FakePage):
    """A page whose `evaluate` never returns, modelling a never-resolving promise."""

    async def evaluate(self, expression: str) -> object:
        await asyncio.Event().wait()
        return None  # pragma: no cover -- unreachable; the wait is cancelled


class _FakeInstallerProcess:
    def __init__(self, *, returncode: int, output: bytes, hang: bool = False) -> None:
        self.returncode = returncode
        self.output = output
        self.hang = hang
        self.communicate_started = asyncio.Event()
        self.terminated = False
        self.waited = False

    async def communicate(self) -> tuple[bytes, None]:
        self.communicate_started.set()
        if self.hang:
            await asyncio.Event().wait()
        return self.output, None

    def terminate(self) -> None:
        self.terminated = True

    async def wait(self) -> int:
        self.waited = True
        return self.returncode


def _install_fake_installer_process(monkeypatch: pytest.MonkeyPatch, process: _FakeInstallerProcess) -> None:
    async def _create_subprocess_exec(*args: str, **kwargs: int) -> _FakeInstallerProcess:
        return process

    monkeypatch.setattr(asyncio, 'create_subprocess_exec', _create_subprocess_exec)


class _FakePlaywrightBrowser:
    def __init__(self, page: _FakePage, *, close_error: Exception | None = None) -> None:
        self._page = page
        self._close_error = close_error
        self.closed = False
        self.contexts: list[_FakeBrowserContext] = []

    async def new_context(
        self, *, storage_state: StorageState | None = None, service_workers: str | None = None
    ) -> _FakeBrowserContext:
        context = _FakeBrowserContext(self._page, storage_state=storage_state, service_workers=service_workers)
        self._page._context = context
        self.contexts.append(context)
        return context

    async def close(self) -> None:
        self.closed = True
        if self._close_error is not None:
            raise self._close_error


class _FakeChromium:
    def __init__(
        self,
        page: _FakePage,
        *,
        executable_missing: bool = False,
        launch_error: Exception | None = None,
        close_error: Exception | None = None,
    ) -> None:
        self._page = page
        # An existing path so the real `os.path.exists` pre-check takes the launch
        # branch; a bogus path models a missing Chromium binary.
        self._executable_path = '/nonexistent/chromium-binary' if executable_missing else sys.executable
        self._launch_error = launch_error
        self._close_error = close_error
        self.launched: list[bool] = []
        self.connected: list[str] = []
        self.browser: _FakePlaywrightBrowser | None = None

    @property
    def executable_path(self) -> str:
        return self._executable_path

    async def launch(self, *, headless: bool) -> _FakePlaywrightBrowser:
        self.launched.append(headless)
        if self._launch_error is not None:
            raise self._launch_error
        self.browser = _FakePlaywrightBrowser(self._page, close_error=self._close_error)
        return self.browser

    async def connect_over_cdp(self, endpoint_url: str) -> _FakePlaywrightBrowser:
        self.connected.append(endpoint_url)
        self.browser = _FakePlaywrightBrowser(self._page, close_error=self._close_error)
        return self.browser


class _FakeDriver:
    def __init__(self, chromium: _FakeChromium) -> None:
        self.chromium = chromium


class _FakeDriverCM:
    def __init__(self, driver: _FakeDriver) -> None:
        self._driver = driver
        self.entered = False
        self.exited = False

    async def __aenter__(self) -> _FakeDriver:
        self.entered = True
        return self._driver

    async def __aexit__(self, *exc: object) -> bool:
        self.exited = True
        return False


def _toolset(
    page: _FakePage,
    *,
    allowed_domains: list[str] | None = None,
    block_private_addresses: bool = True,
    screenshot_on_navigate: bool = False,
    max_content_tokens: int = DEFAULT_MAX_CONTENT_TOKENS,
) -> PlaywrightBrowserToolset[None]:
    """Build a toolset whose active page is the given double."""
    state = PlaywrightBrowserState()
    state.page = page
    return PlaywrightBrowserToolset[None](
        state=state,
        allowed_domains=allowed_domains,
        block_private_addresses=block_private_addresses,
        screenshot_on_navigate=screenshot_on_navigate,
        max_content_tokens=max_content_tokens,
    )


def _ctx() -> RunContext[None]:
    return RunContext[None](deps=None, model=TestModel(), usage=RunUsage())


def _install_fake_driver(
    monkeypatch: pytest.MonkeyPatch,
    page: _FakePage,
    *,
    executable_missing: bool = False,
    launch_error: Exception | None = None,
    close_error: Exception | None = None,
) -> _FakeDriverCM:
    """Point the capability's `async_playwright` at a fake driver chain."""
    chromium = _FakeChromium(
        page, executable_missing=executable_missing, launch_error=launch_error, close_error=close_error
    )
    cm = _FakeDriverCM(_FakeDriver(chromium))
    monkeypatch.setattr(capability_module, 'async_playwright', lambda: cm)
    return cm


# --- Tool behavior ----------------------------------------------------------


class TestPlaywrightBrowserTools:
    def test_check_allowed_domain_rejects_malformed_url(self) -> None:
        assert toolset_module.check_allowed_domain('https://[::1', None) is False
        assert toolset_module.check_allowed_domain('https://[::1', ['::1']) is False

    @pytest.mark.parametrize('url', ['mailto:a@b.com', 'about:blank'])
    def test_check_allowed_domain_rejects_hostless_url_with_open_egress(self, url: str) -> None:
        assert toolset_module.check_allowed_domain(url, None) is False

    def test_check_allowed_domain_rejects_trailing_dot_host_for_blank_entries(self) -> None:
        url = 'https://169.254.169.254./'
        assert toolset_module.check_allowed_domain(url, ['']) is False
        assert toolset_module.check_allowed_domain(url, [' \t ']) is False

    def test_check_allowed_domain_rejects_userinfo_host_spoof(self) -> None:
        # CVE-2025-47241 class: the real host is `evil.com`; `allowed.com` is only
        # the userinfo, so `.hostname` resolves it correctly and the match fails.
        assert toolset_module.check_allowed_domain('https://allowed.com:pass@evil.com/', ['allowed.com']) is False

    def test_check_allowed_domain_rejects_backslash_before_url_parsing(self) -> None:
        assert toolset_module.check_allowed_domain(r'https://evil.com\@example.com/', ['example.com']) is False

    @pytest.mark.parametrize('host', ['evil-example.com', 'example.com.attacker.com'])
    def test_check_allowed_domain_rejects_sibling_domain_tricks(self, host: str) -> None:
        assert toolset_module.check_allowed_domain(f'https://{host}/', ['example.com']) is False

    def test_check_allowed_domain_matches_idn_against_punycode_allowlist(self) -> None:
        # A Unicode host and its `xn--` spelling get the same verdict against an
        # ASCII allowlist entry, in both directions.
        assert toolset_module.check_allowed_domain('https://пример.рф/path', ['xn--e1afmkfd.xn--p1ai']) is True
        assert toolset_module.check_allowed_domain('https://xn--e1afmkfd.xn--p1ai/path', ['пример.рф']) is True

    def test_check_allowed_domain_falls_back_when_entry_not_idna_encodable(self) -> None:
        # An over-long label cannot be IDNA-encoded; the entry falls back to its
        # lowercased form rather than crashing, and simply does not match.
        assert toolset_module.check_allowed_domain('https://example.com/', ['a' * 64]) is False

    async def test_navigate_returns_url_title_and_text(self) -> None:
        toolset = _toolset(_FakePage(title='Docs', body='Page text here'))
        result = await toolset.navigate('https://example.com/')
        assert result == 'URL: https://example.com/\nTitle: Docs\n\nPage text here'

    async def test_navigate_rejects_domain_outside_allowlist(self) -> None:
        page = _FakePage()
        toolset = _toolset(page, allowed_domains=['example.com'])
        result = await toolset.navigate('https://evil.com/')
        assert result == 'Error: domain not in allowed_domains: https://evil.com/'
        assert page.goto_calls == []

    async def test_navigate_allows_exact_subdomain_and_port(self) -> None:
        for url in ('https://example.com/', 'https://docs.example.com/', 'https://example.com:8443/x'):
            page = _FakePage()
            toolset = _toolset(page, allowed_domains=['example.com'])
            result = await toolset.navigate(url)
            assert isinstance(result, str) and result.startswith('URL:')
            assert page.goto_calls == [url]

    async def test_navigate_allows_ipv6_host_in_allowlist(self) -> None:
        # A bracketed IPv6 literal must match its allowlist entry (regression: the
        # old netloc.split(':') turned '[::1]' into '['). Loopback needs the
        # private-address opt-out on top of the allowlist entry.
        page = _FakePage(url='http://[::1]:8080/')
        toolset = _toolset(page, allowed_domains=['::1'], block_private_addresses=False)
        result = await toolset.navigate('http://[::1]:8080/')
        assert isinstance(result, str) and result.startswith('URL:')
        assert page.goto_calls == ['http://[::1]:8080/']

    @pytest.mark.parametrize('url', ['mailto:a@b.com', 'about:blank'])
    async def test_navigate_rejects_url_without_host(self, url: str) -> None:
        page = _FakePage()
        toolset = _toolset(page)
        result = await toolset.navigate(url)
        assert result == f'Error: domain not in allowed_domains: {url}'
        assert page.goto_calls == []

    async def test_navigate_rejects_trailing_dot_host_for_blank_allowlist_entry(self) -> None:
        page = _FakePage()
        toolset = _toolset(page, allowed_domains=[' \t '])
        url = 'https://evil.example./'
        result = await toolset.navigate(url)
        assert result == f'Error: domain not in allowed_domains: {url}'
        assert page.goto_calls == []

    async def test_navigate_rejects_malformed_url(self) -> None:
        page = _FakePage()
        toolset = _toolset(page)
        result = await toolset.navigate('https://[::1')
        assert result == 'Error: domain not in allowed_domains: https://[::1'
        assert page.goto_calls == []

    async def test_navigate_rejects_backslash_url_without_opening_page(self) -> None:
        page = _FakePage()
        url = r'https://evil.com\@example.com/'
        result = await _toolset(page, allowed_domains=['example.com']).navigate(url)
        assert result == f'Error: domain not in allowed_domains: {url}'
        assert page.goto_calls == []

    async def test_navigate_attaches_screenshot_when_configured(self) -> None:
        toolset = _toolset(_FakePage(), screenshot_on_navigate=True)
        result = await toolset.navigate('https://example.com/')
        assert isinstance(result, ToolReturn)
        assert isinstance(result.return_value, str) and result.return_value.startswith('URL:')
        assert result.content is not None
        image = result.content[0]
        assert isinstance(image, BinaryContent)
        assert image.data == b'PNG-BYTES' and image.media_type == 'image/png'

    async def test_navigate_truncates_long_page_text(self) -> None:
        toolset = _toolset(_FakePage(body='X' * 40), max_content_tokens=1)
        result = await toolset.navigate('https://example.com/')
        assert result == 'URL:'

    async def test_navigate_truncates_url_and_title_within_shared_budget(self) -> None:
        url = f'https://example.com/{"u" * 40}'
        toolset = _toolset(_FakePage(title='T' * 40), max_content_tokens=2)
        result = await toolset.navigate(url)
        assert result == 'URL: htt'

    async def test_navigate_bounces_on_redirect_to_disallowed_host(self) -> None:
        page = _FakePage(redirect_to='https://evil.com/landing')
        toolset = _toolset(page, allowed_domains=['example.com'])
        result = await toolset.navigate('https://example.com/start')
        assert result == 'Error: navigate reached a domain not in allowed_domains: https://evil.com/landing'
        assert page.goto_calls == ['https://example.com/start', 'about:blank']

    async def test_navigate_empty_allowlist_blocks_every_host(self) -> None:
        page = _FakePage()
        toolset = _toolset(page, allowed_domains=[])
        result = await toolset.navigate('https://example.com/')
        assert result == 'Error: domain not in allowed_domains: https://example.com/'
        assert page.goto_calls == []

    async def test_navigate_reports_landing_url_after_allowed_redirect(self) -> None:
        page = _FakePage(redirect_to='https://docs.example.com/landing')
        toolset = _toolset(page, allowed_domains=['example.com'])
        result = await toolset.navigate('https://example.com/start')
        assert isinstance(result, str)
        assert result.startswith('URL: https://docs.example.com/landing')
        assert page.goto_calls == ['https://example.com/start']

    async def test_navigate_waits_for_domcontentloaded(self) -> None:
        page = _FakePage()
        await _toolset(page).navigate('https://example.com/')
        assert page.load_states == ['domcontentloaded']

    async def test_concurrent_navigations_return_their_own_page_state(self) -> None:
        page = _ControlledNavigationPage()
        toolset = _toolset(page)
        first = asyncio.create_task(toolset.navigate('https://example.com/first'))
        await page.first_navigation_started.wait()

        second = asyncio.create_task(toolset.navigate('https://example.com/second'))
        await asyncio.sleep(0)
        assert page.goto_calls == ['https://example.com/first']

        page.release_first_navigation.set()
        first_result, second_result = await asyncio.gather(first, second)
        assert isinstance(first_result, str)
        assert first_result.startswith('URL: https://example.com/first')
        assert isinstance(second_result, str)
        assert second_result.startswith('URL: https://example.com/second')

    async def test_click_css_selector(self) -> None:
        page = _FakePage(body='after click')
        toolset = _toolset(page)
        result = await toolset.click('button#go')
        assert page.clicked == ['button#go']
        assert result == "Clicked 'button#go'. URL: https://example.com/\n\nafter click"

    async def test_click_pixel_coordinates(self) -> None:
        page = _FakePage()
        toolset = _toolset(page)
        await toolset.click('450,300')
        assert page.mouse.calls == [('click', 450, 300)]

    async def test_click_negative_pixel_coordinates(self) -> None:
        page = _FakePage()
        toolset = _toolset(page)
        await toolset.click('-10,-20')
        assert page.mouse.calls == [('click', -10, -20)]

    async def test_click_malformed_coordinates_are_treated_as_css(self) -> None:
        page = _FakePage()
        await _toolset(page).click('--1,2')
        assert page.clicked == ['--1,2']
        assert page.mouse.calls == []

    async def test_click_bounces_off_disallowed_domain(self) -> None:
        page = _FakePage(url='https://evil.com/landing')
        toolset = _toolset(page, allowed_domains=['example.com'])
        result = await toolset.click('a.external')
        assert result == 'Error: click reached a domain not in allowed_domains: https://evil.com/landing'
        assert page.goto_calls == ['about:blank']

    async def test_type_text_fills_field(self) -> None:
        page = _FakePage(body='typed')
        toolset = _toolset(page)
        result = await toolset.type_text('input#q', 'hello')
        assert page.filled == [('input#q', 'hello')]
        assert result == "Typed into 'input#q'.\n\ntyped"

    async def test_screenshot_returns_binary_content(self) -> None:
        toolset = _toolset(_FakePage(url='https://example.com/p'))
        result = await toolset.screenshot()
        assert isinstance(result, ToolReturn)
        assert result.return_value == 'Screenshot captured. URL: https://example.com/p'
        assert result.content is not None
        image = result.content[0]
        assert isinstance(image, BinaryContent)
        assert image.data == b'PNG-BYTES'

    async def test_screenshot_full_page(self) -> None:
        toolset = _toolset(_FakePage())
        result = await toolset.screenshot(full_page=True)
        assert isinstance(result, ToolReturn)

    async def test_screenshot_bounds_text_without_dropping_image(self) -> None:
        page = _FakePage(url=f'https://example.com/{"u" * 40}')
        result = await _toolset(page, max_content_tokens=1).screenshot()
        assert isinstance(result, ToolReturn)
        assert result.return_value == 'Scre'
        assert result.content is not None
        image = result.content[0]
        assert isinstance(image, BinaryContent)
        assert image.data == b'PNG-BYTES'

    async def test_screenshot_over_size_limit_returns_error_not_image(self) -> None:
        png = b'x' * (toolset_module._MAX_SCREENSHOT_BYTES + 1)
        result = await _toolset(_FakePage(screenshot_bytes=png)).screenshot(full_page=True)
        assert isinstance(result, str)
        assert result.startswith(f'Error: screenshot is {len(png)} bytes')
        assert 'full_page=False' in result

    async def test_navigate_omits_oversized_screenshot_attachment(self) -> None:
        png = b'x' * (toolset_module._MAX_SCREENSHOT_BYTES + 1)
        toolset = _toolset(_FakePage(screenshot_bytes=png), screenshot_on_navigate=True)
        result = await toolset.navigate('https://example.com/')
        assert isinstance(result, str)  # no ToolReturn: the image is dropped, the text result survives
        assert result.startswith('URL: https://example.com/')
        assert 'Error: screenshot is' in result

    async def test_get_text_with_selector(self) -> None:
        toolset = _toolset(_FakePage())
        assert await toolset.get_text('h1') == 'text:h1'

    async def test_get_text_selector_error(self) -> None:
        toolset = _toolset(_FakePage(selector_raises=True))
        result = await toolset.get_text('#missing')
        assert result.startswith("Error getting text from '#missing':")

    async def test_get_text_surfaces_playwright_error(self) -> None:
        toolset = _toolset(_FakePage(inner_text_error=PlaywrightTimeoutError('inner_text timed out')))
        result = await toolset.get_text('h1')
        assert result == "Error getting text from 'h1': inner_text timed out"

    async def test_get_text_full_page_surfaces_playwright_error(self) -> None:
        toolset = _toolset(_FakePage(inner_text_error=PlaywrightTimeoutError('body timed out')))
        result = await toolset.get_text()
        assert result == (
            'Error: get_text timed out after 30000ms. The element may not exist or the page may be slow; '
            'try a different selector, or navigate again.'
        )

    async def test_get_text_full_page(self) -> None:
        toolset = _toolset(_FakePage(body='full page text'))
        assert await toolset.get_text() == 'full page text'

    async def test_get_text_selector_truncated(self) -> None:
        toolset = _toolset(_FakePage(element_text='Y' * 40), max_content_tokens=1)
        result = await toolset.get_text('article')
        assert result == 'Y' * 4

    async def test_scroll_window(self) -> None:
        page = _FakePage(body='scrolled')
        toolset = _toolset(page)
        result = await toolset.scroll('down')
        assert result == 'Scrolled down.\n\nscrolled'
        assert page.mouse.calls == []

    async def test_scroll_localized(self) -> None:
        page = _FakePage()
        toolset = _toolset(page)
        await toolset.scroll('up', 5, 6)
        assert page.mouse.calls == [('move', 5, 6), ('wheel', 0, -300)]

    async def test_scroll_invalid_direction(self) -> None:
        toolset = _toolset(_FakePage())
        result = await toolset.scroll('sideways')
        assert result == "Error: invalid direction 'sideways'; use up/down/left/right"

    async def test_go_back(self) -> None:
        toolset = _toolset(_FakePage(url='https://example.com/prev', body='prev'))
        result = await toolset.go_back()
        assert result == 'Went back. URL: https://example.com/prev\n\nprev'

    async def test_go_back_bounces_off_disallowed_domain(self) -> None:
        page = _FakePage(url='https://evil.com/')
        toolset = _toolset(page, allowed_domains=['example.com'])
        result = await toolset.go_back()
        assert result == 'Error: go_back reached a domain not in allowed_domains: https://evil.com/'

    async def test_go_back_reports_empty_history(self) -> None:
        result = await _toolset(_FakePage(go_back_result=None)).go_back()
        assert result == 'No previous page in browser history.'

    async def test_go_forward(self) -> None:
        toolset = _toolset(_FakePage(url='https://example.com/next', body='next'))
        result = await toolset.go_forward()
        assert result == 'Went forward. URL: https://example.com/next\n\nnext'

    async def test_go_forward_bounces_off_disallowed_domain(self) -> None:
        page = _FakePage(url='https://evil.com/')
        toolset = _toolset(page, allowed_domains=['example.com'])
        result = await toolset.go_forward()
        assert result == 'Error: go_forward reached a domain not in allowed_domains: https://evil.com/'

    async def test_go_forward_reports_empty_history(self) -> None:
        result = await _toolset(_FakePage(go_forward_result=None)).go_forward()
        assert result == 'No next page in browser history.'

    async def test_execute_js_string_result(self) -> None:
        toolset = _toolset(_FakePage(evaluate_result='the title'))
        assert await toolset.execute_js('document.title') == 'the title'

    async def test_execute_js_json_result(self) -> None:
        toolset = _toolset(_FakePage(evaluate_result={'a': 1}))
        assert await toolset.execute_js('({a:1})') == '{"a": 1}'

    @pytest.mark.parametrize(
        ('value', 'prefix'),
        [
            ('S' * 40, 'SSSS'),
            ({'value': 'J' * 40}, '{"va'),
        ],
    )
    async def test_execute_js_truncates_string_and_serialized_results(self, value: object, prefix: str) -> None:
        toolset = _toolset(_FakePage(evaluate_result=value), max_content_tokens=1)
        assert await toolset.execute_js('largeResult') == prefix

    def test_truncation_marker_fits_inside_budget(self) -> None:
        result = toolset_module._truncate('X' * 200, 80)
        assert len(result) == 80
        assert result.endswith('[... tool output truncated at 80 characters]')
        assert toolset_module._truncate('content', 0) == ''

    async def test_execute_js_null_result(self) -> None:
        toolset = _toolset(_FakePage(evaluate_result=None))
        assert await toolset.execute_js('void 0') == 'undefined'

    async def test_execute_js_error(self) -> None:
        toolset = _toolset(_FakePage(evaluate_raises=ValueError('boom')))
        assert await toolset.execute_js('bad(') == 'JS error: boom'

    async def test_execute_js_bounces_off_disallowed_domain(self) -> None:
        page = _FakePage(url='https://evil.com/', evaluate_result='x')
        toolset = _toolset(page, allowed_domains=['example.com'])
        result = await toolset.execute_js('location.href="https://evil.com"')
        assert result == 'Error: execute_js reached a domain not in allowed_domains: https://evil.com/'


_BLOCKED_ADDRESS_HOSTS = [
    '127.0.0.1',
    '10.0.0.5',
    '172.16.0.1',
    '192.168.1.1',
    '169.254.169.254',
    '0.0.0.0',
    '100.64.0.1',
    '224.0.0.1',
    '240.0.0.1',
    '::1',
    'fe80::1',
    'fd00::1',
    '::ffff:127.0.0.1',
    'localhost',
    'LOCALHOST',
    'app.localhost',
    'localhost.',
    '169.254.169.254.',
]

_PUBLIC_ADDRESS_HOSTS = ['example.com', '8.8.8.8', '2606:4700:4700::1111', 'localhost.example.com', 'my-localhost.dev']


class TestPrivateAddressBlocking:
    @pytest.mark.parametrize('host', _BLOCKED_ADDRESS_HOSTS)
    def test_is_blocked_address_detects_reserved_ranges(self, host: str) -> None:
        assert toolset_module.is_blocked_address(host) is True

    @pytest.mark.parametrize('host', _PUBLIC_ADDRESS_HOSTS)
    def test_is_blocked_address_allows_public_hosts(self, host: str) -> None:
        assert toolset_module.is_blocked_address(host) is False

    def test_blocked_navigation_reason_names_the_denying_policy(self) -> None:
        reason = toolset_module.blocked_navigation_reason
        assert reason('https://example.com/', None, True) is None
        assert reason('http://127.0.0.1/', None, True) == 'blocked private or link-local address'
        assert reason('http://127.0.0.1/', None, False) is None
        # An allowlist miss wins the message even for a private address...
        assert reason('http://127.0.0.1/', ['example.com'], True) == 'domain not in allowed_domains'
        # ...but an allowlisted private address is still blocked (deny over allow).
        assert reason('http://127.0.0.1/', ['127.0.0.1'], True) == 'blocked private or link-local address'
        assert reason('about:blank', None, True) == 'domain not in allowed_domains'

    async def test_navigate_blocks_private_address_under_open_egress(self) -> None:
        page = _FakePage()
        toolset = _toolset(page)
        result = await toolset.navigate('http://169.254.169.254/latest/meta-data/')
        assert result == 'Error: blocked private or link-local address: http://169.254.169.254/latest/meta-data/'
        assert page.goto_calls == []

    async def test_navigate_opt_out_reaches_private_address(self) -> None:
        page = _FakePage(url='http://127.0.0.1:8000/')
        toolset = _toolset(page, block_private_addresses=False)
        result = await toolset.navigate('http://127.0.0.1:8000/')
        assert isinstance(result, str) and result.startswith('URL:')
        assert page.goto_calls == ['http://127.0.0.1:8000/']

    async def test_navigate_blocks_allowlisted_private_address_without_opt_out(self) -> None:
        page = _FakePage()
        toolset = _toolset(page, allowed_domains=['127.0.0.1'])
        result = await toolset.navigate('http://127.0.0.1:8000/admin')
        assert result == 'Error: blocked private or link-local address: http://127.0.0.1:8000/admin'
        assert page.goto_calls == []

    async def test_navigate_bounces_on_redirect_to_private_address(self) -> None:
        page = _FakePage(redirect_to='http://169.254.169.254/')
        toolset = _toolset(page)
        result = await toolset.navigate('https://example.com/start')
        assert result == 'Error: navigate reached a blocked private or link-local address: http://169.254.169.254/'
        assert page.goto_calls == ['https://example.com/start', 'about:blank']

    async def test_click_bounces_off_private_address(self) -> None:
        page = _FakePage(url='http://127.0.0.1:8000/admin')
        toolset = _toolset(page)
        result = await toolset.click('a.local')
        assert result == 'Error: click reached a blocked private or link-local address: http://127.0.0.1:8000/admin'
        assert page.goto_calls == ['about:blank']

    async def test_execute_js_bounces_off_private_address(self) -> None:
        page = _FakePage(url='http://169.254.169.254/', evaluate_result='x')
        toolset = _toolset(page)
        result = await toolset.execute_js('location.href="http://169.254.169.254/"')
        assert result == 'Error: execute_js reached a blocked private or link-local address: http://169.254.169.254/'


class TestPlaywrightErrorHandling:
    async def test_navigate_timeout_returns_bounded_error(self) -> None:
        page = _FakePage(goto_error=PlaywrightTimeoutError('Timeout 30000ms exceeded.'))
        result = await _toolset(page).navigate('https://example.com/')
        assert result == (
            'Error: navigate timed out after 30000ms. The element may not exist or the page may be slow; '
            'try a different selector, or navigate again.'
        )

    async def test_navigate_preserves_net_error_code(self) -> None:
        page = _FakePage(goto_error=PlaywrightError('page.goto: net::ERR_NAME_NOT_RESOLVED at https://nope.invalid/'))
        result = await _toolset(page).navigate('https://nope.invalid/')
        assert isinstance(result, str)
        assert 'net::ERR_NAME_NOT_RESOLVED' in result

    async def test_click_preserves_strict_mode_match_count(self) -> None:
        page = _FakePage(click_error=PlaywrightError('strict mode violation: locator resolved to 3 elements'))
        result = await _toolset(page).click('button')
        assert result == 'Error: click failed: strict mode violation: locator resolved to 3 elements'

    async def test_click_timeout_returns_bounded_error(self) -> None:
        page = _FakePage(click_error=PlaywrightTimeoutError('Timeout 30000ms exceeded.'))
        result = await _toolset(page).click('button#missing')
        assert result.startswith('Error: click timed out after 30000ms.')

    async def test_type_text_timeout_returns_bounded_error(self) -> None:
        page = _FakePage(fill_error=PlaywrightTimeoutError('Timeout 30000ms exceeded.'))
        result = await _toolset(page).type_text('input#missing', 'hi')
        assert result.startswith('Error: type_text timed out after 30000ms.')

    async def test_scroll_playwright_error_returns_error_string(self) -> None:
        page = _FakePage(evaluate_raises=PlaywrightError('scroll blew up'))
        result = await _toolset(page).scroll('down')
        assert result == 'Error: scroll failed: scroll blew up'

    async def test_screenshot_playwright_error_returns_error_string(self) -> None:
        page = _FakePage(screenshot_error=PlaywrightError('screenshot failed'))
        result = await _toolset(page).screenshot()
        assert result == 'Error: screenshot failed: screenshot failed'

    async def test_go_back_target_closed_reports_crash(self) -> None:
        page = _FakePage(go_back_error=TargetClosedError())
        result = await _toolset(page).go_back()
        assert result == (
            'Error: go_back failed: the browser or page was closed unexpectedly. '
            'Browser tools may be unavailable for the rest of this run.'
        )

    async def test_go_forward_playwright_error_returns_error_string(self) -> None:
        page = _FakePage(go_forward_error=PlaywrightError('history unavailable'))
        result = await _toolset(page).go_forward()
        assert result == 'Error: go_forward failed: history unavailable'


_NEGATIVE_TIMEOUT_CALLS: list[Callable[[PlaywrightBrowserToolset[None]], Awaitable[object]]] = [
    lambda t: t.navigate('https://example.com/', timeout_ms=-1),
    lambda t: t.click('button', timeout_ms=-1),
    lambda t: t.type_text('input', 'x', timeout_ms=-1),
    lambda t: t.get_text('h1', timeout_ms=-1),
    lambda t: t.screenshot(timeout_ms=-1),
    lambda t: t.go_back(timeout_ms=-1),
    lambda t: t.go_forward(timeout_ms=-1),
    lambda t: t.wait_for(selector='.x', timeout_ms=-1),
    lambda t: t.snapshot(timeout_ms=-1),
    lambda t: t.scroll('down', timeout_ms=-1),
    lambda t: t.execute_js('1 + 1', timeout_ms=-1),
]


class TestPerCallTimeout:
    async def test_navigate_override_reaches_every_playwright_operation(self) -> None:
        page = _FakePage()
        result = await _toolset(page, screenshot_on_navigate=True).navigate('https://example.com/', timeout_ms=1111)
        assert isinstance(result, ToolReturn)
        assert page.timeouts == {
            'goto': 1111,
            'wait_for_load_state': 1111,
            'inner_text': 1111,
            'screenshot': 1111,
        }

    async def test_zero_timeout_disables_title_deadline(self) -> None:
        result = await _toolset(_FakePage()).navigate('https://example.com/', timeout_ms=0)
        assert isinstance(result, str)
        assert result.startswith('URL:')

    async def test_navigate_title_honors_timeout_override(self) -> None:
        # The reported deadline is the override the call actually ran under, not
        # the capability default.
        result = await _toolset(_HangingTitlePage()).navigate('https://example.com/', timeout_ms=1)
        assert result == (
            'Error: navigate timed out after 1ms. The element may not exist or the page may be slow; '
            'try a different selector, or navigate again.'
        )

    async def test_execute_js_bounds_unresolved_promise(self) -> None:
        # `page.evaluate` waits on returned promises indefinitely; the external
        # deadline turns that into a bounded error instead of a held lock.
        result = await _toolset(_HangingEvaluatePage()).execute_js('new Promise(() => {})', timeout_ms=1)
        assert isinstance(result, str)
        assert result.startswith('Error: execute_js timed out after 1ms.')

    async def test_scroll_bounds_hung_evaluate(self) -> None:
        result = await _toolset(_HangingEvaluatePage()).scroll('down', timeout_ms=1)
        assert result.startswith('Error: scroll timed out after 1ms.')

    async def test_override_reaches_each_playwright_call(self) -> None:
        page = _FakePage()
        toolset = _toolset(page)
        await toolset.navigate('https://example.com/', timeout_ms=1111)
        assert page.timeouts['goto'] == 1111
        # Trailing operations (load wait, page-text read) run under the same
        # override, not the capability default.
        await toolset.click('button#go', timeout_ms=2222)
        assert page.timeouts['click'] == 2222
        assert page.timeouts['wait_for_load_state'] == 2222
        assert page.timeouts['inner_text'] == 2222
        await toolset.type_text('input#q', 'hi', timeout_ms=3333)
        assert page.timeouts['fill'] == 3333
        assert page.timeouts['inner_text'] == 3333
        await toolset.get_text('h1', timeout_ms=4444)
        assert page.timeouts['inner_text'] == 4444
        await toolset.get_text(timeout_ms=4545)
        assert page.timeouts['inner_text'] == 4545
        await toolset.screenshot(timeout_ms=5555)
        assert page.timeouts['screenshot'] == 5555
        await toolset.go_back(timeout_ms=6666)
        assert page.timeouts['go_back'] == 6666
        assert page.timeouts['wait_for_load_state'] == 6666
        assert page.timeouts['inner_text'] == 6666
        await toolset.go_forward(timeout_ms=7777)
        assert page.timeouts['go_forward'] == 7777
        assert page.timeouts['wait_for_load_state'] == 7777
        await toolset.wait_for(selector='.ready', timeout_ms=8888)
        assert page.timeouts['wait_for_selector'] == 8888
        assert page.timeouts['inner_text'] == 8888
        await toolset.snapshot(timeout_ms=9999)
        assert page.timeouts['aria_snapshot'] == 9999
        await toolset.scroll('down', timeout_ms=1234)
        assert page.timeouts['inner_text'] == 1234

    async def test_none_falls_back_to_capability_default(self) -> None:
        page = _FakePage()
        await _toolset(page).click('button#go')
        assert page.timeouts['click'] == DEFAULT_TIMEOUT_MS

    @pytest.mark.parametrize('call', _NEGATIVE_TIMEOUT_CALLS)
    async def test_negative_override_returns_bounded_error(
        self, call: Callable[[PlaywrightBrowserToolset[None]], Awaitable[object]]
    ) -> None:
        result = await call(_toolset(_FakePage()))
        assert result == 'Error: timeout_ms must be greater than or equal to 0.'


class TestWaitFor:
    async def test_wait_for_selector_returns_page_text(self) -> None:
        page = _FakePage(body='loaded content')
        result = await _toolset(page).wait_for(selector='.ready')
        assert result == "Found '.ready'.\n\nloaded content"

    async def test_wait_for_text_uses_text_engine(self) -> None:
        page = _FakePage(body='dynamic text')
        result = await _toolset(page).wait_for(text='Submit')
        assert result == "Found 'text=Submit'.\n\ndynamic text"

    async def test_wait_for_requires_exactly_one_argument(self) -> None:
        toolset = _toolset(_FakePage())
        expected = 'Error: wait_for requires exactly one of selector or text.'
        assert await toolset.wait_for() == expected
        assert await toolset.wait_for(selector='.x', text='y') == expected

    async def test_wait_for_page_text_error_returns_mapped_error(self) -> None:
        # The page can close between a successful `wait_for_selector` and the
        # body read; that must stay a tool result, not abort the run.
        page = _FakePage(inner_text_error=TargetClosedError('Target page closed'))
        result = await _toolset(page).wait_for(selector='.ready')
        assert result == (
            'Error: wait_for failed: the browser or page was closed unexpectedly. '
            'Browser tools may be unavailable for the rest of this run.'
        )

    async def test_wait_for_timeout_returns_mapped_error(self) -> None:
        page = _FakePage(wait_for_error=PlaywrightTimeoutError('Timeout 30000ms exceeded.'))
        result = await _toolset(page).wait_for(selector='.never')
        assert result == (
            'Error: wait_for timed out after 30000ms. The element may not exist or the page may be slow; '
            'try a different selector, or navigate again.'
        )


class TestSnapshot:
    async def test_snapshot_returns_aria_tree(self) -> None:
        result = await _toolset(_FakePage()).snapshot()
        assert result == '- heading "Example" [ref=e1]\n- button "Go" [ref=e2]'

    async def test_snapshot_truncated_to_budget(self) -> None:
        page = _FakePage(aria_snapshot_tree='Z' * 40)
        result = await _toolset(page, max_content_tokens=1).snapshot()
        assert result == 'Z' * 4

    async def test_snapshot_playwright_error_returns_error_string(self) -> None:
        page = _FakePage(aria_snapshot_error=PlaywrightError('snapshot failed'))
        result = await _toolset(page).snapshot()
        assert result == 'Error: snapshot failed: snapshot failed'

    async def test_click_aria_ref_uses_selector_path_not_coordinates(self) -> None:
        page = _FakePage()
        await _toolset(page).click('aria-ref=e2')
        assert page.clicked == ['aria-ref=e2']
        assert page.mouse.calls == []


# --- State / ensure_page ----------------------------------------------------


class TestPlaywrightBrowserState:
    def test_toolset_validates_max_content_tokens(self) -> None:
        state = PlaywrightBrowserState()
        with pytest.raises(ValueError, match='^max_content_tokens must be greater than or equal to 0$'):
            PlaywrightBrowserToolset[None](state=state, max_content_tokens=-1)
        PlaywrightBrowserToolset[None](state=state, max_content_tokens=0)

    def test_toolset_validates_timeout_ms(self) -> None:
        state = PlaywrightBrowserState()
        with pytest.raises(ValueError, match='^timeout_ms must be greater than or equal to 0$'):
            PlaywrightBrowserToolset[None](state=state, timeout_ms=-1)
        PlaywrightBrowserToolset[None](state=state, timeout_ms=0)  # 0 = no deadline, accepted

    async def test_tool_raises_when_wrap_run_not_active(self) -> None:
        toolset = PlaywrightBrowserToolset[None](state=PlaywrightBrowserState())
        with pytest.raises(RuntimeError, match='PlaywrightBrowser is not running'):
            await toolset.screenshot()

    async def test_tool_raises_on_launch_error(self) -> None:
        state = PlaywrightBrowserState()
        state.launch_error = 'Chromium is not installed.'
        toolset = PlaywrightBrowserToolset[None](state=state)
        with pytest.raises(RuntimeError, match='Chromium is not installed'):
            await toolset.screenshot()

    async def test_concurrent_ensure_page_launches_once(self) -> None:
        # Two tool calls that race before the page exists must launch Chromium once.
        state = PlaywrightBrowserState()
        launches: list[int] = []

        async def _launch() -> None:
            launches.append(1)
            await asyncio.sleep(0)  # yield so the second caller blocks on the lock
            state.page = _FakePage()

        state.lazy_launcher = _launch
        first, second = await asyncio.gather(state.ensure_page(), state.ensure_page())
        assert launches == [1]
        assert first is second

    async def test_combined_operation_and_launch_lock_launches_once(self) -> None:
        # Two first tool calls contend both the toolset operation lock and the
        # lazy-launch lock. The operation lock is outer and the launch lock inner,
        # so the launch runs once and both calls observe the same page state.
        state = PlaywrightBrowserState()
        launches: list[int] = []

        async def _launch() -> None:
            launches.append(1)
            await asyncio.sleep(0)
            state.page = _FakePage(body='shared body')

        state.lazy_launcher = _launch
        toolset = PlaywrightBrowserToolset[None](state=state)
        first, second = await asyncio.gather(toolset.get_text(), toolset.get_text())
        assert launches == [1]
        assert first == second == 'shared body'

    async def test_concurrent_ensure_page_failed_launch_raises_once(self) -> None:
        state = PlaywrightBrowserState()
        launches: list[int] = []

        async def _launch() -> None:
            launches.append(1)
            await asyncio.sleep(0)
            state.launch_error = 'Chromium is not installed.'

        state.lazy_launcher = _launch
        results = await asyncio.gather(state.ensure_page(), state.ensure_page(), return_exceptions=True)
        assert launches == [1]  # second caller sees the error, does not relaunch
        assert all(isinstance(r, RuntimeError) for r in results)


# --- Capability hooks -------------------------------------------------------


class TestPlaywrightBrowserHooks:
    def test_capability_validates_max_content_tokens(self) -> None:
        with pytest.raises(ValueError, match='^max_content_tokens must be greater than or equal to 0$'):
            PlaywrightBrowser[None](max_content_tokens=-1)
        PlaywrightBrowser[None](max_content_tokens=0)

    def test_capability_validates_timeout_ms(self) -> None:
        with pytest.raises(ValueError, match='^timeout_ms must be greater than or equal to 0$'):
            PlaywrightBrowser[None](timeout_ms=-1)
        PlaywrightBrowser[None](timeout_ms=0)

    def test_get_instructions_reports_allowlist(self) -> None:
        instructions = PlaywrightBrowser[None](allowed_domains=['a.com', 'b.com']).get_instructions()
        text = instructions(_ctx())
        assert text is not None and 'Allowed domains: a.com, b.com' in text

    def test_get_instructions_reports_all_domains(self) -> None:
        instructions = PlaywrightBrowser[None]().get_instructions()
        text = instructions(_ctx())
        assert text is not None and 'Allowed domains: all' in text

    def test_get_instructions_reports_empty_allowlist_as_none(self) -> None:
        # An empty allowlist blocks every domain, so the model must be told 'none',
        # not 'all' (which list-truthiness would collapse it to).
        instructions = PlaywrightBrowser[None](allowed_domains=[]).get_instructions()
        text = instructions(_ctx())
        assert text is not None and 'Allowed domains: none' in text

    def test_get_instructions_notes_private_address_block(self) -> None:
        text = PlaywrightBrowser[None]().get_instructions()(_ctx())
        assert text is not None
        assert 'Allowed domains: all (private/internal addresses blocked)' in text

    def test_get_instructions_omits_private_note_when_opted_out(self) -> None:
        text = PlaywrightBrowser[None](block_private_addresses=False).get_instructions()(_ctx())
        assert text is not None
        assert 'private/internal addresses blocked' not in text
        assert 'Allowed domains: all' in text

    def test_get_instructions_suppressed_on_launch_error(self) -> None:
        browser = PlaywrightBrowser[None]()
        browser._state.launch_error = 'boom'
        assert browser.get_instructions()(_ctx()) is None

    async def test_prepare_tools_preserves_upstream_unapproved_kind(self) -> None:
        browser = PlaywrightBrowser[None]()
        defs = [
            ToolDefinition(
                name='navigate', parameters_json_schema={'type': 'object'}, kind='unapproved', toolset_id='playwright'
            ),
            ToolDefinition(name='other', parameters_json_schema={'type': 'object'}, kind='function', toolset_id='misc'),
        ]
        result = await browser.prepare_tools(_ctx(), defs)
        by_name = {td.name: td for td in result}
        assert by_name['navigate'].kind == 'unapproved'
        assert by_name['other'].kind == 'function'

    async def test_prepare_tools_hides_tools_on_launch_error(self) -> None:
        browser = PlaywrightBrowser[None]()
        browser._state.launch_error = 'boom'
        defs = [
            ToolDefinition(
                name='navigate', parameters_json_schema={'type': 'object'}, kind='function', toolset_id='playwright'
            ),
            ToolDefinition(name='other', parameters_json_schema={'type': 'object'}, kind='function', toolset_id='misc'),
        ]
        result = await browser.prepare_tools(_ctx(), defs)
        assert [td.name for td in result] == ['other']

    async def test_prepare_tools_ignores_same_named_foreign_tool(self) -> None:
        # A `navigate` from another toolset must be left untouched (not re-approved,
        # not hidden), since matching is by `toolset_id`, not tool name.
        foreign = ToolDefinition(
            name='navigate', parameters_json_schema={'type': 'object'}, kind='unapproved', toolset_id='other_toolset'
        )
        browser = PlaywrightBrowser[None]()
        assert (await browser.prepare_tools(_ctx(), [foreign]))[0].kind == 'unapproved'
        browser._state.launch_error = 'boom'
        assert [td.name for td in await browser.prepare_tools(_ctx(), [foreign])] == ['navigate']

    async def test_for_run_isolates_state(self) -> None:
        browser = PlaywrightBrowser[None]()
        first = await browser.for_run(_ctx())
        second = await browser.for_run(_ctx())
        assert first._state is not second._state
        assert first._state is not browser._state

    def test_from_spec_round_trips_fields(self) -> None:
        browser = PlaywrightBrowser[None].from_spec(
            headless=False,
            allowed_domains=['x.com'],
            block_private_addresses=False,
            screenshot_on_navigate=True,
            max_content_tokens=100,
            timeout_ms=5000,
            auto_install_chromium=True,
            cdp_url='http://localhost:9222',
        )
        assert browser.headless is False
        assert browser.allowed_domains == ['x.com']
        assert browser.block_private_addresses is False
        assert browser.screenshot_on_navigate is True
        assert browser.max_content_tokens == 100
        assert browser.timeout_ms == 5000
        assert browser.auto_install_chromium is True
        assert browser.cdp_url == 'http://localhost:9222'

    def test_from_spec_defaults_to_open_egress(self) -> None:
        browser = PlaywrightBrowser[None].from_spec()
        assert browser.allowed_domains is None
        assert browser.block_private_addresses is True
        assert browser.auto_install_chromium is False
        assert browser.cdp_url is None
        assert browser.storage_state is None

    def test_repr_omits_storage_state(self) -> None:
        # `repr` reaches diagnostics and logs; session cookies must not ride along.
        rendered = repr(PlaywrightBrowser[None](storage_state=_STORAGE_STATE, cdp_url='http://localhost:9222'))
        assert 'abc' not in rendered
        assert 'storage_state' not in rendered
        assert 'http://localhost:9222' in rendered  # non-secret configuration still shows

    def test_from_spec_refuses_storage_state(self) -> None:
        # Session credentials stay out of a spec: naming it fails loudly rather
        # than moving cookies into whatever stores the spec.
        with pytest.raises(TypeError, match='storage_state'):
            PlaywrightBrowser[None].from_spec(storage_state=_STORAGE_STATE)  # pyright: ignore[reportCallIssue]


class TestDurabilityRejection:
    def test_accepts_innermost_non_durability_capability(self) -> None:
        # `innermost` is not a durability marker: `InputGuard` declares it too,
        # so ordering alone would reject the supported guard-plus-browser pairing.
        class _Innermost(AbstractCapability[object]):
            def get_ordering(self) -> CapabilityOrdering:
                return CapabilityOrdering(position='innermost')

        Agent(TestModel(), capabilities=[PlaywrightBrowser(), _Innermost()])

    def test_accepts_capability_with_non_durability_ordering(self) -> None:
        class _Outermost(AbstractCapability[object]):
            def get_ordering(self) -> CapabilityOrdering:
                return CapabilityOrdering(position='outermost')

        Agent(TestModel(), capabilities=[PlaywrightBrowser(), _Outermost()])

    def test_rejects_temporal_durability_at_construction(self) -> None:
        pytest.importorskip('temporalio')
        from pydantic_ai.durable_exec.temporal import TemporalDurability

        with pytest.raises(UserError, match='does not support durable execution'):
            Agent(TestModel(), capabilities=[PlaywrightBrowser(), TemporalDurability()])


class TestChromiumAutoInstall:
    @pytest.mark.parametrize(
        ('returncode', 'output', 'expected'),
        [
            (0, b'', None),
            (1, b'download failed', 'download failed'),
        ],
    )
    async def test_returns_installer_result(
        self,
        monkeypatch: pytest.MonkeyPatch,
        returncode: int,
        output: bytes,
        expected: str | None,
    ) -> None:
        process = _FakeInstallerProcess(returncode=returncode, output=output)
        _install_fake_installer_process(monkeypatch, process)
        assert await capability_module._auto_install_chromium() == expected
        assert process.terminated is False
        assert process.waited is False

    async def test_cancellation_terminates_and_waits_for_installer(self, monkeypatch: pytest.MonkeyPatch) -> None:
        process = _FakeInstallerProcess(returncode=-15, output=b'', hang=True)
        _install_fake_installer_process(monkeypatch, process)
        task = asyncio.create_task(capability_module._auto_install_chromium())
        await process.communicate_started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert process.terminated is True
        assert process.waited is True


# --- Lifecycle through Agent + wrap_run -------------------------------------


class TestPlaywrightBrowserLifecycle:
    async def test_lazy_launch_and_cleanup(self, monkeypatch: pytest.MonkeyPatch) -> None:
        page = _FakePage()
        cm = _install_fake_driver(monkeypatch, page)
        agent = Agent(TestModel(call_tools=['screenshot']), capabilities=[PlaywrightBrowser()])
        await agent.run('use the browser')
        chromium = cm._driver.chromium
        assert chromium.launched == [True]
        assert page.popup_events == ['popup']
        assert page.context.routes == ['**/*']  # the default private-address block installs the route guard
        assert chromium.browser is not None and chromium.browser.closed is True
        assert cm.exited is True

    async def test_storage_state_reaches_new_context(self, monkeypatch: pytest.MonkeyPatch) -> None:
        page = _FakePage()
        cm = _install_fake_driver(monkeypatch, page)
        agent = Agent(
            TestModel(call_tools=['screenshot']),
            capabilities=[PlaywrightBrowser(storage_state=_STORAGE_STATE)],
        )
        await agent.run('screenshot the page')
        browser = cm._driver.chromium.browser
        assert browser is not None
        assert [c.storage_state for c in browser.contexts] == [_STORAGE_STATE]

    async def test_cdp_url_attaches_instead_of_launching(self, monkeypatch: pytest.MonkeyPatch) -> None:
        page = _FakePage()
        cm = _install_fake_driver(monkeypatch, page)
        agent = Agent(
            TestModel(call_tools=['screenshot']),
            capabilities=[PlaywrightBrowser(cdp_url='http://localhost:9222')],
        )
        await agent.run('screenshot the page')
        chromium = cm._driver.chromium
        assert chromium.connected == ['http://localhost:9222']
        assert chromium.launched == []
        assert chromium.browser is not None and chromium.browser.closed is True

    async def test_cdp_url_skips_the_missing_binary_check(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Attaching needs no local Chromium, so an absent binary must not hide the
        # tools or trigger the install hint.
        page = _FakePage()
        cm = _install_fake_driver(monkeypatch, page, executable_missing=True)
        agent = Agent(
            TestModel(call_tools=['screenshot']),
            capabilities=[PlaywrightBrowser(cdp_url='http://localhost:9222', auto_install_chromium=True)],
        )
        await agent.run('screenshot the page')
        chromium = cm._driver.chromium
        assert chromium.connected == ['http://localhost:9222']
        assert chromium.launched == []  # no local binary consulted, so no install hint and no download

    async def test_cdp_url_context_still_applies_storage_state_and_guards(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        page = _FakePage()
        cm = _install_fake_driver(monkeypatch, page)
        agent = Agent(
            TestModel(call_tools=['screenshot']),
            capabilities=[PlaywrightBrowser(cdp_url='http://localhost:9222', storage_state=_STORAGE_STATE)],
        )
        await agent.run('screenshot the page')
        browser = cm._driver.chromium.browser
        assert browser is not None
        assert [c.storage_state for c in browser.contexts] == [_STORAGE_STATE]
        assert [c.service_workers for c in browser.contexts] == ['block']
        assert page.context.routes == ['**/*']

    async def test_storage_state_defaults_to_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        page = _FakePage()
        cm = _install_fake_driver(monkeypatch, page)
        agent = Agent(TestModel(call_tools=['screenshot']), capabilities=[PlaywrightBrowser()])
        await agent.run('screenshot the page')
        browser = cm._driver.chromium.browser
        assert browser is not None
        assert [c.storage_state for c in browser.contexts] == [None]

    async def test_context_blocks_service_workers(self, monkeypatch: pytest.MonkeyPatch) -> None:
        page = _FakePage()
        cm = _install_fake_driver(monkeypatch, page)
        agent = Agent(TestModel(call_tools=['screenshot']), capabilities=[PlaywrightBrowser()])
        await agent.run('screenshot the page')
        browser = cm._driver.chromium.browser
        assert browser is not None
        # Service-worker traffic bypasses context routes, so workers are blocked
        # to keep the route guard authoritative for all requests.
        assert [c.service_workers for c in browser.contexts] == ['block']

    async def test_no_route_guard_when_open_egress_and_private_opt_out(self, monkeypatch: pytest.MonkeyPatch) -> None:
        page = _FakePage()
        _install_fake_driver(monkeypatch, page)
        agent = Agent(
            TestModel(call_tools=['screenshot']),
            capabilities=[PlaywrightBrowser(block_private_addresses=False)],
        )
        await agent.run('screenshot the page')
        assert page.context.routes == []

    async def test_route_guard_blocks_private_navigation_under_open_egress(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        page = _FakePage()
        _install_fake_driver(monkeypatch, page)
        agent = Agent(TestModel(call_tools=['screenshot']), capabilities=[PlaywrightBrowser()])
        await agent.run('screenshot the page')
        request_page = _FakeRequestPage()
        blocked = await page.context.dispatch(
            _FakeRequest('http://169.254.169.254/latest/', navigation=True, frame=request_page.main_frame)
        )
        assert blocked.aborted is True
        assert blocked.continued is False
        allowed = await page.context.dispatch(
            _FakeRequest('https://example.com/', navigation=True, frame=request_page.main_frame)
        )
        assert allowed.aborted is False
        assert allowed.continued is True

    async def test_only_popup_listener_registered_not_dialog(self, monkeypatch: pytest.MonkeyPatch) -> None:
        page = _FakePage()
        _install_fake_driver(monkeypatch, page)
        agent = Agent(TestModel(call_tools=['screenshot']), capabilities=[PlaywrightBrowser()])
        await agent.run('screenshot the page')
        # Only a 'popup' listener is registered, never a 'dialog' one: the capability
        # deliberately relies on Playwright auto-dismissing dialogs (alert/confirm/
        # beforeunload) when no handler is attached, rather than managing them.
        assert page.popup_events == ['popup']

    async def test_cancellation_mid_tool_call_tears_down_browser(self, monkeypatch: pytest.MonkeyPatch) -> None:
        page = _HangingScreenshotPage()
        cm = _install_fake_driver(monkeypatch, page)
        agent = Agent(TestModel(call_tools=['screenshot']), capabilities=[PlaywrightBrowser()])
        run = asyncio.create_task(agent.run('screenshot the page'))
        await page.screenshot_started.wait()
        run.cancel()
        with pytest.raises(asyncio.CancelledError):
            await run
        chromium = cm._driver.chromium
        assert chromium.browser is not None and chromium.browser.closed is True
        assert cm.exited is True

    async def test_allowlist_registers_route_guard_on_browser_context(self, monkeypatch: pytest.MonkeyPatch) -> None:
        page = _FakePage()
        _install_fake_driver(monkeypatch, page)
        agent = Agent(
            TestModel(call_tools=['screenshot']), capabilities=[PlaywrightBrowser(allowed_domains=['example.com'])]
        )
        await agent.run('screenshot the page')
        assert page.context.routes == ['**/*']

    async def test_context_route_guard_applies_to_each_pages_top_level_navigation(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        page = _FakePage()
        _install_fake_driver(monkeypatch, page)
        agent = Agent(
            TestModel(call_tools=['screenshot']), capabilities=[PlaywrightBrowser(allowed_domains=['example.com'])]
        )
        await agent.run('screenshot the page')

        popup_page = _FakeRequestPage()
        blocked = await page.context.dispatch(
            _FakeRequest('https://evil.com/popup', navigation=True, frame=popup_page.main_frame)
        )
        assert blocked.aborted is True
        assert blocked.continued is False

        allowed = await page.context.dispatch(
            _FakeRequest('https://example.com/popup', navigation=True, frame=popup_page.main_frame)
        )
        assert allowed.aborted is False
        assert allowed.continued is True

        non_navigation = await page.context.dispatch(
            _FakeRequest(
                'https://evil.com/script.js',
                navigation=False,
                frame_error=PlaywrightError('frame must not be accessed'),
            )
        )
        assert non_navigation.aborted is False
        assert non_navigation.continued is True

        popup_before_frame_creation = await page.context.dispatch(
            _FakeRequest(
                'https://evil.com/first-popup-request',
                navigation=True,
                frame_error=PlaywrightError('frame is not available yet'),
            )
        )
        assert popup_before_frame_creation.aborted is True
        assert popup_before_frame_creation.continued is False

    async def test_route_guard_blocks_private_addresses_in_subframes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # `snapshot()` reads the ARIA tree of cross-origin child frames (verified
        # against real Chromium), so a private-IP subframe would hand the model
        # the response body the block exists to withhold.
        page = _FakePage()
        _install_fake_driver(monkeypatch, page)
        agent = Agent(
            TestModel(call_tools=['screenshot']), capabilities=[PlaywrightBrowser(allowed_domains=['example.com'])]
        )
        await agent.run('screenshot the page')

        host_page = _FakeRequestPage()
        subframe = _FakeFrame(host_page)  # not `host_page.main_frame`

        metadata = await page.context.dispatch(
            _FakeRequest('http://169.254.169.254/latest/meta-data/', navigation=True, frame=subframe)
        )
        assert metadata.aborted is True
        assert metadata.continued is False

        # The allowlist stays top-level: a page's own third-party frames (identity
        # providers, payment steps) still load.
        third_party = await page.context.dispatch(
            _FakeRequest('https://evil.com/embed', navigation=True, frame=subframe)
        )
        assert third_party.aborted is False
        assert third_party.continued is True

    async def test_route_guard_subframe_block_honors_opt_out(self, monkeypatch: pytest.MonkeyPatch) -> None:
        page = _FakePage()
        _install_fake_driver(monkeypatch, page)
        agent = Agent(
            TestModel(call_tools=['screenshot']),
            capabilities=[PlaywrightBrowser(allowed_domains=['example.com'], block_private_addresses=False)],
        )
        await agent.run('screenshot the page')

        host_page = _FakeRequestPage()
        local = await page.context.dispatch(
            _FakeRequest('http://127.0.0.1:8080/panel', navigation=True, frame=_FakeFrame(host_page))
        )
        assert local.aborted is False
        assert local.continued is True

    async def test_popup_is_closed_without_navigating_main_page(self, monkeypatch: pytest.MonkeyPatch) -> None:
        popup = _FakePage(url='https://example.com/popup')
        page = _FakePage()
        page._popup_on_screenshot = popup
        _install_fake_driver(monkeypatch, page)
        agent = Agent(TestModel(call_tools=['screenshot']), capabilities=[PlaywrightBrowser()])
        await agent.run('screenshot the page')
        assert popup.closed is True
        assert page.goto_calls == []

    async def test_popup_close_error_is_observed_and_does_not_fail_run(self, monkeypatch: pytest.MonkeyPatch) -> None:
        popup = _FakePage(url='https://example.com/popup', close_error=RuntimeError('popup close failed'))
        page = _FakePage()
        page._popup_on_screenshot = popup
        _install_fake_driver(monkeypatch, page)
        browser = PlaywrightBrowser[object]()

        async def _same_instance(self: PlaywrightBrowser[object], ctx: RunContext[object]) -> PlaywrightBrowser[object]:
            return self

        monkeypatch.setattr(PlaywrightBrowser, 'for_run', _same_instance)
        agent = Agent(TestModel(call_tools=['screenshot']), capabilities=[browser])
        await agent.run('screenshot the page')
        assert popup.closed is True
        assert browser._popup_tasks == set()

    async def test_cancelled_popup_task_is_discarded(self) -> None:
        browser = PlaywrightBrowser[None]()
        task = asyncio.create_task(asyncio.sleep(1))
        browser._popup_tasks.add(task)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        browser._popup_done(task)
        assert browser._popup_tasks == set()

    async def test_run_without_browser_tool_skips_launch(self, monkeypatch: pytest.MonkeyPatch) -> None:
        page = _FakePage()
        cm = _install_fake_driver(monkeypatch, page)
        agent = Agent(TestModel(call_tools=[]), capabilities=[PlaywrightBrowser()])
        await agent.run('do nothing with the browser')
        assert cm.entered is False
        assert cm.exited is False

    async def test_missing_binary_raises_install_hint(self, monkeypatch: pytest.MonkeyPatch) -> None:
        page = _FakePage()
        cm = _install_fake_driver(monkeypatch, page, executable_missing=True)
        agent = Agent(TestModel(call_tools=['screenshot']), capabilities=[PlaywrightBrowser()])
        with pytest.raises(RuntimeError, match='playwright install chromium'):
            await agent.run('screenshot the page')
        assert cm._driver.chromium.launched == []  # never attempted launch on a missing binary
        assert cm.exited is True  # Playwright driver still cleaned up

    async def test_launch_failure_with_binary_present_surfaces_own_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        installs: list[bool] = []

        async def _spy_install() -> str | None:  # pragma: no cover -- asserted never called
            installs.append(True)
            return None

        monkeypatch.setattr(capability_module, '_auto_install_chromium', _spy_install)
        page = _FakePage()
        cm = _install_fake_driver(monkeypatch, page, launch_error=RuntimeError('sandbox denied'))
        # auto_install_chromium=True proves a real launch failure does not trigger a download.
        agent = Agent(
            TestModel(call_tools=['screenshot']), capabilities=[PlaywrightBrowser(auto_install_chromium=True)]
        )
        with pytest.raises(RuntimeError, match='sandbox denied'):
            await agent.run('screenshot the page')
        assert installs == []  # binary present -> no install attempt
        assert cm.exited is True  # driver still cleaned up

    async def test_auto_install_retry_when_install_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        page = _FakePage()
        _install_fake_driver(monkeypatch, page, executable_missing=True)

        async def _fake_install() -> str | None:
            return 'download error: HTTP 403 forbidden'

        monkeypatch.setattr(capability_module, '_auto_install_chromium', _fake_install)
        agent = Agent(
            TestModel(call_tools=['screenshot']), capabilities=[PlaywrightBrowser(auto_install_chromium=True)]
        )
        # A failed auto-install carries the installer output tail so it is diagnosable,
        # not just the generic missing-binary hint.
        with pytest.raises(RuntimeError, match='HTTP 403 forbidden') as exc_info:
            await agent.run('screenshot the page')
        assert 'Chromium is not installed' in str(exc_info.value)

    async def test_close_error_still_exits_driver(self, monkeypatch: pytest.MonkeyPatch) -> None:
        page = _FakePage()
        cm = _install_fake_driver(monkeypatch, page, close_error=RuntimeError('close failed'))
        agent = Agent(TestModel(call_tools=['screenshot']), capabilities=[PlaywrightBrowser()])
        with pytest.raises(RuntimeError, match='close failed'):
            await agent.run('screenshot the page')
        assert cm.exited is True  # driver exited despite the close error

    async def test_teardown_error_does_not_mask_run_exception(self, monkeypatch: pytest.MonkeyPatch) -> None:
        page = _FakePage()
        cm = _install_fake_driver(monkeypatch, page, close_error=RuntimeError('close failed'))
        browser = PlaywrightBrowser()

        class _RunFailure(Exception):
            pass

        async def _handler() -> AgentRunResult[None]:
            await browser._state.ensure_page()
            raise _RunFailure('run failed')

        # The run's own exception wins; the close error raised during teardown is dropped.
        with pytest.raises(_RunFailure, match='run failed'):
            await browser.wrap_run(_ctx(), handler=_handler)
        # cm.exited is set in the teardown finally after browser.close() raised, so the
        # driver still tore down and only the masking close error was swallowed.
        assert cm.exited is True

    async def test_pending_popup_tasks_cancelled_on_run_end(self, monkeypatch: pytest.MonkeyPatch) -> None:
        page = _FakePage()
        _install_fake_driver(monkeypatch, page)
        browser = PlaywrightBrowser()

        async def _same_instance(self: PlaywrightBrowser[None], ctx: RunContext[None]) -> PlaywrightBrowser[None]:
            return self

        monkeypatch.setattr(PlaywrightBrowser, 'for_run', _same_instance)
        pending = asyncio.ensure_future(asyncio.sleep(3600))
        browser._popup_tasks.add(pending)
        agent = Agent(TestModel(call_tools=['screenshot']), capabilities=[browser])
        await agent.run('screenshot the page')
        assert pending.cancelled()
        assert browser._popup_tasks == set()


# --- Package surface --------------------------------------------------------


def test_public_exports() -> None:
    assert DEFAULT_MAX_CONTENT_TOKENS == 4000
    assert DEFAULT_TIMEOUT_MS == 30_000
    assert issubclass(PlaywrightBrowser, object)
    assert PlaywrightBrowserToolset is not None
    assert PlaywrightBrowserState is not None
