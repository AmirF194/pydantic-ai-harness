"""Tests for the Browser capability.

The Playwright API surface is mocked throughout: no real Chromium is launched.
An in-memory page double (`_FakePage`) backs the toolset, and a fake Playwright
driver chain backs the `wrap_run` lifecycle, so the suite runs in CI with only
the `playwright` Python package installed (no browser binary).
"""

from __future__ import annotations

import pytest
from pydantic_ai import Agent
from pydantic_ai.messages import BinaryContent, ToolReturn
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import RunContext, ToolDefinition
from pydantic_ai.usage import RunUsage

import pydantic_ai_harness.browser._capability as capability_module
from pydantic_ai_harness.browser import (
    DEFAULT_MAX_CONTENT_TOKENS,
    DEFAULT_TIMEOUT_MS,
    Browser,
    BrowserState,
    BrowserToolset,
)

pytestmark = pytest.mark.anyio


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
        self.mouse = _FakeMouse()
        self.goto_calls: list[str] = []
        self.filled: list[tuple[str, str]] = []
        self.clicked: list[str] = []
        self.routes: list[str] = []
        self.popup_events: list[str] = []

    @property
    def url(self) -> str:
        return self._url

    async def goto(self, url: str, *, timeout: float | None = None) -> None:
        self.goto_calls.append(url)
        # A configured redirect lands the page on a different host than requested,
        # modelling a 3xx to a disallowed domain (but never for the bounce itself).
        self._url = self._redirect_to if self._redirect_to is not None and url != 'about:blank' else url

    async def wait_for_load_state(self, state: str) -> None:
        return None

    async def title(self) -> str:
        return self._title

    async def inner_text(self, selector: str, *, timeout: float | None = None) -> str:
        if selector == 'body':
            return self._body
        if self._selector_raises:
            raise RuntimeError('element not found')
        return self._element_text if self._element_text is not None else f'text:{selector}'

    async def click(self, selector: str, *, timeout: float | None = None) -> None:
        self.clicked.append(selector)

    async def fill(self, selector: str, value: str, *, timeout: float | None = None) -> None:
        self.filled.append((selector, value))

    async def screenshot(self, *, full_page: bool = False) -> bytes:
        return self._screenshot_bytes

    async def evaluate(self, expression: str) -> object:
        if self._evaluate_raises is not None:
            raise self._evaluate_raises
        return self._evaluate_result

    async def go_back(self, *, timeout: float | None = None) -> None:
        return None

    async def go_forward(self, *, timeout: float | None = None) -> None:
        return None

    async def route(self, url: str, handler: object) -> None:
        self.routes.append(url)

    def on(self, event: str, handler: object) -> None:
        self.popup_events.append(event)


class _FakeBrowser:
    def __init__(self, page: _FakePage) -> None:
        self._page = page
        self.closed = False

    async def new_page(self) -> _FakePage:
        return self._page

    async def close(self) -> None:
        self.closed = True


class _FakeChromium:
    def __init__(self, page: _FakePage, *, launch_error: bool = False) -> None:
        self._page = page
        self._launch_error = launch_error
        self.launched: list[bool] = []
        self.browser: _FakeBrowser | None = None

    async def launch(self, *, headless: bool) -> _FakeBrowser:
        self.launched.append(headless)
        if self._launch_error:
            raise RuntimeError('chromium binary missing')
        self.browser = _FakeBrowser(self._page)
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
    screenshot_on_navigate: bool = False,
    max_content_tokens: int = DEFAULT_MAX_CONTENT_TOKENS,
) -> BrowserToolset[None]:
    """Build a toolset whose active page is the given double."""
    state = BrowserState()
    state.page = page
    return BrowserToolset[None](
        state=state,
        allowed_domains=allowed_domains,
        screenshot_on_navigate=screenshot_on_navigate,
        max_content_tokens=max_content_tokens,
    )


def _ctx() -> RunContext[None]:
    return RunContext[None](deps=None, model=TestModel(), usage=RunUsage())


def _install_fake_driver(
    monkeypatch: pytest.MonkeyPatch, page: _FakePage, *, launch_error: bool = False
) -> _FakeDriverCM:
    """Point the capability's `async_playwright` at a fake driver chain."""
    chromium = _FakeChromium(page, launch_error=launch_error)
    cm = _FakeDriverCM(_FakeDriver(chromium))
    monkeypatch.setattr(capability_module, 'async_playwright', lambda: cm)
    return cm


# --- Tool behavior ----------------------------------------------------------


class TestBrowserTools:
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
        assert isinstance(result, str)
        assert 'page text truncated at 4 characters' in result

    async def test_navigate_bounces_on_redirect_to_disallowed_host(self) -> None:
        page = _FakePage(redirect_to='https://evil.com/landing')
        toolset = _toolset(page, allowed_domains=['example.com'])
        result = await toolset.navigate('https://example.com/start')
        assert result == 'Error: navigate reached a domain not in allowed_domains: https://evil.com/landing'
        assert page.goto_calls == ['https://example.com/start', 'about:blank']

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
        assert result.return_value == 'Screenshot captured. URL: https://example.com/p'
        assert result.content is not None
        image = result.content[0]
        assert isinstance(image, BinaryContent)
        assert image.data == b'PNG-BYTES'

    async def test_screenshot_full_page(self) -> None:
        toolset = _toolset(_FakePage())
        result = await toolset.screenshot(full_page=True)
        assert isinstance(result, ToolReturn)

    async def test_get_text_with_selector(self) -> None:
        toolset = _toolset(_FakePage())
        assert await toolset.get_text('h1') == 'text:h1'

    async def test_get_text_selector_error(self) -> None:
        toolset = _toolset(_FakePage(selector_raises=True))
        result = await toolset.get_text('#missing')
        assert result.startswith("Error getting text from '#missing':")

    async def test_get_text_full_page(self) -> None:
        toolset = _toolset(_FakePage(body='full page text'))
        assert await toolset.get_text() == 'full page text'

    async def test_get_text_selector_truncated(self) -> None:
        toolset = _toolset(_FakePage(element_text='Y' * 40), max_content_tokens=1)
        result = await toolset.get_text('article')
        assert result.startswith('Y' * 4)
        assert 'page text truncated at 4 characters' in result

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

    async def test_go_forward(self) -> None:
        toolset = _toolset(_FakePage(url='https://example.com/next', body='next'))
        result = await toolset.go_forward()
        assert result == 'Went forward. URL: https://example.com/next\n\nnext'

    async def test_go_forward_bounces_off_disallowed_domain(self) -> None:
        page = _FakePage(url='https://evil.com/')
        toolset = _toolset(page, allowed_domains=['example.com'])
        result = await toolset.go_forward()
        assert result == 'Error: go_forward reached a domain not in allowed_domains: https://evil.com/'

    async def test_execute_js_string_result(self) -> None:
        toolset = _toolset(_FakePage(evaluate_result='the title'))
        assert await toolset.execute_js('document.title') == 'the title'

    async def test_execute_js_json_result(self) -> None:
        toolset = _toolset(_FakePage(evaluate_result={'a': 1}))
        assert await toolset.execute_js('({a:1})') == '{"a": 1}'

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


# --- State / ensure_page ----------------------------------------------------


class TestBrowserState:
    async def test_tool_raises_when_wrap_run_not_active(self) -> None:
        toolset = BrowserToolset[None](state=BrowserState())
        with pytest.raises(RuntimeError, match='Browser is not running'):
            await toolset.screenshot()

    async def test_tool_raises_on_launch_error(self) -> None:
        state = BrowserState()
        state.launch_error = 'Chromium is not installed.'
        toolset = BrowserToolset[None](state=state)
        with pytest.raises(RuntimeError, match='Chromium is not installed'):
            await toolset.screenshot()


# --- Capability hooks -------------------------------------------------------


class TestBrowserHooks:
    def test_get_instructions_reports_allowlist(self) -> None:
        instructions = Browser[None](allowed_domains=['a.com', 'b.com']).get_instructions()
        text = instructions(_ctx())
        assert text is not None and 'Allowed domains: a.com, b.com' in text

    def test_get_instructions_reports_all_domains(self) -> None:
        instructions = Browser[None]().get_instructions()
        text = instructions(_ctx())
        assert text is not None and 'Allowed domains: all' in text

    def test_get_instructions_suppressed_on_launch_error(self) -> None:
        browser = Browser[None]()
        browser._state.launch_error = 'boom'
        assert browser.get_instructions()(_ctx()) is None

    async def test_prepare_tools_reset_unapproved(self) -> None:
        browser = Browser[None]()
        defs = [
            ToolDefinition(name='navigate', parameters_json_schema={'type': 'object'}, kind='unapproved'),
            ToolDefinition(name='other', parameters_json_schema={'type': 'object'}, kind='function'),
        ]
        result = await browser.prepare_tools(_ctx(), defs)
        by_name = {td.name: td for td in result}
        assert by_name['navigate'].kind == 'function'
        assert by_name['other'].kind == 'function'

    async def test_prepare_tools_hides_tools_on_launch_error(self) -> None:
        browser = Browser[None]()
        browser._state.launch_error = 'boom'
        defs = [
            ToolDefinition(name='navigate', parameters_json_schema={'type': 'object'}, kind='function'),
            ToolDefinition(name='other', parameters_json_schema={'type': 'object'}, kind='function'),
        ]
        result = await browser.prepare_tools(_ctx(), defs)
        assert [td.name for td in result] == ['other']

    async def test_for_run_isolates_state(self) -> None:
        browser = Browser[None]()
        first = await browser.for_run(_ctx())
        second = await browser.for_run(_ctx())
        assert first._state is not second._state
        assert first._state is not browser._state

    def test_from_spec_round_trips_fields(self) -> None:
        browser = Browser[None].from_spec(
            headless=False,
            allowed_domains=['x.com'],
            screenshot_on_navigate=True,
            max_content_tokens=100,
            timeout_ms=5000,
            auto_install=True,
        )
        assert browser.headless is False
        assert browser.allowed_domains == ['x.com']
        assert browser.screenshot_on_navigate is True
        assert browser.max_content_tokens == 100
        assert browser.timeout_ms == 5000
        assert browser.auto_install is True

    def test_from_spec_defaults_to_open_egress(self) -> None:
        browser = Browser[None].from_spec()
        assert browser.allowed_domains is None
        assert browser.auto_install is False


# --- Lifecycle through Agent + wrap_run -------------------------------------


class TestBrowserLifecycle:
    async def test_lazy_launch_and_cleanup(self, monkeypatch: pytest.MonkeyPatch) -> None:
        page = _FakePage()
        cm = _install_fake_driver(monkeypatch, page)
        agent = Agent(TestModel(call_tools=['screenshot']), capabilities=[Browser()])
        await agent.run('use the browser')
        chromium = cm._driver.chromium
        assert chromium.launched == [True]
        assert page.popup_events == ['popup']
        assert page.routes == []  # no allowlist -> no route guard registered
        assert chromium.browser is not None and chromium.browser.closed is True
        assert cm.exited is True

    async def test_allowlist_registers_route_guard(self, monkeypatch: pytest.MonkeyPatch) -> None:
        page = _FakePage()
        _install_fake_driver(monkeypatch, page)
        agent = Agent(TestModel(call_tools=['screenshot']), capabilities=[Browser(allowed_domains=['example.com'])])
        await agent.run('screenshot the page')
        assert page.routes == ['**/*']

    async def test_run_without_browser_tool_skips_launch(self, monkeypatch: pytest.MonkeyPatch) -> None:
        page = _FakePage()
        cm = _install_fake_driver(monkeypatch, page)
        agent = Agent(TestModel(call_tools=[]), capabilities=[Browser()])
        await agent.run('do nothing with the browser')
        assert cm.entered is False
        assert cm.exited is False

    async def test_missing_binary_raises_install_hint(self, monkeypatch: pytest.MonkeyPatch) -> None:
        page = _FakePage()
        cm = _install_fake_driver(monkeypatch, page, launch_error=True)
        agent = Agent(TestModel(call_tools=['screenshot']), capabilities=[Browser()])
        with pytest.raises(RuntimeError, match='playwright install chromium'):
            await agent.run('screenshot the page')
        assert cm.exited is True  # Playwright driver still cleaned up

    async def test_auto_install_retry_when_install_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        page = _FakePage()
        _install_fake_driver(monkeypatch, page, launch_error=True)

        async def _fake_install() -> bool:
            return False

        monkeypatch.setattr(capability_module, '_auto_install_chromium', _fake_install)
        agent = Agent(TestModel(call_tools=['screenshot']), capabilities=[Browser(auto_install=True)])
        with pytest.raises(RuntimeError, match='Chromium is not installed'):
            await agent.run('screenshot the page')


# --- Package surface --------------------------------------------------------


def test_public_exports() -> None:
    assert DEFAULT_MAX_CONTENT_TOKENS == 4000
    assert DEFAULT_TIMEOUT_MS == 30_000
    assert issubclass(Browser, object)
    assert BrowserToolset is not None
    assert BrowserState is not None
