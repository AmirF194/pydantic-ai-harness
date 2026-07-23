"""Playwright capability -- a real, stateful Chromium browser for agents."""

from __future__ import annotations

import asyncio
import functools
import os
import sys
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

from pydantic_ai import AgentRunResult, RunContext
from pydantic_ai.capabilities import AbstractCapability, WrapRunHandler
from pydantic_ai.tools import AgentDepsT, ToolDefinition

from pydantic_ai_harness.playwright._toolset import (
    DEFAULT_MAX_CONTENT_TOKENS,
    DEFAULT_TIMEOUT_MS,
    PlaywrightBrowserState,
    PlaywrightBrowserToolset,
    async_playwright,
    check_allowed_domain,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from playwright.async_api import Browser as PlaywrightBrowserHandle
    from playwright.async_api import Page as PlaywrightPage
    from playwright.async_api import Playwright as PlaywrightDriver
    from playwright.async_api import Request as PlaywrightRequest
    from playwright.async_api import Route as PlaywrightRoute

_CHROMIUM_MISSING_MESSAGE = (
    'Chromium is not installed. Run `playwright install chromium` and restart the agent to enable browser tools.'
)

_INSTRUCTIONS = """\
You have a real web browser powered by Playwright. Use it for pages the lighter tools cannot handle:
pages behind login or session cookies, JavaScript-heavy SPAs, interactive multi-step flows (clicking,
filling forms), and dynamically loaded content. For looking up information or reading a static, public
URL, prefer web search or web fetch.

Tools: `navigate(url)`, `click(selector)` (CSS selector or 'x,y' pixel coordinates),
`type_text(selector, text)`, `screenshot(full_page?)`, `get_text(selector?)`,
`scroll(direction)`, `go_back()`, `go_forward()`, `execute_js(script)`.

Page text is returned truncated to roughly {max_content_tokens} tokens; use `get_text` with a CSS
selector to read a specific section of a large page. The browser is single-tab. Allowed domains: {allowed_domains}.
"""


async def _auto_install_chromium() -> bool:  # pragma: no cover
    """Run `playwright install chromium` in this interpreter; return whether it succeeded.

    Only invoked when `auto_install_chromium=True` and the binary is missing. It
    shells out to a subprocess and downloads a browser, so it runs outside the
    mocked test surface.
    """
    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        '-m',
        'playwright',
        'install',
        'chromium',
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    await proc.communicate()
    return proc.returncode == 0


@dataclass
class PlaywrightBrowser(AbstractCapability[AgentDepsT]):
    """A real, stateful Chromium browser for an agent, via async Playwright.

    Adds nine tools -- navigate, click, type_text, screenshot, get_text, scroll,
    go_back, go_forward, execute_js -- backed by one Chromium page that persists
    across tool calls within a run. Reach for it when the lighter web tools fall
    short: pages behind login/session cookies, JavaScript-rendered SPAs, and
    interactive multi-step flows. For query-based research prefer
    [`ExaSearch`][pydantic_ai_harness.exa.ExaSearch]; for a static URL prefer a
    web-fetch tool.

    ```python
    from pydantic_ai import Agent
    from pydantic_ai_harness.playwright import PlaywrightBrowser

    agent = Agent('anthropic:claude-sonnet-4-6', capabilities=[PlaywrightBrowser()])
    ```

    Requires the `playwright` optional extra and the Chromium binary:

    ```bash
    pip install 'pydantic-ai-harness[playwright]'
    playwright install chromium
    ```

    Egress: `allowed_domains=None` (the default) places no restriction on the
    URLs the agent can reach, including internal or link-local addresses (for
    example `169.254.169.254` or `localhost`). Pass `allowed_domains=[...]` to
    restrict navigation to an allowlist. The allowlist governs page navigation
    (top-level document requests), not requests made by in-page JavaScript
    (`fetch`/XHR via `execute_js`). Blocking private/link-local ranges by default
    and constraining in-page requests are tracked in
    https://github.com/pydantic/pydantic-ai-harness/issues/415; until then, set
    `allowed_domains` when the agent may act on untrusted input.

    Chromium starts lazily on the first browser-tool call and is closed when the
    run ends (on success, error, or cancellation); runs that never call a browser
    tool pay no Playwright cost. When the Chromium binary is missing the tools are
    hidden from the model and calling one raises with a `playwright install
    chromium` hint; set `auto_install_chromium=True` to fetch it automatically
    instead.
    """

    headless: bool = True
    """Run Chromium without a visible window. `True` suits servers and CI."""

    allowed_domains: list[str] | None = None
    """Navigation allowlist. `None` (default) allows all domains -- see the egress note above.

    Each entry matches its exact host and any subdomain. Enforced at two layers:
    a network route guard aborts disallowed top-level navigations (covering
    clicks, `execute_js`, and history moves, not just `navigate`), and each
    tool re-checks the resulting URL and bounces to `about:blank` so disallowed
    content never reaches the model.
    """

    screenshot_on_navigate: bool = False
    """Attach a screenshot (as image content) to every `navigate` result."""

    max_content_tokens: int = DEFAULT_MAX_CONTENT_TOKENS
    """Approximate token budget for page text returned by the tools."""

    timeout_ms: int = DEFAULT_TIMEOUT_MS
    """Default Playwright navigation/action timeout in milliseconds."""

    auto_install_chromium: bool = False
    """Fetch the Chromium binary via `playwright install chromium` on the first miss.

    Off by default: a library should not download a browser as a side effect. When
    the binary is missing the browser tools are hidden and calling one raises a
    clear install hint. Set `True` to opt into the automatic download instead.
    """

    _state: PlaywrightBrowserState = field(default_factory=PlaywrightBrowserState, init=False, repr=False)
    _toolset: PlaywrightBrowserToolset[AgentDepsT] = field(init=False, repr=False)
    _browser: PlaywrightBrowserHandle | None = field(default=None, init=False, repr=False)
    _popup_tasks: set[asyncio.Task[None]] = field(default_factory=set[asyncio.Task[None]], init=False, repr=False)

    def __post_init__(self) -> None:
        self._toolset = PlaywrightBrowserToolset[AgentDepsT](
            state=self._state,
            allowed_domains=self.allowed_domains,
            screenshot_on_navigate=self.screenshot_on_navigate,
            max_content_tokens=self.max_content_tokens,
            timeout_ms=self.timeout_ms,
        )

    async def for_run(self, ctx: RunContext[AgentDepsT]) -> PlaywrightBrowser[AgentDepsT]:
        """Return a fresh instance per run so concurrent runs never share a page or browser."""
        return replace(self)

    def get_toolset(self) -> PlaywrightBrowserToolset[AgentDepsT]:
        """Provide the nine browser tools."""
        return self._toolset

    def get_instructions(self) -> Callable[[RunContext[AgentDepsT]], str | None]:
        """When-to-use guidance for the browser, suppressed while a launch error is set."""

        def _instructions(ctx: RunContext[AgentDepsT]) -> str | None:
            if self._state.launch_error is not None:
                return None
            if self.allowed_domains is None:
                domains = 'all'
            elif self.allowed_domains:
                domains = ', '.join(self.allowed_domains)
            else:
                domains = 'none'
            return _INSTRUCTIONS.format(max_content_tokens=self.max_content_tokens, allowed_domains=domains)

        return _instructions

    async def prepare_tools(self, ctx: RunContext[AgentDepsT], tool_defs: list[ToolDefinition]) -> list[ToolDefinition]:
        """Hide the browser tools when Chromium is unavailable; keep them approval-free otherwise.

        Tools are matched by `toolset_id` so a same-named tool from another toolset
        (e.g. a different `navigate`) is left untouched.
        """
        toolset_id = self._toolset.id
        if self._state.launch_error is not None:
            return [td for td in tool_defs if td.toolset_id != toolset_id]
        return [
            replace(td, kind='function') if td.toolset_id == toolset_id and td.kind == 'unapproved' else td
            for td in tool_defs
        ]

    async def _install_and_retry(self, pw: PlaywrightDriver) -> PlaywrightBrowserHandle | None:
        """Fetch Chromium and relaunch once; `None` when the install itself failed."""
        if not await _auto_install_chromium():
            return None
        return await pw.chromium.launch(headless=self.headless)  # pragma: no cover

    async def _route_guard(  # pragma: no cover
        self, page: PlaywrightPage, route: PlaywrightRoute, request: PlaywrightRequest
    ) -> None:
        """Network-layer allowlist: abort disallowed top-level navigations, pass the rest.

        Runs only as a real Playwright route callback, so it is outside the mocked
        test surface.
        """
        if (
            request.is_navigation_request()
            and request.frame == page.main_frame
            and not check_allowed_domain(request.url, self.allowed_domains)
        ):
            await route.abort()
            return
        await route.continue_()

    async def _handle_popup(self, page: PlaywrightPage, popup: PlaywrightPage) -> None:  # pragma: no cover
        """Single-tab: close a popup and redirect the main tab to it if the allowlist permits.

        Runs only as a real Playwright popup callback, so it is outside the mocked
        test surface.
        """
        new_url = popup.url
        await popup.close()
        if check_allowed_domain(new_url, self.allowed_domains):
            await page.goto(new_url)

    def _on_popup(self, page: PlaywrightPage, popup: PlaywrightPage) -> None:  # pragma: no cover
        """Schedule popup handling, keeping a strong task reference until it finishes."""
        task = asyncio.ensure_future(self._handle_popup(page, popup))
        self._popup_tasks.add(task)
        task.add_done_callback(self._popup_tasks.discard)

    async def wrap_run(self, ctx: RunContext[AgentDepsT], *, handler: WrapRunHandler) -> AgentRunResult[AgentDepsT]:
        """Install a lazy Chromium launcher and guarantee cleanup when the run ends.

        Playwright and Chromium start only on the first browser-tool call. A
        `finally` block closes the browser and exits the Playwright driver whether
        the run returns, raises, or is cancelled.
        """
        pw_cm = async_playwright()
        entered = False

        async def _launch() -> None:
            nonlocal entered
            pw = await pw_cm.__aenter__()
            entered = True
            if not os.path.exists(pw.chromium.executable_path):
                # Binary genuinely absent: raise a clear install hint, or fetch it
                # when opted in. A launch failure with the binary present (sandbox,
                # missing system libs, no display) is left to surface as its own
                # error rather than being masked as "Chromium is not installed".
                browser = await self._install_and_retry(pw) if self.auto_install_chromium else None
                if browser is None:  # pragma: no branch
                    self._state.launch_error = _CHROMIUM_MISSING_MESSAGE
                    return
            else:
                browser = await pw.chromium.launch(headless=self.headless)
            self._browser = browser
            page = await browser.new_page()
            if self.allowed_domains is not None:
                await page.route('**/*', functools.partial(self._route_guard, page))
            page.on('popup', functools.partial(self._on_popup, page))
            self._state.page = page

        self._state.lazy_launcher = _launch
        try:
            return await handler()
        finally:
            self._state.lazy_launcher = None
            if self._popup_tasks:
                for task in self._popup_tasks:
                    task.cancel()
                await asyncio.gather(*self._popup_tasks, return_exceptions=True)
                self._popup_tasks.clear()
            browser = self._browser
            if browser is not None:
                self._state.page = None
                self._browser = None
                try:
                    await browser.close()
                finally:
                    await pw_cm.__aexit__(None, None, None)
            elif entered:
                await pw_cm.__aexit__(None, None, None)

    @classmethod
    def from_spec(
        cls,
        *,
        headless: bool = True,
        allowed_domains: list[str] | None = None,
        screenshot_on_navigate: bool = False,
        max_content_tokens: int = DEFAULT_MAX_CONTENT_TOKENS,
        timeout_ms: int = DEFAULT_TIMEOUT_MS,
        auto_install_chromium: bool = False,
    ) -> PlaywrightBrowser[AgentDepsT]:
        """Construct the capability from serializable spec options (all fields are plain scalars/lists)."""
        return cls(
            headless=headless,
            allowed_domains=list(allowed_domains) if allowed_domains is not None else None,
            screenshot_on_navigate=screenshot_on_navigate,
            max_content_tokens=max_content_tokens,
            timeout_ms=timeout_ms,
            auto_install_chromium=auto_install_chromium,
        )
