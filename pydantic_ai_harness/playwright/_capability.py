"""Playwright capability -- a real, stateful Chromium browser for agents."""

from __future__ import annotations

import asyncio
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
    PlaywrightError,
    async_playwright,
    blocked_navigation_reason,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from playwright.async_api import Browser as PlaywrightBrowserHandle
    from playwright.async_api import Page as PlaywrightPage
    from playwright.async_api import Playwright as PlaywrightDriver
    from playwright.async_api import Request as PlaywrightRequest
    from playwright.async_api import Route as PlaywrightRoute

_CHROMIUM_MISSING_MESSAGE = (
    'Chromium is not installed. Run `playwright install chromium` (on a fresh Linux or CI image use '
    '`playwright install --with-deps chromium` to also install the required system libraries) and restart '
    'the agent to enable browser tools.'
)

_INSTRUCTIONS = """\
You have a real web browser powered by Playwright. Use it for pages the lighter tools cannot handle:
pages behind login or session cookies, JavaScript-heavy SPAs, interactive multi-step flows (clicking,
filling forms), and dynamically loaded content. For looking up information or reading a static, public
URL, prefer web search or web fetch.

Tools: `navigate(url)`, `snapshot()`, `click(selector)` (CSS selector, 'x,y' pixel coordinates, or an
`aria-ref=` handle from `snapshot`), `type_text(selector, text)`, `wait_for(selector?, text?)`,
`screenshot(full_page?)`, `get_text(selector?)`, `scroll(direction)`, `go_back()`, `go_forward()`,
`execute_js(script)`. Every page action takes an optional `timeout_ms` override.

Prefer `snapshot` to read the page structure and obtain `aria-ref` handles, then target elements by
`aria-ref=` for reliable clicks. Use `wait_for` for content that loads after an action, and `screenshot`
only for visual checks (charts, layout).

Textual tool results are truncated to roughly {max_content_tokens} tokens; use `get_text` with a CSS
selector to read a specific section of a large page. The browser is single-tab. Allowed domains: {allowed_domains}.
"""


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


@dataclass
class PlaywrightBrowser(AbstractCapability[AgentDepsT]):
    """A real, stateful Chromium browser for an agent, via async Playwright.

    Adds eleven tools -- navigate, snapshot, click, type_text, wait_for,
    screenshot, get_text, scroll, go_back, go_forward, execute_js -- backed by one
    Chromium page that persists across tool calls within a run. Reach for it when
    the lighter web tools fall short: pages behind login/session cookies,
    JavaScript-rendered SPAs, and interactive multi-step flows. For query-based
    research prefer
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

    Egress: `allowed_domains=None` (the default) places no domain restriction on
    the URLs the agent can reach; pass `allowed_domains=[...]` to restrict
    navigation to an allowlist. Independently, `block_private_addresses=True`
    (the default) refuses navigation to private, loopback, link-local, and other
    reserved IP literals (for example `169.254.169.254`, `127.0.0.1`, or
    `localhost`), even under open egress. Both policies govern page navigation
    (top-level document requests), not requests made by in-page JavaScript
    (`fetch`/XHR via `execute_js`), and the private-address block matches IP
    literals rather than resolving hostnames; those gaps are tracked in
    https://github.com/pydantic/pydantic-ai-harness/issues/415. Set
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

    block_private_addresses: bool = True
    """Refuse navigation to private, loopback, link-local, and other reserved IP literals.

    Covers the cloud metadata endpoint (`169.254.169.254`), loopback
    (`127.0.0.1`, `::1`, `localhost`), and the RFC 1918 ranges, independent of
    `allowed_domains` -- open egress still cannot reach them. Only IP literals
    and `localhost` names are matched; hostnames are not resolved (see the
    egress note above). Set `False` when the agent should reach a local app or
    an internal dashboard.
    """

    screenshot_on_navigate: bool = False
    """Attach a screenshot (as image content) to every `navigate` result."""

    max_content_tokens: int = DEFAULT_MAX_CONTENT_TOKENS
    """Approximate token budget for textual tool results."""

    timeout_ms: int = DEFAULT_TIMEOUT_MS
    """Default Playwright navigation/action timeout in milliseconds."""

    auto_install_chromium: bool = False
    """Fetch the Chromium binary via `playwright install chromium` on the first miss.

    Off by default: a library should not download a browser as a side effect. When
    the binary is missing the browser tools are hidden and calling one raises a
    clear install hint. Set `True` to opt into the automatic download instead.
    """

    storage_state: str | None = None
    """Path to a Playwright storage state JSON file (cookies, localStorage).

    Produced by `context.storage_state(path=...)` or
    `playwright codegen --save-storage`, and loaded into the browser context at
    launch so the agent starts with that session. The file is credential
    material -- treat it as a secret: never commit it, and delete it when the
    session expires.
    """

    _state: PlaywrightBrowserState = field(default_factory=PlaywrightBrowserState, init=False, repr=False)
    _toolset: PlaywrightBrowserToolset[AgentDepsT] = field(init=False, repr=False)
    _browser: PlaywrightBrowserHandle | None = field(default=None, init=False, repr=False)
    _popup_tasks: set[asyncio.Task[None]] = field(default_factory=set[asyncio.Task[None]], init=False, repr=False)

    def __post_init__(self) -> None:
        self._toolset = PlaywrightBrowserToolset[AgentDepsT](
            state=self._state,
            allowed_domains=self.allowed_domains,
            block_private_addresses=self.block_private_addresses,
            screenshot_on_navigate=self.screenshot_on_navigate,
            max_content_tokens=self.max_content_tokens,
            timeout_ms=self.timeout_ms,
        )

    async def for_run(self, ctx: RunContext[AgentDepsT]) -> PlaywrightBrowser[AgentDepsT]:
        """Return a fresh instance per run so concurrent runs never share a page or browser."""
        return replace(self)

    def get_toolset(self) -> PlaywrightBrowserToolset[AgentDepsT]:
        """Provide the eleven browser tools."""
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
            if self.block_private_addresses:
                domains += ' (private/internal addresses blocked)'
            return _INSTRUCTIONS.format(max_content_tokens=self.max_content_tokens, allowed_domains=domains)

        return _instructions

    async def prepare_tools(self, ctx: RunContext[AgentDepsT], tool_defs: list[ToolDefinition]) -> list[ToolDefinition]:
        """Hide the browser tools when Chromium is unavailable.

        Tools are matched by `toolset_id` so a same-named tool from another toolset
        (e.g. a different `navigate`) is left untouched.
        """
        toolset_id = self._toolset.id
        if self._state.launch_error is not None:
            return [td for td in tool_defs if td.toolset_id != toolset_id]
        return tool_defs

    async def _install_and_retry(self, pw: PlaywrightDriver) -> PlaywrightBrowserHandle | None:
        """Fetch Chromium and relaunch once.

        Returns the browser on success. On install failure it records a launch
        error carrying a bounded tail of the installer output (so the failure is
        diagnosable) and returns `None`.
        """
        install_output = await _auto_install_chromium()
        if install_output is None:
            return await pw.chromium.launch(headless=self.headless)  # pragma: no cover
        self._state.launch_error = f'{_CHROMIUM_MISSING_MESSAGE}\nAuto-install failed:\n{install_output[-300:]}'
        return None

    async def _route_guard(self, route: PlaywrightRoute, request: PlaywrightRequest) -> None:  # pragma: no cover
        """Network-layer navigation policy: abort disallowed top-level navigations, pass the rest.

        Applies both the allowlist and the private-address block. Runs only as a
        real Playwright route callback, so it is outside the mocked test surface.
        """
        permitted = blocked_navigation_reason(request.url, self.allowed_domains, self.block_private_addresses) is None
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
                    if self._state.launch_error is None:
                        self._state.launch_error = _CHROMIUM_MISSING_MESSAGE
                    return
            else:
                browser = await pw.chromium.launch(headless=self.headless)
            self._browser = browser
            # Service workers can issue requests that context routes never see, so
            # they are blocked to keep all traffic on the routable path.
            context = await browser.new_context(storage_state=self.storage_state, service_workers='block')
            page = await context.new_page()
            if self.allowed_domains is not None or self.block_private_addresses:
                await context.route('**/*', self._route_guard)
            page.on('popup', self._on_popup)
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
            run_failed = sys.exc_info()[0] is not None
            try:
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
            except Exception:
                # A teardown error must not mask the run's real exception. When the
                # run already failed, drop the close/exit error; otherwise surface it.
                if not run_failed:
                    raise

    @classmethod
    def from_spec(
        cls,
        *,
        headless: bool = True,
        allowed_domains: list[str] | None = None,
        block_private_addresses: bool = True,
        screenshot_on_navigate: bool = False,
        max_content_tokens: int = DEFAULT_MAX_CONTENT_TOKENS,
        timeout_ms: int = DEFAULT_TIMEOUT_MS,
        auto_install_chromium: bool = False,
        storage_state: str | None = None,
    ) -> PlaywrightBrowser[AgentDepsT]:
        """Construct the capability from serializable spec options (all fields are plain scalars/lists).

        `storage_state` is a path string, which is spec-safe: the secret lives in
        the referenced file, not in the spec.
        """
        return cls(
            headless=headless,
            allowed_domains=list(allowed_domains) if allowed_domains is not None else None,
            block_private_addresses=block_private_addresses,
            screenshot_on_navigate=screenshot_on_navigate,
            max_content_tokens=max_content_tokens,
            timeout_ms=timeout_ms,
            auto_install_chromium=auto_install_chromium,
            storage_state=storage_state,
        )
