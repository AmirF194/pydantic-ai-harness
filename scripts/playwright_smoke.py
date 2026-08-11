"""Hand-run smoke test for the Playwright browser capability.

Not collected by pytest and not wired into CI: it launches a real Chromium and
reaches the public internet, which the mocked unit suite deliberately avoids. Run
it after installing the browser binary to verify the real integration end to end:

    playwright install chromium
    uv run python scripts/playwright_smoke.py

It exercises the public `PlaywrightBrowser` surface and checks five things:

- lazy launch plus a real navigation to https://example.com (prints the title),
- the allowlist bounce (a disallowed host returns an error result, not content),
- a `storage_state` round-trip: a cookie captured from a real context is visible
  to the agent after relaunching the capability with that state,
- attaching over `cdp_url` to a browser that already holds a session: the run gets
  its own context (the existing cookie is not visible), the allowlist still bounces,
  and the host browser's own page survives teardown,
- clean teardown: each scenario runs its own capability, whose `wrap_run` closes
  the browser when the run ends. After it prints `all checks passed`, confirm no
  Chromium lingered, e.g. `pgrep -fl chromium` shows nothing this script started.
"""

from __future__ import annotations

import asyncio
import socket
from tempfile import TemporaryDirectory

from playwright.async_api import StorageState, async_playwright
from pydantic_ai import Agent
from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel

from pydantic_ai_harness.playwright import PlaywrightBrowser

_COOKIE = 'smoke_session=abc123'


async def _run_tools(browser: PlaywrightBrowser[object], calls: list[tuple[str, dict[str, object]]]) -> list[str]:
    """Drive a fixed sequence of tool calls through one agent run, in order.

    Emitting one scripted `ToolCallPart` per model turn keeps every call in the
    same run, so the launched browser (and its `storage_state` session) persists
    across the sequence. Returns each tool's stringified result.
    """
    results: list[str] = []
    index = 0

    async def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal index
        for part in messages[-1].parts:
            if isinstance(part, ToolReturnPart):
                results.append(str(part.content))
        if index < len(calls):
            name, args = calls[index]
            index += 1
            return ModelResponse(parts=[ToolCallPart(tool_name=name, args=args)])
        return ModelResponse(parts=[TextPart('done')])

    agent = Agent(FunctionModel(model), capabilities=[browser])
    await agent.run('smoke')
    return results


async def _check_navigate() -> None:
    """Lazy launch, then navigate to a real page and report its title."""
    (result,) = await _run_tools(PlaywrightBrowser[object](), [('navigate', {'url': 'https://example.com'})])
    assert 'Title:' in result, result
    title_line = next(line for line in result.splitlines() if line.startswith('Title:'))
    print(f'navigate ok -- {title_line}')


async def _check_allowlist_bounce() -> None:
    """A navigation to a host outside the allowlist returns an error result."""
    browser = PlaywrightBrowser[object](allowed_domains=['example.com'])
    (result,) = await _run_tools(browser, [('navigate', {'url': 'https://www.iana.org/'})])
    assert 'not in allowed_domains' in result, result
    print('allowlist bounce ok')


async def _capture_storage_state() -> StorageState:
    """Log a cookie into a real context and hand back its storage state."""
    async with async_playwright() as pw:
        chromium = await pw.chromium.launch(headless=True)
        try:
            context = await chromium.new_context()
            page = await context.new_page()
            await page.goto('https://example.com')
            await page.evaluate(f"document.cookie = '{_COOKIE}; path=/'")
            return await context.storage_state()
        finally:
            await chromium.close()


async def _check_storage_state_round_trip() -> None:
    """A cookie captured from a real context is visible to the agent after relaunch."""
    browser = PlaywrightBrowser[object](storage_state=await _capture_storage_state())
    _, cookies = await _run_tools(
        browser,
        [('navigate', {'url': 'https://example.com'}), ('execute_js', {'script': 'document.cookie'})],
    )
    assert _COOKIE in cookies, cookies
    print('storage_state round-trip ok')


async def _check_cdp_attach() -> None:
    """Attaching over CDP gets a fresh context, not the sessions already open there.

    The mocked suite cannot answer this: it turns on whether a real Chrome honours
    `Target.createBrowserContext` for a browser Playwright did not launch. A cookie
    is logged into the attached browser's own default context, then the capability
    attaches and reads `document.cookie` on the same origin. The allowlist is
    checked in the same run, since the route guard is installed on the context the
    capability creates rather than on the browser it connected to.
    """
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', 0))
        port = probe.getsockname()[1]
    with TemporaryDirectory() as user_data_dir:
        async with async_playwright() as pw:
            default_context = await pw.chromium.launch_persistent_context(
                user_data_dir, headless=True, args=[f'--remote-debugging-port={port}']
            )
            try:
                page = await default_context.new_page()
                await page.goto('https://example.com')
                await page.evaluate(f"document.cookie = '{_COOKIE}; path=/'")
                assert _COOKIE in await page.evaluate('document.cookie')

                browser = PlaywrightBrowser[object](cdp_url=f'http://127.0.0.1:{port}', allowed_domains=['example.com'])
                _, cookies, bounced = await _run_tools(
                    browser,
                    [
                        ('navigate', {'url': 'https://example.com'}),
                        ('execute_js', {'script': 'document.cookie'}),
                        ('navigate', {'url': 'https://www.iana.org/'}),
                    ],
                )
                assert _COOKIE not in cookies, f'attached run inherited the default context session: {cookies}'
                assert 'not in allowed_domains' in bounced, bounced
                assert not default_context.pages[0].is_closed(), 'teardown closed a page it did not create'
            finally:
                await default_context.close()
    print('cdp attach ok -- isolated context, allowlist enforced, host browser left open')


async def _main() -> None:
    """Run every smoke scenario in sequence."""
    await _check_navigate()
    await _check_allowlist_bounce()
    await _check_storage_state_round_trip()
    await _check_cdp_attach()
    print('all checks passed')


if __name__ == '__main__':
    asyncio.run(_main())
