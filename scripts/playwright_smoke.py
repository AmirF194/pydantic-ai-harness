"""Hand-run smoke test for the Playwright browser capability.

Not collected by pytest and not wired into CI: it launches a real Chromium and
reaches the public internet, which the mocked unit suite deliberately avoids. Run
it after installing the browser binary to verify the real integration end to end:

    playwright install chromium
    uv run python scripts/playwright_smoke.py

It exercises the public `PlaywrightBrowser` surface and checks four things:

- lazy launch plus a real navigation to https://example.com (prints the title),
- the allowlist bounce (a disallowed host returns an error result, not content),
- a `storage_state` round-trip: a cookie captured from a real context is visible
  to the agent after relaunching the capability with that state,
- clean teardown: each scenario runs its own capability, whose `wrap_run` closes
  the browser when the run ends. After it prints `all checks passed`, confirm no
  Chromium lingered, e.g. `pgrep -fl chromium` shows nothing this script started.
"""

from __future__ import annotations

import asyncio

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


async def _main() -> None:
    """Run every smoke scenario in sequence."""
    await _check_navigate()
    await _check_allowlist_bounce()
    await _check_storage_state_round_trip()
    print('all checks passed')


if __name__ == '__main__':
    asyncio.run(_main())
