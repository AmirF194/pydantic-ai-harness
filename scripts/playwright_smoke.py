"""Hand-run smoke test for the Playwright browser capability.

Not collected by pytest and not wired into CI: it launches a real Chromium and
reaches the public internet, which the mocked unit suite deliberately avoids. Run
it after installing the browser binary to verify the real integration end to end:

    playwright install chromium
    uv run python scripts/playwright_smoke.py

It exercises the public `PlaywrightBrowser` surface and checks eight things:

- lazy launch plus a real navigation to https://example.com (prints the title),
- the allowlist bounce (a disallowed host returns an error result, not content),
- the private-address block against a real local server, including the decimal
  spelling of an IP (which only Chromium's canonicalization resolves) and an
  in-page `fetch`, each with an opt-out control proving the server is reachable,
- a `storage_state` round-trip: a cookie captured from a real context is visible
  to the agent after relaunching the capability with that state,
- embedded content: text inside a real cross-origin iframe reaches the model, a
  `snapshot` ref from that frame clicks inside it, and `wait_for` matches content
  that appears there,
- the browser event log: console output and a failed request land in
  `console_messages` / `network_requests` (a refused request is checked in the
  private-address scenario, where the guard is what refuses it),
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
_SECRET = 'private-address-smoke-secret'


async def _run_tools(
    browser: PlaywrightBrowser[object],
    calls: list[tuple[str, dict[str, object]]],
    resolve_ref: tuple[str, int] | None = None,
) -> list[str]:
    """Drive a fixed sequence of tool calls through one agent run, in order.

    Emitting one scripted `ToolCallPart` per model turn keeps every call in the
    same run, so the launched browser (and its `storage_state` session) persists
    across the sequence. Returns each tool's stringified result.
    """
    results: list[str] = []
    index = 0

    def _with_ref(args: dict[str, object]) -> dict[str, object]:
        """Substitute `aria-ref=REF` with the ref of the named node in the last snapshot.

        A ref is only knowable at run time, so the scripted call carries a
        placeholder and the ref is read out of the snapshot that preceded it.
        """
        if resolve_ref is None or args.get('selector') != 'aria-ref=REF':
            return args
        label, snapshot_index = resolve_ref
        line = next(line for line in results[snapshot_index].splitlines() if label in line and '[ref=' in line)
        return {**args, 'selector': f'aria-ref={line.split("[ref=")[1].split("]")[0]}'}

    async def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal index
        for part in messages[-1].parts:
            if isinstance(part, ToolReturnPart):
                results.append(str(part.content))
        if index < len(calls):
            name, args = calls[index]
            index += 1
            return ModelResponse(parts=[ToolCallPart(tool_name=name, args=_with_ref(args))])
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


async def _serve_secret() -> tuple[asyncio.Server, int]:
    """Serve a fixed body over HTTP on a loopback port, so a leak is observable."""

    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        await reader.readline()
        body = _SECRET.encode()
        writer.write(
            b'HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\n'
            b'Access-Control-Allow-Origin: *\r\n'
            b'Content-Length: ' + str(len(body)).encode() + b'\r\n\r\n' + body
        )
        await writer.drain()
        writer.close()

    server = await asyncio.start_server(handle, '127.0.0.1', 0)
    return server, server.sockets[0].getsockname()[1]


async def _check_private_address_block() -> None:
    """The block must survive both an exotic IP spelling and an in-page fetch.

    Neither is decided by the pre-check: `2130706433` is not parseable as an
    address until Chromium canonicalizes it, and a `fetch` never goes through
    `navigate` at all. Both therefore rest on the route guard, and the mocked
    suite feeds that guard already-canonical URLs. Each assertion is paired with
    a `block_private_addresses=False` control, so a server that simply failed to
    start could not pass the check by accident.
    """
    server, port = await _serve_secret()
    async with server:
        decimal_url = f'http://2130706433:{port}/'
        fetch = f"fetch('http://127.0.0.1:{port}/').then(r => r.text())"

        blocked_nav, refusals = await _run_tools(
            PlaywrightBrowser[object](), [('navigate', {'url': decimal_url}), ('network_requests', {})]
        )
        assert _SECRET not in blocked_nav, blocked_nav
        # The refusal is the route guard's, so it carries the reason rather than
        # arriving as a bare network failure.
        assert 'request_blocked' in refusals and 'private or link-local' in refusals, refusals
        (open_nav,) = await _run_tools(
            PlaywrightBrowser[object](block_private_addresses=False), [('navigate', {'url': decimal_url})]
        )
        assert _SECRET in open_nav, f'control failed, server unreachable: {open_nav}'
        print('decimal-IP navigation blocked ok (control reached the server)')

        (blocked_fetch,) = await _run_tools(PlaywrightBrowser[object](), [('execute_js', {'script': fetch})])
        assert _SECRET not in blocked_fetch, blocked_fetch
        open_fetch = await _run_tools(
            PlaywrightBrowser[object](block_private_addresses=False),
            [('navigate', {'url': f'http://127.0.0.1:{port}/'}), ('execute_js', {'script': fetch})],
        )
        assert _SECRET in open_fetch[-1], f'control failed, fetch never reached the server: {open_fetch}'
        print('in-page fetch to a private address blocked ok (control reached the server)')


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


async def _serve_html(body: bytes) -> tuple[asyncio.Server, int]:
    """Serve a fixed HTML body over HTTP on a loopback port."""

    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        await reader.readline()
        writer.write(
            b'HTTP/1.1 200 OK\r\nContent-Type: text/html\r\n'
            b'Content-Length: ' + str(len(body)).encode() + b'\r\n\r\n' + body
        )
        await writer.drain()
        writer.close()

    server = await asyncio.start_server(handle, '127.0.0.1', 0)
    return server, server.sockets[0].getsockname()[1]


_INNER_FRAME = (
    b'<html><body><h1>Inner schedule</h1>'
    b'<button id="more">Show more</button>'
    b'<div id="late" style="display:none">LATE-CONTENT</div>'
    b"<script>document.getElementById('more').onclick = () => "
    b"{ document.getElementById('late').style.display = 'block' }</script>"
    b'</body></html>'
)


async def _check_embedded_frame() -> None:
    """Content inside a real iframe: readable, clickable by ref, waitable."""
    inner, inner_port = await _serve_html(_INNER_FRAME)
    outer, outer_port = await _serve_html(
        f'<html><body><h1>Conference</h1><iframe src="http://127.0.0.1:{inner_port}/" '
        f'width="600" height="400"></iframe></body></html>'.encode()
    )
    async with inner, outer:
        browser = PlaywrightBrowser[object](block_private_addresses=False)
        page_text, snapshot, _, waited = await _run_tools(
            browser,
            [
                ('navigate', {'url': f'http://127.0.0.1:{outer_port}/'}),
                ('snapshot', {}),
                ('click', {'selector': 'aria-ref=REF'}),
                ('wait_for', {'text': 'LATE-CONTENT', 'timeout_ms': 3000}),
            ],
            resolve_ref=('Show more', 1),
        )
        assert 'Inner schedule' in page_text, f'iframe text missing from the page read: {page_text}'
        assert 'f1e' in snapshot, f'snapshot carries no frame-scoped refs: {snapshot}'
        assert 'timed out' not in waited, waited
    print('embedded frame ok -- read, clicked by ref, and waited inside the iframe')


async def _check_browser_event_log() -> None:
    """Console output, a failed request, and a refused request all reach the tools."""
    server, port = await _serve_html(
        b'<html><body><script>console.error("page boom");'
        b'fetch("http://127.0.0.1:1/never").catch(() => {});'
        b'</script></body></html>'
    )
    async with server:
        browser = PlaywrightBrowser[object](block_private_addresses=False, allowed_domains=['127.0.0.1'])
        _, console, network = await _run_tools(
            browser,
            [
                ('navigate', {'url': f'http://127.0.0.1:{port}/'}),
                ('console_messages', {'errors_only': True}),
                ('network_requests', {}),
            ],
        )
        assert 'page boom' in console, console
        assert '127.0.0.1:1/never' in network, network
    print('browser event log ok -- console error and failed request recorded')


async def _main() -> None:
    """Run every smoke scenario in sequence."""
    await _check_navigate()
    await _check_allowlist_bounce()
    await _check_private_address_block()
    await _check_storage_state_round_trip()
    await _check_embedded_frame()
    await _check_browser_event_log()
    await _check_cdp_attach()
    print('all checks passed')


if __name__ == '__main__':
    asyncio.run(_main())
