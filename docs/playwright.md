---
title: Playwright Browser
description: Give a Pydantic AI agent a real, stateful Chromium browser via async Playwright -- navigate, click, type, scroll, extract page text, run JavaScript, and screenshot JS-heavy or authenticated pages.
---

# Playwright Browser

`PlaywrightBrowser` gives an agent a real, stateful Chromium browser via async
[Playwright](https://playwright.dev/python/): navigate, click, type, scroll,
move through history, extract page text, run JavaScript, and screenshot.

Reach for it when the lighter web tools fall short. A web search tool answers a
research question without loading a page, and a web-fetch tool handles a known
static URL. This capability covers what neither can reach: pages behind login or session cookies,
JavaScript-rendered SPAs, and interactive multi-step flows.

[Source](https://github.com/pydantic/pydantic-ai-harness/tree/main/pydantic_ai_harness/playwright/)

!!! note "Which browser capability?"
    This page covers driving the browser from the host model: typed,
    deterministic actions, one tool call at a time. To hand an open-ended goal
    to an autonomous agent that drives the browser for you, see
    [Browser Use](browser-use.md). Give an agent one or the other: each
    capability runs its own browser, so a session opened by one is not visible
    to the other.

## Installation

The `playwright` extra pulls in Playwright, and Chromium is a separate binary
download:

```bash
pip install 'pydantic-ai-harness[playwright]'
playwright install chromium
```

If the Chromium binary is missing at runtime, the first browser tool call fails
the run with a `playwright install chromium` hint. Set
`auto_install_chromium=True` to fetch the binary automatically on the first miss.

## Usage

```python
from pydantic_ai import Agent
from pydantic_ai_harness.playwright import PlaywrightBrowser

agent = Agent('anthropic:claude-sonnet-4-6', capabilities=[PlaywrightBrowser()])
result = await agent.run('Open https://example.com and tell me the page title.')
```

`PlaywrightBrowser` is a [capability](/ai/core-concepts/capabilities/): it registers the
browser toolset, injects short when-to-use guidance into the system prompt, and
manages the Chromium lifecycle for the run.

## Tools

| Tool | Signature | Returns |
|---|---|---|
| `navigate` | `(url, timeout_ms=None)` | page URL, title, and visible text (truncated) |
| `snapshot` | `(timeout_ms=None)` | the accessibility tree with `aria-ref` handles (truncated) |
| `click` | `(selector, timeout_ms=None)` | page text after the click; `selector` is a CSS selector, an `aria-ref=` handle, or `'x,y'` pixel coordinates |
| `type_text` | `(selector, text, timeout_ms=None)` | page text after typing (replaces the field value) |
| `wait_for` | `(selector=None, text=None, timeout_ms=None)` | page text once the element/text appears; pass exactly one of `selector`/`text` |
| `screenshot` | `(full_page=False, timeout_ms=None)` | a note with the page URL, plus the PNG as image content |
| `get_text` | `(selector=None, timeout_ms=None)` | the element's text, or the full page's visible text |
| `scroll` | `(direction, x=None, y=None, timeout_ms=None)` | page text after scrolling; `direction` is up/down/left/right |
| `go_back` | `(timeout_ms=None)` | the previous page's text |
| `go_forward` | `(timeout_ms=None)` | the next page's text |
| `execute_js` | `(script, timeout_ms=None)` | the JavaScript result (string as-is, objects as JSON, `null` as `undefined`) |

Every page action accepts an optional `timeout_ms` to override the capability's
default `timeout_ms` for that one call. An override has to be greater than 0:
`0` disables the deadline entirely, which stays available as the capability
default but not as an argument the model picks.

`snapshot` returns the page's accessibility tree, the low-cost structured way for
the model to read the page and obtain `aria-ref=eN` handles. Targeting an element
by its `aria-ref=` handle (passed to `click` or `type_text`) is more reliable than
a model-authored CSS selector. The snapshot includes iframe content, so it
partially covers the iframe read limitation noted below. Reach for `screenshot`
only when a visual check is needed (charts, layout).

`screenshot` (and the optional `screenshot_on_navigate` attachment) return the
image as [`BinaryContent`](/ai/api/messages/#pydantic_ai.messages.BinaryContent)
rather than a base64 string, so vision models see the image natively instead of
a wall of base64 in the text context. A capture over 5 MB (typically a full-page
screenshot of a long page) is returned as a bounded error instead of image
content, because model providers reject oversized images and the failure would
otherwise abort the run; capture the viewport or scroll and capture sections
instead.

Browser tool failures -- a timeout, a selector that matches no element, a
navigation error, or a browser that closed mid-run -- are returned to the model
as error strings it can act on (retry, try another selector, navigate again),
not raised to abort the agent run.

## Options

| Option | Default | Purpose |
|---|---|---|
| `headless` | `True` | Run Chromium without a visible window (suits servers and CI). |
| `allowed_domains` | `None` | Navigation allowlist; `None` allows all domains (see [Egress](#egress-and-ssrf)). |
| `block_private_addresses` | `True` | Refuse navigation to private, loopback, link-local, and other reserved IP literals (see [Egress](#egress-and-ssrf)). |
| `screenshot_on_navigate` | `False` | Attach a screenshot to every `navigate` result. |
| `max_content_tokens` | `4000` | Approximate token budget for every textual tool result. |
| `timeout_ms` | `30000` | Default Playwright navigation/action timeout. `0` disables it. |
| `auto_install_chromium` | `False` | Fetch Chromium automatically when the binary is missing. |
| `storage_state` | `None` | Playwright storage state (cookies + localStorage) loaded at launch; see [Authenticated sites](#authenticated-sites). |
| `cdp_url` | `None` | Attach to a Chromium already running at this CDP endpoint instead of launching one; see [Attaching to a running browser](#attaching-to-a-running-browser). |

## Authenticated sites

Pass `storage_state` to start the browser already logged in. It is a Playwright
[storage state](https://playwright.dev/python/docs/auth) object -- cookies plus
localStorage -- loaded into the browser context at launch, so the first
navigation is already authenticated.

Capture it once, in your own code, by logging in with a visible browser:

```python
from playwright.async_api import async_playwright


async def capture_state() -> object:
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=False)
        context = await browser.new_context()
        page = await context.new_page()
        await page.goto('https://example.com/login')
        # log in by hand in the window that opened
        state = await context.storage_state()
        await browser.close()
        return state
```

`playwright codegen https://example.com --save-storage=auth.json` writes the same
structure to a file, which you load with `json.loads(Path('auth.json').read_text())`.
Either way, hand the object to the capability:

```python
from pydantic_ai import Agent
from pydantic_ai_harness.playwright import PlaywrightBrowser

state = ...  # captured above

agent = Agent(
    'anthropic:claude-sonnet-4-6',
    capabilities=[
        PlaywrightBrowser(storage_state=state, allowed_domains=['example.com']),
    ],
)
```

The option takes an object rather than a path because agents commonly run in
shared environments where the local filesystem is not somewhere to put anything
durable: where the state lives is your decision, not the capability's. For the
same reason `PlaywrightBrowser.from_spec` does not accept `storage_state` --
a spec would carry the cookies into whatever stores it. Set it on the
constructed capability instead.

The capability runs headless by default and does not drive the login flow
itself: you capture the state out of band. The usual pattern is to log in once
with a visible browser, then reuse the state for headless runs.

Treat the state as credential material: it can impersonate the account. Keep it
out of source control and out of logs, store it with restrictive permissions if
you do persist it, and discard it when the session expires. This mirrors
Playwright's own [auth-guide](https://playwright.dev/python/docs/auth) warning.
Prefer a minimal-scope state (log in to only the target site when capturing it)
over reusing a full browser profile.

## Attaching to a running browser

Set `cdp_url` to connect to a Chromium that is already running at a
[Chrome DevTools Protocol](https://playwright.dev/python/docs/api/class-browsertype#browser-type-connect-over-cdp)
endpoint instead of launching one:

```python
from pydantic_ai import Agent
from pydantic_ai_harness.playwright import PlaywrightBrowser

agent = Agent(
    'anthropic:claude-sonnet-4-6',
    capabilities=[PlaywrightBrowser(cdp_url='http://localhost:9222')],
)
```

No local Chromium binary is involved, so the install hint and
`auto_install_chromium` do not apply. This is what managed-browser providers and
benchmark harnesses expect: they start a browser and hand you its endpoint.

The run still gets its own browser context, so `storage_state`, the domain
allowlist, and the private-address block apply exactly as they do for a launched
browser, and the agent does not inherit the sessions already open in that
Chrome. Pointing `cdp_url` at your personal everyday browser is still the
higher-risk choice -- that process holds every account you are logged into, and
anything the agent can reach through it is reachable in one prompt injection.
Prefer a browser started for the agent, with a scoped `storage_state`. Provider
endpoints sometimes embed an auth token in the URL; treat those as secrets.

## Lifecycle

Chromium starts lazily on the first browser-tool call and is closed when the run
ends -- on success, error, or cancellation. Runs that never call a browser tool
pay no Playwright cost (no subprocess, no window). Each agent run gets its own
page and browser, so concurrent `agent.run()` calls never share a tab.

## Limitations

- The browser is single-tab: popups are closed automatically, so flows that
  depend on a second window do not complete.
- Page-level selectors cannot reach content inside iframes (payment widgets,
  some CAPTCHAs); `snapshot` does surface iframe content for reading.
- Download-triggering clicks are not handled.
- Durable execution (e.g. `TemporalDurability`) is rejected at agent
  construction: a live Chromium page cannot survive activity replay or worker
  restart.
- The model targets elements by `aria-ref=` handle (from `snapshot`), CSS
  selector, or pixel coordinates.

## Egress and SSRF

By default the browser refuses navigation to IP literals that are not globally
routable -- `169.254.169.254` (the cloud metadata endpoint), `127.0.0.1`, `::1`,
`localhost` and `*.localhost` names, and the RFC 1918 private ranges -- even
when no allowlist is set. Set `block_private_addresses=False` when the agent
should reach a local app or an internal dashboard.

With `allowed_domains=None` (the default) the agent can reach any public URL.
When the agent may act on untrusted input, set `allowed_domains` to an explicit
allowlist. Each entry matches its exact host and any subdomain. The two
policies are independent: an allowlisted private address is still refused until
you opt out of `block_private_addresses`. Both are enforced at two layers: a
network route guard aborts disallowed requests before they leave (covering
clicks, `execute_js`, and history moves, not just `navigate`), and each tool
re-checks the resulting URL and bounces to `about:blank` so disallowed content
never reaches the model. Service workers are blocked in the browser context so their
traffic cannot slip past the route guard.

The two policies differ in how far they reach. The allowlist governs top-level
navigation only, so a page's own subresources and third-party frames (identity
providers, payment steps) still load. The private-address block applies to
**every** frame and **every** resource type, including a `fetch` or XHR issued
by `execute_js`. Both channels end at the model: `snapshot()` reads the ARIA
tree of cross-origin child frames, and a page can read back a subresource it
fetched itself, so either would otherwise hand over the response body the block
exists to withhold.

Neither policy is a general security boundary. Microsoft's own playwright-mcp
disclaims its origin filter the same way. The allowlist governs navigation, not
requests initiated by in-page JavaScript (`fetch`/XHR via `execute_js`); both
policies match IP literals and `localhost` names, not hostnames that resolve to
private addresses (DNS rebinding); and the route guard does not see WebSocket
connections. Constraining in-page requests against the allowlist and
resolution-based blocking are tracked in
[#415](https://github.com/pydantic/pydantic-ai-harness/issues/415).

For untrusted-input scenarios, run the browser in a container or VM with an
egress firewall, or front it with a proxy, and pair it with the harness's
tool-approval hooks for consequential actions. Treat these as defense in depth,
not a guarantee.

## Stability

This capability is available today and its API may change as the harness
evolves. Pin the harness version if you need a stable surface.

## API reference

::: pydantic_ai_harness.playwright.PlaywrightBrowser
