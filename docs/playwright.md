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
| `type_text` | `(selector, text, timeout_ms=None)` | page text after typing (replaces the field value, does not submit) |
| `press_key` | `(key, selector=None, timeout_ms=None)` | page text after the key press; `key` is a Playwright key name (`Enter`, `Escape`, `Tab`, `Control+a`) |
| `select_option` | `(selector, values, timeout_ms=None)` | page text after choosing options in a `<select>` |
| `hover` | `(selector, timeout_ms=None)` | page text after hovering, revealing hover-only menus |
| `wait_for` | `(selector=None, text=None, timeout_ms=None)` | page text once the element/text appears, in the page or any frame; pass exactly one of `selector`/`text` |
| `screenshot` | `(full_page=False, timeout_ms=None)` | a note with the page URL, plus the PNG as image content |
| `get_text` | `(selector=None, timeout_ms=None)` | the element's text, or the full page's visible text |
| `scroll` | `(direction, x=None, y=None, timeout_ms=None)` | page text after scrolling; `direction` is up/down/left/right |
| `go_back` | `(timeout_ms=None)` | the previous page's text |
| `go_forward` | `(timeout_ms=None)` | the next page's text |
| `execute_js` | `(script, timeout_ms=None)` | the JavaScript result (string as-is, objects as JSON, `null` as `undefined`) |
| `console_messages` | `(errors_only=False)` | console output and uncaught script errors, oldest first |
| `network_requests` | `(url_contains=None)` | requests the page made with their status, including ones the egress policy refused |

Every page action accepts an optional `timeout_ms` to override both defaults for
that one call. An override has to be greater than 0: `0` disables the deadline
entirely, which stays available as a capability default but not as an argument
the model picks.

Two defaults rather than one, because the two failures differ. An action that
misses (`click`, `get_text`, `wait_for`) is normally a selector matching nothing,
and `action_timeout_ms` (5s) turns that into a fast, readable failure instead of
a wait long enough to read as a hung agent. A page load legitimately takes
longer, so navigation, load settling, and starting or attaching to the browser
use `navigation_timeout_ms` (60s).

`snapshot` returns the page's accessibility tree, the low-cost structured way for
the model to read the page and obtain `aria-ref=eN` handles. Targeting an element
by its `aria-ref=` handle (passed to `click` or `type_text`) is more reliable than
a model-authored CSS selector. The snapshot includes iframe content (see
[Embedded content](#embedded-content-iframes)). Reach for `screenshot` only when
a visual check is needed (charts, layout).

`type_text` fills a field but does not submit it; `press_key('Enter')` does. A
native `<select>` does not open as page content, so `select_option` operates it
rather than `click`.

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
| `action_timeout_ms` | `5000` | Default deadline for element actions (click, type, read, wait). `0` disables it. |
| `navigation_timeout_ms` | `60000` | Default deadline for navigation and load settling, and for starting or attaching to the browser. `0` disables it. |
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

That lifecycle lives in `PlaywrightBrowserSession`, which the capability creates
per run. It is exported for the case where you want the same guarded browser
without an agent around it -- the allowlist, the private-address block, the
service-worker block, and the single-tab behavior all come with it:

```python
from pydantic_ai_harness.playwright import PlaywrightBrowserSession

async with PlaywrightBrowserSession() as session:
    page = await session.ensure_page()  # Chromium starts here, not on entry
    await page.goto('https://example.com')
```

`PlaywrightBrowserToolset` is exported on the same basis: pass it a session to
get the sixteen tools without the capability's hooks. The policy lives on the
session and only there, so the guard the session installs and the checks the
tools run cannot diverge: they are two layers of one decision, applied at
different moments.

## Embedded content (iframes)

Page-level selectors stop at the frame boundary, so an embedded schedule,
checkout step, or chat widget is not reachable through a CSS selector, and a page
that is mostly an embed can look almost empty. The capability closes that gap in
three places:

- Tools that return page text (`navigate`, `click`, `get_text` without a
  selector, and the rest) append the text of each child frame that has any,
  inside the same token budget.
- `wait_for` watches the page and every child frame at once; the first match
  wins.
- `snapshot` includes frame content, and its refs carry the frame they came
  from (`f1e4` rather than `e4`). Passing such a ref to `click`, `type_text`,
  `hover` or `get_text` resolves inside that frame -- the one handle that reaches
  embedded content.

The sweep over child frames is bounded, so one unresponsive embed cannot consume
the action's deadline; whatever the other frames returned is kept.

## Debugging a run

Three things make a browser agent hard to follow: the page is invisible, a
missing element looks the same as a slow one, and the interesting failures happen
between tool calls.

- Set `headless=False` to watch the run in a real window.
- Every browser operation opens an OpenTelemetry span named `browser <action>`
  (`browser click`, `browser navigate`), carrying `browser.action`,
  `browser.timeout_ms`, `browser.outcome`, and the resulting `url.full`. What the
  page did during that operation is attached as span events: console output,
  uncaught script errors, responses, requests the egress policy refused, and
  popups the session closed. The spans go to the run's own tracer, so an agent
  instrumented for [Logfire](/ai/guides/logfire/) reports them with everything
  else.
- The agent can read the same log through `console_messages` and
  `network_requests`, which is often how it recovers from a page that renders
  from an API rather than from HTML. Recorded URLs keep their host, path and
  parameter names but lose `user:password@` credentials and the values of
  credential-bearing parameters (`token`, `code`, `signature`, and the rest),
  since those reach both the model and the telemetry backend.
- A wait that seems to hang is usually an action timeout. `action_timeout_ms`
  defaults to 5s so the failure arrives quickly; lower it further while
  debugging, and read the timeout value in the error string to tell a slow page
  from a wrong selector.

Logfire's default scrubbing redacts values matching `session` (and `auth`), which
matches page content more often than you would expect -- a conference site whose
every heading says "session" comes back as `[Scrubbed due to 'session']`. Keep
tool results readable by scrubbing them selectively:

```python {test="skip"}
import logfire


def keep_browser_results(match: logfire.ScrubMatch) -> str | None:
    if match.path[:2] == ('attributes', 'tool_response'):
        return match.value
    return None


logfire.configure(scrubbing=logfire.ScrubbingOptions(callback=keep_browser_results))
```

Returning `match.value` keeps the original text; returning `None` leaves the
redaction in place.

## Limitations

- The browser is single-tab: popups are closed automatically, so flows that
  depend on a second window do not complete.
- CSS selectors cannot reach content inside iframes; reading and acting there
  goes through `snapshot` refs (see
  [Embedded content](#embedded-content-iframes)).
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
allowlist. Each entry matches its exact host and any subdomain, compared in the
ASCII form Chromium itself uses, so an internationalized host and its `xn--`
spelling get the same verdict. The two
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
by `execute_js`, and WebSocket connections, which a network route never sees, get
their own guard that refuses a socket to a private address and records why. Both channels end at the model: `snapshot()` reads the ARIA
tree of cross-origin child frames, and a page can read back a subresource it
fetched itself, so either would otherwise hand over the response body the block
exists to withhold.

Neither policy is a general security boundary. Microsoft's own playwright-mcp
disclaims its origin filter the same way. The allowlist governs navigation, not
requests initiated by in-page JavaScript (`fetch`/XHR via `execute_js`); both
policies match IP literals and `localhost` names, not hostnames that resolve to
private addresses (DNS rebinding). Constraining in-page requests against the allowlist and
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
