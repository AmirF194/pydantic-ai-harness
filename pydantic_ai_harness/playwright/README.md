# Playwright Browser

Give an agent a real, stateful Chromium browser via async
[Playwright](https://playwright.dev/python/): navigate, click, type, scroll,
move through history, extract page text, run JavaScript, and screenshot. Unlike
query-based web research ([`ExaSearch`](../exa/)) or a static HTTP fetch, the
browser renders JavaScript-heavy SPAs, carries session cookies through login and
multi-step flows, and reads content that only appears after interaction.

[Source](https://github.com/pydantic/pydantic-ai-harness/tree/main/pydantic_ai_harness/playwright/)

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
| `scroll` | `(direction, x=None, y=None)` | page text after scrolling; `direction` is up/down/left/right |
| `go_back` | `(timeout_ms=None)` | the previous page's text |
| `go_forward` | `(timeout_ms=None)` | the next page's text |
| `execute_js` | `(script)` | the JavaScript result (string as-is, objects as JSON, `null` as `undefined`) |

Every page action accepts an optional `timeout_ms` to override the capability's
default `timeout_ms` for that one call.

`snapshot` returns the page's accessibility tree, the low-cost structured way for
the model to read the page and obtain `aria-ref=eN` handles. Targeting an element
by its `aria-ref=` handle (passed to `click` or `type_text`) is more reliable than
a model-authored CSS selector. The snapshot includes iframe content, so it
partially covers the iframe read limitation noted below. Reach for `screenshot`
only when a visual check is needed (charts, layout).

`screenshot` (and the optional `screenshot_on_navigate` attachment) return the
image as [`BinaryContent`](https://ai.pydantic.dev/api/messages/#pydantic_ai.messages.BinaryContent)
rather than a base64 string, so vision models see the image natively instead of
a wall of base64 in the text context.

Browser tool failures -- a timeout, a selector that matches no element, a
navigation error, or a browser that closed mid-run -- are returned to the model
as error strings it can act on (retry, try another selector, navigate again),
not raised to abort the agent run.

## Options

| Option | Default | Purpose |
|---|---|---|
| `headless` | `True` | Run Chromium without a visible window (suits servers and CI). |
| `allowed_domains` | `None` | Navigation allowlist; `None` allows all domains (see [Egress](#egress-and-ssrf)). |
| `screenshot_on_navigate` | `False` | Attach a screenshot to every `navigate` result. |
| `max_content_tokens` | `4000` | Approximate token budget for every textual tool result. |
| `timeout_ms` | `30000` | Default Playwright navigation/action timeout. |
| `auto_install_chromium` | `False` | Fetch Chromium automatically when the binary is missing. |
| `storage_state` | `None` | Path to a Playwright storage-state JSON file (cookies + localStorage) loaded at launch; see [Authenticated sites](#authenticated-sites). |

## Authenticated sites

Pass `storage_state` to start the browser already logged in. It is the path to a
Playwright [storage state](https://playwright.dev/python/docs/auth) JSON file
(cookies plus localStorage) loaded into the browser context at launch, so the
first navigation is already authenticated.

Produce the file once by logging in by hand, either with the Playwright CLI:

```bash
playwright codegen https://example.com --save-storage=auth.json
```

or from your own script with `context.storage_state(path='auth.json')`. Then
point the capability at it:

```python
from pydantic_ai import Agent
from pydantic_ai_harness.playwright import PlaywrightBrowser

agent = Agent(
    'anthropic:claude-sonnet-4-6',
    capabilities=[
        PlaywrightBrowser(storage_state='auth.json', allowed_domains=['example.com']),
    ],
)
```

The capability runs headless by default and does not drive the login flow
itself: you produce the state file out of band. The usual pattern is to log in
once with a visible browser (`playwright codegen`, or your own Playwright
script) to write the file, then reuse it for headless runs.

Treat the state file as credential material: it can impersonate the account.
gitignore it, never commit it, store it with restrictive permissions, and delete
it when the session expires. This mirrors Playwright's own
[auth-guide](https://playwright.dev/python/docs/auth) warning. Prefer a
minimal-scope file (log in to only the target site when producing it) over
sharing a full browser profile. Attaching the agent to your real running Chrome
(the browser-extension or connect-to-Chrome approach) is the higher-risk
alternative: it exposes every session that browser is logged into, so prefer the
scoped state file.

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
- The model targets elements by `aria-ref=` handle (from `snapshot`), CSS
  selector, or pixel coordinates.

## Egress and SSRF

With `allowed_domains=None` (the default) the agent can reach any URL, including
internal or link-local addresses such as `169.254.169.254` (cloud metadata) or
`localhost`. **When the agent may act on untrusted input, set `allowed_domains`
to an explicit allowlist.** Each entry matches its exact host and any subdomain.
The allowlist is enforced at two layers: a network route guard aborts disallowed
top-level navigations (covering clicks, `execute_js`, and history moves, not just
`navigate`), and each tool re-checks the resulting URL and bounces to
`about:blank` so disallowed content never reaches the model.

`allowed_domains` constrains top-level navigation; it is not a general security
boundary. Microsoft's own playwright-mcp disclaims its origin filter the same
way. The allowlist governs page navigation (top-level document requests), not
requests initiated by in-page JavaScript (`fetch`/XHR via `execute_js`);
constraining those is tracked in #415. WebSocket connections and service-worker
traffic are likewise not covered by the navigation allowlist.

For untrusted-input scenarios, run the browser in a container or VM with an
egress firewall, or front it with a proxy, and pair it with the harness's
tool-approval hooks for consequential actions. Treat these as defense in depth,
not a guarantee.

Blocking private and link-local address ranges by default (so open egress is
safe out of the box) is tracked in
[#415](https://github.com/pydantic/pydantic-ai-harness/issues/415); this
capability ships the opt-in allowlist and documents the default, and does not yet
block internal addresses on its own.

## Stability

This capability is available today and its API may change as the harness
evolves. Pin the harness version if you need a stable surface.
