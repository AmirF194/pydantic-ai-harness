---
title: Playwright Browser
description: Give a Pydantic AI agent a real, stateful Chromium browser via async Playwright -- navigate, click, type, scroll, extract page text, run JavaScript, and screenshot JS-heavy or authenticated pages.
---

# Playwright Browser

`PlaywrightBrowser` gives an agent a real, stateful Chromium browser via async
[Playwright](https://playwright.dev/python/): navigate, click, type, scroll,
move through history, extract page text, run JavaScript, and screenshot. Reach
for it when the lighter web tools fall short -- pages behind login or session
cookies, JavaScript-rendered SPAs, and interactive multi-step flows. For
query-based research prefer [Exa Search](exa-search.md); for a static URL prefer
a web-fetch tool.

[Source](https://github.com/pydantic/pydantic-ai-harness/tree/main/pydantic_ai_harness/playwright/)

## Installation

The `playwright` extra pulls in Playwright, and Chromium is a separate binary
download:

```bash
pip install 'pydantic-ai-harness[playwright]'
playwright install chromium
```

If the Chromium binary is missing at runtime, the browser tools are hidden from
the model and calling one raises with a `playwright install chromium` hint. Set
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
| `navigate` | `(url)` | page URL, title, and visible text (truncated) |
| `click` | `(selector)` | page text after the click; `selector` is a CSS selector or `'x,y'` pixel coordinates |
| `type_text` | `(selector, text)` | page text after typing (replaces the field value) |
| `screenshot` | `(full_page=False)` | a note with the page URL, plus the PNG as image content |
| `get_text` | `(selector=None)` | the element's text, or the full page's visible text |
| `scroll` | `(direction, x=None, y=None)` | page text after scrolling; `direction` is up/down/left/right |
| `go_back` | `()` | the previous page's text |
| `go_forward` | `()` | the next page's text |
| `execute_js` | `(script)` | the JavaScript result (string as-is, objects as JSON, `null` as `undefined`) |

`screenshot` (and the optional `screenshot_on_navigate` attachment) return the
image as [`BinaryContent`](/ai/api/messages/#pydantic_ai.messages.BinaryContent)
rather than a base64 string, so vision models see the image natively instead of
a wall of base64 in the text context.

## Options

| Option | Default | Purpose |
|---|---|---|
| `headless` | `True` | Run Chromium without a visible window (suits servers and CI). |
| `allowed_domains` | `None` | Navigation allowlist; `None` allows all domains (see [Egress](#egress-and-ssrf)). |
| `screenshot_on_navigate` | `False` | Attach a screenshot to every `navigate` result. |
| `max_content_tokens` | `4000` | Approximate token budget for page text. |
| `timeout_ms` | `30000` | Default Playwright navigation/action timeout. |
| `auto_install_chromium` | `False` | Fetch Chromium automatically when the binary is missing. |

## Lifecycle

Chromium starts lazily on the first browser-tool call and is closed when the run
ends -- on success, error, or cancellation. Runs that never call a browser tool
pay no Playwright cost (no subprocess, no window). Each agent run gets its own
page and browser, so concurrent `agent.run()` calls never share a tab.

## Egress and SSRF

With `allowed_domains=None` (the default) the agent can reach any URL, including
internal or link-local addresses such as `169.254.169.254` (cloud metadata) or
`localhost`. When the agent may act on untrusted input, set `allowed_domains` to
an explicit allowlist. Each entry matches its exact host and any subdomain. The
allowlist is enforced at two layers: a network route guard aborts disallowed
top-level navigations (covering clicks, `execute_js`, and history moves, not just
`navigate`), and each tool re-checks the resulting URL and bounces to
`about:blank` so disallowed content never reaches the model.

The allowlist governs page navigation (top-level document requests), not requests
initiated by in-page JavaScript (`fetch`/XHR via `execute_js`); constraining those
is tracked in #415.

Blocking private and link-local address ranges by default (so open egress is
safe out of the box) is tracked in
[#415](https://github.com/pydantic/pydantic-ai-harness/issues/415); this
capability ships the opt-in allowlist and documents the default, and does not yet
block internal addresses on its own.

## Stability

This capability is available today and its API may change as the harness
evolves. Pin the harness version if you need a stable surface.

## API reference

::: pydantic_ai_harness.playwright.PlaywrightBrowser
