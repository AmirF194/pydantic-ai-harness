"""Browser capability: a real, stateful Chromium browser for agents, via async Playwright."""

from pydantic_ai_harness.browser._capability import Browser
from pydantic_ai_harness.browser._toolset import (
    DEFAULT_MAX_CONTENT_TOKENS,
    DEFAULT_TIMEOUT_MS,
    BrowserState,
    BrowserToolset,
)

__all__ = [
    'DEFAULT_MAX_CONTENT_TOKENS',
    'DEFAULT_TIMEOUT_MS',
    'Browser',
    'BrowserState',
    'BrowserToolset',
]
