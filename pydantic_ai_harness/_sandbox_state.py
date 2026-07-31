"""Shared default location for per-capability persistent state inside `ctx.sandbox`."""

from __future__ import annotations

import posixpath

from pydantic_ai.sandboxes import Sandbox

_ROOT_SEGMENT = '.pydantic-ai-harness'


async def default_state_dir(sandbox: Sandbox, capability: str) -> str:
    """Return `<sandbox.working_dir()>/.pydantic-ai-harness/<capability>/` and ensure it exists."""
    if not capability or '/' in capability or capability in ('.', '..'):
        raise ValueError(f'invalid capability name: {capability!r}')
    root = posixpath.join(await sandbox.working_dir(), _ROOT_SEGMENT, capability)
    await sandbox.fs.make_dir(root)
    return root
