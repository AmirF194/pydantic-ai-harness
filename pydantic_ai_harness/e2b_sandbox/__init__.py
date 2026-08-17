"""E2B sandbox capability: gives agents an isolated cloud computer to work in.

`E2BSandbox` is the supported entry point; build an agent with it and use its
tools. `E2BSandboxSession` exposes lower-level lifecycle, command, and file access
for applications that need to share a caller-owned sandbox across runs. The
model-facing toolset remains an implementation detail of the capability.
"""

from pydantic_ai_harness.e2b_sandbox._capability import E2BSandbox
from pydantic_ai_harness.e2b_sandbox._session import (
    E2BSandboxAuthError,
    E2BSandboxError,
    E2BSandboxExecResult,
    E2BSandboxSession,
    E2BSandboxTerminalError,
    E2BSandboxUnavailableError,
)

__all__ = [
    'E2BSandbox',
    'E2BSandboxAuthError',
    'E2BSandboxError',
    'E2BSandboxExecResult',
    'E2BSandboxSession',
    'E2BSandboxTerminalError',
    'E2BSandboxUnavailableError',
]
