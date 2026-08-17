"""Modal sandbox capability: gives agents an isolated cloud sandbox to work in.

`ModalSandbox` is the supported entry point; build an agent with it and use its tools.
`ModalSandboxBackend` is the Modal implementation of Pydantic AI's sandbox backend protocol,
public for applications that want to create or attach to a sandbox themselves and pass it to
a run as `sandbox=`. The model-facing toolset remains an implementation detail of the
capability.
"""

from pydantic_ai_harness.modal_sandbox._backend import (
    ModalSandboxAuthError,
    ModalSandboxBackend,
    ModalSandboxCommandTimeoutError,
    ModalSandboxError,
    ModalSandboxTerminalError,
    ModalSandboxUnavailableError,
)
from pydantic_ai_harness.modal_sandbox._capability import ModalSandbox

__all__ = [
    'ModalSandbox',
    'ModalSandboxAuthError',
    'ModalSandboxBackend',
    'ModalSandboxCommandTimeoutError',
    'ModalSandboxError',
    'ModalSandboxTerminalError',
    'ModalSandboxUnavailableError',
]
