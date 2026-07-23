"""The batteries for your Pydantic AI agent -- the official capability library."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .code_mode import CodeMode
    from .filesystem import FileSystem
    from .guardrails import (
        GuardrailError,
        GuardResult,
        InputBlocked,
        InputGuard,
        InputGuardFunc,
        OutputBlocked,
        OutputGuard,
        OutputGuardFunc,
    )
    from .logfire import (
        AgentControl,
        ManagedPrompt,
    )
    from .shell import LLM_API_KEY_ENV_PATTERNS, Shell

__all__ = [
    'CodeMode',
    'FileSystem',
    'GuardResult',
    'GuardrailError',
    'InputBlocked',
    'InputGuard',
    'InputGuardFunc',
    'LLM_API_KEY_ENV_PATTERNS',
    'AgentControl',
    'ManagedPrompt',
    'OutputBlocked',
    'OutputGuard',
    'OutputGuardFunc',
    'Shell',
]

_GUARDRAIL_EXPORTS = {
    'GuardResult',
    'GuardrailError',
    'InputBlocked',
    'InputGuard',
    'InputGuardFunc',
    'OutputBlocked',
    'OutputGuard',
    'OutputGuardFunc',
}


def __getattr__(name: str) -> object:
    if name == 'CodeMode':
        from .code_mode import CodeMode

        return CodeMode
    if name in _GUARDRAIL_EXPORTS:
        from . import guardrails

        return getattr(guardrails, name)
    if name == 'FileSystem':
        from .filesystem import FileSystem

        return FileSystem
    if name == 'AgentControl':
        from .logfire import AgentControl

        return AgentControl
    if name == 'ManagedPrompt':
        from .logfire import ManagedPrompt

        return ManagedPrompt
    if name == 'Shell':
        from .shell import Shell

        return Shell
    if name == 'LLM_API_KEY_ENV_PATTERNS':
        from .shell import LLM_API_KEY_ENV_PATTERNS

        return LLM_API_KEY_ENV_PATTERNS
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
