"""Logfire-backed capabilities: drive agent configuration from Logfire managed variables."""

from pydantic_ai_harness.logfire._managed_agent import (
    AgentConfig,
    AgentConfigSettings,
    ManagedAgent,
    ToolDefinitionOverride,
)
from pydantic_ai_harness.logfire._managed_prompt import ManagedPrompt
from pydantic_ai_harness.logfire._managed_variable import resolution_reason

__all__ = [
    'AgentConfig',
    'AgentConfigSettings',
    'ManagedAgent',
    'ManagedPrompt',
    'ToolDefinitionOverride',
    'resolution_reason',
]
