"""Logfire-backed capabilities: drive agent configuration from Logfire managed variables."""

from pydantic_ai_harness.logfire._agent_control import (
    AgentConfig,
    AgentConfigSettings,
    AgentControl,
    ToolDefinitionOverride,
)
from pydantic_ai_harness.logfire._managed_prompt import ManagedPrompt
from pydantic_ai_harness.logfire._managed_variable import resolution_reason

__all__ = [
    'AgentControl',
    'AgentConfig',
    'AgentConfigSettings',
    'ManagedPrompt',
    'ToolDefinitionOverride',
    'resolution_reason',
]
