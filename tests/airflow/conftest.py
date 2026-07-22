"""Test helpers for the Airflow durability capability.

No Airflow, no Docker, no network: the capability memoizes into an in-memory
`InMemoryDurableStorage` (which JSON-round-trips every value, so a value that would
be skipped in production is observable here), and an Airflow task retry is simulated
by running the agent a second time against the same populated store. `RunLog` counts
how many model requests and tool calls actually executed, so a replay can assert zero
re-execution.
"""

from __future__ import annotations

from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel


class RunLog:
    """Counts operations that actually executed (not served from the store)."""

    def __init__(self) -> None:
        self.model_requests: int = 0
        self.tool_calls: list[str] = []


def _answered(messages: list[ModelMessage]) -> bool:
    return any(isinstance(part, ToolReturnPart) for message in messages for part in message.parts)


def tool_then_text(log: RunLog | None = None, *, tool_name: str = 'act') -> FunctionModel:
    """A model that calls `tool_name` once, then answers `'done'`.

    `log`, when given, records each model request so a replay can assert the model was not called.
    """

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if log is not None:
            log.model_requests += 1
        if _answered(messages):
            return ModelResponse(parts=[TextPart('done')])
        return ModelResponse(parts=[ToolCallPart(tool_name, {})])

    return FunctionModel(model_fn)
