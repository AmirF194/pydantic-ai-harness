"""Shared private model helpers for Absurd capability tests."""

from __future__ import annotations

from collections.abc import AsyncIterator

from absurd_sdk import JsonValue
from pydantic_ai.messages import ModelMessage, ModelResponse, RetryPromptPart, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel


def _text_model(counter: dict[str, int] | None = None) -> FunctionModel:
    tally = counter if counter is not None else {'calls': 0}

    def fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        tally['calls'] += 1
        return ModelResponse(parts=[TextPart(content='ok')])

    async def stream_fn(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        tally['calls'] += 1
        yield 'ok'

    return FunctionModel(fn, stream_function=stream_fn, model_name='fn')


def _tool_then_done_model(tool_name: str, args: dict[str, JsonValue]) -> FunctionModel:
    def fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        answered = any(
            isinstance(part, (ToolReturnPart, RetryPromptPart)) for message in messages for part in message.parts
        )
        if not answered:
            return ModelResponse(parts=[ToolCallPart(tool_name=tool_name, args=args)])
        return ModelResponse(parts=[TextPart(content='done')])

    return FunctionModel(fn, model_name='fn')
