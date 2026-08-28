from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from typing import Any, Literal

from aws_durable_execution_sdk_python.config import StepConfig
from aws_durable_execution_sdk_python.exceptions import ExecutionError
from pydantic_ai.durable_exec import (
    CallableOperationBackend,
    DurableOperationId,
    JournalOperationNamer,
    OperationConfigRole,
    ToolsetCallToolId,
    ToolsetGetInstructionsId,
    ToolsetGetToolsId,
    ToolsetKind,
    ToolsetValidateToolArgumentsId,
)
from pydantic_core import PydanticSerializationError

from ._bridge import current_bridge


class AWSLambdaOperationConfig:
    def __init__(
        self,
        *,
        base: Mapping[str, Any] | None,
        tool: Callable[[ToolsetKind, object | None, str], StepConfig | Literal[False] | None],
    ) -> None:
        self._base = base
        self._tool = tool

    def base(self, role: OperationConfigRole, *, operation_id: DurableOperationId) -> StepConfig | None:
        if role == 'capability':
            return None
        if role in ('model', 'event'):
            return _step_config(self._base)
        if isinstance(operation_id, ToolsetGetInstructionsId):
            kind: ToolsetKind = 'mcp'
        else:
            assert isinstance(operation_id, ToolsetGetToolsId | ToolsetCallToolId | ToolsetValidateToolArgumentsId)
            kind = operation_id.toolset_kind
        config = self._tool(kind, None, '')
        assert config is not False
        return config

    def for_tool(
        self,
        role: OperationConfigRole,
        *,
        operation_id: DurableOperationId,
        tool: object | None,
        tool_name: str,
    ) -> StepConfig | Literal[False] | None:
        assert role == 'tool'
        assert isinstance(operation_id, ToolsetCallToolId | ToolsetValidateToolArgumentsId)
        return self._tool(operation_id.toolset_kind, tool, tool_name)


class AWSLambdaOperationBackend(CallableOperationBackend[StepConfig | None]):
    def __init__(
        self,
        *,
        agent_name: str,
        default_model_id: str | None,
        config: AWSLambdaOperationConfig,
    ) -> None:
        super().__init__(
            namer=JournalOperationNamer(agent_name, default_model_id=default_model_id or 'default'), config=config
        )

    async def execute(
        self,
        *,
        operation_id: DurableOperationId,
        name: str,
        body: Callable[[], Awaitable[object]],
        cache_key: tuple[object, ...],
        config: StepConfig | None,
    ) -> object:
        del operation_id, cache_key

        async def operation() -> object:
            try:
                return await body()
            except PydanticSerializationError as exc:
                raise ExecutionError(str(exc)) from exc

        bridge = current_bridge()
        assert bridge is not None  # pragma: no cover - execution in a durable context guarantees a bridge
        return await bridge.run_step(name, operation, config)


def _step_config(config: Mapping[str, Any] | None) -> StepConfig | None:
    if not config:
        return None
    return StepConfig(**config)
