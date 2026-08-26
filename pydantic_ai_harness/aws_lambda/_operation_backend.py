from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from typing import Any, Literal

from aws_durable_execution_sdk_python.config import StepConfig
from aws_durable_execution_sdk_python.exceptions import ExecutionError
from pydantic_ai.durable_exec._operation import (
    CallToolId,
    DurableOperationId,
    GetInstructionsId,
    GetToolsId,
    OperationConfigRole,
    ToolsetKind,
    ValidateToolArgumentsId,
)
from pydantic_ai.durable_exec._operation_backend import CallableOperationBackend
from pydantic_ai.durable_exec._operation_names import JournalOperationNamer
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

    def base(self, role: OperationConfigRole, operation_id: DurableOperationId) -> StepConfig | None:
        if role is OperationConfigRole.CAPABILITY:
            return None
        if role in (OperationConfigRole.MODEL, OperationConfigRole.EVENT):
            return _step_config(self._base)
        if isinstance(operation_id, GetInstructionsId):
            kind: ToolsetKind = 'mcp'
        else:
            assert isinstance(operation_id, GetToolsId | CallToolId | ValidateToolArgumentsId)
            kind = operation_id.toolset_kind
        config = self._tool(kind, None, '')
        assert config is not False
        return config

    def for_tool(
        self,
        role: OperationConfigRole,
        operation_id: DurableOperationId,
        tool: object | None,
        tool_name: str,
    ) -> StepConfig | Literal[False] | None:
        assert role in (OperationConfigRole.TOOL_CALL, OperationConfigRole.TOOL_VALIDATION)
        assert isinstance(operation_id, CallToolId | ValidateToolArgumentsId)
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

    async def _execute(
        self,
        *,
        name: str,
        body: Callable[[], Awaitable[object]],
        cache_key: tuple[object, ...],
        config: object,
    ) -> object:
        del cache_key
        assert config is None or isinstance(config, StepConfig)

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
