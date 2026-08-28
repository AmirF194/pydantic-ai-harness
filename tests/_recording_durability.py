from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Literal

from pydantic_ai.durable_exec import (
    JSON_CODEC,
    BaseDurabilityCapability,
    CallableOperationBackend,
    DurabilityEngineSpec,
    DurableOperationId,
    JournalOperationNamer,
    OperationConfigRole,
)


class _RecordingConfig:
    def base(self, role: OperationConfigRole, *, operation_id: DurableOperationId) -> None:
        return None

    def for_tool(
        self,
        role: OperationConfigRole,
        *,
        operation_id: DurableOperationId,
        tool: object | None,
        tool_name: str,
    ) -> None | Literal[False]:
        return None


class _RecordingBackend(CallableOperationBackend[None]):
    def __init__(self, agent_name: str, calls: list[tuple[str, tuple[object, ...]]]) -> None:
        super().__init__(namer=JournalOperationNamer(agent_name), config=_RecordingConfig())
        self.calls = calls

    async def execute(
        self,
        *,
        operation_id: DurableOperationId,
        name: str,
        body: Callable[[], Awaitable[object]],
        cache_key: tuple[object, ...],
        config: None,
    ) -> object:
        self.calls.append((name, cache_key))
        return await body()


class RecordingDurability(BaseDurabilityCapability[object]):
    engine_spec = DurabilityEngineSpec(
        engine_name='recording',
        durable_unit_noun='unit',
        durable_container_noun='journal',
        codec=JSON_CODEC,
        wrapped_toolset_kinds=frozenset(),
    )

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[tuple[str, tuple[object, ...]]] = []

    @property
    def in_durable_context(self) -> bool:
        return True

    def get_durable_operation_backend(self) -> _RecordingBackend:
        return _RecordingBackend(self.name, self.calls)
