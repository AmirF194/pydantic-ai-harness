from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from typing import Any, ClassVar

from pydantic_ai.durable_exec._base import BaseDurabilityCapability
from pydantic_ai.durable_exec._codec import JSON_CODEC
from pydantic_ai.durable_exec._operation import ToolsetKind
from pydantic_ai.durable_exec._toolset import Lifecycle


class RecordingDurability(BaseDurabilityCapability[Any]):
    engine_name = 'recording'
    _durable_unit_noun = 'unit'
    _durable_container_noun = 'journal'
    _codec: ClassVar = JSON_CODEC
    _unsupported_runtime_toolset_kinds: ClassVar = frozenset()
    _wrapped_toolset_kinds: ClassVar = frozenset()
    _toolset_lifecycles: ClassVar[Mapping[ToolsetKind, Lifecycle]] = {
        'function': 'enter-always',
        'mcp': 'enter-always',
        'dynamic': 'enter-never',
    }

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[tuple[str, tuple[object, ...]]] = []

    @property
    def in_durable_context(self) -> bool:
        return True

    async def run_durable_unit(
        self,
        name: str,
        fn: Callable[[], Awaitable[Any]],
        *,
        inputs: tuple[Any, ...],
        config: Any,
    ) -> Any:
        self.calls.append((name, inputs))
        return await fn()
