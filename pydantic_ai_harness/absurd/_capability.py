"""Durable execution for Pydantic AI agents on Absurd."""

from __future__ import annotations

try:
    import absurd_sdk  # noqa: F401  # pyright: ignore[reportUnusedImport]
except ImportError as _import_error:  # pragma: no cover
    raise ImportError(
        'The Absurd durability capability requires the `absurd` extra. '
        'Install it with `pip install "pydantic-ai-harness[absurd]"` or '
        '`uv add "pydantic-ai-harness[absurd]"`.'
    ) from _import_error

from collections.abc import Mapping
from dataclasses import dataclass
from threading import Lock
from typing import Any, ClassVar, Literal, get_args

from absurd_sdk import AsyncTaskContext, JsonValue
from pydantic import TypeAdapter
from pydantic_ai import FunctionToolset
from pydantic_ai.agent import EventStreamHandler, ParallelExecutionMode
from pydantic_ai.agent.abstract import AbstractAgent
from pydantic_ai.capabilities.abstract import WrapModelRequestHandler, WrapRunHandler
from pydantic_ai.durable_exec._base import BaseDurabilityCapability
from pydantic_ai.durable_exec._runtime_toolsets import RuntimeToolsetKind, reject_unsupported_runtime_toolsets
from pydantic_ai.durable_exec._utils import DurableModel, StreamedActivityResult, capture_event_stream
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import AgentStreamEvent, ModelResponse, ModelResponseStreamEvent
from pydantic_ai.models import CompletedStreamedResponse, Model, ModelRequestContext
from pydantic_ai.run import AgentRunResult
from pydantic_ai.tools import AgentDepsT, RunContext
from pydantic_ai.toolsets import AbstractToolset, WrapperToolset

from ._function_toolset import AbsurdFunctionToolset
from ._utils import current_async_context

AbsurdParallelExecutionMode = Literal['sequential', 'parallel_ordered_events']
"""Tool-call execution modes supported by Absurd."""

_response_adapter: TypeAdapter[ModelResponse] = TypeAdapter(ModelResponse)
_events_adapter: TypeAdapter[list[ModelResponseStreamEvent]] = TypeAdapter(list[ModelResponseStreamEvent])
_allowed_parallel_execution_modes: tuple[str, ...] = get_args(AbsurdParallelExecutionMode)
_active_run_namespaces: set[tuple[AsyncTaskContext, str]] = set()
_active_run_namespaces_lock = Lock()


def _reject_hash_in_step_suffix(value: str, *, kind: str) -> None:
    if '#' in value:
        raise UserError(f'{kind} {value!r} contains `#`, which Absurd uses to disambiguate repeated steps.')


def _serialize_response(response: ModelResponse) -> JsonValue:
    return _response_adapter.dump_python(response, mode='json')


def _deserialize_response(payload: JsonValue) -> ModelResponse:
    return _response_adapter.validate_python(payload)


@dataclass(init=False)
class AbsurdDurability(BaseDurabilityCapability[AgentDepsT]):
    """Checkpoint model, MCP, and function-tool I/O in Absurd steps.

    Use `capabilities=[AbsurdDurability()]` and run the agent from an async Absurd task
    handler. Outside an Absurd task, the capability behaves like a normal capability.
    """

    engine_name = 'Absurd'
    _unsupported_runtime_toolset_kinds: ClassVar[frozenset[RuntimeToolsetKind]] = frozenset(
        {'function', 'mcp', 'dynamic'}
    )
    _durable_unit_noun = 'step'
    _durable_container_noun = 'task'

    def __init__(
        self,
        *,
        models: Mapping[str, Model] | None = None,
        event_stream_handler: EventStreamHandler[AgentDepsT] | None = None,
        name: str | None = None,
        parallel_execution_mode: AbsurdParallelExecutionMode = 'sequential',
    ) -> None:
        super().__init__(models=models, event_stream_handler=event_stream_handler, name=name)
        if parallel_execution_mode not in _allowed_parallel_execution_modes:
            allowed = ' or '.join(repr(mode) for mode in _allowed_parallel_execution_modes)
            raise UserError(f'Absurd parallel execution mode must be {allowed}, got {parallel_execution_mode!r}.')
        self._parallel_execution_mode: ParallelExecutionMode = parallel_execution_mode
        self._wrappers_by_leaf: dict[int, WrapperToolset[AgentDepsT]] = {}
        self._construction_leaves: set[int] = set()
        self._default_model_id: str | None = None

    @property
    def in_durable_context(self) -> bool:
        return current_async_context() is not None

    def _bind_to_agent(self, agent: AbstractAgent[AgentDepsT, Any]) -> None:
        self._default_model_id = agent.model if isinstance(agent.model, str) else None
        self._wrappers_by_leaf = {}
        self._construction_leaves = set()
        seen_ids: dict[str, AbstractToolset[AgentDepsT]] = {}

        def register(toolset: AbstractToolset[AgentDepsT]) -> AbstractToolset[AgentDepsT]:
            if toolset.id is not None:
                existing = seen_ids.get(toolset.id)
                if existing is not None and existing is not toolset:
                    raise UserError(f'Two toolsets have the same `id` {toolset.id!r}.')
                seen_ids[toolset.id] = toolset
            leaf_id = id(toolset)
            if leaf_id not in self._construction_leaves:
                self._construction_leaves.add(leaf_id)
                wrapper = self._wrap_leaf_toolset(toolset)
                if wrapper is not None:
                    self._wrappers_by_leaf[leaf_id] = wrapper
            return toolset

        for toolset in agent.toolsets:
            toolset.visit_and_replace(register)

    def _wrap_leaf_toolset(self, ts: AbstractToolset[AgentDepsT]) -> WrapperToolset[AgentDepsT] | None:
        if isinstance(ts, FunctionToolset):
            return AbsurdFunctionToolset(
                wrapped=ts, step_name_prefix=self.name, durable_run_context_scope=self._durable_run_context_scope
            )
        try:
            from pydantic_ai.mcp import MCPToolset
        except ImportError:  # pragma: no cover
            return None
        if isinstance(ts, MCPToolset):
            from ._mcp import AbsurdMCPToolset

            return AbsurdMCPToolset(
                wrapped=ts, step_name_prefix=self.name, durable_run_context_scope=self._durable_run_context_scope
            )
        return None

    def get_wrapper_toolset(self, toolset: AbstractToolset[AgentDepsT]) -> AbstractToolset[AgentDepsT] | None:
        in_durable_context = self.in_durable_context
        runtime_leaves: list[AbstractToolset[AgentDepsT]] = []

        def swap(leaf: AbstractToolset[AgentDepsT]) -> AbstractToolset[AgentDepsT]:
            if in_durable_context and id(leaf) not in self._construction_leaves:
                runtime_leaves.append(leaf)
            return self._wrappers_by_leaf.get(id(leaf), leaf)

        swapped = toolset.visit_and_replace(swap)
        reject_unsupported_runtime_toolsets(
            runtime_leaves,
            unsupported_kinds=self._unsupported_runtime_toolset_kinds,
            engine=self.engine_name,
        )
        return swapped

    async def _dispatch_event_stream_event(self, ctx: RunContext[AgentDepsT], event: AgentStreamEvent) -> None:
        task_ctx = current_async_context()
        assert task_ctx is not None  # pragma: no cover
        handler = self._event_stream_handler
        assert handler is not None  # pragma: no cover

        async def inner() -> None:
            with self._durable_run_context_scope(ctx) as step_ctx:
                await handler(step_ctx, self._single_event_stream(event))

        await task_ctx.step(f'{self.name}__event_stream_handler', inner)

    async def wrap_run(self, ctx: RunContext[AgentDepsT], *, handler: WrapRunHandler) -> AgentRunResult[Any]:
        agent = self._agent
        task_ctx = current_async_context()
        if agent is None:  # pragma: no cover
            return await handler()
        if task_ctx is None:
            return await handler()
        namespace_key = (task_ctx, self.name)
        with _active_run_namespaces_lock:
            if namespace_key in _active_run_namespaces:
                raise UserError(
                    f'Concurrent Absurd agent runs with checkpoint namespace {self.name!r} are not supported '
                    'in the same task context. Await one run before starting another, or use a distinct '
                    '`AbsurdDurability(name=...)` value or task context.'
                )
            _active_run_namespaces.add(namespace_key)
        try:
            with agent.parallel_tool_call_execution_mode(self._parallel_execution_mode):
                return await handler()
        finally:
            with _active_run_namespaces_lock:
                _active_run_namespaces.remove(namespace_key)

    async def wrap_model_request(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        request_context: ModelRequestContext,
        handler: WrapModelRequestHandler,
    ) -> ModelResponse:
        task_ctx = current_async_context()
        if task_ctx is None:
            return await handler(request_context)

        model_id = self._model_id_for_request(ctx, request_context)
        if model_id == self._default_model_id:
            model_id = None
        if model_id is not None:
            _reject_hash_in_step_suffix(model_id, kind='Model ID')
        step_suffix = '' if model_id is None else f'.{model_id}'
        model = request_context.model

        async def request_segment(request: ModelRequestContext) -> ModelResponse:
            async def inner() -> JsonValue:
                with self._durable_run_context_scope(ctx):
                    response = await request.model.request(
                        request.messages, request.model_settings, request.model_request_parameters
                    )
                return _serialize_response(response)

            payload = await task_ctx.step(f'{self.name}__model.request{step_suffix}', inner)
            return _deserialize_response(payload)

        async def request_stream_segment(request: ModelRequestContext) -> StreamedActivityResult:
            async def inner() -> dict[str, JsonValue]:
                with self._durable_run_context_scope(ctx) as step_ctx:
                    async with request.model.request_stream(
                        request.messages, request.model_settings, request.model_request_parameters, step_ctx
                    ) as streamed:
                        events = await capture_event_stream(
                            run_context=step_ctx, stream=streamed, handler=self._event_stream_handler
                        )
                return {
                    'response': _serialize_response(streamed.get()),
                    'events': _events_adapter.dump_python(events, mode='json'),
                }

            payload = await task_ctx.step(f'{self.name}__model.request_stream{step_suffix}', inner)
            if 'response' not in payload:
                response = _deserialize_response(payload)
                completed = CompletedStreamedResponse(
                    response, model_request_parameters=request.model_request_parameters, replay_events=True
                )
                return StreamedActivityResult(response=response, events=[event async for event in completed])
            return StreamedActivityResult(
                response=_deserialize_response(payload['response']),
                events=_events_adapter.validate_python(payload['events']),
            )

        async def cancel_suspended_response_segment(response: ModelResponse) -> None:
            async def inner() -> None:
                with self._durable_run_context_scope(ctx):
                    await model.cancel_suspended_response(response)

            await task_ctx.step(f'{self.name}__model.cancel_suspended_response{step_suffix}', inner)

        request_context.model = DurableModel(
            request_context.model,
            request_segment=request_segment,
            request_stream_segment=request_stream_segment,
            cancel_suspended_response_segment=cancel_suspended_response_segment,
        )
        return await handler(request_context)
