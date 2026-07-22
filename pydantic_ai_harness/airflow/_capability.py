"""Durable execution for Pydantic AI agents on Apache Airflow, shaped as a capability.

`AirflowDurability` memoizes an agent's I/O -- model requests, function tool calls, MCP calls, and
dynamic-toolset resolution -- into a pluggable storage backend, keyed positionally and verified by a
content fingerprint. It is the capability-shaped reference version of Airflow's
`apache-airflow-providers-common-ai` durable integration, which memoizes the same operations but is
attached via an operator and keyed the same positional-plus-fingerprint way.

There is no suspend/resume here and no per-operation retry: the durable unit is the whole agent run
(an Airflow task), and recovery is a task retry that re-runs the agent from the top and replays each
completed operation from the store. See the module README for what is reference-quality versus what a
production Airflow backend would add.
"""

from __future__ import annotations

import json
import warnings
from collections.abc import Awaitable, Callable, Mapping, Sequence
from contextvars import ContextVar
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, ClassVar, TypeVar

from pydantic import TypeAdapter
from pydantic_ai import FunctionToolset, ToolsetTool
from pydantic_ai._enqueue import PendingMessage  # pyright: ignore[reportPrivateUsage]
from pydantic_ai.agent import EventStreamHandler
from pydantic_ai.agent.abstract import AbstractAgent
from pydantic_ai.capabilities.abstract import WrapModelRequestHandler, WrapRunHandler
from pydantic_ai.durable_exec._base import BaseDurabilityCapability  # pyright: ignore[reportPrivateUsage]
from pydantic_ai.durable_exec._runtime_toolsets import RuntimeToolsetKind  # pyright: ignore[reportPrivateUsage]
from pydantic_ai.durable_exec._toolset import (  # pyright: ignore[reportPrivateUsage]
    CallToolResult,
    DurableDynamicToolset,
    DurableFunctionToolset,
    DurableMCPToolset,
    DynamicToolsResult,
    EnqueueGuard,
    ToolConfig,
    call_dynamic_tool,
    get_dynamic_tools,
    resolve_tool_durable_config,
    unwrap_recorded_tool_call_result,
    wrap_tool_call_result,
)
from pydantic_ai.durable_exec._utils import (  # pyright: ignore[reportPrivateUsage]
    DurableModel,
    StreamedActivityResult,
    capture_event_stream,
)
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import AgentStreamEvent, InstructionPart, ModelResponse, ModelResponseStreamEvent
from pydantic_ai.models import Model, ModelRequestContext
from pydantic_ai.run import AgentRunResult
from pydantic_ai.tools import AgentDepsT, RunContext, ToolDefinition
from pydantic_ai.toolsets import AbstractToolset, WrapperToolset
from pydantic_ai.toolsets._dynamic import DynamicToolset  # pyright: ignore[reportPrivateUsage]
from pydantic_core import PydanticSerializationError

from ._fingerprint import fingerprint_model_request, fingerprint_token, fingerprint_tool_call
from ._storage import DURABLE_KEY_PREFIX, DurableStorage, InMemoryDurableStorage

if TYPE_CHECKING:
    from pydantic import JsonValue
    from pydantic_ai.mcp import MCPToolset

_ENGINE_NAME = 'Airflow'
_TOOL_CONFIG_KEY = 'airflow'
_TOOL_CONFIG_LABEL = 'Airflow durable config'
_NO_FALLBACK_CONFIG: Mapping[str, ToolConfig] = {}

_Instructions = str | InstructionPart | Sequence[str | InstructionPart] | None
_ResultT = TypeVar('_ResultT')

_response_adapter: TypeAdapter[ModelResponse] = TypeAdapter(ModelResponse)
_events_adapter: TypeAdapter[list[ModelResponseStreamEvent]] = TypeAdapter(list[ModelResponseStreamEvent])
_call_tool_result_adapter: TypeAdapter[CallToolResult] = TypeAdapter(CallToolResult)
_tool_defs_adapter: TypeAdapter[dict[str, ToolDefinition]] = TypeAdapter(dict[str, ToolDefinition])
_dynamic_tools_adapter: TypeAdapter[DynamicToolsResult] = TypeAdapter(DynamicToolsResult)
_instructions_adapter: TypeAdapter[_Instructions] = TypeAdapter(_Instructions)


class AirflowDurabilityWarning(UserWarning):
    """Warns that a step did not replay from the store and ran live instead.

    Emitted when a stored entry's fingerprint does not match the current request (the run diverged
    from the attempt that produced it), or when a result could not be serialized and so was not
    memoized. Neither is an error: the operation still runs and returns normally.
    """


# Per-run monotonic step counter, isolated per run via `wrap_run`. The counter object is mutable and
# shared by reference across any tasks the run copies the context into, so a cache hit and a live run
# advance the same sequence and positional keys line up on replay.
_step_counter: ContextVar[_StepCounter | None] = ContextVar('pai_airflow_step_counter', default=None)


class _StepCounter:
    def __init__(self) -> None:
        self._step = 0

    def next_step(self) -> int:
        step = self._step
        self._step += 1
        return step


def _current_counter() -> _StepCounter:
    counter = _step_counter.get()
    assert counter is not None  # pragma: no cover - set by `wrap_run` before any operation runs
    return counter


def _serialize_json(adapter: TypeAdapter[_ResultT]) -> Callable[[_ResultT], JsonValue]:
    def dump(value: _ResultT) -> JsonValue:
        return adapter.dump_python(value, mode='json')

    return dump


def _deserialize_json(adapter: TypeAdapter[_ResultT]) -> Callable[[JsonValue], _ResultT]:
    def load(payload: JsonValue) -> _ResultT:
        return adapter.validate_python(payload)

    return load


def _guard_enqueue(ctx: RunContext[AgentDepsT]) -> RunContext[AgentDepsT]:
    """Make `ctx.enqueue()` raise inside any code that runs within a memoized unit.

    A retry serves a memoized unit's stored result without re-running its code, so messages enqueued
    inside it (from a tool call, or from an `event_stream_handler` delivered events inside the model
    or event-handler unit) would be silently dropped. Enqueue from task-level code instead. Applied
    at every site that hands a run context to user code: tool calls and both event-delivery paths,
    matching pydantic-ai #6671. When this targets slim 2.16.0 it collapses onto the shared
    `_durable_run_context` / `_durable_run_context_scope` helpers that ship there.
    """
    guard: list[PendingMessage] = EnqueueGuard(
        '`ctx.enqueue()` is not supported inside an Airflow memoized unit: a retry serves the stored '
        'result without re-running the code, so the enqueued messages would be dropped. Enqueue '
        'messages from task-level code instead.'
    )
    return replace(ctx, pending_messages=guard)


def _resolve_function_tool_config(tool: ToolsetTool[Any] | None, tool_name: str) -> ToolConfig:
    config = resolve_tool_durable_config(
        tool, tool_name, _NO_FALLBACK_CONFIG, metadata_key=_TOOL_CONFIG_KEY, config_type_label=_TOOL_CONFIG_LABEL
    )
    if config is not False and config:
        raise UserError(
            f'Tool {tool_name!r} sets a non-empty {_TOOL_CONFIG_KEY!r} durable config, but Airflow '
            'memoization takes no per-tool options, so the config would have no effect. Only '
            f'metadata={{{_TOOL_CONFIG_KEY!r}: False}} (run the tool inline, un-memoized) is '
            'supported; remove the config.'
        )
    return config


def _resolve_mcp_tool_config(tool: ToolsetTool[Any] | None, tool_name: str) -> ToolConfig:
    config = _resolve_function_tool_config(tool, tool_name)
    if config is False:
        raise UserError(
            f'{_ENGINE_NAME} memoization for MCP tool {tool_name!r} was disabled with '
            f'metadata={{{_TOOL_CONFIG_KEY!r}: False}}, but MCP tools perform I/O and so cannot run '
            'outside a memoized step. Remove the metadata so the call stays memoized.'
        )
    return config


@dataclass(init=False)
class AirflowDurability(BaseDurabilityCapability[AgentDepsT]):
    """Capability that memoizes an agent's I/O into a pluggable durable store, for Airflow-style retry recovery.

    Attach it with `capabilities=[AirflowDurability(storage=...)]`. Every model request, function
    tool call, MCP call, and dynamic-toolset resolution is memoized under a positional step key plus
    a fingerprint of the request that produced it. If the agent run fails, an Airflow task retry
    re-runs it from the top: each completed operation is served from the store without re-executing,
    while the operation that failed (and anything after it) runs live. A stored entry whose
    fingerprint no longer matches the current request is discarded and the step re-runs, so a changed
    prompt, model, or tool call does not replay a stale result.

    This is the capability-shaped reference version of Airflow's `common.ai` durable provider. It
    differs from the durable-execution capabilities for out-of-process engines: there is no
    suspend/resume, the durable unit is the whole run (an Airflow task), and `in_durable_context` is
    a construction-time flag, not a runtime check -- attaching the capability (with `durable=True`,
    the default) makes the run memoized, mirroring the provider's `AgentOperator(durable=...)` flag.
    Construct with `durable=False` for a transparent, non-memoized run.

    Improvements over the provider it models: structured tool results (`ToolReturn`,
    `BinaryContent`) and control-flow signals (`ModelRetry`, `ApprovalRequired`, `CallDeferred`)
    round-trip through the store, and MCP tool listings/instructions and dynamic-toolset resolution
    are memoized (the provider re-runs those live). A result that cannot be serialized is skipped
    with a warning rather than crashing the run.

    Example:
        ```python {test="skip"}
        from pydantic_ai import Agent
        from pydantic_ai_harness.airflow import AirflowDurability, InMemoryDurableStorage

        storage = InMemoryDurableStorage()
        agent = Agent('openai:gpt-5', name='analyst', capabilities=[AirflowDurability(storage=storage)])


        @agent.tool_plain
        def get_weather(city: str) -> str:
            return f'It is sunny in {city}.'
        ```
    """

    engine_name = _ENGINE_NAME
    _unsupported_runtime_toolset_kinds: ClassVar[frozenset[RuntimeToolsetKind]] = frozenset(
        {'function', 'mcp', 'dynamic'}
    )
    _durable_unit_noun = 'step'
    _durable_container_noun = 'task'
    _tool_config_key = _TOOL_CONFIG_KEY

    def __init__(
        self,
        *,
        storage: DurableStorage | None = None,
        models: Mapping[str, Model] | None = None,
        event_stream_handler: EventStreamHandler[AgentDepsT] | None = None,
        name: str | None = None,
        durable: bool = True,
    ) -> None:
        """Create an `AirflowDurability` capability.

        The agent's model, name, and toolsets are discovered when the capability is bound.

        Args:
            storage: The [`DurableStorage`][pydantic_ai_harness.airflow.DurableStorage] backend that
                holds memoized operations across retries. Defaults to a fresh
                [`InMemoryDurableStorage`][pydantic_ai_harness.airflow.InMemoryDurableStorage],
                which persists only within the process; a real deployment passes a backend that
                writes to Airflow's ObjectStorage or task state store.
            models: Optional additional models keyed by ID for run-time model switching via
                `agent.run(model='<id>')`. The ID is folded into the model step key so a retry maps
                each entry back to the model it was recorded for.
            event_stream_handler: Optional event stream handler. Model events are handled live inside
                the model step; each agent-level event is memoized in its own step so its side
                effects do not repeat on a retry.
            name: Unique agent name used as the prefix for every step key. Defaults to the agent's
                `name` when the capability is bound.
            durable: Whether memoization is active. Defaults to `True`. When `False` the capability
                is transparent and the run is an ordinary, non-memoized agent run. This is a
                construction-time flag because, unlike an out-of-process engine, there is no
                Airflow-task context to detect at run time.
        """
        super().__init__(models=models, event_stream_handler=event_stream_handler, name=name)
        self._storage: DurableStorage = storage if storage is not None else InMemoryDurableStorage()
        self._durable = durable
        self._default_model_id: str | None = None

    @property
    def in_durable_context(self) -> bool:
        return self._durable

    def _bind_to_agent(self, agent: AbstractAgent[AgentDepsT, Any]) -> None:
        # Memoization happens through ad-hoc store lookups, so there is nothing to register up front
        # beyond the durable toolset wrappers. Recording the string default lets a request that
        # carries the raw default string as provenance key under the suffix-less model step.
        self._default_model_id = agent.model if isinstance(agent.model, str) else None
        self._register_toolsets(agent)

    # --- The durable "execute" verb ---

    async def _memoize(
        self,
        *,
        kind: str,
        suffix: str,
        fingerprint: str | None,
        run: Callable[[], Awaitable[_ResultT]],
        dump: Callable[[_ResultT], JsonValue],
        load: Callable[[JsonValue], _ResultT],
    ) -> _ResultT:
        """Positional-plus-fingerprint memoization of one operation.

        The counter is advanced before the store is consulted (even on a hit) so that a replay,
        where earlier steps are served from the store, still reaches later steps at the same
        positions it did on the recorded attempt.
        """
        step = _current_counter().next_step()
        key = f'{DURABLE_KEY_PREFIX}{kind}_step_{step}{suffix}'
        entry = self._storage.load(key)
        if entry is not None:
            if entry.fingerprint == fingerprint:
                return load(entry.value)
            warnings.warn(
                f'Airflow durability: stored {kind} step {step} does not match the current request; '
                'running it live instead of replaying.',
                AirflowDurabilityWarning,
                stacklevel=2,
            )
        value = await run()
        payload = _try_dump(value, dump)
        if isinstance(payload, _Skip):
            warnings.warn(
                f'Airflow durability: {kind} step {step} result is not serializable; running it live '
                'without memoizing.',
                AirflowDurabilityWarning,
                stacklevel=2,
            )
            return value
        self._storage.save(key, payload, fingerprint=fingerprint)
        return value

    # --- Toolset wrapping ---

    def _wrap_leaf_toolset(self, ts: AbstractToolset[AgentDepsT]) -> WrapperToolset[AgentDepsT] | None:
        if isinstance(ts, FunctionToolset):
            return self._build_function_toolset(ts)
        if isinstance(ts, DynamicToolset):
            return self._build_dynamic_toolset(ts)
        try:
            from pydantic_ai.mcp import MCPToolset
        except ImportError:  # pragma: no cover - MCP wrapping only applies when the mcp extra is installed
            return None
        if isinstance(ts, MCPToolset):
            return self._build_mcp_toolset(ts)
        return None

    def _build_function_toolset(self, toolset: FunctionToolset[AgentDepsT]) -> DurableFunctionToolset[AgentDepsT]:
        async def call_tool_operation(
            tool_name: str,
            tool_args: dict[str, Any],
            ctx: RunContext[AgentDepsT],
            tool: ToolsetTool[AgentDepsT],
            config: Mapping[str, Any],
        ) -> Any:
            result = await self._memoize(
                kind='tool',
                suffix='',
                fingerprint=fingerprint_tool_call(tool_name, tool_args, ctx.tool_call_id),
                run=lambda: wrap_tool_call_result(toolset.call_tool(tool_name, tool_args, _guard_enqueue(ctx), tool)),
                dump=_serialize_json(_call_tool_result_adapter),
                load=_deserialize_json(_call_tool_result_adapter),
            )
            return unwrap_recorded_tool_call_result(result)

        return DurableFunctionToolset(
            toolset,
            in_durable_context=self._in_durable_context,
            call_tool_operation=call_tool_operation,
            resolve_tool_config=_resolve_function_tool_config,
            lifecycle='enter-always',
        )

    def _build_mcp_toolset(self, toolset: MCPToolset[AgentDepsT]) -> DurableMCPToolset[AgentDepsT]:
        toolset_token = toolset.id or ''

        async def get_tools_operation(ctx: RunContext[AgentDepsT]) -> dict[str, ToolDefinition]:
            return await self._memoize(
                kind='mcp_get_tools',
                suffix='',
                fingerprint=fingerprint_token(toolset_token),
                run=lambda: _mcp_tool_defs(toolset, ctx),
                dump=_serialize_json(_tool_defs_adapter),
                load=_deserialize_json(_tool_defs_adapter),
            )

        async def get_instructions_operation(ctx: RunContext[AgentDepsT]) -> _Instructions:
            return await self._memoize(
                kind='mcp_get_instructions',
                suffix='',
                fingerprint=fingerprint_token(toolset_token),
                run=lambda: toolset.get_instructions(ctx),
                dump=_serialize_json(_instructions_adapter),
                load=_deserialize_json(_instructions_adapter),
            )

        async def call_tool_operation(
            tool_name: str,
            tool_args: dict[str, Any],
            ctx: RunContext[AgentDepsT],
            tool: ToolsetTool[AgentDepsT],
            config: Mapping[str, Any],
        ) -> Any:
            result = await self._memoize(
                kind='mcp_call_tool',
                suffix='',
                fingerprint=fingerprint_tool_call(tool_name, tool_args, ctx.tool_call_id),
                run=lambda: wrap_tool_call_result(toolset.call_tool(tool_name, tool_args, _guard_enqueue(ctx), tool)),
                dump=_serialize_json(_call_tool_result_adapter),
                load=_deserialize_json(_call_tool_result_adapter),
            )
            return unwrap_recorded_tool_call_result(result)

        return DurableMCPToolset(
            toolset,
            in_durable_context=self._in_durable_context,
            get_tools_operation=get_tools_operation,
            get_instructions_operation=get_instructions_operation,
            call_tool_operation=call_tool_operation,
            resolve_tool_config=_resolve_mcp_tool_config,
            lifecycle='enter-always',
        )

    def _build_dynamic_toolset(self, toolset: DynamicToolset[AgentDepsT]) -> DurableDynamicToolset[AgentDepsT]:
        toolset_token = toolset.id or ''

        async def get_tools_operation(ctx: RunContext[AgentDepsT]) -> DynamicToolsResult:
            return await self._memoize(
                kind='dynamic_get_tools',
                suffix='',
                fingerprint=fingerprint_token(toolset_token),
                run=lambda: get_dynamic_tools(toolset, ctx),
                dump=_serialize_json(_dynamic_tools_adapter),
                load=_deserialize_json(_dynamic_tools_adapter),
            )

        async def call_tool_operation(
            tool_name: str,
            tool_args: dict[str, Any],
            ctx: RunContext[AgentDepsT],
            tool: ToolsetTool[AgentDepsT],
            config: Mapping[str, Any],
        ) -> Any:
            result = await self._memoize(
                kind='dynamic_call_tool',
                suffix='',
                fingerprint=fingerprint_tool_call(tool_name, tool_args, ctx.tool_call_id),
                run=lambda: wrap_tool_call_result(
                    call_dynamic_tool(toolset, tool_name, tool_args, _guard_enqueue(ctx))
                ),
                dump=_serialize_json(_call_tool_result_adapter),
                load=_deserialize_json(_call_tool_result_adapter),
            )
            return unwrap_recorded_tool_call_result(result)

        return DurableDynamicToolset(
            toolset,
            in_durable_context=self._in_durable_context,
            get_tools_operation=get_tools_operation,
            call_tool_operation=call_tool_operation,
            resolve_tool_config=_resolve_function_tool_config,
            lifecycle='enter-never',
        )

    def _in_durable_context(self) -> bool:
        return self._durable

    # --- Capability hooks ---

    async def _dispatch_event_stream_event(self, ctx: RunContext[AgentDepsT], event: AgentStreamEvent) -> None:
        handler = self._event_stream_handler
        assert handler is not None  # pragma: no cover - only dispatched when a handler is set
        # The handler is user code running inside the memoized event-handler unit, so an enqueue
        # here would be dropped on a retry (pydantic-ai #6671); hand it a guarded context.
        guarded_ctx = _guard_enqueue(ctx)

        async def run() -> None:
            await handler(guarded_ctx, self._single_event_stream(event))

        # Memoize the handler call so its side effects do not repeat on a retry. There is no stable
        # per-attempt content to fingerprint, so replay is positional only.
        await self._memoize(
            kind='event_stream_handler',
            suffix='',
            fingerprint=None,
            run=run,
            dump=lambda _: None,
            load=lambda _: None,
        )

    async def wrap_run(self, ctx: RunContext[AgentDepsT], *, handler: WrapRunHandler) -> AgentRunResult[Any]:
        """Isolate a per-run step counter and run tool calls sequentially inside a memoized run.

        A step's identity is its position in the operation sequence, so concurrently scheduled tool
        calls could claim each other's positions on a retry. Outside a memoized run (`durable=False`)
        the agent's configured parallelism is left untouched.
        """
        agent = self._agent
        if agent is None or not self._durable:
            return await handler()
        token = _step_counter.set(_StepCounter())
        try:
            with agent.parallel_tool_call_execution_mode('sequential'):
                return await handler()
        finally:
            _step_counter.reset(token)

    async def wrap_model_request(
        self, ctx: RunContext[AgentDepsT], *, request_context: ModelRequestContext, handler: WrapModelRequestHandler
    ) -> ModelResponse:
        """Memoize each model-request segment into a positional step."""
        if not self._durable:
            return await handler(request_context)

        # The step runs in the same process, so the live model is used directly; the model id only
        # disambiguates the step key. A string default carries itself as provenance, so suppress the
        # suffix for it to keep one key for the default model.
        model_id = self._model_id_for_request(ctx, request_context)
        if model_id is not None and model_id == self._default_model_id:
            model_id = None
        suffix = '' if model_id is None else f'.{model_id}'
        model = request_context.model

        async def request_segment(request: ModelRequestContext) -> ModelResponse:
            prepared_settings, prepared_parameters = request.model.prepare_request(
                request.model_settings, request.model_request_parameters
            )
            fingerprint = fingerprint_model_request(
                request.model.model_id, request.messages, prepared_settings, prepared_parameters
            )
            return await self._memoize(
                kind='model',
                suffix=suffix,
                fingerprint=fingerprint,
                run=lambda: request.model.request(
                    request.messages, request.model_settings, request.model_request_parameters
                ),
                dump=_serialize_json(_response_adapter),
                load=_deserialize_json(_response_adapter),
            )

        async def request_stream_segment(request: ModelRequestContext) -> StreamedActivityResult:
            prepared_settings, prepared_parameters = request.model.prepare_request(
                request.model_settings, request.model_request_parameters
            )
            fingerprint = fingerprint_model_request(
                request.model.model_id, request.messages, prepared_settings, prepared_parameters
            )

            async def run() -> StreamedActivityResult:
                async with request.model.request_stream(
                    request.messages, request.model_settings, request.model_request_parameters, ctx
                ) as streamed:
                    # Model events are delivered to the handler inside this memoized model unit, so
                    # the handler gets a guarded context (an enqueue would be dropped on a retry;
                    # pydantic-ai #6671).
                    events = await capture_event_stream(
                        run_context=_guard_enqueue(ctx), stream=streamed, handler=self._event_stream_handler
                    )
                return StreamedActivityResult(response=streamed.get(), events=events)

            return await self._memoize(
                kind='model_stream',
                suffix=suffix,
                fingerprint=fingerprint,
                run=run,
                dump=_dump_streamed,
                load=_load_streamed,
            )

        async def cancel_suspended_response_segment(response: ModelResponse) -> None:
            await self._memoize(
                kind='model_cancel',
                suffix=suffix,
                fingerprint=None,
                run=lambda: model.cancel_suspended_response(response),
                dump=lambda _: None,
                load=lambda _: None,
            )

        request_context.model = DurableModel(
            request_context.model,
            request_segment=request_segment,
            request_stream_segment=request_stream_segment,
            cancel_suspended_response_segment=cancel_suspended_response_segment,
        )
        return await handler(request_context)


class _Skip:
    """Sentinel: the operation's result could not be serialized, so it was not memoized."""


_SKIP = _Skip()


def _try_dump(value: _ResultT, dump: Callable[[_ResultT], JsonValue]) -> JsonValue | _Skip:
    """Serialize `value`, returning `_SKIP` if it is not JSON-round-trippable.

    Both a Pydantic serialization failure and a value that dumps but cannot be JSON-encoded
    (`json.dumps` probe) degrade to a skip, so a non-serializable result never crashes the run and
    is simply not memoized.
    """
    try:
        payload = dump(value)
    except (PydanticSerializationError, TypeError, ValueError):
        return _SKIP
    try:
        json.dumps(payload)
    except (TypeError, ValueError):  # pragma: no cover - `mode='json'` output is JSON-safe or raises above
        return _SKIP
    return payload


async def _mcp_tool_defs(toolset: MCPToolset[AgentDepsT], ctx: RunContext[AgentDepsT]) -> dict[str, ToolDefinition]:
    tools = await toolset.get_tools(ctx)
    return {name: tool.tool_def for name, tool in tools.items()}


def _dump_streamed(result: StreamedActivityResult) -> JsonValue:
    return {
        'response': _response_adapter.dump_python(result.response, mode='json'),
        'events': _events_adapter.dump_python(result.events, mode='json'),
    }


def _load_streamed(payload: JsonValue) -> StreamedActivityResult:
    assert isinstance(payload, dict)
    return StreamedActivityResult(
        response=_response_adapter.validate_python(payload['response']),
        events=_events_adapter.validate_python(payload['events']),
    )
