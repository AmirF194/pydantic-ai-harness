"""Durable execution for Pydantic AI agents on the Absurd engine.

Absurd (`absurd-sdk`) is a Postgres-based durable-execution engine. This module
checkpoints an agent's I/O -- model requests, MCP calls, and function tool calls
-- into Absurd steps (`ctx.step(...)`), so a worker crash mid-run resumes from
the last completed step instead of restarting the run.

For toolsets with explicit IDs, step names are byte-compatible with the
`pydantic-ai-absurd` package. Raw tool-result checkpoints written by that package
are accepted unchanged. New tool-result checkpoints use a versioned envelope for
control-flow wrappers, so compatibility is one-way.
"""

from __future__ import annotations

try:
    import absurd_sdk  # noqa: F401  # pyright: ignore[reportUnusedImport]
except ImportError as _import_error:  # pragma: no cover
    raise ImportError(
        'Please install the `absurd-sdk` package to use the Absurd durability capability, '
        'you can use the `absurd` optional group -- `pip install "pydantic-ai-harness[absurd]"`'
    ) from _import_error

from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from threading import Lock
from typing import TYPE_CHECKING, Any, ClassVar, Literal, get_args

from absurd_sdk import AsyncTaskContext, JsonValue, get_current_context
from pydantic import TypeAdapter
from pydantic_ai import FunctionToolset, ToolsetTool
from pydantic_ai.agent import EventStreamHandler, ParallelExecutionMode
from pydantic_ai.agent.abstract import AbstractAgent
from pydantic_ai.capabilities.abstract import WrapModelRequestHandler, WrapRunHandler
from pydantic_ai.durable_exec._base import BaseDurabilityCapability
from pydantic_ai.durable_exec._runtime_toolsets import RuntimeToolsetKind
from pydantic_ai.durable_exec._toolset import (
    CallToolResult,
    DurableDynamicToolset,
    DurableFunctionToolset,
    DurableMCPToolset,
    DynamicToolsResult,
    ToolConfig,
    call_dynamic_tool,
    get_dynamic_tools,
    resolve_tool_durable_config,
    unwrap_recorded_tool_call_result,
    wrap_tool_call_result,
)
from pydantic_ai.durable_exec._utils import DurableModel, StreamedActivityResult, capture_event_stream
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import AgentStreamEvent, InstructionPart, ModelResponse, ModelResponseStreamEvent
from pydantic_ai.models import Model, ModelRequestContext
from pydantic_ai.run import AgentRunResult
from pydantic_ai.tools import AgentDepsT, RunContext, ToolDefinition
from pydantic_ai.toolsets import AbstractToolset, WrapperToolset
from pydantic_ai.toolsets._dynamic import DynamicToolset

if TYPE_CHECKING:
    from pydantic_ai.mcp import MCPToolset

AbsurdParallelExecutionMode = Literal['sequential', 'parallel_ordered_events']
"""Tool-call execution modes usable with Absurd. A subset of `ParallelExecutionMode`.

Absurd disambiguates repeated step names by encounter order (the second `ctx.step(name, ...)` for a
given `name` records under `name#2`, the third under `name#3`, ...). A replay lines up with its
checkpoints only if each repeated step name claims the same slot it did on the first run.

The slot is claimed synchronously when `ctx.step(...)` is entered, before the step body runs. Tool
calls are scheduled in the model's tool-call order under both parallel modes, so their step names are
assigned in that order regardless of which call finishes first -- completion order does not move a
tool call's slot. `'parallel'` is nonetheless excluded because it emits tool-result events (and so
the per-event `event_stream_handler` steps) in completion order, which races and could assign one of
those repeated step names a different slot on replay. `'parallel_ordered_events'` emits those events
in the model's tool-call order once the whole batch completes, so every repeated step name -- tool
calls and event-handler steps alike -- lines up on replay."""

_ENGINE_NAME = 'Absurd'
_TOOL_CONFIG_KEY = 'absurd'
_TOOL_CONFIG_LABEL = 'Absurd step config'
_CALL_TOOL_RESULT_ENVELOPE_KEY = '__pydantic_ai_harness_call_tool_result__'
_CALL_TOOL_RESULT_ENVELOPE_VERSION = 1
_NO_FALLBACK_CONFIG: Mapping[str, ToolConfig] = {}
_ALLOWED_PARALLEL_EXECUTION_MODES: tuple[str, ...] = get_args(AbsurdParallelExecutionMode)

_Instructions = str | InstructionPart | Sequence[str | InstructionPart] | None
_DurableRunContextScope = Callable[[RunContext[AgentDepsT]], AbstractContextManager[RunContext[AgentDepsT]]]

_response_adapter: TypeAdapter[ModelResponse] = TypeAdapter(ModelResponse)
_events_adapter: TypeAdapter[list[ModelResponseStreamEvent]] = TypeAdapter(list[ModelResponseStreamEvent])
_call_tool_result_adapter: TypeAdapter[CallToolResult] = TypeAdapter(CallToolResult)
_tool_defs_adapter: TypeAdapter[dict[str, ToolDefinition]] = TypeAdapter(dict[str, ToolDefinition])
_instructions_adapter: TypeAdapter[_Instructions] = TypeAdapter(_Instructions)
_dynamic_tools_adapter: TypeAdapter[DynamicToolsResult] = TypeAdapter(DynamicToolsResult)

# Absurd disambiguates repeated step names by encounter order within one task. Two overlapping
# runs using the same task context and namespace would therefore claim each other's checkpoints.
# Keep the active context itself as the value so an object-id reuse cannot release a newer claim.
_active_run_namespaces: dict[tuple[int, str], AsyncTaskContext] = {}
_active_run_namespaces_lock = Lock()


def _current_async_task_context() -> AsyncTaskContext | None:
    """Return the active Absurd async task context, or `None` outside a task.

    A synchronous `TaskContext` raises `UserError`: an agent run is async and a sync task
    cannot await it, so the run cannot be checkpointed from one.
    """
    ctx = get_current_context()
    if ctx is None:
        return None
    if isinstance(ctx, AsyncTaskContext):
        return ctx
    raise UserError(
        f'{_ENGINE_NAME} durability requires an async Absurd task context, but a synchronous '
        '`TaskContext` is active. Agent runs are async, so run the agent from an async task '
        'handler (`AsyncTaskContext`), not a synchronous one.'
    )


def _in_durable_context() -> bool:
    return _current_async_task_context() is not None


def _serialize_response(response: ModelResponse) -> JsonValue:
    return _response_adapter.dump_python(response, mode='json')


def _deserialize_response(payload: JsonValue) -> ModelResponse:
    return _response_adapter.validate_python(payload)


def _serialize_call_tool_result(result: CallToolResult) -> JsonValue:
    # The outer marker distinguishes this wrapper from a legacy raw dictionary whose `kind` and
    # `result` fields happen to match one of the wrapper dataclasses.
    return {
        _CALL_TOOL_RESULT_ENVELOPE_KEY: {
            'version': _CALL_TOOL_RESULT_ENVELOPE_VERSION,
            'result': _call_tool_result_adapter.dump_python(result, mode='json'),
        }
    }


def _unwrap_checkpointed_call_tool_result(payload: JsonValue) -> Any:
    """Unwrap a current checkpoint, passing any un-enveloped legacy recording through unchanged.

    Older Absurd integrations stored the raw tool result directly. The outer envelope is required
    because a raw dictionary such as `{'kind': 'tool_return', 'result': 'x'}` is otherwise
    indistinguishable from a serialized `CallToolResult` wrapper.
    """
    if not isinstance(payload, dict):
        return payload
    if _CALL_TOOL_RESULT_ENVELOPE_KEY not in payload:
        return payload
    envelope = payload.get(_CALL_TOOL_RESULT_ENVELOPE_KEY)
    if not isinstance(envelope, dict):
        raise UserError('Malformed Absurd tool-result checkpoint envelope: expected an object.')
    version = envelope.get('version')
    if type(version) is not int or version != _CALL_TOOL_RESULT_ENVELOPE_VERSION:
        raise UserError(f'Unsupported Absurd tool-result checkpoint envelope version: {version!r}.')
    result = _call_tool_result_adapter.validate_python(envelope.get('result'))
    return unwrap_recorded_tool_call_result(result)


def _reject_model_id_hash(model_id: str) -> None:
    """Reject a model id containing `#`.

    A request's model id is folded into the model step name as a `.{model_id}` suffix, and Absurd
    disambiguates repeated step names by appending `#2`, `#3`, ... So a model id containing `#`
    would collide with that counter suffix (e.g. id `cheap#2` resolves to the `#2` checkpoint of
    model `cheap`). Byte-compatibility with `pydantic-ai-absurd` forbids escaping the id (it would
    change the step name for every such id, and the reference package has the same collision), so
    the id is rejected instead.
    """
    if '#' in model_id:
        raise UserError(
            f'Model id {model_id!r} contains {"#"!r}, which Absurd uses to disambiguate repeated '
            'step names. Folded into the model step name it would collide with that suffix. Choose '
            'a model id without `#`.'
        )


def _reject_tool_name_hash(tool_name: str) -> None:
    """Reject a tool name containing `#`.

    Same collision as `_reject_model_id_hash`: function-tool and dynamic-toolset calls are
    checkpointed under a `.call_tool:{tool_name}` step name, so a tool named `foo#2` would land on
    the checkpoint Absurd assigns to the second call of a tool named `foo`.
    """
    if '#' in tool_name:
        raise UserError(
            f'Tool name {tool_name!r} contains {"#"!r}, which Absurd uses to disambiguate repeated '
            'step names. Folded into the tool call step name it would collide with that suffix. '
            'Rename the tool without `#`.'
        )


def _reject_step_options(config: ToolConfig, tool_name: str) -> None:
    """Reject a non-empty `absurd` config mapping.

    `ctx.step(...)` takes no per-call options, so a mapping under the `absurd` metadata key has
    nothing to apply. Only `False` (inline opt-out) is meaningful, so a populated mapping is a
    mistake rather than something quietly dropped -- Temporal takes the same stance for metadata it
    cannot use.
    """
    if config is not False and config:
        raise UserError(
            f'Tool {tool_name!r} sets a non-empty {_TOOL_CONFIG_KEY!r} step config, but Absurd steps '
            f'take no per-tool options, so the config would have no effect. Only '
            f'metadata={{{_TOOL_CONFIG_KEY!r}: False}} (run the tool inline, uncheckpointed) is '
            'supported; remove the config.'
        )


def _resolve_function_tool_config(tool: ToolsetTool[Any] | None, tool_name: str) -> ToolConfig:
    config = resolve_tool_durable_config(
        tool, tool_name, _NO_FALLBACK_CONFIG, metadata_key=_TOOL_CONFIG_KEY, config_type_label=_TOOL_CONFIG_LABEL
    )
    _reject_step_options(config, tool_name)
    return config


def _resolve_mcp_tool_config(tool: ToolsetTool[Any] | None, tool_name: str) -> ToolConfig:
    config = resolve_tool_durable_config(
        tool, tool_name, _NO_FALLBACK_CONFIG, metadata_key=_TOOL_CONFIG_KEY, config_type_label=_TOOL_CONFIG_LABEL
    )
    if config is False:
        raise UserError(
            f'Absurd checkpointing for MCP tool {tool_name!r} was disabled with '
            f'metadata={{{_TOOL_CONFIG_KEY!r}: False}}, but MCP tools perform I/O and so cannot run '
            'outside a step. Remove the metadata so the call stays checkpointed.'
        )
    _reject_step_options(config, tool_name)
    return config


def _build_function_toolset(
    toolset: FunctionToolset[AgentDepsT],
    *,
    step_name_prefix: str,
    durable_run_context_scope: _DurableRunContextScope[AgentDepsT],
) -> DurableFunctionToolset[AgentDepsT]:
    name = f'{step_name_prefix}__function_toolset__{toolset.id}'

    async def call_tool_operation(
        tool_name: str,
        tool_args: dict[str, Any],
        ctx: RunContext[AgentDepsT],
        tool: ToolsetTool[AgentDepsT],
        config: Mapping[str, Any],
    ) -> Any:
        task_ctx = _current_async_task_context()
        assert task_ctx is not None  # pragma: no cover - only called inside a durable context
        _reject_tool_name_hash(tool_name)

        async def _inner() -> JsonValue:
            with durable_run_context_scope(ctx) as step_ctx:
                result = await wrap_tool_call_result(toolset.call_tool(tool_name, tool_args, step_ctx, tool))
            return _serialize_call_tool_result(result)

        payload = await task_ctx.step(f'{name}.call_tool:{tool_name}', _inner)
        return _unwrap_checkpointed_call_tool_result(payload)

    return DurableFunctionToolset(
        toolset,
        in_durable_context=_in_durable_context,
        call_tool_operation=call_tool_operation,
        resolve_tool_config=_resolve_function_tool_config,
        # Entering a `FunctionToolset` is a no-op -- it holds no connection or session -- so there
        # is nothing to keep open for the run.
        lifecycle='enter-never',
    )


def _build_mcp_toolset(
    toolset: MCPToolset[AgentDepsT],
    *,
    step_name_prefix: str,
    durable_run_context_scope: _DurableRunContextScope[AgentDepsT],
) -> DurableMCPToolset[AgentDepsT]:
    name = f'{step_name_prefix}__mcp_server__{toolset.id}'

    async def get_tools_operation(ctx: RunContext[AgentDepsT]) -> dict[str, ToolDefinition]:
        task_ctx = _current_async_task_context()
        assert task_ctx is not None  # pragma: no cover - only called inside a durable context

        async def _inner() -> JsonValue:
            with durable_run_context_scope(ctx) as step_ctx:
                tools = await toolset.get_tools(step_ctx)
            return _tool_defs_adapter.dump_python({n: t.tool_def for n, t in tools.items()}, mode='json')

        payload = await task_ctx.step(f'{name}.get_tools', _inner)
        return _tool_defs_adapter.validate_python(payload)

    async def get_instructions_operation(ctx: RunContext[AgentDepsT]) -> _Instructions:
        task_ctx = _current_async_task_context()
        assert task_ctx is not None  # pragma: no cover - only called inside a durable context

        async def _inner() -> JsonValue:
            with durable_run_context_scope(ctx) as step_ctx:
                return _instructions_adapter.dump_python(await toolset.get_instructions(step_ctx), mode='json')

        payload = await task_ctx.step(f'{name}.get_instructions', _inner)
        return _instructions_adapter.validate_python(payload)

    async def call_tool_operation(
        tool_name: str,
        tool_args: dict[str, Any],
        ctx: RunContext[AgentDepsT],
        tool: ToolsetTool[AgentDepsT],
        config: Mapping[str, Any],
    ) -> Any:
        task_ctx = _current_async_task_context()
        assert task_ctx is not None  # pragma: no cover - only called inside a durable context

        async def _inner() -> JsonValue:
            with durable_run_context_scope(ctx) as step_ctx:
                result = await wrap_tool_call_result(toolset.call_tool(tool_name, tool_args, step_ctx, tool))
            return _serialize_call_tool_result(result)

        payload = await task_ctx.step(f'{name}.call_tool', _inner)
        return _unwrap_checkpointed_call_tool_result(payload)

    return DurableMCPToolset(
        toolset,
        in_durable_context=_in_durable_context,
        get_tools_operation=get_tools_operation,
        get_instructions_operation=get_instructions_operation,
        call_tool_operation=call_tool_operation,
        resolve_tool_config=_resolve_mcp_tool_config,
        # Enter the wrapped server for the run's duration (the analog for a gated in-process engine,
        # as in Prefect). With `'enter-never'` the wrapper's `__aenter__` is a no-op, so outside a
        # task every `get_tools`/`call_tool` opens its own implicit session, and inside a task each
        # call pays a fresh `initialize`. Entering here keeps one session for a stateful server. The
        # checkpointed operations and their step names are unchanged, so this does not affect the
        # persistence format.
        lifecycle='enter-always',
    )


def _build_dynamic_toolset(
    toolset: DynamicToolset[AgentDepsT],
    *,
    step_name_prefix: str,
    durable_run_context_scope: _DurableRunContextScope[AgentDepsT],
) -> DurableDynamicToolset[AgentDepsT]:
    name = f'{step_name_prefix}__dynamic_toolset__{toolset.id}'

    async def get_tools_operation(ctx: RunContext[AgentDepsT]) -> DynamicToolsResult:
        task_ctx = _current_async_task_context()
        assert task_ctx is not None  # pragma: no cover - only called inside a durable context

        async def _inner() -> JsonValue:
            with durable_run_context_scope(ctx) as step_ctx:
                result = await get_dynamic_tools(toolset, step_ctx)
            return _dynamic_tools_adapter.dump_python(result, mode='json')

        payload = await task_ctx.step(f'{name}.get_tools', _inner)
        return _dynamic_tools_adapter.validate_python(payload)

    async def call_tool_operation(
        tool_name: str,
        tool_args: dict[str, Any],
        ctx: RunContext[AgentDepsT],
        tool: ToolsetTool[AgentDepsT],
        config: Mapping[str, Any],
    ) -> Any:
        task_ctx = _current_async_task_context()
        assert task_ctx is not None  # pragma: no cover - only called inside a durable context
        _reject_tool_name_hash(tool_name)

        async def _inner() -> JsonValue:
            # `call_dynamic_tool` re-validates the args against the re-resolved tool, and
            # `wrap_tool_call_result` only captures control-flow exceptions, so a `ValidationError`
            # from that re-validation propagates out of the step uncaught and `ToolManager` converts
            # it to a retry prompt. The failing step still claims its checkpoint slot -- Absurd
            # claims it when the step is entered, before the body runs -- which is safe because the
            # failure is deterministic: a replay reaches the same step, fails the same way, and
            # claims the same slot, so every later step name lines up with its checkpoint.
            with durable_run_context_scope(ctx) as step_ctx:
                result = await wrap_tool_call_result(call_dynamic_tool(toolset, tool_name, tool_args, step_ctx))
            return _serialize_call_tool_result(result)

        payload = await task_ctx.step(f'{name}.call_tool:{tool_name}', _inner)
        return _unwrap_checkpointed_call_tool_result(payload)

    return DurableDynamicToolset(
        toolset,
        in_durable_context=_in_durable_context,
        get_tools_operation=get_tools_operation,
        call_tool_operation=call_tool_operation,
        resolve_tool_config=_resolve_function_tool_config,
        # The wrapped toolset is only a factory, so there is nothing to enter for the run: each
        # step enters the toolset it resolves.
        lifecycle='enter-never',
    )


@dataclass(init=False)
class AbsurdDurability(BaseDurabilityCapability[AgentDepsT]):
    """Capability that makes an agent durable by checkpointing its I/O into Absurd steps.

    Attach it via `capabilities=[AbsurdDurability()]` and call `agent.run()` inside an Absurd
    task handler: every model request, MCP call, function tool call, and dynamic-toolset
    resolution is wrapped in `ctx.step(...)`, so a worker crash mid-run resumes from the last
    completed step instead of restarting. A completed step is served from its checkpoint on replay
    instead of being recomputed, so tokens are not re-spent on work that already finished. A step
    is checkpointed after it runs, so a crash between a tool's side effect and its checkpoint
    re-runs the tool on recovery: keep tool side effects idempotent. Outside a task the capability
    is transparent and the run is a normal, non-durable agent run.

    The capability discovers the agent's model, name, and toolsets automatically when it is bound
    to the agent. Step results are stored in Postgres as JSON, so a checkpointed tool's return
    value must be JSON-serializable.

    For toolsets with explicit IDs, step names are compatible with the `pydantic-ai-absurd`
    package. A run started under that package can resume here because raw tool-result checkpoints
    are accepted unchanged. New tool-result checkpoints use a versioned envelope for control-flow
    wrappers, so a run started here should not be resumed under the standalone package.

    Example:
        ```python {test="skip"}
        from absurd_sdk import AsyncAbsurd, AsyncTaskContext, JsonValue
        from pydantic_ai import Agent
        from pydantic_ai_harness import AbsurdDurability

        absurd = AsyncAbsurd('postgresql://localhost/absurd', queue_name='agents')
        agent = Agent('openai:gpt-5.6-sol', name='analyst', capabilities=[AbsurdDurability()])


        @absurd.register_task(name='analyse')
        async def analyse(params: JsonValue, ctx: AsyncTaskContext) -> JsonValue:
            assert isinstance(params, dict)
            result = await agent.run(params['prompt'])
            return {'output': result.output}
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
        models: Mapping[str, Model] | None = None,
        event_stream_handler: EventStreamHandler[AgentDepsT] | None = None,
        name: str | None = None,
        parallel_execution_mode: AbsurdParallelExecutionMode = 'sequential',
    ) -> None:
        """Create an `AbsurdDurability` capability.

        The agent's model, name, and toolsets are discovered automatically.

        Args:
            models: Optional additional models keyed by ID for runtime model switching via
                `agent.run(model='<id>')`. The agent's primary model is always registered as
                `'default'`; the ID is folded into the checkpoint step name so a replay resolves
                to the same model.
            event_stream_handler: Optional event stream handler. Model events are handled live
                inside the model-request step; each tool event is handled in its own checkpointed
                step.
            name: Unique agent name used as the prefix for every checkpoint step. Defaults to the
                agent's `name` when the capability is bound.
            parallel_execution_mode: Tool-call execution mode applied for a run inside a task.
                Defaults to `'sequential'`. `'parallel'` is excluded by type and rejected here,
                because it emits tool-result and event-handler steps in completion order, which
                races with Absurd's encounter-order step naming; see
                `AbsurdParallelExecutionMode` for the full invariant. Outside a task the agent's
                configured mode is left untouched.
        """
        super().__init__(models=models, event_stream_handler=event_stream_handler, name=name)
        for model_id in models or {}:
            _reject_model_id_hash(model_id)
        # The type excludes `'parallel'`, but an untyped caller can still reach the mode whose
        # completion-order steps race with encounter-order step naming.
        if parallel_execution_mode not in _ALLOWED_PARALLEL_EXECUTION_MODES:
            allowed = ' or '.join(repr(mode) for mode in _ALLOWED_PARALLEL_EXECUTION_MODES)
            raise UserError(
                f'{parallel_execution_mode!r} is not a supported {_ENGINE_NAME} parallel execution mode. Use {allowed}.'
            )
        self._parallel_execution_mode: ParallelExecutionMode = parallel_execution_mode
        self._default_model_id: str | None = None

    @property
    def in_durable_context(self) -> bool:
        return _in_durable_context()

    def _bind_to_agent(self, agent: AbstractAgent[AgentDepsT, Any]) -> None:
        # Absurd steps are ad-hoc `ctx.step(...)` calls, so unlike Temporal there is nothing to
        # register up front beyond the durable toolset wrappers, which `_register_toolsets` builds
        # and indexes by toolset `id`. Recording the string default lets a request that carries the
        # raw default string as provenance checkpoint under the suffix-less step name.
        self._default_model_id = agent.model if isinstance(agent.model, str) else None
        self._register_toolsets(agent)

    def _wrap_leaf_toolset(self, ts: AbstractToolset[AgentDepsT]) -> WrapperToolset[AgentDepsT] | None:
        if isinstance(ts, FunctionToolset):
            return _build_function_toolset(
                ts, step_name_prefix=self.name, durable_run_context_scope=self._durable_run_context_scope
            )
        if isinstance(ts, DynamicToolset):
            return _build_dynamic_toolset(
                ts, step_name_prefix=self.name, durable_run_context_scope=self._durable_run_context_scope
            )
        try:
            from pydantic_ai.mcp import MCPToolset
        except ImportError:  # pragma: no cover - MCP wrapping only applies when the mcp extra is installed
            return None
        if isinstance(ts, MCPToolset):
            return _build_mcp_toolset(
                ts, step_name_prefix=self.name, durable_run_context_scope=self._durable_run_context_scope
            )
        return None

    async def _dispatch_event_stream_event(self, ctx: RunContext[AgentDepsT], event: AgentStreamEvent) -> None:
        task_ctx = _current_async_task_context()
        assert task_ctx is not None  # pragma: no cover - only dispatched inside a durable context
        handler = self._event_stream_handler
        assert handler is not None  # pragma: no cover - only dispatched when a handler is set

        async def _inner() -> None:
            with self._durable_run_context_scope(ctx) as step_ctx:
                await handler(step_ctx, self._single_event_stream(event))

        # Checkpoint the handler call so its side effects do not re-run on recovery.
        await task_ctx.step(f'{self.name}__event_stream_handler', _inner)

    async def wrap_run(self, ctx: RunContext[AgentDepsT], *, handler: WrapRunHandler) -> AgentRunResult[Any]:
        """Apply the configured parallel-execution mode for a run inside a task.

        The mode only matters for checkpoint-name determinism inside a task, so outside one the
        capability stays transparent and leaves the agent's configured mode untouched. (Unlike an
        out-of-process engine, whose override is harmless off the durable path, forcing a mode here
        would change ordinary non-durable runs.)
        """
        agent = self._agent
        task_ctx = _current_async_task_context()
        if agent is None or task_ctx is None:
            return await handler()

        namespace_key = (id(task_ctx), self.name)
        with _active_run_namespaces_lock:
            if namespace_key in _active_run_namespaces:
                raise UserError(
                    f'Concurrent Absurd agent runs with checkpoint namespace {self.name!r} are not supported '
                    'in the same task context. Await one run before starting another, or use a distinct '
                    '`AbsurdDurability(name=...)` value or task context.'
                )
            _active_run_namespaces[namespace_key] = task_ctx
        try:
            with agent.parallel_tool_call_execution_mode(self._parallel_execution_mode):
                return await handler()
        finally:
            with _active_run_namespaces_lock:
                if _active_run_namespaces.get(namespace_key) is task_ctx:
                    del _active_run_namespaces[namespace_key]

    async def wrap_model_request(
        self, ctx: RunContext[AgentDepsT], *, request_context: ModelRequestContext, handler: WrapModelRequestHandler
    ) -> ModelResponse:
        """Checkpoint model requests into Absurd steps when inside a task."""
        task_ctx = _current_async_task_context()
        if task_ctx is None:
            return await handler(request_context)

        # The step runs in-process, so the model needs no cross-boundary rebuild; the model id is
        # folded into the step name so a replay maps each checkpoint back to the model it was
        # recorded for. A string default carries itself as provenance but was checkpointed without
        # a suffix, so suppress the suffix for it.
        model_id = self._model_id_for_request(ctx, request_context)
        if model_id is not None and model_id == self._default_model_id:
            model_id = None
        if model_id is not None:
            # A runtime `agent.run(model=...)` string never passed through the `models=` check.
            _reject_model_id_hash(model_id)
        step_suffix = '' if model_id is None else f'.{model_id}'
        model = request_context.model

        async def request_segment(request: ModelRequestContext) -> ModelResponse:
            async def _inner() -> JsonValue:
                with self._durable_run_context_scope(ctx):
                    response = await request.model.request(
                        request.messages, request.model_settings, request.model_request_parameters
                    )
                return _serialize_response(response)

            payload = await task_ctx.step(f'{self.name}__model.request{step_suffix}', _inner)
            return _deserialize_response(payload)

        async def request_stream_segment(request: ModelRequestContext) -> StreamedActivityResult:
            async def _inner() -> JsonValue:
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

            payload = await task_ctx.step(f'{self.name}__model.request_stream{step_suffix}', _inner)
            assert isinstance(payload, dict)
            return StreamedActivityResult(
                response=_deserialize_response(payload['response']),
                events=_events_adapter.validate_python(payload['events']),
            )

        async def cancel_suspended_response_segment(response: ModelResponse) -> None:
            async def _inner() -> None:
                with self._durable_run_context_scope(ctx):
                    await model.cancel_suspended_response(response)

            await task_ctx.step(f'{self.name}__model.cancel_suspended_response{step_suffix}', _inner)

        request_context.model = DurableModel(
            request_context.model,
            request_segment=request_segment,
            request_stream_segment=request_stream_segment,
            cancel_suspended_response_segment=cancel_suspended_response_segment,
        )
        return await handler(request_context)
