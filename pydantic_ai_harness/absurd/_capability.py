"""Durable execution for Pydantic AI agents on the Absurd engine.

Absurd (`absurd-sdk`) is a Postgres-based durable-execution engine. This module
checkpoints an agent's I/O -- model requests, MCP calls, and function tool calls
-- into Absurd steps (`ctx.step(...)`), so a worker crash mid-run resumes from
the last completed step instead of restarting the run.

The step names and checkpoint payload shapes are kept byte-compatible with the
`pydantic-ai-absurd` package by Marcelo Trylesinski, so a run started under one
package can resume under the other.
"""

from __future__ import annotations

try:
    import absurd_sdk  # noqa: F401  # pyright: ignore[reportUnusedImport]
except ImportError as _import_error:  # pragma: no cover
    raise ImportError(
        'Please install the `absurd-sdk` package to use the Absurd durability capability, '
        'you can use the `absurd` optional group -- `pip install "pydantic-ai-harness[absurd]"`'
    ) from _import_error

import copy
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Literal

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
    DurableFunctionToolset,
    DurableMCPToolset,
    ToolConfig,
    resolve_tool_durable_config,
    unwrap_tool_call_result,
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
from pydantic_ai.toolsets.external import TOOL_SCHEMA_VALIDATOR

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
_NO_FALLBACK_CONFIG: Mapping[str, ToolConfig] = {}

_Instructions = str | InstructionPart | Sequence[str | InstructionPart] | None

_response_adapter: TypeAdapter[ModelResponse] = TypeAdapter(ModelResponse)
_events_adapter: TypeAdapter[list[ModelResponseStreamEvent]] = TypeAdapter(list[ModelResponseStreamEvent])
_call_tool_result_adapter: TypeAdapter[CallToolResult] = TypeAdapter(CallToolResult)
_tool_defs_adapter: TypeAdapter[dict[str, ToolDefinition]] = TypeAdapter(dict[str, ToolDefinition])
_tool_def_adapter: TypeAdapter[ToolDefinition] = TypeAdapter(ToolDefinition)
_instructions_adapter: TypeAdapter[_Instructions] = TypeAdapter(_Instructions)


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
    return _call_tool_result_adapter.dump_python(result, mode='json')


def _deserialize_call_tool_result(payload: JsonValue) -> CallToolResult:
    return _call_tool_result_adapter.validate_python(payload)


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
    toolset: FunctionToolset[AgentDepsT], *, step_name_prefix: str
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

        async def _inner() -> JsonValue:
            result = await wrap_tool_call_result(toolset.call_tool(tool_name, tool_args, ctx, tool))
            return _serialize_call_tool_result(result)

        payload = await task_ctx.step(f'{name}.call_tool:{tool_name}', _inner)
        return unwrap_tool_call_result(_deserialize_call_tool_result(payload))

    return DurableFunctionToolset(
        toolset,
        in_durable_context=_in_durable_context,
        call_tool_operation=call_tool_operation,
        resolve_tool_config=_resolve_function_tool_config,
        lifecycle='enter-never',
    )


def _build_mcp_toolset(toolset: MCPToolset[AgentDepsT], *, step_name_prefix: str) -> DurableMCPToolset[AgentDepsT]:
    name = f'{step_name_prefix}__mcp_server__{toolset.id}'

    async def get_tools_operation(ctx: RunContext[AgentDepsT]) -> dict[str, ToolDefinition]:
        task_ctx = _current_async_task_context()
        assert task_ctx is not None  # pragma: no cover - only called inside a durable context

        async def _inner() -> JsonValue:
            tools = await toolset.get_tools(ctx)
            return _tool_defs_adapter.dump_python({n: t.tool_def for n, t in tools.items()}, mode='json')

        payload = await task_ctx.step(f'{name}.get_tools', _inner)
        return _tool_defs_adapter.validate_python(payload)

    async def get_instructions_operation(ctx: RunContext[AgentDepsT]) -> _Instructions:
        task_ctx = _current_async_task_context()
        assert task_ctx is not None  # pragma: no cover - only called inside a durable context

        async def _inner() -> JsonValue:
            async with toolset:
                return _instructions_adapter.dump_python(await toolset.get_instructions(ctx), mode='json')

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
            result = await wrap_tool_call_result(toolset.call_tool(tool_name, tool_args, ctx, tool))
            return _serialize_call_tool_result(result)

        payload = await task_ctx.step(f'{name}.call_tool', _inner)
        return unwrap_tool_call_result(_deserialize_call_tool_result(payload))

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


class _AbsurdDynamicToolset(WrapperToolset[AgentDepsT]):
    """Durable wrapper for a construction-time `DynamicToolset`.

    A `DynamicToolset` resolves its inner toolset lazily per run (or per run step) by calling a
    user factory, which may do I/O. Left unwrapped, that resolution and the inner tool calls run
    inline in task code and re-run on recovery. This wrapper moves them into Absurd steps, mirroring
    the shape of `pydantic_ai.durable_exec.temporal._dynamic_toolset.TemporalDynamicToolset` (there
    is no shared core primitive for durable dynamic toolsets, so, as Temporal does, the wrapper is
    engine-local): `get_tools` resolves the toolset inside one step and checkpoints its tool
    definitions and instructions; `call_tool` re-resolves inside its own step and checkpoints the
    result. On replay each step is served from its checkpoint, so the factory and the tool are not
    re-invoked.

    Outside a task the wrapper is transparent and delegates to the wrapped `DynamicToolset`.
    """

    def __init__(
        self,
        wrapped: DynamicToolset[AgentDepsT],
        *,
        step_name_prefix: str,
        in_durable_context: Callable[[], bool],
    ) -> None:
        super().__init__(wrapped)
        # `wrapped.id` may be `None` here: the base capability calls `_wrap_leaf_toolset` before it
        # rejects an id-less leaf, and the name is never used in that case (binding raises first).
        self._name = f'{step_name_prefix}__dynamic_toolset__{wrapped.id}'
        self._in_durable_context = in_durable_context
        # Set by `get_tools`, read by `get_instructions` in the same run step. Lives on the per-run
        # copy so concurrent runs of the shared bound wrapper do not clobber each other.
        self._run_instructions: _Instructions = None

    @property
    def id(self) -> str:
        assert self.wrapped.id is not None
        return self.wrapped.id

    def visit_and_replace(
        self, visitor: Callable[[AbstractToolset[AgentDepsT]], AbstractToolset[AgentDepsT]]
    ) -> AbstractToolset[AgentDepsT]:
        # A durable wrapper is a barrier: it must not be descended into and re-wrapped, so it returns
        # itself rather than the `WrapperToolset` default (which would `replace(...)` and fail this
        # keyword-only constructor). Mirrors core's `DurableToolsetBase`. No public composition
        # currently revisits an already-swapped leaf, so this is a defensive guard.
        return self  # pragma: no cover

    def _run_copy(self, wrapped: AbstractToolset[AgentDepsT]) -> _AbsurdDynamicToolset[AgentDepsT]:
        run_copy = copy.copy(self)
        run_copy.wrapped = wrapped
        run_copy._run_instructions = None
        return run_copy

    async def for_run(self, ctx: RunContext[AgentDepsT]) -> AbstractToolset[AgentDepsT]:
        if not self._in_durable_context():
            new_wrapped = await self.wrapped.for_run(ctx)
            return self if new_wrapped is self.wrapped else self._run_copy(new_wrapped)
        # Resolution is deferred into the durable `get_tools`/`call_tool` steps; only isolate the
        # per-run instructions slot from the process-shared bound wrapper.
        return self._run_copy(self.wrapped)

    async def for_run_step(self, ctx: RunContext[AgentDepsT]) -> AbstractToolset[AgentDepsT]:
        if not self._in_durable_context():
            new_wrapped = await self.wrapped.for_run_step(ctx)
            return self if new_wrapped is self.wrapped else self._run_copy(new_wrapped)
        return self

    async def get_tools(self, ctx: RunContext[AgentDepsT]) -> dict[str, ToolsetTool[AgentDepsT]]:
        if not self._in_durable_context():
            return await super().get_tools(ctx)
        task_ctx = _current_async_task_context()
        assert task_ctx is not None  # pragma: no cover - gated by `_in_durable_context`

        async def _inner() -> JsonValue:
            run_toolset = await self.wrapped.for_run(ctx)
            async with run_toolset:
                resolved = await run_toolset.for_run_step(ctx)
                tools = await resolved.get_tools(ctx)
                instructions = await resolved.get_instructions(ctx)
                return {
                    'tools': {
                        tool_name: {
                            'tool_def': _tool_def_adapter.dump_python(tool.tool_def, mode='json'),
                            'max_retries': tool.max_retries,
                        }
                        for tool_name, tool in tools.items()
                    },
                    'instructions': _instructions_adapter.dump_python(instructions, mode='json'),
                }

        payload = await task_ctx.step(f'{self._name}.get_tools', _inner)
        assert isinstance(payload, dict)
        self._run_instructions = _instructions_adapter.validate_python(payload['instructions'])
        tools_payload = payload['tools']
        assert isinstance(tools_payload, dict)
        return {tool_name: self._tool_from_payload(info) for tool_name, info in tools_payload.items()}

    def _tool_from_payload(self, info: JsonValue) -> ToolsetTool[AgentDepsT]:
        assert isinstance(info, dict)
        max_retries = info['max_retries']
        assert isinstance(max_retries, int)
        return ToolsetTool(
            toolset=self,
            tool_def=_tool_def_adapter.validate_python(info['tool_def']),
            max_retries=max_retries,
            args_validator=TOOL_SCHEMA_VALIDATOR,
        )

    async def get_instructions(self, ctx: RunContext[AgentDepsT]) -> _Instructions:
        if not self._in_durable_context():
            return await super().get_instructions(ctx)
        # Collected by `get_tools`, which the framework runs earlier in each run step.
        return self._run_instructions

    async def call_tool(
        self, name: str, tool_args: dict[str, Any], ctx: RunContext[AgentDepsT], tool: ToolsetTool[AgentDepsT]
    ) -> Any:
        if not self._in_durable_context():
            return await super().call_tool(name, tool_args, ctx, tool)
        task_ctx = _current_async_task_context()
        assert task_ctx is not None  # pragma: no cover - gated by `_in_durable_context`

        async def _inner() -> JsonValue:
            run_toolset = await self.wrapped.for_run(ctx)
            async with run_toolset:
                resolved = await run_toolset.for_run_step(ctx)
                tools = await resolved.get_tools(ctx)
                inner_tool = tools.get(name)
                if inner_tool is None:  # pragma: no cover - factory returned a different toolset
                    raise UserError(
                        f'Tool {name!r} not found in dynamic toolset {self.id!r} when re-resolving it '
                        'inside the durable step. The toolset factory returned a different toolset than '
                        'when the tool was listed.'
                    )
                result = await wrap_tool_call_result(resolved.call_tool(name, tool_args, ctx, inner_tool))
                return _serialize_call_tool_result(result)

        payload = await task_ctx.step(f'{self._name}.call_tool:{name}', _inner)
        return unwrap_tool_call_result(_deserialize_call_tool_result(payload))


def _build_dynamic_toolset(
    toolset: DynamicToolset[AgentDepsT], *, step_name_prefix: str
) -> _AbsurdDynamicToolset[AgentDepsT]:
    return _AbsurdDynamicToolset(toolset, step_name_prefix=step_name_prefix, in_durable_context=_in_durable_context)


@dataclass(init=False)
class AbsurdDurability(BaseDurabilityCapability[AgentDepsT]):
    """Capability that makes an agent durable by checkpointing its I/O into Absurd steps.

    Attach it via `capabilities=[AbsurdDurability()]` and call `agent.run()` inside an Absurd
    task handler: every model request, MCP call, function tool call, and dynamic-toolset
    resolution is wrapped in `ctx.step(...)`, so a worker crash mid-run resumes from the last
    completed step instead of
    restarting. A completed step is served from its checkpoint on replay instead of being
    recomputed, so tokens are not re-spent on work that already finished. A step is checkpointed
    after it runs, so a crash between a tool's side effect and its checkpoint re-runs the tool on
    recovery: keep tool side effects idempotent. Outside a task the capability is transparent and
    the run is a normal, non-durable agent run.

    The capability discovers the agent's model, name, and toolsets automatically when it is bound
    to the agent. Step results are stored in Postgres as JSON, so a checkpointed tool's return
    value must be JSON-serializable.

    Step names and checkpoint payload shapes are compatible with the `pydantic-ai-absurd` package,
    so a run can migrate between the two.

    Example:
        ```python {test="skip"}
        from absurd_sdk import AsyncAbsurd, AsyncTaskContext, JsonValue
        from pydantic_ai import Agent
        from pydantic_ai_harness.absurd import AbsurdDurability

        absurd = AsyncAbsurd('postgresql://localhost/absurd', queue_name='agents')
        agent = Agent('openai:gpt-5', name='analyst', capabilities=[AbsurdDurability()])


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
                Defaults to `'sequential'`. `'parallel'` is excluded by type because it emits
                tool-result and event-handler steps in completion order, which races with Absurd's
                encounter-order step naming; see `AbsurdParallelExecutionMode` for the full
                invariant. Outside a task the agent's configured mode is left untouched.
        """
        super().__init__(models=models, event_stream_handler=event_stream_handler, name=name)
        for model_id in models or {}:
            _reject_model_id_hash(model_id)
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
            return _build_function_toolset(ts, step_name_prefix=self.name)
        if isinstance(ts, DynamicToolset):
            return _build_dynamic_toolset(ts, step_name_prefix=self.name)
        try:
            from pydantic_ai.mcp import MCPToolset
        except ImportError:  # pragma: no cover - MCP wrapping only applies when the mcp extra is installed
            return None
        if isinstance(ts, MCPToolset):
            return _build_mcp_toolset(ts, step_name_prefix=self.name)
        return None

    async def _dispatch_event_stream_event(self, ctx: RunContext[AgentDepsT], event: AgentStreamEvent) -> None:
        task_ctx = _current_async_task_context()
        assert task_ctx is not None  # pragma: no cover - only dispatched inside a durable context
        handler = self._event_stream_handler
        assert handler is not None  # pragma: no cover - only dispatched when a handler is set

        async def _inner() -> None:
            await handler(ctx, self._single_event_stream(event))

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
        if agent is None or not self.in_durable_context:
            return await handler()
        with agent.parallel_tool_call_execution_mode(self._parallel_execution_mode):
            return await handler()

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
        step_suffix = '' if model_id is None else f'.{model_id}'
        model = request_context.model

        async def request_segment(request: ModelRequestContext) -> ModelResponse:
            async def _inner() -> JsonValue:
                response = await request.model.request(
                    request.messages, request.model_settings, request.model_request_parameters
                )
                return _serialize_response(response)

            payload = await task_ctx.step(f'{self.name}__model.request{step_suffix}', _inner)
            return _deserialize_response(payload)

        async def request_stream_segment(request: ModelRequestContext) -> StreamedActivityResult:
            async def _inner() -> JsonValue:
                async with request.model.request_stream(
                    request.messages, request.model_settings, request.model_request_parameters, ctx
                ) as streamed:
                    events = await capture_event_stream(
                        run_context=ctx, stream=streamed, handler=self._event_stream_handler
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
                await model.cancel_suspended_response(response)

            await task_ctx.step(f'{self.name}__model.cancel_suspended_response{step_suffix}', _inner)

        request_context.model = DurableModel(
            request_context.model,
            request_segment=request_segment,
            request_stream_segment=request_stream_segment,
            cancel_suspended_response_segment=cancel_suspended_response_segment,
        )
        return await handler(request_context)
