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

from collections.abc import Mapping, Sequence
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

if TYPE_CHECKING:
    from pydantic_ai.mcp import MCPToolset

AbsurdParallelExecutionMode = Literal['sequential', 'parallel_ordered_events']
"""Tool-call execution modes usable with Absurd. A subset of `ParallelExecutionMode`: `'parallel'`
is excluded because Absurd disambiguates repeated step names with an encounter-order counter, so
checkpoints only line up with a replay when the steps are reached in a deterministic order."""

_ENGINE_NAME = 'Absurd'
_TOOL_CONFIG_KEY = 'absurd'
_TOOL_CONFIG_LABEL = 'Absurd step config'
_NO_FALLBACK_CONFIG: Mapping[str, ToolConfig] = {}

_Instructions = str | InstructionPart | Sequence[str | InstructionPart] | None

_response_adapter: TypeAdapter[ModelResponse] = TypeAdapter(ModelResponse)
_events_adapter: TypeAdapter[list[ModelResponseStreamEvent]] = TypeAdapter(list[ModelResponseStreamEvent])
_call_tool_result_adapter: TypeAdapter[CallToolResult] = TypeAdapter(CallToolResult)
_tool_defs_adapter: TypeAdapter[dict[str, ToolDefinition]] = TypeAdapter(dict[str, ToolDefinition])
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


def _resolve_function_tool_config(tool: ToolsetTool[Any] | None, tool_name: str) -> ToolConfig:
    return resolve_tool_durable_config(
        tool, tool_name, _NO_FALLBACK_CONFIG, metadata_key=_TOOL_CONFIG_KEY, config_type_label=_TOOL_CONFIG_LABEL
    )


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
        lifecycle='enter-never',
    )


@dataclass(init=False)
class AbsurdDurability(BaseDurabilityCapability[AgentDepsT]):
    """Capability that makes an agent durable by checkpointing its I/O into Absurd steps.

    Attach it via `capabilities=[AbsurdDurability()]` and call `agent.run()` inside an Absurd
    task handler: every model request, MCP call, and function tool call is wrapped in
    `ctx.step(...)`, so a worker crash mid-run resumes from the last completed step instead of
    restarting -- no tokens are re-spent, and a tool side effect runs once. Outside a task the
    capability is transparent and the run is a normal, non-durable agent run.

    The capability discovers the agent's model, name, and toolsets automatically when it is bound
    to the agent. Step results are stored in Postgres as JSON, so a checkpointed tool's return
    value must be JSON-serializable.

    Step names and checkpoint payload shapes are compatible with the `pydantic-ai-absurd` package,
    so a run can migrate between the two.

    Example:
        ```python {test="skip"}
        from pydantic_ai import Agent
        from pydantic_ai_harness.absurd import AbsurdDurability

        agent = Agent('openai:gpt-5', name='analyst', capabilities=[AbsurdDurability()])


        @absurd.register_task(name='analyse')
        async def analyse(params, ctx):
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
            parallel_execution_mode: Tool-call execution mode applied for the duration of every
                run. Defaults to `'sequential'`. `'parallel'` is excluded by type: Absurd
                disambiguates repeated step names with an encounter-order counter, so steps must
                be reached in a deterministic order for a replay to line up with its checkpoints.
        """
        super().__init__(models=models, event_stream_handler=event_stream_handler, name=name)
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
        """Apply the configured parallel-execution mode for every entry point."""
        agent = self._agent
        if agent is None:  # pragma: no cover - `for_agent` always binds the agent before a run
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
