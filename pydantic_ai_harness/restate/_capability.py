"""Durable execution for Pydantic AI agents on the Restate engine.

Restate (`restate-sdk`) is a durable-execution engine. This module journals an agent's I/O --
model requests, MCP calls, function tool calls, and dynamic-toolset resolution -- into Restate
run steps (`ctx.run_typed(...)`), so a handler that crashes or is retried mid-run replays from
the journal instead of repeating the work.

Restate's durable primitive is async and its journal is positional: `ctx.run_typed(name, fn, ...)`
records `fn`'s result by encounter order, and on replay serves the recorded bytes without
re-running `fn`. The `name` is a label, not the journal identity. An agent run is async, so it
drives the primitive directly with no thread bridge. Outside a Restate context the capability is
transparent and the run is an ordinary, non-durable agent run.
"""

from __future__ import annotations

try:
    import restate  # noqa: F401  # pyright: ignore[reportUnusedImport]
except ImportError as _import_error:  # pragma: no cover
    raise ImportError(
        'Please install the `restate-sdk` package to use the Restate durability capability, '
        'you can use the `restate` optional group -- `pip install "pydantic-ai-harness[restate]"`'
    ) from _import_error

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, ClassVar

from pydantic import TypeAdapter
from pydantic_ai import FunctionToolset, ToolsetTool
from pydantic_ai._enqueue import PendingMessage  # pyright: ignore[reportPrivateUsage]
from pydantic_ai.agent import EventStreamHandler
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
    EnqueueGuard,
    ToolConfig,
    call_dynamic_tool,
    get_dynamic_tools,
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
from pydantic_ai.toolsets._dynamic import DynamicToolset  # pyright: ignore[reportPrivateUsage]
from restate.context import Context, RunOptions
from restate.serde import JsonSerde
from restate.server_context import current_context

if TYPE_CHECKING:
    from pydantic_ai.mcp import MCPToolset

_ENGINE_NAME = 'Restate'
_TOOL_CONFIG_KEY = 'restate'
_TOOL_CONFIG_LABEL = 'Restate step config'
_NO_FALLBACK_CONFIG: Mapping[str, ToolConfig] = {}

_Instructions = str | InstructionPart | Sequence[str | InstructionPart] | None
# Journaled values are reduced to JSON-able Python via the `TypeAdapter`s below, then written to
# the Restate journal as bytes by `JsonSerde`. `Any` is the pragmatic type for a round-tripped
# JSON value, as it is for the sibling durability engines.
_JsonValue = Any

_response_adapter: TypeAdapter[ModelResponse] = TypeAdapter(ModelResponse)
_events_adapter: TypeAdapter[list[ModelResponseStreamEvent]] = TypeAdapter(list[ModelResponseStreamEvent])
_call_tool_result_adapter: TypeAdapter[CallToolResult] = TypeAdapter(CallToolResult)
_tool_defs_adapter: TypeAdapter[dict[str, ToolDefinition]] = TypeAdapter(dict[str, ToolDefinition])
_dynamic_tools_adapter: TypeAdapter[DynamicToolsResult] = TypeAdapter(DynamicToolsResult)
_instructions_adapter: TypeAdapter[_Instructions] = TypeAdapter(_Instructions)

# The journal stores bytes. Each operation reduces its result to a JSON-able value first, so a
# single JSON serde round-trips every step. A non-serializable payload fails at the `TypeAdapter`
# dump inside the operation, before it reaches the journal, the same in production as in tests.
_JSON_SERDE: JsonSerde[_JsonValue] = JsonSerde()
_RUN_OPTIONS: RunOptions[_JsonValue] = RunOptions(serde=_JSON_SERDE)


def _current_restate_context() -> Context | None:
    """Return the active Restate context, or `None` outside a Restate invocation.

    `restate.server_context.current_context()` reads a `ContextVar` that has no default, so it
    raises `LookupError` outside an invocation rather than returning `None`. Catch it here so the
    capability is transparent off the durable path.
    """
    try:
        return current_context()
    except LookupError:
        return None


def _in_durable_context() -> bool:
    return _current_restate_context() is not None


def _require_context() -> Context:
    ctx = _current_restate_context()
    assert ctx is not None  # pragma: no cover - callers gate on `_in_durable_context`
    return ctx


def _guard_enqueue(ctx: RunContext[AgentDepsT]) -> RunContext[AgentDepsT]:
    """Make `ctx.enqueue()` raise inside any code that runs within a journaled step.

    A replay serves the recorded journal entry without re-running the step, so messages enqueued
    inside it would be dropped. This covers tool calls and both event-stream delivery paths (model
    events handled inside the model-request step, and agent events dispatched in their own step),
    each of which hands a run context to user code.

    This guards only the passed context. pydantic-ai 2.16.0 adds a shared
    `_durable_run_context`/`_durable_run_context_scope` helper that guards both the passed and the
    ambient context from one place and derives the message from `_durable_unit_noun`/
    `_durable_container_noun`; collapse this onto it in the 2.16.0 sweep.
    """
    pending: list[PendingMessage] = EnqueueGuard(
        '`ctx.enqueue()` is not supported inside a Restate journaled step because a replay serves '
        'the recorded step output and would drop the enqueued messages. Enqueue messages from '
        'handler-level code instead.'
    )
    return replace(ctx, pending_messages=pending)


def _serialize_response(response: ModelResponse) -> _JsonValue:
    return _response_adapter.dump_python(response, mode='json')


def _deserialize_response(payload: _JsonValue) -> ModelResponse:
    return _response_adapter.validate_python(payload)


def _serialize_call_tool_result(result: CallToolResult) -> _JsonValue:
    return _call_tool_result_adapter.dump_python(result, mode='json')


def _deserialize_call_tool_result(payload: _JsonValue) -> CallToolResult:
    return _call_tool_result_adapter.validate_python(payload)


def _reject_step_options(config: ToolConfig, tool_name: str) -> None:
    """Reject a non-empty `restate` config mapping.

    The capability journals every step through one shared `RunOptions`, so a mapping under the
    `restate` metadata key has nothing to apply. Only `False` (inline opt-out) is meaningful, so a
    populated mapping is a mistake rather than something quietly dropped.
    """
    if config is not False and config:
        raise UserError(
            f'Tool {tool_name!r} sets a non-empty {_TOOL_CONFIG_KEY!r} step config, but Restate '
            'run steps take no per-tool options here, so the config would have no effect. Only '
            f'metadata={{{_TOOL_CONFIG_KEY!r}: False}} (run the tool inline, unjournaled) is '
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
            f'Restate journaling for MCP tool {tool_name!r} was disabled with '
            f'metadata={{{_TOOL_CONFIG_KEY!r}: False}}, but MCP tools perform I/O and so cannot run '
            'outside a step. Remove the metadata so the call stays journaled.'
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
        async def _inner() -> _JsonValue:
            result = await wrap_tool_call_result(toolset.call_tool(tool_name, tool_args, _guard_enqueue(ctx), tool))
            return _serialize_call_tool_result(result)

        payload = await _require_context().run_typed(f'{name}.call_tool:{tool_name}', _inner, _RUN_OPTIONS)
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
        async def _inner() -> _JsonValue:
            tools = await toolset.get_tools(ctx)
            return _tool_defs_adapter.dump_python({n: t.tool_def for n, t in tools.items()}, mode='json')

        payload = await _require_context().run_typed(f'{name}.get_tools', _inner, _RUN_OPTIONS)
        return _tool_defs_adapter.validate_python(payload)

    async def get_instructions_operation(ctx: RunContext[AgentDepsT]) -> _Instructions:
        async def _inner() -> _JsonValue:
            return _instructions_adapter.dump_python(await toolset.get_instructions(ctx), mode='json')

        payload = await _require_context().run_typed(f'{name}.get_instructions', _inner, _RUN_OPTIONS)
        return _instructions_adapter.validate_python(payload)

    async def call_tool_operation(
        tool_name: str,
        tool_args: dict[str, Any],
        ctx: RunContext[AgentDepsT],
        tool: ToolsetTool[AgentDepsT],
        config: Mapping[str, Any],
    ) -> Any:
        async def _inner() -> _JsonValue:
            result = await wrap_tool_call_result(toolset.call_tool(tool_name, tool_args, _guard_enqueue(ctx), tool))
            return _serialize_call_tool_result(result)

        payload = await _require_context().run_typed(f'{name}.call_tool:{tool_name}', _inner, _RUN_OPTIONS)
        return unwrap_tool_call_result(_deserialize_call_tool_result(payload))

    return DurableMCPToolset(
        toolset,
        in_durable_context=_in_durable_context,
        get_tools_operation=get_tools_operation,
        get_instructions_operation=get_instructions_operation,
        call_tool_operation=call_tool_operation,
        resolve_tool_config=_resolve_mcp_tool_config,
        # Keep one session for the run rather than an implicit session per call, which is what a
        # plain (non-durable) run does. The journaled operations and their step names are
        # unchanged, so this does not affect the persistence format.
        lifecycle='enter-always',
    )


def _build_dynamic_toolset(
    toolset: DynamicToolset[AgentDepsT], *, step_name_prefix: str
) -> DurableDynamicToolset[AgentDepsT]:
    name = f'{step_name_prefix}__dynamic_toolset__{toolset.id}'

    async def get_tools_operation(ctx: RunContext[AgentDepsT]) -> DynamicToolsResult:
        async def _inner() -> _JsonValue:
            return _dynamic_tools_adapter.dump_python(await get_dynamic_tools(toolset, ctx), mode='json')

        payload = await _require_context().run_typed(f'{name}.get_tools', _inner, _RUN_OPTIONS)
        return _dynamic_tools_adapter.validate_python(payload)

    async def call_tool_operation(
        tool_name: str,
        tool_args: dict[str, Any],
        ctx: RunContext[AgentDepsT],
        tool: ToolsetTool[AgentDepsT],
        config: Mapping[str, Any],
    ) -> Any:
        async def _inner() -> _JsonValue:
            # Arg re-validation happens inside `call_dynamic_tool`: a `ValidationError` from it is
            # deterministic, so letting it leave the step unjournaled is safe, and `ToolManager`
            # turns it into a retry prompt.
            result = await wrap_tool_call_result(call_dynamic_tool(toolset, tool_name, tool_args, _guard_enqueue(ctx)))
            return _serialize_call_tool_result(result)

        payload = await _require_context().run_typed(f'{name}.call_tool:{tool_name}', _inner, _RUN_OPTIONS)
        return unwrap_tool_call_result(_deserialize_call_tool_result(payload))

    return DurableDynamicToolset(
        toolset,
        in_durable_context=_in_durable_context,
        get_tools_operation=get_tools_operation,
        call_tool_operation=call_tool_operation,
        resolve_tool_config=_resolve_function_tool_config,
        lifecycle='enter-never',
    )


@dataclass(init=False)
class RestateDurability(BaseDurabilityCapability[AgentDepsT]):
    """Capability that makes an agent durable by journaling its I/O into Restate run steps.

    Attach it via `capabilities=[RestateDurability()]` and call `agent.run()` inside a Restate
    service handler: every model request, MCP call, function tool call, and dynamic-toolset
    resolution is wrapped in `ctx.run_typed(...)`, so a handler that crashes or is retried mid-run
    replays from the journal instead of repeating the work. A completed step is served from its
    journal entry on replay instead of being recomputed, so tokens are not re-spent on work that
    already finished. A step is journaled after it runs, so a crash between a tool's side effect
    and its journal entry re-runs the tool on recovery: keep tool side effects idempotent. Outside
    a Restate context the capability is transparent and the run is a normal, non-durable agent run.

    The capability discovers the agent's model, name, and toolsets automatically when it is bound
    to the agent. Step results are written to the Restate journal as JSON bytes, so a journaled
    tool's return value must be JSON-serializable. Control-flow signals (`ModelRetry`,
    `ApprovalRequired`, `CallDeferred`, `ToolFailed`) cross the journal as values rather than
    exceptions, with their metadata preserved, so approval and deferred-tool flows work inside a
    durable run.

    Example:
        ```python {test="skip"}
        import restate
        from pydantic_ai import Agent
        from pydantic_ai_harness.restate import RestateDurability

        agent = Agent('openai:gpt-5', name='analyst', capabilities=[RestateDurability()])
        analyst = restate.Service('analyst')


        @analyst.handler()
        async def analyse(ctx: restate.Context, prompt: str) -> str:
            result = await agent.run(prompt)
            return result.output


        app = restate.app([analyst])
        ```
    """

    engine_name = _ENGINE_NAME
    _unsupported_runtime_toolset_kinds: ClassVar[frozenset[RuntimeToolsetKind]] = frozenset(
        {'function', 'mcp', 'dynamic'}
    )
    _durable_unit_noun = 'step'
    _durable_container_noun = 'handler'
    _tool_config_key = _TOOL_CONFIG_KEY

    def __init__(
        self,
        *,
        models: Mapping[str, Model] | None = None,
        event_stream_handler: EventStreamHandler[AgentDepsT] | None = None,
        name: str | None = None,
    ) -> None:
        """Create a `RestateDurability` capability.

        The agent's model, name, and toolsets are discovered automatically.

        Args:
            models: Optional additional models keyed by ID for runtime model switching via
                `agent.run(model='<id>')`. The agent's primary model is always registered as
                `'default'`; the ID is folded into the step name so a replay maps each journal
                entry back to the model it was recorded for.
            event_stream_handler: Optional event stream handler. Model events are handled live
                inside the model-request step; each agent-level event is handled in its own
                journaled step.
            name: Unique agent name used as the prefix for every step name. Defaults to the
                agent's `name` when the capability is bound.
        """
        super().__init__(models=models, event_stream_handler=event_stream_handler, name=name)
        self._default_model_id: str | None = None

    @property
    def in_durable_context(self) -> bool:
        return _in_durable_context()

    def _bind_to_agent(self, agent: AbstractAgent[AgentDepsT, Any]) -> None:
        # Restate steps are ad-hoc `ctx.run_typed(...)` calls, so there is nothing to register up
        # front beyond the durable toolset wrappers, which `_register_toolsets` builds and indexes
        # by toolset `id`. Recording the string default lets a request that carries the raw default
        # string as provenance journal under the suffix-less step name.
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
        handler = self._event_stream_handler
        assert handler is not None  # pragma: no cover - only dispatched when a handler is set

        async def _inner() -> _JsonValue:
            # The handler runs inside the journaled step, so guard `ctx.enqueue()`: a replay serves
            # the recorded entry without re-running it, which would drop enqueued messages.
            await handler(_guard_enqueue(ctx), self._single_event_stream(event))
            return None

        # Journal the handler call so its side effects do not re-run on recovery.
        await _require_context().run_typed(f'{self.name}__event_stream_handler', _inner, _RUN_OPTIONS)

    async def wrap_run(self, ctx: RunContext[AgentDepsT], *, handler: WrapRunHandler) -> AgentRunResult[Any]:
        """Run tool calls one at a time inside a Restate handler.

        A journal entry's identity is its encounter order, so concurrently scheduled tool calls
        could claim each other's entries on replay. Outside a Restate context the agent's
        configured mode is left untouched, so an ordinary run keeps its parallelism.
        """
        agent = self._agent
        if agent is None or not self.in_durable_context:
            return await handler()
        with agent.parallel_tool_call_execution_mode('sequential'):
            return await handler()

    async def wrap_model_request(
        self, ctx: RunContext[AgentDepsT], *, request_context: ModelRequestContext, handler: WrapModelRequestHandler
    ) -> ModelResponse:
        """Journal each model-request segment into a Restate run step when inside a handler."""
        restate_ctx = _current_restate_context()
        if restate_ctx is None:
            return await handler(request_context)

        # The step runs in-process, so the model needs no cross-boundary rebuild; the model id is
        # folded into the step name so a replay maps each entry back to the model it was recorded
        # for. A string default carries itself as provenance but was journaled without a suffix, so
        # suppress the suffix for it.
        model_id = self._model_id_for_request(ctx, request_context)
        if model_id is not None and model_id == self._default_model_id:
            model_id = None
        step_suffix = '' if model_id is None else f'.{model_id}'
        model = request_context.model

        async def request_segment(request: ModelRequestContext) -> ModelResponse:
            async def _inner() -> _JsonValue:
                response = await request.model.request(
                    request.messages, request.model_settings, request.model_request_parameters
                )
                return _serialize_response(response)

            payload = await restate_ctx.run_typed(f'{self.name}__model.request{step_suffix}', _inner, _RUN_OPTIONS)
            return _deserialize_response(payload)

        async def request_stream_segment(request: ModelRequestContext) -> StreamedActivityResult:
            async def _inner() -> _JsonValue:
                async with request.model.request_stream(
                    request.messages, request.model_settings, request.model_request_parameters, ctx
                ) as streamed:
                    # Model events reach the handler inside this journaled step, so it receives a
                    # guarded context: enqueueing there would be dropped on replay.
                    events = await capture_event_stream(
                        run_context=_guard_enqueue(ctx), stream=streamed, handler=self._event_stream_handler
                    )
                return {
                    'response': _serialize_response(streamed.get()),
                    'events': _events_adapter.dump_python(events, mode='json'),
                }

            payload = await restate_ctx.run_typed(
                f'{self.name}__model.request_stream{step_suffix}', _inner, _RUN_OPTIONS
            )
            assert isinstance(payload, dict)
            return StreamedActivityResult(
                response=_deserialize_response(payload['response']),
                events=_events_adapter.validate_python(payload['events']),
            )

        async def cancel_suspended_response_segment(response: ModelResponse) -> None:
            async def _inner() -> _JsonValue:
                await model.cancel_suspended_response(response)
                return None

            await restate_ctx.run_typed(
                f'{self.name}__model.cancel_suspended_response{step_suffix}', _inner, _RUN_OPTIONS
            )

        request_context.model = DurableModel(
            request_context.model,
            request_segment=request_segment,
            request_stream_segment=request_stream_segment,
            cancel_suspended_response_segment=cancel_suspended_response_segment,
        )
        return await handler(request_context)
