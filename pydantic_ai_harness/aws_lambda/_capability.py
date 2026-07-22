"""Durable execution for Pydantic AI agents on AWS Lambda durable functions.

Checkpoints an agent's I/O -- model requests, function tool calls, MCP calls, and
dynamic-toolset resolution -- into Lambda durable steps (`DurableContext.step(...)`), so an
invocation that times out, fails, or is retried resumes from the last completed step instead of
replaying the work.

Lambda's durable API is synchronous and an agent run is async, so the capability drives every
step through the bridge in `_bridge.py`. See `run_durable`, which is how a handler enters a run.
"""

from __future__ import annotations

try:
    import aws_durable_execution_sdk_python  # noqa: F401  # pyright: ignore[reportUnusedImport]
except ImportError as _import_error:  # pragma: no cover
    raise ImportError(
        'Please install the `aws-durable-execution-sdk-python` package to use the AWS Lambda durability '
        'capability, you can use the `aws-lambda` optional group -- '
        '`pip install "pydantic-ai-harness[aws-lambda]"`'
    ) from _import_error

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, ClassVar

from aws_durable_execution_sdk_python.config import StepConfig
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
from pydantic_ai.toolsets._dynamic import DynamicToolset  # pyright: ignore[reportPrivateUsage]

from ._bridge import current_bridge, in_durable_context

if TYPE_CHECKING:
    from pydantic_ai.mcp import MCPToolset

_ENGINE_NAME = 'AWS Lambda'
_TOOL_CONFIG_KEY = 'aws_lambda'
_TOOL_CONFIG_LABEL = 'AWS Lambda step config'
_NO_FALLBACK_CONFIG: Mapping[str, ToolConfig] = {}
_STEP_CONFIG_FIELDS = frozenset({'retry_strategy', 'step_semantics', 'serdes'})

_Instructions = str | InstructionPart | Sequence[str | InstructionPart] | None
_JsonValue = Any

_response_adapter: TypeAdapter[ModelResponse] = TypeAdapter(ModelResponse)
_events_adapter: TypeAdapter[list[ModelResponseStreamEvent]] = TypeAdapter(list[ModelResponseStreamEvent])
_call_tool_result_adapter: TypeAdapter[CallToolResult] = TypeAdapter(CallToolResult)
_tool_defs_adapter: TypeAdapter[dict[str, ToolDefinition]] = TypeAdapter(dict[str, ToolDefinition])
_dynamic_tools_adapter: TypeAdapter[DynamicToolsResult] = TypeAdapter(DynamicToolsResult)
_instructions_adapter: TypeAdapter[_Instructions] = TypeAdapter(_Instructions)


def _require_bridge() -> Any:
    bridge = current_bridge()
    assert bridge is not None  # pragma: no cover - callers gate on `in_durable_context`
    return bridge


def _step_config(config: Mapping[str, Any] | None) -> StepConfig | None:
    """Build a `StepConfig` from a per-tool metadata mapping."""
    if not config:
        return None
    unknown = sorted(set(config) - _STEP_CONFIG_FIELDS)
    if unknown:
        raise UserError(
            f'Unknown {_TOOL_CONFIG_KEY!r} step config {"key" if len(unknown) == 1 else "keys"} '
            f'{", ".join(repr(key) for key in unknown)}. Supported keys are '
            f'{", ".join(repr(field) for field in sorted(_STEP_CONFIG_FIELDS))}.'
        )
    return StepConfig(**config)


def _guard_enqueue(ctx: RunContext[AgentDepsT]) -> RunContext[AgentDepsT]:
    """Make `ctx.enqueue()` raise inside a step-wrapped tool call.

    A replay serves the recorded step output without re-running the tool, so messages enqueued
    inside the step would be dropped.
    """
    pending: list[PendingMessage] = EnqueueGuard(
        '`ctx.enqueue()` is not supported inside AWS Lambda step-wrapped tools because a replay '
        'serves the recorded step output and would drop the enqueued messages. Enqueue messages '
        'from handler-level code instead.'
    )
    return replace(ctx, pending_messages=pending)


def _serialize_call_tool_result(result: CallToolResult) -> _JsonValue:
    return _call_tool_result_adapter.dump_python(result, mode='json')


def _deserialize_call_tool_result(payload: _JsonValue) -> CallToolResult:
    return _call_tool_result_adapter.validate_python(payload)


def _resolve_function_tool_config(tool: ToolsetTool[Any] | None, tool_name: str) -> ToolConfig:
    return resolve_tool_durable_config(
        tool, tool_name, _NO_FALLBACK_CONFIG, metadata_key=_TOOL_CONFIG_KEY, config_type_label=_TOOL_CONFIG_LABEL
    )


def _resolve_mcp_tool_config(tool: ToolsetTool[Any] | None, tool_name: str) -> ToolConfig:
    config = _resolve_function_tool_config(tool, tool_name)
    if config is False:
        raise UserError(
            f'{_ENGINE_NAME} checkpointing for MCP tool {tool_name!r} was disabled with '
            f'metadata={{{_TOOL_CONFIG_KEY!r}: False}}, but MCP tools perform I/O and so cannot run '
            'outside a step. Remove the metadata so the call stays checkpointed.'
        )
    return config


def _build_function_toolset(
    toolset: FunctionToolset[AgentDepsT], *, step_name_prefix: str, base_config: Mapping[str, Any] | None
) -> DurableFunctionToolset[AgentDepsT]:
    name = f'{step_name_prefix}__function_toolset__{toolset.id}'

    async def call_tool_operation(
        tool_name: str,
        tool_args: dict[str, Any],
        ctx: RunContext[AgentDepsT],
        tool: ToolsetTool[AgentDepsT],
        config: Mapping[str, Any],
    ) -> Any:
        async def operation() -> _JsonValue:
            result = await wrap_tool_call_result(toolset.call_tool(tool_name, tool_args, _guard_enqueue(ctx), tool))
            return _serialize_call_tool_result(result)

        payload = await _require_bridge().run_step(
            f'{name}.call_tool:{tool_name}', operation, _step_config(config or base_config)
        )
        return unwrap_recorded_tool_call_result(_deserialize_call_tool_result(payload))

    return DurableFunctionToolset(
        toolset,
        in_durable_context=in_durable_context,
        call_tool_operation=call_tool_operation,
        resolve_tool_config=_resolve_function_tool_config,
        # The durable path is gated on `in_durable_context`, so outside a durable handler the
        # wrapper must enter the wrapped toolset itself, as the plain run would.
        lifecycle='enter-always',
        durable_config=base_config,
    )


def _build_mcp_toolset(
    toolset: MCPToolset[AgentDepsT], *, step_name_prefix: str, base_config: Mapping[str, Any] | None
) -> DurableMCPToolset[AgentDepsT]:
    name = f'{step_name_prefix}__mcp_server__{toolset.id}'

    async def get_tools_operation(ctx: RunContext[AgentDepsT]) -> dict[str, ToolDefinition]:
        async def operation() -> _JsonValue:
            tools = await toolset.get_tools(ctx)
            return _tool_defs_adapter.dump_python({n: t.tool_def for n, t in tools.items()}, mode='json')

        payload = await _require_bridge().run_step(f'{name}.get_tools', operation, _step_config(base_config))
        return _tool_defs_adapter.validate_python(payload)

    async def get_instructions_operation(ctx: RunContext[AgentDepsT]) -> _Instructions:
        async def operation() -> _JsonValue:
            return _instructions_adapter.dump_python(await toolset.get_instructions(ctx), mode='json')

        payload = await _require_bridge().run_step(f'{name}.get_instructions', operation, _step_config(base_config))
        return _instructions_adapter.validate_python(payload)

    async def call_tool_operation(
        tool_name: str,
        tool_args: dict[str, Any],
        ctx: RunContext[AgentDepsT],
        tool: ToolsetTool[AgentDepsT],
        config: Mapping[str, Any],
    ) -> Any:
        async def operation() -> _JsonValue:
            result = await wrap_tool_call_result(toolset.call_tool(tool_name, tool_args, _guard_enqueue(ctx), tool))
            return _serialize_call_tool_result(result)

        payload = await _require_bridge().run_step(
            f'{name}.call_tool:{tool_name}', operation, _step_config(config or base_config)
        )
        return unwrap_recorded_tool_call_result(_deserialize_call_tool_result(payload))

    return DurableMCPToolset(
        toolset,
        in_durable_context=in_durable_context,
        get_tools_operation=get_tools_operation,
        get_instructions_operation=get_instructions_operation,
        call_tool_operation=call_tool_operation,
        resolve_tool_config=_resolve_mcp_tool_config,
        # Keep one session for the run rather than an implicit session per call, which is what a
        # plain (non-durable) run does. Prefect takes the same stance for a gated engine.
        lifecycle='enter-always',
        durable_config=base_config,
    )


def _build_dynamic_toolset(
    toolset: DynamicToolset[AgentDepsT], *, step_name_prefix: str, base_config: Mapping[str, Any] | None
) -> DurableDynamicToolset[AgentDepsT]:
    name = f'{step_name_prefix}__dynamic_toolset__{toolset.id}'

    async def get_tools_operation(ctx: RunContext[AgentDepsT]) -> DynamicToolsResult:
        async def operation() -> _JsonValue:
            return _dynamic_tools_adapter.dump_python(await get_dynamic_tools(toolset, ctx), mode='json')

        payload = await _require_bridge().run_step(f'{name}.get_tools', operation, _step_config(base_config))
        return _dynamic_tools_adapter.validate_python(payload)

    async def call_tool_operation(
        tool_name: str,
        tool_args: dict[str, Any],
        ctx: RunContext[AgentDepsT],
        tool: ToolsetTool[AgentDepsT],
        config: Mapping[str, Any],
    ) -> Any:
        async def operation() -> _JsonValue:
            # Arg re-validation happens inside `call_dynamic_tool`: a `ValidationError` from it is
            # deterministic, so letting it leave the step un-checkpointed is safe and `ToolManager`
            # turns it into a retry prompt.
            result = await wrap_tool_call_result(call_dynamic_tool(toolset, tool_name, tool_args, _guard_enqueue(ctx)))
            return _serialize_call_tool_result(result)

        payload = await _require_bridge().run_step(
            f'{name}.call_tool:{tool_name}', operation, _step_config(config or base_config)
        )
        return unwrap_recorded_tool_call_result(_deserialize_call_tool_result(payload))

    return DurableDynamicToolset(
        toolset,
        in_durable_context=in_durable_context,
        get_tools_operation=get_tools_operation,
        call_tool_operation=call_tool_operation,
        resolve_tool_config=_resolve_function_tool_config,
        lifecycle='enter-never',
        durable_config=base_config,
    )


@dataclass(init=False)
class LambdaDurability(BaseDurabilityCapability[AgentDepsT]):
    """Capability that checkpoints an agent's I/O into AWS Lambda durable steps.

    Attach it with `capabilities=[LambdaDurability()]` and enter the run from a durable handler
    with `run_durable`: every model request, function tool call, MCP call, and dynamic-toolset
    resolution is wrapped in `DurableContext.step(...)`. A completed step is served from its
    checkpoint when the execution resumes, so finished work is not repeated and tokens are not
    re-spent.

    A step is checkpointed after it runs, so an interruption between a tool's side effect and its
    checkpoint re-runs the tool when the execution resumes: keep tool side effects idempotent, or
    set `step_semantics` to `AT_MOST_ONCE_PER_RETRY` for the tools that cannot tolerate it.

    Outside a durable handler the capability is transparent and the run is an ordinary agent run.

    Step results are checkpointed through the Lambda SDK's serializer, so a checkpointed tool's
    return value must survive that round trip. Control-flow signals (`ModelRetry`,
    `ApprovalRequired`, `CallDeferred`, `ToolFailed`) cross the boundary as values rather than
    exceptions, so approval and deferred-tool flows work inside a durable execution.

    Example:
        ```python {test="skip"}
        from aws_durable_execution_sdk_python import DurableContext, durable_execution
        from pydantic_ai import Agent
        from pydantic_ai_harness.aws_lambda import LambdaDurability, run_durable

        agent = Agent('bedrock:us.amazon.nova-pro-v1:0', name='support', capabilities=[LambdaDurability()])


        @agent.tool_plain
        def get_weather(city: str) -> str:
            return f'It is sunny in {city}.'


        @durable_execution
        def handler(event: dict[str, object], context: DurableContext) -> str:
            result = run_durable(lambda: agent.run(str(event['prompt'])), context=context)
            return result.output
        ```
    """

    engine_name = _ENGINE_NAME
    _unsupported_runtime_toolset_kinds: ClassVar[frozenset[RuntimeToolsetKind]] = frozenset(
        {'function', 'mcp', 'dynamic'}
    )
    _durable_unit_noun = 'step'
    _durable_container_noun = 'durable execution'
    _tool_config_key = _TOOL_CONFIG_KEY

    def __init__(
        self,
        *,
        models: Mapping[str, Model] | None = None,
        event_stream_handler: EventStreamHandler[AgentDepsT] | None = None,
        name: str | None = None,
        step_config: Mapping[str, Any] | None = None,
    ) -> None:
        """Create a `LambdaDurability` capability.

        The agent's model, name, and toolsets are discovered when the capability is bound.

        Args:
            models: Optional additional models keyed by ID for run-time model switching via
                `agent.run(model='<id>')`. The ID is folded into the step name so a resumed
                execution maps each checkpoint back to the model it was recorded for.
            event_stream_handler: Optional event stream handler. Model events are handled live
                inside the model-request step; each agent-level event is handled in its own
                checkpointed step.
            name: Unique agent name used as the prefix for every step name. Defaults to the
                agent's `name` when the capability is bound.
            step_config: Base `StepConfig` fields applied to every step, as a mapping of
                `retry_strategy`, `step_semantics`, and `serdes`. Per-tool
                `metadata={'aws_lambda': {...}}` replaces it for that tool.
        """
        super().__init__(models=models, event_stream_handler=event_stream_handler, name=name)
        # Validate eagerly so a bad config fails at construction rather than mid-execution.
        _step_config(step_config)
        self._base_step_config = step_config
        self._default_model_id: str | None = None

    @property
    def in_durable_context(self) -> bool:
        return in_durable_context()

    def _bind_to_agent(self, agent: AbstractAgent[AgentDepsT, Any]) -> None:
        # Lambda durable steps are ad-hoc `context.step(...)` calls, so there is nothing to
        # register up front beyond the durable toolset wrappers.
        self._default_model_id = agent.model if isinstance(agent.model, str) else None
        self._register_toolsets(agent)

    def _wrap_leaf_toolset(self, ts: AbstractToolset[AgentDepsT]) -> WrapperToolset[AgentDepsT] | None:
        if isinstance(ts, FunctionToolset):
            return _build_function_toolset(ts, step_name_prefix=self.name, base_config=self._base_step_config)
        if isinstance(ts, DynamicToolset):
            return _build_dynamic_toolset(ts, step_name_prefix=self.name, base_config=self._base_step_config)
        try:
            from pydantic_ai.mcp import MCPToolset
        except ImportError:  # pragma: no cover - only when the mcp extra is absent
            return None
        if isinstance(ts, MCPToolset):
            return _build_mcp_toolset(ts, step_name_prefix=self.name, base_config=self._base_step_config)
        return None

    async def _dispatch_event_stream_event(self, ctx: RunContext[AgentDepsT], event: AgentStreamEvent) -> None:
        handler = self._event_stream_handler
        assert handler is not None  # pragma: no cover - only dispatched when a handler is set

        async def operation() -> None:
            await handler(ctx, self._single_event_stream(event))

        # Checkpoint the handler call so its side effects are not repeated when the execution resumes.
        await _require_bridge().run_step(
            f'{self.name}__event_stream_handler', operation, _step_config(self._base_step_config)
        )

    async def wrap_run(self, ctx: RunContext[AgentDepsT], *, handler: WrapRunHandler) -> AgentRunResult[Any]:
        """Run tool calls one at a time inside a durable execution.

        A step's identity comes from the order steps are reached, so concurrently scheduled tool
        calls could claim each other's checkpoints when the execution resumes. Outside a durable
        handler the agent's configured mode is left untouched, so an ordinary run keeps its
        parallelism.
        """
        agent = self._agent
        if agent is None or not self.in_durable_context:
            return await handler()
        with agent.parallel_tool_call_execution_mode('sequential'):
            return await handler()

    async def wrap_model_request(
        self, ctx: RunContext[AgentDepsT], *, request_context: ModelRequestContext, handler: WrapModelRequestHandler
    ) -> ModelResponse:
        """Checkpoint each model-request segment into a durable step."""
        if not self.in_durable_context:
            return await handler(request_context)

        # The step runs in the same process, so the model needs no cross-boundary rebuild; the
        # model id only disambiguates the step name. A string default carries itself as
        # provenance, so suppress the suffix for it to keep one name for the default model.
        model_id = self._model_id_for_request(ctx, request_context)
        if model_id is not None and model_id == self._default_model_id:
            model_id = None
        suffix = '' if model_id is None else f'.{model_id}'
        config = _step_config(self._base_step_config)
        model = request_context.model

        async def request_segment(request: ModelRequestContext) -> ModelResponse:
            async def operation() -> _JsonValue:
                response = await request.model.request(
                    request.messages, request.model_settings, request.model_request_parameters
                )
                return _response_adapter.dump_python(response, mode='json')

            payload = await _require_bridge().run_step(f'{self.name}__model.request{suffix}', operation, config)
            return _response_adapter.validate_python(payload)

        async def request_stream_segment(request: ModelRequestContext) -> StreamedActivityResult:
            async def operation() -> _JsonValue:
                async with request.model.request_stream(
                    request.messages, request.model_settings, request.model_request_parameters, ctx
                ) as streamed:
                    events = await capture_event_stream(
                        run_context=ctx, stream=streamed, handler=self._event_stream_handler
                    )
                return {
                    'response': _response_adapter.dump_python(streamed.get(), mode='json'),
                    'events': _events_adapter.dump_python(events, mode='json'),
                }

            payload = await _require_bridge().run_step(f'{self.name}__model.request_stream{suffix}', operation, config)
            return StreamedActivityResult(
                response=_response_adapter.validate_python(payload['response']),
                events=_events_adapter.validate_python(payload['events']),
            )

        async def cancel_suspended_response_segment(response: ModelResponse) -> None:
            async def operation() -> None:
                await model.cancel_suspended_response(response)

            await _require_bridge().run_step(f'{self.name}__model.cancel_suspended_response{suffix}', operation, config)

        request_context.model = DurableModel(
            request_context.model,
            request_segment=request_segment,
            request_stream_segment=request_stream_segment,
            cancel_suspended_response_segment=cancel_suspended_response_segment,
        )
        return await handler(request_context)
