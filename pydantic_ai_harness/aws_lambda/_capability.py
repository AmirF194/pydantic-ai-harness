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

from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import Any, ClassVar, Literal, TypeAlias

from aws_durable_execution_sdk_python.config import StepConfig
from pydantic import TypeAdapter
from pydantic_ai.agent import AbstractAgent, EventStreamHandler
from pydantic_ai.durable_exec import (
    JSON_CODEC,
    BaseDurabilityCapability,
    DurabilityEngineSpec,
    DurableOperationBackend,
)
from pydantic_ai.exceptions import UserError
from pydantic_ai.models import Model
from pydantic_ai.tools import AgentDepsT
from pydantic_ai.toolsets import ToolsetTool
from typing_extensions import Self

from ._bridge import ENGINE_NAME as _ENGINE_NAME
from ._bridge import in_durable_context
from ._operation_backend import AWSLambdaOperationBackend, AWSLambdaOperationConfig

_TOOL_CONFIG_KEY = 'aws_lambda'
_ToolsetKind: TypeAlias = Literal['function', 'mcp', 'dynamic']
# Derived rather than listed so a new SDK `StepConfig` field is accepted, not rejected as unknown.
_STEP_CONFIG_FIELDS = frozenset(f.name for f in fields(StepConfig))
_STEP_CONFIG_MAPPING_ADAPTER = TypeAdapter(dict[str, object])


def _step_config(config: Mapping[str, Any] | None) -> StepConfig | None:
    """Build a `StepConfig` from a step configuration mapping."""
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


def _toolset_tool(value: object | None) -> ToolsetTool[Any] | None:
    if isinstance(value, ToolsetTool):
        return value  # pyright: ignore[reportUnknownVariableType]
    return None


@dataclass(init=False)
class AWSLambdaDurability(BaseDurabilityCapability[AgentDepsT]):
    """Capability that checkpoints an agent's I/O into AWS Lambda durable steps.

    Attach it with `capabilities=[AWSLambdaDurability()]` and enter the run from a durable handler
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
        from pydantic_ai_harness.aws_lambda import AWSLambdaDurability, run_durable

        agent = Agent('bedrock:us.amazon.nova-pro-v1:0', name='support', capabilities=[AWSLambdaDurability()])


        @agent.tool_plain
        def get_weather(city: str) -> str:
            return f'It is sunny in {city}.'


        @durable_execution
        def handler(event: dict[str, object], context: DurableContext) -> str:
            result = run_durable(lambda: agent.run(str(event['prompt'])), context=context)
            return result.output
        ```
    """

    engine_spec: ClassVar = DurabilityEngineSpec(
        engine_name=_ENGINE_NAME,
        durable_unit_noun='step',
        durable_container_noun='handler',
        codec=JSON_CODEC,
        tool_call_result_upgrade_lenient=True,
        sequential_tools_in_durable_context=True,
        unsupported_runtime_toolset_kinds=frozenset({'function', 'mcp', 'dynamic'}),
        tool_config_key=_TOOL_CONFIG_KEY,
    )

    def __init__(
        self,
        *,
        models: Mapping[str, Model] | None = None,
        event_stream_handler: EventStreamHandler[AgentDepsT] | None = None,
        name: str | None = None,
        step_config: Mapping[str, Any] | None = None,
    ) -> None:
        """Create an `AWSLambdaDurability` capability.

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
                `metadata={'aws_lambda': {...}}` overrides it key by key for that tool.
        """
        super().__init__(models=models, event_stream_handler=event_stream_handler, name=name)
        _step_config(step_config)
        self._base_step_config = step_config
        self._lambda_default_model_id: str | None = None

    def for_agent(self, agent: AbstractAgent[AgentDepsT, Any]) -> Self:
        self._lambda_default_model_id = agent.model if isinstance(agent.model, str) else None
        return super().for_agent(agent)

    @property
    def in_durable_context(self) -> bool:
        return in_durable_context()

    def get_durable_operation_backend(self) -> DurableOperationBackend[StepConfig | None]:
        def tool_config(kind: _ToolsetKind, tool: object | None, tool_name: str) -> StepConfig | Literal[False] | None:
            toolset_tool = _toolset_tool(tool)
            metadata = toolset_tool.tool_def.metadata if toolset_tool is not None else None
            metadata_config: object = metadata.get(_TOOL_CONFIG_KEY) if metadata is not None else None
            if metadata_config is False:
                return False
            if metadata_config is not None and not isinstance(metadata_config, dict):
                raise UserError(
                    f'Tool {tool_name!r} has invalid {_TOOL_CONFIG_KEY!r} metadata: expected a dict '
                    f'(`{_ENGINE_NAME} durable config`) or `False`, got {type(metadata_config).__name__}.'
                )
            config = dict(self._base_step_config or {})
            if metadata_config:
                config.update(_STEP_CONFIG_MAPPING_ADAPTER.validate_python(metadata_config, strict=True))
            return _step_config(config)

        return AWSLambdaOperationBackend(
            agent_name=self.name,
            default_model_id=self._lambda_default_model_id,
            config=AWSLambdaOperationConfig(base=self._base_step_config, tool=tool_config),
        )
