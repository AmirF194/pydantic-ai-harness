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

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, fields
from typing import Any, ClassVar

from aws_durable_execution_sdk_python.config import StepConfig
from aws_durable_execution_sdk_python.exceptions import ExecutionError
from pydantic_ai.agent import EventStreamHandler
from pydantic_ai.durable_exec._base import BaseDurabilityCapability, ToolsetKind
from pydantic_ai.durable_exec._codec import JSON_CODEC
from pydantic_ai.durable_exec._runtime_toolsets import RuntimeToolsetKind
from pydantic_ai.durable_exec._toolset import Lifecycle
from pydantic_ai.exceptions import UserError
from pydantic_ai.models import Model
from pydantic_ai.tools import AgentDepsT

from ._bridge import ENGINE_NAME as _ENGINE_NAME
from ._bridge import StepBridge, current_bridge, in_durable_context

_TOOL_CONFIG_KEY = 'aws_lambda'
# Derived rather than listed so a new SDK `StepConfig` field is accepted, not rejected as unknown.
_STEP_CONFIG_FIELDS = frozenset(f.name for f in fields(StepConfig))


def _require_bridge() -> StepBridge:
    bridge = current_bridge()
    assert bridge is not None  # pragma: no cover - callers gate on `in_durable_context`
    return bridge


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

    engine_name = _ENGINE_NAME
    _codec: ClassVar = JSON_CODEC
    _unsupported_runtime_toolset_kinds: ClassVar[frozenset[RuntimeToolsetKind]] = frozenset(
        {'function', 'mcp', 'dynamic'}
    )
    _wrapped_toolset_kinds: ClassVar[frozenset[ToolsetKind]] = frozenset({'function', 'mcp', 'dynamic'})
    _toolset_lifecycles: ClassVar[Mapping[ToolsetKind, Lifecycle]] = {
        'function': 'enter-always',
        'mcp': 'enter-always',
        'dynamic': 'enter-never',
    }
    _tool_call_result_upgrade_lenient: ClassVar[bool] = True
    _journal_discovery: ClassVar[bool] = True
    _force_sequential_tools_in_durable_context: ClassVar[bool] = True
    _durable_unit_noun = 'step'
    _durable_container_noun = 'handler'
    _tool_config_key = _TOOL_CONFIG_KEY

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

    @property
    def in_durable_context(self) -> bool:
        return in_durable_context()

    async def run_durable_unit(
        self, name: str, fn: Callable[[], Awaitable[Any]], *, inputs: tuple[Any, ...], config: Any
    ) -> Any:
        """Run the base-built operation through Lambda's synchronous step bridge."""
        del inputs
        assert config is None or isinstance(config, StepConfig)

        async def operation() -> Any:
            return await fn()

        return await _require_bridge().run_step(name, operation, config)

    def _model_unit_config(self) -> StepConfig | None:
        return _step_config(self._base_step_config)

    def _event_unit_config(self) -> StepConfig | None:
        return _step_config(self._base_step_config)

    def _toolset_base_config(self, kind: ToolsetKind) -> Mapping[str, Any] | None:
        del kind
        return self._base_step_config

    def _normalize_unit_config(self, config: Any) -> StepConfig | None:
        return _step_config(config)

    def _serialization_failure(self, exc: Exception) -> BaseException:
        """Map serialization failures to the SDK's non-retryable execution error.

        This fails invalid checkpoint values without consuming SDK step retries.
        """
        return ExecutionError(str(exc))
