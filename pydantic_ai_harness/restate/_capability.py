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
    import restate
except ImportError as _import_error:
    raise ImportError(
        'Please install the `restate-sdk` package to use the Restate durability capability, '
        'you can use the `restate` optional group -- `pip install "pydantic-ai-harness[restate]"`'
    ) from _import_error

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any, ClassVar

from pydantic_ai.agent import EventStreamHandler
from pydantic_ai.durable_exec._base import BaseDurabilityCapability, ToolsetKind
from pydantic_ai.durable_exec._codec import JSON_CODEC
from pydantic_ai.durable_exec._runtime_toolsets import RuntimeToolsetKind
from pydantic_ai.durable_exec._toolset import Lifecycle
from pydantic_ai.exceptions import UserError
from pydantic_ai.models import Model
from pydantic_ai.tools import AgentDepsT
from restate.context import RunOptions
from restate.exceptions import TerminalError
from restate.extensions import current_context
from restate.serde import JsonSerde

_ENGINE_NAME = 'Restate'
_TOOL_CONFIG_KEY = 'restate'


def _current_restate_context() -> restate.Context | None:
    """Return the active Restate context, or `None` outside a Restate invocation."""
    try:
        return current_context()
    except LookupError:
        return None


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
    tool's return value must be JSON-serializable. Serialization failures are terminal so Restate
    does not retry them as transient invocation failures. Control-flow signals (`ModelRetry`,
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
    _codec: ClassVar = JSON_CODEC
    _unsupported_runtime_toolset_kinds: ClassVar[frozenset[RuntimeToolsetKind]] = frozenset(
        {'function', 'mcp', 'dynamic'}
    )
    _wrapped_toolset_kinds: ClassVar[frozenset[ToolsetKind]] = frozenset({'function', 'mcp', 'dynamic'})
    _toolset_lifecycles: ClassVar[Mapping[ToolsetKind, Lifecycle]] = {
        'function': 'enter-never',
        'mcp': 'enter-always',
        'dynamic': 'enter-never',
    }
    _tool_call_result_upgrade_lenient: ClassVar[bool] = False
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

    @property
    def in_durable_context(self) -> bool:
        return _current_restate_context() is not None

    async def run_durable_unit(
        self, name: str, fn: Callable[[], Awaitable[Any]], *, inputs: tuple[Any, ...], config: Any
    ) -> Any:
        """Run the base-built operation through a Restate run step."""
        del inputs
        assert not config
        context = current_context()
        assert context is not None
        return await context.run_typed(name, fn, RunOptions(serde=JsonSerde()))

    def _normalize_unit_config(self, config: Any) -> Any:
        if config:
            raise UserError('Restate run steps take no per-tool options; remove the config.')
        return config

    def _serialization_failure(self, exc: Exception) -> BaseException:
        """Fail non-serializable step results without retrying the Restate invocation."""
        return TerminalError(str(exc))
