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

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any, ClassVar, Literal

from absurd_sdk import AsyncTaskContext, get_current_context
from pydantic_ai.agent import EventStreamHandler, ParallelExecutionMode
from pydantic_ai.capabilities.abstract import WrapRunHandler
from pydantic_ai.durable_exec._base import BaseDurabilityCapability, ToolsetKind
from pydantic_ai.durable_exec._codec import JSON_CODEC
from pydantic_ai.durable_exec._runtime_toolsets import RuntimeToolsetKind
from pydantic_ai.durable_exec._toolset import Lifecycle
from pydantic_ai.exceptions import UserError
from pydantic_ai.models import Model
from pydantic_ai.run import AgentRunResult
from pydantic_ai.tools import AgentDepsT, RunContext

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


def _current_async_task_context() -> AsyncTaskContext | None:
    """Return the active Absurd async task context, or `None` outside a task."""
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


def _reject_model_id_hash(model_id: str) -> None:
    """Reject a model id containing `#`, which Absurd reserves for encounter counters."""
    if '#' in model_id:
        raise UserError(
            f'Model id {model_id!r} contains {"#"!r}, which Absurd uses to disambiguate repeated '
            'step names. Folded into the model step name it would collide with that suffix. Choose '
            'a model id without `#`.'
        )


@dataclass(init=False)
class AbsurdDurability(BaseDurabilityCapability[AgentDepsT]):
    """Capability that makes an agent durable by checkpointing its I/O into Absurd steps.

    Attach it via `capabilities=[AbsurdDurability()]` and call `agent.run()` inside an Absurd
    task handler: every model request, MCP call, function tool call, and dynamic-toolset
    resolution is wrapped in `ctx.step(...)`, so a worker crash mid-run resumes from the last
    completed step instead of restarting. A completed step is served from its checkpoint on
    replay instead of being recomputed, so tokens are not re-spent on work that already finished.
    A step is checkpointed after it runs, so a crash between a tool's side effect and its checkpoint
    re-runs the tool on recovery: keep tool side effects idempotent. Outside a task the capability
    is transparent and the run is a normal, non-durable agent run.

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
    _journal_discovery: ClassVar[bool] = True
    _force_sequential_tools_in_durable_context: ClassVar[bool] = True
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

    @property
    def in_durable_context(self) -> bool:
        return _current_async_task_context() is not None

    async def run_durable_unit(
        self, name: str, fn: Callable[[], Awaitable[Any]], *, inputs: tuple[Any, ...], config: Any
    ) -> Any:
        """Run the base-built operation through an Absurd step."""
        del inputs
        assert config is None
        task_ctx = _current_async_task_context()
        assert task_ctx is not None
        return await task_ctx.step(name, fn)

    def _normalize_unit_config(self, config: Any) -> None:
        if config:
            raise UserError(
                f'Absurd steps take no per-tool options, so non-empty {_TOOL_CONFIG_KEY!r} metadata '
                'would have no effect. Remove the config.'
            )

    async def wrap_run(self, ctx: RunContext[AgentDepsT], *, handler: WrapRunHandler) -> AgentRunResult[Any]:
        """Allow Absurd's encounter-safe ordered parallel mode when explicitly configured."""
        if self._parallel_execution_mode == 'sequential':
            return await super().wrap_run(ctx, handler=handler)
        agent = self._agent
        if agent is None or not self.in_durable_context:
            return await handler()
        with agent.parallel_tool_call_execution_mode(self._parallel_execution_mode):
            return await handler()
