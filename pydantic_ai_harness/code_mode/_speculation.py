"""Speculative launch of sandboxed tool calls while `run_code` arguments stream.

Implements the streaming half of speculative programmatic tool calling (sPTC,
<https://alexzhang13.github.io/blog/2026/spec-ptc/>) for `CodeMode`: while the model is still
emitting the `run_code` tool call, the partial `code` argument is decoded from the accumulated
JSON, parsed into closed top-level statements, and scanned for calls to allowlisted sandbox
functions whose arguments are all literals. Those calls launch immediately as tasks; when the
completed snippet later executes and dispatches the same call, the in-flight task is adopted
instead of starting the tool cold.

The other half of the trick, overlapping independent calls with each other during execution,
already exists: `MontyExecutor` defers external calls as futures and only forces them when the
sandbox needs the value.

Scope is deliberately the blog's "Case 1" (all-literal arguments). Calls whose arguments carry
dependencies would need a shadow interpreter to materialize values; with Monty that shrinks to
forking a session, which is future work and out of scope here.

Only tools named in the allowlist are ever launched early. Launching early is observationally
equivalent to the normal call only for tools without side effects, so the allowlist is an
explicit user promise, mirroring the reference implementation's `speculatable=True, pure=True`
contract. Speculated calls run through the same nested `ToolManager` path as cold calls, so
capability tool hooks fire at launch time rather than at adoption time -- a documented POC
trade-off.
"""

from __future__ import annotations

import ast
import asyncio
import json
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from pydantic import TypeAdapter, ValidationError
from pydantic_ai.messages import (
    AgentStreamEvent,
    PartDeltaEvent,
    PartStartEvent,
    ToolCallPart,
    ToolCallPartDelta,
)
from pydantic_ai.tool_manager import ToolManager
from pydantic_ai.tools import RunContext, ToolDefinition
from pydantic_ai.toolsets.abstract import AbstractToolset, ToolsetTool
from pydantic_core import from_json
from typing_extensions import NotRequired, TypedDict

_RUN_CODE_TOOL_NAME = 'run_code'

# Upper bound on early launches per streamed `run_code` part. Wrong-branch conditionals and
# rewritten lines make some speculations garbage; the cap bounds how much garbage one part can
# start. Adoption is unaffected once a call is in flight.
_MAX_SPECULATIONS_PER_PART = 32


@dataclass
class SpeculationStats:
    """Aggregate speculation counters, shared across the runs of one `CodeMode` instance."""

    launched: int = 0
    """Calls started early from partial code."""

    adopted: int = 0
    """Early calls whose result the executing snippet actually claimed."""

    evicted: int = 0
    """Early calls cancelled or discarded without being claimed."""


@dataclass
class _SpecOutcome:
    """What a speculative task settles to. Exactly one of `error`/the value fields is meaningful."""

    content: Any = None
    """The plain tool return value, before sandbox serialization."""

    serialized: Any = None
    """The sandbox-ready form of `content` (the value the snippet receives)."""

    metadata: Any = None
    """`ToolReturn.metadata` when the tool returned a `ToolReturn`."""

    error: BaseException | None = None
    """The exception the cold path would have raised at the sandbox call site."""

    denied_message: str | None = None
    """Set when a handler denied the call, so adoption can record `outcome='denied'`."""


@dataclass
class SpeculativeCall:
    """One in-flight early launch, waiting to be claimed by the executing snippet."""

    sandbox_name: str
    original_name: str
    kwargs: dict[str, Any]
    task: asyncio.Task[_SpecOutcome]


@dataclass
class _StepIngredients:
    """Everything the launcher needs to run a nested call, stashed by the toolset each step."""

    wrapped: AbstractToolset[Any]
    wrapped_tools: dict[str, ToolsetTool[Any]]
    sanitized_to_original: dict[str, str]
    eligible: frozenset[str]
    """Sandbox (possibly sanitized) names that may launch early this step."""

    serialize: Callable[[Any], Any]
    """Serializer matching the cold path's sandbox return serialization."""


@dataclass
class _PartWatch:
    """Accumulated state for one streamed `run_code` tool call part."""

    tool_call_id: str
    args_text: str = ''
    args_dict: dict[str, Any] | None = None
    consumed_statements: int = 0
    launched: int = 0
    calls: dict[str, deque[SpeculativeCall]] = field(default_factory=dict[str, deque[SpeculativeCall]])
    """FIFO per canonical key: the k-th identical dispatch claims the k-th launch, so results
    of a nondeterministic tool called twice with the same arguments are never collapsed."""


def _canonical_key(sandbox_name: str, kwargs: dict[str, Any]) -> str:
    """Claim identity for one concrete call: both launch and claim hash through here."""
    return json.dumps([sandbox_name, kwargs], sort_keys=True, default=repr)


class _PartialArgs(TypedDict):
    """Lenient view of partially streamed `run_code` arguments: only the code we scan."""

    code: NotRequired[object]


_PARTIAL_ARGS_ADAPTER: TypeAdapter[_PartialArgs] = TypeAdapter(_PartialArgs)


def _decode_partial_code(args_text: str) -> str | None:
    """Recover the `code` string prefix from partially streamed JSON arguments."""
    if not args_text:
        return None
    try:
        decoded = _PARTIAL_ARGS_ADAPTER.validate_python(from_json(args_text, allow_partial='trailing-strings'))
    except (ValueError, ValidationError):
        return None
    code = decoded.get('code')
    return code if isinstance(code, str) else None


def _closed_statements(code: str, consumed: int) -> tuple[list[ast.stmt], int]:
    """Return newly closed top-level statements in `code`, past the first `consumed`.

    Only fully streamed lines participate, and the final parsed statement always stays
    provisional: a trailing compound (`for`, `if`, `try`) can still grow an indented body, so a
    statement counts as closed only once a later top-level statement follows it. A prefix that
    does not parse yields nothing -- either an open bracket/triple-quote closes later, or the
    model wrote broken code and the real run will surface the error.

    The full prefix is re-parsed on each delta. That is quadratic in snippet length, which is
    acceptable for model-written snippets; the reference implementation's incremental scanner is
    the known upgrade path.
    """
    end = code.rfind('\n')
    if end < 0:
        return [], consumed
    try:
        tree = ast.parse(code[: end + 1])
    except SyntaxError:
        return [], consumed
    closed = tree.body[:-1]
    return closed[consumed:], max(consumed, len(closed))


_SKIP_CONTAINERS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)


def _iter_calls(node: ast.AST) -> list[ast.Call]:
    """Collect `ast.Call` nodes in execution position, skipping bodies that don't run yet.

    `def`/`class`/`lambda` bodies execute only when later invoked, usually with non-literal
    arguments, so calls inside them are excluded. Conditional and loop bodies are included:
    launching a pure call from a branch that never runs wastes the call but stays correct,
    since unclaimed launches are evicted.
    """
    found: list[ast.Call] = []
    if isinstance(node, ast.Call):
        found.append(node)
    for child in ast.iter_child_nodes(node):
        if isinstance(child, _SKIP_CONTAINERS):
            continue
        found.extend(_iter_calls(child))
    return found


def _literal_calls(statements: Sequence[ast.stmt], eligible: frozenset[str]) -> list[tuple[str, dict[str, Any]]]:
    """Extract eligible sandbox calls whose arguments are entirely keyword literals.

    Positional arguments are never speculated: the sandbox rejects them at execution time, so an
    early launch would run a call the real snippet cannot claim.
    """
    out: list[tuple[str, dict[str, Any]]] = []
    for statement in statements:
        for call in _iter_calls(statement):
            func = call.func
            if not isinstance(func, ast.Name) or func.id not in eligible:
                continue
            if call.args:
                continue
            kwargs: dict[str, Any] = {}
            literal = True
            for keyword in call.keywords:
                if keyword.arg is None:
                    literal = False
                    break
                try:
                    kwargs[keyword.arg] = ast.literal_eval(keyword.value)
                except ValueError:
                    literal = False
                    break
            if literal:
                out.append((func.id, kwargs))
    return out


@dataclass
class SpeculationState:
    """Per-run speculation store shared between the `CodeMode` capability and its toolset.

    The capability's `wrap_run_event_stream` feeds streamed `run_code` argument deltas in;
    the toolset's dispatch path claims matching in-flight calls out. The capability creates one
    instance per run in `for_run`, and `CodeModeToolset` carries it by reference through its
    `for_run`/`for_run_step` copies.
    """

    allowlist: frozenset[str]
    """Original tool names the user declared side-effect free."""

    stats: SpeculationStats

    _step: _StepIngredients | None = field(default=None, init=False)
    _parts: dict[str, _PartWatch] = field(default_factory=dict[str, _PartWatch], init=False)
    _index_to_part: dict[int, str] = field(default_factory=dict[int, str], init=False)

    def stash_step(
        self,
        *,
        wrapped: AbstractToolset[Any],
        wrapped_tools: dict[str, ToolsetTool[Any]],
        sanitized_to_original: dict[str, str],
        callable_defs: dict[str, ToolDefinition],
        serialize: Callable[[Any], Any],
    ) -> None:
        """Record this step's dispatch ingredients; called by `CodeModeToolset.get_tools`.

        Eligibility is resolved here: allowlisted names, minus `sequential` tools, whose
        rendering as `def` gives execution an ordering contract that an early launch would break.
        """
        eligible = frozenset(
            name
            for name, tool_def in callable_defs.items()
            if not tool_def.sequential and sanitized_to_original.get(name, name) in self.allowlist
        )
        self._step = _StepIngredients(
            wrapped=wrapped,
            wrapped_tools=wrapped_tools,
            sanitized_to_original=sanitized_to_original,
            eligible=eligible,
            serialize=serialize,
        )

    # -- streaming side ---------------------------------------------------------------------

    def observe(self, event: AgentStreamEvent, ctx: RunContext[Any]) -> None:
        """Feed one stream event; launches tasks for any newly speculatable calls."""
        if isinstance(event, PartStartEvent):
            part = event.part
            if isinstance(part, ToolCallPart) and part.tool_name == _RUN_CODE_TOOL_NAME:
                watch = _PartWatch(tool_call_id=part.tool_call_id)
                if isinstance(part.args, str):
                    watch.args_text = part.args
                elif isinstance(part.args, dict):
                    watch.args_dict = dict(part.args)
                self._parts[part.tool_call_id] = watch
                self._index_to_part[event.index] = part.tool_call_id
                self._scan(watch, ctx)
            else:
                self._index_to_part.pop(event.index, None)
        elif isinstance(event, PartDeltaEvent):
            part_id = self._index_to_part.get(event.index)
            delta = event.delta
            if part_id is None or not isinstance(delta, ToolCallPartDelta):
                return
            watch = self._parts[part_id]
            if isinstance(delta.args_delta, str):
                watch.args_text += delta.args_delta
            elif isinstance(delta.args_delta, dict):
                watch.args_dict = {**(watch.args_dict or {}), **delta.args_delta}
            self._scan(watch, ctx)

    def _scan(self, watch: _PartWatch, ctx: RunContext[Any]) -> None:
        step = self._step
        if step is None or not step.eligible:
            return
        if watch.args_dict is not None:
            code = watch.args_dict.get('code')
            if not isinstance(code, str):
                return
        else:
            maybe_code = _decode_partial_code(watch.args_text)
            if maybe_code is None:
                return
            code = maybe_code
        fresh, watch.consumed_statements = _closed_statements(code, watch.consumed_statements)
        if not fresh:
            return
        for sandbox_name, kwargs in _literal_calls(fresh, step.eligible):
            if watch.launched >= _MAX_SPECULATIONS_PER_PART:
                return
            self._launch(watch, step, sandbox_name, kwargs, ctx)

    def _launch(
        self,
        watch: _PartWatch,
        step: _StepIngredients,
        sandbox_name: str,
        kwargs: dict[str, Any],
        ctx: RunContext[Any],
    ) -> None:
        original_name = step.sanitized_to_original.get(sandbox_name, sandbox_name)
        parent_manager = ctx.tool_manager
        tool_manager = ToolManager(
            toolset=step.wrapped,
            root_capability=parent_manager.root_capability if parent_manager is not None else None,
            ctx=ctx,
            tools=step.wrapped_tools,
        )
        watch.launched += 1
        self.stats.launched += 1
        provisional_id = f'{watch.tool_call_id}__spec_{watch.launched}'
        task = asyncio.ensure_future(
            _run_speculative(tool_manager, original_name, provisional_id, kwargs, step.serialize)
        )
        call = SpeculativeCall(sandbox_name=sandbox_name, original_name=original_name, kwargs=kwargs, task=task)
        self.calls_for(watch, sandbox_name, kwargs).append(call)

    def calls_for(self, watch: _PartWatch, sandbox_name: str, kwargs: dict[str, Any]) -> deque[SpeculativeCall]:
        return watch.calls.setdefault(_canonical_key(sandbox_name, kwargs), deque())

    # -- execution side ---------------------------------------------------------------------

    def claim(self, parent_tool_call_id: str, sandbox_name: str, kwargs: dict[str, Any]) -> SpeculativeCall | None:
        """Pop the oldest in-flight launch matching this dispatch, if any."""
        watch = self._parts.get(parent_tool_call_id)
        if watch is None:
            return None
        queue = watch.calls.get(_canonical_key(sandbox_name, kwargs))
        if not queue:
            return None
        return queue.popleft()

    async def evict_part(self, parent_tool_call_id: str) -> None:
        """Drop unclaimed launches for one executed `run_code` part.

        Called when the snippet finishes (successfully or not): whatever was not claimed --
        wrong-branch conditionals, rewritten lines -- is garbage for this part, and a retry
        arrives under a fresh tool call id.
        """
        watch = self._parts.pop(parent_tool_call_id, None)
        if watch is not None:
            await _cancel_watch(watch, self.stats)

    async def close(self) -> None:
        """Run-end cleanup: cancel every launch no snippet ever claimed."""
        parts, self._parts = self._parts, {}
        self._index_to_part.clear()
        for watch in parts.values():
            await _cancel_watch(watch, self.stats)


async def _cancel_watch(watch: _PartWatch, stats: SpeculationStats) -> None:
    tasks: list[asyncio.Task[_SpecOutcome]] = []
    for queue in watch.calls.values():
        for call in queue:
            call.task.cancel()
            tasks.append(call.task)
            stats.evicted += 1
    if tasks:
        # Await the cancellations so dispatched work has fully unwound before the run moves on,
        # mirroring `MontyExecutor.run`'s cleanup. Outcomes are deliberately discarded.
        await asyncio.gather(*tasks, return_exceptions=True)


async def _run_speculative(
    tool_manager: ToolManager[Any],
    original_name: str,
    provisional_id: str,
    kwargs: dict[str, Any],
    serialize: Callable[[Any], Any],
) -> _SpecOutcome:
    """Run one early launch through the same nested-manager path as a cold sandbox call.

    Mirrors the essential behavior of the cold dispatch in `CodeModeToolset.call_tool`
    (`run_tool_call`), settling into a `_SpecOutcome` instead of recording message parts: the
    provisional tool call id is replaced at adoption, when the executing snippet's own id and
    call counter exist. Failures settle rather than raise so an unclaimed failed launch never
    surfaces as an unretrieved task exception.
    """
    from pydantic_ai.exceptions import ApprovalRequired, CallDeferred, UserError
    from pydantic_ai.messages import ToolReturn
    from pydantic_ai.tools import ToolDenied

    call_part = ToolCallPart(tool_name=original_name, args=kwargs, tool_call_id=provisional_id)
    try:
        result = await tool_manager.handle_call(call_part, wrap_validation_errors=False)
    except (CallDeferred, ApprovalRequired) as e:
        return _SpecOutcome(
            error=UserError(
                f'Tool {original_name!r} raised {type(e).__name__} inside code mode, '
                'but no `HandleDeferredToolCalls` capability resolved it. Add a handler '
                'capability on the agent so deferred and approval-required calls can '
                'be resolved inline.'
            )
        )
    except Exception as e:
        return _SpecOutcome(error=e)

    if isinstance(result, ToolDenied):
        return _SpecOutcome(
            error=RuntimeError(f'Tool {original_name!r} call denied: {result.message}'),
            denied_message=result.message,
        )

    metadata: Any = None
    if isinstance(result, ToolReturn):
        metadata = result.metadata
        result = result.return_value
    return _SpecOutcome(content=result, serialized=serialize(result), metadata=metadata)
