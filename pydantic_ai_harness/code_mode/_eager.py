"""Eager execution for `CodeMode`: feed streamed `run_code` statements into the live REPL.

The second tier of streaming code execution. Speculation (`_speculation.py`) shadow-launches
allowlisted pure calls and reconciles them with claim/evict; eager execution instead runs the
real program, in order, in the real session, as each top-level statement closes in the stream.
Nothing is predicted, so nothing can miss or be wasted -- and nothing can be rolled back:
side effects land before the tool call is committed, which is why the tier is opt-in.

A failed statement leaves the session exactly as a failed whole-snippet feed does today:
assignments made before the failing line persist and the error surfaces to the model as the
`run_code` result. The only semantic difference from non-eager execution is timing.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

from pydantic_ai import RunContext
from pydantic_ai.messages import (
    AgentStreamEvent,
    CapabilityEvent,
    PartDeltaEvent,
    PartStartEvent,
    ToolCallPart,
    ToolCallPartDelta,
    ToolReturnPart,
)

from ._events import SpeculativeCodeUpdateEvent
from ._speculation import _closed_statements, _decode_partial_code  # pyright: ignore[reportPrivateUsage]

if TYPE_CHECKING:
    from ._toolset import CodeModeToolset


@dataclass
class _EagerPart:
    """Accumulated state for one streamed `run_code` tool call part under eager execution."""

    tool_call_id: str
    args_text: str = ''
    args_dict: dict[str, Any] | None = None
    consumed: int = 0
    """Top-level statements handed to the feed pump so far."""

    scanned_newlines: int = -1
    """Newline count at the last scan; statements only close on line boundaries."""

    fed_line_count: int = 0
    """Source lines covered by pumped statements; the executed prefix ends here."""

    fed_prefix: str = ''
    """Exact source prefix handed to the pump, for divergence detection at execution."""

    queue: list[str] = field(default_factory=list[str])
    """Statement sources waiting for the pump, in program order."""

    pump: asyncio.Task[None] | None = None
    error: BaseException | None = None
    feed_count: int = 0
    output: str = ''
    """Concatenated print output from completed fragment feeds."""

    nested_calls: dict[str, ToolCallPart] = field(default_factory=dict[str, ToolCallPart])
    nested_returns: dict[str, ToolReturnPart] = field(default_factory=dict[str, ToolReturnPart])


class EagerState:
    """Watch streamed `run_code` parts and feed closed statements into the REPL serially.

    One instance per run, owned by the `CodeMode` clone and bound to its `CodeModeToolset`.
    The stream watcher enqueues statement sources; a single pump task per part feeds them
    through the toolset's normal `run_code` pipeline, so budget accounting, error mapping,
    and tool hooks behave exactly as a model-issued snippet would.
    """

    def __init__(self) -> None:
        self._parts: dict[str, _EagerPart] = {}
        self._toolset: CodeModeToolset[Any] | None = None

    def bind(self, toolset: CodeModeToolset[Any]) -> None:
        """Attach the owning toolset; feeds go through its `run_code` pipeline."""
        self._toolset = toolset

    # -- stream side ------------------------------------------------------------------------

    async def observe(self, event: AgentStreamEvent, ctx: RunContext[Any]) -> list[CapabilityEvent]:
        """Track one stream event, enqueueing newly closed statements for the pump."""
        produced: list[CapabilityEvent] = []
        if isinstance(event, PartStartEvent) and isinstance(event.part, ToolCallPart):
            if event.part.tool_name != 'run_code':
                return produced
            part_id = event.part.tool_call_id
            watch = self._parts.setdefault(part_id, _EagerPart(tool_call_id=part_id))
            args = event.part.args
            if isinstance(args, str):
                watch.args_text += args
            elif isinstance(args, dict):
                watch.args_dict = args
            self._scan(watch, ctx, produced)
        elif isinstance(event, PartDeltaEvent) and isinstance(event.delta, ToolCallPartDelta):
            watch = self._find_watch(event.delta.tool_call_id)
            if watch is None:
                return produced
            delta = event.delta.args_delta
            if isinstance(delta, str):
                watch.args_text += delta
            elif isinstance(delta, dict):
                merged = dict(watch.args_dict or {})
                merged.update(delta)
                watch.args_dict = merged
            self._scan(watch, ctx, produced)
        return produced

    def _find_watch(self, tool_call_id: str | None) -> _EagerPart | None:
        if tool_call_id is not None:
            return self._parts.get(tool_call_id)
        if len(self._parts) == 1:  # pragma: no cover - delta without id, single-part fallback
            return next(iter(self._parts.values()))
        return None

    def _current_code(self, watch: _EagerPart) -> str | None:
        if watch.args_dict is not None:
            code = watch.args_dict.get('code')
            return code if isinstance(code, str) else None
        return _decode_partial_code(watch.args_text)

    def _scan(self, watch: _EagerPart, ctx: RunContext[Any], produced: list[CapabilityEvent]) -> None:
        code = self._current_code(watch)
        if code is None:
            return
        newlines = code.count('\n')
        if newlines == watch.scanned_newlines:
            produced.append(
                SpeculativeCodeUpdateEvent(tool_call_id=watch.tool_call_id, code=code, closed_statements=watch.consumed)
            )
            return
        watch.scanned_newlines = newlines
        fresh, watch.consumed = _closed_statements(code, watch.consumed)
        lines = code.split('\n')
        for statement in fresh:
            end = statement.end_lineno or statement.lineno
            source = '\n'.join(lines[watch.fed_line_count : end])
            watch.queue.append(source)
            watch.fed_line_count = end
        watch.fed_prefix = '\n'.join(lines[: watch.fed_line_count])
        produced.append(
            SpeculativeCodeUpdateEvent(tool_call_id=watch.tool_call_id, code=code, closed_statements=watch.consumed)
        )
        if watch.queue:
            self._ensure_pump(watch, ctx)

    def _ensure_pump(self, watch: _EagerPart, ctx: RunContext[Any]) -> None:
        if watch.pump is None or watch.pump.done():
            watch.pump = asyncio.ensure_future(self._run_pump(watch, ctx))

    async def _run_pump(self, watch: _EagerPart, ctx: RunContext[Any]) -> None:
        """Feed queued statements one at a time; an error stops the part for good."""
        toolset = self._toolset
        assert toolset is not None, 'EagerState must be bound to its CodeModeToolset'
        while watch.queue and watch.error is None:
            source = watch.queue.pop(0)
            watch.feed_count += 1
            feed_ctx = replace(ctx, tool_call_id=f'{watch.tool_call_id}~e{watch.feed_count}', tool_name='run_code')
            try:
                await toolset.feed_eager_fragment(watch, source, feed_ctx)
            except BaseException as e:
                watch.error = e
                watch.queue.clear()

    # -- execution side ---------------------------------------------------------------------

    def take(self, tool_call_id: str, code: str) -> _EagerPart | None:
        """Pop the watch for an executing part, tolerating provider-rewritten ids.

        Falls back to any part whose executed prefix matches the submitted code, since a
        re-keyed part is still the same program.
        """
        watch = self._parts.pop(tool_call_id, None)
        if watch is not None:
            return watch
        for part_id, candidate in list(self._parts.items()):
            if code.startswith(candidate.fed_prefix):
                del self._parts[part_id]
                return candidate
        return None

    async def drain(self, watch: _EagerPart) -> None:
        """Wait until every enqueued statement has been fed (or the part errored).

        At most one pump exists per part: `_scan` enqueues before ensuring the pump, so a
        finished pump means an empty queue, and no new statements arrive once the part's
        arguments are complete.
        """
        if watch.pump is not None:
            await watch.pump

    async def close(self) -> None:
        """Cancel pumps and drop all watches at run end."""
        parts, self._parts = list(self._parts.values()), {}
        for watch in parts:
            pump = watch.pump
            if pump is None or pump.done():
                continue
            pump.cancel()
            try:
                await pump
            except (asyncio.CancelledError, Exception):  # pragma: no cover - cancellation race
                pass
