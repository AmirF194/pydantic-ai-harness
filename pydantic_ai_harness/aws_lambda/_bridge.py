"""Bridge between Lambda's synchronous durable API and the async agent loop.

`DurableContext.step()` is synchronous, must be called from the handler's own thread, and
blocks until the step body returns. An agent run is async. The bridge connects them with two
threads and a queue:

- the handler thread runs `run_durable`, which drains a queue of step requests and calls
  `context.step(...)` for each one, so every step is created on the handler's thread in one
  continuous sequence;
- a background thread runs a persistent event loop that hosts the agent run and the actual
  model and tool calls.

A step body dispatched on the handler thread schedules its async operation back onto the agent
loop and blocks on the result, so the loop stays free while the handler thread waits.

The active bridge is published in a `ContextVar` rather than passed around, so the capability
does not have to be handed a `DurableContext` per invocation.

The invariant the bridge holds: **every queued step request is resolved exactly once**. The
handler thread blocks on the result of the step it is servicing, so a request that is never
resolved wedges the invocation until the function times out.
"""

from __future__ import annotations

import asyncio
import contextvars
import threading
from collections.abc import Callable, Coroutine
from concurrent.futures import Future
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import dataclass
from queue import Queue
from typing import TYPE_CHECKING, Any, Generic, Protocol, TypeVar

from pydantic_ai.exceptions import UserError

if TYPE_CHECKING:
    from aws_durable_execution_sdk_python.config import StepConfig

T = TypeVar('T')

ENGINE_NAME = 'AWS Lambda'

DEFAULT_CANCEL_TIMEOUT_SECONDS = 5.0
"""Default for `run_durable(cancel_timeout=...)`: how long to wait for an abandoned run to unwind.

Bounded so a tool whose cleanup hangs cannot wedge the handler, which is the failure mode the
bridge exists to avoid everywhere else. The value is a heuristic -- long enough for ordinary
`__aexit__`/`finally` cleanup (closing an HTTP client, cancelling a task group), short enough to
leave the handler room to return. Raise it for a workload with genuinely slow cleanup; the wait
timing out is handled rather than ignored, so the ceiling is a latency knob, not a correctness one.
"""

_LOOP_LIVENESS_POLL_SECONDS = 5.0
"""How often the handler thread re-checks that the agent loop is still alive while blocked.

A step body blocks the handler thread on a result the agent loop will deliver. Steps can
legitimately run for minutes, so the wait cannot simply time out; instead it wakes periodically to
check that the loop that owes it a result still exists.
"""


class AgentLoopGone(BaseException):
    """The agent loop stopped while a durable step was in flight, so its result can never arrive.

    A `BaseException` rather than an `Exception` on purpose: `consume()` routes ordinary step
    failures back into the agent run so it can handle them, and that is precisely what cannot work
    here -- the loop that would receive them is the thing that is gone. Like the SDK's own control
    flow, this has to leave the handler instead.
    """


class DurableStepContext(Protocol):
    """The part of `DurableContext` this package uses.

    Structural so the durable handler's real `DurableContext` satisfies it without this module
    depending on the concrete class, and so tests can supply a recording stand-in.
    """

    def step(  # pragma: no cover - structural declaration, never executed
        self,
        func: Callable[[Any], Any],
        name: str | None = None,
        config: StepConfig | None = None,
    ) -> Any: ...


class _AgentLoop:
    """A background event loop reused across invocations of a warm execution environment.

    Reusing it keeps loop-bound async resources (a provider's cached HTTP client, for example)
    valid between invocations, which a fresh loop per invocation would invalidate. The tradeoff is
    that a run abandoned mid-flight would otherwise survive into the next invocation, so
    `run_durable` cancels the run it started before returning, and retires the loop if that
    cancellation does not finish in time.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None

    def get(self) -> asyncio.AbstractEventLoop:
        """A loop that is confirmed to be running.

        `is_closed()` alone is not enough to decide a loop is reusable: a stopped-but-open loop
        answers `False` to it while still never running a callback, so `run_durable` would queue
        `schedule_run` on a loop that can never execute it and then block in `consume()` until
        Lambda timed the function out. Hence the `is_running()` check -- and hence, in turn,
        waiting for a freshly-built loop to actually start before returning it, so that check
        cannot race a loop whose thread has not reached `run_forever` yet.
        """
        with self._lock:
            loop = self._loop
            if loop is not None and not loop.is_closed() and loop.is_running():
                return loop
            loop = asyncio.new_event_loop()
            running = threading.Event()

            def run(loop: asyncio.AbstractEventLoop = loop) -> None:
                asyncio.set_event_loop(loop)
                loop.call_soon(running.set)
                loop.run_forever()

            threading.Thread(target=run, daemon=True, name='pydantic-ai-lambda-agent').start()
            running.wait()
            self._loop = loop
            return loop

    def retire(self, loop: asyncio.AbstractEventLoop) -> None:
        """Stop handing `loop` to later invocations.

        Called when an abandoned run outlives the cancellation wait. The loop is deliberately left
        running: the run is still unwinding on it, and stopping it out from under a `finally` or
        `__aexit__` would strand whatever that cleanup holds. Dropping the reference means the next
        invocation builds a fresh loop with its own loop-bound resources, so the leftover cleanup
        cannot reach into the execution that follows it -- which is the whole reason for the wait.
        """
        with self._lock:
            if self._loop is loop:
                self._loop = None


_agent_loop = _AgentLoop()

_in_step_body: contextvars.ContextVar[bool] = contextvars.ContextVar(
    'pydantic_ai_harness_aws_lambda_in_step_body', default=False
)
"""Set in the context a step's operation runs under, so a nested step can be detected.

A step body blocks the handler thread inside `context.step(...)`, so a step requested from within
another step's operation could never be serviced. Concurrent *sibling* steps are fine: they queue,
and the handler thread runs them one at a time.
"""


@dataclass
class _StepRequest:
    name: str
    body: Callable[[Any], Any]
    config: StepConfig | None
    reply: Callable[[bool, Any], None]
    context: contextvars.Context


@dataclass
class _Finished(Generic[T]):
    result: T | None
    error: BaseException | None


class StepBridge:
    """Runs durable steps on the handler thread on behalf of the agent loop."""

    def __init__(self, context: DurableStepContext) -> None:
        self._context = context
        self._queue: Queue[_StepRequest | _Finished[Any]] = Queue()
        # Serialises step requests so their queue order -- and so the order Lambda assigns
        # checkpoint identity in -- is first-come, rather than depending on how the event loop
        # interleaves concurrent callers (two MCP servers being listed in parallel, say).
        self._order = asyncio.Lock()

    async def run_step(
        self,
        name: str,
        operation: Callable[[], Coroutine[Any, Any, T]],
        config: StepConfig | None = None,
    ) -> T:
        """Checkpoint `operation` as a durable step. Runs on the agent loop."""
        if _in_step_body.get():
            raise UserError(
                f'A durable step was requested from inside another {ENGINE_NAME} durable step. Lambda '
                'durable steps cannot be nested, and the handler thread is blocked servicing the outer '
                'step, so the inner one could never run. This usually means a tool starts a nested agent '
                'run that also has `AWSLambdaDurability` attached; drop the capability from the nested '
                "agent, or opt the tool out of checkpointing with `metadata={'aws_lambda': False}`."
            )

        loop = asyncio.get_running_loop()

        def body(_step_context: Any) -> T:
            # Runs on the handler thread, inside `context.step(...)`. Hand the real work back to
            # the agent loop and block until it finishes.
            result: Future[T] = Future()
            step_context = contextvars.copy_context()
            step_context.run(_in_step_body.set, True)

            def schedule() -> None:
                try:
                    # Creating the task inside `step_context` makes it the task's context, which is
                    # what `create_task(context=...)` does on 3.11+, spelled so it also type-checks
                    # against the repo's 3.10 target.
                    task: asyncio.Task[T] = step_context.run(lambda: loop.create_task(operation()))
                except BaseException as exc:  # pragma: no cover - task creation failing is not reproducible
                    result.set_exception(exc)
                    return
                task.add_done_callback(lambda finished: _forward(finished, result))

            loop.call_soon_threadsafe(schedule)
            while True:
                try:
                    return result.result(timeout=_LOOP_LIVENESS_POLL_SECONDS)
                except FutureTimeoutError:
                    # A step can legitimately run for minutes, so the wait itself must not time
                    # out. What it must not do is block forever on a loop that is no longer there
                    # to deliver the result: the loop is what runs the done-callback, so if it has
                    # stopped, nothing will ever resolve `result`.
                    if loop.is_running() or result.done():
                        continue
                    raise AgentLoopGone(
                        f'The {ENGINE_NAME} agent event loop stopped while durable step {name!r} was in '
                        'flight, so its result can never arrive. This should not happen; please report '
                        'it at https://github.com/pydantic/pydantic-ai-harness/issues.'
                    ) from None

        async with self._order:
            future: asyncio.Future[T] = loop.create_future()

            def reply(succeeded: bool, value: Any) -> None:
                if succeeded:
                    loop.call_soon_threadsafe(_set_result_if_pending, future, value)
                else:
                    loop.call_soon_threadsafe(_set_exception_if_pending, future, value)

            self._queue.put(
                _StepRequest(name=name, body=body, config=config, reply=reply, context=contextvars.copy_context())
            )
            return await future

    def finish(self, result: Any = None, error: BaseException | None = None) -> None:
        self._queue.put(_Finished(result=result, error=error))

    def consume(self) -> Any:
        """Run queued steps on the handler thread until the agent run finishes.

        A `BaseException` that is not an `Exception` is the SDK's own control flow, most importantly
        `SuspendExecution`, which is how a step retry ends the invocation so Lambda can re-invoke it
        later. Those have to reach the SDK's handler wrapper unchanged, so they propagate out of here
        rather than being routed into the agent, and the queue stops being serviced.
        """
        while True:
            item = self._queue.get()
            if isinstance(item, _Finished):
                if item.error is not None:
                    raise item.error
                return item.result
            try:
                value = item.context.run(self._context.step, item.body, name=item.name, config=item.config)
            except Exception as exc:
                # An ordinary step failure, already past the SDK's retry policy. Hand it to the agent
                # so the run can surface or handle it.
                item.reply(False, exc)
            except BaseException as exc:
                # SDK control flow (suspension, interruption). Resolve the waiting step so the agent
                # task is not left on a future, then let it out of the handler untouched.
                item.reply(False, exc)
                raise
            else:
                item.reply(True, value)


def _forward(task: asyncio.Task[T], target: Future[T]) -> None:
    """Resolve `target` from a finished task, without ever leaving it unresolved.

    `Task.exception()` *raises* `CancelledError` for a cancelled task, so the naive spelling lets an
    exception escape this done-callback and strands `target`, wedging the handler thread.
    """
    try:
        if task.cancelled():
            target.set_exception(asyncio.CancelledError())
            return
        exc = task.exception()
        if exc is not None:
            target.set_exception(exc)
        else:
            target.set_result(task.result())
    except BaseException as exc:  # pragma: no cover - defensive: the resolve-exactly-once invariant
        target.set_exception(exc)


def _set_result_if_pending(future: asyncio.Future[T], value: T) -> None:
    if not future.done():  # pragma: no branch - the future is only resolved here
        future.set_result(value)


def _set_exception_if_pending(future: asyncio.Future[T], error: BaseException) -> None:
    if not future.done():  # pragma: no branch - the future is only resolved here
        future.set_exception(error)


_active_bridge: contextvars.ContextVar[StepBridge | None] = contextvars.ContextVar(
    'pydantic_ai_harness_aws_lambda_bridge', default=None
)


def current_bridge() -> StepBridge | None:
    return _active_bridge.get()


def in_durable_context() -> bool:
    return _active_bridge.get() is not None


def run_durable(
    agent_run: Callable[[], Coroutine[Any, Any, T]],
    *,
    context: DurableStepContext,
    cancel_timeout: float = DEFAULT_CANCEL_TIMEOUT_SECONDS,
) -> T:
    """Run an async agent call from a synchronous Lambda durable handler.

    Hosts `agent_run()` on a background event loop and services its durable steps on the calling
    (handler) thread, so every `context.step(...)` is created in one continuous sequence on the
    thread Lambda invoked. Returns whatever `agent_run()` returns.

    Args:
        agent_run: Callable returning the coroutine to run, e.g. `lambda: agent.run(prompt)`.
            It is called once per handler invocation, including each replay.
        context: The `DurableContext` the durable handler was invoked with.
        cancel_timeout: Seconds to wait for a run abandoned by a suspension or an error to finish
            unwinding before returning. Raise it for a workload whose cleanup is genuinely slow.
            Exceeding it is safe: the background event loop is retired instead of reused, so the
            leftover cleanup cannot overlap the next warm invocation.

    Example:
        ```python {test="skip"}
        @durable_execution
        def handler(event: dict[str, Any], context: DurableContext) -> str:
            result = run_durable(lambda: agent.run(event['prompt']), context=context)
            return result.output
        ```
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        pass
    else:
        raise UserError(
            '`run_durable()` blocks the calling thread until the agent run finishes, so it cannot be '
            f'called from a running event loop. An {ENGINE_NAME} durable handler is synchronous, so '
            'call it directly from the handler; if you already have an event loop, await the agent '
            'run instead.'
        )
    if _active_bridge.get() is not None:
        raise UserError(
            f'`run_durable()` is already active on this thread. An {ENGINE_NAME} durable handler runs '
            'one agent run at a time; call `run_durable()` once per handler invocation.'
        )

    bridge = StepBridge(context)
    token = _active_bridge.set(bridge)
    loop = _agent_loop.get()
    run_context = contextvars.copy_context()
    tasks: list[asyncio.Task[None]] = []
    started = threading.Event()

    async def run() -> None:
        try:
            bridge.finish(result=await agent_run())
        except BaseException as exc:
            bridge.finish(error=exc)

    def schedule_run() -> None:
        try:
            tasks.append(run_context.run(lambda: loop.create_task(run())))
        except BaseException as exc:  # pragma: no cover - task creation failing is not reproducible
            # Nothing else will ever call `bridge.finish()` if the run never started, so
            # `consume()` below would block until Lambda timed the function out. Resolve it here.
            bridge.finish(error=exc)
        finally:
            # Always, so the `finally` below cannot wait on a flag a failed scheduling never set.
            started.set()

    try:
        loop.call_soon_threadsafe(schedule_run)
        return bridge.consume()
    finally:
        # The loop outlives the invocation, so a run abandoned by a suspension or an error escaping
        # the handler would otherwise keep running into the next warm invocation. Wait for it to
        # finish unwinding: an agent coroutine's `finally`/`__aexit__` cleanup runs during
        # cancellation, and letting that overlap the next invocation would touch shared provider
        # resources after the execution it belonged to was abandoned.
        started.wait()
        _active_bridge.reset(token)
        if tasks:  # pragma: no branch - only empty when scheduling itself failed
            task = tasks[0]
            finished = threading.Event()

            def cancel_and_notify() -> None:
                if task.done():
                    finished.set()
                    return
                task.add_done_callback(lambda _: finished.set())
                task.cancel()

            loop.call_soon_threadsafe(cancel_and_notify)
            if not finished.wait(timeout=cancel_timeout):
                # Cleanup is still running and we are out of budget to wait for it. Returning while
                # it holds the shared loop is the leak this wait exists to prevent, so give the loop
                # up: the next invocation gets a fresh one instead of inheriting this one's
                # half-torn-down state.
                _agent_loop.retire(loop)
