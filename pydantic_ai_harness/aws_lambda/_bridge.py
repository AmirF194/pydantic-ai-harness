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

The active bridge is published in a `ContextVar` rather than passed around, which is what lets
`LambdaDurability` be attached at agent construction (the point at which it can wrap toolsets)
instead of per invocation.
"""

from __future__ import annotations

import asyncio
import contextvars
import threading
from collections.abc import Callable, Coroutine
from concurrent.futures import Future
from dataclasses import dataclass
from queue import Queue
from typing import TYPE_CHECKING, Any, Generic, Protocol, TypeVar

from pydantic_ai.exceptions import UserError

if TYPE_CHECKING:
    from aws_durable_execution_sdk_python.config import StepConfig

T = TypeVar('T')

_ENGINE_NAME = 'AWS Lambda'


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
    valid between invocations, which a fresh loop per invocation would invalidate.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None

    def get(self) -> asyncio.AbstractEventLoop:
        with self._lock:
            loop = self._loop
            if loop is None or loop.is_closed():
                loop = asyncio.new_event_loop()
                threading.Thread(target=loop.run_forever, daemon=True, name='pydantic-ai-lambda-agent').start()
                self._loop = loop
            return loop


_agent_loop = _AgentLoop()


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
        self._lock = threading.Lock()
        self._step_in_flight = False

    async def run_step(
        self,
        name: str,
        operation: Callable[[], Coroutine[Any, Any, T]],
        config: StepConfig | None = None,
    ) -> T:
        """Checkpoint `operation` as a durable step. Runs on the agent loop."""
        with self._lock:
            if self._step_in_flight:
                raise UserError(
                    f'A {_ENGINE_NAME} durable step was requested while another step is still running. '
                    'Lambda durable steps cannot be nested, and the handler thread is blocked until the '
                    'outer step returns, so this would deadlock. This usually means a tool starts a '
                    'nested agent run that also has `LambdaDurability` attached; drop the capability '
                    'from the nested agent, or opt the tool out of checkpointing with '
                    "`metadata={'aws_lambda': False}`."
                )
            self._step_in_flight = True

        loop = asyncio.get_running_loop()
        future: asyncio.Future[T] = loop.create_future()

        def reply(succeeded: bool, value: Any) -> None:
            if succeeded:
                loop.call_soon_threadsafe(_set_result_if_pending, future, value)
            else:
                loop.call_soon_threadsafe(_set_exception_if_pending, future, value)

        def body(_step_context: Any) -> T:
            # Runs on the handler thread, inside `context.step(...)`. Hand the real work back to
            # the agent loop and block until it finishes.
            result: Future[T] = Future()
            step_context = contextvars.copy_context()

            def schedule() -> None:
                try:
                    # Creating the task inside `step_context` makes it the task's context, which
                    # `create_task(context=...)` would do directly on 3.11+.
                    task: asyncio.Task[T] = step_context.run(lambda: loop.create_task(operation()))
                except BaseException as exc:  # pragma: no cover - task creation failing is not reproducible
                    result.set_exception(exc)
                    return
                task.add_done_callback(lambda finished: _forward(finished, result))

            loop.call_soon_threadsafe(schedule)
            return result.result()

        self._queue.put(
            _StepRequest(name=name, body=body, config=config, reply=reply, context=contextvars.copy_context())
        )
        try:
            return await future
        finally:
            with self._lock:
                self._step_in_flight = False

    def finish(self, result: Any = None, error: BaseException | None = None) -> None:
        self._queue.put(_Finished(result=result, error=error))

    def consume(self) -> Any:
        """Run queued steps on the handler thread until the agent run finishes."""
        while True:
            item = self._queue.get()
            if isinstance(item, _Finished):
                if item.error is not None:
                    raise item.error
                return item.result
            try:
                value = item.context.run(self._context.step, item.body, name=item.name, config=item.config)
            except BaseException as exc:
                # The step failed even after Lambda applied its retry policy. Hand the error to
                # the agent so the run can surface it, rather than aborting the handler here.
                item.reply(False, exc)
            else:
                item.reply(True, value)


def _forward(task: asyncio.Task[T], target: Future[T]) -> None:
    exc = task.exception()
    if exc is not None:
        target.set_exception(exc)
    else:
        target.set_result(task.result())


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


def run_durable(agent_run: Callable[[], Coroutine[Any, Any, T]], *, context: DurableStepContext) -> T:
    """Run an async agent call from a synchronous Lambda durable handler.

    Hosts `agent_run()` on a background event loop and services its durable steps on the calling
    (handler) thread, so every `context.step(...)` is created in one continuous sequence on the
    thread Lambda invoked. Returns whatever `agent_run()` returns.

    Args:
        agent_run: Callable returning the coroutine to run, e.g. `lambda: agent.run(prompt)`.
            It is called once per handler invocation, including each replay.
        context: The `DurableContext` the durable handler was invoked with.

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
            f'called from a running event loop. An {_ENGINE_NAME} durable handler is synchronous, so '
            'call it directly from the handler; if you already have an event loop, await the agent '
            'run instead.'
        )
    if _active_bridge.get() is not None:
        raise UserError(
            f'`run_durable()` is already active on this thread. An {_ENGINE_NAME} durable handler runs '
            'one agent run at a time; call `run_durable()` once per handler invocation.'
        )

    bridge = StepBridge(context)
    token = _active_bridge.set(bridge)
    try:
        loop = _agent_loop.get()
        run_context = contextvars.copy_context()

        async def run() -> None:
            try:
                bridge.finish(result=await agent_run())
            except BaseException as exc:
                bridge.finish(error=exc)

        def schedule_run() -> None:
            run_context.run(lambda: loop.create_task(run()))

        loop.call_soon_threadsafe(schedule_run)
        return bridge.consume()
    finally:
        _active_bridge.reset(token)
