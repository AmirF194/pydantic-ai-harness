"""In-process execution loop for due schedules."""

from __future__ import annotations

import inspect
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Generic, Literal

import anyio
from pydantic_ai.agent import AbstractAgent
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.tools import AgentDepsT
from pydantic_ai.usage import UsageLimits

from pydantic_ai_harness.scheduling._store import ScheduleStore
from pydantic_ai_harness.scheduling._types import (
    CronTrigger,
    IntervalTrigger,
    OnceTrigger,
    Schedule,
    ScheduleResult,
    ScheduleResultCallback,
    next_run_time,
)

scheduled_run_var: ContextVar[str | None] = ContextVar('scheduled_run', default=None)
"""Id of the schedule running in the current context, or `None`."""


@dataclass(frozen=True)
class _Claim:
    schedule: Schedule
    execute: bool
    claimed_at: datetime


class ScheduleRunner(Generic[AgentDepsT]):
    """Execute due schedules against an agent.

    A runner assumes exclusive ownership of its store. Occurrences are advanced
    before agent execution for at-most-once behavior, and `stop()` drains runs
    already in flight before `run_until_stopped()` returns.
    """

    def __init__(
        self,
        agent: AbstractAgent[AgentDepsT, Any],
        *,
        deps: AgentDepsT,
        store: ScheduleStore | None = None,
        on_result: ScheduleResultCallback | None = None,
        tick_interval: float = 60.0,
        misfire_grace: timedelta = timedelta(minutes=10),
        run_timeout: float | None = None,
        usage_limits: UsageLimits | None = None,
    ) -> None:
        """Initialize a schedule runner.

        Args:
            agent: Agent used for each isolated scheduled run.
            deps: Dependencies passed to every agent run.
            store: Schedule store, or `None` to find the agent's `Scheduling` store.
            on_result: Optional sync or async outcome callback.
            tick_interval: Seconds between claims in the continuous loop.
            misfire_grace: Maximum lateness before one-shot schedules are missed.
            run_timeout: Optional wall-clock limit for one agent run.
            usage_limits: Default usage limits for schedules without their own limits.

        Raises:
            ValueError: If configuration is invalid or no unambiguous store can be found.
        """
        if tick_interval <= 0:
            raise ValueError('tick_interval must be greater than zero')
        if misfire_grace < timedelta(0):
            raise ValueError('misfire_grace must not be negative')
        if run_timeout is not None and run_timeout <= 0:
            raise ValueError('run_timeout must be greater than zero')
        self._agent = agent
        self._store = store or self._store_from_agent(agent)
        self._deps = deps
        self._on_result = on_result
        self._tick_interval = tick_interval
        self._misfire_grace = misfire_grace
        self._run_timeout = run_timeout
        self._usage_limits = usage_limits
        self._running: set[str] = set()
        self._stop_event = anyio.Event()

    @staticmethod
    def _store_from_agent(agent: AbstractAgent[AgentDepsT, Any]) -> ScheduleStore:
        from pydantic_ai_harness.scheduling._capability import Scheduling

        found: ScheduleStore | None = None
        multiple = False

        def inspect_capability(capability: AbstractCapability[AgentDepsT]) -> None:
            nonlocal found, multiple
            if isinstance(capability, Scheduling):
                if found is not None:
                    multiple = True
                else:
                    found = capability.resolved_store

        agent.root_capability.apply(inspect_capability)
        if multiple:
            raise ValueError('The agent has multiple Scheduling capabilities; pass `store=` explicitly.')
        if found is None:
            raise ValueError('No Scheduling capability was found on the agent; pass `store=` explicitly.')
        return found

    async def _claim_due(self, now: datetime) -> list[_Claim]:
        claims: list[_Claim] = []
        for schedule in await self._store.list():
            due_at = schedule.next_run_at
            if not schedule.enabled or due_at is None or due_at > now or schedule.id in self._running:
                continue
            self._running.add(schedule.id)
            try:
                overdue = now - due_at > self._misfire_grace
                if overdue and isinstance(schedule.trigger, OnceTrigger):
                    schedule.next_run_at = None
                    schedule.last_status = 'missed'
                    schedule.last_error = 'The runner was not running within the allowed grace period.'
                    await self._store.save(schedule)
                    claims.append(_Claim(schedule, execute=False, claimed_at=now))
                    continue

                if isinstance(schedule.trigger, (CronTrigger, IntervalTrigger)):
                    schedule.next_run_at = next_run_time(schedule.trigger, after=now, timezone=schedule.timezone)
                else:
                    schedule.next_run_at = None
                schedule.runs_completed += 1
                if schedule.max_runs is not None and schedule.runs_completed >= schedule.max_runs:
                    schedule.next_run_at = None
                await self._store.save(schedule)
                claims.append(_Claim(schedule, execute=True, claimed_at=now))
            except BaseException:
                self._running.discard(schedule.id)
                raise
        return claims

    async def _deliver(self, result: ScheduleResult) -> None:
        if self._on_result is None:
            return
        result.schedule.last_delivery_error = None
        try:
            callback_result = self._on_result(result)
            if inspect.isawaitable(callback_result):
                await callback_result
        except Exception as exc:
            result.schedule.last_delivery_error = f'{type(exc).__name__}: {exc}'[:1000]
        await self._store.save(result.schedule)

    async def _execute(self, claim: _Claim) -> ScheduleResult:
        schedule = claim.schedule
        started_at = datetime.now(timezone.utc)
        try:
            if not claim.execute:
                finished_at = datetime.now(timezone.utc)
                result = ScheduleResult(
                    schedule=schedule,
                    status='missed',
                    output=None,
                    error=schedule.last_error,
                    started_at=started_at,
                    finished_at=finished_at,
                    usage=None,
                )
                await self._deliver(result)
                return result

            token = scheduled_run_var.set(schedule.id)
            try:
                run = self._agent.run(
                    schedule.prompt,
                    deps=self._deps,
                    usage_limits=schedule.usage_limits or self._usage_limits,
                )
                if self._run_timeout is None:
                    agent_result = await run
                else:
                    with anyio.fail_after(self._run_timeout):
                        agent_result = await run
            except Exception as exc:
                status: Literal['success', 'error', 'missed'] = 'error'
                output = None
                error = f'{type(exc).__name__}: {exc}'[:1000]
                usage = None
            else:
                status = 'success'
                output = str(agent_result.output)
                error = None
                usage = agent_result.usage
            finally:
                scheduled_run_var.reset(token)

            schedule.last_run_at = claim.claimed_at
            schedule.last_status = status
            schedule.last_output = output if status == 'success' else schedule.last_output
            schedule.last_error = error
            await self._store.save(schedule)
            result = ScheduleResult(
                schedule=schedule,
                status=status,
                output=output,
                error=error,
                started_at=started_at,
                finished_at=datetime.now(timezone.utc),
                usage=usage,
            )
            await self._deliver(result)
            return result
        finally:
            self._running.discard(schedule.id)

    async def _execute_collect(self, claim: _Claim, results: list[ScheduleResult]) -> None:
        results.append(await self._execute(claim))

    async def tick(self, now: datetime | None = None) -> list[ScheduleResult]:
        """Claim and concurrently execute schedules due at `now`.

        This method does not require the continuous runner loop, so an external
        scheduler can drive it from system cron, a workflow engine, or serverless
        infrastructure.

        Args:
            now: UTC-aware claim time override for deterministic callers and tests.

        Returns:
            Results from every occurrence claimed by this tick.
        """
        reference = now or datetime.now(timezone.utc)
        results: list[ScheduleResult] = []
        claims = await self._claim_due(reference)
        async with anyio.create_task_group() as task_group:
            for claim in claims:
                task_group.start_soon(self._execute_collect, claim, results)
        return results

    async def run_until_stopped(self) -> None:
        """Claim on each interval until stopped, then drain in-flight runs."""
        async with anyio.create_task_group() as task_group:
            while not self._stop_event.is_set():
                claims = await self._claim_due(datetime.now(timezone.utc))
                for claim in claims:
                    task_group.start_soon(self._execute, claim)
                with anyio.move_on_after(self._tick_interval):
                    await self._stop_event.wait()

    def stop(self) -> None:
        """Request an idempotent graceful stop of the continuous loop."""
        self._stop_event.set()
