"""Budget windows, and the store keys they accumulate under.

A window decides the key a budget counts against and nothing else, so a window
rolls over by producing a different key rather than by anyone resetting a
counter. A new day is a new key.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Literal

from pydantic_ai.exceptions import UserError
from pydantic_ai.tools import RunContext
from typing_extensions import assert_never

Window = Literal['run', 'conversation', 'day', 'month', 'total']
"""The period a budget counts over."""

SEPARATOR = '|'
"""Joins name, scope, and bucket into a store key.

`|` rather than `:`, which appears inside model references and tenant
identifiers.
"""

_ANY_SCOPE = '*'

_TTLS: dict[Window, timedelta | None] = {
    'run': timedelta(hours=24),
    'conversation': timedelta(hours=24),
    'day': timedelta(hours=48),
    'month': timedelta(days=62),
    'total': None,
}


@dataclass(frozen=True, kw_only=True)
class Budget:
    """One spend window: what it limits, over what period, for whom.

    A budget with neither `usd` nor `tokens` is a pure counter: it accumulates
    and reports, and never stops a run. That is how per-tenant accounting with
    no cap is expressed.

    ```python
    from decimal import Decimal

    from pydantic_ai_harness.spend import Budget

    Budget(usd=Decimal('100'), window='day')
    Budget(usd=Decimal('10'), window='day', scope=lambda ctx: ctx.deps.tenant_id)
    Budget(window='month', name='accounting')  # counts, never blocks
    ```
    """

    usd: Decimal | None = None
    """Ceiling in US dollars. `None` means this budget does not limit spend."""

    tokens: int | None = None
    """Ceiling in total tokens. `None` means this budget does not limit tokens."""

    window: Window = 'day'
    """The period the ceiling applies to."""

    scope: Callable[[RunContext[Any]], str] | None = None
    """Partitions the counter -- per tenant, per user, per agent. `None` counts globally."""

    warn_at: float | None = None
    """Fraction of the ceiling past which `BudgetStatus.warning` is set. Never blocks."""

    name: str = 'default'
    """Distinguishes budgets sharing a window and scope. Part of the store key."""

    def __post_init__(self) -> None:
        """Reject configurations that would quietly misbehave rather than fail.

        A negative ceiling makes a budget exhausted before anything is spent, so
        the first request is refused with no way to tell that from a real
        overspend. A `warn_at` on a budget with no ceiling has nothing to be a
        fraction of, so it can never fire. Both read as configuration and behave
        as breakage, which is why they are errors here rather than surprises
        later.
        """
        if not self.name or SEPARATOR in self.name:
            raise UserError(f'Budget.name must be non-empty and must not contain {SEPARATOR!r}; got {self.name!r}.')
        if self.usd is not None and self.usd < 0:
            raise UserError(f'Budget.usd must not be negative; got {self.usd}.')
        if self.tokens is not None and self.tokens < 0:
            raise UserError(f'Budget.tokens must not be negative; got {self.tokens}.')
        if self.warn_at is not None:
            if not 0 < self.warn_at <= 1:
                raise UserError(f'Budget.warn_at must be a fraction in (0, 1]; got {self.warn_at!r}.')
            if not self.enforces:
                raise UserError(
                    f'Budget {self.name!r} sets warn_at but no `usd` or `tokens` ceiling, so the warning could '
                    'never fire. Add a ceiling, or drop warn_at to leave it a plain counter.'
                )

    @property
    def enforces(self) -> bool:
        """Whether this budget can refuse a request, rather than only counting."""
        return self.usd is not None or self.tokens is not None

    @property
    def ttl(self) -> timedelta | None:
        """How long a store may keep this window's counter after its last write."""
        return _TTLS[self.window]


def bucket(window: Window, ctx: RunContext[Any] | None, now: datetime) -> str | None:
    """The period identifier for `window`, or `None` when it needs a run and none was given.

    `run` and `conversation` are meaningless outside an agent run, which is what
    the `None` reports back to the caller.
    """
    match window:
        case 'run':
            return None if ctx is None else _run_identity(ctx.run_id, 'run')
        case 'conversation':
            return None if ctx is None else _run_identity(ctx.conversation_id, 'conversation')
        case 'day':
            return now.date().isoformat()
        case 'month':
            return f'{now.year:04d}-{now.month:02d}'
        case 'total':
            return 'total'
        case _:  # pragma: no cover - assert_never exhaustiveness guard
            assert_never(window)


def _run_identity(identity: str | None, window: Window) -> str:
    """The run or conversation id, refusing to silently share a bucket when it is absent."""
    if identity is None:
        raise UserError(
            f"A Budget with window='{window}' needs the run's {window} id, but this run reports none. "
            f'Use a time window, or run the agent through `Agent.run`, which always sets one.'
        )
    return identity


def scope_key(budget: Budget, ctx: RunContext[Any] | None, explicit: str | None) -> str:
    """The scope segment of a budget's store key.

    `explicit` is what `SpendGuard.status` was given, for use outside a run. It
    is ignored by a budget that declares no `scope`, since such a budget counts
    globally.
    """
    if budget.scope is None:
        return _ANY_SCOPE
    resolved = budget.scope(ctx) if ctx is not None else explicit
    if resolved is None:  # pragma: no cover - callers filter these budgets out first
        raise UserError(f'Budget {budget.name!r} declares a scope, which cannot be resolved without a run.')
    if not resolved or SEPARATOR in resolved:
        raise UserError(
            f'The scope of budget {budget.name!r} must be non-empty and must not contain '
            f'{SEPARATOR!r}; got {resolved!r}.'
        )
    return resolved


def store_key(budget: Budget, bucket_id: str, scope: str) -> str:
    """Join the parts of a budget's store key.

    `name` and `scope` are checked to be free of `SEPARATOR`, which is what
    makes a key identify exactly one `(name, scope, bucket)`: the first two
    separators delimit unambiguously, so whatever follows is the bucket. That
    is why `bucket_id` needs no check of its own even though a caller-supplied
    conversation id may contain anything.
    """
    return f'{budget.name}{SEPARATOR}{scope}{SEPARATOR}{bucket_id}'
