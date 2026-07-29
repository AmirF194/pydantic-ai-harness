"""Track what an agent spends, and stop it when a budget is gone.

`UsageLimits` in Pydantic AI caps tokens and requests for the duration of one
run. `SpendLimits` covers what that leaves: money, periods longer than a run,
partitioning by tenant or user, and a counter that several worker processes
share. It prices each response with
[`ModelResponse.cost()`][pydantic_ai.messages.ModelResponse.cost], adds it to
every configured window, and refuses the next request once a window is spent.

The gate is local and immediate. Provider usage APIs and observability backends
aggregate after the fact and are read by polling, so a number there moves only
once the requests behind it have already been made -- enough to reconcile a
counter, not enough to stop the request a runaway loop is about to make.
"""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Any, Literal, TypeGuard

from pydantic_ai.capabilities import AbstractCapability, CapabilityOrdering
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import ModelResponse
from pydantic_ai.tools import AgentDepsT, RunContext

from pydantic_ai_harness.spend._budget import Budget, bucket, scope_key, store_key
from pydantic_ai_harness.spend._exceptions import SpendLimitExceeded, UnpricedModelError
from pydantic_ai_harness.spend._snapshot import BudgetStatus, SpendSnapshot, Spent
from pydantic_ai_harness.spend._store import InMemorySpendStore, SpendStore, utc_now

if TYPE_CHECKING:
    from pydantic_ai.models import ModelRequestContext
    from pydantic_ai.toolsets import AgentToolset


SpendCallback = Callable[[SpendSnapshot], None | Awaitable[None]]
"""Called after each model response with what it cost and where the budgets stand."""

PriceFunc = Callable[[ModelResponse], Decimal | None]
"""Prices a response. Return `None` to fall back to the `genai-prices` registry."""

_RUN_SCOPED_WINDOWS = ('run', 'conversation')
_UNPRICED_POLICIES = frozenset({'zero', 'raise'})


@dataclass
class SpendLimits(AbstractCapability[AgentDepsT]):
    """Accumulate spend per window and refuse a request once a window is exhausted.

    ```python
    from decimal import Decimal

    from pydantic_ai import Agent
    from pydantic_ai_harness.spend import Budget, SpendLimits

    agent = Agent(
        'openai:gpt-5.4',
        capabilities=[SpendLimits(budgets=[Budget(usd=Decimal('100'), window='day')])],
    )
    ```

    With no budgets the capability only reports, through `on_spend`. Add a
    `Budget` with no ceiling to keep a running total that never blocks.

    What the gate guarantees: no request **starts** after a budget is
    exhausted. What it does not: that spend stays under the ceiling. The
    request that crosses the line completes, and concurrent runs can each pass
    the check before any of them records anything. This is a brake on a runaway
    loop, not an accounting ledger.

    State lives across runs on purpose, so `for_run` is left alone: a daily
    budget that reset every run would not be a daily budget. Per-run isolation
    comes from `Budget(window='run')`, whose key carries the run id.

    Durable execution: the hooks run in the workflow, while the model request
    itself is the activity. A store that talks over the network therefore reads
    and writes from workflow code, which Temporal replays. Use the in-process
    store there, and enforce a shared budget before starting the workflow with
    `exhausted()`. Temporal's workflow sandbox also restricts the clock these
    hooks read, so the workflow runner needs
    `SandboxRestrictions.default.with_passthrough_modules('pydantic_ai_harness')`.
    """

    budgets: Sequence[Budget] = ()
    """Windows to accumulate against, and which of them can refuse a request."""

    store: SpendStore = field(default_factory=InMemorySpendStore)
    """Where counters live. The default holds them for the lifetime of the process."""

    price: PriceFunc | None = None
    """Prices a response before the registry is consulted.

    Returning `None` falls through to `genai-prices`. This is the way to charge
    a self-hosted model, or a negotiated rate the public registry does not know.
    """

    on_spend: SpendCallback | None = None
    """Called with a `SpendSnapshot` after each response. May be sync or async."""

    on_unpriced: Literal['zero', 'raise'] = 'zero'
    """What to do when a response cannot be priced.

    `'zero'` counts it as free and increments `Spent.unpriced_requests`, so the
    gap shows up instead of disappearing. `'raise'` fails the run with
    `UnpricedModelError`. Tokens are counted either way, so a token ceiling
    still holds for a model the registry does not know.
    """

    expose_tools: bool = False
    """Offer the agent a `get_spend` tool.

    Off by default: a tool costs schema tokens on every request, and most
    applications want the number on a screen rather than in the model's context.
    """

    clock: Callable[[], datetime] = utc_now
    """Supplies the time that day and month windows are derived from.

    It does not reach a default-constructed `store`, which keeps its own `utc_now` for
    expiry. Both remain absolute instants, so a custom clock buckets on one and expires on
    the other; pass the same callable to the store when that matters.
    """

    def __post_init__(self) -> None:
        """Reject an `on_unpriced` that arrived as plain data and is not one of the two policies.

        Anything other than `'raise'` behaves as `'zero'`, so a typo in a spec
        would quietly turn unpriced responses free instead of failing the run.
        """
        # Budgets sharing a name, window and scope share a counter deliberately, which is how
        # one window carries a USD and a token ceiling. Two ceilings of the SAME kind on one
        # counter is not that: it reads as two independent limits and behaves as the smaller
        # one. Comparing against a single remembered budget per slot missed any collision a
        # later budget displaced, so the slot carries every attribute the counter key does.
        seen: set[tuple[str, str, str, int]] = set()
        for budget in self.budgets:
            for kind, ceiling in (('usd', budget.usd), ('tokens', budget.tokens)):
                if ceiling is None:
                    continue
                slot = (budget.name, kind, budget.window, id(budget.scope))
                if slot in seen:
                    raise UserError(
                        f'Two budgets named {budget.name!r} both set a `{kind}` ceiling on the same window and '
                        'scope, so they would share one counter and only the smaller would ever apply. '
                        'Give them different `name`s.'
                    )
                seen.add(slot)
        if self.on_unpriced not in _UNPRICED_POLICIES:
            raise UserError(
                f'SpendLimits.on_unpriced must be one of {sorted(_UNPRICED_POLICIES)}; got {self.on_unpriced!r}.'
            )

    @classmethod
    def get_serialization_name(cls) -> str | None:
        """Serialization name for agent-spec support."""
        return 'SpendLimits'

    def get_ordering(self) -> CapabilityOrdering:
        """Sit innermost, so a capability listed after this one cannot reject a billed response.

        `after_model_request` reaches innermost capabilities first. Anywhere else in the
        chain, a capability listed after this one -- which is how anyone would write it --
        can raise `ModelRetry` on a response that was already generated and billed, and the
        counter never sees it.

        This orders against non-innermost capabilities only. Innermost members are not
        ordered among themselves, and the one listed later runs `after_model_request`
        first, so another innermost capability (`TemporalDurability`, `InputGuardrail`)
        placed after this one can still reject a response before it is counted.
        """
        return CapabilityOrdering(position='innermost')

    def get_toolset(self) -> AgentToolset[AgentDepsT] | None:
        """Offer `get_spend` when `expose_tools` is set."""
        if not self.expose_tools:
            return None
        from pydantic_ai_harness.spend._toolset import build_toolset

        return build_toolset(self)

    async def before_model_request(
        self,
        ctx: RunContext[AgentDepsT],
        request_context: ModelRequestContext,
    ) -> ModelRequestContext:
        """Refuse the request if any budget with a ceiling is already spent."""
        read: dict[str, Spent] = {}
        for budget, key in self._keyed(ctx):
            if not budget.enforces:
                continue
            if key not in read:
                read[key] = await self.store.get(key)
            self._check(budget, read[key], ctx)
        return request_context

    async def after_model_request(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        request_context: ModelRequestContext,
        response: ModelResponse,
    ) -> ModelResponse:
        """Price the response, add it to every window, and report the result."""
        usd, priced = self._price_of(response)
        accrued: dict[str, Spent] = {}
        statuses: list[BudgetStatus] = []
        for budget, key in self._keyed(ctx):
            # Budgets sharing a name, window, and scope share a counter, which is
            # how one window carries both a USD and a token ceiling. Adding the
            # response once per budget would double-count it and halve them both.
            if key not in accrued:
                accrued[key] = await self.store.add(
                    key,
                    usd=usd,
                    tokens=response.usage.total_tokens,
                    requests=1,
                    unpriced=0 if priced else 1,
                    ttl=budget.ttl,
                )
            statuses.append(_status(budget, key, accrued[key]))

        if self.on_spend is not None:
            snapshot = SpendSnapshot(
                model=response.model_name,
                usage=response.usage,
                usd=usd,
                priced=priced,
                budgets=tuple(statuses),
            )
            result = self.on_spend(snapshot)
            if inspect.isawaitable(result):
                await result

        if not priced and self.on_unpriced == 'raise':
            # Raised last, after the store is updated and `on_spend` has seen the
            # response. The request happened and its tokens were really spent, so
            # dropping them would leave a token ceiling understating what the
            # model was asked to do, and an audit that skipped exactly the
            # unpriced responses would be missing the ones worth knowing about.
            raise UnpricedModelError(
                f'No price for model {response.model_name or "<unnamed>"}. Supply `SpendLimits.price`, '
                "or set on_unpriced='zero' to count the request as free."
            )
        return response

    async def status(
        self,
        ctx: RunContext[AgentDepsT] | None = None,
        *,
        scope: str | None = None,
    ) -> tuple[BudgetStatus, ...]:
        """Where each budget stands.

        Inside a run, pass `ctx` and every budget resolves. Without one -- the
        reading a cost display wants, and the check to make before starting a
        durable workflow whose hooks cannot reach a shared store -- budgets on a
        `run` or `conversation` window are omitted, since those periods have no
        meaning outside a run, and a budget declaring a `scope` is omitted
        unless `scope` names the partition to read, since its callable has no
        run context to resolve against.

        Reach for [`exhausted`][pydantic_ai_harness.spend.SpendLimits.exhausted] when the
        answer gates something: `any(s.exhausted for s in ...)` over a tuple that happens to
        be empty is a brake that reads as enforcement and inspects nothing, and a `SpendLimits`
        whose budgets are all scoped returns exactly that tuple.
        """
        statuses, _ = await self._resolve(ctx, scope)
        return statuses

    async def _resolve(
        self, ctx: RunContext[AgentDepsT] | None, scope: str | None
    ) -> tuple[tuple[BudgetStatus, ...], tuple[str, ...]]:
        """The readable budgets, and the names of the ones this call cannot resolve.

        `ctx` and `scope` are two answers to the same question, so supplying both is refused
        rather than resolved by precedence: a run's own scope silently winning would report
        one tenant's money under another tenant's name, and the caller has no way to tell.
        """
        if ctx is not None and scope is not None:
            raise UserError(
                'Pass either a run context or `scope=`, not both: `ctx` already names the partition to read, '
                'so a second answer here would be reporting one scope under the name of another. '
                'Drop `scope=` to read the run, or drop `ctx` to read another partition.'
            )
        now = self._now()
        statuses: list[BudgetStatus] = []
        unresolved: list[str] = []
        for budget in self.budgets:
            if ctx is None and (budget.window in _RUN_SCOPED_WINDOWS or (budget.scope is not None and scope is None)):
                unresolved.append(budget.name)
                continue
            key = self._key(budget, ctx, now, scope)
            statuses.append(_status(budget, key, await self.store.get(key)))
        return tuple(statuses), tuple(unresolved)

    async def exhausted(
        self,
        ctx: RunContext[AgentDepsT] | None = None,
        *,
        scope: str | None = None,
    ) -> bool:
        """Whether any budget this call can read is exhausted, refusing to guess about the rest.

        The pre-flight check a durable workflow makes before starting: its hooks cannot reach
        a shared store mid-run, so the decision has to be made up front. `status()` omits what
        it cannot resolve, and `any(...)` over the remainder is a brake that silently checks
        nothing when every budget is scoped -- so this raises instead, naming the budgets that
        need a `scope` or a `ctx`.
        """
        statuses, unresolved = await self._resolve(ctx, scope)
        if unresolved:
            raise UserError(
                f'Cannot read budget(s) {sorted(unresolved)} without a run context or a `scope`, so this check '
                'would pass having inspected nothing. Pass `scope=` for a scoped budget, or call it inside a run.'
            )
        return any(status.exhausted for status in statuses)

    @classmethod
    def from_spec(cls, *args: Any, **kwargs: Any) -> SpendLimits[Any]:
        """Build from an agent spec, covering the fields a spec can express.

        `budgets` arrive as mappings and become `Budget` instances, with `usd`
        accepted as a string so YAML cannot round a price through a float. The
        callables and the store have no spec representation and are rejected
        rather than dropped: a spec that promises per-tenant scoping and does
        not deliver it is worse than a spec that refuses to load.
        """
        unsupported = sorted({'store', 'price', 'on_spend', 'clock'} & kwargs.keys())
        if unsupported:
            raise UserError(
                f'SpendLimits cannot be built from a spec with {unsupported}: these take callables or a live '
                'store. Construct the capability in code to use them.'
            )
        budgets = [_budget_from_spec(entry) for entry in kwargs.pop('budgets', [])]
        return cls(*args, budgets=budgets, **kwargs)

    def _now(self) -> datetime:
        """The current time, naming the fix when Temporal's sandbox refuses the clock.

        These hooks run in workflow code, and the default clock calls `datetime.now`, which
        Temporal's workflow sandbox restricts. The sandbox's own error names
        `datetime.datetime.now` and not the setting that resolves it, so it is translated
        here. Matched by class name rather than by importing `temporalio`, which this package
        does not depend on.
        """
        try:
            return self.clock()
        except Exception as error:
            if type(error).__name__ != 'RestrictedWorkflowAccessError':
                raise
            raise UserError(
                "SpendLimits reads the clock from workflow code, which Temporal's workflow sandbox "
                'restricts. Pass this package through the sandbox with '
                "`SandboxRestrictions.default.with_passthrough_modules('pydantic_ai_harness')`, "
                'and keep the in-process store inside the workflow.'
            ) from error

    def _key(
        self,
        budget: Budget,
        ctx: RunContext[AgentDepsT] | None,
        now: datetime,
        scope: str | None,
    ) -> str:
        """The store key this budget accumulates under right now."""
        bucket_id = bucket(budget.window, ctx, now)
        if bucket_id is None:  # pragma: no cover - callers filter run-scoped windows out first
            raise UserError(f"Budget {budget.name!r} uses window='{budget.window}', which needs a run.")
        return store_key(budget, bucket_id, scope_key(budget, ctx, scope))

    def _check(self, budget: Budget, spent: Spent, ctx: RunContext[AgentDepsT]) -> None:
        """Raise if `spent` has reached either of the budget's ceilings."""
        if budget.usd is not None and spent.usd >= budget.usd:
            self._refuse(budget, ctx, f'spent ${spent.usd} of ${budget.usd}')
        if budget.tokens is not None and spent.tokens >= budget.tokens:
            self._refuse(budget, ctx, f'used {spent.tokens} of {budget.tokens} tokens')

    def _refuse(self, budget: Budget, ctx: RunContext[AgentDepsT], detail: str) -> None:
        """Record the refusal as a span and raise."""
        attributes: dict[str, str] = {'spend.budget': budget.name, 'spend.window': budget.window}
        if ctx.trace_include_content and budget.scope is not None:
            # A scope key is usually a tenant or user id, and a trace has a wider
            # audience than the application that produced it.
            attributes['spend.scope'] = budget.scope(ctx)
        ctx.tracer.start_span('spend budget exhausted', attributes=attributes).end()
        raise SpendLimitExceeded(f'Budget {budget.name!r} exhausted for this {budget.window}: {detail}')

    def _keyed(self, ctx: RunContext[AgentDepsT]) -> list[tuple[Budget, str]]:
        """Each budget paired with the store key it accumulates under right now.

        The construction-time check compares the shapes a key is built from; here the keys
        themselves are known, so two budgets whose scopes are different callables returning
        the same string are caught as well. That is the common way it happens: a scope written
        twice as `lambda ctx: ctx.deps.tenant` is two objects and one key.
        """
        now = self._now()
        keyed = [(budget, self._key(budget, ctx, now, None)) for budget in self.budgets]
        claimed: dict[tuple[str, str], Budget] = {}
        for budget, key in keyed:
            for kind, ceiling in (('usd', budget.usd), ('tokens', budget.tokens)):
                if ceiling is None:
                    continue
                prior = claimed.get((key, kind))
                if prior is not None and prior is not budget:
                    raise UserError(
                        f'Budgets {prior.name!r} and {budget.name!r} accumulate under the same key {key!r} and '
                        f'both set a `{kind}` ceiling, so only the smaller would ever apply. '
                        'Give them different `name`s, or one scope.'
                    )
                claimed[(key, kind)] = budget
        return keyed

    def _price_of(self, response: ModelResponse) -> tuple[Decimal, bool]:
        """What the response cost, and whether that number is real."""
        if self.price is not None:
            supplied = self.price(response)
            if supplied is not None:
                if supplied < 0:
                    # A credit would move a budget away from its ceiling, which
                    # turns a bug in the pricing function into a gate that never
                    # closes. Corrections belong in the store, not here.
                    raise UserError(f'`SpendLimits.price` returned a negative amount ({supplied}) for a response.')
                return supplied, True
        if response.model_name:
            try:
                return response.cost().total_price, True
            except (LookupError, ValueError):
                # LookupError: the registry has no entry. ValueError: it rejects the
                # usage shape. Either way this is a pricing failure, and letting it
                # escape would skip `on_unpriced` and drop the accrual with it.
                pass
        return Decimal(0), False


def _status(budget: Budget, key: str, spent: Spent) -> BudgetStatus:
    """Pair a budget with what its window has accumulated."""
    remaining_usd = None if budget.usd is None else budget.usd - spent.usd
    remaining_tokens = None if budget.tokens is None else budget.tokens - spent.tokens
    return BudgetStatus(
        budget=budget,
        key=key,
        spent=spent,
        remaining_usd=remaining_usd,
        remaining_tokens=remaining_tokens,
        warning=_warning(budget, spent),
        exhausted=(remaining_usd is not None and remaining_usd <= 0)
        or (remaining_tokens is not None and remaining_tokens <= 0),
    )


def _warning(budget: Budget, spent: Spent) -> bool:
    """Whether spend has crossed the budget's warning fraction."""
    if budget.warn_at is None:
        return False
    fraction = Decimal(str(budget.warn_at))
    if budget.usd is not None and spent.usd >= budget.usd * fraction:
        return True
    return budget.tokens is not None and spent.tokens >= budget.tokens * fraction


def _is_spec_mapping(entry: object) -> TypeGuard[Mapping[str, Any]]:
    """Whether a spec entry has the shape a `Budget` can be built from.

    Only the mapping shape is checked; a non-string key fails against
    `Budget`'s own signature, in an error that names the field.
    """
    return isinstance(entry, Mapping)


def _budget_from_spec(entry: Any) -> Budget:
    """Build one `Budget` from its spec mapping."""
    if not _is_spec_mapping(entry):
        raise UserError(f'Each SpendLimits budget in a spec must be a mapping; got {entry!r}.')
    fields = dict(entry)
    if 'scope' in fields:
        raise UserError('A SpendLimits budget scope is a callable and cannot be expressed in a spec.')
    # Every number a spec can carry, not just `usd`: the others reached
    # `Budget.__post_init__` as strings and compared str to int there, which
    # surfaces as a bare TypeError naming no field.
    for name, convert in (('usd', Decimal), ('tokens', int), ('warn_at', float)):
        if name in fields:
            try:
                fields[name] = convert(str(fields[name]))
            except (TypeError, ValueError) as error:
                raise UserError(f'SpendLimits budget {name!r} is not a number: {fields[name]!r}.') from error
    return Budget(**fields)
