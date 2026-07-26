---
title: Spend
description: Track what an agent costs and refuse the next request once a budget is spent, with windows longer than a run, per-tenant scopes, and a counter shared across worker processes.
---

# Spend

Track what an agent costs, and stop it when a budget is gone.

> The API may change between releases. Where practical, breaking changes ship with a deprecation warning.

## The problem

A loop that calls a model until a condition it never reaches will keep calling until something stops it. `UsageLimits` in Pydantic AI is that stop for one run: it caps tokens and requests, in token counts, for the duration of a single `run()`. What it does not cover is money, a period longer than one run, a per-tenant share of a shared allowance, or a counter that several worker processes agree on. A daily ceiling spread across a queue's workers is exactly the case where each worker independently believes it has the whole budget.

Provider usage APIs do not close that gap. OpenAI's and Anthropic's report minutes to hours late; Logfire, measured, lands around 25 seconds behind. Those numbers are fine for reconciling a ledger and useless for refusing the request a runaway loop is about to make.

## The solution

`SpendGuard` prices every model response with [`ModelResponse.cost()`](https://pydantic.dev/docs/ai/api/messages/), adds it to each window you configure, and refuses the next request once a window is spent.

```python
from decimal import Decimal

from pydantic_ai import Agent
from pydantic_ai_harness.spend import Budget, SpendGuard

agent = Agent(
    'openai:gpt-5.4',
    capabilities=[SpendGuard(budgets=[Budget(usd=Decimal('100'), window='day')])],
)
```

Past $100 in a UTC day, the next request raises `SpendLimitExceeded`.

## Budgets

A budget is a ceiling, a period, and optionally a partition. They compose, so several apply at once:

```python
from decimal import Decimal

from pydantic_ai_harness.spend import Budget, SpendGuard

SpendGuard(
    budgets=[
        Budget(usd=Decimal('5'), window='run'),  # one runaway run
        Budget(usd=Decimal('100'), window='day'),  # the whole deployment, per day
        Budget(usd=Decimal('2000'), window='month', warn_at=0.8),
        Budget(usd=Decimal('10'), window='day', scope=lambda ctx: ctx.deps.tenant_id, name='tenant'),
    ]
)
```

| Field | Meaning |
|---|---|
| `usd` / `tokens` | ceilings; set either, both, or neither |
| `window` | `run`, `conversation`, `day`, `month`, `total` |
| `scope` | derives a partition key from the run, so tenants count separately |
| `warn_at` | fraction past which `BudgetStatus.warning` is set; never blocks |
| `name` | distinguishes budgets sharing a window and scope |

A window rolls over by producing a different store key rather than by resetting a counter, so a new day is simply a new key and nothing has to run at midnight. A `total` counter never expires. `run` and `conversation` buckets never roll over either, so expiry there hands back the ceiling rather than starting a new period -- but each mints a key per run or per conversation, so they carry a long horizon (24 hours and 30 days) instead, past which the counter is dropped rather than kept forever.

Budgets that share a `name`, `window`, and `scope` share one counter, which is how a single window carries both a USD and a token ceiling. The response is added to that counter once, not once per budget.

**A budget with no ceiling is a counter.** It accumulates and reports and never refuses anything, which is how per-tenant accounting with no cap is expressed:

```python
from pydantic_ai_harness.spend import Budget, SpendGuard

SpendGuard(budgets=[Budget(window='month', scope=lambda ctx: ctx.deps.tenant_id, name='chargeback')])
```

## What the gate guarantees

No request **starts** after a budget is exhausted.

Not: that spend stays under the ceiling. The request that crosses the line completes, and concurrent runs can each pass the check before any of them records anything. Two further gaps are worth knowing rather than discovering: a stream the caller abandons part-way never reaches the accounting hook, so its tokens are billed by the provider and invisible here, and a capability that answers from a cache without calling a provider is charged the registry price for the response it returns. Treat this as a brake on a runaway loop, not as an accounting ledger; reconcile against the provider's own numbers if you need the second thing.

## Reading the numbers

```python
from decimal import Decimal

from pydantic_ai_harness.spend import Budget, SpendGuard, SpendSnapshot


def show(snapshot: SpendSnapshot) -> None:
    print(f'{snapshot.model} cost ${snapshot.usd}')
    for status in snapshot.budgets:
        print(f'  {status.budget.name}: ${status.remaining_usd} left')


SpendGuard(budgets=[Budget(usd=Decimal('100'))], on_spend=show)
```

`on_spend` fires after every response, sync or async, with a `SpendSnapshot` -- including one that `on_unpriced='raise'` is about to reject, since a report that skipped exactly the unpriced responses would be missing the ones worth knowing about. It carries the response's `usage` unchanged, so cache reads and writes are available without this capability modelling them.

`status()` reads the same numbers without a run, which is what a cost display in a UI wants:

```python
async def report(guard: SpendGuard[None]) -> None:
    for status in await guard.status(scope='acme'):
        print(status.budget.name, status.spent.usd, status.exhausted)
```

Without a run context, budgets on a `run` or `conversation` window are omitted, and so is a budget declaring a `scope` unless `scope=` names the partition to read. Pass `ctx` inside a run and every budget resolves.

Set `expose_tools=True` to give the agent a `get_spend` tool. It is off by default: a tool costs schema tokens on every request, and most applications want the number on a screen rather than in the model's context.

## Sharing a counter across processes

The default store keeps counters in the process, which catches a runaway loop inside one worker and does nothing for a budget spread across a queue. `RedisSpendStore` is the shared counter:

```python
from decimal import Decimal

from redis.asyncio import Redis

from pydantic_ai_harness.spend import Budget, RedisSpendStore, SpendGuard

store = RedisSpendStore(Redis.from_url('redis://localhost'))
guard = SpendGuard(budgets=[Budget(usd=Decimal('100'), window='day')], store=store)
```

It adds no dependency: `RedisClient` is a protocol of the three coroutines used, so any compatible client satisfies it. Amounts are stored as integer billionths of a dollar and incremented with `HINCRBY`, because `INCRBYFLOAT` accumulates rounding error over the tens of thousands of requests a busy day produces. Billionths rather than millionths because the residue does not average out: an agent repeats requests of near-identical shape, so the same fraction rounds the same way every time.

The default store is built per capability, so two `SpendGuard` instances do not quietly share one counter. Pass the same store object to both when you want them to.

A store that fails does not fail quietly. An error reading the counter refuses the request, which is the safe direction. An error writing it propagates out of the run after the model has already answered and been charged, and the increment may be half applied. That is deliberate: a swallowed write would drift the counter down and weaken the gate, which is worse than a visible failure. If your deployment would rather keep the answer than the count, wrap the store and decide there.

Any object with `get` and `add` works, so a Postgres or DynamoDB counter is a small class rather than a fork.

## Pricing

Prices come from [genai-prices](https://github.com/pydantic/genai-prices) via `ModelResponse.cost()`, per response: cache and tier pricing are per request, so summing usage across requests and pricing the total gives the wrong number.

A model the registry does not know -- a local deployment, a negotiated rate -- is handled by `price`:

```python
from decimal import Decimal

from pydantic_ai_harness.spend import SpendGuard

SpendGuard(price=lambda response: Decimal('0.002') if response.model_name == 'internal-7b' else None)
```

Returning `None` falls through to the registry. When nothing can price a response, `on_unpriced` decides: `'zero'` (the default) counts it as free and increments `Spent.unpriced_requests` so the gap is visible, and `'raise'` fails the run with `UnpricedModelError`. Either way the response is recorded first and the tokens are counted, so a token ceiling still holds for a model with no price and an application that catches the error does not carry on against an understated counter.

## Composition

State lives across runs deliberately, so `for_run` is not overridden: a daily budget that reset every run would not be a daily budget. Per-run isolation comes from `Budget(window='run')`, whose key carries the run id.

The capability declares itself innermost. `after_model_request` runs innermost first, so anywhere else in the chain a capability listed after this one could raise `ModelRetry` on a response that was already generated and billed, and the counter would never see it. Being innermost is what makes "counted exactly once" true rather than dependent on the order you happened to write.

**Durable execution.** The capability hooks run in the workflow; the model request itself is the activity. A store that talks over the network therefore reads and writes from workflow code, which Temporal replays -- so a shared store is not workflow-safe here. Use the in-process store inside the workflow, and enforce the shared budget before starting it:

```python
async def start_if_funded(guard: SpendGuard[None], tenant_id: str) -> None:
    if await guard.exhausted(scope=tenant_id):
        raise RuntimeError('daily budget exhausted')
    await workflow_handle.execute(...)
```

`exhausted` rather than `any(s.exhausted for s in await guard.status(...))`: `status` omits
the budgets it cannot resolve, and `any()` over what is left is a brake that passes having
inspected nothing -- which is exactly what a guard whose budgets are all scoped returns when
the scope is missing. `exhausted` raises there instead, naming the budgets that need a
`scope` or a run context. Use `status` for a reading, `exhausted` for a decision.

## Tracing

A refusal emits a `spend budget exhausted` span with `spend.budget` and `spend.window`. Accrual emits nothing: a span per model request would double the size of a trace without adding a decision. `spend.scope` is attached only when `RunContext.trace_include_content` is set, since a scope key is usually a tenant or user id and a trace has a wider audience than the application that produced it.

## Specs

`Agent.from_spec` supports the part of the configuration a spec can express:

```yaml
- SpendGuard:
    budgets:
      - {usd: '100', window: day}
      - {usd: '2000', window: month, warn_at: 0.8}
    on_unpriced: raise
```

`store`, `price`, `on_spend`, `clock`, and a budget's `scope` take callables or live objects. A spec naming them is rejected rather than silently ignored, because a spec that promises per-tenant scoping and does not deliver it is worse than one that refuses to load.

## API

```python {test="skip"}
SpendGuard(
    budgets: Sequence[Budget] = (),
    store: SpendStore = ...,  # a fresh InMemorySpendStore per capability
    price: Callable[[ModelResponse], Decimal | None] | None = None,
    on_spend: Callable[[SpendSnapshot], None | Awaitable[None]] | None = None,
    on_unpriced: Literal['zero', 'raise'] = 'zero',
    expose_tools: bool = False,
    clock: Callable[[], datetime] = utc_now,
)

Budget(
    usd: Decimal | None = None,
    tokens: int | None = None,
    window: Literal['run', 'conversation', 'day', 'month', 'total'] = 'day',
    scope: Callable[[RunContext[Any]], str] | None = None,
    warn_at: float | None = None,
    name: str = 'default',
)


class SpendStore(Protocol):
    async def get(self, key: str) -> Spent: ...
    async def add(
        self, key: str, *, usd: Decimal, tokens: int, requests: int, unpriced: int, ttl: timedelta | None
    ) -> Spent: ...


InMemorySpendStore(clock: Callable[[], datetime] = utc_now)
RedisSpendStore(client: RedisClient, prefix: str = 'pydantic-ai-harness:spend')
```

`SpendLimitExceeded` subclasses `UsageLimitExceeded`, so code that already stops on a usage limit stops here too, while code that needs to tell a spent daily budget from an over-long run can catch it specifically. `UnpricedModelError` subclasses `UserError`.

Source: [`pydantic_ai_harness/spend/`](https://github.com/pydantic/pydantic-ai-harness/tree/main/pydantic_ai_harness/spend/).
