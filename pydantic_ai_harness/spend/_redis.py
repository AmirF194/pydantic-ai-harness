"""A spend counter shared across processes, backed by Redis.

Takes a client protocol rather than a Redis dependency, the way
`pydantic_ai_harness.memory` takes a Postgres connection protocol: any client
exposing these two coroutines works, and installing the harness pulls in
nothing extra.
"""

from __future__ import annotations

from collections.abc import Awaitable, Mapping, Sequence
from dataclasses import dataclass
from datetime import timedelta
from decimal import ROUND_HALF_UP, Decimal, localcontext
from typing import Protocol, runtime_checkable

from pydantic_ai.exceptions import UserError

from pydantic_ai_harness.spend._snapshot import Spent
from pydantic_ai_harness.spend._store import DEFAULT_DEDUP_RETAIN, SpendEntry

_SCALE = Decimal(10) ** 9
_PRECISION = 40

_USD_FIELD = 'usd_nanos'
_TOKENS_FIELD = 'tokens'
_REQUESTS_FIELD = 'requests'
_UNPRICED_FIELD = 'unpriced'

_ADD_SCRIPT = f"""
local n = tonumber(ARGV[1])
local out = {{}}
for i = 1, n do
  local key = KEYS[i]
  local base = 1 + (i - 1) * 6
  local marker_ttl = tonumber(ARGV[base + 6])
  local existed = redis.call('EXISTS', key)
  local apply = true
  if marker_ttl > 0 and not redis.call('SET', KEYS[n + i], '1', 'NX', 'EX', marker_ttl) then
    apply = false
    existed = 1
  end
  if apply then
    redis.call('HINCRBY', key, '{_USD_FIELD}', ARGV[base + 1])
    redis.call('HINCRBY', key, '{_TOKENS_FIELD}', ARGV[base + 2])
    redis.call('HINCRBY', key, '{_REQUESTS_FIELD}', ARGV[base + 3])
    redis.call('HINCRBY', key, '{_UNPRICED_FIELD}', ARGV[base + 4])
    local ttl = tonumber(ARGV[base + 5])
    if ttl > 0 then
      redis.call('EXPIRE', key, ttl)
    else
      redis.call('PERSIST', key)
    end
  end
  local totals = redis.call('HMGET', key, '{_USD_FIELD}', '{_TOKENS_FIELD}', '{_REQUESTS_FIELD}', '{_UNPRICED_FIELD}')
  for j = 1, 4 do
    if not totals[j] then totals[j] = '0' end
  end
  totals[5] = tostring(existed)
  out[i] = totals
end
return out
"""
"""Applies one response to every window it counts against, and returns each window's totals.

KEYS is one key per entry followed by one marker key per entry; ARGV is the entry count
followed by six values per entry: the four increments, the key's horizon in seconds, and
the marker's, where a zero horizon means "keep this key" and "claim no marker".

Redis runs a script to completion without interleaving another command, so no other client
observes a response part-applied -- neither across the four counters of one window nor across
the windows themselves. Issued as separate commands instead, a failure between them leaves a
window counting some of a response and not the rest, which reads as a smaller number than was
really spent and so releases the brake later than it should.

The totals come back as bulk strings, read with `HMGET` after the increments rather than taken
from what `HINCRBY` returns. `HINCRBY` is exact 64-bit integer arithmetic and the increment
arrives as a string, so the stored counter never rounds; its *reply* is an integer that becomes
a Lua number, and Lua 5.1 numbers are doubles, so a total past 2**53 would come back rounded
even though Redis holds it exactly. A Lua string is returned as a bulk string with no double in
the path.

What is left is `HINCRBY`'s own range: a counter passing the signed 64-bit range, around $9.22
billion or 9.2 quintillion tokens against a single key, which Redis refuses before writing that
field. It aborts the script there, and Redis does not roll back what earlier commands in the
script already did, so a multi-window response would keep the windows applied before the one
that overflowed.

`SET ... NX` on a per-entry marker is how an entry carrying a `SpendEntry.token` is applied at
most once. A claim that fails means this response already reached this window, so the entry is
skipped and the current totals are returned unchanged.

A zero `ttl` means "keep this key", and says so with `PERSIST` rather than by doing nothing.
`HINCRBY` leaves an existing expiry in place, so a budget moved from a finite `retain` to
`'forever'` would otherwise keep expiring on the old horizon -- handing back the ceiling on a
schedule nothing in the configuration mentions any more, and diverging from
`InMemorySpendStore`, which drops the expiry on the next write.

The fifth value of each row reports whether the key existed before this write, which is what
tells the store a key may still have a counter under the pre-0.18 name. See
`RedisSpendStore._carry_legacy`.
"""


@runtime_checkable
class RedisClient(Protocol):
    """The part of a Redis client `RedisSpendStore` uses.

    `redis.asyncio.Redis` satisfies this. So does any wrapper or fake exposing the
    same two methods, `async def` included.

    Declared as returning an `Awaitable` rather than as `async def`, which would
    narrow the requirement to a `Coroutine` and is what an implementation actually
    has to hand back. `redis.asyncio.Redis` types these as `Awaitable`, so an
    `async def` protocol refused the one client this exists to accept.
    """

    def hgetall(self, name: str) -> Awaitable[Mapping[str | bytes, str | bytes]]:
        """Every field of a hash. An absent hash reads as empty."""
        ...  # pragma: no cover

    def eval(self, script: str, numkeys: int, *keys_and_args: str | int) -> Awaitable[Sequence[Sequence[str | bytes]]]:
        """Run a Lua script server-side and return what it returns.

        Rows of strings, because that is what `_ADD_SCRIPT` returns and what keeps a
        total past 2**53 exact. `str` or `bytes` depending on the client's
        `decode_responses`.
        """
        ...  # pragma: no cover


def _to_nanos(usd: Decimal) -> int:
    """US dollars as whole billionths.

    Counters are integers because `INCRBYFLOAT` accumulates binary rounding
    error over the tens of thousands of requests a busy day produces, while
    `HINCRBY` on integers is both exact and atomic.

    Billionths rather than millionths because the residue does not average out:
    an agent repeats requests of near-identical shape, so the same fraction is
    rounded the same way every time. At a cheap model's per-request price,
    rounding to a millionth drifts by tens of percent over a day; a billionth
    keeps that under a part in ten thousand.

    The local context is pinned so an application that lowered `Decimal`
    precision for its own arithmetic cannot silently truncate money here.
    """
    with localcontext() as context:
        context.prec = _PRECISION
        return int((usd * _SCALE).to_integral_value(rounding=ROUND_HALF_UP))


def _from_nanos(nanos: int) -> Decimal:
    """Whole billionths back to US dollars."""
    with localcontext() as context:
        context.prec = _PRECISION
        return Decimal(nanos) / _SCALE


def _expiry_seconds(ttl: timedelta) -> int:
    """A positive `timedelta` as whole seconds, rounded up.

    `EXPIRE` takes seconds, and the script reads zero as "keep this key". Rounding down
    would turn any horizon under a second -- which `Budget.retain` accepts -- into no
    expiry at all, the opposite of what was asked for and a silent divergence from
    `InMemorySpendStore`, which honours the horizon exactly.

    Ceiling division on the `timedelta` itself rather than on `total_seconds()`, which is
    a float: `timedelta` divides in whole microseconds, so a horizon with a sub-millisecond
    remainder rounds up like any other instead of being truncated away before the ceiling
    is taken.
    """
    return max(1, -(-ttl // timedelta(seconds=1)))


def _text(value: str | bytes) -> str:
    """One value of a reply, whichever way the client decodes."""
    return value.decode() if isinstance(value, bytes) else value


def _field(fields: Mapping[str | bytes, str | bytes], name: str) -> int:
    """One integer field of a hash, treating an absent field as zero."""
    for key, value in fields.items():
        if _text(key) == name:
            return int(_text(value))
    return 0


def _spent(fields: Mapping[str | bytes, str | bytes]) -> Spent:
    """A window's counters, read from its hash."""
    return Spent(
        usd=_from_nanos(_field(fields, _USD_FIELD)),
        tokens=_field(fields, _TOKENS_FIELD),
        requests=_field(fields, _REQUESTS_FIELD),
        unpriced_requests=_field(fields, _UNPRICED_FIELD),
    )


@dataclass
class RedisSpendStore:
    """Spend counters in Redis, so every worker enforces one budget.

    One hash per window, holding the four counters as integers.

    ```python
    from redis.asyncio import Redis

    from pydantic_ai_harness.spend import RedisSpendStore

    store = RedisSpendStore(Redis.from_url('redis://localhost'))
    ```

    A read and the increment that follows it are separate round trips, so
    concurrent runs can each observe a budget as unexhausted and push past it
    together. That is the same overshoot the in-process store has, widened by
    the number of workers; see the README on what the gate does and does not
    guarantee.
    """

    client: RedisClient
    """Any client exposing `hgetall` and `eval`."""

    prefix: str = 'pydantic-ai-harness:spend'
    """Namespace for the keys, so a shared Redis stays tidy.

    Every key this store writes is `{prefix}:...` with the braces literal, which is a
    Redis Cluster hash tag: the slot is computed from the prefix alone, so all of a
    store's keys land in one slot and a script may take several of them at once.
    Applying one response to a day and a month window in one script is what that buys,
    and the cost is that a cluster cannot spread this store's keys across its nodes.
    """

    dedup_retain: timedelta | None = DEFAULT_DEDUP_RETAIN
    """How long an applied `SpendEntry.token` is remembered, or `None` to apply every entry.

    Each remembered token is one small key per response per window, expiring on this
    horizon. Shorten it where the write rate makes that memory matter more than an
    accrual replayed by a durable engine after a long recovery.
    """

    def __post_init__(self) -> None:
        """Reject a prefix that would break the hash tag it is wrapped in.

        A brace inside the prefix moves or truncates the tag, so two windows of one
        budget would hash to different slots and a cluster would refuse the script that
        applies them together. Checked here for the same reason `Budget.name` is checked
        against its separator: the failure otherwise arrives as a `CROSSSLOT` error on a
        model request.
        """
        if '{' in self.prefix or '}' in self.prefix:
            raise UserError(
                f'RedisSpendStore.prefix must not contain braces; got {self.prefix!r}. The prefix is wrapped in '
                'a Redis Cluster hash tag, and a brace inside it would change which slot the keys hash to.'
            )

    async def get(self, key: str) -> Spent:
        """What `key` has accumulated. Deprecated in favour of `get_many`, removed in 0.20.0."""
        return (await self.get_many([key]))[key]

    async def add(
        self,
        key: str,
        *,
        usd: Decimal,
        tokens: int,
        requests: int,
        unpriced: int,
        ttl: timedelta | None,
    ) -> Spent:
        """Add to `key` and return the result. Deprecated in favour of `add_many`, removed in 0.20.0.

        One window per call, so a response counting against a day and a month budget is
        two calls and a failure between them leaves the day counted and the month not.
        `add_many` is one script over every window, which is what closes that.
        """
        entry = SpendEntry(key=key, usd=usd, tokens=tokens, requests=requests, unpriced=unpriced, ttl=ttl)
        return (await self.add_many([entry]))[key]

    async def get_many(self, keys: Sequence[str]) -> Mapping[str, Spent]:
        """What each key has accumulated. An absent hash reads as zero.

        One round trip per key, and a second for a key that reads as absent and may still
        have a counter under the pre-0.18 name; see `_carry_legacy`.
        """
        totals: dict[str, Spent] = {}
        for key in keys:
            fields = await self.client.hgetall(self._name(key))
            if not fields:
                fields = await self.client.hgetall(self._legacy_name(key))
            totals[key] = _spent(fields)
        return totals

    async def add_many(self, entries: Sequence[SpendEntry]) -> Mapping[str, Spent]:
        """Apply every entry as one script and return each key's new total.

        One round trip, and one unit of work: every window of the response lands or none
        does, and the script returns each new total, so the result needs no second read.

        A failure before the server runs the script -- the client cannot connect, the
        request never lands -- writes nothing. A failure after it does not say which:
        the connection can drop once `EVAL` has already committed, so an error here
        means the outcome is unknown rather than that nothing happened. Retrying
        therefore risks counting the response twice, which is why
        `SpendLimits.wrap_model_request` does not retry and lets the error end the run
        instead. Over-counting a response the provider did bill is the direction a
        brake can survive; under-counting is not.

        A `SpendEntry.token` makes the retry that *is* safe: a durable engine
        re-executing an accrual it already committed claims a marker that is already
        held, so the entry is skipped and the current total is returned.
        """
        if not entries:
            return {}
        applied = await self._apply(entries)
        cold = [entry for entry in entries if not applied[entry.key][1]]
        carried: Mapping[str, Spent] = await self._carry_legacy(cold) if cold else {}
        return {key: carried.get(key, spent) for key, (spent, _) in applied.items()}

    async def _apply(self, entries: Sequence[SpendEntry]) -> dict[str, tuple[Spent, bool]]:
        """Run `_ADD_SCRIPT` over the entries, returning each key's totals and whether it existed."""
        keys = [self._name(entry.key) for entry in entries]
        markers = [self._marker_name(entry) for entry in entries]
        arguments: list[str | int] = [len(entries)]
        for entry in entries:
            arguments += [
                _to_nanos(entry.usd),
                entry.tokens,
                entry.requests,
                entry.unpriced,
                0 if entry.ttl is None else _expiry_seconds(entry.ttl),
                self._marker_seconds(entry),
            ]
        rows = await self.client.eval(_ADD_SCRIPT, 2 * len(entries), *keys, *markers, *arguments)
        return {
            entry.key: (
                Spent(
                    usd=_from_nanos(int(_text(row[0]))),
                    tokens=int(_text(row[1])),
                    requests=int(_text(row[2])),
                    unpriced_requests=int(_text(row[3])),
                ),
                _text(row[4]) == '1',
            )
            for entry, row in zip(entries, rows)
        }

    # Everything below carries counters written before 0.18, when the keys had no hash
    # tag. Delete `_carry_legacy`, `_legacy_name`, and their two call sites (`get_many`
    # and `add_many`) in 0.20.0, by which point every window written under the old name
    # has passed the longest `Budget.retain` default.
    async def _carry_legacy(self, entries: Sequence[SpendEntry]) -> Mapping[str, Spent]:
        """Move any pre-0.18 counter into the tagged key, for entries whose key was cold.

        Which worker folds is settled by the script that just ran: it applies the whole
        entry atomically, so exactly one caller can see a key that did not exist before
        its own write, and only that caller reaches here. The legacy hash is read outside
        the script because it hashes to a different slot, which a cluster would refuse
        inside one.

        A process that dies between the write and this fold leaves the old counter where
        it is: no other caller will see that key cold again, so the amount is not carried.
        That is the one gap, and it is bounded by the pre-0.18 counters that still exist.
        """
        carried: list[SpendEntry] = []
        for entry in entries:
            fields = await self.client.hgetall(self._legacy_name(entry.key))
            if fields:
                previous = _spent(fields)
                carried.append(
                    SpendEntry(
                        key=entry.key,
                        usd=previous.usd,
                        tokens=previous.tokens,
                        requests=previous.requests,
                        unpriced=previous.unpriced_requests,
                        ttl=entry.ttl,
                    )
                )
        if not carried:
            return {}
        return {key: spent for key, (spent, _) in (await self._apply(carried)).items()}

    def _legacy_name(self, key: str) -> str:
        """The Redis key a pre-0.18 harness wrote this budget key under."""
        return f'{self.prefix}:{key}'

    def _name(self, key: str) -> str:
        """The Redis key for a budget key, hash-tagged so one script may take several."""
        return f'{{{self.prefix}}}:{key}'

    def _marker_name(self, entry: SpendEntry) -> str:
        """The key claimed to record that this entry's response reached this window.

        An entry with no token still needs a name here, because the script reads a marker
        per entry: it gets one nothing writes to, since its horizon is zero.
        """
        return self._name('dedup' if entry.token is None else f'dedup|{entry.key}|{entry.token}')

    def _marker_seconds(self, entry: SpendEntry) -> int:
        """How long this entry's marker is held, or zero to apply the entry unconditionally."""
        if entry.token is None or self.dedup_retain is None:
            return 0
        return _expiry_seconds(self.dedup_retain)
