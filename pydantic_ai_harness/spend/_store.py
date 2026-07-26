"""Where spend counters live.

`SpendStore` is the seam between the gate and its counter. The default keeps
counters in the process, which catches a runaway loop inside one worker; a
shared store is what makes a budget hold across the workers of a queue.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Protocol, runtime_checkable

from pydantic_ai_harness.spend._snapshot import Spent

_Entries = dict[str, tuple[Spent, 'datetime | None']]
"""Each key's counter and the moment it stops counting, if it ever does."""


def utc_now() -> datetime:
    """Current UTC time. The default clock for windows and expiry."""
    return datetime.now(timezone.utc)


@runtime_checkable
class SpendStore(Protocol):
    """Reads and accumulates the counter behind each budget window.

    `add` returns the state **after** the increment so an atomic backend can
    answer without a second round trip. `requests` is an explicit parameter
    rather than an implied `+= 1` so a reconciler correcting drift against an
    external source can post a delta (including a negative `usd`) without
    inflating the request count, and without the protocol growing a second
    method for it.
    """

    async def get(self, key: str) -> Spent:
        """What `key` has accumulated. A key that was never written reads as zero."""
        ...  # pragma: no cover

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
        """Add to `key` and return the result. `ttl` is how long the key may be kept."""
        ...  # pragma: no cover


@dataclass
class InMemorySpendStore:
    """Counters for the lifetime of one process.

    Catches a runaway loop inside the worker it runs in. It does not enforce a
    budget across processes: every worker of a queue would keep its own count,
    which is what a shared store such as
    [`RedisSpendStore`][pydantic_ai_harness.spend.RedisSpendStore] is for.
    """

    clock: Callable[[], datetime] = utc_now
    """Supplies the time expiry is measured against."""

    _entries: _Entries = field(default_factory=_Entries, init=False, repr=False)

    def __len__(self) -> int:
        """How many windows are being held.

        Bounded by the budgets and scopes still live, since a write sweeps the
        rolled-over ones. Worth watching if scopes are high-cardinality.
        """
        return len(self._entries)

    async def get(self, key: str) -> Spent:
        """What `key` has accumulated, treating an expired key as absent."""
        return self._live(key)

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
        """Add to `key` and return the result.

        The mutation spans no `await`, so concurrent runs in one event loop
        cannot interleave halfway through it.
        """
        self._sweep()
        current = self._live(key)
        updated = Spent(
            usd=current.usd + usd,
            tokens=current.tokens + tokens,
            requests=current.requests + requests,
            unpriced_requests=current.unpriced_requests + unpriced,
        )
        self._entries[key] = (updated, None if ttl is None else self.clock() + ttl)
        return updated

    def _sweep(self) -> None:
        """Drop every entry whose window has rolled over.

        Expiry cannot wait for the next read of a key: a day window produces a
        new key each day, so yesterday's is never asked for again and would sit
        in the dict forever. Sweeping on write keeps the dict to the windows
        that are still live, which is a handful per budget and scope.
        """
        now = self.clock()
        stale = [key for key, (_, expires_at) in self._entries.items() if expires_at is not None and now >= expires_at]
        for key in stale:
            del self._entries[key]

    def _live(self, key: str) -> Spent:
        """The entry at `key`, dropping it first if its window has rolled over.

        A read also expires the key it touches, so a rolled-over window reads as
        zero even between writes.
        """
        entry = self._entries.get(key)
        if entry is None:
            return Spent()
        spent, expires_at = entry
        if expires_at is not None and self.clock() >= expires_at:
            del self._entries[key]
            return Spent()
        return spent
