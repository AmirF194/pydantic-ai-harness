"""A spend counter shared across processes, backed by Redis.

Takes a client protocol rather than a Redis dependency, the way
`pydantic_ai_harness.memory` takes a Postgres connection protocol: any client
exposing these three coroutines works, and installing the harness pulls in
nothing extra.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import timedelta
from decimal import ROUND_HALF_UP, Decimal
from typing import Protocol, runtime_checkable

from pydantic_ai_harness.spend._snapshot import Spent

_MICROS = Decimal(10) ** 6

_USD_FIELD = 'usd_micros'
_TOKENS_FIELD = 'tokens'
_REQUESTS_FIELD = 'requests'
_UNPRICED_FIELD = 'unpriced'


@runtime_checkable
class RedisClient(Protocol):
    """The part of a Redis client `RedisSpendStore` uses.

    `redis.asyncio.Redis` satisfies this. So does any wrapper or fake exposing
    the same three coroutines.
    """

    async def hincrby(self, name: str, key: str, amount: int) -> int:
        """Add `amount` to a hash field and return the new value."""
        ...  # pragma: no cover

    async def hgetall(self, name: str) -> Mapping[str | bytes, str | bytes]:
        """Every field of a hash. An absent hash reads as empty."""
        ...  # pragma: no cover

    async def expire(self, name: str, time: int) -> bool:
        """Set a key's time to live, in seconds."""
        ...  # pragma: no cover


def _to_micros(usd: Decimal) -> int:
    """US dollars as whole millionths.

    Counters are integers because `INCRBYFLOAT` accumulates binary rounding
    error over the tens of thousands of requests a busy day produces, while
    `HINCRBY` on integers is both exact and atomic.
    """
    return int((usd * _MICROS).to_integral_value(rounding=ROUND_HALF_UP))


def _from_micros(micros: int) -> Decimal:
    """Whole millionths back to US dollars."""
    return Decimal(micros) / _MICROS


def _field(fields: Mapping[str | bytes, str | bytes], name: str) -> int:
    """One integer field of a hash, treating an absent field as zero."""
    for key, value in fields.items():
        decoded = key.decode() if isinstance(key, bytes) else key
        if decoded == name:
            return int(value.decode() if isinstance(value, bytes) else value)
    return 0


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
    """Any client exposing `hincrby`, `hgetall`, and `expire`."""

    prefix: str = 'pydantic-ai-harness:spend'
    """Namespace for the keys, so a shared Redis stays tidy."""

    async def get(self, key: str) -> Spent:
        """What `key` has accumulated. An absent hash reads as zero."""
        fields = await self.client.hgetall(self._name(key))
        return Spent(
            usd=_from_micros(_field(fields, _USD_FIELD)),
            tokens=_field(fields, _TOKENS_FIELD),
            requests=_field(fields, _REQUESTS_FIELD),
            unpriced_requests=_field(fields, _UNPRICED_FIELD),
        )

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

        Each `HINCRBY` returns its own new value, so the result needs no second
        read. The four are not one atomic group, which does not matter: no
        decision compares these fields against each other.
        """
        name = self._name(key)
        micros = await self.client.hincrby(name, _USD_FIELD, _to_micros(usd))
        total_tokens = await self.client.hincrby(name, _TOKENS_FIELD, tokens)
        total_requests = await self.client.hincrby(name, _REQUESTS_FIELD, requests)
        total_unpriced = await self.client.hincrby(name, _UNPRICED_FIELD, unpriced)
        if ttl is not None:
            await self.client.expire(name, int(ttl.total_seconds()))
        return Spent(
            usd=_from_micros(micros),
            tokens=total_tokens,
            requests=total_requests,
            unpriced_requests=total_unpriced,
        )

    def _name(self, key: str) -> str:
        """The Redis key for a budget key, namespaced so a shared server stays tidy."""
        return f'{self.prefix}:{key}'
