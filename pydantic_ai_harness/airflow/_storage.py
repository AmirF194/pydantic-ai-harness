"""Storage backends for the Airflow durability capability.

The capability memoizes each model and tool operation behind a small
[`DurableStorage`][pydantic_ai_harness.airflow.DurableStorage] Protocol, so the
persistence layer is pluggable. Two offline backends ship here: an in-memory
store for tests and single-process use, and a JSON-file store. A backend that
writes to Airflow's ObjectStorage or task state store would live in the Airflow
provider and satisfy the same Protocol.

Stored values are already-serialized `JsonValue` payloads (the capability dumps
each operation's result through a Pydantic `TypeAdapter` before handing it over),
so a backend only stores and retrieves plain JSON plus the request fingerprint
that produced it.
"""

from __future__ import annotations

import contextlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

from pydantic import JsonValue

# Prefix for durable cache keys. Mirrors the real Airflow provider so a key namespace
# shared with user code (e.g. a task state store) keeps durable steps from colliding.
DURABLE_KEY_PREFIX = '__pai_airflow_durable__'


@dataclass(frozen=True)
class StoredEntry:
    """One memoized operation: its serialized value and the request fingerprint that produced it."""

    value: JsonValue
    fingerprint: str | None


@runtime_checkable
class DurableStorage(Protocol):
    """Persistence contract for the Airflow durability capability.

    A backend maps a positional step key (`{prefix}model_step_3`, `{prefix}tool_step_4`, ...) to the
    serialized result of that operation plus the fingerprint of the request that produced it. The
    capability compares the current request's fingerprint against the stored one before replaying,
    so a backend does not need to understand fingerprints, only round-trip them.

    Values handed to `save` are JSON-compatible (`dict`/`list`/`str`/`int`/`float`/`bool`/`None`),
    so a backend can persist them with any JSON-capable store.
    """

    def load(self, key: str) -> StoredEntry | None:
        """Return the stored entry for `key`, or `None` if nothing was memoized under it."""
        ...  # pragma: no cover

    def save(self, key: str, value: JsonValue, *, fingerprint: str | None) -> None:
        """Store `value` (and the request `fingerprint`) under `key`, overwriting any prior entry."""
        ...  # pragma: no cover

    def cleanup(self) -> None:
        """Discard the store's memoized entries.

        A caller invokes this once the durable unit (the Airflow task) has finished successfully and
        no retry can replay from it. The capability does not call it: it cannot observe the task
        boundary, only the agent run.
        """
        ...  # pragma: no cover


class InMemoryDurableStorage:
    """In-process `DurableStorage` backed by a dict, for tests and single-process runs.

    `save` round-trips each value through `json.dumps`/`json.loads` so the stored form is the same
    plain JSON a persistent backend would hold. A value that is not JSON-serializable raises here,
    the same way it would against Postgres or a file, rather than being smuggled through as a live
    Python object.
    """

    def __init__(self) -> None:
        self._entries: dict[str, StoredEntry] = {}

    def load(self, key: str) -> StoredEntry | None:
        return self._entries.get(key)

    def save(self, key: str, value: JsonValue, *, fingerprint: str | None) -> None:
        round_tripped: JsonValue = json.loads(json.dumps(value))
        self._entries[key] = StoredEntry(value=round_tripped, fingerprint=fingerprint)

    def cleanup(self) -> None:
        self._entries.clear()

    @property
    def keys(self) -> list[str]:
        """The step keys with a stored entry, in insertion order (test/inspection helper)."""
        return list(self._entries)


class JSONFileDurableStorage:
    """`DurableStorage` that persists all entries as a single JSON file.

    The whole store is one file so a step's write is atomic relative to the map of steps: the file
    survives across process restarts (the offline analog of an Airflow task retry reading a cache
    that outlived the failed attempt). `cleanup` deletes the file.
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)

    def _read(self) -> dict[str, dict[str, JsonValue]]:
        try:
            loaded: JsonValue = json.loads(self._path.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            return {}
        # A file present but not shaped like our store is treated as empty rather than crashing.
        if not isinstance(loaded, dict):  # pragma: no cover - defensive against a foreign file
            return {}
        return {key: value for key, value in loaded.items() if isinstance(value, dict)}

    def load(self, key: str) -> StoredEntry | None:
        raw = self._read().get(key)
        if raw is None:
            return None
        fingerprint = raw.get('fingerprint')
        return StoredEntry(
            value=raw.get('value'),
            fingerprint=fingerprint if isinstance(fingerprint, str) else None,
        )

    def save(self, key: str, value: JsonValue, *, fingerprint: str | None) -> None:
        data = self._read()
        data[key] = {'value': value, 'fingerprint': fingerprint}
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(data))

    def cleanup(self) -> None:
        with contextlib.suppress(FileNotFoundError):
            self._path.unlink()
