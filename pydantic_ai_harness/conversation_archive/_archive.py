"""Read/write of the append-only conversation-history archive file.

The archive is a flat JSON file holding the full message history serialized via
`ModelMessagesTypeAdapter`, matching the format the `search_conversation_history`
tool reads. This module owns the write path -- capturing messages before a
compaction capability prunes them from the live history -- and the robust loader
shared with the search toolset.
"""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path

from pydantic_ai.messages import (
    ModelMessage,
    ModelMessagesTypeAdapter,
    ModelRequest,
    SystemPromptPart,
)

_logger = logging.getLogger(__name__)

_SUMMARY_PREFIX = 'Summary of previous conversation:'
"""Prefix a `SummarizingCompaction` writes into the summary artifact it inserts.

Mirrors `pydantic_ai_harness.compaction._summarizing_compaction._SUMMARY_PREFIX`.
Kept as a local literal rather than an import so this capability does not couple to
compaction internals: the archive stores the originals a summary replaced, never the
derived summary, so any later turn whose live history already carries the summary
artifact must skip it.
"""


def message_hash(message: ModelMessage) -> str:
    """Return a stable content hash of a single message.

    Dedup keys off serialized content, not object identity: durable executors
    (Temporal, DBOS) serialize state between steps, so the same message arrives as a
    fresh object each turn and identity-based dedup would re-append it. The hash is
    computed over the same `ModelMessagesTypeAdapter` bytes used to persist the
    archive, so it is stable across re-instantiation and replay.
    """
    return hashlib.sha256(ModelMessagesTypeAdapter.dump_json([message])).hexdigest()


def is_summary_artifact(message: ModelMessage) -> bool:
    """Return whether a message is a compaction summary artifact (never archived)."""
    if not isinstance(message, ModelRequest):
        return False
    return any(
        isinstance(part, SystemPromptPart) and part.content.startswith(_SUMMARY_PREFIX) for part in message.parts
    )


def load_archive(path: str | os.PathLike[str]) -> list[ModelMessage]:
    """Load archived messages, distinguishing "no archive" from "corrupt archive".

    A missing file is the normal empty case (silent). An unreadable or unparsable
    file is logged and treated as empty so a corrupt or schema-drifted archive is not
    silently reported to the model as an empty history.
    """
    file_path = Path(path)
    if not file_path.exists():
        return []
    try:
        raw = file_path.read_bytes()
    except OSError:
        _logger.warning('Could not read conversation archive %s', file_path, exc_info=True)
        return []
    if not raw:
        return []
    try:
        return list(ModelMessagesTypeAdapter.validate_json(raw))
    except Exception:
        _logger.warning(
            'Conversation archive %s exists but failed to parse; treating as empty',
            file_path,
            exc_info=True,
        )
        return []


def write_archive(path: str | os.PathLike[str], messages: list[ModelMessage]) -> None:
    """Persist the full archive as `ModelMessagesTypeAdapter` JSON."""
    file_path = Path(path)
    parent = file_path.parent
    if parent != Path(''):
        parent.mkdir(parents=True, exist_ok=True)
    file_path.write_bytes(ModelMessagesTypeAdapter.dump_json(messages, indent=2))
