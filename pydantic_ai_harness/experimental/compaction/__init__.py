"""Deprecated import location for `pydantic_ai_harness.compaction`.

This capability graduated out of `experimental`; importing from here still works but
emits a `HarnessDeprecationWarning`. Import from `pydantic_ai_harness.compaction` instead.
"""

from pydantic_ai_harness.compaction import (
    ClampOversizedMessages,
    ClearToolResults,
    CompactionStrategy,
    DeduplicateFileReads,
    SlidingWindowCompaction,
    SummarizingCompaction,
    TieredCompaction,
    TranscriptStore,
    WarningKind,
    WarnNearLimits,
    estimate_token_count,
    format_receipt,
    is_pinned,
    pin,
)
from pydantic_ai_harness.experimental._warn import warn_moved

warn_moved('compaction', 'compaction')

LimitWarner = WarnNearLimits
SlidingWindow = SlidingWindowCompaction

__all__ = [
    'ClampOversizedMessages',
    'ClearToolResults',
    'CompactionStrategy',
    'DeduplicateFileReads',
    'LimitWarner',
    'SlidingWindow',
    'SummarizingCompaction',
    'TieredCompaction',
    'TranscriptStore',
    'WarningKind',
    'estimate_token_count',
    'format_receipt',
    'is_pinned',
    'pin',
]
