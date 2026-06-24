"""Shared presentation helpers for file-read and command-output tools.

Pure formatting with no I/O and no backend coupling, so any toolset can opt in to the
same windowing and truncation behavior. Ported from the execution-environment work in
PR #261 (`_truncate.py` / `_read_file`).

`read_file`-style tools want `render_file_window` (line-addressable, head-first, with a
continuation offset). Free-form command output wants `truncate_output` (tail-first, so
errors and exit status survive). Both share `truncate`, which never emits a partial line.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

from pydantic_ai.exceptions import ModelRetry

# Two independent caps; whichever is hit first wins.
DEFAULT_MAX_LINES = 2000
DEFAULT_MAX_BYTES = 50 * 1024  # 50KB

TruncatedBy: TypeAlias = Literal['lines', 'bytes'] | None


@dataclass(kw_only=True, frozen=True)
class TruncationResult:
    """What fit under the caps, and why we stopped. The caller turns this into a note."""

    truncated_lines: list[str]
    truncated_by: TruncatedBy = None
    # A single line wider than the byte cap can't be shown partially; this lets the
    # caller emit a "line too big" message instead of a normal continuation note.
    first_line_exceeded: bool = False

    @property
    def truncated(self) -> bool:
        # Derived, never stored, so it can't disagree with truncated_by.
        return self.truncated_by is not None


def format_size(num_bytes: int) -> str:
    """Render a byte count the way the continuation notes show it to the model."""
    if num_bytes < 1024:
        return f'{num_bytes}B'
    if num_bytes < 1024 * 1024:
        return f'{num_bytes / 1024:.1f}KB'
    return f'{num_bytes / (1024 * 1024):.1f}MB'


def truncate(
    lines: list[str],
    *,
    max_lines: int = DEFAULT_MAX_LINES,
    max_bytes: int = DEFAULT_MAX_BYTES,
    direction: Literal['head', 'tail'] = 'head',
) -> TruncationResult:
    """Keep the lines that fit under both caps; never emit a partial line.

    `head` keeps the first lines (file reads, top-down); `tail` keeps the last (shell
    output, where errors and the exit status live). We reverse up front so the rest of the
    function -- including the first-line guard -- always operates on "the line we'd keep
    first" regardless of direction.
    """
    if direction == 'tail':
        lines = lines[::-1]

    # A line wider than the byte cap can't be shown partially (we never split a line), so
    # keep nothing and flag it -- the caller reports the line's size and omits it.
    if lines and len(lines[0].encode('utf-8')) > max_bytes:
        return TruncationResult(truncated_lines=[], truncated_by='bytes', first_line_exceeded=True)

    kept: list[str] = []
    running_byte_size = 0
    truncated_by: TruncatedBy = None

    for line in lines:
        if len(kept) >= max_lines:
            truncated_by = 'lines'
            break
        # +1 for the '\n' that '\n'.join inserts before every line except the first.
        cost = len(line.encode('utf-8')) + (1 if kept else 0)
        if running_byte_size + cost > max_bytes:
            truncated_by = 'bytes'
            break
        kept.append(line)
        running_byte_size += cost

    if direction == 'tail':
        kept = kept[::-1]
    return TruncationResult(truncated_lines=kept, truncated_by=truncated_by)


def truncate_output(
    text: str,
    *,
    max_lines: int = DEFAULT_MAX_LINES,
    max_bytes: int = DEFAULT_MAX_BYTES,
    direction: Literal['head', 'tail'] = 'tail',
) -> str:
    """Cap free-form tool output (e.g. shell) and mark it when anything was dropped.

    Unlike `render_file_window`, this output is not line-addressable, so the model gets a
    marker rather than a continuation offset. Defaults to `tail`: command errors and exit
    status live at the end.
    """
    result = truncate(text.split('\n'), max_lines=max_lines, max_bytes=max_bytes, direction=direction)
    body = '\n'.join(result.truncated_lines)
    if not result.truncated:
        return body
    kept = 'last' if direction == 'tail' else 'first'
    marker = f'[... output truncated to the {kept} {format_size(max_bytes)} ...]'
    return f'{marker}\n{body}' if direction == 'tail' else f'{body}\n{marker}'


def render_file_window(
    data: bytes,
    *,
    offset: int | None = None,
    limit: int | None = None,
    max_lines: int = DEFAULT_MAX_LINES,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> str:
    """Decode, window, and truncate a file's bytes for a `read_file`-style tool.

    `offset`/`limit` are 1-indexed line counts (to agree with `grep -n`, editors, and
    stack traces). When the safety caps or `limit` stop the read short, the returned text
    ends with a note pointing at the next `offset` so the model can page the rest. Raises
    `ModelRetry` for bad bounds or non-UTF-8 content so the model can react.
    """
    if offset is not None and offset < 1:
        raise ModelRetry(f'offset must be >= 1 (lines are 1-indexed), got {offset}')
    if limit is not None and limit < 1:
        raise ModelRetry(f'limit must be >= 1, got {limit}')

    try:
        text = data.decode('utf-8')
    except UnicodeDecodeError:
        raise ModelRetry('file is not valid UTF-8 text (it may be a binary file).')

    # Split on '\n' only, not str.splitlines(): splitlines() also breaks on '\r', '\v',
    # '\f', and Unicode separators, which would make line numbers disagree with editors
    # and grep -n. A trailing '\n' yields a final '' element, so total_lines counts it.
    lines = text.split('\n')
    total_lines = len(lines)

    start = offset - 1 if offset is not None else 0
    if start >= total_lines:
        raise ModelRetry(f'offset {offset} is beyond end of file ({total_lines} lines total)')

    end = min(start + limit, total_lines) if limit is not None else total_lines
    window = lines[start:end]

    result = truncate(window, max_lines=max_lines, max_bytes=max_bytes, direction='head')
    start_display = start + 1  # 1-indexed line the window starts on

    if result.first_line_exceeded:
        line_size = format_size(len(lines[start].encode('utf-8')))
        return f'[Line {start_display} is {line_size}, exceeds the {format_size(max_bytes)} limit and was omitted.]'

    body = '\n'.join(result.truncated_lines)

    if result.truncated:
        # A safety cap stopped us; point the model at the exact next line.
        end_display = start_display + len(result.truncated_lines) - 1
        next_offset = end_display + 1
        limit_note = f' ({format_size(max_bytes)} limit)' if result.truncated_by == 'bytes' else ''
        return (
            f'{body}\n\n[Showing lines {start_display}-{end_display} of {total_lines}{limit_note}. '
            f'Use offset={next_offset} to continue.]'
        )

    if limit is not None and end < total_lines:
        # The model's own limit stopped us early (not the safety cap); tell it where to resume.
        remaining = total_lines - end
        return f'{body}\n\n[{remaining} more lines in file. Use offset={end + 1} to continue.]'

    return body
