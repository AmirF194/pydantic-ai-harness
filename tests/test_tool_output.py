"""Tests for the shared tool-output presentation helpers."""

from __future__ import annotations

import pytest
from pydantic_ai.exceptions import ModelRetry

from pydantic_ai_harness._tool_output import (
    format_size,
    guard_read_size,
    render_file_window,
    truncate,
    truncate_output,
)


class TestGuardReadSize:
    def test_allows_at_or_under_limit(self) -> None:
        guard_read_size(1000, max_bytes=1000)  # exactly at the limit: no raise

    def test_refuses_over_limit(self) -> None:
        with pytest.raises(ModelRetry, match='File is 2.0KB, over the 1000B read limit'):
            guard_read_size(2048, max_bytes=1000)


class TestFormatSize:
    @pytest.mark.parametrize(
        ('num_bytes', 'expected'),
        [(512, '512B'), (2048, '2.0KB'), (3 * 1024 * 1024, '3.0MB')],
    )
    def test_units(self, num_bytes: int, expected: str) -> None:
        assert format_size(num_bytes) == expected


class TestTruncate:
    def test_keeps_everything_under_caps(self) -> None:
        result = truncate(['a', 'b', 'c'])
        assert result.truncated_lines == ['a', 'b', 'c']
        assert result.truncated is False

    def test_line_cap(self) -> None:
        result = truncate(['x', 'y', 'z'], max_lines=2)
        assert result.truncated_lines == ['x', 'y']
        assert result.truncated_by == 'lines'

    def test_byte_cap(self) -> None:
        result = truncate(['aaa', 'bbb', 'ccc'], max_bytes=4)
        # First line (3B) fits; second would cost 3 + 1 newline = 4 -> 3+4 > 4 -> stop.
        assert result.truncated_lines == ['aaa']
        assert result.truncated_by == 'bytes'

    def test_tail_keeps_last_lines(self) -> None:
        result = truncate(['a', 'b', 'c'], max_lines=2, direction='tail')
        assert result.truncated_lines == ['b', 'c']
        assert result.truncated_by == 'lines'

    def test_first_line_exceeded(self) -> None:
        result = truncate(['x' * 100], max_bytes=10)
        assert result.truncated_lines == []
        assert result.first_line_exceeded is True
        assert result.truncated_by == 'bytes'

    def test_tail_keeps_partial_suffix_of_oversized_last_line(self) -> None:
        # Command output (tail) keeps the end of a single over-long line, not nothing, so
        # the error/exit region survives.
        result = truncate(['x' * 100], max_bytes=10, direction='tail')
        assert result.truncated_lines == ['x' * 10]
        assert result.truncated_by == 'bytes'
        assert result.first_line_exceeded is False

    def test_tail_suffix_drops_partial_multibyte_char(self) -> None:
        # Cutting the byte-suffix mid-character must not raise and must drop the partial byte.
        result = truncate(['é' * 10], max_bytes=5, direction='tail')  # 'é' is 2 bytes
        assert result.truncated_lines == ['éé']  # 4 bytes; the partial leading byte is dropped
        assert result.truncated_by == 'bytes'


class TestTruncateOutput:
    def test_untruncated_returns_body(self) -> None:
        assert truncate_output('a\nb') == 'a\nb'

    def test_tail_marker_on_top(self) -> None:
        out = truncate_output('a\nb\nc', max_lines=1)
        assert out.startswith('[... output truncated')
        assert out.endswith('c')

    def test_head_marker_on_bottom(self) -> None:
        out = truncate_output('a\nb\nc', max_lines=1, direction='head')
        assert out.startswith('a')
        # Head keeps the first lines, so the marker must say "first", not "last".
        assert 'truncated to the first' in out
        assert out.rstrip().endswith(']')

    def test_tail_marker_says_last(self) -> None:
        out = truncate_output('a\nb\nc', max_lines=1, direction='tail')
        assert 'truncated to the last' in out

    def test_line_cap_marker_names_lines_not_bytes(self) -> None:
        # The line cap fired, so the marker must report lines, not the byte cap.
        out = truncate_output('a\nb\nc', max_lines=1, direction='head')
        assert 'truncated to the first 1 lines' in out

    def test_byte_cap_marker_names_bytes(self) -> None:
        out = truncate_output('aaaa\nbbbb\ncccc', max_bytes=4, direction='head')
        assert 'truncated to the first 4B' in out

    def test_oversized_single_line_command_keeps_suffix(self) -> None:
        out = truncate_output('y' * 100, max_bytes=10, direction='tail')
        assert out.startswith('[... output truncated to the last 10B ...]')
        assert out.rstrip().endswith('y' * 10)


class TestRenderFileWindow:
    def test_returns_full_small_file(self) -> None:
        assert render_file_window(b'one\ntwo') == 'one\ntwo'

    def test_offset_and_limit_window(self) -> None:
        data = b'l1\nl2\nl3\nl4\nl5'
        out = render_file_window(data, offset=2, limit=2)
        assert out.startswith('l2\nl3')
        assert '2 more lines in file. Use offset=4 to continue.' in out

    def test_offset_below_one_rejected(self) -> None:
        with pytest.raises(ModelRetry, match='offset must be >= 1'):
            render_file_window(b'x', offset=0)

    def test_limit_below_one_rejected(self) -> None:
        with pytest.raises(ModelRetry, match='limit must be >= 1'):
            render_file_window(b'x', limit=0)

    def test_offset_beyond_end_rejected(self) -> None:
        with pytest.raises(ModelRetry, match='beyond end of file'):
            render_file_window(b'one\ntwo', offset=99)

    def test_binary_rejected(self) -> None:
        with pytest.raises(ModelRetry, match='not valid UTF-8'):
            render_file_window(b'\xff\xfe\x00')

    def test_byte_cap_emits_continuation_offset(self) -> None:
        data = b'\n'.join(b'line%02d' % i for i in range(10))
        out = render_file_window(data, max_bytes=12)
        assert 'limit). Use offset=' in out

    def test_line_cap_emits_continuation_offset(self) -> None:
        data = b'\n'.join(b'l' for _ in range(10))
        out = render_file_window(data, max_lines=3)
        assert 'Showing lines 1-3 of 10. Use offset=4 to continue.' in out
        assert 'limit)' not in out

    def test_first_line_too_big_is_omitted(self) -> None:
        # A single oversized line with nothing after it: no continuation to offer.
        out = render_file_window(b'x' * 200, max_bytes=10)
        assert out == '[Line 1 is 200B, exceeds the 10B limit and was omitted.]'

    def test_first_line_too_big_points_past_it(self) -> None:
        # An oversized first line with more lines after it must tell the model how to page on.
        out = render_file_window(b'x' * 200 + b'\nfoo', max_bytes=10)
        assert out == '[Line 1 is 200B, exceeds the 10B limit and was omitted. Use offset=2 to continue.]'

    def test_limit_reaching_eof_has_no_note(self) -> None:
        out = render_file_window(b'a\nb\nc', offset=1, limit=10)
        assert out == 'a\nb\nc'
