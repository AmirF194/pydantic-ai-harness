"""Shared helpers for model-visible tool output."""


def truncate_tail(text: str, max_chars: int, *, preserve_prefix_chars: int = 0) -> str:
    """Limit text to `max_chars`, including an accurate truncation marker."""
    if max_chars <= 0:
        return ''
    if len(text) <= max_chars:
        return text
    if preserve_prefix_chars:
        prefix_chars = min(preserve_prefix_chars, max_chars)
        return text[:prefix_chars] + truncate_tail(text[preserve_prefix_chars:], max_chars - prefix_chars)

    def marker(tail_chars: int) -> str:
        return f'[... output truncated, showing last {tail_chars} chars]\n'

    tail_chars = max(0, max_chars - len(marker(max_chars)))
    if tail_chars + 1 + len(marker(tail_chars + 1)) <= max_chars:
        tail_chars += 1
    if tail_chars == 0:
        return text[-max_chars:]
    return marker(tail_chars) + text[-tail_chars:]
