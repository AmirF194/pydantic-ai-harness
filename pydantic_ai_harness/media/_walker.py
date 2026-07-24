"""JSON walkers that swap inline large parts for externalized markers.

Used by step-persistence backends to keep large payloads out of snapshot
records. The walkers operate on JSON-shaped trees (`list`/`dict`/scalars)
so they never need to know about `pydantic_ai.messages` types directly --
they recognize the serialized shapes emitted by
`ModelMessagesTypeAdapter.dump_json`:

- `BinaryContent`: `{"kind": "binary", "data": "<b64>", ...}`.
- Any part carrying a string `content` (`TextPart`, `ToolReturnPart`,
  a string-valued `UserPromptPart`, ...): `{"content": "<text>", ...}`.

Both kinds go to the same content-addressed `MediaStore`; the marker records
which field was externalized so `restore_media` re-inlines the right one.
Externalizing large text (not just binary) keeps a single snapshot below
MongoDB's 16MB document cap when a tool returns a large string.

Round-trip: `externalize_media` (pre-write) and `restore_media` (post-read)
are inverses for payloads whose byte length is >= threshold. Payloads below
the threshold pass through unchanged.
"""

from __future__ import annotations

import base64
from typing import TypeGuard

from pydantic_ai_harness.media._store import MediaContext, MediaStore

_EXTERNAL_MARKER = '__harness_external_media__'
_TEXT_MARKER = '__harness_external_text__'


def _is_json_dict(node: object) -> TypeGuard[dict[str, object]]:
    """Treat any `dict` returned by `json.loads` as `dict[str, object]`.

    JSON guarantees string keys, so the runtime `isinstance(..., dict)` check
    is sufficient -- pyright just needs the explicit TypeGuard to propagate
    the value-type narrowing through the walker.
    """
    return isinstance(node, dict)


def _is_json_list(node: object) -> TypeGuard[list[object]]:
    """Treat any `list` returned by `json.loads` as `list[object]`.

    Same rationale as `_is_json_dict` -- JSON lists contain JSON-compatible
    scalars, and we walk every element regardless.
    """
    return isinstance(node, list)


def _is_binary_part(node: dict[str, object]) -> bool:
    return node.get('kind') == 'binary' and isinstance(node.get('data'), str)


def _is_text_part(node: dict[str, object]) -> bool:
    """A serialized message part carrying inline text.

    Requires a string `part_kind` -- the discriminator every `pydantic_ai`
    message part serializes -- as well as a string `content`. Gating on
    `part_kind` keeps arbitrary nested data that merely happens to carry a
    string `content` key (e.g. a tool-return payload with its own `content`
    field) from being mistaken for a part and having its fields rewritten.
    Covers `TextPart`, `ToolReturnPart` with a string return, and
    string-valued `UserPromptPart`. A bare string element inside a
    `UserPromptPart.content` list is not a dict, so it is never a candidate
    (replacing it with a marker dict would break `UserContent` validation).
    """
    return isinstance(node.get('part_kind'), str) and isinstance(node.get('content'), str)


def _is_external_marker(node: dict[str, object]) -> bool:
    return node.get(_EXTERNAL_MARKER) is True


async def externalize_media(node: object, *, media_store: MediaStore, threshold_bytes: int) -> object:
    """Walk `node` and replace inline binary/text parts >= threshold.

    Each qualifying part becomes a marker dict carrying the canonical
    `media+sha256://` URI; the raw bytes are written to `media_store`.
    Returns a new tree -- input is not mutated.
    """
    if _is_json_list(node):
        out_list: list[object] = []
        for item in node:
            out_list.append(await externalize_media(item, media_store=media_store, threshold_bytes=threshold_bytes))
        return out_list
    if _is_json_dict(node):
        if _is_binary_part(node):
            replaced = await _maybe_externalize_binary(node, media_store, threshold_bytes)
            if replaced is not None:
                return replaced
        elif _is_text_part(node):
            replaced = await _maybe_externalize_text(node, media_store, threshold_bytes)
            if replaced is not None:
                return replaced
        out_dict: dict[str, object] = {}
        for key, value in node.items():
            out_dict[key] = await externalize_media(value, media_store=media_store, threshold_bytes=threshold_bytes)
        return out_dict
    return node


async def _maybe_externalize_binary(
    node: dict[str, object],
    media_store: MediaStore,
    threshold_bytes: int,
) -> dict[str, object] | None:
    # `_is_binary_part` already verified `data` is a string; the cast is safe.
    data_value = node['data']
    assert isinstance(data_value, str)
    raw = base64.b64decode(data_value)
    if len(raw) < threshold_bytes:
        return None
    media_type_value = node.get('media_type')
    media_type = media_type_value if isinstance(media_type_value, str) else None
    uri = await media_store.put(raw, context=MediaContext(media_type=media_type))
    # Preserve every field except the externalized `data`, so the round-trip
    # survives new `BinaryContent` fields added by pydantic_ai upstream. Fields
    # are recursively walked so any nested externalizable payload still goes out.
    marker = await _preserve_fields(node, 'data', media_store, threshold_bytes)
    marker[_EXTERNAL_MARKER] = True
    marker['uri'] = uri
    return marker


async def _maybe_externalize_text(
    node: dict[str, object],
    media_store: MediaStore,
    threshold_bytes: int,
) -> dict[str, object] | None:
    # `_is_text_part` already verified `content` is a string.
    content_value = node['content']
    assert isinstance(content_value, str)
    raw = content_value.encode('utf-8')
    if len(raw) < threshold_bytes:
        return None
    uri = await media_store.put(raw, context=MediaContext(media_type='text/plain'))
    # Preserve every field except the externalized `content`, recursively walked
    # so a nested payload (e.g. binary in `metadata`) still goes external too.
    # `_TEXT_MARKER` tells `restore_media` to decode UTF-8 back into `content`
    # rather than base64 into `data`.
    marker = await _preserve_fields(node, 'content', media_store, threshold_bytes)
    marker[_EXTERNAL_MARKER] = True
    marker[_TEXT_MARKER] = True
    marker['uri'] = uri
    return marker


async def _preserve_fields(
    node: dict[str, object],
    externalized_key: str,
    media_store: MediaStore,
    threshold_bytes: int,
) -> dict[str, object]:
    """Recursively externalize every field of `node` except `externalized_key`."""
    out: dict[str, object] = {}
    for key, value in node.items():
        if key == externalized_key:
            continue
        out[key] = await externalize_media(value, media_store=media_store, threshold_bytes=threshold_bytes)
    return out


async def restore_media(node: object, *, media_store: MediaStore) -> object:
    """Inverse of `externalize_media`. Walks `node` and re-inlines external refs.

    Each marker dict's `uri` is resolved via `media_store.get`; a text marker
    re-inlines `content`, a binary marker re-inlines base64 `data`, so the
    result round-trips through `ModelMessagesTypeAdapter.validate_python`.
    """
    if _is_json_list(node):
        out_list: list[object] = []
        for item in node:
            out_list.append(await restore_media(item, media_store=media_store))
        return out_list
    if _is_json_dict(node):
        if _is_external_marker(node):
            return await _restore_external(node, media_store)
        out_dict: dict[str, object] = {}
        for key, value in node.items():
            out_dict[key] = await restore_media(value, media_store=media_store)
        return out_dict
    return node


async def _restore_external(node: dict[str, object], media_store: MediaStore) -> dict[str, object]:
    uri_value = node.get('uri')
    if not isinstance(uri_value, str):
        raise ValueError(f'externalized media marker missing string uri: {node!r}')
    raw = await media_store.get(uri_value)
    # Inverse of `_maybe_externalize_*`: drop the marker keys, restore the
    # externalized field, keep every other field the original part carried.
    # Preserved fields are recursively restored so a nested marker (from the
    # externalize-side recursion) is re-inlined too.
    restored: dict[str, object] = {}
    for key, value in node.items():
        if key in (_EXTERNAL_MARKER, _TEXT_MARKER, 'uri'):
            continue
        restored[key] = await restore_media(value, media_store=media_store)
    if node.get(_TEXT_MARKER) is True:
        restored['content'] = raw.decode('utf-8')
    else:
        restored['data'] = base64.b64encode(raw).decode('ascii')
    return restored
