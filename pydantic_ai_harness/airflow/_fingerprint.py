"""Request fingerprints for fingerprint-verified positional replay.

Durable memoization keys each operation positionally (`model_step_{N}`, `tool_step_{N}`, ...).
Position alone cannot tell whether a stored entry still corresponds to the current request: if the
prompt, model, toolset, or message history changed between the failed attempt and the retry,
replaying by position would feed the agent a response recorded for a different conversation.

Each entry therefore stores a fingerprint of the request that produced it. On a positional hit the
stored fingerprint is compared against the current request; a mismatch is treated as a miss and the
step re-runs live. A model divergence cascades to downstream tool steps: a fresh model response
carries newly generated `tool_call_id` values, and `tool_call_id` is part of the tool fingerprint,
so stale tool results recorded under the old conversation no longer match.

Fields that Pydantic AI regenerates on every attempt (message-level `timestamp`/`run_id`/
`conversation_id` and part-level `timestamp`) are excluded, following the same set the Prefect
integration excludes in its cache policy. A request that cannot be serialized fingerprints as
`None`, which degrades that step to unverified positional replay rather than disabling memoization.
"""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING, Any

from pydantic import TypeAdapter
from pydantic_ai.messages import ModelMessagesTypeAdapter
from pydantic_ai.models import ModelRequestParameters

if TYPE_CHECKING:
    from pydantic_ai.messages import ModelMessage
    from pydantic_ai.settings import ModelSettings

_MODEL_REQUEST_PARAMETERS_ADAPTER: TypeAdapter[ModelRequestParameters] = TypeAdapter(ModelRequestParameters)

# Message-level fields Pydantic AI regenerates on every attempt. Matches the set the Prefect
# integration's `_cache_policies` strips, so a faithful replay hashes identically.
_VOLATILE_MESSAGE_KEYS = frozenset({'timestamp', 'run_id', 'conversation_id'})

# Settings that control transport, not response content. Excluded so changing them does not
# invalidate a stored response, and so a non-JSON value (e.g. an `httpx.Timeout` under `timeout`)
# does not force the whole fingerprint to `None` and silently disable replay verification.
_TRANSPORT_ONLY_SETTINGS = frozenset({'timeout'})


def _content_settings(model_settings: ModelSettings | None) -> dict[str, Any] | None:
    if not model_settings:
        return None
    content = {key: value for key, value in model_settings.items() if key not in _TRANSPORT_ONLY_SETTINGS}
    return content or None


def _strip_message(message: Any) -> Any:
    """Drop per-attempt fields from one dumped message.

    Only the levels Pydantic AI regenerates are touched (message-level ids and timestamps,
    part-level timestamps); user data such as tool arguments is never recursed into, so an argument
    named `run_id` still affects the fingerprint. A dumped message's `parts` is always a list of
    dicts, so each part's `timestamp` is stripped.
    """
    cleaned = {k: v for k, v in message.items() if k not in _VOLATILE_MESSAGE_KEYS}
    if 'parts' in cleaned:
        cleaned['parts'] = [{k: v for k, v in part.items() if k != 'timestamp'} for part in cleaned['parts']]
    return cleaned


def _strip_volatile(messages_dump: Any) -> list[Any]:
    return [_strip_message(message) for message in messages_dump]


def _digest(payload: Any) -> str:
    # No `default=` fallback: a non-JSON value must raise so callers degrade to an unverifiable
    # (`None`) fingerprint instead of hashing a process-local `repr` that never matches on retry.
    canonical = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()


def fingerprint_model_request(
    model_identifier: str,
    messages: list[ModelMessage],
    model_settings: ModelSettings | None,
    model_request_parameters: ModelRequestParameters,
) -> str | None:
    """Fingerprint a model request: model identity, message history, settings, and request parameters.

    The full `ModelRequestParameters` is hashed (tool definitions, output mode and schema, native
    tools) so any change to what is sent to the model invalidates the stored response. Returns
    `None` when the request cannot be serialized, so it degrades to unverified positional replay
    rather than disabling memoization.
    """
    try:
        dumped: Any = ModelMessagesTypeAdapter.dump_python(messages, mode='json')
        params: Any = _MODEL_REQUEST_PARAMETERS_ADAPTER.dump_python(model_request_parameters, mode='json')
        return _digest(
            {
                'model': model_identifier,
                'messages': _strip_volatile(dumped),
                'settings': _content_settings(model_settings),
                'params': params,
            }
        )
    except (TypeError, ValueError):
        # A non-JSON value anywhere in the request (messages, params, or content settings) degrades
        # to an unverifiable fingerprint rather than crashing the run.
        return None


def fingerprint_tool_call(name: str, tool_args: dict[str, Any], tool_call_id: str | None) -> str | None:
    """Fingerprint a tool call: tool name, arguments, and the model-issued call id.

    `tool_call_id` round-trips through the model-response memo, so it is stable under faithful
    replay but regenerated whenever a live model call replaces a stored response, cascading
    invalidation to downstream tool steps.
    """
    try:
        return _digest({'name': name, 'args': tool_args, 'tool_call_id': tool_call_id})
    except (TypeError, ValueError):
        return None


def fingerprint_token(token: str) -> str:
    """Fingerprint a no-argument operation (MCP tool listing/instructions, dynamic resolution).

    These carry no per-attempt input, so the toolset id is used as the token: it verifies that the
    step at this position still belongs to the same toolset, catching a shape change that a bare
    positional key would miss.
    """
    return _digest(token)
