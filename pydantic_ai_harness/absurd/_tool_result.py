from __future__ import annotations

from absurd_sdk import JsonValue
from pydantic import TypeAdapter
from pydantic_ai.durable_exec._toolset import (
    CallToolResult,
    _ToolContentResult,  # pyright: ignore[reportPrivateUsage]
    _ToolReturn,  # pyright: ignore[reportPrivateUsage]
    unwrap_recorded_tool_call_result,
)
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import ToolReturn, ToolReturnContent

_ENVELOPE_KEY = '__pydantic_ai_harness_call_tool_result__'
_ENVELOPE_VERSION = 1
_result_adapter: TypeAdapter[CallToolResult] = TypeAdapter(CallToolResult)


def serialize_tool_call_result(result: CallToolResult) -> JsonValue | ToolReturn | ToolReturnContent:
    """Serialize a result, retaining standalone's raw shape for successful calls."""
    if isinstance(result, _ToolReturn):
        return result.result
    if isinstance(result, _ToolContentResult):
        return result.result
    return {
        _ENVELOPE_KEY: {
            'version': _ENVELOPE_VERSION,
            'result': _result_adapter.dump_python(result, mode='json'),
        }
    }


def unwrap_tool_call_checkpoint(payload: JsonValue) -> JsonValue:
    """Decode current checkpoints while preserving raw standalone checkpoints."""
    if not isinstance(payload, dict) or _ENVELOPE_KEY not in payload:
        return payload

    envelope = payload[_ENVELOPE_KEY]
    if not isinstance(envelope, dict):
        raise UserError('Malformed Absurd tool-result checkpoint envelope: expected an object.')
    version = envelope.get('version')
    if type(version) is not int or version != _ENVELOPE_VERSION:
        raise UserError(f'Unsupported Absurd tool-result checkpoint envelope version: {version!r}.')

    result = _result_adapter.validate_python(envelope.get('result'))
    return unwrap_recorded_tool_call_result(result)
