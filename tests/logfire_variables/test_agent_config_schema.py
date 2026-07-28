"""Pin `AGENT_CONFIG_JSON_SCHEMA` against `AgentConfig` and against the Logfire UI's copy of it.

The schema is stored on the `agent__<name>` variable by whichever side creates it first -- this SDK
or the Logfire Agent Control UI -- and the Logfire backend validates every later version of the value
against it. So the schema has to describe everything `AgentConfig` can emit, and reject nothing
`AgentConfig` tolerates, or a value one half of the contract writes becomes unwritable or unreadable
for the other.

`jsonschema` is not a dependency of this package, so the checks below run against a small validator
covering the keywords the canonical schema actually uses.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from pydantic_ai_harness.logfire import (
    AGENT_CONFIG_JSON_SCHEMA,
    AgentConfig,
    AgentConfigSettings,
    ToolDefinitionOverride,
)

LOCKSTEP = (
    'AGENT_CONFIG_JSON_SCHEMA is one half of a contract with the Logfire UI: update '
    'src/services/logfire-frontend/src/app/project/managed-agents/agent-config.ts in the Logfire '
    'platform repo in lockstep, or a UI-created variable and an SDK-created one will store different '
    'schemas for the same agent.'
)

SETTINGS_SCHEMA: dict[str, Any] = AGENT_CONFIG_JSON_SCHEMA['properties']['settings']
TOOL_OVERRIDE_SCHEMA: dict[str, Any] = AGENT_CONFIG_JSON_SCHEMA['properties']['tool_definitions'][
    'additionalProperties'
]

# Every section populated, with every canonical setting and every override field, so the assertions
# below cover the whole surface rather than the fields that happen to be interesting.
FULL_VALUE: dict[str, Any] = {
    'instructions': 'Be concise.',
    'model': 'openai:gpt-5',
    'settings': {
        'max_tokens': 2048,
        'temperature': 0.4,
        'top_p': 0.9,
        'top_k': 40,
        'seed': 7,
        'presence_penalty': 0.1,
        'frequency_penalty': 0.2,
        'parallel_tool_calls': True,
        'timeout': 30.0,
        'stop_sequences': ['STOP'],
        'thinking': 'high',
        'service_tier': 'flex',
        'provider_options': {'anthropic': {'thinking': {'type': 'enabled', 'budget_tokens': 16384}}},
    },
    'tool_definitions': {
        'get_weather': {
            'new_name': 'lookup_weather',
            'description': 'Look up the current weather for a city.',
            'parameter_descriptions': {'city': "City name, e.g. 'London'"},
        }
    },
}

_JSON_TYPES: dict[str, type | tuple[type, ...]] = {
    'object': dict,
    'array': list,
    'string': str,
    'integer': int,
    'number': (int, float),
    'boolean': bool,
}


def schema_errors(schema: dict[str, Any], value: Any, path: str = 'value') -> list[str]:
    """Validate `value` against the JSON Schema subset the canonical schema uses."""
    errors: list[str] = []
    if 'anyOf' in schema:
        if all(schema_errors(option, value, path) for option in schema['anyOf']):
            errors.append(f'{path}: matches none of the allowed types')
        return errors
    expected: str = schema['type']
    # `bool` is an `int` subclass, so a numeric schema has to reject `True` explicitly.
    if not isinstance(value, _JSON_TYPES[expected]) or (expected in ('integer', 'number') and isinstance(value, bool)):
        return [f'{path}: expected {expected}, got {type(value).__name__}']
    if expected == 'string' and len(value) < schema.get('minLength', 0):
        errors.append(f'{path}: shorter than minLength')
    elif expected == 'object':
        members: dict[str, Any] = value
        properties: dict[str, Any] = schema.get('properties', {})
        additional: dict[str, Any] | bool = schema.get('additionalProperties', True)
        for key, item in members.items():
            item_schema = properties.get(key, additional)
            if item_schema is False:
                errors.append(f'{path}.{key}: additional properties are not allowed')
            elif item_schema is not True:
                errors.extend(schema_errors(item_schema, item, f'{path}.{key}'))
    elif expected == 'array':
        entries: list[Any] = value
        for index, item in enumerate(entries):
            errors.extend(schema_errors(schema['items'], item, f'{path}[{index}]'))
    return errors


def subschemas(schema: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """The schema and every subschema reachable from it."""
    yield schema
    properties: dict[str, Any] = schema.get('properties', {})
    for value in properties.values():
        yield from subschemas(value)
    additional: dict[str, Any] | bool | None = schema.get('additionalProperties')
    if isinstance(additional, dict):
        yield from subschemas(additional)
    items: dict[str, Any] | None = schema.get('items')
    if items is not None:
        yield from subschemas(items)
    options: list[dict[str, Any]] = schema.get('anyOf', [])
    for option in options:
        yield from subschemas(option)


def test_every_model_field_is_described() -> None:
    assert set(AgentConfig.model_fields) == set(AGENT_CONFIG_JSON_SCHEMA['properties']), LOCKSTEP
    assert set(AgentConfigSettings.model_fields) == set(SETTINGS_SCHEMA['properties']), LOCKSTEP
    assert set(ToolDefinitionOverride.model_fields) == set(TOOL_OVERRIDE_SCHEMA['properties']), LOCKSTEP


def test_schema_is_permissive_at_every_level() -> None:
    # A closed schema anywhere would make the Logfire backend reject a write that `AgentConfig` is
    # built to tolerate, so the forward-compatibility contract lives in the stored schema too.
    for subschema in subschemas(AGENT_CONFIG_JSON_SCHEMA):
        assert subschema.get('additionalProperties') is not False, LOCKSTEP
        assert 'required' not in subschema, LOCKSTEP
        assert 'enum' not in subschema, LOCKSTEP
        # Optional by omission, not by a null union, and flat: no `$defs`/`$ref` indirection and none
        # of Pydantic's `title`/`default` noise, which render badly in a form editor.
        assert subschema.get('type') != 'null', LOCKSTEP
        assert not {'$defs', '$ref', 'title', 'default'} & set(subschema), LOCKSTEP


def test_everything_agent_config_emits_validates() -> None:
    dumped = AgentConfig.model_validate(FULL_VALUE).model_dump(exclude_none=True)
    assert dumped == FULL_VALUE, LOCKSTEP
    assert schema_errors(AGENT_CONFIG_JSON_SCHEMA, dumped) == [], LOCKSTEP


def test_schema_shaped_value_round_trips() -> None:
    assert schema_errors(AGENT_CONFIG_JSON_SCHEMA, FULL_VALUE) == [], LOCKSTEP
    config = AgentConfig.model_validate(FULL_VALUE)
    assert config.instructions == FULL_VALUE['instructions']
    assert config.model == FULL_VALUE['model']
    assert config.settings == AgentConfigSettings.model_validate(FULL_VALUE['settings'])
    assert config.tool_definitions == {
        'get_weather': ToolDefinitionOverride.model_validate(FULL_VALUE['tool_definitions']['get_weather'])
    }


def test_boolean_thinking_and_empty_sections_validate() -> None:
    assert schema_errors(AGENT_CONFIG_JSON_SCHEMA, {'settings': {'thinking': True}, 'tool_definitions': {}}) == []


def test_unknown_keys_are_accepted_everywhere() -> None:
    # `AgentConfig` ignores keys it doesn't know so a value written by a newer UI degrades to the
    # sections this SDK understands. The stored schema has to accept them for that value to be
    # writable at all.
    value = {
        'future_section': {'anything': 1},
        'settings': {'future_setting': 'raw json'},
        'tool_definitions': {'get_weather': {'future_override': ['x']}},
    }
    assert schema_errors(AGENT_CONFIG_JSON_SCHEMA, value) == [], LOCKSTEP
    assert AgentConfig.model_validate(value).settings is not None


def test_wrong_types_are_rejected() -> None:
    assert schema_errors(AGENT_CONFIG_JSON_SCHEMA, {'instructions': 5}) == [
        'value.instructions: expected string, got int'
    ]
    assert schema_errors(AGENT_CONFIG_JSON_SCHEMA, {'settings': {'max_tokens': True}}) == [
        'value.settings.max_tokens: expected integer, got bool'
    ]
    assert schema_errors(AGENT_CONFIG_JSON_SCHEMA, {'settings': {'thinking': 5}}) == [
        'value.settings.thinking: matches none of the allowed types'
    ]
    assert schema_errors(AGENT_CONFIG_JSON_SCHEMA, {'settings': {'stop_sequences': [1]}}) == [
        'value.settings.stop_sequences[0]: expected string, got int'
    ]
    # An empty `new_name` is the one value constraint the schema keeps: it is structural rather than
    # versioned, so it is worth refusing at write time instead of leaving the SDK to drop the override
    # (see `test_agent_config_skew.py`) on every read.
    assert schema_errors(AGENT_CONFIG_JSON_SCHEMA, {'tool_definitions': {'t': {'new_name': ''}}}) == [
        'value.tool_definitions.t.new_name: shorter than minLength'
    ]


def test_closed_schema_would_reject_a_newer_uis_key() -> None:
    # What an `additionalProperties: false` root (the schema the UI used to write) does to the same
    # value, and why neither side stores one.
    closed = {**AGENT_CONFIG_JSON_SCHEMA, 'additionalProperties': False}
    assert schema_errors(closed, {'future_section': {}}) == [
        'value.future_section: additional properties are not allowed'
    ]
