"""What an `AgentConfig` written by a newer Logfire UI or Pydantic AI costs an older SDK.

The stored schema is permissive at every level so a newer writer's keys and values reach the SDK at
all, which makes this module the other half of that contract: whatever gets through has to degrade
the narrowest unit that contains it. An `AgentConfig` that fails validation is reverted *whole* by
Logfire's resolution fallback, so a value this SDK can't act on must cost one setting or one tool
override -- never the instructions, the model, and every other override along with it.
"""

from __future__ import annotations

import warnings
from typing import Any

import pytest
from pydantic import ValidationError
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from pydantic_ai_harness.logfire import (
    AgentConfig,
    AgentConfigSettings,
    AgentControl,
    ToolDefinitionOverride,
    _agent_control,
)

pytestmark = pytest.mark.anyio


@pytest.fixture(autouse=True)
def _forget_warned_drops() -> None:
    """Start each test with an empty once-per-process guard so every drop warns independently."""
    _agent_control._warned_drops.clear()


def full_value(**settings: Any) -> dict[str, Any]:
    """A config with every section populated, so a drop's blast radius is visible in the rest."""
    return {
        'instructions': 'Be concise.',
        'model': 'openai:gpt-5',
        'settings': {'temperature': 0.4, **settings},
        'tool_definitions': {'get_weather': {'description': 'Look it up.'}},
    }


def assert_other_sections_survived(config: AgentConfig) -> None:
    assert config.instructions == 'Be concise.'
    assert config.model == 'openai:gpt-5'
    assert config.settings is not None and config.settings.temperature == 0.4
    assert config.tool_definitions == {'get_weather': ToolDefinitionOverride(description='Look it up.')}


def test_unrecognized_thinking_drops_only_that_setting() -> None:
    with pytest.warns(UserWarning, match=r"sets 'thinking' to 'ultra', which this version of the SDK"):
        config = AgentConfig.model_validate(full_value(thinking='ultra', service_tier='flex'))
    assert config.settings is not None
    assert config.settings.thinking is None
    assert config.settings.service_tier == 'flex'
    # Dropped, not smuggled through as an extra key, so `_lower_settings` never emits it.
    assert config.settings.model_extra == {}
    assert_other_sections_survived(config)


def test_unrecognized_service_tier_drops_only_that_setting() -> None:
    with pytest.warns(UserWarning, match=r"sets 'service_tier' to 'realtime'"):
        config = AgentConfig.model_validate(full_value(thinking='high', service_tier='realtime'))
    assert config.settings is not None
    assert config.settings.thinking == 'high'
    assert config.settings.service_tier is None
    assert_other_sections_survived(config)


def test_recognized_values_are_untouched() -> None:
    with warnings.catch_warnings(record=True) as caught:
        settings = AgentConfigSettings.model_validate({'thinking': True, 'service_tier': 'priority'})
    assert caught == []
    assert (settings.thinking, settings.service_tier) == (True, 'priority')


def test_unknown_keys_still_flow_through_untouched() -> None:
    # The key-level tolerance this SDK already promised: only *known* fields with unrecognized values
    # are dropped, so a newer UI's keys keep reaching `extra='allow'` and the ignored top level.
    with warnings.catch_warnings(record=True) as caught:
        config = AgentConfig.model_validate(
            {
                'future_section': {'anything': 1},
                'settings': {'future_setting': 'raw json', 'thinking': 'high'},
                'tool_definitions': {'get_weather': {'future_override': ['x'], 'description': 'Look it up.'}},
            }
        )
    assert caught == []
    assert config.settings is not None
    assert config.settings.model_extra == {'future_setting': 'raw json'}
    assert config.settings.thinking == 'high'
    assert config.tool_definitions == {'get_weather': ToolDefinitionOverride(description='Look it up.')}


def test_invalid_override_drops_only_that_tool() -> None:
    with pytest.warns(UserWarning, match=r"override for 'get_weather' is invalid -- new_name=''"):
        config = AgentConfig.model_validate(
            {
                'instructions': 'Be concise.',
                'tool_definitions': {
                    'get_weather': {'new_name': '', 'description': 'Dropped with its entry.'},
                    'get_forecast': {'new_name': 'forecast'},
                },
            }
        )
    assert config.instructions == 'Be concise.'
    assert config.tool_definitions == {'get_forecast': ToolDefinitionOverride(new_name='forecast')}


def test_override_that_is_not_an_object_drops_only_that_tool() -> None:
    with pytest.warns(UserWarning, match=r"override for 'get_weather' is invalid -- override='nope'"):
        config = AgentConfig.model_validate(
            {'tool_definitions': {'get_weather': 'nope', 'get_forecast': {'description': 'Kept.'}}}
        )
    assert config.tool_definitions == {'get_forecast': ToolDefinitionOverride(description='Kept.')}


def test_repeated_resolutions_warn_once() -> None:
    # A managed config is resolved on every run, so the same unrecognized value must not warn per run.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        for _ in range(3):
            AgentConfig.model_validate(full_value(thinking='ultra'))
        AgentConfig.model_validate(full_value(thinking='hyper'))
    assert [str(warning.message).split(',')[0] for warning in caught] == [
        "Managed agent config sets 'thinking' to 'ultra'",
        "Managed agent config sets 'thinking' to 'hyper'",
    ]


def test_malformed_sections_are_left_to_pydantic() -> None:
    # Neither validator can assume it was handed a mapping. A section that isn't one is malformed
    # rather than merely newer -- the stored schema rejects it at write time -- so it stays a
    # validation error, and the whole value degrades to code through Logfire's resolution fallback.
    with warnings.catch_warnings(record=True) as caught:
        for value in ({'settings': 'nope'}, {'tool_definitions': ['nope']}):
            with pytest.raises(ValidationError):
                AgentConfig.model_validate(value)
        with pytest.raises(ValidationError):
            AgentConfigSettings.model_validate(['nope'])
    assert caught == []


async def test_agent_keeps_managed_config_around_a_dropped_setting() -> None:
    seen: list[dict[str, object]] = []

    def capture(_messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen.append(dict(info.model_settings or {}))
        return ModelResponse(parts=[TextPart('done')])

    with pytest.warns(UserWarning, match=r"sets 'thinking' to 'ultra'"):
        config = AgentConfig.model_validate({'settings': {'temperature': 0.4, 'thinking': 'ultra'}})
    await Agent(FunctionModel(capture), capabilities=[AgentControl('skew', default=config)]).run('hello')
    assert seen == [{'temperature': 0.4}]
