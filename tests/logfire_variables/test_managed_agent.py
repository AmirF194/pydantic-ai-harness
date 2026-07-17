from __future__ import annotations

import json
import warnings
from typing import Any, cast

import logfire
import pytest
from logfire.testing import CaptureLogfire
from logfire.variables import LabeledValue, Rollout, Variable, VariableConfig, VariablesConfig
from pydantic_ai import Agent, RunContext, Tool
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import ToolDefinition

from pydantic_ai_harness import ManagedAgent as RootManagedAgent
from pydantic_ai_harness.logfire import (
    AgentConfig,
    AgentConfigSettings,
    ManagedAgent,
    ToolDefinitionOverride,
    _managed_variable,
)

from ._helpers import advertised, capture_tools, get_weather, variables_provider

pytestmark = pytest.mark.anyio

assert RootManagedAgent is ManagedAgent


def instructions_seen(messages: list[ModelMessage]) -> list[str]:
    return [m.instructions for m in messages if isinstance(m, ModelRequest) and m.instructions is not None]


async def test_empty_config_keeps_code_behavior() -> None:
    result = await Agent(TestModel(), instructions='code', capabilities=[ManagedAgent('empty')]).run('hello')
    assert instructions_seen(result.all_messages()) == ['code']


async def test_instructions_apply_independently() -> None:
    capability = ManagedAgent('instructions', default=AgentConfig(instructions='managed'))
    result = await Agent(TestModel(), instructions='code', capabilities=[capability]).run('hello')
    assert instructions_seen(result.all_messages()) == ['code\n\nmanaged']


async def test_tool_definition_patches() -> None:
    seen: list[ToolDefinition] = []
    capability = ManagedAgent(
        'tools',
        default=AgentConfig(
            tool_definitions={
                'get_weather': ToolDefinitionOverride(
                    description='Managed.', parameter_descriptions={'city': 'Managed city.'}
                )
            }
        ),
    )
    await Agent(capture_tools(seen), tools=[get_weather], capabilities=[capability]).run('hello')
    assert advertised(seen) == {'get_weather': 'Managed.'}
    assert seen[0].parameters_json_schema['properties']['city']['description'] == 'Managed city.'


async def test_settings_schema_and_lowering() -> None:
    seen: list[dict[str, object]] = []

    def capture_settings(_messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen.append(dict(info.model_settings or {}))
        return ModelResponse(parts=[TextPart('done')])

    capability = ManagedAgent(
        'settings',
        default=AgentConfig(
            settings=AgentConfigSettings.model_validate(
                {
                    'temperature': 0.2,
                    'top_p': 0.4,
                    'provider_options': {'openai': {'temperature': 0.8}, 'custom': {'flag': True}},
                    'openai_temperature': 0.6,
                }
            )
        ),
    )
    await Agent(
        FunctionModel(capture_settings),
        model_settings={'temperature': 0.1, 'top_k': 3},
        capabilities=[capability],
    ).run('hello', model_settings={'temperature': 0.9})
    assert seen == [
        {
            'temperature': 0.9,
            'top_k': 3,
            'top_p': 0.4,
            'openai_temperature': 0.8,
            'custom_flag': True,
        }
    ]


def test_agent_config_ignores_forward_keys() -> None:
    assert AgentConfig.model_validate({'instructions': 'x', 'future': True}) == AgentConfig(instructions='x')


def test_prebuilt_variable() -> None:
    variable = Variable(
        'agent__prebuilt',
        type=AgentConfig,
        default=AgentConfig(model='test'),
        logfire_instance=logfire.DEFAULT_LOGFIRE_INSTANCE,
    )
    assert ManagedAgent(variable).get_model() == 'test'


async def test_auto_create_uses_request_snapshot(capfire: CaptureLogfire, monkeypatch: pytest.MonkeyPatch) -> None:
    _managed_variable._reset_auto_create_guard()
    created: list[VariableConfig] = []

    def create_inline(variable: Variable[object], config: VariableConfig | None = None) -> None:
        assert config is not None
        created.append(config)
        _managed_variable._create_variable(variable, config)

    monkeypatch.setattr(_managed_variable, '_spawn_create', create_inline)
    config = VariablesConfig(variables={})

    def lookup(city: str) -> str:
        """Look up a city.

        Args:
            city: City to look up.
        """
        return city

    def raw() -> str:
        return 'raw'

    raw_tool = Tool.from_schema(
        raw,
        name='raw',
        description=None,
        json_schema={
            'type': 'object',
            'properties': {'plain': 'not-a-schema', 'count': {'description': 5}, 'named': {'description': 'Named.'}},
        },
    )
    empty_tool = Tool.from_schema(raw, name='empty', description=None, json_schema={'type': 'object'})

    with variables_provider(capfire, config):
        agent = Agent(
            TestModel(),
            name='snapshot_agent',
            instructions='Code instructions.',
            model_settings={'temperature': 0.3},
            tools=[lookup, raw_tool, empty_tool],
            capabilities=[ManagedAgent()],
        )
        await agent.run('hello')

    assert len(created) == 1
    example = json.loads(created[0].example or '{}')
    assert example == {
        'instructions': 'Code instructions.',
        'model': 'test:test',
        'settings': {'temperature': 0.3},
        'tool_definitions': {
            'lookup': {
                'description': 'Look up a city.',
                'parameter_descriptions': {'city': 'City to look up.'},
            },
            'raw': {'parameter_descriptions': {'named': 'Named.'}},
            'empty': {},
        },
    }


async def test_applied_sections_baggage() -> None:
    seen: list[object] = []

    def inspect_baggage() -> str:
        seen.append(logfire.get_baggage().get('logfire.managed.applied_sections'))
        return 'ok'

    capability = ManagedAgent(
        'baggage',
        default=AgentConfig(instructions='managed', settings=AgentConfigSettings(temperature=0.2)),
    )
    await Agent(TestModel(), tools=[inspect_baggage], capabilities=[capability]).run('hello')
    assert seen == ['instructions,settings']


async def test_empty_config_has_no_applied_sections_baggage() -> None:
    seen: list[object] = []

    def inspect_baggage() -> str:
        seen.append(logfire.get_baggage().get('logfire.managed.applied_sections'))
        return 'ok'

    await Agent(TestModel(), tools=[inspect_baggage], capabilities=[ManagedAgent('empty_baggage')]).run('hello')
    assert seen == [None]


async def test_rename_round_trip_preserves_original_context_name() -> None:
    calls = 0
    context_names: list[str | None] = []

    def model(_messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal calls
        calls += 1
        if calls == 1:
            assert [tool.name for tool in info.function_tools] == ['weather_now']
            return ModelResponse(parts=[ToolCallPart('weather_now', {'city': 'Paris'}, tool_call_id='call')])
        return ModelResponse(parts=[TextPart('done')])

    def weather(ctx: RunContext[object], city: str) -> str:
        context_names.append(ctx.tool_name)
        return city

    capability = ManagedAgent(
        'rename',
        default=AgentConfig(tool_definitions={'weather': ToolDefinitionOverride(new_name='weather_now')}),
    )
    await Agent(FunctionModel(model), tools=[weather], capabilities=[capability]).run('hello')
    assert context_names == ['weather']


async def test_rename_collision_warns_and_keeps_other_patches() -> None:
    seen: list[ToolDefinition] = []
    calls = 0

    def first() -> str:
        return 'first'

    def second() -> str:  # pragma: no cover - advertised only
        return 'second'

    def model(_messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal calls
        calls += 1
        seen.extend(info.function_tools)
        if calls == 1:
            return ModelResponse(parts=[ToolCallPart('first', {}, tool_call_id='call')])
        return ModelResponse(parts=[TextPart('done')])

    capability = ManagedAgent(
        'collision',
        default=AgentConfig(
            tool_definitions={'first': ToolDefinitionOverride(new_name='second', description='Managed first.')}
        ),
    )
    with pytest.warns(UserWarning, match='already advertised'):
        await Agent(FunctionModel(model), tools=[first, second], capabilities=[capability]).run('hello')
    assert advertised(seen[:2]) == {'first': 'Managed first.', 'second': None}


async def test_unknown_tool_and_parameter_keys_are_inert() -> None:
    seen: list[ToolDefinition] = []
    capability = ManagedAgent(
        'unknown_tool',
        default=AgentConfig(
            tool_definitions={
                'missing': ToolDefinitionOverride(description='ignored'),
                'get_weather': ToolDefinitionOverride(parameter_descriptions={'missing': 'ignored'}),
            }
        ),
    )
    with warnings.catch_warnings(record=True) as caught:
        await Agent(capture_tools(seen), tools=[get_weather], capabilities=[capability]).run('hello')
    assert caught == []
    assert advertised(seen) == {'get_weather': None}
    assert get_weather('Paris') == 'sunny in Paris'


async def test_schema_without_properties_is_tolerated() -> None:
    seen: list[ToolDefinition] = []

    def raw_tool() -> str:  # pragma: no cover - advertised only
        return 'ok'

    tool = Tool.from_schema(raw_tool, name='raw_tool', description='Original.', json_schema={'type': 'object'})
    capability = ManagedAgent(
        'raw_schema',
        default=AgentConfig(
            tool_definitions={'raw_tool': ToolDefinitionOverride(parameter_descriptions={'missing': 'ignored'})}
        ),
    )
    await Agent(capture_tools(seen), tools=[tool], capabilities=[capability]).run('hello')
    assert seen[0].parameters_json_schema == {'type': 'object'}


async def test_managed_model_runs_model_less_agent_and_run_model_wins() -> None:
    managed = ManagedAgent('managed_model', default=AgentConfig(model='test'))
    assert (await Agent(None, capabilities=[managed]).run('hello')).output.startswith('success')

    def call_site(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart('call-site')])

    result = await Agent(None, capabilities=[managed]).run('hello', model=FunctionModel(call_site))
    assert result.output == 'call-site'


async def test_nameless_model_selector_resolves_once_per_run(monkeypatch: pytest.MonkeyPatch) -> None:
    # A nameless capability's selector is evaluated once per request step, but the managed model is a
    # run-stable value, so it memoizes and resolves the variable exactly once even across steps.
    resolves: list[str] = []
    original = ManagedAgent[Any]._resolve_model_value

    def counting(self: ManagedAgent[Any], variable: Variable[Any]) -> str | None:
        resolves.append(variable.name)
        return original(self, variable)

    monkeypatch.setattr(ManagedAgent, '_resolve_model_value', counting)

    def a_tool() -> str:
        return 'ok'

    capability = ManagedAgent(default=AgentConfig(model='test'))
    # A model-less agent with one tool: `TestModel` calls the tool (step 1) then answers (step 2).
    await Agent(None, name='multi_step', tools=[a_tool], capabilities=[capability]).run('hello')
    assert resolves == ['agent__multi_step']


async def test_known_variable_skips_snapshot_build(capfire: CaptureLogfire, monkeypatch: pytest.MonkeyPatch) -> None:
    config = VariableConfig(name='agent__known', labels={}, rollout=Rollout(labels={}), overrides=[])

    def fail_to_config(self: Variable[object]) -> VariableConfig:  # pragma: no cover - failure sentinel
        raise AssertionError('snapshot should not be built')

    monkeypatch.setattr(Variable, 'to_config', fail_to_config)
    with variables_provider(capfire, VariablesConfig(variables={'agent__known': config})):
        await Agent(TestModel(), capabilities=[ManagedAgent('known')]).run('hello')


async def test_before_model_request_outside_run_is_inert() -> None:
    capability = ManagedAgent('outside_run')
    request = cast(Any, object())
    assert await capability.before_model_request(cast(Any, None), request) is request


async def test_object_value_composition(capfire: CaptureLogfire) -> None:
    config = VariablesConfig(
        variables={
            'agent__composed': VariableConfig(
                name='agent__composed',
                labels={
                    'production': LabeledValue(version=1, serialized_value='{"instructions":"@{prompt__shared}@"}')
                },
                rollout=Rollout(labels={'production': 1}),
                overrides=[],
            ),
            'prompt__shared': VariableConfig(
                name='prompt__shared',
                labels={'production': LabeledValue(version=1, serialized_value='"Shared instructions."')},
                rollout=Rollout(labels={'production': 1}),
                overrides=[],
            ),
        }
    )
    with variables_provider(capfire, config):
        result = await Agent(TestModel(), capabilities=[ManagedAgent('composed')]).run('hello')
    assert instructions_seen(result.all_messages()) == ['Shared instructions.']
