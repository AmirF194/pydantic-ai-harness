from __future__ import annotations

import asyncio
from collections.abc import Sequence

import pytest
from inline_snapshot import snapshot
from pydantic_ai import AdvisorTool, Agent
from pydantic_ai.capabilities import AbstractCapability, PrefixTools
from pydantic_ai.exceptions import UsageLimitExceeded, UserError
from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.models.function import AgentInfo, FunctionDef, FunctionModel
from pydantic_ai.profiles import ModelProfile
from pydantic_ai.settings import ModelSettings
from pydantic_ai.usage import UsageLimits

from pydantic_ai_harness.advisor import Advisor, AdvisorMode

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


class _ProviderFunctionModel(FunctionModel):
    def __init__(
        self,
        provider: str,
        function: FunctionDef,
        *,
        supports_advisor: bool = False,
    ) -> None:
        profile = ModelProfile(supported_native_tools=frozenset({AdvisorTool}) if supports_advisor else frozenset())
        super().__init__(function, profile=profile)
        self._system_name = provider

    @property
    def system(self) -> str:
        return self._system_name


class _ActiveDurability(AbstractCapability[None]):
    engine_name = 'Test'

    @property
    def in_durable_context(self) -> bool:
        return True


class _InactiveDurability(AbstractCapability[None]):
    engine_name = 'Test'

    @property
    def in_durable_context(self) -> bool:
        return False


def _has_tool_return(messages: Sequence[ModelMessage]) -> bool:
    return any(isinstance(part, ToolReturnPart) for message in messages for part in message.parts)


class TestAdvisor:
    async def test_uses_native_advisor_for_matching_provider(self) -> None:
        seen: list[ModelRequestParameters] = []

        def executor(_messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            seen.append(info.model_request_parameters)
            return ModelResponse(parts=[TextPart('done')])

        model = _ProviderFunctionModel('anthropic', executor, supports_advisor=True)
        agent = Agent(
            model,
            capabilities=[
                Advisor(
                    'anthropic:claude-opus-4-8',
                    mode='native',
                    max_tokens=2048,
                    caching='5m',
                )
            ],
        )

        await agent.run('Review this plan.')

        assert seen[0].function_tools == []
        assert seen[0].native_tools == [AdvisorTool(model='claude-opus-4-8', max_tokens=2048, caching='5m')]

    async def test_model_instance_uses_local_advisor_and_preserves_identity(self) -> None:
        advisor_prompts: list[str] = []
        advisor_settings: list[ModelSettings | None] = []

        def advisor_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            advisor_settings.append(info.model_settings)
            prompt = messages[-1].parts[-1]
            assert isinstance(prompt, UserPromptPart)
            assert isinstance(prompt.content, str)
            advisor_prompts.append(prompt.content)
            return ModelResponse(parts=[TextPart('Use a staged rollout.')])

        def executor(messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
            if not _has_tool_return(messages):
                return ModelResponse(
                    parts=[ToolCallPart('advisor', {'prompt': 'How should I deploy this?'}, tool_call_id='advisor-1')]
                )
            return ModelResponse(parts=[TextPart('I will use a staged rollout.')])

        advisor = _ProviderFunctionModel('anthropic', advisor_model)
        executor_model = _ProviderFunctionModel('anthropic', executor, supports_advisor=True)
        agent = Agent(executor_model, capabilities=[Advisor(advisor, max_tokens=2048)])

        await agent.run('Plan the deployment.')
        await agent.run('Plan another deployment.')

        assert advisor_prompts == ['How should I deploy this?', 'How should I deploy this?']
        assert advisor_settings == [ModelSettings(max_tokens=2048), ModelSettings(max_tokens=2048)]

    async def test_unsupported_native_profile_exposes_local_tool(self) -> None:
        seen: list[ModelRequestParameters] = []

        def executor(_messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            seen.append(info.model_request_parameters)
            return ModelResponse(parts=[TextPart('done')])

        model = _ProviderFunctionModel('anthropic', executor)
        agent = Agent(model, capabilities=[Advisor('anthropic:claude-opus-4-8')])

        await agent.run('Review this plan.')

        assert seen[0].native_tools == []
        assert [tool.name for tool in seen[0].function_tools] == ['advisor']
        assert seen[0].function_tools[0].sequential is True

    @pytest.mark.parametrize('prefix', [None, 'consult'])
    async def test_max_uses_is_deterministic_for_parallel_calls(self, prefix: str | None) -> None:
        advisor_calls = 0
        tool_returns: list[str] = []
        tool_name = f'{prefix}_advisor' if prefix is not None else 'advisor'

        def advisor_model(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
            nonlocal advisor_calls
            advisor_calls += 1
            return ModelResponse(parts=[TextPart('first advice')])

        def executor(messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
            nonlocal tool_returns
            if not _has_tool_return(messages):
                return ModelResponse(
                    parts=[
                        ToolCallPart(tool_name, {'prompt': 'first'}, tool_call_id='advisor-1'),
                        ToolCallPart(tool_name, {'prompt': 'second'}, tool_call_id='advisor-2'),
                    ]
                )
            tool_returns = [
                part.content
                for message in messages
                for part in message.parts
                if isinstance(part, ToolReturnPart) and isinstance(part.content, str)
            ]
            return ModelResponse(parts=[TextPart('done')])

        advisor = _ProviderFunctionModel('anthropic', advisor_model)
        capability: AbstractCapability[object] = Advisor(advisor, max_uses=1)
        if prefix is not None:
            capability = PrefixTools(capability, prefix=prefix)
        agent = Agent(FunctionModel(executor), capabilities=[capability])

        await agent.run('Ask twice.')

        assert advisor_calls == 1
        assert tool_returns == snapshot(
            [
                'first advice',
                'Advisor consultation limit reached for this model request. Continue without further advice.',
            ]
        )

    async def test_invalid_call_does_not_consume_max_uses(self) -> None:
        advisor_calls = 0
        executor_calls = 0

        def advisor_model(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
            nonlocal advisor_calls
            advisor_calls += 1
            return ModelResponse(parts=[TextPart('valid advice')])

        def executor(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
            nonlocal executor_calls
            executor_calls += 1
            if executor_calls == 1:
                return ModelResponse(
                    parts=[
                        ToolCallPart('advisor', {}, tool_call_id='invalid'),
                        ToolCallPart('advisor', {'prompt': 'valid'}, tool_call_id='valid'),
                    ]
                )
            return ModelResponse(parts=[TextPart('done')])

        advisor = _ProviderFunctionModel('anthropic', advisor_model)
        agent = Agent(FunctionModel(executor), capabilities=[Advisor(advisor, max_uses=1)])

        await agent.run('Ask twice.')

        assert advisor_calls == 1

    async def test_max_uses_resets_for_each_executor_request(self) -> None:
        advisor_calls = 0
        executor_calls = 0

        def advisor_model(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
            nonlocal advisor_calls
            advisor_calls += 1
            return ModelResponse(parts=[TextPart(f'advice {advisor_calls}')])

        def executor(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
            nonlocal executor_calls
            executor_calls += 1
            if executor_calls <= 2:
                return ModelResponse(
                    parts=[
                        ToolCallPart(
                            'advisor',
                            {'prompt': f'question {executor_calls}'},
                            tool_call_id=f'advisor-{executor_calls}',
                        )
                    ]
                )
            return ModelResponse(parts=[TextPart('done')])

        advisor = _ProviderFunctionModel('anthropic', advisor_model)
        agent = Agent(FunctionModel(executor), capabilities=[Advisor(advisor, max_uses=1)])

        await agent.run('Ask on consecutive requests.')

        assert advisor_calls == 2

    async def test_concurrent_runs_have_independent_max_uses(self) -> None:
        advisor_calls = 0

        async def advisor_model(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
            nonlocal advisor_calls
            advisor_calls += 1
            await asyncio.sleep(0)
            return ModelResponse(parts=[TextPart('advice')])

        def executor(messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
            if not _has_tool_return(messages):
                return ModelResponse(parts=[ToolCallPart('advisor', {'prompt': 'question'})])
            return ModelResponse(parts=[TextPart('done')])

        advisor = _ProviderFunctionModel('anthropic', advisor_model)
        agent = Agent(FunctionModel(executor), capabilities=[Advisor(advisor, max_uses=1)])

        await asyncio.gather(agent.run('first'), agent.run('second'))

        assert advisor_calls == 2

    async def test_openrouter_max_uses_selects_local_fallback(self) -> None:
        seen: list[ModelRequestParameters] = []

        def executor(_messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            seen.append(info.model_request_parameters)
            return ModelResponse(parts=[TextPart('done')])

        model = _ProviderFunctionModel('openrouter', executor, supports_advisor=True)
        agent = Agent(
            model,
            capabilities=[Advisor('openrouter:anthropic/claude-opus-4.8', max_uses=1)],
        )

        await agent.run('Review this plan.')

        assert seen[0].native_tools == []
        assert [tool.name for tool in seen[0].function_tools] == ['advisor']

    async def test_openrouter_without_max_uses_uses_native_advisor(self) -> None:
        seen: list[ModelRequestParameters] = []

        def executor(_messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            seen.append(info.model_request_parameters)
            return ModelResponse(parts=[TextPart('done')])

        model = _ProviderFunctionModel('openrouter', executor, supports_advisor=True)
        agent = Agent(
            model,
            capabilities=[Advisor('openrouter:anthropic/claude-opus-4.8')],
        )

        await agent.run('Review this plan.')

        assert seen[0].function_tools == []
        assert seen[0].native_tools == [AdvisorTool(model='anthropic/claude-opus-4.8')]

    async def test_local_advisor_shares_parent_usage_and_limits(self) -> None:
        def executor(messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
            if not _has_tool_return(messages):
                return ModelResponse(parts=[ToolCallPart('advisor', {'prompt': 'Review this.'})])
            return ModelResponse(parts=[TextPart('done')])

        advisor = FunctionModel(lambda _messages, _info: ModelResponse(parts=[TextPart('advice')]))
        agent = Agent(FunctionModel(executor), capabilities=[Advisor(advisor)])

        result = await agent.run('Ask for advice.')

        assert result.usage.requests == 3

        with pytest.raises(UsageLimitExceeded, match='request_limit'):
            await agent.run('Ask for advice.', usage_limits=UsageLimits(request_limit=1))

    @pytest.mark.parametrize('mode', ['auto', 'local'])
    async def test_local_capable_mode_runs_outside_durable_context(self, mode: AdvisorMode) -> None:
        advisor_calls = 0

        def advisor_model(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
            nonlocal advisor_calls
            advisor_calls += 1
            return ModelResponse(parts=[TextPart('Use a staged rollout.')])

        def executor(messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
            if not _has_tool_return(messages):
                return ModelResponse(
                    parts=[ToolCallPart('advisor', {'prompt': 'Review this.'}, tool_call_id='advisor-1')]
                )
            return ModelResponse(parts=[TextPart('done')])

        model = FunctionModel(advisor_model)
        agent = Agent(
            FunctionModel(executor),
            name='executor',
            deps_type=type(None),
            capabilities=[Advisor(model, mode=mode), _InactiveDurability()],
        )

        await agent.run('Review this.')

        assert advisor_calls == 1

    @pytest.mark.parametrize('mode', ['auto', 'local'])
    async def test_local_capable_mode_rejects_active_durable_context(self, mode: AdvisorMode) -> None:
        model_called = False

        def executor(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:  # pragma: no cover
            nonlocal model_called
            model_called = True
            return ModelResponse(parts=[TextPart('done')])

        model = FunctionModel(executor)
        advisor: Advisor[None] = Advisor(model, mode=mode)
        active: AbstractCapability[None] = _ActiveDurability()
        agent = Agent[None, str](
            model,
            name='executor',
            deps_type=type(None),
            capabilities=[advisor, active],
        )

        with pytest.raises(UserError, match="modes 'auto' and 'local'.*active durable execution"):
            await agent.run('Review this.')
        assert model_called is False

    async def test_native_mode_runs_inside_active_durable_context(self) -> None:
        seen: list[ModelRequestParameters] = []

        def executor(_messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            seen.append(info.model_request_parameters)
            return ModelResponse(parts=[TextPart('done')])

        model = _ProviderFunctionModel('anthropic', executor, supports_advisor=True)
        advisor: Advisor[None] = Advisor('anthropic:claude-opus-4-8', mode='native')
        active: AbstractCapability[None] = _ActiveDurability()
        agent = Agent[None, str](
            model,
            deps_type=type(None),
            capabilities=[advisor, active],
        )

        await agent.run('Review this.')

        assert seen[0].function_tools == []
        assert seen[0].native_tools == [AdvisorTool(model='claude-opus-4-8')]

    async def test_auto_mode_requires_explicit_native_opt_in_during_durable_execution(self) -> None:
        model_called = False

        def executor(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:  # pragma: no cover
            nonlocal model_called
            model_called = True
            return ModelResponse(parts=[TextPart('done')])

        model = _ProviderFunctionModel('anthropic', executor, supports_advisor=True)
        advisor: Advisor[None] = Advisor('anthropic:claude-opus-4-8')
        active: AbstractCapability[None] = _ActiveDurability()
        agent = Agent[None, str](
            model,
            deps_type=type(None),
            capabilities=[advisor, active],
        )

        with pytest.raises(UserError, match="mode='native'"):
            await agent.run('Review this.')
        assert model_called is False

    async def test_rejects_temporal_durability(self, monkeypatch: pytest.MonkeyPatch) -> None:
        pytest.importorskip('temporalio')
        from pydantic_ai.durable_exec.temporal import TemporalDurability

        await self._assert_rejects_durability(TemporalDurability(), monkeypatch)

    async def test_rejects_dbos_durability(self, monkeypatch: pytest.MonkeyPatch) -> None:
        pytest.importorskip('dbos')
        from pydantic_ai.durable_exec.dbos import DBOSDurability

        await self._assert_rejects_durability(DBOSDurability(), monkeypatch)

    async def test_rejects_prefect_durability(  # pragma: lax no cover
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pytest.importorskip('prefect')
        from pydantic_ai.durable_exec.prefect import PrefectDurability

        await self._assert_rejects_durability(PrefectDurability(), monkeypatch)

    @staticmethod
    async def _assert_rejects_durability(
        durability: AbstractCapability[None],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        model = FunctionModel(lambda _messages, _info: ModelResponse(parts=[TextPart('done')]))
        agent = Agent(
            model,
            name=f'advisor-{type(durability).__name__}',
            deps_type=type(None),
            capabilities=[Advisor(model, mode='local'), durability],
        )
        monkeypatch.setattr(type(durability), 'in_durable_context', property(lambda _self: True))

        with pytest.raises(UserError, match='active durable execution'):
            await agent.run('Review this.')

    def test_explicit_execution_modes(self) -> None:
        local = Advisor('anthropic:claude-opus-4-8', mode='local')
        assert local.get_native_tools() == []
        assert local.get_toolset() is not None

        other_provider = Advisor('openai:gpt-5.4')
        assert other_provider.get_toolset() is not None

        native = Advisor('anthropic:claude-opus-4-8', mode='native', max_uses=2)
        assert len(native.get_native_tools()) == 1
        assert native.get_toolset() is None

    @pytest.mark.parametrize(
        ('executor_provider', 'advisor_model', 'required_provider'),
        [
            ('openrouter', 'anthropic:claude-opus-4-8', 'anthropic'),
            ('anthropic', 'openrouter:anthropic/claude-opus-4.8', 'openrouter'),
        ],
    )
    async def test_native_mode_rejects_cross_provider_model_id(
        self,
        executor_provider: str,
        advisor_model: str,
        required_provider: str,
    ) -> None:
        executor = _ProviderFunctionModel(
            executor_provider,
            lambda _messages, _info: ModelResponse(parts=[TextPart('done')]),
            supports_advisor=True,
        )
        agent = Agent(executor, capabilities=[Advisor(advisor_model, mode='native')])

        with pytest.raises(UserError, match=f'requires a {required_provider} executor'):
            await agent.run('Review this.')

    async def test_rejects_duplicate_capability_or_tool_name(self) -> None:
        model = FunctionModel(lambda _messages, _info: ModelResponse(parts=[TextPart('done')]))

        with pytest.raises(UserError, match="Capability id 'advisor' is used by multiple capabilities"):
            Agent(model, capabilities=[Advisor(model), Advisor(model)])

        def advisor(prompt: str) -> str:  # pragma: no cover
            return prompt

        agent = Agent(model, capabilities=[Advisor(model)], tools=[advisor])
        with pytest.raises(UserError, match='tool whose name conflicts'):
            await agent.run('Review this.')

    def test_rejects_invalid_limits(self) -> None:
        model = FunctionModel(lambda _messages, _info: ModelResponse(parts=[TextPart('done')]))
        with pytest.raises(ValueError, match='Advisor.mode'):
            Advisor(model, mode='invalid')  # pyright: ignore[reportArgumentType]
        with pytest.raises(ValueError, match='Advisor.max_uses must be at least 1'):
            Advisor(model, max_uses=0)
        with pytest.raises(ValueError, match='Advisor.max_tokens must be at least 1024'):
            Advisor(model, max_tokens=1023)
        with pytest.raises(ValueError, match="mode='native'.*model name"):
            Advisor(model, mode='native')
        with pytest.raises(ValueError, match="mode='native'.*model name"):
            Advisor('test', mode='native')
        with pytest.raises(ValueError, match='not supported by OpenRouter'):
            Advisor('openrouter:anthropic/claude-opus-4.8', mode='native', max_uses=1)

    def test_is_not_spec_serializable(self) -> None:
        assert Advisor.get_serialization_name() is None
