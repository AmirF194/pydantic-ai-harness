from __future__ import annotations

import pytest
from pydantic_ai import Agent
from pydantic_ai.exceptions import UserError
from pydantic_ai.models.test import TestModel

from pydantic_ai_harness.logfire import AgentConfig, AgentControl, ManagedPrompt

pytestmark = pytest.mark.anyio


async def test_nameless_prompt_normalizes_agent_name() -> None:
    capability = ManagedPrompt(default='hello')
    agent = Agent(TestModel(), name=' Checkout Assistant #2 ', capabilities=[capability])
    await agent.run('hello')
    assert capability._variable.name == 'prompt__checkout_assistant_2'


async def test_nameless_agent_normalizes_agent_name() -> None:
    capability = AgentControl()
    agent = Agent(TestModel(), name=' Checkout Assistant #2 ', capabilities=[capability])
    await agent.run('hello')
    assert capability._variable.name == 'agent__checkout_assistant_2'


async def test_hyphenated_agent_name() -> None:
    capability = AgentControl(default=AgentConfig(instructions='hello'))
    await Agent(TestModel(), name='pydanty-explorer', capabilities=[capability]).run('hello')
    assert capability._variable.name == 'agent__pydanty_explorer'


async def test_nameless_without_agent_name_raises() -> None:
    agent = Agent(TestModel(), capabilities=[AgentControl()])
    with pytest.raises(UserError, match='without an explicit `name`'):
        await agent.run('hello', infer_name=False)


def test_explicit_name_rules_unchanged() -> None:
    assert AgentControl('Checkout-Agent')._variable.name == 'agent__Checkout_Agent'
    with pytest.raises(ValueError, match='invalid variable name'):
        AgentControl('Checkout Agent')


def test_nameless_get_model_returns_selector() -> None:
    # A nameless capability can't source the model statically (no agent yet), so `get_model` hands
    # back a selector Pydantic AI evaluates once it has a `ModelSelectionContext`.
    assert callable(AgentControl().get_model())


async def test_nameless_sources_model_for_model_less_agent() -> None:
    # The nameless selector derives `agent__solo` from the agent's name and drives a model-less agent.
    capability = AgentControl(default=AgentConfig(model='test'))
    result = await Agent(None, name='solo', capabilities=[capability]).run('hello')
    assert result.output.startswith('success')
    assert capability._variable.name == 'agent__solo'


async def test_nameless_model_less_agent_without_managed_model_raises() -> None:
    agent = Agent(None, name='unpublished', capabilities=[AgentControl()])
    with pytest.raises(UserError, match='no model to run'):
        await agent.run('hello')
