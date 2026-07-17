from __future__ import annotations

import pytest
from pydantic_ai import Agent
from pydantic_ai.exceptions import UserError
from pydantic_ai.models.test import TestModel

from pydantic_ai_harness.logfire import AgentConfig, ManagedAgent, ManagedPrompt

pytestmark = pytest.mark.anyio


async def test_nameless_prompt_normalizes_agent_name() -> None:
    capability = ManagedPrompt(default='hello')
    agent = Agent(TestModel(), name=' Checkout Assistant #2 ', capabilities=[capability])
    await agent.run('hello')
    assert capability._variable.name == 'prompt__checkout_assistant_2'


async def test_nameless_agent_normalizes_agent_name() -> None:
    capability = ManagedAgent()
    agent = Agent(TestModel(), name=' Checkout Assistant #2 ', capabilities=[capability])
    await agent.run('hello')
    assert capability._variable.name == 'agent__checkout_assistant_2'


async def test_hyphenated_agent_name() -> None:
    capability = ManagedAgent(default=AgentConfig(instructions='hello'))
    await Agent(TestModel(), name='pydanty-explorer', capabilities=[capability]).run('hello')
    assert capability._variable.name == 'agent__pydanty_explorer'


async def test_nameless_without_agent_name_raises() -> None:
    agent = Agent(TestModel(), capabilities=[ManagedAgent()])
    with pytest.raises(UserError, match='without an explicit `name`'):
        await agent.run('hello', infer_name=False)


def test_explicit_name_rules_unchanged() -> None:
    assert ManagedAgent('Checkout-Agent')._variable.name == 'agent__Checkout_Agent'
    with pytest.raises(ValueError, match='invalid variable name'):
        ManagedAgent('Checkout Agent')


def test_nameless_get_model_returns_none() -> None:
    assert ManagedAgent().get_model() is None
