from pydantic_ai import Agent
from pydantic_ai.capabilities import Capability, WebSearch
from pydantic_ai.models.test import TestModel
from pydantic_ai.native_tools import WebSearchTool

from pydantic_ai_harness.code_mode import CodeMode
from pydantic_ai_harness.researcher import DEFAULT_RESEARCHER_INSTRUCTIONS, Researcher, researcher_agent
from pydantic_ai_harness.tool_output_limits import ToolOutputLimits


def test_researcher_constructs_agent() -> None:
    agent = Agent(TestModel(), capabilities=[Researcher()])

    assert isinstance(agent, Agent)


def test_researcher_agent_is_model_less_and_composed() -> None:
    assert isinstance(researcher_agent, Agent)
    assert researcher_agent.model is None
    assert researcher_agent.name == 'researcher'
    assert any(isinstance(capability, CodeMode) for capability in researcher_agent.root_capability.capabilities)


def test_researcher_unknown_export() -> None:
    import pytest

    import pydantic_ai_harness.researcher

    with pytest.raises(AttributeError, match='has no attribute'):
        pydantic_ai_harness.researcher.__getattr__('missing')


def test_researcher_members_are_transparent() -> None:
    researcher = Researcher()

    assert [type(capability) for capability in researcher.capabilities] == [
        CodeMode,
        Capability,
        WebSearch,
        ToolOutputLimits,
    ]
    capability = next(capability for capability in researcher.capabilities if isinstance(capability, Capability))
    assert capability.get_instructions() == [DEFAULT_RESEARCHER_INSTRUCTIONS]
    web_search = next(capability for capability in researcher.capabilities if isinstance(capability, WebSearch))
    assert isinstance(web_search.native, WebSearchTool)
    assert web_search.local is not None


def test_researcher_threads_instructions() -> None:
    researcher = Researcher(instructions='Custom instructions')

    capability = next(capability for capability in researcher.capabilities if isinstance(capability, Capability))
    assert capability.get_instructions() == ['Custom instructions']


def test_researcher_none_disables_instructions() -> None:
    researcher = Researcher(instructions=None)

    assert not any(isinstance(capability, Capability) for capability in researcher.capabilities)


def test_researcher_for_agent_preserves_subclass() -> None:
    researcher = Researcher()
    bound = researcher.for_agent(Agent(TestModel()))

    assert isinstance(bound, Researcher)
