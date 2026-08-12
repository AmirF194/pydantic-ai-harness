from pydantic_ai import Agent
from pydantic_ai.capabilities import Capability, WebSearch
from pydantic_ai.models.test import TestModel

from pydantic_ai_harness.code_mode import CodeMode
from pydantic_ai_harness.researcher import DEFAULT_RESEARCHER_INSTRUCTIONS, Researcher
from pydantic_ai_harness.tool_output_limits import ToolOutputLimits


def test_researcher_constructs_agent() -> None:
    agent = Agent(TestModel(), capabilities=[Researcher()])

    assert isinstance(agent, Agent)


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


def test_researcher_threads_instructions() -> None:
    researcher = Researcher(instructions='Custom instructions')

    capability = next(capability for capability in researcher.capabilities if isinstance(capability, Capability))
    assert capability.get_instructions() == ['Custom instructions']


def test_researcher_for_agent_preserves_subclass() -> None:
    researcher = Researcher()
    bound = researcher.for_agent(Agent(TestModel()))

    assert isinstance(bound, Researcher)
