# pyright: reportMissingImports=false, reportUnknownVariableType=false, reportUnknownParameterType=false
# pyright: reportUnknownMemberType=false, reportUntypedFunctionDecorator=false

from __future__ import annotations

import uuid
from collections.abc import Generator
from typing import Any

import pytest

try:
    from dbos import DBOS, DBOSConfig, SetWorkflowID
    from pydantic_ai.durable_exec.dbos import DBOSDurability
except ImportError:  # pragma: lax no cover
    pytest.skip('dbos not installed', allow_module_level=True)

from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, SystemPromptPart, TextPart, UserPromptPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RunUsage

from pydantic_ai_harness.compaction import SummarizingCompaction


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


@pytest.fixture
def dbos(tmp_path: Any) -> Generator[DBOS, None, None]:
    config: DBOSConfig = {
        'name': 'durable_summarizing_compaction',
        'system_database_url': f'sqlite:///{tmp_path / "dbos.sqlite"}',
        'run_admin_server': False,
    }
    instance = DBOS(config=config)
    DBOS.launch()
    try:
        yield instance
    finally:
        DBOS.destroy()


def _history() -> list[ModelMessage]:
    return [
        ModelRequest(parts=[UserPromptPart('first')]),
        ModelResponse(parts=[TextPart('response')]),
        ModelRequest(parts=[UserPromptPart('second')]),
        ModelResponse(parts=[TextPart('response')]),
    ]


@pytest.mark.anyio
async def test_dbos_replays_the_recorded_summary(dbos: DBOS) -> None:
    summary_calls = 0

    async def summarize(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        del messages, info
        nonlocal summary_calls
        summary_calls += 1
        return ModelResponse(parts=[TextPart(f'summary {summary_calls}')])

    compaction: SummarizingCompaction[None] = SummarizingCompaction(
        model=FunctionModel(summarize),
        max_messages=1,
        keep_messages=1,
        preserve_first_user_message=False,
    )
    agent: Agent[None, str] = Agent(
        TestModel(),
        name='durable_summary',
        deps_type=type(None),
        capabilities=[compaction, DBOSDurability[None]()],
    )
    workflow_id = str(uuid.uuid4())

    @DBOS.workflow(name=f'durable_summary_{workflow_id}')
    async def workflow() -> tuple[str, int]:
        result = await agent.run('continue', message_history=_history())
        first = result.all_messages()[0]
        assert isinstance(first, ModelRequest)
        summary_part = first.parts[-1]
        assert isinstance(summary_part, SystemPromptPart)
        return summary_part.content, result.usage.requests

    with SetWorkflowID(workflow_id):
        assert await workflow() == ('Summary of previous conversation:\n\nsummary 1', 2)
    with SetWorkflowID(workflow_id):
        assert await workflow() == ('Summary of previous conversation:\n\nsummary 1', 2)

    assert summary_calls == 1
    steps = await dbos.list_workflow_steps_async(workflow_id)
    assert 'durable_summary__capability__summarizing_compaction.summarize' in [step['function_name'] for step in steps]


@pytest.mark.anyio
async def test_non_durable_summary_behavior_and_usage_are_unchanged() -> None:
    usage = RunUsage()
    compaction: SummarizingCompaction[None] = SummarizingCompaction(
        model=TestModel(custom_output_text='stable summary'),
        max_messages=1,
        keep_messages=1,
        preserve_first_user_message=False,
    )
    agent: Agent[None, str] = Agent(TestModel(), deps_type=type(None), capabilities=[compaction])

    result = await agent.run('continue', message_history=_history(), usage=usage)

    first = result.all_messages()[0]
    assert isinstance(first, ModelRequest)
    summary_part = first.parts[-1]
    assert isinstance(summary_part, SystemPromptPart)
    assert summary_part.content == 'Summary of previous conversation:\n\nstable summary'
    assert result.usage is usage
    assert usage.requests == 2
