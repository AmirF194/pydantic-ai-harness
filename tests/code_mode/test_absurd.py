"""Composition test: `CodeMode` under `AbsurdDurability`.

`AbsurdDurability` is transparent outside a task, so a passing run alone proves nothing about
durability. The test runs inside a `FakeAsyncTaskContext` and asserts a model-request step was
checkpointed, which is what proves the two capabilities compose while the run is durable.

Both extras can be absent on a CI leg, so the module `importorskip`s `absurd_sdk` and
`pydantic_monty` (CodeMode's sandbox).
"""

from __future__ import annotations

import pytest

pytest.importorskip('absurd_sdk')
pytest.importorskip('pydantic_monty')

from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.toolsets import FunctionToolset

from pydantic_ai_harness import CodeMode
from pydantic_ai_harness.absurd import AbsurdDurability
from tests.absurd.conftest import (  # pyright: ignore[reportMissingTypeStubs]
    FakeAsyncTaskContext,
    absurd_task_context,
)

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


def _text_model() -> FunctionModel:
    def fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart(content='ok')])

    return FunctionModel(fn, model_name='fn')


class TestCodeModeUnderAbsurd:
    async def test_run_is_durable_with_code_mode(self) -> None:
        toolset = FunctionToolset(id='tools')

        @toolset.tool_plain
        def search(query: str) -> str:  # pragma: no cover - the model returns text without calling it
            return f'results for {query}'

        agent = Agent(_text_model(), name='composed', toolsets=[toolset], capabilities=[CodeMode(), AbsurdDurability()])

        ctx = FakeAsyncTaskContext()
        with absurd_task_context(ctx):
            result = await agent.run('hi')

        assert result.output == 'ok'
        assert 'composed__model.request' in ctx.stored
