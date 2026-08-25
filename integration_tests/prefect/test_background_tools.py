"""Prefect integration tests for `BackgroundTools`."""

# pyright: reportAttributeAccessIssue=false, reportMissingImports=false
# pyright: reportUnknownVariableType=false, reportUntypedFunctionDecorator=false

from __future__ import annotations

import asyncio
import gc
import uuid
from collections.abc import Generator

import pytest

try:
    from prefect import flow
    from prefect.testing.utilities import prefect_test_harness
    from pydantic_ai.durable_exec.prefect import PrefectDurability
except ImportError:  # pragma: lax no cover
    pytest.skip('prefect not installed', allow_module_level=True)

from pydantic_ai import Agent
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel

from pydantic_ai_harness.background_tools import BackgroundTools

pytestmark = [
    pytest.mark.anyio,
    # Prefect's temporary API server leaves an HTTP socket for the GC on Python 3.14.
    pytest.mark.filterwarnings('ignore:unclosed <socket.socket:ResourceWarning'),
]


@pytest.fixture(scope='module')
def anyio_backend() -> str:
    """Prefect runs on asyncio."""
    return 'asyncio'


@pytest.fixture(scope='module', autouse=True)
def prefect_harness() -> Generator[None, None, None]:
    """Run the test against an isolated Prefect API and database."""
    with prefect_test_harness():
        yield
    gc.collect()


def _has_ack(messages: list[ModelMessage]) -> bool:
    return any(
        isinstance(part, ToolReturnPart) and 'running in background' in str(part.content)
        for message in messages
        if isinstance(message, ModelRequest)
        for part in message.parts
    )


def _completion_count(messages: list[ModelMessage]) -> int:
    return sum(
        1
        for message in messages
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, UserPromptPart)
        and isinstance(part.content, str)
        and "Background tool 'research'" in part.content
    )


async def test_background_result_is_reconstructed_on_prefect_flow_retry() -> None:
    """A flow retry rebuilds the task while Prefect restores its completed handler."""
    tool_calls = 0
    flow_attempts = 0
    run_id = str(uuid.uuid4())

    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if _completion_count(messages):
            return ModelResponse(parts=[TextPart(content='done')])
        if _has_ack(messages):
            return ModelResponse(parts=[TextPart(content='waiting')])
        return ModelResponse(parts=[ToolCallPart(tool_name='research', args={}, tool_call_id='research-call')])

    agent = Agent(
        FunctionModel(model),
        name='background_tools_prefect_agent',
        capabilities=[BackgroundTools(), PrefectDurability()],
    )

    @agent.tool_plain(metadata={'background': True})
    async def research() -> str:  # pyright: ignore[reportUnusedFunction]
        nonlocal tool_calls
        tool_calls += 1
        await asyncio.sleep(0.05)
        return 'durable result'

    @flow(name='background-tools-prefect-retry', retries=1, retry_delay_seconds=0)
    async def run_flow() -> str:
        nonlocal flow_attempts
        flow_attempts += 1
        result = await agent.run('go', run_id=run_id)
        assert _completion_count(result.all_messages()) == 1
        if flow_attempts == 1:
            raise RuntimeError('retry after the durable tasks completed')
        return str(result.output)

    assert await run_flow() == 'done'
    assert flow_attempts == 2
    assert tool_calls == 1
