"""End-to-end Absurd durability tests against a real PostgreSQL instance."""

from __future__ import annotations

from absurd_sdk import AsyncAbsurd, AsyncTaskContext, JsonValue
from pydantic_ai import Agent, FunctionToolset, ModelMessage, ModelResponse
from pydantic_ai.messages import TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from pydantic_ai_harness.absurd import AbsurdDurability


async def test_spawned_agent_task_completes_against_postgres(absurd_client: AsyncAbsurd) -> None:
    """Exercise the real worker, task context, checkpoint store, and completion path."""
    agent: Agent[None, str] = Agent(
        FunctionModel(lambda messages, info: ModelResponse(parts=[TextPart(content='ok')]), model_name='fn'),
        name='analyst',
        capabilities=[AbsurdDurability()],
    )

    @absurd_client.register_task(name='analyse')
    async def analyse(params: JsonValue, ctx: AsyncTaskContext) -> JsonValue:
        assert isinstance(params, dict)
        prompt = params['prompt']
        assert isinstance(prompt, str)
        result = await agent.run(prompt)
        return {'output': result.output}

    spawned = await absurd_client.spawn('analyse', {'prompt': 'go'})
    await absurd_client.work_batch(batch_size=1)

    result = await absurd_client.fetch_task_result(spawned['task_id'])
    assert result is not None and result.state == 'completed'
    assert result.result == {'output': 'ok'}


async def test_worker_retry_replays_postgres_checkpoints(absurd_client: AsyncAbsurd) -> None:
    """A fresh worker attempt must reuse model and tool results persisted by PostgreSQL."""
    calls = {'attempts': 0, 'model': 0, 'tool': 0}
    tools = FunctionToolset[object](id='billing')

    @tools.tool_plain
    def charge_card(amount: int) -> str:
        calls['tool'] += 1
        return f'charged {amount}'

    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        calls['model'] += 1
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart(tool_name='charge_card', args={'amount': 42})])
        return ModelResponse(parts=[TextPart(content='done')])

    agent: Agent[object, str] = Agent(
        FunctionModel(model, model_name='fn'),
        name='checkout',
        toolsets=[tools],
        capabilities=[AbsurdDurability()],
    )

    @absurd_client.register_task(name='checkout')
    async def checkout(params: JsonValue, ctx: AsyncTaskContext) -> JsonValue:
        calls['attempts'] += 1
        result = await agent.run('charge it')
        if calls['attempts'] == 1:
            raise RuntimeError('simulated worker crash after checkpoints were committed')
        return {'output': result.output}

    spawned = await absurd_client.spawn(
        'checkout',
        None,
        max_attempts=2,
        retry_strategy={'kind': 'fixed', 'base_seconds': 0},
    )
    await absurd_client.work_batch(batch_size=1)
    await absurd_client.work_batch(batch_size=1)

    result = await absurd_client.fetch_task_result(spawned['task_id'])
    assert result is not None and result.state == 'completed'
    assert result.result == {'output': 'done'}
    assert calls == {'attempts': 2, 'model': 2, 'tool': 1}
