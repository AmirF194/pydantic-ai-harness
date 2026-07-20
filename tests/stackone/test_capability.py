"""Tests for the `StackOne` capability through the public `Agent(capabilities=[...])` surface."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from pydantic_ai import Agent
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import ModelMessage, ModelRequest, ToolCallPart
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import RunContext

from pydantic_ai_harness.stackone import StackOne

if TYPE_CHECKING:
    from fastmcp import FastMCP

pytestmark = pytest.mark.anyio


def tool_call_names(messages: list[ModelMessage]) -> set[str]:
    return {part.tool_name for message in messages for part in message.parts if isinstance(part, ToolCallPart)}


def request_instructions(messages: list[ModelMessage]) -> str:
    first = messages[0]
    assert isinstance(first, ModelRequest)
    return first.instructions or ''


class TestStackOne:
    def test_serialization_name(self):
        assert StackOne.get_serialization_name() == 'StackOne'

    def test_missing_api_key_fails_at_agent_construction(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv('STACKONE_API_KEY', raising=False)
        with pytest.raises(UserError, match='STACKONE_API_KEY'):
            Agent(TestModel(), capabilities=[StackOne(account_id='45320')])

    def test_api_key_from_environment(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv('STACKONE_API_KEY', 'env-key')
        assert StackOne(account_id='45320').get_toolset() is not None

    def test_construction_does_not_warn(self, recwarn: pytest.WarningsRecorder):
        StackOne(account_id='45320', api_key='key', actions=['*_list_*'])
        StackOne(account_id='45320', api_key='key', id='stackone', defer_loading=True)
        assert not recwarn.list

    def test_rejects_actions_in_explicit_search_execute(self):
        with pytest.raises(UserError, match='cannot apply in `search_execute` mode'):
            StackOne(account_id='45320', api_key='key', tool_mode='search_execute', actions=['*_list_*'])

    async def test_agent_run_calls_stackone_tools(self, stackone_server: FastMCP):
        capability = StackOne(account_id='45320', api_key='key', client=stackone_server, tool_mode='individual')
        agent = Agent(TestModel(), capabilities=[capability])
        result = await agent.run('list employees')
        assert 'bamboohr_list_employees' in tool_call_names(result.all_messages())
        assert 'Ada' in result.output

    async def test_actions_glob_filter(self, stackone_server: FastMCP):
        capability = StackOne(account_id='45320', api_key='key', client=stackone_server, actions=['*_list_*'])
        agent = Agent(TestModel(), capabilities=[capability])
        result = await agent.run('list employees')
        assert tool_call_names(result.all_messages()) == {'bamboohr_list_employees'}

    async def test_instructions_follow_the_resolved_mode(self, stackone_server: FastMCP):
        individual = StackOne(account_id='45320', api_key='key', client=stackone_server, actions=['*_list_*'])
        agent = Agent(TestModel(), capabilities=[individual])
        result = await agent.run('list employees')
        assert '{connector}_{action}_{entity}' in request_instructions(result.all_messages())
        default = StackOne(account_id='45320', api_key='key', client=stackone_server)
        agent = Agent(TestModel(), capabilities=[default])
        result = await agent.run('list employees')
        assert 'must never be guessed' in request_instructions(result.all_messages())

    async def test_instructions_disabled(self, stackone_server: FastMCP):
        capability = StackOne(account_id='45320', api_key='key', client=stackone_server, include_instructions=False)
        agent = Agent(TestModel(), capabilities=[capability])
        result = await agent.run('list employees')
        assert '{connector}_{action}_{entity}' not in request_instructions(result.all_messages())

    async def test_search_execute_mode(self, search_execute_server: FastMCP):
        capability = StackOne(
            account_id='45320', api_key='key', tool_mode='search_execute', client=search_execute_server
        )
        agent = Agent(TestModel(), capabilities=[capability])
        result = await agent.run('list employees')
        assert tool_call_names(result.all_messages()) == {'bamboohr_search_actions', 'bamboohr_execute_action'}
        assert 'must never be guessed' in request_instructions(result.all_messages())

    async def test_metadata_merged_onto_tools(self, stackone_server: FastMCP, run_context: RunContext[None]):
        capability = StackOne(
            account_id='45320', api_key='key', client=stackone_server, metadata={'code_mode': True, 'team': 'hr'}
        )
        toolset = capability.get_toolset()
        async with toolset:
            tools = await toolset.get_tools(run_context)
        assert tools
        for tool in tools.values():
            assert tool.tool_def.metadata is not None
            assert tool.tool_def.metadata.get('code_mode') is True
            assert tool.tool_def.metadata.get('team') == 'hr'
