"""Tests for `StackOneToolset` wire construction, filtering, and tool execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import pytest
from pydantic_ai.exceptions import UserError
from pydantic_ai.tools import RunContext
from pydantic_ai.toolsets import FunctionToolset

from pydantic_ai_harness.stackone import StackOneToolset

if TYPE_CHECKING:
    from fastmcp import FastMCP

pytestmark = pytest.mark.anyio


@dataclass
class MCPToolsetRecorder:
    """Records `MCPToolset` constructor calls, standing in a no-op toolset for each."""

    calls: list[tuple[Any, dict[str, Any]]] = field(default_factory=list[tuple[Any, dict[str, Any]]])

    def __call__(self, client: Any, **kwargs: Any) -> FunctionToolset[None]:
        self.calls.append((client, kwargs))
        return FunctionToolset[None](id=kwargs.get('id'))


@pytest.fixture
def mcp_recorder(monkeypatch: pytest.MonkeyPatch) -> MCPToolsetRecorder:
    recorder = MCPToolsetRecorder()
    monkeypatch.setattr('pydantic_ai_harness.stackone._toolset.MCPToolset', recorder)
    return recorder


class TestStackOneToolset:
    def test_default_url_headers_and_id(self, mcp_recorder: MCPToolsetRecorder):
        StackOneToolset(account_id='45320', api_key='key')
        (client, kwargs) = mcp_recorder.calls[0]
        assert client == 'https://api.stackone.com/mcp?tool-mode=search_execute'
        assert kwargs['headers']['Authorization'].startswith('Basic ')
        assert kwargs['headers']['x-account-id'] == '45320'
        assert kwargs['id'] == 'stackone'

    def test_default_mode_resolution(self, mcp_recorder: MCPToolsetRecorder):
        StackOneToolset(account_id='1', api_key='key', actions=['*_list_*'])
        assert mcp_recorder.calls[0][0] == 'https://api.stackone.com/mcp'
        StackOneToolset(account_id='1', api_key='key', tool_mode='individual')
        assert mcp_recorder.calls[1][0] == 'https://api.stackone.com/mcp'

    def test_custom_id_reaches_the_connection(self, mcp_recorder: MCPToolsetRecorder):
        StackOneToolset(account_id='45320', api_key='key', id='stackone_eu')
        assert mcp_recorder.calls[0][1]['id'] == 'stackone_eu'

    def test_custom_base_url(self, mcp_recorder: MCPToolsetRecorder):
        StackOneToolset(
            account_id='45320', api_key='key', base_url='https://api.eu1.stackone.com/', tool_mode='individual'
        )
        assert mcp_recorder.calls[0][0] == 'https://api.eu1.stackone.com/mcp'

    def test_search_execute_url(self, mcp_recorder: MCPToolsetRecorder):
        StackOneToolset(account_id='45320', api_key='key', tool_mode='search_execute')
        assert mcp_recorder.calls[0][0] == 'https://api.stackone.com/mcp?tool-mode=search_execute'

    def test_search_execute_param_appended_to_custom_urls(self, mcp_recorder: MCPToolsetRecorder):
        StackOneToolset(account_id='1', api_key='key', tool_mode='search_execute', client='https://proxy.example/mcp')
        assert mcp_recorder.calls[0][0] == 'https://proxy.example/mcp?tool-mode=search_execute'
        StackOneToolset(
            account_id='1', api_key='key', tool_mode='search_execute', client='https://proxy.example/mcp?region=eu'
        )
        assert mcp_recorder.calls[1][0] == 'https://proxy.example/mcp?region=eu&tool-mode=search_execute'

    def test_no_headers_for_non_url_clients(self, stackone_server: FastMCP, mcp_recorder: MCPToolsetRecorder):
        StackOneToolset(account_id='45320', api_key='key', client=stackone_server)
        (client, kwargs) = mcp_recorder.calls[0]
        assert client is stackone_server
        assert kwargs['headers'] is None

    def test_missing_api_key_fails_at_construction(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv('STACKONE_API_KEY', raising=False)
        with pytest.raises(UserError, match='STACKONE_API_KEY'):
            StackOneToolset(account_id='45320')

    def test_api_key_from_environment(self, monkeypatch: pytest.MonkeyPatch, mcp_recorder: MCPToolsetRecorder):
        monkeypatch.setenv('STACKONE_API_KEY', 'env-key')
        StackOneToolset(account_id='45320')
        assert mcp_recorder.calls[0][1]['headers']['Authorization'].startswith('Basic ')

    def test_warns_on_actions_in_search_execute(self):
        with pytest.warns(UserWarning, match='`actions` filters are ignored'):
            StackOneToolset(account_id='1', api_key='key', tool_mode='search_execute', actions=['*_list_*'])

    async def test_actions_filter_is_case_insensitive(self, stackone_server: FastMCP, run_context: RunContext[None]):
        toolset = StackOneToolset(account_id='1', api_key='key', client=stackone_server, actions=['*_LIST_*'])
        async with toolset:
            tools = await toolset.get_tools(run_context)
        assert set(tools) == {'bamboohr_list_employees'}

    async def test_call_tool_executes_via_the_connection(self, stackone_server: FastMCP, run_context: RunContext[None]):
        toolset = StackOneToolset(account_id='1', api_key='key', client=stackone_server)
        async with toolset:
            tools = await toolset.get_tools(run_context)
            result = await toolset.call_tool(
                'bamboohr_create_employee', {'name': 'Grace'}, run_context, tools['bamboohr_create_employee']
            )
        assert 'Grace' in str(result)

    async def test_prebuilt_fastmcp_client(self, stackone_server: FastMCP, run_context: RunContext[None]):
        from fastmcp import Client

        toolset = StackOneToolset(account_id='1', api_key='key', client=Client(stackone_server))
        async with toolset:
            tools = await toolset.get_tools(run_context)
        assert 'bamboohr_list_employees' in tools
