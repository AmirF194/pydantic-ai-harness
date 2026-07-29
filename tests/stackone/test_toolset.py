"""Tests for `StackOneToolset` wire construction, filtering, and tool execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, NamedTuple

import pytest
from pydantic import AnyUrl
from pydantic_ai.exceptions import UserError
from pydantic_ai.mcp import MCPToolsetClient
from pydantic_ai.messages import BinaryContent
from pydantic_ai.tools import RunContext
from pydantic_ai.toolsets import FunctionToolset

from pydantic_ai_harness.stackone import StackOneToolset

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

pytestmark = pytest.mark.anyio


class RecordedCall(NamedTuple):
    client: MCPToolsetClient
    id: str
    headers: dict[str, str] | None


@dataclass
class MCPToolsetRecorder:
    """Records `MCPToolset` constructor calls, standing in a no-op toolset for each."""

    calls: list[RecordedCall] = field(default_factory=list[RecordedCall])

    def __call__(self, client: MCPToolsetClient, *, id: str, headers: dict[str, str] | None) -> FunctionToolset[None]:
        self.calls.append(RecordedCall(client, id, headers))
        return FunctionToolset[None](id=id)


@pytest.fixture
def mcp_recorder(monkeypatch: pytest.MonkeyPatch) -> MCPToolsetRecorder:
    recorder = MCPToolsetRecorder()
    monkeypatch.setattr('pydantic_ai_harness.stackone._toolset.MCPToolset', recorder)
    return recorder


class TestStackOneToolset:
    def test_default_url_headers_and_id(self, mcp_recorder: MCPToolsetRecorder):
        StackOneToolset(account_id='45320', api_key='key')
        call = mcp_recorder.calls[0]
        assert call.client == 'https://api.stackone.com/mcp?tool-mode=search_execute'
        assert call.headers is not None
        assert call.headers['Authorization'].startswith('Basic ')
        assert call.headers['x-account-id'] == '45320'
        assert call.id == 'stackone'

    def test_default_mode_resolution(self, mcp_recorder: MCPToolsetRecorder):
        StackOneToolset(account_id='1', api_key='key', actions=['*_list_*'])
        assert mcp_recorder.calls[0].client == 'https://api.stackone.com/mcp'
        StackOneToolset(account_id='1', api_key='key', tool_mode='individual')
        assert mcp_recorder.calls[1].client == 'https://api.stackone.com/mcp'

    def test_custom_id_reaches_the_connection(self, mcp_recorder: MCPToolsetRecorder):
        StackOneToolset(account_id='45320', api_key='key', id='stackone_eu')
        assert mcp_recorder.calls[0].id == 'stackone_eu'

    def test_custom_base_url(self, mcp_recorder: MCPToolsetRecorder):
        StackOneToolset(
            account_id='45320', api_key='key', base_url='https://api.eu1.stackone.com/', tool_mode='individual'
        )
        assert mcp_recorder.calls[0].client == 'https://api.eu1.stackone.com/mcp'

    @pytest.mark.parametrize('base_url', ['ftp://api.stackone.com', 'localhost:9999', 'https://'])
    def test_rejects_invalid_base_url(self, base_url: str):
        with pytest.raises(UserError, match='`base_url` must be an absolute HTTPS URL'):
            StackOneToolset(account_id='45320', api_key='key', base_url=base_url)

    @pytest.mark.parametrize('suffix', ['?region=eu', '#region-eu'])
    def test_rejects_base_url_query_and_fragment(self, suffix: str):
        with pytest.raises(UserError, match='`base_url` must not contain a query or fragment'):
            StackOneToolset(account_id='45320', api_key='key', base_url=f'https://proxy.example{suffix}')

    @pytest.mark.parametrize('client', ['http://api.stackone.com/mcp', AnyUrl('http://api.stackone.com/mcp')])
    def test_rejects_insecure_http_client(self, client: str | AnyUrl):
        with pytest.raises(UserError, match='`client` must be an absolute HTTPS URL'):
            StackOneToolset(account_id='45320', api_key='key', client=client)

    def test_non_url_client_ignores_base_url(self, mcp_recorder: MCPToolsetRecorder):
        StackOneToolset(account_id='45320', client='server.py', base_url='not-used')
        assert mcp_recorder.calls[0].client == 'server.py'

    def test_search_execute_url(self, mcp_recorder: MCPToolsetRecorder):
        StackOneToolset(account_id='45320', api_key='key', tool_mode='search_execute')
        assert mcp_recorder.calls[0].client == 'https://api.stackone.com/mcp?tool-mode=search_execute'

    def test_search_execute_param_appended_to_custom_urls(self, mcp_recorder: MCPToolsetRecorder):
        StackOneToolset(account_id='1', api_key='key', tool_mode='search_execute', client='https://proxy.example/mcp')
        assert mcp_recorder.calls[0].client == 'https://proxy.example/mcp?tool-mode=search_execute'
        StackOneToolset(
            account_id='1', api_key='key', tool_mode='search_execute', client='https://proxy.example/mcp?region=eu'
        )
        assert mcp_recorder.calls[1].client == 'https://proxy.example/mcp?region=eu&tool-mode=search_execute'

    def test_any_url_gets_headers_and_tool_mode(self, mcp_recorder: MCPToolsetRecorder):
        StackOneToolset(
            account_id='1',
            api_key='key',
            tool_mode='search_execute',
            client=AnyUrl('https://proxy.example/mcp?region=eu'),
        )
        call = mcp_recorder.calls[0]
        assert call.client == 'https://proxy.example/mcp?region=eu&tool-mode=search_execute'
        assert call.headers is not None
        assert call.headers['Authorization'].startswith('Basic ')
        assert call.headers['x-account-id'] == '1'

    def test_url_scheme_is_case_insensitive(self, mcp_recorder: MCPToolsetRecorder):
        StackOneToolset(account_id='1', api_key='key', client='HTTPS://proxy.example/mcp')
        assert mcp_recorder.calls[0].headers is not None

    def test_custom_url_tool_mode_matches_configuration(self, mcp_recorder: MCPToolsetRecorder):
        search_execute_url = 'https://proxy.example/mcp?tool%2Dmode=search%5Fexecute&signature=a%2fb%20c&flag#fragment'
        StackOneToolset(
            account_id='1',
            api_key='key',
            tool_mode='search_execute',
            client=search_execute_url,
        )
        assert mcp_recorder.calls[0].client == search_execute_url
        individual_url = 'https://proxy.example/mcp?signature=a%2fb%20c&tool-mode=individual&flag#fragment'
        StackOneToolset(
            account_id='1',
            api_key='key',
            tool_mode='individual',
            client=individual_url,
        )
        assert mcp_recorder.calls[1].client == individual_url

    def test_custom_url_conflicting_tool_mode_is_replaced(self, mcp_recorder: MCPToolsetRecorder):
        StackOneToolset(
            account_id='1',
            api_key='key',
            tool_mode='search_execute',
            client=(
                'https://proxy.example/mcp?opaque=a%2fb%20c&flag&not-tool-mode=individual'
                '&tool%2Dmode=individual#fragment'
            ),
        )
        assert (
            mcp_recorder.calls[0].client == 'https://proxy.example/mcp?opaque=a%2fb%20c&flag&not-tool-mode=individual'
            '&tool-mode=search_execute#fragment'
        )
        StackOneToolset(
            account_id='1',
            api_key='key',
            tool_mode='individual',
            client='https://proxy.example/mcp?region=eu&tool-mode=search_execute',
        )
        assert mcp_recorder.calls[1].client == 'https://proxy.example/mcp?region=eu'

    def test_no_headers_for_non_url_clients(self, stackone_server: FastMCP, mcp_recorder: MCPToolsetRecorder):
        StackOneToolset(account_id='45320', api_key='key', client=stackone_server)
        call = mcp_recorder.calls[0]
        assert call.client is stackone_server
        assert call.headers is None

    def test_prebuilt_http_client_keeps_its_configuration(self, mcp_recorder: MCPToolsetRecorder):
        from fastmcp import Client

        client = Client('http://proxy.example/mcp')
        StackOneToolset(account_id='45320', api_key='key', client=client)
        call = mcp_recorder.calls[0]
        assert call.client is client
        assert call.headers is None

    def test_script_path_string_is_not_treated_as_url(self, mcp_recorder: MCPToolsetRecorder):
        StackOneToolset(account_id='45320', api_key='key', client='server.py')
        call = mcp_recorder.calls[0]
        assert call.client == 'server.py'
        assert call.headers is None

    def test_missing_api_key_fails_at_construction(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv('STACKONE_API_KEY', raising=False)
        with pytest.raises(UserError, match='STACKONE_API_KEY'):
            StackOneToolset(account_id='45320')

    def test_api_key_from_environment(self, monkeypatch: pytest.MonkeyPatch, mcp_recorder: MCPToolsetRecorder):
        monkeypatch.setenv('STACKONE_API_KEY', 'env-key')
        StackOneToolset(account_id='45320')
        call = mcp_recorder.calls[0]
        assert call.headers is not None
        assert call.headers['Authorization'].startswith('Basic ')

    def test_rejects_actions_in_search_execute(self):
        with pytest.raises(UserError, match='cannot apply in `search_execute` mode'):
            StackOneToolset(account_id='1', api_key='key', tool_mode='search_execute', actions=['*_list_*'])

    def test_rejects_invalid_output_limits(self):
        with pytest.raises(UserError, match='`max_output_bytes` must be a positive integer'):
            StackOneToolset(account_id='1', api_key='key', max_output_bytes=0)
        with pytest.raises(UserError, match='`max_output_bytes` must be a positive integer'):
            StackOneToolset(account_id='1', api_key='key', max_output_bytes=True)
        with pytest.raises(UserError, match='`max_output_lines` must be a positive integer'):
            StackOneToolset(account_id='1', api_key='key', max_output_lines=0)

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

    async def test_output_byte_limit_includes_marker(self, stackone_server: FastMCP, run_context: RunContext[None]):
        toolset = StackOneToolset(
            account_id='1',
            client=stackone_server,
            max_output_bytes=64,
            max_output_lines=10,
        )
        async with toolset:
            tools = await toolset.get_tools(run_context)
            result = await toolset.call_tool(
                'bamboohr_export_employees',
                {'lines': 20},
                run_context,
                tools['bamboohr_export_employees'],
            )
        assert isinstance(result, str)
        assert len(result.encode()) <= 64
        assert result.count('\n') + 1 <= 10
        assert 'truncated' in result

    async def test_output_line_limit_includes_marker(self, stackone_server: FastMCP, run_context: RunContext[None]):
        toolset = StackOneToolset(
            account_id='1',
            client=stackone_server,
            max_output_bytes=1_000,
            max_output_lines=2,
        )
        async with toolset:
            tools = await toolset.get_tools(run_context)
            result = await toolset.call_tool(
                'bamboohr_export_employees',
                {'lines': 20},
                run_context,
                tools['bamboohr_export_employees'],
            )
        assert isinstance(result, str)
        assert result.count('\n') + 1 == 2
        assert result.endswith('lines]')

    async def test_output_line_limit_handles_carriage_returns(
        self, stackone_server: FastMCP, run_context: RunContext[None]
    ):
        toolset = StackOneToolset(
            account_id='1',
            client=stackone_server,
            max_output_bytes=1_000,
            max_output_lines=2,
        )
        async with toolset:
            tools = await toolset.get_tools(run_context)
            result = await toolset.call_tool(
                'bamboohr_export_employees',
                {'lines': 20, 'separator': '\r'},
                run_context,
                tools['bamboohr_export_employees'],
            )
        assert isinstance(result, str)
        assert len(result.splitlines()) == 2
        assert result.endswith('lines]')

    async def test_structured_output_is_preserved_when_within_limits(
        self, stackone_server: FastMCP, run_context: RunContext[None]
    ):
        toolset = StackOneToolset(account_id='1', client=stackone_server, tool_mode='individual')
        async with toolset:
            tools = await toolset.get_tools(run_context)
            result = await toolset.call_tool(
                'bamboohr_create_employee',
                {'name': 'Grace'},
                run_context,
                tools['bamboohr_create_employee'],
            )
        assert not isinstance(result, str)
        assert 'Grace' in str(result)

    async def test_oversized_structured_output_is_omitted(
        self, stackone_server: FastMCP, run_context: RunContext[None]
    ):
        toolset = StackOneToolset(
            account_id='1',
            client=stackone_server,
            tool_mode='individual',
            max_output_bytes=50,
        )
        async with toolset:
            tools = await toolset.get_tools(run_context)
            result = await toolset.call_tool(
                'bamboohr_create_employee',
                {'name': 'Grace Hopper' * 20},
                run_context,
                tools['bamboohr_create_employee'],
            )
        assert isinstance(result, str)
        assert len(result.encode()) <= 50
        assert result.startswith('[StackOne output omitted')

    async def test_oversized_binary_output_is_omitted(
        self, monkeypatch: pytest.MonkeyPatch, run_context: RunContext[None]
    ):
        def binary_result() -> BinaryContent:
            """Return an image."""
            return BinaryContent(data=b'x' * 100, media_type='image/png')

        function_toolset = FunctionToolset[None]([binary_result])

        def build_toolset(
            client: MCPToolsetClient, *, id: str, headers: dict[str, str] | None
        ) -> FunctionToolset[None]:
            return function_toolset

        monkeypatch.setattr('pydantic_ai_harness.stackone._toolset.MCPToolset', build_toolset)
        toolset = StackOneToolset(account_id='1', client='server.py', max_output_bytes=50)
        async with toolset:
            tools = await toolset.get_tools(run_context)
            result = await toolset.call_tool('binary_result', {}, run_context, tools['binary_result'])
        assert isinstance(result, str)
        assert len(result.encode()) <= 50
        assert result.startswith('[StackOne output omitted')

    async def test_toolset_rewrite_preserves_output_limits(
        self, stackone_server: FastMCP, run_context: RunContext[None]
    ):
        toolset = StackOneToolset(
            account_id='1',
            client=stackone_server,
            max_output_bytes=64,
            max_output_lines=10,
        )
        rewritten = toolset.visit_and_replace(lambda inner: inner)
        assert isinstance(rewritten, StackOneToolset)
        async with rewritten:
            tools = await rewritten.get_tools(run_context)
            result = await rewritten.call_tool(
                'bamboohr_export_employees',
                {'lines': 20},
                run_context,
                tools['bamboohr_export_employees'],
            )
        assert isinstance(result, str)
        assert len(result.encode()) <= 64
        assert 'truncated' in result

    async def test_tiny_output_limit_remains_strict(self, stackone_server: FastMCP, run_context: RunContext[None]):
        toolset = StackOneToolset(
            account_id='1',
            client=stackone_server,
            max_output_bytes=5,
            max_output_lines=1,
        )
        async with toolset:
            tools = await toolset.get_tools(run_context)
            result = await toolset.call_tool(
                'bamboohr_export_employees',
                {'lines': 2},
                run_context,
                tools['bamboohr_export_employees'],
            )
        assert isinstance(result, str)
        assert len(result.encode()) <= 5
        assert '\n' not in result

    async def test_prebuilt_fastmcp_client(self, stackone_server: FastMCP, run_context: RunContext[None]):
        from fastmcp import Client

        toolset = StackOneToolset(account_id='1', api_key='key', client=Client(stackone_server))
        async with toolset:
            tools = await toolset.get_tools(run_context)
        assert 'bamboohr_list_employees' in tools
