"""Tests for the GitHub capability.

These exercise the capability through its public `get_toolset()` composition:
unwrap the toolset it registers to inspect the `MCPServerStdio` the agent will
run, and the wired `FilteredToolset.filter_func` for tool limiting. A live
`docker run` is out of scope for CI (no Docker, no token, no network), so the
behavior that can regress here is the server invocation and the filter wiring.
"""

from __future__ import annotations

import sys
from collections.abc import Awaitable, Callable

import pytest
from pydantic_ai import Agent
from pydantic_ai.exceptions import UserError
from pydantic_ai.mcp import MCPServerStdio
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import RunContext, ToolDefinition
from pydantic_ai.toolsets import AbstractToolset, FilteredToolset, PrefixedToolset
from pydantic_ai.usage import RunUsage

from pydantic_ai_harness import GitHub

pytestmark = pytest.mark.anyio

PAT = 'GITHUB_PERSONAL_ACCESS_TOKEN'


def _run_context() -> RunContext[None]:
    return RunContext[None](
        deps=None,
        model=TestModel(),
        usage=RunUsage(),
        prompt=None,
        messages=[],
        run_step=0,
    )


def _public_toolset(capability: GitHub[None]) -> AbstractToolset[None]:
    """The toolset the capability registers with the agent (narrowed off the callable form)."""
    toolset = capability.get_toolset()
    assert toolset is not None and not callable(toolset)
    return toolset


def _server(capability: GitHub[None]) -> MCPServerStdio:
    """The `MCPServerStdio` leaf inside the capability's public toolset."""
    found: list[MCPServerStdio] = []

    def _visit(ts: AbstractToolset[None]) -> AbstractToolset[None]:
        if isinstance(ts, MCPServerStdio):
            found.append(ts)
        return ts

    _public_toolset(capability).visit_and_replace(_visit)
    assert len(found) == 1
    return found[0]


def _filter_func(
    capability: GitHub[None],
) -> Callable[[RunContext[None], ToolDefinition], bool | Awaitable[bool]]:
    """The filter wired into the public `FilteredToolset` (exercises real tool limiting)."""
    toolset = _public_toolset(capability)
    assert isinstance(toolset, FilteredToolset)
    return toolset.filter_func


@pytest.fixture(autouse=True)
def _clear_token_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests control the token explicitly; never let an ambient PAT leak in."""
    monkeypatch.delenv('GITHUB_PERSONAL_ACCESS_TOKEN', raising=False)
    monkeypatch.delenv('GITHUB_TOKEN', raising=False)


class TestTokenResolution:
    def test_explicit_token(self) -> None:
        assert _server(GitHub[None](token='explicit')).env == {PAT: 'explicit'}

    def test_token_from_pat_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv('GITHUB_PERSONAL_ACCESS_TOKEN', 'from-pat')
        assert _server(GitHub[None]()).env == {PAT: 'from-pat'}

    def test_token_from_github_token_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv('GITHUB_TOKEN', 'from-token')
        assert _server(GitHub[None]()).env == {PAT: 'from-token'}

    def test_blank_env_token_is_ignored(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv('GITHUB_PERSONAL_ACCESS_TOKEN', '')
        monkeypatch.setenv('GITHUB_TOKEN', 'real')
        assert _server(GitHub[None]()).env == {PAT: 'real'}

    def test_empty_explicit_token_falls_back_to_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # An explicit '' is treated as unset, consistent with a blank env var.
        monkeypatch.setenv('GITHUB_TOKEN', 'from-env')
        assert _server(GitHub[None](token='')).env == {PAT: 'from-env'}

    def test_empty_explicit_token_with_no_env_raises(self) -> None:
        with pytest.raises(UserError, match='needs a token'):
            GitHub[None](token='').get_toolset()

    def test_missing_token_raises(self) -> None:
        with pytest.raises(UserError, match='needs a token'):
            GitHub[None]().get_toolset()


class TestServerInvocation:
    def test_bare_defaults(self) -> None:
        server = _server(GitHub[None](token='t'))
        assert server.command == 'docker'
        assert server.env == {PAT: 't'}
        # No trailing `stdio` arg: the image's default command already speaks stdio.
        assert server.args == ['run', '-i', '--rm', '-e', PAT, 'ghcr.io/github/github-mcp-server']
        assert server.id == 'github'
        assert server.timeout == 30.0
        assert server.read_timeout == 300.0

    def test_toolsets_read_only_dynamic_host(self) -> None:
        server = _server(
            GitHub[None](
                token='t',
                toolsets=['repos', 'issues'],
                read_only=True,
                dynamic_toolsets=True,
                host='ghe.example.com',
            )
        )
        assert server.env == {
            PAT: 't',
            'GITHUB_TOOLSETS': 'repos,issues',
            'GITHUB_READ_ONLY': '1',
            'GITHUB_DYNAMIC_TOOLSETS': '1',
            'GITHUB_HOST': 'ghe.example.com',
        }
        # Every env var the server reads is forwarded into the container by name (value never in argv).
        assert server.env is not None
        for name in server.env:
            assert f'{name}=' not in ' '.join(server.args)
            i = server.args.index(name)
            assert server.args[i - 1] == '-e'

    def test_env_cannot_override_reserved_keys(self) -> None:
        # A user must not be able to silently flip a safety setting through `env`.
        with pytest.raises(UserError, match='reserved keys'):
            GitHub[None](token='t', read_only=True, env={'GITHUB_READ_ONLY': '0'}).get_toolset()

    def test_extra_env_merged_and_forwarded(self) -> None:
        server = _server(GitHub[None](token='t', env={'HTTP_PROXY': 'http://proxy:3128'}))
        assert server.env == {PAT: 't', 'HTTP_PROXY': 'http://proxy:3128'}
        assert ['-e', 'HTTP_PROXY'] == server.args[
            server.args.index('HTTP_PROXY') - 1 : server.args.index('HTTP_PROXY') + 1
        ]

    def test_docker_overrides(self) -> None:
        server = _server(
            GitHub[None](
                token='t',
                docker_command='podman',
                docker_image='example.com/ghmcp:dev',
                docker_args=['-v', '/cache:/cache'],
                id='gh-1',
                init_timeout=10.0,
                read_timeout=60.0,
            )
        )
        assert server.command == 'podman'
        assert server.args[-1] == 'example.com/ghmcp:dev'
        # Extra docker args land after the forwarded `-e` flags and before the image.
        assert server.args[-3:-1] == ['-v', '/cache:/cache']
        assert server.id == 'gh-1'
        assert server.timeout == 10.0
        assert server.read_timeout == 60.0


class TestToolFiltering:
    def test_bare_returns_unwrapped_server(self) -> None:
        assert isinstance(_public_toolset(GitHub[None](token='t')), MCPServerStdio)

    def test_allowed_tools_wrap_in_filter(self) -> None:
        assert isinstance(_public_toolset(GitHub[None](token='t', allowed_tools=['get_issue'])), FilteredToolset)

    def test_prefix_wraps_outermost(self) -> None:
        toolset = _public_toolset(GitHub[None](token='t', tool_prefix='gh'))
        assert isinstance(toolset, PrefixedToolset)
        assert toolset.prefix == 'gh'

    def test_prefix_and_filter_compose(self) -> None:
        toolset = _public_toolset(GitHub[None](token='t', tool_prefix='gh', allowed_tools=['get_issue']))
        assert isinstance(toolset, PrefixedToolset)
        assert isinstance(toolset.wrapped, FilteredToolset)

    def test_allowed_list(self) -> None:
        f = _filter_func(GitHub[None](token='t', allowed_tools=['get_issue']))
        ctx = _run_context()
        assert f(ctx, ToolDefinition(name='get_issue')) is True
        assert f(ctx, ToolDefinition(name='create_issue')) is False

    def test_denied_list(self) -> None:
        f = _filter_func(GitHub[None](token='t', denied_tools=['delete_repository']))
        ctx = _run_context()
        assert f(ctx, ToolDefinition(name='delete_repository')) is False
        assert f(ctx, ToolDefinition(name='get_issue')) is True

    def test_allow_and_deny_combine(self) -> None:
        f = _filter_func(
            GitHub[None](token='t', allowed_tools=['get_issue', 'create_issue'], denied_tools=['create_issue'])
        )
        ctx = _run_context()
        assert f(ctx, ToolDefinition(name='get_issue')) is True
        assert f(ctx, ToolDefinition(name='create_issue')) is False

    def test_sync_predicate(self) -> None:
        f = _filter_func(GitHub[None](token='t', tool_filter=lambda _ctx, td: td.name.startswith('get_')))
        ctx = _run_context()
        assert f(ctx, ToolDefinition(name='get_issue')) is True
        assert f(ctx, ToolDefinition(name='create_issue')) is False

    async def test_async_predicate(self) -> None:
        async def predicate(_ctx: RunContext[None], td: ToolDefinition) -> bool:
            return td.name == 'get_issue'

        f = _filter_func(GitHub[None](token='t', tool_filter=predicate))
        ctx = _run_context()
        result = f(ctx, ToolDefinition(name='get_issue'))
        assert isinstance(result, Awaitable)
        assert await result is True


class TestAgentIntegration:
    def test_capability_registers_server(self) -> None:
        agent = Agent(TestModel(), capabilities=[GitHub(token='t', read_only=True)])
        leaves: list[type[AbstractToolset[None]]] = []

        def _visit(ts: AbstractToolset[None]) -> AbstractToolset[None]:
            leaves.append(type(ts))
            return ts

        for toolset in agent.toolsets:
            toolset.visit_and_replace(_visit)
        assert MCPServerStdio in leaves


class TestMissingExtra:
    def test_constructs_without_mcp_extra(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # A slim install (no `mcp`) can still import and construct the capability;
        # only building the toolset needs the extra.
        monkeypatch.setitem(sys.modules, 'pydantic_ai.mcp', None)
        assert GitHub(token='t').token == 't'

    def test_missing_mcp_extra_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setitem(sys.modules, 'pydantic_ai.mcp', None)
        with pytest.raises(UserError, match='requires the `mcp` extra'):
            GitHub(token='t').get_toolset()
