"""StackOne wire contract and toolset.

StackOne (https://docs.stackone.com) exposes actions on a user's linked SaaS
accounts (HRIS, ATS, CRM, and more) through an MCP endpoint. This module owns
the wire contract: endpoint path, header names, and the tool naming
convention. A StackOne API change should be a diff to this file only.

Wire contract (https://docs.stackone.com/mcp/quickstart):

- `POST {base_url}/mcp` -- MCP over streamable HTTP; lists and executes tools.
  `?tool-mode=search_execute` switches from one-tool-per-action to two
  search/execute meta-tools.
- Auth is `Authorization: Basic base64('{api_key}:')` plus an `x-account-id`
  header selecting the linked account.
- Tool names follow `{connector}_{action}_{entity}`, e.g.
  `bamboohr_list_employees`.
"""

from __future__ import annotations

import base64
import os
import warnings
from collections.abc import Callable, Sequence
from fnmatch import fnmatch
from typing import Literal

from pydantic_ai.exceptions import UserError
from pydantic_ai.tools import AgentDepsT, RunContext, ToolDefinition
from pydantic_ai.toolsets import AbstractToolset, WrapperToolset

try:
    from pydantic_ai.mcp import MCPToolset, MCPToolsetClient
except ImportError as _import_error:  # pragma: no cover
    raise ImportError(
        'MCP support is required for the StackOne capability. '
        'Install it with: pip install "pydantic-ai-harness[stackone]"'
    ) from _import_error

__all__ = (
    'STACKONE_API_KEY_ENV',
    'STACKONE_BASE_URL',
    'StackOneToolset',
    'ToolMode',
)

STACKONE_API_KEY_ENV = 'STACKONE_API_KEY'
"""Environment variable consulted when `api_key` is not passed explicitly."""

STACKONE_BASE_URL = 'https://api.stackone.com'
"""Default StackOne API host."""

ToolMode = Literal['individual', 'search_execute']
"""How StackOne registers tools: one tool per action, or two search/execute meta-tools."""

_MCP_PATH = '/mcp'
_SEARCH_EXECUTE_QUERY = 'tool-mode=search_execute'


def resolve_tool_mode(tool_mode: ToolMode | None, actions: Sequence[str]) -> ToolMode:
    """Resolve the default tool mode: `search_execute`, or `individual` when `actions` are given.

    `search_execute` keeps the prompt footprint constant regardless of catalog
    size (provider catalogs can exceed model context windows in `individual`
    mode), while `actions` globs only apply to individually registered tools.
    """
    if tool_mode is not None:
        return tool_mode
    return 'individual' if actions else 'search_execute'


def resolve_api_key(api_key: str | None) -> str:
    """Return the given API key, or the one from `STACKONE_API_KEY`.

    Raises:
        UserError: If neither is set.
    """
    resolved = api_key or os.environ.get(STACKONE_API_KEY_ENV)
    if not resolved:
        raise UserError(
            f'A StackOne API key is required: pass `api_key` or set the `{STACKONE_API_KEY_ENV}` '
            'environment variable. Keys are managed at https://app.stackone.com.'
        )
    return resolved


def _basic_auth(api_key: str) -> str:
    token = base64.b64encode(f'{api_key}:'.encode()).decode()
    return f'Basic {token}'


def _with_tool_mode(url: str, tool_mode: ToolMode) -> str:
    if tool_mode != 'search_execute' or 'tool-mode=' in url:
        return url
    separator = '&' if '?' in url else '?'
    return f'{url}{separator}{_SEARCH_EXECUTE_QUERY}'


def warn_ignored_actions(tool_mode: ToolMode | None, actions: Sequence[str], *, stacklevel: int) -> None:
    """Warn that `actions` globs cannot apply in explicitly requested `search_execute` mode."""
    if tool_mode == 'search_execute' and actions:
        warnings.warn(
            '`actions` filters are ignored in `search_execute` mode: individual action names '
            'never reach the agent. Use `individual` mode to filter actions.',
            stacklevel=stacklevel + 1,
        )


def _action_filter(actions: Sequence[str]) -> Callable[[RunContext[AgentDepsT], ToolDefinition], bool]:
    """A `FilteredToolset` predicate matching tool names against the given globs.

    The lowered globs are computed once here rather than per tool per step,
    which is how often `FilteredToolset` evaluates the predicate.
    """
    lowered = tuple(glob.lower() for glob in actions)

    def action_filter(ctx: RunContext[AgentDepsT], tool_def: ToolDefinition) -> bool:
        return any(fnmatch(tool_def.name.lower(), glob) for glob in lowered)

    return action_filter


class StackOneToolset(WrapperToolset[AgentDepsT]):
    """StackOne actions on one linked SaaS account, as an agent toolset.

    A thin wrapper over an `MCPToolset` connected to StackOne's MCP endpoint,
    with StackOne auth and account headers applied and `actions` globs turned
    into a `FilteredToolset`. Being composed from core toolsets at construction
    keeps it visible to anything that rewrites toolsets, such as durable
    execution wrappers.

    Most users want the `StackOne` capability, which adds usage instructions
    and agent-spec support; use this class directly for toolset-level control,
    e.g. combinators like `approval_required()`.
    """

    def __init__(
        self,
        *,
        account_id: str,
        api_key: str | None = None,
        base_url: str = STACKONE_BASE_URL,
        actions: Sequence[str] = (),
        tool_mode: ToolMode | None = None,
        code_mode: bool = False,
        client: MCPToolsetClient | None = None,
        id: str = 'stackone',
    ) -> None:
        """Build the toolset.

        Args:
            account_id: The linked account to act on (one account is one provider connection).
            api_key: StackOne API key. Defaults to the `STACKONE_API_KEY` environment variable.
            base_url: StackOne API host. Point at a regional or staging host if needed.
            actions: `fnmatch` globs over full tool names (case-insensitive), e.g. `['*_list_*']`.
                Only apply in `individual` mode; explicitly requesting `search_execute`
                alongside `actions` warns.
            tool_mode: `individual` registers one tool per enabled action; `search_execute`
                registers two server-side meta-tools that search the catalog and execute
                actions by id, keeping the prompt footprint constant for large catalogs.
                `None` picks `search_execute`, or `individual` when `actions` are given.
            code_mode: Tag every tool with `code_mode=True` metadata so a
                `CodeMode(tools={'code_mode': True})` capability exposes StackOne calls
                inside its sandbox.
            client: Replacement for the default `{base_url}/mcp` connection: anything
                `MCPToolset` accepts (URL, `FastMCP` server, prebuilt `fastmcp.Client`).
                Auth headers and the `search_execute` query parameter are only applied
                when the client is a URL string.
            id: Stable toolset id. Give each instance a distinct id when an agent uses
                several StackOne toolsets.
        """
        warn_ignored_actions(tool_mode, actions, stacklevel=2)
        mode = resolve_tool_mode(tool_mode, actions)
        resolved = client if client is not None else f'{base_url.rstrip("/")}{_MCP_PATH}'
        headers: dict[str, str] | None = None
        if isinstance(resolved, str):
            resolved = _with_tool_mode(resolved, mode)
            headers = {'Authorization': _basic_auth(resolve_api_key(api_key)), 'x-account-id': account_id}
        toolset: AbstractToolset[AgentDepsT] = MCPToolset(resolved, id=id, headers=headers)
        if mode == 'individual' and actions:
            toolset = toolset.filtered(_action_filter(actions))
        if code_mode:
            toolset = toolset.with_metadata(code_mode=True)
        super().__init__(wrapped=toolset)
