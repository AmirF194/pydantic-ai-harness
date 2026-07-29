"""StackOne wire contract and toolset.

Wire contract, verified 2026-07-29:

- `POST {base_url}/mcp` -- MCP over streamable HTTP; lists and executes tools.
  `?tool-mode=search_execute` switches from one-tool-per-action to two
  search/execute meta-tools.
- Auth is `Authorization: Basic base64('{api_key}:')` plus an `x-account-id`
  header selecting the linked account. StackOne requires HTTPS.
- Tool names follow `{connector}_{action}_{entity}`, e.g.
  `bamboohr_list_employees`.
- Search/execute names end in `_search_actions` and `_execute_action`;
  the first returns runtime `action_id` values consumed by the second.

Sources: https://docs.stackone.com/mcp/quickstart and
https://docs.stackone.com/mcp/auth-security. Re-check with the documented
initialize and tools/list requests in both tool modes.
"""

from __future__ import annotations

import base64
import os
from collections.abc import Callable, Mapping, Sequence
from copy import copy
from fnmatch import fnmatch
from typing import Any, Literal, TypeGuard
from urllib.parse import unquote_plus, urlsplit, urlunsplit

from pydantic import AnyUrl
from pydantic_ai.exceptions import UserError
from pydantic_ai.tools import AgentDepsT, RunContext, ToolDefinition
from pydantic_ai.toolsets import AbstractToolset, ToolsetTool, WrapperToolset
from pydantic_core import to_json
from typing_extensions import Self

try:
    from pydantic_ai.mcp import MCPToolset, MCPToolsetClient
except ImportError as _import_error:  # pragma: no cover
    raise ImportError(
        'MCP support is required for the StackOne capability. Install it with: uv add "pydantic-ai-harness[stackone]"'
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
DEFAULT_MAX_OUTPUT_BYTES = 50 * 1024
DEFAULT_MAX_OUTPUT_LINES = 2000


def _is_sequence(value: object) -> TypeGuard[Sequence[object]]:
    return isinstance(value, Sequence)


def validate_configuration(tool_mode: object, actions: object) -> tuple[ToolMode | None, tuple[str, ...]]:
    if tool_mode is None:
        resolved_mode = None
    elif tool_mode == 'individual':
        resolved_mode = 'individual'
    elif tool_mode == 'search_execute':
        resolved_mode = 'search_execute'
    else:
        raise UserError('`tool_mode` must be `individual`, `search_execute`, or `None`.')

    if isinstance(actions, str):
        resolved_actions = (actions,)
    elif _is_sequence(actions) and not isinstance(actions, bytes):
        action_patterns: list[str] = []
        for action in actions:
            if not isinstance(action, str):
                raise UserError('`actions` must contain only string patterns.')
            action_patterns.append(action)
        resolved_actions = tuple(action_patterns)
    else:
        raise UserError('`actions` must be a string pattern or a sequence of string patterns.')

    if resolved_mode == 'search_execute' and resolved_actions:
        raise UserError(
            '`actions` filters cannot apply in `search_execute` mode because that mode registers '
            'only the search and execute tools. Use `individual` mode to filter action tools.'
        )
    return resolved_mode, resolved_actions


def validate_output_limits(max_output_bytes: object, max_output_lines: object) -> tuple[int, int]:
    if isinstance(max_output_bytes, bool) or not isinstance(max_output_bytes, int) or max_output_bytes <= 0:
        raise UserError('`max_output_bytes` must be a positive integer.')
    if isinstance(max_output_lines, bool) or not isinstance(max_output_lines, int) or max_output_lines <= 0:
        raise UserError('`max_output_lines` must be a positive integer.')
    return max_output_bytes, max_output_lines


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
            f'A StackOne API key is required: pass `api_key` or set the `{STACKONE_API_KEY_ENV}` environment variable.'
        )
    return resolved


def _basic_auth(api_key: str) -> str:
    token = base64.b64encode(f'{api_key}:'.encode()).decode()
    return f'Basic {token}'


def _with_tool_mode(url: str, tool_mode: ToolMode) -> str:
    parts = urlsplit(url)
    fields = parts.query.split('&') if parts.query else []
    tool_mode_fields = [field for field in fields if unquote_plus(field.partition('=')[0]) == 'tool-mode']
    if len(tool_mode_fields) == 1 and unquote_plus(tool_mode_fields[0].partition('=')[2]) == tool_mode:
        return url
    query = [field for field in fields if unquote_plus(field.partition('=')[0]) != 'tool-mode']
    if tool_mode == 'individual' and query == fields:
        return url
    if tool_mode == 'search_execute':
        query.append(_SEARCH_EXECUTE_QUERY)
    return urlunsplit(parts._replace(query='&'.join(query)))


def _validate_https_url(url: str, *, name: str) -> None:
    parts = urlsplit(url)
    if parts.scheme.lower() != 'https' or parts.hostname is None:
        raise UserError(f'`{name}` must be an absolute HTTPS URL.')


def _truncate_utf8(text: str, max_bytes: int) -> str:
    return text.encode('utf-8')[:max_bytes].decode('utf-8', errors='ignore')


def _limit_tool_output(value: object, *, max_bytes: int, max_lines: int) -> object:
    is_text = isinstance(value, str)
    if is_text:
        text = value
        data = value.encode('utf-8')
    else:
        data = to_json(value)
        text = data.decode('utf-8', errors='replace')

    line_count = len(text.splitlines()) or 1
    if len(data) <= max_bytes and line_count <= max_lines:
        return value

    action = 'truncated' if is_text else 'omitted'
    marker = f'[StackOne output {action} at {max_bytes} bytes or {max_lines} lines]'
    marker = _truncate_utf8(marker, max_bytes)
    if not is_text or max_lines == 1 or len(marker.encode('utf-8')) >= max_bytes:
        return marker

    preview_lines = text.splitlines()[: max_lines - 1]
    preview = '\n'.join(preview_lines)
    preview_budget = max_bytes - len(marker.encode('utf-8')) - 1
    preview = _truncate_utf8(preview, preview_budget)
    return f'{preview}\n{marker}' if preview else marker


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

    A thin wrapper over an `MCPToolset` connected to StackOne's MCP endpoint.
    URL clients receive StackOne auth and account headers. `actions` globs
    become a `FilteredToolset`. Prebuilt clients keep their own transport
    configuration.

    Use the `StackOne` capability for usage instructions and agent-spec support.
    Use this class for toolset combinators such as `approval_required()`.
    """

    def __init__(
        self,
        *,
        account_id: str,
        api_key: str | None = None,
        base_url: str = STACKONE_BASE_URL,
        actions: Sequence[str] = (),
        tool_mode: ToolMode | None = None,
        metadata: Mapping[str, object] | None = None,
        client: MCPToolsetClient | None = None,
        max_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES,
        max_output_lines: int = DEFAULT_MAX_OUTPUT_LINES,
        id: str = 'stackone',
    ) -> None:
        """Build a StackOne MCP toolset.

        Args:
            account_id: Linked account used for StackOne requests.
            api_key: API key, or `STACKONE_API_KEY` when omitted.
            base_url: HTTPS StackOne API host.
            actions: Case-insensitive globs over individual action tool names. Selects
                `individual`; incompatible with explicit `search_execute`.
            tool_mode: Individual tools or the search/execute pair. Inferred when omitted.
            metadata: Metadata merged onto each tool definition.
            client: URL, `FastMCP`, or prebuilt client accepted by `MCPToolset`.
                Non-URL clients must configure their own transport and auth.
            max_output_bytes: Serialized result byte cap. Oversized text is truncated;
                structured and binary results are omitted.
            max_output_lines: Serialized result line cap with the same lossy behavior.
            id: Toolset ID; use distinct values for multiple accounts.
        """
        tool_mode, actions = validate_configuration(tool_mode, actions)
        max_output_bytes, max_output_lines = validate_output_limits(max_output_bytes, max_output_lines)
        mode = resolve_tool_mode(tool_mode, actions)
        if client is None:
            _validate_https_url(base_url, name='base_url')
            parts = urlsplit(base_url)
            if parts.query or parts.fragment:
                raise UserError('`base_url` must not contain a query or fragment.')
            resolved: MCPToolsetClient = urlunsplit(parts._replace(path=f'{parts.path.rstrip("/")}{_MCP_PATH}'))
        else:
            resolved = client
        headers: dict[str, str] | None = None
        is_url = isinstance(resolved, AnyUrl) or (
            isinstance(resolved, str) and urlsplit(resolved).scheme.lower() in ('http', 'https')
        )
        if is_url:
            _validate_https_url(str(resolved), name='client')
            resolved = _with_tool_mode(str(resolved), mode)
            headers = {'Authorization': _basic_auth(resolve_api_key(api_key)), 'x-account-id': account_id}
        toolset: AbstractToolset[AgentDepsT] = MCPToolset(resolved, id=id, headers=headers)
        if mode == 'individual' and actions:
            toolset = toolset.filtered(_action_filter(actions))
        if metadata:
            toolset = toolset.with_metadata(**metadata)
        self._max_output_bytes = max_output_bytes
        self._max_output_lines = max_output_lines
        super().__init__(wrapped=toolset)

    def _with_wrapped(self, wrapped: AbstractToolset[AgentDepsT]) -> Self:
        result = copy(self)
        result.wrapped = wrapped
        return result

    async def for_run(self, ctx: RunContext[AgentDepsT]) -> AbstractToolset[AgentDepsT]:
        wrapped = await self.wrapped.for_run(ctx)
        return self if wrapped is self.wrapped else self._with_wrapped(wrapped)

    async def for_run_step(self, ctx: RunContext[AgentDepsT]) -> AbstractToolset[AgentDepsT]:
        wrapped = await self.wrapped.for_run_step(ctx)
        return self if wrapped is self.wrapped else self._with_wrapped(wrapped)

    def visit_and_replace(
        self, visitor: Callable[[AbstractToolset[AgentDepsT]], AbstractToolset[AgentDepsT]]
    ) -> AbstractToolset[AgentDepsT]:
        return self._with_wrapped(self.wrapped.visit_and_replace(visitor))

    async def call_tool(
        self, name: str, tool_args: dict[str, Any], ctx: RunContext[AgentDepsT], tool: ToolsetTool[AgentDepsT]
    ) -> Any:
        result = await self.wrapped.call_tool(name, tool_args, ctx, tool)
        return _limit_tool_output(
            result,
            max_bytes=self._max_output_bytes,
            max_lines=self._max_output_lines,
        )
