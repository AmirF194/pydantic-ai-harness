"""StackOne capability that gives agents access to actions on a linked SaaS account."""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from dataclasses import KW_ONLY, dataclass
from typing import Any

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.tools import AgentDepsT

from pydantic_ai_harness.stackone._toolset import (
    STACKONE_BASE_URL,
    MCPToolsetClient,
    StackOneToolset,
    ToolMode,
    check_actions_apply,
    resolve_tool_mode,
)

_INDIVIDUAL_INSTRUCTIONS = (
    "The StackOne tools operate on the user's linked SaaS account (HRIS, ATS, CRM, and more). "
    'Tool names follow `{connector}_{action}_{entity}`, for example `bamboohr_list_employees`. '
    'Results are JSON from the underlying provider. Prefer list actions with filters over unbounded listings.'
)
_SEARCH_EXECUTE_INSTRUCTIONS = (
    'StackOne is available through two tools: a search tool (name ending in `_search_actions`) that finds '
    'available actions from a natural-language query, and an execute tool (name ending in `_execute_action`) that '
    'runs one action by id. Always search first: `action_id` values are runtime identifiers returned by the search '
    'tool and must never be guessed.'
)


@dataclass
class StackOne(AbstractCapability[AgentDepsT]):
    """Actions on the user's SaaS account (HRIS, ATS, CRM, and more) via StackOne.

    StackOne (https://docs.stackone.com) is an integration platform exposing
    actions across 400+ SaaS providers through linked accounts. This
    capability connects the agent to one linked account's actions over
    StackOne's MCP endpoint, handling authentication, tool filtering, and
    usage instructions.
    """

    account_id: str
    """The linked account to act on (one account is one provider connection)."""

    _: KW_ONLY

    api_key: str | None = None
    """StackOne API key. Defaults to the `STACKONE_API_KEY` environment variable."""

    base_url: str = STACKONE_BASE_URL
    """StackOne API host. Point at a regional or staging host if needed."""

    actions: Sequence[str] = ()
    """`fnmatch` globs over full tool names (case-insensitive), e.g. `['*_list_*']`.
    Giving `actions` switches the default `tool_mode` to `individual`, where the globs apply."""

    tool_mode: ToolMode | None = None
    """`individual` registers one tool per enabled action; `search_execute` registers two
    server-side meta-tools (search the catalog, execute an action by id) whose prompt
    footprint stays constant however large the catalog is. `None` picks `search_execute`,
    or `individual` when `actions` are given."""

    include_instructions: bool = True
    """Inject StackOne usage instructions into the system prompt."""

    metadata: Mapping[str, Any] | None = None
    """Metadata merged onto every tool, available to tool-selection machinery such as
    `CodeMode(tools={'code_mode': True})` or custom `prepare_tools` hooks."""

    client: MCPToolsetClient | None = None
    """Replacement for the default `{base_url}/mcp` connection; see `StackOneToolset`."""

    def __post_init__(self) -> None:
        check_actions_apply(self.tool_mode, self.actions)
        if resolve_tool_mode(self.tool_mode, self.actions) == 'search_execute' and self.defer_loading:
            # stacklevel 3: user code -> generated `__init__` -> `__post_init__`.
            warnings.warn(
                '`defer_loading` hides the two `search_execute` meta-tools behind `tool_search`, adding '
                "a discovery hop on top of StackOne's own search. Consider `tool_mode='individual'` with "
                '`defer_loading`, or `search_execute` without it.',
                stacklevel=3,
            )

    def get_toolset(self) -> StackOneToolset[AgentDepsT]:
        """Build the StackOne toolset, failing fast if no API key is configured."""
        return StackOneToolset[AgentDepsT](
            account_id=self.account_id,
            api_key=self.api_key,
            base_url=self.base_url,
            actions=self.actions,
            tool_mode=self.tool_mode,
            metadata=self.metadata,
            client=self.client,
            id=self.id or 'stackone',
        )

    def get_instructions(self) -> str | None:
        """StackOne usage guidance; the underlying MCP toolset provides none itself."""
        if not self.include_instructions:
            return None
        mode = resolve_tool_mode(self.tool_mode, self.actions)
        return _SEARCH_EXECUTE_INSTRUCTIONS if mode == 'search_execute' else _INDIVIDUAL_INSTRUCTIONS

    @classmethod
    def get_serialization_name(cls) -> str:
        """Serialization name for agent-spec support.

        All fields are YAML-expressible; keep the API key out of spec files and
        rely on the `STACKONE_API_KEY` environment variable instead.
        """
        return 'StackOne'
