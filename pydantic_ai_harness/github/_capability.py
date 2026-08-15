"""GitHub capability backed by the official GitHub MCP server running in Docker."""

from __future__ import annotations

import os
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.exceptions import UserError
from pydantic_ai.tools import AgentDepsT, RunContext, ToolDefinition
from pydantic_ai.toolsets import AbstractToolset, AgentToolset

_DEFAULT_IMAGE = 'ghcr.io/github/github-mcp-server'
_TOKEN_ENV_VARS = ('GITHUB_PERSONAL_ACCESS_TOKEN', 'GITHUB_TOKEN')
_PAT_ENV_VAR = 'GITHUB_PERSONAL_ACCESS_TOKEN'
_RESERVED_ENV = frozenset(
    {_PAT_ENV_VAR, 'GITHUB_TOOLSETS', 'GITHUB_READ_ONLY', 'GITHUB_DYNAMIC_TOOLSETS', 'GITHUB_HOST'}
)
"""Environment keys owned by first-class fields; `env` may not set them (see `_build_environment`)."""


@dataclass
class GitHub(AbstractCapability[AgentDepsT]):
    """Give an agent the GitHub MCP server, run as a Docker subprocess over stdio.

    Spawns the official [`github/github-mcp-server`](https://github.com/github/github-mcp-server)
    container and exposes its tools to the agent. Tools can be limited at two levels:

    - **server-side** (`toolsets`, `read_only`, `dynamic_toolsets`): the server only
      advertises the selected tool groups, so the model never sees the rest and no
      tokens are spent describing them.
    - **client-side** (`allowed_tools`, `denied_tools`, `tool_filter`): the advertised
      tools are filtered before reaching the model, for fine-grained control.

    ```python
    from pydantic_ai import Agent
    from pydantic_ai_harness import GitHub

    agent = Agent(
        'anthropic:claude-sonnet-4-6',
        capabilities=[
            GitHub(toolsets=['repos', 'issues', 'pull_requests'], read_only=True),
        ],
    )
    ```

    The token is read from `token` or, when unset, the `GITHUB_PERSONAL_ACCESS_TOKEN`
    or `GITHUB_TOKEN` environment variables. Requires the `mcp` extra
    (`pip install "pydantic-ai-harness[github]"`) and a working Docker (or compatible)
    runtime on the host.
    """

    token: str | None = None
    """GitHub personal access token. An empty value is treated as unset and falls back to the
    `GITHUB_PERSONAL_ACCESS_TOKEN` then `GITHUB_TOKEN` environment variables."""

    toolsets: Sequence[str] | None = None
    """Server-side toolset groups to enable (e.g. `['repos', 'issues', 'pull_requests']`).

    Maps to the server's `GITHUB_TOOLSETS`. `None` leaves the server default. Use `['all']`
    to enable everything. See the github-mcp-server docs for the full list of groups.
    """

    read_only: bool = False
    """Run the server in read-only mode, so no tool can mutate GitHub state."""

    dynamic_toolsets: bool = False
    """Let the model discover and enable toolsets on demand instead of listing them all up front."""

    allowed_tools: Sequence[str] | None = None
    """If set, only tools whose (unprefixed) name is listed are exposed to the model."""

    denied_tools: Sequence[str] | None = None
    """Tools whose (unprefixed) name is listed are hidden from the model.

    Combines with `allowed_tools`: a tool must be allowed *and* not denied.
    """

    tool_filter: Callable[[RunContext[AgentDepsT], ToolDefinition], bool | Awaitable[bool]] | None = None
    """Optional predicate for fine-grained filtering, applied after `allowed_tools`/`denied_tools`."""

    tool_prefix: str | None = None
    """Prefix added to every GitHub tool name (e.g. `gh` makes `get_issue` into `gh_get_issue`)."""

    host: str | None = None
    """GitHub host for GitHub Enterprise Server (maps to `GITHUB_HOST`). `None` uses github.com."""

    docker_image: str = _DEFAULT_IMAGE
    """Container image reference for the GitHub MCP server."""

    docker_command: str = 'docker'
    """Executable used to run the container (set to `podman` to use Podman)."""

    docker_args: Sequence[str] = field(default_factory=tuple)
    """Extra arguments inserted into `docker run` before the image (e.g. extra `-e`/`-v` flags)."""

    env: Mapping[str, str] | None = None
    """Extra environment variables to set in the server process and forward into the container.

    Keys owned by the first-class fields (`GITHUB_PERSONAL_ACCESS_TOKEN`, `GITHUB_TOOLSETS`,
    `GITHUB_READ_ONLY`, `GITHUB_DYNAMIC_TOOLSETS`, `GITHUB_HOST`) are rejected here so `env`
    cannot silently override a safety setting -- use the matching field instead.
    """

    init_timeout: float = 30.0
    """Seconds to wait for the server to initialize (a cold `docker run` may pull the image first)."""

    read_timeout: float = 300.0
    """Seconds to wait for new messages on the established connection before it is considered stale."""

    include_instructions: bool = True
    """Include the server's own instructions in the agent's system prompt."""

    id: str | None = 'github'
    """Stable identifier for the MCP server, used by durable execution backends.

    Give each one a distinct `id` when attaching more than one GitHub server to an agent.
    """

    def _resolve_token(self) -> str:
        if self.token:
            return self.token
        for name in _TOKEN_ENV_VARS:
            value = os.environ.get(name)
            if value:
                return value
        raise UserError(
            f'GitHub() needs a token: pass `token=...` or set the {" or ".join(_TOKEN_ENV_VARS)} environment variable.'
        )

    def _build_environment(self) -> dict[str, str]:
        """Environment for the server process; every key is also forwarded into the container."""
        environment: dict[str, str] = {_PAT_ENV_VAR: self._resolve_token()}
        if self.toolsets is not None:
            environment['GITHUB_TOOLSETS'] = ','.join(self.toolsets)
        if self.read_only:
            environment['GITHUB_READ_ONLY'] = '1'
        if self.dynamic_toolsets:
            environment['GITHUB_DYNAMIC_TOOLSETS'] = '1'
        if self.host is not None:
            environment['GITHUB_HOST'] = self.host
        if self.env is not None:
            reserved = _RESERVED_ENV.intersection(self.env)
            if reserved:
                raise UserError(
                    f'GitHub(env=...) cannot set reserved keys ({", ".join(sorted(reserved))}); '
                    'use the matching GitHub() fields (token, toolsets, read_only, dynamic_toolsets, host) instead.'
                )
            environment.update(self.env)
        return environment

    def _tool_filter(self) -> Callable[[RunContext[AgentDepsT], ToolDefinition], bool | Awaitable[bool]] | None:
        """Combine the allow list, deny list, and predicate into one filter over raw tool names.

        Names are matched before any `tool_prefix` is applied, so the lists stay stable
        regardless of prefix. Returns `None` when no client-side filtering is configured.
        """
        if self.allowed_tools is None and self.denied_tools is None and self.tool_filter is None:
            return None
        allowed = frozenset(self.allowed_tools) if self.allowed_tools is not None else None
        denied = frozenset(self.denied_tools) if self.denied_tools is not None else None
        predicate = self.tool_filter

        def tool_filter(ctx: RunContext[AgentDepsT], tool_def: ToolDefinition) -> bool | Awaitable[bool]:
            if allowed is not None and tool_def.name not in allowed:
                return False
            if denied is not None and tool_def.name in denied:
                return False
            if predicate is not None:
                return predicate(ctx, tool_def)
            return True

        return tool_filter

    def _build_server(self) -> AbstractToolset[AgentDepsT]:
        try:
            from pydantic_ai.mcp import MCPServerStdio
        except ImportError as e:
            raise UserError('GitHub() requires the `mcp` extra — `pip install "pydantic-ai-harness[github]"`.') from e

        environment = self._build_environment()
        docker_args = ['run', '-i', '--rm']
        for name in environment:
            docker_args += ['-e', name]
        docker_args += list(self.docker_args)
        docker_args.append(self.docker_image)

        return MCPServerStdio(
            command=self.docker_command,
            args=docker_args,
            env=environment,
            timeout=self.init_timeout,
            read_timeout=self.read_timeout,
            include_instructions=self.include_instructions,
            id=self.id,
        )

    def get_toolset(self) -> AgentToolset[AgentDepsT] | None:
        """Build the GitHub MCP toolset, applying tool filtering and prefixing."""
        toolset = self._build_server()
        tool_filter = self._tool_filter()
        if tool_filter is not None:
            toolset = toolset.filtered(tool_filter)
        if self.tool_prefix is not None:
            toolset = toolset.prefixed(self.tool_prefix)
        return toolset
