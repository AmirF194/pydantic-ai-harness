"""Provider-adaptive advisor capability."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic_ai import AdvisorTool, Agent, AgentRunResult
from pydantic_ai.capabilities import NativeOrLocalTool, ValidatedToolArgs, WrapRunHandler
from pydantic_ai.exceptions import SkipToolExecution, UserError
from pydantic_ai.messages import ModelResponse, ToolCallPart
from pydantic_ai.models import KnownModelName, Model, ModelRequestContext
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import AgentDepsT, AgentNativeTool, RunContext, Tool, ToolDefinition

AdvisorModel = Model | KnownModelName | str
"""A Pydantic AI model instance or model name used for advisor consultations."""

AdvisorMode = Literal['auto', 'native', 'local']
"""How advisor consultations are executed."""

_LOCAL_TOOL_NAME = 'advisor'
_LOCAL_TOOL_DESCRIPTION = (
    'Consult a stronger model about a difficult or high-impact decision. '
    'Include the complete question and all relevant context in `prompt` because '
    'conversation history may not be available to the advisor.'
)
_LIMIT_REACHED = 'Advisor consultation limit reached for this model request. Continue without further advice.'


@runtime_checkable
class _DurabilityCapability(Protocol):
    """Structural view of core durability capabilities without importing optional integrations."""

    engine_name: str

    @property
    def in_durable_context(self) -> bool: ...  # pragma: no cover


@dataclass(init=False)
class Advisor(NativeOrLocalTool[AgentDepsT]):
    """Let an agent consult another model through a provider-native tool or local fallback.

    In `auto` mode, `Advisor` uses Pydantic AI's native `AdvisorTool` when an
    explicit provider-qualified model name matches a compatible Anthropic or
    OpenRouter executor. On every other model, it exposes an `advisor` function
    tool backed by a separate Pydantic AI agent.

    ```python
    from pydantic_ai import Agent
    from pydantic_ai_harness.advisor import Advisor

    agent = Agent(
        'openai:gpt-5.4',
        capabilities=[Advisor('anthropic:claude-opus-4-8')],
    )
    ```
    """

    model: AdvisorModel
    """The model to consult.

    Accepts the same model names and model instances as `Agent`. In `auto`
    mode, model instances use local execution so their provider configuration
    is preserved.
    """

    mode: AdvisorMode
    """How advisor consultations are executed.

    `auto` uses a native advisor only for an explicit same-provider model name.
    `native` requires a provider-native advisor, and `local` always runs a
    separate Pydantic AI agent.
    """

    max_uses: int | None
    """Maximum consultations in one executor model request.

    The limit resets on the next executor request. OpenRouter's native advisor
    does not honor this option, so setting it selects the local fallback there.
    """

    max_tokens: int | None
    """Maximum output tokens for each advisor consultation.

    Values below 1024 are rejected so the setting remains valid on every native
    and local execution path.
    """

    caching: Literal['5m', '1h'] | None
    """Anthropic-native advisor prompt caching.

    This is an opportunistic optimization. OpenRouter and the local fallback do
    not provide an equivalent cache control.
    """

    forward_history: bool
    """Whether local consultations receive the executor's completed message history.

    Native execution keeps the provider's transcript behavior unchanged.
    """

    _local_uses: dict[str, int] = field(init=False, repr=False)

    def __init__(
        self,
        model: AdvisorModel,
        *,
        mode: AdvisorMode = 'auto',
        max_uses: int | None = None,
        max_tokens: int | None = None,
        caching: Literal['5m', '1h'] | None = None,
        forward_history: bool = False,
    ) -> None:
        if mode not in {'auto', 'native', 'local'}:
            raise ValueError("Advisor.mode must be 'auto', 'native', or 'local'")
        if max_uses is not None and max_uses < 1:
            raise ValueError('Advisor.max_uses must be at least 1')
        if max_tokens is not None and max_tokens < 1024:
            raise ValueError('Advisor.max_tokens must be at least 1024')

        self.model = model
        self.mode = mode
        self.max_uses = max_uses
        self.max_tokens = max_tokens
        self.caching = caching
        self.forward_history = forward_history
        self._local_uses = {}
        native_provider, _ = self._parse_native_model(model)
        if mode == 'native' and native_provider is None:
            raise ValueError(
                "Advisor(mode='native') requires an 'anthropic:<model>' or 'openrouter:<model>' model name"
            )
        if mode == 'native' and native_provider == 'openrouter' and max_uses is not None:
            raise ValueError("Advisor.max_uses is not supported by OpenRouter in mode='native'")

        native: AgentNativeTool[AgentDepsT] | bool
        local: Tool[AgentDepsT] | bool
        if mode == 'native':
            native = self._required_native_advisor
            local = False
        else:

            async def local_advisor(ctx: RunContext[AgentDepsT], prompt: str) -> str:
                settings = ModelSettings(max_tokens=max_tokens) if max_tokens is not None else None
                advisor = Agent(
                    model,
                    instructions=(
                        'You are an expert advisor. Give concise, actionable advice to the executor model '
                        'about the question it sends you. Do not address the end user.'
                    ),
                    model_settings=settings,
                )
                result = await advisor.run(
                    prompt,
                    message_history=ctx.messages[:-1] if forward_history else None,
                    usage=ctx.usage,
                    usage_limits=ctx.usage_limits,
                )
                return result.output

            local = Tool(
                local_advisor,
                name=_LOCAL_TOOL_NAME,
                description=_LOCAL_TOOL_DESCRIPTION,
            )
            native = False if mode == 'local' else self._native_advisor
        super().__init__(
            native=native,
            local=local,
            id='advisor',
        )

    async def after_model_request(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        request_context: ModelRequestContext,
        response: ModelResponse,
    ) -> ModelResponse:
        """Reset the local consultation allowance for each executor response."""
        self._local_uses[ctx.run_id or ''] = 0
        return response

    async def before_tool_execute(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: ValidatedToolArgs,
    ) -> ValidatedToolArgs:
        """Enforce `max_uses` after argument validation."""
        if tool_def.capability_id == self.id and self.max_uses is not None:
            run_id = ctx.run_id or ''
            local_uses = self._local_uses.get(run_id, 0)
            if local_uses >= self.max_uses:
                raise SkipToolExecution(_LIMIT_REACHED)
            self._local_uses[run_id] = local_uses + 1
        return args

    async def wrap_run(self, ctx: RunContext[AgentDepsT], *, handler: WrapRunHandler) -> AgentRunResult[Any]:
        """Run the agent, then discard its local consultation count."""
        try:
            return await handler()
        finally:
            self._local_uses.pop(ctx.run_id or '', None)

    def _native_unique_id(self) -> str:
        """Identify the native tool paired with the local fallback."""
        return AdvisorTool.kind

    def _native_advisor(self, ctx: RunContext[AgentDepsT]) -> AdvisorTool | None:
        """Build a native tool only when an explicit advisor model name is lossless."""
        native_provider, native_model_name = self._parse_native_model(self.model)
        provider = ctx.model.system
        if provider != native_provider or native_model_name is None:
            return None
        if provider == 'openrouter' and self.max_uses is not None:
            return None
        return self._advisor_tool(native_model_name)

    def _required_native_advisor(self, ctx: RunContext[AgentDepsT]) -> AdvisorTool:
        """Build a required native tool or reject a cross-provider executor."""
        native_provider, native_model_name = self._parse_native_model(self.model)
        if ctx.model.system != native_provider:
            raise UserError(f"Advisor(mode='native') requires a {native_provider} executor, not {ctx.model.system}")
        assert native_model_name is not None
        return self._advisor_tool(native_model_name)

    def _advisor_tool(self, model_name: str) -> AdvisorTool:
        return AdvisorTool(
            model=model_name,
            max_uses=self.max_uses,
            max_tokens=self.max_tokens,
            caching=self.caching,
        )

    async def before_run(self, ctx: RunContext[AgentDepsT]) -> None:
        """Reject local-capable execution inside an active durable container."""
        if self.mode != 'native' and any(
            isinstance(capability, _DurabilityCapability) and capability.in_durable_context
            for capability in ctx.capabilities.values()
        ):
            raise UserError(
                "Advisor modes 'auto' and 'local' are not compatible with active durable execution; "
                "use mode='native' with a supported provider"
            )

    @staticmethod
    def _parse_native_model(
        model: AdvisorModel,
    ) -> tuple[Literal['anthropic', 'openrouter'] | None, str | None]:
        """Extract a native model ID only from an explicit provider-qualified name."""
        if not isinstance(model, str):
            return None, None
        provider, separator, model_name = model.partition(':')
        if separator and model_name:
            if provider == 'anthropic':
                return 'anthropic', model_name
            if provider == 'openrouter':
                return 'openrouter', model_name
        return None, None

    @classmethod
    def get_serialization_name(cls) -> str | None:
        """Model instances are not spec-serializable."""
        return None
