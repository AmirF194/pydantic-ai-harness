"""Provider-adaptive advisor capability."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

from pydantic_ai import AdvisorTool, Agent
from pydantic_ai.capabilities import NativeOrLocalTool
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import ModelMessage, ModelResponse, ToolCallPart
from pydantic_ai.models import KnownModelName, Model
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import AgentDepsT, AgentNativeTool, RunContext, Tool

AdvisorModel = Model | KnownModelName | str
"""A Pydantic AI model instance or model name used for advisor consultations."""

AdvisorMode = Literal['auto', 'native', 'local']
"""How advisor consultations are executed."""

_LOCAL_TOOL_NAME = 'advisor'
_LOCAL_TOOL_DESCRIPTION = (
    'Consult a stronger model about a difficult or high-impact decision. '
    'Include the complete question and all relevant context in `prompt` because '
    'the local advisor does not receive this conversation automatically.'
)
_LIMIT_REACHED = 'Advisor consultation limit reached for this model request. Continue without further advice.'


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

    _advisor_agent: Agent[None, str] | None = field(init=False, repr=False)
    _native_provider: Literal['anthropic', 'openrouter'] | None = field(init=False, repr=False)
    _native_model_name: str | None = field(init=False, repr=False)

    def __init__(
        self,
        model: AdvisorModel,
        *,
        mode: AdvisorMode = 'auto',
        max_uses: int | None = None,
        max_tokens: int | None = None,
        caching: Literal['5m', '1h'] | None = None,
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
        self._advisor_agent = None
        self._native_provider, self._native_model_name = self._parse_native_model(model)
        if mode == 'native' and self._native_provider is None:
            raise ValueError(
                "Advisor(mode='native') requires an 'anthropic:<model>' or 'openrouter:<model>' model name"
            )
        if mode == 'native' and self._native_provider == 'openrouter' and max_uses is not None:
            raise ValueError("Advisor.max_uses is not supported by OpenRouter in mode='native'")

        local_tool: Tool[AgentDepsT] = Tool(
            self._consult_advisor,
            name=_LOCAL_TOOL_NAME,
            description=_LOCAL_TOOL_DESCRIPTION,
            sequential=True,
        )
        native: AgentNativeTool[AgentDepsT] | bool
        local: Tool[AgentDepsT] | bool
        if mode == 'local':
            native = False
            local = local_tool
        elif mode == 'native':
            native = self._required_native_advisor
            local = False
        else:
            native = self._native_advisor
            local = local_tool
        super().__init__(
            native=native,
            local=local,
            id='advisor',
        )

    def _native_unique_id(self) -> str:
        """Identify the native tool paired with the local fallback."""
        return AdvisorTool.kind

    def _native_advisor(self, ctx: RunContext[AgentDepsT]) -> AdvisorTool | None:
        """Build a native tool only when an explicit advisor model name is lossless."""
        provider = ctx.model.system
        if provider != self._native_provider or self._native_model_name is None:
            return None
        if provider == 'openrouter' and self.max_uses is not None:
            return None
        return self._advisor_tool(self._native_model_name)

    def _required_native_advisor(self, ctx: RunContext[AgentDepsT]) -> AdvisorTool:
        """Build a required native tool or reject a cross-provider executor."""
        if ctx.model.system != self._native_provider:
            raise UserError(
                f"Advisor(mode='native') requires a {self._native_provider} executor, not {ctx.model.system}"
            )
        assert self._native_model_name is not None
        return self._advisor_tool(self._native_model_name)

    def _advisor_tool(self, model_name: str) -> AdvisorTool:
        return AdvisorTool(
            model=model_name,
            max_uses=self.max_uses,
            max_tokens=self.max_tokens,
            caching=self.caching,
        )

    async def before_run(self, ctx: RunContext[AgentDepsT]) -> None:
        """Reject local-capable execution before a durable run starts."""
        if self.mode != 'native' and any(
            any(
                base.__module__ == 'pydantic_ai.durable_exec._base' and base.__name__ == 'BaseDurabilityCapability'
                for base in type(capability).__mro__
            )
            for capability in ctx.capabilities.values()
        ):
            raise UserError(
                "Advisor modes 'auto' and 'local' are not compatible with durable execution; "
                "use mode='native' with a supported provider"
            )

    async def _consult_advisor(self, ctx: RunContext[AgentDepsT], prompt: str) -> str:
        """Consult the configured advisor model.

        Args:
            ctx: The executor run context.
            prompt: A self-contained question with all context the advisor needs.
        """
        try:
            messages = ctx.messages
        except UserError as e:
            raise UserError(
                'Advisor local execution is not compatible with a serialized durable tool activity; '
                "use mode='native' with a supported provider"
            ) from e

        if self.max_uses is not None and self._call_ordinal(messages, ctx.tool_call_id) > self.max_uses:
            return _LIMIT_REACHED
        if self._advisor_agent is None:
            settings = ModelSettings(max_tokens=self.max_tokens) if self.max_tokens is not None else None
            self._advisor_agent = Agent(
                self.model,
                instructions=(
                    'You are an expert advisor. Give concise, actionable advice to the executor model '
                    'about the question it sends you. Do not address the end user.'
                ),
                model_settings=settings,
            )
        result = await self._advisor_agent.run(
            prompt,
            usage=ctx.usage,
            usage_limits=ctx.usage_limits,
        )
        return result.output

    @staticmethod
    def _call_ordinal(messages: Sequence[ModelMessage], tool_call_id: str | None) -> int:
        """Return this advisor call's 1-based position in the current executor response."""
        for message in reversed(messages):
            if not isinstance(message, ModelResponse):  # pragma: no cover - the current response is last
                continue
            calls = [
                part for part in message.parts if isinstance(part, ToolCallPart) and part.tool_name == _LOCAL_TOOL_NAME
            ]
            for index, call in enumerate(calls, start=1):
                if call.tool_call_id == tool_call_id:
                    return index
            break  # pragma: no cover - every executing local tool has a matching call
        return 1  # pragma: no cover - agent tool execution always follows a model response

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
