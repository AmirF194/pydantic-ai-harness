"""Back an agent's Agent Control configuration with one Logfire variable."""

from __future__ import annotations

import json
import warnings
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, ClassVar, Literal, TypeAlias, cast

import logfire
from logfire.variables import Variable
from pydantic import BaseModel, ConfigDict, Field
from pydantic_ai import AbstractToolset, RunContext, TemplateStr, ToolDefinition, WrapperToolset
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import ModelRequest
from pydantic_ai.models import ModelRequestContext
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import AgentDepsT
from pydantic_ai.toolsets.abstract import ToolsetTool

from pydantic_ai_harness.logfire._managed_variable import ManagedVariableCapability

if TYPE_CHECKING:
    from pydantic_ai.agent.abstract import AgentModelSettings
    from pydantic_ai.capabilities import AgentModel, ModelSelection
    from pydantic_ai.capabilities.abstract import WrapRunHandler
    from pydantic_ai.models import ModelSelectionContext
    from pydantic_ai.run import AgentRunResult

_AGENT_VARIABLE_PREFIX = 'agent__'


class AgentConfigSettings(BaseModel):
    """Canonical model settings managed as one section of an `AgentConfig`.

    Every field name matches a key in [`ModelSettings`][pydantic_ai.settings.ModelSettings], so the
    payload lowers without translation. Unset fields keep their code-defined values. Extra fields
    are allowed so canonical settings introduced by a newer Logfire UI can flow through older SDKs.
    The model itself is the sibling `model` field on `AgentConfig`.
    """

    model_config = ConfigDict(extra='allow')

    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    seed: int | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    parallel_tool_calls: bool | None = None
    timeout: float | None = None
    stop_sequences: list[str] | None = None
    thinking: bool | Literal['minimal', 'low', 'medium', 'high', 'xhigh'] | None = None
    service_tier: Literal['auto', 'default', 'flex', 'priority'] | None = None
    provider_options: dict[str, dict[str, Any]] | None = None
    """Per-provider escape hatch.

    `provider_options[provider][key]` lowers to the flat `<provider>_<key>` model setting. Provider
    options are applied after canonical fields, so a provider-specific value wins on collision.
    """


def _lower_settings(value: AgentConfigSettings) -> ModelSettings:
    """Lower an `AgentConfigSettings` value to Pydantic AI's flat `ModelSettings` shape.

    Only explicitly set, non-`None` canonical fields are emitted. `provider_options` entries are
    flattened to `<provider>_<key>` afterwards, giving them precedence over canonical fields that
    lower to the same key.
    """
    settings: dict[str, Any] = value.model_dump(exclude_none=True, exclude={'provider_options'})
    if value.provider_options:
        for provider, options in value.provider_options.items():
            for key, option in options.items():
                settings[f'{provider}_{key}'] = option
    return cast(ModelSettings, settings)


class ToolDefinitionOverride(BaseModel):
    """A patch over a tool's LLM-facing definition.

    Overrides change only what the model is shown. Schema structure, validation, and execution
    remain code-defined. The dictionary key in `AgentConfig.tool_definitions` looks up the tool by
    its original code-side name.
    """

    new_name: str | None = Field(default=None, min_length=1)
    """Replacement name shown to the model; `None` keeps the original.

    A renamed call still routes back to the original code implementation.
    """
    description: str | None = None
    """Replacement description shown to the model; `None` keeps the code-defined description."""
    parameter_descriptions: dict[str, str] | None = None
    """Replacement description text for named top-level parameters.

    Names, types, and requiredness stay exactly as defined in code, so argument validation is
    unaffected. Unknown parameter names are ignored.
    """


class AgentConfig(BaseModel):
    """The schema contract shared with the Logfire Managed agents UI.

    Every managed value is a patch on the code-defined agent. A key present in the value is managed
    from Logfire; an absent key keeps code-defined behavior. Removing a key in Logfire is therefore
    a deliberate revert to code. Extra keys retain Pydantic's default `ignore` behavior: a key
    written by a newer UI must not fail validation, because the SDK's validation fallback would
    otherwise revert every section to code rather than only ignoring the unknown section.
    """

    model_config = ConfigDict(protected_namespaces=())

    instructions: str | None = None
    """Instructions replacing what the model is shown when present.

    `{{...}}` runtime placeholders are rendered against `deps` only when
    `AgentControl.render_template` is set.
    """
    model: str | None = None
    """A Pydantic AI model string such as `'openai:gpt-5'`; `None` keeps the code model."""
    settings: AgentConfigSettings | None = None
    """Canonical model settings patch; see `AgentConfigSettings`."""
    tool_definitions: dict[str, ToolDefinitionOverride] | None = None
    """LLM-facing overlays keyed by each tool's original code-side name."""


OverridesProvider: TypeAlias = Callable[[], Mapping[str, ToolDefinitionOverride]]


def _with_parameter_descriptions(
    parameters_json_schema: dict[str, Any], parameter_descriptions: dict[str, str]
) -> dict[str, Any]:
    """Patch top-level parameter descriptions while preserving all schema structure."""
    if not isinstance(parameters_json_schema.get('properties'), dict):
        return parameters_json_schema
    properties: dict[str, Any] = parameters_json_schema['properties']
    new_properties: dict[str, Any] = {}
    changed = False
    for name, schema in properties.items():
        if name in parameter_descriptions and isinstance(schema, dict):
            param_schema: dict[str, Any] = properties[name]
            new_properties[name] = {**param_schema, 'description': parameter_descriptions[name]}
            changed = True
        else:
            new_properties[name] = schema
    if not changed:
        return parameters_json_schema
    return {**parameters_json_schema, 'properties': new_properties}


def _apply_override(tool_def: ToolDefinition, override: ToolDefinitionOverride) -> ToolDefinition:
    """Apply the LLM-facing parts of an override, returning the original definition for a no-op."""
    changes: dict[str, Any] = {}
    if override.new_name is not None and override.new_name != tool_def.name:
        changes['name'] = override.new_name
    if override.description is not None and override.description != tool_def.description:
        changes['description'] = override.description
    if override.parameter_descriptions:
        schema = _with_parameter_descriptions(tool_def.parameters_json_schema, override.parameter_descriptions)
        if schema is not tool_def.parameters_json_schema:
            changes['parameters_json_schema'] = schema
    # `replace` preserves concrete `ToolDefinition` subclasses and fields added by the framework.
    return replace(tool_def, **changes) if changes else tool_def


@dataclass
class _ToolDefinitionOverridesToolset(WrapperToolset[AgentDepsT]):
    """Overlay advertised definitions while routing calls to their code-side tool names."""

    get_overrides: OverridesProvider = field(repr=False, compare=False)

    def _effective_tools(
        self, tools: dict[str, ToolsetTool[AgentDepsT]], *, warn: bool
    ) -> dict[str, tuple[str, ToolsetTool[AgentDepsT]]]:
        """Build the deterministic advertised-name to original-tool mapping.

        Listing and routing both use this method so collision handling cannot disagree between the
        schema the model saw and the call path. A colliding rename is dropped while its other
        patches remain, preserving every tool under a callable name.
        """
        overrides = self.get_overrides()
        result: dict[str, tuple[str, ToolsetTool[AgentDepsT]]] = {}
        for original_name, tool in tools.items():
            override = overrides.get(original_name)
            new_tool_def = tool.tool_def if override is None else _apply_override(tool.tool_def, override)
            new_name = new_tool_def.name
            if new_name != original_name and (new_name in tools or new_name in result):
                if warn:
                    warnings.warn(
                        f'Managed tool definition override renames {original_name!r} to {new_name!r}, '
                        f'which is already advertised by another tool; keeping the original name {original_name!r}.',
                        stacklevel=2,
                    )
                new_tool_def = replace(new_tool_def, name=original_name)
                new_name = original_name
            result[new_name] = (
                original_name,
                tool if new_tool_def is tool.tool_def else replace(tool, toolset=self, tool_def=new_tool_def),
            )
        return result

    async def get_tools(self, ctx: RunContext[AgentDepsT]) -> dict[str, ToolsetTool[AgentDepsT]]:
        """Return tools with managed definitions and collision-safe advertised names."""
        tools = await super().get_tools(ctx)
        if not self.get_overrides():
            return tools
        return {name: tool for name, (_, tool) in self._effective_tools(tools, warn=True).items()}

    async def call_tool(
        self, name: str, tool_args: dict[str, Any], ctx: RunContext[AgentDepsT], tool: ToolsetTool[AgentDepsT]
    ) -> Any:
        """Translate a renamed model call back to its original implementation and context name."""
        overrides = self.get_overrides()
        if not overrides or not any(o.new_name for o in overrides.values()):
            return await super().call_tool(name, tool_args, ctx, tool)
        # Recompute from live tools for correctness if a dynamic toolset changes between listing and calling;
        # this path runs only when at least one rename exists, so the micro-optimization is not worthwhile.
        entry = self._effective_tools(await super().get_tools(ctx), warn=False).get(name)
        if entry is not None and entry[0] != name:
            original_name = entry[0]
            ctx = replace(ctx, tool_name=original_name)
            tool = replace(tool, tool_def=replace(tool.tool_def, name=original_name))
            return await super().call_tool(original_name, tool_args, ctx, tool)
        return await super().call_tool(name, tool_args, ctx, tool)


@dataclass
class AgentControl(ManagedVariableCapability[AgentDepsT, AgentConfig]):
    """Manage an agent's config through one `agent__<name>` Logfire variable.

    The variable holds an `AgentConfig`. Each present section -- `instructions`, `model`, `settings`,
    or `tool_definitions` -- patches the code-defined agent, while an absent section keeps the
    code-defined behavior. Removing a section in Logfire deliberately reverts that section to code.

    When `name` is omitted, the variable name is derived from the agent's telemetry name using the
    same normalization as the Logfire UI. The variable is resolved once per run, and its label and
    version baggage remains active for the whole run. Managed settings override constructor settings;
    settings passed to `run()` override both.

    The managed `model` is sourced during model selection, so it slots in with the right precedence:
    a call-site `run(model=...)` beats it, it beats the agent's constructor model, and a fully
    model-less `Agent(None, ...)` -- named or nameless -- can be driven entirely from managed config.
    A named capability supplies the model statically (resolved once at run setup); a nameless one
    supplies a selector that derives its variable from the agent when the model is first selected,
    then reuses that choice for the rest of the run. Callable `targeting_key`/`attributes` don't
    participate in model selection (it runs before a run context exists) -- only the static `label`
    and static targeting inputs do.

    Tool overrides change only the definitions shown to the model. Renames route back to the original
    implementation, collisions retain the original name with a warning, and unknown tool keys are
    inert. Parameter names, types, requiredness, validation, and implementation stay code-owned.

    Missing, invalid, or unreachable remote values degrade to the code-defined agent through Logfire's
    resolution fallback. If the provider does not know the variable, auto-create is attempted once per
    process in the background. Its `example` is an `AgentConfig`-shaped snapshot of the code-side agent
    from the first model request -- the baseline read by the Logfire UI and optimizer.

    ```python
    import logfire
    from pydantic_ai import Agent

    from pydantic_ai_harness.logfire import AgentControl

    logfire.configure()
    agent = Agent('openai:gpt-5', name='checkout_assistant', capabilities=[AgentControl(label='production')])
    result = agent.run_sync('Refund my last order.')
    ```

    Runtime `{{...}}` placeholders pass through unless `render_template=True`. During a run,
    `logfire.managed.applied_sections` lists the present sections. The `model` section is reported
    when present even if a call-site model outranked it for that run.
    """

    name: str | Variable[AgentConfig] | None = None
    """Bare variable name, pre-built variable, or `None` to derive it from the agent name.

    A nameless capability derives its variable (and can source the model) from the agent's `name`
    the first time it's needed in a run; the agent must then have a `name`.
    """
    default: AgentConfig | None = None
    """Code-side fallback config; omitted sections preserve the corresponding agent behavior."""
    render_template: bool = False
    """Render managed instruction `{{...}}` placeholders against run dependencies when enabled."""

    _auto_create_in_wrap_run: ClassVar[bool] = False

    def __post_init__(self) -> None:
        self._setup_variable(
            self.name, prefix=_AGENT_VARIABLE_PREFIX, value_type=AgentConfig, default=self.default or AgentConfig()
        )

    def get_instructions(self) -> Callable[[RunContext[AgentDepsT]], str | None]:
        """Contribute managed instructions, optionally rendered against the run dependencies."""

        def instructions(ctx: RunContext[AgentDepsT]) -> str | None:
            resolved = self.resolved
            if resolved is None or resolved.value.instructions is None:
                return None
            value = resolved.value.instructions
            return TemplateStr[AgentDepsT](value).render(ctx.deps) if self.render_template else value

        return instructions

    def get_model_settings(self) -> AgentModelSettings[AgentDepsT] | None:
        """Contribute the lowered managed settings patch for each model request."""

        def model_settings(ctx: RunContext[AgentDepsT]) -> ModelSettings:
            resolved = self.resolved
            if resolved is None or resolved.value.settings is None:
                return ModelSettings()
            return _lower_settings(resolved.value.settings)

        return model_settings

    def get_model(self) -> AgentModel[AgentDepsT] | None:
        """Source the managed model with the right precedence (`run(model=...)` > managed > constructor).

        When the backing variable is already known (an explicit `name` or `Variable`), the model is
        sourced statically here, so Pydantic AI resolves it once at run setup. When the capability is
        nameless, the variable is derived from the agent's `name`, which isn't available until a
        [`ModelSelectionContext`][pydantic_ai.models.ModelSelectionContext] exists, so a selector
        callable is returned instead: it derives and resolves the variable the first time Pydantic AI
        selects the model for the run. Either way a fully model-less agent can be driven entirely
        from Logfire, and a call-site `run(model=...)` still wins.

        Both paths read the value with a **bare** `variable.get()` -- they read `.value` and never
        enter the [`ResolvedVariable`][logfire.variables.ResolvedVariable] as a context manager -- so
        model selection contributes no baggage: [`wrap_run`][pydantic_ai_harness.logfire.AgentControl.wrap_run]
        stays the sole owner of the run's resolution baggage. Model selection runs before a
        `RunContext` exists, so callable `targeting_key`/`attributes` can't participate (a
        `ModelSelectionContext` is deliberately narrower); only the static `label` and static
        targeting inputs do. That early read and `wrap_run`'s authoritative resolve return a
        consistent value -- `get()` is a cheap lookup over the SDK's cached config, and resolution is
        deterministic within a run.
        """
        if not self._name_omitted:
            return self._resolve_model_value(self._variable)
        return self._model_selector()

    def _resolve_model_value(self, variable: Variable[AgentConfig]) -> str | None:
        """Bare-read the managed value's `model` via static targeting -- no CM entry, so no baggage."""
        targeting_key = None if callable(self.targeting_key) else self.targeting_key
        attributes = None if callable(self.attributes) else self.attributes
        return variable.get(targeting_key=targeting_key, attributes=attributes, label=self.label).value.model

    def _model_selector(self) -> Callable[[ModelSelectionContext[AgentDepsT]], ModelSelection]:
        """Build the per-run selector a nameless `get_model` returns.

        Pydantic AI evaluates a selector once per new request step, but the managed model is a
        run-stable config value, so the selector memoizes its first choice and every later step
        reuses it -- one resolve per run, matching the static (named) path rather than re-reading the
        variable each step. A fresh selector (and fresh memo) is built per `get_model` call, i.e. per
        run, so nothing leaks across runs. On the first evaluation it derives the backing variable
        from `ctx.agent` (the same derivation `wrap_run` performs), reads the managed model, and
        falls back to the model Pydantic AI already selected (`ctx.model`) when none is managed --
        raising only when there is no model at all (a nameless, model-less agent with nothing
        published yet), so the misconfiguration surfaces clearly instead of downstream.
        """
        selected: list[ModelSelection] = []

        def select(ctx: ModelSelectionContext[AgentDepsT]) -> ModelSelection:
            if not selected:
                model = self._resolve_model_value(self._ensure_variable_for_agent(ctx.agent))
                if model is not None:
                    selected.append(model)
                elif ctx.model is not None:
                    selected.append(ctx.model)
                else:
                    raise UserError(
                        'A nameless `AgentControl` on a model-less agent has no model to run: the agent '
                        'defines no model and none is published in Logfire yet. Give the agent a model, '
                        'pass one to `run(model=...)`, or publish a `model` in the managed config.'
                    )
            return selected[0]

        return select

    def get_wrapper_toolset(self, toolset: AbstractToolset[AgentDepsT]) -> AbstractToolset[AgentDepsT]:
        """Wrap the agent toolset with managed LLM-facing definition overlays."""
        return _ToolDefinitionOverridesToolset(wrapped=toolset, get_overrides=self._current_overrides)

    def _current_overrides(self) -> Mapping[str, ToolDefinitionOverride]:
        """Return the active run's overrides, or an empty mapping outside a resolved run."""
        resolved = self.resolved
        return {} if resolved is None else resolved.value.tool_definitions or {}

    async def wrap_run(self, ctx: RunContext[AgentDepsT], *, handler: WrapRunHandler) -> AgentRunResult[Any]:
        """Add applied-section baggage inside the base's once-per-run resolution context."""

        async def wrapped() -> AgentRunResult[Any]:
            resolved = self.resolved
            assert resolved is not None  # set by the base's wrap_run before it invokes the handler
            sections = ','.join(
                name
                for name in ('instructions', 'model', 'settings', 'tool_definitions')
                if getattr(resolved.value, name) is not None
            )
            if sections:
                with logfire.set_baggage(**{'logfire.managed.applied_sections': sections}):
                    return await handler()
            return await handler()

        return await super().wrap_run(ctx, handler=wrapped)

    async def before_model_request(
        self, ctx: RunContext[AgentDepsT], request_context: ModelRequestContext
    ) -> ModelRequestContext:
        """Capture the code-side creation baseline on the first eligible model request.

        The managed model itself is sourced at run setup by
        [`get_model`][pydantic_ai_harness.logfire.AgentControl.get_model], so this hook only snapshots
        the code-side agent for auto-create and leaves the request untouched.
        """
        self._auto_create_snapshot(request_context)
        return request_context

    def _auto_create_snapshot(self, request_context: ModelRequestContext) -> None:
        """Create the code-side `AgentConfig` baseline at the first eligible model request.

        The request context carries the real tool definitions, instructions, and settings after the
        agent has assembled them. Auto-create only fires when the provider does not know the variable,
        so this effective request is the code baseline. The snapshot is built synchronously while the
        HTTP creation itself remains in the background.
        """
        resolved = self.resolved
        if resolved is None:
            return
        if not self._should_auto_create_for(resolved):
            return
        instructions = next(
            (
                message.instructions
                for message in reversed(request_context.messages)
                if isinstance(message, ModelRequest)
            ),
            None,
        )
        tool_definitions: dict[str, ToolDefinitionOverride] = {}
        for tool in request_context.model_request_parameters.function_tools:
            descriptions: dict[str, str] = {}
            properties = tool.parameters_json_schema.get('properties')
            if isinstance(properties, dict):
                typed_properties: dict[str, Any] = tool.parameters_json_schema['properties']
                for name, schema in typed_properties.items():
                    if isinstance(schema, dict):
                        parameter_schema: dict[str, Any] = typed_properties[name]
                        description = parameter_schema.get('description')
                        if isinstance(description, str):
                            descriptions[name] = description
            tool_definitions[tool.name] = ToolDefinitionOverride(
                description=tool.description or None, parameter_descriptions=descriptions or None
            )
        example = AgentConfig(
            instructions=instructions,
            model=f'{request_context.model.system}:{request_context.model.model_name}',
            settings=AgentConfigSettings.model_validate(request_context.model_settings)
            if request_context.model_settings
            else None,
            tool_definitions=tool_definitions or None,
        )
        config = self._variable.to_config().model_copy(
            update={'example': json.dumps(example.model_dump(exclude_none=True), indent=2)}
        )
        self._maybe_auto_create(self._variable, config)
