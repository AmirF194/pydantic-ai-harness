"""Shared base for capabilities backed by a Logfire managed variable.

`ManagedPrompt` and `ManagedAgent` both resolve a Logfire
[managed variable](https://logfire.pydantic.dev/docs/reference/advanced/managed-variables/) once per
run and keep its baggage active for the whole run. This base owns that shared plumbing -- the
targeting inputs, the per-run resolution context variable, `get_ordering`, and `wrap_run` -- so each
capability only declares its own variable and exposes the resolved value through its own surface.
"""

from __future__ import annotations

import re
import threading
import warnings
from collections.abc import Callable, Mapping
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, Generic

import logfire
from logfire.variables import Variable, VariableAlreadyExistsError
from logfire.variables.abstract import NoOpVariableProvider
from pydantic_ai.capabilities import AbstractCapability, CapabilityOrdering, Instrumentation
from pydantic_ai.exceptions import UserError
from pydantic_ai.tools import AgentDepsT, RunContext
from typing_extensions import TypeVar

if TYPE_CHECKING:
    from logfire import Logfire
    from logfire.variables import ResolvedVariable, VariableConfig
    from pydantic_ai.agent.abstract import AbstractAgent
    from pydantic_ai.capabilities.abstract import WrapRunHandler
    from pydantic_ai.run import AgentRunResult

# `AgentDepsT` carries a PEP 696 default, so `ValueT` needs one too to follow it in the type
# parameter list. Subclasses always bind it explicitly, so the default is never actually used.
ValueT = TypeVar('ValueT', default=object)


def resolution_reason(resolved: ResolvedVariable[Any]) -> str | None:
    """The reason a variable resolved the way it did (`'unrecognized_variable'`, `'code_default'`, ...).

    Reads the public `reason` attribute where the logfire SDK exposes it, falling back to the private
    `_reason` on older versions that predate it. Kept in one place so callers (and the demo) don't
    each reimplement the compatibility shim.
    """
    reason = getattr(resolved, 'reason', None)
    if reason is not None:
        return reason
    return getattr(resolved, '_reason', None)


# Names of variables we have already attempted to auto-create in this process, guarded by a lock.
# The contract is one attempt per process per name: we mark a name when spawning the creation thread
# (not on success), so a failed create -- e.g. a read-only token -- does not retry on every run.
_auto_create_attempted: set[str] = set()
_auto_create_lock = threading.Lock()

# Resolution reasons that mean "the value fell back to the code default because the provider had no
# value for it". logfire >= 4.37 collapses the "provider doesn't recognize this variable" case into
# the general `'code_default'` reason; older SDKs surface `'unrecognized_variable'` directly. We
# accept both and then confirm the actual cause against the provider (see `_maybe_auto_create_for`).
_CODE_DEFAULT_REASONS = frozenset({'code_default', 'unrecognized_variable'})


def _reset_auto_create_guard() -> None:  # pyright: ignore[reportUnusedFunction]
    """Clear the once-per-process auto-create guard. Intended for tests only."""
    with _auto_create_lock:
        _auto_create_attempted.clear()


def _normalize_agent_name(name: str) -> str:
    """Normalize a telemetry agent name exactly as the Logfire managed-agents UI does.

    The rule trims and lowercases the name, replaces hyphens and every other non-`[a-z0-9_]`
    character with `_`, collapses runs of underscores, and strips underscores from both ends. The
    SDK and UI must apply the same rule so they land on the same variable for a given agent name.
    """
    return re.sub(r'_+', '_', re.sub(r'[^a-z0-9_]', '_', name.strip().lower().replace('-', '_'))).strip('_')


def _spawn_create(variable: Variable[Any], config: VariableConfig | None = None) -> None:
    """Run the (blocking, sync-HTTP) creation off the run's thread so it never blocks or fails it.

    Isolated as a module-level function so tests can monkeypatch it to run `_create_variable`
    inline for determinism.
    """
    threading.Thread(target=_create_variable, args=(variable, config), daemon=True).start()


def _create_variable(variable: Variable[Any], config: VariableConfig | None = None) -> None:
    """Create the variable in Logfire from its code default, JSON schema, and description.

    Best-effort: an already-existing variable (a race with another process or the UI) is fine, and
    any other failure is surfaced once as a warning rather than crashing the background thread.
    """
    provider = variable.logfire_instance.config.get_variable_provider()
    # Duck-type the write path: a provider without it (or without persistence) can't be created into.
    create = getattr(provider, 'create_variable', None)
    if not callable(create):  # pragma: no cover
        return
    try:
        create(config or variable.to_config())
    except VariableAlreadyExistsError:
        # The variable already exists server-side (another process or the UI created it first).
        pass
    except Exception as exc:
        warnings.warn(f'Failed to auto-create Logfire managed variable {variable.name!r}: {exc}')


@dataclass(frozen=True)
class _DeferredVariable(Generic[ValueT]):
    """The inputs to build the backing variable lazily.

    Remembered by [`ManagedVariableCapability._setup_variable`][] when the capability's `name` was
    omitted, so the backing variable can be constructed as `<prefix><agent name>` from the running
    agent's own `name` on first run-time use (there is no agent at construction time on this
    pydantic-ai version).
    """

    prefix: str
    value_type: type[ValueT]
    default: ValueT


@dataclass
class ManagedVariableCapability(AbstractCapability[AgentDepsT], Generic[AgentDepsT, ValueT]):
    """Base for capabilities that resolve a Logfire managed variable once per run.

    Subclasses call `_setup_variable` from their `__post_init__` with their prefix, value type, and
    default. When an explicit `name` (or a pre-built [`Variable`][logfire.variables.Variable]) is
    given, the backing variable is built eagerly; when `name` is omitted, its construction is
    deferred to the first run-time use, where it is derived from the running agent's own `name` (see
    `_ensure_variable`). Either way, the capability exposes the active run's resolution through its
    own surface (instructions, a toolset wrapper, ...).
    """

    label: str | None = field(default=None, kw_only=True)
    """Explicit targeting label on the Logfire managed variable to resolve (e.g. `'production'`).
    When `None`, the targeting rules on the managed variable select the label."""

    targeting_key: str | Callable[[RunContext[AgentDepsT]], str | None] | None = field(default=None, kw_only=True)
    """Stable key that seeds Logfire's deterministic rollout assignment -- the same key always
    lands in the same percentage bucket. Accepts a static value or a callable that derives it from
    the [`RunContext`][pydantic_ai.tools.RunContext]. When `None`, Logfire falls back to its own
    targeting context and then the active trace id."""

    attributes: Mapping[str, Any] | Callable[[RunContext[AgentDepsT]], Mapping[str, Any] | None] | None = field(
        default=None, kw_only=True
    )
    """Attributes for condition-based targeting rules, or a callable that derives them
    from the [`RunContext`][pydantic_ai.tools.RunContext]."""

    logfire_instance: Logfire | None = field(default=None, kw_only=True)
    """Logfire instance to resolve the variable on. When `None`, the global default instance is
    used. Ignored when the capability is given a pre-built `Variable`."""

    auto_create: bool = field(default=True, kw_only=True)
    """Whether to create the variable in Logfire the first time it is used but doesn't exist there yet.

    When the variable is unknown to the configured Logfire provider, it is created in the background
    with the code default as its value (plus the payload's JSON schema and description), so the
    Logfire UI becomes the editing surface without a manual create-in-UI step. Until someone
    configures a label there, resolution keeps falling back to the code default. Creation happens
    off the run's thread and never blocks or fails the run; it is attempted at most once per process
    per variable. Set to `False` to opt out."""

    _variable: Variable[ValueT] = field(init=False, repr=False, compare=False)
    """The managed variable backing this capability. Assigned eagerly in `_setup_variable` for an
    explicit name/`Variable`; for a nameless capability it is left unset until `_ensure_variable`
    builds it from the agent's `name` on first run-time use (use `_built_variable` to read it safely
    before then)."""

    _deferred: _DeferredVariable[ValueT] | None = field(init=False, default=None, repr=False, compare=False)
    """The inputs to build `_variable` lazily, set only when `name` was omitted; `None` otherwise."""

    _build_lock: threading.Lock = field(init=False, default_factory=threading.Lock, repr=False, compare=False)
    """Guards the lazy build of `_variable` so concurrent first runs don't each construct one."""

    _resolved: ContextVar[ResolvedVariable[ValueT] | None] = field(init=False, repr=False, compare=False)
    """Per-run resolution, isolated across concurrent runs via the context variable."""

    _auto_create_in_wrap_run: ClassVar[bool] = True
    """Whether the base run wrapper triggers auto-create; subclasses with richer config set `False`."""

    def _new_resolved(self) -> ContextVar[ResolvedVariable[ValueT] | None]:
        """A fresh per-run resolution context variable; `None` means nothing is resolved yet."""
        return ContextVar('managed_variable_resolved', default=None)

    def _setup_variable(
        self, name: str | Variable[ValueT] | None, *, prefix: str, value_type: type[ValueT], default: ValueT
    ) -> None:
        """Wire up the backing variable from `name`, deferring construction when `name` was omitted.

        Called from each subclass's `__post_init__`. A str `name` builds `<prefix><name>` eagerly; a
        pre-built [`Variable`][logfire.variables.Variable] is used as-is (with full `get_model`
        support); `None` records the build inputs so the variable is derived from the running agent's
        own `name` on first run-time use (this pydantic-ai version has no construction-time agent hook).
        """
        self._resolved = self._new_resolved()
        if isinstance(name, str):
            self._variable = self._build_managed_variable(name, prefix=prefix, value_type=value_type, default=default)
        elif name is not None:
            self._warn_logfire_instance_ignored('name')
            self._variable = name
        else:
            # Nameless: leave `_variable` unset (there is no agent yet) and remember the build inputs,
            # so `_ensure_variable` can derive `<prefix><agent name>` on the first run.
            self._deferred = _DeferredVariable(prefix=prefix, value_type=value_type, default=default)

    @property
    def _name_omitted(self) -> bool:
        """Whether the capability was constructed without an explicit `name`.

        When it was, the backing variable is derived from the agent's own `name` at run time rather
        than built at construction.
        """
        return self._deferred is not None

    @property
    def _built_variable(self) -> Variable[ValueT] | None:
        """The backing variable if it has been built yet, else `None`.

        For a nameless capability `_variable` is unset until the first run builds it, so read it
        through this rather than touching `_variable` directly outside a run.
        """
        return getattr(self, '_variable', None)

    def _ensure_variable(self, ctx: RunContext[AgentDepsT]) -> Variable[ValueT]:
        """Return the backing variable, building it from the running agent's `name` on first use.

        Thin wrapper over [`_ensure_variable_for_agent`][] reading the agent off a `RunContext`. Model
        selection (which has a `ModelSelectionContext`, not a `RunContext`) calls the agent-based
        method directly, so both entry points build the same variable and share the cache.
        """
        return self._ensure_variable_for_agent(ctx.agent)

    def _ensure_variable_for_agent(self, agent: AbstractAgent[AgentDepsT, Any] | None) -> Variable[ValueT]:
        """Return the backing variable, building it from the agent's `name` on first use.

        For an eagerly-built variable (an explicit `name` or `Variable` was given) this just returns
        it. For a nameless capability it derives `<prefix><agent name>` from `agent.name` once,
        caching the result on the capability so later runs and the other run-time surfaces reuse it.
        Takes the agent directly (rather than a `RunContext`) so both the run-time hooks and model
        selection -- which sees a narrower `ModelSelectionContext` before any `RunContext` exists --
        derive the same variable. Raises [`UserError`][pydantic_ai.exceptions.UserError] when there
        is no agent name to derive from -- a nameless managed capability requires the agent to have a
        `name`.
        """
        variable = self._built_variable
        if variable is not None:
            return variable
        deferred = self._deferred
        assert deferred is not None  # `_variable` is unset only in the nameless (deferred) case.
        with self._build_lock:
            # A concurrent first run may have built the variable while we waited for the lock.
            variable = self._built_variable
            if variable is not None:
                return variable
            agent_name = agent.name if agent is not None else None
            if not agent_name:
                raise UserError(
                    'A managed capability without an explicit `name` derives its backing variable from '
                    "the agent's `name`, but this agent has none. Give the agent a `name=...`, or pass an "
                    'explicit `name` to the capability.'
                )
            variable = self._build_managed_variable(
                _normalize_agent_name(agent_name),
                prefix=deferred.prefix,
                value_type=deferred.value_type,
                default=deferred.default,
            )
            self._variable = variable
            return variable

    def _warn_logfire_instance_ignored(self, field_name: str) -> None:
        if self.logfire_instance is not None:
            warnings.warn(
                f'`logfire_instance` is ignored when `{field_name}` is a `Variable`; '
                'the variable already carries its own Logfire instance.',
                # 1=warn, 2=_warn_logfire_instance_ignored, 3=__post_init__,
                # 4=dataclass-generated __init__, 5=user's `ManagedTool(...)` call.
                stacklevel=4,
            )

    def _build_managed_variable(
        self, name: str, *, prefix: str, value_type: type[ValueT], default: ValueT
    ) -> Variable[ValueT]:
        """Declare the backing variable as `<prefix><name>`, normalizing and validating the name."""
        # Strip the prefix if the user accidentally passed it so we can still apply
        # hyphen-to-underscore normalization, then re-add the prefix below.
        if name.startswith(prefix):
            warnings.warn(
                f'The {prefix!r} prefix is added automatically; pass the bare name rather than {name!r}.',
                # Same chain as `_warn_logfire_instance_ignored`: helper → __post_init__ → dataclass __init__ → user.
                stacklevel=4,
            )
            name = name[len(prefix) :]

        variable_name = f'{prefix}{name.replace("-", "_")}'
        if not variable_name.isidentifier():
            raise ValueError(
                f'Name {name!r} produces an invalid variable name {variable_name!r}; '
                'names may only contain letters, digits, hyphens, and underscores.'
            )

        # Construct the variable directly (rather than via `logfire.var`) so redeclaring the
        # same name is idempotent: `logfire.var` registers in a per-instance registry and raises
        # on a duplicate name, which would break sharing one variable across agents.
        instance = self.logfire_instance if self.logfire_instance is not None else logfire.DEFAULT_LOGFIRE_INSTANCE
        return Variable(variable_name, type=value_type, default=default, logfire_instance=instance)

    @property
    def resolved(self) -> ResolvedVariable[ValueT] | None:
        """The resolution for the active run, or `None` outside a run.

        Exposes the full [`ResolvedVariable`][logfire.variables.ResolvedVariable] (`value`, `label`,
        `version`, `reason`, ...) so callers can inspect which version is in play.
        """
        return self._resolved.get()

    def get_ordering(self) -> CapabilityOrdering:
        """Run outermost so the resolution's baggage envelops the whole run, including the run span."""
        return CapabilityOrdering(position='outermost', wraps=[Instrumentation])

    def _maybe_auto_create(self, variable: Variable[Any], config: VariableConfig | None = None) -> None:
        """Kick off background creation of the backing variable, at most once per process per name."""
        name = variable.name
        with _auto_create_lock:
            if name in _auto_create_attempted:
                return
            # Mark before spawning: one attempt per process, so a failed create doesn't retry.
            _auto_create_attempted.add(name)
        _spawn_create(variable, config)

    def _should_auto_create_for(self, resolved: ResolvedVariable[ValueT]) -> bool:
        """Whether a configured provider does not recognize this variable and creation is still eligible.

        Auto-create is for exactly one case: a provider is configured but has no entry for this name,
        so resolution fell back to the code default. logfire >= 4.37 reports that as `'code_default'`
        (older SDKs as `'unrecognized_variable'`), but `'code_default'` also covers "no provider
        configured" and "known variable with no targeted value" -- neither of which should create
        anything. The once-per-process guard is only peeked at here; `_maybe_auto_create` re-checks
        and marks it under the same lock, making concurrent callers race-safe.
        """
        if not self.auto_create or resolution_reason(resolved) not in _CODE_DEFAULT_REASONS:
            return False
        variable = self._variable
        with _auto_create_lock:
            if variable.name in _auto_create_attempted:
                return False
        provider = variable.logfire_instance.config.get_variable_provider()
        if isinstance(provider, NoOpVariableProvider):
            return False
        return provider.get_variable_config(variable.name) is None

    def _maybe_auto_create_for(self, resolved: ResolvedVariable[ValueT], config: VariableConfig | None = None) -> None:
        """Trigger background auto-create when a configured provider doesn't recognize the variable yet.

        Auto-create is for exactly one case: a provider is configured but has no entry for this name,
        so resolution fell back to the code default. logfire >= 4.37 reports that as `'code_default'`
        (older SDKs as `'unrecognized_variable'`), but `'code_default'` also covers "no provider
        configured" and "known variable with no targeted value" -- neither of which should create
        anything. So we confirm against the provider itself: it must be a real (non-`NoOp`) provider
        that has no config for this name. A `resolved`/`context_override` value isn't a candidate at
        all, and is filtered out by the reason check up front.
        """
        if self._should_auto_create_for(resolved):
            self._maybe_auto_create(self._variable, config)

    def _resolve(self, ctx: RunContext[AgentDepsT]) -> ResolvedVariable[ValueT]:
        """Resolve the backing variable for this run using the capability's targeting inputs.

        Shared by `wrap_run` (the base's per-run resolution point) and subclasses that must resolve
        earlier -- e.g. in `for_run`, where the resolved value drives what the run is assembled from.
        Builds the backing variable from the agent's `name` first when the capability is nameless.
        """
        variable = self._ensure_variable(ctx)

        if callable(self.targeting_key):
            targeting_key = self.targeting_key(ctx)
        else:
            targeting_key = self.targeting_key

        if callable(self.attributes):
            attributes = self.attributes(ctx)
        else:
            attributes = self.attributes

        return variable.get(targeting_key=targeting_key, attributes=attributes, label=self.label)

    async def wrap_run(self, ctx: RunContext[AgentDepsT], *, handler: WrapRunHandler) -> AgentRunResult[Any]:
        """Resolve the variable once and keep its baggage active for the duration of the run."""
        resolved = self._resolve(ctx)
        if self._auto_create_in_wrap_run:
            self._maybe_auto_create_for(resolved)
        with resolved:
            token = self._resolved.set(resolved)
            try:
                return await handler()
            finally:
                self._resolved.reset(token)
