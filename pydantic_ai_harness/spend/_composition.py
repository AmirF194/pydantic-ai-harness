"""Report a capability arrangement that can bill a response the accrual never sees.

`SpendLimits` accrues inside its own `wrap_model_request`, so anything nested further in
can reject a response the counter has not recorded yet. Pydantic AI sorts the `innermost`
tier against everything else but not against itself, so the arrangement is reached by
listing capabilities in a particular order rather than by anything going wrong, and the
resulting chain is readable from
[`RunContext.root_capability`][pydantic_ai.tools.RunContext.root_capability].

What is read here is the ordering. Whether a nested wrapper actually rejects a billed
response depends on its own configuration and on how a given run races, none of which is
visible from the chain, so this reports an arrangement rather than an under-count that has
happened. That is also why it warns rather than refuses.
"""

from __future__ import annotations

import warnings

from pydantic_ai.capabilities import AbstractCapability, CombinedCapability, Hooks, WrapperCapability
from pydantic_ai.durable_exec._base import BaseDurabilityCapability  # pyright: ignore[reportPrivateUsage]
from pydantic_ai.tools import AgentDepsT

from pydantic_ai_harness.spend._exceptions import SpendCompositionWarning


def warn_about_inner_wrappers(
    root: AbstractCapability[AgentDepsT] | None,
    capability: AbstractCapability[AgentDepsT],
    reported: set[str],
) -> None:
    """Warn when a capability in `root` wraps inside `capability`'s `wrap_model_request`.

    `reported` accumulates the arrangements already warned about and is read and written here,
    so the same one reports once however many requests or runs it survives. Deduplicating on
    the arrangement rather than on the first call is what lets a reused agent be read again:
    `agent.run(capabilities=[...])` can put a different chain around the same capability
    instance on every run, and a flag set by the first, safe chain would hide every later one.
    """
    inner = _inner_wrappers(root, capability)
    if not inner:
        return
    listed = ', '.join(inner)
    if listed in reported:
        return
    reported.add(listed)
    name = type(capability).__name__
    warnings.warn(
        f'These capabilities are listed after `{name}`, so they wrap inside it: {listed}. '
        'If one of them rejects a response it has already awaited, the provider billed that '
        'response and the accrual never sees it. This reads the ordering, not what those '
        f'capabilities do with it. List `{name}` last among the innermost capabilities to rule it out.',
        SpendCompositionWarning,
        stacklevel=2,
    )


def _inner_wrappers(
    root: AbstractCapability[AgentDepsT] | None,
    capability: AbstractCapability[AgentDepsT],
) -> list[str]:
    """Names of the capabilities in `root` whose own `wrap_model_request` runs inside `capability`'s.

    A run carries its sorted chain as a `CombinedCapability`, which flattens nested ones, so
    position in that list is the whole answer: everything after `capability` nests inside it.
    Anything else -- no chain to read, or a `capability` reached through a wrapper rather than
    listed in the chain -- leaves nothing to compare against and reports nothing.
    """
    if not isinstance(root, CombinedCapability):
        return []
    chain: list[AbstractCapability[AgentDepsT]] = list(root.capabilities)
    for position, member in enumerate(chain):
        if member is capability:
            return [type(inner).__name__ for inner in chain[position + 1 :] if _may_reject_a_billed_response(inner)]
    return []


def _may_reject_a_billed_response(capability: AbstractCapability[AgentDepsT]) -> bool:
    """Whether nesting this capability inside the accrual is worth reporting.

    The question is whether it brings a `wrap_model_request` of its own that could await a
    response and then raise. Two kinds of capability define the method unconditionally, so
    defining it says nothing about them:

    - [`Hooks`][pydantic_ai.capabilities.Hooks] dispatches to whatever hook functions were
      registered, and the registry that would say whether one was is private. Core publishes
      `has_wrap_node_run` and `has_wrap_run_event_stream` but no equivalent for model requests;
      asked for in [pydantic-ai#7177](https://github.com/pydantic/pydantic-ai/issues/7177).
      Read as "no" until that lands.
    - [`WrapperCapability`][pydantic_ai.capabilities.WrapperCapability] delegates straight to
      `self.wrapped`, which is the capability that actually decides, so the question moves
      there. A wrapper over a real rejector is still reported, under the wrapper's own name,
      which is the name the reader put in the list.

    A subclass of either that overrides the method supplies its own and is answered on that.

    A durable-execution capability is read as "no" for a different reason: the signal is right,
    and the correction is the problem. Core requires the durable dispatch to be the innermost
    wrapper (`BaseDurabilityCapability.get_ordering`), so listing `SpendLimits` after it is the
    one thing a reader must not do, and the report would name an unavailable fix. That
    combination has its own, louder report: `SpendLimits` refuses the workflow clock and names
    <https://github.com/pydantic/pydantic-ai-harness/issues/531>. Matched by `isinstance`
    against the base the bundled Temporal, DBOS and Prefect integrations share, the same way
    `PlaywrightBrowser.for_agent` matches it, so both sites move together if core renames
    `pydantic_ai.durable_exec._base`. A public route is the fourth item on
    [pydantic-ai#7177](https://github.com/pydantic/pydantic-ai/issues/7177).

    Everything else is reported, including a capability that would not have rejected anything
    on the run in hand. `InputGuardrail` is the case in point: it can reach a billed response
    only under `parallel=True`, and that field is deliberately not read. `parallel` can be
    flipped without moving anything in the list, so the ordering is the durable property and
    the one the reader controls; and a capability that silenced its own report by reading a
    sibling's field would go quiet the day that field stopped meaning what it means here.
    """
    implementation = type(capability).wrap_model_request
    if isinstance(capability, WrapperCapability) and implementation is WrapperCapability.wrap_model_request:
        return _may_reject_a_billed_response(capability.wrapped)
    if isinstance(capability, BaseDurabilityCapability):
        return False
    return (
        implementation is not AbstractCapability.wrap_model_request and implementation is not Hooks.wrap_model_request
    )
