"""Report a capability arrangement that can bill a response the accrual never sees.

`SpendLimits` accrues inside its own `wrap_model_request`, so anything nested further in
can reject a response the counter has not recorded yet. Pydantic AI sorts the `innermost`
tier against everything else but not against itself, so the arrangement is reached by
listing capabilities in a particular order rather than by anything going wrong, and the
resulting chain is readable from
[`RunContext.root_capability`][pydantic_ai.tools.RunContext.root_capability].

Reported rather than refused: the under-count needs the inner wrapper to reject a response
it has already awaited, which a guard that wins its race against the provider never does.
"""

from __future__ import annotations

import warnings
from typing import Any

from pydantic_ai.capabilities import AbstractCapability, CombinedCapability, Hooks

from pydantic_ai_harness.spend._exceptions import SpendCompositionWarning


def warn_about_inner_wrappers(root: AbstractCapability[Any] | None, capability: AbstractCapability[Any]) -> None:
    """Warn when a capability in `root` wraps inside `capability`'s `wrap_model_request`.

    Call this once per capability instance rather than once per request: the chain is
    fixed for the run, and a per-request warning would report the same arrangement as
    many times as the agent happens to call the model.
    """
    inner = _inner_wrappers(root, capability)
    if not inner:
        return
    name = type(capability).__name__
    warnings.warn(
        f'These capabilities are listed after `{name}`, so they wrap inside it: {", ".join(inner)}. '
        'A response one of them rejects after awaiting it is billed by the provider and never counted. '
        f'List `{name}` last among the innermost capabilities to close that.',
        SpendCompositionWarning,
        stacklevel=2,
    )


def _inner_wrappers(root: AbstractCapability[Any] | None, capability: AbstractCapability[Any]) -> list[str]:
    """Names of the capabilities in `root` whose own `wrap_model_request` runs inside `capability`'s.

    A run carries its sorted chain as a `CombinedCapability`, which flattens nested ones, so
    position in that list is the whole answer: everything after `capability` nests inside it.
    Anything else -- no chain to read, or a `capability` reached through a wrapper rather than
    listed in the chain -- leaves nothing to compare against and reports nothing.
    """
    if not isinstance(root, CombinedCapability):
        return []
    chain = list(root.capabilities)
    for position, member in enumerate(chain):
        if member is capability:
            return [type(inner).__name__ for inner in chain[position + 1 :] if _may_reject_a_billed_response(inner)]
    return []


def _may_reject_a_billed_response(capability: AbstractCapability[Any]) -> bool:
    """Whether nesting this capability inside the accrual is worth reporting.

    The question is whether it brings a `wrap_model_request` that can await a response and
    then raise. Two cases are read as "no", both because the report would land on an
    arrangement the reader cannot correct:

    - [`Hooks`][pydantic_ai.capabilities.Hooks] defines the method unconditionally and
      dispatches to whatever hook functions were registered, so the definition says nothing
      about whether one was, and the registry that would say is private. Core publishes
      `has_wrap_node_run` and `has_wrap_run_event_stream` but no equivalent for model
      requests; asked for in
      [pydantic-ai#7165](https://github.com/pydantic/pydantic-ai/issues/7165). A subclass
      that overrides the method supplies its own and is still reported.
    - A durable-execution capability, identified by the `engine_name` its base declares,
      routes the request into an activity/step/task rather than rejecting what comes back,
      and core requires its dispatch to be the innermost wrapper. Reordering is the one
      thing a reader must not do here, so there would be nothing to act on. That
      combination has its own, louder report: `SpendLimits` refuses the workflow clock and
      names <https://github.com/pydantic/pydantic-ai-harness/issues/531>.

    A missed report is preferred over one on a correct arrangement, which the reader could
    only silence by changing correct code.
    """
    if hasattr(type(capability), 'engine_name'):
        return False
    implementation = type(capability).wrap_model_request
    return (
        implementation is not AbstractCapability.wrap_model_request and implementation is not Hooks.wrap_model_request
    )
