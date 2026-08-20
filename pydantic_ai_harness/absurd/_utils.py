from __future__ import annotations

from absurd_sdk import AsyncTaskContext, get_current_context
from pydantic_ai.exceptions import UserError


def current_async_context() -> AsyncTaskContext | None:
    """Return the current Absurd async task context, or ``None`` outside a task."""
    ctx = get_current_context()
    if ctx is None:
        return None
    if isinstance(ctx, AsyncTaskContext):
        return ctx
    raise UserError('Absurd durability requires an async Absurd task context; got a sync one.')
