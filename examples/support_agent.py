"""A customer support triage agent with guardrails and per-customer memory.

`InputGuard` rejects prompt injection before any tokens are spent, `OutputGuard`
sends policy-violating replies back to the model to fix, and `Memory` gives each
customer a persistent notebook so the next conversation starts informed. The
reply is a typed `TriageDecision`, ready for a ticketing system.

    uv run examples/support_agent.py customer-42 "The export button times out on big projects"
"""

import os
import re
import sys
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models import Model

from pydantic_ai_harness import GuardResult, InputGuard, OutputGuard
from pydantic_ai_harness.memory import FileStore, Memory

DEFAULT_MODEL = os.environ.get('PYDANTIC_AI_MODEL', 'anthropic:claude-opus-4-7')


# Phrases that read as attempts to override the agent's instructions. A regex
# is a demo seam, not a security control: a real deployment would put a
# classifier or moderation-API call here instead (guards may be async).
_INJECTION_PATTERNS = re.compile(
    r'(ignore|disregard|forget) (all|your|previous|prior) (instructions|directions|rules)'
    r'|reveal .{0,30}(system )?prompt'
    r'|you are now',
    re.IGNORECASE,
)
# The agent must never promise money on its own authority. Deliberately coarse:
# it also bounces refusals like "I cannot issue a refund", which costs one
# retry and still converges on an escalation without the loaded word.
_FORBIDDEN_PROMISES = re.compile(r'refund|chargeback|compensat|free month', re.IGNORECASE)


class TriageDecision(BaseModel):
    """The structured outcome of one support message."""

    category: Literal['bug', 'billing', 'how_to', 'feature_request', 'abuse']
    urgency: Literal['low', 'normal', 'high']
    reply: str = Field(description='The message to send to the customer.')
    escalate: bool = Field(description='True when a human must take over.')
    escalation_reason: str = Field(default='', description='Why a human is needed, when escalate is true.')


def reject_injection(prompt: str) -> GuardResult:
    """Refuse messages that try to rewrite the agent's instructions, spending no tokens."""
    if _INJECTION_PATTERNS.search(prompt):
        return GuardResult.block("I can only help with product support questions. Let's stick to your issue.")
    return GuardResult.allow()


def enforce_policy(output: object) -> GuardResult:
    """Send policy-violating replies back to the model to fix, rather than failing the run."""
    if isinstance(output, TriageDecision) and _FORBIDDEN_PROMISES.search(output.reply):
        return GuardResult.retry(
            'Your reply promises money or credit, which only a human can authorize. '
            'Rewrite the reply without financial promises and set escalate=true with the reason.'
        )
    return GuardResult.allow()


INSTRUCTIONS = """\
You are the first-line support agent for a developer tools product.

- Triage each message: category, urgency, and whether a human must take over.
- Billing disputes, legal threats, security reports, and anything involving money
  always escalate.
- Record durable customer facts in memory (their plan, stack, recurring issues) so
  the next conversation starts informed. Do not store message transcripts.
- Be direct and specific in replies. If you need information to proceed, ask for
  exactly what you need.
"""


def build_agent(model: Model | str = DEFAULT_MODEL, customer_id: str = 'anonymous') -> Agent[None, TriageDecision]:
    """Build the support agent with memory namespaced to `customer_id`."""
    return Agent(
        model,
        capabilities=[
            InputGuard(guard=reject_injection),
            OutputGuard(guard=enforce_policy),
            # One notebook per customer, persisted on disk, injected into the prompt.
            # The env var is read here (not at import) so callers can redirect it.
            Memory(
                store=FileStore(Path(os.environ.get('SUPPORT_MEMORY_DIR', '.support-memory'))),
                namespace=customer_id,
            ),
        ],
        instructions=INSTRUCTIONS,
        output_type=TriageDecision,
    )


def main() -> None:
    """Triage one customer message from the command line."""
    customer_id = sys.argv[1] if len(sys.argv) > 1 else 'customer-42'
    message = ' '.join(sys.argv[2:]) or 'The export button times out on big projects.'
    decision = build_agent(customer_id=customer_id).run_sync(message).output
    print(f'[{decision.category}/{decision.urgency}] escalate={decision.escalate}')
    print(decision.reply)


if __name__ == '__main__':
    main()
