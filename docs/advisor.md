---
title: Advisor
description: Let an executor model consult a separate advisor model through a provider-native tool or a local Pydantic AI fallback.
---

# Advisor

Give an executor model a way to consult a separate advisor model before it answers or commits to a decision.

[Source](https://github.com/pydantic/pydantic-ai-harness/tree/main/pydantic_ai_harness/advisor/)

> The API may change between releases. Breaking changes ship deprecation warnings where practical.

## Usage

Pass the advisor model as the first argument. The model can be any model name or model instance accepted by Pydantic AI:

```python
from pydantic_ai import Agent
from pydantic_ai_harness.advisor import Advisor

agent = Agent(
    'openai:gpt-5.4',
    capabilities=[
        Advisor(
            'anthropic:claude-opus-4-8',
            max_uses=1,
            max_tokens=4096,
        )
    ],
)

result = agent.run_sync(
    'Design a zero-downtime database migration. Consult the advisor before choosing a plan.'
)
print(result.output)
```

The executor decides when to consult. Ask it explicitly in the user prompt or the agent's instructions when a consultation is required.

## Provider adaptation

`Advisor` exposes one logical tool through two execution paths:

- **Native:** when the executor and advisor are both on a compatible Anthropic provider, or both use OpenRouter, Pydantic AI's provider-native [`AdvisorTool`](/ai/tools-toolsets/native-tools/#advisor-tool) runs the consultation.
- **Local fallback:** every other pairing gets an `advisor` function tool. Calling it runs a separate Pydantic AI agent with the configured advisor model.

In the default `auto` mode, native selection is conservative. The capability only reuses an explicit provider-qualified model name when the executor and advisor share a provider, so it does not guess how an Anthropic model ID maps to an OpenRouter catalog slug. For example:

```python
from pydantic_ai_harness.advisor import Advisor

# Native for an Anthropic executor; local for OpenAI, Google, and other executors.
anthropic_advisor = Advisor('anthropic:claude-opus-4-8')

# Native for an OpenRouter executor.
openrouter_advisor = Advisor('openrouter:anthropic/claude-opus-4.8')
```

Passing a `Model` instance selects local execution in `auto` mode. This preserves that instance's provider, client, credentials, base URL, and instrumentation. String model names are resolved only if local execution is selected, so a native consultation uses the executor provider's existing configuration.

Pydantic AI's resolved executor model profile makes the final support decision. A client or model that does not support the native tool uses the local fallback.

## Options

| Option | Default | Behavior |
|---|---:|---|
| `model` | required | Advisor model name or `Model` instance. |
| `mode` | `'auto'` | Execution policy: `'auto'`, `'native'`, or `'local'`. |
| `max_uses` | `None` | Maximum consultations in one executor model request. Must be at least `1`. |
| `max_tokens` | `None` | Maximum output tokens for each consultation. Must be at least `1024`. |
| `caching` | `None` | Anthropic-native prompt-cache TTL: `'5m'` or `'1h'`. |

Use `mode='native'` when the consultation must stay inside the executor provider, or `mode='local'` when the configured advisor provider must receive a separate request. Native mode requires an `anthropic:<model>` or `openrouter:<model>` string and an executor on that same provider. It does not fall back when the executor lacks support.

`max_uses` has the same per-request scope as Anthropic's native tool. Only calls whose arguments validate consume this allowance. It resets when the executor makes its next model request. OpenRouter ignores native `max_uses`, so `auto` mode selects the local fallback when this option is set. Combining OpenRouter, `mode='native'`, and `max_uses` is rejected.

`caching` is an opportunistic Anthropic-native optimization. OpenRouter and the local fallback have no equivalent control.

## Context passed to the advisor

The context depends on the execution path:

| Path | Advisor context |
|---|---|
| Anthropic native | The provider supplies the full transcript, including system instructions, tool definitions, earlier turns and results, and executor text produced so far. |
| OpenRouter native | The executor supplies a consultation prompt. Pydantic AI configures `forward_transcript=false`. |
| Local fallback | The executor supplies a consultation prompt through the `advisor` function tool. |

For portable behavior, tell the executor to put the question and all relevant evidence in its consultation prompt. The local tool description reinforces this requirement.

The local fallback sends that prompt to the configured advisor model and provider. Native execution uses the executor's provider configuration. Treat this distinction as a data-routing choice when reviewing credentials, transcript sharing, and provider policies.

## Usage, failures, and observability

Local advisor requests share the parent run's [`RunUsage`](/ai/api/pydantic-ai/usage/#pydantic_ai.usage.RunUsage) and [`UsageLimits`](/ai/api/pydantic-ai/usage/#pydantic_ai.usage.UsageLimits), so their requests and tokens count toward the agent tree's normal limits. Native providers report advisor usage according to their own protocol. Anthropic records advisor-specific values in `RequestUsage.details`, while OpenRouter exposes aggregate server-tool counts in response provider details.

Invalid option combinations fail when `Advisor` is constructed. Executor and provider compatibility is validated when a run prepares its model request. Local model resolution, authentication, provider, and request errors propagate through the normal Pydantic AI tool execution path. When a local call exceeds `max_uses`, the tool returns a bounded message telling the executor to continue without more advice, matching the native tool's non-fatal limit behavior.

## Composition

The capability needs no ordering constraint. It composes with other capabilities and ordinary toolsets through Pydantic AI's [native-or-local tool selection](/ai/capabilities/overview/#provider-adaptive-tools).

The advisor tool is always visible and is not deferred through [Tool Search](/ai/tools-toolsets/tools-advanced/#tool-search). It reserves the tool name and toolset ID `advisor`. One `Advisor` instance is supported per agent because the native tool has one stable identity.

During streaming, the executor stream pauses while an advisor consultation runs and resumes when the completed advice is available. The local fallback does not splice the advisor model's token deltas into the executor stream.

Native advice is compatible with durable execution because it remains part of the executor model request. Local execution cannot yet preserve the same semantics across every durable backend. Temporal and Prefect can checkpoint the returned advice, but changes to the activity-local or task-local [`RunUsage`](/ai/api/pydantic-ai/usage/#pydantic_ai.usage.RunUsage) do not merge back into the outer run. DBOS does not checkpoint ordinary function-tool calls, so a local advisor request could run again during workflow replay.

When using the `TemporalDurability`, `DBOSDurability`, or `PrefectDurability` capability, modes `auto` and `local` therefore reject runs inside an active durable workflow or flow. The same configured agent still works normally outside that durable container. Use `mode='native'` with a supported provider when running the agent durably.

The deprecated `TemporalAgent`, `DBOSAgent`, and `PrefectAgent` wrappers do not expose their active durable state to capability hooks. Local-capable Advisor modes are not supported with those wrappers. Migrate to the corresponding durability capability, or use `mode='native'`.

## API reference

::: pydantic_ai_harness.advisor.Advisor
