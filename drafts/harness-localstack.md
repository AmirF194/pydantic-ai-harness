---
title: Ten agents, ten clouds, one answer
description: Give each agent in a fan-out its own disposable cloud, and a team of them can build and test many designs at once. It's the experiment you'd never run on real AWS. The pattern scales to ten; the demo runs three.
draft: true
series: Pydantic AI Harness
---

> [!NOTE]
> Part of Harness week. Earlier: [You've built this agent before](https://pydantic.dev/articles/harness-week),
> [When agents build agents](https://pydantic.dev/articles/when-agents-build-agents),
> [The agent writes faster than you can review](https://pydantic.dev/articles/harness-macroscope).

"You're absolutely right. I'll drop the old table and redeploy." And it can. It has
credentials, it's fast, and it's confident in the way that reads right, up until
CloudFormation spends four minutes half-applying a stack it then rolls back. Now you're
reading the events tab at human speed, in an account with a real bill, to find out what
your absolutely-right agent just did.

Give an agent a real cloud task, like "add a rate limiter to the checkout API," and it
writes clean CDK in seconds. Then it stops being fast. Every deploy waits minutes on a
control plane in another region. Every experiment is a line item. Every mistake lands
in an account something real depends on, and the cloud charges for the conversation.

Wiring it up takes almost nothing. `Shell` gives the agent a terminal, your AWS
credentials are already in the environment, and `aws` is one command away:

```python
from pydantic_ai import Agent
from pydantic_ai_harness import Shell

agent = Agent(
    'anthropic:claude-sonnet-4-6',
    capabilities=[Shell()],
)

result = agent.run_sync('Deploy the checkout stack and smoke-test the endpoint.')
```

It works. The agent creates the bucket, ships the function, wires the trigger. Then
you remember it's doing all of that in a real account.

## The cloud punishes what the agent is good at

So you put a human in front of it. You approve each deploy and read each plan, which
makes you the slowest component in the system again. The agent could try ten designs;
you let it try one, because ten costs ten times as much, takes ten times as long, and
leaves ten times the mess. It could fail fast and learn; you can't let it fail, because
failure here has a blast radius.

Speed, fearlessness, a willingness to try the wrong thing on the way to the right one:
that is what makes an agent good at this, and it is everything a real cloud account is
built to punish. You supervise it not because it's bad at the task but because the
environment is unforgiving, it's your name on the account, and your credit card pays
the invoice.

## Change what it deploys against

The agent isn't the problem. The cloud is. The [LocalStack capability](https://github.com/pydantic/pydantic-ai-harness/tree/main/pydantic_ai_harness/localstack/)
gives an agent an emulated AWS: the real service APIs, running in a container. It
injects the endpoint and credentials, so the agent just issues plain `aws` commands.
One line puts it in reach:

```python
from pydantic_ai import Agent
from pydantic_ai_harness.localstack import LocalStack

agent = Agent(
    'anthropic:claude-sonnet-4-6',
    capabilities=[LocalStack(manage_container=True)],
)
```

`manage_container=True` starts a fresh LocalStack container for the run and stops it at
the end. (It needs a free LocalStack auth token in `LOCALSTACK_AUTH_TOKEN`, or an older
tokenless `image=`.) The agent's `aws` commands don't change. What changes is everything the
account made expensive:

- The four-minute deploy takes seconds. There's no remote control plane to wait on.
- The bill is zero. Nothing is provisioned anywhere you pay for.
- The blast radius is nothing. There's no production, only a container you throw away.

And because each run gets its own container, "reset to before the agent touched it" is
the next run, not an afternoon of cleanup. Nothing carries over between runs.

## A cloud each

A throwaway cloud does more than speed the agent up. It removes a constraint you
didn't notice you were obeying.

You would never point ten agents at one AWS account at once. Ten agents, one
production, shared state, ten times the bill: obviously not. But a cloud that costs
nothing to run and starts clean every time takes that off the table. So don't give one
agent one cloud. Give each agent its own.

Say you want to choose a rate-limiter design, and you have three candidates: a
fixed-window counter, a sliding-window log, a token bucket, each backed by DynamoDB.
Instead of asking one agent to reason about all three on paper, hand each to its own
engineer with its own LocalStack, and let each build its design, deploy it, and drive
requests through it until it should start rejecting. Then compare the three that ran, on
what happened rather than what was promised.

On real AWS you could run this. You never would. Here each engineer is an agent with
its own container:

```python
from pydantic_ai import Agent
from pydantic_ai_harness.localstack import LocalStack
from pydantic_ai_harness.dynamic_workflow import DynamicWorkflow

designs = {
    'fixed': 'DynamoDB fixed-window counter (conditional increment, reset each window)',
    'sliding': 'DynamoDB sliding-window log (timestamped items with TTL, count the recent ones)',
    'bucket': 'DynamoDB token bucket (conditional decrement, refill over time)',
}

# One engineer per design, each with its own throwaway cloud on its own port.
engineers = [
    Agent(
        'anthropic:claude-sonnet-4-6',
        name=f'build_{key}',
        description=f'Builds and tests one design: {spec}',
        instructions=(
            'Deploy your design to your LocalStack with the AWS CLI, send requests through '
            'it past the limit, and report whether it actually starts rejecting and the '
            'DynamoDB calls it takes to decide.'
        ),
        capabilities=[
            LocalStack(manage_container=True, endpoint_url=f'http://localhost.localstack.cloud:{port}'),
        ],
    )
    for port, (key, spec) in zip((4566, 4576, 4586), designs.items())
]

judge = Agent(
    'anthropic:claude-sonnet-4-6',
    name='judge',
    description='Compares the tested designs and names a winner.',
)

architect = Agent(
    'anthropic:claude-opus-4-8',
    instructions='Have each design built and tested in parallel, then recommend the one the evidence supports.',
    capabilities=[DynamicWorkflow(agents=[*engineers, judge], max_agent_calls=20)],
)
```

The distinct ports aren't incidental. Each engineer gets its own container, so three of
them run at once without stepping on each other.

Given the task, the architect doesn't make a dozen tool calls and narrate the results
back to itself. `DynamicWorkflow` hands it one tool, `run_workflow`, and it writes a
script:

```python
import asyncio

prompt = 'Deploy your design, send requests past the limit, and report whether it actually starts rejecting.'

reports = await asyncio.gather(
    build_fixed(task=prompt),
    build_sliding(task=prompt),
    build_bucket(task=prompt),
)

await judge(task='Recommend one rate-limiter design, citing which actually held the limit:\n\n'
                 + '\n\n---\n\n'.join(reports))
```

Three real deployments, built and pushed past their limits in parallel, each on its own
cloud, judged on what actually held. The whole tree runs inside one `run_workflow` call,
so only the recommendation reaches the architect, not the deploy logs. The choreography
moved out of the conversation and into code, and the cloud it runs on is one you can
afford three of.

The disposable cloud removes the cloud bill, but the model still costs tokens.
`max_agent_calls` caps the number of sub-agent runs exactly, even under fan-out, so a
workflow that explores a dozen designs instead of three stops at the ceiling you set.

## Where this goes

There's a version of this that outlives one run: a migration that saves its progress
between steps and resumes after a crash. Pydantic AI's durable execution and
`DynamicWorkflow`'s durable workflows are heading there; no harness ships it end to end
yet.

For now the smaller thing is enough. The agent can be wrong. It can drop the wrong
table, ship a limiter that never rejects, break the deploy outright, and find out in
seconds, for free, on a cloud that starts clean the next time you run it.

"You're absolutely right" stops being the sentence you brace for. It becomes a
hypothesis you can afford to test.
