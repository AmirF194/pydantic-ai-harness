---
title: Ten agents, ten clouds, one answer
description: Give each agent in a fan-out its own disposable cloud, and a team of them can build and test many designs at once. It's the experiment real AWS can't run at any price. The pattern scales to ten; the demo runs three.
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
in an account something real depends on. The agent runs at agent speed; the cloud
answers at cloud speed, and charges for the conversation.

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

It works. The agent creates the bucket, writes the function, wires the trigger. Then
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

## The agent isn't the problem. The cloud is.

So don't change the agent. Change what it deploys against.

The [LocalStack capability](https://github.com/pydantic/pydantic-ai-harness/tree/main/pydantic_ai_harness/localstack/)
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
the end. The agent's `aws` commands don't change. What changes is everything the
account made expensive:

- The four-minute deploy takes seconds. There's no remote control plane to wait on.
- The bill is zero. Nothing is provisioned anywhere you pay for.
- The blast radius is nothing. There's no production, only a container you throw away.

And because each run gets its own container, "reset to before the agent touched it"
isn't an afternoon of cleanup. It's the next run. Nothing persists unless you ask it
to.

## The part that isn't just faster

A throwaway cloud does more than speed the agent up. It removes a constraint you
didn't notice you were obeying.

You would never point five agents at one AWS account at once. Five agents, one
production, shared state, five times the bill: obviously not. But a cloud that costs
nothing to run and starts clean every time takes that off the table. So don't give one
agent one cloud. Give each agent its own.

Say you want to choose a rate-limiter design, and you have three candidates: a DynamoDB
token bucket, API Gateway usage plans, a Redis sliding window. Instead of asking one
agent to reason about all three on paper, hand each to its own engineer with its own
LocalStack, and let each build its design, deploy it, and exercise it against an
identical clean cloud. Then compare the three that ran, on what happened rather than
what was promised.

On real AWS this isn't expensive. It's impossible. Here each engineer is an agent with
its own container:

```python
from pydantic_ai import Agent
from pydantic_ai_harness.localstack import LocalStack
from pydantic_ai_harness.dynamic_workflow import DynamicWorkflow

designs = {
    'dynamo': 'DynamoDB on-demand + a token bucket in the handler',
    'apigw': 'API Gateway usage plans + API keys',
    'redis': 'ElastiCache (Redis) sliding-window counter',
}

# One engineer per design, each with its own throwaway cloud on its own port.
engineers = [
    Agent(
        'anthropic:claude-sonnet-4-6',
        name=f'build_{key}',
        description=f'Builds and load-tests: {spec}',
        instructions=(
            'Deploy your design to your LocalStack with the AWS CLI, exercise it under '
            'load, and report throughput, errors, and the IAM it needs.'
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
    'anthropic:claude-opus-4-7',
    instructions='Have each design built and tested in parallel, then recommend the one the evidence supports.',
    capabilities=[DynamicWorkflow(agents=[*engineers, judge], max_agent_calls=12)],
)
```

The distinct ports aren't incidental. They're the idea made literal: each engineer gets
its own cloud, so three of them run at once without stepping on each other.

Given the task, the architect doesn't make a dozen tool calls and narrate the results
back to itself. `DynamicWorkflow` hands it one tool, `run_workflow`, and it writes a
script:

```python
import asyncio

prompt = 'Build and load-test your design. Report throughput, errors, and the IAM it needs.'

reports = await asyncio.gather(
    build_dynamo(task=prompt),
    build_apigw(task=prompt),
    build_redis(task=prompt),
)

await judge(task='Recommend one rate-limiter design, citing the load and IAM evidence:\n\n'
                 + '\n\n---\n\n'.join(reports))
```

Three real deployments, built and load-tested in parallel, each on its own cloud, judged
on what held up. The whole tree runs inside one `run_workflow` call. The architect's
context never fills with deploy logs; only the recommendation comes back. The
choreography moved out of the conversation and into code, and the cloud it choreographs
is one you can afford to run three of.

Cheap to run isn't free. `max_agent_calls` caps the number of sub-agent runs exactly,
even under fan-out, so a workflow that decides to explore a dozen designs instead of
three stops at the ceiling you set. The disposable cloud removes the cloud bill; the
budget keeps the model bill honest.

## Where this goes

There's a version of this that outlives one run. A migration that takes an afternoon,
moving every service to least-privilege IAM one at a time, wants to keep its progress
when the process dies and pick up where it left off. Pydantic AI's durable execution and
`DynamicWorkflow`'s durable workflows are heading there. No harness ships that end to end
yet.

For now the smaller thing is enough, and it isn't small. The agent can be wrong. It can
drop the table, over-scope the IAM, ship the design that folds under load, and find out
in seconds, for free, on a cloud that starts clean the next time you run it.

"You're absolutely right" stops being the sentence you brace for. It becomes a
hypothesis you can afford to test.
