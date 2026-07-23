---
title: You're absolutely right, my agent has access to AWS
description: An agent is fast, fearless, and willing to be wrong on the way to being right. A real cloud account punishes all three. Give it a disposable one instead -- and a superpower falls out.
draft: true
series: Pydantic AI Harness
---

> [!NOTE]
> Part of Harness week. Earlier: [You've built this agent before](https://pydantic.dev/articles/harness-week),
> [When agents build agents](https://pydantic.dev/articles/when-agents-build-agents),
> [The agent writes faster than you can review](https://pydantic.dev/articles/harness-macroscope).

"You're absolutely right -- I'll drop the old table and redeploy." And it can. It has
credentials, it is fast, and it is confident in exactly the way that reads right, up
until CloudFormation spends four minutes half-applying a stack it then rolls back.
Now you are reading the events tab at human speed, in an account with a real bill, to
find out what your absolutely-right agent just did.

Give an agent a real cloud task -- "add a rate limiter to the checkout API" -- and it
writes clean CDK in seconds. Then it stops being fast. Every deploy is minutes of
waiting on a control plane in another region. Every experiment is a line item. Every
mistake lands in an account something real depends on. The agent runs at agent speed;
the cloud answers at cloud speed, and charges for the conversation.

## The cloud punishes exactly what the agent is good at

So you do the sensible thing and put a human in front of it. You approve each deploy.
You read each plan. You are, once again, the slowest component in the system. The
agent could try ten designs; you let it try one, because ten costs ten times as much,
takes ten times as long, and leaves ten times the debris. It could fail fast and
learn; you cannot let it fail, because failure here has a blast radius.

Everything that makes an agent good at this kind of work -- speed, fearlessness, a
willingness to try the wrong thing to find the right one -- is exactly what a real
cloud account is built to punish. You are not supervising the agent because it is bad
at the task. You are supervising it because the environment is unforgiving, and it is
your name on the account.

## The agent is not the problem. The cloud is.

So don't change the agent. Change what it deploys against.

[LocalStack](https://www.localstack.cloud/) runs the AWS APIs -- around 120 services,
the real request and response shapes -- as a container on your laptop. The agent's CDK
does not change. `aws` does not change. What changes is everything the account made
expensive:

- The four-minute deploy becomes seconds. There is no control plane in another region
  to wait on.
- The line item becomes zero. Nothing is provisioned anywhere you pay for.
- The blast radius becomes nothing. There is no production -- there is a container you
  can throw away.

And one more, which turns out to matter most: you can snapshot it. LocalStack's cloud
pods save and restore the whole environment's state, so "reset to the way it was
before the agent touched it" is one call, not an afternoon.

It is emulation, not the real thing; the last mile of fidelity still belongs to a
staging account. But for the part of the loop where an agent is trying things and being
wrong on the way to being right, a disposable cloud is the environment the work
actually wants.

## The part that isn't just faster

Here is where it stops being a quicker version of the same thing and becomes a
different thing.

Go back to the constraint we swallowed a moment ago: you would never point five agents
at one AWS account at once. Of course not -- five agents, one production, shared state,
five times the bill. But a cloud that is free to run and restores from a snapshot in a
second removes the reason. So don't run one agent against one cloud. Give every agent
its own.

Restore the same snapshot into five isolated instances. Hand each to a sub-agent with a
different idea -- a DynamoDB token bucket, API Gateway usage plans, a Redis sliding
window -- and let each one build it, deploy it, and test it for real, against a
byte-identical starting point. Then compare the ones that actually stood up, on the
evidence of what happened rather than what was promised.

On real AWS this is not expensive. It is impossible. On a disposable cloud it is
Tuesday.

Coordinating it is what [`DynamicWorkflow`](https://pydantic.dev/articles/when-agents-build-agents)
is for. Each specialist is a normal agent, wired to the LocalStack MCP server. The one
on top gets a single tool, `run_workflow`, and a catalog of the others:

```python
from pydantic_ai import Agent
from pydantic_ai_harness.dynamic_workflow import DynamicWorkflow, WorkflowAgent

# engineer and judge are ordinary agents; engineer holds the LocalStack MCP toolset.
architect = Agent(
    'anthropic:claude-opus-4-7',
    instructions='Evaluate the rate-limiter designs and recommend the one the evidence supports.',
    capabilities=[
        DynamicWorkflow(
            agents=[
                WorkflowAgent(engineer, name='engineer',
                              description='Builds and tests one design on its own LocalStack cloud.'),
                WorkflowAgent(judge, name='judge',
                              description='Compares tested designs and picks a winner.'),
            ],
            max_agent_calls=20,
        ),
    ],
)
```

Given the task, the architect does not make ten tool calls and narrate the results
back to itself. It writes one script:

```python
import asyncio

designs = [
    'DynamoDB on-demand + token bucket in the handler',
    'API Gateway usage plans + API keys',
    'ElastiCache (Redis) sliding-window counter',
]

reports = await asyncio.gather(*[
    engineer(task=f'Restore cloud-pod prod-baseline into a fresh LocalStack instance. '
                  f'Build: {d}. Deploy it, run the load test, inject a fault, and check '
                  f'the IAM policy. Return a scored report with the evidence.')
    for d in designs
])

await judge(task='Recommend one design, citing the load, fault, and IAM evidence:\n\n'
                 + '\n\n---\n\n'.join(reports))
```

Three real deployments, built and tested in parallel, each on its own cloud, judged on
what actually happened under load and fault injection. The whole tree runs inside one
`run_workflow` call. The architect's context never fills with deploy logs; only the
recommendation comes back. The choreography moved out of the conversation and into code
-- and the cloud it choreographs is one you can afford to run five of.

Cheap to run is not the same as free. `max_agent_calls` caps the number of sub-agent
runs exactly, even under fan-out, so a workflow that decides to explore fifty designs
instead of three stops at the ceiling you set. A disposable cloud removes the cloud
bill; the budget keeps the model bill honest.

## Where this goes

There is a version of this that outlives a single run. A migration that takes an
afternoon -- move every service to least-privilege IAM, one at a time -- wants to
snapshot the environment between steps, and it wants to survive the process being
killed and resume where it left off. Pydantic AI's durable execution and cloud pods'
snapshots are two halves of that; `DynamicWorkflow`'s durable workflows are coming to
join them. No harness ships that end-to-end yet.

For now the smaller thing is enough, and it is not small. The agent can be wrong. It
can drop the table, blow the IAM policy, deploy the design that falls over under load
-- and find out, in seconds, for free, on a cloud that resets to exactly where it
started.

"You're absolutely right" stops being the sentence you brace for. It becomes a
hypothesis you can afford to test.
