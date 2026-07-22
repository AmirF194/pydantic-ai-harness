---
title: AWS Lambda Durability
description: Checkpoint a Pydantic AI agent's model requests and tool calls into AWS Lambda durable steps.
---

# AWS Lambda Durability

`LambdaDurability` makes an agent resumable on [AWS Lambda durable
functions](https://docs.aws.amazon.com/lambda/latest/dg/durable-functions.html). Every model
request, function tool call, MCP call, and dynamic-toolset resolution is checkpointed as a durable
step, so an invocation that times out, fails, or is retried continues from the last completed step
instead of repeating the work already paid for.

Lambda keeps a log of durable operations. When an execution resumes, the handler runs again from
the top and completed steps return their stored results rather than executing. Without
checkpointing, a resumed run would repeat every model request and tool call.

## Installation

```bash
pip install "pydantic-ai-harness[aws-lambda]"
```

The AWS Durable Execution SDK requires Python 3.11 or newer.

## Quick start

Attach the capability when you build the agent, and enter the run from the durable handler with
`run_durable`:

```python {title="handler.py" test="skip" lint="skip"}
from typing import Any

from aws_durable_execution_sdk_python import DurableContext, durable_execution
from pydantic_ai import Agent

from pydantic_ai_harness.aws_lambda import LambdaDurability, run_durable

agent = Agent(
    'bedrock:us.amazon.nova-pro-v1:0',
    name='support',
    capabilities=[LambdaDurability()],
)


@agent.tool_plain
def get_weather(city: str) -> str:
    return f'It is sunny in {city}.'


@durable_execution
def handler(event: dict[str, Any], context: DurableContext) -> str:
    result = run_durable(lambda: agent.run(str(event['prompt'])), context=context)
    return result.output
```

Attach the capability at construction, not per invocation: binding is when the agent's toolsets are
wrapped, so a capability added per run cannot checkpoint tool calls.

Deploy with a durable configuration and invoke a published version, since in-flight executions are
pinned to the version that started them:

```bash
aws lambda create-function \
  --function-name support-agent \
  --runtime python3.13 \
  --handler handler.handler \
  --role <ROLE_ARN> \
  --zip-file fileb://support-agent.zip \
  --timeout 300 --memory-size 1024 \
  --durable-config '{"ExecutionTimeout":3600,"RetentionPeriodInDays":7}'

aws lambda publish-version --function-name support-agent
```

## How the sync handler and the async agent connect

Lambda's durable API is synchronous: `context.step(...)` blocks, and every step has to be created
on the thread Lambda invoked. An agent run is async. `run_durable` bridges the two: it hosts the
agent run on a background event loop and services its steps on the handler thread, so all steps are
created in one continuous sequence. A step body hands its work back to the agent loop and blocks
until it finishes, which keeps the loop free while the handler thread waits.

Two consequences are worth knowing:

- `run_durable` blocks the calling thread, so it cannot be called from inside a running event loop.
  Call it directly from the synchronous handler.
- Durable steps cannot nest. A tool that starts another durable agent run is rejected with an
  explanatory error rather than deadlocking.

## What gets checkpointed

Step names are built from the agent's `name` and each toolset's `id`:

| Step name | Operation |
|---|---|
| `{name}__model.request` | one model request segment |
| `{name}__model.request_stream` | one streamed model request segment |
| `{name}__model.cancel_suspended_response` | tearing down a suspended response |
| `{name}__function_toolset__{id}.call_tool:{tool}` | a function tool call |
| `{name}__mcp_server__{id}.get_tools` | listing an MCP server's tools |
| `{name}__mcp_server__{id}.get_instructions` | an MCP server's instructions |
| `{name}__mcp_server__{id}.call_tool:{tool}` | an MCP tool call |
| `{name}__dynamic_toolset__{id}.get_tools` | resolving a dynamic toolset |
| `{name}__dynamic_toolset__{id}.call_tool:{tool}` | a dynamic toolset's tool call |
| `{name}__event_stream_handler` | one event delivered to an `event_stream_handler` |

When a run uses more than one model, the model ID is appended to the model step names
(`{name}__model.request.{model_id}`) so a resumed execution maps each checkpoint back to the model
it was recorded for.

## Constraints

- **Steps are at least once.** A step is checkpointed after it runs, so an interruption between a
  tool's side effect and its checkpoint re-runs the tool when the execution resumes. Keep tool side
  effects idempotent, or set `step_semantics` to `AT_MOST_ONCE_PER_RETRY` for the tools that cannot
  tolerate a repeat.
- **Tool calls run one at a time.** A step's identity comes from the order steps are reached, so
  concurrently scheduled tool calls could claim each other's checkpoints when the execution
  resumes. Inside a durable handler the run is switched to sequential tool execution. Outside one
  the agent keeps its configured parallelism.
- **Renaming breaks in-flight executions.** The agent's `name` and each toolset's `id` are part of
  every step name, so they should not change once the agent is deployed: a rename orphans the
  checkpoints of in-flight executions, which resume under the old names and re-run their steps.
- **Step results must survive the SDK serializer.** Results are checkpointed through the Lambda
  SDK's serializer. Tool results are encoded by Pydantic first, so structured returns such as
  `ToolReturn` and `BinaryContent` round-trip; a value Pydantic cannot serialize does not.
- **Streaming a run out of a durable execution is not supported.** There is no channel out of a
  running durable execution, so `run_stream` and `iter` are not available inside the handler. An
  `event_stream_handler` still works: model events are handled live inside the model step and each
  agent-level event is checkpointed in its own step.
- **`ctx.enqueue()` is not available inside a checkpointed tool**, because a resumed execution
  serves the recorded step output and would drop the enqueued messages. Enqueue from handler-level
  code instead.
- **Budgets.** A durable execution allows 3,000 operations and 100 MB of cumulative checkpointed
  state. A turn costs one model step plus one step per tool call, so the operation budget is
  generous, but large tool results consume the state budget: return a reference (an S3 key, say)
  rather than a blob.

## Per-tool configuration

Tool metadata under the `aws_lambda` key configures that tool's step. It accepts the `StepConfig`
fields `retry_strategy`, `step_semantics`, and `serdes`:

```python {test="skip" lint="skip"}
from aws_durable_execution_sdk_python.config import StepSemantics
from pydantic_ai.toolsets import FunctionToolset

toolset = FunctionToolset(id='billing')


@toolset.tool_plain(metadata={'aws_lambda': {'step_semantics': StepSemantics.AT_MOST_ONCE_PER_RETRY}})
def charge_card(amount: int) -> str:
    return f'charged {amount}'
```

`metadata={'aws_lambda': False}` opts a tool out of checkpointing entirely, so it runs inline on
every attempt. Use it for cheap, side-effect-free tools whose result is not worth a checkpoint. MCP
tools cannot opt out, because they perform I/O that must not re-run when the execution resumes.

`LambdaDurability(step_config=...)` sets the base configuration for every step, which per-tool
metadata replaces.

## Composition with other capabilities

`LambdaDurability` orders itself innermost, so any other capability's contribution to a model
request is already applied inside the durable step. Attach it alongside other capabilities as usual.

## Relation to the AWS example

The AWS Durable Execution SDK repository carries a
[Pydantic AI example](https://github.com/aws/aws-durable-execution-sdk-python) showing the
sync-to-async bridge. This capability builds on the same idea and adds the pieces a production
integration needs: control-flow signals crossing a step as values, tool-result serialization, MCP
and dynamic toolsets, transparency outside a durable handler, and per-tool step configuration.

## Further reading

- [AWS Lambda durable functions](https://docs.aws.amazon.com/lambda/latest/dg/durable-functions.html)
- [AWS Durable Execution SDK for Python](https://github.com/aws/aws-durable-execution-sdk-python)
- [Pydantic AI durable execution](https://pydantic.dev/docs/ai/durable_execution/temporal/)
- [AWS Lambda Durability source code](https://github.com/pydantic/pydantic-ai-harness/tree/main/pydantic_ai_harness/aws_lambda/)

## API reference

::: pydantic_ai_harness.aws_lambda.LambdaDurability

::: pydantic_ai_harness.aws_lambda.run_durable
