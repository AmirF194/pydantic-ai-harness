---
title: Modal Sandbox
description: Give a Pydantic AI agent a per-run Modal sandbox with command and file tools.
---

# Modal Sandbox

Give an agent an isolated Modal container for running commands and managing files
without using the host filesystem or process space.

> [!NOTE]
> Import this capability from its submodule. It is not re-exported from
> `pydantic_ai_harness`:
>
> ```python
> from pydantic_ai_harness.modal_sandbox import ModalSandbox
> ```

Modal Sandbox is a released, non-experimental capability. Pydantic AI Harness is
still on 0.x releases, so the API may change between minor releases. See the
[version policy](index.md#version-policy).

[Source](https://github.com/pydantic/pydantic-ai-harness/tree/main/pydantic_ai_harness/modal_sandbox/)

## Usage

Install the optional Modal dependency and configure Modal credentials:

```bash
uv add "pydantic-ai-harness[modal]"
export MODAL_TOKEN_ID=...
export MODAL_TOKEN_SECRET=...
```

Pass `ModalSandbox` through the agent's `capabilities` parameter:

```python
from pydantic_ai import Agent
from pydantic_ai_harness.modal_sandbox import ModalSandbox

agent = Agent(
    'anthropic:claude-sonnet-4-6',
    capabilities=[ModalSandbox()],
)

result = agent.run_sync('Create a Python script and run its tests.')
print(result.output)
```

The capability contributes four tools:

| Tool | Purpose |
| --- | --- |
| `run_command` | Run a shell command through `sh -c`. |
| `read_file` | Read a UTF-8 text file with bounded output and line paging. |
| `write_file` | Write a UTF-8 text file and create parent directories. |
| `list_directory` | List directory entries, marking directories with `/`. |

Command output labels stdout and stderr and reports non-zero exit codes to the
model. It keeps the tail when truncating, so later diagnostics remain visible.
File reads keep the head and return the next line offset when more content is
available.

## Lifecycle

By default, each agent run creates an owned sandbox and requests its termination
when the run exits. Teardown waits for confirmation for a bounded period; if the
control plane does not respond, `sandbox_timeout` remains the server-side
cleanup backstop. The sandbox is provisioned when the run enters the capability
toolset, even if no sandbox tool is called. Deferred tool loading controls which
tool definitions reach the model; it does not defer toolset lifecycle.

Attach to a sandbox managed elsewhere by ID:

```python
from pydantic_ai_harness.modal_sandbox import ModalSandbox

ModalSandbox(sandbox_id='sb-abc123')
```

To share a sandbox across runs while controlling its lifetime, create and enter a
`ModalSandboxSession` yourself:

```python
from pydantic_ai import Agent
from pydantic_ai_harness.modal_sandbox import ModalSandbox, ModalSandboxSession

async with ModalSandboxSession(image='python:3.12-slim') as session:
    agent = Agent(
        'anthropic:claude-sonnet-4-6',
        capabilities=[ModalSandbox(session=session)],
    )
    await agent.run('Install the project dependencies.')
    await agent.run('Run the test suite in the same sandbox.')
```

Attached and injected sandboxes are left running when an agent run ends. They
share a filesystem and process space, so do not use the same sandbox for
overlapping runs that need isolation.

## Timeouts and output limits

Every model-facing command receives a finite deadline.
`default_command_timeout` supplies the default and `max_command_timeout`
caps model-supplied values. Modal accepts whole-second deadlines, so fractional
values round up without exceeding the configured integer ceiling.

Modal does not expose a per-command kill operation. Cancelling the client wait
does not stop the remote command immediately; it continues until its command
deadline or the sandbox is terminated.

Each command stream retains the last `max_output_bytes` after every transport
chunk, and the output payload is also truncated by `max_output_lines`. Labels,
truncation or continuation notes, and command status add a small amount beyond
those payload limits. One transport chunk can temporarily be larger than the
byte limit. Invalid UTF-8 is decoded with replacement characters.

`read_file` checks file metadata before reading and checks the returned byte
count again. A file that grows between those operations can temporarily exceed
`max_read_bytes` in client memory before being rejected. Modal's filesystem
API does not expose a bounded read, so use a bounded shell command for virtual
files or other paths whose reported size may be misleading.

`list_directory` materializes the complete directory listing before truncating
it. Listing a directory with many entries therefore uses memory proportional to
the number of entries; use a narrowed shell command for unusually large
directories.

Modal's SDK is asyncio-native. The capability requires an asyncio event loop and
does not run under trio.

## Errors and composition

Recoverable command and filesystem failures become model retry prompts. A
terminated sandbox or rejected Modal credentials raises a terminal public error
instead of retrying against the same unusable sandbox.

The toolset is an implementation detail. The public lower-level API consists of
`ModalSandboxSession`, `ModalSandboxExecResult`, and the typed sandbox error
classes.

Do not combine this capability with another unprefixed capability that registers
`run_command`, `read_file`, `write_file`, or `list_directory`. Pydantic
AI rejects duplicate tool names. Prefix the capability with
`PrefixTools(wrapped=ModalSandbox(), prefix='modal')` before composing it with
another capability that uses the same names.

## Configuration

```python
from pydantic_ai_harness.modal_sandbox import ModalSandbox

ModalSandbox(
    image='python:3.12-slim',
    sandbox_id=None,
    session=None,
    app_name='pydantic-ai-harness',
    create_app_if_missing=True,
    sandbox_timeout=300,
    workdir=None,
    env=None,
    default_command_timeout=60.0,
    max_command_timeout=None,
    max_output_bytes=50 * 1024,
    max_output_lines=2000,
    max_read_bytes=5 * 1024 * 1024,
    include_instructions=True,
)
```

Settings used only when creating a sandbox cannot be combined with
`sandbox_id` or an injected `session`. These conflicts fail at construction
instead of being ignored.

## Agent specs

Register `ModalSandbox` as a custom capability type when loading an agent spec:

```yaml
model: anthropic:claude-sonnet-4-6
capabilities:
  - ModalSandbox:
      image: python:3.12-slim
      sandbox_timeout: 600
```

```python
from pydantic_ai import Agent
from pydantic_ai_harness.modal_sandbox import ModalSandbox

agent = Agent.from_file('agent.yaml', custom_capability_types=[ModalSandbox])
```

## API reference

- [Pydantic AI capabilities](/ai/core-concepts/capabilities/)
- [Pydantic AI toolsets](/ai/tools-toolsets/toolsets/)
- [Modal sandboxes](https://modal.com/docs/guide/sandbox)

::: pydantic_ai_harness.modal_sandbox.ModalSandbox

::: pydantic_ai_harness.modal_sandbox.ModalSandboxSession

::: pydantic_ai_harness.modal_sandbox.ModalSandboxExecResult

::: pydantic_ai_harness.modal_sandbox.ModalSandboxError

::: pydantic_ai_harness.modal_sandbox.ModalSandboxTerminalError

::: pydantic_ai_harness.modal_sandbox.ModalSandboxUnavailableError
