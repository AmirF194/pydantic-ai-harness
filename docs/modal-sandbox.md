---
title: Modal Sandbox
description: Give a Pydantic AI agent a per-run Modal sandbox with command and file tools.
---

# Modal Sandbox

`ModalSandbox` gives an agent an isolated cloud container for running commands
and working with files. Use it for coding, data processing, and other tasks that
should not execute model-generated commands on the application host.

The capability adds shell and file tools backed by a
[Modal sandbox](https://modal.com/docs/guide/sandbox). By default, every agent
run gets a fresh sandbox created from a container image. The capability requests
termination when the run ends. You can also attach an existing sandbox or reuse
one across several runs.

> While Pydantic AI Harness is on 0.x releases, the API may change between minor releases; when it does, deprecation warnings and release-note migration guidance tell you (or your agent) exactly how to upgrade. See the [version policy](index.md#version-policy).

## Quick start

Install the `modal` extra and authenticate with the Modal CLI. In CI, set
`MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET` instead.

```bash
uv add "pydantic-ai-harness[modal]"
modal token new                # writes ~/.modal.toml
# or, e.g. in CI:
export MODAL_TOKEN_ID=...
export MODAL_TOKEN_SECRET=...
```

Add `ModalSandbox` to the agent:

```python
from pydantic_ai import Agent
from pydantic_ai_harness import ModalSandbox

agent = Agent(
    'anthropic:claude-sonnet-4-6',
    capabilities=[ModalSandbox(image='python:3.12-slim')],
)

result = agent.run_sync('Create a Python script and run its tests.')
print(result.output)
```

During the run, the agent can create files, inspect its working directory, run
commands, and react to command failures. The sandbox is separate from the host
filesystem and process space.

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

The capability supplies the run's sandbox through Pydantic AI's lifecycle hooks:
`create_sandbox` provisions the environment when the run starts, `get_sandbox`
connects to it, and `destroy_sandbox` tears it down when the run ends, including
on failure. The tools read `ctx.sandbox`, so a `sandbox=` run argument or an
earlier capability's sandbox is what they operate on instead.

By default, each agent run creates an owned sandbox and terminates it when the run
exits, so expect a cold-start cost per run. Teardown waits for confirmation for a
bounded period; if the control plane does not respond, `sandbox_timeout` remains
the server-side cleanup backstop. The sandbox is provisioned when the run starts,
even if no sandbox tool is called.

Modal has no create-or-reuse key for sandboxes, so a durable engine that retries
the provisioning step after a crash creates a second one; the abandoned sandbox is
left to its own `sandbox_timeout`.

Attach to a sandbox managed elsewhere by ID to reuse it across runs. It is never
terminated by the capability:

```python
from pydantic_ai_harness import ModalSandbox

ModalSandbox(sandbox_id='sb-abc123', max_command_timeout=600)
```

The capability cannot see an attached sandbox's real lifetime, so each command
there is capped at 300s unless `max_command_timeout` raises the ceiling.

To control the environment yourself, create the backend and attach it by ID:

```python
from pydantic_ai import Agent
from pydantic_ai_harness import ModalSandbox
from pydantic_ai_harness.modal_sandbox import ModalSandboxBackend

backend = await ModalSandboxBackend.create(image='python:3.12-slim', sandbox_timeout=1800)
try:
    agent = Agent(
        'anthropic:claude-sonnet-4-6',
        capabilities=[ModalSandbox(sandbox_id=backend.sandbox_id, max_command_timeout=600)],
    )
    await agent.run('Install the project dependencies.')
    await agent.run('Run the test suite in the same sandbox.')
finally:
    await backend.close(terminate=True)
```

Attached sandboxes are left running when an agent run ends. They share a
filesystem and process space, so do not use the same sandbox for overlapping runs
that need isolation.

## Timeouts and output limits

Every model-facing command receives a finite deadline.
`default_command_timeout` supplies the default and `max_command_timeout`
caps model-supplied values. Modal accepts whole-second deadlines, so fractional
values round up without exceeding the configured integer ceiling.

Modal does not expose a per-command kill operation, so `SandboxProcess.kill()`
raises `NotImplementedError` naming the deadline as the alternative. Cancelling
the client wait does not stop the remote command immediately; it continues until
its command deadline or the sandbox is terminated. A command that hits its
deadline raises `ModalSandboxCommandTimeoutError`, a builtin `TimeoutError`
carrying the output produced before the kill, which `run_command` reports as that
output plus a `[timed out after Ns]` note.

Each stream's payload is truncated separately by `max_output_bytes` and
`max_output_lines` in the tool output, so a large stderr cannot crowd out stdout
and the `[stdout]` / `[stderr]` labels always survive. Any cut is marked. Labels,
truncation or continuation notes, and command status add a small amount beyond
those payload limits. These caps bound what reaches the model, not what crosses
the wire: the sandbox protocol delivers a command's whole output, so bound it
in-command (`| tail -c 10000`) when the transfer itself matters. Invalid UTF-8 is
decoded with replacement characters.

`read_file` checks file metadata before reading and checks the returned byte
count again. A file that grows between those operations can temporarily exceed
`max_read_bytes` in client memory before being rejected. The filesystem API
exposes no bounded read, so use a bounded shell command for virtual files or other
paths whose reported size may be misleading.

`list_directory` materializes the complete directory listing before truncating
it. Listing a directory with many entries therefore uses memory proportional to
the number of entries; use a narrowed shell command for unusually large
directories.

Modal's SDK is asyncio-native. The capability requires an asyncio event loop and
does not run under trio.

## Errors and composition

Recoverable command and filesystem failures become model retry prompts, including
a missing path, which the backend reports as the builtin `FileNotFoundError` the
sandbox protocol requires. A terminated sandbox raises
`ModalSandboxUnavailableError` and rejected Modal credentials raise
`ModalSandboxAuthError` (both `ModalSandboxTerminalError` subclasses) instead of
retrying against the same unusable sandbox.

The toolset is an implementation detail. The public lower-level API consists of
`ModalSandboxBackend` -- the `SandboxBackend` implementation, which also
implements `SupportsFilesystem`, `SupportsStart`, and `SupportsStream` -- and the
typed sandbox error classes.

Do not combine this capability with another unprefixed capability that registers
`run_command`, `read_file`, `write_file`, or `list_directory` (e.g. the Shell or
FileSystem capabilities). Pydantic AI rejects duplicate tool names. Prefix the
capability before composing it with another capability that uses the same names:

```python
from pydantic_ai.capabilities import PrefixTools

from pydantic_ai_harness import ModalSandbox

sandbox = PrefixTools(
    wrapped=ModalSandbox(
        instructions=(
            'You have a Modal cloud sandbox. Use the modal_-prefixed tools to run '
            'shell commands and manage files in it.'
        )
    ),
    prefix='modal',
)
```

Prefixing renames the tools (`modal_run_command`, ...) but does not rewrite the
capability's default instructions, which name the unprefixed tools -- pass
`instructions` with text that matches the prefixed names.

## Configuration

```python
from pydantic_ai_harness import ModalSandbox

ModalSandbox(
    image='python:3.12-slim',
    sandbox_id=None,
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
    instructions=None,
)
```

The default instructions state the tools, the command timeout, and its ceiling.
Set `instructions=''` to add none, or pass your own text to replace the default.

Settings used only when creating a sandbox cannot be combined with `sandbox_id`.
These conflicts fail at construction instead of being ignored.

## Not yet supported

- Streaming command output to the model: `run_command` returns once the command
  finishes (or hits its deadline), not incrementally. `ModalSandboxBackend`
  implements `SupportsStream`, so a caller can stream a command it starts itself.
- Custom-built images, mounts, or `modal.Secret`: `image` takes a registry tag,
  and `env` takes plain environment variables. For anything richer, create the
  sandbox yourself with the Modal SDK and attach it via `sandbox_id`.
- Spilling full output to a file: truncated file reads end with the next
  `offset` to page from and oversized files get a shell-slice hint; truncated
  command output gets a truncation marker. Nothing is written to a file in the
  sandbox for the model to open.

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
from pydantic_ai_harness import ModalSandbox

agent = Agent.from_file('agent.yaml', custom_capability_types=[ModalSandbox])
```

## API reference

- [Pydantic AI capabilities](/ai/core-concepts/capabilities/)
- [Pydantic AI toolsets](/ai/tools-toolsets/toolsets/)
- [Modal sandboxes](https://modal.com/docs/guide/sandbox)
- [Modal Sandbox source code](https://github.com/pydantic/pydantic-ai-harness/tree/main/pydantic_ai_harness/modal_sandbox/)
- [Pydantic AI Harness version policy](index.md#version-policy)


::: pydantic_ai_harness.modal_sandbox.ModalSandbox

::: pydantic_ai_harness.modal_sandbox.ModalSandboxBackend

::: pydantic_ai_harness.modal_sandbox.ModalSandboxError

::: pydantic_ai_harness.modal_sandbox.ModalSandboxTerminalError

::: pydantic_ai_harness.modal_sandbox.ModalSandboxAuthError

::: pydantic_ai_harness.modal_sandbox.ModalSandboxUnavailableError

::: pydantic_ai_harness.modal_sandbox.ModalSandboxCommandTimeoutError
