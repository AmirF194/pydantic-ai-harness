---
title: E2B Sandbox
description: Give a Pydantic AI agent a per-run E2B sandbox with command and file tools.
---

# E2B Sandbox

`E2BSandbox` gives an agent an isolated cloud computer for running commands and
working with files. Use it for coding, tests, data processing, and other tasks
that should not execute model-generated commands on the application host.

The capability adds shell and file tools backed by an
[E2B sandbox](https://e2b.dev/docs). By default, every agent run gets a fresh
sandbox created from a template. The capability kills it when the run ends. You
can also attach an existing sandbox and reuse it across runs.

> While Pydantic AI Harness is on 0.x releases, the API may change between minor releases; when it does, deprecation warnings and release-note migration guidance tell you (or your agent) exactly how to upgrade. See the [version policy](index.md#version-policy).

## Quick start

Install the `e2b` extra and set an API key:

```bash
uv add "pydantic-ai-harness[e2b]"
export E2B_API_KEY=...
```

Add `E2BSandbox` to the agent:

```python
from pydantic_ai import Agent
from pydantic_ai_harness.e2b_sandbox import E2BSandbox

agent = Agent(
    'anthropic:claude-sonnet-4-6',
    capabilities=[E2BSandbox(template='base')],
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
| `run_command` | Run a shell command in the sandbox. |
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

By default, each agent run creates an owned sandbox and kills it when the run
exits, so expect a cold-start cost per run. Teardown waits for confirmation for a
bounded period; if the control plane does not respond, `sandbox_timeout` remains
the server-side cleanup backstop. The sandbox is provisioned when the run starts,
even if no sandbox tool is called.

E2B has no create-or-reuse key for sandboxes, so a durable engine that retries the
provisioning step after a crash creates a second one; the abandoned sandbox is
left to its own `sandbox_timeout`.

Attach to a sandbox managed elsewhere by ID to reuse it across runs. It is never
killed by the capability:

```python
from pydantic_ai_harness.e2b_sandbox import E2BSandbox

E2BSandbox(sandbox_id='sbx-abc123', max_command_timeout=600)
```

Attaching to a paused sandbox resumes it: E2B treats connect as "resume if
needed", so a paused environment comes back with its filesystem intact rather than
failing the run.

The capability cannot see an attached sandbox's real lifetime, so each command
there is capped at 300s unless `max_command_timeout` raises the ceiling.

To control the environment yourself, create the backend and attach it by ID:

```python
from pydantic_ai import Agent
from pydantic_ai_harness.e2b_sandbox import E2BSandbox, E2BSandboxBackend

backend = await E2BSandboxBackend.create(template='base', sandbox_timeout=1800)
try:
    agent = Agent(
        'anthropic:claude-sonnet-4-6',
        capabilities=[E2BSandbox(sandbox_id=backend.sandbox_id, max_command_timeout=600)],
    )
    await agent.run('Install the project dependencies.')
    await agent.run('Run the test suite in the same sandbox.')
finally:
    await backend.close(terminate=True)
```

Attached sandboxes are left running when an agent run ends. They share a
filesystem and process space, so do not use the same sandbox for overlapping runs
that need isolation.

## How commands run

E2B executes every command through `/bin/bash -l -c`, which has two consequences
worth knowing:

- An argv sequence has no direct E2B equivalent, so `E2BSandboxBackend.run` quotes
  it with `shlex.join` into a single shell word string. The shell still parses the
  result, but the quoting keeps each element one literal word.
- Bash login startup files run before every command. Keep them silent in custom
  templates: their output is part of the command's output.

## Timeouts and cancellation

Every model-facing command receives a finite deadline.
`default_command_timeout` supplies the default and `max_command_timeout` caps
model-supplied values, falling back to `sandbox_timeout`. Fractional values are
used as given.

E2B's own command `timeout` bounds the output stream rather than the command: when
it expires the SDK stops listening and the command keeps running. The backend
therefore owns the deadline and calls E2B's per-command kill when it expires, and
again when the caller is cancelled, so a cancelled run does not knowingly leave a
command running. A command that hits its deadline raises
`E2BSandboxCommandTimeoutError`, a builtin `TimeoutError` carrying the output
produced before the kill, which `run_command` reports as that output plus a
`[timed out after Ns]` note.

That kill signals the command's own process. A process the command started in the
background is not reached by it and lives until the sandbox is torn down, which for
an owned run is the end of the run.

## Output limits

Each stream's payload is truncated separately by `max_output_bytes` and
`max_output_lines` in the tool output, so a large stderr cannot crowd out stdout
and the `[stdout]` / `[stderr]` labels always survive. Any cut is marked. Labels,
truncation or continuation notes, and command status add a small amount beyond
those payload limits. These caps bound what reaches the model, not what crosses
the wire: the sandbox protocol delivers a command's whole output and E2B's SDK
accumulates it in the command handle, so bound it in-command (`| tail -c 10000`)
when the transfer itself matters. Invalid UTF-8 is decoded with replacement
characters.

`read_file` checks file metadata before reading and checks the returned byte count
again. A file that grows between those operations can temporarily exceed
`max_read_bytes` in client memory before being rejected. The filesystem API
exposes no bounded read, so use a bounded shell command for virtual files or other
paths whose reported size may be misleading.

`list_directory` materializes the complete directory listing before truncating it.
Listing a directory with many entries therefore uses memory proportional to the
number of entries; use a narrowed shell command for unusually large directories.

E2B's async SDK is asyncio-native (its command handles are asyncio tasks). The
capability requires an asyncio event loop and does not run under trio.

## Errors and composition

Recoverable command and filesystem failures become model retry prompts, including
a missing path, which the backend reports as the builtin `FileNotFoundError` the
sandbox protocol requires. A sandbox that is gone raises
`E2BSandboxUnavailableError` and rejected credentials raise `E2BSandboxAuthError`
(both `E2BSandboxTerminalError` subclasses) instead of retrying against the same
unusable sandbox. E2B reports an unanswered request as a timeout whether the
sandbox is merely slow or already gone, so a failure of that shape is classified by
asking E2B whether the sandbox is still running.

The toolset is an implementation detail. The public lower-level API consists of
`E2BSandboxBackend` -- the `SandboxBackend` implementation, which also implements
`SupportsFilesystem` and `SupportsStart` -- and the typed sandbox error classes.

Do not combine this capability with another unprefixed capability that registers
`run_command`, `read_file`, `write_file`, or `list_directory` (e.g. the Shell or
FileSystem capabilities). Pydantic AI rejects duplicate tool names. Prefix the
capability before composing it with another capability that uses the same names:

```python
from pydantic_ai.capabilities import PrefixTools

from pydantic_ai_harness.e2b_sandbox import E2BSandbox

sandbox = PrefixTools(
    wrapped=E2BSandbox(
        instructions=(
            'You have an E2B cloud sandbox. Use the e2b_-prefixed tools to run '
            'shell commands and manage files in it.'
        )
    ),
    prefix='e2b',
)
```

Prefixing renames the tools (`e2b_run_command`, ...) but does not rewrite the
capability's default instructions, which name the unprefixed tools -- pass
`instructions` with text that matches the prefixed names.

## Configuration

```python
from pydantic_ai_harness.e2b_sandbox import E2BSandbox

E2BSandbox(
    template=None,
    sandbox_id=None,
    sandbox_timeout=300,
    workdir=None,
    env=None,
    metadata=None,
    allow_internet_access=True,
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

Settings used only when creating a sandbox (`template`, `sandbox_timeout`, `env`,
`metadata`, `allow_internet_access`) cannot be combined with `sandbox_id`. These
conflicts fail at construction instead of being ignored. `workdir` is the
exception: E2B sets the working directory per command rather than at creation, so
it still applies to an attached sandbox.

## Not yet supported

- Streaming command output to the model: `run_command` returns once the command
  finishes (or hits its deadline), not incrementally. `E2BSandboxBackend` does not
  implement `SupportsStream` either: E2B delivers live output through callbacks its
  own event pump awaits, with no async iterator behind them, so a `stream()` here
  could only be a replay of buffered output.
- E2B features beyond commands and files: PTYs, port forwarding, snapshots, volume
  mounts, and MCP servers are reachable through the underlying
  `E2BSandboxBackend.sandbox` object, but the capability exposes none of them.
- Spilling full output to a file: truncated file reads end with the next `offset`
  to page from and oversized files get a shell-slice hint; truncated command output
  gets a truncation marker. Nothing is written to a file in the sandbox for the
  model to open.

## Agent specs

Register `E2BSandbox` as a custom capability type when loading an agent spec:

```yaml
model: anthropic:claude-sonnet-4-6
capabilities:
  - E2BSandbox:
      template: base
      sandbox_timeout: 600
```

```python
from pydantic_ai import Agent
from pydantic_ai_harness.e2b_sandbox import E2BSandbox

agent = Agent.from_file('agent.yaml', custom_capability_types=[E2BSandbox])
```

## API reference

- [Pydantic AI capabilities](/ai/core-concepts/capabilities/)
- [Pydantic AI toolsets](/ai/tools-toolsets/toolsets/)
- [E2B documentation](https://e2b.dev/docs)
- [E2B Sandbox source code](https://github.com/pydantic/pydantic-ai-harness/tree/main/pydantic_ai_harness/e2b_sandbox/)
- [Pydantic AI Harness version policy](index.md#version-policy)


::: pydantic_ai_harness.e2b_sandbox.E2BSandbox

::: pydantic_ai_harness.e2b_sandbox.E2BSandboxBackend

::: pydantic_ai_harness.e2b_sandbox.E2BSandboxError

::: pydantic_ai_harness.e2b_sandbox.E2BSandboxTerminalError

::: pydantic_ai_harness.e2b_sandbox.E2BSandboxAuthError

::: pydantic_ai_harness.e2b_sandbox.E2BSandboxUnavailableError

::: pydantic_ai_harness.e2b_sandbox.E2BSandboxCommandTimeoutError
