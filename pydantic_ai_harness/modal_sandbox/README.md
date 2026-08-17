# Modal Sandbox

`ModalSandbox` gives an agent an isolated cloud container for running commands
and working with files. Use it for coding, data processing, and other tasks that
should not execute model-generated commands on the application host.

The capability supplies the run's
[sandbox](https://pydantic.dev/docs/ai/sandbox/) from a
[Modal sandbox](https://modal.com/docs/guide/sandbox), and adds shell and file
tools that work in it. By default, every agent run gets a fresh sandbox created
from a container image, terminated when the run ends. You can also attach an
existing sandbox and reuse it across runs.

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

## Tools

| Tool | Purpose |
|---|---|
| `run_command` | Run a shell command (`sh -c`) in the sandbox. Pipes, redirection, `&&`, and globs work. Returns labelled stdout/stderr plus an exit code on failure. |
| `read_file` | Read a text file from the sandbox. |
| `write_file` | Write text to a file (creating parent directories). |
| `list_directory` | List a directory's entries (directories shown with a trailing `/`). |

Output is labelled with `[stdout]` / `[stderr]` markers and an `[exit code: N]`
line on non-zero exit. Each command stream (and each file read) is truncated
separately by `max_output_bytes` (UTF-8 bytes) and `max_output_lines` (lines),
whichever is hit first, so a large stderr cannot crowd out stdout and the labels
always survive. Labels, truncation or continuation notes, and command status add
a small amount beyond those payload limits. For commands the **tail** is kept, so
errors survive truncation; file reads keep the head and return the next `offset`
to page from. A non-zero exit from `run_command` is reported, not raised, so the
model can react to it; file-tool failures (missing path, etc.) come back as a
retry prompt.

`max_output_bytes` and `max_output_lines` bound what reaches the model, not what
crosses the wire: the sandbox protocol delivers a command's whole output, so a
command that floods stdout is capped after the transfer, not during it. Bound it
in-command (`| tail -c 10000`) when the transfer itself matters. Command output is
read as bytes and decoded as UTF-8 with `errors='replace'`, so binary or invalid
UTF-8 output is reported with replacement characters instead of crashing the run.

`run_command` runs through `/bin/sh -c`; `read_file`, `write_file`, and
`list_directory` go through the sandbox filesystem API (no shell), so writes
stream the content rather than passing it as a command argument, and parent
directories are created on write. A relative path given to a file tool is
resolved against the sandbox working directory (discovered once with `pwd` and
cached) and normalized, keeping both views of the tree consistent. Resolution is
a spelling convenience, not confinement: isolation is the sandbox's job.

Because the tools read the run's sandbox rather than owning a connection, they
also work against any other backend the run attached (a `sandbox=` run argument,
or another capability's sandbox). Behavior that only Modal reports -- the output
that preceded a deadline kill, the terminal error taxonomy -- degrades to what
that backend reports.

## Failure handling

Failures split into two kinds:

- **Recoverable** -- a bad path, a command that exits non-zero, a transient
  sandbox-side error. These come back to the model as a retry (`ModelRetry`) or,
  for `run_command`, as reported output it can react to. Retrying can plausibly
  work, so the run continues.
- **Terminal** -- the sandbox itself is gone (terminated, or expired at its
  `sandbox_timeout`), raising `ModalSandboxUnavailableError`, or the credentials
  were rejected, raising `ModalSandboxAuthError`. Re-running the command cannot
  fix these, so the tool lets them propagate (both are `ModalSandboxTerminalError`
  subclasses) and the run ends with an actionable message instead of looping the
  model against a dead sandbox. If owned runs legitimately hit the lifetime,
  raise `sandbox_timeout`.

A command that hits its deadline raises `ModalSandboxCommandTimeoutError`, a
builtin `TimeoutError` carrying the output produced before the kill. `run_command`
turns it back into that output plus a `[timed out after Ns]` note. A missing path
raises the builtin `FileNotFoundError`, which the file tools report as a retry.

## Sandbox lifetime

The capability implements Pydantic AI's sandbox lifecycle hooks: `create_sandbox`
provisions the environment at the start of a run, `get_sandbox` connects to it,
and `destroy_sandbox` tears it down when the run ends (including on failure).

By default the capability is **owned**: each run creates a fresh sandbox and
terminates it when the run ends. Teardown waits for confirmation for a bounded
period; if Modal's control plane does not respond, `sandbox_timeout` remains the
server-side cleanup backstop. Each owned run spins up its own sandbox, so expect a
cold-start cost per run, and the sandbox is provisioned even if the model never
calls a sandbox tool.

Modal has no create-or-reuse key for sandboxes, so a durable engine that retries
the provisioning step after a crash creates a second one; the abandoned sandbox is
left to its own `sandbox_timeout`.

**Attach** to a sandbox you manage elsewhere (e.g. created via the Modal CLI) by
id, to reuse it across runs. It is never terminated by the capability:

```python
from pydantic_ai_harness import ModalSandbox

ModalSandbox(sandbox_id='sb-abc123', max_command_timeout=600)
```

The capability cannot see an attached sandbox's real lifetime, so each command
there is capped at 300s unless `max_command_timeout` raises the ceiling.

An attached sandbox is not concurrency-safe across overlapping runs: they share
one filesystem and one process space. Use separate sandboxes for runs that overlap
in time.

To control the environment yourself, create the backend and pass it to the run
instead of adding the capability's lifecycle:

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
    await agent.run('clone the repo and install deps')   # same sandbox...
    await agent.run('run the test suite')                # ...reused across runs
finally:
    await backend.close(terminate=True)
```

## Cancellation

Modal does not currently expose a way to kill a single running command, so a
command is stopped by its own deadline or by the whole sandbox being terminated.
The capability is built around that:

- A cancelled run stops waiting for the command immediately, but the command
  keeps running in the sandbox until its deadline. Every `run_command` carries
  one (`default_command_timeout`, or the per-call `timeout_seconds`), so a
  cancelled or abandoned command is reaped within that window rather than running
  on. Lower `default_command_timeout` to shorten the worst-case window. A
  model-supplied `timeout_seconds` is capped at `max_command_timeout` (which
  defaults to `sandbox_timeout`), so the model cannot ask for an unbounded one.
- `SandboxProcess.kill()` raises `NotImplementedError` naming that alternative,
  rather than pretending a single command can be stopped.
- When an owned run ends or is cancelled, the capability terminates the sandbox
  and waits for a bounded period. `sandbox_timeout` remains the server-side
  backstop if the teardown RPC cannot be confirmed.
- An attached sandbox is never terminated by the capability (its owner controls
  that), so an in-flight command there is bounded only by its deadline.

## Lower-level access

`ModalSandbox` is the main entry point. The toolset is an implementation detail.
`ModalSandboxBackend` is public: it implements Pydantic AI's
[`SandboxBackend`](https://pydantic.dev/docs/ai/sandbox/) protocol over a Modal
sandbox, including `SupportsFilesystem`, `SupportsStart`, and `SupportsStream`, so
it can be passed to any run or used directly.

```python
from pydantic_ai_harness.modal_sandbox import ModalSandboxBackend

backend = await ModalSandboxBackend.create(image='python:3.12-slim')
try:
    result = await backend.run(['echo', 'hello'])
    print(result.stdout, result.exit_code)
finally:
    await backend.close(terminate=True)
```

## Configuration

```python
from pydantic_ai_harness import ModalSandbox

ModalSandbox(
    image='python:3.12-slim',     # registry image for owned sandboxes
    sandbox_id=None,              # attach to an existing sandbox instead of creating one
    app_name='pydantic-ai-harness',  # Modal app the owned sandbox runs under
    create_app_if_missing=True,   # create the app if it does not exist
    sandbox_timeout=300,          # max lifetime (seconds) of an owned sandbox
    workdir=None,                 # working directory for commands (Modal default when None)
    env=None,                     # environment variables for an owned sandbox (dict)
    default_command_timeout=60.0, # default timeout for one run_command (seconds; fractions round up)
    max_command_timeout=None,     # hard ceiling for one command; None -> sandbox_timeout
    max_output_bytes=50 * 1024,   # per-stream payload cap in UTF-8 bytes before annotations
    max_output_lines=2000,        # per-stream payload line cap before annotations
    max_read_bytes=5 * 1024 * 1024,  # refuse read_file on files larger than this
    instructions=None,            # None: default usage instructions; '': none; str: your own
)
```

Modal enforces whole-second command deadlines, so a fractional
`default_command_timeout` or `timeout_seconds` rounds up (0.5 behaves as 1).
The default instructions state the tools, the command timeout, and its ceiling;
set `instructions=''` to add none, or pass your own text (needed when prefixing,
see below).

`read_file` loads a file fully before returning a window of it, so it refuses
files larger than `max_read_bytes` and tells the model to slice them with a shell
command (`head`, `tail`, `sed -n`, `grep`) instead. That guard reads the size from
a `stat` first and checks the returned byte count again. A file that grows
between those calls can temporarily exceed the limit in client memory before it is
rejected. The guard is not a defense against special or virtual files whose
reported size is misleading, because the filesystem API exposes no bounded read.
Use `run_command` with a bounded shell command for those paths.

`list_directory` reads the whole directory listing before capping it (Modal has
no streaming list API), so listing a directory with a very large number of
entries costs memory proportional to the entry count. Point the model at a
narrowed `run_command` (`ls | head`, `find -maxdepth`) for directories that big.

## Not yet supported

- Streaming command output to the model: `run_command` returns once the command
  finishes (or hits its deadline), not incrementally. `ModalSandboxBackend`
  implements `SupportsStream`, so a caller can stream a command it starts itself.
- Custom-built images, mounts, or `modal.Secret`: `image` takes a registry tag,
  and `env` takes plain environment variables. For anything richer, create the
  sandbox yourself with the Modal SDK and attach it via `sandbox_id`.
- Spilling full output to a file: truncated file reads end with the next
  `offset` to page from and oversized files get a shell-slice hint (`head`,
  `tail`, `sed -n`); truncated command output gets a truncation marker. Nothing
  is written to a file in the sandbox for the model to open. This is a
  deliberate choice for now.

Modal's SDK is asyncio-native, so the capability drives its async (`.aio`) API
directly and requires an asyncio event loop (it does not run under trio).

## Composing with other capabilities

Do not combine this capability with another unprefixed capability that registers
`run_command`, `read_file`, `write_file`, or `list_directory` (e.g. the Shell or
FileSystem capabilities). Pydantic AI rejects duplicate tool names. If an agent
needs both sets of tools, prefix one of the capabilities:

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

## Agent spec (YAML/JSON)

`ModalSandbox` works with Pydantic AI's
[agent spec](https://pydantic.dev/docs/ai/core-concepts/agent-spec/):

```yaml
# agent.yaml
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

## Further reading

- [Modal sandboxes](https://modal.com/docs/guide/sandbox)
- [Pydantic AI capabilities](https://pydantic.dev/docs/ai/core-concepts/capabilities/)
- [Pydantic AI toolsets](https://pydantic.dev/docs/ai/tools-toolsets/toolsets/)
- [Modal Sandbox source code](https://github.com/pydantic/pydantic-ai-harness/tree/main/pydantic_ai_harness/modal_sandbox/)
- [Pydantic AI Harness version policy](https://github.com/pydantic/pydantic-ai-harness#version-policy)
