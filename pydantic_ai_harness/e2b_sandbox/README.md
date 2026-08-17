# E2B Sandbox

`E2BSandbox` gives an agent an isolated cloud computer for running commands and
working with files. Use it for coding, tests, data processing, and other tasks
that should not execute model-generated commands on the application host.

The capability supplies the run's
[sandbox](https://pydantic.dev/docs/ai/sandbox/) from an
[E2B sandbox](https://e2b.dev/docs), and adds shell and file tools that work in
it. By default, every agent run gets a fresh sandbox created from a template,
killed when the run ends. You can also attach an existing sandbox and reuse it
across runs.

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

## Tools

| Tool | Purpose |
|---|---|
| `run_command` | Run a shell command in the sandbox. Pipes, redirection, `&&`, and globs work. Returns labelled stdout/stderr plus an exit code on failure. |
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
crosses the wire: the sandbox protocol delivers a command's whole output, and
E2B's SDK accumulates it in the command handle, so a command that floods stdout
is capped after the transfer, not during it. Bound it in-command
(`| tail -c 10000`) when the transfer itself matters. E2B decodes command output
as UTF-8 with replacement characters, so binary output is reported rather than
crashing the run.

`run_command` runs through E2B's shell; `read_file`, `write_file`, and
`list_directory` go through the sandbox filesystem API (no shell), so writes
stream the content rather than passing it as a command argument, and parent
directories are created on write. A relative path given to a file tool is
resolved against the sandbox working directory (`workdir`, or the template's own,
discovered once with `pwd` and cached) and normalized, keeping both views of the
tree consistent. Resolution is a spelling convenience, not confinement: isolation
is the sandbox's job.

Because the tools read the run's sandbox rather than owning a connection, they
also work against any other backend the run attached (a `sandbox=` run argument,
or another capability's sandbox). Behavior that only E2B reports -- the output
that preceded a deadline kill, the terminal error taxonomy -- degrades to what
that backend reports.

## How commands run

E2B executes every command through `/bin/bash -l -c`, which has two consequences
worth knowing:

- An argv sequence has no direct E2B equivalent, so `E2BSandboxBackend.run`
  quotes it with `shlex.join` into a single shell word string. The shell still
  parses the result, but the quoting keeps each element one literal word.
- Bash login startup files run before every command. Keep them silent in custom
  templates: their output is part of the command's output.

## Failure handling

Failures split into two kinds:

- **Recoverable** -- a bad path, a command that exits non-zero, a transient
  sandbox-side error. These come back to the model as a retry (`ModelRetry`) or,
  for `run_command`, as reported output it can react to. Retrying can plausibly
  work, so the run continues.
- **Terminal** -- the sandbox itself is gone (killed, or expired at its
  `sandbox_timeout`), raising `E2BSandboxUnavailableError`, or the credentials
  were rejected, raising `E2BSandboxAuthError`. Re-running the command cannot fix
  these, so the tool lets them propagate (both are `E2BSandboxTerminalError`
  subclasses) and the run ends with an actionable message instead of looping the
  model against a dead sandbox. If owned runs legitimately hit the lifetime,
  raise `sandbox_timeout`.

E2B reports an envd request the sandbox never answered as a timeout whether the
sandbox is merely slow or already gone, so a failure of that shape is classified
by asking E2B whether the sandbox is still running before deciding which of the
two kinds it is.

A command that hits its deadline raises `E2BSandboxCommandTimeoutError`, a
builtin `TimeoutError` carrying the output produced before the kill.
`run_command` turns it back into that output plus a `[timed out after Ns]` note.
A missing path raises the builtin `FileNotFoundError`, which the file tools report
as a retry.

## Sandbox lifetime

The capability implements Pydantic AI's sandbox lifecycle hooks: `create_sandbox`
provisions the environment at the start of a run, `get_sandbox` connects to it,
and `destroy_sandbox` tears it down when the run ends (including on failure).

By default the capability is **owned**: each run creates a fresh sandbox and kills
it when the run ends. Teardown waits for confirmation for a bounded period; if
E2B's control plane does not respond, `sandbox_timeout` remains the server-side
cleanup backstop. Each owned run spins up its own sandbox, so expect a cold-start
cost per run, and the sandbox is provisioned even if the model never calls a
sandbox tool.

E2B has no create-or-reuse key for sandboxes, so a durable engine that retries the
provisioning step after a crash creates a second one; the abandoned sandbox is
left to its own `sandbox_timeout`.

**Attach** to a sandbox you manage elsewhere (e.g. created via the E2B dashboard
or SDK) by id, to reuse it across runs. It is never killed by the capability:

```python
from pydantic_ai_harness.e2b_sandbox import E2BSandbox

E2BSandbox(sandbox_id='sbx-abc123', max_command_timeout=600)
```

Attaching to a **paused** sandbox resumes it: E2B treats connect as "resume if
needed", so a paused environment comes back with its filesystem intact rather
than failing the run.

The capability cannot see an attached sandbox's real lifetime, so each command
there is capped at 300s unless `max_command_timeout` raises the ceiling.

An attached sandbox is not concurrency-safe across overlapping runs: they share
one filesystem and one process space. Use separate sandboxes for runs that overlap
in time.

To control the environment yourself, create the backend and pass it to the run
instead of adding the capability's lifecycle:

```python
from pydantic_ai import Agent
from pydantic_ai_harness.e2b_sandbox import E2BSandbox, E2BSandboxBackend

backend = await E2BSandboxBackend.create(template='base', sandbox_timeout=1800)
try:
    agent = Agent(
        'anthropic:claude-sonnet-4-6',
        capabilities=[E2BSandbox(sandbox_id=backend.sandbox_id, max_command_timeout=600)],
    )
    await agent.run('clone the repo and install deps')   # same sandbox...
    await agent.run('run the test suite')                # ...reused across runs
finally:
    await backend.close(terminate=True)
```

## Cancellation

E2B's own command `timeout` bounds the output stream rather than the command: when
it expires the SDK stops listening and the command keeps running. The backend
therefore owns the deadline itself, and kills the command when it expires or when
the caller is cancelled:

- Every `run_command` carries a deadline (`default_command_timeout`, or the
  per-call `timeout_seconds`, capped by `max_command_timeout`, which defaults to
  `sandbox_timeout`). At the deadline the command is killed and the output it
  produced first is reported to the model.
- A cancelled run kills the command on its way out rather than leaving it running.
  The kill is best effort; if the request fails, the sandbox's own lifetime is the
  backstop.
- The kill signals the command's own process. A process the command started in the
  background is not reached by that signal, and lives until the sandbox is torn
  down. For an owned run that is the end of the run.
- When an owned run ends or is cancelled, the capability kills the sandbox and
  waits for a bounded period. `sandbox_timeout` remains the server-side backstop
  if the request cannot be confirmed.
- An attached sandbox is never killed by the capability (its owner controls that).

## Lower-level access

`E2BSandbox` is the main entry point. The toolset is an implementation detail.
`E2BSandboxBackend` is public: it implements Pydantic AI's
[`SandboxBackend`](https://pydantic.dev/docs/ai/sandbox/) protocol over an E2B
sandbox, including `SupportsFilesystem` and `SupportsStart`, so it can be passed
to any run or used directly.

```python
from pydantic_ai_harness.e2b_sandbox import E2BSandboxBackend

backend = await E2BSandboxBackend.create(template='base')
try:
    result = await backend.run(['echo', 'hello'])
    print(result.stdout, result.exit_code)
finally:
    await backend.close(terminate=True)
```

## Configuration

```python
from pydantic_ai_harness.e2b_sandbox import E2BSandbox

E2BSandbox(
    template=None,                # E2B template name/id for owned sandboxes; None uses E2B's default
    sandbox_id=None,              # attach to an existing sandbox instead of creating one
    sandbox_timeout=300,          # max lifetime (seconds) of an owned sandbox
    workdir=None,                 # working directory for commands (the template's own when None)
    env=None,                     # environment variables for an owned sandbox (dict)
    metadata=None,                # E2B metadata recorded on an owned sandbox (dict)
    allow_internet_access=True,   # whether an owned sandbox may reach the internet
    default_command_timeout=60.0, # default timeout for one run_command (seconds)
    max_command_timeout=None,     # hard ceiling for one command; None -> sandbox_timeout
    max_output_bytes=50 * 1024,   # per-stream payload cap in UTF-8 bytes before annotations
    max_output_lines=2000,        # per-stream payload line cap before annotations
    max_read_bytes=5 * 1024 * 1024,  # refuse read_file on files larger than this
    instructions=None,            # None: default usage instructions; '': none; str: your own
)
```

Settings that only apply when creating a sandbox (`template`, `sandbox_timeout`,
`env`, `metadata`, `allow_internet_access`) cannot be combined with `sandbox_id`;
these conflicts fail at construction instead of being ignored. `workdir` is the
exception: E2B sets the working directory per command rather than at creation, so
it still applies to an attached sandbox.

The default instructions state the tools, the command timeout, and its ceiling;
set `instructions=''` to add none, or pass your own text (needed when prefixing,
see below).

`read_file` loads a file fully before returning a window of it, so it refuses
files larger than `max_read_bytes` and tells the model to slice them with a shell
command (`head`, `tail`, `sed -n`, `grep`) instead. That guard reads the size from
a `stat` first and checks the returned byte count again. A file that grows between
those calls can temporarily exceed the limit in client memory before it is
rejected. The guard is not a defense against special or virtual files whose
reported size is misleading, because the filesystem API exposes no bounded read.
Use `run_command` with a bounded shell command for those paths.

`list_directory` reads the whole directory listing before capping it (E2B has no
streaming list API), so listing a directory with a very large number of entries
costs memory proportional to the entry count. Point the model at a narrowed
`run_command` (`ls | head`, `find -maxdepth`) for directories that big.

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
  to page from and oversized files get a shell-slice hint (`head`, `tail`,
  `sed -n`); truncated command output gets a truncation marker. Nothing is written
  to a file in the sandbox for the model to open. This is a deliberate choice for
  now.

E2B's async SDK is asyncio-native (its command handles are asyncio tasks), so the
capability requires an asyncio event loop and does not run under trio.

## Composing with other capabilities

Do not combine this capability with another unprefixed capability that registers
`run_command`, `read_file`, `write_file`, or `list_directory` (e.g. the Shell or
FileSystem capabilities). Pydantic AI rejects duplicate tool names. If an agent
needs both sets of tools, prefix one of the capabilities:

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

## Agent spec (YAML/JSON)

`E2BSandbox` works with Pydantic AI's
[agent spec](https://pydantic.dev/docs/ai/core-concepts/agent-spec/):

```yaml
# agent.yaml
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

## Further reading

- [E2B documentation](https://e2b.dev/docs)
- [E2B Python SDK](https://github.com/e2b-dev/E2B/tree/main/packages/python-sdk)
- [Pydantic AI capabilities](https://pydantic.dev/docs/ai/core-concepts/capabilities/)
- [Pydantic AI toolsets](https://pydantic.dev/docs/ai/tools-toolsets/toolsets/)
- [E2B Sandbox source code](https://github.com/pydantic/pydantic-ai-harness/tree/main/pydantic_ai_harness/e2b_sandbox/)
- [Pydantic AI Harness version policy](https://github.com/pydantic/pydantic-ai-harness#version-policy)
