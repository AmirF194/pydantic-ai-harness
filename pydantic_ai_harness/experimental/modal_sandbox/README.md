# ModalSandbox

Give an agent an isolated, ephemeral cloud sandbox, powered by
[Modal](https://modal.com), to run commands and manage files in without
touching the host.

## The problem

Agents that write and run code need somewhere safe to do it. Running
model-generated commands on the host machine is risky; spinning up and tearing
down isolated environments by hand is boilerplate. You want the agent to get a
clean container, use it for a task, and have it disposed of automatically.

## The solution

`ModalSandbox` gives the agent shell and file tools wired to a
[Modal sandbox](https://modal.com/docs/guide/sandbox). By default each run gets a
fresh sandbox created from an image and terminated when the run ends; the
container is the isolation boundary.

```python
from pydantic_ai import Agent
from pydantic_ai_harness.experimental.modal_sandbox import ModalSandbox

agent = Agent('anthropic:claude-sonnet-4-6', capabilities=[ModalSandbox()])

result = agent.run_sync('Write a Python script that prints the first 10 primes and run it.')
print(result.output)
```

## Setup

Install the `modal` extra and provide Modal credentials in the environment:

```bash
pip install "pydantic-ai-harness[modal]"
export MODAL_TOKEN_ID=...      # from `modal token new`
export MODAL_TOKEN_SECRET=...
```

The capability authenticates from those standard environment variables — the
same ones the Modal CLI and SDK use.

## Tools

| Tool | Purpose |
|---|---|
| `run_command` | Run a shell command (`sh -c`) in the sandbox. Pipes, redirection, `&&`, and globs work. Returns labelled stdout/stderr plus an exit code on failure. |
| `read_file` | Read a text file from the sandbox. |
| `write_file` | Write text to a file (creating parent directories). |
| `list_directory` | List a directory's entries (directories shown with a trailing `/`). |

Output is labelled with `[stdout]` / `[stderr]` markers and an `[exit code: N]`
line on non-zero exit. Output is capped by two limits, whichever is hit first:
`max_output_bytes` (UTF-8 bytes) and `max_output_lines` (lines). For commands the
**tail** is kept, so errors survive truncation; file reads keep the head and
return the next `offset` to page from. A non-zero exit from `run_command` is reported, not
raised, so the model can react to it; file-tool failures (missing path, etc.)
come back as a retry prompt.

The cap also bounds memory, not just what the model sees: a command that floods
`stdout` has only its last `max_output_bytes` retained client-side (whole output
chunks are dropped from the front, so a multi-byte character is never split), so
runaway output cannot exhaust memory.

## Failure handling

Failures split into two kinds:

- **Recoverable** -- a bad path, a command that exits non-zero, a transient
  sandbox-side error. These come back to the model as a retry (`ModelRetry`) or,
  for `run_command`, as reported output it can react to. Retrying can plausibly
  work, so the run continues.
- **Terminal** -- the sandbox itself is gone (terminated, or expired at its
  `sandbox_timeout`) or the credentials were rejected. Re-running the command
  cannot fix these, so the tool raises `ModalSandboxUnavailableError` /
  `ModalSandboxTerminalError` and the run ends with an actionable message instead
  of looping the model against a dead sandbox. If owned runs legitimately hit the
  lifetime, raise `sandbox_timeout`.

## Sandbox lifetime

By default the capability is **owned**: each run creates a fresh sandbox and
terminates it when the run ends, so runs are isolated and nothing leaks. Because
each owned run spins up its own sandbox, expect a cold-start cost per run; reuse a
sandbox across runs when you want to avoid it. There are two ways to reuse one.

**Attach** to a sandbox you manage elsewhere (e.g. created via the Modal CLI) by
id. It is never terminated by the capability:

```python
ModalSandbox(sandbox_id='sb-abc123')   # attach to an existing sandbox
```

**Inject a session** you own to reuse one sandbox across runs while controlling
its lifetime yourself. The capability uses the session but never opens or
terminates it, so the owner decides when the sandbox goes away, and can read its
`sandbox_id`:

```python
from pydantic_ai import Agent
from pydantic_ai_harness.experimental.modal_sandbox import ModalSandbox, ModalSandboxSession

async with ModalSandboxSession(image='python:3.12-slim') as session:
    print(session.sandbox_id)   # the running sandbox id
    agent = Agent('anthropic:claude-sonnet-4-6', capabilities=[ModalSandbox(session=session)])
    await agent.run('clone the repo and install deps')   # same sandbox...
    await agent.run('run the test suite')                # ...reused across runs
# the session and its sandbox are torn down here, by the code that owns them
```

A reused sandbox (attach or injected session) is not concurrency-safe across
overlapping runs: they share one filesystem and one process space. Use separate
sandboxes for runs that overlap in time.

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
- An owned sandbox is terminated when its run ends or is cancelled; Modal tears
  it down asynchronously, which also stops anything still running in it.
- An attached or injected sandbox is never terminated by the capability (its
  owner controls that), so an in-flight command there is bounded only by its
  deadline.

`ModalSandbox` is the supported entry point. The capability is built in two
layers -- a session that owns the sandbox mechanism (commands, file access,
lifecycle) and a toolset that presents it to the model -- kept separate so the
internals can change without affecting the tools. The session is also usable on
its own as a lower-level async context manager:

```python
from pydantic_ai_harness.experimental.modal_sandbox import ModalSandboxSession

async with ModalSandboxSession(image='python:3.12-slim') as session:
    result = await session.exec(['echo', 'hello'])
    print(result.stdout, result.returncode)
```

## Configuration

```python
ModalSandbox(
    image='python:3.12-slim',     # registry image for owned sandboxes
    sandbox_id=None,              # attach to an existing sandbox instead of creating one
    session=None,                 # reuse a ModalSandboxSession you own across runs
    app_name='pydantic-ai-harness',  # Modal app the owned sandbox runs under
    create_app_if_missing=True,   # create the app if it does not exist
    sandbox_timeout=300,          # max lifetime (seconds) of an owned sandbox
    workdir=None,                 # working directory for commands (Modal default when None)
    env=None,                     # environment variables for an owned sandbox (dict)
    default_command_timeout=60.0, # default timeout for one run_command (seconds)
    max_command_timeout=None,     # hard ceiling for one command; None -> sandbox_timeout
    max_output_bytes=50 * 1024,   # output cap returned to the model (UTF-8 bytes)
    max_output_lines=2000,        # output cap in lines; whichever cap is hit first wins
    max_read_bytes=5 * 1024 * 1024,  # refuse read_file on files larger than this
    include_instructions=True,    # add usage instructions to the prompt
)
```

`read_file` loads a file fully before returning a window of it, so it refuses
files larger than `max_read_bytes` and tells the model to slice them with a shell
command (`head`, `tail`, `sed -n`, `grep`) instead. That guard reads the size from
a `stat` first, so it bounds regular files with an honest, stable size. It is not
a defense against a special or virtual file (a device like `/dev/zero`, a FIFO)
whose reported size is misleading: Modal's filesystem API has no bounded read, so
such a path would still transfer unboundedly. Use `run_command` for those. The
worst case is bounded by the owned sandbox's `sandbox_timeout` (which reaps it),
but it can still cost memory within that window, so keep the file tools to real
files and reach for a shell command otherwise.

`list_directory` reads the whole directory listing before capping it (Modal has
no streaming list API), so listing a directory with a very large number of
entries costs memory proportional to the entry count. Point the model at a
narrowed `run_command` (`ls | head`, `find -maxdepth`) for directories that big.

## Not yet supported

- Streaming command output: `run_command` returns once the command finishes (or
  hits its deadline), not incrementally.
- Custom-built images, mounts, or `modal.Secret`: `image` takes a registry tag,
  and `env` takes plain environment variables. For anything richer, create the
  sandbox yourself with the Modal SDK and pass it via `sandbox_id` or `session`.
- Spilling full output to a file: when command output or a file read is
  truncated, the model is pointed at a shell slice (`head`, `tail`, `sed -n`,
  `offset`) rather than the whole output being written to a file in the sandbox
  for it to open. This is a deliberate choice for now; the shell-slice hint
  covers the same need without managing temporary files.

Modal's SDK is asyncio-native, so the capability drives its async (`.aio`) API
directly and requires an asyncio event loop (it does not run under trio).
`run_command` runs through `sh -c`; `read_file`, `write_file`, and
`list_directory` use Modal's filesystem API directly (no shell), so writes stream
the content rather than passing it as a command argument and `write_file` creates
parent directories. Modal's filesystem API only accepts absolute paths, so a
relative path given to a file tool is resolved against the working directory used
by `run_command` (queried once with `pwd` and cached), keeping both views of the
tree consistent.

## Agent spec (YAML/JSON)

`ModalSandbox` works with Pydantic AI's
[agent spec](https://ai.pydantic.dev/agent-spec/):

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
from pydantic_ai_harness.experimental.modal_sandbox import ModalSandbox

agent = Agent.from_file('agent.yaml', custom_capability_types=[ModalSandbox])
```

## Further reading

- [Modal sandboxes](https://modal.com/docs/guide/sandbox)
- [Pydantic AI capabilities](https://ai.pydantic.dev/capabilities/)
- [Toolsets](https://ai.pydantic.dev/toolsets/)
