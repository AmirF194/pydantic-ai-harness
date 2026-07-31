# Port harness capabilities to `ctx.sandbox`

Draft PR against `pydantic/pydantic-ai-harness`. Depends on
[pydantic/pydantic-ai#6492](https://github.com/pydantic/pydantic-ai/pull/6492) (the
`Sandbox` protocol + `RunContext.sandbox` + `get_sandbox` capability hook).

`pydantic-ai-slim` is pinned via `[tool.uv.sources]` to the `sandbox-concept` branch
head; the pin flips back to a released version once #6492 lands. See `pyproject.toml`.

Delete this file before merge.

## Progress

Migrated so far (each with tests green + docstring covering the sandbox contract):

- `filesystem` -- tool methods take `ctx: RunContext[AgentDepsT]` as first arg; every
  read/write/stat routes through `ctx.sandbox.fs`. No `sandbox=` constructor kwarg
  and no toolset-owned sandbox state; sandbox lifetime is tied to the run.
- `shell` -- same shape (`ctx` first arg, `sandbox = ctx.sandbox`). `for_run` retained
  only for `_cwd`/`_background` per-run isolation. `persist_cwd` marker parsing fix,
  `env -i` strict-env wrap for backends that overlay host env (e.g. `LocalSandbox`).
- `pydantic_ai_docs` -- `read_pyai_docs` takes `ctx` and reads from `ctx.sandbox.fs`;
  remote fallback unchanged.
- `experimental/acp` client toolsets -- `AcpFileSystemToolset.write_file` takes `ctx`
  and threads it through to any configured local writer; the `for_run` / bind-sandbox
  dance is gone.
- `repo_context` -- walk-up discovery reads `ctx.sandbox.fs` from `before_run` and
  `wrap_tool_execute`. No toolset-owned sandbox state.

Remaining (in migration-table order):

- Storage variants: `SandboxOverflowStore`, `SandboxMediaStore`, `SandboxStepStore`,
  `SandboxMemoryStore` -- each ships alongside the existing `Disk*` store, and the
  matching capability's default flips to the sandbox-backed variant via
  `default_state_dir(sandbox, '<capability>')` resolved in `before_run` /
  `for_run`.
- `subagents/_disk.py` -- markdown definitions currently loaded synchronously at
  capability construction; needs deferral to `before_run` so file reads can consult
  `ctx.sandbox`. Non-trivial because `_by_name` and `get_instructions` depend on the
  loaded roster.
- `capability_creation` (formerly `runtime_authoring`) -- writes through
  `ctx.sandbox.fs`; validation stays local.
- README and `docs/<capability>.md` updates for each migrated capability.
- Adversarial review rounds across correctness / api-consistency / security /
  test-coverage.

## Motivation

Once `ctx.sandbox` is the framework's single execution surface, every capability that
touches files or spawns processes has one place to go. Today each does it differently:
`filesystem` uses `pathlib`, `shell` uses `anyio.open_process`, six store dataclasses
hard-code host paths through `pathlib`, `context`/`docs` load files at capability
construction. This PR routes all of them through `ctx.sandbox`.

## Design

### One helper, one convention

`pydantic_ai_harness/_sandbox_state.py` exports one function:

```python
async def default_state_dir(sandbox: Sandbox, capability: str) -> str:
    """Return `<sandbox.working_dir()>/.pydantic-ai-harness/<capability>/`."""
```

Every capability that persists state calls this to compute its default location. Users
override per capability by passing an explicit path. Users who want ephemeral behavior
override `wrap_run` and `mktemp -d` on their own — the framework already gives them
that hook and it already has `ctx.sandbox` live.

### No new core concept

We do **not** propose `Sandbox.scratch_dir()` upstream. The composable answer already
exists: `wrap_run` receives `ctx.sandbox` (verified against `capabilities/abstract.py`
in the sandbox-concept branch).

## Migration table

Every capability that touches host FS or spawns processes today.

| Capability | Today | After |
|---|---|---|
| `filesystem` | `pathlib` reads/writes/globs, `os.path.realpath` containment | `ctx.sandbox.fs.*`, `ctx.sandbox.read_file`; recursive walkers shell out via `ctx.sandbox.run('find ...' / 'grep ...')` with `list_dir` recursion fallback. Allow/deny/protected globs, binary detection, content-hash concurrency preserved. Symlink-realpath containment dropped (was host-only; isolation is the sandbox's job per PR #6492). |
| `shell` | `anyio.open_process`, `tempfile.mkstemp` for cwd capture | `ctx.sandbox.run`/`ctx.sandbox.start`. `persist_cwd` overlays cwd on top of `sandbox.working_dir()`. Denied commands/operators/env-strip preserved. |
| `context` | `Path(...).read_text()` at construction | Async load inside `before_run` via `ctx.sandbox.read_file`. Loaded contents cached on the per-run capability. |
| `docs` | Same as `context` | Same treatment. |
| `overflowing_tool_output` | `DiskOverflowStore` writes to `tempfile.gettempdir() / 'pyai_harness_overflow'` | New `SandboxOverflowStore` variant writes through `ctx.sandbox.fs` at `default_state_dir(sandbox, 'overflow')`. Capability default flips. `DiskOverflowStore` kept for explicit host-path users. |
| `media` | `DiskMediaStore` writes under a required `directory` | New `SandboxMediaStore` mirrors the same API on `ctx.sandbox.fs`. Metadata sidecar (`.meta.json`) written the same way. Capability default flips. `DiskMediaStore` kept. |
| `step_persistence` | `FileStepStore` writes under a configurable root | New `SandboxStepStore` variant. Traversal guards preserved. `FileStepStore` kept. |
| `memory` | `FilesystemMemoryStore` (host path) | New `SandboxMemoryStore` implementing the async `MemoryStore` protocol on `ctx.sandbox.fs`. Version-based optimistic concurrency preserved. `FilesystemMemoryStore` kept. |
| `subagents/_disk.py` | Local disk state | Route through `ctx.sandbox.fs` at `default_state_dir(sandbox, 'subagents')`. |
| `runtime_authoring` | Writes runtime files with `Path.write_text` | Writes through `ctx.sandbox.fs`. Validation stays local. |

`modal_sandbox` is already a sandbox provider (contributes `get_sandbox`); nothing to
port. Guardrails, planning, memory-search, and the other pure-instruction capabilities
don't touch FS today; nothing to port.

## Follow-up (not in this PR)

`code_mode`'s `mount` + `os_access` route through `pydantic-monty`. Making
`CodeMode` consume `ctx.sandbox` needs monty to accept a `Sandbox` value and route
its own OS/mount calls through it. Not attempted here. Filed as harness follow-up
issue and referenced in the PR body; monty issue filed against
`pydantic/pydantic-monty` separately.

## Testing

Every migrated capability keeps its existing test suite (which now runs against the
sandbox-backed path via the framework's default `LocalSandbox`). New stores
(`SandboxOverflowStore`, `SandboxMediaStore`, `SandboxStepStore`, `SandboxMemoryStore`)
each get their own test module hitting a `LocalSandbox` under `tmp_path`. 100% branch
coverage per harness house rules.

## What we deliberately do NOT do

- Redesign store abstractions. Every existing `DiskXxxStore` stays. Only the default
  changes; users with explicit host paths are unaffected.
- Add fallbacks or backwards-compatibility shims. The pin swap on #6492's merge is a
  one-line pyproject change; no runtime detection.
- Widen fixes beyond the stated capabilities. Anything not in the migration table
  above is out of scope.

## Cutlines

If review pushes back:

- **Symlink-realpath containment dropped in `filesystem`** — restore by shelling out
  via `sandbox.run('readlink -f')` before every access. Adds a subprocess per op.
- **`search_files`/`find_files` shelling out to `rg`/`find`** — fall back purely to
  `list_dir` recursion. Slower but no sandbox dependency on host tooling.
- **Store default flip** — leave existing defaults unchanged; make the sandbox-backed
  variant opt-in.
