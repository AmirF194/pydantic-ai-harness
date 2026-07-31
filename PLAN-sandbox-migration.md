# Port harness capabilities to `ctx.sandbox`

Draft PR against `pydantic/pydantic-ai-harness`. Depends on
[pydantic/pydantic-ai#6492](https://github.com/pydantic/pydantic-ai/pull/6492) (the
`Sandbox` protocol + `RunContext.sandbox` + `get_sandbox` capability hook).

`pydantic-ai-slim` is pinned via `[tool.uv.sources]` to the `sandbox-concept` branch
head; the pin flips back to a released version once #6492 lands. See `pyproject.toml`.

Delete this file before merge.

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
