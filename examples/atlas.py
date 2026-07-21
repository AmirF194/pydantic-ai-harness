"""Atlas: build and maintain a knowledge map of a repository for agents to consult.

Atlas surveys a repo and writes a compact set of linked pages under `atlas/`
describing its subsystems, with source references -- so other agents (and
humans) can orient themselves without re-exploring the codebase from scratch.
Re-running it refreshes only what changed.

The design splits work between deterministic code and model judgment:

- **Code decides *whether* and *with what evidence* to run.** `main()` keeps a
  state file recording the last-documented commit, exits without spending tokens
  when nothing changed, and assembles the git change evidence itself.
- **The model decides *what the changes mean* for the map.** `Planning` structures
  the survey; `FileSystem` and a read-only `git_history` tool ground every page
  in real files and history.

Build or refresh the map for the current repository:

    uv run examples/atlas.py
"""

import json
import os
import subprocess
from pathlib import Path

from pydantic_ai import Agent, ModelRetry
from pydantic_ai.models import Model

from pydantic_ai_harness import FileSystem
from pydantic_ai_harness.compaction import SlidingWindow
from pydantic_ai_harness.planning import Planning

DEFAULT_MODEL = os.environ.get('PYDANTIC_AI_MODEL', 'anthropic:claude-opus-4-7')
STATE_FILE = 'atlas/.state.json'

INSTRUCTIONS = """\
You maintain `atlas/`, this repository's knowledge map: a compact set of linked
pages that gives future agents (and humans) fast, trustworthy context.

Structure:
- `atlas/overview.md` is the front door. It answers "what is this repo, and how
  is it laid out?" and links to every other page.
- Give a subsystem, workflow, or data model its own page only when there is
  enough to say about it; fold minor topics into the nearest related page
  rather than minting stubs.
- A good page answers four questions: purpose, design rationale, entry points
  (with concrete source paths), and the invariants an editor must respect.
- Where one part of the system builds on another, link to that page at the
  point in the prose where the dependency is discussed, so the link carries
  meaning rather than decorating a list.

Accuracy:
- Every statement must trace to something you actually opened: a source file,
  an existing doc, or git history. If you can't verify it, leave it out.
- Use the `git_history` tool (log, blame, show) on load-bearing files to
  recover design rationale, not just current shape.

Refresh runs (the prompt will include a summary of recent changes):
- Read the current map first, decide which pages the changes invalidate, edit
  those, and leave the rest untouched. Favor the smallest correct edit; keep
  wording that is still accurate.
- A refresh that finds nothing to update is a success: report that and stop.

Tools: survey the repo with the read-only file tools and `git_history`. Write
map pages with the `atlas_`-prefixed tools, whose paths are relative to
`atlas/` itself -- write `overview.md`, not `atlas/overview.md`. Those are the
only tools that write, and they can only reach the map.
"""


# Inspection-only git subcommands. A whole-command allowlist ('git') would also
# admit mutating subcommands like `git checkout` or `git apply`, quietly
# bypassing the write boundary below.
_GIT_READ_SUBCOMMANDS = frozenset({'log', 'show', 'blame', 'diff', 'status', 'shortlog', 'rev-parse'})

# The filesystem tools that only read.
_READ_TOOLS = frozenset({'read_file', 'list_directory', 'search_files', 'find_files', 'file_info'})


def build_agent(model: Model | str = DEFAULT_MODEL, workspace: Path | None = None) -> Agent:
    """Build the atlas agent for `workspace` (defaults to the current directory)."""
    workspace = workspace or Path.cwd()
    (workspace / 'atlas').mkdir(exist_ok=True)
    # The write boundary holds by construction, not by instruction: surveying
    # the repo uses a toolset filtered to the read-only tools, and the only
    # write-capable toolset is rooted at atlas/ (its tools carry an atlas_
    # prefix, with paths relative to that root).
    survey = FileSystem(root_dir=workspace).get_toolset().filtered(lambda ctx, tool: tool.name in _READ_TOOLS)
    map_edit = FileSystem(root_dir=workspace / 'atlas').get_toolset().prefixed('atlas')

    def git_history(subcommand: str, args: list[str]) -> str:
        """Run a read-only git inspection subcommand (log, show, blame, diff, status).

        Args:
            subcommand: One of log, show, blame, diff, status, shortlog, rev-parse.
            args: Arguments after the subcommand, e.g. `['-L', '10,20:src/x.py']`.
        """
        if subcommand not in _GIT_READ_SUBCOMMANDS:
            raise ModelRetry(f'git {subcommand!r} is not available; choose from {sorted(_GIT_READ_SUBCOMMANDS)}.')
        if any(a.startswith('--output') for a in args):
            raise ModelRetry('--output is not available; read the command output directly.')
        result = subprocess.run(
            ['git', '--no-pager', subcommand, *args], cwd=workspace, capture_output=True, text=True, timeout=30
        )
        if result.returncode != 0:
            raise ModelRetry(f'git {subcommand} failed: {result.stderr.strip()[:500]}')
        return result.stdout

    return Agent(
        model,
        capabilities=[
            Planning(),
            SlidingWindow(max_tokens=150_000, keep_messages=40),
        ],
        toolsets=[survey, map_edit],
        tools=[git_history],
        instructions=INSTRUCTIONS,
    )


def _git(workspace: Path, *args: str) -> str:
    result = subprocess.run(
        ['git', '--no-pager', *args], cwd=workspace, capture_output=True, text=True, timeout=30, check=False
    )
    if result.returncode != 0:
        # Fail loudly (e.g. not a git repository) instead of feeding the agent
        # empty evidence and silently rebuilding the map on every run.
        raise SystemExit(f'git {args[0]} failed in {workspace}: {result.stderr.strip()}')
    return result.stdout.strip()


def change_evidence(workspace: Path, last_commit: str | None) -> str:
    """Summarize what changed since the last run, computed here so the model starts from facts."""
    if last_commit:
        commits = _git(workspace, 'log', '--oneline', '--name-only', f'{last_commit}..HEAD')
    else:
        commits = _git(workspace, 'log', '--oneline', '--name-only', '-25')
    pending = _git(workspace, 'status', '--porcelain')
    return (
        f'Recent commits and the files they touched:\n{commits or "(none)"}\n\n'
        f'Uncommitted work in progress:\n{pending or "(none)"}'
    )


def main() -> None:
    """Build the map on first run; on later runs, refresh it only if the repo changed."""
    workspace = Path.cwd()
    state_path = workspace / STATE_FILE
    state: dict[str, str] = json.loads(state_path.read_text()) if state_path.exists() else {}
    head = _git(workspace, 'rev-parse', 'HEAD')

    # The zero-token gate: clean tree and unchanged HEAD means there is nothing to do.
    if state.get('commit') == head and not _git(workspace, 'status', '--short'):
        print('Knowledge map is current; nothing to do.')
        return

    if state:
        prompt = 'Refresh the knowledge map under `atlas/` for this repository.\n\n' + change_evidence(
            workspace, state.get('commit')
        )
    else:
        prompt = 'Build the initial knowledge map under `atlas/` for this repository.'

    result = build_agent(workspace=workspace).run_sync(prompt)
    state_path.parent.mkdir(exist_ok=True)
    state_path.write_text(json.dumps({'commit': head}, indent=2) + '\n')
    print(result.output)


if __name__ == '__main__':
    main()
