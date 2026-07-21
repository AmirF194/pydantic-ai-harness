"""A code review agent that reviews a git diff with specialist sub-reviewers.

A lead reviewer delegates to correctness and security specialists (all provably
read-only) and synthesizes their reports into a validated `ReviewReport` -- a
typed result that can go straight into CI annotations or a bot comment.

Review the working tree against a base ref:

    uv run examples/code_review_agent.py [base-ref]   # defaults to HEAD~1
"""

import os
import subprocess
import sys
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models import Model

from pydantic_ai_harness import FileSystem, Shell
from pydantic_ai_harness.subagents import SubAgent, SubAgents

DEFAULT_MODEL = os.environ.get('PYDANTIC_AI_MODEL', 'anthropic:claude-opus-4-7')


class Finding(BaseModel):
    """One review finding, anchored to a file and line."""

    file: str = Field(description='Repo-relative path of the file the finding is in.')
    line: int = Field(description='1-indexed line number the finding anchors to.')
    severity: Literal['blocker', 'important', 'minor']
    summary: str = Field(description='One-sentence statement of the defect or risk.')
    suggestion: str = Field(description='Concrete change that would resolve the finding.')


class ReviewReport(BaseModel):
    """The full review: a verdict plus findings ordered most-severe first."""

    verdict: Literal['approve', 'request_changes']
    findings: list[Finding]


LEAD_INSTRUCTIONS = """\
You are the lead reviewer for a code change.

1. Get the diff (`git diff <base>...` or as instructed) and read enough surrounding
   code to understand each change in context. Judge the diff, not the whole repo.
   The shell allows only `git` and `rg`; use the filesystem tools to read files.
2. Delegate one pass to the `correctness` reviewer and one to the `security`
   reviewer, giving each the base ref and the list of changed files.
3. Verify their findings against the code before including them: drop anything you
   cannot confirm, deduplicate, and add anything they missed.
4. Report only findings a maintainer would act on. Zero findings is a valid outcome.
"""

CORRECTNESS_BRIEF = """\
You review code changes for correctness only: logic errors, broken edge cases,
race conditions, error handling that swallows failures, and behavior that
contradicts nearby tests or docs. Read the diff and surrounding code with the
tools provided. Report each suspected defect with file, line, and a concrete
failure scenario. Do not comment on style.
"""

SECURITY_BRIEF = """\
You review code changes for security only: injection, path traversal, secrets in
code or logs, missing authorization, unsafe deserialization, and dependency
misuse. Read the diff and surrounding code with the tools provided. Report each
risk with file, line, and how it could be exploited. Do not comment on style.
"""


def build_agent(model: Model | str = DEFAULT_MODEL, workspace: Path | None = None) -> Agent[None, ReviewReport]:
    """Build the review agent for `workspace` (defaults to the current directory)."""
    workspace = workspace or Path.cwd()
    # Every path is readable, no path is writable: reviewers can't "fix" the code.
    read_only_fs = FileSystem(root_dir=workspace, protected_patterns=['**'])
    git_and_grep = Shell(cwd=workspace, denied_commands=[], allowed_commands=['git', 'rg'], default_timeout=60.0)

    def specialist(name: str, brief: str) -> SubAgent:
        return SubAgent(
            Agent(model, name=name, instructions=brief),
            max_calls=1,
        )

    return Agent(
        model,
        capabilities=[
            read_only_fs,
            git_and_grep,
            SubAgents(
                agents=[
                    specialist('correctness', CORRECTNESS_BRIEF),
                    specialist('security', SECURITY_BRIEF),
                ],
                # The specialists get the same read-only filesystem and shell as the lead.
                shared_capabilities=[read_only_fs, git_and_grep],
            ),
        ],
        instructions=LEAD_INSTRUCTIONS,
        output_type=ReviewReport,
    )


def main() -> None:
    """Review the working tree against the base ref given on the command line."""
    base = sys.argv[1] if len(sys.argv) > 1 else 'HEAD~1'
    # Refuse anything that isn't a resolvable ref before it reaches the agent's
    # prompt (and, through it, the shell's `git` invocations).
    check = subprocess.run(['git', 'rev-parse', '--verify', '--quiet', f'{base}^{{commit}}'], capture_output=True)
    if check.returncode != 0:
        raise SystemExit(f'{base!r} is not a resolvable git ref in this repository.')
    report = build_agent().run_sync(f'Review the changes since `{base}` in this repository.').output
    print(f'verdict: {report.verdict}')
    for f in report.findings:
        print(f'[{f.severity}] {f.file}:{f.line} {f.summary}\n    fix: {f.suggestion}')


if __name__ == '__main__':
    main()
