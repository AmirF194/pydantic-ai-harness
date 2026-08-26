---
name: final-review
description: Run a final review pass on a harness PR that has already been iterated on - bots have posted, threads have been resolved, and commits have landed since the first review. Use it before asking a maintainer to merge, or when a PR has changed a lot since it was last read. Do NOT use it as a first read of a fresh PR: it audits what earlier rounds claimed, so with no earlier rounds most of it has nothing to work on. Do NOT use it to re-report per-line correctness or security findings - Macroscope and Veria already post those on every PR.
---

# Final review

Audit an iterated harness PR: what earlier rounds claimed, what the diff grew
into, and what nobody ran.

Two rules decide everything below.

**Every charter declares its lane, and Lane A charters name a probe they will
run.** A charter that runs nothing and reads nothing cannot fail, so it always
reports clean.

**Reproduce a claim before accepting it and before dismissing it.** This applies
to bot findings, to resolved threads, to the PR body's own verification tables,
and to anything a previous session wrote down.

## Phase 0 - ground

Run this before writing any charter.

1. `git fetch origin main`. Record whether the branch is behind, and by how much.
2. If the branch is behind, establish every claim about main with
   `git show origin/main:<file>`. Never read main from the merge-base. A stale
   worktree produces findings about code that main already fixed.
3. Build the diff once. Strip cassettes.
4. List the iteration history: resolved review threads, dismissed bot findings,
   and commits pushed since the first review. This list is the input to the claim
   charters.

Then run the mechanical pre-pass. Each check is one command and settles a class
no model judgement improves:

- Every pinned `uses:` SHA in a changed workflow resolves:
  `gh api repos/<owner>/<repo>/commits/<sha>`.
- A guarantee sentence changed in one doc surface is changed in the other.
  README and `docs/<capability>.md` ship as a pair.
- New test files import no `_`-prefixed module member.
- The diff adds no `Any`, no `cast`, and no `getattr` by string name on a type
  the repo declares.
- Every reviewer that normally posts on a harness PR actually posted a review.
  A quota notice is not a review, and a skipped verdict is not an approval.

## Phase 1 - charter

Write 2 to 6 charters. Fewer than 2 means the PR did not need this pass. More
than 6 means the charters are too small.

Each charter states:

- the concern, in one sentence
- its lane, A or B
- the files it covers
- what would falsify it

Reject a charter that cannot state what would falsify it.

### Lane A - the charter runs code

Use Lane A when two versions of the code can disagree on a printed result.

A Lane A charter also names the two versions and the input set.

These concerns belong in Lane A. Each earned its place by changing a merged PR:

- **Guard grammar against consumer grammar.** A validator rejects what the
  downstream accepts, or forwards what it rejects. Check both directions.
- **Version and OS cells.** A branch that exists on one Python version or one OS
  is untested on the others. Coverage stays at 100% while a cell is wrong.
- **Vacuity.** Run the test that appears to guard the changed behavior against a
  reverted fix. A test that still passes guards nothing.
- **Cap accounting.** Measure the limit on the final value the caller receives,
  after framing, truncation markers, envelopes and metadata.
- **Option to consumer.** Run once per value of a new option. Print which branch
  of the consumer each value reached.
- **Positional signature compatibility.** Construct the changed dataclass
  positionally with the previous argument list. Print what binds to what.
- **Resolution compatibility.** `pydantic-ai-harness` and `pydantic-ai-slim` ship
  separately. Resolve under `--locked`, `--resolution lowest-direct`, and latest,
  and print the resolved versions. An `override-dependencies` entry replaces the
  candidate's own requirement, so a harness ceiling can veto a floor core raised
  and report no conflict.
- **Durability composition.** A capability that holds state or overrides
  `for_run` needs a public `Agent` test per supported wrapper. Assert state
  survives a replay boundary, or assert the composition fails before the first
  tool call.

Probe primitives, strongest first:

1. **Two live modules in one process.** Load main's module beside the branch's
   with `importlib.util.spec_from_file_location`, then run one input through
   both.
2. **Source swap.** `git show origin/main:<file> > <file>`, run, restore, and
   prove the tree is clean with `git status --short`.
3. **Case matrix.** N inputs by M versions. Print one row per input and flag the
   rows that differ. Print one field or one key per row. An aggregate comparison
   hides which side owns the difference.
4. **Control arm.** Add an arm that must not fail, so a failure pins to one
   component.
5. **Scratch patch.** Copy the file, apply the proposed fix to the copy, and run
   the matrix against it. Never patch the branch to test a recommendation.

### Lane B - the charter reads

Use Lane B when no probe can settle the concern. Do not invent a probe to make a
Lane B charter look like Lane A.

These concerns belong in Lane B:

- **Prose against code.** For every guarantee sentence the diff adds or leaves
  standing in a README, a `docs/` page, or a docstring: is the sentence true of
  the code as written? Docs parity tests check that both surfaces exist and
  match each other. No test checks whether either one is true. In this repo the
  documented guarantee is consistently stronger than the code, never weaker.
- **Workflow graph semantics.** `needs` against a skipped dependency,
  `always()` against `!cancelled()`, the permission ceiling a caller imposes on a
  reusable workflow, a job-level `environment` shadowing a `workflow_call`
  secret. Decide these from the workflow file. Reproducing them means waiting on
  a schedule.
- **Cross-PR interference.** Name every open PR that edits the same function.
  State which mechanism becomes redundant, whose comment rationale goes stale,
  and what a test-merge conflicts on.
- **Boundary and naming.** Whether the behavior belongs in core rather than
  harness, whether the capability name follows the repo's naming rule, and
  whether a link to an open issue carries the required comment on that issue.

### Claim charters

Always write at least one claim charter when the iteration history is not empty.

Audit each of these:

- a finding a bot marked resolved - open the resolving commit and confirm it
  addresses the finding
- a finding someone agreed to and never applied
- a thread resolved without a verifiable fix
- the PR body's own verification tables

## Phase 2 - investigate

Dispatch one agent per charter, in parallel. Give each agent its charter and
nothing else.

Instruct each agent to falsify its charter. A charter falsified with evidence is
a complete review that found nothing. An invented finding is the failure this
instruction prevents.

Never hand an investigating agent your own conclusion. An agent told the answer
confirms it.

Report a probe's result as the printed matrix, not as a summary of the matrix.

## Phase 3 - adjudicate

Dispatch one agent over all findings after every charter returns.

Assign each finding one verdict:

- **CONFIRMED** - the evidence holds and the finding is about this diff
- **RESCOPED** - the behavior is real but main owns it, not this diff. Open the
  tracking artifact in this run
- **REFUTED** - positive contradicting evidence exists
- **UNSETTLED** - the evidence is absent

REFUTED requires evidence that contradicts the finding. Absence of confirming
evidence is UNSETTLED. A wrongly refuted finding is deleted silently.

Settle provenance by running both versions. This repo has no blame map in the
review path.

## Phase 4 - report

Post one comment per run. Do not edit an earlier run's comment. A final review
audits what earlier rounds said, so earlier rounds stay readable.

Give every finding a severity and a verdict. They are independent:

- severity is BLOCKING or WARNING
- verdict is `do`, `defer`, or `skip`, judged on net value, with its reason on
  the same line

A WARNING whose verdict is `do` is a required action. A `defer` without a
tracking artifact is invalid - open the artifact or change the verdict.

Report a charter that failed as failed. An incomplete review is the honest
verdict.

## Constraints on your own behavior

- Never run repository-wide pyright, pytest, or coverage. Run targeted files and
  nodes.
- Treat 100% branch coverage as a property of combined matrix data. A local gap
  is not evidence.
- Write no em dashes, no superlatives, and no marketing adjectives in anything
  you post.
- Change no file on the branch. This pass reports.
