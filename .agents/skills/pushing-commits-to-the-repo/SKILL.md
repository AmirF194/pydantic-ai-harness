---
name: pushing-commits-to-the-repo
description: Open and advance a PR -- write a current title and body, label it, watch CI, and
  triage every comment. Use whenever you open a PR or push a commit to one.
---

# pushing-commits-to-the-repo

Pushing starts a loop; it does not end the task. **Work stops only when CI is green AND no comment
is left unresolved.**

## When you open the PR

### Write the title and body

Follow the title and template rules in the root `AGENTS.md`.

Keep visible body content within 40 lines. Exclude template lines and collapsed `<details>`
contents from the count. For a feature or behavior change, use this order:

1. **Why we make these changes** -- State the problem and decision in a few sentences. Link the issue.
2. **New public surface** -- List each new maintained symbol. Write `none` when there is none.
3. **User-visible behavior** -- Show the smallest before-and-after example. Replace it with a
   call-path diff when the changed call chain explains the behavior; do not include both.
4. **Verification** -- Link the exact proving tests from the PR diff. Put a minimal runnable
   playground in `<details>` only when it helps reviewers reproduce the behavior.
5. **What changes for existing users** -- State the effect in one sentence. `Nothing` is valid.

Use one collapsed `<details>` section per goal only when the PR has multiple independent goals.
For a trivial PR, use the issue link, a short summary, and the test plan.

#### User-visible call-path diff

Use one fenced `diff` tree from the public entry point to the changed observable result.

- Format each node as `path/file.py :: Class.method()` or `path/file.py :: function()`.
- Indent each callee beneath its caller with `└─`. Preserve enough unchanged nodes to show each edge.
- Collapse irrelevant intermediate calls as `... unchanged machinery ...`.
- Include arguments only when they explain the change.
- Include results only on relevant leaves.
- Keep the shared caller prefix unmarked. Mark only diverging nodes, relevant arguments, or results.
- Target 12 content lines inside the fence. Never exceed 20; collapse secondary branches instead.

Apply a label -- the repo triages and filters by them. Fetch the real list first with
`gh label list --limit 100`, because the set changes and a guessed label silently fails to apply.
Pick the one naming what the PR *is* (`bug`, `enhancement`, `documentation`) and add a topic label
where one fits (`capability`, `primitive`, `agent-feature`, `core-change`, `code-mode`, `media`,
`externalization`, ...): `gh pr edit <number> --add-label <label>`.

Labelling needs triage permission on the repo. If it fails, quote the actual error rather than
concluding you lack permission.

## Before you push
- Attempt the push. If it fails, read the real error — do not preemptively decide you lack
  permission from a flag or setting.
- Leave nothing unstaged or uncommitted locally, unless the user's instructions override this.

## After you push — the loop
1. **Watch CI to a terminal state.** Don't idle. If it fails, diagnose: fix if the failure is
   yours; if it's a known flake or pre-existing on main, say so with evidence.
2. **Triage every comment** (bots and humans alike). For each one:
   - **Valid** → fix it, then reply saying what changed, and react 👍.
   - **Invalid** → reply explaining concretely why (with code evidence), and react 👎.
   - Never silently ignore a comment, and never resolve a thread without a reply.
3. **Escalate real trade-offs, don't guess.** If a comment needs a maintainer decision (a design
   choice, an API trade-off, a behavioral default), leave a comment containing: the background,
   your reasoning, the decision that needs making, the trade-offs (pros/cons of each option), and
   your recommendation. Then **poll every 30 minutes for a reply** and continue when it lands.
4. Repeat until CI is green and no comment is outstanding.

## Before handing the PR back

Run this final metadata check after CI and comments have settled:

1. Dispatch a fresh subagent that has not worked on the PR.
2. Give it the PR URL, linked issue, current `base...HEAD` diff, final test status, title, and body.
3. Ask it to check only the title and body against this section and the root `AGENTS.md`.
4. Require either `current` or an exact replacement title and body.
5. Apply every correction. Code changes restart the post-push loop; metadata-only changes do not.
6. After a replacement, repeat the check with another fresh subagent.
7. Hand the PR back only after the check reports `current`.
8. Report the human-only AI-code checkbox separately.
