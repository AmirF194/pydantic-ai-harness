# Docs Conventions

How the harness's user-facing docs stay correct, consistent, and discoverable. Read this before
touching `README.md`, `docs/`, a capability README, or `docs/nav.json` — and before adding a
capability, since every capability lands with docs or fails CI.

## The Parity Contract (enforced by `tests/test_docs_parity.py`)

Every capability package must, or CI fails:

1. Have its own `README.md` (purpose-first lead, source-module link, spaced-words H1).
2. Be linked from the top-level `README.md` capability tables.
3. Have a `docs/<slug>.md` page registered in `_CAPABILITY_PAGE_META` (source module + exact H1).
4. Appear in `docs/nav.json`, and every nav slug must have a page. Nav groups nest
   (`{label, items: [...]}`); the parity test parses them recursively.

## Page Conventions

- **Purpose-first lead**: the opening paragraph says what the capability is for and when to use
  it. Lifecycle hook names (`before_model_request`, …) never appear in the lead — mechanism goes
  below the purpose.
- **H1 = the capability's spaced name** (`Code Mode`, not `CodeMode`), matching filename and nav
  label. Allowlisted ClassName exceptions live in the parity test.
- **Every page links its own source module** on GitHub.
- **Experimental framing only on ACP.** Everything else uses the README's stability note.
- **Example style**: comments one line max; outputs elided (`#> ...`) rather than long canned
  text; models are the current generation (verify against core's `_known_model_names.py` — as of
  2026-08 that means `claude-fable-5` / `gpt-5.6-sol`, not older defaults) unless a page
  intentionally pins an older one.

## The Capability Index Spans Two Repos

`README.md` and `docs/index.md` present **one capability index across core and Harness** (the
"Ships in" column): core capabilities (Web Search, MCP, Tool Search, Thinking, tool approval, …)
appear here linked to the pydantic-ai docs, and pydantic-ai's docs mirror harness entries in the
other direction.

**This mirroring is a maintenance contract.** If you add, rename, re-categorize, or remove a
capability here, check whether pydantic-ai's docs (its capabilities overview and front pages)
reference it — and open a companion PR there when they do. The same applies in reverse when core
capabilities change. Until an automated docs-review workflow enforces this, the reminder lives
here: a rename PR is not done until both repos agree.

## Categories

Capabilities are grouped by what they give the agent, not by implementation detail:
**Harnesses** (complete combined capabilities like `Coder`) · **Tools & environments** (what the
agent can touch) · **Web & research** · **Context efficiency** (how it spends the context window
— Code Mode lives here, not under execution: it changes *how* the agent executes, not *where*) ·
**Knowledge & memory** · **Delegation & planning** · **Steering & safety** · **Self-extension**.
A new capability goes in the category matching its user-facing benefit; if none fits, raise it in
the PR rather than inventing a ninth silently. Keep every table's "Ships in" column and one-line
description style intact.

## Harness Pages Specifically

A harness page (`Coder`, `Researcher`, …) must state near the top that it is a **regular combined
capability composing other capabilities** and show the blown-out equivalent `capabilities=[...]`
list — the transparency promise is part of the product. Harness classes are named like the agent
they create (`Coder`), not like capabilities; the word "preset" is not used.
