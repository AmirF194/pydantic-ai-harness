# Skills

Load filesystem [Agent Skills](https://agentskills.io/specification) as
[on-demand Pydantic AI capabilities](https://pydantic.dev/docs/ai/capabilities/on-demand/).
The model initially sees each skill's name and description. When it calls the core
`load_capability` tool, it receives that skill's instructions.

[Source](https://github.com/pydantic/pydantic-ai-harness/tree/main/pydantic_ai_harness/skills/)

> The API may change between releases. Where practical, breaking changes ship with a deprecation warning.

## Scope

v1 exposes instructions only. Each skill is its `SKILL.md` frontmatter (`name`,
`description`) plus the Markdown body, rendered as a deferred capability. In
this release, Skills reads `SKILL.md` from the configured local paths when it is
constructed. This does not grant the model access to those paths. Skills does
not inspect other bundled files, resolve paths in the body, run scripts, or
grant filesystem permissions. Behavioral frontmatter fields that other clients
act on are accepted but ignored with a warning (see below).

## Installation

Skills uses PyYAML for frontmatter parsing:

```bash
uv add "pydantic-ai-harness[skills]"
```

## Skill layout

Pass one explicit skill-library directory or a sequence of directories. Each
immediate child directory containing `SKILL.md` is a skill:

```text
.agents/skills/
  code-review/
    SKILL.md
  release-notes/
    SKILL.md
```

```markdown
---
name: code-review
description: Review a change for correctness and repository conventions.
---

Inspect the change and report findings by severity.
```

`name` may be omitted, in which case the parent directory name is used. If present,
it must match the directory name. `description` is required. A `SKILL.md` nested
deeper than an immediate child directory (for example inside a `scripts/` folder) is
not a skill.

Skills are discovered once when `Skills(...)` is constructed. The skill catalog and
parsed `SKILL.md` instructions are a snapshot. Construct a new instance to rescan.

## Usage

```python
from pydantic_ai import Agent
from pydantic_ai_harness.skills import Skills

agent = Agent(
    'anthropic:claude-sonnet-4-6',
    capabilities=[Skills('.agents/skills')],
)
```

Skills does not discover `.agents`, `.claude`, or home-directory libraries implicitly.
Pass every library the application intends to expose.

## Select skills per agent

Use `include` when several agents share a library but each workload should see a
specific set:

```python
from pydantic_ai import Agent
from pydantic_ai_harness.skills import Skills

invoice_agent = Agent(
    'anthropic:claude-sonnet-4-6',
    capabilities=[
        Skills(
            '.agents/skills',
            include=['invoice-review', 'vat-policy'],
        )
    ],
)
```

Use `exclude` when an agent should see all but a small number of skills:

```python
from pydantic_ai import Agent
from pydantic_ai_harness.skills import Skills

support_agent = Agent(
    'anthropic:claude-sonnet-4-6',
    capabilities=[
        Skills(
            '.agents/skills',
            exclude=['deployment', 'incident-response'],
        )
    ],
)
```

`include` and `exclude` are mutually exclusive catalog-selection modes, enforced
by the typed constructor and validated at runtime for agent specs and untyped
callers. Passing both is invalid even when either collection is empty. Unknown
names fail during construction. An empty `include` exposes no skills; an empty
`exclude` has the same effect as omitting it. Selection happens before
`SKILL.md` is parsed, so unselected skills do not add validation or prompt
context to that agent.

These options control which skills appear in the deferred capability catalog.
They are not filesystem permissions or an access-control boundary.

## Bundled files and filesystem access

Agent Skills may bundle files next to `SKILL.md`:

- `references/` commonly contains documentation or data to read when needed;
- `assets/` commonly contains templates or files used in generated output;
- `scripts/` commonly contains executable helpers.

Skills does not enumerate, read, or execute these files. Relative references and
placeholders such as `${CLAUDE_SKILL_DIR}` remain unchanged in the loaded
instructions. v1 does not provide a path that the model can use to resolve them.

Future resource, asset, and script support should perform its I/O through
`RunContext.sandbox`. That keeps execution authority in the sandbox instead of
coupling Skills to model-facing filesystem or shell tools. Moving initial skill
discovery behind the sandbox also requires Pydantic AI core to make the sandbox
available before capability hooks need it.

## Claude Code compatibility

The loader keeps the parts of the portable format that carry into an
instructions-only capability. `name` may be omitted and derived from the
directory. Resource-relative paths and placeholders such as
`${CLAUDE_SKILL_DIR}` remain unchanged because v1 does not resolve bundled
files.

Only metadata that changes instructions is implemented. The following behavioral
frontmatter fields are accepted but ignored with one aggregated `UserWarning` during
construction:

```text
agent, allowed-tools, argument-hint, arguments, context, dependencies,
disable-model-invocation, disallowed-tools, effort, hooks, model, paths, shell,
tools, user-invocable, when_to_use
```

Fields such as `license`, `compatibility`, and `metadata` are accepted as valid
frontmatter but do not change runtime behavior. Unknown non-behavioral fields are also
accepted.

## Agent spec

Skills works with Pydantic AI's YAML/JSON agent spec:

```yaml
model: anthropic:claude-sonnet-4-6
capabilities:
  - Skills:
      directories: .agents/skills
      include:
        - invoice-review
        - vat-policy
```

```python
from pydantic_ai import Agent
from pydantic_ai_harness.skills import Skills

agent = Agent.from_file('agent.yaml', custom_capability_types=[Skills])
```

The `[skills]` extra provides the same PyYAML parser used for direct construction.

## Code-defined capabilities

Use Pydantic AI's core `Capability` for instructions or tools defined in Python:

```python
from pydantic_ai.capabilities import Capability

refunds = Capability(
    id='refunds',
    description='Use for refund policy questions.',
    instructions='Check the refund policy before answering.',
    defer_loading=True,
)
```

`Skills` is the package loader for portable `SKILL.md` libraries. It does not
replace the core API for code-defined capabilities.

## API

```python {test="skip"}
Skills(
    directories: str | Path | Sequence[str | Path],
    *,
    include: Collection[str] | None = None,
    exclude: Collection[str] | None = None,
)
```

- `directories` accepts one skill-library path or a sequence of paths and scans
  immediate child skill packages.
- `include` exposes only the named skills.
- `exclude` omits the named skills from the catalog.

Malformed required metadata, invalid or mismatched names, duplicate skill names,
unknown selections, missing roots, and non-directory roots fail during
construction. Ordinary files and child directories without `SKILL.md` are
ignored.

Every selected skill uses its frontmatter `description` and is always deferred
individually. This is part of the Skills design, not a configurable option: the
model sees the skill catalog first and loads a skill's full instructions only
when needed.

## Further reading

- [Agent Skills specification](https://agentskills.io/specification)
- [Adding skills support to an agent](https://agentskills.io/client-implementation/adding-skills-support)
- [Pydantic AI on-demand capabilities](https://pydantic.dev/docs/ai/capabilities/on-demand/)
