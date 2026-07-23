---
title: Skills
description: Load filesystem Agent Skills as on-demand Pydantic AI capabilities, exposing each skill's instructions to the model.
---

# Skills

`Skills` loads filesystem [Agent Skills](https://agentskills.io/specification) as
[on-demand Pydantic AI capabilities](/ai/capabilities/on-demand/). The model
initially sees each skill's name and description. When it calls the core
`load_capability` tool, it receives that skill's instructions.

[Source](https://github.com/pydantic/pydantic-ai-harness/tree/main/pydantic_ai_harness/skills/)

> The API may change between releases. Where practical, breaking changes ship with a deprecation warning.

## The problem

Agent Skills use progressive disclosure: metadata helps a model choose a skill,
while the full instructions are loaded only when needed. `Skills` maps each skill
to Pydantic AI's deferred-capability primitive, so the loaded content joins the
run through the same `load_capability` flow as any other on-demand capability.

## Scope

v1 exposes instructions only. Each skill is its `SKILL.md` frontmatter (`name`,
`description`) plus the Markdown body, rendered as a deferred capability. Skills
reads `SKILL.md` when it is constructed. It does not read other bundled files,
run scripts, or grant filesystem permissions. Behavioral frontmatter fields that
other clients act on are accepted but ignored with a warning (see below).

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

`name` may be omitted, in which case the parent directory name is used. If
present, it must match the directory name. `description` is required. A
`SKILL.md` nested deeper than an immediate child directory (for example inside a
`scripts/` folder) is not a skill.

Skills are discovered once when `Skills(...)` is constructed. The skill catalog
and parsed `SKILL.md` instructions are a snapshot. Construct a new instance to
rescan.

## Usage

```python
from pydantic_ai import Agent
from pydantic_ai_harness.skills import Skills

agent = Agent(
    'anthropic:claude-sonnet-4-6',
    capabilities=[Skills('.agents/skills')],
)
```

Skills does not discover `.agents`, `.claude`, or home-directory libraries
implicitly. Pass every library the application intends to expose.

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

`include` and `exclude` match exact skill names and cannot be combined. Unknown
names fail during construction. An empty `include` exposes no skills. Selection
happens before `SKILL.md` is parsed, so unselected skills do not add validation
or prompt context to that agent.

Use `include` when the set is an access or workload policy. A skill added to the
shared directory is not exposed until the agent's allowlist includes it.

## Bundled files and filesystem access

Agent Skills may bundle files next to `SKILL.md`:

- `references/` commonly contains documentation or data to read when needed;
- `assets/` commonly contains templates or files used in generated output;
- `scripts/` commonly contains executable helpers.

Skills does not enumerate, read, or execute these files. Loaded instructions
include the absolute skill directory and expand `${CLAUDE_SKILL_DIR}` to that
path. If a body references `references/guide.md`, the model can resolve it
relative to the displayed skill directory.

The application decides whether the agent can access that path. For example,
compose Skills with `FileSystem` and choose a root that contains the skill
library:

```python
from pathlib import Path

from pydantic_ai import Agent
from pydantic_ai_harness.filesystem import FileSystem
from pydantic_ai_harness.skills import Skills

skill_library = Path('.agents/skills').resolve()

agent = Agent(
    'anthropic:claude-sonnet-4-6',
    capabilities=[
        FileSystem(
            root_dir=skill_library,
            protected_patterns=['**'],
        ),
        Skills(skill_library),
    ],
)
```

`FileSystem` includes write, edit, and create tools. Setting
`protected_patterns=['**']` makes every path under this root read-only, so those
tools reject changes while read and search operations remain available.

Without an application-provided tool that authorizes the path, the skill
receives its instructions but cannot read bundled files. Skills does not inspect
other tools or infer filesystem authority from tool names.

## Claude Code compatibility

The loader keeps the parts of the portable format that carry into an
instructions-only capability:

- `${CLAUDE_SKILL_DIR}` is replaced with the skill's absolute directory;
- `name` may be omitted and derived from the directory.

Only metadata that changes instructions is implemented. The following behavioral
frontmatter fields are accepted but ignored with one aggregated `UserWarning`
during construction:

```text
agent, allowed-tools, argument-hint, arguments, context, dependencies,
disable-model-invocation, disallowed-tools, effort, hooks, model, paths, shell,
tools, user-invocable, when_to_use
```

Fields such as `license`, `compatibility`, and `metadata` are accepted as valid
frontmatter but do not change runtime behavior. Unknown non-behavioral fields
are also accepted.

## Agent spec

Skills works with Pydantic AI's [YAML/JSON agent spec](/ai/core-concepts/agent-spec/):

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

The `skills` extra provides the same PyYAML parser used for direct construction.

## Configuration

```python {test="skip"}
Skills(
    directories: str | Path | Sequence[str | Path],
    *,
    include: Collection[str] | None = None,
    exclude: Collection[str] = (),
)
```

- `directories` accepts one skill-library path or a sequence of paths and scans
  immediate child skill packages.
- `include` exposes only the named skills.
- `exclude` hides the named skills.

Malformed required metadata, invalid or mismatched names, duplicate skill
names, unknown selections, missing roots, and non-directory roots fail during
construction. Ordinary files and child directories without `SKILL.md` are
ignored.

Every selected skill uses its frontmatter `description` and is deferred
individually. `Skills` is a composite and does not accept capability-level `id`,
`description`, or `defer_loading` options.

## Further reading

- [Agent Skills specification](https://agentskills.io/specification)
- [Adding skills support to an agent](https://agentskills.io/client-implementation/adding-skills-support)
- [Pydantic AI on-demand capabilities](/ai/capabilities/on-demand/)
- [the capabilities overview](index.md)

## API reference

::: pydantic_ai_harness.skills.Skills
