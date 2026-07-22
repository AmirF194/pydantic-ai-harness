# Skills

Load filesystem [Agent Skills](https://agentskills.io/specification) as
[on-demand Pydantic AI capabilities](https://pydantic.dev/docs/ai/capabilities/on-demand/).
The model initially sees each skill's name and description. When it calls the core
`load_capability` tool, it receives that skill's instructions.

[Source](https://github.com/pydantic/pydantic-ai-harness/tree/main/pydantic_ai_harness/skills/)

> The API may change between releases. Where practical, breaking changes ship with a deprecation warning.

## Scope

v1 exposes instructions only. Each skill is its `SKILL.md` frontmatter (`name`,
`description`) plus the Markdown body, rendered as a deferred capability. Skills
does not read a skill's supporting files, run bundled scripts, or expose any
assets. Behavioral frontmatter fields that other clients act on are accepted but
ignored with a warning (see below).

Resource and script support is planned for a later version, once skills sit on a
sandbox abstraction that owns file and command access.

## Installation

Skills uses PyYAML for frontmatter parsing:

```bash
uv add "pydantic-ai-harness[skills]"
```

## Skill layout

Pass one or more explicit skill-library directories. Each immediate child directory
containing `SKILL.md` is a skill:

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
    capabilities=[Skills(directories=['.agents/skills'])],
)
```

Skills does not discover `.agents`, `.claude`, or home-directory libraries implicitly.
Pass every library the application intends to expose.

## What v1 does not do

- No file reading. A skill body that references `references/guide.md` is passed to the
  model verbatim; Skills does not read that file or grant permission to read it.
- No script execution. A skill body that references `scripts/verify.py` is passed
  through unchanged; Skills does not run it.
- No assets. Skills does not enumerate a skill directory's contents.

The rendered instructions include the skill's absolute directory and expand
`${CLAUDE_SKILL_DIR}` to that path. These are cheap spec-compatibility: a portable
skill body degrades gracefully if the application happens to expose its own file
tools, but Skills itself grants no access.

## Claude Code compatibility

The loader keeps the parts of the portable format that carry into an
instructions-only capability:

- `${CLAUDE_SKILL_DIR}` is replaced with the skill's absolute directory;
- `name` may be omitted and derived from the directory.

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
      directories:
        - .agents/skills
```

```python
from pydantic_ai import Agent
from pydantic_ai_harness.skills import Skills

agent = Agent.from_file('agent.yaml', custom_capability_types=[Skills])
```

The `[skills]` extra provides the same PyYAML parser used for direct construction.

## API

```python {test="skip"}
Skills(
    directories: Sequence[str | Path],
)
```

- `directories` scans immediate child skill packages under the listed roots.

Malformed required metadata, invalid or mismatched names, duplicate skill names,
missing roots, and non-directory roots fail during construction. Ordinary files and
child directories without `SKILL.md` are ignored.

## Further reading

- [Agent Skills specification](https://agentskills.io/specification)
- [Adding skills support to an agent](https://agentskills.io/client-implementation/adding-skills-support)
- [Pydantic AI on-demand capabilities](https://pydantic.dev/docs/ai/capabilities/on-demand/)
