# Skills

Load filesystem [Agent Skills](https://agentskills.io/specification) as
[on-demand Pydantic AI capabilities](https://pydantic.dev/docs/ai/capabilities/on-demand/).
The model initially sees each skill's name and description. When it calls the core
`load_capability` tool, it receives that skill's instructions and directory.

[Source](https://github.com/pydantic/pydantic-ai-harness/tree/main/pydantic_ai_harness/skills/)

> The API may change between releases. Where practical, breaking changes ship with a deprecation warning.

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
    references/
      checklist.md
    scripts/
      verify.py
  release-notes/
    SKILL.md
```

```markdown
---
name: code-review
description: Review a change for correctness and repository conventions.
---

Read `references/checklist.md`, inspect the change, and report findings by severity.
Run `scripts/verify.py` when command execution is available.
```

`name` may be omitted, in which case the parent directory name is used. If present,
it must match the directory name. `description` is required. Files below a skill
directory are resources, not nested skills, even if a resource is named `SKILL.md`.

Skills are discovered once when `Skills(...)` is constructed. The skill catalog,
parsed `SKILL.md` instructions, and supporting-file path inventory are snapshots.
Contents of existing supporting files are read lazily and remain live. Construct a new
instance to rescan skill metadata or file paths.

## Knowledge-only skills

A skill whose instructions are fully contained in `SKILL.md` needs no filesystem tool:

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

## References and scripts

Skills reveals instructions. It does not grant filesystem or command-execution access.
Compose it with the relevant Harness capabilities when skills contain referenced files
or scripts:

```python
from pydantic_ai import Agent
from pydantic_ai_harness.filesystem import FileSystem
from pydantic_ai_harness.shell import Shell
from pydantic_ai_harness.skills import Skills

agent = Agent(
    'anthropic:claude-sonnet-4-6',
    capabilities=[
        Skills(directories=['.agents/skills']),
        FileSystem(
            root_dir='.',
            protected_patterns=[
                '.git/*',
                '.env',
                '.env.*',
                '*.pem',
                '*.key',
                '**/secrets*',
                '.agents/skills/*',
            ],
        ),
        Shell(cwd='.'),
    ],
)
```

On activation, a resource-bearing skill requires a `FileSystem` capability whose
`root_dir` contains the skill directory. Skills tells the model to use `read_file` with
the exact path prefix for that root. It raises `UserError` if `FileSystem` is missing or
the skill is outside every configured root. Merely registering another tool named
`read_file` does not establish filesystem authority. Resource contents are not loaded
eagerly.

Scripts are optional. When a `Shell` capability is present, activation instructions tell
the model how to use `run_command`. Without `Shell` or a custom executor, the model is
told not to run bundled scripts and can continue using the rest of the skill. Scripts
are supporting files, so reading one still requires `FileSystem` or a custom reader.

`CodeMode()` composes without Skills-specific routing. Skills names the operation and
path (`read_file` or `run_command`); CodeMode owns whether that operation is presented
directly or as a function inside `run_code`. This keeps wrapper presentation out of the
filesystem-authority check.

## Custom or prefixed tools

Configure the model-facing tool names when an application uses a custom reader,
prefixes tool names, or exposes a different execution environment:

```python
Skills(
    directories=['/opt/agent-skills'],
    reader_tool='workspace_read',
    reader_root='/opt',
    executor_tool='workspace_exec',
)
```

`reader_root` is the root expected by the custom reader. It requires `reader_tool`.
Skills calculates the prefix from that root to each skill directory. When it is omitted,
instructions use absolute paths.

Custom tool names are caller declarations. Skills does not inspect the final model tool
catalog to verify them because wrappers can change how tools are presented. The
application is responsible for composing a reader or executor that makes the declared
operation usable; a bad declaration surfaces when the model tries to use it.

## Claude Code compatibility

The loader preserves the filesystem behavior needed by portable Claude Code skills:

- supporting files stay relative to the skill directory;
- the original instructions decide which references and scripts to read;
- `${CLAUDE_SKILL_DIR}` is replaced with the skill's absolute directory;
- `name` may be omitted and derived from the directory.

Only metadata and behavior that can be represented safely by this capability are
implemented. The following behavioral frontmatter fields are accepted but ignored with
one aggregated `UserWarning` during construction:

```text
agent, allowed-tools, argument-hint, arguments, context, dependencies,
disable-model-invocation, disallowed-tools, effort, hooks, model, paths, shell,
tools, user-invocable, when_to_use
```

Fields such as `license`, `compatibility`, and `metadata` are accepted as valid
frontmatter but do not change runtime behavior. Unknown non-behavioral fields are also
accepted. Tool permissions remain the responsibility of the composed Pydantic AI
capabilities.

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
    reader_tool: str | None = None,
    reader_root: str | Path | None = None,
    executor_tool: str | None = None,
)
```

- `directories` scans immediate child skill packages under the listed roots.
- `reader_tool` declares a custom resource reader instead of requiring `FileSystem`.
- `reader_root` defines the path root expected by a custom reader and requires `reader_tool`.
- `executor_tool` declares a custom script executor instead of using an available `Shell`.

Malformed required metadata, invalid or mismatched names, duplicate skill names,
missing roots, and non-directory roots fail during construction. Ordinary files and
child directories without `SKILL.md` are ignored.

## Further reading

- [Agent Skills specification](https://agentskills.io/specification)
- [Adding skills support to an agent](https://agentskills.io/client-implementation/adding-skills-support)
- [Pydantic AI on-demand capabilities](https://pydantic.dev/docs/ai/capabilities/on-demand/)
