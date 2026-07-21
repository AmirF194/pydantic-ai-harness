# Skills

Load filesystem [Agent Skills](https://agentskills.io/specification) as
[on-demand Pydantic AI capabilities](https://pydantic.dev/docs/ai/capabilities/on-demand/).
The model initially sees each skill's name and description. When it calls the core
`load_capability` tool, it receives that skill's instructions and directory.

[Source](https://github.com/pydantic/pydantic-ai-harness/tree/main/pydantic_ai_harness/skills/)

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

Skills are discovered once when `Skills(...)` is constructed. Adding, removing, or
editing files does not change that instance's catalog or resource snapshot. Construct
a new instance to rescan.

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

On activation, a resource-bearing skill checks that `read_file` is available, either
directly or through `CodeMode`, and tells the model the exact path prefix to use. It
raises `UserError` if no compatible reader exists or the skill is outside the detected
`FileSystem` root. The resource contents are not loaded eagerly.

Scripts are optional. When `run_command` is available, the activation instructions tell
the model how to invoke it. Without an executor, the model is told not to run bundled
scripts and can continue using the rest of the skill. Setting `executor_tool` makes a
missing executor an activation error.

`CodeMode()` is supported. Skills tells the model to call `read_file(...)` and
`run_command(...)` inside `run_code` when those tools are sandboxed there.

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

`reader_root` is the root expected by the reader tool. Skills calculates the prefix from
that root to each skill directory. When it is omitted for a custom reader, instructions
use absolute paths. The configured tool names must be visible to the model, either
directly or as functions in `run_code`.

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
- `reader_tool` overrides the default `read_file` resource reader.
- `reader_root` defines the path root expected by a custom reader.
- `executor_tool` overrides the optional default `run_command` script executor.

Malformed required metadata, invalid or mismatched names, duplicate skill names,
missing roots, and non-directory roots fail during construction. Ordinary files and
child directories without `SKILL.md` are ignored.

## Further reading

- [Agent Skills specification](https://agentskills.io/specification)
- [Adding skills support to an agent](https://agentskills.io/client-implementation/adding-skills-support)
- [Pydantic AI on-demand capabilities](https://pydantic.dev/docs/ai/capabilities/on-demand/)
