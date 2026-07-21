"""Filesystem Agent Skills composed from deferred Pydantic AI capabilities."""

from __future__ import annotations

import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from pydantic_ai.capabilities import AbstractCapability, Capability
from pydantic_ai.exceptions import UserError
from pydantic_ai.tools import AgentDepsT, RunContext

from pydantic_ai_harness.skills._loader import SkillDefinition, load_skill_libraries

_DEFAULT_READER_TOOL = 'read_file'
_DEFAULT_EXECUTOR_TOOL = 'run_command'


def _relative_to_root(directory: Path, root: str | Path) -> str | None:
    try:
        return str(directory.relative_to(Path(root).resolve()))
    except ValueError:
        return None


def _first_party_reader_path(ctx: RunContext[AgentDepsT], directory: Path) -> tuple[bool, str | None]:
    from pydantic_ai_harness.filesystem import FileSystem

    found = False
    for capability in ctx.capabilities.values():
        if isinstance(capability, FileSystem):
            found = True
            relative = _relative_to_root(directory, capability.root_dir)
            if relative is not None:
                return found, relative
    return found, None


def _first_party_executor_path(ctx: RunContext[AgentDepsT], directory: Path) -> tuple[bool, str | None]:
    from pydantic_ai_harness.shell import Shell

    found = False
    for capability in ctx.capabilities.values():
        if isinstance(capability, Shell):
            found = True
            relative = _relative_to_root(directory, capability.cwd)
            if relative is not None:
                return found, relative
    return found, None


def _tool_instruction(tool_name: str, action: str, path: str, *, absolute: bool = False) -> str:
    if absolute:
        path_direction = f'Pass absolute paths rooted at `{path}`.'
    elif path == '.':
        path_direction = 'The tool is rooted at this skill directory, so pass skill-relative paths as written.'
    else:
        path_direction = f'Prefix skill-relative paths with `{path}/` when calling the tool.'
    return f'Use `{tool_name}` to {action}. {path_direction}'


@dataclass
class Skills(AbstractCapability[AgentDepsT]):
    """Load filesystem Agent Skills through core deferred capabilities.

    Every immediate child directory containing `SKILL.md` becomes an independent
    deferred capability. Pydantic AI's `load_capability` tool handles discovery,
    activation, and message-history replay.
    """

    directories: Sequence[str | Path]
    """Skill-library roots whose immediate child directories are scanned."""

    reader_tool: str | None = None
    """Caller-declared resource reader. By default, require `FileSystem`."""

    reader_root: str | Path | None = None
    """Filesystem root for a custom `reader_tool`; paths are absolute when omitted."""

    executor_tool: str | None = None
    """Caller-declared script executor. By default, use `Shell` when present."""

    _skill_capabilities: tuple[Capability[AgentDepsT], ...] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if isinstance(self.directories, (str, Path)):
            raise TypeError('directories must be a sequence of skill-library paths, for example `["./skills"]`.')
        if self.reader_root is not None and self.reader_tool is None:
            raise ValueError('reader_root requires reader_tool because the default reader uses the `FileSystem` root.')
        self.directories = tuple(self.directories)
        definitions = load_skill_libraries(self.directories)
        ignored = [
            f'{skill.name}: {", ".join(skill.ignored_behavioral_fields)}'
            for skill in definitions
            if skill.ignored_behavioral_fields
        ]
        if ignored:
            warnings.warn(
                'Ignoring unsupported Agent Skill behavioral frontmatter fields: ' + '; '.join(ignored),
                UserWarning,
                stacklevel=2,
            )
        self._skill_capabilities = tuple(self._to_capability(skill) for skill in definitions)

    def apply(self, visitor: Callable[[AbstractCapability[AgentDepsT]], None]) -> None:
        """Expose each loaded skill as a leaf capability to Pydantic AI."""
        for capability in self._skill_capabilities:
            capability.apply(visitor)

    def _to_capability(self, skill: SkillDefinition) -> Capability[AgentDepsT]:
        def instructions(ctx: RunContext[AgentDepsT]) -> str:
            return self._render_instructions(skill, ctx)

        return Capability[AgentDepsT](
            id=skill.name,
            description=skill.description,
            instructions=instructions,
            defer_loading=True,
        )

    def _render_instructions(self, skill: SkillDefinition, ctx: RunContext[AgentDepsT]) -> str:
        absolute_directory = str(skill.directory)
        lines = [
            f'# Skill: {skill.name}',
            '',
            f'Skill directory: `{absolute_directory}`.',
            'Resolve relative paths in this skill against that directory.',
        ]

        if skill.supporting_files:
            lines.extend(['', self._reader_instruction(skill, ctx)])
        if skill.script_files:
            lines.extend(['', self._executor_instruction(skill, ctx)])

        body = skill.body.replace('${CLAUDE_SKILL_DIR}', absolute_directory)
        if body:
            lines.extend(['', body])
        return '\n'.join(lines)

    def _reader_instruction(self, skill: SkillDefinition, ctx: RunContext[AgentDepsT]) -> str:
        if self.reader_tool is not None:
            tool_name = self.reader_tool
            if self.reader_root is None:
                return _tool_instruction(
                    tool_name,
                    'read supporting skill files',
                    str(skill.directory),
                    absolute=True,
                )
            reader_path = _relative_to_root(skill.directory, self.reader_root)
            if reader_path is None:
                raise UserError(
                    f'Skill {skill.name!r} at {skill.directory} is outside configured reader_root '
                    f'{Path(self.reader_root).resolve()}.'
                )
            return _tool_instruction(tool_name, 'read supporting skill files', reader_path)

        found_reader, reader_path = _first_party_reader_path(ctx, skill.directory)
        if not found_reader:
            raise UserError(
                f'Skill {skill.name!r} contains supporting files in {skill.directory}, but no `FileSystem` '
                'capability is configured. Add `FileSystem(root_dir=...)` or configure '
                '`Skills(reader_tool=..., reader_root=...)`.'
            )
        if reader_path is None:
            raise UserError(f'Skill {skill.name!r} at {skill.directory} is outside every configured `FileSystem` root.')
        return _tool_instruction(_DEFAULT_READER_TOOL, 'read supporting skill files', reader_path)

    def _executor_instruction(self, skill: SkillDefinition, ctx: RunContext[AgentDepsT]) -> str:
        if self.executor_tool is not None:
            return _tool_instruction(
                self.executor_tool,
                'run bundled skill scripts',
                str(skill.directory),
                absolute=True,
            )

        found_executor, executor_path = _first_party_executor_path(ctx, skill.directory)
        if not found_executor:
            return (
                'Bundled scripts are present, but no script-execution tool is configured. Do not attempt to run '
                'them; continue without them or explain that script execution is required.'
            )

        if executor_path is None:
            return _tool_instruction(
                _DEFAULT_EXECUTOR_TOOL,
                'run bundled skill scripts',
                str(skill.directory),
                absolute=True,
            )
        return _tool_instruction(_DEFAULT_EXECUTOR_TOOL, 'run bundled skill scripts', executor_path)
