"""Agent Skills exposed as deferred Pydantic AI capabilities."""

from __future__ import annotations

import warnings
from collections.abc import Callable, Collection, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from pydantic_ai.capabilities import AbstractCapability, Capability
from pydantic_ai.tools import AgentDepsT

from pydantic_ai_harness.skills._loader import SkillDefinition, load_skill_libraries


@dataclass(init=False)
class Skills(AbstractCapability[AgentDepsT]):
    """Load filesystem Agent Skills as deferred instructions.

    Every immediate child directory containing `SKILL.md` becomes an independent
    deferred capability whose description and body come from the skill's
    frontmatter and Markdown. Pydantic AI's `load_capability` tool handles
    discovery, activation, and message-history replay. Every discovered skill is
    deferred by design; there is no option to change this.

    v1 exposes instructions only. It does not read supporting files or run
    bundled scripts. Applications can grant access to bundled files separately
    through a filesystem capability whose root contains the skill directory.
    """

    id: str | None = field(init=False, default=None, repr=False, compare=False)
    description: str | None = field(init=False, default=None, repr=False, compare=False)
    defer_loading: bool = field(init=False, default=False, repr=False, compare=False)

    directories: tuple[str | Path, ...]
    """Skill-library roots whose immediate child directories are scanned."""

    include: frozenset[str] | None
    """Exact skill names to expose, or `None` to expose every discovered skill."""

    exclude: frozenset[str]
    """Exact skill names to hide."""

    _skill_capabilities: tuple[Capability[AgentDepsT], ...] = field(init=False, repr=False, compare=False)

    def __init__(
        self,
        directories: str | Path | Sequence[str | Path],
        *,
        include: Collection[str] | None = None,
        exclude: Collection[str] = (),
    ) -> None:
        """Build an immutable snapshot of selected Agent Skills.

        Args:
            directories: One skill-library path, or a sequence of paths.
            include: Exact skill names to expose. Omit to expose all discovered skills.
            exclude: Exact skill names to hide. Cannot be combined with `include`.
        """
        if include is not None and exclude:
            raise ValueError('include and exclude cannot be used together.')

        self.id = None
        self.description = None
        self.defer_loading = False
        self.directories = self._normalize_directories(directories)
        self.include = self._normalize_selection('include', include) if include is not None else None
        self.exclude = self._normalize_selection('exclude', exclude)

        definitions = load_skill_libraries(self.directories, include=self.include, exclude=self.exclude)
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

    @staticmethod
    def _normalize_directories(
        directories: str | Path | Sequence[str | Path],
    ) -> tuple[str | Path, ...]:
        if isinstance(directories, (str, Path)):
            return (directories,)
        return tuple(directories)

    @staticmethod
    def _normalize_selection(name: str, values: Collection[object]) -> frozenset[str]:
        if isinstance(values, str):
            raise TypeError(f'{name} must be a collection of skill names, not a string.')
        normalized: set[str] = set()
        for value in values:
            if not isinstance(value, str):
                raise TypeError(f'{name} must contain only skill names as strings.')
            normalized.add(value)
        return frozenset(normalized)

    def apply(self, visitor: Callable[[AbstractCapability[AgentDepsT]], None]) -> None:
        """Expose each loaded skill as a leaf capability to Pydantic AI."""
        for capability in self._skill_capabilities:
            capability.apply(visitor)

    def _to_capability(self, skill: SkillDefinition) -> Capability[AgentDepsT]:
        return Capability[AgentDepsT](
            id=skill.name,
            description=skill.description,
            instructions=self._render_instructions(skill),
            defer_loading=True,
        )

    def _render_instructions(self, skill: SkillDefinition) -> str:
        absolute_directory = str(skill.directory)
        lines = [
            f'# Skill: {skill.name}',
            '',
            f'Skill directory: `{absolute_directory}`.',
            'Files referenced by this skill are relative to that directory.',
            'Access them only through filesystem tools provided by the application.',
        ]
        body = skill.body.replace('${CLAUDE_SKILL_DIR}', absolute_directory)
        if body:
            lines.extend(['', body])
        return '\n'.join(lines)
