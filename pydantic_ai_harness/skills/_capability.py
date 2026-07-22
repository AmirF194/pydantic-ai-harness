"""Agent Skills exposed as deferred Pydantic AI capabilities."""

from __future__ import annotations

import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from pydantic_ai.capabilities import AbstractCapability, Capability
from pydantic_ai.tools import AgentDepsT

from pydantic_ai_harness.skills._loader import SkillDefinition, load_skill_libraries


@dataclass
class Skills(AbstractCapability[AgentDepsT]):
    """Load filesystem Agent Skills as deferred instructions.

    Every immediate child directory containing `SKILL.md` becomes an independent
    deferred capability whose description and body come from the skill's
    frontmatter and Markdown. Pydantic AI's `load_capability` tool handles
    discovery, activation, and message-history replay.

    v1 exposes instructions only. It does not read supporting files or run
    bundled scripts. Resource and script support returns once skills sit on a
    sandbox abstraction.
    """

    directories: Sequence[str | Path]
    """Skill-library roots whose immediate child directories are scanned."""

    _skill_capabilities: tuple[Capability[AgentDepsT], ...] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if isinstance(self.directories, (str, Path)):
            raise TypeError('directories must be a sequence of skill-library paths, for example `["./skills"]`.')
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
        ]
        body = skill.body.replace('${CLAUDE_SKILL_DIR}', absolute_directory)
        if body:
            lines.extend(['', body])
        return '\n'.join(lines)
