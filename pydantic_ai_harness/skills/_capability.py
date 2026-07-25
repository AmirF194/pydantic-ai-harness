"""Load Agent Skill instructions as deferred Pydantic AI capabilities."""

from __future__ import annotations

import warnings
from collections.abc import Callable, Collection, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import overload

from pydantic_ai.capabilities import AbstractCapability, Capability
from pydantic_ai.tools import AgentDepsT

from pydantic_ai_harness.skills._loader import SkillDefinition, load_skill_libraries


@dataclass(init=False)
class Skills(AbstractCapability[AgentDepsT]):
    """Load a library of Agent Skill instructions on demand.

    Each selected `SKILL.md` becomes a deferred capability. The model sees its
    name and description in the initial catalog, then receives its Markdown body
    after calling Pydantic AI's `load_capability` tool.

    Libraries are scanned once when this class is constructed. Only immediate
    child directories are discovered. This release reads `SKILL.md` only; it
    does not load bundled files, resolve resource paths, or run scripts.
    """

    id: str | None = field(init=False, default=None, repr=False, compare=False)
    description: str | None = field(init=False, default=None, repr=False, compare=False)
    defer_loading: bool = field(init=False, default=False, repr=False, compare=False)

    directories: tuple[str | Path, ...]
    """Skill-library paths scanned during construction."""

    include: frozenset[str] | None
    """Exact skill names to expose, or `None` to expose all discovered skills."""

    exclude: frozenset[str]
    """Exact skill names to omit from the deferred capability catalog."""

    _skill_capabilities: tuple[Capability[AgentDepsT], ...] = field(init=False, repr=False, compare=False)

    @overload
    def __init__(  # pragma: no cover - overload is enforced by static type checking
        self,
        directories: str | Path | Sequence[str | Path],
        *,
        include: Collection[str],
        exclude: None = None,
    ) -> None: ...

    @overload
    def __init__(  # pragma: no cover - overload is enforced by static type checking
        self,
        directories: str | Path | Sequence[str | Path],
        *,
        include: None = None,
        exclude: Collection[str] | None = None,
    ) -> None: ...

    def __init__(
        self,
        directories: str | Path | Sequence[str | Path],
        *,
        include: Collection[str] | None = None,
        exclude: Collection[str] | None = None,
    ) -> None:
        """Build a snapshot of the selected Agent Skills.

        Args:
            directories: One skill-library path or a sequence of paths.
            include: Exact names to expose. Omit to expose all discovered skills.
            exclude: Exact names to omit. Cannot be combined with `include`.
        """
        if include is not None and exclude is not None:
            raise ValueError('include and exclude cannot be used together.')

        self.id = None
        self.description = None
        self.defer_loading = False
        self.directories = self._normalize_directories(directories)
        self.include = self._normalize_selection('include', include) if include is not None else None
        self.exclude = self._normalize_selection('exclude', exclude) if exclude is not None else frozenset()

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
            instructions=f'# Skill: {skill.name}\n\n{skill.body}' if skill.body else f'# Skill: {skill.name}',
            defer_loading=True,
        )
