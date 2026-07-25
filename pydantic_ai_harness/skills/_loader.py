"""Discover and parse Agent Skill packages from local libraries."""

from __future__ import annotations

import re
from collections.abc import Collection, Hashable, Sequence
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError, field_validator

_SKILL_NAME_PATTERN = re.compile(r'[a-z0-9]+(?:-[a-z0-9]+)*')
_FRONTMATTER_TA = TypeAdapter(dict[str, object])

# These fields affect invocation, permissions, model selection, execution, or
# prompt rendering in clients that implement them. Skills accepts their files
# for compatibility but reports that the behavior is not active.
BEHAVIORAL_FRONTMATTER_FIELDS = frozenset(
    {
        'agent',
        'allowed-tools',
        'argument-hint',
        'arguments',
        'context',
        'dependencies',
        'disable-model-invocation',
        'disallowed-tools',
        'effort',
        'hooks',
        'model',
        'paths',
        'shell',
        'tools',
        'user-invocable',
        'when_to_use',
    }
)


class _SkillFrontmatter(BaseModel):
    model_config = ConfigDict(extra='allow')

    name: str | None = None
    description: str = Field(min_length=1, max_length=1024)

    @field_validator('name', 'description', mode='after')
    @classmethod
    def _strip_scalar(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        if not stripped:
            raise ValueError('must not be empty')
        return stripped


@dataclass(frozen=True)
class SkillDefinition:
    """Validated skill metadata and body from a single `SKILL.md`."""

    name: str
    description: str
    body: str
    ignored_behavioral_fields: tuple[str, ...]


def _extract_frontmatter(text: str, source: Path) -> tuple[str, str]:
    lines = text.splitlines()
    if not lines or lines[0] != '---':
        raise ValueError(f'{source} must start with YAML frontmatter delimited by `---`.')

    closing = next((index for index, line in enumerate(lines[1:], start=1) if line == '---'), None)
    if closing is None:
        raise ValueError(f'{source} has unclosed YAML frontmatter.')

    frontmatter = '\n'.join(lines[1:closing])
    body = '\n'.join(lines[closing + 1 :]).strip()
    return frontmatter, body


def _parse_yaml(frontmatter: str, source: Path) -> dict[str, object]:
    try:
        import yaml
    except ImportError:  # pragma: no cover - exercised in an environment without the skills extra
        raise ImportError(
            'PyYAML is required to load Agent Skills. Install it with: pip install "pydantic-ai-harness[skills]"'
        ) from None

    class UniqueKeyLoader(yaml.SafeLoader):
        def construct_mapping(self, node: yaml.MappingNode, deep: bool = False) -> dict[Hashable, object]:
            keys: set[str] = set()
            for key_node, _ in node.value:
                if not isinstance(key_node, yaml.ScalarNode):
                    raise yaml.constructor.ConstructorError(
                        'while constructing a mapping',
                        node.start_mark,
                        'found a non-scalar key',
                        key_node.start_mark,
                    )
                key: str = key_node.value
                if key in keys:
                    raise yaml.constructor.ConstructorError(
                        'while constructing a mapping',
                        node.start_mark,
                        f'found duplicate key {key!r}',
                        key_node.start_mark,
                    )
                keys.add(key)
            return super().construct_mapping(node, deep=deep)

    try:
        parsed = yaml.load(frontmatter, Loader=UniqueKeyLoader)
    except yaml.YAMLError as exc:
        raise ValueError(f'Invalid YAML frontmatter in {source}: {exc}') from exc

    try:
        return _FRONTMATTER_TA.validate_python(parsed)
    except ValidationError as exc:
        raise ValueError(f'YAML frontmatter in {source} must be a mapping.') from exc


def _validate_name(name: str, source: Path) -> None:
    if len(name) > 64 or _SKILL_NAME_PATTERN.fullmatch(name) is None:
        raise ValueError(
            f'Invalid skill name {name!r} in {source}; expected at most 64 lowercase letters, numbers, and '
            'single hyphens, without a leading or trailing hyphen.'
        )


def load_skill(skill_file: Path) -> SkillDefinition:
    """Load one `SKILL.md` into a validated construction-time definition."""
    frontmatter_text, body = _extract_frontmatter(skill_file.read_text(encoding='utf-8'), skill_file)
    raw_frontmatter = _parse_yaml(frontmatter_text, skill_file)
    try:
        frontmatter = _SkillFrontmatter.model_validate(raw_frontmatter)
    except ValidationError as exc:
        raise ValueError(f'Invalid Agent Skill frontmatter in {skill_file}: {exc}') from exc

    directory_name = skill_file.parent.name
    name = frontmatter.name or directory_name
    _validate_name(name, skill_file)
    if frontmatter.name is not None and name != directory_name:
        raise ValueError(f'Skill name {name!r} in {skill_file} must match its parent directory {directory_name!r}.')

    ignored_fields = tuple(sorted(BEHAVIORAL_FRONTMATTER_FIELDS.intersection(raw_frontmatter)))
    return SkillDefinition(
        name=name,
        description=frontmatter.description,
        body=body,
        ignored_behavioral_fields=ignored_fields,
    )


def load_skill_libraries(
    directories: Sequence[str | Path],
    *,
    include: Collection[str] | None,
    exclude: Collection[str],
) -> tuple[SkillDefinition, ...]:
    """Discover immediate child skill packages under explicit library roots."""
    roots: list[Path] = []
    seen_roots: set[Path] = set()
    for configured in directories:
        root = Path(configured)
        resolved = root.resolve()
        if resolved in seen_roots:
            continue
        if not root.exists():
            raise ValueError(f'Skill library directory does not exist: {root}')
        if not root.is_dir():
            raise ValueError(f'Skill library path is not a directory: {root}')
        if (root / 'SKILL.md').is_file():
            raise ValueError(
                f'Skill library path points to a skill package: {root}. Pass its parent directory instead.'
            )
        seen_roots.add(resolved)
        roots.append(root)

    skill_files: list[Path] = []
    for root in roots:
        skill_files.extend(
            sorted(
                (child / 'SKILL.md' for child in root.iterdir() if child.is_dir() and (child / 'SKILL.md').is_file()),
                key=lambda path: path.parent.name,
            )
        )

    available = frozenset(skill_file.parent.name for skill_file in skill_files)
    _validate_selection('include', include, available)
    _validate_selection('exclude', exclude, available)
    selected = include if include is not None else available.difference(exclude)

    selected_files: list[Path] = []
    names: dict[str, Path] = {}
    for skill_file in skill_files:
        name = skill_file.parent.name
        if name not in selected:
            continue
        resolved = skill_file.resolve()
        if previous := names.get(name):
            raise ValueError(f'Duplicate skill name {name!r}: {previous} and {resolved}.')
        names[name] = resolved
        selected_files.append(skill_file)

    return tuple(load_skill(skill_file) for skill_file in selected_files)


def _validate_selection(
    name: str,
    selected: Collection[str] | None,
    available: Collection[str],
) -> None:
    if selected is None:
        return
    unknown = sorted(set(selected).difference(available))
    if not unknown:
        return
    noun = 'skill' if len(unknown) == 1 else 'skills'
    available_text = ', '.join(sorted(available)) or '(none)'
    raise ValueError(f'Unknown {noun} in {name}: {", ".join(unknown)}. Available skills: {available_text}.')
