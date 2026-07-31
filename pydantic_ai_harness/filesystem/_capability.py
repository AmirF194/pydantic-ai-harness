"""Filesystem capability that provides sandboxed file system access."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.tools import AgentDepsT

from pydantic_ai_harness.filesystem._toolset import FileSystemToolset

_DEFAULT_PROTECTED: list[str] = [
    '.git/*',
    '.env',
    '.env.*',
    '*.pem',
    '*.key',
    '**/secrets*',
]


@dataclass
class FileSystem(AbstractCapability[AgentDepsT]):
    """File system access scoped to a sub-root inside `ctx.sandbox`.

    Paths resolve relative to `root_dir`, itself a path inside the run's sandbox
    working directory. Traversal above the root is rejected textually. Symlink
    containment is the sandbox's responsibility.
    """

    root_dir: str | Path = '.'
    """Path inside the sandbox working directory. Defaults to the working directory itself."""

    allowed_patterns: Sequence[str] = field(default_factory=list[str])
    """If non-empty, only paths matching at least one glob pattern are accessible."""

    denied_patterns: Sequence[str] = field(default_factory=list[str])
    """Paths matching any of these glob patterns are rejected."""

    protected_patterns: Sequence[str] = field(default_factory=lambda: list(_DEFAULT_PROTECTED))
    """Paths matching these patterns are read-only (writes are rejected).

    Defaults to protecting `.git/`, `.env`, key files, and secrets.
    Set to an empty list to disable protection.
    """

    max_read_lines: int = 2000
    """Maximum number of lines returned by a single `read_file` call."""

    max_search_results: int = 1000
    """Maximum number of matches returned by `search_files`."""

    max_find_results: int = 1000
    """Maximum number of matches returned by `find_files`."""

    def __post_init__(self) -> None:
        values: dict[str, Any] = {
            'max_read_lines': self.max_read_lines,
            'max_search_results': self.max_search_results,
            'max_find_results': self.max_find_results,
        }
        for name, value in values.items():
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f'{name} must be a positive integer, got {value!r}')

    def get_toolset(self) -> FileSystemToolset[AgentDepsT]:
        return FileSystemToolset[AgentDepsT](
            root_dir=str(self.root_dir),
            allowed_patterns=self.allowed_patterns,
            denied_patterns=self.denied_patterns,
            protected_patterns=self.protected_patterns,
            max_read_lines=self.max_read_lines,
            max_search_results=self.max_search_results,
            max_find_results=self.max_find_results,
        )
