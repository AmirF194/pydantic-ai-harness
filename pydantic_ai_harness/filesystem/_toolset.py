"""Filesystem toolset providing sandboxed file operations backed by `ctx.sandbox`."""

from __future__ import annotations

import fnmatch
import functools
import hashlib
import posixpath
import re
import shlex
from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path
from typing import Concatenate, ParamSpec

from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.sandboxes import Sandbox
from pydantic_ai.tools import AgentDepsT, RunContext
from pydantic_ai.toolsets import AbstractToolset, FunctionToolset

_P = ParamSpec('_P')

_RECOVERABLE_ERRORS = (PermissionError, FileNotFoundError, NotADirectoryError, IsADirectoryError, ValueError, OSError)


def _recoverable(
    fn: Callable[Concatenate[FileSystemToolset, _P], Awaitable[str]],
) -> Callable[Concatenate[FileSystemToolset, _P], Awaitable[str]]:
    @functools.wraps(fn)
    async def wrapper(self: FileSystemToolset, *args: _P.args, **kwargs: _P.kwargs) -> str:
        try:
            return await fn(self, *args, **kwargs)
        except _RECOVERABLE_ERRORS as e:
            raise ModelRetry(str(e)) from e

    return wrapper


def _format_lines(lines: Sequence[str], offset: int, limit: int) -> str:
    total = len(lines)

    if total == 0:
        return '(empty file)\n'

    if offset >= total:
        raise ValueError(f'Offset {offset} exceeds file length ({total} lines).')

    selected = lines[offset : offset + limit]
    numbered = [f'{i:>6}\t{line}' for i, line in enumerate(selected, start=offset + 1)]
    result = ''.join(numbered)
    if not result.endswith('\n'):
        result += '\n'

    remaining = total - (offset + len(selected))
    if remaining > 0:
        next_offset = offset + len(selected)
        result += f'... ({remaining} more lines. Use offset={next_offset} to continue reading.)\n'

    return result


def _is_binary(data: bytes, sample_size: int = 8192) -> bool:
    return b'\x00' in data[:sample_size]


def _content_hash(content: str) -> str:
    return hashlib.sha256(content.encode('utf-8')).hexdigest()[:12]


class FileSystemToolset(FunctionToolset[AgentDepsT]):
    """Filesystem operations scoped to a sub-root inside `ctx.sandbox`.

    Paths resolve relative to `root_dir`, itself resolved against
    [`Sandbox.working_dir`][pydantic_ai.sandboxes.Sandbox.working_dir]. Every read,
    write, and listing goes through
    [`ctx.sandbox.fs`][pydantic_ai.sandboxes.Sandbox.fs]; recursive walkers use
    `list_dir` recursion.

    Textual path validation rejects `..` traversal and absolute-path escape.
    Symlink-realpath containment is not enforced here: isolation is the sandbox's
    responsibility, per the sandbox protocol.
    """

    def __init__(
        self,
        *,
        root_dir: str | Path = '.',
        allowed_patterns: Sequence[str],
        denied_patterns: Sequence[str],
        protected_patterns: Sequence[str],
        max_read_lines: int,
        max_search_results: int,
        max_find_results: int,
        sandbox: Sandbox | None = None,
    ) -> None:
        super().__init__()
        self._root_dir = str(root_dir)
        self._allowed_patterns = list(allowed_patterns)
        self._denied_patterns = list(denied_patterns)
        self._protected_patterns = list(protected_patterns)
        self._max_read_lines = max_read_lines
        self._max_search_results = max_search_results
        self._max_find_results = max_find_results
        self._sandbox = sandbox

        self.add_function(self.read_file, name='read_file')
        self.add_function(self.write_file, name='write_file')
        self.add_function(self.edit_file, name='edit_file')
        self.add_function(self.list_directory, name='list_directory')
        self.add_function(self.search_files, name='search_files')
        self.add_function(self.find_files, name='find_files')
        self.add_function(self.create_directory, name='create_directory')
        self.add_function(self.file_info, name='file_info')

    async def for_run(self, ctx: RunContext[AgentDepsT]) -> AbstractToolset[AgentDepsT]:
        return FileSystemToolset[AgentDepsT](
            root_dir=self._root_dir,
            allowed_patterns=self._allowed_patterns,
            denied_patterns=self._denied_patterns,
            protected_patterns=self._protected_patterns,
            max_read_lines=self._max_read_lines,
            max_search_results=self._max_search_results,
            max_find_results=self._max_find_results,
            sandbox=ctx.sandbox,
        )

    @property
    def sandbox(self) -> Sandbox:
        if self._sandbox is None:
            raise RuntimeError(
                'FileSystemToolset has no sandbox; construct it with sandbox=... or use it inside an agent run.'
            )
        return self._sandbox

    def _matches(self, path: str, pattern: str) -> bool:
        if fnmatch.fnmatch(path, pattern):
            return True
        if pattern.startswith('**/'):
            return fnmatch.fnmatch(path, pattern[3:])
        return False

    def _first_matching_pattern(self, path: str, patterns: list[str]) -> str | None:
        return next((p for p in patterns if self._matches(path, p)), None)

    async def _resolved_root(self) -> str:
        return await self.sandbox.resolve(self._root_dir)

    async def _resolve_path(self, path: str) -> tuple[str, str]:
        root = await self._resolved_root()
        joined = path if posixpath.isabs(path) else posixpath.join(root, path)
        candidate = posixpath.normpath(joined)
        relative = posixpath.relpath(candidate, root)
        if relative == '..' or relative.startswith('../'):
            raise PermissionError(f'Path {path!r} resolves outside the root directory.')
        return candidate, '' if relative == '.' else relative

    def _check_access(self, path: str, *, write: bool = False, check_allowed: bool = True) -> None:
        if write and self._protected_patterns:
            matched = self._first_matching_pattern(path, self._protected_patterns)
            if matched:
                raise PermissionError(f'Path {path!r} is protected (matches {matched!r}).')

        if self._denied_patterns:
            matched = self._first_matching_pattern(path, self._denied_patterns)
            if matched:
                raise PermissionError(f'Path {path!r} is denied by pattern {matched!r}.')

        if check_allowed and self._allowed_patterns:
            if not any(self._matches(path, p) for p in self._allowed_patterns):
                raise PermissionError(f'Path {path!r} does not match any allowed pattern.')

    def _is_accessible(self, path: str, *, write: bool = False) -> bool:
        if write and self._protected_patterns:
            if self._first_matching_pattern(path, self._protected_patterns) is not None:
                return False
        if self._denied_patterns:
            if self._first_matching_pattern(path, self._denied_patterns) is not None:
                return False
        if self._allowed_patterns and not any(self._matches(path, p) for p in self._allowed_patterns):
            return False
        return True

    async def _safe_resolve(self, path: str, *, write: bool = False, check_allowed: bool = True) -> str:
        absolute, relative = await self._resolve_path(path)
        self._check_access(relative, write=write, check_allowed=check_allowed)
        return absolute

    @_recoverable
    async def read_file(self, path: str, *, offset: int = 0, limit: int | None = None) -> str:
        """Read a text file with line numbers.

        Args:
            path: File path relative to the root directory.
            offset: Zero-based line offset to start reading from.
            limit: Maximum number of lines to return (default: 2000).
        """
        if limit is None:
            limit = self._max_read_lines
        absolute = await self._safe_resolve(path)
        entry = await self.sandbox.fs.stat(absolute)
        if entry.is_dir:
            raise FileNotFoundError(f"'{path}' is a directory, not a file.")

        raw = await self.sandbox.fs.read_bytes(absolute)
        if _is_binary(raw):
            return f'[Binary file: {len(raw)} bytes. Use a binary-aware tool to inspect.]'

        text = raw.decode('utf-8', errors='replace')
        lines = text.splitlines(keepends=True)
        content_hash = _content_hash(text)

        header = f'[{path} | {len(lines)} lines | hash:{content_hash}]\n'
        return header + _format_lines(lines, offset, limit)

    @_recoverable
    async def write_file(self, path: str, content: str, *, expected_hash: str | None = None) -> str:
        """Create or overwrite a file with conflict detection.

        Args:
            path: File path relative to the root directory.
            content: The text content to write.
            expected_hash: If provided, the write is rejected when the file exists and its
                current hash doesn't match (optimistic concurrency).
        """
        absolute = await self._safe_resolve(path, write=True)

        exists = await self.sandbox.fs.exists(absolute)
        if expected_hash is not None and exists:
            current = (await self.sandbox.fs.read_bytes(absolute)).decode('utf-8', errors='replace')
            current_hash = _content_hash(current)
            if current_hash != expected_hash:
                raise ValueError(
                    f'Conflict: file {path!r} has changed (expected hash:{expected_hash}, '
                    f'got hash:{current_hash}). Re-read the file and retry.'
                )

        parent = posixpath.dirname(absolute)
        if parent and not await self.sandbox.fs.exists(parent):
            parent_rel = posixpath.relpath(parent, await self._resolved_root())
            raise FileNotFoundError(f"Parent directory '{parent_rel}' does not exist. Use create_directory first.")

        await self.sandbox.fs.write_bytes(absolute, content.encode('utf-8'))
        new_hash = _content_hash(content)
        lines = len(content.splitlines())
        return f'Wrote {len(content)} chars ({lines} lines) to {path}. [hash:{new_hash}]'

    @_recoverable
    async def edit_file(self, path: str, old_text: str, new_text: str, *, expected_hash: str | None = None) -> str:
        """Edit a file by exact string replacement with conflict detection.

        `old_text` must appear exactly once in the file. Include surrounding
        context lines to ensure uniqueness.

        Args:
            path: File path relative to the root directory.
            old_text: The exact text to find (must appear exactly once).
            new_text: The replacement text.
            expected_hash: If provided, rejects the edit when the file's current hash doesn't match.
        """
        absolute = await self._safe_resolve(path, write=True)
        raw = await self.sandbox.fs.read_bytes(absolute)

        text = raw.decode('utf-8', errors='replace')
        current_hash = _content_hash(text)

        if expected_hash is not None and current_hash != expected_hash:
            raise ValueError(
                f'Conflict: file {path!r} has changed (expected hash:{expected_hash}, '
                f'got hash:{current_hash}). Re-read the file and retry.'
            )

        count = text.count(old_text)
        if count == 0:
            raise ValueError(f'old_text not found in {path}.')
        if count > 1:
            raise ValueError(
                f'old_text found {count} times in {path}. Include more surrounding context to make the match unique.'
            )

        new_content = text.replace(old_text, new_text, 1)
        await self.sandbox.fs.write_bytes(absolute, new_content.encode('utf-8'))
        new_hash = _content_hash(new_content)
        return f'Edited {path}. [hash:{new_hash}]'

    @_recoverable
    async def list_directory(self, path: str = '.') -> str:
        """List the contents of a directory.

        Args:
            path: Directory path relative to the root directory.
        """
        absolute = await self._safe_resolve(path, check_allowed=False)
        entry = await self.sandbox.fs.stat(absolute)
        if not entry.is_dir:
            raise NotADirectoryError(f'Not a directory: {path}')

        root = await self._resolved_root()
        entries: list[str] = []
        for child in await self.sandbox.fs.list_dir(absolute):
            rel = posixpath.relpath(child.path, root)
            if any(part.startswith('.') for part in rel.split('/')):
                continue
            if not self._is_accessible(rel, write=True):
                continue
            if child.is_dir:
                entries.append(f'{rel}/')
            else:
                size = child.size if child.size is not None else 0
                entries.append(f'{rel}  ({size} bytes)')
        return '\n'.join(entries) if entries else '(empty directory)'

    @_recoverable
    async def search_files(self, pattern: str, *, path: str = '.', include_glob: str | None = None) -> str:
        """Search file contents using a regular expression.

        Args:
            pattern: Regex pattern to search for.
            path: Directory to search in, relative to the root directory.
            include_glob: If provided, only search files matching this glob (e.g. `*.py`).
        """
        absolute = await self._safe_resolve(path, check_allowed=False)
        try:
            compiled = re.compile(pattern)
        except re.error as e:
            raise ValueError(f'Invalid regex pattern: {e}') from e

        candidate_files = await self._walk_files(absolute)
        results: list[str] = []
        root = await self._resolved_root()
        for file_path in candidate_files:
            rel = posixpath.relpath(file_path, root)
            if any(part.startswith('.') for part in rel.split('/')):
                continue
            if not self._is_accessible(rel, write=True):
                continue
            if include_glob and not fnmatch.fnmatch(rel, include_glob):
                continue
            try:
                raw = await self.sandbox.fs.read_bytes(file_path)
            except OSError:  # pragma: no cover
                continue
            if _is_binary(raw):
                continue
            text = raw.decode('utf-8', errors='replace')
            for line_num, line in enumerate(text.splitlines(), start=1):
                if compiled.search(line):
                    results.append(f'{rel}:{line_num}:{line}')
            if len(results) >= self._max_search_results:
                results.append(f'[... truncated at {self._max_search_results} matches]')
                break

        return '\n'.join(results) if results else 'No matches found.'

    @_recoverable
    async def find_files(self, pattern: str, *, path: str = '.') -> str:
        """Find files by glob pattern (name matching, not content search).

        Args:
            pattern: Glob pattern to match (e.g. `*.py`, `**/*.json`).
            path: Directory to search in, relative to the root directory.
        """
        absolute = await self._safe_resolve(path, check_allowed=False)
        entry = await self.sandbox.fs.stat(absolute)
        if not entry.is_dir:
            raise NotADirectoryError(f'Not a directory: {path}')

        root = await self._resolved_root()
        entries = sorted(await self._walk(absolute))
        matches: list[str] = []
        for entry_path, is_dir in entries:
            rel = posixpath.relpath(entry_path, root)
            if any(part.startswith('.') for part in rel.split('/')):
                continue
            if not fnmatch.fnmatch(rel, pattern):
                continue
            if not self._is_accessible(rel, write=True):
                continue
            suffix = '/' if is_dir else ''
            matches.append(f'{rel}{suffix}')
            if len(matches) >= self._max_find_results:
                matches.append(f'[... truncated at {self._max_find_results} matches]')
                break

        return '\n'.join(matches) if matches else 'No matches found.'

    @_recoverable
    async def create_directory(self, path: str) -> str:
        """Create a directory and any missing parents.

        Args:
            path: Directory path relative to the root directory.
        """
        absolute = await self._safe_resolve(path, write=True)
        await self.sandbox.fs.make_dir(absolute)
        return f'Created directory: {path}'

    @_recoverable
    async def file_info(self, path: str) -> str:
        """Get metadata about a file or directory.

        Args:
            path: File or directory path relative to the root directory.
        """
        absolute = await self._safe_resolve(path)
        entry = await self.sandbox.fs.stat(absolute)

        parts = [f'path: {path}', f'type: {"directory" if entry.is_dir else "file"}', f'size: {entry.size or 0} bytes']

        if not entry.is_dir:
            raw = await self.sandbox.fs.read_bytes(absolute)
            is_bin = _is_binary(raw)
            parts.append(f'binary: {is_bin}')
            if not is_bin:
                text = raw.decode('utf-8', errors='replace')
                parts.append(f'lines: {len(text.splitlines())}')
                parts.append(f'hash: {_content_hash(text)}')

        target = await self._readlink(absolute)
        if target is not None:
            parts.append(f'symlink_target: {target}')

        return '\n'.join(parts)

    async def _walk(self, root: str) -> list[tuple[str, bool]]:
        entries: list[tuple[str, bool]] = []
        stack = [root]
        while stack:
            current = stack.pop()
            children = await self.sandbox.fs.list_dir(current)
            for child in children:
                entries.append((child.path, bool(child.is_dir)))
                if child.is_dir:
                    stack.append(child.path)
        return entries

    async def _walk_files(self, root: str) -> list[str]:
        try:
            entry = await self.sandbox.fs.stat(root)
        except FileNotFoundError:
            return []
        if not entry.is_dir:
            return [root]
        return sorted(path for path, is_dir in await self._walk(root) if not is_dir)

    async def _readlink(self, path: str) -> str | None:
        result = await self.sandbox.run(f'readlink {shlex.quote(path)}', shell=True)
        if result.exit_code != 0:
            return None
        target = result.stdout.strip()
        return target or None
