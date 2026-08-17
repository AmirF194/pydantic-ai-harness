"""Filesystem toolset providing sandboxed file operations."""

from __future__ import annotations

import errno
import fnmatch
import functools
import hashlib
import os
import posixpath
import re
import stat
from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path, PurePath
from typing import Concatenate, Literal, ParamSpec

from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.tools import AgentDepsT
from pydantic_ai.toolsets import FunctionToolset

_P = ParamSpec('_P')

READ_ONLY_TOOL_NAMES: frozenset[str] = frozenset(
    {'read_file', 'list_directory', 'search_files', 'find_files', 'file_info'}
)
"""Names of filesystem tools that do not modify the workspace."""

# Errors that mean "the model asked for something the tool couldn't do" -- a
# missing file, a denied path, a stale edit. pyai only feeds `ModelRetry` back
# to the model; any other exception aborts the whole run. `_recoverable`
# converts these so the agent can correct itself and continue.
_RECOVERABLE_ERRORS = (PermissionError, FileNotFoundError, NotADirectoryError, IsADirectoryError, ValueError)

_HAS_FD_TRAVERSAL = (
    {os.open, os.mkdir, os.readlink} <= os.supports_dir_fd and hasattr(os, 'O_NOFOLLOW') and hasattr(os, 'O_DIRECTORY')
)
"""Whether the platform can walk paths descriptor-relative (`openat` style).

True on Linux and macOS. Where it is False (Windows), containment falls back
to the pathname-based `realpath` check, which cannot bind the check to the
object the I/O later touches.
"""

# Directories along the walk only anchor the next `dir_fd` step. `O_PATH`
# (Linux) grants exactly that without needing read permission; platforms
# without it fall back to `O_RDONLY`.
_DIR_OPEN_FLAGS = (getattr(os, 'O_PATH', os.O_RDONLY) | os.O_DIRECTORY | os.O_NOFOLLOW) if _HAS_FD_TRAVERSAL else 0

# `O_NONBLOCK` keeps an open from waiting on a FIFO; it has no effect on
# regular files. `O_NOFOLLOW` refuses a symlink swapped in after the walk
# resolved the component.
_FILE_OPEN_FLAGS = (os.O_NOFOLLOW | os.O_NONBLOCK) if _HAS_FD_TRAVERSAL else 0

# A symlink refused by `O_NOFOLLOW` surfaces as `ELOOP` on Linux and macOS and
# as `EMLINK` on some BSDs.
_SYMLINK_ERRNOS = frozenset({errno.ELOOP, errno.EMLINK})

# Entries that exist but cannot be opened as files: sockets (`EOPNOTSUPP` on
# macOS, `ENXIO` on Linux) and device nodes without a driver (`ENODEV`).
_SPECIAL_FILE_ERRNOS = frozenset({errno.ENXIO, errno.ENODEV, getattr(errno, 'EOPNOTSUPP', errno.ENXIO)})

_MAX_SYMLINK_HOPS = 40
"""Symlink resolutions allowed per lookup before reporting a loop, mirroring the kernel's own bound."""

_Intent = Literal['read', 'write', 'edit', 'mkdir']

_Kind = Literal['missing', 'dir', 'file', 'other']


def _recoverable(
    fn: Callable[Concatenate[FileSystemToolset, _P], Awaitable[str]],
) -> Callable[Concatenate[FileSystemToolset, _P], Awaitable[str]]:
    """Surface model-correctable tool errors as `ModelRetry`."""

    @functools.wraps(fn)
    async def wrapper(self: FileSystemToolset, *args: _P.args, **kwargs: _P.kwargs) -> str:
        try:
            return await fn(self, *args, **kwargs)
        except _RECOVERABLE_ERRORS as e:
            raise ModelRetry(str(e)) from e

    return wrapper


def _format_lines(lines: Sequence[str], offset: int, limit: int) -> str:
    """Format pre-split lines with line numbers and continuation hint."""
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
    """Detect binary content by checking for null bytes in the sample."""
    return b'\x00' in data[:sample_size]


def _matching_lines(text: str, compiled: re.Pattern[str], rel_str: str, limit: int) -> tuple[list[str], bool]:
    """Match one file's lines, keeping at most `limit` of them.

    Returns the formatted matches and whether a further match had to be
    dropped, so the caller reports truncation only when output was cut. A
    `limit` of zero or less keeps nothing.
    """
    matches: list[str] = []
    for line_num, line in enumerate(text.splitlines(), start=1):
        if compiled.search(line):
            if len(matches) >= limit:
                return matches, True
            matches.append(f'{rel_str}:{line_num}:{line}')
    return matches, False


def _content_hash(content: str) -> str:
    """Compute a short content hash for conflict detection."""
    return hashlib.sha256(content.encode('utf-8')).hexdigest()[:12]


def _kind_of(st: os.stat_result | None) -> _Kind:
    """Classify a stat result by what I/O the target supports."""
    if st is None:
        return 'missing'
    if stat.S_ISDIR(st.st_mode):
        return 'dir'
    if stat.S_ISREG(st.st_mode):
        return 'file'
    return 'other'


def _readlink_or_none(name: str, dir_fd: int) -> str | None:
    """The symlink target of `name` in the directory `dir_fd`, or None when it is not a symlink."""
    try:
        return os.readlink(name, dir_fd=dir_fd)
    except OSError as e:
        # `EINVAL`: a real (non-symlink) entry. `ENOENT`: nothing there; the
        # open step reports it with the caller's message.
        if e.errno in (errno.EINVAL, errno.ENOENT):
            return None
        raise  # pragma: no cover


class _SpecialFileError(Exception):
    """Internal: the walk's final component exists but cannot be opened (socket, device node)."""


def _too_many_links(path: str) -> PermissionError:
    """The error for a lookup that exhausted the symlink budget (a loop, or an absurd chain)."""
    return PermissionError(f'Path {path!r} resolves through too many levels of symbolic links.')


def _normalized_remainder(path: str, canonical: list[str], name: str, pending: list[str]) -> str:
    """Lexically collapse the un-walked tail of a missing path for pattern checks and messages.

    Components past a missing directory cannot contain symlinks, so collapsing
    them lexically matches what `realpath` would have produced.
    """
    remainder = posixpath.normpath('/'.join([*canonical, name, *reversed(pending)]))
    if remainder == '..' or remainder.startswith('../'):
        raise PermissionError(f'Path {path!r} resolves outside the root directory.')
    return remainder


def _restart_collapsed(remainder: str, fds: list[int], canonical: list[str], pending: list[str]) -> None:
    """Restart the walk from the root on a lexically collapsed remainder."""
    while len(fds) > 1:
        os.close(fds.pop())
    canonical.clear()
    pending.clear()
    if remainder != '.':
        pending.extend(reversed(remainder.split('/')))


class _Resolved:
    """One authorized filesystem location, ready for I/O.

    With descriptor traversal, `fd` was opened by walking every component
    `O_NOFOLLOW` relative to its parent, so the object the checks authorized is
    the object the I/O touches. On fallback platforms `fd` is None and I/O uses
    the pathname, keeping the legacy best-effort guarantee.
    """

    __slots__ = ('fd', 'path', 'canonical', 'created')

    def __init__(self, fd: int | None, path: Path, canonical: str, *, created: bool = False) -> None:
        self.fd = fd
        self.path = path
        self.canonical = canonical
        self.created = created

    def __enter__(self) -> _Resolved:
        return self

    def __exit__(self, *_exc_info: object) -> None:
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None

    def stat(self) -> os.stat_result | None:
        """Stat the target, or None when it does not exist."""
        try:
            return os.fstat(self.fd) if self.fd is not None else os.stat(self.path)
        except OSError:
            return None

    def read_bytes(self) -> bytes:
        """Read the target's full content as bytes."""
        if self.fd is None:
            return self.path.read_bytes()
        os.lseek(self.fd, 0, os.SEEK_SET)
        chunks = bytearray()
        while chunk := os.read(self.fd, 1 << 16):
            chunks += chunk
        return bytes(chunks)

    def read_text(self) -> str:
        """Strict UTF-8 text with universal newlines, matching `Path.read_text`."""
        if self.fd is None:
            return self.path.read_text(encoding='utf-8')
        return self.read_bytes().decode('utf-8').replace('\r\n', '\n').replace('\r', '\n')

    def replace_text(self, content: str) -> None:
        """Replace the target's content, through the descriptor when there is one."""
        if self.fd is None:
            self.path.write_text(content, encoding='utf-8')
            return
        os.lseek(self.fd, 0, os.SEEK_SET)
        os.truncate(self.fd, 0)
        data = content.encode('utf-8')
        while data:
            data = data[os.write(self.fd, data) :]


class FileSystemToolset(FunctionToolset[AgentDepsT]):
    """Toolset providing filesystem operations scoped to a root directory.

    Security model:
    - All paths resolved relative to root with canonical path checks
    - Where the platform supports `openat`-style traversal, every path
      component is opened `O_NOFOLLOW` relative to its parent and I/O happens
      on that descriptor, so a path swapped mid-operation cannot redirect it
    - Glob-based allow/deny filtering, matched against resolved locations
    - Protected path patterns (e.g. `.git/`, `.env`)
    - Binary file detection blocks text operations
    """

    def __init__(
        self,
        *,
        root_dir: Path,
        allowed_patterns: Sequence[str],
        denied_patterns: Sequence[str],
        protected_patterns: Sequence[str],
        max_read_lines: int,
        max_list_results: int,
        max_search_results: int,
        max_find_results: int,
    ) -> None:
        super().__init__()
        self._root = root_dir.resolve()
        self._real_root = Path(os.path.realpath(self._root))
        self._allowed_patterns = list(allowed_patterns)
        self._denied_patterns = list(denied_patterns)
        self._protected_patterns = list(protected_patterns)
        self._max_read_lines = max_read_lines
        self._max_list_results = max_list_results
        self._max_search_results = max_search_results
        self._max_find_results = max_find_results

        self.add_function(self.read_file, name='read_file')
        self.add_function(self.write_file, name='write_file')
        self.add_function(self.edit_file, name='edit_file')
        self.add_function(self.list_directory, name='list_directory')
        self.add_function(self.search_files, name='search_files')
        self.add_function(self.find_files, name='find_files')
        self.add_function(self.create_directory, name='create_directory')
        self.add_function(self.file_info, name='file_info')

    def _matches(self, path: str, pattern: str) -> bool:
        """Glob-match a relative path, treating a leading `**/` as 'any directory, including the root'.

        `fnmatch` has no recursive `**`, so a bare `**/secrets*` would miss a
        root-level `secrets.yaml` -- there's no leading directory to match.
        Retrying with the `**/` prefix stripped covers the zero-directory case.
        """
        if fnmatch.fnmatch(path, pattern):
            return True
        if pattern.startswith('**/'):
            return fnmatch.fnmatch(path, pattern[3:])
        return False

    def _first_matching_pattern(self, path: str, patterns: list[str]) -> str | None:
        """Return the first pattern that matches path, or None."""
        return next((p for p in patterns if self._matches(path, p)), None)

    def _resolve_path(self, path: str) -> Path:
        """Resolve path relative to root the legacy way, rejecting traversal.

        Fallback for platforms without descriptor traversal: `os.path.realpath`
        resolves symlinks before the containment check, but nothing binds that
        check to the object later I/O touches.
        """
        candidate = (self._root / path).resolve()
        real = Path(os.path.realpath(candidate))
        if not real.is_relative_to(self._real_root):
            raise PermissionError(f'Path {path!r} resolves outside the root directory.')

        return real

    def _check_access(self, path: str, *, write: bool = False, check_allowed: bool = True) -> None:
        """Validate path against allow/deny/protected patterns.

        `check_allowed=False` skips the `allowed_patterns` gate. Walkers
        (`list_directory`, `search_files`, `find_files`) pass it so their root
        directory isn't required to match `allowed_patterns` itself -- `.` or
        `src` would never match a file pattern like `src/*.py`. The walk's
        entries are still filtered against `allowed_patterns` per-entry via
        `_resolve_entry`. Denied patterns continue to gate the root.
        """
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

    def _relative_to_root(self, resolved: Path) -> str:
        """Canonical path of a resolved location relative to the real root."""
        return str(resolved.relative_to(self._real_root))

    def _split_components(self, path: str) -> list[str]:
        """Split a tool path into components to walk from the root.

        `..` components are kept for the walk to apply physically. An absolute
        path is accepted only when it names a location under the root; the
        walk still verifies every component of it.
        """
        pure = PurePath(path)
        if pure.is_absolute():
            try:
                pure = pure.relative_to(self._real_root)
            except ValueError:
                # The absolute spelling may reach the root through symlinks
                # (e.g. macOS `/tmp` vs `/private/tmp`). `realpath` here is
                # advisory: the walk still verifies every component it yields.
                try:
                    pure = PurePath(os.path.realpath(path)).relative_to(self._real_root)
                except ValueError:
                    raise PermissionError(f'Path {path!r} resolves outside the root directory.') from None
        return [c for c in pure.parts if c != '.']

    def _resolve_for(
        self,
        path: str,
        *,
        intent: _Intent = 'read',
        write: bool = False,
        check_allowed: bool = True,
        missing: str,
        missing_parent: Callable[[str], str] | None = None,
        need_read: bool = False,
    ) -> _Resolved:
        """Authorize `path` and open it for the given intent.

        With descriptor traversal, the returned handle's descriptor is the very
        object the containment and pattern checks authorized. On fallback
        platforms the handle is pathname-bound and the checks stay best-effort.
        """
        if not _HAS_FD_TRAVERSAL:
            resolved = self._resolve_path(path)
            canonical = self._relative_to_root(resolved)
            self._check_access(canonical, write=write, check_allowed=check_allowed)
            return _Resolved(None, resolved, canonical)
        fd, canonical, created = self._walk_beneath(
            path,
            intent=intent,
            write=write,
            check_allowed=check_allowed,
            missing=missing,
            missing_parent=missing_parent,
            need_read=need_read,
        )
        return _Resolved(fd, self._real_root / canonical, canonical, created=created)

    def _resolve_entry(self, rel: str) -> _Resolved | None:
        """Authorize one entry of a directory walk, or return None to skip it.

        Callers must do their I/O through the returned handle. Resolving and
        opening in one authorized step means a symlink can neither escape the
        root nor alias a file past a rule its own name would trip, since the
        patterns are matched against the resolved location.
        """
        try:
            return self._resolve_for(rel, missing=f'File not found: {rel}')
        except OSError:
            return None

    def _walk_beneath(
        self,
        path: str,
        *,
        intent: _Intent,
        write: bool,
        check_allowed: bool,
        missing: str,
        missing_parent: Callable[[str], str] | None,
        need_read: bool,
    ) -> tuple[int | None, str, bool]:
        """Open `path` by walking each component relative to the previous one.

        Every step uses `os.open(..., O_NOFOLLOW, dir_fd=parent)`, symlinks are
        resolved manually and re-checked for containment, and `..` pops the
        directory stack so it can never climb past the root. The returned
        descriptor is therefore the object the checks authorized, closing the
        gap between pathname authorization and pathname I/O (#632).

        Returns `(fd, canonical_relative_path, created)`. `fd` is None only for
        a special file (socket, device node) that exists but cannot be opened;
        the caller falls back to metadata-only handling for it.
        """
        pending = list(reversed(self._split_components(path)))
        hops = _MAX_SYMLINK_HOPS
        fds = [os.open(self._real_root, os.O_RDONLY | os.O_DIRECTORY)]
        canonical: list[str] = []
        try:
            while pending:
                name = pending.pop()
                if name == '..':
                    if len(fds) == 1:
                        raise PermissionError(f'Path {path!r} resolves outside the root directory.')
                    os.close(fds.pop())
                    canonical.pop()
                    continue
                target = _readlink_or_none(name, fds[-1])
                if target is not None:
                    hops -= 1
                    if hops <= 0:
                        raise _too_many_links(path)
                    self._splice_link(target, path, fds, canonical, pending)
                    continue
                if pending:
                    outcome = self._descend(
                        name,
                        fds,
                        canonical,
                        pending,
                        path=path,
                        intent=intent,
                        write=write,
                        check_allowed=check_allowed,
                        missing=missing,
                        missing_parent=missing_parent,
                    )
                    if outcome == 'retry_symlink':
                        hops -= 1
                        if hops <= 0:
                            raise _too_many_links(path)
                    continue
                rel = '/'.join([*canonical, name])
                self._check_access(rel, write=write, check_allowed=check_allowed)
                try:
                    opened = self._open_final(
                        name, fds[-1], rel=rel, path=path, intent=intent, missing=missing, need_read=need_read
                    )
                except _SpecialFileError:
                    return None, rel, False
                if opened is None:
                    # The component turned into a symlink after it was resolved
                    # as a plain entry; go around to resolve it safely.
                    hops -= 1
                    if hops <= 0:
                        raise _too_many_links(path)
                    pending.append(name)
                    continue
                return opened[0], rel, opened[1]
            # No components left: the path names the walked directory itself.
            rel = '/'.join(canonical) or '.'
            self._check_access(rel, write=write, check_allowed=check_allowed)
            if intent == 'write':
                raise IsADirectoryError(f'Is a directory: {path}')
            if intent == 'edit':
                raise FileNotFoundError(missing)
            return os.dup(fds[-1]), rel, False
        finally:
            for fd in fds:
                os.close(fd)

    def _splice_link(self, target: str, path: str, fds: list[int], canonical: list[str], pending: list[str]) -> None:
        """Queue a symlink target's components for the walk.

        An absolute target restarts the walk at the root. `realpath` here is
        advisory -- it produces the canonical candidate to compare and walk --
        while enforcement stays with the `O_NOFOLLOW` open of each component.
        """
        if os.path.isabs(target):
            real_target = Path(os.path.realpath(target))
            if not real_target.is_relative_to(self._real_root):
                raise PermissionError(f'Path {path!r} resolves outside the root directory.')
            while len(fds) > 1:
                os.close(fds.pop())
            canonical.clear()
            parts: Sequence[str] = real_target.relative_to(self._real_root).parts
        else:
            parts = [c for c in PurePath(target).parts if c != '.']
        pending.extend(reversed(parts))

    def _descend(
        self,
        name: str,
        fds: list[int],
        canonical: list[str],
        pending: list[str],
        *,
        path: str,
        intent: _Intent,
        write: bool,
        check_allowed: bool,
        missing: str,
        missing_parent: Callable[[str], str] | None,
    ) -> Literal['descended', 'retry', 'retry_symlink']:
        """Open intermediate directory `name` and push it onto the walk stack.

        Returns 'descended' after pushing the opened directory, 'retry' when
        the queue should simply be reprocessed (a directory was created for
        `create_directory`, or a lexical `..` past an un-walkable component was
        collapsed), and 'retry_symlink' when the component changed into a
        symlink underneath the walk, which must count against the symlink
        budget. Raises for a missing or non-directory component.
        """
        try:
            fd = os.open(name, _DIR_OPEN_FLAGS, dir_fd=fds[-1])
        except OSError as e:
            if e.errno in _SYMLINK_ERRNOS:
                pending.append(name)
                return 'retry_symlink'
            if e.errno in (errno.ENOENT, errno.ENOTDIR) and '..' in pending:
                # A lexical `..` beyond a missing or non-directory component:
                # restart the walk on the collapsed path, matching what
                # `realpath` produces for a tail that cannot be walked.
                _restart_collapsed(_normalized_remainder(path, canonical, name, pending), fds, canonical, pending)
                return 'retry'
            if e.errno == errno.ENOENT:
                remainder = _normalized_remainder(path, canonical, name, pending)
                self._check_access(remainder, write=write, check_allowed=check_allowed)
                if intent == 'mkdir':
                    try:
                        os.mkdir(name, dir_fd=fds[-1])
                    except FileExistsError:
                        # Created concurrently; reopening it below is fine.
                        pass
                    pending.append(name)
                    return 'retry'
                if missing_parent is not None:
                    raise FileNotFoundError(missing_parent(posixpath.dirname(remainder))) from e
                raise FileNotFoundError(missing) from e
            if e.errno == errno.ENOTDIR:
                remainder = _normalized_remainder(path, canonical, name, pending)
                self._check_access(remainder, write=write, check_allowed=check_allowed)
                raise NotADirectoryError(f'Not a directory: {"/".join([*canonical, name])}') from e
            e.filename = '/'.join([*canonical, name])
            raise
        fds.append(fd)
        canonical.append(name)
        return 'descended'

    def _open_final(
        self,
        name: str,
        dir_fd: int,
        *,
        rel: str,
        path: str,
        intent: _Intent,
        missing: str,
        need_read: bool,
    ) -> tuple[int, bool] | None:
        """Open the walk's fully resolved final component, `O_NOFOLLOW`.

        Returns `(fd, created)`, or None when the component became a symlink
        (or vanished mid-create) and the walk must re-resolve it. Raises
        `_SpecialFileError` for an entry that exists but cannot be opened.
        """
        if intent == 'mkdir':
            return self._open_final_dir(name, dir_fd)
        if intent == 'write':
            return self._open_final_write(name, dir_fd, rel=rel, path=path, need_read=need_read)
        flags = os.O_RDWR if intent == 'edit' else os.O_RDONLY
        try:
            return os.open(name, flags | _FILE_OPEN_FLAGS, dir_fd=dir_fd), False
        except OSError as e:
            if e.errno in _SYMLINK_ERRNOS:
                return None
            if e.errno in (errno.ENOENT, errno.EISDIR):
                # `EISDIR` is `edit` only: a directory cannot be edited, and
                # the tool reports that the same way as a missing file.
                raise FileNotFoundError(missing) from e
            if e.errno in _SPECIAL_FILE_ERRNOS:
                raise _SpecialFileError from e
            e.filename = rel
            raise

    def _open_final_write(
        self, name: str, dir_fd: int, *, rel: str, path: str, need_read: bool
    ) -> tuple[int, bool] | None:
        """Open (or create) the final component for `write_file`.

        Creation is attempted first with `O_EXCL` so the caller knows whether
        the file existed and an `expected_hash` must be honored. Opening
        without truncation lets the caller classify the descriptor and check
        the hash before any content changes.
        """
        base = (os.O_RDWR if need_read else os.O_WRONLY) | _FILE_OPEN_FLAGS
        try:
            return os.open(name, base | os.O_CREAT | os.O_EXCL, 0o666, dir_fd=dir_fd), True
        except FileExistsError:
            pass
        try:
            return os.open(name, base, dir_fd=dir_fd), False
        except OSError as e:
            if e.errno in _SYMLINK_ERRNOS or e.errno == errno.ENOENT:
                # Swapped for a symlink, or deleted between the two opens.
                return None
            if e.errno == errno.EISDIR:
                raise IsADirectoryError(f'Is a directory: {path}') from e
            if e.errno in _SPECIAL_FILE_ERRNOS:
                # For example a FIFO with no reader, or a socket; without
                # `O_NONBLOCK` the FIFO open would hang.
                raise ValueError(f'Path {path!r} exists and is not a regular file.') from e
            e.filename = rel
            raise

    def _open_final_dir(self, name: str, dir_fd: int) -> tuple[int, bool]:
        """Create (or reuse) the final directory for `create_directory`."""
        try:
            os.mkdir(name, dir_fd=dir_fd)
        except FileExistsError as exists_error:
            try:
                return os.open(name, _DIR_OPEN_FLAGS, dir_fd=dir_fd), False
            except OSError as e:
                if e.errno == errno.ENOTDIR or e.errno in _SYMLINK_ERRNOS:
                    # Exists but is not a directory: report it as `mkdir` would.
                    raise exists_error from e
                raise  # pragma: no cover
        return os.open(name, _DIR_OPEN_FLAGS, dir_fd=dir_fd), True

    @_recoverable
    async def read_file(self, path: str, *, offset: int = 0, limit: int | None = None) -> str:
        """Read a text file with line numbers.

        Args:
            path: File path relative to the root directory.
            offset: Zero-based line offset to start reading from.
            limit: Maximum number of lines to return (default: 2000).

        Returns:
            File content with line numbers, plus metadata header.
        """
        if limit is None:
            limit = self._max_read_lines
        with self._resolve_for(path, missing=f'File not found: {path}') as resolved:
            kind = _kind_of(resolved.stat())
            if kind == 'dir':
                raise FileNotFoundError(f"'{path}' is a directory, not a file.")
            if kind != 'file':
                raise FileNotFoundError(f'File not found: {path}')
            raw = resolved.read_bytes()

        if _is_binary(raw):
            size = len(raw)
            return f'[Binary file: {size} bytes. Use a binary-aware tool to inspect.]'

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
            expected_hash: If provided, the write is rejected when the file exists
                and its current hash doesn't match (optimistic concurrency).

        Returns:
            Confirmation message with new hash.
        """

        def missing_parent(parent: str) -> str:
            return f"Parent directory '{parent}' does not exist. Use create_directory first."

        with self._resolve_for(
            path,
            intent='write',
            write=True,
            missing=f'File not found: {path}',
            missing_parent=missing_parent,
            need_read=expected_hash is not None,
        ) as resolved:
            if resolved.fd is None:
                # Pathname fallback: the same checks, without descriptor binding.
                if expected_hash is not None and resolved.path.is_file():
                    self._check_expected_hash(path, expected_hash, resolved.read_text())
                if not resolved.path.parent.exists():
                    parent_rel = str(resolved.path.parent.relative_to(self._root))
                    raise FileNotFoundError(missing_parent(parent_rel))
            else:
                if _kind_of(resolved.stat()) == 'other':
                    raise ValueError(f'Path {path!r} exists and is not a regular file.')
                if expected_hash is not None and not resolved.created:
                    self._check_expected_hash(path, expected_hash, resolved.read_text())
            resolved.replace_text(content)
        new_hash = _content_hash(content)
        lines = len(content.splitlines())
        return f'Wrote {len(content)} chars ({lines} lines) to {path}. [hash:{new_hash}]'

    def _check_expected_hash(self, path: str, expected_hash: str, current: str) -> None:
        """Reject a stale write or edit (optimistic concurrency)."""
        current_hash = _content_hash(current)
        if current_hash != expected_hash:
            raise ValueError(
                f'Conflict: file {path!r} has changed (expected hash:{expected_hash}, '
                f'got hash:{current_hash}). Re-read the file and retry.'
            )

    @_recoverable
    async def edit_file(self, path: str, old_text: str, new_text: str, *, expected_hash: str | None = None) -> str:
        """Edit a file by exact string replacement with conflict detection.

        The old_text must appear exactly once in the file. Include surrounding
        context lines to ensure uniqueness.

        Args:
            path: File path relative to the root directory.
            old_text: The exact text to find (must appear exactly once).
            new_text: The replacement text.
            expected_hash: If provided, rejects the edit when the file's
                current hash doesn't match (optimistic concurrency).

        Returns:
            Summary with new hash for subsequent operations.
        """
        with self._resolve_for(
            path, intent='edit', write=True, missing=f'File not found: {path}', need_read=True
        ) as resolved:
            if _kind_of(resolved.stat()) != 'file':
                raise FileNotFoundError(f'File not found: {path}')

            text = resolved.read_text()

            if expected_hash is not None:
                self._check_expected_hash(path, expected_hash, text)

            count = text.count(old_text)
            if count == 0:
                raise ValueError(f'old_text not found in {path}.')
            if count > 1:
                raise ValueError(
                    f'old_text found {count} times in {path}. '
                    'Include more surrounding context to make the match unique.'
                )

            new_content = text.replace(old_text, new_text, 1)
            resolved.replace_text(new_content)
        new_hash = _content_hash(new_content)
        return f'Edited {path}. [hash:{new_hash}]'

    @_recoverable
    async def list_directory(self, path: str = '.') -> str:
        """List the contents of a directory.

        Args:
            path: Directory path relative to the root directory.

        Returns:
            A newline-separated listing with type indicators and sizes.
        """
        # The listing root is gated by denied patterns but not by
        # allowed_patterns: a directory like '.' never matches a file pattern.
        # Entries are filtered per-entry against allowed_patterns below.
        with self._resolve_for(path, check_allowed=False, missing=f'Not a directory: {path}') as resolved:
            if _kind_of(resolved.stat()) != 'dir':
                raise NotADirectoryError(f'Not a directory: {path}')
            base = resolved.path

        entries: list[str] = []
        for entry in sorted(base.iterdir()):
            try:
                rel_path = entry.relative_to(self._real_root)
            except ValueError:  # pragma: no cover
                continue
            # Skip dotfiles and dot-directories, matching search_files and
            # find_files so the three walkers agree on what exists.
            if any(part.startswith('.') for part in rel_path.parts):
                continue
            target = self._resolve_entry(str(rel_path))
            if target is None:
                continue
            with target:
                st = target.stat()
            if st is None:
                # A dangling symlink, or an entry deleted mid-walk: it has
                # no size to report, so leave it out of the listing.
                continue
            rel = str(rel_path)
            if _kind_of(st) == 'dir':
                line = f'{rel}/'
            else:
                line = f'{rel}  ({st.st_size} bytes)'
            # Only a listing that actually dropped an entry is marked truncated,
            # so one that merely fills the cap reads as complete.
            if len(entries) >= self._max_list_results:
                entries.append(f'[... truncated at {self._max_list_results} entries]')
                break
            entries.append(line)
        return '\n'.join(entries) if entries else '(empty directory)'

    @_recoverable
    async def search_files(self, pattern: str, *, path: str = '.', include_glob: str | None = None) -> str:
        """Search file contents using a regular expression.

        Args:
            pattern: Regex pattern to search for.
            path: Directory to search in, relative to the root directory.
            include_glob: If provided, only search files matching this glob (e.g. '*.py').

        Returns:
            str: Matching lines formatted as file:line_number:text.
        """
        # See list_directory: the search root isn't gated by allowed_patterns;
        # matched files are filtered per-entry below.
        root_kind: _Kind = 'missing'
        base: Path | None = None
        try:
            with self._resolve_for(path, check_allowed=False, missing=f'File not found: {path}') as resolved:
                root_kind = _kind_of(resolved.stat())
                base = resolved.path
        except (FileNotFoundError, NotADirectoryError):
            # A root that does not name a directory tree has no matches in it.
            pass
        try:
            compiled = re.compile(pattern)
        except re.error as e:
            raise ValueError(f'Invalid regex pattern: {e}') from e

        results: list[str] = []

        if base is None or root_kind in ('missing', 'other'):
            files: list[Path] = []
        elif root_kind == 'file':
            files = [base]
        else:
            files = sorted(base.rglob('*'))

        for file_path in files:
            try:
                rel_path = file_path.relative_to(self._real_root)
            except ValueError:  # pragma: no cover
                continue
            if any(part.startswith('.') for part in rel_path.parts):
                continue
            rel_str = str(rel_path)
            if include_glob and not fnmatch.fnmatch(rel_str, include_glob):
                continue
            target = self._resolve_entry(rel_str)
            if target is None:
                continue
            with target:
                if _kind_of(target.stat()) != 'file':
                    continue
                try:
                    raw = target.read_bytes()
                except OSError:  # pragma: no cover
                    continue
            if _is_binary(raw):
                continue
            text = raw.decode('utf-8', errors='replace')
            matches, truncated = _matching_lines(text, compiled, rel_str, self._max_search_results - len(results))
            results.extend(matches)
            if truncated:
                results.append(f'[... truncated at {self._max_search_results} matches]')
                break

        return '\n'.join(results) if results else 'No matches found.'

    @_recoverable
    async def find_files(self, pattern: str, *, path: str = '.') -> str:
        """Find files by glob pattern (name matching, not content search).

        Args:
            pattern: Glob pattern to match, relative to `path` (e.g. '*.py',
                '**/*.json'). Absolute patterns are rejected.
            path: Directory to search in, relative to the root directory.

        Returns:
            Newline-separated list of matching file paths relative to root.
        """
        if os.path.isabs(pattern):
            raise ValueError(f'Pattern {pattern!r} must be relative to the search path, not absolute.')

        # See list_directory: the find root isn't gated by allowed_patterns;
        # matched entries are filtered per-entry below.
        with self._resolve_for(path, check_allowed=False, missing=f'Not a directory: {path}') as resolved:
            if _kind_of(resolved.stat()) != 'dir':
                raise NotADirectoryError(f'Not a directory: {path}')
            base = resolved.path

        matches: list[str] = []
        for match in sorted(base.glob(pattern)):
            try:
                rel_path = match.relative_to(self._real_root)
            except ValueError:  # pragma: no cover
                continue
            if any(part.startswith('.') for part in rel_path.parts):
                continue
            target = self._resolve_entry(str(rel_path))
            if target is None:
                continue
            with target:
                kind = _kind_of(target.stat())
            if kind == 'missing':
                # A dangling symlink resolves inside the root but names nothing.
                continue
            if len(matches) >= self._max_find_results:
                matches.append(f'[... truncated at {self._max_find_results} matches]')
                break
            rel = str(rel_path)
            suffix = '/' if kind == 'dir' else ''
            matches.append(f'{rel}{suffix}')

        return '\n'.join(matches) if matches else 'No matches found.'

    @_recoverable
    async def create_directory(self, path: str) -> str:
        """Create a directory and any missing parents.

        Args:
            path: Directory path relative to the root directory.

        Returns:
            Confirmation message.
        """
        with self._resolve_for(path, intent='mkdir', write=True, missing=f'Path not found: {path}') as resolved:
            if resolved.fd is None:
                resolved.path.mkdir(parents=True, exist_ok=True)
        return f'Created directory: {path}'

    @_recoverable
    async def file_info(self, path: str) -> str:
        """Get metadata about a file or directory.

        Args:
            path: File or directory path relative to the root directory.

        Returns:
            Formatted metadata including size, type, and permissions.
        """
        with self._resolve_for(path, missing=f'Path not found: {path}') as resolved:
            st = resolved.stat()
            if st is None:
                raise FileNotFoundError(f'Path not found: {path}')
            kind = _kind_of(st)

            # Whether the path as given is a symlink, and where it points. This
            # is display-only metadata about the name, not the object read, so
            # a pathname lookup is fine here.
            original = self._root / path
            is_link = original.is_symlink()

            parts = [f'path: {path}', f'type: {"directory" if kind == "dir" else "file"}', f'size: {st.st_size} bytes']

            if kind == 'file':
                raw = resolved.read_bytes()
                is_bin = _is_binary(raw)
                parts.append(f'binary: {is_bin}')
                if not is_bin:
                    text = raw.decode('utf-8', errors='replace')
                    parts.append(f'lines: {len(text.splitlines())}')
                    parts.append(f'hash: {_content_hash(text)}')

            if is_link:
                parts.append(f'symlink_target: {os.readlink(original)}')

        return '\n'.join(parts)
