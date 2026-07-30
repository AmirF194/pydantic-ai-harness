"""Command policy normalization shared by the Shell capability and toolset."""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass
from typing import Literal

_DEFAULT_DENIED_COMMANDS: frozenset[str] = frozenset(
    {
        'rm',
        'rmdir',
        'mkfs',
        'dd',
        'format',
        'shutdown',
        'reboot',
        'halt',
        'poweroff',
        'init',
    }
)


@dataclass(frozen=True)
class CommandPolicy:
    mode: Literal['allow', 'deny']
    commands: frozenset[str]

    def denial_message(self, executable: str) -> str | None:
        if self.mode == 'allow' and executable not in self.commands:
            return f'Command {executable!r} is not in the allowed list.'
        if self.mode == 'deny' and executable in self.commands:
            return f'Command {executable!r} is denied.'
        return None


def normalize_command_policy(
    allowed_commands: Collection[object] | None,
    denied_commands: Collection[object] | None,
) -> CommandPolicy:
    if allowed_commands is not None and denied_commands is not None:
        raise ValueError('Specify allowed_commands or denied_commands, not both.')

    if allowed_commands is not None:
        return CommandPolicy('allow', _normalize_commands('allowed_commands', allowed_commands))
    if denied_commands is not None:
        return CommandPolicy('deny', _normalize_commands('denied_commands', denied_commands))
    return CommandPolicy('deny', _DEFAULT_DENIED_COMMANDS)


def _normalize_commands(name: str, commands: Collection[object]) -> frozenset[str]:
    if isinstance(commands, str):
        raise TypeError(f'{name} must be a collection of command names, not a string.')

    normalized: set[str] = set()
    for command in commands:
        if not isinstance(command, str):
            raise TypeError(f'{name} must contain only command names as strings.')
        normalized.add(command)
    return frozenset(normalized)
