from __future__ import annotations

import posixpath
from pathlib import Path

import pytest
from pydantic_ai.sandboxes import LocalSandbox, Sandbox

from pydantic_ai_harness._sandbox_state import default_state_dir


@pytest.mark.anyio
async def test_default_state_dir_creates_and_returns(tmp_path: Path) -> None:
    async with LocalSandbox(root=tmp_path) as backend:
        sandbox = Sandbox.wrap(backend)
        state = await default_state_dir(sandbox, 'demo')
        assert state == posixpath.join(await sandbox.working_dir(), '.pydantic-ai-harness', 'demo')
        assert (tmp_path / '.pydantic-ai-harness' / 'demo').is_dir()


@pytest.mark.anyio
@pytest.mark.parametrize('name', ['', '.', '..', 'a/b'])
async def test_default_state_dir_rejects_bad_name(name: str) -> None:
    async with LocalSandbox() as backend:
        with pytest.raises(ValueError, match='invalid capability name'):
            await default_state_dir(Sandbox.wrap(backend), name)
