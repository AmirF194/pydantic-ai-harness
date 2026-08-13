import os
from pathlib import Path

import pytest
from pydantic_ai import Agent
from pydantic_ai.messages import ToolCallPart

from pydantic_ai_harness import Coder

pytestmark = pytest.mark.anyio


@pytest.mark.vcr
async def test_coder_completes_task(
    tmp_path: Path, allow_model_requests: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    if 'ANTHROPIC_API_KEY' not in os.environ:
        monkeypatch.setenv('ANTHROPIC_API_KEY', 'replay-key')
    workspace = tmp_path / 'workspace'
    workspace.mkdir()
    (workspace / 'calculator.py').write_text('def add(a: int, b: int) -> int:\n    return a - b\n')
    (workspace / 'test_add.py').write_text(
        'from calculator import add\n\n\ndef test_add() -> None:\n    assert add(2, 3) == 5\n'
    )
    (workspace / 'README.md').write_text('# Calculator\n\nA tiny calculator used by the test suite.\n')
    (workspace / 'AGENTS.md').write_text('Run tests with `pytest`. Keep the implementation minimal.\n')

    agent = Agent('anthropic:claude-fable-5', capabilities=[Coder(workspace)])
    result = await agent.run('Run the tests, fix the bug they catch, and re-run them to confirm.')

    assert 'return a + b' in (workspace / 'calculator.py').read_text()
    tool_names = {
        part.tool_name for message in result.all_messages() for part in message.parts if isinstance(part, ToolCallPart)
    }
    assert tool_names & {'edit_file', 'write_file'}
    assert 'run_command' in tool_names
