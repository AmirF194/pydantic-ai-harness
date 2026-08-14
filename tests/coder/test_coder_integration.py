import os
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
from inline_snapshot import snapshot
from pydantic_ai import (
    Agent,
    ModelRequest,
    ModelResponse,
    RequestUsage,
    RetryPromptPart,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)

from pydantic_ai_harness import Coder
from pydantic_ai_harness.repo_context import AgentContextInventory, AssetRoot

if TYPE_CHECKING:

    def IsDatetime(*args: Any, **kwargs: Any) -> datetime: ...
    def IsInstance(expected_type: type[RequestUsage], **kwargs: Any) -> RequestUsage: ...
    def IsStr(*args: Any, **kwargs: Any) -> str: ...
else:
    from dirty_equals import IsDatetime, IsInstance, IsStr

pytestmark = pytest.mark.anyio


@pytest.mark.vcr
async def test_coder_completes_task(
    tmp_path: Path, allow_model_requests: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv('ANTHROPIC_API_KEY', os.environ.get('ANTHROPIC_API_KEY', 'replay-key'))
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

    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='Run the tests, fix the bug they catch, and re-run them to confirm.',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsDatetime(),
                instructions="""\
<context-file path="AGENTS.md">
Run tests with `pytest`. Keep the implementation minimal.

</context-file>

Call `inventory_agent_context` to map where this repo keeps its coding-assistant setup (instruction dirs, skills, sub-agents, and hooks) so you can read and translate it.
You have a planning tool, `write_plan`. For multi-step work, call it first to lay out the steps, then keep it current: mark exactly one step `in_progress`, and mark a step `completed` as soon as it is fully done. Pass the full plan every time you call `write_plan`. Use `add_task` to append a single step, `update_task_status`/`update_task_statuses` to move steps between statuses, and `read_plan` to see step ids before a granular edit.
You can delegate self-contained tasks to these sub-agents using the `delegate_task` tool. Each runs in its own fresh context and does not see this conversation, so pass everything it needs.

Available sub-agents:
- explorer: Explore the codebase and answer questions without modifying anything\
""",
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content='',
                        signature='CAISmgIKiAEIEBgCKkAc4Hu4EDWflpzHaAR+bnDts1WgZX7qNHMU5n61GWmwotlYslqkJr7FIMr9lu9FUeD3l1csLe79smb6qt3HLNfUMg5jbGF1ZGUtZmFibGUtNTgBQgh0aGlua2luZ1okNDRhZTY3NmMtOTU4Zi00ZDY4LTkxMDgtZWFlOWRlN2IzNjZiEgy4AcyLNDcG3R420sIaDGJqTuylYpcYAp0S4SIwNbDHobVqwgFyQ6KIXanzdrhgCGIK+9hqaq+62buq09yiCRQvmIb55aLxl5USWB/6Kj+K1uUfWmwjK6Qvj9PM+5I64eivNb9+u3iRH/BhivIG61LDir2IM9J0a5s+qGwDkmjJ5TJBW3z+wrWVQ7tcLg0YAQ==',
                        provider_name='anthropic',
                    ),
                    ToolCallPart(
                        tool_name='inventory_agent_context',
                        args={},
                        tool_call_id='toolu_01KqR5a3RB88eBkqHwZiDgHx',
                    ),
                    ToolCallPart(
                        tool_name='run_command',
                        args={'command': 'pytest'},
                        tool_call_id='toolu_01FWCZRAfoGykchXthSaSjSv',
                    ),
                ],
                usage=IsInstance(RequestUsage),
                model_name='claude-fable-5',
                timestamp=IsDatetime(),
                provider_name='anthropic',
                provider_url='https://api.anthropic.com',
                provider_details={'finish_reason': 'tool_use'},
                provider_response_id='msg_011Ce3J22m5dPagVUrxfaRHA',
                finish_reason='tool_call',
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='inventory_agent_context',
                        content=AgentContextInventory(
                            roots=[
                                AssetRoot(root='.claude', exists=False),
                                AssetRoot(root='.agents', exists=False),
                                AssetRoot(
                                    root='.codex',
                                    exists=False,
                                    notes='Codex uses TOML config; assets are derived from the .claude/.agents setup.',
                                ),
                                AssetRoot(
                                    root='.grok',
                                    exists=False,
                                    notes='Grok setup is derived from the .claude/.agents setup.',
                                ),
                            ]
                        ),
                        tool_call_id='toolu_01KqR5a3RB88eBkqHwZiDgHx',
                        timestamp=IsDatetime(),
                    ),
                    ToolReturnPart(
                        tool_name='run_command',
                        content=IsStr(
                            regex="""\
\\[stdout\\]\\
=============================\\ test\\ session\\ starts\\ ==============================\\
platform\\ linux\\ \\-\\-\\ Python\\ 3\\.14\\.3,\\ pytest\\-9\\.0\\.3,\\ pluggy\\-1\\.6\\.0\\
rootdir:\\ /tmp/pytest\\-of\\-DouweM/pytest\\-\\d+/test_coder_completes_task0/workspace\\
plugins:\\ examples\\-0\\.0\\.18,\\ recording\\-0\\.13\\.4,\\ inline\\-snapshot\\-0\\.33\\.0,\\ anyio\\-4\\.12\\.1,\\ logfire\\-4\\.33\\.0\\
collected\\ 1\\ item\\
\\
test_add\\.py\\ F\\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\[100%\\]\\
\\
===================================\\ FAILURES\\ ===================================\\
___________________________________\\ test_add\\ ___________________________________\\
\\
\\ \\ \\ \\ def\\ test_add\\(\\)\\ \\->\\ None:\\
>\\ \\ \\ \\ \\ \\ \\ assert\\ add\\(2,\\ 3\\)\\ ==\\ 5\\
E\\ \\ \\ \\ \\ \\ \\ assert\\ \\-1\\ ==\\ 5\\
E\\ \\ \\ \\ \\ \\ \\ \\ \\+\\ \\ where\\ \\-1\\ =\\ add\\(2,\\ 3\\)\\
\\
test_add\\.py:5:\\ AssertionError\\
===========================\\ short\\ test\\ summary\\ info\\ ============================\\
FAILED\\ test_add\\.py::test_add\\ \\-\\ assert\\ \\-1\\ ==\\ 5\\
==============================\\ 1\\ failed\\ in\\ 0\\.05s\\ ===============================\\
\\
\\[exit\\ code:\\ 1\\]\
"""
                        ),
                        tool_call_id='toolu_01FWCZRAfoGykchXthSaSjSv',
                        timestamp=IsDatetime(),
                    ),
                ],
                timestamp=IsDatetime(),
                instructions="""\
<context-file path="AGENTS.md">
Run tests with `pytest`. Keep the implementation minimal.

</context-file>

Call `inventory_agent_context` to map where this repo keeps its coding-assistant setup (instruction dirs, skills, sub-agents, and hooks) so you can read and translate it.
You have a planning tool, `write_plan`. For multi-step work, call it first to lay out the steps, then keep it current: mark exactly one step `in_progress`, and mark a step `completed` as soon as it is fully done. Pass the full plan every time you call `write_plan`. Use `add_task` to append a single step, `update_task_status`/`update_task_statuses` to move steps between statuses, and `read_plan` to see step ids before a granular edit.
You can delegate self-contained tasks to these sub-agents using the `delegate_task` tool. Each runs in its own fresh context and does not see this conversation, so pass everything it needs.

Available sub-agents:
- explorer: Explore the codebase and answer questions without modifying anything\
""",
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='read_file',
                        args={'path': 'add.py'},
                        tool_call_id='toolu_01UC7qkB7QKhTSFTGyM23xPU',
                    )
                ],
                usage=IsInstance(RequestUsage),
                model_name='claude-fable-5',
                timestamp=IsDatetime(),
                provider_name='anthropic',
                provider_url='https://api.anthropic.com',
                provider_details={'finish_reason': 'tool_use'},
                provider_response_id='msg_011Ce3J2Q6izo4VLGxbRUG9q',
                finish_reason='tool_call',
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    RetryPromptPart(
                        content='File not found: add.py',
                        tool_name='read_file',
                        tool_call_id='toolu_01UC7qkB7QKhTSFTGyM23xPU',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsDatetime(),
                instructions="""\
<context-file path="AGENTS.md">
Run tests with `pytest`. Keep the implementation minimal.

</context-file>

Call `inventory_agent_context` to map where this repo keeps its coding-assistant setup (instruction dirs, skills, sub-agents, and hooks) so you can read and translate it.
You have a planning tool, `write_plan`. For multi-step work, call it first to lay out the steps, then keep it current: mark exactly one step `in_progress`, and mark a step `completed` as soon as it is fully done. Pass the full plan every time you call `write_plan`. Use `add_task` to append a single step, `update_task_status`/`update_task_statuses` to move steps between statuses, and `read_plan` to see step ids before a granular edit.
You can delegate self-contained tasks to these sub-agents using the `delegate_task` tool. Each runs in its own fresh context and does not see this conversation, so pass everything it needs.

Available sub-agents:
- explorer: Explore the codebase and answer questions without modifying anything\
""",
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='read_file',
                        args={'path': 'test_add.py'},
                        tool_call_id='toolu_01JJSQj57CaEdQMuoERGDmyy',
                    ),
                    ToolCallPart(
                        tool_name='list_directory',
                        args={},
                        tool_call_id='toolu_01BGuPfxqCbNxewpRLHQ8tnS',
                    ),
                ],
                usage=IsInstance(RequestUsage),
                model_name='claude-fable-5',
                timestamp=IsDatetime(),
                provider_name='anthropic',
                provider_url='https://api.anthropic.com',
                provider_details={'finish_reason': 'tool_use'},
                provider_response_id='msg_011Ce3J2fWBFJJ3iyrR7fg4C',
                finish_reason='tool_call',
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='read_file',
                        content="""\
[test_add.py | 5 lines | hash:6cb6c49877c7]
     1	from calculator import add
     2
     3
     4	def test_add() -> None:
     5	    assert add(2, 3) == 5
""",
                        tool_call_id='toolu_01JJSQj57CaEdQMuoERGDmyy',
                        timestamp=IsDatetime(),
                    ),
                    ToolReturnPart(
                        tool_name='list_directory',
                        content="""\
AGENTS.md  (58 bytes)
README.md  (56 bytes)
__pycache__/
calculator.py  (49 bytes)
test_add.py  (79 bytes)\
""",
                        tool_call_id='toolu_01BGuPfxqCbNxewpRLHQ8tnS',
                        timestamp=IsDatetime(),
                    ),
                ],
                timestamp=IsDatetime(),
                instructions="""\
<context-file path="AGENTS.md">
Run tests with `pytest`. Keep the implementation minimal.

</context-file>

Call `inventory_agent_context` to map where this repo keeps its coding-assistant setup (instruction dirs, skills, sub-agents, and hooks) so you can read and translate it.
You have a planning tool, `write_plan`. For multi-step work, call it first to lay out the steps, then keep it current: mark exactly one step `in_progress`, and mark a step `completed` as soon as it is fully done. Pass the full plan every time you call `write_plan`. Use `add_task` to append a single step, `update_task_status`/`update_task_statuses` to move steps between statuses, and `read_plan` to see step ids before a granular edit.
You can delegate self-contained tasks to these sub-agents using the `delegate_task` tool. Each runs in its own fresh context and does not see this conversation, so pass everything it needs.

Available sub-agents:
- explorer: Explore the codebase and answer questions without modifying anything\
""",
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='read_file',
                        args={'path': 'calculator.py'},
                        tool_call_id='toolu_01SMpkPz3c11M9od4rBk3Sqs',
                    )
                ],
                usage=IsInstance(RequestUsage),
                model_name='claude-fable-5',
                timestamp=IsDatetime(),
                provider_name='anthropic',
                provider_url='https://api.anthropic.com',
                provider_details={'finish_reason': 'tool_use'},
                provider_response_id='msg_011Ce3J2vUqKebCVE6bgeNPY',
                finish_reason='tool_call',
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='read_file',
                        content="""\
[calculator.py | 2 lines | hash:9445c47952ab]
     1	def add(a: int, b: int) -> int:
     2	    return a - b
""",
                        tool_call_id='toolu_01SMpkPz3c11M9od4rBk3Sqs',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsDatetime(),
                instructions="""\
<context-file path="AGENTS.md">
Run tests with `pytest`. Keep the implementation minimal.

</context-file>

Call `inventory_agent_context` to map where this repo keeps its coding-assistant setup (instruction dirs, skills, sub-agents, and hooks) so you can read and translate it.
You have a planning tool, `write_plan`. For multi-step work, call it first to lay out the steps, then keep it current: mark exactly one step `in_progress`, and mark a step `completed` as soon as it is fully done. Pass the full plan every time you call `write_plan`. Use `add_task` to append a single step, `update_task_status`/`update_task_statuses` to move steps between statuses, and `read_plan` to see step ids before a granular edit.
You can delegate self-contained tasks to these sub-agents using the `delegate_task` tool. Each runs in its own fresh context and does not see this conversation, so pass everything it needs.

Available sub-agents:
- explorer: Explore the codebase and answer questions without modifying anything\
""",
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content='',
                        signature='CAIS/QEKiAEIEBgCKkCEzsLt8dXU3sNjWCjfh/BFFl9ZVxugbCZLmtHU7iWugyYytJb2Fix70MdxrVByFtzskocciBZDoMoQGlIL9byyMg5jbGF1ZGUtZmFibGUtNTgBQgh0aGlua2luZ1okNDRhZTY3NmMtOTU4Zi00ZDY4LTkxMDgtZWFlOWRlN2IzNjZiEgxN32FmDlQo1AdplnMaDMZ8SxTUjTtaOqvexyIwybbfzp7W+mhnBsXhKNEC+GlQyhXOFrZ/ajvC1a2cEt5MeT1Gb4p+vw0GAtfXpPiBKiKeEsuyfIKo771ggKjXAC0F9jA+V5PPbcQeaneERhX8km9rGAE=',
                        provider_name='anthropic',
                    ),
                    ToolCallPart(
                        tool_name='edit_file',
                        args={
                            'path': 'calculator.py',
                            'old_text': '    return a - b',
                            'new_text': '    return a + b',
                            'expected_hash': '9445c47952ab',
                        },
                        tool_call_id='toolu_01P1BRhTQtkZ2Vvq4K9ZFgas',
                    ),
                ],
                usage=IsInstance(RequestUsage),
                model_name='claude-fable-5',
                timestamp=IsDatetime(),
                provider_name='anthropic',
                provider_url='https://api.anthropic.com',
                provider_details={'finish_reason': 'tool_use'},
                provider_response_id='msg_011Ce3J39ZPipbAc4QBWCg1i',
                finish_reason='tool_call',
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='edit_file',
                        content='Edited calculator.py. [hash:b414e3e8a1cc]',
                        tool_call_id='toolu_01P1BRhTQtkZ2Vvq4K9ZFgas',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsDatetime(),
                instructions="""\
<context-file path="AGENTS.md">
Run tests with `pytest`. Keep the implementation minimal.

</context-file>

Call `inventory_agent_context` to map where this repo keeps its coding-assistant setup (instruction dirs, skills, sub-agents, and hooks) so you can read and translate it.
You have a planning tool, `write_plan`. For multi-step work, call it first to lay out the steps, then keep it current: mark exactly one step `in_progress`, and mark a step `completed` as soon as it is fully done. Pass the full plan every time you call `write_plan`. Use `add_task` to append a single step, `update_task_status`/`update_task_statuses` to move steps between statuses, and `read_plan` to see step ids before a granular edit.
You can delegate self-contained tasks to these sub-agents using the `delegate_task` tool. Each runs in its own fresh context and does not see this conversation, so pass everything it needs.

Available sub-agents:
- explorer: Explore the codebase and answer questions without modifying anything\
""",
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='run_command',
                        args={'command': 'pytest'},
                        tool_call_id='toolu_01SkSf5RzibFcAgodKPctjL5',
                    )
                ],
                usage=IsInstance(RequestUsage),
                model_name='claude-fable-5',
                timestamp=IsDatetime(),
                provider_name='anthropic',
                provider_url='https://api.anthropic.com',
                provider_details={'finish_reason': 'tool_use'},
                provider_response_id='msg_011Ce3J3V5tX1ZHusNLvzqb9',
                finish_reason='tool_call',
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='run_command',
                        content=IsStr(
                            regex="""\
\\[stdout\\]\\
=============================\\ test\\ session\\ starts\\ ==============================\\
platform\\ linux\\ \\-\\-\\ Python\\ 3\\.14\\.3,\\ pytest\\-9\\.0\\.3,\\ pluggy\\-1\\.6\\.0\\
rootdir:\\ /tmp/pytest\\-of\\-DouweM/pytest\\-\\d+/test_coder_completes_task0/workspace\\
plugins:\\ examples\\-0\\.0\\.18,\\ recording\\-0\\.13\\.4,\\ inline\\-snapshot\\-0\\.33\\.0,\\ anyio\\-4\\.12\\.1,\\ logfire\\-4\\.33\\.0\\
collected\\ 1\\ item\\
\\
test_add\\.py\\ \\.\\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\[100%\\]\\
\\
==============================\\ 1\\ passed\\ in\\ 0\\.00s\\ ===============================\\
"""
                        ),
                        tool_call_id='toolu_01SkSf5RzibFcAgodKPctjL5',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsDatetime(),
                instructions="""\
<context-file path="AGENTS.md">
Run tests with `pytest`. Keep the implementation minimal.

</context-file>

Call `inventory_agent_context` to map where this repo keeps its coding-assistant setup (instruction dirs, skills, sub-agents, and hooks) so you can read and translate it.
You have a planning tool, `write_plan`. For multi-step work, call it first to lay out the steps, then keep it current: mark exactly one step `in_progress`, and mark a step `completed` as soon as it is fully done. Pass the full plan every time you call `write_plan`. Use `add_task` to append a single step, `update_task_status`/`update_task_statuses` to move steps between statuses, and `read_plan` to see step ids before a granular edit.
You can delegate self-contained tasks to these sub-agents using the `delegate_task` tool. Each runs in its own fresh context and does not see this conversation, so pass everything it needs.

Available sub-agents:
- explorer: Explore the codebase and answer questions without modifying anything\
""",
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    TextPart(
                        content='Done. The bug was in `calculator.py`: `add` was subtracting (`a - b`) instead of adding. I changed it to `a + b`, and the test suite now passes (1 passed).'
                    )
                ],
                usage=IsInstance(RequestUsage),
                model_name='claude-fable-5',
                timestamp=IsDatetime(),
                provider_name='anthropic',
                provider_url='https://api.anthropic.com',
                provider_details={'finish_reason': 'end_turn'},
                provider_response_id='msg_011Ce3J3kpwCdVYGrmBpABcM',
                finish_reason='stop',
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )
    assert 'return a + b' in (workspace / 'calculator.py').read_text()
    tool_names = {
        part.tool_name for message in result.all_messages() for part in message.parts if isinstance(part, ToolCallPart)
    }
    assert tool_names & {'edit_file', 'write_file'}
    assert 'run_command' in tool_names
