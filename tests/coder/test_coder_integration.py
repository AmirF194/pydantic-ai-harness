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
                        signature='CAISlwIKiAEIEBgCKkDLDkNar0Nh8/XX2HNp4w26tyoe9Ke6/UMoq1N8JkVAlg72EiA1H/My1Hz6Wc1mulmLP7rylcyionq3KXi/wRygMg5jbGF1ZGUtZmFibGUtNTgBQgh0aGlua2luZ1okNDRhZTY3NmMtOTU4Zi00ZDY4LTkxMDgtZWFlOWRlN2IzNjZiEgwErt4wCiH1K3TKQckaDIADW4fIZzop7XqQ6CIw3+uqO/ijL36zMzpUyhJyrA7Xk1Lyf9mGQKxSbqYRDQmJg+dfZkPBiTjtpFbovo2iKjyk6csWtVcARDb03ZR1dyELCtcoBN/PjHQMTFCrg2ilLbjGubyhmXETiH9X8Rjq0x4++TiVNIrW5LIxhDEYAQ==',
                        provider_name='anthropic',
                    ),
                    ToolCallPart(
                        tool_name='inventory_agent_context',
                        args={},
                        tool_call_id='toolu_01VoeLSNUJcBQvr5ycMFWh75',
                    ),
                    ToolCallPart(
                        tool_name='run_command',
                        args={'command': 'pytest -x -q 2>&1 | tail -30'},
                        tool_call_id='toolu_01XxfoZCLa3rKmP11WXmvgxR',
                    ),
                ],
                usage=IsInstance(RequestUsage),
                model_name='claude-fable-5',
                timestamp=IsDatetime(),
                provider_name='anthropic',
                provider_url='https://api.anthropic.com',
                provider_details={'finish_reason': 'tool_use'},
                provider_response_id='msg_011Ce3KjtBvw3dRA1pSVWkuA',
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
                        tool_call_id='toolu_01VoeLSNUJcBQvr5ycMFWh75',
                        timestamp=IsDatetime(),
                    ),
                    ToolReturnPart(
                        tool_name='run_command',
                        content="""\
[stdout]
F
=================================== FAILURES ===================================
___________________________________ test_add ___________________________________

    def test_add() -> None:
>       assert add(2, 3) == 5
E       assert -1 == 5
E        +  where -1 = add(2, 3)

test_add.py:5: AssertionError
=========================== short test summary info ============================
FAILED test_add.py::test_add - assert -1 == 5
!!!!!!!!!!!!!!!!!!!!!!!!!! stopping after 1 failures !!!!!!!!!!!!!!!!!!!!!!!!!!!
1 failed in 0.05s
""",
                        tool_call_id='toolu_01XxfoZCLa3rKmP11WXmvgxR',
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
                        tool_name='run_command',
                        args={'command': 'cat test_add.py; grep -rn "def add" --include=*.py .'},
                        tool_call_id='toolu_01B8DTjAgCPzgEztaACLwjKT',
                    )
                ],
                usage=IsInstance(RequestUsage),
                model_name='claude-fable-5',
                timestamp=IsDatetime(),
                provider_name='anthropic',
                provider_url='https://api.anthropic.com',
                provider_details={'finish_reason': 'tool_use'},
                provider_response_id='msg_011Ce3KkGaaGdrt4ZDHUjDRY',
                finish_reason='tool_call',
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='run_command',
                        content="""\
[stdout]
from calculator import add


def test_add() -> None:
    assert add(2, 3) == 5
./calculator.py:1:def add(a: int, b: int) -> int:
""",
                        tool_call_id='toolu_01B8DTjAgCPzgEztaACLwjKT',
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
                        args={'path': 'calculator.py'},
                        tool_call_id='toolu_011sziyqgZcvhDiQRL6yv4x3',
                    )
                ],
                usage=IsInstance(RequestUsage),
                model_name='claude-fable-5',
                timestamp=IsDatetime(),
                provider_name='anthropic',
                provider_url='https://api.anthropic.com',
                provider_details={'finish_reason': 'tool_use'},
                provider_response_id='msg_011Ce3KkXWzrnheQifwBQ32Y',
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
                        tool_call_id='toolu_011sziyqgZcvhDiQRL6yv4x3',
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
                        tool_name='edit_file',
                        args={
                            'path': 'calculator.py',
                            'old_text': '    return a - b',
                            'new_text': '    return a + b',
                            'expected_hash': '9445c47952ab',
                        },
                        tool_call_id='toolu_01BsYsyjFhFHDPA4fj2W2iAw',
                    )
                ],
                usage=IsInstance(RequestUsage),
                model_name='claude-fable-5',
                timestamp=IsDatetime(),
                provider_name='anthropic',
                provider_url='https://api.anthropic.com',
                provider_details={'finish_reason': 'tool_use'},
                provider_response_id='msg_011Ce3Kkn85og9uF8immfDzP',
                finish_reason='tool_call',
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='edit_file',
                        content='Edited calculator.py. [hash:b414e3e8a1cc]',
                        tool_call_id='toolu_01BsYsyjFhFHDPA4fj2W2iAw',
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
                        args={'command': 'pytest -q'},
                        tool_call_id='toolu_01V1o93RHfN7o4NCp5RKWAUi',
                    )
                ],
                usage=IsInstance(RequestUsage),
                model_name='claude-fable-5',
                timestamp=IsDatetime(),
                provider_name='anthropic',
                provider_url='https://api.anthropic.com',
                provider_details={'finish_reason': 'tool_use'},
                provider_response_id='msg_011Ce3Km6nzHNCeehvzhTaEN',
                finish_reason='tool_call',
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='run_command',
                        content="""\
[stdout]
.                                                                        [100%]
1 passed in 0.00s
""",
                        tool_call_id='toolu_01V1o93RHfN7o4NCp5RKWAUi',
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
                        content='Done. The bug: `add` in `calculator.py` was subtracting (`a - b`) instead of adding. I changed it to `return a + b`, and the test suite now passes (1 passed).'
                    )
                ],
                usage=IsInstance(RequestUsage),
                model_name='claude-fable-5',
                timestamp=IsDatetime(),
                provider_name='anthropic',
                provider_url='https://api.anthropic.com',
                provider_details={'finish_reason': 'end_turn'},
                provider_response_id='msg_011Ce3KmPHBF1JcAEEc9k7B1',
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
