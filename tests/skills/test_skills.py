from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest
from pydantic_ai import Agent
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import LoadCapabilityReturnPart, ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from pydantic_ai_harness.code_mode import CodeMode
from pydantic_ai_harness.filesystem import FileSystem
from pydantic_ai_harness.shell import Shell
from pydantic_ai_harness.skills import Skills

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


def _write_skill(
    library: Path,
    name: str,
    *,
    description: str = 'Help with the task.',
    body: str = 'Follow these directions.',
    frontmatter: str | None = None,
    files: Mapping[str, str] | None = None,
) -> Path:
    directory = library / name
    directory.mkdir(parents=True)
    metadata = frontmatter if frontmatter is not None else f'description: {description}'
    (directory / 'SKILL.md').write_text(f'---\n{metadata}\n---\n\n{body}\n', encoding='utf-8')
    for relative, content in (files or {}).items():
        path = directory / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding='utf-8')
    return directory


async def _activate(
    capabilities: Sequence[AbstractCapability[object]],
    skill_name: str,
) -> tuple[str, list[list[str]], str]:
    observed_tools: list[list[str]] = []
    initial_instructions = ''
    loaded_instructions = ''

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal initial_instructions, loaded_instructions
        observed_tools.append([tool.name for tool in info.function_tools])
        returns = [part for message in messages for part in message.parts if isinstance(part, LoadCapabilityReturnPart)]
        if not returns:
            initial_instructions = info.instructions or ''
            return ModelResponse(parts=[ToolCallPart(tool_name='load_capability', args={'id': skill_name})])
        instructions = returns[-1].content.get('instructions')
        assert instructions is not None
        loaded_instructions = instructions
        return ModelResponse(parts=[TextPart('done')])

    agent: Agent[object, str] = Agent(FunctionModel(model_fn), capabilities=capabilities)
    await agent.run('use the skill')
    return initial_instructions, observed_tools, loaded_instructions


class TestSkills:
    async def test_each_skill_is_a_deferred_core_capability(self, tmp_path: Path) -> None:
        library = tmp_path / 'skills'
        _write_skill(library, 'alpha', description='Alpha help.')
        _write_skill(library, 'beta', description='Beta help.')

        initial, tools, loaded = await _activate([Skills(directories=[library])], 'beta')

        assert '- alpha: Alpha help.' in initial
        assert '- beta: Beta help.' in initial
        assert tools[0] == ['load_capability']
        assert '# Skill: beta' in loaded
        assert 'Follow these directions.' in loaded
        assert 'description: Beta help.' not in loaded

    async def test_knowledge_only_skill_needs_no_filesystem(self, tmp_path: Path) -> None:
        library = tmp_path / 'skills'
        directory = _write_skill(library, 'knowledge', body='Answer from this embedded guidance.')

        _, _, loaded = await _activate([Skills(directories=[library])], 'knowledge')

        assert f'Skill directory: `{directory.resolve()}`.' in loaded
        assert 'Answer from this embedded guidance.' in loaded

    async def test_empty_skill_body_still_loads_path_directions(self, tmp_path: Path) -> None:
        library = tmp_path / 'skills'
        _write_skill(library, 'empty', body='')

        _, _, loaded = await _activate([Skills(directories=[library])], 'empty')

        assert '# Skill: empty' in loaded
        assert loaded.endswith('Resolve relative paths in this skill against that directory.')

    async def test_resource_skill_directs_first_party_filesystem(self, tmp_path: Path) -> None:
        library = tmp_path / 'skills'
        _write_skill(
            library,
            'research',
            body='Read references/guide.md.',
            files={'references/guide.md': 'guide'},
        )

        _, tools, loaded = await _activate(
            [Skills(directories=[library]), FileSystem(root_dir=tmp_path)],
            'research',
        )

        assert 'read_file' in tools[-1]
        assert 'Use `read_file` to read supporting skill files.' in loaded
        assert 'Prefix skill-relative paths with `skills/research/`' in loaded

    @pytest.mark.parametrize('dynamic_catalog', [False, True])
    async def test_resource_skill_composes_with_code_mode(self, tmp_path: Path, dynamic_catalog: bool) -> None:
        library = tmp_path / 'skills'
        _write_skill(library, 'research', files={'references/guide.md': 'guide'})

        _, tools, loaded = await _activate(
            [
                Skills(directories=[library]),
                FileSystem(root_dir=tmp_path),
                CodeMode[object](dynamic_catalog=dynamic_catalog),
            ],
            'research',
        )

        assert 'run_code' in tools[-1]
        assert 'read_file' not in tools[-1]
        assert 'Use `read_file` to read supporting skill files.' in loaded
        assert 'run_code' not in loaded

    async def test_visible_read_file_does_not_replace_filesystem_authority(self, tmp_path: Path) -> None:
        library = tmp_path / 'skills'
        _write_skill(library, 'research', files={'guide.md': 'guide'})
        skills = Skills[object](directories=[library])
        calls = 0

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            nonlocal calls
            calls += 1
            if calls == 1:
                return ModelResponse(parts=[ToolCallPart(tool_name='load_capability', args={'id': 'research'})])
            return ModelResponse(parts=[TextPart('done')])  # pragma: no cover - activation fails first

        agent: Agent[object, str] = Agent(FunctionModel(model_fn), capabilities=[skills])

        @agent.tool_plain
        def read_file(path: str) -> str:
            return path  # pragma: no cover - only the tool definition is inspected

        with pytest.raises(UserError, match='no `FileSystem` capability is configured'):
            await agent.run('go')

    async def test_reader_root_can_make_skill_directory_the_tool_root(self, tmp_path: Path) -> None:
        library = tmp_path / 'skills'
        directory = _write_skill(library, 'research', files={'guide.md': 'guide'})
        skills = Skills(directories=[library], reader_tool='fetch_asset', reader_root=directory)

        observed = ''

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            nonlocal observed
            returns = [
                part for message in messages for part in message.parts if isinstance(part, LoadCapabilityReturnPart)
            ]
            if not returns:
                return ModelResponse(parts=[ToolCallPart(tool_name='load_capability', args={'id': 'research'})])
            instructions = returns[-1].content.get('instructions')
            assert instructions is not None
            observed = instructions
            return ModelResponse(parts=[TextPart('done')])

        agent: Agent[object, str] = Agent(FunctionModel(model_fn), capabilities=[skills])

        await agent.run('go')
        assert 'Use `fetch_asset`' in observed
        assert 'pass skill-relative paths as written' in observed

    async def test_custom_reader_without_root_uses_absolute_paths(self, tmp_path: Path) -> None:
        library = tmp_path / 'skills'
        directory = _write_skill(library, 'research', files={'guide.md': 'guide'})
        skills = Skills(directories=[library], reader_tool='fetch_asset')

        observed = ''

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            nonlocal observed
            returns = [
                part for message in messages for part in message.parts if isinstance(part, LoadCapabilityReturnPart)
            ]
            if not returns:
                return ModelResponse(parts=[ToolCallPart(tool_name='load_capability', args={'id': 'research'})])
            instructions = returns[-1].content.get('instructions')
            assert instructions is not None
            observed = instructions
            return ModelResponse(parts=[TextPart('done')])

        agent: Agent[object, str] = Agent(FunctionModel(model_fn), capabilities=[skills])

        await agent.run('go')
        assert f'Pass absolute paths rooted at `{directory.resolve()}`.' in observed

    async def test_resource_skill_errors_when_reader_is_missing(self, tmp_path: Path) -> None:
        library = tmp_path / 'skills'
        _write_skill(library, 'research', files={'guide.md': 'guide'})

        with pytest.raises(UserError, match='no `FileSystem` capability is configured'):
            await _activate([Skills(directories=[library])], 'research')

    async def test_resource_skill_errors_when_outside_filesystem_root(self, tmp_path: Path) -> None:
        library = tmp_path / 'skills'
        _write_skill(library, 'research', files={'guide.md': 'guide'})
        other_root = tmp_path / 'other'
        other_root.mkdir()

        with pytest.raises(UserError, match='outside every configured `FileSystem` root'):
            await _activate(
                [Skills(directories=[library]), FileSystem(root_dir=other_root)],
                'research',
            )

    async def test_reader_root_must_contain_skill(self, tmp_path: Path) -> None:
        library = tmp_path / 'skills'
        _write_skill(library, 'research', files={'guide.md': 'guide'})

        with pytest.raises(UserError, match='outside configured reader_root'):
            await _activate(
                [
                    Skills(directories=[library], reader_tool='read_file', reader_root=tmp_path / 'other'),
                    FileSystem(root_dir=tmp_path),
                ],
                'research',
            )

    async def test_scripts_are_directed_to_shell(self, tmp_path: Path) -> None:
        library = tmp_path / 'skills'
        _write_skill(library, 'build', files={'scripts/check.py': 'print("ok")'})

        _, _, loaded = await _activate(
            [Skills(directories=[library]), FileSystem(root_dir=tmp_path), Shell(cwd=tmp_path)],
            'build',
        )

        assert 'Use `run_command` to run bundled skill scripts.' in loaded
        assert 'Prefix skill-relative paths with `skills/build/`' in loaded

    async def test_scripts_remain_optional_without_executor(self, tmp_path: Path) -> None:
        library = tmp_path / 'skills'
        _write_skill(library, 'build', files={'scripts/check.py': 'print("ok")'})

        _, _, loaded = await _activate(
            [Skills(directories=[library]), FileSystem(root_dir=tmp_path)],
            'build',
        )

        assert 'no script-execution tool is configured' in loaded
        assert 'Do not attempt to run them' in loaded

    async def test_custom_executor_is_a_caller_declaration(self, tmp_path: Path) -> None:
        library = tmp_path / 'skills'
        directory = _write_skill(library, 'build', files={'scripts/check.py': 'print("ok")'})
        skills = Skills[object](directories=[library], executor_tool='execute_script')
        observed = ''

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            nonlocal observed
            returns = [
                part for message in messages for part in message.parts if isinstance(part, LoadCapabilityReturnPart)
            ]
            if not returns:
                return ModelResponse(parts=[ToolCallPart(tool_name='load_capability', args={'id': 'build'})])
            instructions = returns[-1].content.get('instructions')
            assert instructions is not None
            observed = instructions
            return ModelResponse(parts=[TextPart('done')])

        agent: Agent[object, str] = Agent(FunctionModel(model_fn), capabilities=[skills, FileSystem(root_dir=tmp_path)])
        await agent.run('go')
        assert 'Use `execute_script` to run bundled skill scripts.' in observed
        assert f'Pass absolute paths rooted at `{directory.resolve()}`.' in observed

    async def test_shell_outside_skill_tree_uses_absolute_paths(self, tmp_path: Path) -> None:
        library = tmp_path / 'skills'
        directory = _write_skill(library, 'build', files={'scripts/check.py': 'print("ok")'})
        shell_root = tmp_path / 'shell'
        shell_root.mkdir()

        _, _, loaded = await _activate(
            [Skills(directories=[library]), FileSystem(root_dir=tmp_path), Shell(cwd=shell_root)],
            'build',
        )

        assert f'Pass absolute paths rooted at `{directory.resolve()}`.' in loaded

    async def test_claude_skill_dir_is_expanded(self, tmp_path: Path) -> None:
        library = tmp_path / 'skills'
        directory = _write_skill(library, 'portable', body='Read ${CLAUDE_SKILL_DIR}/guide.md.')

        _, _, loaded = await _activate([Skills(directories=[library])], 'portable')

        assert f'Read {directory.resolve()}/guide.md.' in loaded

    async def test_construction_is_a_snapshot(self, tmp_path: Path) -> None:
        library = tmp_path / 'skills'
        _write_skill(library, 'first')
        skills = Skills(directories=[library])
        _write_skill(library, 'later')

        initial, _, _ = await _activate([skills], 'first')

        assert '- first:' in initial
        assert '- later:' not in initial

    async def test_agent_spec_constructs_skills(self, tmp_path: Path) -> None:
        library = tmp_path / 'skills'
        _write_skill(library, 'from-spec')
        spec = tmp_path / 'agent.yaml'
        spec.write_text(
            f'capabilities:\n  - Skills:\n      directories:\n        - {library}\n',
            encoding='utf-8',
        )

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            assert 'from-spec' in (info.instructions or '')
            return ModelResponse(parts=[TextPart('done')])

        agent = Agent.from_file(spec, custom_capability_types=[Skills], model=FunctionModel(model_fn))
        result = await agent.run('go')
        assert result.output == 'done'


class TestSkillValidation:
    def test_reader_root_requires_custom_reader(self, tmp_path: Path) -> None:
        library = tmp_path / 'skills'
        _write_skill(library, 'knowledge')

        with pytest.raises(ValueError, match='reader_root requires reader_tool'):
            Skills(directories=[library], reader_root=tmp_path)

    def test_name_can_be_derived_from_directory(self, tmp_path: Path) -> None:
        library = tmp_path / 'skills'
        _write_skill(library, 'derived-name')
        Skills(directories=[library])

    def test_explicit_null_name_is_derived_from_directory(self, tmp_path: Path) -> None:
        library = tmp_path / 'skills'
        _write_skill(library, 'derived-name', frontmatter='name: null\ndescription: Help')
        Skills(directories=[library])

    def test_explicit_name_must_match_directory(self, tmp_path: Path) -> None:
        library = tmp_path / 'skills'
        _write_skill(library, 'actual', frontmatter='name: different\ndescription: Help')

        with pytest.raises(ValueError, match='must match its parent directory'):
            Skills(directories=[library])

    @pytest.mark.parametrize('name', ['Uppercase', '-leading', 'trailing-', 'two--hyphens'])
    def test_invalid_derived_name_is_rejected(self, tmp_path: Path, name: str) -> None:
        library = tmp_path / 'skills'
        _write_skill(library, name)

        with pytest.raises(ValueError, match='Invalid skill name'):
            Skills(directories=[library])

    def test_name_length_is_limited(self, tmp_path: Path) -> None:
        library = tmp_path / 'skills'
        _write_skill(library, 'a' * 65)

        with pytest.raises(ValueError, match='at most 64'):
            Skills(directories=[library])

    @pytest.mark.parametrize(
        ('text', 'error'),
        [
            ('description: no delimiters', 'must start with YAML frontmatter'),
            ('---\ndescription: unclosed', 'unclosed YAML frontmatter'),
            ('---\ndescription: [invalid\n---', 'Invalid YAML frontmatter'),
            ('---\n- description\n---', 'must be a mapping'),
            ('---\nname: okay\n---', 'Invalid Agent Skill frontmatter'),
            ('---\ndescription: "   "\n---', 'must not be empty'),
        ],
    )
    def test_invalid_frontmatter_is_rejected(self, tmp_path: Path, text: str, error: str) -> None:
        directory = tmp_path / 'skills' / 'invalid'
        directory.mkdir(parents=True)
        (directory / 'SKILL.md').write_text(text, encoding='utf-8')

        with pytest.raises(ValueError, match=error):
            Skills(directories=[tmp_path / 'skills'])

    def test_description_is_limited(self, tmp_path: Path) -> None:
        library = tmp_path / 'skills'
        _write_skill(library, 'verbose', description='x' * 1025)

        with pytest.raises(ValueError, match='Invalid Agent Skill frontmatter'):
            Skills(directories=[library])

    def test_duplicate_names_across_roots_are_rejected(self, tmp_path: Path) -> None:
        first = tmp_path / 'first'
        second = tmp_path / 'second'
        _write_skill(first, 'duplicate')
        _write_skill(second, 'duplicate')

        with pytest.raises(ValueError, match="Duplicate skill name 'duplicate'"):
            Skills(directories=[first, second])

    def test_duplicate_root_is_scanned_once(self, tmp_path: Path) -> None:
        library = tmp_path / 'skills'
        _write_skill(library, 'once')
        Skills(directories=[library, library.resolve()])

    def test_missing_root_is_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match='does not exist'):
            Skills(directories=[tmp_path / 'missing'])

    def test_file_root_is_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / 'skills'
        path.write_text('not a directory', encoding='utf-8')

        with pytest.raises(ValueError, match='is not a directory'):
            Skills(directories=[path])

    def test_single_string_is_rejected_with_sequence_hint(self, tmp_path: Path) -> None:
        with pytest.raises(TypeError, match='sequence of skill-library paths'):
            Skills(directories=str(tmp_path))  # type: ignore[arg-type]

    def test_non_skill_children_are_ignored(self, tmp_path: Path) -> None:
        library = tmp_path / 'skills'
        library.mkdir()
        (library / 'README.md').write_text('ordinary file', encoding='utf-8')
        (library / 'not-a-skill').mkdir()
        Skills(directories=[library])

    async def test_nested_skill_md_is_a_supporting_file(self, tmp_path: Path) -> None:
        library = tmp_path / 'skills'
        _write_skill(library, 'outer', files={'references/SKILL.md': 'reference'})

        with pytest.raises(UserError, match='contains supporting files'):
            await _activate([Skills(directories=[library])], 'outer')

    def test_behavioral_fields_are_ignored_with_one_warning(self, tmp_path: Path) -> None:
        library = tmp_path / 'skills'
        _write_skill(
            library,
            'first',
            frontmatter='description: First\nallowed-tools: Read\nmodel: sonnet',
        )
        _write_skill(
            library,
            'second',
            frontmatter='description: Second\ndisable-model-invocation: true',
        )

        with pytest.warns(UserWarning) as caught:
            Skills(directories=[library])

        assert len(caught) == 1
        message = str(caught[0].message)
        assert 'first: allowed-tools, model' in message
        assert 'second: disable-model-invocation' in message

    def test_standard_non_behavioral_fields_are_accepted(self, tmp_path: Path) -> None:
        library = tmp_path / 'skills'
        _write_skill(
            library,
            'standard',
            frontmatter='description: Standard\nlicense: Apache-2.0\ncompatibility: Python\nmetadata:\n  owner: pydantic',
        )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            Skills(directories=[library])

        assert not caught
