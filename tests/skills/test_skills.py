from __future__ import annotations

import inspect
import warnings
from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest
from pydantic_ai import Agent
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import (
    LoadCapabilityReturnPart,
    ModelMessage,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel

from pydantic_ai_harness.filesystem import FileSystem
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
    def test_public_constructor_only_exposes_skill_library_configuration(self) -> None:
        assert tuple(inspect.signature(Skills).parameters) == ('directories', 'include', 'exclude')

    async def test_single_library_path_is_accepted(self, tmp_path: Path) -> None:
        library = tmp_path / 'skills'
        _write_skill(library, 'alpha', description='Alpha help.')

        initial, _, _ = await _activate([Skills(library)], 'alpha')

        assert '- alpha: Alpha help.' in initial

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

    async def test_include_exposes_only_selected_skills(self, tmp_path: Path) -> None:
        library = tmp_path / 'skills'
        _write_skill(library, 'alpha', description='Alpha help.')
        _write_skill(library, 'beta', description='Beta help.')

        initial, _, _ = await _activate([Skills(library, include=['beta'])], 'beta')

        assert '- alpha:' not in initial
        assert '- beta: Beta help.' in initial

    async def test_exclude_hides_selected_skills(self, tmp_path: Path) -> None:
        library = tmp_path / 'skills'
        _write_skill(library, 'alpha', description='Alpha help.')
        _write_skill(library, 'beta', description='Beta help.')

        initial, _, _ = await _activate([Skills(library, exclude=['alpha'])], 'beta')

        assert '- alpha:' not in initial
        assert '- beta: Beta help.' in initial

    async def test_skill_body_is_exposed_with_its_directory(self, tmp_path: Path) -> None:
        library = tmp_path / 'skills'
        directory = _write_skill(library, 'knowledge', body='Answer from this embedded guidance.')

        _, _, loaded = await _activate([Skills(directories=[library])], 'knowledge')

        assert f'Skill directory: `{directory.resolve()}`.' in loaded
        assert 'Files referenced by this skill are relative to that directory.' in loaded
        assert 'Answer from this embedded guidance.' in loaded

    async def test_empty_skill_body_still_loads_the_directory_line(self, tmp_path: Path) -> None:
        library = tmp_path / 'skills'
        directory = _write_skill(library, 'empty', body='')

        _, _, loaded = await _activate([Skills(directories=[library])], 'empty')

        assert '# Skill: empty' in loaded
        assert f'Skill directory: `{directory.resolve()}`.' in loaded
        assert loaded.endswith('Access them only through filesystem tools provided by the application.')

    async def test_claude_skill_dir_is_expanded(self, tmp_path: Path) -> None:
        library = tmp_path / 'skills'
        directory = _write_skill(library, 'portable', body='Read ${CLAUDE_SKILL_DIR}/guide.md.')

        _, _, loaded = await _activate([Skills(directories=[library])], 'portable')

        assert f'Read {directory.resolve()}/guide.md.' in loaded

    async def test_bundled_file_can_be_read_when_application_grants_access(self, tmp_path: Path) -> None:
        library = tmp_path / 'skills'
        directory = _write_skill(
            library,
            'portable',
            body='Read ${CLAUDE_SKILL_DIR}/references/guide.md.',
            files={'references/guide.md': 'Use the documented workflow.'},
        )
        observed_content = ''

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            nonlocal observed_content
            returns = [
                part for message in messages for part in message.parts if isinstance(part, LoadCapabilityReturnPart)
            ]
            tool_returns = [
                part
                for message in messages
                for part in message.parts
                if isinstance(part, ToolReturnPart) and part.tool_name == 'read_file'
            ]
            if not returns:
                return ModelResponse(parts=[ToolCallPart(tool_name='load_capability', args={'id': 'portable'})])
            if not tool_returns:
                return ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_name='read_file',
                            args={'path': str(directory / 'references' / 'guide.md')},
                        )
                    ]
                )
            observed_content = str(tool_returns[-1].content)
            return ModelResponse(parts=[TextPart('done')])

        agent = Agent(
            FunctionModel(model_fn),
            capabilities=[FileSystem(root_dir=library, protected_patterns=['**']), Skills(library)],
        )
        await agent.run('use the skill')

        assert 'Use the documented workflow.' in observed_content

    async def test_construction_is_a_snapshot(self, tmp_path: Path) -> None:
        library = tmp_path / 'skills'
        _write_skill(library, 'first')
        skills = Skills(directories=[library])
        _write_skill(library, 'later')

        initial, _, _ = await _activate([skills], 'first')

        assert '- first:' in initial
        assert '- later:' not in initial

    @pytest.mark.parametrize(
        ('suffix', 'content'),
        [
            (
                '.yaml',
                'capabilities:\n  - Skills:\n      directories: {library}\n      include:\n        - from-spec\n',
            ),
            (
                '.json',
                '{{"capabilities": [{{"Skills": {{"directories": "{library}", "include": ["from-spec"]}}}}]}}',
            ),
        ],
    )
    async def test_agent_spec_constructs_skills(self, tmp_path: Path, suffix: str, content: str) -> None:
        library = tmp_path / 'skills'
        _write_skill(library, 'from-spec')
        _write_skill(library, 'not-selected')
        spec = tmp_path / f'agent{suffix}'
        spec.write_text(
            content.format(library=library),
            encoding='utf-8',
        )

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            assert 'from-spec' in (info.instructions or '')
            assert 'not-selected' not in (info.instructions or '')
            return ModelResponse(parts=[TextPart('done')])

        agent = Agent.from_file(spec, custom_capability_types=[Skills], model=FunctionModel(model_fn))
        result = await agent.run('go')
        assert result.output == 'done'


class TestSkillValidation:
    def test_include_and_exclude_are_mutually_exclusive(self, tmp_path: Path) -> None:
        library = tmp_path / 'skills'
        _write_skill(library, 'alpha')

        with pytest.raises(ValueError, match='include and exclude cannot be used together'):
            Skills(library, include=['alpha'], exclude=['alpha'])

    @pytest.mark.parametrize('selector', ['include', 'exclude'])
    def test_unknown_selected_skill_is_rejected(self, tmp_path: Path, selector: str) -> None:
        library = tmp_path / 'skills'
        _write_skill(library, 'available')
        kwargs = {selector: ['missing']}

        with pytest.raises(
            ValueError, match=r'Unknown skill in (include|exclude): missing.*Available skills: available'
        ):
            if selector == 'include':
                Skills(library, include=kwargs[selector])
            else:
                Skills(library, exclude=kwargs[selector])

    @pytest.mark.parametrize('selector', ['include', 'exclude'])
    def test_selector_must_not_be_a_string(self, tmp_path: Path, selector: str) -> None:
        library = tmp_path / 'skills'
        _write_skill(library, 'alpha')
        kwargs = {selector: 'alpha'}

        with pytest.raises(TypeError, match=f'{selector} must be a collection of skill names'):
            if selector == 'include':
                Skills(library, include=kwargs[selector])
            else:
                Skills(library, exclude=kwargs[selector])

    def test_selector_entries_must_be_strings(self, tmp_path: Path) -> None:
        library = tmp_path / 'skills'
        _write_skill(library, 'alpha')

        with pytest.raises(TypeError, match='include must contain only skill names as strings'):
            Skills(library, include=[1])  # type: ignore[list-item]

    def test_multiple_unknown_skills_report_an_empty_library(self, tmp_path: Path) -> None:
        library = tmp_path / 'skills'
        library.mkdir()

        with pytest.raises(
            ValueError,
            match=r'Unknown skills in include: first, second\. Available skills: \(none\)\.',
        ):
            Skills(library, include=['second', 'first'])

    def test_selection_happens_before_frontmatter_parsing(self, tmp_path: Path) -> None:
        library = tmp_path / 'skills'
        _write_skill(library, 'selected')
        _write_skill(library, 'ignored', frontmatter='not: [valid')

        Skills(library, include=['selected'])

    async def test_empty_include_exposes_no_skills(self, tmp_path: Path) -> None:
        library = tmp_path / 'skills'
        _write_skill(library, 'alpha')

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            assert 'alpha' not in (info.instructions or '')
            assert 'load_capability' not in [tool.name for tool in info.function_tools]
            return ModelResponse(parts=[TextPart('done')])

        agent = Agent(FunctionModel(model_fn), capabilities=[Skills(library, include=[])])
        result = await agent.run('go')
        assert result.output == 'done'

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

    def test_non_skill_children_are_ignored(self, tmp_path: Path) -> None:
        library = tmp_path / 'skills'
        library.mkdir()
        (library / 'README.md').write_text('ordinary file', encoding='utf-8')
        (library / 'not-a-skill').mkdir()
        Skills(directories=[library])

    async def test_nested_skill_md_is_not_a_skill(self, tmp_path: Path) -> None:
        library = tmp_path / 'skills'
        _write_skill(library, 'outer', files={'references/SKILL.md': 'reference'})

        initial, _, loaded = await _activate([Skills(directories=[library])], 'outer')

        assert '- outer:' in initial
        assert '# Skill: outer' in loaded
        assert '- references:' not in initial

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
