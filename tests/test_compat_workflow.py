from pathlib import Path


def test_compat_commands_do_not_resync_editable_overrides() -> None:
    workflow = (Path(__file__).parents[1] / '.github' / 'workflows' / 'compat-test.yml').read_text()

    commands = (
        'ruff format --check',
        'ruff check',
        'pyright',
        'pytest',
    )
    for command in commands:
        assert f'uv run --no-sync {command}' in workflow
