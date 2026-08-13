from typing import Any

import pytest


@pytest.fixture(scope='module')
def vcr_config() -> dict[str, Any]:
    return {
        'filter_headers': [
            ('authorization', 'REDACTED'),
            ('x-api-key', 'REDACTED'),
        ],
        'match_on': ['method', 'uri'],
    }


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'
