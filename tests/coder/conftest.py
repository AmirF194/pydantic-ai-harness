import re
from datetime import datetime
from typing import Any

import pytest
from dirty_equals import IsDatetime, IsInstance, IsStr
from inline_snapshot.plugin import customize
from pydantic_ai.usage import RequestUsage


class InlineSnapshotPlugin:
    @customize(tryfirst=True)
    def nondeterministic_values(self, value: object, builder: Any) -> Any:
        if isinstance(value, datetime):
            return builder.create_call(IsDatetime)
        if isinstance(value, RequestUsage):
            return builder.create_call(IsInstance, [RequestUsage])
        if isinstance(value, str) and re.fullmatch(r'01[a-z0-9]{6}(?:-[a-z0-9]{4}){3}-[a-z0-9]{12}', value):
            return builder.create_call(IsStr)
        if isinstance(value, str) and (
            '/tmp/pytest-of-' in value
            or re.search(r'01[a-z0-9]{6}-', value)
            or re.search(r'\bin \d+(?:\.\d+)?s\b', value)
            or re.search(r'\b[0-9a-f]{8}\b', value)
        ):
            normalized = re.sub(r'pytest-\d+', 'PYTEST_RUN', value)
            normalized = re.sub(r'01[a-z0-9]{6}(?:-[a-z0-9]{4}){3}-[a-z0-9]{12}', 'RUN_ID', normalized)
            normalized = re.sub(r'\bin \d+(?:\.\d+)?s\b', 'in ELAPSED_TIME', normalized)
            normalized = re.sub(r'\b[0-9a-f]{8}\b', 'TASK_ID', normalized)
            pattern = re.escape(normalized).replace('PYTEST_RUN', r'pytest\-\d+')
            pattern = pattern.replace('RUN_ID', r'01[a-z0-9]{6}(?:\-[a-z0-9]{4}){3}\-[a-z0-9]{12}')
            pattern = pattern.replace('ELAPSED_TIME', r'\d+(?:\.\d+)?s')
            pattern = pattern.replace('TASK_ID', r'[0-9a-f]{8}')
            return builder.create_call(IsStr, [], {'regex': pattern})


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
