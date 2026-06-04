"""Shared fixtures for ModalSandbox tests."""

from __future__ import annotations

import sys
from collections.abc import Iterator

import pytest

from .fake_modal import FakeModal


@pytest.fixture
def fake_modal(monkeypatch: pytest.MonkeyPatch) -> Iterator[FakeModal]:
    """Inject a fake `modal` module and yield its control surface."""
    control = FakeModal()
    monkeypatch.setitem(sys.modules, 'modal', control.module)
    yield control
