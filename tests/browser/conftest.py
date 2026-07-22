"""Shared collection rules for the Browser capability tests."""

from __future__ import annotations

import importlib.util

# `playwright` is gated on the `browser` extra, so slim CI runs (no extras) can't
# import these modules. Ignore them at collection. A conditional expression rather
# than an `if` statement: branch coverage traces statement arcs, and no single
# environment can take both arms of an install-dependent branch.
collect_ignore = ['test_browser.py'] if importlib.util.find_spec('playwright') is None else []
