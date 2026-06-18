"""Test fixtures for the agents package.

Keeps the tool-selection-policy tests offline by default: each test gets
a fresh `tooluniverse_adapter` cache and, unless a test explicitly
installs a fake universe, the adapter's metadata helpers behave as if
ToolUniverse is unavailable (returning `None` / `{}`). That way the
selector exercises the CAPABILITY_REGISTRY fallback path here without
loading the real 2 000+ tool TU registry.

Tests that want to exercise the official-spec path explicitly install a
`FakeUniverse` via the `install_universe` fixture in `tests/mcp/conftest.py`,
or patch `tooluniverse_adapter._get_universe` directly.
"""

from __future__ import annotations

import pytest

from app.mcp import tooluniverse_adapter


@pytest.fixture(autouse=True)
def _isolate_tooluniverse_adapter(monkeypatch: pytest.MonkeyPatch) -> None:
    tooluniverse_adapter._reset_for_tests()
    monkeypatch.setattr(
        tooluniverse_adapter,
        "_get_universe",
        lambda: (_ for _ in ()).throw(
            tooluniverse_adapter.ToolUniverseAdapterError(
                "offline test default — install a FakeUniverse to opt in"
            )
        ),
    )
    yield
    tooluniverse_adapter._reset_for_tests()
