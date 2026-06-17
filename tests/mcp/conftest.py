"""Shared test fixtures for the MCP test package.

Provides a fake ToolUniverse stand-in so wrapper `_live=True` tests can
exercise the adapter routing path without loading the real 2 000+ tool
registry or hitting the network.
"""

from __future__ import annotations

from typing import Any, Callable

import pytest

from app.mcp import tooluniverse_adapter
from app.settings import get_settings


class FakeUniverse:
    """Minimal stand-in for `tooluniverse.ToolUniverse`."""

    def __init__(
        self,
        *,
        tools: dict[str, Callable[[dict], Any]] | None = None,
        names: list[str] | None = None,
    ) -> None:
        self._tools = tools or {}
        self._names = names or list(self._tools.keys())
        self.calls: list[dict] = []

    def load_tools(self, **_kwargs: Any) -> None:
        return None

    def get_available_tools(self, name_only: bool = True) -> list[str]:
        return list(self._names)

    def run_one_function(
        self,
        function_call_json: dict,
        validate: bool = True,
        use_cache: bool = False,
    ) -> Any:
        self.calls.append(function_call_json)
        handler = self._tools.get(function_call_json["name"])
        if handler is None:
            return {
                "status": "error",
                "error": f"Tool '{function_call_json['name']}' not found",
                "error_details": {"type": "ToolUnavailableError"},
            }
        return handler(function_call_json.get("arguments") or {})


@pytest.fixture
def install_universe(monkeypatch: pytest.MonkeyPatch):
    """Install a fake universe for the duration of a test."""

    def _install(
        *,
        tools: dict[str, Callable[[dict], Any]] | None = None,
        names: list[str] | None = None,
    ) -> FakeUniverse:
        fake = FakeUniverse(tools=tools, names=names)
        tooluniverse_adapter._reset_for_tests()
        monkeypatch.setattr(tooluniverse_adapter, "_get_universe", lambda: fake)
        return fake

    yield _install
    tooluniverse_adapter._reset_for_tests()


@pytest.fixture(autouse=True)
def _reset_settings_cache_for_mcp_tests() -> None:
    """Drop the `get_settings` lru_cache so env mutations apply per test."""
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
