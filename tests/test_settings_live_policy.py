"""MCP live-mode policy (`Settings.should_use_live`).

Policy under test:
- live OFF -> never live (the default, so unit tests never hit the network).
- live ON + non-empty allowlist -> only listed tools (constrained smoke/debug).
- live ON + empty allowlist -> every scoped tool goes live (production all-live).
"""

from __future__ import annotations

import pytest

from app.settings import Settings, get_settings


def _settings(**overrides) -> Settings:
    base = {"mcp_live_tools": False, "mcp_live_tool_allowlist": ""}
    base.update(overrides)
    return Settings(**base)


def test_live_off_never_live():
    s = _settings(mcp_live_tools=False, mcp_live_tool_allowlist="")
    assert s.should_use_live("ADMETAI_predict_toxicity") is False
    # Even a populated allowlist stays off while the master flag is off.
    s2 = _settings(mcp_live_tools=False, mcp_live_tool_allowlist="ADMETAI_predict_toxicity")
    assert s2.should_use_live("ADMETAI_predict_toxicity") is False


def test_live_on_empty_allowlist_is_all_live():
    s = _settings(mcp_live_tools=True, mcp_live_tool_allowlist="")
    assert s.should_use_live("ADMETAI_predict_toxicity") is True
    assert s.should_use_live("any_arbitrary_tool_name") is True
    assert s.should_use_live("PROSITE_scan_sequence") is True


def test_live_on_nonempty_allowlist_limits_to_listed():
    s = _settings(
        mcp_live_tools=True,
        mcp_live_tool_allowlist="SwissADME_check_druglikeness, PROSITE_scan_sequence",
    )
    assert s.should_use_live("SwissADME_check_druglikeness") is True
    assert s.should_use_live("PROSITE_scan_sequence") is True
    # Whitespace around entries is tolerated.
    assert s.should_use_live("ADMETAI_predict_toxicity") is False
    assert s.should_use_live("not_listed_tool") is False


def test_default_settings_keep_live_off():
    """Default construction must keep the network off for unit tests."""
    s = _settings()
    assert s.mcp_live_tools is False
    assert s.should_use_live("ADMETAI_predict_toxicity") is False


def test_env_empty_allowlist_all_live(monkeypatch):
    monkeypatch.setenv("MCP_LIVE_TOOLS", "true")
    monkeypatch.delenv("MCP_LIVE_TOOL_ALLOWLIST", raising=False)
    get_settings.cache_clear()
    try:
        assert get_settings().should_use_live("ChEMBL_search_molecules") is True
    finally:
        get_settings.cache_clear()
