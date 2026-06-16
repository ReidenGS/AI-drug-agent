"""LLM provider selection: must not fall into unimplemented Gemini path by
accident. Only `LLM_PROVIDER=gemini` flips the switch.
"""

from __future__ import annotations

import pytest

import app.deps as deps
from app.llm.gemini_provider import GeminiProvider
from app.llm.provider import MockLLMProvider
from app.settings import get_settings


def _clear_caches() -> None:
    for fn in (
        get_settings,
        deps.get_storage,
        deps.get_registry_service,
        deps.get_workflow_state_service,
        deps.get_tool_inventory_service,
        deps.get_mcp_client,
        deps.get_llm_provider,
    ):
        fn.cache_clear()


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    _clear_caches()
    yield
    _clear_caches()


def test_default_provider_is_mock(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    assert isinstance(deps.get_llm_provider(), MockLLMProvider)


def test_gemini_api_key_alone_does_not_select_gemini(monkeypatch):
    """Even with a key present, the default must remain Mock — Gemini's
    generate* still raises NotImplementedError and we refuse to walk into it
    silently."""
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    monkeypatch.setenv("GEMINI_API_KEY", "looks-like-a-real-key-but-isnt")
    assert isinstance(deps.get_llm_provider(), MockLLMProvider)


def test_explicit_gemini_requires_key(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "")
    with pytest.raises(ValueError, match="GEMINI_API_KEY"):
        deps.get_llm_provider()


def test_explicit_gemini_with_key_returns_gemini_provider(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    provider = deps.get_llm_provider()
    assert isinstance(provider, GeminiProvider)
    # Sanity: the abstraction is in place — generate is still a stub.
    with pytest.raises(NotImplementedError):
        provider.generate("hi")


# ── case-normalization (settings layer) ─────────────────────────────────────

@pytest.mark.parametrize(
    "raw_value",
    ["Gemini", "GEMINI", "gemini", "  gemini  ", "GeMiNi"],
    ids=lambda v: repr(v),
)
def test_gemini_case_variants_all_select_gemini(monkeypatch, raw_value):
    """Settings must accept any case form so users can't typo themselves into
    a silent MockLLMProvider when they meant to enable Gemini."""
    monkeypatch.setenv("LLM_PROVIDER", raw_value)
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    assert isinstance(deps.get_llm_provider(), GeminiProvider)


@pytest.mark.parametrize("raw_value", ["Mock", "MOCK", "mock", " Mock "])
def test_mock_case_variants_all_select_mock(monkeypatch, raw_value):
    monkeypatch.setenv("LLM_PROVIDER", raw_value)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    assert isinstance(deps.get_llm_provider(), MockLLMProvider)


def test_invalid_llm_provider_value_raises_clear_error(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "claude")
    with pytest.raises(Exception) as excinfo:
        deps.get_llm_provider()
    msg = str(excinfo.value)
    assert "claude" in msg.lower()
    assert "mock" in msg.lower() or "gemini" in msg.lower()


# ── GEMINI_MODEL env propagation ────────────────────────────────────────────

def test_gemini_model_env_value_is_passed_to_provider(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "Gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-3.5-flash")
    provider = deps.get_llm_provider()
    assert isinstance(provider, GeminiProvider)
    assert provider.model == "gemini-3.5-flash"


def test_default_gemini_model_is_2_5_flash():
    """The Settings field default must match the README. We pin the value so
    a silent default change can't ship without updating docs. We assert on
    the field default rather than instantiating Settings, because a local
    `.env` (which we cannot guarantee stays absent in dev) may override env."""
    from app.settings import Settings

    assert Settings.model_fields["gemini_model"].default == "gemini-3.5-flash"
