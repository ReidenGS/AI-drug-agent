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
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    assert isinstance(deps.get_llm_provider(), MockLLMProvider)


def test_gemini_api_key_alone_does_not_select_gemini(monkeypatch):
    """Even with a key present, the default must remain Mock — Gemini's
    generate* still raises NotImplementedError and we refuse to walk into it
    silently."""
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
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
