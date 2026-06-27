"""LLM provider selection: must not fall into unimplemented Gemini path by
accident. Only `LLM_PROVIDER=gemini` flips the switch.
"""

from __future__ import annotations

import pytest

import app.deps as deps
from app.llm.gemini_provider import GeminiProvider
from app.llm.provider import MockLLMProvider
from app.llm.qwen_provider import QwenProvider
from app.settings import Settings, get_settings


def _clear_caches() -> None:
    for fn in (
        get_settings,
        deps.get_settings,
        deps.get_storage,
        deps.get_registry_service,
        deps.get_workflow_state_service,
        deps.get_tool_inventory_service,
        deps.get_mcp_client,
        deps.get_llm_provider,
    ):
        cache_clear = getattr(fn, "cache_clear", None)
        if cache_clear:
            cache_clear()


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    _clear_caches()
    monkeypatch.setattr(deps, "get_settings", lambda: Settings(_env_file=None))
    yield
    _clear_caches()


def test_default_provider_is_mock(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    assert isinstance(deps.get_llm_provider(), MockLLMProvider)


def test_provider_selection_tests_ignore_project_dotenv(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    settings = deps.get_settings()
    assert settings.llm_provider == "openai"
    assert settings.openai_api_key == ""


def test_gemini_api_key_alone_does_not_select_gemini(monkeypatch):
    """Even with a key present, the default must remain Mock — Gemini's
    generate* still raises NotImplementedError and we refuse to walk into it
    silently."""
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    monkeypatch.setenv("GEMINI_API_KEY", "looks-like-a-real-key-but-isnt")
    assert isinstance(deps.get_llm_provider(), MockLLMProvider)


def test_qwen_api_key_alone_does_not_select_qwen(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    monkeypatch.setenv("QWEN_API_KEY", "qwen-fake-key")
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


# ── OpenAI provider selection ───────────────────────────────────────────────


from app.llm.openai_provider import OpenAIProvider  # noqa: E402


@pytest.mark.parametrize(
    "raw_value",
    ["openai", "OpenAI", "OPENAI", "  openai  ", "OpEnAi"],
    ids=lambda v: repr(v),
)
def test_openai_case_variants_all_select_openai(monkeypatch, raw_value):
    monkeypatch.setenv("LLM_PROVIDER", raw_value)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake-key")
    assert isinstance(deps.get_llm_provider(), OpenAIProvider)


def test_explicit_openai_requires_key(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError) as excinfo:
        deps.get_llm_provider()
    msg = str(excinfo.value)
    assert "OPENAI_API_KEY" in msg
    # No key material leakage in the error.
    assert "sk-" not in msg


def test_openai_api_key_alone_does_not_select_openai(monkeypatch):
    """Mirrors the Gemini guard: a key in env must NOT silently flip the
    provider; only explicit ``LLM_PROVIDER=openai`` does."""
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake-key")
    assert isinstance(deps.get_llm_provider(), MockLLMProvider)


def test_openai_model_env_value_is_passed_to_provider(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake-key")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4.1-nano")
    provider = deps.get_llm_provider()
    assert isinstance(provider, OpenAIProvider)
    assert provider.model == "gpt-4.1-nano"


def test_default_openai_model_pinned():
    from app.settings import Settings

    assert Settings.model_fields["openai_model"].default == "gpt-4.1-mini"


# ── Qwen provider selection ────────────────────────────────────────────────


@pytest.mark.parametrize(
    "raw_value",
    ["qwen", "Qwen", "QWEN", "  qwen  ", "QwEn"],
    ids=lambda v: repr(v),
)
def test_qwen_case_variants_all_select_qwen(monkeypatch, raw_value):
    monkeypatch.setenv("LLM_PROVIDER", raw_value)
    monkeypatch.setenv("QWEN_API_KEY", "qwen-fake-key")
    assert isinstance(deps.get_llm_provider(), QwenProvider)


def test_explicit_qwen_requires_key(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "qwen")
    monkeypatch.delenv("QWEN_API_KEY", raising=False)
    with pytest.raises(ValueError) as excinfo:
        deps.get_llm_provider()
    msg = str(excinfo.value)
    assert "QWEN_API_KEY" in msg
    assert "qwen-fake-key" not in msg


def test_qwen_model_env_value_is_passed_to_provider(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "qwen")
    monkeypatch.setenv("QWEN_API_KEY", "qwen-fake-key")
    monkeypatch.setenv("QWEN_MODEL", "qwen-max")
    provider = deps.get_llm_provider()
    assert isinstance(provider, QwenProvider)
    assert provider.model == "qwen-max"


def test_qwen_base_url_env_value_is_passed_to_provider(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "qwen")
    monkeypatch.setenv("QWEN_API_KEY", "qwen-fake-key")
    monkeypatch.setenv("QWEN_BASE_URL", "https://example.test/compatible/v1")
    provider = deps.get_llm_provider()
    assert isinstance(provider, QwenProvider)
    assert provider.base_url == "https://example.test/compatible/v1"


def test_default_qwen_model_pinned():
    from app.settings import Settings

    assert Settings.model_fields["qwen_model"].default == "qwen-plus"
