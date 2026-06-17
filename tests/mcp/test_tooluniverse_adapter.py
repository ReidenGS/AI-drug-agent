"""ToolUniverse adapter tests (no network, no real TU load).

The real `tooluniverse` registry holds 2 000+ tools and `load_tools()` is
slow + memory-heavy. These tests inject a fake universe via the shared
`install_universe` fixture (`tests/mcp/conftest.py`) so we exercise the
envelope normalization paths offline.

After the integration audit, `_live=True` on migrated wrappers routes
through the adapter unconditionally — there is no gate setting. Tests
here cover the adapter contract; per-wrapper routing tests live in
`test_p0_live_wrappers.py` and `test_live_public_wrappers.py`.
"""

from __future__ import annotations

from typing import Any

import pytest

from app.mcp import tooluniverse_adapter
from app.mcp.tools.alphafold import alphafold_get_prediction
from app.mcp.tools.developability_compounds import DrugProps_calculate_qed
from app.mcp.tools.evidence import EuropePMC_search_articles


# ── adapter envelope normalization ─────────────────────────────────────────

def test_adapter_success_payload(install_universe):
    install_universe(tools={"EuropePMC_search_articles": lambda args: {"results": [{"id": "1"}]}})
    out = tooluniverse_adapter.call_tool("EuropePMC_search_articles", {"query": "x"})
    assert out["status"] == "ok"
    assert out["executor"] == "tooluniverse"
    assert out["arguments"] == {"query": "x"}
    assert out["payload"] == {"results": [{"id": "1"}]}


def test_adapter_empty_list_payload(install_universe):
    install_universe(tools={"EuropePMC_search_articles": lambda args: {"results": []}})
    out = tooluniverse_adapter.call_tool("EuropePMC_search_articles", {"query": "x"})
    assert out["status"] == "empty"


def test_adapter_upstream_error_envelope(install_universe):
    install_universe(
        tools={
            "EuropePMC_search_articles": lambda args: {
                "status": "error",
                "error": "ToolValidationError: missing query",
                "error_details": {"type": "ToolValidationError"},
            }
        }
    )
    out = tooluniverse_adapter.call_tool("EuropePMC_search_articles", {})
    assert out["status"] == "upstream_error"
    assert "missing query" in out["error_message"]
    assert out["error_details"]["type"] == "ToolValidationError"


def test_adapter_unknown_tool(install_universe):
    install_universe(tools={})
    out = tooluniverse_adapter.call_tool("NotARealTool", {})
    assert out["status"] == "upstream_error"
    assert "not found" in out["error_message"].lower()


def test_adapter_exception_is_caught(install_universe, monkeypatch):
    fake = install_universe(tools={})

    def _boom(*args, **kwargs):
        raise RuntimeError("kaboom")

    monkeypatch.setattr(fake, "run_one_function", _boom)
    out = tooluniverse_adapter.call_tool("X", {})
    assert out["status"] == "upstream_error"
    assert "kaboom" in out["error_message"]


def test_adapter_unavailable_when_tu_missing(monkeypatch):
    """If tooluniverse is not installed at import time, adapter surfaces a
    clean upstream_error envelope rather than raising."""
    tooluniverse_adapter._reset_for_tests()
    real_import = __import__

    def _fake_import(name, *args, **kwargs):
        if name == "tooluniverse":
            raise ImportError("simulated: tooluniverse not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", _fake_import)
    out = tooluniverse_adapter.call_tool("Anything", {})
    assert out["status"] == "upstream_error"
    assert "tooluniverse" in out["error_message"].lower()


# ── migrated wrappers route through TU unconditionally on _live=True ───────

def test_europepmc_mock_unchanged():
    out = EuropePMC_search_articles("any query")
    assert out["status"] == "mocked"


def test_europepmc_live_always_uses_adapter(install_universe):
    """No gate setting — `_live=True` always routes through TU."""
    fake = install_universe(
        tools={"EuropePMC_search_articles": lambda args: {"results": [{"id": "x"}]}}
    )
    out = EuropePMC_search_articles("HER2 ADC", _live=True)
    assert out["executor"] == "tooluniverse"
    assert fake.calls[0]["arguments"]["query"] == "HER2 ADC"


def test_alphafold_live_routes_through_tu(install_universe):
    fake = install_universe(
        tools={"alphafold_get_prediction": lambda args: {"predictions": [{"qualifier": args["qualifier"]}]}}
    )
    out = alphafold_get_prediction("P00533", _live=True)
    assert out["executor"] == "tooluniverse"
    assert fake.calls[0]["arguments"]["qualifier"] == "P00533"


# ── DrugProps_calculate_qed: was _ni, now wired ────────────────────────────

def test_drugprops_qed_mock_unchanged():
    out = DrugProps_calculate_qed("CC(=O)Oc1ccccc1C(=O)O")
    assert out["status"] == "mocked"
    assert out["qed"] is None


def test_drugprops_qed_requires_smiles():
    with pytest.raises(ValueError):
        DrugProps_calculate_qed("")


def test_drugprops_qed_live_routes_through_tu(install_universe):
    fake = install_universe(
        tools={"DrugProps_calculate_qed": lambda args: {"qed": 0.55, "smiles": args["smiles"]}}
    )
    out = DrugProps_calculate_qed("CC(=O)Oc1ccccc1C(=O)O", _live=True)
    assert out["executor"] == "tooluniverse"
    assert out["status"] == "ok"
    assert out["payload"]["qed"] == 0.55


def test_drugprops_qed_live_surfaces_rdkit_missing_as_upstream_error(install_universe):
    install_universe(
        tools={
            "DrugProps_calculate_qed": lambda args: {
                "status": "error",
                "error": "RDKit is required for drug property calculations.",
            }
        }
    )
    out = DrugProps_calculate_qed("CCO", _live=True)
    assert out["status"] == "upstream_error"
    assert "rdkit" in out["error_message"].lower()


# ── env hydration ──────────────────────────────────────────────────────────

import os  # noqa: E402


def test_hydrate_env_injects_when_settings_present_and_env_missing(monkeypatch):
    """Settings object holds a key; os.environ doesn't → hydrate injects it.

    We monkeypatch the settings attribute directly (not via env reload)
    so the local `.env`'s real key cannot leak into the assertion diff.
    """
    from app.mcp.tooluniverse_adapter import _hydrate_env_from_settings
    from app.settings import get_settings

    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    get_settings.cache_clear()
    settings = get_settings()
    monkeypatch.setattr(settings, "gemini_api_key", "sentinel-key-A")
    _hydrate_env_from_settings()
    assert os.environ.get("GEMINI_API_KEY") == "sentinel-key-A"


def test_hydrate_env_does_not_overwrite_existing_env(monkeypatch):
    from app.mcp.tooluniverse_adapter import _hydrate_env_from_settings
    from app.settings import get_settings

    monkeypatch.setenv("GEMINI_API_KEY", "operator-set")
    get_settings.cache_clear()
    settings = get_settings()
    monkeypatch.setattr(settings, "gemini_api_key", "from-dotenv-different")
    _hydrate_env_from_settings()
    assert os.environ["GEMINI_API_KEY"] == "operator-set"


def test_hydrate_env_does_nothing_when_settings_empty(monkeypatch):
    from app.mcp.tooluniverse_adapter import _hydrate_env_from_settings
    from app.settings import get_settings

    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_MODEL_ID", raising=False)
    monkeypatch.delenv("TOOLUNIVERSE_LLM_MODEL_DEFAULT", raising=False)
    get_settings.cache_clear()
    settings = get_settings()
    monkeypatch.setattr(settings, "gemini_api_key", "")
    monkeypatch.setattr(settings, "gemini_model", "")
    _hydrate_env_from_settings()
    assert "GEMINI_API_KEY" not in os.environ
    assert "GEMINI_MODEL_ID" not in os.environ
    assert "TOOLUNIVERSE_LLM_MODEL_DEFAULT" not in os.environ


def test_hydrate_env_never_logs_the_key(monkeypatch, capsys):
    from app.mcp.tooluniverse_adapter import _hydrate_env_from_settings
    from app.settings import get_settings

    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    get_settings.cache_clear()
    settings = get_settings()
    monkeypatch.setattr(settings, "gemini_api_key", "super-secret-do-not-print")
    _hydrate_env_from_settings()
    captured = capsys.readouterr()
    assert "super-secret-do-not-print" not in captured.out
    assert "super-secret-do-not-print" not in captured.err


def test_hydrate_env_bridges_model_to_tu_compatible_names(monkeypatch):
    from app.mcp.tooluniverse_adapter import _hydrate_env_from_settings
    from app.settings import get_settings

    monkeypatch.delenv("GEMINI_MODEL_ID", raising=False)
    monkeypatch.delenv("TOOLUNIVERSE_LLM_MODEL_DEFAULT", raising=False)
    get_settings.cache_clear()
    settings = get_settings()
    monkeypatch.setattr(settings, "gemini_api_key", "x")
    monkeypatch.setattr(settings, "gemini_model", "gemini-3.5-flash")
    _hydrate_env_from_settings()
    assert os.environ.get("GEMINI_MODEL_ID") == "gemini-3.5-flash"
    assert os.environ.get("TOOLUNIVERSE_LLM_MODEL_DEFAULT") == "gemini-3.5-flash"
