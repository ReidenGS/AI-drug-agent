"""Live-mode wrappers — TU-routed (no manual httpx).

After the integration audit, EuropePMC / PubChem patents / FDA Orange
Book have no parallel manual httpx implementation. `_live=True`
unconditionally routes through `ToolUniverseAdapter`. These tests:

- verify `_live=False` mock behavior is unchanged;
- verify `_live=True` invokes the adapter with the right arguments;
- verify TU upstream errors surface as `status=upstream_error`;
- exercise the `LocalMCPClient` policy that decides whether to inject
  `_live=True` at all (`MCP_LIVE_TOOLS` + allowlist).

Smoke-script helper tests for sample selection / acceptance rules live
here too because the smoke script imports those wrappers directly.
"""

from __future__ import annotations

from typing import Any

import pytest

from app.mcp.client import LocalMCPClient
from app.mcp.tools.evidence import EuropePMC_search_articles
from app.mcp.tools.patent import (
    FDA_OrangeBook_get_patent_info,
    PubChem_get_associated_patents_by_CID,
)
from app.settings import get_settings


# ── EuropePMC ──────────────────────────────────────────────────────────────

def test_europepmc_mock_unchanged():
    out = EuropePMC_search_articles("HER2 ADC")
    assert out == {
        "status": "mocked",
        "source": "EuropePMC_search_articles",
        "query": "HER2 ADC",
        "results": [],
    }


def test_europepmc_live_routes_through_tu(install_universe):
    fake = install_universe(
        tools={
            "EuropePMC_search_articles": lambda args: {
                "results": [{"pmid": "12345", "title": "A paper"}],
                "hitCount": 1,
            }
        }
    )
    out = EuropePMC_search_articles("HER2 ADC", _live=True)
    assert out["executor"] == "tooluniverse"
    assert out["status"] == "ok"
    assert fake.calls[0]["name"] == "EuropePMC_search_articles"
    # Unspecified options are omitted so ToolUniverse applies official defaults.
    assert fake.calls[0]["arguments"] == {"query": "HER2 ADC"}


def test_europepmc_live_empty_results(install_universe):
    install_universe(tools={"EuropePMC_search_articles": lambda args: {"results": []}})
    out = EuropePMC_search_articles("nothing", _live=True)
    assert out["status"] == "empty"


def test_europepmc_live_upstream_error(install_universe):
    install_universe(
        tools={
            "EuropePMC_search_articles": lambda args: {
                "status": "error",
                "error": "timeout",
            }
        }
    )
    out = EuropePMC_search_articles("HER2", _live=True)
    assert out["status"] == "upstream_error"
    assert "timeout" in out["error_message"]


# ── PubChem patents by CID ─────────────────────────────────────────────────

def test_pubchem_patents_mock_unchanged():
    out = PubChem_get_associated_patents_by_CID("2244")
    assert out["status"] == "mocked"
    assert out["cid"] == "2244"
    assert out["patents"] == []


def test_pubchem_patents_live_routes_through_tu(install_universe):
    fake = install_universe(
        tools={
            "PubChem_get_associated_patents_by_CID": lambda args: {
                "patents": ["US-1234-A1", "US-5678-B2"]
            }
        }
    )
    out = PubChem_get_associated_patents_by_CID("5957", _live=True)
    assert out["executor"] == "tooluniverse"
    assert out["status"] == "ok"
    assert fake.calls[0]["arguments"] == {"cid": "5957"}


def test_pubchem_patents_live_empty(install_universe):
    install_universe(tools={"PubChem_get_associated_patents_by_CID": lambda args: {"patents": []}})
    out = PubChem_get_associated_patents_by_CID("5957", _live=True)
    assert out["status"] == "empty"


def test_pubchem_patents_live_upstream_error(install_universe):
    install_universe(
        tools={
            "PubChem_get_associated_patents_by_CID": lambda args: {
                "status": "error",
                "error": "read timeout",
            }
        }
    )
    out = PubChem_get_associated_patents_by_CID("2244", _live=True)
    assert out["status"] == "upstream_error"


# ── FDA Orange Book ────────────────────────────────────────────────────────

def test_orange_book_mock_unchanged():
    out = FDA_OrangeBook_get_patent_info(brand_name="LIPITOR")
    assert out["status"] == "mocked"
    assert out["records"] == []


def test_orange_book_requires_one_arg():
    with pytest.raises(ValueError):
        FDA_OrangeBook_get_patent_info()


def test_orange_book_live_routes_through_tu_by_brand_name(install_universe):
    fake = install_universe(
        tools={
            "FDA_OrangeBook_get_patent_info": lambda args: {
                "records": [{"application_number": "021436", "trade_name": "LIPITOR"}],
            }
        }
    )
    out = FDA_OrangeBook_get_patent_info(brand_name="LIPITOR", _live=True)
    assert out["executor"] == "tooluniverse"
    assert out["status"] == "ok"
    assert fake.calls[0]["arguments"] == {"brand_name": "LIPITOR"}


def test_orange_book_live_routes_through_tu_by_application_number(install_universe):
    fake = install_universe(
        tools={"FDA_OrangeBook_get_patent_info": lambda args: {"records": []}}
    )
    FDA_OrangeBook_get_patent_info(application_number="NDA021436", _live=True)
    assert fake.calls[0]["arguments"] == {"application_number": "NDA021436"}


def test_orange_book_live_upstream_error(install_universe):
    install_universe(
        tools={
            "FDA_OrangeBook_get_patent_info": lambda args: {
                "status": "error",
                "error": "zip unavailable",
            }
        }
    )
    out = FDA_OrangeBook_get_patent_info(brand_name="LIPITOR", _live=True)
    assert out["status"] == "upstream_error"


# ── LocalMCPClient live-mode policy injection ──────────────────────────────

def test_local_client_does_not_inject_live_by_default(monkeypatch):
    monkeypatch.setenv("MCP_LIVE_TOOLS", "false")
    monkeypatch.delenv("MCP_LIVE_TOOL_ALLOWLIST", raising=False)
    get_settings.cache_clear()

    captured: dict[str, Any] = {}

    def _spy(**kwargs: Any) -> dict:
        captured.update(kwargs)
        return {"status": "mocked"}

    client = LocalMCPClient(bindings={"EuropePMC_search_articles": _spy})
    out = client.call_tool(
        agent_name="evidence_agent",
        step_id="step_13",
        tool_name="EuropePMC_search_articles",
        query="HER2",
    )
    assert out["run_status"] == "failed"
    assert out["executor"] == "mock"
    assert "_live" not in captured


def test_local_client_injects_live_when_on_allowlist(monkeypatch):
    monkeypatch.setenv("MCP_LIVE_TOOLS", "true")
    monkeypatch.setenv(
        "MCP_LIVE_TOOL_ALLOWLIST",
        "EuropePMC_search_articles,PubChem_get_associated_patents_by_CID",
    )
    get_settings.cache_clear()

    captured: dict[str, Any] = {}

    def _spy(**kwargs: Any) -> dict:
        captured.update(kwargs)
        return {"status": "mocked"}

    client = LocalMCPClient(bindings={"EuropePMC_search_articles": _spy})
    client.call_tool(
        agent_name="evidence_agent",
        step_id="step_13",
        tool_name="EuropePMC_search_articles",
        query="HER2",
    )
    assert captured.get("_live") is True


def test_local_client_does_not_inject_live_when_not_on_allowlist(monkeypatch):
    monkeypatch.setenv("MCP_LIVE_TOOLS", "true")
    monkeypatch.setenv("MCP_LIVE_TOOL_ALLOWLIST", "PubChem_get_associated_patents_by_CID")
    get_settings.cache_clear()

    captured: dict[str, Any] = {}

    def _spy(**kwargs: Any) -> dict:
        captured.update(kwargs)
        return {"status": "mocked"}

    client = LocalMCPClient(bindings={"EuropePMC_search_articles": _spy})
    client.call_tool(
        agent_name="evidence_agent",
        step_id="step_13",
        tool_name="EuropePMC_search_articles",
        query="HER2",
    )
    assert "_live" not in captured


def test_local_client_injects_live_for_all_tools_with_empty_allowlist(monkeypatch):
    """Production all-live: live ON + empty allowlist injects `_live=True`
    for an arbitrary scoped tool, not just allowlisted ones."""
    monkeypatch.setenv("MCP_LIVE_TOOLS", "true")
    monkeypatch.delenv("MCP_LIVE_TOOL_ALLOWLIST", raising=False)
    get_settings.cache_clear()

    captured: dict[str, Any] = {}

    def _spy(**kwargs: Any) -> dict:
        captured.update(kwargs)
        return {"status": "mocked"}

    client = LocalMCPClient(bindings={"EuropePMC_search_articles": _spy})
    client.call_tool(
        agent_name="evidence_agent",
        step_id="step_13",
        tool_name="EuropePMC_search_articles",
        query="HER2",
    )
    assert captured.get("_live") is True


def test_local_client_caller_live_kwarg_takes_precedence(monkeypatch):
    monkeypatch.setenv("MCP_LIVE_TOOLS", "true")
    monkeypatch.setenv("MCP_LIVE_TOOL_ALLOWLIST", "EuropePMC_search_articles")
    get_settings.cache_clear()

    captured: dict[str, Any] = {}

    def _spy(**kwargs: Any) -> dict:
        captured.update(kwargs)
        return {"status": "mocked"}

    client = LocalMCPClient(bindings={"EuropePMC_search_articles": _spy})
    client.call_tool(
        agent_name="evidence_agent",
        step_id="step_13",
        tool_name="EuropePMC_search_articles",
        query="HER2",
        _live=False,
    )
    assert captured.get("_live") is False


# ── Live smoke script: sample + acceptance rules ───────────────────────────

def _import_smoke_module():
    import importlib.util
    import pathlib

    script = pathlib.Path(__file__).resolve().parents[2] / "scripts" / "run_live_public_wrappers_smoke.py"
    spec = importlib.util.spec_from_file_location("live_wrappers_smoke", script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_smoke_uses_stable_orange_book_brand():
    mod = _import_smoke_module()
    assert mod.FDA_SMOKE_BRAND_NAME != "HERCEPTIN"
    assert mod.FDA_SMOKE_BRAND_NAME == "LIPITOR"


def test_smoke_uses_bounded_pubchem_cid():
    mod = _import_smoke_module()
    assert mod.PUBCHEM_SMOKE_CID != "2244"
