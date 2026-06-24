"""Tests for the live-smoke MCP allowlist wrapper.

The smoke uses a small local wrapper to narrow the Step 6 LLM catalog
to the LIVE_ALLOWLIST so the LLM cannot pick a tool that would later
fail ``attempted_live=true``. The wrapper must:

- restrict ``list_tools`` to the allowlist (so Stage 1 catalog narrows);
- delegate ``call_tool`` unchanged (so no scope is widened);
- NOT touch production agents that build their own MCP client.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys

import pytest

# Import the smoke module — its top-level body sets MCP_LIVE_TOOLS etc.,
# but the import is side-effect-safe because it only sets env vars and
# defines classes/constants. We carefully clear the live env vars after
# the test so other tests don't inherit them.
_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(_SCRIPTS))


@pytest.fixture
def smoke_module(monkeypatch):
    # Save current env to restore.
    saved = {
        "MCP_LIVE_TOOLS": os.environ.get("MCP_LIVE_TOOLS"),
        "MCP_LIVE_TOOL_ALLOWLIST": os.environ.get("MCP_LIVE_TOOL_ALLOWLIST"),
    }
    import importlib
    if "run_live_llm_step1_6_pdb_smoke" in sys.modules:
        del sys.modules["run_live_llm_step1_6_pdb_smoke"]
    mod = importlib.import_module("run_live_llm_step1_6_pdb_smoke")
    yield mod
    # Restore env.
    for k, v in saved.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


# ── 1. wrapper narrows list_tools ─────────────────────────────────────────


def test_allowlist_wrapper_narrows_list_tools(smoke_module):
    class _Inner:
        def list_tools(self, *, agent_name, step_id):
            return [
                "EBIProteins_get_features", "EBIProteins_get_epitopes",
                "ProteinsPlus_profile_structure_quality",
                "ADMETAI_predict_toxicity",  # NOT in allowlist below
                "SwissADME_check_druglikeness",  # NOT in allowlist below
            ]
        def call_tool(self, *, agent_name, step_id, tool_name, **kwargs):
            return {"run_status": "success", "tool_name": tool_name, "kwargs": kwargs}

    allowlist = (
        "EBIProteins_get_features",
        "EBIProteins_get_epitopes",
        "ProteinsPlus_profile_structure_quality",
    )
    wrapper = smoke_module._AllowlistMCPClient(_Inner(), allowlist)
    out = wrapper.list_tools(agent_name="developability_agent", step_id="step_06")
    assert set(out) == set(allowlist), out
    # ADMETAI and SwissADME_check_druglikeness must be invisible to the LLM.
    assert "ADMETAI_predict_toxicity" not in out
    assert "SwissADME_check_druglikeness" not in out


def test_allowlist_wrapper_call_tool_delegates_unchanged(smoke_module):
    seen = []

    class _Inner:
        def list_tools(self, *, agent_name, step_id):
            return []
        def call_tool(self, *, agent_name, step_id, tool_name, **kwargs):
            seen.append((agent_name, step_id, tool_name, dict(kwargs)))
            return {"run_status": "success", "tool_name": tool_name}

    wrapper = smoke_module._AllowlistMCPClient(_Inner(), ("anything",))
    res = wrapper.call_tool(
        agent_name="developability_agent", step_id="step_06",
        tool_name="ChEMBL_search_activities", molecule_chembl_id="CHEMBL1",
    )
    assert res["run_status"] == "success"
    assert seen == [(
        "developability_agent", "step_06",
        "ChEMBL_search_activities",
        {"molecule_chembl_id": "CHEMBL1"},
    )]


# ── 2. production parity: DevelopabilityAgent's surface is unchanged ──────


def test_production_developability_tool_surface_unchanged_by_smoke_wrapper(smoke_module):
    """The smoke's allowlist wrapper must NOT influence production code that
    builds its own MCP client via deps.get_mcp_client().
    """
    from app.mcp.client import LocalMCPClient  # noqa: PLC0415
    from app.services.tool_inventory_service import ToolInventoryService  # noqa: PLC0415

    xlsx = Path(__file__).resolve().parents[3] / "项目文件" / "ToolUniversity_inventory_v0.2.xlsx"
    if not xlsx.exists():
        pytest.skip(f"Inventory xlsx not at {xlsx}")
    inventory = ToolInventoryService(str(xlsx))
    bare = LocalMCPClient(inventory=inventory)
    bare_tools = set(bare.list_tools(agent_name="developability_agent", step_id="step_06"))

    # Wrap a SUBSET that excludes ADMETAI.
    narrow = (
        "EBIProteins_get_features", "EBIProteins_get_epitopes",
        "ProteinsPlus_profile_structure_quality",
        "DrugProps_pains_filter", "DrugProps_lipinski_filter",
        "ChEMBL_search_activities", "SwissADME_calculate_adme",
    )
    wrapped = smoke_module._AllowlistMCPClient(bare, narrow)

    # 1. The wrapper-presented catalog is narrower.
    wrapped_tools = set(wrapped.list_tools(agent_name="developability_agent", step_id="step_06"))
    assert wrapped_tools.issubset(set(narrow))
    assert "ADMETAI_predict_toxicity" in bare_tools  # production sees it
    assert "ADMETAI_predict_toxicity" not in wrapped_tools  # smoke does not

    # 2. The underlying LocalMCPClient instance is untouched (same identity,
    #    same surface). The wrapper is a local view, NOT a global mutation.
    again = set(bare.list_tools(agent_name="developability_agent", step_id="step_06"))
    assert again == bare_tools
    assert "ADMETAI_predict_toxicity" in again


# ── 3. chembl ID counting: occurrence vs unique ──────────────────────────


def test_chembl_id_counting_distinguishes_occurrence_and_unique(smoke_module):
    payload = {
        "data": {"molecules": [
            {"molecule_chembl_id": "CHEMBL1",
             "molecule_structures": {"chembl_id": "CHEMBL1", "canonical_smiles": "CCO"}},
            {"molecule_chembl_id": "CHEMBL2"},
            {"molecule_chembl_id": "CHEMBL1"},  # duplicate
        ]},
    }
    occurrences, smiles, unique = smoke_module._extract_chembl_ids_and_smiles(payload)
    assert occurrences == 4  # CHEMBL1 twice in row 0, plus rows 1+2
    assert unique == {"CHEMBL1", "CHEMBL2"}
    assert smiles == 1


# ── 4. known_dependency_gap classification ──────────────────────────────


def test_proteins_plus_listed_as_known_dependency_gap_constant(smoke_module):
    assert "ProteinsPlus_profile_structure_quality" in smoke_module.KNOWN_LIVE_DEPENDENCY_GAPS
