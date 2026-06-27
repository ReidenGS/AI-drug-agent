"""Production-parity live-smoke configuration tests."""

from __future__ import annotations

import os
from pathlib import Path
import sys

import pytest

_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(_SCRIPTS))


@pytest.fixture
def smoke_module():
    saved = {
        "MCP_LIVE_TOOLS": os.environ.get("MCP_LIVE_TOOLS"),
        "MCP_LIVE_TOOL_ALLOWLIST": os.environ.get("MCP_LIVE_TOOL_ALLOWLIST"),
    }
    import importlib
    if "run_live_llm_step1_6_pdb_smoke" in sys.modules:
        del sys.modules["run_live_llm_step1_6_pdb_smoke"]
    mod = importlib.import_module("run_live_llm_step1_6_pdb_smoke")
    yield mod
    for key, value in saved.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def test_smoke_has_no_catalog_narrowing_wrapper(smoke_module):
    assert not hasattr(smoke_module, "_AllowlistMCPClient")


def test_smoke_uses_full_production_developability_scope(smoke_module):
    from app.mcp.client import LocalMCPClient
    from app.services.tool_inventory_service import ToolInventoryService

    xlsx = Path(__file__).resolve().parents[3] / "项目文件" / "ToolUniversity_inventory_v0.2.xlsx"
    if not xlsx.exists():
        pytest.skip(f"Inventory xlsx not at {xlsx}")
    mcp = LocalMCPClient(inventory=ToolInventoryService(str(xlsx)))
    tools = set(mcp.list_tools(agent_name="developability_agent", step_id="step_06"))
    assert len(tools) == 53
    assert "ADMETAI_predict_toxicity" in tools
    assert "SwissADME_check_druglikeness" in tools
    assert "SwissADME_check_druglikeness" in smoke_module.LIVE_ALLOWLIST


def test_live_allowlist_does_not_narrow_step6_catalog(smoke_module):
    from app.mcp.client import LocalMCPClient
    from app.services.tool_inventory_service import ToolInventoryService

    xlsx = Path(__file__).resolve().parents[3] / "项目文件" / "ToolUniversity_inventory_v0.2.xlsx"
    if not xlsx.exists():
        pytest.skip(f"Inventory xlsx not at {xlsx}")
    mcp = LocalMCPClient(inventory=ToolInventoryService(str(xlsx)))
    catalog_names = set(mcp.list_tools(agent_name="developability_agent", step_id="step_06"))
    assert len(catalog_names) == 53
    step6_live_allowlist = set(smoke_module.LIVE_ALLOWLIST) & catalog_names
    assert step6_live_allowlist < catalog_names
    assert "SwissADME_check_druglikeness" in step6_live_allowlist
    assert {
        "ADMETAI_predict_toxicity",
        "ProteinsPlus_profile_structure_quality",
    } <= catalog_names
    # ADMETAI was migrated to a live wrapper; it now belongs to the
    # live allowlist and is NOT a known dependency gap. ProteinsPlus
    # remains deferred.
    assert "ADMETAI_predict_toxicity" in smoke_module.LIVE_ALLOWLIST
    assert "ProteinsPlus_profile_structure_quality" in smoke_module.KNOWN_LIVE_DEPENDENCY_GAPS
    assert "ADMETAI_predict_toxicity" not in smoke_module.KNOWN_LIVE_DEPENDENCY_GAPS


def test_chembl_id_counting_distinguishes_occurrence_and_unique(smoke_module):
    payload = {
        "data": {"molecules": [
            {"molecule_chembl_id": "CHEMBL1",
             "molecule_structures": {"chembl_id": "CHEMBL1", "canonical_smiles": "CCO"}},
            {"molecule_chembl_id": "CHEMBL2"},
            {"molecule_chembl_id": "CHEMBL1"},
        ]},
    }
    occurrences, smiles, unique = smoke_module._extract_chembl_ids_and_smiles(payload)
    assert occurrences == 4
    assert unique == {"CHEMBL1", "CHEMBL2"}
    assert smiles == 1


def test_known_dependency_gap_constants_are_explicit(smoke_module):
    """ProteinsPlus_profile_structure_quality is the only Step 6 tool
    still classified as a known live dependency gap after the ADMETAI
    live-wiring migration."""
    assert "ProteinsPlus_profile_structure_quality" in smoke_module.KNOWN_LIVE_DEPENDENCY_GAPS
    assert "ADMETAI_predict_toxicity" not in smoke_module.KNOWN_LIVE_DEPENDENCY_GAPS
