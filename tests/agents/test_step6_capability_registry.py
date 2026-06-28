from pathlib import Path

from app.agents.step_06_capability_registry import (
    STEP_06_CAPABILITY_REGISTRY,
    eligible_capabilities,
)
from app.mcp.client import LocalMCPClient
from app.services.tool_inventory_service import ToolInventoryService


PROJECT_ROOT = Path(__file__).resolve().parents[2].parent
INVENTORY = PROJECT_ROOT / "项目文件" / "ToolUniversity_inventory_v0.2.xlsx"


def test_step6_registry_covers_each_inventory_tool_exactly_once():
    inventory_names = {
        entry.tool_name for entry in ToolInventoryService(str(INVENTORY)).scope_for(step_id="6")
    }
    registry_names = [entry.tool_name for entry in STEP_06_CAPABILITY_REGISTRY]
    assert len(registry_names) == len(set(registry_names)) == 53
    assert set(registry_names) == inventory_names


def test_step6_registry_tools_exist_in_production_mcp_scope():
    mcp = LocalMCPClient(inventory=ToolInventoryService(str(INVENTORY)))
    scoped = set(mcp.list_tools(agent_name="developability_agent", step_id="step_06"))
    assert len(scoped) == 53
    assert {entry.tool_name for entry in STEP_06_CAPABILITY_REGISTRY} <= scoped


def test_future_and_dependency_unavailable_tools_do_not_enter_live_catalog():
    scoped = {entry.tool_name for entry in STEP_06_CAPABILITY_REGISTRY}
    eligible, excluded = eligible_capabilities(
        "payload_linker_compound_liability",
        signals={"smiles": True},
        scoped_tools=scoped,
    )
    eligible_names = {entry.tool_name for entry in eligible}
    # ADMETAI tools are now live-wired through ToolUniverseAdapter and
    # belong in the eligible set alongside DrugProps / SwissADME.
    assert eligible_names == {
        "DrugProps_pains_filter",
        "DrugProps_lipinski_filter",
        "DrugProps_calculate_qed",
        "SwissADME_calculate_adme",
        "SwissADME_check_druglikeness",
        "ADMETAI_predict_toxicity",
        "ADMETAI_predict_physicochemical_properties",
        "ADMETAI_predict_solubility_lipophilicity_hydration",
        "ADMETAI_predict_CYP_interactions",
        "ADMETAI_predict_bioavailability",
        "ADMETAI_predict_clearance_distribution",
        "ADMETAI_predict_stress_response",
        "ADMETAI_predict_nuclear_receptor_activity",
    }
    excluded_by_name = {entry["tool_name"]: entry["reason"] for entry in excluded}
    # ADMETAI must NOT appear in the excluded-by-dependency bucket any
    # more; the only dependency_unavailable smiles-lane tool today is
    # ProteinsPlus (lane=structure) which is not in this lane's catalog.
    assert "ADMETAI_predict_toxicity" not in excluded_by_name
    assert not any(name.startswith(("DNA_", "RNAcentral_", "Rfam_", "miRBase_")) for name in eligible_names)
