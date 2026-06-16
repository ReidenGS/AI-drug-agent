"""Inventory-scoped LocalMCPClient: out-of-step/out-of-inventory tools skipped."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from app.mcp.client import LocalMCPClient
from app.services.tool_inventory_service import ToolInventoryService


PROJECT_ROOT = Path(__file__).resolve().parents[2].parent
DEFAULT_XLSX = PROJECT_ROOT / "项目文件" / "ToolUniversity_inventory_v0.2.xlsx"


@pytest.fixture
def inventory() -> ToolInventoryService:
    xlsx = os.environ.get("TOOL_INVENTORY_XLSX", str(DEFAULT_XLSX))
    if not Path(xlsx).exists():
        pytest.skip(f"Inventory xlsx not at {xlsx}")
    return ToolInventoryService(xlsx)


def test_inventory_scoped_client_lists_only_step_5_tools(inventory):
    client = LocalMCPClient(inventory=inventory)
    tools = client.list_tools(agent_name="candidate_context_agent", step_id="step_05")
    # SAbDab / ChEMBL_search_molecules are canonical Step 5 tools in v0.2.
    assert "SAbDab_search_structures" in tools
    assert "ChEMBL_search_molecules" in tools
    # Step 6+ tools must not leak into the Step 5 scope:
    assert "ProteinsPlus_profile_structure_quality" not in tools  # step 6
    assert "alphafold_get_prediction" not in tools  # step 7 in v0.2
    assert "NvidiaNIM_alphafold2_multimer" not in tools  # step 8


def test_inventory_scoped_client_skips_out_of_step_tool(inventory):
    client = LocalMCPClient(inventory=inventory)
    res = client.call_tool(
        agent_name="candidate_context_agent",
        step_id="step_05",
        tool_name="ProteinsPlus_profile_structure_quality",  # Step 6 tool
        pdb_id="1n8z",
    )
    assert res["run_status"] == "skipped"
    assert res["skip_reason"] == "tool_not_in_agent_scope"


def test_inventory_scoped_client_skips_unknown_tool(inventory):
    client = LocalMCPClient(inventory=inventory)
    res = client.call_tool(
        agent_name="candidate_context_agent",
        step_id="step_05",
        tool_name="FakeTool_does_not_exist",
    )
    assert res["run_status"] == "skipped"
