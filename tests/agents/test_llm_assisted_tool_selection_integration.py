from __future__ import annotations

import os
from pathlib import Path

import pytest

from app.graph.adc_graph import build_step1_14_graph
from app.mcp.client import LocalMCPClient
from app.services.tool_inventory_service import ToolInventoryService


PROJECT_ROOT = Path(__file__).resolve().parents[2].parent
DEFAULT_XLSX = PROJECT_ROOT / "\u9879\u76ee\u6587\u4ef6" / "ToolUniversity_inventory_v0.2.xlsx"


@pytest.fixture
def inventory():
    xlsx = os.environ.get("TOOL_INVENTORY_XLSX", str(DEFAULT_XLSX))
    if not Path(xlsx).exists():
        pytest.skip(f"Inventory xlsx not at {xlsx}")
    return ToolInventoryService(xlsx)


def test_step6_9_13_14_record_selection_metadata(
    local_storage, registry_service, workflow_state_service, inventory
):
    graph = build_step1_14_graph(
        storage=local_storage,
        registry=registry_service,
        workflow_state=workflow_state_service,
        mcp_client=LocalMCPClient(inventory=inventory),
    )
    final = graph.invoke(
        {
            "intake_request": {
                "raw_user_query": "Design ADC against HER2 using PDB 1N8Z; vc-MMAE payload",
                "user_provided_context": {
                    "target_or_antigen_text": "HER2",
                    "candidate_text": "Trastuzumab analog",
                    "payload_linker_text": "vc-MMAE",
                },
            }
        }
    )
    run_id = final["run_id"]

    step6 = local_storage.read_json(local_storage.run_key(run_id, "structured_liability_summary.json"))
    step6_calls = [
        tc
        for cand in step6["candidate_liability_results"]
        for lane in cand["lane_results"]
        for tc in lane["tool_call_records"]
    ]
    assert step6_calls
    assert any("selection_policy_version" in tc["tool_input_summary"] for tc in step6_calls)

    step9 = local_storage.read_json(local_storage.run_key(run_id, "compound_screening_artifact.json"))
    assert step9["tool_call_records"]
    assert all("selection_policy_version" in tc["tool_input_summary"] for tc in step9["tool_call_records"])
    assert "ZINC22" not in str(step9)

    step13 = local_storage.read_json(local_storage.run_key(run_id, "scientific_evidence_table.json"))
    assert step13["tool_call_records"]
    assert any("selection_policy_version" in tc["tool_input_summary"] for tc in step13["tool_call_records"])

    step14 = local_storage.read_json(local_storage.run_key(run_id, "patent_prior_art_table.json"))
    assert step14["tool_call_records"]
    assert any("selection_policy_version" in tc["tool_input_summary"] for tc in step14["tool_call_records"])
