"""End-to-end Step 1→14 graph happy path against LocalStorage."""

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


def test_step1_14_pipeline_completes_all_fourteen_steps(
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
                "raw_user_query": "HER2 ADC vc-MMAE",
                "user_provided_context": {
                    "target_or_antigen_text": "HER2",
                    "candidate_text": "Trastuzumab analog",
                    "payload_linker_text": "vc-MMAE",
                },
            }
        }
    )
    run_id = final["run_id"]
    artifacts = final["artifacts"]
    for key in (
        "raw_request_record", "structured_query", "input_readiness_status",
        "run_step_plan", "candidate_context_table", "structured_liability_summary",
        "prepared_structure_input_package",
        "structure_prediction_and_interface_results",
        "structure_variant_and_compound_screening",
        "scoring_handoff_package", "scoring_validation", "ranking_table",
        "scientific_evidence_table", "patent_prior_art_table",
    ):
        assert artifacts.get(key), f"missing {key}"

    state = workflow_state_service.get(run_id)
    for s in (f"step_{i:02d}" for i in range(1, 15)):
        assert state["steps"][s] == "completed", state["steps"][s]

    # Step 11/12 honest "awaiting" because there's no external scoring file.
    assert final["results"]["step_11"]["validation_status"] == "awaiting_external_input"
    assert final["results"]["step_12"]["ranking_status"] == "awaiting_external_scoring"
    # Step 13/14 actually ran.
    assert final["results"]["step_13"]["executed"] is True
    assert final["results"]["step_14"]["executed"] is True
