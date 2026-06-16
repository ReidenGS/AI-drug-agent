"""End-to-end LangGraph Step 1→12 against LocalStorage + inventory-scoped MCP."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from app.graph.adc_graph import build_step1_12_graph
from app.mcp.client import LocalMCPClient
from app.services.tool_inventory_service import ToolInventoryService


PROJECT_ROOT = Path(__file__).resolve().parents[2].parent
DEFAULT_XLSX = PROJECT_ROOT / "项目文件" / "ToolUniversity_inventory_v0.2.xlsx"


@pytest.fixture
def inventory():
    xlsx = os.environ.get("TOOL_INVENTORY_XLSX", str(DEFAULT_XLSX))
    if not Path(xlsx).exists():
        pytest.skip(f"Inventory xlsx not at {xlsx}")
    return ToolInventoryService(xlsx)


def test_step1_12_pipeline_happy_path_stops_at_awaiting_external(
    local_storage, registry_service, workflow_state_service, inventory
):
    """Without any external_scoring_result.json, Step 1-12 must complete with
    Step 11 in `awaiting_external_input` and Step 12 in
    `awaiting_external_scoring`. We never invent a ranking."""
    graph = build_step1_12_graph(
        storage=local_storage,
        registry=registry_service,
        workflow_state=workflow_state_service,
        mcp_client=LocalMCPClient(inventory=inventory),
    )
    final = graph.invoke(
        {
            "intake_request": {
                "raw_user_query": "HER2 ADC with vc-MMAE",
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
    ):
        assert artifacts.get(key), f"missing {key}"

    state = workflow_state_service.get(run_id)
    for s in (f"step_{i:02d}" for i in range(1, 13)):
        assert state["steps"][s] == "completed", state["steps"][s]

    assert final["results"]["step_10"]["handoff_status"] == "awaiting_external_scoring"
    assert final["results"]["step_11"]["validation_status"] == "awaiting_external_input"
    assert final["results"]["step_12"]["ranking_status"] == "awaiting_external_scoring"


def test_step1_12_pipeline_skips_steps_under_wait_for_input(
    local_storage, registry_service, workflow_state_service, inventory
):
    """Payload absent → plan_status=wait_for_input → Step 5-12 all skipped."""
    graph = build_step1_12_graph(
        storage=local_storage,
        registry=registry_service,
        workflow_state=workflow_state_service,
        mcp_client=LocalMCPClient(inventory=inventory),
    )
    final = graph.invoke(
        {
            "intake_request": {
                "raw_user_query": "HER2 only",
                "user_provided_context": {
                    "target_or_antigen_text": "HER2",
                    "candidate_text": "Trastuzumab",
                },
            }
        }
    )
    for s in ("step_05", "step_06", "step_07", "step_08", "step_09",
              "step_10", "step_11", "step_12"):
        assert final["results"][s]["executed"] is False
        assert final["results"][s]["plan_status"] == "wait_for_input"
