"""End-to-end LangGraph Step 1→9 against LocalStorage + inventory-scoped MCP."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from app.graph.adc_graph import build_step1_9_graph
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


def test_step1_9_pipeline_happy_path(
    local_storage, registry_service, workflow_state_service, inventory
):
    graph = build_step1_9_graph(
        storage=local_storage,
        registry=registry_service,
        workflow_state=workflow_state_service,
        mcp_client=LocalMCPClient(inventory=inventory),
    )
    final = graph.invoke(
        {
            "intake_request": {
                "raw_user_query": (
                    "Design ADC against HER2 using PDB 1N8Z; vc-MMAE payload"
                ),
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
        "raw_request_record",
        "structured_query",
        "input_readiness_status",
        "run_step_plan",
        "candidate_context_table",
        "structured_liability_summary",
        "prepared_structure_input_package",
        "structure_prediction_and_interface_results",
        "structure_variant_and_compound_screening",
    ):
        assert artifacts.get(key), f"missing artifact id for {key}"

    state = workflow_state_service.get(run_id)
    for s in (
        "step_01", "step_02", "step_03", "step_04", "step_05",
        "step_06", "step_07", "step_08", "step_09",
    ):
        assert state["steps"][s] == "completed", f"{s} not completed: {state['steps'][s]}"

    reg = registry_service.get(run_id)
    assert reg.active_artifacts.prepared_structure_input_package_id
    assert reg.active_artifacts.structure_prediction_and_interface_results_id
    assert reg.active_artifacts.structure_variant_and_compound_screening_id


def test_step1_9_pipeline_step7_9_skip_on_wait_for_input(
    local_storage, registry_service, workflow_state_service, inventory
):
    """Without payload, Step 4 plan_status=wait_for_input; Step 5-9 nodes must
    all short-circuit and not execute their agents."""
    graph = build_step1_9_graph(
        storage=local_storage,
        registry=registry_service,
        workflow_state=workflow_state_service,
        mcp_client=LocalMCPClient(inventory=inventory),
    )
    final = graph.invoke(
        {
            "intake_request": {
                "raw_user_query": "Build something",
                "user_provided_context": {
                    "target_or_antigen_text": "HER2",
                    "candidate_text": "Trastuzumab",
                },
            }
        }
    )
    run_id = final["run_id"]
    for s in ("step_05", "step_06", "step_07", "step_08", "step_09"):
        assert final["results"][s]["executed"] is False
        assert final["results"][s]["plan_status"] == "wait_for_input"

    reg = registry_service.get(run_id)
    assert reg.active_artifacts.prepared_structure_input_package_id is None
    assert reg.active_artifacts.structure_prediction_and_interface_results_id is None
    assert reg.active_artifacts.structure_variant_and_compound_screening_id is None
