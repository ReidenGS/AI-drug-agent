"""Step 6 DevelopabilityAgent MVP tests."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from app.agents.candidate_context_agent import CandidateContextAgent
from app.agents.developability_agent import DevelopabilityAgent
from app.agents.supervisor_agent import SupervisorAgent
from app.llm.provider import MockLLMProvider
from app.mcp.client import LocalMCPClient
from app.services.input_readiness_service import InputReadinessService
from app.services.intake_service import IntakeService
from app.services.structured_query_service import StructuredQueryService
from app.services.tool_inventory_service import ToolInventoryService
from app.services.workflow_setup_service import WorkflowSetupService
from app.utils.errors import WorkflowStateError


PROJECT_ROOT = Path(__file__).resolve().parents[2].parent
DEFAULT_XLSX = PROJECT_ROOT / "项目文件" / "ToolUniversity_inventory_v0.2.xlsx"


def _bindings(canned: dict[str, dict]) -> dict:
    def make(payload):
        def _fn(**_kwargs):
            return payload
        return _fn
    return {name: make(p) for name, p in canned.items()}


def _seed_through_step_5(
    local_storage,
    registry_service,
    workflow_state_service,
    *,
    step5_bindings: dict[str, dict] | None = None,
) -> str:
    intake = IntakeService(local_storage, registry_service, workflow_state_service)
    rec = intake.submit(
        raw_user_query="HER2 ADC with vc-MMAE",
        user_provided_context={
            "target_or_antigen_text": "HER2",
            "candidate_text": "Trastuzumab analog",
            "payload_linker_text": "vc-MMAE",
        },
    )
    StructuredQueryService(
        local_storage, registry_service, workflow_state_service, SupervisorAgent(llm=MockLLMProvider())
    ).parse(rec.run_id)
    InputReadinessService(local_storage, registry_service, workflow_state_service).check(rec.run_id)
    WorkflowSetupService(local_storage, registry_service, workflow_state_service).plan(rec.run_id)
    CandidateContextAgent(
        storage=local_storage,
        registry=registry_service,
        workflow_state=workflow_state_service,
        mcp_client=LocalMCPClient(bindings=step5_bindings or _bindings({
            "SAbDab_search_structures": {"hits": [{"pdb_id": "1n8z"}]},
            "ChEMBL_search_molecules": {"hits": [{"chembl_id": "CHEMBL1201585"}]},
            "ChEMBL_search_substructure": {"hits": [{"chembl_id": "CHEMBL_linker"}]},
        })),
    ).run(rec.run_id)
    return rec.run_id


# ── 1. missing Step 5 artifact ───────────────────────────────────────────────

def test_step6_requires_step5_artifact(
    local_storage, registry_service, workflow_state_service
):
    intake = IntakeService(local_storage, registry_service, workflow_state_service)
    rec = intake.submit(
        raw_user_query="hello", user_provided_context={"target_or_antigen_text": "HER2"}
    )
    agent = DevelopabilityAgent(
        storage=local_storage,
        registry=registry_service,
        workflow_state=workflow_state_service,
        mcp_client=LocalMCPClient(),
    )
    with pytest.raises(WorkflowStateError, match="Step 5"):
        agent.run(rec.run_id)


# ── 2. happy path: produces structured_liability_summary ─────────────────────

def test_step6_produces_summary_from_step5(
    local_storage, registry_service, workflow_state_service
):
    run_id = _seed_through_step_5(local_storage, registry_service, workflow_state_service)
    agent = DevelopabilityAgent(
        storage=local_storage,
        registry=registry_service,
        workflow_state=workflow_state_service,
        mcp_client=LocalMCPClient(),
    )
    summary = agent.run(run_id)
    assert summary.run_id == run_id
    assert summary.step_id == "step_06_developability"
    assert summary.candidate_liability_results, "Step 6 should produce one row per Step 5 candidate"
    # registry updated
    reg = registry_service.get(run_id)
    assert reg.active_artifacts.structured_liability_summary_id is not None
    # workflow_state updated
    state = workflow_state_service.get(run_id)
    assert state["steps"]["step_06"] == "completed"


# ── 3. inventory scope: Step 6 agent only calls Step 6 tools ─────────────────

def test_step6_only_calls_step6_inventory_tools(
    local_storage, registry_service, workflow_state_service
):
    xlsx = os.environ.get("TOOL_INVENTORY_XLSX", str(DEFAULT_XLSX))
    if not Path(xlsx).exists():
        pytest.skip(f"Inventory xlsx not available at {xlsx}")

    # Use inventory-scoped client: any non-Step-6 tool the agent tries would
    # come back as "skipped". The agent's lane router only ever picks Step 6
    # tool names, so we expect no skipped-by-scope outcomes here.
    run_id = _seed_through_step_5(local_storage, registry_service, workflow_state_service)
    inventory = ToolInventoryService(xlsx)
    mcp = LocalMCPClient(inventory=inventory)

    agent = DevelopabilityAgent(
        storage=local_storage,
        registry=registry_service,
        workflow_state=workflow_state_service,
        mcp_client=mcp,
    )
    agent.run(run_id)
    persisted = local_storage.read_json(
        local_storage.run_key(run_id, "structured_liability_summary.json")
    )

    # Every recorded tool call must be a Step 6 tool from v0.2 inventory.
    step6_tool_names = {
        e.tool_name
        for e in inventory.load()
        if (e.step_id or "").strip() == "6"
    }
    assert step6_tool_names, "inventory has no Step 6 tools — fixture problem"

    for cand in persisted["candidate_liability_results"]:
        for lane in cand["lane_results"]:
            for tc in lane["tool_call_records"]:
                assert tc["tool_name"] in step6_tool_names, (
                    f"Step 6 agent called non-Step-6 tool: {tc['tool_name']}"
                )
                # Nothing the agent calls should be rejected by scope — if it
                # were, the lane routing has a bug.
                assert tc["run_status"] != "skipped" or lane["input_status"] == "missing"


# ── 4. unwired wrappers → dependency_unavailable, status partial ─────────────

def test_step6_handles_unwired_wrappers_gracefully(
    local_storage, registry_service, workflow_state_service
):
    """Without injected bindings or inventory, the default LocalMCPClient
    returns `dependency_unavailable` for all unwired Step 6 tools. The step
    must still produce a partial summary, not crash."""
    run_id = _seed_through_step_5(local_storage, registry_service, workflow_state_service)
    agent = DevelopabilityAgent(
        storage=local_storage,
        registry=registry_service,
        workflow_state=workflow_state_service,
        mcp_client=LocalMCPClient(),  # no inventory, no bindings overrides
    )
    summary = agent.run(run_id)
    persisted = local_storage.read_json(
        local_storage.run_key(run_id, "structured_liability_summary.json")
    )
    assert summary.prefilter_status in {"partial", "completed_with_missing_lanes"}
    # at least one tool call recorded as dependency_unavailable
    dep_unavail = [
        tc
        for cand in persisted["candidate_liability_results"]
        for lane in cand["lane_results"]
        for tc in lane["tool_call_records"]
        if tc["run_status"] == "dependency_unavailable"
    ]
    assert dep_unavail, "expected dependency_unavailable for unwired wrappers"


# ── 5. raw payload isolation ─────────────────────────────────────────────────

def test_step6_raw_payload_does_not_leak_into_normalized_records(
    local_storage, registry_service, workflow_state_service
):
    run_id = _seed_through_step_5(local_storage, registry_service, workflow_state_service)
    bindings = _bindings({
        "DrugProps_pains_filter": {"hits": [{"alert": "michael_acceptor"}]},
        "PROSITE_scan_sequence": {"hits": [{"motif": "GLYCOSYLATION"}]},
        "EBIProteins_get_features": {"hits": [{"feature": "epitope"}]},
        "ProteinsPlus_profile_structure_quality": {"hits": [{"quality": "low"}]},
        "ChEMBL_search_activities": {"hits": [{"assay_id": "A1"}]},
    })
    agent = DevelopabilityAgent(
        storage=local_storage,
        registry=registry_service,
        workflow_state=workflow_state_service,
        mcp_client=LocalMCPClient(bindings=bindings),
    )
    agent.run(run_id)
    persisted = local_storage.read_json(
        local_storage.run_key(run_id, "structured_liability_summary.json")
    )

    # No raw `hits` key should appear in normalized liability fields.
    for cand in persisted["candidate_liability_results"]:
        for lane in cand["lane_results"]:
            # liability_flags and lane_summary must not embed raw payload
            assert "hits" not in json.dumps(lane["liability_flags"])
            assert "hits" not in (lane.get("lane_summary") or "")
            for tc in lane["tool_call_records"]:
                if tc["run_status"] == "success":
                    assert tc["tool_output_ref"]
                    assert local_storage.exists(tc["tool_output_ref"])
                    raw = local_storage.read_json(tc["tool_output_ref"])
                    assert "output" in raw  # raw payload lives only here
