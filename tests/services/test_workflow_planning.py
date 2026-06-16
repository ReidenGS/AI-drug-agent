"""Step 4 deterministic planning engine tests."""

from __future__ import annotations

import pytest

from app.services.input_readiness_service import InputReadinessService
from app.services.intake_service import IntakeService
from app.services.workflow_setup_service import WorkflowSetupService, planned_step_for
from app.schemas.step_02_structured_query import (
    MentionedEntities,
    SourceRawRequestRef,
    StructuredQuery,
    TaskIntent,
)
from app.utils.errors import WorkflowStateError
from app.utils.ids import new_artifact_id
from app.utils.time import now_iso


def _bootstrap(
    local_storage, registry_service, workflow_state_service, *,
    target=None, candidate=None, payload=None, linker=None, referenced_inputs=None,
    raw_context: dict | None = None, uploaded_files: list | None = None,
) -> str:
    intake = IntakeService(local_storage, registry_service, workflow_state_service)
    rec = intake.submit(
        raw_user_query="x",
        user_provided_context=raw_context or {},
        uploaded_files=uploaded_files,
    )
    reg = registry_service.get(rec.run_id)
    sq = StructuredQuery(
        run_id=rec.run_id,
        parsed_at=now_iso(),
        source_raw_request_ref=SourceRawRequestRef(
            raw_request_record_id=reg.active_artifacts.raw_request_record_id
        ),
        task_intent=TaskIntent(task_type="adc_design"),
        mentioned_entities=MentionedEntities(
            target_or_antigen_text=target,
            antibody_candidate_text=candidate,
            payload_text=payload,
            linker_text=linker,
        ),
        referenced_inputs=referenced_inputs or [],
    )
    sq_id = new_artifact_id("structured_query")
    local_storage.write_json(
        local_storage.run_key(rec.run_id, "inputs/structured_query.json"),
        {"artifact_id": sq_id, **sq.model_dump()},
    )
    registry_service.update_active(rec.run_id, structured_query_id=sq_id)
    workflow_state_service.mark(rec.run_id, "step_02", "completed")
    InputReadinessService(local_storage, registry_service, workflow_state_service).check(rec.run_id)
    return rec.run_id


def _plan(local_storage, registry_service, workflow_state_service, run_id):
    return WorkflowSetupService(local_storage, registry_service, workflow_state_service).plan(run_id)


# ── 1. fully sufficient input → Step 5/6 runnable ────────────────────────────

def test_full_input_plans_step5_and_step6_run(
    local_storage, registry_service, workflow_state_service
):
    run_id = _bootstrap(
        local_storage, registry_service, workflow_state_service,
        target="HER2", candidate="Trastuzumab", payload="MMAE", linker="vc",
        raw_context={
            "target_or_antigen_text": "HER2",
            "candidate_text": "Trastuzumab",
            "payload_linker_text": "vc-MMAE",
        },
    )
    plan = _plan(local_storage, registry_service, workflow_state_service, run_id)
    assert plan.plan_status == "ready_to_execute"
    s5 = planned_step_for(plan, "step_05_candidate_context")
    s6 = planned_step_for(plan, "step_06_developability")
    assert s5 and s5.planned_status == "run"
    assert s6 and s6.planned_status == "run"


# ── 2. missing target → blocked ──────────────────────────────────────────────

def test_missing_target_blocks_plan(
    local_storage, registry_service, workflow_state_service
):
    """Readiness is blocked when target is missing; Step 4 must raise."""
    run_id = _bootstrap(local_storage, registry_service, workflow_state_service)
    with pytest.raises(WorkflowStateError, match="blocked"):
        _plan(local_storage, registry_service, workflow_state_service, run_id)


# ── 3. missing antibody → Step 5 runs, Step 6 partial ────────────────────────

def test_missing_antibody_runs_step5_but_partial_step6(
    local_storage, registry_service, workflow_state_service
):
    run_id = _bootstrap(
        local_storage, registry_service, workflow_state_service,
        target="HER2", payload="MMAE", linker="vc",
        raw_context={"target_or_antigen_text": "HER2", "payload_linker_text": "vc-MMAE"},
    )
    plan = _plan(local_storage, registry_service, workflow_state_service, run_id)
    s5 = planned_step_for(plan, "step_05_candidate_context")
    s6 = planned_step_for(plan, "step_06_developability")
    assert s5.planned_status == "run"
    assert s5.lane_flags.get("antibody_discovery_lane") is True
    assert s6.planned_status == "partial"
    assert s6.lane_flags.get("antibody_lane") is False


# ── 4. missing payload → compound lane partial ───────────────────────────────

def test_missing_payload_marks_compound_lane_partial(
    local_storage, registry_service, workflow_state_service
):
    run_id = _bootstrap(
        local_storage, registry_service, workflow_state_service,
        target="HER2", candidate="Trastuzumab",
        raw_context={"target_or_antigen_text": "HER2", "candidate_text": "Trastuzumab"},
    )
    plan = _plan(local_storage, registry_service, workflow_state_service, run_id)
    s6 = planned_step_for(plan, "step_06_developability")
    assert s6.planned_status == "partial"
    assert s6.lane_flags.get("compound_lane") is False
    s5 = planned_step_for(plan, "step_05_candidate_context")
    assert s5.lane_flags.get("compound_lane") is False


# ── 5. missing structure/sequence → Step 7-9 partial ────────────────────────

def test_missing_structure_marks_step7_to_9_partial(
    local_storage, registry_service, workflow_state_service
):
    run_id = _bootstrap(
        local_storage, registry_service, workflow_state_service,
        target="HER2", candidate="Trastuzumab", payload="MMAE", linker="vc",
        raw_context={
            "target_or_antigen_text": "HER2",
            "candidate_text": "Trastuzumab",
            "payload_linker_text": "vc-MMAE",
        },
    )
    plan = _plan(local_storage, registry_service, workflow_state_service, run_id)
    for sid in (
        "step_07_structure_input",
        "step_08_structure_evaluation",
        "step_09_structure_design",
    ):
        p = planned_step_for(plan, sid)
        assert p.planned_status == "partial"
        assert p.lane_flags.get("structure_lane") is False


# ── 6. graph respects plan: Step 5 blocked → graph skips Step 5/6 ───────────

def test_graph_respects_plan_when_step5_skipped(
    local_storage, registry_service, workflow_state_service
):
    """If we tamper with the plan to mark Step 5/6 as skip, the graph nodes
    must not run them. Direct unit-test of nodes here avoids needing a fully
    consistent pipeline through Step 2."""
    run_id = _bootstrap(
        local_storage, registry_service, workflow_state_service,
        target="HER2", candidate="Trastuzumab", payload="MMAE", linker="vc",
        raw_context={
            "target_or_antigen_text": "HER2",
            "candidate_text": "Trastuzumab",
            "payload_linker_text": "vc-MMAE",
        },
    )
    plan = _plan(local_storage, registry_service, workflow_state_service, run_id)
    # patch persisted plan: skip Step 5 + Step 6
    plan_dict = plan.model_dump()
    for p in plan_dict["planned_steps"]:
        if p["step_id"] in {"step_05_candidate_context", "step_06_developability"}:
            p["planned_status"] = "skip"
            p["reason"] = "test override"
    local_storage.write_json(
        local_storage.run_key(run_id, "inputs/run_step_plan.json"),
        {"artifact_id": "rp_x", **plan_dict},
    )

    from app.graph.nodes import make_node_step_05, make_node_step_06
    from app.mcp.client import LocalMCPClient

    mcp = LocalMCPClient()
    s5 = make_node_step_05(local_storage, registry_service, workflow_state_service, mcp)
    s6 = make_node_step_06(local_storage, registry_service, workflow_state_service, mcp)
    state = {"run_id": run_id}
    state5 = s5(state)
    state6 = s6(state5)
    assert state5["results"]["step_05"]["executed"] is False
    assert state6["results"]["step_06"]["executed"] is False
    # Registry should NOT have Step 5/6 artifact ids
    reg = registry_service.get(run_id)
    assert reg.active_artifacts.candidate_context_table_id is None
    assert reg.active_artifacts.structured_liability_summary_id is None
    # Workflow state marked skipped
    state = workflow_state_service.get(run_id)
    assert state["steps"]["step_05"] == "skipped"
    assert state["steps"]["step_06"] == "skipped"
