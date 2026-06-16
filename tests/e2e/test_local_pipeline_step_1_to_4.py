"""End-to-end: run Step 1 → 2 (stub) → 3 → 4 on local storage."""

from __future__ import annotations

from app.services.input_readiness_service import InputReadinessService
from app.services.intake_service import IntakeService
from app.services.workflow_setup_service import WorkflowSetupService
from app.schemas.step_02_structured_query import (
    SourceRawRequestRef,
    StructuredQuery,
    TaskIntent,
)
from app.utils.ids import new_artifact_id
from app.utils.time import now_iso


def test_step_1_to_4_local_chain(local_storage, registry_service, workflow_state_service):
    # Step 1
    intake = IntakeService(local_storage, registry_service, workflow_state_service)
    rec = intake.submit(
        raw_user_query="HER2 ADC, MMAE payload",
        user_provided_context={
            "target_or_antigen_text": "HER2",
            "candidate_text": "Trastuzumab analog",
            "payload_linker_text": "vc-MMAE",
        },
    )

    # Step 2 (stub)
    reg = registry_service.get(rec.run_id)
    sq = StructuredQuery(
        run_id=rec.run_id,
        parsed_at=now_iso(),
        source_raw_request_ref=SourceRawRequestRef(
            raw_request_record_id=reg.active_artifacts.raw_request_record_id
        ),
        task_intent=TaskIntent(task_type="adc_design"),
    )
    sq_id = new_artifact_id("structured_query")
    local_storage.write_json(
        local_storage.run_key(rec.run_id, "inputs/structured_query.json"),
        {"artifact_id": sq_id, **sq.model_dump()},
    )
    registry_service.update_active(rec.run_id, structured_query_id=sq_id)
    workflow_state_service.mark(rec.run_id, "step_02", "completed")

    # Step 3
    readiness = InputReadinessService(
        local_storage, registry_service, workflow_state_service
    ).check(rec.run_id)
    assert readiness.input_readiness_status == "ready"

    # Step 4
    plan = WorkflowSetupService(
        local_storage, registry_service, workflow_state_service
    ).plan(rec.run_id)
    assert plan.plan_status == "ready_to_execute"

    # Final registry has all four step artifacts
    final_reg = registry_service.get(rec.run_id)
    assert final_reg.active_artifacts.raw_request_record_id
    assert final_reg.active_artifacts.structured_query_id
    assert final_reg.active_artifacts.input_readiness_status_id
    assert final_reg.active_artifacts.run_step_plan_id
