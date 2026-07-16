from __future__ import annotations

import pytest

from app.services.input_readiness_service import InputReadinessService
from app.services.intake_service import IntakeService
from app.schemas.step_02_structured_query import (
    SourceRawRequestRef,
    StructuredQuery,
    TaskIntent,
)
from app.utils.ids import new_artifact_id
from app.utils.time import now_iso


def _bootstrap_step_2(local_storage, registry_service, workflow_state_service, run_id: str) -> None:
    reg = registry_service.get(run_id)
    sq = StructuredQuery(
        run_id=run_id,
        parsed_at=now_iso(),
        source_raw_request_ref=SourceRawRequestRef(
            raw_request_record_id=reg.active_artifacts.raw_request_record_id
        ),
        task_intent=TaskIntent(task_type="adc_design"),
    )
    sq_id = new_artifact_id("structured_query")
    local_storage.write_json(
        local_storage.run_key(run_id, "inputs/structured_query.json"),
        {"artifact_id": sq_id, **sq.model_dump()},
    )
    registry_service.update_active(run_id, structured_query_id=sq_id)
    workflow_state_service.mark(run_id, "step_02", "completed")


def test_readiness_requests_user_input_without_target(
    local_storage, registry_service, workflow_state_service
):
    intake = IntakeService(local_storage, registry_service, workflow_state_service)
    rec = intake.submit(raw_user_query="design something", user_provided_context={})
    _bootstrap_step_2(local_storage, registry_service, workflow_state_service, rec.run_id)
    out = InputReadinessService(local_storage, registry_service, workflow_state_service).check(rec.run_id)
    assert out.input_readiness_status == "needs_user_input"


def test_readiness_needs_user_input_when_payload_missing(
    local_storage, registry_service, workflow_state_service
):
    """Payload missing is a warning (compound lanes will be partial). With only
    target + candidate present, readiness is `needs_user_input`, not `ready`."""
    intake = IntakeService(local_storage, registry_service, workflow_state_service)
    rec = intake.submit(
        raw_user_query="design HER2 ADC",
        user_provided_context={
            "target_or_antigen_text": "HER2",
            "candidate_text": "Trastuzumab",
        },
    )
    _bootstrap_step_2(local_storage, registry_service, workflow_state_service, rec.run_id)
    out = InputReadinessService(local_storage, registry_service, workflow_state_service).check(rec.run_id)
    assert out.input_readiness_status == "needs_user_input"
    cats = {m.category for m in out.missing_input_checklist}
    assert "payload_or_linker" in cats


def test_readiness_ready_with_full_adc_context(
    local_storage, registry_service, workflow_state_service
):
    intake = IntakeService(local_storage, registry_service, workflow_state_service)
    rec = intake.submit(
        raw_user_query="design HER2 ADC with vc-MMAE payload",
        user_provided_context={
            "target_or_antigen_text": "HER2",
            "candidate_text": "Trastuzumab",
            "payload_linker_text": "vc-MMAE",
        },
    )
    _bootstrap_step_2(local_storage, registry_service, workflow_state_service, rec.run_id)
    out = InputReadinessService(local_storage, registry_service, workflow_state_service).check(rec.run_id)
    assert out.input_readiness_status == "ready"


def test_readiness_requires_step1_and_step2(local_storage, registry_service, workflow_state_service):
    intake = IntakeService(local_storage, registry_service, workflow_state_service)
    rec = intake.submit(raw_user_query="x", user_provided_context={"target_or_antigen_text": "x"})
    # no Step 2 → should raise
    with pytest.raises(ValueError):
        InputReadinessService(local_storage, registry_service, workflow_state_service).check(rec.run_id)
