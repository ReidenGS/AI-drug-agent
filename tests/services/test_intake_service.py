from __future__ import annotations

from app.services.intake_service import IntakeService


def test_intake_creates_run_and_registry(local_storage, registry_service, workflow_state_service):
    svc = IntakeService(
        storage=local_storage, registry=registry_service, workflow_state=workflow_state_service
    )
    rec = svc.submit(
        raw_user_query="Design ADC against HER2",
        user_provided_context={"target_or_antigen_text": "HER2"},
    )
    assert rec.run_id.startswith("run_")
    reg = registry_service.get(rec.run_id)
    assert reg.active_artifacts.raw_request_record_id is not None
    state = workflow_state_service.get(rec.run_id)
    assert state["steps"]["step_01"] == "completed"
