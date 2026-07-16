from __future__ import annotations

from pydantic import TypeAdapter
import pytest

from app.services.intake_service import IntakeService
from app.utils.ids import SessionId


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


def test_intake_generates_session_and_allows_isolated_runs_to_reuse_it(
    local_storage, registry_service, workflow_state_service
):
    service = IntakeService(
        storage=local_storage,
        registry=registry_service,
        workflow_state=workflow_state_service,
    )
    first = service.submit(raw_user_query="first isolated run")
    session_id = TypeAdapter(SessionId).validate_python(
        first.session_id, strict=True
    )
    second = service.submit(
        raw_user_query="second isolated run", session_id=session_id
    )

    assert first.run_id != second.run_id
    assert first.session_id == second.session_id == session_id
    first_raw = local_storage.read_json(
        local_storage.run_key(first.run_id, "inputs/raw_request_record.json")
    )
    second_raw = local_storage.read_json(
        local_storage.run_key(second.run_id, "inputs/raw_request_record.json")
    )
    assert first_raw["session_id"] == second_raw["session_id"] == session_id
    assert first_raw["run_id"] != second_raw["run_id"]
    assert (
        registry_service.get(first.run_id).run_artifact_registry_id
        != registry_service.get(second.run_id).run_artifact_registry_id
    )


def test_invalid_explicit_session_is_rejected_before_any_write(
    local_storage, registry_service, workflow_state_service
):
    service = IntakeService(
        storage=local_storage,
        registry=registry_service,
        workflow_state=workflow_state_service,
    )

    with pytest.raises(ValueError, match="^session_id_invalid$") as caught:
        service.submit(
            raw_user_query="must not persist",
            run_id="run_20260715_deadbeef",
            session_id="sk-live-invalid-session",  # type: ignore[arg-type]
        )

    assert "sk-live" not in repr(caught.value)
    assert local_storage.list_prefix(local_storage.prefix) == []
