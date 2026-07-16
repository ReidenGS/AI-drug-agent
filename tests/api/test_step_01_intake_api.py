"""Verify Step 1 API re-execute does NOT silently mint a new run_id."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import app.deps as deps
from app.main import app
from app.services.intake_service import IntakeService


@pytest.fixture
def client(tmp_path: Path, monkeypatch):
    # Force a fresh local storage rooted in tmp_path for the duration of the test
    # and reset the lru_cache that holds storage/registry/workflow_state singletons.
    monkeypatch.setenv("STORAGE_MODE", "local")
    monkeypatch.setenv("LOCAL_STORAGE_ROOT", str(tmp_path / "store"))
    from app.settings import get_settings

    get_settings.cache_clear()
    deps.get_storage.cache_clear()
    deps.get_registry_service.cache_clear()
    deps.get_workflow_state_service.cache_clear()
    yield TestClient(app)
    get_settings.cache_clear()
    deps.get_storage.cache_clear()
    deps.get_registry_service.cache_clear()
    deps.get_workflow_state_service.cache_clear()


def test_step_01_reexecute_unknown_run_returns_404(client: TestClient):
    resp = client.post("/runs/run_does_not_exist/steps/1/execute")
    assert resp.status_code == 404
    body = resp.json()
    assert body["code"] == "not_found"


def test_step_01_reexecute_known_run_returns_existing_artifact(client: TestClient):
    intake = IntakeService(
        storage=deps.get_storage(),
        registry=deps.get_registry_service(),
        workflow_state=deps.get_workflow_state_service(),
    )
    rec = intake.submit(
        raw_user_query="HER2 ADC",
        user_provided_context={"target_or_antigen_text": "HER2"},
    )
    real_run_id = rec.run_id

    resp = client.post(f"/runs/{real_run_id}/steps/1/execute")
    assert resp.status_code == 200
    body = resp.json()
    assert body["run_id"] == real_run_id, "Step 1 re-execute must not allocate a new run_id"


def test_create_and_get_run_return_generated_session(client: TestClient):
    created = client.post("/runs", json={"raw_user_query": "Analyze HER2"})
    assert created.status_code == 200
    body = created.json()
    session_id = body["session_id"]
    assert session_id.startswith("sess_") and len(session_id) == 21
    assert body["raw_request_record"]["session_id"] == session_id

    fetched = client.get(f"/runs/{body['run_id']}")
    assert fetched.status_code == 200
    assert fetched.json()["session_id"] == session_id


def test_create_run_reuses_valid_session_but_keeps_run_authority_isolated(
    client: TestClient,
):
    session_id = "sess_0123456789abcdef"
    first = client.post(
        "/runs", json={"raw_user_query": "first", "session_id": session_id}
    ).json()
    second = client.post(
        "/runs", json={"raw_user_query": "second", "session_id": session_id}
    ).json()

    assert first["session_id"] == second["session_id"] == session_id
    assert first["run_id"] != second["run_id"]
    assert (
        first["raw_request_record"]["run_artifact_registry_id"]
        != second["raw_request_record"]["run_artifact_registry_id"]
    )


def test_invalid_json_session_is_422_and_writes_no_artifact(client: TestClient):
    before = list(deps.get_storage().list_prefix(deps.get_storage().prefix))
    response = client.post(
        "/runs",
        json={"raw_user_query": "invalid session", "session_id": "session-secret"},
    )
    after = list(deps.get_storage().list_prefix(deps.get_storage().prefix))

    assert response.status_code == 422
    assert response.json()["detail"] == "session_id_invalid"
    assert "session-secret" not in response.text
    assert before == after == []


def test_get_run_rejects_tampered_raw_identity_with_compact_error(
    client: TestClient,
):
    created = client.post(
        "/runs", json={"raw_user_query": "identity authority"}
    ).json()
    storage = deps.get_storage()
    key = storage.run_key(
        created["run_id"], "inputs/raw_request_record.json"
    )
    body = storage.read_json(key)
    body["artifact_id"] = "sk-live-tampered-artifact"
    storage.write_json(key, body)

    response = client.get(f"/runs/{created['run_id']}")

    assert response.status_code == 409
    assert response.json()["detail"] == "raw_request_record_identity_mismatch"
    assert "sk-live" not in response.text


def test_get_run_rejects_invalid_persisted_session_without_leaking_it(
    client: TestClient,
):
    created = client.post(
        "/runs", json={"raw_user_query": "session authority"}
    ).json()
    storage = deps.get_storage()
    key = storage.run_key(
        created["run_id"], "inputs/raw_request_record.json"
    )
    body = storage.read_json(key)
    body["session_id"] = "sk-live-invalid-persisted-session"
    storage.write_json(key, body)

    response = client.get(f"/runs/{created['run_id']}")

    assert response.status_code == 409
    assert response.json()["detail"] == "raw_request_record_schema_invalid"
    assert "sk-live" not in response.text
