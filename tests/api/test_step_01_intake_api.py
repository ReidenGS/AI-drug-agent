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
