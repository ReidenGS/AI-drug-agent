"""Step 7/8/9 API plan_status gating — same behavior as Step 5/6."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import app.deps as deps
from app.agents.supervisor_agent import SupervisorAgent
from app.llm.provider import MockLLMProvider
from app.main import app
from app.services.input_readiness_service import InputReadinessService
from app.services.structured_query_service import StructuredQueryService
from app.services.workflow_setup_service import WorkflowSetupService


@pytest.fixture
def client(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("STORAGE_MODE", "local")
    monkeypatch.setenv("LOCAL_STORAGE_ROOT", str(tmp_path / "store"))
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    from app.settings import get_settings

    for fn in (
        get_settings, deps.get_storage, deps.get_registry_service,
        deps.get_workflow_state_service, deps.get_tool_inventory_service,
        deps.get_mcp_client, deps.get_llm_provider,
    ):
        fn.cache_clear()
    yield TestClient(app)
    for fn in (
        get_settings, deps.get_storage, deps.get_registry_service,
        deps.get_workflow_state_service, deps.get_tool_inventory_service,
        deps.get_mcp_client, deps.get_llm_provider,
    ):
        fn.cache_clear()


def _seed_through_step_4(client: TestClient, *, with_payload=True) -> str:
    ctx = {"target_or_antigen_text": "HER2", "candidate_text": "Trastuzumab"}
    if with_payload:
        ctx["payload_linker_text"] = "vc-MMAE"
    resp = client.post("/runs", json={"raw_user_query": "HER2 ADC", "user_provided_context": ctx})
    assert resp.status_code == 200
    run_id = resp.json()["run_id"]

    storage = deps.get_storage()
    reg = deps.get_registry_service()
    ws = deps.get_workflow_state_service()
    StructuredQueryService(storage, reg, ws, SupervisorAgent(llm=MockLLMProvider())).parse(run_id)
    InputReadinessService(storage, reg, ws).check(run_id)
    WorkflowSetupService(storage, reg, ws).plan(run_id)
    return run_id


@pytest.mark.parametrize("step_num", [7, 8, 9])
def test_step789_api_returns_409_when_plan_status_wait_for_input(client: TestClient, step_num: int):
    run_id = _seed_through_step_4(client, with_payload=False)  # payload absent → wait_for_input
    plan = deps.get_storage().read_json(
        deps.get_storage().run_key(run_id, "inputs/run_step_plan.json")
    )
    assert plan["plan_status"] == "wait_for_input"

    resp = client.post(f"/runs/{run_id}/steps/{step_num}/execute")
    assert resp.status_code == 409
    body = resp.json()
    assert body["code"] == "workflow_state_error"
    assert body["detail"]["plan_status"] == "wait_for_input"
    assert body["detail"]["step_id"].startswith(f"step_{step_num:02d}_")


@pytest.mark.parametrize("step_num", [7, 8, 9])
def test_step789_api_returns_409_when_plan_status_blocked(client: TestClient, step_num: int):
    run_id = _seed_through_step_4(client, with_payload=True)
    storage = deps.get_storage()
    key = storage.run_key(run_id, "inputs/run_step_plan.json")
    plan = storage.read_json(key)
    plan["plan_status"] = "blocked"
    storage.write_json(key, plan)

    resp = client.post(f"/runs/{run_id}/steps/{step_num}/execute")
    assert resp.status_code == 409
    assert resp.json()["detail"]["plan_status"] == "blocked"
