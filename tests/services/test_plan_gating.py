"""Step 4 plan_status as a global execution gate.

`wait_for_input` and `blocked` plans must prevent Step 5/6 from running both
through the LangGraph node path and through the direct Step 5/6 API path.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import app.deps as deps
from app.agents.supervisor_agent import SupervisorAgent
from app.llm.provider import MockLLMProvider
from app.main import app
from app.mcp.client import LocalMCPClient
from app.services.input_readiness_service import InputReadinessService
from app.services.intake_service import IntakeService
from app.services.structured_query_service import StructuredQueryService
from app.services.workflow_setup_service import WorkflowSetupService, execution_decision


# ── shared helpers ───────────────────────────────────────────────────────────

def _seed_through_step_4(
    local_storage, registry_service, workflow_state_service, *, with_payload=True
):
    intake = IntakeService(local_storage, registry_service, workflow_state_service)
    rec = intake.submit(
        raw_user_query="HER2 ADC",
        user_provided_context={
            "target_or_antigen_text": "HER2",
            "candidate_text": "Trastuzumab",
            **({"payload_linker_text": "vc-MMAE"} if with_payload else {}),
        },
    )
    StructuredQueryService(
        local_storage, registry_service, workflow_state_service,
        SupervisorAgent(llm=MockLLMProvider()),
    ).parse(rec.run_id)
    InputReadinessService(
        local_storage, registry_service, workflow_state_service
    ).check(rec.run_id)
    WorkflowSetupService(
        local_storage, registry_service, workflow_state_service
    ).plan(rec.run_id)
    return rec.run_id


def _overwrite_plan_status(local_storage, run_id: str, new_status: str) -> None:
    key = local_storage.run_key(run_id, "inputs/run_step_plan.json")
    plan = local_storage.read_json(key)
    plan["plan_status"] = new_status
    local_storage.write_json(key, plan)


# ── execution_decision unit-level checks ────────────────────────────────────

def test_execution_decision_blocks_on_wait_for_input():
    plan = {"plan_status": "wait_for_input", "planned_steps": []}
    d = execution_decision(plan, "step_05_candidate_context")
    assert d.allow is False
    assert d.plan_status == "wait_for_input"
    assert "wait_for_input" in d.reason


def test_execution_decision_blocks_on_blocked_plan():
    plan = {"plan_status": "blocked", "planned_steps": []}
    d = execution_decision(plan, "step_06_developability")
    assert d.allow is False
    assert d.plan_status == "blocked"


def test_execution_decision_blocks_when_no_plan():
    d = execution_decision(None, "step_05_candidate_context")
    assert d.allow is False
    assert d.plan_status == "missing"


def test_execution_decision_allows_when_ready_and_step_run():
    plan = {
        "plan_status": "ready_to_execute",
        "planned_steps": [
            {"step_id": "step_05_candidate_context", "planned_status": "run",
             "reason": "ok", "required_artifact_refs": [], "lane_flags": {}},
        ],
    }
    d = execution_decision(plan, "step_05_candidate_context")
    assert d.allow is True
    assert d.planned_status == "run"


def test_execution_decision_blocks_per_step_skip_even_when_ready():
    plan = {
        "plan_status": "ready_to_execute",
        "planned_steps": [
            {"step_id": "step_06_developability", "planned_status": "skip",
             "reason": "test", "required_artifact_refs": [], "lane_flags": {}},
        ],
    }
    d = execution_decision(plan, "step_06_developability")
    assert d.allow is False
    assert d.planned_status == "skip"


# ── graph nodes honor the gate ──────────────────────────────────────────────

def test_graph_step5_node_skips_when_plan_status_wait_for_input(
    local_storage, registry_service, workflow_state_service
):
    """Sanity-build a plan via the real services, then tamper with
    plan_status to wait_for_input. The Step 5 node must NOT execute the
    agent — registry stays without candidate_context_table_id, workflow
    state shows step_05 = skipped, and results carries the reason."""
    run_id = _seed_through_step_4(
        local_storage, registry_service, workflow_state_service
    )
    _overwrite_plan_status(local_storage, run_id, "wait_for_input")

    from app.graph.nodes import make_node_step_05

    node = make_node_step_05(
        local_storage, registry_service, workflow_state_service, LocalMCPClient()
    )
    state = node({"run_id": run_id})

    assert state["results"]["step_05"]["executed"] is False
    assert state["results"]["step_05"]["plan_status"] == "wait_for_input"
    assert "wait_for_input" in state["results"]["step_05"]["reason"]

    reg = registry_service.get(run_id)
    assert reg.active_artifacts.candidate_context_table_id is None
    ws = workflow_state_service.get(run_id)
    assert ws["steps"]["step_05"] == "skipped"


def test_graph_step6_node_skips_when_plan_status_blocked(
    local_storage, registry_service, workflow_state_service
):
    run_id = _seed_through_step_4(
        local_storage, registry_service, workflow_state_service
    )
    _overwrite_plan_status(local_storage, run_id, "blocked")

    from app.graph.nodes import make_node_step_06

    node = make_node_step_06(
        local_storage, registry_service, workflow_state_service, LocalMCPClient()
    )
    state = node({"run_id": run_id})

    assert state["results"]["step_06"]["executed"] is False
    assert state["results"]["step_06"]["plan_status"] == "blocked"
    reg = registry_service.get(run_id)
    assert reg.active_artifacts.structured_liability_summary_id is None
    ws = workflow_state_service.get(run_id)
    assert ws["steps"]["step_06"] == "skipped"


# ── Step 5/6 APIs honor the gate ────────────────────────────────────────────

@pytest.fixture
def gated_client(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("STORAGE_MODE", "local")
    monkeypatch.setenv("LOCAL_STORAGE_ROOT", str(tmp_path / "store"))
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    from app.settings import get_settings

    for fn in (
        get_settings,
        deps.get_storage,
        deps.get_registry_service,
        deps.get_workflow_state_service,
        deps.get_tool_inventory_service,
        deps.get_mcp_client,
        deps.get_llm_provider,
    ):
        fn.cache_clear()
    yield TestClient(app)
    for fn in (
        get_settings,
        deps.get_storage,
        deps.get_registry_service,
        deps.get_workflow_state_service,
        deps.get_tool_inventory_service,
        deps.get_mcp_client,
        deps.get_llm_provider,
    ):
        fn.cache_clear()


def _seed_via_client(client: TestClient, *, with_payload=True) -> str:
    body = {
        "raw_user_query": "HER2 ADC",
        "user_provided_context": {
            "target_or_antigen_text": "HER2",
            "candidate_text": "Trastuzumab",
        },
    }
    if with_payload:
        body["user_provided_context"]["payload_linker_text"] = "vc-MMAE"
    resp = client.post("/runs", json=body)
    assert resp.status_code == 200
    run_id = resp.json()["run_id"]

    storage = deps.get_storage()
    reg = deps.get_registry_service()
    ws = deps.get_workflow_state_service()
    StructuredQueryService(storage, reg, ws, SupervisorAgent(llm=MockLLMProvider())).parse(run_id)
    InputReadinessService(storage, reg, ws).check(run_id)
    WorkflowSetupService(storage, reg, ws).plan(run_id)
    return run_id


def test_step5_api_returns_409_when_plan_status_wait_for_input(gated_client: TestClient):
    run_id = _seed_via_client(gated_client, with_payload=False)  # payload absent → wait_for_input

    plan = deps.get_storage().read_json(
        deps.get_storage().run_key(run_id, "inputs/run_step_plan.json")
    )
    assert plan["plan_status"] == "wait_for_input"

    resp = gated_client.post(f"/runs/{run_id}/steps/5/execute")
    assert resp.status_code == 409
    body = resp.json()
    assert body["code"] == "workflow_state_error"
    assert body["detail"]["plan_status"] == "wait_for_input"
    assert body["detail"]["step_id"] == "step_05_candidate_context"
    # No Step 5 artifact landed
    reg = deps.get_registry_service().get(run_id)
    assert reg.active_artifacts.candidate_context_table_id is None


def test_step6_api_returns_409_when_plan_status_wait_for_input(gated_client: TestClient):
    run_id = _seed_via_client(gated_client, with_payload=False)
    resp = gated_client.post(f"/runs/{run_id}/steps/6/execute")
    assert resp.status_code == 409
    body = resp.json()
    assert body["detail"]["plan_status"] == "wait_for_input"
    assert body["detail"]["step_id"] == "step_06_developability"
    reg = deps.get_registry_service().get(run_id)
    assert reg.active_artifacts.structured_liability_summary_id is None


def test_step5_api_returns_409_when_plan_status_blocked(gated_client: TestClient):
    run_id = _seed_via_client(gated_client, with_payload=True)
    storage = deps.get_storage()
    _overwrite_plan_status(storage, run_id, "blocked")
    resp = gated_client.post(f"/runs/{run_id}/steps/5/execute")
    assert resp.status_code == 409
    assert resp.json()["detail"]["plan_status"] == "blocked"


def test_step5_api_runs_when_plan_ready_to_execute(gated_client: TestClient):
    """Happy path: ready_to_execute plan lets Step 5 API run normally."""
    run_id = _seed_via_client(gated_client, with_payload=True)
    plan = deps.get_storage().read_json(
        deps.get_storage().run_key(run_id, "inputs/run_step_plan.json")
    )
    assert plan["plan_status"] == "ready_to_execute"

    resp = gated_client.post(f"/runs/{run_id}/steps/5/execute")
    assert resp.status_code == 200
    body = resp.json()
    assert body["run_id"] == run_id
    assert body["candidate_records"]
    # registry updated
    reg = deps.get_registry_service().get(run_id)
    assert reg.active_artifacts.candidate_context_table_id
