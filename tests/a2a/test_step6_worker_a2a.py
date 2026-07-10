"""Turn C1 — Step 6 worker over the real shared HTTP A2A transport.

The HTTP cases exercise:

    A2AClient -> localhost TCP -> A2AServer/A2AWorkerAdapter
      -> Step6A2AWorker -> DevelopabilityAgent.run_from_artifacts

The LLM and MCP fixtures are deterministic/local to isolate external networks.
They are NOT live LLM/MCP smokes. They still execute the real Step 6 candidate
projection, progressive disclosure, Stage 1/Stage 2 selection, runtime resolver,
MCP call path, normalized persistence, registry update, and workflow-state path.
No production mock/fallback branch is introduced.
"""

from __future__ import annotations

import contextlib
import json
import threading

import pytest
import requests
from werkzeug.serving import make_server

from python_a2a import A2AClient, Message, MessageRole, Task, TaskState, TextContent

from app.a2a.agent_cards import validate_adc_agent_contract
from app.a2a.contracts import (
    A2ATaskMetadata,
    InputArtifactRef,
    InputProjection,
    OrchestratorRoutingDecisionRef,
    PrivacyConstraints,
    WorkerExecutionRequest,
    WorkerRequestSpec,
)
from app.a2a.step6_worker import Step6A2AWorker, create_step6_flask_app
from app.agents.developability_agent import DevelopabilityAgent
from app.llm.provider import MockLLMProvider
from app.mcp.client import LocalMCPClient
from app.services.intake_service import IntakeService
from app.utils.ids import new_artifact_id
from app.utils.time import now_iso

MAIN_THREAD = threading.current_thread().name
_CCT_FIELD_KEYS = ["candidate_records"]


@pytest.fixture(autouse=True)
def _no_proxy(monkeypatch):
    """Test-only localhost proxy isolation; no production environment change."""
    for variable in ("NO_PROXY", "no_proxy"):
        monkeypatch.setenv(variable, "127.0.0.1,localhost")
    for variable in (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "http_proxy",
        "https_proxy",
        "ALL_PROXY",
        "all_proxy",
    ):
        monkeypatch.delenv(variable, raising=False)


def _success_mcp() -> LocalMCPClient:
    """Test-only mocked local success binding, not a live MCP execution.

    This fixture does not demonstrate success from a real ToolUniverse tool.
    Production has no corresponding mock-success fallback.
    """

    def _pains(**_kwargs):
        return {"status": "mocked", "alerts": [], "passes": True}

    return LocalMCPClient(bindings={"DrugProps_pains_filter": _pains})


def _dependency_unavailable_mcp() -> LocalMCPClient:
    """Deterministically exercises LocalMCPClient dependency semantics."""

    def _not_wired(**_kwargs):
        raise NotImplementedError

    return LocalMCPClient(bindings={"DrugProps_pains_filter": _not_wired})


def _candidate(candidate_id: str = "cand_step6_http") -> dict:
    return {
        "candidate_id": candidate_id,
        "candidate_label": "payload fixture",
        "candidate_type": "compound_component",
        "source_records": [],
        "identifiers": [],
        "materials": [
            {
                "material_id": "mat_payload_smiles",
                "material_type": "payload_smiles",
                "value": "CCO",
                "value_format": "smiles",
                "extraction_status": "extracted",
                "validation_status": "valid",
                "role": "payload",
                "role_status": "explicit",
            }
        ],
        "adc_links": {
            "target_material_ids": [],
            "antibody_material_ids": [],
            "payload_material_ids": ["mat_payload_smiles"],
            "linker_material_ids": [],
            "dar_material_ids": [],
        },
        "candidate_status": "partially_ready_for_step6",
        "candidate_notes": None,
        "candidate_role": "user_provided_candidate",
        "is_generated_candidate": False,
        "context_status": "partial",
        "data_gaps": [],
        "missing_material_roles": [],
        "context_notes": [],
    }


def _seed_candidate_context(
    local_storage,
    registry_service,
    workflow_state_service,
    *,
    candidates: list[dict] | None = None,
) -> str:
    record = IntakeService(
        local_storage,
        registry_service,
        workflow_state_service,
    ).submit(
        raw_user_query="Step 6 request-based HTTP fixture",
        user_provided_context={},
    )
    run_id = record.run_id
    artifact_id = new_artifact_id("candidate_context_table")
    body = {
        "artifact_id": artifact_id,
        "run_id": run_id,
        "step_id": "step_05_candidate_context",
        "created_at": now_iso(),
        "context_build_status": "ok",
        "candidate_records": [_candidate()] if candidates is None else candidates,
        "missing_context_flags": [],
        "tool_call_records": [],
        "downstream_query_hints": [],
        "enrichment_selection_audit": {},
    }
    local_storage.write_json(
        local_storage.run_key(run_id, "candidate_context_table.json"),
        body,
    )
    registry_service.update_active(
        run_id,
        candidate_context_table_id=artifact_id,
    )
    # Deliberately no WorkflowSetupService.plan(): request-based Step 6 must not
    # depend on the legacy Step 4 run_step_plan registry pointer.
    assert registry_service.get(run_id).active_artifacts.run_step_plan_id is None
    return run_id


class _RecordingStep6Worker(Step6A2AWorker):
    """HTTP-boundary recorder; delegates all business logic to the real agent."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.execute_threads: list[str] = []
        self.agent_run_count = 0

    def execute_request(self, request):  # type: ignore[override]
        self.execute_threads.append(threading.current_thread().name)
        return super().execute_request(request)

    def _default_agent_factory(self):
        outer = self
        real = DevelopabilityAgent(
            storage=outer._storage,
            registry=outer._registry,
            workflow_state=outer._workflow_state,
            mcp_client=outer._mcp_client,
            llm=outer._llm,
        )

        class _CountingAgent:
            def run_from_artifacts(self, run_id, **kwargs):
                outer.agent_run_count += 1
                return real.run_from_artifacts(run_id, **kwargs)

        return _CountingAgent()


class _IdentityTamperingStep6Worker(Step6A2AWorker):
    """TEST STUB: run the real agent, then corrupt one persisted identity.

    Only the persisted test artifact is changed. The registry pointer and the
    production worker identity-validation behavior are not bypassed or altered.
    """

    def __init__(self, *args, identity_field: str, **kwargs):
        self._identity_field = identity_field
        super().__init__(*args, **kwargs)

    def _default_agent_factory(self):
        outer = self
        real = DevelopabilityAgent(
            storage=outer._storage,
            registry=outer._registry,
            workflow_state=outer._workflow_state,
            mcp_client=outer._mcp_client,
            llm=outer._llm,
        )

        class _TamperingAgent:
            def run_from_artifacts(self, run_id, **kwargs):
                summary = real.run_from_artifacts(run_id, **kwargs)
                key = real.storage.run_key(
                    run_id,
                    "structured_liability_summary.json",
                )
                persisted = real.storage.read_json(key)
                persisted[outer._identity_field] = (
                    f"tampered_{outer._identity_field}_test_only"
                )
                real.storage.write_json(key, persisted)
                return summary

        return _TamperingAgent()


class _ServerHandle:
    def __init__(self, base_url, worker, server, thread):
        self.base_url = base_url
        self.worker = worker
        self._server = server
        self._thread = thread

    def close(self):
        self._server.shutdown()
        self._thread.join(timeout=5)


def _new_worker(
    local_storage,
    registry_service,
    workflow_state_service,
    *,
    mcp_client=None,
) -> _RecordingStep6Worker:
    return _RecordingStep6Worker(
        url="http://step6-worker:8006",
        storage=local_storage,
        registry=registry_service,
        workflow_state=workflow_state_service,
        mcp_client=mcp_client or _success_mcp(),
        llm=MockLLMProvider(),
    )


@contextlib.contextmanager
def _serve(worker):
    app = create_step6_flask_app(worker)
    server = make_server("127.0.0.1", 0, app, threaded=False)
    thread = threading.Thread(
        target=server.serve_forever,
        name="step6-worker-http",
        daemon=True,
    )
    thread.start()
    try:
        yield _ServerHandle(
            f"http://127.0.0.1:{server.server_port}",
            worker,
            server,
            thread,
        )
    finally:
        server.shutdown()
        thread.join(timeout=5)


@pytest.fixture
def worker_server(local_storage, registry_service, workflow_state_service):
    worker = _new_worker(
        local_storage,
        registry_service,
        workflow_state_service,
    )
    with _serve(worker) as handle:
        yield handle


def _artifact_ref(registry_service, run_id: str) -> InputArtifactRef:
    artifact_id = (
        registry_service.get(run_id).active_artifacts.candidate_context_table_id
    )
    return InputArtifactRef(
        artifact_id=artifact_id,
        run_id=run_id,
        artifact_type="candidate_context_table",
        entity_type="candidate",
        selection_mode="all_in_artifact",
        field_keys=list(_CCT_FIELD_KEYS),
        can_read_from_db=True,
    )


def _request(
    run_id: str,
    *,
    refs: dict[str, InputArtifactRef],
) -> WorkerExecutionRequest:
    return WorkerExecutionRequest(
        payload_type="worker_execution_request",
        payload_version="v1",
        run_id=run_id,
        task_id=f"task_step6_{run_id}",
        routing_plan_id="wrp_step6_001",
        routing_decision_id="route_developability_prefiltering",
        agent_id=Step6A2AWorker.AGENT_ID,
        capability_id=Step6A2AWorker.CAPABILITY_ID,
        created_by="step_04_orchestrator_planner",
        worker_request=WorkerRequestSpec(
            objective="Run developability pre-filtering"
        ),
        orchestrator_routing_decision=OrchestratorRoutingDecisionRef(
            planned_status="run",
            dispatch_mode="python_a2a",
            expected_outputs=["structured_liability_summary"],
        ),
        input_projection=InputProjection(
            compact_inputs={"candidate_context_available": True},
            input_artifact_refs=refs,
        ),
        privacy_constraints=PrivacyConstraints(),
    )


def _task(request: WorkerExecutionRequest) -> Task:
    message = Message(
        content=TextContent(text=request.model_dump_json()),
        role=MessageRole.USER,
    )
    metadata = A2ATaskMetadata(
        adc_payload_type="worker_execution_request",
        adc_payload_version="v1",
        run_id=request.run_id,
        task_id=request.task_id,
        routing_plan_id=request.routing_plan_id,
        routing_decision_id=request.routing_decision_id,
        agent_id=request.agent_id,
        capability_id=request.capability_id,
        created_by=request.created_by,
    ).model_dump()
    return Task(
        id=request.task_id,
        message=message.to_dict(),
        metadata=metadata,
    )


def _result(result_task: Task) -> dict:
    artifacts = result_task.artifacts or []
    assert artifacts, "Step 6 result task carried no compact result artifact"
    return json.loads(artifacts[0]["parts"][0]["text"])


async def _send(base_url: str, task: Task) -> Task:
    return await A2AClient(base_url).send_task_async(task)


async def _run_valid(handle, registry_service, run_id: str):
    request = _request(
        run_id,
        refs={"candidate_context_table": _artifact_ref(registry_service, run_id)},
    )
    task = _task(request)
    assert task.id == request.task_id
    result_task = await _send(handle.base_url, task)
    return result_task, _result(result_task)


def _persisted_records(persisted: dict) -> list[dict]:
    return [
        record
        for candidate in persisted["candidate_liability_results"]
        for lane in candidate["lane_results"]
        for record in lane["tool_call_records"]
    ]


def _expected_tool_summary(records: list[dict]) -> dict[str, int]:
    expected = {
        "attempted": 0,
        "success": 0,
        "failed": 0,
        "dependency_unavailable": 0,
        "skipped": 0,
    }
    for record in records:
        status = record["run_status"]
        if status in {"skipped", "not_run"}:
            expected["skipped"] += 1
            continue
        expected["attempted"] += 1
        if status == "success":
            expected["success"] += 1
        elif status == "dependency_unavailable":
            expected["dependency_unavailable"] += 1
        else:
            expected["failed"] += 1
    return expected


def test_health_endpoint(worker_server):
    response = requests.get(f"{worker_server.base_url}/health", timeout=5)
    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "agent_id": "step_06_developability_agent",
        "capabilities": ["step_06_developability"],
    }


def test_agent_card_endpoint(worker_server):
    response = requests.get(f"{worker_server.base_url}/agent-card", timeout=5)
    assert response.status_code == 200
    contract = response.json()["capabilities"]["adc_agent_contract"]
    capability = contract["capabilities"][0]
    assert contract["agent_id"] == "step_06_developability_agent"
    assert [
        ref["artifact_name"] for ref in capability["required_input_artifacts"]
    ] == ["candidate_context_table"]
    assert capability["optional_input_artifacts"] == []
    assert capability["required_artifact_fields"][
        "candidate_context_table"
    ]["required_field_keys"] == ["candidate_records"]
    validate_adc_agent_contract(worker_server.worker.agent_card)


async def test_valid_request_uses_real_http_and_runs_without_run_step_plan(
    worker_server,
    local_storage,
    registry_service,
    workflow_state_service,
):
    run_id = _seed_candidate_context(
        local_storage,
        registry_service,
        workflow_state_service,
    )
    result_task, result = await _run_valid(
        worker_server,
        registry_service,
        run_id,
    )

    assert result_task.status.state == TaskState.COMPLETED
    assert result["result_status"] == "success"
    assert result["execution_status"] == "completed"
    assert worker_server.worker.agent_run_count == 1
    assert worker_server.worker.execute_threads
    assert all(
        thread_name != MAIN_THREAD
        for thread_name in worker_server.worker.execute_threads
    )
    assert any(
        name.startswith("step6-worker-http")
        for name in worker_server.worker.execute_threads
    )
    assert registry_service.get(run_id).active_artifacts.run_step_plan_id is None


@pytest.mark.parametrize(
    "mutate,error_code",
    [
        (lambda ref: ref.model_copy(update={"run_id": "wrong_run"}),
         "artifact_ref_run_id_mismatch"),
        (lambda ref: ref.model_copy(update={"artifact_type": "structured_query"}),
         "artifact_ref_type_mismatch"),
        (lambda ref: ref.model_copy(update={"can_read_from_db": False}),
         "artifact_ref_not_db_readable"),
        (lambda ref: ref.model_copy(update={"artifact_id": "wrong_artifact"}),
         "artifact_ref_id_mismatch"),
        (lambda ref: ref.model_copy(update={"field_keys": []}),
         "artifact_ref_field_keys_missing"),
        (lambda ref: ref.model_copy(update={"entity_type": None}),
         "artifact_ref_entity_type_mismatch"),
        (lambda ref: ref.model_copy(update={"entity_type": "compound"}),
         "artifact_ref_entity_type_mismatch"),
        (lambda ref: ref.model_copy(update={"selection_mode": None}),
         "artifact_ref_selection_mode_unsupported"),
        (lambda ref: ref.model_copy(update={"selection_mode": "selected_entities"}),
         "artifact_ref_selection_mode_unsupported"),
    ],
)
async def test_invalid_candidate_context_ref_does_not_run_agent(
    worker_server,
    local_storage,
    registry_service,
    workflow_state_service,
    mutate,
    error_code,
):
    run_id = _seed_candidate_context(
        local_storage,
        registry_service,
        workflow_state_service,
    )
    ref = mutate(_artifact_ref(registry_service, run_id))
    request = _request(run_id, refs={"candidate_context_table": ref})
    result_task = await _send(worker_server.base_url, _task(request))
    result = _result(result_task)

    assert result_task.status.state == TaskState.FAILED
    assert result["result_status"] == "validation_failed"
    assert result["execution_status"] == "failed"
    assert result["error_code"] == error_code
    assert worker_server.worker.agent_run_count == 0


@pytest.mark.parametrize("identity_field", ["artifact_id", "run_id"])
async def test_persisted_step6_identity_mismatch_is_compact_tool_failure_over_http(
    local_storage,
    registry_service,
    workflow_state_service,
    identity_field,
):
    run_id = _seed_candidate_context(
        local_storage,
        registry_service,
        workflow_state_service,
    )
    worker = _IdentityTamperingStep6Worker(
        url="http://step6-worker:8006",
        storage=local_storage,
        registry=registry_service,
        workflow_state=workflow_state_service,
        mcp_client=_success_mcp(),
        llm=MockLLMProvider(),
        identity_field=identity_field,
    )
    with _serve(worker) as handle:
        request = _request(
            run_id,
            refs={
                "candidate_context_table": _artifact_ref(
                    registry_service,
                    run_id,
                )
            },
        )
        result_task = await _send(handle.base_url, _task(request))
    result = _result(result_task)

    assert result_task.status.state == TaskState.FAILED
    assert result["result_status"] == "tool_failed"
    assert result["execution_status"] == "failed"
    assert result["error_code"] == (
        "structured_liability_artifact_identity_mismatch"
    )
    assert result["output_artifact_refs"] == {}
    assert identity_field in result["error_summary"]
    compact_blob = json.dumps(result).lower()
    assert "tampered_" not in compact_blob
    assert "candidate_liability_results" not in compact_blob


async def test_missing_candidate_context_ref_does_not_run_agent(
    worker_server,
    local_storage,
    registry_service,
    workflow_state_service,
):
    run_id = _seed_candidate_context(
        local_storage,
        registry_service,
        workflow_state_service,
    )
    result_task = await _send(
        worker_server.base_url,
        _task(_request(run_id, refs={})),
    )
    result = _result(result_task)
    assert result_task.status.state == TaskState.FAILED
    assert result["error_code"] == "missing_required_input_artifact_refs"
    assert worker_server.worker.agent_run_count == 0


async def test_candidate_context_body_missing_required_field_does_not_run_agent(
    worker_server,
    local_storage,
    registry_service,
    workflow_state_service,
):
    run_id = _seed_candidate_context(
        local_storage,
        registry_service,
        workflow_state_service,
    )
    key = local_storage.run_key(run_id, "candidate_context_table.json")
    body = local_storage.read_json(key)
    body.pop("candidate_records")
    local_storage.write_json(key, body)

    result_task, result = await _run_valid(
        worker_server,
        registry_service,
        run_id,
    )
    assert result_task.status.state == TaskState.FAILED
    assert result["error_code"] == "artifact_required_fields_missing"
    assert worker_server.worker.agent_run_count == 0


@pytest.mark.parametrize("identity_field", ["artifact_id", "run_id"])
async def test_persisted_candidate_context_identity_mismatch_rejected_over_http(
    worker_server,
    local_storage,
    registry_service,
    workflow_state_service,
    identity_field,
):
    run_id = _seed_candidate_context(
        local_storage,
        registry_service,
        workflow_state_service,
    )
    key = local_storage.run_key(run_id, "candidate_context_table.json")
    persisted = local_storage.read_json(key)
    persisted[identity_field] = f"tampered_test_only_{identity_field}"
    local_storage.write_json(key, persisted)

    request = _request(
        run_id,
        refs={"candidate_context_table": _artifact_ref(registry_service, run_id)},
    )
    result_task = await _send(worker_server.base_url, _task(request))
    result = _result(result_task)

    assert result_task.status.state == TaskState.FAILED
    assert result["result_status"] == "validation_failed"
    assert result["execution_status"] == "failed"
    assert result["error_code"] == "input_artifact_identity_mismatch"
    assert worker_server.worker.agent_run_count == 0
    assert "candidate_context_table" in result["error_summary"]
    assert identity_field in result["error_summary"]
    compact_blob = json.dumps(result).lower()
    assert "tampered_test_only" not in compact_blob
    assert "candidate_records" not in compact_blob


async def test_candidate_context_storage_artifact_missing_does_not_run_agent(
    worker_server,
    local_storage,
    registry_service,
    workflow_state_service,
):
    run_id = _seed_candidate_context(
        local_storage,
        registry_service,
        workflow_state_service,
    )
    local_storage.delete(
        local_storage.run_key(run_id, "candidate_context_table.json")
    )

    result_task, result = await _run_valid(
        worker_server,
        registry_service,
        run_id,
    )
    assert result_task.status.state == TaskState.FAILED
    assert result["error_code"] == "artifact_not_found"
    assert worker_server.worker.agent_run_count == 0


async def test_persisted_artifact_and_compact_result_match(
    worker_server,
    local_storage,
    registry_service,
    workflow_state_service,
):
    run_id = _seed_candidate_context(
        local_storage,
        registry_service,
        workflow_state_service,
    )
    _result_task, result = await _run_valid(
        worker_server,
        registry_service,
        run_id,
    )
    artifact_key = local_storage.run_key(
        run_id,
        "structured_liability_summary.json",
    )
    assert local_storage.exists(artifact_key)
    persisted = local_storage.read_json(artifact_key)
    registry_id = (
        registry_service.get(run_id)
        .active_artifacts.structured_liability_summary_id
    )
    ref = result["output_artifact_refs"]["structured_liability_summary"]
    assert ref["artifact_id"] == registry_id == persisted["artifact_id"]
    assert ref["artifact_type"] == "structured_liability_summary"
    assert ref["storage_key"] == "structured_liability_summary.json"
    assert ref["run_id"] == run_id

    candidates = persisted["candidate_liability_results"]
    lanes = [lane for candidate in candidates for lane in candidate["lane_results"]]
    records = _persisted_records(persisted)
    compact = result["compact_summary"]
    assert compact == {
        "prefilter_status": persisted["prefilter_status"],
        "candidate_count": len(candidates),
        "lane_count": len(lanes),
        "assessed_lane_count": sum(c["assessed_lane_count"] for c in candidates),
        "not_assessed_lane_count": sum(
            c["not_assessed_lane_count"] for c in candidates
        ),
        "missing_input_flags_count": len(persisted["missing_input_flags"]),
        "output_artifact_present": True,
    }
    assert result["tool_call_summary"] == _expected_tool_summary(records)
    assert result["skipped_or_failed_tools"] == sorted(
        {
            record["tool_name"]
            for record in records
            if record["run_status"] != "success"
        }
    )


async def test_completed_with_missing_lanes_is_partial_but_a2a_completed(
    worker_server,
    local_storage,
    registry_service,
    workflow_state_service,
):
    no_material_candidate = _candidate("cand_missing_lanes")
    no_material_candidate["materials"] = []
    no_material_candidate["adc_links"]["payload_material_ids"] = []
    run_id = _seed_candidate_context(
        local_storage,
        registry_service,
        workflow_state_service,
        candidates=[no_material_candidate],
    )
    result_task, result = await _run_valid(
        worker_server,
        registry_service,
        run_id,
    )
    assert result_task.status.state == TaskState.COMPLETED
    assert result["compact_summary"]["prefilter_status"] == (
        "completed_with_missing_lanes"
    )
    assert result["result_status"] == "partial"
    assert result["execution_status"] == "completed"


async def test_dependency_unavailable_maps_to_partial_and_is_auditable(
    local_storage,
    registry_service,
    workflow_state_service,
):
    run_id = _seed_candidate_context(
        local_storage,
        registry_service,
        workflow_state_service,
    )
    worker = _new_worker(
        local_storage,
        registry_service,
        workflow_state_service,
        mcp_client=_dependency_unavailable_mcp(),
    )
    with _serve(worker) as handle:
        result_task, result = await _run_valid(handle, registry_service, run_id)
    persisted = local_storage.read_json(
        local_storage.run_key(run_id, "structured_liability_summary.json")
    )
    records = _persisted_records(persisted)
    unavailable = [
        record
        for record in records
        if record["run_status"] == "dependency_unavailable"
    ]

    assert result_task.status.state == TaskState.COMPLETED
    assert result["compact_summary"]["prefilter_status"] == "partial"
    assert result["result_status"] == "partial"
    assert result["execution_status"] == "completed"
    assert unavailable
    assert result["tool_call_summary"]["dependency_unavailable"] == len(
        unavailable
    )
    assert "DrugProps_pains_filter" in result["skipped_or_failed_tools"]


async def test_failed_prefilter_is_blocked_and_a2a_failed(
    worker_server,
    local_storage,
    registry_service,
    workflow_state_service,
):
    run_id = _seed_candidate_context(
        local_storage,
        registry_service,
        workflow_state_service,
        candidates=[],
    )
    result_task, result = await _run_valid(
        worker_server,
        registry_service,
        run_id,
    )
    persisted = local_storage.read_json(
        local_storage.run_key(run_id, "structured_liability_summary.json")
    )

    assert persisted["prefilter_status"] == "failed"
    assert result_task.status.state == TaskState.FAILED
    assert result["result_status"] == "blocked"
    assert result["execution_status"] == "failed"
    assert result["error_code"] == "developability_prefilter_blocked"
    assert "structured_liability_summary" in result["output_artifact_refs"]
    assert result["compact_summary"]["output_artifact_present"] is True


async def test_compact_result_contains_no_raw_material_or_secrets(
    worker_server,
    local_storage,
    registry_service,
    workflow_state_service,
):
    run_id = _seed_candidate_context(
        local_storage,
        registry_service,
        workflow_state_service,
    )
    _result_task, result = await _run_valid(
        worker_server,
        registry_service,
        run_id,
    )
    blob = json.dumps(result).lower()
    for forbidden in (
        "cco",
        "alerts",
        "raw_sequence",
        "fasta",
        "pdb_body",
        "cif_body",
        "a3m",
        "api_key",
        "raw_tooluniverse_payload",
        "full_prompt",
        "raw_llm_response",
    ):
        assert forbidden not in blob, f"compact result leaked {forbidden!r}"
