"""Turn B — Step 5 worker real HTTP A2A transport.

Transport integration cases exercise the genuine path:

    A2AClient(url).send_task_async(task)
      -> real localhost TCP socket
      -> Flask / python-a2a A2AServer route
      -> A2AWorkerAdapter.handle_task(task)   (generic adapter)
      -> Step5A2AWorker.execute_request(request)  (Step 5 core)
      -> CandidateContextAgent.run_from_artifacts (worker-local execution)

The server runs single-threaded in one named thread on a real EPHEMERAL TCP
port (``make_server("127.0.0.1", 0, ...)``); handler code records the thread it
ran on so the tests prove the call crossed the socket rather than a direct call.

Test-environment isolation (DISCLOSED, not production behavior): this machine
has a macOS system proxy (Aurora listens on 127.0.0.1:29290) that would
otherwise hijack localhost HTTP and yield 502/timeout. The ``_no_proxy`` fixture
sets ``NO_PROXY``/``no_proxy`` and clears ``*_PROXY`` env vars so requests /
A2AClient talk straight to the local TCP server. This changes nothing in
production code.

MCP fixture: ``_mcp()`` uses DETERMINISTIC LOCAL bindings (canned tool outputs)
to isolate the test from external networks while still driving the REAL Step 5
production business path (run_from_artifacts + candidate construction + registry
write). It is NOT a live MCP test and is NOT a mocked success in production code.
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
from app.a2a.step5_worker import Step5A2AWorker, create_step5_flask_app
from app.agents.candidate_context_agent import CandidateContextAgent
from app.agents.supervisor_agent import SupervisorAgent
from app.llm.provider import MockLLMProvider
from app.mcp.client import LocalMCPClient
from app.services.intake_service import IntakeService
from app.services.input_readiness_service import InputReadinessService
from app.services.structured_query_service import StructuredQueryService
from app.services.workflow_setup_service import WorkflowSetupService

MAIN_THREAD = threading.current_thread().name

_RAW_FIELD_KEYS = ["raw_user_query", "user_provided_context", "uploaded_files"]
_SQ_FIELD_KEYS = [
    "mentioned_entities",
    "referenced_inputs",
    "normalized_entities",
    "entity_decompositions",
]


# ── proxy isolation (test-env only, disclosed) ───────────────────────────────
@pytest.fixture(autouse=True)
def _no_proxy(monkeypatch):
    for var in ("NO_PROXY", "no_proxy"):
        monkeypatch.setenv(var, "127.0.0.1,localhost")
    for var in (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "http_proxy",
        "https_proxy",
        "ALL_PROXY",
        "all_proxy",
    ):
        monkeypatch.delenv(var, raising=False)


# ── run setup (production services, no shortcuts) ────────────────────────────
def _setup_run(local_storage, registry_service, workflow_state_service, *, plan: bool = True) -> str:
    intake = IntakeService(local_storage, registry_service, workflow_state_service)
    rec = intake.submit(
        raw_user_query="Design ADC against HER2 with vc-MMAE payload",
        user_provided_context={
            "target_or_antigen_text": "HER2",
            "candidate_text": "Trastuzumab analog",
            "payload_linker_text": "vc-MMAE",
        },
    )
    supervisor = SupervisorAgent(llm=MockLLMProvider())
    StructuredQueryService(
        local_storage, registry_service, workflow_state_service, supervisor
    ).parse(rec.run_id)
    InputReadinessService(local_storage, registry_service, workflow_state_service).check(rec.run_id)
    if plan:
        WorkflowSetupService(local_storage, registry_service, workflow_state_service).plan(rec.run_id)
    return rec.run_id


def _mcp() -> LocalMCPClient:
    """Deterministic LOCAL MCP bindings (NOT live)."""
    def make(payload):
        def _fn(**_kwargs):
            return payload

        return _fn

    canned = {
        "SAbDab_search_structures": {"hits": [{"pdb_id": "1n8z"}]},
        "ChEMBL_search_molecules": {"hits": [{"chembl_id": "CHEMBL1201585"}]},
        "ChEMBL_search_substructure": {"hits": [{"chembl_id": "CHEMBL_linker"}]},
    }
    return LocalMCPClient(bindings={name: make(p) for name, p in canned.items()})


class _RecordingStep5Worker(Step5A2AWorker):
    """Records the thread each execute_request ran on (proof of HTTP boundary)
    and counts real run_from_artifacts invocations (so validation-failure tests
    can assert the domain core was never reached)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.execute_threads: list[str] = []
        self.agent_run_count = 0

    def execute_request(self, request):  # type: ignore[override]
        self.execute_threads.append(threading.current_thread().name)
        return super().execute_request(request)

    def _default_agent_factory(self):
        outer = self
        real = CandidateContextAgent(
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


class _IdentityTamperingStep5Worker(Step5A2AWorker):
    """TEST STUB: run the real agent, then corrupt one persisted identity.

    It changes only the persisted test artifact after genuine agent execution;
    the registry pointer remains untouched and production has no such branch.
    """

    def __init__(self, *args, identity_field: str, **kwargs):
        self._identity_field = identity_field
        super().__init__(*args, **kwargs)

    def _default_agent_factory(self):
        outer = self
        real = CandidateContextAgent(
            storage=outer._storage,
            registry=outer._registry,
            workflow_state=outer._workflow_state,
            mcp_client=outer._mcp_client,
            llm=outer._llm,
        )

        class _TamperingAgent:
            def run_from_artifacts(self, run_id, **kwargs):
                table = real.run_from_artifacts(run_id, **kwargs)
                key = real.storage.run_key(run_id, "candidate_context_table.json")
                persisted = real.storage.read_json(key)
                persisted[outer._identity_field] = (
                    f"tampered_{outer._identity_field}_test_only"
                )
                real.storage.write_json(key, persisted)
                return table

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


@pytest.fixture
def worker_server(local_storage, registry_service, workflow_state_service):
    worker = _RecordingStep5Worker(
        url="http://step5-worker:8005",
        storage=local_storage,
        registry=registry_service,
        workflow_state=workflow_state_service,
        mcp_client=_mcp(),
        llm=MockLLMProvider(),
    )
    app = create_step5_flask_app(worker)
    # OS-assigned ephemeral port (never hardcode 8005; Dify/RAGFlow coexistence).
    # Single-threaded in one named thread so the handler thread is deterministic.
    server = make_server("127.0.0.1", 0, app, threaded=False)
    port = server.server_port
    thread = threading.Thread(target=server.serve_forever, name="step5-worker-http")
    thread.daemon = True
    thread.start()
    handle = _ServerHandle(f"http://127.0.0.1:{port}", worker, server, thread)
    try:
        yield handle
    finally:
        handle.close()


# ── request / task builders ──────────────────────────────────────────────────
def _base_refs(registry_service, run_id) -> dict[str, InputArtifactRef]:
    active = registry_service.get(run_id).active_artifacts
    return {
        "raw_request_record": InputArtifactRef(
            artifact_id=active.raw_request_record_id,
            run_id=run_id,
            artifact_type="raw_request_record",
            field_keys=list(_RAW_FIELD_KEYS),
            can_read_from_db=True,
        ),
        "structured_query": InputArtifactRef(
            artifact_id=active.structured_query_id,
            run_id=run_id,
            artifact_type="structured_query",
            field_keys=list(_SQ_FIELD_KEYS),
            can_read_from_db=True,
        ),
    }


def _request(
    run_id,
    *,
    refs: dict[str, InputArtifactRef],
    agent_id: str = Step5A2AWorker.AGENT_ID,
    capability_id: str = Step5A2AWorker.CAPABILITY_ID,
    privacy: PrivacyConstraints | None = None,
) -> WorkerExecutionRequest:
    return WorkerExecutionRequest(
        payload_type="worker_execution_request",
        payload_version="v1",
        run_id=run_id,
        task_id=f"task_step5_{run_id}",
        routing_plan_id="wrp_001",
        routing_decision_id="route_candidate_context",
        agent_id=agent_id,
        capability_id=capability_id,
        created_by="step_04_orchestrator_planner",
        worker_request=WorkerRequestSpec(objective="Build candidate context"),
        orchestrator_routing_decision=OrchestratorRoutingDecisionRef(
            planned_status="run",
            dispatch_mode="python_a2a",
            expected_outputs=["candidate_context_table"],
        ),
        input_projection=InputProjection(
            compact_inputs={"target_name": "HER2"},
            input_artifact_refs=refs,
        ),
        privacy_constraints=privacy or PrivacyConstraints(),
    )


def _metadata_from_request(request: WorkerExecutionRequest, **overrides) -> dict:
    meta = A2ATaskMetadata(
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
    meta.update(overrides)
    return meta


def _task(
    request: WorkerExecutionRequest,
    *,
    metadata: dict | None = "auto",
    transport_task_id: str | None = None,
) -> Task:
    msg = Message(content=TextContent(text=request.model_dump_json()), role=MessageRole.USER)
    if metadata == "auto":
        metadata = _metadata_from_request(request)
    if metadata is None:
        return Task(id=transport_task_id or request.task_id, message=msg.to_dict())
    return Task(
        id=transport_task_id or request.task_id,
        message=msg.to_dict(),
        metadata=metadata,
    )


def _task_from_raw(body_text: str, metadata: dict) -> Task:
    """Build a Task from an arbitrary (possibly schema-invalid) body string so
    version / malformed-body cases can be exercised past pydantic construction."""
    msg = Message(content=TextContent(text=body_text), role=MessageRole.USER)
    return Task(id=str(metadata["task_id"]), message=msg.to_dict(), metadata=metadata)


@contextlib.contextmanager
def _serve(worker):
    app = create_step5_flask_app(worker)
    server = make_server("127.0.0.1", 0, app, threaded=False)
    thread = threading.Thread(target=server.serve_forever, name="step5-worker-http", daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        thread.join(timeout=5)


def _result(result_task: Task) -> dict:
    artifacts = result_task.artifacts or []
    assert artifacts, "result task carried no artifacts"
    return json.loads(artifacts[0]["parts"][0]["text"])


async def _send(base_url: str, task: Task) -> Task:
    return await A2AClient(base_url).send_task_async(task)


async def _run_valid(worker_server, registry_service, run_id) -> dict:
    refs = _base_refs(registry_service, run_id)
    task = _task(_request(run_id, refs=refs))
    return _result(await _send(worker_server.base_url, task))


# ── 1. health endpoint (real HTTP) ───────────────────────────────────────────
def test_health_endpoint(worker_server):
    resp = requests.get(f"{worker_server.base_url}/health", timeout=5)
    assert resp.status_code == 200
    assert resp.json() == {
        "status": "ok",
        "agent_id": "step_05_candidate_context_agent",
        "capabilities": ["step_05_candidate_context"],
    }


# ── 2. AgentCard endpoint (real HTTP) ────────────────────────────────────────
def test_agent_card_endpoint(worker_server):
    resp = requests.get(f"{worker_server.base_url}/agent-card", timeout=5)
    assert resp.status_code == 200
    card = resp.json()
    assert card["capabilities"]["adc_agent_contract"]["agent_id"] == (
        "step_05_candidate_context_agent"
    )
    validate_adc_agent_contract(worker_server.worker.agent_card)


# ── 3. metadata missing ──────────────────────────────────────────────────────
async def test_metadata_missing_rejected(worker_server, local_storage, registry_service, workflow_state_service):
    run_id = _setup_run(local_storage, registry_service, workflow_state_service)
    task = _task(_request(run_id, refs=_base_refs(registry_service, run_id)), metadata=None)
    result = _result(await _send(worker_server.base_url, task))
    assert result["result_status"] == "validation_failed"
    assert result["error_code"] == "task_metadata_missing"
    assert worker_server.worker.agent_run_count == 0


# ── 4. metadata/body identity mismatch (every identity field) ────────────────
@pytest.mark.parametrize(
    "field, bad_value",
    [
        ("run_id", "run_other"),
        ("task_id", "task_other"),
        ("routing_plan_id", "wrp_other"),
        ("routing_decision_id", "route_other"),
        ("agent_id", "step_06_developability_agent"),
        ("capability_id", "step_06_developability"),
        ("created_by", "someone_else"),
    ],
)
async def test_metadata_identity_mismatch_rejected(
    worker_server, local_storage, registry_service, workflow_state_service, field, bad_value
):
    run_id = _setup_run(local_storage, registry_service, workflow_state_service)
    request = _request(run_id, refs=_base_refs(registry_service, run_id))
    task = _task(request, metadata=_metadata_from_request(request, **{field: bad_value}))
    result = _result(await _send(worker_server.base_url, task))
    assert result["result_status"] == "validation_failed"
    assert result["error_code"] == "task_metadata_body_mismatch"
    assert worker_server.worker.agent_run_count == 0


# ── outer python-a2a Task.id must match the validated ADC task_id ─────────────
async def test_transport_task_id_mismatch_rejected_before_step5_core(
    worker_server,
    local_storage,
    registry_service,
    workflow_state_service,
):
    run_id = _setup_run(local_storage, registry_service, workflow_state_service)
    request = _request(run_id, refs=_base_refs(registry_service, run_id))
    task = _task(request, transport_task_id="wrong_transport_task_id")
    result_task = await _send(worker_server.base_url, task)
    result = _result(result_task)

    assert result_task.status.state == TaskState.FAILED
    assert result["result_status"] == "validation_failed"
    assert result["execution_status"] == "failed"
    assert result["error_code"] == "task_transport_id_mismatch"
    assert worker_server.worker.execute_threads == []
    assert worker_server.worker.agent_run_count == 0
    # The compact failure is correlated to the validated request, not the
    # deliberately wrong outer transport id.
    assert result["run_id"] == request.run_id
    assert result["task_id"] == request.task_id
    assert result["routing_plan_id"] == request.routing_plan_id
    assert result["routing_decision_id"] == request.routing_decision_id
    assert result["agent_id"] == request.agent_id
    assert result["capability_id"] == request.capability_id


# ── 5. wrong agent_id ────────────────────────────────────────────────────────
async def test_wrong_agent_id_rejected(worker_server, local_storage, registry_service, workflow_state_service):
    run_id = _setup_run(local_storage, registry_service, workflow_state_service)
    request = _request(run_id, refs=_base_refs(registry_service, run_id), agent_id="step_06_developability_agent")
    result = _result(await _send(worker_server.base_url, _task(request)))
    assert result["result_status"] == "validation_failed"
    assert result["error_code"] == "agent_id_mismatch"
    assert worker_server.worker.agent_run_count == 0


# ── 6. wrong capability_id ────────────────────────────────────────────────────
async def test_wrong_capability_id_rejected(worker_server, local_storage, registry_service, workflow_state_service):
    run_id = _setup_run(local_storage, registry_service, workflow_state_service)
    request = _request(run_id, refs=_base_refs(registry_service, run_id), capability_id="step_06_developability")
    result = _result(await _send(worker_server.base_url, _task(request)))
    assert result["result_status"] == "validation_failed"
    assert result["error_code"] == "capability_id_mismatch"
    assert worker_server.worker.agent_run_count == 0


# ── 7. missing required input artifact ref ───────────────────────────────────
async def test_missing_required_input_artifact_ref_rejected(worker_server, local_storage, registry_service, workflow_state_service):
    run_id = _setup_run(local_storage, registry_service, workflow_state_service)
    refs = _base_refs(registry_service, run_id)
    del refs["raw_request_record"]
    result = _result(await _send(worker_server.base_url, _task(_request(run_id, refs=refs))))
    assert result["result_status"] == "validation_failed"
    assert result["error_code"] == "missing_required_input_artifact_refs"
    assert worker_server.worker.agent_run_count == 0


# ── 8. ref.run_id mismatch ───────────────────────────────────────────────────
async def test_ref_run_id_mismatch_rejected(worker_server, local_storage, registry_service, workflow_state_service):
    run_id = _setup_run(local_storage, registry_service, workflow_state_service)
    refs = _base_refs(registry_service, run_id)
    refs["structured_query"] = refs["structured_query"].model_copy(update={"run_id": "run_other"})
    result = _result(await _send(worker_server.base_url, _task(_request(run_id, refs=refs))))
    assert result["result_status"] == "validation_failed"
    assert result["error_code"] == "artifact_ref_run_id_mismatch"
    assert worker_server.worker.agent_run_count == 0


# ── 9. ref.artifact_type mismatch ────────────────────────────────────────────
async def test_ref_artifact_type_mismatch_rejected(worker_server, local_storage, registry_service, workflow_state_service):
    run_id = _setup_run(local_storage, registry_service, workflow_state_service)
    refs = _base_refs(registry_service, run_id)
    refs["structured_query"] = refs["structured_query"].model_copy(update={"artifact_type": "run_step_plan"})
    result = _result(await _send(worker_server.base_url, _task(_request(run_id, refs=refs))))
    assert result["result_status"] == "validation_failed"
    assert result["error_code"] == "artifact_ref_type_mismatch"
    assert worker_server.worker.agent_run_count == 0


# ── 10. ref.artifact_id != registry active id ────────────────────────────────
async def test_ref_artifact_id_mismatch_rejected(worker_server, local_storage, registry_service, workflow_state_service):
    run_id = _setup_run(local_storage, registry_service, workflow_state_service)
    refs = _base_refs(registry_service, run_id)
    refs["raw_request_record"] = refs["raw_request_record"].model_copy(update={"artifact_id": "not_the_registry_id"})
    result = _result(await _send(worker_server.base_url, _task(_request(run_id, refs=refs))))
    assert result["result_status"] == "validation_failed"
    assert result["error_code"] == "artifact_ref_id_mismatch"
    assert worker_server.worker.agent_run_count == 0


# ── 11. can_read_from_db=false ───────────────────────────────────────────────
async def test_ref_not_db_readable_rejected(worker_server, local_storage, registry_service, workflow_state_service):
    run_id = _setup_run(local_storage, registry_service, workflow_state_service)
    refs = _base_refs(registry_service, run_id)
    refs["structured_query"] = refs["structured_query"].model_copy(update={"can_read_from_db": False})
    result = _result(await _send(worker_server.base_url, _task(_request(run_id, refs=refs))))
    assert result["result_status"] == "validation_failed"
    assert result["error_code"] == "artifact_ref_not_db_readable"
    assert worker_server.worker.agent_run_count == 0


# ── 12. field_keys missing a required key ────────────────────────────────────
async def test_ref_field_keys_missing_required_rejected(worker_server, local_storage, registry_service, workflow_state_service):
    run_id = _setup_run(local_storage, registry_service, workflow_state_service)
    refs = _base_refs(registry_service, run_id)
    refs["structured_query"] = refs["structured_query"].model_copy(
        update={"field_keys": ["mentioned_entities"]}  # drops the other required keys
    )
    result = _result(await _send(worker_server.base_url, _task(_request(run_id, refs=refs))))
    assert result["result_status"] == "validation_failed"
    assert result["error_code"] == "artifact_ref_field_keys_missing"
    assert worker_server.worker.agent_run_count == 0


# ── 13. storage artifact not found ───────────────────────────────────────────
async def test_storage_artifact_not_found_rejected(worker_server, local_storage, registry_service, workflow_state_service):
    run_id = _setup_run(local_storage, registry_service, workflow_state_service)
    refs = _base_refs(registry_service, run_id)
    # Ref passes every id/type/field check, but the stored body is gone.
    local_storage.delete(local_storage.run_key(run_id, "inputs/structured_query.json"))
    result = _result(await _send(worker_server.base_url, _task(_request(run_id, refs=refs))))
    assert result["result_status"] == "validation_failed"
    assert result["error_code"] == "artifact_not_found"
    assert worker_server.worker.agent_run_count == 0


# ── 14. artifact body missing a required field ───────────────────────────────
async def test_artifact_body_missing_required_field_rejected(worker_server, local_storage, registry_service, workflow_state_service):
    run_id = _setup_run(local_storage, registry_service, workflow_state_service)
    refs = _base_refs(registry_service, run_id)
    key = local_storage.run_key(run_id, "inputs/structured_query.json")
    body = local_storage.read_json(key)
    body.pop("entity_decompositions", None)  # remove a declared required key from the body
    local_storage.write_json(key, body)
    result = _result(await _send(worker_server.base_url, _task(_request(run_id, refs=refs))))
    assert result["result_status"] == "validation_failed"
    assert result["error_code"] == "artifact_required_fields_missing"
    assert worker_server.worker.agent_run_count == 0


@pytest.mark.parametrize(
    "artifact_name,storage_path,identity_field",
    [
        ("raw_request_record", "inputs/raw_request_record.json", "artifact_id"),
        ("raw_request_record", "inputs/raw_request_record.json", "run_id"),
        ("structured_query", "inputs/structured_query.json", "artifact_id"),
        ("structured_query", "inputs/structured_query.json", "run_id"),
    ],
)
async def test_persisted_input_identity_mismatch_rejected_over_real_http(
    worker_server,
    local_storage,
    registry_service,
    workflow_state_service,
    artifact_name,
    storage_path,
    identity_field,
):
    run_id = _setup_run(local_storage, registry_service, workflow_state_service)
    key = local_storage.run_key(run_id, storage_path)
    persisted = local_storage.read_json(key)
    persisted[identity_field] = f"tampered_test_only_{identity_field}"
    local_storage.write_json(key, persisted)

    result_task = await _send(
        worker_server.base_url,
        _task(_request(run_id, refs=_base_refs(registry_service, run_id))),
    )
    result = _result(result_task)

    assert result_task.status.state == TaskState.FAILED
    assert result["result_status"] == "validation_failed"
    assert result["execution_status"] == "failed"
    assert result["error_code"] == "input_artifact_identity_mismatch"
    assert worker_server.worker.agent_run_count == 0
    assert artifact_name in result["error_summary"]
    assert identity_field in result["error_summary"]
    compact_blob = json.dumps(result).lower()
    assert "tampered_test_only" not in compact_blob
    assert "raw_user_query" not in compact_blob


# ── 15. malformed JSON payload ───────────────────────────────────────────────
async def test_malformed_json_payload_rejected(worker_server):
    msg = Message(content=TextContent(text="this is not json {["), role=MessageRole.USER)
    task = Task(message=msg.to_dict(), metadata={"run_id": "x", "task_id": "y"})
    result = _result(await _send(worker_server.base_url, task))
    assert result["result_status"] == "validation_failed"
    assert result["error_code"] in {"malformed_task_json", "malformed_task_no_message_text"}
    assert worker_server.worker.agent_run_count == 0


# ── 16. privacy constraint disabled ──────────────────────────────────────────
async def test_privacy_constraint_disabled_rejected(worker_server, local_storage, registry_service, workflow_state_service):
    run_id = _setup_run(local_storage, registry_service, workflow_state_service)
    request = _request(
        run_id,
        refs=_base_refs(registry_service, run_id),
        privacy=PrivacyConstraints(no_api_keys=False),
    )
    result = _result(await _send(worker_server.base_url, _task(request)))
    assert result["result_status"] == "validation_failed"
    assert result["error_code"] == "privacy_constraints_disabled"
    assert worker_server.worker.agent_run_count == 0


# ── 17. valid task via real A2AClient -> HTTP server -> handler ──────────────
async def test_valid_task_uses_real_http_a2a_path(worker_server, local_storage, registry_service, workflow_state_service):
    run_id = _setup_run(local_storage, registry_service, workflow_state_service)
    result = await _run_valid(worker_server, registry_service, run_id)

    # Handler ran on the werkzeug server thread, not the test's main thread.
    assert worker_server.worker.execute_threads, "execute_request was never invoked"
    assert all(name != MAIN_THREAD for name in worker_server.worker.execute_threads)
    assert any(name.startswith("step5-worker-http") for name in worker_server.worker.execute_threads)
    assert result["result_status"] in {"success", "partial"}
    assert worker_server.worker.agent_run_count == 1


# ── 18. valid request needs NO run_step_plan ref ─────────────────────────────
async def test_valid_request_needs_no_run_step_plan_ref(worker_server, local_storage, registry_service, workflow_state_service):
    run_id = _setup_run(local_storage, registry_service, workflow_state_service)
    refs = _base_refs(registry_service, run_id)
    assert "run_step_plan" not in refs
    result = _result(await _send(worker_server.base_url, _task(_request(run_id, refs=refs))))
    assert result["result_status"] in {"success", "partial"}


# ── 19. request-based core does not depend on registry.run_step_plan_id ──────
async def test_request_core_runs_without_step4_run_step_plan(worker_server, local_storage, registry_service, workflow_state_service):
    # No WorkflowSetupService.plan() -> no run_step_plan artifact / registry id.
    run_id = _setup_run(local_storage, registry_service, workflow_state_service, plan=False)
    assert registry_service.get(run_id).active_artifacts.run_step_plan_id is None

    result = await _run_valid(worker_server, registry_service, run_id)
    assert result["result_status"] in {"success", "partial"}
    assert worker_server.worker.agent_run_count == 1
    assert local_storage.exists(local_storage.run_key(run_id, "candidate_context_table.json"))


# ── 20. legacy run(run_id) keeps its Step4 gate (no regression) ──────────────
def test_legacy_run_still_requires_run_step_plan(local_storage, registry_service, workflow_state_service):
    run_id = _setup_run(local_storage, registry_service, workflow_state_service, plan=False)
    agent = CandidateContextAgent(
        storage=local_storage,
        registry=registry_service,
        workflow_state=workflow_state_service,
        mcp_client=_mcp(),
        llm=MockLLMProvider(),
    )
    with pytest.raises(ValueError, match="run_step_plan"):
        agent.run(run_id)


# ── 21. valid task writes candidate_context_table + matching ref ─────────────
async def test_valid_task_writes_artifact_and_matches_ref(worker_server, local_storage, registry_service, workflow_state_service):
    run_id = _setup_run(local_storage, registry_service, workflow_state_service)
    result = await _run_valid(worker_server, registry_service, run_id)

    assert result["execution_status"] == "completed"
    assert local_storage.exists(local_storage.run_key(run_id, "candidate_context_table.json"))
    artifact_id = registry_service.get(run_id).active_artifacts.candidate_context_table_id
    assert artifact_id

    ref = result["output_artifact_refs"]["candidate_context_table"]
    assert ref["artifact_id"] == artifact_id
    assert ref["artifact_type"] == "candidate_context_table"
    assert ref["storage_key"] == "candidate_context_table.json"
    assert ref["run_id"] == run_id

    cs = result["compact_summary"]
    for key in (
        "context_build_status",
        "candidate_count",
        "tool_call_count",
        "missing_context_flags_count",
        "missing_context_flags",
        "output_artifact_present",
    ):
        assert key in cs
    assert cs["output_artifact_present"] is True

    tcs = result["tool_call_summary"]
    for key in ("attempted", "success", "failed", "dependency_unavailable", "skipped"):
        assert key in tcs


@pytest.mark.parametrize("identity_field", ["artifact_id", "run_id"])
async def test_persisted_step5_identity_mismatch_is_compact_tool_failure_over_http(
    local_storage,
    registry_service,
    workflow_state_service,
    identity_field,
):
    run_id = _setup_run(
        local_storage,
        registry_service,
        workflow_state_service,
    )
    worker = _IdentityTamperingStep5Worker(
        url="http://step5-worker:8005",
        storage=local_storage,
        registry=registry_service,
        workflow_state=workflow_state_service,
        mcp_client=_mcp(),
        llm=MockLLMProvider(),
        identity_field=identity_field,
    )
    with _serve(worker) as base_url:
        request = _request(
            run_id,
            refs=_base_refs(registry_service, run_id),
        )
        result_task = await _send(base_url, _task(request))
    result = _result(result_task)

    assert result_task.status.state == TaskState.FAILED
    assert result["result_status"] == "tool_failed"
    assert result["execution_status"] == "failed"
    assert result["error_code"] == (
        "candidate_context_artifact_identity_mismatch"
    )
    assert result["output_artifact_refs"] == {}
    assert identity_field in result["error_summary"]
    compact_blob = json.dumps(result).lower()
    assert "tampered_" not in compact_blob
    assert "candidate_records" not in compact_blob


# ── 22. compact result carries no raw material / secrets ─────────────────────
async def test_result_has_no_raw_material_or_secrets(worker_server, local_storage, registry_service, workflow_state_service):
    run_id = _setup_run(local_storage, registry_service, workflow_state_service)
    result = await _run_valid(worker_server, registry_service, run_id)
    blob = json.dumps(result).lower()
    for needle in (
        "hits",  # raw MCP tool payload key from the canned bindings
        "api_key",
        "authorization",
        "bearer",
        "raw_sequence",
        "fasta",
        "pdb_body",
        "cif_body",
        "a3m",
        "full_prompt",
        "raw_llm_response",
        "tooluniverse_payload",
    ):
        assert needle not in blob, f"result leaked forbidden token: {needle}"


# ── 23. cross-check the real candidate_context_table vs the compact result ───
async def test_compact_result_matches_real_candidate_context_table(worker_server, local_storage, registry_service, workflow_state_service):
    run_id = _setup_run(local_storage, registry_service, workflow_state_service)
    result = await _run_valid(worker_server, registry_service, run_id)

    persisted = local_storage.read_json(
        local_storage.run_key(run_id, "candidate_context_table.json")
    )
    records = persisted["tool_call_records"]

    # candidate count + build status agree with the persisted artifact.
    assert result["compact_summary"]["candidate_count"] == len(persisted["candidate_records"])
    assert result["compact_summary"]["context_build_status"] == persisted["context_build_status"]
    assert result["compact_summary"]["tool_call_count"] == len(records)

    # tool-call status distribution recomputed from the real records matches.
    def _bucket(status: str) -> str:
        if status in {"skipped", "not_run"}:
            return "skipped"
        if status == "success":
            return "success"
        if status == "dependency_unavailable":
            return "dependency_unavailable"
        return "failed"

    expected = {"attempted": 0, "success": 0, "failed": 0, "dependency_unavailable": 0, "skipped": 0}
    for rec in records:
        bucket = _bucket(rec["run_status"])
        expected[bucket] += 1
        if bucket != "skipped":
            expected["attempted"] += 1
    assert result["tool_call_summary"] == expected

    # skipped_or_failed_tools == the non-success tool names in the artifact.
    non_success = sorted({r["tool_name"] for r in records if r["run_status"] != "success"})
    assert result["skipped_or_failed_tools"] == non_success

    # This deterministic-binding run should resolve all tool calls successfully.
    assert expected["attempted"] >= 1
    assert expected["success"] == expected["attempted"]
    assert expected["failed"] == 0
    assert expected["dependency_unavailable"] == 0
    assert result["result_status"] == "success"


# ── payload version is locked to v1 (no silent normalize) ────────────────────
@pytest.mark.parametrize(
    "missing_field",
    ["payload_type", "payload_version", "privacy_constraints"],
)
async def test_required_request_markers_cannot_be_defaulted_on_wire(
    worker_server,
    local_storage,
    registry_service,
    workflow_state_service,
    missing_field,
):
    run_id = _setup_run(local_storage, registry_service, workflow_state_service)
    request = _request(run_id, refs=_base_refs(registry_service, run_id))
    body = request.model_dump()
    body.pop(missing_field)
    task = _task_from_raw(json.dumps(body), _metadata_from_request(request))
    result = _result(await _send(worker_server.base_url, task))
    assert result["result_status"] == "validation_failed"
    assert result["error_code"] == "request_schema_invalid"
    assert worker_server.worker.agent_run_count == 0


@pytest.mark.parametrize("missing_field", ["adc_payload_type", "adc_payload_version"])
async def test_required_metadata_markers_cannot_be_defaulted_on_wire(
    worker_server,
    local_storage,
    registry_service,
    workflow_state_service,
    missing_field,
):
    run_id = _setup_run(local_storage, registry_service, workflow_state_service)
    request = _request(run_id, refs=_base_refs(registry_service, run_id))
    metadata = _metadata_from_request(request)
    metadata.pop(missing_field)
    result = _result(
        await _send(
            worker_server.base_url,
            _task_from_raw(request.model_dump_json(), metadata),
        )
    )
    assert result["result_status"] == "validation_failed"
    assert result["error_code"] == "task_metadata_invalid"
    assert worker_server.worker.agent_run_count == 0


async def test_request_payload_version_v2_rejected(worker_server, local_storage, registry_service, workflow_state_service):
    run_id = _setup_run(local_storage, registry_service, workflow_state_service)
    request = _request(run_id, refs=_base_refs(registry_service, run_id))
    body = request.model_dump()
    body["payload_version"] = "v2"  # unsupported
    task = _task_from_raw(json.dumps(body), _metadata_from_request(request))
    result = _result(await _send(worker_server.base_url, task))
    assert result["result_status"] == "validation_failed"
    assert result["error_code"] == "request_schema_invalid"
    assert worker_server.worker.agent_run_count == 0


async def test_metadata_payload_version_v2_rejected(worker_server, local_storage, registry_service, workflow_state_service):
    run_id = _setup_run(local_storage, registry_service, workflow_state_service)
    request = _request(run_id, refs=_base_refs(registry_service, run_id))
    meta = _metadata_from_request(request, adc_payload_version="v2")  # unsupported
    task = _task_from_raw(request.model_dump_json(), meta)
    result = _result(await _send(worker_server.base_url, task))
    assert result["result_status"] == "validation_failed"
    assert result["error_code"] == "task_metadata_invalid"
    assert worker_server.worker.agent_run_count == 0


async def test_inconsistent_versions_rejected(worker_server, local_storage, registry_service, workflow_state_service):
    # Both Literal["v1"] fields are the primary gate: a body claiming v2 while
    # metadata is v1 is caught (as request_schema_invalid) and never executed.
    run_id = _setup_run(local_storage, registry_service, workflow_state_service)
    request = _request(run_id, refs=_base_refs(registry_service, run_id))
    body = request.model_dump()
    body["payload_version"] = "v2"
    task = _task_from_raw(json.dumps(body), _metadata_from_request(request))  # metadata v1
    result = _result(await _send(worker_server.base_url, task))
    assert result["result_status"] == "validation_failed"
    assert result["error_code"] == "request_schema_invalid"
    assert worker_server.worker.agent_run_count == 0


# ── malformed body but valid metadata -> full metadata correlation ───────────
async def test_malformed_body_with_valid_metadata_keeps_correlation(worker_server, local_storage, registry_service, workflow_state_service):
    run_id = _setup_run(local_storage, registry_service, workflow_state_service)
    request = _request(run_id, refs=_base_refs(registry_service, run_id))
    meta = _metadata_from_request(request)  # valid v1 metadata
    task = _task_from_raw("this is not json {[", meta)
    result = _result(await _send(worker_server.base_url, task))

    assert result["result_status"] == "validation_failed"
    assert result["error_code"] in {"malformed_task_json", "malformed_task_no_message_text"}
    # Correlation identity comes from the VALID metadata, not python-a2a Task.id.
    assert result["run_id"] == request.run_id
    assert result["task_id"] == request.task_id
    assert result["routing_plan_id"] == request.routing_plan_id
    assert result["routing_decision_id"] == request.routing_decision_id
    assert result["agent_id"] == request.agent_id
    assert result["capability_id"] == request.capability_id
    assert worker_server.worker.agent_run_count == 0


# ── context_build_status=failed maps to an A2A FAILED / tool_failed result ───
class _FailedBuildStep5Worker(Step5A2AWorker):
    """TEST STUB (disclosed): wraps the REAL agent — which truly persists the
    candidate_context_table + registry pointer — then synchronizes ONLY the
    returned/persisted context_build_status to 'failed' so this test can exercise
    honest failure-status mapping. It does NOT fake success, bypass persistence,
    or change the production CandidateContextAgent status calculation."""

    def _default_agent_factory(self):
        real = CandidateContextAgent(
            storage=self._storage,
            registry=self._registry,
            workflow_state=self._workflow_state,
            mcp_client=self._mcp_client,
            llm=self._llm,
        )

        class _FailedAgent:
            def run_from_artifacts(self, run_id, **kwargs):
                table = real.run_from_artifacts(run_id, **kwargs)
                failed_table = table.model_copy(update={"context_build_status": "failed"})

                # TEST-ONLY failure-status mapping stub: the real agent already
                # wrote the artifact and registry pointer. Keep that persisted
                # source of truth aligned with the returned table before the
                # worker builds its compact failed result.
                artifact_key = real.storage.run_key(run_id, "candidate_context_table.json")
                persisted = real.storage.read_json(artifact_key)
                persisted["context_build_status"] = "failed"
                real.storage.write_json(artifact_key, persisted)
                return failed_table

        return _FailedAgent()


async def test_failed_build_is_not_a2a_completed(local_storage, registry_service, workflow_state_service):
    run_id = _setup_run(local_storage, registry_service, workflow_state_service)
    worker = _FailedBuildStep5Worker(
        url="http://step5-worker:8005",
        storage=local_storage,
        registry=registry_service,
        workflow_state=workflow_state_service,
        mcp_client=_mcp(),
        llm=MockLLMProvider(),
    )
    with _serve(worker) as base_url:
        refs = _base_refs(registry_service, run_id)
        result_task = await _send(base_url, _task(_request(run_id, refs=refs)))
    result = _result(result_task)
    artifact_key = local_storage.run_key(run_id, "candidate_context_table.json")
    persisted = local_storage.read_json(artifact_key)
    records = persisted["tool_call_records"]

    assert result["result_status"] == "tool_failed"
    assert result["execution_status"] == "failed"
    assert result["error_code"] == "candidate_context_build_failed"
    # A2A task status is FAILED, never COMPLETED.
    assert result_task.status.state == TaskState.FAILED
    # Artifact was really persisted, so the compact ref/summary are kept for audit.
    assert local_storage.exists(artifact_key)
    assert "candidate_context_table" in result["output_artifact_refs"]
    assert persisted["context_build_status"] == "failed"
    assert result["compact_summary"]["context_build_status"] == "failed"
    assert result["compact_summary"]["candidate_count"] == len(
        persisted["candidate_records"]
    )
    assert result["compact_summary"]["tool_call_count"] == len(records)

    expected_tool_summary = {
        "attempted": 0,
        "success": 0,
        "failed": 0,
        "dependency_unavailable": 0,
        "skipped": 0,
    }
    for record in records:
        status = record["run_status"]
        if status in {"skipped", "not_run"}:
            expected_tool_summary["skipped"] += 1
            continue
        expected_tool_summary["attempted"] += 1
        if status == "success":
            expected_tool_summary["success"] += 1
        elif status == "dependency_unavailable":
            expected_tool_summary["dependency_unavailable"] += 1
        else:
            expected_tool_summary["failed"] += 1
    assert result["tool_call_summary"] == expected_tool_summary
    assert result["skipped_or_failed_tools"] == sorted(
        {record["tool_name"] for record in records if record["run_status"] != "success"}
    )
