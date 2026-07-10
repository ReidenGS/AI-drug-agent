"""Turn C2 Structure worker integration over the real HTTP A2A transport.

Every dispatch case crosses ``A2AClient -> localhost TCP -> A2AServer``. The
LLM is ``MockLLMProvider`` and MCP bindings are deterministic, test-only local
fixtures: this is not a live LLM/MCP smoke, mocked tool success does not prove
real ToolUniverse success, and production contains no mock-success fallback.
The production Step 7/8/9 projection, routing, runtime resolution, persistence,
registry, and workflow-state implementations remain in the exercised path.
"""

from __future__ import annotations

import contextlib
import json
import threading

import pytest
import requests
from werkzeug.serving import make_server

from python_a2a import A2AClient, Message, MessageRole, Task, TaskState, TextContent

from app.a2a.agent_cards import (
    CAP_STRUCTURE_DESIGN_WORKFLOW,
    STEP_07_STRUCTURE_INPUT,
    STEP_08_STRUCTURE_EVALUATION,
    STEP_09_STRUCTURE_DESIGN,
    validate_adc_agent_contract,
)
from app.a2a.contracts import (
    A2ATaskMetadata,
    InputArtifactRef,
    InputProjection,
    OrchestratorRoutingDecisionRef,
    PrivacyConstraints,
    WorkerExecutionRequest,
    WorkerRequestSpec,
)
from app.a2a.structure_worker import (
    StructureA2AWorker,
    create_structure_flask_app,
)
from app.agents.structure_and_design_agent import StructureAndDesignAgent
from app.agents.supervisor_agent import SupervisorAgent
from app.llm.provider import MockLLMProvider
from app.mcp.client import LocalMCPClient
from app.services.intake_service import IntakeService
from app.services.structured_query_service import StructuredQueryService
from app.utils.ids import new_artifact_id
from app.utils.time import now_iso
from app.utils.errors import WorkflowStateError


MAIN_THREAD = threading.current_thread().name
_ORDER = [
    STEP_07_STRUCTURE_INPUT,
    STEP_08_STRUCTURE_EVALUATION,
    STEP_09_STRUCTURE_DESIGN,
]
_FIELD_KEYS = {
    "raw_request_record": [
        "raw_user_query",
        "user_provided_context",
        "uploaded_files",
    ],
    "structured_query": [
        "task_intent",
        "referenced_inputs",
        "requested_outputs",
        "user_constraints",
        "normalized_entities",
        "canonical_query",
    ],
    "candidate_context_table": [
        "candidate_records",
        "downstream_query_hints",
    ],
}
_OUTPUTS = [
    (
        "prepared_structure_input_package",
        "prepared_structure_input_package.json",
        "prepared_structure_input_package_id",
    ),
    (
        "structure_prediction_and_interface_results",
        "structure_prediction_and_interface_results.json",
        "structure_prediction_and_interface_results_id",
    ),
    (
        "structure_variant_and_compound_screening",
        "compound_screening_artifact.json",
        "structure_variant_and_compound_screening_id",
    ),
]


@pytest.fixture(autouse=True)
def _no_proxy(monkeypatch):
    """Test-only localhost proxy isolation; no production environment gate."""
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


def _mocked_local_mcp() -> LocalMCPClient:
    """Test-only mocked local binding; not a live ToolUniverse success."""

    def _entry(**kwargs):
        return {"status": "mocked", "pdb_id": kwargs.get("pdb_id")}

    return LocalMCPClient(bindings={"RCSBData_get_entry": _entry})


def _auditable_local_mcp() -> LocalMCPClient:
    """Test-only bindings producing honest success/dependency/failed records."""

    def _success(**kwargs):
        return {"status": "mocked", "pdb_id": kwargs.get("pdb_id")}

    def _dependency(**_kwargs):
        raise NotImplementedError

    def _failed(**_kwargs):
        raise RuntimeError("test-only upstream failure")

    return LocalMCPClient(
        bindings={
            "RCSBData_get_entry": _success,
            "RCSBData_get_assembly": _dependency,
            "get_refinement_resolution_by_pdb_id": _success,
            "PDBePISA_get_interfaces": _failed,
        }
    )


def _candidate(*, known_pdb: bool = False) -> dict:
    if known_pdb:
        materials = []
        identifiers = [{"id_type": "pdb_id", "id_value": "1N8Z"}]
    else:
        materials = [
            {
                "material_id": "mat_target_sequence",
                "material_type": "target_sequence",
                "value": "MKTAYIAKQNNVG",
                "role": "target",
            }
        ]
        identifiers = []
    return {
        "candidate_id": "cand_structure_http",
        "candidate_label": "HER2 test target",
        "candidate_type": "target_antigen",
        "source_records": [],
        "identifiers": identifiers,
        "materials": materials,
        "adc_links": {},
        "candidate_status": "partially_ready_for_step6",
        "candidate_role": "user_provided_candidate",
        "is_generated_candidate": False,
        "context_status": "partial",
        "data_gaps": [],
        "missing_material_roles": [],
        "context_notes": [],
    }


def _seed_run(
    local_storage,
    registry_service,
    workflow_state_service,
    *,
    known_pdb: bool = False,
) -> str:
    record = IntakeService(
        local_storage,
        registry_service,
        workflow_state_service,
    ).submit(
        raw_user_query="Prepare HER2 structure workflow",
        user_provided_context={"target_or_antigen_text": "HER2"},
    )
    run_id = record.run_id
    StructuredQueryService(
        local_storage,
        registry_service,
        workflow_state_service,
        SupervisorAgent(llm=MockLLMProvider()),
    ).parse(run_id)

    artifact_id = new_artifact_id("candidate_context_table")
    local_storage.write_json(
        local_storage.run_key(run_id, "candidate_context_table.json"),
        {
            "artifact_id": artifact_id,
            "run_id": run_id,
            "step_id": "step_05_candidate_context",
            "created_at": now_iso(),
            "context_build_status": "ok",
            "candidate_records": [_candidate(known_pdb=known_pdb)],
            "missing_context_flags": [],
            "tool_call_records": [],
            "downstream_query_hints": [],
            "enrichment_selection_audit": {},
        },
    )
    registry_service.update_active(
        run_id,
        candidate_context_table_id=artifact_id,
    )
    assert registry_service.get(run_id).active_artifacts.run_step_plan_id is None
    return run_id


class _RecordingStructureWorker(StructureA2AWorker):
    """Observe the real workflow without replacing any production step."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.execute_threads: list[str] = []
        self.agent_run_count = 0
        self.internal_events: list[str] = []

    def execute_request(self, request):  # type: ignore[override]
        self.execute_threads.append(threading.current_thread().name)
        return super().execute_request(request)

    def _default_agent_factory(self):
        outer = self

        class _ObservedAgent(StructureAndDesignAgent):
            def run_workflow_from_artifacts(self, run_id, **kwargs):
                outer.agent_run_count += 1
                return super().run_workflow_from_artifacts(run_id, **kwargs)

            def _run_step_7_from_artifacts(self, run_id, **kwargs):
                outer.internal_events.append(STEP_07_STRUCTURE_INPUT)
                return super()._run_step_7_from_artifacts(run_id, **kwargs)

            def run_step_8(self, run_id):
                assert outer._storage.exists(
                    outer._storage.run_key(
                        run_id,
                        "prepared_structure_input_package.json",
                    )
                )
                outer.internal_events.append(STEP_08_STRUCTURE_EVALUATION)
                return super().run_step_8(run_id)

            def run_step_9(self, run_id):
                assert outer._storage.exists(
                    outer._storage.run_key(
                        run_id,
                        "structure_prediction_and_interface_results.json",
                    )
                )
                outer.internal_events.append(STEP_09_STRUCTURE_DESIGN)
                return super().run_step_9(run_id)

        return _ObservedAgent(
            storage=outer._storage,
            registry=outer._registry,
            workflow_state=outer._workflow_state,
            mcp_client=outer._mcp_client,
            llm=outer._llm,
        )


class _IdentityTamperingStructureWorker(StructureA2AWorker):
    """TEST STUB: run all real steps, then corrupt one persisted identity."""

    def __init__(self, *args, output_name: str, identity_field: str, **kwargs):
        self._output_name = output_name
        self._identity_field = identity_field
        super().__init__(*args, **kwargs)

    def _default_agent_factory(self):
        outer = self
        real = StructureAndDesignAgent(
            storage=outer._storage,
            registry=outer._registry,
            workflow_state=outer._workflow_state,
            mcp_client=outer._mcp_client,
            llm=outer._llm,
        )

        class _TamperingAgent:
            def run_workflow_from_artifacts(self, run_id, **kwargs):
                results = real.run_workflow_from_artifacts(run_id, **kwargs)
                path = next(
                    path for name, path, _field in _OUTPUTS if name == outer._output_name
                )
                key = real.storage.run_key(run_id, path)
                persisted = real.storage.read_json(key)
                persisted[outer._identity_field] = (
                    f"tampered_test_only_{outer._identity_field}"
                )
                real.storage.write_json(key, persisted)
                return results

        return _TamperingAgent()


class _StatusMappingStructureWorker(StructureA2AWorker):
    """TEST STUB: run the real workflow, then set persisted status fields.

    This isolates compact status mapping only. It does not bypass any internal
    step, persistence operation, registry update, LLM route, or MCP route.
    """

    def __init__(self, *args, status_updates: dict[str, tuple[str, str]], **kwargs):
        self._status_updates = status_updates
        super().__init__(*args, **kwargs)

    def _default_agent_factory(self):
        outer = self
        real = StructureAndDesignAgent(
            storage=outer._storage,
            registry=outer._registry,
            workflow_state=outer._workflow_state,
            mcp_client=outer._mcp_client,
            llm=outer._llm,
        )

        class _StatusAgent:
            def run_workflow_from_artifacts(self, run_id, **kwargs):
                results = real.run_workflow_from_artifacts(run_id, **kwargs)
                for path, (field, value) in outer._status_updates.items():
                    key = real.storage.run_key(run_id, path)
                    persisted = real.storage.read_json(key)
                    persisted[field] = value
                    real.storage.write_json(key, persisted)
                return results

        return _StatusAgent()


class _MissingInternalArtifactStructureWorker(StructureA2AWorker):
    """TEST STUB: delete one real internal output at a workflow boundary."""

    def __init__(self, *args, missing_step: str, **kwargs):
        self._missing_step = missing_step
        self.step8_called = False
        self.step9_called = False
        super().__init__(*args, **kwargs)

    def _default_agent_factory(self):
        outer = self

        class _MissingArtifactAgent(StructureAndDesignAgent):
            def _run_step_7_from_artifacts(self, run_id, **kwargs):
                result = super()._run_step_7_from_artifacts(run_id, **kwargs)
                if outer._missing_step == STEP_07_STRUCTURE_INPUT:
                    self.storage.delete(
                        self.storage.run_key(
                            run_id,
                            "prepared_structure_input_package.json",
                        )
                    )
                return result

            def run_step_8(self, run_id):
                outer.step8_called = True
                result = super().run_step_8(run_id)
                if outer._missing_step == STEP_08_STRUCTURE_EVALUATION:
                    self.storage.delete(
                        self.storage.run_key(
                            run_id,
                            "structure_prediction_and_interface_results.json",
                        )
                    )
                return result

            def run_step_9(self, run_id):
                outer.step9_called = True
                return super().run_step_9(run_id)

        return _MissingArtifactAgent(
            storage=outer._storage,
            registry=outer._registry,
            workflow_state=outer._workflow_state,
            mcp_client=outer._mcp_client,
            llm=outer._llm,
        )


class _BoundaryIdentityTamperingStructureWorker(StructureA2AWorker):
    """TEST STUB: corrupt one identity after a real step, before its boundary."""

    def __init__(self, *args, boundary_step: str, identity_field: str, **kwargs):
        self._boundary_step = boundary_step
        self._identity_field = identity_field
        self.step8_called = False
        self.step9_called = False
        super().__init__(*args, **kwargs)

    def _default_agent_factory(self):
        outer = self

        class _BoundaryTamperingAgent(StructureAndDesignAgent):
            def _tamper(self, run_id, path):
                key = self.storage.run_key(run_id, path)
                persisted = self.storage.read_json(key)
                persisted[outer._identity_field] = (
                    f"tampered_boundary_test_only_{outer._identity_field}"
                )
                self.storage.write_json(key, persisted)

            def _run_step_7_from_artifacts(self, run_id, **kwargs):
                result = super()._run_step_7_from_artifacts(run_id, **kwargs)
                if outer._boundary_step == STEP_07_STRUCTURE_INPUT:
                    self._tamper(run_id, "prepared_structure_input_package.json")
                return result

            def run_step_8(self, run_id):
                outer.step8_called = True
                result = super().run_step_8(run_id)
                if outer._boundary_step == STEP_08_STRUCTURE_EVALUATION:
                    self._tamper(
                        run_id,
                        "structure_prediction_and_interface_results.json",
                    )
                return result

            def run_step_9(self, run_id):
                outer.step9_called = True
                return super().run_step_9(run_id)

        return _BoundaryTamperingAgent(
            storage=outer._storage,
            registry=outer._registry,
            workflow_state=outer._workflow_state,
            mcp_client=outer._mcp_client,
            llm=outer._llm,
        )


class _BlockedStructureWorker(StructureA2AWorker):
    """TEST STUB: run all real steps, then mark persisted Step 7 failed."""

    def _default_agent_factory(self):
        real = StructureAndDesignAgent(
            storage=self._storage,
            registry=self._registry,
            workflow_state=self._workflow_state,
            mcp_client=self._mcp_client,
            llm=self._llm,
        )

        class _FailedStep7Agent:
            def run_workflow_from_artifacts(self, run_id, **kwargs):
                results = real.run_workflow_from_artifacts(run_id, **kwargs)
                key = real.storage.run_key(
                    run_id,
                    "prepared_structure_input_package.json",
                )
                persisted = real.storage.read_json(key)
                persisted["structure_preparation_status"] = "failed"
                real.storage.write_json(key, persisted)
                return results

        return _FailedStep7Agent()


class _ServerHandle:
    def __init__(self, base_url, worker):
        self.base_url = base_url
        self.worker = worker


def _new_worker(
    local_storage,
    registry_service,
    workflow_state_service,
    *,
    mcp_client=None,
):
    return _RecordingStructureWorker(
        url="http://structure-worker:8009",
        storage=local_storage,
        registry=registry_service,
        workflow_state=workflow_state_service,
        mcp_client=mcp_client or _mocked_local_mcp(),
        llm=MockLLMProvider(),
    )


@contextlib.contextmanager
def _serve(worker):
    app = create_structure_flask_app(worker)
    server = make_server("127.0.0.1", 0, app, threaded=False)
    thread = threading.Thread(
        target=server.serve_forever,
        name="structure-worker-http",
        daemon=True,
    )
    thread.start()
    try:
        yield _ServerHandle(f"http://127.0.0.1:{server.server_port}", worker)
    finally:
        server.shutdown()
        thread.join(timeout=5)


@pytest.fixture
def worker_server(local_storage, registry_service, workflow_state_service):
    with _serve(
        _new_worker(
            local_storage,
            registry_service,
            workflow_state_service,
        )
    ) as handle:
        yield handle


def _refs(registry_service, run_id: str) -> dict[str, InputArtifactRef]:
    active = registry_service.get(run_id).active_artifacts
    refs = {
        name: InputArtifactRef(
            artifact_id=getattr(active, f"{name}_id"),
            run_id=run_id,
            artifact_type=name,
            field_keys=list(fields),
            can_read_from_db=True,
        )
        for name, fields in _FIELD_KEYS.items()
    }
    refs["candidate_context_table"] = refs["candidate_context_table"].model_copy(
        update={
            "entity_type": "candidate",
            "selection_mode": "all_in_artifact",
        }
    )
    return refs


def _request(
    run_id: str,
    *,
    refs: dict[str, InputArtifactRef],
) -> WorkerExecutionRequest:
    return WorkerExecutionRequest(
        payload_type="worker_execution_request",
        payload_version="v1",
        run_id=run_id,
        task_id=f"task_structure_{run_id}",
        routing_plan_id="wrp_structure_001",
        routing_decision_id="route_structure_design_workflow",
        agent_id=StructureA2AWorker.AGENT_ID,
        capability_id=StructureA2AWorker.CAPABILITY_ID,
        created_by="step_04_orchestrator_planner",
        worker_request=WorkerRequestSpec(
            objective="Run the complete structure design workflow"
        ),
        orchestrator_routing_decision=OrchestratorRoutingDecisionRef(
            planned_status="run",
            dispatch_mode="python_a2a",
            expected_outputs=[name for name, _path, _field in _OUTPUTS],
        ),
        input_projection=InputProjection(
            compact_inputs={"structure_workflow_requested": True},
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
    return Task(id=request.task_id, message=message.to_dict(), metadata=metadata)


def _result(result_task: Task) -> dict:
    artifacts = result_task.artifacts or []
    assert artifacts, "Structure task returned no compact result artifact"
    return json.loads(artifacts[0]["parts"][0]["text"])


async def _send(base_url: str, task: Task) -> Task:
    return await A2AClient(base_url).send_task_async(task)


async def _run_valid(handle, registry_service, run_id: str):
    request = _request(run_id, refs=_refs(registry_service, run_id))
    task = _task(request)
    assert task.id == request.task_id
    result_task = await _send(handle.base_url, task)
    return result_task, _result(result_task)


def _persisted_outputs(local_storage, run_id: str) -> dict[str, dict]:
    return {
        name: local_storage.read_json(local_storage.run_key(run_id, path))
        for name, path, _field in _OUTPUTS
    }


def _records(outputs: dict[str, dict]) -> list[dict]:
    return [
        *outputs["prepared_structure_input_package"][
            "structure_tool_call_records"
        ],
        *outputs["structure_prediction_and_interface_results"][
            "tool_call_records"
        ],
        *outputs["structure_variant_and_compound_screening"][
            "tool_call_records"
        ],
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
        "agent_id": "structure_and_design_agent",
        "capabilities": ["structure_design_workflow"],
    }


def test_agent_card_exposes_one_workflow_capability(worker_server):
    response = requests.get(f"{worker_server.base_url}/agent-card", timeout=5)
    assert response.status_code == 200
    card = response.json()
    contract = card["capabilities"]["adc_agent_contract"]
    assert [cap["capability_id"] for cap in contract["capabilities"]] == [
        CAP_STRUCTURE_DESIGN_WORKFLOW
    ]
    assert [skill["id"] for skill in card["skills"]] == [
        CAP_STRUCTURE_DESIGN_WORKFLOW
    ]
    validate_adc_agent_contract(worker_server.worker.agent_card)


async def test_real_http_runs_one_task_in_strict_order_without_run_step_plan(
    worker_server,
    local_storage,
    registry_service,
    workflow_state_service,
):
    run_id = _seed_run(
        local_storage,
        registry_service,
        workflow_state_service,
    )
    result_task, result = await _run_valid(worker_server, registry_service, run_id)

    assert result_task.status.state == TaskState.COMPLETED
    assert result["result_status"] in {"success", "partial"}
    assert worker_server.worker.agent_run_count == 1
    assert worker_server.worker.internal_events == _ORDER
    assert result["compact_summary"]["internal_execution_order"] == _ORDER
    assert result["compact_summary"]["completed_internal_steps"] == _ORDER
    assert registry_service.get(run_id).active_artifacts.run_step_plan_id is None
    assert worker_server.worker.execute_threads
    assert all(name != MAIN_THREAD for name in worker_server.worker.execute_threads)
    assert any(
        name.startswith("structure-worker-http")
        for name in worker_server.worker.execute_threads
    )


@pytest.mark.parametrize("missing_name", list(_FIELD_KEYS))
async def test_missing_required_ref_does_not_run_agent(
    worker_server,
    local_storage,
    registry_service,
    workflow_state_service,
    missing_name,
):
    run_id = _seed_run(local_storage, registry_service, workflow_state_service)
    refs = _refs(registry_service, run_id)
    refs.pop(missing_name)
    result_task = await _send(
        worker_server.base_url,
        _task(_request(run_id, refs=refs)),
    )
    result = _result(result_task)
    assert result_task.status.state == TaskState.FAILED
    assert result["error_code"] == "missing_required_input_artifact_refs"
    assert worker_server.worker.agent_run_count == 0


@pytest.mark.parametrize(
    "artifact_name,updates,error_code",
    [
        ("raw_request_record", {"run_id": "wrong"}, "artifact_ref_run_id_mismatch"),
        ("structured_query", {"artifact_type": "wrong"}, "artifact_ref_type_mismatch"),
        ("raw_request_record", {"artifact_id": "wrong"}, "artifact_ref_id_mismatch"),
        ("structured_query", {"can_read_from_db": False}, "artifact_ref_not_db_readable"),
        ("candidate_context_table", {"field_keys": []}, "artifact_ref_field_keys_missing"),
    ],
)
async def test_invalid_required_ref_does_not_run_agent(
    worker_server,
    local_storage,
    registry_service,
    workflow_state_service,
    artifact_name,
    updates,
    error_code,
):
    run_id = _seed_run(local_storage, registry_service, workflow_state_service)
    refs = _refs(registry_service, run_id)
    refs[artifact_name] = refs[artifact_name].model_copy(update=updates)
    result_task = await _send(
        worker_server.base_url,
        _task(_request(run_id, refs=refs)),
    )
    result = _result(result_task)
    assert result_task.status.state == TaskState.FAILED
    assert result["result_status"] == "validation_failed"
    assert result["error_code"] == error_code
    assert worker_server.worker.agent_run_count == 0


@pytest.mark.parametrize(
    "updates,error_code",
    [
        ({"entity_type": None}, "artifact_ref_entity_type_mismatch"),
        ({"entity_type": "compound"}, "artifact_ref_entity_type_mismatch"),
        ({"selection_mode": None}, "artifact_ref_selection_mode_unsupported"),
        (
            {"selection_mode": "selected_entities"},
            "artifact_ref_selection_mode_unsupported",
        ),
    ],
)
async def test_candidate_selection_contract_fails_closed_before_agent(
    worker_server,
    local_storage,
    registry_service,
    workflow_state_service,
    updates,
    error_code,
):
    run_id = _seed_run(local_storage, registry_service, workflow_state_service)
    refs = _refs(registry_service, run_id)
    refs["candidate_context_table"] = refs[
        "candidate_context_table"
    ].model_copy(update=updates)
    result_task = await _send(
        worker_server.base_url,
        _task(_request(run_id, refs=refs)),
    )
    result = _result(result_task)
    assert result_task.status.state == TaskState.FAILED
    assert result["execution_status"] == "failed"
    assert result["error_code"] == error_code
    assert worker_server.worker.agent_run_count == 0


@pytest.mark.parametrize(
    "artifact_name,storage_path,identity_field",
    [
        ("raw_request_record", "inputs/raw_request_record.json", "artifact_id"),
        ("raw_request_record", "inputs/raw_request_record.json", "run_id"),
        ("structured_query", "inputs/structured_query.json", "artifact_id"),
        ("structured_query", "inputs/structured_query.json", "run_id"),
        ("candidate_context_table", "candidate_context_table.json", "artifact_id"),
        ("candidate_context_table", "candidate_context_table.json", "run_id"),
    ],
)
async def test_required_input_body_identity_mismatch_fails_before_agent(
    worker_server,
    local_storage,
    registry_service,
    workflow_state_service,
    artifact_name,
    storage_path,
    identity_field,
):
    run_id = _seed_run(local_storage, registry_service, workflow_state_service)
    key = local_storage.run_key(run_id, storage_path)
    persisted = local_storage.read_json(key)
    persisted[identity_field] = f"tampered_test_only_{identity_field}"
    local_storage.write_json(key, persisted)

    result_task = await _send(
        worker_server.base_url,
        _task(_request(run_id, refs=_refs(registry_service, run_id))),
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
    assert "candidate_records" not in compact_blob


async def test_optional_liability_ref_is_not_required(
    worker_server,
    local_storage,
    registry_service,
    workflow_state_service,
):
    run_id = _seed_run(local_storage, registry_service, workflow_state_service)
    refs = _refs(registry_service, run_id)
    assert "structured_liability_summary" not in refs
    result_task = await _send(
        worker_server.base_url,
        _task(_request(run_id, refs=refs)),
    )
    assert result_task.status.state == TaskState.COMPLETED


def test_legacy_step7_still_requires_run_step_plan(
    local_storage,
    registry_service,
    workflow_state_service,
):
    run_id = _seed_run(local_storage, registry_service, workflow_state_service)
    agent = StructureAndDesignAgent(
        storage=local_storage,
        registry=registry_service,
        workflow_state=workflow_state_service,
        mcp_client=_mocked_local_mcp(),
        llm=MockLLMProvider(),
    )
    with pytest.raises(WorkflowStateError, match="run_step_plan"):
        agent.run_step_7(run_id)


async def test_provided_optional_liability_ref_must_match_registry(
    worker_server,
    local_storage,
    registry_service,
    workflow_state_service,
):
    run_id = _seed_run(local_storage, registry_service, workflow_state_service)
    refs = _refs(registry_service, run_id)
    refs["structured_liability_summary"] = InputArtifactRef(
        artifact_id="not_active",
        run_id=run_id,
        artifact_type="structured_liability_summary",
        can_read_from_db=True,
    )
    result_task = await _send(
        worker_server.base_url,
        _task(_request(run_id, refs=refs)),
    )
    result = _result(result_task)
    assert result_task.status.state == TaskState.FAILED
    assert result["error_code"] == "artifact_ref_id_mismatch"
    assert worker_server.worker.agent_run_count == 0


@pytest.mark.parametrize("identity_field", ["artifact_id", "run_id"])
async def test_optional_liability_body_identity_mismatch_fails_before_agent(
    worker_server,
    local_storage,
    registry_service,
    workflow_state_service,
    identity_field,
):
    run_id = _seed_run(local_storage, registry_service, workflow_state_service)
    artifact_id = new_artifact_id("structured_liability_summary")
    key = local_storage.run_key(run_id, "structured_liability_summary.json")
    local_storage.write_json(
        key,
        {
            "artifact_id": artifact_id,
            "run_id": run_id,
            "prefilter_status": "completed",
        },
    )
    registry_service.update_active(
        run_id,
        structured_liability_summary_id=artifact_id,
    )
    refs = _refs(registry_service, run_id)
    refs["structured_liability_summary"] = InputArtifactRef(
        artifact_id=artifact_id,
        run_id=run_id,
        artifact_type="structured_liability_summary",
        can_read_from_db=True,
    )
    persisted = local_storage.read_json(key)
    persisted[identity_field] = f"tampered_test_only_{identity_field}"
    local_storage.write_json(key, persisted)

    result_task = await _send(
        worker_server.base_url,
        _task(_request(run_id, refs=refs)),
    )
    result = _result(result_task)

    assert result_task.status.state == TaskState.FAILED
    assert result["result_status"] == "validation_failed"
    assert result["execution_status"] == "failed"
    assert result["error_code"] == "input_artifact_identity_mismatch"
    assert worker_server.worker.agent_run_count == 0
    assert "structured_liability_summary" in result["error_summary"]
    assert identity_field in result["error_summary"]
    assert "tampered_test_only" not in json.dumps(result).lower()


async def test_three_persisted_artifacts_refs_counts_and_tools_reconcile(
    worker_server,
    local_storage,
    registry_service,
    workflow_state_service,
):
    run_id = _seed_run(local_storage, registry_service, workflow_state_service)
    _result_task, result = await _run_valid(worker_server, registry_service, run_id)
    outputs = _persisted_outputs(local_storage, run_id)
    active = registry_service.get(run_id).active_artifacts

    assert list(result["output_artifact_refs"]) == [name for name, _p, _f in _OUTPUTS]
    for name, path, registry_field in _OUTPUTS:
        persisted = outputs[name]
        ref = result["output_artifact_refs"][name]
        registry_id = getattr(active, registry_field)
        assert persisted["artifact_id"] == registry_id == ref["artifact_id"]
        assert persisted["run_id"] == run_id == ref["run_id"]
        assert ref["storage_key"] == path

    step7 = outputs["prepared_structure_input_package"]
    step8 = outputs["structure_prediction_and_interface_results"]
    step9 = outputs["structure_variant_and_compound_screening"]
    compact = result["compact_summary"]
    assert compact == {
        "internal_execution_order": _ORDER,
        "completed_internal_steps": _ORDER,
        "step7_status": step7["structure_preparation_status"],
        "step7_prepared_input_count": len(step7["prepared_structure_inputs"]),
        "step7_unresolved_resource_count": len(step7["unresolved_resource_refs"]),
        "step7_preparation_warning_count": len(step7["preparation_warnings"]),
        "step8_status": step8["structure_modeling_status"],
        "step8_candidate_result_count": len(step8["candidate_structure_results"]),
        "step8_output_artifact_count": len(step8["output_artifacts"]),
        "step9_status": step9["screening_status"],
        "step9_stage1_selected_tool_count": len(step9["step9_stage1_selected_tools"]),
        "step9_stage2_mapped_tool_count": len(step9["step9_stage2_mapped_tools"]),
        "step9_executed_tool_count": len(step9["step9_runtime_executed_tools"]),
        "output_artifact_count": 3,
    }
    records = _records(outputs)
    assert result["tool_call_summary"] == _expected_tool_summary(records)
    assert result["skipped_or_failed_tools"] == sorted(
        {
            record["tool_name"]
            for record in records
            if record["run_status"] != "success"
        }
    )


async def test_all_non_success_tool_statuses_remain_auditable(
    local_storage,
    registry_service,
    workflow_state_service,
):
    run_id = _seed_run(
        local_storage,
        registry_service,
        workflow_state_service,
        known_pdb=True,
    )
    worker = _new_worker(
        local_storage,
        registry_service,
        workflow_state_service,
        mcp_client=_auditable_local_mcp(),
    )
    with _serve(worker) as handle:
        result_task, result = await _run_valid(handle, registry_service, run_id)
    outputs = _persisted_outputs(local_storage, run_id)
    records = _records(outputs)
    statuses = {record["run_status"] for record in records}

    assert result_task.status.state == TaskState.COMPLETED
    assert {"success", "failed", "dependency_unavailable", "skipped"} <= statuses
    assert result["tool_call_summary"] == _expected_tool_summary(records)
    assert result["tool_call_summary"]["failed"] >= 1
    assert result["tool_call_summary"]["dependency_unavailable"] >= 1
    assert result["tool_call_summary"]["skipped"] >= 1


async def test_step9_skipped_is_neutral_not_a2a_failure(
    worker_server,
    local_storage,
    registry_service,
    workflow_state_service,
):
    run_id = _seed_run(local_storage, registry_service, workflow_state_service)
    result_task, result = await _run_valid(worker_server, registry_service, run_id)
    assert result["compact_summary"]["step9_status"] == "skipped"
    assert result_task.status.state == TaskState.COMPLETED
    assert result["execution_status"] == "completed"
    assert result["result_status"] in {"success", "partial"}


async def test_clean_ok_ok_skipped_mapping_is_success(
    local_storage,
    registry_service,
    workflow_state_service,
):
    run_id = _seed_run(local_storage, registry_service, workflow_state_service)
    worker = _StatusMappingStructureWorker(
        url="http://structure-worker:8009",
        storage=local_storage,
        registry=registry_service,
        workflow_state=workflow_state_service,
        mcp_client=_mocked_local_mcp(),
        llm=MockLLMProvider(),
        status_updates={
            "prepared_structure_input_package.json": (
                "structure_preparation_status",
                "ok",
            ),
            "structure_prediction_and_interface_results.json": (
                "structure_modeling_status",
                "ok",
            ),
            "compound_screening_artifact.json": ("screening_status", "skipped"),
        },
    )
    with _serve(worker) as handle:
        result_task, result = await _run_valid(handle, registry_service, run_id)
    assert result_task.status.state == TaskState.COMPLETED
    assert result["result_status"] == "success"
    assert result["execution_status"] == "completed"
    assert result["error_code"] is None


@pytest.mark.parametrize(
    "path,field",
    [
        (
            "structure_prediction_and_interface_results.json",
            "structure_modeling_status",
        ),
        ("compound_screening_artifact.json", "screening_status"),
    ],
)
async def test_downstream_failed_with_upstream_artifacts_maps_partial(
    local_storage,
    registry_service,
    workflow_state_service,
    path,
    field,
):
    run_id = _seed_run(local_storage, registry_service, workflow_state_service)
    worker = _StatusMappingStructureWorker(
        url="http://structure-worker:8009",
        storage=local_storage,
        registry=registry_service,
        workflow_state=workflow_state_service,
        mcp_client=_mocked_local_mcp(),
        llm=MockLLMProvider(),
        status_updates={
            "prepared_structure_input_package.json": (
                "structure_preparation_status",
                "ok",
            ),
            path: (field, "failed"),
        },
    )
    with _serve(worker) as handle:
        result_task, result = await _run_valid(handle, registry_service, run_id)
    assert result_task.status.state == TaskState.COMPLETED
    assert result["result_status"] == "partial"
    assert result["execution_status"] == "completed"
    assert len(result["output_artifact_refs"]) == 3


async def test_step7_failed_is_blocked_after_all_three_audit_artifacts(
    local_storage,
    registry_service,
    workflow_state_service,
):
    run_id = _seed_run(local_storage, registry_service, workflow_state_service)
    worker = _BlockedStructureWorker(
        url="http://structure-worker:8009",
        storage=local_storage,
        registry=registry_service,
        workflow_state=workflow_state_service,
        mcp_client=_mocked_local_mcp(),
        llm=MockLLMProvider(),
    )
    with _serve(worker) as handle:
        result_task, result = await _run_valid(handle, registry_service, run_id)

    assert result_task.status.state == TaskState.FAILED
    assert result["result_status"] == "blocked"
    assert result["execution_status"] == "failed"
    assert result["error_code"] == "structure_workflow_blocked"
    assert list(result["output_artifact_refs"]) == [name for name, _p, _f in _OUTPUTS]
    assert all(
        local_storage.exists(local_storage.run_key(run_id, path))
        for _name, path, _field in _OUTPUTS
    )


@pytest.mark.parametrize(
    "missing_step,step8_called,step9_called",
    [
        (STEP_07_STRUCTURE_INPUT, False, False),
        (STEP_08_STRUCTURE_EVALUATION, True, False),
    ],
)
async def test_missing_internal_artifact_stops_downstream_steps(
    local_storage,
    registry_service,
    workflow_state_service,
    missing_step,
    step8_called,
    step9_called,
):
    run_id = _seed_run(local_storage, registry_service, workflow_state_service)
    worker = _MissingInternalArtifactStructureWorker(
        url="http://structure-worker:8009",
        storage=local_storage,
        registry=registry_service,
        workflow_state=workflow_state_service,
        mcp_client=_mocked_local_mcp(),
        llm=MockLLMProvider(),
        missing_step=missing_step,
    )
    with _serve(worker) as handle:
        result_task, result = await _run_valid(handle, registry_service, run_id)

    assert result_task.status.state == TaskState.FAILED
    assert result["result_status"] == "tool_failed"
    assert result["execution_status"] == "failed"
    assert result["error_code"] == "worker_execution_error"
    assert result["output_artifact_refs"] == {}
    assert worker.step8_called is step8_called
    assert worker.step9_called is step9_called


@pytest.mark.parametrize(
    "boundary_step,identity_field,step8_called,step9_called",
    [
        (STEP_07_STRUCTURE_INPUT, "artifact_id", False, False),
        (STEP_07_STRUCTURE_INPUT, "run_id", False, False),
        (STEP_08_STRUCTURE_EVALUATION, "artifact_id", True, False),
        (STEP_08_STRUCTURE_EVALUATION, "run_id", True, False),
    ],
)
async def test_internal_boundary_identity_corruption_stops_downstream_over_http(
    local_storage,
    registry_service,
    workflow_state_service,
    boundary_step,
    identity_field,
    step8_called,
    step9_called,
):
    run_id = _seed_run(local_storage, registry_service, workflow_state_service)
    worker = _BoundaryIdentityTamperingStructureWorker(
        url="http://structure-worker:8009",
        storage=local_storage,
        registry=registry_service,
        workflow_state=workflow_state_service,
        mcp_client=_mocked_local_mcp(),
        llm=MockLLMProvider(),
        boundary_step=boundary_step,
        identity_field=identity_field,
    )
    with _serve(worker) as handle:
        result_task, result = await _run_valid(handle, registry_service, run_id)

    assert result_task.status.state == TaskState.FAILED
    assert result["result_status"] == "tool_failed"
    assert result["execution_status"] == "failed"
    assert result["error_code"] == "worker_execution_error"
    assert result["output_artifact_refs"] == {}
    assert worker.step8_called is step8_called
    assert worker.step9_called is step9_called
    compact_blob = json.dumps(result).lower()
    assert "tampered_boundary_test_only" not in compact_blob
    assert "prepared_structure_inputs" not in compact_blob


@pytest.mark.parametrize(
    "output_name,identity_field",
    [
        (name, identity_field)
        for name, _path, _registry_field in _OUTPUTS
        for identity_field in ("artifact_id", "run_id")
    ],
)
async def test_output_identity_tampering_is_compact_tool_failure_over_http(
    local_storage,
    registry_service,
    workflow_state_service,
    output_name,
    identity_field,
):
    run_id = _seed_run(local_storage, registry_service, workflow_state_service)
    worker = _IdentityTamperingStructureWorker(
        url="http://structure-worker:8009",
        storage=local_storage,
        registry=registry_service,
        workflow_state=workflow_state_service,
        mcp_client=_mocked_local_mcp(),
        llm=MockLLMProvider(),
        output_name=output_name,
        identity_field=identity_field,
    )
    with _serve(worker) as handle:
        result_task, result = await _run_valid(handle, registry_service, run_id)

    assert result_task.status.state == TaskState.FAILED
    assert result["result_status"] == "tool_failed"
    assert result["execution_status"] == "failed"
    assert result["error_code"] == "structure_workflow_artifact_identity_mismatch"
    assert result["output_artifact_refs"] == {}
    compact_blob = json.dumps(result).lower()
    assert "tampered_test_only_" not in compact_blob
    assert "prepared_structure_inputs" not in compact_blob
    assert "candidate_structure_results" not in compact_blob


async def test_compact_result_contains_no_raw_material_or_secrets(
    worker_server,
    local_storage,
    registry_service,
    workflow_state_service,
):
    run_id = _seed_run(local_storage, registry_service, workflow_state_service)
    _result_task, result = await _run_valid(worker_server, registry_service, run_id)
    blob = json.dumps(result).lower()
    for forbidden in (
        "mktayiakqnnvg",
        "raw_sequence",
        "fasta_body",
        "pdb_body",
        "cif_body",
        "a3m_body",
        "api_key",
        "raw_tooluniverse_payload",
        "full_prompt",
        "raw_llm_response",
        "status\": \"mocked",
    ):
        assert forbidden not in blob
