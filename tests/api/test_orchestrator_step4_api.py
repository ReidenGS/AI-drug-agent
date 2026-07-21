"""Turn G production Step 4 application/lifespan API integration tests.

The synthetic localhost A2AServer is a transport fixture only. It is not a
live LLM, MCP, ToolUniverse, or biomedical-worker success claim.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

import pytest
from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import InMemorySaver

from app.a2a.orchestrator_application_service import (
    OrchestratorApplicationService,
)
from app.a2a.orchestrator_discovery import DispatchTarget
from app.a2a.orchestrator_discovery import (
    ExpectedWorkerEndpoint,
    WorkerDiscoveryService,
)
from app.a2a.agent_cards import (
    AGENT_ID_PATENT_EVIDENCE,
    AGENT_ID_STEP5,
    AGENT_ID_STEP6,
    AGENT_ID_STRUCTURE,
    CAP_PATENT_EVIDENCE_WORKFLOW,
    CAP_STEP5_CANDIDATE_CONTEXT,
    CAP_STEP6_DEVELOPABILITY,
    CAP_STRUCTURE_DESIGN_WORKFLOW,
)
from app.a2a.orchestrator_routing_service import OrchestratorRoutingService
from app.agents.supervisor_agent import SupervisorAgent
from app.graph.orchestrator_execution_graph import (
    build_orchestrator_execution_graph,
    execution_graph_config,
)
from app.graph.orchestrator_checkpoint_runtime import (
    OrchestratorCheckpointRuntimeError,
)
from app.main import create_app
from app.llm.provider import MockLLMProvider
from app.mcp.client import LocalMCPClient
from app.services.input_readiness_service import InputReadinessService
from app.services.intake_service import IntakeService
from app.services.structured_query_service import StructuredQueryService
from app.settings import get_settings
from app.utils.ids import new_file_id
from tests.a2a.test_orchestrator_dispatch import _serve_task_handler
from tests.a2a.test_orchestrator_dispatch import _RecordingStep5, _local_mcp
from tests.a2a.test_parallel_worker_http_smoke import _independent_services, _serve
from tests.a2a.test_patent_evidence_worker_a2a import (
    _RecordingWorker as _RecordingPatentEvidenceWorker,
    _bindings as _patent_evidence_bindings,
)
from tests.a2a.test_step6_worker_a2a import _RecordingStep6Worker, _success_mcp
from tests.a2a.test_structure_worker_a2a import (
    _RecordingStructureWorker,
    _auditable_local_mcp,
)
from app.a2a.step5_worker import create_step5_flask_app
from app.a2a.step6_worker import create_step6_flask_app
from app.a2a.structure_worker import create_structure_flask_app
from app.a2a.patent_evidence_worker import create_patent_evidence_flask_app
from tests.a2a.test_orchestrator_post_ingestion import (
    RUN_ID,
    _contract,
    _DeterministicLLM,
    _FrozenDiscovery,
    _proposal,
    _seed_inputs,
)
from tests.a2a.test_orchestrator_retry_loop import _card, _SyntheticWorker
from tests.a2a.test_orchestrator_routing_service import (
    _seed_inputs as _seed_routing_inputs,
    _service as _routing_service_fixture,
)


@pytest.fixture(autouse=True)
def _localhost_proxy_isolation(monkeypatch):
    for name in ("NO_PROXY", "no_proxy"):
        monkeypatch.setenv(name, "127.0.0.1,localhost")
    for name in (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "http_proxy",
        "https_proxy",
        "ALL_PROXY",
        "all_proxy",
    ):
        monkeypatch.delenv(name, raising=False)


class _MemoryRuntime:
    def __init__(self):
        self.saver = InMemorySaver()
        self.graph = build_orchestrator_execution_graph(checkpointer=self.saver)
        self.startup_count = 0
        self.shutdown_count = 0

    async def startup(self):
        self.startup_count += 1
        return self

    async def shutdown(self):
        self.shutdown_count += 1

    @asynccontextmanager
    async def run_lock(self, _run_id):
        yield


class _StartupFailureRuntime:
    async def startup(self):
        raise RuntimeError("postgresql://private:secret@unavailable/raw")

    async def shutdown(self):
        raise AssertionError("never started")


class _CountingApplicationService:
    """Expose the real readiness gate while counting forbidden execute calls."""

    def __init__(self, delegate):
        self.delegate = delegate
        self.execute_count = 0

    def ensure_input_readiness_ready(self, run_id):
        return self.delegate.ensure_input_readiness_ready(run_id)

    async def execute(self, run_id):
        self.execute_count += 1
        return await self.delegate.execute(run_id)


def _configure_lifespan(monkeypatch):
    monkeypatch.setenv(
        "LANGGRAPH_CHECKPOINT_DATABASE_URL",
        "postgresql://test_only:private@checkpoint.invalid/adc",
    )
    monkeypatch.setenv("ORCHESTRATOR_WORKER_TIMEOUT_SECONDS", "2")
    get_settings.cache_clear()


def test_missing_dsn_keeps_health_but_step4_fails_without_memory_fallback(
    monkeypatch,
):
    monkeypatch.delenv("LANGGRAPH_CHECKPOINT_DATABASE_URL", raising=False)
    monkeypatch.setenv("ORCHESTRATOR_WORKER_TIMEOUT_SECONDS", "2")
    get_settings.cache_clear()
    runtime_factory_calls = []
    app = create_app(
        checkpoint_runtime_factory=lambda settings: runtime_factory_calls.append(
            settings
        ),
    )

    with TestClient(app) as client:
        assert client.get("/healthz").status_code == 200
        response = client.post(f"/runs/{RUN_ID}/steps/4/execute")
        invalid = client.get(
            "/runs/sk-live-RAW-PATH-SECRET/steps/4/status"
        )

    assert response.status_code == 503
    assert response.json()["error_code"] == (
        "orchestrator_checkpoint_database_url_required"
    )
    assert response.json()["outcome"] == "unavailable"
    assert runtime_factory_calls == []
    assert invalid.status_code == 422
    assert invalid.json()["run_id"] is None
    assert invalid.json()["error_code"] == "orchestrator_request_invalid"
    assert "sk-live" not in invalid.text
    assert "RAW-PATH-SECRET" not in invalid.text


def test_configured_postgres_startup_failure_aborts_lifespan_without_fallback(
    monkeypatch,
):
    _configure_lifespan(monkeypatch)
    app = create_app(
        checkpoint_runtime_factory=lambda _settings: _StartupFailureRuntime()
    )
    with pytest.raises(
        OrchestratorCheckpointRuntimeError,
        match="^checkpoint_runtime_startup_failed$",
    ) as caught:
        with TestClient(app):
            pass
    assert "private" not in repr(caught.value)
    assert "secret" not in str(caught.value)


def test_not_ready_execute_is_409_before_orchestrator_side_effects(
    monkeypatch, local_storage, registry_service
):
    _configure_lifespan(monkeypatch)
    run_ids = {
        "needs_user_input": "run_20260715_abcde001",
        "blocked": "run_20260715_abcde002",
    }
    for readiness, run_id in run_ids.items():
        _seed_routing_inputs(
            local_storage,
            registry_service,
            run_id=run_id,
            readiness=readiness,
        )
    routing_service, llm, discovery = _routing_service_fixture(
        local_storage, registry_service
    )
    runtime = _MemoryRuntime()
    production_service = OrchestratorApplicationService(
        checkpoint_runtime=runtime,
        routing_service=routing_service,
        discovery=discovery,
        registry=registry_service,
        storage=local_storage,
        worker_timeout_seconds=2,
        max_worker_retries=3,
    )
    counting = _CountingApplicationService(production_service)
    app = create_app(
        checkpoint_runtime_factory=lambda _settings: runtime,
        orchestrator_service_factory=lambda _runtime: counting,
    )

    with TestClient(app) as client:
        responses = {
            status: client.post(f"/runs/{run_id}/steps/4/execute")
            for status, run_id in run_ids.items()
        }

    assert {response.status_code for response in responses.values()} == {409}
    assert {
        response.json()["error_code"] for response in responses.values()
    } == {"input_readiness_not_ready"}
    assert counting.execute_count == 0
    assert llm.call_count == discovery.discover_count == 0
    assert list(runtime.saver.list(None)) == []
    for run_id in run_ids.values():
        active = registry_service.get(run_id).active_artifacts
        assert active.worker_discovery_snapshot_id is None
        assert active.worker_routing_plan_id is None
        assert active.worker_routing_plan_control_id is None
        assert active.candidate_context_table_id is None
        assert active.structured_liability_summary_id is None
        assert active.prepared_structure_input_package_id is None
        assert not local_storage.exists(
            local_storage.run_key(run_id, "worker_routing_plan.json")
        )


@pytest.mark.parametrize(
    ("id_type", "value"),
    [
        ("target_sequence", "ACDE?FG"),
        ("uniprot_id", "not-a-uniprot-accession"),
    ],
)
def test_invalid_typed_structure_input_is_blocked_before_step4_or_a2a(
    monkeypatch,
    local_storage,
    registry_service,
    workflow_state_service,
    id_type,
    value,
):
    _configure_lifespan(monkeypatch)

    class _InvalidSequenceProvider:
        """Test-only Step 2 fixture; readiness and Step 4 gate are real."""

        name = "test-only-invalid-sequence"
        model = "test-only"

        def __init__(self):
            self.inner = MockLLMProvider()
            self.call_count = 0

        def generate_json(self, prompt, *, schema, system=None):
            self.call_count += 1
            result = self.inner.generate_json(
                prompt, schema=schema, system=system
            )
            result["referenced_inputs"] = [
                {
                    "id_type": id_type,
                    "value": value,
                    "source": "user",
                }
            ]
            result["missing_slots"] = []
            result["response"] = None
            return result

    step2_provider = _InvalidSequenceProvider()
    raw = IntakeService(
        local_storage, registry_service, workflow_state_service
    ).submit(
        raw_user_query="Analyze the HER2 structure with the supplied sequence.",
        user_provided_context={},
    )
    StructuredQueryService(
        local_storage,
        registry_service,
        workflow_state_service,
        SupervisorAgent(llm=step2_provider),
    ).parse(raw.run_id)
    readiness = InputReadinessService(
        local_storage, registry_service, workflow_state_service
    ).check(raw.run_id)
    assert readiness.input_readiness_status == "needs_user_input"

    routing_service, routing_llm, discovery = _routing_service_fixture(
        local_storage, registry_service
    )
    runtime = _MemoryRuntime()
    production_service = OrchestratorApplicationService(
        checkpoint_runtime=runtime,
        routing_service=routing_service,
        discovery=discovery,
        registry=registry_service,
        storage=local_storage,
        worker_timeout_seconds=2,
        max_worker_retries=3,
    )
    counting = _CountingApplicationService(production_service)
    app = create_app(
        checkpoint_runtime_factory=lambda _settings: runtime,
        orchestrator_service_factory=lambda _runtime: counting,
    )

    with TestClient(app) as client:
        response = client.post(f"/runs/{raw.run_id}/steps/4/execute")

    assert response.status_code == 409
    assert response.json()["error_code"] == "input_readiness_not_ready"
    assert step2_provider.call_count == 1
    assert counting.execute_count == 0
    assert routing_llm.call_count == 0
    assert discovery.discover_count == 0
    assert list(runtime.saver.list(None)) == []
    active = registry_service.get(raw.run_id).active_artifacts
    assert active.worker_discovery_snapshot_id is None
    assert active.worker_routing_plan_id is None
    assert active.worker_routing_plan_control_id is None
    assert active.candidate_context_table_id is None
    assert active.structured_liability_summary_id is None
    assert active.prepared_structure_input_package_id is None
    assert not local_storage.exists(
        local_storage.run_key(raw.run_id, "worker_routing_plan.json")
    )
    assert not any(
        "tool_call_records" in key
        for key in local_storage.list_prefix(local_storage.run_key(raw.run_id))
    )


def test_unassigned_uploaded_fasta_blocks_before_step4_or_a2a(
    monkeypatch,
    local_storage,
    registry_service,
    workflow_state_service,
):
    _configure_lifespan(monkeypatch)
    file_id = new_file_id()

    class _GenericFastaProvider:
        """Test-only Step 2 fixture; production normalization/gates are real."""

        name = "test-only-generic-fasta"
        model = "test-only"

        def __init__(self):
            self.inner = MockLLMProvider()
            self.call_count = 0

        def generate_json(self, prompt, *, schema, system=None):
            self.call_count += 1
            result = self.inner.generate_json(prompt, schema=schema, system=system)
            result["referenced_inputs"] = [
                {
                    "id_type": "uploaded_file",
                    "value": file_id,
                    "source": "uploaded_file",
                }
            ]
            result["missing_slots"] = []
            result["response"] = None
            return result

    step2_provider = _GenericFastaProvider()
    raw = IntakeService(
        local_storage, registry_service, workflow_state_service
    ).submit(
        raw_user_query=(
            "Analyze the target heavy light sequence keywords in this upload."
        ),
        uploaded_files=[
            {
                "file_id": file_id,
                "original_filename": "target_heavy_light_sequence.fasta",
                "storage_path": f"inputs/files/{file_id}.fasta",
                "content_type": "text/x-fasta",
            }
        ],
    )
    sq = StructuredQueryService(
        local_storage,
        registry_service,
        workflow_state_service,
        SupervisorAgent(llm=step2_provider),
    ).parse(raw.run_id)
    readiness = InputReadinessService(
        local_storage, registry_service, workflow_state_service
    ).check(raw.run_id)
    assert [ref.get("source") for ref in sq.referenced_inputs] == [
        "uploaded_file"
    ]
    assert [slot.slot_name for slot in sq.missing_slots] == ["sequence_role"]
    assert readiness.input_readiness_status == "needs_user_input"
    assert any(
        request.slot_name == "sequence_role"
        for request in readiness.clarification_requests
    )

    routing_service, routing_llm, discovery = _routing_service_fixture(
        local_storage, registry_service
    )
    runtime = _MemoryRuntime()
    production_service = OrchestratorApplicationService(
        checkpoint_runtime=runtime,
        routing_service=routing_service,
        discovery=discovery,
        registry=registry_service,
        storage=local_storage,
        worker_timeout_seconds=2,
        max_worker_retries=3,
    )
    counting = _CountingApplicationService(production_service)
    app = create_app(
        checkpoint_runtime_factory=lambda _settings: runtime,
        orchestrator_service_factory=lambda _runtime: counting,
    )
    with TestClient(app) as client:
        response = client.post(f"/runs/{raw.run_id}/steps/4/execute")

    assert response.status_code == 409
    assert response.json()["error_code"] == "input_readiness_not_ready"
    assert step2_provider.call_count == 1
    assert counting.execute_count == 0
    assert routing_llm.call_count == 0
    assert discovery.discover_count == 0
    assert list(runtime.saver.list(None)) == []
    active = registry_service.get(raw.run_id).active_artifacts
    assert active.worker_discovery_snapshot_id is None
    assert active.worker_routing_plan_id is None
    assert active.worker_routing_plan_control_id is None
    assert active.candidate_context_table_id is None
    assert not local_storage.exists(
        local_storage.run_key(raw.run_id, "worker_routing_plan.json")
    )


def test_fresh_concurrent_execute_status_and_resume_are_idempotent_over_http(
    monkeypatch, local_storage, registry_service
):
    _configure_lifespan(monkeypatch)
    _seed_inputs(local_storage, registry_service)
    contract = _contract("agent_alpha")
    discovery = _FrozenDiscovery([contract])
    llm = _DeterministicLLM(_proposal("agent_alpha"))
    routing_service = OrchestratorRoutingService(
        discovery=discovery,
        storage=local_storage,
        registry=registry_service,
        llm=llm,
    )
    worker = _SyntheticWorker(
        agent_id="agent_alpha",
        storage=local_storage,
        registry=registry_service,
    )
    handle = _serve_task_handler(_card("agent_alpha"), worker.handle)

    def resolve(
        run_id, *, agent_id, capability_id, dispatch_mode="python_a2a"
    ):
        assert run_id == RUN_ID
        return DispatchTarget(
            agent_id=agent_id,
            capability_id=capability_id,
            dispatch_url=handle.url,
            dispatch_mode=dispatch_mode,
        )

    discovery.resolve_dispatch_target = resolve
    runtime = _MemoryRuntime()

    def service_factory(started_runtime):
        assert started_runtime is runtime
        return OrchestratorApplicationService(
            checkpoint_runtime=runtime,
            routing_service=routing_service,
            discovery=discovery,
            registry=registry_service,
            storage=local_storage,
            worker_timeout_seconds=2,
            max_worker_retries=3,
        )

    app = create_app(
        checkpoint_runtime_factory=lambda _settings: runtime,
        orchestrator_service_factory=service_factory,
    )
    try:
        with TestClient(app) as client:
            missing_status = client.get(
                f"/runs/{RUN_ID}/steps/4/status"
            )
            missing_resume = client.post(
                f"/runs/{RUN_ID}/steps/4/resume"
            )
            with ThreadPoolExecutor(max_workers=2) as pool:
                responses = list(
                    pool.map(
                        lambda _index: client.post(
                            f"/runs/{RUN_ID}/steps/4/execute"
                        ),
                        range(2),
                    )
                )
            status = client.get(f"/runs/{RUN_ID}/steps/4/status")
            resumed = client.post(f"/runs/{RUN_ID}/steps/4/resume")
    finally:
        handle.close()
        get_settings.cache_clear()

    assert runtime.startup_count == runtime.shutdown_count == 1
    assert missing_status.status_code == missing_resume.status_code == 404
    assert missing_status.json()["error_code"] == (
        "orchestrator_checkpoint_not_found"
    )
    assert missing_resume.json()["error_code"] == (
        "orchestrator_checkpoint_not_found"
    )
    assert all(item.status_code == 200 for item in responses)
    assert {item.json()["outcome"] for item in responses} == {"completed"}
    assert sum(item.json()["llm_routing_called"] for item in responses) == 1
    assert sorted(item.json()["checkpoint_reused"] for item in responses) == [
        False,
        True,
    ]
    assert llm.call_count == 1
    assert handle.hits["task"] == 1
    assert len(worker.requests) == 1
    assert worker.requests[0].session_id == "sess_0123456789abcdef"
    checkpoint = runtime.graph.get_state(execution_graph_config(RUN_ID))
    assert checkpoint.values["session_id"] == "sess_0123456789abcdef"
    assert status.status_code == resumed.status_code == 200
    assert status.json()["outcome"] == resumed.json()["outcome"] == "completed"
    assert status.json()["task_counts"]["completed"] == 1
    assert status.json()["artifact_counts"]["available"] == 1
    assert resumed.json()["dispatch_attempt_count"] == 0
    assert set(status.json()) == {
        "run_id",
        "routing_plan_id",
        "outcome",
        "run_status",
        "orchestrator_status",
        "next_wakeup",
        "checkpoint_reused",
        "llm_routing_called",
        "dispatch_attempt_count",
        "decision_counts",
        "task_counts",
        "artifact_counts",
        "artifact_refs",
        "action_code",
        "error_code",
    }
    assert set(status.json()["artifact_refs"][0]) == {
        "artifact_name",
        "status",
        "artifact_id",
        "producer_task_id",
        "safe_summary_ref",
    }

    serialized = repr([item.json() for item in [*responses, status, resumed]])
    for forbidden in (
        "WorkerExecutionRequest",
        "WorkerExecutionResult",
        "PreparedA2ATask",
        "http://127.0.0.1",
        "storage_key",
        "raw_tooluniverse_payload",
        "full_prompt",
        "api_key",
    ):
        assert forbidden not in serialized


def test_step4_api_runs_patent_evidence_dependency_and_replays_without_http(
    monkeypatch, local_storage
):
    _configure_lifespan(monkeypatch)
    base_storage, base_registry, base_workflow = _independent_services(local_storage)
    raw = IntakeService(base_storage, base_registry, base_workflow).submit(
        raw_user_query=(
            "Review scientific literature evidence and patent prior art for a "
            "HER2 ADC using PubChem CID:2244."
        ),
        user_provided_context={
            "target_or_antigen_text": "HER2",
            "candidate_text": "trastuzumab-like antibody",
            "payload_linker_text": "vc-MMAE",
        },
    )
    structured = StructuredQueryService(
        base_storage,
        base_registry,
        base_workflow,
        SupervisorAgent(llm=MockLLMProvider()),
    ).parse(raw.run_id)
    assert {"literature_review_summary", "patent_or_ip_summary"} <= set(
        structured.requested_outputs
    )
    assert any(
        item.get("id_type") == "pubchem_cid"
        for item in structured.referenced_inputs
    )
    assert InputReadinessService(
        base_storage, base_registry, base_workflow
    ).check(raw.run_id).input_readiness_status == "ready"

    step5_services = _independent_services(local_storage)
    step6_services = _independent_services(local_storage)
    structure_services = _independent_services(local_storage)
    patent_services = _independent_services(local_storage)
    step5 = _serve(
        _RecordingStep5,
        create_step5_flask_app,
        storage=step5_services[0],
        registry=step5_services[1],
        workflow=step5_services[2],
        mcp=_local_mcp(),
    )
    step6 = _serve(
        _RecordingStep6Worker,
        create_step6_flask_app,
        storage=step6_services[0],
        registry=step6_services[1],
        workflow=step6_services[2],
        mcp=_success_mcp(),
    )
    structure = _serve(
        _RecordingStructureWorker,
        create_structure_flask_app,
        storage=structure_services[0],
        registry=structure_services[1],
        workflow=structure_services[2],
        mcp=_auditable_local_mcp(),
    )
    patent = _serve(
        _RecordingPatentEvidenceWorker,
        create_patent_evidence_flask_app,
        storage=patent_services[0],
        registry=patent_services[1],
        workflow=patent_services[2],
        mcp=LocalMCPClient(bindings=_patent_evidence_bindings()),
    )
    discovery = WorkerDiscoveryService(
        expected_workers=[
            ExpectedWorkerEndpoint(
                AGENT_ID_STEP5, (CAP_STEP5_CANDIDATE_CONTEXT,), step5.url
            ),
            ExpectedWorkerEndpoint(
                AGENT_ID_STEP6, (CAP_STEP6_DEVELOPABILITY,), step6.url
            ),
            ExpectedWorkerEndpoint(
                AGENT_ID_STRUCTURE, (CAP_STRUCTURE_DESIGN_WORKFLOW,), structure.url
            ),
            ExpectedWorkerEndpoint(
                AGENT_ID_PATENT_EVIDENCE,
                (CAP_PATENT_EVIDENCE_WORKFLOW,),
                patent.url,
            ),
        ],
        storage=base_storage,
        registry=base_registry,
        discovery_timeout_seconds=3,
        health_timeout_seconds=3,
    )
    routing_llm = MockLLMProvider()
    routing_service = OrchestratorRoutingService(
        discovery=discovery,
        storage=base_storage,
        registry=base_registry,
        llm=routing_llm,
    )
    runtime = _MemoryRuntime()

    def service_factory(started_runtime):
        assert started_runtime is runtime
        return OrchestratorApplicationService(
            checkpoint_runtime=runtime,
            routing_service=routing_service,
            discovery=discovery,
            registry=base_registry,
            storage=base_storage,
            worker_timeout_seconds=60,
            max_worker_retries=3,
        )

    app = create_app(
        checkpoint_runtime_factory=lambda _settings: runtime,
        orchestrator_service_factory=service_factory,
    )
    try:
        with TestClient(app) as client:
            first = client.post(f"/runs/{raw.run_id}/steps/4/execute")
            status = client.get(f"/runs/{raw.run_id}/steps/4/status")
            replay = client.post(f"/runs/{raw.run_id}/steps/4/execute")
    finally:
        step5.close()
        step6.close()
        structure.close()
        patent.close()
        get_settings.cache_clear()

    print(
        "STEP4_PATENT_INSPECTION="
        + repr(
            {
                "first": first.json(),
                "status": status.json(),
                "replay_status_code": replay.status_code,
                "replay": replay.json(),
                "posts": {
                    "step5": step5.hits["task"],
                    "step6": step6.hits["task"],
                    "structure": structure.hits["task"],
                    "patent": patent.hits["task"],
                },
            }
        )
    )
    assert first.status_code == status.status_code == replay.status_code == 200
    assert first.json()["outcome"] == status.json()["outcome"] == "completed"
    assert first.json()["dispatch_attempt_count"] == 3
    assert replay.json()["checkpoint_reused"] is True
    assert replay.json()["llm_routing_called"] is False
    assert replay.json()["dispatch_attempt_count"] == 0
    assert step5.hits["task"] == patent.hits["task"] == 1
    assert step6.hits["task"] == 1
    assert structure.hits["task"] == 0
    assert {
        "step5": step5.hits["card"],
        "step6": step6.hits["card"],
        "structure": structure.hits["card"],
        "patent": patent.hits["card"],
    } == {"step5": 3, "step6": 3, "structure": 2, "patent": 3}
    assert all(worker.hits["health"] == 1 for worker in (step5, step6, structure, patent))
    active = base_registry.get(raw.run_id).active_artifacts
    assert active.candidate_context_table_id is not None
    assert active.scientific_evidence_table_id is not None
    assert active.patent_prior_art_table_id is not None
    assert status.json()["task_counts"]["completed"] == 3
