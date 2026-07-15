"""Turn G production-parity FastAPI/Postgres/HTTP worker-core smoke.

MockLLMProvider and deterministic local MCP bindings are test/offline
fixtures. They do not prove live LLM, MCP, ToolUniverse, or biomedical-tool
success, and no production mock-success fallback is installed.
"""

from __future__ import annotations

import copy
import json
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import psycopg
import pytest
from fastapi.testclient import TestClient

from app.a2a.agent_cards import (
    AGENT_ID_STEP5,
    AGENT_ID_STEP6,
    AGENT_ID_STRUCTURE,
    CAP_STEP5_CANDIDATE_CONTEXT,
    CAP_STEP6_DEVELOPABILITY,
    CAP_STRUCTURE_DESIGN_WORKFLOW,
)
from app.a2a.orchestrator_application_service import (
    OrchestratorApplicationService,
)
from app.a2a.orchestrator_discovery import (
    ExpectedWorkerEndpoint,
    WorkerDiscoveryService,
)
from app.a2a.orchestrator_routing_service import OrchestratorRoutingService
from app.a2a.step5_worker import create_step5_flask_app
from app.a2a.step6_worker import create_step6_flask_app
from app.a2a.structure_worker import create_structure_flask_app
from app.agents.supervisor_agent import SupervisorAgent
from app.graph.orchestrator_checkpoint_runtime import (
    OrchestratorCheckpointRuntimeError,
    OrchestratorPostgresCheckpointRuntime,
)
from app.llm.provider import MockLLMProvider
from app.main import create_app
from app.services.input_readiness_service import InputReadinessService
from app.services.intake_service import IntakeService
from app.services.structured_query_service import StructuredQueryService
from app.settings import get_settings
from tests.a2a.test_orchestrator_dispatch import _local_mcp
from tests.a2a.test_orchestrator_routing_intent import QUERY
from tests.a2a.test_parallel_worker_http_smoke import (
    _TimedStep5,
    _TimedStep6,
    _TimedStructure,
    _independent_services,
    _record_status,
    _serve,
    _tool_records,
)
from tests.a2a.test_step6_worker_a2a import _success_mcp
from tests.a2a.test_structure_worker_a2a import _auditable_local_mcp
from tests.integration.test_orchestrator_postgres_checkpoint import _database_url
from tests.integration.test_orchestrator_postgres_checkpoint import (
    _serve_discoverable_worker,
)
from tests.a2a import test_orchestrator_post_ingestion as post_ingestion_fixtures
from tests.a2a.test_orchestrator_post_ingestion import (
    _DeterministicLLM,
    _contract,
    _proposal,
    _seed_inputs,
)
from tests.a2a.test_orchestrator_retry_loop import _SyntheticWorker
from app.utils.ids import new_run_id


class _CountingMockLLM(MockLLMProvider):
    """Test-only counter around the real deterministic Mock provider path."""

    def __init__(self):
        super().__init__()
        self.routing_call_count = 0

    def generate_json(self, prompt, *, schema, system=None):
        if (schema or {}).get("task") == "orchestrator_worker_routing":
            self.routing_call_count += 1
        return super().generate_json(prompt, schema=schema, system=system)


class _BlockingRoutingLLM(_DeterministicLLM):
    """Test-only deterministic LLM with a planning concurrency barrier."""

    def __init__(self, decisions):
        super().__init__(decisions)
        self.entered = threading.Event()
        self.release = threading.Event()

    def generate_json(self, prompt, *, schema, system=None):
        self.call_count += 1
        self.entered.set()
        if not self.release.wait(timeout=5):
            raise RuntimeError("test planning barrier timed out")
        return {
            "loop_decision": "dispatch_next_workers",
            "decisions": copy.deepcopy(self.decisions),
            "decision_summary": "Validate one synthetic generic worker.",
        }


@pytest.mark.asyncio
async def test_postgres_run_lock_rejects_competing_backend_mutation():
    runtime_a = OrchestratorPostgresCheckpointRuntime(_database_url())
    runtime_b = OrchestratorPostgresCheckpointRuntime(_database_url())
    await runtime_a.startup()
    await runtime_b.startup()
    try:
        async with runtime_a.run_lock("run_20260715_abcdef12"):
            with pytest.raises(
                OrchestratorCheckpointRuntimeError,
                match="^checkpoint_run_lock_unavailable$",
            ):
                async with runtime_b.run_lock("run_20260715_abcdef12"):
                    pass
        async with runtime_b.run_lock("run_20260715_abcdef12"):
            pass
    finally:
        await runtime_a.shutdown()
        await runtime_b.shutdown()


def _build_generic_backend_app(
    *, storage, registry, contract, worker_url, llm
):
    discovery = WorkerDiscoveryService(
        expected_workers=[
            ExpectedWorkerEndpoint(
                contract.agent_id,
                (contract.capabilities[0].capability_id,),
                worker_url,
            )
        ],
        storage=storage,
        registry=registry,
        discovery_timeout_seconds=3,
        health_timeout_seconds=3,
    )
    routing = OrchestratorRoutingService(
        discovery=discovery,
        storage=storage,
        registry=registry,
        llm=llm,
    )
    planned = []
    original = routing.plan_for_run

    def record(run_id):
        result = original(run_id)
        planned.append(result)
        return result

    routing.plan_for_run = record

    def factory(runtime):
        return OrchestratorApplicationService(
            checkpoint_runtime=runtime,
            routing_service=routing,
            discovery=discovery,
            registry=registry,
            storage=storage,
            worker_timeout_seconds=10,
            max_worker_retries=3,
        )

    return create_app(orchestrator_service_factory=factory), planned, discovery


def _checkpoint_count(run_id):
    with psycopg.connect(_database_url()) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT count(*) FROM checkpoints WHERE thread_id = %s",
                (run_id,),
            )
            return int(cursor.fetchone()[0])


def test_two_fastapi_backends_share_postgres_exactly_once_authority(
    monkeypatch, local_storage, registry_service
):
    """Two real runtimes contend before checkpoint/task side effects."""
    run_id = new_run_id()
    monkeypatch.setattr(post_ingestion_fixtures, "RUN_ID", run_id)
    _seed_inputs(local_storage, registry_service)
    contract = _contract("agent_alpha")
    contract = contract.model_copy(
        update={
            "capabilities": [
                contract.capabilities[0].model_copy(
                    update={"required_artifact_fields": {}}
                )
            ]
        }
    )
    worker = _SyntheticWorker(
        agent_id="agent_alpha",
        storage=local_storage,
        registry=registry_service,
    )
    handle = _serve_discoverable_worker(contract, worker.handle)
    llm_a = _BlockingRoutingLLM(_proposal("agent_alpha"))
    llm_b = _DeterministicLLM(_proposal("agent_alpha"))
    app_a, planned_a, _discovery_a = _build_generic_backend_app(
        storage=local_storage,
        registry=registry_service,
        contract=contract,
        worker_url=handle.url,
        llm=llm_a,
    )
    app_b, planned_b, _discovery_b = _build_generic_backend_app(
        storage=local_storage,
        registry=registry_service,
        contract=contract,
        worker_url=handle.url,
        llm=llm_b,
    )
    monkeypatch.setenv("LANGGRAPH_CHECKPOINT_DATABASE_URL", _database_url())
    monkeypatch.setenv("ORCHESTRATOR_WORKER_TIMEOUT_SECONDS", "10")
    get_settings.cache_clear()

    try:
        with TestClient(app_a) as client_a, TestClient(app_b) as client_b:
            assert (
                app_a.state.orchestrator_checkpoint_runtime
                is not app_b.state.orchestrator_checkpoint_runtime
            )
            with ThreadPoolExecutor(max_workers=2) as pool:
                first_future = pool.submit(
                    client_a.post, f"/runs/{run_id}/steps/4/execute"
                )
                assert llm_a.entered.wait(timeout=3)
                before_competitor = dict(handle.hits)
                assert _checkpoint_count(run_id) == 0
                competing = client_b.post(
                    f"/runs/{run_id}/steps/4/execute"
                )
                assert competing.status_code == 503
                assert competing.json()["error_code"] == "orchestrator_run_busy"
                assert llm_b.call_count == 0
                assert planned_b == []
                assert dict(handle.hits) == before_competitor
                assert _checkpoint_count(run_id) == 0
                llm_a.release.set()
                first = first_future.result(timeout=15)

            counts_after_first = dict(handle.hits)
            checkpoint_count_after_first = _checkpoint_count(run_id)
            assert checkpoint_count_after_first > 0
            replay_execute = client_b.post(
                f"/runs/{run_id}/steps/4/execute"
            )
            replay_resume = client_b.post(
                f"/runs/{run_id}/steps/4/resume"
            )
            status = client_b.get(f"/runs/{run_id}/steps/4/status")
            checkpoint_count_after_replay = _checkpoint_count(run_id)
    finally:
        llm_a.release.set()
        handle.close()
        get_settings.cache_clear()

    assert first.status_code == 200
    assert first.json()["outcome"] == "completed"
    assert replay_execute.status_code == replay_resume.status_code == 200
    assert replay_execute.json()["outcome"] == "completed"
    assert replay_resume.json()["outcome"] == "completed"
    assert status.status_code == 200
    assert llm_a.call_count == 1
    assert llm_b.call_count == 0
    assert len(planned_a) == 1
    assert planned_b == []
    assert handle.hits["task"] == 1
    assert len(worker.requests) == 1
    assert dict(handle.hits) == counts_after_first
    assert checkpoint_count_after_replay == checkpoint_count_after_first

    active = registry_service.get(run_id).active_artifacts
    plan = local_storage.read_json(
        local_storage.run_key(run_id, "inputs/worker_routing_plan.json")
    )
    assert plan["artifact_id"] == active.worker_routing_plan_id
    assert plan["routing_plan_id"] == active.worker_routing_plan_control_id
    assert status.json()["routing_plan_id"] == plan["routing_plan_id"]
    print(
        "TURN_G_TWO_BACKEND_EXACTLY_ONCE="
        + json.dumps(
            {
                "llm_calls": {"backend_a": 1, "backend_b": 0},
                "http_counts": dict(handle.hits),
                "busy_before_side_effects": True,
                "routing_plan_artifact_id": plan["artifact_id"],
                "routing_plan_id": plan["routing_plan_id"],
                "checkpoint_routing_plan_id": status.json()["routing_plan_id"],
                "checkpoint_writes": checkpoint_count_after_first,
                "replay_http_delta": 0,
                "replay_checkpoint_delta": 0,
            },
            sort_keys=True,
        )
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


def _build_app(
    *, storage, registry, urls, llm, monkeypatch
):
    discovery = WorkerDiscoveryService(
        expected_workers=[
            ExpectedWorkerEndpoint(
                AGENT_ID_STEP5,
                (CAP_STEP5_CANDIDATE_CONTEXT,),
                urls[AGENT_ID_STEP5],
            ),
            ExpectedWorkerEndpoint(
                AGENT_ID_STEP6,
                (CAP_STEP6_DEVELOPABILITY,),
                urls[AGENT_ID_STEP6],
            ),
            ExpectedWorkerEndpoint(
                AGENT_ID_STRUCTURE,
                (CAP_STRUCTURE_DESIGN_WORKFLOW,),
                urls[AGENT_ID_STRUCTURE],
            ),
        ],
        storage=storage,
        registry=registry,
        discovery_timeout_seconds=3,
        health_timeout_seconds=3,
    )
    routing = OrchestratorRoutingService(
        discovery=discovery,
        storage=storage,
        registry=registry,
        llm=llm,
    )
    planned_results = []
    original_plan_for_run = routing.plan_for_run

    def recording_plan_for_run(run_id):
        result = original_plan_for_run(run_id)
        planned_results.append(result)
        return result

    routing.plan_for_run = recording_plan_for_run

    def service_factory(runtime):
        return OrchestratorApplicationService(
            checkpoint_runtime=runtime,
            routing_service=routing,
            discovery=discovery,
            registry=registry,
            storage=storage,
            worker_timeout_seconds=60,
            max_worker_retries=3,
        )

    monkeypatch.setenv("LANGGRAPH_CHECKPOINT_DATABASE_URL", _database_url())
    monkeypatch.setenv("ORCHESTRATOR_WORKER_TIMEOUT_SECONDS", "60")
    get_settings.cache_clear()
    return (
        create_app(orchestrator_service_factory=service_factory),
        discovery,
        planned_results,
    )


def test_fastapi_postgres_three_worker_core_smoke_and_restart_resume(
    monkeypatch, local_storage
):
    base_storage, base_registry, base_workflow = _independent_services(local_storage)
    record = IntakeService(base_storage, base_registry, base_workflow).submit(
        raw_user_query=QUERY,
        user_provided_context={
            "target_or_antigen_text": "HER2",
            "candidate_text": "trastuzumab-like antibody",
            "payload_linker_text": "vc-MMAE",
        },
    )
    StructuredQueryService(
        base_storage,
        base_registry,
        base_workflow,
        SupervisorAgent(llm=MockLLMProvider()),
    ).parse(record.run_id)
    readiness = InputReadinessService(
        base_storage, base_registry, base_workflow
    ).check(record.run_id)
    assert readiness.input_readiness_status == "ready"

    step5_services = _independent_services(local_storage)
    step6_services = _independent_services(local_storage)
    structure_services = _independent_services(local_storage)
    step5 = _serve(
        _TimedStep5,
        create_step5_flask_app,
        storage=step5_services[0],
        registry=step5_services[1],
        workflow=step5_services[2],
        mcp=_local_mcp(),
    )
    step6 = _serve(
        _TimedStep6,
        create_step6_flask_app,
        storage=step6_services[0],
        registry=step6_services[1],
        workflow=step6_services[2],
        mcp=_success_mcp(),
    )
    structure = _serve(
        _TimedStructure,
        create_structure_flask_app,
        storage=structure_services[0],
        registry=structure_services[1],
        workflow=structure_services[2],
        mcp=_auditable_local_mcp(),
    )
    handles = {
        AGENT_ID_STEP5: step5,
        AGENT_ID_STEP6: step6,
        AGENT_ID_STRUCTURE: structure,
    }
    urls = {name: handle.url for name, handle in handles.items()}
    first_llm = _CountingMockLLM()
    app_a, discovery_a, planned_a = _build_app(
        storage=base_storage,
        registry=base_registry,
        urls=urls,
        llm=first_llm,
        monkeypatch=monkeypatch,
    )
    try:
        with TestClient(app_a) as client:
            executed = client.post(
                f"/runs/{record.run_id}/steps/4/execute"
            )
            status_a = client.get(
                f"/runs/{record.run_id}/steps/4/status"
            )

        counts_before_restart = {
            name: dict(handle.hits) for name, handle in handles.items()
        }
        second_llm = _CountingMockLLM()
        app_b, _discovery_b, planned_b = _build_app(
            storage=base_storage,
            registry=base_registry,
            urls=urls,
            llm=second_llm,
            monkeypatch=monkeypatch,
        )
        with TestClient(app_b) as client:
            resumed = client.post(
                f"/runs/{record.run_id}/steps/4/resume"
            )
            status_b = client.get(
                f"/runs/{record.run_id}/steps/4/status"
            )
    finally:
        for handle in handles.values():
            handle.close()
        get_settings.cache_clear()

    assert executed.status_code == status_a.status_code == 200
    assert resumed.status_code == status_b.status_code == 200
    assert executed.json()["outcome"] == "completed"
    assert executed.json()["dispatch_attempt_count"] == 3
    assert status_a.json()["outcome"] == resumed.json()["outcome"] == "completed"
    assert status_b.json()["outcome"] == "completed"
    assert resumed.json()["dispatch_attempt_count"] == 0
    assert discovery_a.get_full_card_cache(record.run_id)
    assert first_llm.routing_call_count == 1
    assert second_llm.routing_call_count == 0
    assert len(planned_a) == 1
    assert planned_b == []

    for name, handle in handles.items():
        assert handle.hits["card"] == 3
        assert handle.hits["health"] == 1
        assert handle.hits["task"] == 1
        assert handle.hits["get_task"] == 0
        assert dict(handle.hits) == counts_before_restart[name]

    assert step6.worker.window[0] < structure.worker.window[1]
    assert structure.worker.window[0] < step6.worker.window[1]

    active = base_registry.get(record.run_id).active_artifacts
    artifact_specs = {
        "candidate_context_table": (
            "candidate_context_table.json",
            active.candidate_context_table_id,
        ),
        "structured_liability_summary": (
            "structured_liability_summary.json",
            active.structured_liability_summary_id,
        ),
        "prepared_structure_input_package": (
            "prepared_structure_input_package.json",
            active.prepared_structure_input_package_id,
        ),
        "structure_prediction_and_interface_results": (
            "structure_prediction_and_interface_results.json",
            active.structure_prediction_and_interface_results_id,
        ),
        "structure_variant_and_compound_screening": (
            "compound_screening_artifact.json",
            active.structure_variant_and_compound_screening_id,
        ),
    }
    artifacts = {}
    for name, (path, artifact_id) in artifact_specs.items():
        body = base_storage.read_json(base_storage.run_key(record.run_id, path))
        assert artifact_id is not None
        assert body["artifact_id"] == artifact_id
        assert body["run_id"] == record.run_id
        artifacts[name] = body

    workflow = base_workflow.get(record.run_id)
    assert {
        name: workflow["steps"][name]
        for name in ("step_05", "step_06", "step_07", "step_08", "step_09")
    } == {
        "step_05": "completed",
        "step_06": "completed",
        "step_07": "completed",
        "step_08": "completed",
        "step_09": "completed",
    }

    tool_records = _tool_records(artifacts)
    distribution = Counter(_record_status(item) for item in tool_records)
    non_success = [
        {
            "tool_name": item.get("tool_name"),
            "status": item.get("run_status"),
            "reason": item.get("error_code") or item.get("error_message"),
        }
        for item in tool_records
        if item.get("run_status") != "success"
    ]
    response_blob = json.dumps(
        [executed.json(), status_a.json(), resumed.json(), status_b.json()],
        sort_keys=True,
    )
    checkpoint_safe = (
        "WorkerExecutionResult" not in response_blob
        and "WorkerExecutionRequest" not in response_blob
        and "PreparedA2ATask" not in response_blob
        and "http://127.0.0.1" not in response_blob
        and QUERY not in response_blob
    )
    assert checkpoint_safe

    plan_body = base_storage.read_json(
        base_storage.run_key(record.run_id, "inputs/worker_routing_plan.json")
    )
    assert plan_body["artifact_id"] == active.worker_routing_plan_id
    assert plan_body["routing_plan_id"] == (
        active.worker_routing_plan_control_id
    )
    assert status_b.json()["routing_plan_id"] == plan_body["routing_plan_id"]
    public_artifacts = {
        item["artifact_name"]: item
        for item in status_b.json()["artifact_refs"]
    }
    for name, (_path, artifact_id) in artifact_specs.items():
        assert public_artifacts[name]["status"] == "available"
        assert public_artifacts[name]["artifact_id"] == artifact_id
    actual_decisions = {
        item["agent_id"]: item["validation_status"]
        for item in plan_body["validated_decisions"]
    }
    assert actual_decisions == {
        AGENT_ID_STEP5: "ready",
        AGENT_ID_STEP6: "ready",
        AGENT_ID_STRUCTURE: "ready",
    }
    proposal_agents = [
        item["agent_id"] for item in plan_body["proposed_decisions"]
    ]
    assert proposal_agents == [
        AGENT_ID_STEP5,
        AGENT_ID_STEP6,
        AGENT_ID_STRUCTURE,
    ]
    initial_decisions = {
        item.agent_id: item.validation_status
        for item in planned_a[0].plan.validated_decisions
    }
    assert initial_decisions == {
        AGENT_ID_STEP5: "ready",
        AGENT_ID_STEP6: "waiting_for_dependencies",
        AGENT_ID_STRUCTURE: "waiting_for_dependencies",
    }

    inspection = {
        "proposal_agents": proposal_agents,
        "initial_decisions": initial_decisions,
        "final_validated_decisions": actual_decisions,
        "dispatch_attempts": executed.json()["dispatch_attempt_count"],
        "http_counts": {
            name: dict(handle.hits) for name, handle in handles.items()
        },
        "parallel_windows": {
            AGENT_ID_STEP6: step6.worker.window,
            AGENT_ID_STRUCTURE: structure.worker.window,
        },
        "api_task_counts": status_b.json()["task_counts"],
        "api_artifact_counts": status_b.json()["artifact_counts"],
        "active_artifacts": sorted(artifact_specs),
        "routing_identity": {
            "plan_artifact_matches_registry": True,
            "routing_plan_matches_registry_control": True,
            "checkpoint_api_routing_plan_matches": True,
        },
        "workflow_steps": {
            name: workflow["steps"][name]
            for name in ("step_05", "step_06", "step_07", "step_08", "step_09")
        },
        "tool_status_distribution": dict(distribution),
        "non_success_tools": non_success,
        "restart": {
            "outcome": resumed.json()["outcome"],
            "dispatch_attempts": resumed.json()["dispatch_attempt_count"],
            "fresh_discovery_http": 0,
        },
    }
    print("TURN_G_API_POSTGRES_SMOKE=" + json.dumps(inspection, sort_keys=True))
