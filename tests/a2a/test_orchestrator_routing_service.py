"""Turn F1-C2 production routing service tests; no task is ever sent.

Most tests use a frozen in-memory discovery authority to isolate HTTP and focus
on planner persistence/revalidation. The dedicated integration test uses real
localhost python-a2a AgentCard endpoints and /health requests. LLM fixtures are
deterministic and test-only; they are not live-provider results or fallbacks.
"""

from __future__ import annotations

import copy
import inspect
import json
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, contextmanager
from types import SimpleNamespace

import pytest
from flask import jsonify, request
from python_a2a import A2AServer
from python_a2a.server.http import create_flask_app
from werkzeug.serving import make_server

from app.a2a.agent_cards import (
    AGENT_ID_STEP5,
    AGENT_ID_STEP6,
    AGENT_ID_STRUCTURE,
    CAP_STEP5_CANDIDATE_CONTEXT,
    CAP_STEP6_DEVELOPABILITY,
    CAP_STRUCTURE_DESIGN_WORKFLOW,
    build_compact_card_catalog,
    build_step5_agent_card,
    build_step6_agent_card,
    build_structure_agent_card,
    parse_adc_agent_contract,
)
from app.a2a.contracts import (
    WorkerArtifactRef,
    WorkerExecutionRequest,
    WorkerExecutionResult,
)
from app.a2a.orchestrator_discovery import (
    DispatchTarget,
    ExpectedWorkerEndpoint,
    WorkerDiscoveryService,
    WorkerUnavailableError,
)
from app.a2a.orchestrator_routing_service import (
    OrchestratorRoutingService,
    OrchestratorRoutingServiceError,
)
from app.utils.ids import new_artifact_id

_PLAN_KEY = "inputs/worker_routing_plan.json"


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


def _step5_step6_proposal() -> dict:
    return {
        "loop_decision": "dispatch_next_workers",
        "decisions": [
            {
                "agent_id": AGENT_ID_STEP5,
                "capability_id": CAP_STEP5_CANDIDATE_CONTEXT,
                "objective": "Build normalized candidate context.",
                "selection_reason": "Candidate context is required downstream.",
                "priority": "high",
            },
            {
                "agent_id": AGENT_ID_STEP6,
                "capability_id": CAP_STEP6_DEVELOPABILITY,
                "objective": "Assess candidate developability.",
                "selection_reason": "The user requested developability assessment.",
                "priority": "normal",
            },
        ],
        "decision_summary": "Build context, then assess developability.",
    }


class _DeterministicRoutingLLM:
    """Test-only structured provider; no network and no production fallback."""

    name = "deterministic_test"
    model = "deterministic-test-v1"

    def __init__(self, response=None, *, failure: Exception | None = None, delay=0):
        self.response = response or _step5_step6_proposal()
        self.failure = failure
        self.delay = delay
        self.call_count = 0
        self.schemas: list[dict] = []
        self._lock = threading.Lock()

    def generate_json(self, prompt, *, schema, system=None):
        with self._lock:
            self.call_count += 1
            self.schemas.append(copy.deepcopy(schema))
        if self.delay:
            time.sleep(self.delay)
        if self.failure:
            raise self.failure
        return copy.deepcopy(self.response)


class _FrozenDiscovery:
    """Test-only frozen-card authority; no HTTP and no worker execution."""

    def __init__(self, *, unavailable=()):
        self.cards = [
            build_step5_agent_card("http://step5.test:8005"),
            build_step6_agent_card("http://step6.test:8006"),
        ]
        self.contracts = [parse_adc_agent_contract(card) for card in self.cards]
        self.unavailable = set(unavailable)
        self.discover_count = 0
        self.workers = {
            contract.agent_id: SimpleNamespace(
                is_available=contract.agent_id not in self.unavailable,
                contract=contract,
            )
            for contract in self.contracts
        }

    def discover_for_run(self, run_id):
        self.discover_count += 1
        return SimpleNamespace(
            available_agent_ids=[
                item.agent_id
                for item in self.contracts
                if item.agent_id not in self.unavailable
            ],
            unavailable_agent_ids=sorted(self.unavailable),
        )

    def get_compact_card_catalog(self, run_id):
        return build_compact_card_catalog(self.cards)

    def get_full_card_cache(self, run_id):
        return SimpleNamespace(workers=self.workers)

    def resolve_dispatch_target(
        self, run_id, *, agent_id, capability_id, dispatch_mode="python_a2a"
    ):
        worker = self.workers.get(agent_id)
        if worker is None or not worker.is_available:
            raise WorkerUnavailableError("test-only unavailable")
        return DispatchTarget(
            agent_id=agent_id,
            capability_id=capability_id,
            dispatch_url=f"http://{agent_id}.test",
            dispatch_mode=dispatch_mode,
        )


def _persist(local_storage, registry_service, run_id, name, path, fields):
    artifact_id = new_artifact_id(name)
    local_storage.write_json(
        local_storage.run_key(run_id, path),
        {"artifact_id": artifact_id, "run_id": run_id, **fields},
    )
    registry_service.update_active(run_id, **{f"{name}_id": artifact_id})
    return artifact_id


def _seed_inputs(
    local_storage,
    registry_service,
    *,
    run_id="run_routing_service",
    readiness="ready",
    sensitive=False,
):
    registry_service.init_registry(run_id)
    sensitive_text = (
        "sk-private-secret\nHEADER PRIVATE_PDB\nACDEFGHIKLMNPQRSTVWYACDEFGHIK"
        if sensitive
        else ""
    )
    raw_request_id = _persist(
        local_storage,
        registry_service,
        run_id,
        "raw_request_record",
        "inputs/raw_request_record.json",
        {
            "raw_user_query": f"Assess developability {sensitive_text}",
            "user_provided_context": {"private_payload": sensitive_text},
            "uploaded_files": [],
        },
    )
    structured_query_id = _persist(
        local_storage,
        registry_service,
        run_id,
        "structured_query",
        "inputs/structured_query.json",
        {
            "task_intent": {
                "primary_intent": "developability_assessment",
                "secondary_intents": [],
            },
            "canonical_query": f"Assess developability {sensitive_text}",
            "requested_outputs": ["developability_summary"],
            "missing_slots": [],
            "mentioned_entities": {},
            "referenced_inputs": [],
            "normalized_entities": [],
            "entity_decompositions": [],
        },
    )
    _persist(
        local_storage,
        registry_service,
        run_id,
        "input_readiness_status",
        "inputs/input_readiness_status.json",
        {
            "checked_at": "2026-07-13T00:00:00Z",
            "source_refs": {
                "raw_request_record_id": raw_request_id,
                "structured_query_id": structured_query_id,
            },
            "input_readiness_status": readiness,
            "missing_input_checklist": [],
            "blocking_reasons": [sensitive_text] if sensitive else [],
        },
    )
    return run_id


def _service(local_storage, registry_service, *, llm=None, discovery=None):
    llm = llm or _DeterministicRoutingLLM()
    discovery = discovery or _FrozenDiscovery()
    return (
        OrchestratorRoutingService(
            discovery=discovery,
            storage=local_storage,
            registry=registry_service,
            llm=llm,
        ),
        llm,
        discovery,
    )


def _decision(plan, capability_id):
    return next(
        item for item in plan.validated_decisions if item.capability_id == capability_id
    )


def _persist_candidate(local_storage, registry_service, run_id, *, status="ok"):
    return _persist(
        local_storage,
        registry_service,
        run_id,
        "candidate_context_table",
        "candidate_context_table.json",
        {
            "schema_version": "v1",
            "context_build_status": status,
            "candidate_records": [],
            "downstream_query_hints": [],
        },
    )


def _persist_liability(local_storage, registry_service, run_id, *, status="completed"):
    return _persist(
        local_storage,
        registry_service,
        run_id,
        "structured_liability_summary",
        "structured_liability_summary.json",
        {
            "schema_version": "v1",
            "prefilter_status": status,
            "candidate_liability_results": [],
        },
    )


def _completed_result(
    *, run_id, plan, decision, artifact_id, storage_key="candidate_context_table.json"
):
    return WorkerExecutionResult(
        payload_type="worker_execution_result",
        payload_version="v1",
        run_id=run_id,
        task_id=decision.task_id,
        routing_plan_id=plan.routing_plan_id,
        routing_decision_id=decision.routing_decision_id,
        agent_id=decision.agent_id,
        capability_id=decision.capability_id,
        execution_status="completed",
        result_status="success",
        output_artifact_refs={
            "candidate_context_table": WorkerArtifactRef(
                artifact_id=artifact_id,
                artifact_type="candidate_context_table",
                storage_key=storage_key,
                run_id=run_id,
                schema_version="v1",
            )
        },
    )


def _step6_completed_result(*, run_id, plan, decision, artifact_id):
    return WorkerExecutionResult(
        payload_type="worker_execution_result",
        payload_version="v1",
        run_id=run_id,
        task_id=decision.task_id,
        routing_plan_id=plan.routing_plan_id,
        routing_decision_id=decision.routing_decision_id,
        agent_id=decision.agent_id,
        capability_id=decision.capability_id,
        execution_status="completed",
        result_status="success",
        output_artifact_refs={
            "structured_liability_summary": WorkerArtifactRef(
                artifact_id=artifact_id,
                artifact_type="structured_liability_summary",
                storage_key="structured_liability_summary.json",
                run_id=run_id,
                schema_version="v1",
            )
        },
    )


def _persisted_plan_bytes(local_storage, run_id):
    return local_storage.read_bytes(local_storage.run_key(run_id, _PLAN_KEY))


def _routing_authority(registry_service, run_id):
    active = registry_service.get(run_id).active_artifacts
    return (
        active.worker_routing_plan_id,
        active.worker_routing_plan_control_id,
        dict(active.worker_routing_plan_output_baselines),
    )


def _task_request(prepared):
    return WorkerExecutionRequest.model_validate_json(
        prepared.task.message["content"]["text"]
    )


def test_initial_plan_persists_identity_and_only_prepares_step5(
    local_storage, registry_service
):
    run_id = _seed_inputs(local_storage, registry_service)
    service, llm, discovery = _service(local_storage, registry_service)

    result = service.plan_for_run(run_id)

    step5 = _decision(result.plan, CAP_STEP5_CANDIDATE_CONTEXT)
    step6 = _decision(result.plan, CAP_STEP6_DEVELOPABILITY)
    assert result.plan.routing_status == "ready"
    assert step5.validation_status == "ready" and step5.task_id
    assert step6.validation_status == "waiting_for_dependencies"
    assert step6.task_id is None
    assert [edge.artifact_name for edge in result.plan.dependency_edges] == [
        "candidate_context_table"
    ]
    assert len(result.prepared_tasks) == result.plan.ready_task_count == 1
    assert _task_request(result.prepared_tasks[0]).capability_id == (
        CAP_STEP5_CANDIDATE_CONTEXT
    )
    assert llm.call_count == discovery.discover_count == 1

    active = registry_service.get(run_id).active_artifacts
    active_id = active.worker_routing_plan_id
    persisted = local_storage.read_json(local_storage.run_key(run_id, _PLAN_KEY))
    assert persisted["artifact_id"] == active_id == result.plan_artifact_id
    assert persisted["run_id"] == run_id
    assert persisted["routing_plan_id"] == result.plan.routing_plan_id
    assert active.worker_routing_plan_control_id == result.plan.routing_plan_id


def test_old_candidate_does_not_release_selected_producer_dependency(
    local_storage, registry_service
):
    run_id = _seed_inputs(local_storage, registry_service, run_id="run_old_candidate")
    old_id = _persist_candidate(local_storage, registry_service, run_id, status="ok")
    candidate_key = local_storage.run_key(run_id, "candidate_context_table.json")
    candidate_body = local_storage.read_json(candidate_key)
    candidate_body["undeclared_private_field"] = "must-not-be-projected"
    local_storage.write_json(candidate_key, candidate_body)
    service, llm, discovery = _service(local_storage, registry_service)
    initial = service.plan_for_run(run_id)

    candidate_summary = next(
        item
        for item in llm.schemas[0]["available_artifact_summary"]
        if item["artifact_name"] == "candidate_context_table"
    )
    assert candidate_summary == {
        "artifact_name": "candidate_context_table",
        "available": True,
        "present_field_names": ["candidate_records"],
    }
    assert "must-not-be-projected" not in json.dumps(llm.schemas[0])

    step5 = _decision(initial.plan, CAP_STEP5_CANDIDATE_CONTEXT)
    stale_completion = _completed_result(
        run_id=run_id,
        plan=initial.plan,
        decision=step5,
        artifact_id=old_id,
    )
    with pytest.raises(
        OrchestratorRoutingServiceError,
        match="completion_output_artifact_not_new",
    ):
        service.revalidate_for_run(run_id, completed_results=[stale_completion])

    unchanged = service.revalidate_for_run(run_id, completed_results=[])
    step6 = _decision(unchanged.plan, CAP_STEP6_DEVELOPABILITY)
    assert step6.validation_status == "waiting_for_dependencies"
    assert step6.task_id is None
    assert len(unchanged.prepared_tasks) == 1
    assert _task_request(unchanged.prepared_tasks[0]).capability_id == (
        CAP_STEP5_CANDIDATE_CONTEXT
    )
    assert old_id not in _task_request(
        unchanged.prepared_tasks[0]
    ).model_dump_json()
    assert llm.call_count == discovery.discover_count == 1


def test_failed_candidate_is_not_available_in_llm_artifact_summary(
    local_storage, registry_service
):
    run_id = _seed_inputs(local_storage, registry_service, run_id="run_failed_summary")
    _persist_candidate(local_storage, registry_service, run_id, status="failed")
    service, llm, _discovery = _service(local_storage, registry_service)

    service.plan_for_run(run_id)

    item = next(
        entry
        for entry in llm.schemas[0]["available_artifact_summary"]
        if entry["artifact_name"] == "candidate_context_table"
    )
    assert item == {
        "artifact_name": "candidate_context_table",
        "available": False,
        "present_field_names": [],
    }


def test_conflicting_cross_card_artifact_contract_fails_closed(
    local_storage, registry_service
):
    run_id = _seed_inputs(local_storage, registry_service, run_id="run_card_conflict")
    discovery = _FrozenDiscovery()
    step6_contract = discovery.workers[AGENT_ID_STEP6].contract
    step6_contract.capabilities[0].required_input_artifacts[0].storage_path = (
        "conflicting/path.json"
    )
    service, llm, _discovery = _service(
        local_storage,
        registry_service,
        llm=_DeterministicRoutingLLM(),
        discovery=discovery,
    )

    with pytest.raises(
        OrchestratorRoutingServiceError, match="artifact_contract_conflict"
    ):
        service.plan_for_run(run_id)
    assert llm.call_count == 0


@pytest.mark.parametrize("candidate_status", ["ok", "partial"])
def test_completed_producer_revalidation_prepares_only_step6_with_stable_ids(
    local_storage, registry_service, candidate_status
):
    run_id = _seed_inputs(
        local_storage,
        registry_service,
        run_id=f"run_revalidate_{candidate_status}",
    )
    service, llm, discovery = _service(local_storage, registry_service)
    initial = service.plan_for_run(run_id)
    step5_before = _decision(initial.plan, CAP_STEP5_CANDIDATE_CONTEXT)
    step6_before = _decision(initial.plan, CAP_STEP6_DEVELOPABILITY)
    candidate_id = _persist_candidate(
        local_storage, registry_service, run_id, status=candidate_status
    )
    completed = _completed_result(
        run_id=run_id,
        plan=initial.plan,
        decision=step5_before,
        artifact_id=candidate_id,
    )

    updated = service.revalidate_for_run(
        run_id,
        completed_results=[completed],
    )

    step5_after = _decision(updated.plan, CAP_STEP5_CANDIDATE_CONTEXT)
    step6_after = _decision(updated.plan, CAP_STEP6_DEVELOPABILITY)
    assert llm.call_count == discovery.discover_count == 1
    assert updated.plan.routing_plan_id == initial.plan.routing_plan_id
    assert step5_after.routing_decision_id == step5_before.routing_decision_id
    assert step5_after.task_id == step5_before.task_id
    assert step6_after.routing_decision_id == step6_before.routing_decision_id
    assert step6_after.validation_status == "ready"
    assert step6_after.task_id
    assert len(updated.prepared_tasks) == updated.plan.ready_task_count == 1
    request = _task_request(updated.prepared_tasks[0])
    assert request.capability_id == CAP_STEP6_DEVELOPABILITY
    assert request.routing_plan_id == initial.plan.routing_plan_id
    ref = request.input_projection.input_artifact_refs["candidate_context_table"]
    assert ref.artifact_id == candidate_id
    serialized = request.model_dump_json()
    assert '"candidate_records":' not in serialized
    assert '"context_build_status":' not in serialized
    assert "candidate-secret" not in serialized
    assert "candidate_context_table.json" not in serialized

    repeated = service.revalidate_for_run(
        run_id,
        completed_results=[completed],
    )
    assert len(repeated.prepared_tasks) == 1
    assert repeated.plan.ready_task_count == 1
    assert repeated.prepared_tasks[0].task.id == updated.prepared_tasks[0].task.id
    assert _decision(
        repeated.plan, CAP_STEP6_DEVELOPABILITY
    ).task_id == step6_after.task_id
    assert llm.call_count == discovery.discover_count == 1


def test_same_service_plan_for_run_requires_cumulative_completion_proof(
    local_storage, registry_service
):
    run_id = _seed_inputs(local_storage, registry_service, run_id="run_same_proof")
    service, llm, discovery = _service(local_storage, registry_service)
    initial = service.plan_for_run(run_id)
    step5 = _decision(initial.plan, CAP_STEP5_CANDIDATE_CONTEXT)
    candidate_id = _persist_candidate(local_storage, registry_service, run_id)
    step5_result = _completed_result(
        run_id=run_id,
        plan=initial.plan,
        decision=step5,
        artifact_id=candidate_id,
    )
    updated = service.revalidate_for_run(
        run_id, completed_results=[step5_result]
    )
    before = _persisted_plan_bytes(local_storage, run_id)
    authority_before = _routing_authority(registry_service, run_id)

    with pytest.raises(
        OrchestratorRoutingServiceError, match="completion_proof_required"
    ):
        service.plan_for_run(run_id)

    assert _persisted_plan_bytes(local_storage, run_id) == before
    assert _routing_authority(registry_service, run_id) == authority_before
    assert _decision(
        updated.plan, CAP_STEP6_DEVELOPABILITY
    ).validation_status == "ready"
    assert llm.call_count == discovery.discover_count == 1


def test_fresh_service_requires_proof_then_restores_only_step6_identity(
    local_storage, registry_service
):
    run_id = _seed_inputs(local_storage, registry_service, run_id="run_fresh_proof")
    service, _llm, _discovery = _service(local_storage, registry_service)
    initial = service.plan_for_run(run_id)
    step5 = _decision(initial.plan, CAP_STEP5_CANDIDATE_CONTEXT)
    candidate_id = _persist_candidate(local_storage, registry_service, run_id)
    step5_result = _completed_result(
        run_id=run_id,
        plan=initial.plan,
        decision=step5,
        artifact_id=candidate_id,
    )
    updated = service.revalidate_for_run(
        run_id, completed_results=[step5_result]
    )
    step6_task_id = updated.prepared_tasks[0].task.id
    before = _persisted_plan_bytes(local_storage, run_id)
    authority_before = _routing_authority(registry_service, run_id)

    fresh, fresh_llm, fresh_discovery = _service(local_storage, registry_service)
    with pytest.raises(
        OrchestratorRoutingServiceError, match="completion_proof_required"
    ):
        fresh.plan_for_run(run_id)
    assert _persisted_plan_bytes(local_storage, run_id) == before
    assert _routing_authority(registry_service, run_id) == authority_before

    restored = fresh.revalidate_for_run(
        run_id, completed_results=[step5_result]
    )
    assert restored.plan.routing_plan_id == initial.plan.routing_plan_id
    assert len(restored.prepared_tasks) == restored.plan.ready_task_count == 1
    assert restored.prepared_tasks[0].task.id == step6_task_id
    assert _task_request(restored.prepared_tasks[0]).capability_id == (
        CAP_STEP6_DEVELOPABILITY
    )
    assert fresh_llm.call_count == 0
    assert fresh_discovery.discover_count == 1


def test_cumulative_step5_step6_results_complete_plan_and_omission_fails_closed(
    local_storage, registry_service
):
    run_id = _seed_inputs(local_storage, registry_service, run_id="run_all_complete")
    service, llm, discovery = _service(local_storage, registry_service)
    initial = service.plan_for_run(run_id)
    step5 = _decision(initial.plan, CAP_STEP5_CANDIDATE_CONTEXT)
    candidate_id = _persist_candidate(local_storage, registry_service, run_id)
    step5_result = _completed_result(
        run_id=run_id,
        plan=initial.plan,
        decision=step5,
        artifact_id=candidate_id,
    )
    after_step5 = service.revalidate_for_run(
        run_id, completed_results=[step5_result]
    )
    step6 = _decision(after_step5.plan, CAP_STEP6_DEVELOPABILITY)
    liability_id = _persist_liability(local_storage, registry_service, run_id)
    step6_result = _step6_completed_result(
        run_id=run_id,
        plan=after_step5.plan,
        decision=step6,
        artifact_id=liability_id,
    )
    before = _persisted_plan_bytes(local_storage, run_id)
    authority_before = _routing_authority(registry_service, run_id)

    with pytest.raises(
        OrchestratorRoutingServiceError, match="completion_proof_required"
    ):
        service.revalidate_for_run(run_id, completed_results=[step6_result])
    assert _persisted_plan_bytes(local_storage, run_id) == before
    assert _routing_authority(registry_service, run_id) == authority_before

    completed = service.revalidate_for_run(
        run_id, completed_results=[step5_result, step6_result]
    )
    assert completed.plan.routing_status == "completed"
    assert completed.prepared_tasks == ()
    assert completed.plan.ready_task_count == 0
    assert completed.plan.waiting_decision_count == 0
    assert completed.plan.rejected_decision_count == 0
    assert completed.plan.warnings == []
    assert llm.call_count == discovery.discover_count == 1


def test_completion_unexpected_output_ref_fails_without_plan_pollution(
    local_storage, registry_service
):
    run_id = _seed_inputs(local_storage, registry_service, run_id="run_extra_output")
    service, _llm, _discovery = _service(local_storage, registry_service)
    initial = service.plan_for_run(run_id)
    step5 = _decision(initial.plan, CAP_STEP5_CANDIDATE_CONTEXT)
    candidate_id = _persist_candidate(local_storage, registry_service, run_id)
    step5_result = _completed_result(
        run_id=run_id,
        plan=initial.plan,
        decision=step5,
        artifact_id=candidate_id,
    )
    refs = dict(step5_result.output_artifact_refs)
    refs["unexpected_artifact"] = WorkerArtifactRef(
        artifact_id="unexpected_test_only",
        artifact_type="unexpected_artifact",
        run_id=run_id,
    )
    unexpected = step5_result.model_copy(update={"output_artifact_refs": refs})
    before = _persisted_plan_bytes(local_storage, run_id)
    authority_before = _routing_authority(registry_service, run_id)

    with pytest.raises(
        OrchestratorRoutingServiceError,
        match="completion_output_artifacts_unexpected",
    ):
        service.revalidate_for_run(run_id, completed_results=[unexpected])

    assert _persisted_plan_bytes(local_storage, run_id) == before
    assert _routing_authority(registry_service, run_id) == authority_before


@pytest.mark.parametrize(
    ("storage_key", "accepted"),
    [
        (None, True),
        ("candidate_context_table.json", True),
        ("untrusted/output.json", False),
    ],
)
def test_completion_storage_key_matches_shared_agent_card_contract(
    local_storage, registry_service, storage_key, accepted
):
    run_id = _seed_inputs(
        local_storage,
        registry_service,
        run_id=f"run_storage_key_{'none' if storage_key is None else accepted}",
    )
    service, _llm, _discovery = _service(local_storage, registry_service)
    initial = service.plan_for_run(run_id)
    step5 = _decision(initial.plan, CAP_STEP5_CANDIDATE_CONTEXT)
    candidate_id = _persist_candidate(local_storage, registry_service, run_id)
    completed = _completed_result(
        run_id=run_id,
        plan=initial.plan,
        decision=step5,
        artifact_id=candidate_id,
        storage_key=storage_key,
    )
    if accepted:
        result = service.revalidate_for_run(
            run_id, completed_results=[completed]
        )
        assert _decision(
            result.plan, CAP_STEP6_DEVELOPABILITY
        ).validation_status == "ready"
    else:
        before = _persisted_plan_bytes(local_storage, run_id)
        with pytest.raises(
            OrchestratorRoutingServiceError,
            match="^completion_output_artifact_storage_key_mismatch$",
        ):
            service.revalidate_for_run(run_id, completed_results=[completed])
        assert _persisted_plan_bytes(local_storage, run_id) == before


@pytest.mark.parametrize(
    ("field", "value", "code"),
    [
        ("routing_decision_id", "unknown_decision", "completion_unknown_decision"),
        ("run_id", "wrong_run", "completion_identity_mismatch"),
        ("routing_plan_id", "wrong_plan", "completion_identity_mismatch"),
        ("task_id", "wrong_task", "completion_identity_mismatch"),
        ("agent_id", "wrong_agent", "completion_identity_mismatch"),
        ("capability_id", "wrong_capability", "completion_identity_mismatch"),
    ],
)
def test_unknown_or_mismatched_completion_result_fails_closed(
    local_storage, registry_service, field, value, code
):
    run_id = _seed_inputs(
        local_storage, registry_service, run_id=f"run_completion_{field}"
    )
    service, _llm, _discovery = _service(local_storage, registry_service)
    initial = service.plan_for_run(run_id)
    step5 = _decision(initial.plan, CAP_STEP5_CANDIDATE_CONTEXT)
    artifact_id = _persist_candidate(local_storage, registry_service, run_id)
    completed = _completed_result(
        run_id=run_id,
        plan=initial.plan,
        decision=step5,
        artifact_id=artifact_id,
    ).model_copy(update={field: value})

    with pytest.raises(OrchestratorRoutingServiceError, match=code):
        service.revalidate_for_run(run_id, completed_results=[completed])


@pytest.mark.parametrize(
    ("execution_status", "result_status"),
    [("failed", "tool_failed"), ("completed", "blocked")],
)
def test_unsuccessful_completion_result_fails_closed(
    local_storage, registry_service, execution_status, result_status
):
    run_id = _seed_inputs(local_storage, registry_service, run_id="run_bad_completion")
    service, _llm, _discovery = _service(local_storage, registry_service)
    initial = service.plan_for_run(run_id)
    step5 = _decision(initial.plan, CAP_STEP5_CANDIDATE_CONTEXT)
    artifact_id = _persist_candidate(local_storage, registry_service, run_id)
    completed = _completed_result(
        run_id=run_id,
        plan=initial.plan,
        decision=step5,
        artifact_id=artifact_id,
    ).model_copy(
        update={
            "execution_status": execution_status,
            "result_status": result_status,
        }
    )

    with pytest.raises(
        OrchestratorRoutingServiceError, match="completion_status_invalid"
    ):
        service.revalidate_for_run(run_id, completed_results=[completed])


def test_completion_missing_expected_output_ref_fails_closed(
    local_storage, registry_service
):
    run_id = _seed_inputs(local_storage, registry_service, run_id="run_missing_ref")
    service, _llm, _discovery = _service(local_storage, registry_service)
    initial = service.plan_for_run(run_id)
    step5 = _decision(initial.plan, CAP_STEP5_CANDIDATE_CONTEXT)
    completed = _completed_result(
        run_id=run_id,
        plan=initial.plan,
        decision=step5,
        artifact_id="unused",
    ).model_copy(update={"output_artifact_refs": {}})

    with pytest.raises(
        OrchestratorRoutingServiceError,
        match="completion_output_artifacts_missing",
    ):
        service.revalidate_for_run(run_id, completed_results=[completed])


@pytest.mark.parametrize("artifact_state", ["missing", "failed", "corrupt"])
def test_invalid_completed_producer_output_fails_closed(
    local_storage, registry_service, artifact_state
):
    run_id = _seed_inputs(
        local_storage,
        registry_service,
        run_id=f"run_bad_producer_{artifact_state}",
    )
    service, _llm, _discovery = _service(local_storage, registry_service)
    initial = service.plan_for_run(run_id)
    step5 = _decision(initial.plan, CAP_STEP5_CANDIDATE_CONTEXT)
    candidate_id = "missing_artifact"
    if artifact_state != "missing":
        candidate_id = _persist_candidate(
            local_storage,
            registry_service,
            run_id,
            status="failed" if artifact_state == "failed" else "ok",
        )
    if artifact_state == "corrupt":
        key = local_storage.run_key(run_id, "candidate_context_table.json")
        body = local_storage.read_json(key)
        body["artifact_id"] = "tampered_test_only"
        local_storage.write_json(key, body)

    completed = _completed_result(
        run_id=run_id,
        plan=initial.plan,
        decision=step5,
        artifact_id=candidate_id,
    )
    with pytest.raises(
        OrchestratorRoutingServiceError,
        match=(
            "completion_output_artifact_identity_mismatch"
            if artifact_state == "missing"
            else "completion_output_artifact_invalid"
        ),
    ):
        service.revalidate_for_run(run_id, completed_results=[completed])


def test_repeated_and_concurrent_plan_for_run_are_idempotent(
    local_storage, registry_service
):
    run_id = _seed_inputs(
        local_storage, registry_service, run_id="run_concurrent_plan"
    )
    llm = _DeterministicRoutingLLM(delay=0.05)
    service, llm, discovery = _service(
        local_storage, registry_service, llm=llm
    )

    with ThreadPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(lambda _index: service.plan_for_run(run_id), range(6)))

    assert llm.call_count == discovery.discover_count == 1
    assert len({item.plan_artifact_id for item in results}) == 1
    assert len({item.plan.routing_plan_id for item in results}) == 1
    assert all(len(item.prepared_tasks) == 1 for item in results)
    assert len({item.prepared_tasks[0].task.id for item in results}) == 1
    assert sum(not item.reused_existing_plan for item in results) == 1
    repeated = service.plan_for_run(run_id)
    assert repeated.reused_existing_plan is True
    assert len(repeated.prepared_tasks) == 1
    assert repeated.prepared_tasks[0].task.id == results[0].prepared_tasks[0].task.id
    assert llm.call_count == discovery.discover_count == 1


def test_llm_failure_persists_compact_failure_without_fallback(
    local_storage, registry_service
):
    run_id = _seed_inputs(local_storage, registry_service, run_id="run_llm_failure")
    llm = _DeterministicRoutingLLM(
        failure=RuntimeError("sk-secret raw response and private endpoint")
    )
    service, llm, _discovery = _service(
        local_storage, registry_service, llm=llm
    )

    result = service.plan_for_run(run_id)

    assert llm.call_count == 1
    assert result.plan.routing_status == "llm_failed"
    assert result.plan.loop_decision is None
    assert result.plan.proposed_decisions == []
    assert result.plan.validated_decisions == []
    assert result.prepared_tasks == ()
    persisted = json.dumps(
        local_storage.read_json(local_storage.run_key(run_id, _PLAN_KEY))
    )
    assert result.plan.warnings == ["llm_error:deterministic_test:runtimeerror"]
    for forbidden in ("sk-secret", "raw response", "private endpoint", "MockLLM"):
        assert forbidden not in persisted


def test_malformed_llm_proposal_persists_schema_failure_without_tasks(
    local_storage, registry_service
):
    run_id = _seed_inputs(local_storage, registry_service, run_id="run_bad_shape")
    llm = _DeterministicRoutingLLM(
        {
            "loop_decision": "dispatch_next_workers",
            "decisions": [],
            "decision_summary": "",
        }
    )
    service, llm, discovery = _service(
        local_storage, registry_service, llm=llm
    )

    result = service.plan_for_run(run_id)

    assert llm.call_count == discovery.discover_count == 1
    assert result.plan.routing_status == "llm_failed"
    assert result.plan.loop_decision is None
    assert result.plan.warnings == ["llm_response_schema_invalid"]
    assert result.plan.proposed_decisions == []
    assert result.plan.validated_decisions == []
    assert result.prepared_tasks == ()


def test_route_to_final_response_persists_completed_without_tasks(
    local_storage, registry_service
):
    run_id = _seed_inputs(local_storage, registry_service, run_id="run_final")
    llm = _DeterministicRoutingLLM(
        {
            "loop_decision": "route_to_final_response",
            "decisions": [],
            "decision_summary": "No more workers are required.",
        }
    )
    service, llm, discovery = _service(
        local_storage, registry_service, llm=llm
    )

    result = service.plan_for_run(run_id)

    assert llm.call_count == discovery.discover_count == 1
    assert result.plan.routing_status == "completed"
    assert result.plan.loop_decision == "route_to_final_response"
    assert result.plan.ready_task_count == 0
    assert result.prepared_tasks == ()


def test_step3_blocked_skips_llm_and_tasks(local_storage, registry_service):
    run_id = _seed_inputs(
        local_storage,
        registry_service,
        run_id="run_step3_blocked",
        readiness="blocked",
        sensitive=True,
    )
    service, llm, discovery = _service(local_storage, registry_service)

    result = service.plan_for_run(run_id)

    assert discovery.discover_count == 1
    assert llm.call_count == 0
    assert result.plan.routing_status == "blocked"
    assert result.plan.loop_decision is None
    assert result.plan.warnings == ["input_readiness_blocked"]
    assert result.prepared_tasks == ()
    persisted = json.dumps(
        local_storage.read_json(local_storage.run_key(run_id, _PLAN_KEY))
    )
    for forbidden in ("blocking_reasons", "sk-private", "PRIVATE_PDB"):
        assert forbidden not in persisted


def test_step3_needs_user_input_waits_without_llm_or_tasks(
    local_storage, registry_service
):
    run_id = _seed_inputs(
        local_storage,
        registry_service,
        run_id="run_step3_needs_input",
        readiness="needs_user_input",
    )
    service, llm, discovery = _service(local_storage, registry_service)

    result = service.plan_for_run(run_id)

    assert discovery.discover_count == 1
    assert llm.call_count == 0
    assert result.prepared_tasks == ()
    assert result.plan.routing_status == "waiting"
    assert result.plan.loop_decision == "request_user_input"
    assert result.plan.warnings == ["input_readiness_needs_user_input"]
    repeated = service.plan_for_run(run_id)
    assert repeated.plan == result.plan
    assert repeated.prepared_tasks == ()
    assert llm.call_count == 0


@pytest.mark.parametrize("bad_status", [None, "unknown"])
def test_step3_invalid_or_missing_status_fails_closed(
    local_storage, registry_service, bad_status
):
    run_id = _seed_inputs(
        local_storage, registry_service, run_id=f"run_bad_readiness_{bad_status}"
    )
    key = local_storage.run_key(run_id, "inputs/input_readiness_status.json")
    body = local_storage.read_json(key)
    if bad_status is None:
        body.pop("input_readiness_status")
    else:
        body["input_readiness_status"] = bad_status
    local_storage.write_json(key, body)
    service, llm, _discovery = _service(local_storage, registry_service)

    with pytest.raises(
        OrchestratorRoutingServiceError,
        match="input_readiness_status_schema_invalid",
    ):
        service.plan_for_run(run_id)
    assert llm.call_count == 0


def test_unsafe_llm_decision_is_rejected_without_persisted_leak(
    local_storage, registry_service
):
    run_id = _seed_inputs(local_storage, registry_service, run_id="run_unsafe_llm")
    unsafe = _step5_step6_proposal()
    unsafe["decisions"] = [
        {
            **unsafe["decisions"][0],
            "objective": "NVIDIA_API_KEY=super-secret-value",
        }
    ]
    service, _llm, _discovery = _service(
        local_storage,
        registry_service,
        llm=_DeterministicRoutingLLM(unsafe),
    )

    result = service.plan_for_run(run_id)

    assert result.plan.routing_status == "rejected"
    assert result.plan.proposed_decisions == []
    assert result.plan.validated_decisions == []
    assert result.plan.rejected_decisions[0].reason == "unsafe_llm_output"
    assert result.prepared_tasks == ()
    persisted = json.dumps(
        local_storage.read_json(local_storage.run_key(run_id, _PLAN_KEY))
    )
    assert "super-secret-value" not in persisted
    assert "NVIDIA_API_KEY" not in persisted


@pytest.mark.parametrize(
    ("artifact_name", "path", "identity_field"),
    [
        ("structured_query", "inputs/structured_query.json", "artifact_id"),
        ("structured_query", "inputs/structured_query.json", "run_id"),
        (
            "input_readiness_status",
            "inputs/input_readiness_status.json",
            "artifact_id",
        ),
        ("input_readiness_status", "inputs/input_readiness_status.json", "run_id"),
    ],
)
def test_planning_input_identity_tampering_fails_closed(
    local_storage, registry_service, artifact_name, path, identity_field
):
    run_id = _seed_inputs(
        local_storage,
        registry_service,
        run_id=f"run_input_tamper_{artifact_name}_{identity_field}",
    )
    key = local_storage.run_key(run_id, path)
    body = local_storage.read_json(key)
    body[identity_field] = "tampered_test_only"
    local_storage.write_json(key, body)
    service, llm, _discovery = _service(local_storage, registry_service)

    with pytest.raises(
        OrchestratorRoutingServiceError,
        match=f"{artifact_name}_identity_mismatch",
    ):
        service.plan_for_run(run_id)
    assert llm.call_count == 0
    assert registry_service.get(
        run_id
    ).active_artifacts.worker_routing_plan_id is None


@pytest.mark.parametrize("identity_field", ["artifact_id", "run_id", "routing_plan_id"])
def test_persisted_plan_identity_tampering_fails_closed(
    local_storage, registry_service, identity_field
):
    run_id = _seed_inputs(
        local_storage,
        registry_service,
        run_id=f"run_plan_tamper_{identity_field}",
    )
    service, _llm, _discovery = _service(local_storage, registry_service)
    service.plan_for_run(run_id)
    key = local_storage.run_key(run_id, _PLAN_KEY)
    body = local_storage.read_json(key)
    body[identity_field] = "tampered_test_only"
    local_storage.write_json(key, body)

    fresh_service, fresh_llm, fresh_discovery = _service(
        local_storage, registry_service
    )
    with pytest.raises(
        OrchestratorRoutingServiceError,
        match="worker_routing_plan_identity_mismatch",
    ):
        fresh_service.plan_for_run(run_id)
    assert fresh_llm.call_count == 0
    assert fresh_discovery.discover_count == 0


def test_fresh_service_rediscovers_and_rebuilds_same_ready_task_identity(
    local_storage, registry_service
):
    run_id = _seed_inputs(local_storage, registry_service, run_id="run_fresh_restore")
    service, llm, discovery = _service(local_storage, registry_service)
    initial = service.plan_for_run(run_id)
    task_id = initial.prepared_tasks[0].task.id

    fresh, fresh_llm, fresh_discovery = _service(local_storage, registry_service)
    restored = fresh.plan_for_run(run_id)

    assert llm.call_count == discovery.discover_count == 1
    assert fresh_llm.call_count == 0
    assert fresh_discovery.discover_count == 1
    assert restored.reused_existing_plan is True
    assert restored.discovery_performed is True
    assert restored.plan.routing_plan_id == initial.plan.routing_plan_id
    assert restored.plan.ready_task_count == 1
    assert len(restored.prepared_tasks) == 1
    assert restored.prepared_tasks[0].task.id == task_id


def test_fresh_service_rejects_registry_routing_control_tampering(
    local_storage, registry_service
):
    run_id = _seed_inputs(local_storage, registry_service, run_id="run_control_tamper")
    service, _llm, _discovery = _service(local_storage, registry_service)
    service.plan_for_run(run_id)
    registry_service.update_active(
        run_id, worker_routing_plan_control_id="tampered_control_id"
    )
    fresh, fresh_llm, fresh_discovery = _service(local_storage, registry_service)

    with pytest.raises(
        OrchestratorRoutingServiceError,
        match="worker_routing_plan_identity_mismatch",
    ):
        fresh.plan_for_run(run_id)
    assert fresh_llm.call_count == fresh_discovery.discover_count == 0


def test_projected_schema_and_persisted_plan_exclude_private_material(
    local_storage, registry_service
):
    run_id = _seed_inputs(
        local_storage,
        registry_service,
        run_id="run_privacy",
        sensitive=True,
    )
    service, llm, _discovery = _service(local_storage, registry_service)
    result = service.plan_for_run(run_id)

    schema_blob = json.dumps(llm.schemas[0])
    persisted = json.dumps(
        local_storage.read_json(local_storage.run_key(run_id, _PLAN_KEY))
    )
    task_blob = json.dumps(
        [item.task.to_dict() for item in result.prepared_tasks]
    )
    assert "available_artifact_summary" in llm.schemas[0]
    assert "present_field_names" in schema_blob
    for forbidden in (
        "sk-private-secret",
        "PRIVATE_PDB",
        "ACDEFGHIKLMNPQRSTVWYACDEFGHIK",
        "inputs/structured_query.json",
        "http://step5.test",
        "ORCHESTRATOR_ROUTING_SYSTEM_PROMPT",
        "raw LLM response",
    ):
        assert forbidden not in schema_blob
        assert forbidden not in persisted
    assert "python_a2a.Task" not in persisted
    assert "candidate_records" not in task_blob


def test_unavailable_worker_remains_compact_rejected(
    local_storage, registry_service
):
    run_id = _seed_inputs(local_storage, registry_service, run_id="run_unavailable")
    proposal = _step5_step6_proposal()
    proposal["decisions"] = [proposal["decisions"][1]]
    discovery = _FrozenDiscovery(unavailable={AGENT_ID_STEP6})
    service, _llm, _discovery = _service(
        local_storage,
        registry_service,
        llm=_DeterministicRoutingLLM(proposal),
        discovery=discovery,
    )

    result = service.plan_for_run(run_id)

    assert result.plan.routing_status == "rejected"
    assert result.plan.validated_decisions == []
    assert result.plan.rejected_decisions[0].reason == "rejected_unavailable"
    assert result.prepared_tasks == ()


class _HttpStub:
    def __init__(self, url, server, thread, hits):
        self.url = url
        self.server = server
        self.thread = thread
        self.hits = hits

    def close(self):
        self.server.shutdown()
        self.thread.join(timeout=5)


@contextmanager
def _http_worker(builder):
    card = builder("http://placeholder")
    server = A2AServer(agent_card=card, google_a2a_compatible=False)
    app = create_flask_app(server)
    hits = Counter()

    @app.before_request
    def _count():
        if "agent.json" in request.path:
            hits["card"] += 1
        elif request.path == "/health":
            hits["health"] += 1
        else:
            hits["other"] += 1

    httpd = make_server("127.0.0.1", 0, app, threaded=True)
    url = f"http://127.0.0.1:{httpd.server_port}"
    card.url = url

    @app.get("/health")
    def _health():
        contract = card.capabilities["adc_agent_contract"]
        return jsonify(
            {
                "status": "ok",
                "agent_id": contract["agent_id"],
                "capabilities": [
                    item["capability_id"] for item in contract["capabilities"]
                ],
            }
        )

    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    handle = _HttpStub(url, httpd, thread, hits)
    try:
        yield handle
    finally:
        handle.close()


def test_real_localhost_discovery_to_persisted_plan_without_task_send(
    local_storage, registry_service
):
    run_id = _seed_inputs(local_storage, registry_service, run_id="run_real_http")
    with ExitStack() as stack:
        step5 = stack.enter_context(_http_worker(build_step5_agent_card))
        step6 = stack.enter_context(_http_worker(build_step6_agent_card))
        structure = stack.enter_context(_http_worker(build_structure_agent_card))
        discovery = WorkerDiscoveryService(
            expected_workers=[
                ExpectedWorkerEndpoint(
                    AGENT_ID_STEP5,
                    (CAP_STEP5_CANDIDATE_CONTEXT,),
                    step5.url,
                ),
                ExpectedWorkerEndpoint(
                    AGENT_ID_STEP6,
                    (CAP_STEP6_DEVELOPABILITY,),
                    step6.url,
                ),
                ExpectedWorkerEndpoint(
                    AGENT_ID_STRUCTURE,
                    (CAP_STRUCTURE_DESIGN_WORKFLOW,),
                    structure.url,
                ),
            ],
            storage=local_storage,
            registry=registry_service,
            discovery_timeout_seconds=3,
            health_timeout_seconds=3,
        )
        llm = _DeterministicRoutingLLM()
        service = OrchestratorRoutingService(
            discovery=discovery,
            storage=local_storage,
            registry=registry_service,
            llm=llm,
        )

        result = service.plan_for_run(run_id)

        assert result.plan.routing_status == "ready"
        assert len(result.prepared_tasks) == 1
        assert llm.call_count == 1
        # One A2AClient AgentCard fetch plus one declared-card URL check, and
        # one /health request per worker. No A2A task endpoint is invoked.
        assert step5.hits == {"card": 2, "health": 1}
        assert step6.hits == {"card": 2, "health": 1}
        assert structure.hits == {"card": 2, "health": 1}
        assert sum(item.hits.get("other", 0) for item in (step5, step6, structure)) == 0
        assert result.plan.available_agent_ids == [
            AGENT_ID_STEP5,
            AGENT_ID_STEP6,
            AGENT_ID_STRUCTURE,
        ]
        assert [
            item["agent_id"] for item in llm.schemas[0]["compact_card_catalog"]
        ] == [AGENT_ID_STEP5, AGENT_ID_STEP6, AGENT_ID_STRUCTURE]
        assert registry_service.get(
            run_id
        ).active_artifacts.worker_routing_plan_id == result.plan_artifact_id


def test_service_source_has_no_dispatch_or_worker_execution_calls():
    source = inspect.getsource(OrchestratorRoutingService)
    for forbidden in (
        "send_task(",
        "send_task_async(",
        "execute_request(",
        "worker.run(",
        "agent.run_from_artifacts(",
    ):
        assert forbidden not in source
