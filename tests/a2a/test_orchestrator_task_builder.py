"""In-memory Task builder tests; no A2A client or worker execution."""

from __future__ import annotations

import inspect
import json
from types import SimpleNamespace

import pytest

from app.a2a.agent_cards import (
    AGENT_ID_STEP6,
    CAP_STEP6_DEVELOPABILITY,
    build_step5_agent_card,
    build_step6_agent_card,
    build_structure_agent_card,
    parse_adc_agent_contract,
)
from app.a2a.contracts import A2ATaskMetadata, WorkerExecutionRequest
from app.a2a.orchestrator_discovery import DispatchTarget
from app.a2a.orchestrator_routing_validation import validate_orchestrator_routing
from app.a2a.orchestrator_task_builder import (
    build_orchestrator_worker_task,
)
from app.schemas.worker_routing_plan import (
    OrchestratorRoutingProposal,
    RejectedRoutingDecision,
)


class _Discovery:
    def __init__(self):
        contracts = [
            parse_adc_agent_contract(build_step5_agent_card("http://step5")),
            parse_adc_agent_contract(build_step6_agent_card("http://step6")),
            parse_adc_agent_contract(build_structure_agent_card("http://structure")),
        ]
        self.workers = {
            contract.agent_id: SimpleNamespace(is_available=True, contract=contract)
            for contract in contracts
        }

    def get_full_card_cache(self, run_id):
        return SimpleNamespace(workers=self.workers)

    def resolve_dispatch_target(
        self, run_id, *, agent_id, capability_id, dispatch_mode="python_a2a"
    ):
        return DispatchTarget(
            agent_id=agent_id,
            capability_id=capability_id,
            dispatch_url=f"http://private-{agent_id}:8000",
            dispatch_mode=dispatch_mode,
        )


def _ready(local_storage, registry_service):
    run_id = "run_task_builder"
    registry_service.init_registry(run_id)
    artifact_id = "artifact_candidate_context"
    local_storage.write_json(
        local_storage.run_key(run_id, "candidate_context_table.json"),
        {
            "artifact_id": artifact_id,
            "run_id": run_id,
            "schema_version": "v1",
            "candidate_records": [],
            "context_build_status": "ok",
            "test_only_raw_sequence": "ACDEFGHIKLMNPQRSTVWYACDEFGHIK",
            "test_only_pdb_body": "HEADER PRIVATE_PDB_SENTINEL",
            "test_only_api_key": "sk-private-secret-sentinel",
        },
    )
    registry_service.update_active(
        run_id, candidate_context_table_id=artifact_id
    )
    proposal = OrchestratorRoutingProposal(
        loop_decision="dispatch_next_workers",
        decisions=[
            {
                "agent_id": AGENT_ID_STEP6,
                "capability_id": CAP_STEP6_DEVELOPABILITY,
                "objective": "Assess developability",
                "selection_reason": "Requested by the user.",
                "priority": "high",
            }
        ],
        decision_summary="Assess developability.",
    )
    result = validate_orchestrator_routing(
        run_id=run_id,
        proposal=proposal,
        discovery=_Discovery(),
        storage=local_storage,
        registry=registry_service,
    )
    return run_id, result.ready_decisions[0]


def _request_from_task(task):
    return WorkerExecutionRequest.model_validate_json(task.message["content"]["text"])


def test_task_round_trips_request_metadata_and_all_identity_fields(
    local_storage, registry_service
):
    run_id, validated = _ready(local_storage, registry_service)
    prepared = build_orchestrator_worker_task(
        run_id=run_id,
        routing_plan_id="wrp_builder",
        validated=validated,
    )
    request = _request_from_task(prepared.task)
    metadata = A2ATaskMetadata.model_validate(prepared.task.metadata)

    assert prepared.task.id == request.task_id == metadata.task_id
    assert prepared.decision.task_id == request.task_id
    assert request.routing_plan_id == metadata.routing_plan_id == "wrp_builder"
    assert request.routing_decision_id == metadata.routing_decision_id
    assert request.agent_id == metadata.agent_id == validated.decision.agent_id
    assert request.capability_id == metadata.capability_id
    assert request.created_by == metadata.created_by == "step_04_orchestrator"
    assert request.orchestrator_routing_decision.planned_status == "run"
    assert request.orchestrator_routing_decision.dispatch_mode == "python_a2a"
    assert request.orchestrator_routing_decision.deterministic_gate_status == "passed"
    assert request.orchestrator_routing_decision.expected_outputs == [
        "structured_liability_summary"
    ]


def test_task_contains_only_compact_refs_and_url_stays_in_memory(
    local_storage, registry_service
):
    run_id, validated = _ready(local_storage, registry_service)
    prepared = build_orchestrator_worker_task(
        run_id=run_id,
        routing_plan_id="wrp_private",
        validated=validated,
    )
    request = _request_from_task(prepared.task)
    ref = request.input_projection.input_artifact_refs[
        "candidate_context_table"
    ]
    assert ref.model_dump(exclude_none=True) == {
        "artifact_id": "artifact_candidate_context",
        "run_id": run_id,
        "artifact_type": "candidate_context_table",
        "artifact_role": "candidate_context_table",
        "schema_version": "v1",
        "entity_type": "candidate",
        "selection_mode": "all_in_artifact",
        "field_keys": ["candidate_records"],
        "can_read_from_db": True,
    }
    assert request.input_projection.compact_inputs == {}
    assert request.input_projection.runtime_refs == {}
    serialized = request.model_dump_json() + json.dumps(prepared.task.metadata)
    assert prepared.dispatch_target.dispatch_url.startswith("http://private-")
    for forbidden in (
        prepared.dispatch_target.dispatch_url,
        "storage_path",
        "context_build_status",
        "raw_query",
        "ACDEFGHIKLMNPQRSTVWYACDEFGHIK",
        "HEADER PRIVATE_PDB_SENTINEL",
        "sk-private-secret-sentinel",
    ):
        assert forbidden.lower() not in serialized.lower()


@pytest.mark.parametrize("status", ["waiting_for_dependencies", "blocked_missing_dependency"])
def test_non_ready_runtime_decision_raises(
    local_storage, registry_service, status
):
    run_id, validated = _ready(local_storage, registry_service)
    validated.decision = validated.decision.model_copy(
        update={"validation_status": status}
    )
    with pytest.raises(ValueError, match="requires_ready_decision"):
        build_orchestrator_worker_task(
            run_id=run_id,
            routing_plan_id="wrp_not_ready",
            validated=validated,
        )


def test_rejected_decision_raises_and_builder_has_no_send_path(
    local_storage, registry_service
):
    run_id, _ = _ready(local_storage, registry_service)
    rejected = RejectedRoutingDecision(
        routing_decision_id="route_rejected",
        agent_id="safe_agent",
        capability_id="safe_capability",
        reason="unknown_worker",
    )
    with pytest.raises(ValueError, match="runtime_validated_decision"):
        build_orchestrator_worker_task(
            run_id=run_id,
            routing_plan_id="wrp_rejected",
            validated=rejected,
        )
    source = inspect.getsource(build_orchestrator_worker_task)
    assert "send_task" not in source
    assert "send_task_async" not in source
    assert "A2AClient" not in source


def test_builder_rejects_run_identity_mismatch(local_storage, registry_service):
    run_id, validated = _ready(local_storage, registry_service)
    with pytest.raises(ValueError, match="run_id_mismatch"):
        build_orchestrator_worker_task(
            run_id=f"{run_id}_other",
            routing_plan_id="wrp_wrong_run",
            validated=validated,
        )


def test_builder_rejects_non_python_a2a_dispatch_mode(
    local_storage, registry_service
):
    run_id, validated = _ready(local_storage, registry_service)
    validated.dispatch_target = DispatchTarget(
        agent_id=validated.decision.agent_id,
        capability_id=validated.decision.capability_id,
        dispatch_url=validated.dispatch_target.dispatch_url,
        dispatch_mode="local_call",
    )
    with pytest.raises(ValueError, match="dispatch_mode_invalid"):
        build_orchestrator_worker_task(
            run_id=run_id,
            routing_plan_id="wrp_bad_mode",
            validated=validated,
        )


def test_builder_rejects_expected_output_contract_drift(
    local_storage, registry_service
):
    run_id, validated = _ready(local_storage, registry_service)
    validated.decision = validated.decision.model_copy(
        update={"expected_output_artifact_names": ["tampered_output"]}
    )
    with pytest.raises(ValueError, match="expected_outputs_mismatch"):
        build_orchestrator_worker_task(
            run_id=run_id,
            routing_plan_id="wrp_output_drift",
            validated=validated,
        )
