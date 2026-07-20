"""Deterministic infrastructure tests; no worker, LLM, MCP, or dispatch."""

from __future__ import annotations

import json
import socket
import threading
from types import SimpleNamespace

import pytest
from flask import Flask, jsonify
from werkzeug.serving import make_server

from app.a2a.agent_cards import (
    AGENT_ID_PATENT_EVIDENCE,
    AGENT_ID_STEP5,
    AGENT_ID_STEP6,
    AGENT_ID_STRUCTURE,
    CAP_PATENT_EVIDENCE_WORKFLOW,
    CAP_STEP5_CANDIDATE_CONTEXT,
    CAP_STEP6_DEVELOPABILITY,
    CAP_STRUCTURE_DESIGN_WORKFLOW,
    AdcAgentContract,
    AgentCapabilityContract,
    ArtifactFieldRequirement,
    ContractArtifactRef,
    build_patent_evidence_agent_card,
    build_step5_agent_card,
    build_step6_agent_card,
    build_structure_agent_card,
    parse_adc_agent_contract,
)
from app.a2a.orchestrator_discovery import (
    DispatchTarget,
    DispatchTargetValidationError,
    ExpectedWorkerEndpoint,
    WorkerDiscoveryService,
    WorkerUnavailableError,
)
from app.a2a.orchestrator_routing_validation import validate_orchestrator_routing
from app.a2a.orchestrator_task_builder import build_orchestrator_worker_task
from app.schemas.worker_routing_plan import OrchestratorRoutingProposal


class _FrozenDiscovery:
    """Deterministic full-card authority fixture, not a live worker."""

    def __init__(self, contracts: list[AdcAgentContract], unavailable=()):
        self.workers = {
            contract.agent_id: SimpleNamespace(
                is_available=contract.agent_id not in unavailable,
                contract=contract,
            )
            for contract in contracts
        }

    def get_full_card_cache(self, run_id):
        return SimpleNamespace(workers=self.workers)

    def resolve_dispatch_target(
        self, run_id, *, agent_id, capability_id, dispatch_mode="python_a2a"
    ):
        worker = self.workers.get(agent_id)
        if worker is None:
            raise DispatchTargetValidationError("unknown")
        if not worker.is_available:
            raise WorkerUnavailableError("unavailable")
        if capability_id not in {
            cap.capability_id for cap in worker.contract.capabilities
        }:
            raise DispatchTargetValidationError("capability")
        return DispatchTarget(
            agent_id=agent_id,
            capability_id=capability_id,
            dispatch_url=f"http://{agent_id}.internal",
            dispatch_mode=dispatch_mode,
        )


def _contracts():
    return [
        parse_adc_agent_contract(build_step5_agent_card("http://step5")),
        parse_adc_agent_contract(build_step6_agent_card("http://step6")),
        parse_adc_agent_contract(build_structure_agent_card("http://structure")),
        parse_adc_agent_contract(build_patent_evidence_agent_card("http://patent")),
    ]


def _proposal(*routes, loop="dispatch_next_workers"):
    return OrchestratorRoutingProposal(
        loop_decision=loop,
        decisions=[
            {
                "agent_id": agent,
                "capability_id": capability,
                "objective": objective,
                "selection_reason": "Selected from the discovered catalog.",
                "priority": "normal",
            }
            for agent, capability, objective in routes
        ],
        decision_summary="Deterministic test proposal.",
    )


def _init_run(registry_service, name="run_routing_validation"):
    registry_service.init_registry(name)
    return name


def _persist(
    storage,
    registry,
    run_id,
    *,
    name,
    path,
    fields,
    artifact_id=None,
    body_artifact_id=None,
    body_run_id=None,
):
    artifact_id = artifact_id or f"artifact_{name}"
    body = {
        "artifact_id": body_artifact_id or artifact_id,
        "run_id": body_run_id or run_id,
        "schema_version": "v1",
        **fields,
    }
    storage.write_json(storage.run_key(run_id, path), body)
    registry.update_active(run_id, **{f"{name}_id": artifact_id})
    return body


def _persist_step5_inputs(storage, registry, run_id):
    _persist(
        storage,
        registry,
        run_id,
        name="raw_request_record",
        path="inputs/raw_request_record.json",
        fields={
            "raw_user_query": "safe compact query",
            "user_provided_context": {},
            "uploaded_files": [],
        },
    )
    _persist(
        storage,
        registry,
        run_id,
        name="structured_query",
        path="inputs/structured_query.json",
        fields={
            "mentioned_entities": {},
            "referenced_inputs": [],
            "normalized_entities": [],
            "entity_decompositions": [],
            "task_intent": {},
            "requested_outputs": [],
            "user_constraints": [],
            "canonical_query": "safe compact query",
        },
    )


def _validate(storage, registry, run_id, proposal, discovery=None):
    return validate_orchestrator_routing(
        run_id=run_id,
        proposal=proposal,
        discovery=discovery or _FrozenDiscovery(_contracts()),
        storage=storage,
        registry=registry,
    )


def _by_capability(result):
    return {item.decision.capability_id: item for item in result.decisions}


def test_step5_step6_missing_candidate_builds_card_derived_dependency(
    local_storage, registry_service
):
    run_id = _init_run(registry_service)
    _persist_step5_inputs(local_storage, registry_service, run_id)
    result = _validate(
        local_storage,
        registry_service,
        run_id,
        _proposal(
            (AGENT_ID_STEP5, CAP_STEP5_CANDIDATE_CONTEXT, "Build context"),
            (AGENT_ID_STEP6, CAP_STEP6_DEVELOPABILITY, "Assess developability"),
        ),
    )
    by_cap = _by_capability(result)
    assert by_cap[CAP_STEP5_CANDIDATE_CONTEXT].decision.validation_status == "ready"
    assert (
        by_cap[CAP_STEP6_DEVELOPABILITY].decision.validation_status
        == "waiting_for_dependencies"
    )
    assert [edge.artifact_name for edge in result.dependency_edges] == [
        "candidate_context_table"
    ]
    assert [item.decision.capability_id for item in result.ready_decisions] == [
        CAP_STEP5_CANDIDATE_CONTEXT
    ]


def test_step5_patent_dependency_is_derived_from_artifact_contracts(
    local_storage, registry_service
):
    run_id = _init_run(registry_service, "run_step5_patent_contract_dag")
    _persist_step5_inputs(local_storage, registry_service, run_id)
    result = _validate(
        local_storage,
        registry_service,
        run_id,
        _proposal(
            (AGENT_ID_STEP5, CAP_STEP5_CANDIDATE_CONTEXT, "Build context"),
            (
                AGENT_ID_PATENT_EVIDENCE,
                CAP_PATENT_EVIDENCE_WORKFLOW,
                "Review evidence and patent prior art",
            ),
        ),
    )
    by_cap = _by_capability(result)
    assert by_cap[CAP_STEP5_CANDIDATE_CONTEXT].decision.validation_status == "ready"
    patent = by_cap[CAP_PATENT_EVIDENCE_WORKFLOW]
    assert patent.decision.validation_status == "waiting_for_dependencies"
    assert "candidate_context_table" not in patent.input_artifact_refs
    assert [edge.model_dump() for edge in result.dependency_edges] == [
        {
            "artifact_name": "candidate_context_table",
            "producer_agent_id": AGENT_ID_STEP5,
            "producer_capability_id": CAP_STEP5_CANDIDATE_CONTEXT,
            "consumer_agent_id": AGENT_ID_PATENT_EVIDENCE,
            "consumer_capability_id": CAP_PATENT_EVIDENCE_WORKFLOW,
        }
    ]


def test_patent_is_ready_from_existing_candidate_without_selected_producer(
    local_storage, registry_service
):
    run_id = _init_run(registry_service, "run_patent_existing_candidate")
    _persist_step5_inputs(local_storage, registry_service, run_id)
    _persist(
        local_storage,
        registry_service,
        run_id,
        name="candidate_context_table",
        path="candidate_context_table.json",
        fields={
            "candidate_records": [],
            "downstream_query_hints": [],
            "context_build_status": "ok",
        },
    )
    result = _validate(
        local_storage,
        registry_service,
        run_id,
        _proposal(
            (
                AGENT_ID_PATENT_EVIDENCE,
                CAP_PATENT_EVIDENCE_WORKFLOW,
                "Review evidence and patent prior art",
            )
        ),
    )
    patent = _by_capability(result)[CAP_PATENT_EVIDENCE_WORKFLOW]
    assert patent.decision.validation_status == "ready"
    assert set(patent.input_artifact_refs) == {
        "structured_query",
        "candidate_context_table",
    }
    assert result.dependency_edges == []
    assert [item.decision.capability_id for item in result.ready_decisions] == [
        CAP_PATENT_EVIDENCE_WORKFLOW
    ]


@pytest.mark.parametrize("existing_status", ["ok", "failed"])
def test_selected_producer_wins_over_existing_candidate_artifact(
    local_storage, registry_service, existing_status
):
    run_id = _init_run(registry_service, f"run_selected_producer_{existing_status}")
    _persist_step5_inputs(local_storage, registry_service, run_id)
    _persist(
        local_storage,
        registry_service,
        run_id,
        name="candidate_context_table",
        path="candidate_context_table.json",
        fields={
            "candidate_records": [],
            "context_build_status": existing_status,
        },
    )
    result = _validate(
        local_storage,
        registry_service,
        run_id,
        _proposal(
            (AGENT_ID_STEP5, CAP_STEP5_CANDIDATE_CONTEXT, "Rebuild context"),
            (AGENT_ID_STEP6, CAP_STEP6_DEVELOPABILITY, "Assess developability"),
        ),
    )
    by_cap = _by_capability(result)
    assert by_cap[CAP_STEP5_CANDIDATE_CONTEXT].decision.validation_status == "ready"
    consumer = by_cap[CAP_STEP6_DEVELOPABILITY]
    assert consumer.decision.validation_status == "waiting_for_dependencies"
    assert "candidate_context_table" not in consumer.input_artifact_refs
    assert [edge.artifact_name for edge in result.dependency_edges] == [
        "candidate_context_table"
    ]
    ready_tasks = [
        build_orchestrator_worker_task(
            run_id=run_id,
            routing_plan_id="wrp_selected_producer",
            validated=item,
        )
        for item in result.ready_decisions
    ]
    assert [item.decision.capability_id for item in ready_tasks] == [
        CAP_STEP5_CANDIDATE_CONTEXT
    ]


def test_step5_structure_dependency_is_derived_from_cards(
    local_storage, registry_service
):
    run_id = _init_run(registry_service, "run_structure_dag")
    _persist_step5_inputs(local_storage, registry_service, run_id)
    result = _validate(
        local_storage,
        registry_service,
        run_id,
        _proposal(
            (AGENT_ID_STEP5, CAP_STEP5_CANDIDATE_CONTEXT, "Build context"),
            (AGENT_ID_STRUCTURE, CAP_STRUCTURE_DESIGN_WORKFLOW, "Run structure"),
        ),
    )
    structure = _by_capability(result)[CAP_STRUCTURE_DESIGN_WORKFLOW]
    assert structure.decision.validation_status == "waiting_for_dependencies"
    assert structure.decision.dependency_artifact_names == [
        "candidate_context_table"
    ]


def test_valid_candidate_artifact_makes_step6_ready_with_card_fields(
    local_storage, registry_service
):
    run_id = _init_run(registry_service, "run_step6_ready")
    _persist(
        local_storage,
        registry_service,
        run_id,
        name="candidate_context_table",
        path="candidate_context_table.json",
        fields={"candidate_records": [], "context_build_status": "ok"},
    )
    result = _validate(
        local_storage,
        registry_service,
        run_id,
        _proposal((AGENT_ID_STEP6, CAP_STEP6_DEVELOPABILITY, "Assess")),
    )
    item = result.ready_decisions[0]
    ref = item.input_artifact_refs["candidate_context_table"]
    assert ref.entity_type == "candidate"
    assert ref.selection_mode == "all_in_artifact"
    assert ref.field_keys == ["candidate_records"]
    assert "storage" not in ref.model_dump_json()


def test_candidate_readiness_failed_blocks_step6_and_structure_without_producer(
    local_storage, registry_service
):
    run_id = _init_run(registry_service, "run_candidate_failed")
    _persist_step5_inputs(local_storage, registry_service, run_id)
    _persist(
        local_storage,
        registry_service,
        run_id,
        name="candidate_context_table",
        path="candidate_context_table.json",
        fields={
            "candidate_records": [],
            "downstream_query_hints": {},
            "context_build_status": "failed",
        },
    )
    result = _validate(
        local_storage,
        registry_service,
        run_id,
        _proposal(
            (AGENT_ID_STEP6, CAP_STEP6_DEVELOPABILITY, "Assess"),
            (AGENT_ID_STRUCTURE, CAP_STRUCTURE_DESIGN_WORKFLOW, "Run structure"),
        ),
    )
    assert result.ready_decisions == []
    assert {
        (item.decision.capability_id, item.decision.reason)
        for item in result.decisions
    } == {
        (CAP_STEP6_DEVELOPABILITY, "required_artifact_not_ready"),
        (CAP_STRUCTURE_DESIGN_WORKFLOW, "required_artifact_not_ready"),
    }
    assert all(
        "candidate_context_table" not in item.input_artifact_refs
        for item in result.decisions
    )


def test_candidate_readiness_partial_is_consumable_and_selected_optional_is_omitted(
    local_storage, registry_service
):
    run_id = _init_run(registry_service, "run_candidate_partial")
    _persist_step5_inputs(local_storage, registry_service, run_id)
    _persist(
        local_storage,
        registry_service,
        run_id,
        name="candidate_context_table",
        path="candidate_context_table.json",
        fields={
            "candidate_records": [],
            "downstream_query_hints": {},
            "context_build_status": "partial",
        },
    )
    _persist(
        local_storage,
        registry_service,
        run_id,
        name="structured_liability_summary",
        path="structured_liability_summary.json",
        fields={"prefilter_status": "completed"},
    )
    result = _validate(
        local_storage,
        registry_service,
        run_id,
        _proposal(
            (AGENT_ID_STEP6, CAP_STEP6_DEVELOPABILITY, "Reassess"),
            (AGENT_ID_STRUCTURE, CAP_STRUCTURE_DESIGN_WORKFLOW, "Run structure"),
        ),
    )
    assert {
        item.decision.capability_id for item in result.ready_decisions
    } == {CAP_STEP6_DEVELOPABILITY, CAP_STRUCTURE_DESIGN_WORKFLOW}
    structure = _by_capability(result)[CAP_STRUCTURE_DESIGN_WORKFLOW]
    assert "candidate_context_table" in structure.input_artifact_refs
    assert "structured_liability_summary" not in structure.input_artifact_refs
    assert result.dependency_edges == []


@pytest.mark.parametrize(
    ("case_name", "status_present", "status_value"),
    [("missing", False, None), ("integer", True, 7), ("list", True, ["ok"])],
)
def test_missing_or_wrong_type_declared_readiness_status_fails_closed(
    local_storage, registry_service, case_name, status_present, status_value
):
    run_id = _init_run(
        registry_service, f"run_candidate_status_invalid_{case_name}"
    )
    fields = {"candidate_records": []}
    if status_present:
        fields["context_build_status"] = status_value
    _persist(
        local_storage,
        registry_service,
        run_id,
        name="candidate_context_table",
        path="candidate_context_table.json",
        fields=fields,
    )
    result = _validate(
        local_storage,
        registry_service,
        run_id,
        _proposal((AGENT_ID_STEP6, CAP_STEP6_DEVELOPABILITY, "Assess")),
    )
    item = result.decisions[0]
    assert item.decision.validation_status == "blocked_missing_dependency"
    assert item.decision.reason == "required_artifact_not_ready"
    assert item.input_artifact_refs == {}
    assert result.warnings == ["candidate_context_table:artifact_not_ready"]


def test_optional_declared_readiness_not_ready_is_omitted_without_blocking(
    local_storage, registry_service
):
    run_id = _init_run(registry_service, "run_optional_status_not_ready")
    _persist(
        local_storage,
        registry_service,
        run_id,
        name="structured_liability_summary",
        path="structured_liability_summary.json",
        fields={"prefilter_status": "failed"},
    )
    base_cap = _contracts()[0].capabilities[0]
    optional = ContractArtifactRef(
        artifact_name="structured_liability_summary",
        storage_path="structured_liability_summary.json",
        readiness_status_field="prefilter_status",
        ready_status_values=["completed", "partial"],
    )
    capability = base_cap.model_copy(
        update={
            "capability_id": "optional_readiness_consumer",
            "required_input_artifacts": [],
            "optional_input_artifacts": [optional],
            "required_artifact_fields": {},
            "output_artifacts": [
                ContractArtifactRef(
                    artifact_name="candidate_context_table",
                    storage_path="candidate_context_table.json",
                )
            ],
        }
    )
    result = _validate(
        local_storage,
        registry_service,
        run_id,
        _proposal(
            ("optional_consumer", "optional_readiness_consumer", "Consume optional")
        ),
        _FrozenDiscovery([_custom_contract("optional_consumer", capability)]),
    )
    item = result.ready_decisions[0]
    assert item.input_artifact_refs == {}
    assert result.warnings == [
        "structured_liability_summary:optional_artifact_not_ready"
    ]


def test_structure_ready_when_required_valid_and_optional_missing(
    local_storage, registry_service
):
    run_id = _init_run(registry_service, "run_structure_ready")
    _persist_step5_inputs(local_storage, registry_service, run_id)
    _persist(
        local_storage,
        registry_service,
        run_id,
        name="candidate_context_table",
        path="candidate_context_table.json",
        fields={
            "candidate_records": [],
            "downstream_query_hints": {},
            "context_build_status": "ok",
        },
    )
    result = _validate(
        local_storage,
        registry_service,
        run_id,
        _proposal((AGENT_ID_STRUCTURE, CAP_STRUCTURE_DESIGN_WORKFLOW, "Run structure")),
    )
    item = result.ready_decisions[0]
    assert item.decision.validation_status == "ready"
    assert "structured_liability_summary" not in item.input_artifact_refs


@pytest.mark.parametrize(
    "prefilter_status",
    ["completed", "completed_with_missing_lanes", "partial"],
)
def test_real_structure_optional_liability_ready_status_is_referenced(
    local_storage, registry_service, prefilter_status
):
    run_id = _init_run(registry_service, f"run_liability_{prefilter_status}")
    _persist_step5_inputs(local_storage, registry_service, run_id)
    _persist(
        local_storage,
        registry_service,
        run_id,
        name="candidate_context_table",
        path="candidate_context_table.json",
        fields={
            "candidate_records": [],
            "downstream_query_hints": [],
            "context_build_status": "ok",
        },
    )
    _persist(
        local_storage,
        registry_service,
        run_id,
        name="structured_liability_summary",
        path="structured_liability_summary.json",
        fields={"prefilter_status": prefilter_status},
    )
    result = _validate(
        local_storage,
        registry_service,
        run_id,
        _proposal(
            (AGENT_ID_STRUCTURE, CAP_STRUCTURE_DESIGN_WORKFLOW, "Run structure")
        ),
    )
    item = result.ready_decisions[0]
    assert "structured_liability_summary" in item.input_artifact_refs
    assert result.warnings == []


def test_real_structure_optional_failed_liability_is_omitted_not_blocking(
    local_storage, registry_service
):
    run_id = _init_run(registry_service, "run_liability_failed")
    _persist_step5_inputs(local_storage, registry_service, run_id)
    _persist(
        local_storage,
        registry_service,
        run_id,
        name="candidate_context_table",
        path="candidate_context_table.json",
        fields={
            "candidate_records": [],
            "downstream_query_hints": [],
            "context_build_status": "ok",
        },
    )
    _persist(
        local_storage,
        registry_service,
        run_id,
        name="structured_liability_summary",
        path="structured_liability_summary.json",
        fields={"prefilter_status": "failed"},
    )
    result = _validate(
        local_storage,
        registry_service,
        run_id,
        _proposal(
            (AGENT_ID_STRUCTURE, CAP_STRUCTURE_DESIGN_WORKFLOW, "Run structure")
        ),
    )
    item = result.ready_decisions[0]
    assert item.decision.validation_status == "ready"
    assert "structured_liability_summary" not in item.input_artifact_refs
    assert result.warnings == [
        "structured_liability_summary:optional_artifact_not_ready"
    ]


def test_corrupt_optional_artifact_is_omitted_with_compact_warning(
    local_storage, registry_service
):
    run_id = _init_run(registry_service, "run_optional_corrupt")
    _persist_step5_inputs(local_storage, registry_service, run_id)
    _persist(
        local_storage,
        registry_service,
        run_id,
        name="candidate_context_table",
        path="candidate_context_table.json",
        fields={
            "candidate_records": [],
            "downstream_query_hints": {},
            "context_build_status": "ok",
        },
    )
    _persist(
        local_storage,
        registry_service,
        run_id,
        name="structured_liability_summary",
        path="structured_liability_summary.json",
        fields={"candidate_liability_results": []},
        body_run_id="private_wrong_run",
    )
    result = _validate(
        local_storage,
        registry_service,
        run_id,
        _proposal((AGENT_ID_STRUCTURE, CAP_STRUCTURE_DESIGN_WORKFLOW, "Run structure")),
    )
    item = result.ready_decisions[0]
    assert "structured_liability_summary" not in item.input_artifact_refs
    assert result.warnings == [
        "structured_liability_summary:optional_artifact_invalid"
    ]
    assert "private_wrong_run" not in json.dumps(result.warnings)


def test_missing_required_without_producer_is_blocked(local_storage, registry_service):
    run_id = _init_run(registry_service, "run_missing_no_producer")
    result = _validate(
        local_storage,
        registry_service,
        run_id,
        _proposal((AGENT_ID_STEP6, CAP_STEP6_DEVELOPABILITY, "Assess")),
    )
    item = result.decisions[0]
    assert item.decision.validation_status == "blocked_missing_dependency"
    assert result.ready_decisions == []


def test_blocked_producer_propagates_block_to_consumer(
    local_storage, registry_service
):
    run_id = _init_run(registry_service, "run_blocked_producer")
    result = _validate(
        local_storage,
        registry_service,
        run_id,
        _proposal(
            (AGENT_ID_STEP5, CAP_STEP5_CANDIDATE_CONTEXT, "Build context"),
            (AGENT_ID_STEP6, CAP_STEP6_DEVELOPABILITY, "Assess"),
        ),
    )
    by_cap = _by_capability(result)
    assert (
        by_cap[CAP_STEP5_CANDIDATE_CONTEXT].decision.validation_status
        == "blocked_missing_dependency"
    )
    consumer = by_cap[CAP_STEP6_DEVELOPABILITY]
    assert consumer.decision.validation_status == "blocked_missing_dependency"
    assert consumer.decision.reason == "dependency_producer_blocked"
    assert result.ready_decisions == []


@pytest.mark.parametrize(
    ("tamper", "expected_code"),
    [
        ("registry_id", "artifact_id_mismatch"),
        ("body_artifact_id", "artifact_id_mismatch"),
        ("body_run_id", "artifact_run_id_mismatch"),
        ("required_fields", "artifact_required_fields_missing"),
    ],
)
def test_corrupt_required_artifact_fails_closed_without_sensitive_audit(
    local_storage, registry_service, tamper, expected_code
):
    run_id = _init_run(registry_service, f"run_corrupt_{tamper}")
    artifact_id = "secret_registry_identity"
    body_id = artifact_id
    body_run = run_id
    fields = {"candidate_records": [], "context_build_status": "ok"}
    if tamper == "registry_id":
        body_id = "secret_old_identity"
    elif tamper == "body_artifact_id":
        body_id = "secret_tampered_identity"
    elif tamper == "body_run_id":
        body_run = "secret_other_run"
    elif tamper == "required_fields":
        fields = {"wrong_field": []}
    _persist(
        local_storage,
        registry_service,
        run_id,
        name="candidate_context_table",
        path="candidate_context_table.json",
        fields=fields,
        artifact_id=artifact_id,
        body_artifact_id=body_id,
        body_run_id=body_run,
    )
    result = _validate(
        local_storage,
        registry_service,
        run_id,
        _proposal((AGENT_ID_STEP6, CAP_STEP6_DEVELOPABILITY, "Assess")),
    )
    assert result.decisions[0].decision.validation_status == "blocked_missing_dependency"
    audit = json.dumps(
        {
            "decision": result.decisions[0].decision.model_dump(),
            "warnings": result.warnings,
        }
    )
    assert expected_code in audit
    for secret in (artifact_id, body_id, body_run, "candidate_context_table.json"):
        assert secret not in audit


def test_unknown_capability_unavailable_and_duplicates_are_rejected(
    local_storage, registry_service
):
    run_id = _init_run(registry_service, "run_rejections")
    contracts = _contracts()
    proposal = _proposal(
        ("safe_unknown_worker", "safe_cap", "Unknown"),
        (AGENT_ID_STEP5, "safe_unknown_cap", "Unknown capability"),
        (AGENT_ID_STEP6, CAP_STEP6_DEVELOPABILITY, "Unavailable"),
        (AGENT_ID_STEP5, CAP_STEP5_CANDIDATE_CONTEXT, "Duplicate one"),
        (AGENT_ID_STEP5, CAP_STEP5_CANDIDATE_CONTEXT, "Duplicate two"),
    )
    result = _validate(
        local_storage,
        registry_service,
        run_id,
        proposal,
        _FrozenDiscovery(contracts, unavailable={AGENT_ID_STEP6}),
    )
    reasons = [item.reason for item in result.rejected_decisions]
    assert reasons == [
        "unknown_worker",
        "unknown_capability",
        "rejected_unavailable",
        "duplicate_route",
        "duplicate_route",
    ]
    assert result.decisions == []


def test_dispatch_target_validation_failure_has_distinct_compact_reason(
    local_storage, registry_service
):
    class _InvalidTargetDiscovery(_FrozenDiscovery):
        def resolve_dispatch_target(self, *args, **kwargs):
            raise DispatchTargetValidationError(
                "private endpoint and raw target validation detail"
            )

    run_id = _init_run(registry_service, "run_invalid_dispatch_target")
    result = _validate(
        local_storage,
        registry_service,
        run_id,
        _proposal((AGENT_ID_STEP6, CAP_STEP6_DEVELOPABILITY, "Assess")),
        _InvalidTargetDiscovery(_contracts()),
    )
    assert result.decisions == []
    assert result.rejected_decisions[0].reason == "dispatch_target_invalid"
    audit = result.rejected_decisions[0].model_dump_json()
    assert "private endpoint" not in audit
    assert "raw target" not in audit


def test_unsafe_text_is_rejected_without_raw_audit(local_storage, registry_service):
    run_id = _init_run(registry_service, "run_unsafe")
    sentinel = "NVIDIA_API_KEY=super-secret-value"
    proposal = OrchestratorRoutingProposal(
        loop_decision="dispatch_next_workers",
        decisions=[
            {
                "agent_id": AGENT_ID_STEP5,
                "capability_id": CAP_STEP5_CANDIDATE_CONTEXT,
                "objective": sentinel,
                "selection_reason": "raw ToolUniverse payload",
                "priority": "normal",
            }
        ],
        decision_summary="Unsafe proposal test.",
    )
    result = _validate(local_storage, registry_service, run_id, proposal)
    audit = json.dumps([item.model_dump() for item in result.rejected_decisions])
    assert result.rejected_decisions[0].reason == "unsafe_llm_output"
    assert sentinel not in audit
    assert "ToolUniverse" not in audit


@pytest.mark.parametrize("field", ["objective", "selection_reason"])
def test_short_uppercase_sequence_like_routing_text_is_rejected(
    local_storage, registry_service, field
):
    run_id = _init_run(registry_service, f"run_short_sequence_{field}")
    sentinel = "ACDEFGHIKLMNPQRSTVWY"
    decision = {
        "agent_id": AGENT_ID_STEP6,
        "capability_id": CAP_STEP6_DEVELOPABILITY,
        "objective": "Assess developability",
        "selection_reason": "Requested by user",
        "priority": "normal",
    }
    decision[field] = sentinel
    result = _validate(
        local_storage,
        registry_service,
        run_id,
        OrchestratorRoutingProposal(
            loop_decision="dispatch_next_workers",
            decisions=[decision],
            decision_summary="Privacy boundary test.",
        ),
    )
    assert result.decisions == []
    assert result.rejected_decisions[0].reason == "unsafe_llm_output"
    assert sentinel not in result.rejected_decisions[0].model_dump_json()


@pytest.mark.parametrize("field", ["agent_id", "capability_id"])
def test_unsafe_identity_is_not_copied_into_rejected_audit(
    local_storage, registry_service, field
):
    run_id = _init_run(registry_service, f"run_unsafe_{field}")
    decision = {
        "agent_id": AGENT_ID_STEP5,
        "capability_id": CAP_STEP5_CANDIDATE_CONTEXT,
        "objective": "Build context",
        "selection_reason": "Selected safely.",
        "priority": "normal",
    }
    sentinel = "/private/storage/raw_payload.json"
    decision[field] = sentinel
    result = _validate(
        local_storage,
        registry_service,
        run_id,
        OrchestratorRoutingProposal(
            loop_decision="dispatch_next_workers",
            decisions=[decision],
            decision_summary="Unsafe identity test.",
        ),
    )
    rejected = result.rejected_decisions[0]
    assert rejected.reason == "unsafe_llm_output"
    assert rejected.agent_id is None
    assert rejected.capability_id is None
    assert sentinel not in rejected.model_dump_json()


@pytest.mark.parametrize(
    "loop",
    [
        "wait_for_dependencies",
        "route_to_final_response",
        "request_user_input",
        "repair_or_retry",
        "stop_cannot_satisfy",
    ],
)
def test_non_dispatch_loop_with_decision_rejects_every_decision_before_artifacts(
    local_storage, registry_service, loop
):
    run_id = _init_run(registry_service, f"run_invalid_{loop}")
    result = _validate(
        local_storage,
        registry_service,
        run_id,
        _proposal(
            (AGENT_ID_STEP5, CAP_STEP5_CANDIDATE_CONTEXT, "Build"),
            loop=loop,
        ),
    )
    assert result.plan_error_codes == ["invalid_loop_decision"]
    assert result.decisions == []
    assert result.ready_decisions == []
    assert [item.reason for item in result.rejected_decisions] == [
        "invalid_loop_decision"
    ]


@pytest.mark.parametrize(
    "loop",
    [
        "wait_for_dependencies",
        "route_to_final_response",
        "request_user_input",
        "repair_or_retry",
        "stop_cannot_satisfy",
    ],
)
def test_non_dispatch_loop_without_decisions_is_valid_and_taskless(
    local_storage, registry_service, loop
):
    run_id = _init_run(registry_service, f"run_valid_{loop}")
    result = _validate(
        local_storage,
        registry_service,
        run_id,
        OrchestratorRoutingProposal(
            loop_decision=loop,
            decisions=[],
            decision_summary="No new worker dispatch.",
        ),
    )
    assert result.plan_error_codes == []
    assert result.decisions == []
    assert result.rejected_decisions == []
    assert result.ready_decisions == []


def test_empty_dispatch_is_plan_level_invalid(local_storage, registry_service):
    run_id = _init_run(registry_service, "run_empty_dispatch")

    empty_dispatch = _validate(
        local_storage,
        registry_service,
        run_id,
        OrchestratorRoutingProposal(
            loop_decision="dispatch_next_workers",
            decisions=[],
            decision_summary="Empty proposal for deterministic audit.",
        ),
    )
    assert empty_dispatch.plan_error_codes == ["invalid_loop_decision"]
    assert empty_dispatch.decisions == []


def test_unknown_artifact_registry_field_fails_closed(
    local_storage, registry_service
):
    run_id = _init_run(registry_service, "run_unknown_registry_field")
    base_cap = _contracts()[0].capabilities[0]
    unknown = ContractArtifactRef(
        artifact_name="not_registered_artifact",
        storage_path="not_registered.json",
    )
    capability = base_cap.model_copy(
        update={
            "capability_id": "unknown_registry_consumer",
            "required_input_artifacts": [unknown],
            "required_artifact_fields": {
                "not_registered_artifact": ArtifactFieldRequirement(
                    required_field_keys=["value"]
                )
            },
        }
    )
    discovery = _FrozenDiscovery(
        [_custom_contract("unknown_registry_agent", capability)]
    )
    result = _validate(
        local_storage,
        registry_service,
        run_id,
        _proposal(
            (
                "unknown_registry_agent",
                "unknown_registry_consumer",
                "Validate registry field",
            )
        ),
        discovery,
    )
    assert result.decisions[0].decision.validation_status == "blocked_missing_dependency"
    assert result.warnings == [
        "not_registered_artifact:unknown_artifact_registry_field"
    ]


@pytest.mark.parametrize("failure", ["missing", "unreadable", "schema_invalid"])
def test_registry_failure_is_compact_and_blocks_all_target_valid_decisions(
    local_storage, registry_service, failure
):
    class _BrokenRegistry:
        def get(self, run_id):
            if failure == "missing":
                raise FileNotFoundError("/private/registry/current.json")
            if failure == "unreadable":
                raise json.JSONDecodeError("secret registry body", "private", 0)
            from app.schemas.registry import RunArtifactRegistry

            return RunArtifactRegistry.model_validate(
                {"private_registry_body": "secret-id"}
            )

    class _StorageMustNotBeRead:
        def __getattr__(self, name):
            raise AssertionError(f"artifact storage accessed during {failure}: {name}")

    run_id = f"run_registry_{failure}"
    result = validate_orchestrator_routing(
        run_id=run_id,
        proposal=_proposal(
            (AGENT_ID_STEP5, CAP_STEP5_CANDIDATE_CONTEXT, "Build"),
            (AGENT_ID_STEP6, CAP_STEP6_DEVELOPABILITY, "Assess"),
        ),
        discovery=_FrozenDiscovery(_contracts()),
        storage=_StorageMustNotBeRead(),
        registry=_BrokenRegistry(),
    )
    assert result.warnings == ["run_registry_unavailable"]
    assert result.ready_decisions == []
    assert {
        (item.decision.validation_status, item.decision.reason)
        for item in result.decisions
    } == {("blocked_missing_dependency", "run_registry_unavailable")}
    audit = json.dumps(
        {
            "decisions": [item.decision.model_dump() for item in result.decisions],
            "warnings": result.warnings,
        }
    )
    for forbidden in (
        "/private/registry/current.json",
        "secret registry body",
        "private_registry_body",
        "secret-id",
    ):
        assert forbidden not in audit


def _custom_contract(agent_id, capability):
    base = _contracts()[0]
    return base.model_copy(
        update={"agent_id": agent_id, "capabilities": [capability]}
    )


def test_output_producer_conflict_rejects_all_writers_and_blocks_consumer(
    local_storage, registry_service
):
    run_id = _init_run(registry_service, "run_ambiguous")
    base_cap = _contracts()[0].capabilities[0]
    output_x = ContractArtifactRef(
        artifact_name="candidate_context_table",
        storage_path="candidate_context_table.json",
    )
    producer_a = base_cap.model_copy(
        update={"capability_id": "produce_a", "required_input_artifacts": [],
                "required_artifact_fields": {}, "output_artifacts": [output_x]}
    )
    producer_b = producer_a.model_copy(update={"capability_id": "produce_b"})
    consumer = base_cap.model_copy(
        update={
            "capability_id": "consume_x",
            "required_input_artifacts": [output_x],
            "required_artifact_fields": {
                    "candidate_context_table": ArtifactFieldRequirement(
                        required_field_keys=["value"]
                    )
            },
            "output_artifacts": [ContractArtifactRef(artifact_name="y", storage_path="y.json")],
        }
    )
    contracts = [
        _custom_contract("producer_a", producer_a),
        _custom_contract("producer_b", producer_b),
        _custom_contract("consumer", consumer),
    ]
    result = _validate(
        local_storage,
        registry_service,
        run_id,
        _proposal(
            ("producer_a", "produce_a", "Produce"),
            ("producer_b", "produce_b", "Produce"),
            ("consumer", "consume_x", "Consume"),
        ),
        _FrozenDiscovery(contracts),
    )
    consumer_result = _by_capability(result)["consume_x"]
    assert consumer_result.decision.validation_status == "blocked_missing_dependency"
    assert result.ready_decisions == []
    assert result.dependency_edges == []
    assert result.warnings == [
        "candidate_context_table:ambiguous_output_producer"
    ]
    conflicts = [
        item
        for item in result.rejected_decisions
        if item.reason == "output_artifact_conflict"
    ]
    assert {(item.agent_id, item.capability_id) for item in conflicts} == {
        ("producer_a", "produce_a"),
        ("producer_b", "produce_b"),
    }
    for rejected in conflicts:
        with pytest.raises(ValueError, match="runtime_validated_decision"):
            build_orchestrator_worker_task(
                run_id=run_id,
                routing_plan_id="wrp_conflict",
                validated=rejected,
            )

    run_existing = _init_run(registry_service, "run_conflict_existing")
    _persist(
        local_storage,
        registry_service,
        run_existing,
        name="candidate_context_table",
        path="candidate_context_table.json",
        fields={"value": "safe"},
    )
    existing = _validate(
        local_storage,
        registry_service,
        run_existing,
        _proposal(
            ("producer_a", "produce_a", "Produce"),
            ("producer_b", "produce_b", "Produce"),
            ("consumer", "consume_x", "Consume"),
        ),
        _FrozenDiscovery(contracts),
    )
    assert {
        item.capability_id
        for item in existing.rejected_decisions
        if item.reason == "output_artifact_conflict"
    } == {"produce_a", "produce_b"}
    assert all(
        item.decision.capability_id not in {"produce_a", "produce_b"}
        for item in existing.ready_decisions
    )
    existing_consumer = _by_capability(existing)["consume_x"]
    assert existing_consumer.decision.validation_status == "blocked_missing_dependency"
    assert existing_consumer.decision.reason == "dependency_producer_conflict"
    assert existing_consumer.input_artifact_refs == {}


@pytest.mark.parametrize("existing_artifact", [False, True])
def test_output_storage_path_conflict_rejects_differently_named_writers(
    local_storage, registry_service, existing_artifact
):
    run_id = _init_run(
        registry_service, f"run_path_conflict_{existing_artifact}"
    )
    if existing_artifact:
        _persist(
            local_storage,
            registry_service,
            run_id,
            name="candidate_context_table",
            path="candidate_context_table.json",
            fields={
                "candidate_records": [],
                "context_build_status": "ok",
            },
        )
    base_cap = _contracts()[0].capabilities[0]
    producer_a = base_cap.model_copy(
        update={
            "capability_id": "path_producer_a",
            "required_input_artifacts": [],
            "required_artifact_fields": {},
            "output_artifacts": [
                ContractArtifactRef(
                    artifact_name="candidate_context_table",
                    storage_path="shared.json",
                )
            ],
        }
    )
    producer_b = producer_a.model_copy(
        update={
            "capability_id": "path_producer_b",
            "output_artifacts": [
                ContractArtifactRef(
                    artifact_name="structured_liability_summary",
                    storage_path="shared.json",
                )
            ],
        }
    )
    result = _validate(
        local_storage,
        registry_service,
        run_id,
        _proposal(
            ("path_agent_a", "path_producer_a", "Produce A"),
            ("path_agent_b", "path_producer_b", "Produce B"),
        ),
        _FrozenDiscovery(
            [
                _custom_contract("path_agent_a", producer_a),
                _custom_contract("path_agent_b", producer_b),
            ]
        ),
    )
    assert result.decisions == []
    assert result.ready_decisions == []
    assert {
        item.capability_id
        for item in result.rejected_decisions
        if item.reason == "output_artifact_conflict"
    } == {"path_producer_a", "path_producer_b"}
    assert result.warnings == [
        "output_storage_path:ambiguous_output_producer"
    ]
    assert "shared.json" not in json.dumps(result.warnings)


def test_dependency_cycle_fails_closed(local_storage, registry_service):

    run_cycle = _init_run(registry_service, "run_cycle")
    ref_a = ContractArtifactRef(
        artifact_name="candidate_context_table",
        storage_path="candidate_context_table.json",
    )
    ref_b = ContractArtifactRef(
        artifact_name="structured_liability_summary",
        storage_path="structured_liability_summary.json",
    )
    cap_a = AgentCapabilityContract(
        capability_id="cycle_cap_a", skill_name="A", capability_summary="A",
        required_input_artifacts=[ref_b],
        required_artifact_fields={
            "structured_liability_summary": ArtifactFieldRequirement(
                required_field_keys=["v"]
            )
        },
        output_artifacts=[ref_a], uses_llm=False, uses_mcp=False,
    )
    cap_b = AgentCapabilityContract(
        capability_id="cycle_cap_b", skill_name="B", capability_summary="B",
        required_input_artifacts=[ref_a],
        required_artifact_fields={
            "candidate_context_table": ArtifactFieldRequirement(
                required_field_keys=["v"]
            )
        },
        output_artifacts=[ref_b], uses_llm=False, uses_mcp=False,
    )
    cycle = _validate(
        local_storage,
        registry_service,
        run_cycle,
        _proposal(("cycle_a", "cycle_cap_a", "A"), ("cycle_b", "cycle_cap_b", "B")),
        _FrozenDiscovery([
            _custom_contract("cycle_a", cap_a),
            _custom_contract("cycle_b", cap_b),
        ]),
    )
    assert {item.decision.reason for item in cycle.decisions} == {"dependency_cycle"}
    assert cycle.ready_decisions == []


def _free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def test_real_http_discovery_cache_to_validation_and_task_builder(
    local_storage, registry_service, monkeypatch
):
    """Real AgentCard + health HTTP; no A2A task send or worker handler."""
    for variable in ("NO_PROXY", "no_proxy"):
        monkeypatch.setenv(variable, "127.0.0.1,localhost")
    for variable in (
        "HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy"
    ):
        monkeypatch.delenv(variable, raising=False)
    port = _free_port()
    url = f"http://127.0.0.1:{port}"
    card = build_step5_agent_card(url)
    hits = {"card": 0, "health": 0}
    app = Flask(__name__)

    @app.get("/.well-known/agent.json")
    def _card():
        hits["card"] += 1
        return jsonify(card.to_dict())

    @app.get("/health")
    def _health():
        hits["health"] += 1
        return jsonify(
            {
                "status": "ok",
                "agent_id": AGENT_ID_STEP5,
                "capabilities": [CAP_STEP5_CANDIDATE_CONTEXT],
            }
        )

    server = make_server("127.0.0.1", port, app)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        run_id = _init_run(registry_service, "run_http_integration")
        _persist_step5_inputs(local_storage, registry_service, run_id)
        service = WorkerDiscoveryService(
            expected_workers=[
                ExpectedWorkerEndpoint(
                    AGENT_ID_STEP5, (CAP_STEP5_CANDIDATE_CONTEXT,), url
                )
            ],
            storage=local_storage,
            registry=registry_service,
            discovery_timeout_seconds=2,
            health_timeout_seconds=2,
        )
        service.discover_for_run(run_id)
        counts_after_first = dict(hits)
        service.discover_for_run(run_id)
        assert hits == counts_after_first == {"card": 2, "health": 1}
        result = _validate(
            local_storage,
            registry_service,
            run_id,
            _proposal((AGENT_ID_STEP5, CAP_STEP5_CANDIDATE_CONTEXT, "Build")),
            service,
        )
        prepared = build_orchestrator_worker_task(
            run_id=run_id,
            routing_plan_id="wrp_http_integration",
            validated=result.ready_decisions[0],
        )
        assert prepared.task.id == prepared.decision.task_id
        assert hits == {"card": 2, "health": 1}
    finally:
        server.shutdown()
        thread.join(timeout=2)


def test_discovery_rejects_malicious_card_path_before_artifact_storage_access():
    malicious_path = "../../private/registry.json"
    card = build_step5_agent_card("http://step5.invalid")
    card.capabilities["adc_agent_contract"]["capabilities"][0][
        "required_input_artifacts"
    ][0]["storage_path"] = malicious_path
    artifact_storage_calls: list[tuple[str, str]] = []

    class _StorageTrap:
        def exists(self, key):
            artifact_storage_calls.append(("exists", key))
            raise AssertionError("artifact storage must not be consulted")

        def read_json(self, key):
            artifact_storage_calls.append(("read_json", key))
            raise AssertionError("artifact storage must not be consulted")

    service = WorkerDiscoveryService(
        expected_workers=[],
        storage=_StorageTrap(),
        registry=SimpleNamespace(),
        discovery_timeout_seconds=1,
        health_timeout_seconds=1,
    )
    expected = ExpectedWorkerEndpoint(
        AGENT_ID_STEP5,
        (CAP_STEP5_CANDIDATE_CONTEXT,),
        "http://step5.invalid",
    )
    with pytest.raises(Exception) as excinfo:
        service._validate_card(card, expected, expected.endpoint_url)
    assert getattr(excinfo.value, "code", None) == "adc_agent_contract_invalid"
    assert artifact_storage_calls == []
    assert malicious_path not in repr(artifact_storage_calls)
