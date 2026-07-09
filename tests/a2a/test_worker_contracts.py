"""Turn A — WorkerExecutionRequest / WorkerExecutionResult contract schemas.

These exercise the real production schema (extra="forbid" on every model). No
test-only relaxations are introduced.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.a2a.contracts import (
    A2ATaskMetadata,
    InputArtifactRef,
    InputProjection,
    OrchestratorRoutingDecisionRef,
    RuntimeRef,
    ToolCallSummary,
    WorkerArtifactRef,
    WorkerExecutionRequest,
    WorkerExecutionResult,
    WorkerRequestSpec,
    WorkerStatusSummary,
)


def _valid_request_kwargs() -> dict:
    return dict(
        run_id="run_123",
        task_id="task_step6_001",
        routing_plan_id="wrp_001",
        routing_decision_id="route_developability_prefiltering",
        agent_id="step_06_developability_agent",
        capability_id="step_06_developability",
        created_by="step_04_orchestrator_planner",
        worker_request=WorkerRequestSpec(objective="Run developability pre-filtering."),
        orchestrator_routing_decision=OrchestratorRoutingDecisionRef(
            planned_status="run",
            dispatch_mode="python_a2a",
            expected_outputs=["structured_liability_summary"],
        ),
        input_projection=InputProjection(
            compact_inputs={"target_name": "HER2", "candidate_count": 3},
            input_artifact_refs={
                "candidate_context_table": InputArtifactRef(
                    artifact_id="candidate_context_table_abc",
                    run_id="run_123",
                    artifact_type="candidate_context_table",
                    field_keys=["candidate_id", "target_name"],
                )
            },
            runtime_refs={
                "candidate:cand_her2_001:material:mat_pdb_1": RuntimeRef(
                    **{"$ref": "candidate:cand_her2_001:material:mat_pdb_1"},
                    ref_type="artifact_field",
                    expected_runtime_type="pdb_file",
                )
            },
        ),
    )


# ── 8. unknown field on the request is rejected ─────────────────────────────
def test_request_rejects_unknown_field():
    with pytest.raises(ValidationError):
        WorkerExecutionRequest(**_valid_request_kwargs(), some_unknown_field="x")


def test_valid_request_round_trips():
    req = WorkerExecutionRequest(**_valid_request_kwargs())
    assert req.payload_type == "worker_execution_request"
    assert req.privacy_constraints.no_raw_sequence is True
    # RuntimeRef $ref alias survives dump under the alias name.
    dumped = req.model_dump(by_alias=True)
    ref = dumped["input_projection"]["runtime_refs"][
        "candidate:cand_her2_001:material:mat_pdb_1"
    ]
    assert ref["$ref"] == "candidate:cand_her2_001:material:mat_pdb_1"


# ── 9. raw-looking forbidden fields are rejected on request AND result ───────
_FORBIDDEN_FIELDS = [
    "raw_sequence",
    "pdb_body",
    "api_key",
    "raw_tooluniverse_payload",
    "full_prompt",
    "raw_llm_response",
    "fasta",
    "cif_body",
    "a3m",
]


@pytest.mark.parametrize("field", _FORBIDDEN_FIELDS)
def test_request_rejects_raw_forbidden_fields(field):
    with pytest.raises(ValidationError):
        WorkerExecutionRequest(**_valid_request_kwargs(), **{field: "SENSITIVE"})


@pytest.mark.parametrize("field", _FORBIDDEN_FIELDS)
def test_result_rejects_raw_forbidden_fields(field):
    base = dict(
        run_id="run_123",
        task_id="task_step6_001",
        agent_id="step_06_developability_agent",
        capability_id="step_06_developability",
        execution_status="completed",
        result_status="partial",
        output_artifact_refs={
            "structured_liability_summary": WorkerArtifactRef(
                artifact_id="structured_liability_summary_xyz",
                artifact_type="structured_liability_summary",
                storage_key="structured_liability_summary.json",
            )
        },
        compact_summary={"candidate_count": 3},
        tool_call_summary=ToolCallSummary(attempted=6, success=5, failed=1),
    )
    # Sanity: base is valid without the forbidden field.
    WorkerExecutionResult(**base)
    with pytest.raises(ValidationError):
        WorkerExecutionResult(**base, **{field: "SENSITIVE"})


# ── nested models also forbid unknown fields ────────────────────────────────
def test_input_artifact_ref_rejects_raw_body_field():
    with pytest.raises(ValidationError):
        InputArtifactRef(
            artifact_id="a",
            run_id="run_123",
            artifact_type="candidate_context_table",
            pdb_body="ATOM ...",
        )


def test_runtime_ref_rejects_inline_raw_value():
    with pytest.raises(ValidationError):
        RuntimeRef(
            **{"$ref": "material:mat_seq_001"},
            expected_runtime_type="protein_sequence",
            raw_sequence="EVQLVESGGG...",
        )


# ── A2ATaskMetadata carries only compact identifiers ────────────────────────
def test_task_metadata_is_compact_identifiers_only():
    meta = A2ATaskMetadata(
        run_id="run_123",
        task_id="task_step6_001",
        routing_plan_id="wrp_001",
        routing_decision_id="route_developability_prefiltering",
        agent_id="step_06_developability_agent",
        capability_id="step_06_developability",
        created_by="step_04_orchestrator_planner",
    )
    assert meta.adc_payload_type == "worker_execution_request"
    with pytest.raises(ValidationError):
        A2ATaskMetadata(
            run_id="run_123",
            task_id="t",
            routing_plan_id="p",
            routing_decision_id="d",
            agent_id="a",
            capability_id="c",
            created_by="o",
            candidate_context_table={"candidates": []},  # artifact body not allowed
        )


# ── compact-dict privacy guard: reject raw-looking nested keys ──────────────
def test_compact_inputs_rejects_raw_sequence_key():
    with pytest.raises(ValidationError):
        InputProjection(compact_inputs={"raw_sequence": "EVQLVESGGG..."})


def test_compact_inputs_rejects_deep_nested_api_key():
    with pytest.raises(ValidationError):
        InputProjection(compact_inputs={"nested": {"api_key": "sk-secret"}})


def test_runtime_ref_safe_summary_rejects_pdb_body():
    with pytest.raises(ValidationError):
        RuntimeRef(
            **{"$ref": "material:mat_pdb_1"},
            expected_runtime_type="pdb_file",
            safe_summary={"pdb_body": "ATOM      1  N   GLY A   1 ..."},
        )


def test_runtime_ref_safe_summary_rejects_deep_nested_a3m():
    with pytest.raises(ValidationError):
        RuntimeRef(
            **{"$ref": "material:mat_msa_1"},
            expected_runtime_type="a3m_alignment",
            safe_summary={"alignment": {"a3m": ">seq\nEVQ..."}},
        )


def _result_kwargs(**overrides) -> dict:
    base = dict(
        run_id="run_123",
        task_id="task_step6_001",
        agent_id="step_06_developability_agent",
        capability_id="step_06_developability",
        execution_status="completed",
        result_status="partial",
        compact_summary={"candidate_count": 3},
        tool_call_summary=ToolCallSummary(attempted=6, success=5, failed=1),
    )
    base.update(overrides)
    return base


def test_compact_summary_rejects_api_key():
    with pytest.raises(ValidationError):
        WorkerExecutionResult(**_result_kwargs(compact_summary={"api_key": "sk-secret"}))


def test_compact_summary_rejects_deep_nested_raw_tooluniverse_payload():
    with pytest.raises(ValidationError):
        WorkerExecutionResult(
            **_result_kwargs(
                compact_summary={"tool": {"raw_tooluniverse_payload": {"foo": "bar"}}}
            )
        )


def test_safe_compact_summaries_still_pass():
    safe = {"length": 438, "sha256_prefix": "abc123", "alphabet": "protein"}
    # All three guarded fields accept a legitimate compact fingerprint.
    InputProjection(compact_inputs=dict(safe))
    RuntimeRef(
        **{"$ref": "material:mat_seq_001"},
        expected_runtime_type="protein_sequence",
        safe_summary=dict(safe),
    )
    WorkerExecutionResult(**_result_kwargs(compact_summary=dict(safe)))


def test_worker_status_summary_defaults_and_forbid():
    summ = WorkerStatusSummary(agent_id="step_06_developability_agent")
    assert summ.availability == "available"
    assert summ.agent_failure_reason == "none"
    with pytest.raises(ValidationError):
        WorkerStatusSummary(agent_id="x", endpoint_url="http://x:1")
