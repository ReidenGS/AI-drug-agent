"""Step 6 reviewer-facing liability interpretation.

Covers the additive structured fields that translate tool-level outputs into
explicit assessed / not-assessed / signal labels: lane assessment_status,
risk_label, not_assessed_reason, interpreted_findings,
missing_or_unassessed_items, and candidate-level context_completeness /
label / recommended_action / interpretation_summary.

Unit tests exercise the pure interpretation helpers; the end-to-end test
runs the real Step 5 → Step 6 production flow on sequence-only antibody
input and inspects the persisted normalized artifact (and asserts no raw
sequence leakage).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.agents.candidate_context_agent import CandidateContextAgent
from app.agents.developability_agent import DevelopabilityAgent
from app.agents.step_06_interpretation import (
    derive_candidate_interpretation,
    derive_lane_assessment,
    derive_missing_lane_assessment,
    interpreted_findings_from_flags,
)
from app.mcp.client import LocalMCPClient
from app.schemas.step_02_structured_query import (
    SourceRawRequestRef,
    StructuredQuery,
    TaskIntent,
)
from app.schemas.step_06_structured_liability_summary import LaneResult
from app.services.input_readiness_service import InputReadinessService
from app.services.intake_service import IntakeService
from app.services.tool_inventory_service import ToolInventoryService
from app.services.workflow_setup_service import WorkflowSetupService
from app.utils.ids import new_artifact_id
from app.utils.time import now_iso


PROJECT_ROOT = Path(__file__).resolve().parents[2].parent
DEFAULT_XLSX = PROJECT_ROOT / "项目文件" / "ToolUniversity_inventory_v0.2.xlsx"
HEAVY = "EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGK"
LIGHT = "DIQMTQSPSSLSASVGDRVTITCRASQGISSYLNWYQQKPGK"


def _inventory_or_skip() -> ToolInventoryService:
    xlsx = os.environ.get("TOOL_INVENTORY_XLSX", str(DEFAULT_XLSX))
    if not Path(xlsx).exists():
        pytest.skip(f"Inventory xlsx not at {xlsx}")
    return ToolInventoryService(xlsx)


# ── unit: lane assessment ───────────────────────────────────────────────────


def test_missing_input_lane_assessment():
    a = derive_missing_lane_assessment("payload_linker_compound_liability")
    assert a["assessment_status"] == "not_assessed_missing_input"
    assert a["risk_label"] == "not_assessed"
    assert a["not_assessed_reason"]
    items = a["missing_or_unassessed_items"]
    assert items and items[0]["item"] == "payload/linker compound SMILES"
    assert items[0]["blocking"] is False
    assert items[0]["suggested_next_input"]


def test_no_signal_lane_assessment():
    a = derive_lane_assessment(
        lane_type="antibody_protein_sequence_liability",
        plans_present=True,
        flags=[],
        lane_risk_category="low",
        any_success=True,
        all_dependency_unavailable=False,
        has_upstream_error=False,
        any_failed=False,
        tool_records=[],
    )
    assert a["assessment_status"] == "no_signal"
    assert a["risk_label"] == "low"
    assert a["interpreted_findings"] == []


def test_signal_detected_lane_assessment():
    flags = [
        {"flag_type": "motif_match", "severity": "high",
         "evidence_summary": "motif X", "source_tool": "PROSITE_scan_sequence",
         "source_ref": "tool_outputs/step_06/tc_1.json"},
    ]
    a = derive_lane_assessment(
        lane_type="antibody_protein_sequence_liability",
        plans_present=True,
        flags=flags,
        lane_risk_category="high",
        any_success=True,
        all_dependency_unavailable=False,
        has_upstream_error=False,
        any_failed=False,
        tool_records=[],
    )
    assert a["assessment_status"] == "signal_detected"
    assert a["risk_label"] == "high"
    assert a["interpreted_findings"][0]["finding_type"] == "motif_match"
    assert a["interpreted_findings"][0]["label"] == "high"


def test_upstream_error_lane_assessment_not_clean_low():
    a = derive_lane_assessment(
        lane_type="antibody_protein_sequence_liability",
        plans_present=True,
        flags=[],
        lane_risk_category="unknown",
        any_success=True,
        all_dependency_unavailable=False,
        has_upstream_error=True,
        any_failed=False,
        tool_records=[],
    )
    assert a["assessment_status"] == "partial_upstream_error"
    assert a["risk_label"] != "low"
    assert a["risk_label"] == "review"


def test_dependency_unavailable_lane_assessment():
    a = derive_lane_assessment(
        lane_type="payload_linker_compound_liability",
        plans_present=True,
        flags=[],
        lane_risk_category="unknown",
        any_success=False,
        all_dependency_unavailable=True,
        has_upstream_error=False,
        any_failed=False,
        tool_records=[],
    )
    assert a["assessment_status"] == "not_assessed_dependency_unavailable"
    assert a["risk_label"] == "not_assessed"


def test_no_plans_lane_is_missing_input():
    a = derive_lane_assessment(
        lane_type="structure_interface_quality",
        plans_present=False,
        flags=[],
        lane_risk_category="unknown",
        any_success=False,
        all_dependency_unavailable=False,
        has_upstream_error=False,
        any_failed=False,
        tool_records=[],
    )
    assert a["assessment_status"] == "not_assessed_missing_input"
    assert a["missing_or_unassessed_items"]


def test_all_skipped_with_missing_typed_input_is_not_assessed_not_failed():
    records = [
        SimpleNamespace(
            run_status="skipped",
            tool_input_summary={
                "validation_status": "skipped",
                "missing_required_fields": ["required_identifier"],
                "runtime_resolver_audit": [],
            },
            tool_output_ref=None,
            tool_call_id="tc_missing",
        ),
        SimpleNamespace(
            run_status="not_run",
            tool_input_summary={
                "validation_status": "valid",
                "missing_required_fields": [],
                "runtime_resolver_audit": [
                    {"schema_arg": "required_ref", "resolve_status": "unresolved"}
                ],
            },
            tool_output_ref=None,
            tool_call_id="tc_unresolved",
        ),
    ]

    assessment = derive_lane_assessment(
        lane_type="structure_interface_quality",
        plans_present=True,
        flags=[],
        lane_risk_category="unknown",
        any_success=False,
        all_dependency_unavailable=False,
        has_upstream_error=False,
        any_failed=False,
        tool_records=records,
    )

    assert assessment["assessment_status"] == "not_assessed_missing_input"
    assert assessment["risk_label"] == "not_assessed"
    assert "typed or invokable input" in assessment["not_assessed_reason"]
    assert assessment["missing_or_unassessed_items"]
    gap = assessment["missing_or_unassessed_items"][0]
    assert gap["item"] == "required typed tool input"
    assert gap["missing_field_names"] == ["required_identifier", "required_ref"]


def test_attempted_failed_record_is_not_reclassified_as_missing_input():
    assessment = derive_lane_assessment(
        lane_type="structure_interface_quality",
        plans_present=True,
        flags=[],
        lane_risk_category="unknown",
        any_success=False,
        all_dependency_unavailable=False,
        has_upstream_error=False,
        any_failed=True,
        tool_records=[
            SimpleNamespace(
                run_status="failed",
                tool_input_summary={
                    "validation_status": "valid",
                    "missing_required_fields": [],
                    "runtime_resolver_audit": [],
                },
                tool_output_ref=None,
                tool_call_id="tc_failed",
            )
        ],
    )

    assert assessment["assessment_status"] == "failed"
    assert assessment["risk_label"] == "unknown"


def test_policy_only_skip_is_not_reclassified_as_missing_input():
    assessment = derive_lane_assessment(
        lane_type="structure_interface_quality",
        plans_present=True,
        flags=[],
        lane_risk_category="unknown",
        any_success=False,
        all_dependency_unavailable=False,
        has_upstream_error=False,
        any_failed=False,
        tool_records=[
            SimpleNamespace(
                run_status="skipped",
                tool_input_summary={
                    "validation_status": "skipped",
                    "missing_required_fields": [],
                    "runtime_resolver_audit": [],
                },
                tool_output_ref=None,
                tool_call_id="tc_policy_skip",
            )
        ],
    )

    assert assessment["assessment_status"] == "failed"
    assert assessment["risk_label"] == "unknown"


def test_interpreted_findings_map_source_tool_call_ids():
    class _Rec:
        tool_output_ref = "tool_outputs/step_06/tc_42.json"
        tool_call_id = "tc_42"

    flags = [{
        "flag_type": "pains_alert", "severity": "high",
        "evidence_summary": "alert A", "source_tool": "DrugProps_pains_filter",
        "source_ref": "tool_outputs/step_06/tc_42.json",
    }]
    out = interpreted_findings_from_flags(flags, [_Rec()])
    assert out[0]["source_tool_call_ids"] == ["tc_42"]
    assert out[0]["source_tools"] == ["DrugProps_pains_filter"]


# ── unit: candidate aggregation ─────────────────────────────────────────────


def _lane(lane_type: str, status: str, risk: str, items=None) -> LaneResult:
    return LaneResult(
        lane_type=lane_type,
        run_status="ok" if status in {"no_signal", "signal_detected"} else "skipped",
        input_status="sufficient" if status in {"no_signal", "signal_detected"} else "missing",
        assessment_status=status,
        risk_label=risk,
        missing_or_unassessed_items=items or [],
    )


def test_candidate_partial_context_is_review_continue_with_review():
    lanes = [
        _lane("antibody_protein_sequence_liability", "no_signal", "low"),
        _lane("payload_linker_compound_liability", "not_assessed_missing_input", "not_assessed",
              items=[{"item": "SMILES", "reason": "no smiles", "blocking": False,
                      "suggested_next_input": "smiles"}]),
        _lane("antigen_protein_feature_context", "not_assessed_missing_input", "not_assessed"),
        _lane("structure_interface_quality", "not_assessed_missing_input", "not_assessed"),
        _lane("compound_bioactivity_prior_context", "not_assessed_missing_input", "not_assessed"),
    ]
    c = derive_candidate_interpretation(lanes)
    assert c["context_completeness"] == "partial"
    assert c["assessed_lane_count"] == 1
    assert c["not_assessed_lane_count"] == 4
    assert c["candidate_overall_liability_label"] == "review"
    assert c["recommended_action"] == "continue_with_review"
    assert "not fully acceptable" in c["interpretation_summary"]
    # Structured gaps aggregated, not only free text.
    assert any(i["item"] == "SMILES" for i in c["missing_or_unassessed_items"])


def test_candidate_high_signal_is_high_risk_deprioritize():
    lanes = [_lane("payload_linker_compound_liability", "signal_detected", "high")]
    c = derive_candidate_interpretation(lanes)
    assert c["candidate_overall_liability_label"] == "high-risk"
    assert c["recommended_action"] == "deprioritize"


def test_candidate_none_assessed_is_insufficient_data():
    lanes = [_lane("payload_linker_compound_liability", "not_assessed_missing_input", "not_assessed")]
    c = derive_candidate_interpretation(lanes)
    assert c["context_completeness"] == "none"
    assert c["recommended_action"] == "insufficient_data"
    assert c["candidate_overall_liability_label"] == "unknown"


def test_candidate_upstream_error_not_clean_continue():
    lanes = [
        _lane("antibody_protein_sequence_liability", "no_signal", "low"),
        _lane("payload_linker_compound_liability", "partial_upstream_error", "review"),
        _lane("antigen_protein_feature_context", "not_assessed_missing_input", "not_assessed"),
        _lane("structure_interface_quality", "not_assessed_missing_input", "not_assessed"),
        _lane("compound_bioactivity_prior_context", "not_assessed_missing_input", "not_assessed"),
    ]
    c = derive_candidate_interpretation(lanes)
    assert c["recommended_action"] != "continue"
    assert c["candidate_overall_liability_label"] == "review"
    assert "upstream_error" in c["interpretation_summary"]


# ── end-to-end: sequence-only antibody developability ───────────────────────


def _seed_sequence_run(local_storage, registry_service, workflow_state_service):
    rec = IntakeService(local_storage, registry_service, workflow_state_service).submit(
        raw_user_query="developability pre-filter on antibody heavy/light sequences",
        user_provided_context={},
    )
    run_id = rec.run_id
    sq = StructuredQuery(
        run_id=run_id,
        parsed_at=now_iso(),
        source_raw_request_ref=SourceRawRequestRef(
            raw_request_record_id=registry_service.get(run_id).active_artifacts.raw_request_record_id
        ),
        task_intent=TaskIntent(
            task_type="developability_assessment",
            primary_intent="developability_assessment",
        ),
        mentioned_entities={"antibody_candidate_text": "antibody protein sequences"},
        referenced_inputs=[
            {"id_type": "antibody_heavy_chain_sequence", "value": HEAVY, "source": "user"},
            {"id_type": "antibody_light_chain_sequence", "value": LIGHT, "source": "user"},
        ],
        missing_slots=[],
        canonical_query="developability/liability pre-filter for antibody heavy/light sequences",
    )
    sq_id = new_artifact_id("structured_query")
    local_storage.write_json(
        local_storage.run_key(run_id, "inputs/structured_query.json"),
        {"artifact_id": sq_id, **sq.model_dump()},
    )
    registry_service.update_active(run_id, structured_query_id=sq_id)
    workflow_state_service.mark(run_id, "step_02", "completed")
    InputReadinessService(local_storage, registry_service, workflow_state_service).check(run_id)
    WorkflowSetupService(local_storage, registry_service, workflow_state_service).plan(run_id)
    return run_id


def test_sequence_only_end_to_end_structured_summary(
    local_storage, registry_service, workflow_state_service
):
    inventory = _inventory_or_skip()
    run_id = _seed_sequence_run(local_storage, registry_service, workflow_state_service)
    CandidateContextAgent(
        storage=local_storage, registry=registry_service,
        workflow_state=workflow_state_service,
        mcp_client=LocalMCPClient(inventory=inventory),
    ).run(run_id)
    DevelopabilityAgent(
        storage=local_storage, registry=registry_service,
        workflow_state=workflow_state_service,
        mcp_client=LocalMCPClient(
            inventory=inventory,
            bindings={
                "PROSITE_scan_sequence": lambda **k: {"status": "mocked", "motifs": []},
                "IEDB_predict_mhci_binding": lambda **k: {"status": "mocked", "predictions": []},
            },
        ),
    ).run(run_id)

    summary = local_storage.read_json(
        local_storage.run_key(run_id, "structured_liability_summary.json")
    )
    cand = summary["candidate_liability_results"][0]
    lanes = {lr["lane_type"]: lr for lr in cand["lane_results"]}

    # Antibody lane assessed with no signal → low.
    ab = lanes["antibody_protein_sequence_liability"]
    assert ab["assessment_status"] in {"assessed", "no_signal"}
    assert ab["risk_label"] == "low"

    # Other lanes not assessed due to missing input, with structured gaps.
    for lt in (
        "payload_linker_compound_liability",
        "antigen_protein_feature_context",
        "structure_interface_quality",
        "compound_bioactivity_prior_context",
    ):
        lr = lanes[lt]
        assert lr["assessment_status"] == "not_assessed_missing_input"
        assert lr["risk_label"] == "not_assessed"
        assert lr["missing_or_unassessed_items"], f"{lt} should list structured gaps"

    # Candidate-level: incomplete context, not fully acceptable.
    assert cand["context_completeness"] == "partial"
    assert cand["assessed_lane_count"] == 1
    assert cand["candidate_overall_liability_label"] == "review"
    assert cand["recommended_action"] == "continue_with_review"
    assert cand["interpretation_summary"]
    assert cand["missing_or_unassessed_items"]

    # No raw sequence anywhere in the normalized artifact.
    blob = json.dumps(summary)
    assert HEAVY not in blob
    assert LIGHT not in blob
