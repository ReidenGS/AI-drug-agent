"""Step 5 enrichment-semantics tests.

Pinned by the live Step 1→6 smoke observations:

- SAbDab can return ok WITHOUT any heavy/light sequence; Step 6 sequence
  lane must stay honest about that gap.
- ChEMBL substructure search returns chembl_id values that are upper-bound
  identity matches, NOT confirmed exact identity for the user's compound.
- ChEMBL name queries can return zero hits; that gap must be recorded.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.agents.candidate_context_agent import (
    _annotate_antibody_sabdab_outcome,
    _antibody_payload_has_sequence,
    _apply_compound_tool_enrichment,
    _record_chembl_plan_outcome,
)
from app.schemas.common import ToolCallRecord
from app.schemas.step_05_candidate_context_table import (
    CandidateRecord,
    Identifier,
    Material,
)


def _new_antibody_candidate() -> CandidateRecord:
    return CandidateRecord(
        candidate_id="cand_ab_test",
        candidate_label="trastuzumab analog",
        candidate_type="antibody",
        materials=[Material(material_id="m1", material_type="antibody_name", value="trastuzumab analog")],
        identifiers=[],
        candidate_role="user_provided_candidate",
        is_generated_candidate=False,
        context_status="partial",
    )


def _new_compound_candidate() -> CandidateRecord:
    return CandidateRecord(
        candidate_id="cand_cc_test",
        candidate_label="vc-MMAE",
        candidate_type="compound_component",
        materials=[],
        identifiers=[],
        candidate_role="user_provided_candidate",
        is_generated_candidate=False,
        context_status="partial",
    )


class _StubStorage:
    """Minimal storage that returns canned JSON for a known ref."""

    def __init__(self, files: dict[str, dict]):
        self._files = files

    def read_json(self, ref: str) -> dict:
        return self._files[ref]


# ── SAbDab antibody outcome ────────────────────────────────────────────────


def test_sabdab_ok_without_sequence_annotates_data_gap_and_note():
    cand = _new_antibody_candidate()
    storage = _StubStorage({
        "ref_ok_no_seq": {
            "output": {
                "executor": "tooluniverse",
                "status": "ok",
                "source": "SAbDab_search_structures",
                # Live shape with no usable sequence field anywhere.
                "payload": {"data": {"hits": [{"pdb_id": "1n8z", "name": "trastuzumab"}]}},
            },
        }
    })
    tc = ToolCallRecord(
        tool_call_id="tc1",
        tool_name="SAbDab_search_structures",
        run_status="success",
        tool_output_ref="ref_ok_no_seq",
    )
    _annotate_antibody_sabdab_outcome(storage=storage, record=cand, tc=tc)
    assert any(
        "sabdab_no_sequence_field" in g for g in cand.data_gaps
    ), cand.data_gaps
    assert any(
        "SAbDab" in n and "no antibody" in n.lower() for n in cand.context_notes
    ), cand.context_notes


def test_sabdab_ok_with_real_sequence_does_not_add_gap():
    cand = _new_antibody_candidate()
    long_seq = "EVQLVQSGAEVKKPGSSVKVSCKASGGTFSSYAISWVRQAPGQGLEWMGGIIPIFGTANYAQKFQG"
    storage = _StubStorage({
        "ref_ok_with_seq": {
            "output": {
                "executor": "tooluniverse",
                "status": "ok",
                "payload": {"hits": [{"heavy_chain_sequence": long_seq}]},
            }
        }
    })
    tc = ToolCallRecord(
        tool_call_id="tc2",
        tool_name="SAbDab_search_structures",
        run_status="success",
        tool_output_ref="ref_ok_with_seq",
    )
    _annotate_antibody_sabdab_outcome(storage=storage, record=cand, tc=tc)
    assert not any("sabdab_no_sequence_field" in g for g in cand.data_gaps), cand.data_gaps


def test_sabdab_dependency_unavailable_records_dep_gap():
    cand = _new_antibody_candidate()
    tc = ToolCallRecord(
        tool_call_id="tc3",
        tool_name="SAbDab_search_structures",
        run_status="dependency_unavailable",
        error_message=None,
    )
    _annotate_antibody_sabdab_outcome(storage=_StubStorage({}), record=cand, tc=tc)
    assert any("dependency_unavailable" in g for g in cand.data_gaps), cand.data_gaps


def test_antibody_payload_has_sequence_helper_thresholds():
    assert not _antibody_payload_has_sequence(None)
    assert not _antibody_payload_has_sequence({"name": "trastuzumab"})
    # too short
    assert not _antibody_payload_has_sequence({"heavy_chain_sequence": "EVQ"})
    long_seq = "EVQLVQSGAEVKKPGSSVKVSCKASGGTFSSYAISWVRQAPGQGLEWMG"
    assert _antibody_payload_has_sequence({"heavy_chain_sequence": long_seq})
    assert _antibody_payload_has_sequence({"hits": [{"vh_sequence": long_seq}]})


# ── ChEMBL substructure tagging ────────────────────────────────────────────


def test_substructure_search_marks_substructure_derived_in_notes_and_gaps():
    cand = _new_compound_candidate()
    envelope = {
        "executor": "tooluniverse",
        "status": "ok",
        "payload": {"data": {"molecules": [
            {"molecule_chembl_id": "CHEMBL2107839",
             "molecule_structures": {"canonical_smiles": "CC(N)C(=O)O"}},
        ]}},
    }
    n = _apply_compound_tool_enrichment(
        cand, envelope,
        source_artifact_id="tc_subs",
        tool_name="ChEMBL_search_substructure",
        query_kind="smiles",
        query_value="NCC(=O)O",
    )
    assert n == 1
    # Confidence lowered for substructure-derived ids.
    chembl_ids = [i for i in cand.identifiers if i.id_type == "chembl_id"]
    assert chembl_ids and chembl_ids[0].confidence == 0.5
    # Compact provenance note + gap recorded.
    assert any("substructure-derived" in n.lower() for n in cand.context_notes), cand.context_notes
    assert any("substructure_derived_not_exact_identity" in g for g in cand.data_gaps), cand.data_gaps


def test_name_search_keeps_higher_confidence_and_no_substructure_note():
    cand = _new_compound_candidate()
    envelope = {
        "executor": "tooluniverse",
        "status": "ok",
        "payload": {"data": {"molecules": [{"molecule_chembl_id": "CHEMBL999"}]}},
    }
    n = _apply_compound_tool_enrichment(
        cand, envelope,
        source_artifact_id="tc_name",
        tool_name="ChEMBL_search_molecules",
        query_kind="name",
        query_value="monomethyl auristatin E",
    )
    assert n == 1
    chembl_ids = [i for i in cand.identifiers if i.id_type == "chembl_id"]
    assert chembl_ids and chembl_ids[0].confidence == 0.8
    assert not any("substructure-derived" in note.lower() for note in cand.context_notes)


# ── ChEMBL outcome recording (zero hits, upstream_error, dep_unavailable) ──


class _StubPlan:
    def __init__(self, tool_name, query, query_kind):
        self.tool_name = tool_name
        self.query = query
        self.query_kind = query_kind
        self.query_role = "payload"
        self.material_type = "payload_name"


def test_chembl_name_zero_hit_recorded_as_data_gap():
    cand = _new_compound_candidate()
    storage = _StubStorage({
        "ref_zero": {"output": {
            "executor": "tooluniverse",
            "status": "ok",
            "payload": {"data": {"molecules": []}},
        }}
    })
    tc = ToolCallRecord(
        tool_call_id="tc_z",
        tool_name="ChEMBL_search_molecules",
        run_status="success",
        tool_output_ref="ref_zero",
        tool_output_artifact_id="tool_output_zero",
    )
    plan = _StubPlan("ChEMBL_search_molecules", "vc-MMAE", "name")
    _record_chembl_plan_outcome(storage=storage, record=cand, plan=plan, tc=tc)
    assert any(
        "zero_matches_returned" in g and "vc-MMAE" in g for g in cand.data_gaps
    ), cand.data_gaps


def test_chembl_substructure_upstream_error_recorded_as_data_gap():
    cand = _new_compound_candidate()
    storage = _StubStorage({
        "ref_err": {"output": {
            "executor": "tooluniverse",
            "status": "upstream_error",
            "error_message": "ChEMBL API returned HTTP 400",
            "source": "ChEMBL_search_substructure",
        }}
    })
    tc = ToolCallRecord(
        tool_call_id="tc_e",
        tool_name="ChEMBL_search_substructure",
        run_status="success",
        tool_output_ref="ref_err",
        tool_output_artifact_id="tool_output_err",
    )
    plan = _StubPlan("ChEMBL_search_substructure", "CCO", "smiles")
    _record_chembl_plan_outcome(storage=storage, record=cand, plan=plan, tc=tc)
    assert any(
        "upstream_error" in g and "HTTP 400" in g for g in cand.data_gaps
    ), cand.data_gaps


# ── Mutual-exclusion guarantees ────────────────────────────────────────────


def test_chembl_upstream_error_does_not_also_record_zero_matches():
    """Regression: a single ChEMBL call that returned envelope_status=
    upstream_error must NOT also produce a zero_matches_returned gap."""
    cand = _new_compound_candidate()
    storage = _StubStorage({
        "ref_upe": {"output": {
            "executor": "tooluniverse",
            "status": "upstream_error",
            "error_message": "ChEMBL API returned HTTP 400",
        }}
    })
    tc = ToolCallRecord(
        tool_call_id="tc_upe",
        tool_name="ChEMBL_search_substructure",
        run_status="success",
        tool_output_ref="ref_upe",
        tool_output_artifact_id="tool_output_upe",
    )
    plan = _StubPlan("ChEMBL_search_substructure", "CCO", "smiles")
    _record_chembl_plan_outcome(storage=storage, record=cand, plan=plan, tc=tc)
    gaps_for_cco = [
        g for g in cand.data_gaps
        if "ChEMBL_search_substructure" in g and "CCO" in g
    ]
    assert any("upstream_error" in g for g in gaps_for_cco)
    assert not any("zero_matches_returned" in g for g in gaps_for_cco), gaps_for_cco
    # Exactly one outcome row per (tool, query).
    assert len(gaps_for_cco) == 1, gaps_for_cco


def test_chembl_ok_with_empty_molecules_records_zero_matches_only():
    cand = _new_compound_candidate()
    storage = _StubStorage({
        "ref_zero": {"output": {
            "executor": "tooluniverse",
            "status": "ok",
            "payload": {"data": {"molecules": []}},
        }}
    })
    tc = ToolCallRecord(
        tool_call_id="tc_z",
        tool_name="ChEMBL_search_molecules",
        run_status="success",
        tool_output_ref="ref_zero",
        tool_output_artifact_id="tool_output_zero",
    )
    plan = _StubPlan("ChEMBL_search_molecules", "vc-MMAE", "name")
    _record_chembl_plan_outcome(storage=storage, record=cand, plan=plan, tc=tc)
    rows = [g for g in cand.data_gaps if "vc-MMAE" in g]
    assert any("zero_matches_returned" in g for g in rows), rows
    assert not any("upstream_error" in g for g in rows), rows
    assert len(rows) == 1, rows


def test_chembl_ok_with_non_empty_molecules_records_no_outcome_gap():
    cand = _new_compound_candidate()
    storage = _StubStorage({
        "ref_ok": {"output": {
            "executor": "tooluniverse",
            "status": "ok",
            "payload": {"data": {"molecules": [
                {"molecule_chembl_id": "CHEMBL55555"},
            ]}},
        }}
    })
    tc = ToolCallRecord(
        tool_call_id="tc_ok",
        tool_name="ChEMBL_search_molecules",
        run_status="success",
        tool_output_ref="ref_ok",
        tool_output_artifact_id="tool_output_ok",
    )
    plan = _StubPlan("ChEMBL_search_molecules", "monomethyl auristatin E", "name")
    _record_chembl_plan_outcome(storage=storage, record=cand, plan=plan, tc=tc)
    rows = [g for g in cand.data_gaps if "monomethyl auristatin E" in g]
    assert rows == [], (
        "successful ChEMBL call with a real hit must not append an outcome gap"
    )
    assert any(i.id_value == "CHEMBL55555" for i in cand.identifiers)


def test_chembl_dependency_unavailable_does_not_also_record_zero_matches():
    cand = _new_compound_candidate()
    tc = ToolCallRecord(
        tool_call_id="tc_dep",
        tool_name="ChEMBL_search_substructure",
        run_status="dependency_unavailable",
    )
    plan = _StubPlan("ChEMBL_search_substructure", "CCO", "smiles")
    _record_chembl_plan_outcome(storage=_StubStorage({}), record=cand, plan=plan, tc=tc)
    rows = [g for g in cand.data_gaps if "CCO" in g]
    assert any("dependency_unavailable" in g for g in rows), rows
    assert not any("zero_matches_returned" in g for g in rows), rows
    assert len(rows) == 1, rows


# ── Step 6 bioactivity summary annotates substructure-derived chembl_id ────


def test_step6_bioactivity_lane_summary_calls_out_substructure_derived(
    local_storage, registry_service, workflow_state_service,
):
    """When a chembl_id identifier is substructure-derived (marked in the
    candidate's context_notes / data_gaps), Step 6's bioactivity lane
    summary must surface that provenance so reviewers do not mistake
    upper-bound identity for confirmed identity."""
    from app.agents.developability_agent import DevelopabilityAgent  # noqa: PLC0415
    from app.mcp.client import LocalMCPClient  # noqa: PLC0415
    from app.services.intake_service import IntakeService  # noqa: PLC0415
    from app.utils.ids import new_artifact_id  # noqa: PLC0415
    from app.utils.time import now_iso  # noqa: PLC0415

    intake = IntakeService(local_storage, registry_service, workflow_state_service)
    rec = intake.submit(
        raw_user_query="bioactivity substructure provenance",
        user_provided_context={"target_or_antigen_text": "fixture"},
    )
    run_id = rec.run_id
    artifact_id = new_artifact_id("candidate_context_table")
    cct = {
        "artifact_id": artifact_id,
        "run_id": run_id,
        "step_id": "step_05_candidate_context",
        "created_at": now_iso(),
        "context_build_status": "ok",
        "candidate_records": [{
            "candidate_id": "cand_substructure",
            "candidate_label": "vc-MMAE substructure context",
            "candidate_type": "compound_component",
            "source_records": [],
            "identifiers": [{
                "id_type": "chembl_id",
                "id_value": "CHEMBL2107839",
                "source_ids": ["tc_substr"],
                "confidence": 0.5,
            }],
            "materials": [],
            "adc_links": {"target_material_ids": [], "antibody_material_ids": [],
                          "payload_material_ids": [], "linker_material_ids": [],
                          "dar_material_ids": []},
            "candidate_status": "partially_ready_for_step6",
            "candidate_role": "user_provided_candidate",
            "is_generated_candidate": False,
            "context_status": "partial",
            "data_gaps": ["chembl_id_origin:substructure_derived_not_exact_identity"],
            "missing_material_roles": [],
            "context_notes": [
                "ChEMBL substructure-derived chembl_id count=1; not confirmed exact identity"
            ],
        }],
        "missing_context_flags": [],
        "tool_call_records": [],
        "downstream_query_hints": [],
    }
    local_storage.write_json(
        local_storage.run_key(run_id, "candidate_context_table.json"), cct
    )
    registry_service.update_active(run_id, candidate_context_table_id=artifact_id)

    def bind(payload):
        def f(**_):
            return payload
        return f

    bindings = {"ChEMBL_search_activities": bind({"status": "mocked", "activities": []})}
    DevelopabilityAgent(
        storage=local_storage, registry=registry_service,
        workflow_state=workflow_state_service,
        mcp_client=LocalMCPClient(bindings=bindings),
    ).run(run_id)
    persisted = local_storage.read_json(
        local_storage.run_key(run_id, "structured_liability_summary.json")
    )
    bio = [
        lane for cand in persisted["candidate_liability_results"]
        for lane in cand["lane_results"]
        if lane["lane_type"] == "compound_bioactivity_prior_context"
    ][0]
    assert bio["run_status"] in {"ok", "partial"}, bio
    summary = (bio.get("lane_summary") or "").lower()
    assert "substructure" in summary, summary
    assert "not confirmed exact identity" in summary, summary
