"""Verify schemas accept the canonical field values listed in
ADC_Pipeline_IO_Schema_v0.1.md (post-alignment).
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.schemas.common import ToolCallRecord
from app.schemas.step_08_structure_prediction_and_interface_results import (
    CandidateStructureResult,
    ComplexPredictionPlan,
    ComplexStructureRef,
    InterfaceAnalysisRecord,
    Step8DownstreamHandoff,
    StructureConfidenceRecord,
    StructureOutputArtifact,
    StructurePredictionAndInterfaceResults,
)
from app.schemas.step_09_structure_variant_and_compound_screening import CompoundHit
from app.schemas.step_14_patent_prior_art_table import PatentPriorArtTable, PatentRecord


_NOW = "2026-06-15T00:00:00Z"
# Fixture value; not derived from `new_run_id()`. Canonical schema check only
# needs *some* string in this slot.
_RUN = "run_canonical_fixture_001"


# ── ToolCallRecord ───────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "status", ["success", "failed", "skipped", "dependency_unavailable", "partial", "pending", "not_run"]
)
def test_tool_call_record_accepts_canonical_statuses(status):
    rec = ToolCallRecord(tool_call_id="tc_1", tool_name="x", run_status=status)
    assert rec.run_status == status


def test_tool_call_record_supports_tool_output_ref():
    rec = ToolCallRecord(
        tool_call_id="tc_1",
        tool_name="x",
        run_status="success",
        tool_output_artifact_id="art_1",
        tool_output_ref="s3://bucket/adc_pilot/runs/run_x/tool_outputs/step_06/abc.json",
    )
    assert rec.tool_output_ref.startswith("s3://")


def test_tool_call_record_rejects_bogus_status():
    with pytest.raises(ValidationError):
        ToolCallRecord(tool_call_id="tc_1", tool_name="x", run_status="exploded")


# ── Step 8 confidence types + output_artifacts ───────────────────────────────

@pytest.mark.parametrize(
    "ctype", ["refinement_resolution", "crystal_density_validation", "unit_cell_consistency"]
)
def test_step_08_accepts_new_confidence_types(ctype):
    rec = StructureConfidenceRecord(confidence_type=ctype, value=1.85, source="get_refinement_resolution_by_pdb_id")
    assert rec.confidence_type == ctype


def test_step_08_output_artifact_is_structured():
    art = StructureOutputArtifact(
        artifact_id="art_1",
        related_candidate_id="cand_1",
        related_structure_input_id="si_1",
        artifact_type="refinement_or_validation_report",
        storage_ref="s3://bucket/adc_pilot/runs/run_x/tool_outputs/step_08/refinement.json",
        storage_type="s3_path",
        content_type="json",
    )
    table = StructurePredictionAndInterfaceResults(
        run_id=_RUN, created_at=_NOW, output_artifacts=[art]
    )
    assert table.output_artifacts[0].artifact_type == "refinement_or_validation_report"


def test_step_08_business_output_fields_are_additive_and_defaulted():
    result = CandidateStructureResult(
        candidate_id="cand_1",
        structure_input_id="si_1",
        run_case="existing_complex_interface_evaluation",
        run_status="ok",
    )
    assert result.complex_structure_refs == []
    assert result.interface_analysis_records == []
    assert result.downstream_handoff == Step8DownstreamHandoff()
    assert result.complex_prediction_plans == []
    assert result.missing_prediction_inputs == []


def test_step_08_accepts_complex_refs_interface_records_and_handoff():
    result = CandidateStructureResult(
        candidate_id="cand_1",
        structure_input_id="si_1",
        run_case="existing_complex_interface_evaluation",
        run_status="ok",
        complex_structure_refs=[
            ComplexStructureRef(
                source_kind="existing_pdb_complex",
                source_ref="1n8z",
                pdb_id="1n8z",
                structure_format="pdb",
                source_tool_call_id="tc_pisa",
                confidence_summary={"interface_evaluation": "success"},
            )
        ],
        interface_analysis_records=[
            InterfaceAnalysisRecord(
                source_tool="PDBePISA_get_interfaces",
                source_tool_call_id="tc_pisa",
                chain_pair={"chain_id_1": "A", "chain_id_2": "B"},
                interface_residue_count=2,
                interface_area=123.4,
                h_bond_count=2,
                source_ref="1n8z",
            )
        ],
        downstream_handoff=Step8DownstreamHandoff(
            has_complex_structure=True,
            has_interface_features=True,
            structure_for_variant_generation_ref="1n8z",
            interface_quality_available=True,
            refinement_resolution_available=True,
        ),
        complex_prediction_plans=[
            ComplexPredictionPlan(
                tool_name="NvidiaNIM_boltz2",
                input_status="selected_but_deferred",
                runtime_status="runtime_unavailable",
                can_invoke=False,
                sequence_inputs=[{"sequence_id": "antigen_seq", "chain_role": "antigen"}],
                contract_notes=["runtime deferred"],
            )
        ],
        complex_prediction_input_status="selected_but_deferred",
        prediction_runtime_status="runtime_unavailable",
    )
    table = StructurePredictionAndInterfaceResults(
        run_id=_RUN,
        created_at=_NOW,
        candidate_structure_results=[result],
    )
    dumped = table.model_dump()
    roundtrip = StructurePredictionAndInterfaceResults.model_validate(dumped)
    assert roundtrip.candidate_structure_results[0].downstream_handoff.has_complex_structure is True


# ── Step 9 compound source library ───────────────────────────────────────────

@pytest.mark.parametrize("lib", ["ZINC", "ZINC15", "ZINC22"])
def test_step_09_compound_hit_accepts_zinc_family(lib):
    hit = CompoundHit(
        compound_id="c1",
        source_library=lib,
        smiles="CCO",
        source_database_version="unknown",
        source_tool_name="ZINC_search_compounds",
        source_runtime_status="skipped",
    )
    assert hit.source_library == lib
    assert hit.source_database_version == "unknown"
    assert hit.source_tool_name == "ZINC_search_compounds"
    assert hit.source_runtime_status == "skipped"


def test_step_09_compound_hit_does_not_default_to_zinc22():
    """ToolUniverse ZINC_* wrappers must NOT silently claim ZINC22 — the field
    is optional and defaults to None."""
    hit = CompoundHit(compound_id="c1", source_library="ZINC", smiles="CCO")
    assert hit.source_database_version is None


# ── Step 14 patent record ────────────────────────────────────────────────────

def test_step_14_patent_record_supports_fda_orange_book():
    rec = PatentRecord(
        patent_record_id="pr_1",
        candidate_id="cand_1",
        matched_entity_type="drug_application_or_regulatory_reference",
        source_database="FDA_OrangeBook",
        patent_application_number="NDA-123456",
        publication_date="2024-01-01",
        filing_date="2020-06-15",
        source_url="https://www.accessdata.fda.gov/scripts/cder/ob/",
        source_ref="orange_book_raw#row_42",
        notes_limitations="Orange Book product-level fields kept in raw output ref",
    )
    table = PatentPriorArtTable(run_id=_RUN, created_at=_NOW, patent_records=[rec])
    rt = PatentPriorArtTable.model_validate(table.model_dump())
    assert rt.patent_records[0].source_database == "FDA_OrangeBook"
    assert rt.patent_records[0].matched_entity_type == "drug_application_or_regulatory_reference"
    assert rt.legal_disclaimer.startswith("For demonstration purposes only.")
