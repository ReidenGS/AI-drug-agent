"""Round-trip every Step schema with a minimal valid payload."""

from __future__ import annotations

from app.schemas.step_01_raw_request_record import RawRequestRecord
from app.schemas.step_02_structured_query import StructuredQuery, SourceRawRequestRef, TaskIntent
from app.schemas.step_03_input_readiness import InputReadinessStatus, SourceRefs
from app.schemas.step_04_run_step_plan import RunStepPlan
from app.schemas.step_05_candidate_context_table import CandidateContextTable
from app.schemas.step_06_structured_liability_summary import StructuredLiabilitySummary
from app.schemas.step_07_prepared_structure_input_package import PreparedStructureInputPackage
from app.schemas.step_08_structure_prediction_and_interface_results import (
    StructurePredictionAndInterfaceResults,
)
from app.schemas.step_09_structure_variant_and_compound_screening import (
    CompoundScreeningArtifact,
    ProteinDesignArtifact,
)
from app.schemas.step_10_scoring_handoff import ScoringHandoff
from app.schemas.step_11_scoring_validation import ScoringValidation
from app.schemas.step_12_ranking_table import RankingTable
from app.schemas.step_13_scientific_evidence_table import ScientificEvidenceTable
from app.schemas.step_14_patent_prior_art_table import PatentPriorArtTable


_NOW = "2026-06-15T00:00:00Z"
# Fixture value; not derived from `new_run_id()`. Schema round-trip only needs
# *some* string in this slot.
_RUN = "run_schema_fixture_001"


def test_step_01_roundtrip():
    obj = RawRequestRecord(
        run_id=_RUN,
        run_artifact_registry_id="reg_x",
        created_at=_NOW,
        raw_user_query="hello",
    )
    assert RawRequestRecord.model_validate(obj.model_dump()) == obj


def test_step_02_roundtrip():
    obj = StructuredQuery(
        run_id=_RUN,
        parsed_at=_NOW,
        source_raw_request_ref=SourceRawRequestRef(raw_request_record_id="rr1"),
        task_intent=TaskIntent(task_type="adc_design"),
    )
    assert StructuredQuery.model_validate(obj.model_dump()) == obj


def test_step_03_roundtrip():
    obj = InputReadinessStatus(
        run_id=_RUN,
        checked_at=_NOW,
        source_refs=SourceRefs(raw_request_record_id="rr1", structured_query_id="sq1"),
        input_readiness_status="ready",
    )
    assert InputReadinessStatus.model_validate(obj.model_dump()) == obj


def test_step_04_roundtrip():
    obj = RunStepPlan(run_id=_RUN, planned_at=_NOW)
    assert RunStepPlan.model_validate(obj.model_dump()) == obj


def test_steps_05_14_roundtrip():
    for model in (
        CandidateContextTable(run_id=_RUN, created_at=_NOW),
        StructuredLiabilitySummary(run_id=_RUN, created_at=_NOW),
        PreparedStructureInputPackage(run_id=_RUN, created_at=_NOW),
        StructurePredictionAndInterfaceResults(run_id=_RUN, created_at=_NOW),
        ProteinDesignArtifact(run_id=_RUN, created_at=_NOW),
        CompoundScreeningArtifact(run_id=_RUN, created_at=_NOW),
        ScoringHandoff(run_id=_RUN, created_at=_NOW),
        ScoringValidation(run_id=_RUN, created_at=_NOW),
        RankingTable(run_id=_RUN, created_at=_NOW),
        ScientificEvidenceTable(run_id=_RUN, created_at=_NOW),
        PatentPriorArtTable(run_id=_RUN, created_at=_NOW),
    ):
        assert type(model).model_validate(model.model_dump()) == model
