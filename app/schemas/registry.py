"""Run artifact registry — tracks active artifacts and snapshots."""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field


class ActiveArtifacts(BaseModel):
    raw_request_record_id: Optional[str] = None
    structured_query_id: Optional[str] = None
    input_readiness_status_id: Optional[str] = None
    run_step_plan_id: Optional[str] = None
    candidate_context_table_id: Optional[str] = None
    structured_liability_summary_id: Optional[str] = None
    prepared_structure_input_package_id: Optional[str] = None
    structure_prediction_and_interface_results_id: Optional[str] = None
    structure_variant_and_compound_screening_id: Optional[str] = None
    scoring_handoff_id: Optional[str] = None
    scoring_validation_id: Optional[str] = None
    ranking_table_id: Optional[str] = None
    scientific_evidence_table_id: Optional[str] = None
    patent_prior_art_table_id: Optional[str] = None


class RunArtifactRegistry(BaseModel):
    run_id: str
    run_artifact_registry_id: str
    version: int = 1
    created_at: str
    updated_at: str
    active_artifacts: ActiveArtifacts = Field(default_factory=ActiveArtifacts)
