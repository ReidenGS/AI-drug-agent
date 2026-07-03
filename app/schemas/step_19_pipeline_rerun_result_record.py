"""Step 19 — optional pipeline rerun result scaffold record."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class RerunArtifactRefs(BaseModel):
    candidate_context_table_id: Optional[str] = None
    structured_liability_summary_id: Optional[str] = None
    prepared_structure_input_package_id: Optional[str] = None
    structure_prediction_and_interface_results_id: Optional[str] = None
    structural_variant_and_compound_screening_results_id: Optional[str] = None
    scoring_table_id: Optional[str] = None
    scoring_validation_exceptions_id: Optional[str] = None
    ranked_candidate_shortlist_id: Optional[str] = None
    scientific_evidence_table_id: Optional[str] = None
    patent_prior_art_table_id: Optional[str] = None
    ip_risk_integrated_shortlist_id: Optional[str] = None


class RerunTaskResult(BaseModel):
    redesign_task_id: str
    source_candidate_id: Optional[str] = None
    rerun_status: Literal["completed", "completed_with_warnings", "skipped", "failed"] = "skipped"
    rerun_start_step: Literal[
        "step_05",
        "step_06",
        "step_07",
        "step_08",
        "step_09",
        "step_10",
        "step_11",
        "step_12",
        "step_13",
        "step_14",
        "step_15",
        "other",
    ] = "other"
    rerun_end_step: Literal[
        "step_06",
        "step_07",
        "step_08",
        "step_09",
        "step_10",
        "step_11",
        "step_12",
        "step_13",
        "step_14",
        "step_15",
        "step_16",
        "other",
    ] = "other"
    new_or_updated_candidate_ids: list[str] = Field(default_factory=list)
    rerun_artifact_refs: RerunArtifactRefs = Field(default_factory=RerunArtifactRefs)
    comparison_ref: Optional[str] = None
    skip_or_failure_reason: Optional[
        Literal[
            "missing_candidate_id",
            "missing_redesign_task",
            "missing_required_input",
            "rerun_step_failed",
            "tool_unavailable",
            "no_rerun_required",
            "other",
        ]
    ] = "no_rerun_required"


class RerunWarning(BaseModel):
    warning_type: Literal[
        "partial_rerun",
        "missing_input",
        "failed_step",
        "tool_unavailable",
        "comparison_unavailable",
        "other",
    ]
    related_redesign_task_id: Optional[str] = None
    related_candidate_id: Optional[str] = None
    message: str


class PipelineRerunResultRecord(BaseModel):
    run_id: str
    step_id: str = "step_19"
    created_at: str
    rerun_status: Literal[
        "completed",
        "completed_with_warnings",
        "partial",
        "skipped",
        "failed",
    ] = "skipped"
    rerun_iteration_id: Optional[str] = None
    source_redesign_task_record_id: str
    rerun_task_results: list[RerunTaskResult] = Field(default_factory=list)
    rerun_warnings: list[RerunWarning] = Field(default_factory=list)
    rerun_notes: Optional[str] = None
