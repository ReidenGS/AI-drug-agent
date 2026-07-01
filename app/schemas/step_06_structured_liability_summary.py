"""Step 6 — structured_liability_summary (DevelopabilityAgent output)."""

from __future__ import annotations

from typing import Literal, Optional
from pydantic import BaseModel, Field

from .common import ToolCallRecord


LaneType = Literal[
    "payload_linker_compound_liability",
    "antibody_protein_sequence_liability",
    "antigen_protein_feature_context",
    "structure_interface_quality",
    "compound_bioactivity_prior_context",
]


# Reviewer-facing lane interpretation (additive — translates tool execution
# into clear assessed / not-assessed / signal labels).
LaneAssessmentStatus = Literal[
    "assessed",                              # ran, interpreted, no liability signal
    "no_signal",                             # ran successfully, no signal (alias of assessed-clean)
    "signal_detected",                       # ran, interpreted liability signal(s)
    "not_assessed_missing_input",            # required typed input absent
    "not_assessed_dependency_unavailable",   # tool(s) dependency_unavailable
    "partial_upstream_error",                # ran but upstream_error envelope(s)
    "failed",                                # ran but failed (no usable output)
]
LaneRiskLabel = Literal["low", "review", "high", "unknown", "not_assessed"]


class LaneResult(BaseModel):
    lane_type: LaneType
    run_status: Literal["ok", "skipped", "partial", "failed"]
    input_status: Literal["sufficient", "insufficient", "missing"]
    selected_tools: list[str] = Field(default_factory=list)
    tool_call_records: list[ToolCallRecord] = Field(default_factory=list)
    argument_mapping_audit: list[dict] = Field(default_factory=list)
    liability_flags: list[dict] = Field(default_factory=list)
    lane_risk_category: Literal["low", "medium", "high", "unknown"] = "unknown"
    lane_summary: Optional[str] = None
    # ── Additive reviewer-facing interpretation (defaults keep old artifacts
    # valid). `assessment_status` + `risk_label` translate tool-level outputs
    # into explicit business labels; `not_assessed_reason` explains gaps;
    # `interpreted_findings` lists per-signal evidence (by reference, no raw
    # payload); `missing_or_unassessed_items` enumerates structured gaps. ──
    assessment_status: LaneAssessmentStatus = "not_assessed_missing_input"
    risk_label: LaneRiskLabel = "not_assessed"
    not_assessed_reason: Optional[str] = None
    interpreted_findings: list[dict] = Field(default_factory=list)
    missing_or_unassessed_items: list[dict] = Field(default_factory=list)


CandidatePrefilterStatus = Literal["completed", "partial", "not_run", "failed"]
PrefilterStatus = Literal["completed", "completed_with_missing_lanes", "partial", "failed"]


class CandidateLiability(BaseModel):
    candidate_id: str
    candidate_prefilter_status: CandidatePrefilterStatus = "partial"
    lane_results: list[LaneResult] = Field(default_factory=list)
    candidate_overall_liability_label: Literal["acceptable", "review", "high-risk", "unknown"] = "unknown"
    recommended_action: Literal[
        "continue", "continue_with_review", "deprioritize", "insufficient_data"
    ] = "insufficient_data"
    # ── Additive reviewer-facing candidate interpretation. `context_completeness`
    # makes explicit that a candidate with only some lanes assessed is NOT
    # fully acceptable; the counts + summary + aggregated gaps spell out what
    # was assessed vs not assessed. Defaults keep old artifacts valid. ──
    context_completeness: Literal["complete", "partial", "none"] = "none"
    assessed_lane_count: int = 0
    not_assessed_lane_count: int = 0
    interpretation_summary: Optional[str] = None
    missing_or_unassessed_items: list[dict] = Field(default_factory=list)


class StructuredLiabilitySummary(BaseModel):
    run_id: str
    step_id: str = "step_06_developability"
    created_at: str
    prefilter_status: PrefilterStatus = "partial"
    strict_filter_mode: bool = False
    candidate_liability_results: list[CandidateLiability] = Field(default_factory=list)
    missing_input_flags: list[str] = Field(default_factory=list)
    tool_output_artifacts: list[str] = Field(default_factory=list)
    selection_audit: dict = Field(default_factory=dict)
    notes: Optional[str] = None
