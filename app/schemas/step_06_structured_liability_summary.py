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
