"""Step 17 — human-in-the-loop review decision record."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class ReviewedReportRef(BaseModel):
    llm_design_review_report_id: str
    report_storage_ref: Optional[str] = None


class ReviewerInfo(BaseModel):
    reviewer_id: Optional[str] = None
    reviewer_name: Optional[str] = None
    reviewer_role: Literal[
        "scientist",
        "professor",
        "domain_expert",
        "legal_reviewer",
        "other",
        "unknown",
    ] = "unknown"
    reviewed_at: Optional[str] = None


class ReportDecision(BaseModel):
    decision: Literal[
        "approve",
        "approve_with_conditions",
        "request_revision",
        "request_more_data",
        "reject",
        "other",
    ] = "request_more_data"
    decision_rationale: Optional[str] = None
    reviewer_confidence: Literal["high", "medium", "low", "unknown"] = "unknown"


class CandidateDecision(BaseModel):
    candidate_id: str
    source_rank: Optional[float] = None
    decision: Literal["keep", "reject", "redesign", "hold", "request_more_data", "other"]
    decision_rationale: Optional[str] = None
    reviewer_confidence: Literal["high", "medium", "low", "unknown"] = "unknown"
    recommended_next_action: Literal[
        "proceed",
        "redesign",
        "collect_more_data",
        "legal_review",
        "experimental_validation",
        "no_action",
        "other",
    ] = "no_action"
    review_notes: Optional[str] = None


class ReviewFeedback(BaseModel):
    summary: Optional[str] = None
    free_text_feedback: Optional[str] = None


class FollowUpAction(BaseModel):
    action_type: Literal[
        "redesign",
        "rerun_scoring",
        "rerun_structure_prediction",
        "collect_more_evidence",
        "legal_ip_review",
        "experimental_validation",
        "revise_report",
        "no_action",
        "other",
    ]
    related_candidate_ids: list[str] = Field(default_factory=list)
    priority: Literal["high", "medium", "low"] = "medium"
    action_notes: Optional[str] = None


class HumanReviewDecisionRecord(BaseModel):
    run_id: str
    step_id: str = "step_17"
    created_at: str
    review_status: Literal[
        "completed",
        "completed_with_conditions",
        "needs_more_information",
        "rejected",
        "skipped",
    ] = "needs_more_information"
    reviewed_report: ReviewedReportRef
    reviewer_info: ReviewerInfo = Field(default_factory=ReviewerInfo)
    report_decision: ReportDecision = Field(default_factory=ReportDecision)
    candidate_decisions: list[CandidateDecision] = Field(default_factory=list)
    review_feedback: ReviewFeedback = Field(default_factory=ReviewFeedback)
    follow_up_actions: list[FollowUpAction] = Field(default_factory=list)
    next_step_instruction: Literal[
        "proceed_to_output_package",
        "trigger_redesign",
        "revise_report",
        "request_more_data",
        "stop_run",
        "other",
    ] = "request_more_data"
    review_notes: Optional[str] = None
