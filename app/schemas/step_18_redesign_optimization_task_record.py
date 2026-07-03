"""Step 18 — redesign / optimization trigger task record."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class RedesignTriggerSource(BaseModel):
    human_review_decision_record_id: str
    source_fields: list[
        Literal[
            "candidate_decisions",
            "follow_up_actions",
            "next_step_instruction",
            "review_feedback",
            "other",
        ]
    ] = Field(default_factory=list)


class OptimizationGoal(BaseModel):
    goal_type: Literal[
        "affinity",
        "stability",
        "solubility",
        "expression",
        "immunogenicity",
        "developability",
        "structure_quality",
        "payload_linker",
        "ip_risk_reduction",
        "evidence_gap_closure",
        "other",
    ] = "other"
    goal_description: Optional[str] = None
    priority: Literal["high", "medium", "low"] = "medium"
    source_ref: Optional[str] = None


class RedesignTask(BaseModel):
    redesign_task_id: str
    candidate_id: Optional[str] = None
    trigger_reason: Literal[
        "human_requested_redesign",
        "liability_issue",
        "low_confidence_scoring",
        "poor_structure_quality",
        "weak_evidence",
        "ip_risk",
        "request_more_data",
        "other",
    ] = "human_requested_redesign"
    optimization_goals: list[OptimizationGoal] = Field(default_factory=list)
    recommended_redesign_scope: Literal[
        "antibody_sequence",
        "linker_payload",
        "candidate_selection",
        "structure_modeling",
        "evidence_collection",
        "ip_review",
        "other",
        "unknown",
    ] = "unknown"
    requires_new_candidate_generation: bool = False
    requires_pipeline_rerun: bool = False
    suggested_rerun_start_step: Optional[
        Literal[
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
        ]
    ] = None
    task_status: Literal["ready", "needs_clarification", "blocked"] = "ready"
    task_notes: Optional[str] = None


class NonRedesignOutcome(BaseModel):
    reason: Optional[
        Literal[
            "no_redesign_requested",
            "proceed_to_output_package",
            "stop_run",
            "report_revision_only",
            "request_more_data_only",
            "other",
        ]
    ] = None
    next_step_instruction: Optional[
        Literal[
            "proceed_to_output_package",
            "revise_report",
            "request_more_data",
            "stop_run",
            "other",
        ]
    ] = None


class MissingRedesignInput(BaseModel):
    missing_item: Literal["candidate_id", "optimization_goal", "redesign_scope", "reviewer_rationale", "other"]
    severity: Literal["warning", "blocking"] = "warning"
    message: str


class RedesignOptimizationTaskRecord(BaseModel):
    run_id: str
    step_id: str = "step_18"
    created_at: str
    redesign_trigger_status: Literal[
        "triggered",
        "not_triggered",
        "needs_clarification",
        "skipped",
        "failed",
    ] = "not_triggered"
    trigger_source: RedesignTriggerSource
    redesign_tasks: list[RedesignTask] = Field(default_factory=list)
    non_redesign_outcome: NonRedesignOutcome = Field(default_factory=NonRedesignOutcome)
    missing_redesign_inputs: list[MissingRedesignInput] = Field(default_factory=list)
    redesign_trigger_notes: Optional[str] = None
