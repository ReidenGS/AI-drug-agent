"""Step 4 — run_step_plan."""

from __future__ import annotations

from typing import Literal, Optional
from pydantic import BaseModel, Field


class DefaultExecutionPolicy(BaseModel):
    default_behavior: Literal["run_all_adc_main_steps"] = "run_all_adc_main_steps"
    start_step_id: str = "step_01_intake"
    end_step_id: str = "step_14_patent_ip"


class SkippedStep(BaseModel):
    step_id: str
    reason_type: str
    reason: str


PlannedStatus = Literal["run", "skip", "partial", "blocked", "wait_for_input"]


class PlannedStep(BaseModel):
    """Per-step deterministic decision derived from readiness + structured_query.

    Lane flags carry forward intent for Step 6 (compound lane) / Step 7-9
    (structure lane) / Step 13-14 (evidence + patent lanes). Downstream
    agents read these to scope their tool calls.
    """

    step_id: str
    planned_status: PlannedStatus
    reason: str
    required_artifact_refs: list[str] = Field(default_factory=list)
    lane_flags: dict[str, bool] = Field(default_factory=dict)


PlanStatus = Literal["ready_to_execute", "wait_for_input", "blocked"]


class RunStepPlan(BaseModel):
    run_id: str
    step_id: str = "step_04_workflow_setup"
    planned_at: str
    pipeline_scope: Literal["ADC"] = "ADC"
    pipeline_template_version: str = "v0.1"
    plan_status: PlanStatus = "ready_to_execute"
    default_execution_policy: DefaultExecutionPolicy = Field(default_factory=DefaultExecutionPolicy)
    planned_steps: list[PlannedStep] = Field(default_factory=list)
    skipped_steps: list[SkippedStep] = Field(default_factory=list)
    skipped_step_ids: list[str] = Field(default_factory=list)
    planning_warnings: list[str] = Field(default_factory=list)
    planning_notes: Optional[str] = None
