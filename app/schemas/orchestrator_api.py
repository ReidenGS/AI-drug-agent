"""Compact public response contract for the production Step 4 Orchestrator."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .orchestrator_execution_state import (
    ArtifactId,
    ArtifactStatus,
    ArtifactSummaryRef,
    ContractIdentifier,
    NextWakeupState,
    OrchestratorStatus,
    RoutingPlanId,
    RunId,
    RunStatus,
    TaskId,
)

_FORBID = ConfigDict(extra="forbid")

OrchestratorApiOutcome = Literal[
    "completed",
    "failed",
    "waiting_for_input",
    "reconciliation_required",
    "waiting",
    "unavailable",
]


class CompactDecisionCounts(BaseModel):
    model_config = _FORBID

    total: int = Field(ge=0)
    ready: int = Field(ge=0)
    pending_dependency: int = Field(ge=0)
    blocked: int = Field(ge=0)
    dispatched: int = Field(ge=0)
    completed: int = Field(ge=0)
    failed: int = Field(ge=0)
    skipped: int = Field(ge=0)
    planned: int = Field(ge=0)


class CompactTaskCounts(BaseModel):
    model_config = _FORBID

    total: int = Field(ge=0)
    not_dispatched: int = Field(ge=0)
    dispatching: int = Field(ge=0)
    dispatched: int = Field(ge=0)
    dispatch_failed: int = Field(ge=0)
    not_started: int = Field(ge=0)
    running: int = Field(ge=0)
    completed: int = Field(ge=0)
    failed: int = Field(ge=0)
    retry_tasks: int = Field(ge=0)


class CompactArtifactCounts(BaseModel):
    model_config = _FORBID

    total: int = Field(ge=0)
    missing: int = Field(ge=0)
    planned: int = Field(ge=0)
    producing: int = Field(ge=0)
    available: int = Field(ge=0)
    invalid: int = Field(ge=0)


class CompactArtifactRef(BaseModel):
    model_config = _FORBID

    artifact_name: ContractIdentifier
    status: ArtifactStatus
    artifact_id: ArtifactId | None = None
    producer_task_id: TaskId | None = None
    safe_summary_ref: ArtifactSummaryRef | None = None


class OrchestratorStep4Response(BaseModel):
    """No Task, worker result, endpoint, storage path, or artifact body."""

    model_config = _FORBID

    run_id: RunId | None = None
    routing_plan_id: RoutingPlanId | None = None
    outcome: OrchestratorApiOutcome
    run_status: RunStatus | None = None
    orchestrator_status: OrchestratorStatus | None = None
    next_wakeup: NextWakeupState | None = None
    checkpoint_reused: bool
    llm_routing_called: bool
    dispatch_attempt_count: int = Field(default=0, ge=0)
    decision_counts: CompactDecisionCounts | None = None
    task_counts: CompactTaskCounts | None = None
    artifact_counts: CompactArtifactCounts | None = None
    artifact_refs: list[CompactArtifactRef] = Field(default_factory=list)
    action_code: Literal[
        "provide_required_input",
        "reconcile_worker_result",
        "wait_for_dependencies",
        "inspect_compact_failure",
    ] | None = None
    error_code: str | None = Field(
        default=None,
        pattern=r"^[a-z][a-z0-9_]{0,127}$",
    )


__all__ = [
    "CompactArtifactCounts",
    "CompactArtifactRef",
    "CompactDecisionCounts",
    "CompactTaskCounts",
    "OrchestratorApiOutcome",
    "OrchestratorStep4Response",
]
