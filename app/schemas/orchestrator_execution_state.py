"""Strict compact runtime state for the A2A Orchestrator graph."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from app.schemas.worker_routing_plan import LoopDecision

_FORBID = ConfigDict(extra="forbid")

RunStatus = Literal[
    "created", "running", "waiting_for_input", "completed", "failed", "canceled"
]
OrchestratorStatus = Literal[
    "planning",
    "validating",
    "dispatching",
    "waiting_for_workers",
    "evaluating_results",
    "routing_to_final",
    "completed",
    "failed",
]
RoutingDecisionStatus = Literal[
    "planned",
    "ready",
    "pending_dependency",
    "dispatched",
    "skipped",
    "blocked",
    "completed",
    "failed",
]
DispatchStatus = Literal[
    "not_dispatched", "dispatching", "dispatched", "dispatch_failed"
]
ExecutionStatus = Literal["not_started", "running", "completed", "failed", "canceled"]
ResultStatus = Literal[
    "success",
    "partial",
    "validation_failed",
    "tool_failed",
    "blocked",
    "needs_user_input",
]
ArtifactStatus = Literal["missing", "planned", "producing", "available", "invalid"]
BlockingReason = Literal[
    "missing_required_artifact",
    "agent_unavailable",
    "input_not_ready",
    "validation_failed",
    "dispatch_failed",
    "worker_failed",
    "needs_user_input",
    "dependency_failed",
]
AgentFailureReason = Literal[
    "none",
    "discovery_timeout",
    "discovery_connection_failed",
    "card_invalid",
    "health_timeout",
    "health_failed",
    "dispatch_timeout",
    "dispatch_connection_failed",
    "dispatch_transport_error",
    "server_error",
]
RoutingSource = Literal[
    "llm_primary_validated",
    "llm_failed",
    "not_run_step3_needs_user_input",
    "not_run_step3_blocked",
]
NextWakeupTarget = Literal[
    "worker_dispatch", "orchestrator_loop", "user_input", "final_response"
]
NextWakeupReason = Literal[
    "ready_tasks_available",
    "dispatch_in_progress",
    "worker_result_received",
    "dispatch_failed",
    "routing_blocked",
    "dependencies_pending",
    "needs_user_input",
    "routing_completed",
    "routing_failed",
]

RunId = Annotated[
    str,
    StringConstraints(
        min_length=21,
        max_length=21,
        pattern=r"^run_[0-9]{8}_[0-9a-f]{8}$",
    ),
]
RoutingPlanId = Annotated[
    str,
    StringConstraints(
        min_length=20,
        max_length=20,
        pattern=r"^wrp_[0-9a-f]{16}$",
    ),
]
RoutingDecisionId = Annotated[
    str,
    StringConstraints(
        min_length=22,
        max_length=22,
        pattern=r"^route_[0-9a-f]{16}$",
    ),
]
TaskId = Annotated[
    str,
    StringConstraints(
        min_length=21,
        max_length=21,
        pattern=r"^task_[0-9a-f]{16}$",
    ),
]
ContractIdentifier = Annotated[
    str,
    StringConstraints(
        min_length=2,
        max_length=128,
        pattern=r"^[a-z][a-z0-9_]*$",
    ),
]
ArtifactId = Annotated[
    str,
    StringConstraints(
        min_length=15,
        max_length=128,
        pattern=r"^[a-z][a-z0-9_]*_[0-9a-f]{12}$",
    ),
]
OrchestratorSummaryRef = Annotated[
    str,
    StringConstraints(
        min_length=29,
        max_length=29,
        pattern=r"^mem_orchestrator_[0-9a-f]{12}$",
    ),
]
WorkerSummaryRef = Annotated[
    str,
    StringConstraints(
        min_length=27,
        max_length=27,
        pattern=r"^summary_worker_[0-9a-f]{12}$",
    ),
]
FinalResponseRef = Annotated[
    str,
    StringConstraints(
        min_length=31,
        max_length=31,
        pattern=r"^mem_final_response_[0-9a-f]{12}$",
    ),
]
ArtifactSummaryRef = Annotated[
    str,
    StringConstraints(
        min_length=29,
        max_length=29,
        pattern=r"^summary_artifact_[0-9a-f]{12}$",
    ),
]


class OrchestratorState(BaseModel):
    model_config = _FORBID

    status: OrchestratorStatus
    loop_decision: LoopDecision | None = None
    deterministic_validation_status: Literal["passed", "failed"]
    next_wakeup_reason: NextWakeupReason | None = None


class RoutingDecisionExecutionState(BaseModel):
    model_config = _FORBID

    routing_decision_id: RoutingDecisionId
    agent_id: ContractIdentifier
    capability_id: ContractIdentifier
    status: RoutingDecisionStatus
    blocking_reason: BlockingReason | None = None
    required_artifact_names: list[ContractIdentifier] = Field(default_factory=list)
    expected_output_artifact_names: list[ContractIdentifier] = Field(
        default_factory=list
    )
    task_ids: list[TaskId] = Field(default_factory=list)


class RoutingExecutionState(BaseModel):
    model_config = _FORBID

    routing_plan_id: RoutingPlanId
    routing_source: RoutingSource
    decisions: dict[RoutingDecisionId, RoutingDecisionExecutionState] = Field(
        default_factory=dict
    )

    @model_validator(mode="after")
    def validate_decision_keys(self) -> RoutingExecutionState:
        for key, decision in self.decisions.items():
            if key != decision.routing_decision_id:
                raise ValueError("routing_decision_identity_mismatch")
        task_ids = [task_id for item in self.decisions.values() for task_id in item.task_ids]
        if len(task_ids) != len(set(task_ids)):
            raise ValueError("conflicting_duplicate_task_identity")
        return self


class WorkerTaskExecutionState(BaseModel):
    model_config = _FORBID

    task_id: TaskId
    routing_plan_id: RoutingPlanId
    routing_decision_id: RoutingDecisionId
    agent_id: ContractIdentifier
    capability_id: ContractIdentifier
    dispatch_status: DispatchStatus
    execution_status: ExecutionStatus
    result_status: ResultStatus | None = None
    agent_failure_reason: AgentFailureReason = "none"
    retry_of_task_id: TaskId | None = None
    output_artifact_refs: dict[ContractIdentifier, ArtifactId] = Field(
        default_factory=dict
    )

    @model_validator(mode="after")
    def validate_lifecycle_shape(self) -> WorkerTaskExecutionState:
        if self.dispatch_status in {"not_dispatched", "dispatching", "dispatch_failed"}:
            if self.execution_status != "not_started":
                raise ValueError("task_lifecycle_status_conflict")
        if self.dispatch_status == "dispatch_failed":
            if self.agent_failure_reason == "none":
                raise ValueError("dispatch_failure_reason_required")
        elif self.agent_failure_reason != "none":
            raise ValueError("agent_failure_reason_without_dispatch_failure")
        if self.execution_status in {"not_started", "running"} and self.result_status is not None:
            raise ValueError("result_status_before_terminal_execution")
        return self


class ArtifactExecutionState(BaseModel):
    model_config = _FORBID

    artifact_name: ContractIdentifier
    status: ArtifactStatus
    artifact_id: ArtifactId | None = None
    producer_task_id: TaskId | None = None
    safe_summary_ref: ArtifactSummaryRef | None = None

    @model_validator(mode="after")
    def validate_available_identity(self) -> ArtifactExecutionState:
        if self.status == "available" and not self.artifact_id:
            raise ValueError("available_artifact_identity_missing")
        return self


class NextWakeupState(BaseModel):
    model_config = _FORBID

    target: NextWakeupTarget
    reason: NextWakeupReason


class MemoryRefs(BaseModel):
    model_config = _FORBID

    orchestrator_run_summary: OrchestratorSummaryRef | None = None
    completed_worker_summaries: list[WorkerSummaryRef] = Field(default_factory=list)
    final_response_context: FinalResponseRef | None = None


class OrchestratorExecutionState(BaseModel):
    """Only compact runtime state; never task, request, result, card, or artifact bodies."""

    model_config = _FORBID

    run_id: RunId
    run_status: RunStatus
    orchestrator: OrchestratorState
    routing: RoutingExecutionState
    worker_tasks: dict[TaskId, WorkerTaskExecutionState] = Field(default_factory=dict)
    artifacts: dict[ContractIdentifier, ArtifactExecutionState] = Field(
        default_factory=dict
    )
    memory_refs: MemoryRefs = Field(default_factory=MemoryRefs)
    next_wakeup: NextWakeupState | None = None

    @model_validator(mode="after")
    def validate_cross_references(self) -> OrchestratorExecutionState:
        if self.next_wakeup is None:
            if self.orchestrator.next_wakeup_reason is not None:
                raise ValueError("next_wakeup_reason_mismatch")
        elif self.orchestrator.next_wakeup_reason != self.next_wakeup.reason:
            raise ValueError("next_wakeup_reason_mismatch")
        for key, task in self.worker_tasks.items():
            if key != task.task_id:
                raise ValueError("task_identity_mismatch")
            if task.routing_plan_id != self.routing.routing_plan_id:
                raise ValueError("task_routing_plan_identity_mismatch")
            decision = self.routing.decisions.get(task.routing_decision_id)
            if decision is None:
                raise ValueError("task_routing_decision_unknown")
            if (
                decision.agent_id != task.agent_id
                or decision.capability_id != task.capability_id
                or task.task_id not in decision.task_ids
            ):
                raise ValueError("task_routing_decision_identity_mismatch")
        for decision in self.routing.decisions.values():
            for task_id in decision.task_ids:
                if task_id not in self.worker_tasks:
                    raise ValueError("routing_decision_task_unknown")
        for key, artifact in self.artifacts.items():
            if key != artifact.artifact_name:
                raise ValueError("artifact_identity_mismatch")
            if artifact.producer_task_id and artifact.producer_task_id not in self.worker_tasks:
                raise ValueError("artifact_producer_task_unknown")
        return self


__all__ = [
    "AgentFailureReason",
    "ArtifactExecutionState",
    "MemoryRefs",
    "NextWakeupState",
    "OrchestratorExecutionState",
    "OrchestratorState",
    "RoutingDecisionExecutionState",
    "RoutingExecutionState",
    "WorkerTaskExecutionState",
]
