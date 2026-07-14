"""Compact persisted Step 4 Orchestrator worker-routing plan (Turn F1)."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

_FORBID = ConfigDict(extra="forbid")

LoopDecision = Literal[
    "dispatch_next_workers",
    "wait_for_dependencies",
    "route_to_final_response",
    "request_user_input",
    "repair_or_retry",
    "stop_cannot_satisfy",
]
Priority = Literal["low", "normal", "high"]
ValidationStatus = Literal[
    "ready",
    "waiting_for_dependencies",
    "wait_for_input",
    "blocked_missing_dependency",
    "rejected",
]


class OrchestratorRouteDecision(BaseModel):
    model_config = _FORBID
    agent_id: str = Field(min_length=1)
    capability_id: str = Field(min_length=1)
    objective: str = Field(min_length=1)
    selection_reason: str = Field(min_length=1)
    priority: Priority


class OrchestratorRoutingProposal(BaseModel):
    model_config = _FORBID
    loop_decision: LoopDecision
    decisions: list[OrchestratorRouteDecision] = Field(default_factory=list)
    decision_summary: str = Field(min_length=1)


class DependencyEdge(BaseModel):
    model_config = _FORBID
    artifact_name: str
    producer_agent_id: str
    producer_capability_id: str
    consumer_agent_id: str
    consumer_capability_id: str


class ValidatedRoutingDecision(BaseModel):
    model_config = _FORBID
    routing_decision_id: str
    agent_id: str
    capability_id: str
    objective: str
    selection_reason: str
    priority: Priority
    validation_status: ValidationStatus
    required_artifact_names: list[str] = Field(default_factory=list)
    dependency_artifact_names: list[str] = Field(default_factory=list)
    dependency_producers: list[str] = Field(default_factory=list)
    expected_output_artifact_names: list[str] = Field(default_factory=list)
    task_id: Optional[str] = None
    reason: Optional[str] = None


class RejectedRoutingDecision(BaseModel):
    model_config = _FORBID
    routing_decision_id: str
    agent_id: Optional[str] = None
    capability_id: Optional[str] = None
    reason: Literal[
        "duplicate_route",
        "unknown_worker",
        "unknown_capability",
        "rejected_unavailable",
        "dispatch_target_invalid",
        "output_artifact_conflict",
        "unsafe_llm_output",
        "invalid_loop_decision",
    ]


class WorkerRoutingPlan(BaseModel):
    model_config = _FORBID
    run_id: str
    step_id: Literal["step_04_orchestrator_routing"] = "step_04_orchestrator_routing"
    routing_plan_id: str
    planned_at: str
    loop_decision: Optional[LoopDecision]
    routing_status: Literal[
        "ready", "waiting", "blocked", "completed", "rejected", "llm_failed"
    ]
    llm_selection_source: str
    prompt_cache_layout_version: Literal["orchestrator-routing-v1"] = (
        "orchestrator-routing-v1"
    )
    proposed_decisions: list[OrchestratorRouteDecision] = Field(default_factory=list)
    validated_decisions: list[ValidatedRoutingDecision] = Field(default_factory=list)
    rejected_decisions: list[RejectedRoutingDecision] = Field(default_factory=list)
    dependency_edges: list[DependencyEdge] = Field(default_factory=list)
    ready_task_count: int = Field(default=0, ge=0)
    waiting_decision_count: int = Field(default=0, ge=0)
    rejected_decision_count: int = Field(default=0, ge=0)
    available_agent_ids: list[str] = Field(default_factory=list)
    unavailable_agent_ids: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
