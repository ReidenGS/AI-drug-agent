"""Build validated in-memory python-a2a Tasks without dispatching them."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from python_a2a import Message, MessageRole, Task, TextContent

from app.schemas.worker_routing_plan import (
    RejectedRoutingDecision,
    ValidatedRoutingDecision,
)
from app.utils.ids import new_task_id

from .contracts import (
    A2ATaskMetadata,
    InputArtifactRef,
    InputProjection,
    OrchestratorRoutingDecisionRef,
    PrivacyConstraints,
    RetryContext,
    WorkerExecutionRequest,
    WorkerRequestSpec,
)
from .orchestrator_discovery import DispatchTarget
from .orchestrator_routing_validation import RuntimeValidatedDecision


@dataclass(frozen=True)
class PreparedA2ATask:
    """In-memory dispatch preparation; ``dispatch_target`` is never serialized."""

    decision: ValidatedRoutingDecision
    task: Task
    dispatch_target: DispatchTarget
    input_artifact_refs: dict[str, InputArtifactRef]


def build_canonical_worker_execution_request(
    *,
    run_id: str,
    routing_plan_id: str,
    decision: ValidatedRoutingDecision,
    input_artifact_refs: Mapping[str, InputArtifactRef],
    retry_context: RetryContext | None = None,
) -> WorkerExecutionRequest:
    """Build the complete canonical ADC request for initial or retry dispatch.

    Both Task construction and transport validation call this function so
    optional/default request surfaces cannot drift independently.
    """
    if not decision.task_id:
        raise ValueError("canonical_request_task_id_required")
    return WorkerExecutionRequest(
        payload_type="worker_execution_request",
        payload_version="v1",
        run_id=run_id,
        session_id=None,
        task_id=decision.task_id,
        routing_plan_id=routing_plan_id,
        routing_decision_id=decision.routing_decision_id,
        agent_id=decision.agent_id,
        capability_id=decision.capability_id,
        created_by="step_04_orchestrator",
        worker_request=WorkerRequestSpec(
            objective=decision.objective,
            reason=decision.selection_reason,
            priority=decision.priority,
        ),
        orchestrator_routing_decision=OrchestratorRoutingDecisionRef(
            planned_status="run",
            dispatch_mode="python_a2a",
            deterministic_gate_status="passed",
            routing_phase=None,
            expected_outputs=list(decision.expected_output_artifact_names),
            reason=None,
        ),
        input_projection=InputProjection(
            projection_version="v1",
            compact_inputs={},
            input_artifact_refs=dict(input_artifact_refs),
            runtime_refs={},
        ),
        privacy_constraints=PrivacyConstraints(),
        retry_context=retry_context,
    )


def build_orchestrator_worker_task(
    *,
    run_id: str,
    routing_plan_id: str,
    validated: RuntimeValidatedDecision
    | ValidatedRoutingDecision
    | RejectedRoutingDecision,
    task_id: str | None = None,
) -> PreparedA2ATask:
    """Build one Task for a ready decision; never sends or executes it."""
    if not isinstance(validated, RuntimeValidatedDecision):
        raise ValueError("task_builder_requires_runtime_validated_decision")
    if validated.decision.validation_status != "ready":
        raise ValueError("task_builder_requires_ready_decision")
    if not validated.task_build_allowed:
        raise ValueError("task_builder_not_allowed_for_loop_or_validation_state")
    if validated.run_id != run_id:
        raise ValueError("task_builder_run_id_mismatch")
    if validated.dispatch_target.dispatch_mode != "python_a2a":
        raise ValueError("task_builder_dispatch_mode_invalid")
    if (
        validated.dispatch_target.agent_id != validated.decision.agent_id
        or validated.dispatch_target.capability_id
        != validated.decision.capability_id
    ):
        raise ValueError("task_builder_dispatch_target_identity_mismatch")

    if task_id is not None and not task_id:
        raise ValueError("task_builder_task_id_invalid")
    task_id = task_id or new_task_id()
    expected_outputs = [
        artifact.artifact_name for artifact in validated.capability.output_artifacts
    ]
    if expected_outputs != validated.decision.expected_output_artifact_names:
        raise ValueError("task_builder_expected_outputs_mismatch")
    updated_decision = validated.decision.model_copy(
        update={
            "task_id": task_id,
        }
    )
    request = build_canonical_worker_execution_request(
        run_id=run_id,
        routing_plan_id=routing_plan_id,
        decision=updated_decision,
        input_artifact_refs=validated.input_artifact_refs,
    )
    return _assemble_prepared_task(
        decision=updated_decision,
        request=request,
        dispatch_target=validated.dispatch_target,
        input_artifact_refs=validated.input_artifact_refs,
    )


def build_retry_orchestrator_worker_task(
    *,
    run_id: str,
    routing_plan_id: str,
    validated: RuntimeValidatedDecision,
    task_id: str,
    retry_attempt: int,
    max_retry_attempts: int,
    retry_of_task_id: str,
    retry_reason: str,
) -> PreparedA2ATask:
    """Build one retry from freshly reconstructed persisted/frozen authority."""
    if not isinstance(validated, RuntimeValidatedDecision):
        raise ValueError("retry_task_requires_runtime_validated_decision")
    if (
        validated.decision.validation_status != "ready"
        or not validated.task_build_allowed
        or validated.decision.task_id != task_id
    ):
        raise ValueError("retry_task_validation_state_invalid")
    if not task_id or task_id == retry_of_task_id:
        raise ValueError("retry_task_identity_invalid")
    retry_context = RetryContext(
        retry_of_task_id=retry_of_task_id,
        retry_attempt=retry_attempt,
        max_retry_attempts=max_retry_attempts,
        retry_reason=retry_reason,
    )
    decision = validated.decision
    request = build_canonical_worker_execution_request(
        run_id=run_id,
        routing_plan_id=routing_plan_id,
        decision=decision,
        input_artifact_refs=validated.input_artifact_refs,
        retry_context=retry_context,
    )
    return _assemble_prepared_task(
        decision=decision,
        request=request,
        dispatch_target=validated.dispatch_target,
        input_artifact_refs=validated.input_artifact_refs,
    )


def _assemble_prepared_task(
    *,
    decision: ValidatedRoutingDecision,
    request: WorkerExecutionRequest,
    dispatch_target: DispatchTarget,
    input_artifact_refs: Mapping[str, InputArtifactRef],
) -> PreparedA2ATask:
    task_id = decision.task_id
    if not task_id:
        raise ValueError("prepared_task_id_required")
    metadata = A2ATaskMetadata(
        adc_payload_type="worker_execution_request",
        adc_payload_version="v1",
        run_id=request.run_id,
        task_id=task_id,
        routing_plan_id=request.routing_plan_id,
        routing_decision_id=decision.routing_decision_id,
        agent_id=decision.agent_id,
        capability_id=decision.capability_id,
        created_by=request.created_by,
    )
    message = Message(
        content=TextContent(text=request.model_dump_json()),
        role=MessageRole.USER,
    )
    task = Task(
        id=task_id,
        message=message.to_dict(),
        metadata=metadata.model_dump(),
    )
    return PreparedA2ATask(
        decision=decision,
        task=task,
        dispatch_target=dispatch_target,
        input_artifact_refs=dict(input_artifact_refs),
    )


__all__ = [
    "PreparedA2ATask",
    "build_canonical_worker_execution_request",
    "build_orchestrator_worker_task",
    "build_retry_orchestrator_worker_task",
]
