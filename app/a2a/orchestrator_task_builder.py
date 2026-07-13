"""Build validated in-memory python-a2a Tasks without dispatching them."""

from __future__ import annotations

from dataclasses import dataclass

from python_a2a import Message, MessageRole, Task, TextContent

from app.schemas.worker_routing_plan import (
    RejectedRoutingDecision,
    ValidatedRoutingDecision,
)
from app.utils.ids import new_task_id

from .contracts import (
    A2ATaskMetadata,
    InputProjection,
    OrchestratorRoutingDecisionRef,
    PrivacyConstraints,
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


def build_orchestrator_worker_task(
    *,
    run_id: str,
    routing_plan_id: str,
    validated: RuntimeValidatedDecision
    | ValidatedRoutingDecision
    | RejectedRoutingDecision,
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

    task_id = new_task_id()
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
    created_by = "step_04_orchestrator"
    request = WorkerExecutionRequest(
        payload_type="worker_execution_request",
        payload_version="v1",
        run_id=run_id,
        task_id=task_id,
        routing_plan_id=routing_plan_id,
        routing_decision_id=updated_decision.routing_decision_id,
        agent_id=updated_decision.agent_id,
        capability_id=updated_decision.capability_id,
        created_by=created_by,
        worker_request=WorkerRequestSpec(
            objective=updated_decision.objective,
            reason=updated_decision.selection_reason,
            priority=updated_decision.priority,
        ),
        orchestrator_routing_decision=OrchestratorRoutingDecisionRef(
            planned_status="run",
            dispatch_mode="python_a2a",
            deterministic_gate_status="passed",
            expected_outputs=expected_outputs,
        ),
        input_projection=InputProjection(
            compact_inputs={},
            input_artifact_refs=validated.input_artifact_refs,
            runtime_refs={},
        ),
        privacy_constraints=PrivacyConstraints(),
    )
    metadata = A2ATaskMetadata(
        adc_payload_type="worker_execution_request",
        adc_payload_version="v1",
        run_id=run_id,
        task_id=task_id,
        routing_plan_id=routing_plan_id,
        routing_decision_id=updated_decision.routing_decision_id,
        agent_id=updated_decision.agent_id,
        capability_id=updated_decision.capability_id,
        created_by=created_by,
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
        decision=updated_decision,
        task=task,
        dispatch_target=validated.dispatch_target,
    )
