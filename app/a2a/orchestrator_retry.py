"""Deterministic generic retry policy for terminal A2A worker attempts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal

from pydantic import ValidationError

from app.schemas.orchestrator_execution_state import (
    OrchestratorExecutionState,
    WorkerTaskExecutionState,
)
from app.utils.ids import new_task_id

from .contracts import WorkerExecutionResult
from .orchestrator_execution_state import (
    OrchestratorExecutionStateError,
    dispatch_eligible_task_ids,
    recompute_aggregate_state,
)

MAX_WORKER_RETRIES = 3


class OrchestratorRetryError(RuntimeError):
    """Compact fail-closed retry-policy error."""


class OrchestratorRetryResult:
    """Compact deterministic retry/disposition result without Task payloads."""

    __slots__ = ("state", "retry_task_ids", "terminal_disposition")

    def __init__(
        self,
        *,
        state: OrchestratorExecutionState,
        retry_task_ids: tuple[str, ...],
        terminal_disposition: Literal[
            "none",
            "retry_exhausted",
            "non_retryable_failure",
            "needs_user_input",
            "reconciliation_required",
        ],
    ) -> None:
        self.state = state
        self.retry_task_ids = retry_task_ids
        self.terminal_disposition = terminal_disposition


def prepare_orchestrator_retries(
    *,
    state: OrchestratorExecutionState,
    completion_proofs: Mapping[str, WorkerExecutionResult],
    max_worker_retries: int = MAX_WORKER_RETRIES,
) -> OrchestratorRetryResult:
    """Create every eligible retry in one deterministic, concurrency-safe batch."""
    if max_worker_retries != MAX_WORKER_RETRIES:
        raise OrchestratorRetryError("worker_retry_policy_invalid")
    checked = _validated_state(state)
    proofs = _validated_proofs(checked, completion_proofs)
    payload = checked.model_dump()
    retry_ids: list[str] = []
    exhausted: list[str] = []
    non_retryable: list[str] = []
    needs_input: list[str] = []
    reconciliation_required = False

    for decision_id in sorted(checked.routing.decisions):
        decision = checked.routing.decisions[decision_id]
        if not decision.task_ids:
            continue
        previous_id = decision.task_ids[-1]
        previous = checked.worker_tasks[previous_id]
        if previous.execution_status not in {"completed", "failed"}:
            continue
        proof = proofs.get(previous_id)
        if proof is None:
            reconciliation_required = True
            continue
        if proof.retry_of_task_id != previous.retry_of_task_id:
            raise OrchestratorRetryError("worker_retry_proof_identity_mismatch")
        if previous.result_status == "needs_user_input":
            needs_input.append(decision_id)
            continue
        if previous.result_status in {"validation_failed", "blocked"}:
            non_retryable.append(decision_id)
            continue
        if previous.result_status != "tool_failed":
            continue
        if previous.terminal_error_code != proof.error_code:
            raise OrchestratorRetryError("worker_retry_proof_identity_mismatch")
        if previous.retry_attempt >= max_worker_retries:
            exhausted.append(decision_id)
            continue

        task_id = new_task_id()
        attempt = previous.retry_attempt + 1
        payload["worker_tasks"][task_id] = WorkerTaskExecutionState(
            task_id=task_id,
            routing_plan_id=checked.routing.routing_plan_id,
            routing_decision_id=decision_id,
            agent_id=decision.agent_id,
            capability_id=decision.capability_id,
            dispatch_status="not_dispatched",
            execution_status="not_started",
            result_status=None,
            retry_of_task_id=previous_id,
            retry_attempt=attempt,
            max_retry_attempts=max_worker_retries,
            terminal_error_code=None,
            output_artifact_refs={},
        ).model_dump()
        payload["routing"]["decisions"][decision_id].update(
            {
                "status": "ready",
                "blocking_reason": None,
                "task_ids": [*decision.task_ids, task_id],
            }
        )
        for artifact_name in decision.expected_output_artifact_names:
            artifact = payload["artifacts"].get(artifact_name)
            if artifact is None:
                raise OrchestratorRetryError("worker_retry_artifact_state_missing")
            artifact.update(
                {
                    "status": "planned",
                    "artifact_id": None,
                    "producer_task_id": task_id,
                    "safe_summary_ref": None,
                }
            )
        retry_ids.append(task_id)

    terminal_failures = [*exhausted, *non_retryable, *needs_input]
    if terminal_failures:
        _block_failed_dependencies(
            payload,
            exhausted=exhausted,
            non_retryable=non_retryable,
            needs_input=needs_input,
        )
    try:
        updated = recompute_aggregate_state(
            OrchestratorExecutionState.model_validate(payload)
        )
    except (ValidationError, OrchestratorExecutionStateError):
        raise OrchestratorRetryError("worker_retry_state_invalid") from None

    disposition: Literal[
        "none",
        "retry_exhausted",
        "non_retryable_failure",
        "needs_user_input",
        "reconciliation_required",
    ] = "none"
    if reconciliation_required:
        disposition = "reconciliation_required"
    elif needs_input:
        disposition = "needs_user_input"
    elif exhausted:
        disposition = "retry_exhausted"
    elif non_retryable:
        disposition = "non_retryable_failure"

    if disposition != "none" and not _has_active_or_eligible_work(updated):
        final_payload = updated.model_dump()
        if disposition == "needs_user_input":
            final_payload["run_status"] = "waiting_for_input"
            final_payload["orchestrator"].update(
                {"status": "planning", "next_wakeup_reason": "needs_user_input"}
            )
            final_payload["next_wakeup"] = {
                "target": "user_input",
                "reason": "needs_user_input",
            }
        elif disposition == "reconciliation_required":
            final_payload["run_status"] = "running"
            final_payload["orchestrator"].update(
                {
                    "status": "evaluating_results",
                    "next_wakeup_reason": "worker_result_reconciliation_required",
                }
            )
            final_payload["next_wakeup"] = {
                "target": "orchestrator_loop",
                "reason": "worker_result_reconciliation_required",
            }
        else:
            reason = (
                "worker_retry_exhausted"
                if disposition == "retry_exhausted"
                else "worker_non_retryable_failure"
            )
            final_payload["run_status"] = "failed"
            final_payload["orchestrator"].update(
                {"status": "routing_to_final", "next_wakeup_reason": reason}
            )
            final_payload["next_wakeup"] = {
                "target": "final_response",
                "reason": reason,
            }
        try:
            updated = OrchestratorExecutionState.model_validate(final_payload)
        except ValidationError:
            raise OrchestratorRetryError("worker_retry_state_invalid") from None

    return OrchestratorRetryResult(
        state=updated,
        retry_task_ids=tuple(retry_ids),
        terminal_disposition=disposition,
    )


def _validated_state(state: OrchestratorExecutionState) -> OrchestratorExecutionState:
    try:
        return OrchestratorExecutionState.model_validate(state.model_dump())
    except (AttributeError, ValidationError):
        raise OrchestratorRetryError("worker_retry_state_invalid") from None


def _validated_proofs(
    state: OrchestratorExecutionState,
    supplied: Mapping[str, WorkerExecutionResult],
) -> dict[str, WorkerExecutionResult]:
    if not isinstance(supplied, Mapping):
        raise OrchestratorRetryError("worker_retry_proofs_invalid")
    checked: dict[str, WorkerExecutionResult] = {}
    for key, raw in supplied.items():
        try:
            proof = WorkerExecutionResult.model_validate(
                raw.model_dump(mode="python", warnings=False), strict=True
            )
        except (AttributeError, ValidationError):
            raise OrchestratorRetryError("worker_retry_proof_schema_invalid") from None
        task = state.worker_tasks.get(key)
        if (
            key != proof.task_id
            or task is None
            or proof.run_id != state.run_id
            or proof.routing_plan_id != state.routing.routing_plan_id
            or proof.routing_decision_id != task.routing_decision_id
            or proof.agent_id != task.agent_id
            or proof.capability_id != task.capability_id
            or proof.execution_status != task.execution_status
            or proof.result_status != task.result_status
            or proof.retry_of_task_id != task.retry_of_task_id
        ):
            raise OrchestratorRetryError("worker_retry_proof_identity_mismatch")
        checked[key] = proof.model_copy(deep=True)
    return checked


def _block_failed_dependencies(
    payload: dict,
    *,
    exhausted: Sequence[str],
    non_retryable: Sequence[str],
    needs_input: Sequence[str],
) -> None:
    decisions = payload["routing"]["decisions"]
    failed_artifacts: set[str] = set()
    for decision_id in exhausted:
        decisions[decision_id].update(
            {"status": "failed", "blocking_reason": "worker_failed"}
        )
        failed_artifacts.update(decisions[decision_id]["expected_output_artifact_names"])
    for decision_id in non_retryable:
        latest_task_id = decisions[decision_id]["task_ids"][-1]
        result_status = payload["worker_tasks"][latest_task_id]["result_status"]
        decisions[decision_id].update(
            {
                "status": "blocked" if result_status == "blocked" else "failed",
                "blocking_reason": "worker_failed",
            }
        )
        failed_artifacts.update(decisions[decision_id]["expected_output_artifact_names"])
    for decision_id in needs_input:
        decisions[decision_id].update(
            {"status": "blocked", "blocking_reason": "needs_user_input"}
        )
        failed_artifacts.update(decisions[decision_id]["expected_output_artifact_names"])

    changed = True
    while changed:
        changed = False
        for decision in decisions.values():
            if decision["status"] in {"completed", "failed", "blocked", "dispatched"}:
                continue
            if set(decision["required_artifact_names"]) & failed_artifacts:
                decision.update(
                    {"status": "blocked", "blocking_reason": "dependency_failed"}
                )
                failed_artifacts.update(decision["expected_output_artifact_names"])
                changed = True


def _has_active_or_eligible_work(state: OrchestratorExecutionState) -> bool:
    if dispatch_eligible_task_ids(state):
        return True
    return any(
        task.dispatch_status in {"dispatching", "dispatched"}
        and task.execution_status in {"not_started", "running"}
        for task in state.worker_tasks.values()
    )


__all__ = [
    "MAX_WORKER_RETRIES",
    "OrchestratorRetryError",
    "OrchestratorRetryResult",
    "prepare_orchestrator_retries",
]
