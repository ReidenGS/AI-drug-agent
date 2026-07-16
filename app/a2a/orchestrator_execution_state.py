"""Projection and fail-closed lifecycle reducers for compact execution state."""

from __future__ import annotations

from typing import Mapping

from pydantic import ValidationError

from app.schemas.orchestrator_execution_state import (
    AgentFailureReason,
    ArtifactExecutionState,
    NextWakeupState,
    OrchestratorExecutionState,
    OrchestratorState,
    RoutingDecisionExecutionState,
    RoutingExecutionState,
    WorkerTaskExecutionState,
)

from .orchestrator_routing_service import OrchestratorRoutingServiceResult
from .orchestrator_task_builder import PreparedA2ATask

_DISPATCH_FAILURE_REASONS = {
    "dispatch_timeout",
    "dispatch_connection_failed",
    "dispatch_transport_error",
    "server_error",
}


class OrchestratorExecutionStateError(ValueError):
    """Compact fail-closed execution-state error."""


def execution_state_from_routing_result(
    result: OrchestratorRoutingServiceResult,
) -> OrchestratorExecutionState:
    """Project a validated F1 routing result without retaining transport payloads."""
    plan = result.plan
    if plan.run_id == "" or result.plan_artifact_id == "":
        raise OrchestratorExecutionStateError("routing_result_identity_invalid")
    if plan.ready_task_count != len(result.prepared_tasks):
        raise OrchestratorExecutionStateError(
            "routing_result_ready_task_count_mismatch"
        )

    prepared_by_decision = {}
    task_ids: set[str] = set()
    for prepared in result.prepared_tasks:
        decision_id = prepared.decision.routing_decision_id
        task_id = prepared.decision.task_id
        if not task_id or decision_id in prepared_by_decision or task_id in task_ids:
            raise OrchestratorExecutionStateError("conflicting_duplicate_task_identity")
        prepared_by_decision[decision_id] = prepared
        task_ids.add(task_id)

    decisions: dict[str, RoutingDecisionExecutionState] = {}
    worker_tasks: dict[str, WorkerTaskExecutionState] = {}
    artifacts: dict[str, ArtifactExecutionState] = {}
    known_decision_ids: set[str] = set()
    for decision in plan.validated_decisions:
        decision_id = decision.routing_decision_id
        if decision_id in known_decision_ids:
            raise OrchestratorExecutionStateError("routing_decision_identity_conflict")
        known_decision_ids.add(decision_id)
        prepared = prepared_by_decision.get(decision_id)
        mapped_task_ids: list[str] = []
        if prepared is not None:
            task_id = prepared.decision.task_id
            if (
                task_id is None
                or decision.task_id != task_id
                or str(prepared.task.id) != task_id
                or prepared.decision.agent_id != decision.agent_id
                or prepared.decision.capability_id != decision.capability_id
                or prepared.dispatch_target.agent_id != decision.agent_id
                or prepared.dispatch_target.capability_id != decision.capability_id
                or decision.validation_status != "ready"
            ):
                raise OrchestratorExecutionStateError("prepared_task_identity_mismatch")
            mapped_task_ids.append(task_id)
            worker_tasks[task_id] = WorkerTaskExecutionState(
                task_id=task_id,
                routing_plan_id=plan.routing_plan_id,
                routing_decision_id=decision_id,
                agent_id=decision.agent_id,
                capability_id=decision.capability_id,
                dispatch_status="not_dispatched",
                execution_status="not_started",
                result_status=None,
                agent_failure_reason="none",
                output_artifact_refs={},
            )
        decisions[decision_id] = RoutingDecisionExecutionState(
            routing_decision_id=decision_id,
            agent_id=decision.agent_id,
            capability_id=decision.capability_id,
            status=_routing_decision_status(decision.validation_status),
            blocking_reason=_blocking_reason(
                decision.validation_status, decision.reason
            ),
            required_artifact_names=list(decision.required_artifact_names),
            expected_output_artifact_names=list(
                decision.expected_output_artifact_names
            ),
            task_ids=mapped_task_ids,
        )
        producer_task_id = mapped_task_ids[0] if mapped_task_ids else None
        for artifact_name in decision.expected_output_artifact_names:
            if artifact_name in artifacts:
                raise OrchestratorExecutionStateError(
                    "output_artifact_identity_conflict"
                )
            artifacts[artifact_name] = ArtifactExecutionState(
                artifact_name=artifact_name,
                status="planned",
                producer_task_id=producer_task_id,
            )

    if set(prepared_by_decision) - known_decision_ids:
        raise OrchestratorExecutionStateError("prepared_task_decision_unknown")

    for prepared in result.prepared_tasks:
        for artifact_name, ref in prepared.input_artifact_refs.items():
            if (
                ref.run_id != plan.run_id
                or ref.artifact_type != artifact_name
                or not ref.artifact_id
            ):
                raise OrchestratorExecutionStateError(
                    "input_artifact_ref_identity_mismatch"
                )
            existing = artifacts.get(artifact_name)
            if existing is not None:
                if existing.status != "available" or existing.artifact_id != ref.artifact_id:
                    raise OrchestratorExecutionStateError(
                        "artifact_execution_identity_conflict"
                    )
                continue
            artifacts[artifact_name] = ArtifactExecutionState(
                artifact_name=artifact_name,
                status="available",
                artifact_id=ref.artifact_id,
            )

    if plan.routing_status == "completed":
        run_status = "completed"
    elif plan.routing_status in {"blocked", "rejected", "llm_failed"}:
        run_status = "failed"
    elif plan.loop_decision == "request_user_input":
        run_status = "waiting_for_input"
    else:
        run_status = "running"
    try:
        projected = OrchestratorExecutionState(
            run_id=plan.run_id,
            session_id=plan.session_id,
            run_status=run_status,
            orchestrator=OrchestratorState(
                status="planning",
                loop_decision=plan.loop_decision,
                deterministic_validation_status=(
                    "failed"
                    if plan.routing_status in {"rejected", "llm_failed"}
                    else "passed"
                ),
                next_wakeup_reason=None,
            ),
            routing=RoutingExecutionState(
                routing_plan_id=plan.routing_plan_id,
                routing_source=plan.llm_selection_source,
                decisions=decisions,
            ),
            worker_tasks=worker_tasks,
            artifacts=artifacts,
            memory_refs={},
            next_wakeup=None,
        )
    except ValidationError as exc:
        raise OrchestratorExecutionStateError("execution_state_projection_invalid") from exc
    return recompute_aggregate_state(projected)


def dispatch_eligible_task_ids(state: OrchestratorExecutionState) -> tuple[str, ...]:
    """Return every dispatchable task deterministically; no concurrency cap."""
    checked = _validated_copy(state)
    return tuple(
        sorted(
            task_id
            for task_id, task in checked.worker_tasks.items()
            if task.dispatch_status == "not_dispatched"
            and task.execution_status == "not_started"
            and checked.routing.decisions[task.routing_decision_id].status == "ready"
        )
    )


def mark_task_dispatching(
    state: OrchestratorExecutionState, task_id: str
) -> OrchestratorExecutionState:
    return _transition_task(
        state,
        task_id,
        dispatch_status="dispatching",
        execution_status="not_started",
        agent_failure_reason="none",
    )


def mark_task_dispatched(
    state: OrchestratorExecutionState, task_id: str
) -> OrchestratorExecutionState:
    return _transition_task(
        state,
        task_id,
        dispatch_status="dispatched",
        execution_status="not_started",
        agent_failure_reason="none",
    )


def mark_task_dispatch_failed(
    state: OrchestratorExecutionState,
    task_id: str,
    agent_failure_reason: AgentFailureReason,
) -> OrchestratorExecutionState:
    if agent_failure_reason not in _DISPATCH_FAILURE_REASONS:
        raise OrchestratorExecutionStateError("agent_failure_reason_invalid")
    return _transition_task(
        state,
        task_id,
        dispatch_status="dispatch_failed",
        execution_status="not_started",
        agent_failure_reason=agent_failure_reason,
    )


def mark_task_running(
    state: OrchestratorExecutionState, task_id: str
) -> OrchestratorExecutionState:
    return _transition_task(
        state,
        task_id,
        dispatch_status="dispatched",
        execution_status="running",
        agent_failure_reason="none",
    )


def mark_task_result(
    state: OrchestratorExecutionState,
    task_id: str,
    *,
    result_status: str,
    error_code: str | None = None,
    output_artifact_refs: Mapping[str, str],
    available_output_artifact_names: frozenset[str] = frozenset(),
) -> OrchestratorExecutionState:
    """Apply one validated terminal worker result to compact execution state."""
    checked = _validated_copy(state)
    current = checked.worker_tasks.get(task_id)
    if current is None:
        raise OrchestratorExecutionStateError("task_id_unknown")
    productive = result_status in {"success", "partial"}
    if productive and error_code is not None:
        raise OrchestratorExecutionStateError("productive_result_error_code_forbidden")
    if not productive and error_code is None:
        raise OrchestratorExecutionStateError("failed_result_error_code_required")
    execution_status = "completed" if productive else "failed"
    decision = checked.routing.decisions[current.routing_decision_id]
    expected = set(decision.expected_output_artifact_names)
    actual = set(output_artifact_refs)
    if actual - expected:
        raise OrchestratorExecutionStateError("task_result_output_unexpected")
    if productive and actual != expected:
        raise OrchestratorExecutionStateError("task_result_output_missing")
    if productive and available_output_artifact_names != frozenset(actual):
        raise OrchestratorExecutionStateError("task_result_output_not_available")
    if not productive and available_output_artifact_names:
        raise OrchestratorExecutionStateError("failed_task_output_available")

    if current.execution_status in {"completed", "failed", "canceled"}:
        if (
            current.execution_status == execution_status
            and current.result_status == result_status
            and current.terminal_error_code == error_code
            and dict(current.output_artifact_refs) == dict(output_artifact_refs)
        ):
            return checked
        raise OrchestratorExecutionStateError("task_terminal_result_conflict")
    if current.dispatch_status != "dispatched" or current.execution_status not in {
        "not_started",
        "running",
    }:
        raise OrchestratorExecutionStateError("task_result_transition_invalid")

    payload = checked.model_dump()
    payload["worker_tasks"][task_id].update(
        {
            "execution_status": execution_status,
            "result_status": result_status,
            "terminal_error_code": error_code,
            "output_artifact_refs": dict(output_artifact_refs),
        }
    )
    decision_payload = payload["routing"]["decisions"][
        current.routing_decision_id
    ]
    if productive:
        decision_payload.update({"status": "completed", "blocking_reason": None})
    elif result_status in {"blocked", "needs_user_input"}:
        decision_payload.update(
            {
                "status": "blocked",
                "blocking_reason": (
                    "needs_user_input"
                    if result_status == "needs_user_input"
                    else "worker_failed"
                ),
            }
        )
    else:
        decision_payload.update(
            {"status": "failed", "blocking_reason": "worker_failed"}
        )

    for artifact_name in expected:
        artifact = payload["artifacts"].get(artifact_name)
        if artifact is None or artifact["producer_task_id"] != task_id:
            raise OrchestratorExecutionStateError(
                "task_result_artifact_identity_mismatch"
            )
        artifact.update(
            {
                "status": (
                    "available"
                    if artifact_name in available_output_artifact_names
                    else "invalid"
                ),
                "artifact_id": output_artifact_refs.get(artifact_name),
            }
        )
    try:
        transitioned = OrchestratorExecutionState.model_validate(payload)
    except ValidationError as exc:
        raise OrchestratorExecutionStateError("task_result_state_invalid") from exc
    aggregated = recompute_aggregate_state(transitioned)
    result_received = aggregated.model_dump()
    result_received["run_status"] = "running"
    result_received["orchestrator"].update(
        {
            "status": "evaluating_results",
            "next_wakeup_reason": "worker_result_received",
        }
    )
    result_received["next_wakeup"] = {
        "target": "orchestrator_loop",
        "reason": "worker_result_received",
    }
    try:
        return OrchestratorExecutionState.model_validate(result_received)
    except ValidationError as exc:
        raise OrchestratorExecutionStateError(
            "task_result_received_state_invalid"
        ) from exc


def reconcile_execution_state_after_revalidation(
    state: OrchestratorExecutionState,
    result: OrchestratorRoutingServiceResult,
) -> tuple[OrchestratorExecutionState, tuple[PreparedA2ATask, ...]]:
    """Merge one revalidated plan without replacing runtime task history.

    The routing service remains the dependency-DAG authority.  This reducer only
    reconciles its stable decision/task identities into compact runtime state.
    Prepared A2A tasks are returned ephemerally and are never embedded in state.
    """
    checked = _validated_copy(state)
    plan = result.plan
    if plan.run_id != checked.run_id:
        raise OrchestratorExecutionStateError("revalidation_run_identity_mismatch")
    if plan.routing_plan_id != checked.routing.routing_plan_id:
        raise OrchestratorExecutionStateError("revalidation_plan_identity_mismatch")

    plan_decisions = {
        decision.routing_decision_id: decision
        for decision in plan.validated_decisions
    }
    if len(plan_decisions) != len(plan.validated_decisions) or set(plan_decisions) != set(
        checked.routing.decisions
    ):
        raise OrchestratorExecutionStateError("revalidation_decision_set_mismatch")

    prepared_by_decision: dict[str, PreparedA2ATask] = {}
    for prepared in result.prepared_tasks:
        decision_id = prepared.decision.routing_decision_id
        if decision_id in prepared_by_decision:
            raise OrchestratorExecutionStateError("prepared_task_identity_conflict")
        prepared_by_decision[decision_id] = prepared

    payload = checked.model_dump()
    for decision_id, planned in plan_decisions.items():
        current = checked.routing.decisions[decision_id]
        if (
            current.agent_id != planned.agent_id
            or current.capability_id != planned.capability_id
            or current.required_artifact_names != planned.required_artifact_names
            or current.expected_output_artifact_names
            != planned.expected_output_artifact_names
        ):
            raise OrchestratorExecutionStateError(
                "revalidation_decision_identity_mismatch"
            )

        runtime_terminal = current.status in {
            "completed",
            "failed",
            "dispatched",
        } or any(
            checked.worker_tasks[task_id].dispatch_status
            in {"dispatching", "dispatched", "dispatch_failed"}
            or checked.worker_tasks[task_id].execution_status
            in {"running", "completed", "failed", "canceled"}
            for task_id in current.task_ids
        )
        if not runtime_terminal:
            payload["routing"]["decisions"][decision_id].update(
                {
                    "status": _routing_decision_status(planned.validation_status),
                    "blocking_reason": _blocking_reason(
                        planned.validation_status, planned.reason
                    ),
                }
            )

        prepared = prepared_by_decision.get(decision_id)
        if prepared is None:
            continue
        task_id = prepared.decision.task_id
        if (
            task_id is None
            or planned.task_id != task_id
            or prepared.decision.agent_id != planned.agent_id
            or prepared.decision.capability_id != planned.capability_id
            or prepared.dispatch_target.agent_id != planned.agent_id
            or prepared.dispatch_target.capability_id != planned.capability_id
            or str(prepared.task.id) != task_id
            or prepared.decision.validation_status != "ready"
        ):
            raise OrchestratorExecutionStateError("prepared_task_identity_mismatch")

        existing_task = checked.worker_tasks.get(task_id)
        if existing_task is not None:
            if (
                existing_task.routing_plan_id != plan.routing_plan_id
                or existing_task.routing_decision_id != decision_id
                or existing_task.agent_id != planned.agent_id
                or existing_task.capability_id != planned.capability_id
            ):
                raise OrchestratorExecutionStateError(
                    "prepared_task_identity_conflict"
                )
        else:
            if current.task_ids:
                raise OrchestratorExecutionStateError(
                    "prepared_task_identity_conflict"
                )
            payload["worker_tasks"][task_id] = WorkerTaskExecutionState(
                task_id=task_id,
                routing_plan_id=plan.routing_plan_id,
                routing_decision_id=decision_id,
                agent_id=planned.agent_id,
                capability_id=planned.capability_id,
                dispatch_status="not_dispatched",
                execution_status="not_started",
                result_status=None,
                agent_failure_reason="none",
                output_artifact_refs={},
            ).model_dump()
            payload["routing"]["decisions"][decision_id]["task_ids"] = [task_id]

        for artifact_name in planned.expected_output_artifact_names:
            artifact = payload["artifacts"].get(artifact_name)
            if artifact is None:
                payload["artifacts"][artifact_name] = ArtifactExecutionState(
                    artifact_name=artifact_name,
                    status="planned",
                    producer_task_id=task_id,
                ).model_dump()
            elif artifact["producer_task_id"] not in {None, task_id}:
                raise OrchestratorExecutionStateError(
                    "revalidation_artifact_producer_mismatch"
                )
            elif artifact["status"] not in {"available", "invalid"}:
                artifact["producer_task_id"] = task_id
                artifact["status"] = "planned"

        for artifact_name, ref in prepared.input_artifact_refs.items():
            if ref.run_id != plan.run_id or ref.artifact_type != artifact_name:
                raise OrchestratorExecutionStateError(
                    "input_artifact_ref_identity_mismatch"
                )
            artifact = payload["artifacts"].get(artifact_name)
            if artifact is None:
                payload["artifacts"][artifact_name] = ArtifactExecutionState(
                    artifact_name=artifact_name,
                    status="available",
                    artifact_id=ref.artifact_id,
                ).model_dump()
            elif artifact["status"] != "available" or artifact["artifact_id"] != ref.artifact_id:
                raise OrchestratorExecutionStateError(
                    "input_artifact_ref_state_mismatch"
                )

    if set(prepared_by_decision) - set(plan_decisions):
        raise OrchestratorExecutionStateError("prepared_task_decision_unknown")

    if plan.routing_status == "completed":
        payload["run_status"] = "completed"
        payload["orchestrator"]["status"] = "completed"
        payload["orchestrator"]["next_wakeup_reason"] = "routing_completed"
        payload["next_wakeup"] = {
            "target": "final_response",
            "reason": "routing_completed",
        }
    else:
        payload["run_status"] = "running"
        payload["orchestrator"]["status"] = "validating"
        payload["orchestrator"]["next_wakeup_reason"] = "dependencies_pending"
        payload["next_wakeup"] = {
            "target": "orchestrator_loop",
            "reason": "dependencies_pending",
        }
    try:
        reconciled = recompute_aggregate_state(
            OrchestratorExecutionState.model_validate(payload)
        )
    except ValidationError as exc:
        raise OrchestratorExecutionStateError(
            "reconciled_execution_state_invalid"
        ) from exc

    eligible = set(dispatch_eligible_task_ids(reconciled))
    prepared_for_dispatch = tuple(
        prepared
        for prepared in result.prepared_tasks
        if str(prepared.task.id) in eligible
    )
    if {str(item.task.id) for item in prepared_for_dispatch} != eligible:
        raise OrchestratorExecutionStateError("ready_task_preparation_mismatch")
    return reconciled, prepared_for_dispatch


def transition_task_lifecycle(
    state: OrchestratorExecutionState,
    task_id: str,
    *,
    dispatch_status: str,
    execution_status: str,
    agent_failure_reason: str = "none",
) -> OrchestratorExecutionState:
    """Strict lower-level transition seam used by lifecycle commands."""
    return _transition_task(
        state,
        task_id,
        dispatch_status=dispatch_status,
        execution_status=execution_status,
        agent_failure_reason=agent_failure_reason,
    )


def _transition_task(
    state: OrchestratorExecutionState,
    task_id: str,
    *,
    dispatch_status: str,
    execution_status: str,
    agent_failure_reason: str,
) -> OrchestratorExecutionState:
    current_state = _validated_copy(state)
    current = current_state.worker_tasks.get(task_id)
    if current is None:
        raise OrchestratorExecutionStateError("task_id_unknown")
    target = (dispatch_status, execution_status, agent_failure_reason)
    source = (
        current.dispatch_status,
        current.execution_status,
        current.agent_failure_reason,
    )
    if target == source:
        return recompute_aggregate_state(current_state)
    allowed = {
        ("not_dispatched", "not_started", "none"): {
            ("dispatching", "not_started", "none")
        },
        ("dispatching", "not_started", "none"): {
            ("dispatched", "not_started", "none"),
            *{
                ("dispatch_failed", "not_started", reason)
                for reason in _DISPATCH_FAILURE_REASONS
            },
        },
        ("dispatched", "not_started", "none"): {
            ("dispatched", "running", "none")
        },
    }
    if target not in allowed.get(source, set()):
        raise OrchestratorExecutionStateError("task_lifecycle_transition_invalid")

    payload = current_state.model_dump()
    payload["worker_tasks"][task_id].update(
        {
            "dispatch_status": dispatch_status,
            "execution_status": execution_status,
            "agent_failure_reason": agent_failure_reason,
        }
    )
    decision_id = current.routing_decision_id
    if dispatch_status == "dispatched":
        payload["routing"]["decisions"][decision_id]["status"] = "dispatched"
    elif dispatch_status == "dispatch_failed":
        payload["routing"]["decisions"][decision_id].update(
            {"status": "failed", "blocking_reason": "dispatch_failed"}
        )
    try:
        transitioned = OrchestratorExecutionState.model_validate(payload)
    except ValidationError as exc:
        raise OrchestratorExecutionStateError("task_lifecycle_state_invalid") from exc
    return recompute_aggregate_state(transitioned)


def _validated_copy(state: OrchestratorExecutionState) -> OrchestratorExecutionState:
    if not isinstance(state, OrchestratorExecutionState):
        raise OrchestratorExecutionStateError("execution_state_type_invalid")
    try:
        return OrchestratorExecutionState.model_validate(state.model_dump())
    except ValidationError as exc:
        raise OrchestratorExecutionStateError("execution_state_identity_invalid") from exc


def _routing_decision_status(validation_status: str) -> str:
    mapping = {
        "ready": "ready",
        "waiting_for_dependencies": "pending_dependency",
        "blocked_missing_dependency": "blocked",
        "wait_for_input": "blocked",
        "rejected": "blocked",
    }
    try:
        return mapping[validation_status]
    except KeyError as exc:
        raise OrchestratorExecutionStateError(
            "routing_validation_status_invalid"
        ) from exc


def _blocking_reason(validation_status: str, reason: str | None) -> str | None:
    if validation_status == "ready":
        return None
    if validation_status == "waiting_for_dependencies":
        return "missing_required_artifact"
    if validation_status == "wait_for_input":
        return "needs_user_input"
    if reason == "required_artifact_not_ready":
        return "input_not_ready"
    if reason in {"missing_required_artifact", "waiting_for_dependencies"}:
        return "missing_required_artifact"
    return "validation_failed"


def recompute_aggregate_state(
    state: OrchestratorExecutionState,
) -> OrchestratorExecutionState:
    """Recompute top-level orchestration and producer artifact state once."""
    checked = _validated_copy(state)
    payload = checked.model_dump()

    for artifact in payload["artifacts"].values():
        producer_task_id = artifact["producer_task_id"]
        if producer_task_id is None:
            continue
        task = payload["worker_tasks"][producer_task_id]
        if task["dispatch_status"] == "dispatch_failed":
            artifact["status"] = "invalid"
        elif task["dispatch_status"] == "dispatched" and task[
            "execution_status"
        ] in {"not_started", "running"}:
            artifact["status"] = "producing"
        elif task["dispatch_status"] in {"not_dispatched", "dispatching"}:
            artifact["status"] = "planned"

    tasks = payload["worker_tasks"]
    decisions = payload["routing"]["decisions"]
    eligible = [
        task
        for task in tasks.values()
        if task["dispatch_status"] == "not_dispatched"
        and task["execution_status"] == "not_started"
        and decisions[task["routing_decision_id"]]["status"] == "ready"
    ]
    dispatching = [
        task for task in tasks.values() if task["dispatch_status"] == "dispatching"
    ]
    active_workers = [
        task
        for task in tasks.values()
        if task["dispatch_status"] == "dispatched"
        and task["execution_status"] in {"not_started", "running"}
    ]
    dispatch_failures = [
        task
        for task in tasks.values()
        if task["dispatch_status"] == "dispatch_failed"
    ]
    if eligible:
        run_status = "running"
        orchestrator_status = "dispatching"
        wakeup = NextWakeupState(
            target="worker_dispatch", reason="ready_tasks_available"
        )
    elif dispatching:
        run_status = "running"
        orchestrator_status = "dispatching"
        wakeup = NextWakeupState(
            target="worker_dispatch", reason="dispatch_in_progress"
        )
    elif active_workers:
        run_status = "running"
        orchestrator_status = "waiting_for_workers"
        wakeup = NextWakeupState(
            target="orchestrator_loop", reason="worker_result_received"
        )
    elif checked.run_status == "completed":
        run_status = "completed"
        orchestrator_status = "completed"
        wakeup = NextWakeupState(
            target="final_response", reason="routing_completed"
        )
    elif checked.run_status == "waiting_for_input":
        run_status = "waiting_for_input"
        orchestrator_status = "planning"
        wakeup = NextWakeupState(target="user_input", reason="needs_user_input")
    elif (
        checked.run_status == "running"
        and checked.orchestrator.status == "evaluating_results"
        and checked.next_wakeup is not None
        and checked.next_wakeup.target == "orchestrator_loop"
        and checked.next_wakeup.reason == "worker_result_received"
    ):
        # This is an explicit result-received transition, not a state inferred
        # from historical terminal tasks. A later Orchestrator loop may advance
        # the top-level status without old tasks pulling it back here.
        run_status = "running"
        orchestrator_status = "evaluating_results"
        wakeup = NextWakeupState(
            target="orchestrator_loop", reason="worker_result_received"
        )
    elif (
        checked.run_status == "running"
        and checked.orchestrator.status == "evaluating_results"
        and checked.next_wakeup is not None
        and checked.next_wakeup.reason == "worker_result_reconciliation_required"
    ):
        # Transport uncertainty is an explicit stop-the-loop disposition.  It
        # takes precedence over the generic dispatch-failed aggregate so a
        # checkpoint/reconstruction cannot silently become retry-eligible.
        run_status = "running"
        orchestrator_status = "evaluating_results"
        wakeup = checked.next_wakeup
    elif dispatch_failures:
        run_status = "running"
        orchestrator_status = "evaluating_results"
        wakeup = NextWakeupState(
            target="orchestrator_loop", reason="dispatch_failed"
        )
    elif (
        checked.run_status == "failed"
        and checked.orchestrator.status == "routing_to_final"
        and checked.next_wakeup is not None
        and checked.next_wakeup.target == "final_response"
        and checked.next_wakeup.reason
        in {"worker_retry_exhausted", "worker_non_retryable_failure"}
    ):
        run_status = "failed"
        orchestrator_status = "routing_to_final"
        wakeup = checked.next_wakeup
    elif checked.run_status == "failed" or (
        decisions and all(item["status"] in {"blocked", "failed"} for item in decisions.values())
    ):
        run_status = "failed"
        orchestrator_status = "failed"
        reason = (
            "routing_failed"
            if checked.routing.routing_source in {"llm_failed", "llm_primary_validated"}
            and not decisions
            else "routing_blocked"
        )
        wakeup = NextWakeupState(target="orchestrator_loop", reason=reason)
    else:
        run_status = "running"
        orchestrator_status = "validating"
        wakeup = NextWakeupState(
            target="orchestrator_loop", reason="dependencies_pending"
        )

    payload["run_status"] = run_status
    payload["orchestrator"]["status"] = orchestrator_status
    payload["orchestrator"]["next_wakeup_reason"] = wakeup.reason
    payload["next_wakeup"] = wakeup.model_dump()
    try:
        return OrchestratorExecutionState.model_validate(payload)
    except ValidationError as exc:
        raise OrchestratorExecutionStateError("aggregate_execution_state_invalid") from exc


__all__ = [
    "OrchestratorExecutionStateError",
    "dispatch_eligible_task_ids",
    "execution_state_from_routing_result",
    "mark_task_dispatch_failed",
    "mark_task_dispatched",
    "mark_task_dispatching",
    "mark_task_running",
    "mark_task_result",
    "reconcile_execution_state_after_revalidation",
    "recompute_aggregate_state",
    "transition_task_lifecycle",
]
