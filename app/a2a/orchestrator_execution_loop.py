"""Public deterministic dispatch/ingest/revalidate/retry execution loop."""

from __future__ import annotations

import copy
from collections.abc import Callable, Mapping, Sequence
from types import MappingProxyType
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, PrivateAttr, ValidationError
from python_a2a import A2AClient

from app.schemas.orchestrator_execution_state import OrchestratorExecutionState
from app.services.artifact_registry_service import ArtifactRegistryService
from app.services.storage_service import Storage

from .contracts import WorkerExecutionResult
from .orchestrator_dispatch import dispatch_orchestrator_tasks
from .orchestrator_discovery import WorkerDiscoveryService
from .orchestrator_execution_state import dispatch_eligible_task_ids
from .orchestrator_post_ingestion import revalidate_orchestrator_after_ingestion
from .orchestrator_result_ingestion import ingest_orchestrator_worker_results
from .orchestrator_routing_service import OrchestratorRoutingService
from .orchestrator_retry import (
    MAX_WORKER_RETRIES,
    prepare_orchestrator_retries,
)
from .orchestrator_task_builder import PreparedA2ATask


class OrchestratorExecutionLoopError(RuntimeError):
    """Compact fail-closed execution-loop error."""


class ExecutionLoopCheckpointError(OrchestratorExecutionLoopError):
    """Checkpoint failure with compact in-process recovery state."""

    __slots__ = ("_recovery_result",)

    def __init__(self, recovery_result: "OrchestratorExecutionLoopResult") -> None:
        super().__init__("execution_loop_checkpoint_failed")
        self._recovery_result = recovery_result

    @property
    def recovery_result(self) -> "OrchestratorExecutionLoopResult":
        return self._recovery_result

    def __repr__(self) -> str:
        return "ExecutionLoopCheckpointError('execution_loop_checkpoint_failed')"

    def __reduce_ex__(self, protocol: int) -> Any:
        raise TypeError("execution_loop_checkpoint_error_pickle_unsupported")


class OrchestratorExecutionLoopResult(BaseModel):
    """Compact outcome with private same-process continuation authority."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    state: OrchestratorExecutionState
    outcome: Literal[
        "completed",
        "retry_exhausted",
        "non_retryable_failure",
        "needs_user_input",
        "reconciliation_required",
        "waiting",
    ]
    dispatch_round_count: int
    dispatch_attempt_count: int
    _completion_proofs: dict[str, WorkerExecutionResult] = PrivateAttr(
        default_factory=dict
    )
    _prepared_task_history: dict[str, PreparedA2ATask] = PrivateAttr(
        default_factory=dict
    )

    def __init__(self, **data: Any) -> None:
        proofs = data.pop("completion_proofs", {})
        history = data.pop("prepared_task_history", {})
        super().__init__(**data)
        self._completion_proofs = {
            key: value.model_copy(deep=True) for key, value in dict(proofs).items()
        }
        self._prepared_task_history = copy.deepcopy(dict(history))

    @property
    def completion_proofs(self) -> Mapping[str, WorkerExecutionResult]:
        return MappingProxyType(
            {
                key: value.model_copy(deep=True)
                for key, value in self._completion_proofs.items()
            }
        )

    @property
    def prepared_task_history(self) -> Mapping[str, PreparedA2ATask]:
        return MappingProxyType(copy.deepcopy(self._prepared_task_history))

    def __reduce_ex__(self, protocol: int) -> Any:
        raise TypeError("orchestrator_execution_loop_result_pickle_unsupported")


async def execute_orchestrator_worker_loop(
    *,
    run_id: str,
    state: OrchestratorExecutionState,
    prepared_tasks: Sequence[PreparedA2ATask],
    routing_service: OrchestratorRoutingService,
    discovery: WorkerDiscoveryService,
    registry: ArtifactRegistryService,
    storage: Storage,
    execution_graph: Any,
    checkpoint_config: Any,
    timeout_seconds: float,
    max_worker_retries: int,
    completion_proofs: Mapping[str, WorkerExecutionResult] | None = None,
    client_factory: Callable[..., A2AClient] = A2AClient,
) -> OrchestratorExecutionLoopResult:
    """Run complete eligible HTTP batches until success, exhaustion, or uncertainty."""
    if max_worker_retries != MAX_WORKER_RETRIES:
        raise OrchestratorExecutionLoopError("worker_retry_policy_invalid")
    current = _checked_state(run_id, state)
    proofs = dict(completion_proofs or {})
    history: dict[str, PreparedA2ATask] = {}
    current_prepared = tuple(copy.deepcopy(tuple(prepared_tasks)))
    for item in current_prepared:
        history[str(item.task.id)] = copy.deepcopy(item)
    dispatch_round_count = 0
    dispatch_attempt_count = 0

    while True:
        if _has_transport_uncertainty(current):
            recovery = _result(
                state=_mark_reconciliation_required(current),
                outcome="reconciliation_required",
                rounds=dispatch_round_count,
                attempts=dispatch_attempt_count,
                proofs=proofs,
                history=history,
            )
            if recovery.state != current:
                current = await _checkpoint_transition(
                    execution_graph,
                    recovery.state,
                    checkpoint_config,
                    recovery,
                )
            else:
                current = recovery.state
            return _result(
                state=current,
                outcome="reconciliation_required",
                rounds=dispatch_round_count,
                attempts=dispatch_attempt_count,
                proofs=proofs,
                history=history,
            )

        eligible = dispatch_eligible_task_ids(current)
        if eligible:
            if not current_prepared:
                current_prepared = _rebuild_pending_retries(
                    run_id=run_id,
                    state=current,
                    task_ids=eligible,
                    routing_service=routing_service,
                )
                for item in current_prepared:
                    history[str(item.task.id)] = copy.deepcopy(item)
            if {str(item.task.id) for item in current_prepared} != set(eligible):
                raise OrchestratorExecutionLoopError(
                    "execution_loop_prepared_task_set_mismatch"
                )
            dispatched = await dispatch_orchestrator_tasks(
                run_id=run_id,
                state=current,
                prepared_tasks=current_prepared,
                discovery=discovery,
                execution_graph=execution_graph,
                checkpoint_config=checkpoint_config,
                timeout_seconds=timeout_seconds,
                client_factory=client_factory,
                routing_service=routing_service,
            )
            dispatch_round_count += 1
            dispatch_attempt_count += len(current_prepared)
            current = dispatched.state
            transport_uncertain = any(
                receipt.dispatch_status == "dispatch_failed"
                for receipt in dispatched.receipts
            )
            ingested = await ingest_orchestrator_worker_results(
                run_id=run_id,
                dispatch_result=dispatched,
                discovery=discovery,
                registry=registry,
                storage=storage,
                execution_graph=execution_graph,
                checkpoint_config=checkpoint_config,
            )
            proofs.update(dict(ingested.completion_proofs))
            current = ingested.state
            current_prepared = ()
            if transport_uncertain:
                current = _mark_reconciliation_required(current)
                recovery = _result(
                    state=current,
                    outcome="reconciliation_required",
                    rounds=dispatch_round_count,
                    attempts=dispatch_attempt_count,
                    proofs=proofs,
                    history=history,
                )
                current = await _checkpoint_transition(
                    execution_graph,
                    current,
                    checkpoint_config,
                    recovery,
                )
                return _result(
                    state=current,
                    outcome="reconciliation_required",
                    rounds=dispatch_round_count,
                    attempts=dispatch_attempt_count,
                    proofs=proofs,
                    history=history,
                )
            if not _latest_terminal_proof_missing(current, proofs):
                post = await revalidate_orchestrator_after_ingestion(
                    run_id=run_id,
                    ingestion_result=ingested,
                    previous_completion_proofs=proofs,
                    routing_service=routing_service,
                    execution_graph=execution_graph,
                    checkpoint_config=checkpoint_config,
                )
                current = post.state
                proofs = dict(post.completion_proofs)
                current_prepared = tuple(post.prepared_tasks)
                for item in current_prepared:
                    history[str(item.task.id)] = copy.deepcopy(item)

        retry = prepare_orchestrator_retries(
            state=current,
            completion_proofs=proofs,
            max_worker_retries=max_worker_retries,
        )
        if retry.state != current:
            current = retry.state
            retry_prepared = tuple(
                routing_service.rebuild_retry_task(
                    run_id=run_id,
                    execution_state=current,
                    task_id=task_id,
                )
                for task_id in retry.retry_task_ids
            )
            for item in retry_prepared:
                history[str(item.task.id)] = copy.deepcopy(item)
            current_prepared = _merge_prepared(
                current_prepared, retry_prepared
            )
            recovery = _result(
                state=current,
                outcome=_outcome_for_disposition(retry.terminal_disposition),
                rounds=dispatch_round_count,
                attempts=dispatch_attempt_count,
                proofs=proofs,
                history=history,
            )
            current = await _checkpoint_transition(
                execution_graph,
                current,
                checkpoint_config,
                recovery,
            )

        if dispatch_eligible_task_ids(current):
            continue
        if current.run_status == "completed":
            outcome = "completed"
        elif retry.terminal_disposition == "reconciliation_required":
            outcome = "reconciliation_required"
        elif retry.terminal_disposition == "needs_user_input":
            outcome = "needs_user_input"
        elif retry.terminal_disposition == "non_retryable_failure":
            outcome = "non_retryable_failure"
        elif (
            current.run_status == "failed"
            and current.next_wakeup is not None
            and current.next_wakeup.reason == "worker_retry_exhausted"
        ):
            outcome = "retry_exhausted"
        else:
            outcome = "waiting"
        return _result(
            state=current,
            outcome=outcome,
            rounds=dispatch_round_count,
            attempts=dispatch_attempt_count,
            proofs=proofs,
            history=history,
        )


def _merge_prepared(
    existing: Sequence[PreparedA2ATask], retries: Sequence[PreparedA2ATask]
) -> tuple[PreparedA2ATask, ...]:
    merged = {str(item.task.id): item for item in (*existing, *retries)}
    return tuple(merged[key] for key in sorted(merged))


async def _checkpoint_transition(
    graph: Any,
    state: OrchestratorExecutionState,
    config: Any,
    recovery: OrchestratorExecutionLoopResult,
):
    try:
        payload = await graph.ainvoke(state, config=config)
        checked = OrchestratorExecutionState.model_validate(payload)
    except Exception:
        raise ExecutionLoopCheckpointError(recovery) from None
    if checked != state:
        raise OrchestratorExecutionLoopError("execution_loop_checkpoint_mismatch")
    return checked


def _rebuild_pending_retries(*, run_id, state, task_ids, routing_service):
    prepared = []
    for task_id in task_ids:
        task = state.worker_tasks[task_id]
        if task.retry_attempt == 0:
            raise OrchestratorExecutionLoopError(
                "execution_loop_prepared_task_set_mismatch"
            )
        try:
            prepared.append(
                routing_service.rebuild_retry_task(
                    run_id=run_id,
                    execution_state=state,
                    task_id=task_id,
                )
            )
        except Exception:
            raise OrchestratorExecutionLoopError(
                "execution_loop_retry_reconstruction_failed"
            ) from None
    return tuple(prepared)


def _latest_terminal_proof_missing(state, proofs) -> bool:
    for decision in state.routing.decisions.values():
        if not decision.task_ids:
            continue
        task = state.worker_tasks[decision.task_ids[-1]]
        if task.execution_status in {"completed", "failed"} and task.task_id not in proofs:
            return True
    return False


def _has_transport_uncertainty(state: OrchestratorExecutionState) -> bool:
    return any(
        task.dispatch_status == "dispatch_failed"
        for task in state.worker_tasks.values()
    )


def _mark_reconciliation_required(
    state: OrchestratorExecutionState,
) -> OrchestratorExecutionState:
    payload = state.model_dump()
    payload["run_status"] = "running"
    payload["orchestrator"].update(
        {
            "status": "evaluating_results",
            "next_wakeup_reason": "worker_result_reconciliation_required",
        }
    )
    payload["next_wakeup"] = {
        "target": "orchestrator_loop",
        "reason": "worker_result_reconciliation_required",
    }
    try:
        return OrchestratorExecutionState.model_validate(payload)
    except ValidationError:
        raise OrchestratorExecutionLoopError(
            "execution_loop_reconciliation_state_invalid"
        ) from None


def _outcome_for_disposition(disposition):
    return {
        "retry_exhausted": "retry_exhausted",
        "non_retryable_failure": "non_retryable_failure",
        "needs_user_input": "needs_user_input",
        "reconciliation_required": "reconciliation_required",
    }.get(disposition, "waiting")


def _checked_state(run_id: str, state: OrchestratorExecutionState):
    try:
        checked = OrchestratorExecutionState.model_validate(state.model_dump())
    except (AttributeError, ValidationError):
        raise OrchestratorExecutionLoopError("execution_loop_state_invalid") from None
    if checked.run_id != run_id:
        raise OrchestratorExecutionLoopError(
            "execution_loop_run_identity_mismatch"
        )
    return checked


def _result(*, state, outcome, rounds, attempts, proofs, history):
    return OrchestratorExecutionLoopResult(
        state=state,
        outcome=outcome,
        dispatch_round_count=rounds,
        dispatch_attempt_count=attempts,
        completion_proofs=proofs,
        prepared_task_history=history,
    )


__all__ = [
    "OrchestratorExecutionLoopError",
    "ExecutionLoopCheckpointError",
    "OrchestratorExecutionLoopResult",
    "execute_orchestrator_worker_loop",
]
