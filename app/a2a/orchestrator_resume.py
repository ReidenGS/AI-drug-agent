"""Internal cross-process resume entrypoint for durable checkpoints."""

from __future__ import annotations

import re
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, PrivateAttr, ValidationError

from app.graph.orchestrator_execution_graph import execution_graph_config
from app.schemas.orchestrator_execution_state import OrchestratorExecutionState

from .contracts import WorkerExecutionResult
from .orchestrator_execution_loop import execute_orchestrator_worker_loop
from .orchestrator_execution_state import dispatch_eligible_task_ids
from .orchestrator_post_ingestion import revalidate_orchestrator_after_ingestion
from .orchestrator_reconciliation import reconcile_orchestrator_tasks
from .orchestrator_result_ingestion import OrchestratorResultIngestionResult

_COMPACT_CODE = re.compile(r"^[a-z][a-z0-9_]{0,127}$")


class OrchestratorResumeError(RuntimeError):
    """Compact resume failure without checkpoint, task, or endpoint payload."""


class OrchestratorResumeResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    state: OrchestratorExecutionState
    outcome: Literal[
        "completed",
        "failed",
        "waiting_for_input",
        "reconciliation_required",
        "waiting",
    ]
    dispatch_attempt_count: int = 0
    _completion_proofs: dict[str, WorkerExecutionResult] = PrivateAttr(
        default_factory=dict
    )

    def __init__(self, **data: Any) -> None:
        proofs = data.pop("completion_proofs", {})
        super().__init__(**data)
        self._completion_proofs = {
            key: value.model_copy(deep=True)
            for key, value in dict(proofs).items()
        }

    @property
    def completion_proofs(self) -> Mapping[str, WorkerExecutionResult]:
        return MappingProxyType(
            {
                key: value.model_copy(deep=True)
                for key, value in self._completion_proofs.items()
            }
        )

    def __reduce_ex__(self, protocol: int) -> Any:
        raise TypeError("orchestrator_resume_result_pickle_unsupported")


async def resume_orchestrator_run(
    *,
    run_id: str,
    checkpoint_runtime: Any,
    routing_service: Any,
    discovery: Any,
    registry: Any,
    storage: Any,
    timeout_seconds: float,
    max_worker_retries: int,
    client_factory: Any = None,
) -> OrchestratorResumeResult:
    """Resume from Postgres without relying on prior-process Task objects."""
    graph = checkpoint_runtime.graph
    config = execution_graph_config(run_id)
    try:
        snapshot = await graph.aget_state(config)
        state = OrchestratorExecutionState.model_validate(snapshot.values)
        routing_service.validate_execution_state_authority(run_id, state)
    except (AttributeError, ValidationError):
        raise OrchestratorResumeError("resume_checkpoint_state_invalid") from None
    except Exception as exc:
        code = str(exc)
        if not _COMPACT_CODE.fullmatch(code):
            code = "resume_authority_invalid"
        raise OrchestratorResumeError(code) from None

    if state.run_status == "completed":
        return OrchestratorResumeResult(state=state, outcome="completed")
    if state.run_status == "failed" and state.orchestrator.status in {
        "routing_to_final",
        "failed",
    }:
        return OrchestratorResumeResult(state=state, outcome="failed")
    if state.run_status == "waiting_for_input":
        return OrchestratorResumeResult(
            state=state, outcome="waiting_for_input"
        )
    try:
        discovery.discover_for_run(run_id)
    except Exception:
        raise OrchestratorResumeError("resume_discovery_unavailable") from None

    proof_task_ids = tuple(
        sorted(
            task.task_id
            for task in state.worker_tasks.values()
            if task.execution_status in {"completed", "failed"}
            or task.dispatch_status
            in {"dispatching", "dispatched", "dispatch_failed"}
        )
    )
    proofs: dict[str, WorkerExecutionResult] = {}
    if proof_task_ids:
        kwargs = {}
        if client_factory is not None:
            kwargs["client_factory"] = client_factory
        reconciled = await reconcile_orchestrator_tasks(
            run_id=run_id,
            state=state,
            task_ids=proof_task_ids,
            routing_service=routing_service,
            discovery=discovery,
            registry=registry,
            storage=storage,
            execution_graph=graph,
            checkpoint_config=config,
            timeout_seconds=timeout_seconds,
            **kwargs,
        )
        state = reconciled.state
        proofs.update(dict(reconciled.completion_proofs))
        if reconciled.status == "reconciliation_required":
            return OrchestratorResumeResult(
                state=state,
                outcome="reconciliation_required",
                completion_proofs=proofs,
            )
        if not _proofs_already_reflected_by_successors(state, proofs):
            ingestion = OrchestratorResultIngestionResult(
                state=state,
                receipts=(),
                checkpoint_written=True,
                completion_proofs=proofs,
            )
            post = await revalidate_orchestrator_after_ingestion(
                run_id=run_id,
                ingestion_result=ingestion,
                previous_completion_proofs=proofs,
                routing_service=routing_service,
                execution_graph=graph,
                checkpoint_config=config,
            )
            state = post.state
            proofs = dict(post.completion_proofs)

    eligible = dispatch_eligible_task_ids(state)
    prepared = tuple(
        routing_service.rebuild_task_from_execution_state(
            run_id=run_id, execution_state=state, task_id=task_id
        )
        for task_id in eligible
    )
    if not prepared:
        return OrchestratorResumeResult(
            state=state,
            outcome=_resume_outcome(state=state),
            completion_proofs=proofs,
        )
    loop_kwargs = {}
    if client_factory is not None:
        loop_kwargs["client_factory"] = client_factory
    loop = await execute_orchestrator_worker_loop(
        run_id=run_id,
        state=state,
        prepared_tasks=prepared,
        routing_service=routing_service,
        discovery=discovery,
        registry=registry,
        storage=storage,
        execution_graph=graph,
        checkpoint_config=config,
        timeout_seconds=timeout_seconds,
        max_worker_retries=max_worker_retries,
        completion_proofs=proofs,
        **loop_kwargs,
    )
    return OrchestratorResumeResult(
        state=loop.state,
        outcome=_resume_outcome(state=loop.state, loop_outcome=loop.outcome),
        dispatch_attempt_count=loop.dispatch_attempt_count,
        completion_proofs=loop.completion_proofs,
    )


def _resume_outcome(
    *,
    state: OrchestratorExecutionState,
    loop_outcome: str | None = None,
) -> Literal[
    "completed",
    "failed",
    "waiting_for_input",
    "reconciliation_required",
    "waiting",
]:
    if loop_outcome == "reconciliation_required":
        return "reconciliation_required"
    if state.run_status == "completed" or loop_outcome == "completed":
        return "completed"
    if state.run_status == "waiting_for_input" or loop_outcome == "needs_user_input":
        return "waiting_for_input"
    if state.run_status == "failed" or loop_outcome in {
        "retry_exhausted",
        "non_retryable_failure",
    }:
        return "failed"
    return "waiting"


def _proofs_already_reflected_by_successors(
    state: OrchestratorExecutionState,
    proofs: Mapping[str, WorkerExecutionResult],
) -> bool:
    """Return true when every recovered proof already produced a later task.

    A strict checkpoint lineage containing a successor retry is the durable
    evidence that post-ingestion revalidation for the preceding terminal proof
    already ran.  Re-running that reducer would try to prepare the persisted
    root task while the stable retry is already eligible.
    """
    if not proofs:
        return False
    for task_id in proofs:
        task = state.worker_tasks[task_id]
        decision = state.routing.decisions[task.routing_decision_id]
        if task_id not in decision.task_ids:
            return False
        if decision.task_ids.index(task_id) == len(decision.task_ids) - 1:
            return False
    return True


__all__ = [
    "OrchestratorResumeError",
    "OrchestratorResumeResult",
    "resume_orchestrator_run",
]
