"""Production application boundary for the Step 4 HTTP A2A Orchestrator."""

from __future__ import annotations

import asyncio
import math
import threading
from collections import Counter
from typing import Any

from pydantic import ValidationError

from app.graph.orchestrator_execution_graph import execution_graph_config
from app.schemas.orchestrator_api import (
    CompactArtifactCounts,
    CompactArtifactRef,
    CompactDecisionCounts,
    CompactTaskCounts,
    OrchestratorStep4Response,
)
from app.schemas.orchestrator_execution_state import OrchestratorExecutionState

from .orchestrator_execution_loop import execute_orchestrator_worker_loop
from .orchestrator_execution_state import (
    dispatch_eligible_task_ids,
    execution_state_from_routing_result,
)
from .orchestrator_resume import resume_orchestrator_run


class OrchestratorApplicationServiceError(RuntimeError):
    """Fixed compact application-boundary failure."""


class OrchestratorApplicationService:
    """Compose existing planning/checkpoint/HTTP execution and resume APIs.

    The service owns no worker implementation and never calls a domain agent.
    One per-run async lock serializes duplicate/concurrent requests within the
    backend process; the durable checkpoint remains the restart authority.
    """

    def __init__(
        self,
        *,
        checkpoint_runtime: Any,
        routing_service: Any,
        discovery: Any,
        registry: Any,
        storage: Any,
        worker_timeout_seconds: float,
        max_worker_retries: int,
        client_factory: Any = None,
    ) -> None:
        try:
            timeout = float(worker_timeout_seconds)
        except (TypeError, ValueError):
            raise OrchestratorApplicationServiceError(
                "orchestrator_worker_timeout_invalid"
            ) from None
        if not math.isfinite(timeout) or timeout <= 0:
            raise OrchestratorApplicationServiceError(
                "orchestrator_worker_timeout_invalid"
            )
        if int(max_worker_retries) != 3:
            raise OrchestratorApplicationServiceError(
                "worker_retry_policy_invalid"
            )
        self._runtime = checkpoint_runtime
        self._routing_service = routing_service
        self._discovery = discovery
        self._registry = registry
        self._storage = storage
        self._worker_timeout_seconds = timeout
        self._max_worker_retries = int(max_worker_retries)
        self._client_factory = client_factory
        self._meta_lock = threading.Lock()
        self._run_locks: dict[str, asyncio.Lock] = {}

    async def execute(self, run_id: str) -> OrchestratorStep4Response:
        """Plan a fresh run, or resume when a durable checkpoint exists."""
        async with self._run_lock(run_id):
            try:
                async with self._runtime.run_lock(run_id):
                    return await self._execute_locked(run_id)
            except OrchestratorApplicationServiceError:
                raise
            except Exception as exc:
                raise OrchestratorApplicationServiceError(
                    "orchestrator_run_busy"
                    if str(exc) == "checkpoint_run_lock_unavailable"
                    else "orchestrator_run_lock_failed"
                ) from None

    async def _execute_locked(self, run_id: str) -> OrchestratorStep4Response:
        existing = await self._load_state(run_id, required=False)
        if existing is not None:
            return await self._resume_locked(run_id)

        try:
            routing = await _plan_for_run_cancellation_safe(
                self._routing_service, run_id
            )
            initial = execution_state_from_routing_result(routing)
            checkpointed = await self._checkpoint(run_id, initial)
        except OrchestratorApplicationServiceError:
            raise
        except Exception:
            raise OrchestratorApplicationServiceError(
                "orchestrator_planning_failed"
            ) from None

        if not dispatch_eligible_task_ids(checkpointed):
            return compact_orchestrator_response(
                checkpointed,
                checkpoint_reused=False,
                llm_routing_called=routing.llm_called,
            )

        loop_kwargs: dict[str, Any] = {}
        if self._client_factory is not None:
            loop_kwargs["client_factory"] = self._client_factory
        try:
            loop = await execute_orchestrator_worker_loop(
                run_id=run_id,
                state=checkpointed,
                prepared_tasks=routing.prepared_tasks,
                routing_service=self._routing_service,
                discovery=self._discovery,
                registry=self._registry,
                storage=self._storage,
                execution_graph=self._runtime.graph,
                checkpoint_config=execution_graph_config(run_id),
                timeout_seconds=self._worker_timeout_seconds,
                max_worker_retries=self._max_worker_retries,
                **loop_kwargs,
            )
        except Exception:
            raise OrchestratorApplicationServiceError(
                "orchestrator_execution_failed"
            ) from None
        return compact_orchestrator_response(
            loop.state,
            checkpoint_reused=False,
            llm_routing_called=routing.llm_called,
            dispatch_attempt_count=loop.dispatch_attempt_count,
            outcome_hint=loop.outcome,
        )

    async def resume(self, run_id: str) -> OrchestratorStep4Response:
        """Explicitly resume one existing durable checkpoint."""
        async with self._run_lock(run_id):
            try:
                async with self._runtime.run_lock(run_id):
                    await self._load_state(run_id, required=True)
                    return await self._resume_locked(run_id)
            except OrchestratorApplicationServiceError:
                raise
            except Exception as exc:
                raise OrchestratorApplicationServiceError(
                    "orchestrator_run_busy"
                    if str(exc) == "checkpoint_run_lock_unavailable"
                    else "orchestrator_run_lock_failed"
                ) from None

    async def status(self, run_id: str) -> OrchestratorStep4Response:
        """Read the latest durable compact state without discovery or dispatch."""
        state = await self._load_state(run_id, required=True)
        assert state is not None
        try:
            await asyncio.to_thread(
                self._routing_service.validate_execution_state_authority,
                run_id,
                state,
            )
        except Exception:
            raise OrchestratorApplicationServiceError(
                "orchestrator_status_authority_invalid"
            ) from None
        return compact_orchestrator_response(
            state,
            checkpoint_reused=True,
            llm_routing_called=False,
        )

    async def _resume_locked(self, run_id: str) -> OrchestratorStep4Response:
        kwargs: dict[str, Any] = {}
        if self._client_factory is not None:
            kwargs["client_factory"] = self._client_factory
        try:
            resumed = await resume_orchestrator_run(
                run_id=run_id,
                checkpoint_runtime=self._runtime,
                routing_service=self._routing_service,
                discovery=self._discovery,
                registry=self._registry,
                storage=self._storage,
                timeout_seconds=self._worker_timeout_seconds,
                max_worker_retries=self._max_worker_retries,
                **kwargs,
            )
        except Exception:
            raise OrchestratorApplicationServiceError(
                "orchestrator_resume_failed"
            ) from None
        return compact_orchestrator_response(
            resumed.state,
            checkpoint_reused=True,
            llm_routing_called=False,
            dispatch_attempt_count=resumed.dispatch_attempt_count,
            outcome_hint=resumed.outcome,
        )

    async def _load_state(
        self, run_id: str, *, required: bool
    ) -> OrchestratorExecutionState | None:
        try:
            config = execution_graph_config(run_id)
            snapshot = await self._runtime.graph.aget_state(config)
            values = getattr(snapshot, "values", None)
        except Exception:
            raise OrchestratorApplicationServiceError(
                "orchestrator_checkpoint_read_failed"
            ) from None
        if not values:
            if required:
                raise OrchestratorApplicationServiceError(
                    "orchestrator_checkpoint_not_found"
                )
            return None
        try:
            return OrchestratorExecutionState.model_validate(values)
        except ValidationError:
            raise OrchestratorApplicationServiceError(
                "orchestrator_checkpoint_state_invalid"
            ) from None

    async def _checkpoint(
        self, run_id: str, state: OrchestratorExecutionState
    ) -> OrchestratorExecutionState:
        try:
            payload = await self._runtime.graph.ainvoke(
                state, config=execution_graph_config(run_id)
            )
            checked = OrchestratorExecutionState.model_validate(payload)
        except Exception:
            raise OrchestratorApplicationServiceError(
                "orchestrator_initial_checkpoint_failed"
            ) from None
        if checked != state:
            raise OrchestratorApplicationServiceError(
                "orchestrator_initial_checkpoint_mismatch"
            )
        return checked

    def _run_lock(self, run_id: str) -> asyncio.Lock:
        with self._meta_lock:
            return self._run_locks.setdefault(run_id, asyncio.Lock())


async def _plan_for_run_cancellation_safe(
    routing_service: Any, run_id: str
) -> Any:
    """Quiesce the side-effecting planning thread before releasing run locks.

    ``asyncio.to_thread`` cannot stop its worker thread when the awaiting task
    is cancelled.  Shielding and then draining the thread prevents a second
    backend from entering planning while the first planner can still write its
    routing authority.  The original cancellation is propagated after the
    thread has stopped; its result is deliberately not checkpointed/dispatched.
    """
    planning_task = asyncio.create_task(
        asyncio.to_thread(routing_service.plan_for_run, run_id)
    )
    try:
        return await asyncio.shield(planning_task)
    except asyncio.CancelledError:
        while not planning_task.done():
            try:
                await asyncio.shield(planning_task)
            except asyncio.CancelledError:
                continue
        try:
            planning_task.result()
        except BaseException:
            pass
        raise


def compact_orchestrator_response(
    state: OrchestratorExecutionState,
    *,
    checkpoint_reused: bool,
    llm_routing_called: bool,
    dispatch_attempt_count: int = 0,
    outcome_hint: str | None = None,
) -> OrchestratorStep4Response:
    """Project public counts/refs only; transport and proof objects are absent."""
    checked = OrchestratorExecutionState.model_validate(state.model_dump())
    decision_statuses = Counter(
        item.status for item in checked.routing.decisions.values()
    )
    dispatch_statuses = Counter(
        item.dispatch_status for item in checked.worker_tasks.values()
    )
    execution_statuses = Counter(
        item.execution_status for item in checked.worker_tasks.values()
    )
    artifact_statuses = Counter(item.status for item in checked.artifacts.values())
    outcome = _public_outcome(checked, outcome_hint)
    action_code = {
        "waiting_for_input": "provide_required_input",
        "reconciliation_required": "reconcile_worker_result",
        "waiting": "wait_for_dependencies",
        "failed": "inspect_compact_failure",
    }.get(outcome)
    return OrchestratorStep4Response(
        run_id=checked.run_id,
        routing_plan_id=checked.routing.routing_plan_id,
        outcome=outcome,
        run_status=checked.run_status,
        orchestrator_status=checked.orchestrator.status,
        next_wakeup=checked.next_wakeup,
        checkpoint_reused=checkpoint_reused,
        llm_routing_called=llm_routing_called,
        dispatch_attempt_count=dispatch_attempt_count,
        decision_counts=CompactDecisionCounts(
            total=len(checked.routing.decisions),
            **{
                status: decision_statuses[status]
                for status in (
                    "ready",
                    "pending_dependency",
                    "blocked",
                    "dispatched",
                    "completed",
                    "failed",
                    "skipped",
                    "planned",
                )
            },
        ),
        task_counts=CompactTaskCounts(
            total=len(checked.worker_tasks),
            not_dispatched=dispatch_statuses["not_dispatched"],
            dispatching=dispatch_statuses["dispatching"],
            dispatched=dispatch_statuses["dispatched"],
            dispatch_failed=dispatch_statuses["dispatch_failed"],
            not_started=execution_statuses["not_started"],
            running=execution_statuses["running"],
            completed=execution_statuses["completed"],
            failed=execution_statuses["failed"],
            retry_tasks=sum(
                item.retry_attempt > 0 for item in checked.worker_tasks.values()
            ),
        ),
        artifact_counts=CompactArtifactCounts(
            total=len(checked.artifacts),
            **{
                status: artifact_statuses[status]
                for status in (
                    "missing",
                    "planned",
                    "producing",
                    "available",
                    "invalid",
                )
            },
        ),
        artifact_refs=[
            CompactArtifactRef(**checked.artifacts[name].model_dump())
            for name in sorted(checked.artifacts)
        ],
        action_code=action_code,
    )


def unavailable_orchestrator_response(
    run_id: str | None, error_code: str
) -> OrchestratorStep4Response:
    """Return one typed compact unavailable response without runtime details."""
    try:
        return OrchestratorStep4Response(
            run_id=run_id,
            outcome="unavailable",
            checkpoint_reused=False,
            llm_routing_called=False,
            error_code=error_code,
        )
    except ValidationError:
        raise OrchestratorApplicationServiceError(
            "orchestrator_request_invalid"
        ) from None


def _public_outcome(state: OrchestratorExecutionState, hint: str | None) -> str:
    if state.run_status == "completed" or hint == "completed":
        return "completed"
    if state.run_status == "waiting_for_input" or hint in {
        "needs_user_input",
        "waiting_for_input",
    }:
        return "waiting_for_input"
    if hint == "reconciliation_required" or (
        state.next_wakeup is not None
        and state.next_wakeup.reason == "worker_result_reconciliation_required"
    ):
        return "reconciliation_required"
    if state.run_status == "failed" or hint in {
        "retry_exhausted",
        "non_retryable_failure",
        "failed",
    }:
        return "failed"
    return "waiting"


__all__ = [
    "OrchestratorApplicationService",
    "OrchestratorApplicationServiceError",
    "compact_orchestrator_response",
    "unavailable_orchestrator_response",
]
