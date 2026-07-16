"""Cross-process reconciliation through python-a2a get_task over real HTTP."""

from __future__ import annotations

import asyncio
import math
import re
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, PrivateAttr, ValidationError
from python_a2a import A2AClient, Task, TaskState
from requests import RequestException

from app.schemas.orchestrator_execution_state import OrchestratorExecutionState
from app.services.artifact_registry_service import ArtifactRegistryService
from app.services.storage_service import Storage

from .contracts import (
    A2ATaskMetadata,
    WorkerExecutionRequest,
    WorkerExecutionResult,
)
from .orchestrator_dispatch import (
    DispatchReceipt,
    OrchestratorDispatchResult,
)
from .orchestrator_result_ingestion import (
    OrchestratorResultIngestionResult,
    ingest_orchestrator_worker_results,
    validate_reconciled_worker_response,
)

_COMPACT_CODE = re.compile(r"^[a-z][a-z0-9_]{0,127}$")


class OrchestratorReconciliationError(RuntimeError):
    """Fixed compact reconciliation failure without endpoint/response detail."""


class OrchestratorReconciliationResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    state: OrchestratorExecutionState
    status: Literal["resolved", "reconciliation_required"]
    resolved_task_ids: tuple[str, ...] = ()
    unresolved_task_ids: tuple[str, ...] = ()
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
        return {
            key: value.model_copy(deep=True)
            for key, value in self._completion_proofs.items()
        }

    def __reduce_ex__(self, protocol: int) -> Any:
        raise TypeError("orchestrator_reconciliation_result_pickle_unsupported")


async def reconcile_orchestrator_tasks(
    *,
    run_id: str,
    state: OrchestratorExecutionState,
    task_ids: Sequence[str],
    routing_service: Any,
    discovery: Any,
    registry: ArtifactRegistryService,
    storage: Storage,
    execution_graph: Any,
    checkpoint_config: Any,
    timeout_seconds: float,
    client_factory: Callable[..., A2AClient] = A2AClient,
) -> OrchestratorReconciliationResult:
    """Fetch and strictly ingest terminal tasks; never resend uncertain work."""
    timeout = _validated_timeout(timeout_seconds)
    checked = OrchestratorExecutionState.model_validate(state.model_dump())
    if checked.run_id != run_id:
        raise OrchestratorReconciliationError(
            "reconciliation_run_identity_mismatch"
        )
    unique_ids = tuple(sorted(set(task_ids)))
    if len(unique_ids) != len(tuple(task_ids)):
        raise OrchestratorReconciliationError("reconciliation_task_duplicate")

    responses: dict[str, Task] = {}
    receipts: list[DispatchReceipt] = []
    reconciled_state = checked
    unresolved: list[str] = []
    for task_id in unique_ids:
        task_state = checked.worker_tasks.get(task_id)
        if task_state is None:
            raise OrchestratorReconciliationError(
                "reconciliation_task_unknown"
            )
        try:
            prepared = routing_service.rebuild_task_from_execution_state(
                run_id=run_id, execution_state=checked, task_id=task_id
            )
            target = discovery.resolve_dispatch_target(
                run_id,
                agent_id=task_state.agent_id,
                capability_id=task_state.capability_id,
                dispatch_mode="python_a2a",
            )
        except Exception:
            raise OrchestratorReconciliationError(
                "reconciliation_authority_invalid"
            ) from None
        response = await _get_task(
            target.dispatch_url,
            task_id,
            timeout=timeout,
            client_factory=client_factory,
        )
        if response is None:
            unresolved.append(task_id)
            continue
        response_state = _validated_response_state(response)
        if _task_is_unknown(response):
            unresolved.append(task_id)
            continue
        if str(getattr(response, "id", "")) != task_id:
            raise OrchestratorReconciliationError(
                "reconciliation_task_identity_mismatch"
            )
        if response_state not in {TaskState.COMPLETED, TaskState.FAILED}:
            unresolved.append(task_id)
            continue
        if not _request_matches_authority(response, prepared.task):
            raise OrchestratorReconciliationError(
                "reconciliation_request_identity_mismatch"
            )
        try:
            validate_reconciled_worker_response(
                run_id=run_id,
                state=checked,
                task_id=task_id,
                response=response,
                discovery=discovery,
                registry=registry,
                storage=storage,
            )
        except Exception as exc:
            code = str(exc)
            if not _COMPACT_CODE.fullmatch(code):
                code = "reconciliation_result_invalid"
            raise OrchestratorReconciliationError(code) from None
        reconciled_state = _mark_transport_reconciled(
            reconciled_state, task_id
        )
        responses[task_id] = response
        receipts.append(
            DispatchReceipt(
                task_id=task_id,
                routing_decision_id=task_state.routing_decision_id,
                agent_id=task_state.agent_id,
                capability_id=task_state.capability_id,
                dispatch_status="dispatched",
                agent_failure_reason="none",
            )
        )

    if not responses:
        return OrchestratorReconciliationResult(
            state=checked,
            status="reconciliation_required",
            unresolved_task_ids=tuple(unresolved),
        )
    dispatch_result = OrchestratorDispatchResult(
        state=reconciled_state,
        receipts=tuple(receipts),
        response_tasks=responses,
    )
    ingestion: OrchestratorResultIngestionResult = (
        await ingest_orchestrator_worker_results(
            run_id=run_id,
            dispatch_result=dispatch_result,
            discovery=discovery,
            registry=registry,
            storage=storage,
            execution_graph=execution_graph,
            checkpoint_config=checkpoint_config,
        )
    )
    return OrchestratorReconciliationResult(
        state=ingestion.state,
        status=("reconciliation_required" if unresolved else "resolved"),
        resolved_task_ids=tuple(sorted(responses)),
        unresolved_task_ids=tuple(unresolved),
        completion_proofs=ingestion.completion_proofs,
    )


def _mark_transport_reconciled(
    state: OrchestratorExecutionState, task_id: str
) -> OrchestratorExecutionState:
    payload = state.model_dump()
    task = payload["worker_tasks"][task_id]
    if task["dispatch_status"] in {"dispatching", "dispatch_failed"}:
        task["dispatch_status"] = "dispatched"
        task["agent_failure_reason"] = "none"
    try:
        return OrchestratorExecutionState.model_validate(payload)
    except Exception:
        raise OrchestratorReconciliationError(
            "reconciliation_state_transition_invalid"
        ) from None


async def _get_task(
    endpoint: str,
    task_id: str,
    *,
    timeout: float,
    client_factory: Callable[..., A2AClient],
) -> Task | None:
    def call() -> Task:
        client = client_factory(endpoint, timeout=max(1, math.ceil(timeout)))
        return client.get_task(task_id)

    try:
        return await asyncio.wait_for(asyncio.to_thread(call), timeout=timeout)
    except (asyncio.TimeoutError, TimeoutError, ConnectionError, RequestException):
        return None
    except Exception:
        raise OrchestratorReconciliationError(
            "reconciliation_get_task_protocol_error"
        ) from None


def _validated_response_state(task: Any) -> TaskState:
    if not isinstance(task, Task):
        raise OrchestratorReconciliationError(
            "reconciliation_task_protocol_invalid"
        )
    status = getattr(task, "status", None)
    if status is None or not hasattr(status, "state"):
        raise OrchestratorReconciliationError(
            "reconciliation_task_protocol_invalid"
        )
    try:
        return TaskState(status.state)
    except (TypeError, ValueError):
        raise OrchestratorReconciliationError(
            "reconciliation_task_protocol_invalid"
        ) from None


def _task_is_unknown(task: Task) -> bool:
    return not isinstance(task.artifacts, list) or not task.artifacts


def _request_matches_authority(response: Task, authoritative: Task) -> bool:
    try:
        response_request = WorkerExecutionRequest.model_validate_json(
            response.message["content"]["text"]
        )
        expected_request = WorkerExecutionRequest.model_validate_json(
            authoritative.message["content"]["text"]
        )
        response_metadata = A2ATaskMetadata.model_validate(response.metadata)
        expected_metadata = A2ATaskMetadata.model_validate(
            authoritative.metadata
        )
    except (KeyError, TypeError, ValidationError):
        return False
    return (
        response_request == expected_request
        and response_metadata == expected_metadata
    )


def _validated_timeout(value: float) -> float:
    try:
        timeout = float(value)
    except (TypeError, ValueError):
        raise OrchestratorReconciliationError(
            "reconciliation_timeout_invalid"
        ) from None
    if not math.isfinite(timeout) or timeout <= 0:
        raise OrchestratorReconciliationError(
            "reconciliation_timeout_invalid"
        )
    return timeout


__all__ = [
    "OrchestratorReconciliationError",
    "OrchestratorReconciliationResult",
    "reconcile_orchestrator_tasks",
]
