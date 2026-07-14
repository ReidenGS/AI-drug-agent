"""Concurrent HTTP A2A dispatch for already validated Orchestrator tasks.

This layer owns transport only.  It checkpoints the complete eligible batch as
``dispatching`` before any network call, sends every task through
``A2AClient.send_task_async`` concurrently, and checkpoints the merged
``dispatched`` / ``dispatch_failed`` state once after all calls settle.  Worker
result parsing and artifact validation intentionally belong to the next turn.
"""

from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Callable, Literal, Mapping, Sequence

import requests
from pydantic import BaseModel, ConfigDict, PrivateAttr, ValidationError
from python_a2a import A2AClient, Task

from app.schemas.orchestrator_execution_state import (
    AgentFailureReason,
    ContractIdentifier,
    OrchestratorExecutionState,
    RoutingDecisionId,
    TaskId,
)

from .contracts import (
    A2ATaskMetadata,
    InputArtifactRef,
    RetryContext,
    WorkerExecutionRequest,
)
from .orchestrator_context_projection import contains_unsafe_routing_text
from .orchestrator_discovery import DispatchTarget
from .orchestrator_execution_state import (
    OrchestratorExecutionStateError,
    dispatch_eligible_task_ids,
    mark_task_dispatch_failed,
    mark_task_dispatched,
    mark_task_dispatching,
)
from .orchestrator_task_builder import (
    PreparedA2ATask,
    build_canonical_worker_execution_request,
)


class OrchestratorDispatchError(RuntimeError):
    """Fixed compact failure code; never includes a URL or raw exception."""


class DispatchPostCheckpointError(OrchestratorDispatchError):
    """Post-network checkpoint failure with repr-safe in-memory recovery."""

    __slots__ = ("_recovery_result",)

    def __init__(self, recovery_result: "OrchestratorDispatchResult") -> None:
        super().__init__("dispatch_post_checkpoint_failed")
        self._recovery_result = recovery_result

    @property
    def recovery_result(self) -> "OrchestratorDispatchResult":
        return self._recovery_result

    def __repr__(self) -> str:
        return "DispatchPostCheckpointError('dispatch_post_checkpoint_failed')"

    def __reduce_ex__(self, protocol: int) -> Any:
        raise TypeError("dispatch_post_checkpoint_error_pickle_unsupported")


class DispatchReceipt(BaseModel):
    """Checkpoint-safe transport receipt without response payloads."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    task_id: TaskId
    routing_decision_id: RoutingDecisionId
    agent_id: ContractIdentifier
    capability_id: ContractIdentifier
    dispatch_status: Literal["dispatched", "dispatch_failed"]
    agent_failure_reason: AgentFailureReason


class OrchestratorDispatchResult(BaseModel):
    """Merged compact state plus ephemeral, in-memory A2A responses."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
        frozen=True,
    )

    state: OrchestratorExecutionState
    receipts: tuple[DispatchReceipt, ...]
    _response_tasks: dict[str, Task] = PrivateAttr(default_factory=dict)

    def __init__(self, **data: Any) -> None:
        response_tasks = data.pop("response_tasks", None)
        super().__init__(**data)
        self._response_tasks = dict(response_tasks or {})

    @property
    def response_tasks(self) -> Mapping[str, Task]:
        """Read-only B2 handoff; never part of model iteration/serialization."""
        return MappingProxyType(self._response_tasks)

    def __reduce_ex__(self, protocol: int) -> Any:
        raise TypeError("orchestrator_dispatch_result_pickle_unsupported")


@dataclass(frozen=True)
class _ValidatedDispatch:
    prepared: PreparedA2ATask
    canonical_target: DispatchTarget


@dataclass(frozen=True)
class _TransportOutcome:
    task_id: str
    response_task: Task | None
    failure_reason: AgentFailureReason


async def dispatch_orchestrator_tasks(
    *,
    run_id: str,
    state: OrchestratorExecutionState,
    prepared_tasks: Sequence[PreparedA2ATask],
    discovery: Any,
    execution_graph: Any,
    checkpoint_config: Any,
    timeout_seconds: float,
    client_factory: Callable[..., A2AClient] = A2AClient,
    routing_service: Any | None = None,
) -> OrchestratorDispatchResult:
    """Validate, checkpoint, and concurrently dispatch the complete ready batch."""
    timeout = _validated_timeout(timeout_seconds)
    validated_state, batch = _validate_batch(
        run_id=run_id,
        state=state,
        prepared_tasks=prepared_tasks,
        discovery=discovery,
        routing_service=routing_service,
    )
    if not batch:
        return OrchestratorDispatchResult(
            state=validated_state,
            receipts=(),
            response_tasks={},
        )

    dispatching_state = validated_state
    try:
        for item in batch:
            dispatching_state = mark_task_dispatching(
                dispatching_state, str(item.prepared.task.id)
            )
    except OrchestratorExecutionStateError:
        raise OrchestratorDispatchError("dispatch_state_transition_invalid") from None

    dispatching_state = await _checkpoint_state(
        execution_graph,
        dispatching_state,
        checkpoint_config,
        failure_code="dispatch_pre_checkpoint_failed",
    )

    outcomes = await asyncio.gather(
        *(
            _send_one(
                item,
                timeout_seconds=timeout,
                client_factory=client_factory,
            )
            for item in batch
        ),
        return_exceptions=True,
    )

    merged_state = dispatching_state
    receipts: list[DispatchReceipt] = []
    response_tasks: dict[str, Task] = {}
    for item, raw_outcome in zip(batch, outcomes, strict=True):
        task_id = str(item.prepared.task.id)
        if isinstance(raw_outcome, asyncio.CancelledError):
            raise raw_outcome
        if isinstance(raw_outcome, BaseException):
            outcome = _TransportOutcome(
                task_id=task_id,
                response_task=None,
                failure_reason=_classify_transport_failure(raw_outcome),
            )
        else:
            outcome = raw_outcome

        task_state = merged_state.worker_tasks[task_id]
        if outcome.failure_reason == "none":
            merged_state = mark_task_dispatched(merged_state, task_id)
            dispatch_status = "dispatched"
            if outcome.response_task is not None:
                response_tasks[task_id] = outcome.response_task
        else:
            merged_state = mark_task_dispatch_failed(
                merged_state, task_id, outcome.failure_reason
            )
            dispatch_status = "dispatch_failed"
        receipts.append(
            DispatchReceipt(
                task_id=task_id,
                routing_decision_id=task_state.routing_decision_id,
                agent_id=task_state.agent_id,
                capability_id=task_state.capability_id,
                dispatch_status=dispatch_status,
                agent_failure_reason=outcome.failure_reason,
            )
        )

    recovery_result = OrchestratorDispatchResult(
        state=merged_state,
        receipts=tuple(receipts),
        response_tasks=response_tasks,
    )
    try:
        checkpointed_state = await _checkpoint_state(
            execution_graph,
            merged_state,
            checkpoint_config,
            failure_code="dispatch_post_checkpoint_failed",
        )
    except OrchestratorDispatchError:
        raise DispatchPostCheckpointError(recovery_result) from None
    return OrchestratorDispatchResult(
        state=checkpointed_state,
        receipts=recovery_result.receipts,
        response_tasks=response_tasks,
    )


def _validate_batch(
    *,
    run_id: str,
    state: OrchestratorExecutionState,
    prepared_tasks: Sequence[PreparedA2ATask],
    discovery: Any,
    routing_service: Any | None,
) -> tuple[OrchestratorExecutionState, tuple[_ValidatedDispatch, ...]]:
    try:
        checked = OrchestratorExecutionState.model_validate(state.model_dump())
    except (AttributeError, ValidationError):
        raise OrchestratorDispatchError("dispatch_state_invalid") from None
    if checked.run_id != run_id:
        raise OrchestratorDispatchError("dispatch_run_identity_mismatch")

    try:
        eligible_ids = dispatch_eligible_task_ids(checked)
    except OrchestratorExecutionStateError:
        raise OrchestratorDispatchError("dispatch_state_invalid") from None
    supplied_ids: list[str] = []
    for prepared in prepared_tasks:
        if not isinstance(prepared, PreparedA2ATask):
            raise OrchestratorDispatchError("prepared_task_type_invalid")
        supplied_ids.append(str(prepared.task.id))
    if len(supplied_ids) != len(set(supplied_ids)):
        raise OrchestratorDispatchError("prepared_task_identity_duplicate")
    if set(supplied_ids) != set(eligible_ids):
        raise OrchestratorDispatchError("prepared_task_set_mismatch")
    if not eligible_ids:
        return checked, ()

    prepared_by_id = {
        str(prepared.task.id): prepared for prepared in prepared_tasks
    }
    validated: list[_ValidatedDispatch] = []
    for task_id in eligible_ids:
        prepared = prepared_by_id[task_id]
        task_state = checked.worker_tasks[task_id]
        decision = checked.routing.decisions[task_state.routing_decision_id]
        _validate_prepared_identity(
            run_id=run_id,
            state=checked,
            prepared=prepared,
            task_id=task_id,
            task_state=task_state,
            decision=decision,
            routing_service=routing_service,
        )
        try:
            canonical = discovery.resolve_dispatch_target(
                run_id,
                agent_id=task_state.agent_id,
                capability_id=task_state.capability_id,
                dispatch_mode="python_a2a",
            )
        except Exception:
            raise OrchestratorDispatchError(
                "dispatch_target_validation_failed"
            ) from None
        if canonical != prepared.dispatch_target:
            raise OrchestratorDispatchError("dispatch_target_mismatch")
        validated.append(
            _ValidatedDispatch(prepared=prepared, canonical_target=canonical)
        )
    return checked, tuple(validated)


def _validate_prepared_identity(
    *,
    run_id: str,
    state: OrchestratorExecutionState,
    prepared: PreparedA2ATask,
    task_id: str,
    task_state: Any,
    decision: Any,
    routing_service: Any | None,
) -> None:
    prepared_decision = prepared.decision
    if task_state.retry_attempt > 0:
        if routing_service is None:
            raise OrchestratorDispatchError("retry_authority_required")
        try:
            authoritative = routing_service.rebuild_retry_task(
                run_id=run_id,
                execution_state=state,
                task_id=task_id,
            )
        except Exception:
            raise OrchestratorDispatchError(
                "prepared_retry_authority_invalid"
            ) from None
        if not _prepared_tasks_equal(prepared, authoritative):
            raise OrchestratorDispatchError(
                "prepared_task_payload_contract_mismatch"
            )
    expected = (
        task_id,
        state.routing.routing_plan_id,
        task_state.routing_decision_id,
        task_state.agent_id,
        task_state.capability_id,
    )
    if (
        prepared_decision.task_id,
        task_state.routing_plan_id,
        prepared_decision.routing_decision_id,
        prepared_decision.agent_id,
        prepared_decision.capability_id,
    ) != expected:
        raise OrchestratorDispatchError("prepared_task_identity_mismatch")
    if (
        prepared_decision.validation_status != "ready"
        or prepared_decision.expected_output_artifact_names
        != decision.expected_output_artifact_names
        or decision.routing_decision_id != task_state.routing_decision_id
        or decision.agent_id != task_state.agent_id
        or decision.capability_id != task_state.capability_id
        or task_id not in decision.task_ids
    ):
        raise OrchestratorDispatchError("prepared_task_identity_mismatch")
    try:
        metadata = A2ATaskMetadata.model_validate(prepared.task.metadata)
        request = WorkerExecutionRequest.model_validate_json(
            prepared.task.message["content"]["text"]
        )
    except (KeyError, TypeError, ValidationError):
        raise OrchestratorDispatchError("prepared_task_payload_invalid") from None
    metadata_identity = (
        metadata.run_id,
        metadata.task_id,
        metadata.routing_plan_id,
        metadata.routing_decision_id,
        metadata.agent_id,
        metadata.capability_id,
    )
    request_identity = (
        request.run_id,
        request.task_id,
        request.routing_plan_id,
        request.routing_decision_id,
        request.agent_id,
        request.capability_id,
    )
    expected_payload_identity = (run_id, *expected)
    if metadata_identity != expected_payload_identity or request_identity != expected_payload_identity:
        raise OrchestratorDispatchError("prepared_task_payload_identity_mismatch")
    _validate_prepared_input_artifact_refs(
        run_id=run_id,
        state=state,
        input_artifact_refs=prepared.input_artifact_refs,
    )
    try:
        retry_context = None
        if task_state.retry_attempt > 0:
            previous = _checked_previous_task(state, task_state)
            retry_context = RetryContext(
                retry_of_task_id=previous.task_id,
                retry_attempt=task_state.retry_attempt,
                max_retry_attempts=task_state.max_retry_attempts,
                retry_reason=previous.terminal_error_code,
            )
        canonical_request = build_canonical_worker_execution_request(
            run_id=run_id,
            routing_plan_id=state.routing.routing_plan_id,
            decision=prepared_decision,
            input_artifact_refs=prepared.input_artifact_refs,
            retry_context=retry_context,
        )
    except (TypeError, ValueError, ValidationError):
        raise OrchestratorDispatchError(
            "prepared_task_payload_contract_mismatch"
        ) from None
    _validate_request_contract(request, metadata, canonical_request)


def _prepared_tasks_equal(left: PreparedA2ATask, right: PreparedA2ATask) -> bool:
    try:
        left_request = WorkerExecutionRequest.model_validate_json(
            left.task.message["content"]["text"]
        )
        right_request = WorkerExecutionRequest.model_validate_json(
            right.task.message["content"]["text"]
        )
    except (KeyError, TypeError, ValidationError):
        return False
    return (
        left.decision.model_dump() == right.decision.model_dump()
        and str(left.task.id) == str(right.task.id)
        and left_request.model_dump() == right_request.model_dump()
        and left.task.metadata == right.task.metadata
        and left.dispatch_target == right.dispatch_target
        and {
            key: value.model_dump() for key, value in left.input_artifact_refs.items()
        }
        == {
            key: value.model_dump() for key, value in right.input_artifact_refs.items()
        }
    )


def _checked_previous_task(state: OrchestratorExecutionState, task_state: Any) -> Any:
    if task_state.retry_attempt == 0:
        if task_state.retry_of_task_id is not None:
            raise OrchestratorDispatchError("prepared_task_retry_lineage_invalid")
        return None
    previous = state.worker_tasks.get(task_state.retry_of_task_id)
    if (
        previous is None
        or previous.routing_decision_id != task_state.routing_decision_id
        or previous.agent_id != task_state.agent_id
        or previous.capability_id != task_state.capability_id
        or previous.retry_attempt + 1 != task_state.retry_attempt
        or previous.execution_status != "failed"
        or previous.result_status != "tool_failed"
        or previous.terminal_error_code is None
    ):
        raise OrchestratorDispatchError("prepared_task_retry_lineage_invalid")
    return previous


def _validate_request_contract(
    request: WorkerExecutionRequest,
    metadata: A2ATaskMetadata,
    canonical_request: WorkerExecutionRequest,
) -> None:
    if _contains_unsafe_request_text(request.model_dump(mode="json")):
        raise OrchestratorDispatchError("prepared_task_privacy_invalid")
    if metadata.created_by != canonical_request.created_by or request != canonical_request:
        raise OrchestratorDispatchError("prepared_task_payload_contract_mismatch")


def _validate_prepared_input_artifact_refs(
    *,
    run_id: str,
    state: OrchestratorExecutionState,
    input_artifact_refs: Any,
) -> None:
    if not isinstance(input_artifact_refs, dict):
        raise OrchestratorDispatchError("prepared_task_payload_contract_mismatch")
    for artifact_name, raw_ref in input_artifact_refs.items():
        try:
            ref = InputArtifactRef.model_validate(raw_ref)
        except ValidationError:
            raise OrchestratorDispatchError(
                "prepared_task_payload_contract_mismatch"
            ) from None
        artifact = state.artifacts.get(artifact_name)
        if (
            ref.run_id != run_id
            or artifact_name != ref.artifact_type
            or artifact is None
            or artifact.status != "available"
            or artifact.artifact_id != ref.artifact_id
        ):
            raise OrchestratorDispatchError(
                "prepared_task_payload_contract_mismatch"
            )


def _contains_unsafe_request_text(value: Any) -> bool:
    if isinstance(value, str):
        return contains_unsafe_routing_text(value)
    if isinstance(value, dict):
        return any(_contains_unsafe_request_text(nested) for nested in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_unsafe_request_text(item) for item in value)
    return False


async def _send_one(
    item: _ValidatedDispatch,
    *,
    timeout_seconds: float,
    client_factory: Callable[..., A2AClient],
) -> _TransportOutcome:
    task = item.prepared.task
    task_id = str(task.id)
    try:
        async def _construct_and_send() -> Task:
            # A2AClient construction performs AgentCard HTTP discovery and is
            # synchronous in python-a2a.  Run it off-loop so all task-specific
            # transport setup shares the same outer finite timeout and does not
            # serialize sibling dispatches.
            client = await asyncio.to_thread(
                client_factory,
                item.canonical_target.dispatch_url,
                timeout=max(1, math.ceil(timeout_seconds)),
            )
            return await client.send_task_async(task)

        response = await asyncio.wait_for(
            _construct_and_send(), timeout=timeout_seconds
        )
        if not isinstance(response, Task) or str(response.id) != task_id:
            return _TransportOutcome(
                task_id=task_id,
                response_task=None,
                failure_reason="dispatch_transport_error",
            )
        return _TransportOutcome(
            task_id=task_id,
            response_task=response,
            failure_reason="none",
        )
    except BaseException as exc:  # gather must retain every sibling outcome
        if isinstance(exc, (asyncio.CancelledError, KeyboardInterrupt, SystemExit)):
            raise
        return _TransportOutcome(
            task_id=task_id,
            response_task=None,
            failure_reason=_classify_transport_failure(exc),
        )


def _classify_transport_failure(exc: BaseException) -> AgentFailureReason:
    if isinstance(exc, (asyncio.TimeoutError, TimeoutError, requests.Timeout)):
        return "dispatch_timeout"
    if isinstance(exc, requests.HTTPError):
        status = getattr(getattr(exc, "response", None), "status_code", None)
        if isinstance(status, int) and status >= 500:
            return "server_error"
        return "dispatch_transport_error"
    # requests exceptions inherit OSError, so HTTP status failures must be
    # classified before the socket/connection family.
    if isinstance(exc, (requests.ConnectionError, ConnectionError, OSError)):
        return "dispatch_connection_failed"
    return "dispatch_transport_error"


async def _checkpoint_state(
    execution_graph: Any,
    state: OrchestratorExecutionState,
    checkpoint_config: Any,
    *,
    failure_code: str,
) -> OrchestratorExecutionState:
    try:
        snapshot = await execution_graph.ainvoke(state, config=checkpoint_config)
        return OrchestratorExecutionState.model_validate(snapshot)
    except Exception:
        raise OrchestratorDispatchError(failure_code) from None


def _validated_timeout(value: float) -> float:
    try:
        checked = float(value)
    except (TypeError, ValueError):
        raise OrchestratorDispatchError("dispatch_timeout_config_invalid") from None
    if not math.isfinite(checked) or checked <= 0:
        raise OrchestratorDispatchError("dispatch_timeout_config_invalid")
    return checked


__all__ = [
    "DispatchPostCheckpointError",
    "DispatchReceipt",
    "OrchestratorDispatchError",
    "OrchestratorDispatchResult",
    "dispatch_orchestrator_tasks",
]
