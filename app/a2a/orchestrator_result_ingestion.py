"""Generic ingestion of strict A2A worker results into compact graph state."""

from __future__ import annotations

import json
import re
from types import MappingProxyType
from typing import Any, Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, ValidationError
from python_a2a import Task, TaskState

from app.schemas.orchestrator_execution_state import (
    ContractIdentifier,
    OrchestratorExecutionState,
    RoutingDecisionId,
    TaskId,
)
from app.services.artifact_registry_service import ArtifactRegistryService
from app.services.storage_service import Storage

from .contracts import ToolCallSummary, WorkerExecutionResult
from .orchestrator_completion_validation import (
    CompletionArtifactValidationError,
    validate_worker_output_artifacts,
)
from .orchestrator_context_projection import contains_unsafe_routing_text
from .orchestrator_dispatch import OrchestratorDispatchResult
from .orchestrator_execution_state import (
    OrchestratorExecutionStateError,
    mark_task_result,
)

_COMPACT_CODE = re.compile(r"^[a-z][a-z0-9_]{0,127}$")
_PRODUCTIVE_STATUSES = frozenset({"success", "partial"})
_TERMINAL_FAILURE_STATUSES = frozenset(
    {"validation_failed", "tool_failed", "blocked", "needs_user_input"}
)


class OrchestratorResultIngestionError(RuntimeError):
    """Fixed compact failure code; never contains a worker payload."""


class ResultIngestionReceipt(BaseModel):
    """Checkpoint-safe audit of one response ingestion outcome."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    task_id: TaskId
    routing_decision_id: RoutingDecisionId
    agent_id: ContractIdentifier
    capability_id: ContractIdentifier
    ingestion_status: Literal["completed", "failed", "not_received"]
    execution_status: Literal["completed", "failed"] | None = None
    result_status: Literal[
        "success",
        "partial",
        "validation_failed",
        "tool_failed",
        "blocked",
        "needs_user_input",
    ] | None = None
    error_code: str | None = Field(default=None, pattern=r"^[a-z][a-z0-9_]{0,127}$")
    tool_call_summary: ToolCallSummary = Field(default_factory=ToolCallSummary)
    skipped_or_failed_tool_count: int = Field(default=0, ge=0)
    warning_count: int = Field(default=0, ge=0)


class OrchestratorResultIngestionResult(BaseModel):
    """Compact state plus private terminal worker completion proofs.

    Proofs include productive successes and validated terminal failures.  A
    failure proof may attest that a worker advanced an audit-artifact pointer,
    but only productive proofs may satisfy a downstream DAG dependency.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    state: OrchestratorExecutionState
    receipts: tuple[ResultIngestionReceipt, ...]
    checkpoint_written: bool
    _completion_proofs: dict[str, WorkerExecutionResult] = PrivateAttr(
        default_factory=dict
    )

    def __init__(self, **data: Any) -> None:
        proofs = data.pop("completion_proofs", None)
        super().__init__(**data)
        self._completion_proofs = {
            str(task_id): proof.model_copy(deep=True)
            for task_id, proof in dict(proofs or {}).items()
        }

    @property
    def completion_proofs(self) -> Mapping[str, WorkerExecutionResult]:
        return MappingProxyType(
            {
                task_id: proof.model_copy(deep=True)
                for task_id, proof in self._completion_proofs.items()
            }
        )

    def __reduce_ex__(self, protocol: int) -> Any:
        raise TypeError("orchestrator_result_ingestion_pickle_unsupported")


class ResultIngestionPostCheckpointError(OrchestratorResultIngestionError):
    """Checkpoint failure with repr-safe, non-serializable recovery state."""

    __slots__ = ("_recovery_result",)

    def __init__(self, recovery_result: OrchestratorResultIngestionResult) -> None:
        super().__init__("result_ingestion_post_checkpoint_failed")
        self._recovery_result = recovery_result

    @property
    def recovery_result(self) -> OrchestratorResultIngestionResult:
        return self._recovery_result

    def __repr__(self) -> str:
        return (
            "ResultIngestionPostCheckpointError("
            "'result_ingestion_post_checkpoint_failed')"
        )

    def __reduce_ex__(self, protocol: int) -> Any:
        raise TypeError("result_ingestion_post_checkpoint_error_pickle_unsupported")


class _DuplicateJsonKey(ValueError):
    pass


def validate_reconciled_worker_response(
    *,
    run_id: str,
    state: OrchestratorExecutionState,
    task_id: str,
    response: Task,
    discovery: Any,
    registry: ArtifactRegistryService,
    storage: Storage,
) -> WorkerExecutionResult:
    """Preflight a get_task terminal result before any checkpoint mutation."""
    try:
        checked = OrchestratorExecutionState.model_validate(state.model_dump())
        active = registry.get(run_id).active_artifacts
        result = _parse_worker_result(response)
        _validate_result_identity(
            run_id=run_id, state=checked, task_id=task_id, result=result
        )
        task = checked.worker_tasks[task_id]
        decision = checked.routing.decisions[task.routing_decision_id]
        productive = result.result_status in _PRODUCTIVE_STATUSES
        validate_worker_output_artifacts(
            run_id=run_id,
            agent_id=task.agent_id,
            capability_id=task.capability_id,
            expected_output_artifact_names=set(
                decision.expected_output_artifact_names
            ),
            output_artifact_refs=result.output_artifact_refs,
            productive=productive,
            discovery=discovery,
            registry=registry,
            storage=storage,
            active=active,
        )
    except (
        KeyError,
        ValidationError,
        CompletionArtifactValidationError,
        OrchestratorResultIngestionError,
    ) as exc:
        code = str(exc)
        if not _COMPACT_CODE.fullmatch(code):
            code = "worker_reconciliation_result_invalid"
        raise OrchestratorResultIngestionError(code) from None
    return result


async def ingest_orchestrator_worker_results(
    *,
    run_id: str,
    dispatch_result: OrchestratorDispatchResult,
    discovery: Any,
    registry: ArtifactRegistryService,
    storage: Storage,
    execution_graph: Any,
    checkpoint_config: Any,
) -> OrchestratorResultIngestionResult:
    """Validate a complete transport batch and checkpoint one compact merge."""
    state, response_by_task, receipt_by_task = _validate_batch_identity(
        run_id=run_id,
        dispatch_result=dispatch_result,
    )
    try:
        active = registry.get(run_id).active_artifacts
    except Exception:  # noqa: BLE001 - fixed compact batch error
        raise OrchestratorResultIngestionError("result_registry_unavailable") from None

    merged = state
    receipts: list[ResultIngestionReceipt] = []
    proofs: dict[str, WorkerExecutionResult] = {}
    changed = False
    for task_id in sorted(receipt_by_task):
        dispatch_receipt = receipt_by_task[task_id]
        task_state = state.worker_tasks[task_id]
        if dispatch_receipt.dispatch_status == "dispatch_failed":
            receipts.append(
                ResultIngestionReceipt(
                    task_id=task_id,
                    routing_decision_id=task_state.routing_decision_id,
                    agent_id=task_state.agent_id,
                    capability_id=task_state.capability_id,
                    ingestion_status="not_received",
                    error_code="dispatch_failed_no_worker_result",
                )
            )
            continue

        response = response_by_task[task_id]
        try:
            result = _parse_worker_result(response)
            _validate_result_identity(
                run_id=run_id,
                state=state,
                task_id=task_id,
                result=result,
            )
        except OrchestratorResultIngestionError as exc:
            if str(exc) == "worker_result_identity_mismatch":
                raise
            merged = _apply_ingestion_failure(
                merged, task_id, error_code=str(exc)
            )
            receipts.append(
                _failure_receipt(
                    task_state=task_state,
                    error_code=str(exc),
                )
            )
            changed = changed or merged != state
            continue

        productive = result.result_status in _PRODUCTIVE_STATUSES
        try:
            validated_outputs = validate_worker_output_artifacts(
                run_id=run_id,
                agent_id=task_state.agent_id,
                capability_id=task_state.capability_id,
                expected_output_artifact_names=set(
                    state.routing.decisions[
                        task_state.routing_decision_id
                    ].expected_output_artifact_names
                ),
                output_artifact_refs=result.output_artifact_refs,
                productive=productive,
                discovery=discovery,
                registry=registry,
                storage=storage,
                active=active,
            )
        except CompletionArtifactValidationError as exc:
            code = str(exc)
            merged = _apply_ingestion_failure(merged, task_id, error_code=code)
            receipts.append(
                _failure_receipt(
                    task_state=task_state,
                    error_code=code,
                    result=result,
                )
            )
            changed = changed or merged != state
            continue

        output_ids = {
            name: output.artifact_id
            for name, output in validated_outputs.items()
        }
        available = frozenset(
            name
            for name, output in validated_outputs.items()
            if productive and output.ready
        )
        try:
            updated = mark_task_result(
                merged,
                task_id,
                result_status=result.result_status,
                error_code=result.error_code,
                output_artifact_refs=output_ids,
                available_output_artifact_names=available,
            )
        except OrchestratorExecutionStateError:
            raise OrchestratorResultIngestionError(
                "worker_result_state_transition_invalid"
            ) from None
        changed = changed or updated != merged
        merged = updated
        receipts.append(
            ResultIngestionReceipt(
                task_id=task_id,
                routing_decision_id=task_state.routing_decision_id,
                agent_id=task_state.agent_id,
                capability_id=task_state.capability_id,
                ingestion_status="completed" if productive else "failed",
                execution_status=result.execution_status,
                result_status=result.result_status,
                error_code=result.error_code,
                tool_call_summary=result.tool_call_summary,
                skipped_or_failed_tool_count=len(result.skipped_or_failed_tools),
                warning_count=len(result.warnings),
            )
        )
        proofs[task_id] = result

    recovery = OrchestratorResultIngestionResult(
        state=merged,
        receipts=tuple(receipts),
        checkpoint_written=False,
        completion_proofs=proofs,
    )
    if not changed:
        return recovery
    try:
        checkpointed = await _checkpoint_state(
            execution_graph, merged, checkpoint_config
        )
    except OrchestratorResultIngestionError:
        raise ResultIngestionPostCheckpointError(recovery) from None
    return OrchestratorResultIngestionResult(
        state=checkpointed,
        receipts=recovery.receipts,
        checkpoint_written=True,
        completion_proofs=proofs,
    )


def _validate_batch_identity(
    *, run_id: str, dispatch_result: OrchestratorDispatchResult
) -> tuple[OrchestratorExecutionState, dict[str, Task], dict[str, Any]]:
    if not isinstance(dispatch_result, OrchestratorDispatchResult):
        raise OrchestratorResultIngestionError("dispatch_result_type_invalid")
    try:
        state = OrchestratorExecutionState.model_validate(
            dispatch_result.state.model_dump()
        )
    except (AttributeError, ValidationError):
        raise OrchestratorResultIngestionError("dispatch_result_state_invalid") from None
    if state.run_id != run_id:
        raise OrchestratorResultIngestionError("dispatch_result_run_mismatch")

    receipt_by_task: dict[str, Any] = {}
    for receipt in dispatch_result.receipts:
        if receipt.task_id in receipt_by_task:
            raise OrchestratorResultIngestionError("dispatch_receipt_duplicate")
        task = state.worker_tasks.get(receipt.task_id)
        if task is None:
            raise OrchestratorResultIngestionError("dispatch_receipt_task_unknown")
        if (
            receipt.routing_decision_id != task.routing_decision_id
            or receipt.agent_id != task.agent_id
            or receipt.capability_id != task.capability_id
            or receipt.dispatch_status != task.dispatch_status
            or receipt.agent_failure_reason != task.agent_failure_reason
        ):
            raise OrchestratorResultIngestionError(
                "dispatch_receipt_identity_mismatch"
            )
        receipt_by_task[receipt.task_id] = receipt

    response_by_task = dict(dispatch_result.response_tasks)
    if set(response_by_task) - set(receipt_by_task):
        raise OrchestratorResultIngestionError("worker_response_task_unknown")
    for task_id, response in response_by_task.items():
        if not isinstance(response, Task) or str(response.id) != task_id:
            raise OrchestratorResultIngestionError("worker_response_identity_mismatch")
    for task_id, receipt in receipt_by_task.items():
        has_response = task_id in response_by_task
        if receipt.dispatch_status == "dispatched" and not has_response:
            raise OrchestratorResultIngestionError("worker_response_missing")
        if receipt.dispatch_status == "dispatch_failed" and has_response:
            raise OrchestratorResultIngestionError(
                "dispatch_failed_response_unexpected"
            )
    return state, response_by_task, receipt_by_task


def _parse_worker_result(task: Task) -> WorkerExecutionResult:
    artifacts = task.artifacts
    if not isinstance(artifacts, list) or len(artifacts) != 1:
        raise OrchestratorResultIngestionError("worker_result_artifact_shape_invalid")
    artifact = artifacts[0]
    if not isinstance(artifact, dict):
        raise OrchestratorResultIngestionError("worker_result_artifact_shape_invalid")
    parts = artifact.get("parts")
    if not isinstance(parts, list) or len(parts) != 1:
        raise OrchestratorResultIngestionError("worker_result_part_shape_invalid")
    part = parts[0]
    if (
        not isinstance(part, dict)
        or part.get("type") != "text"
        or not isinstance(part.get("text"), str)
    ):
        raise OrchestratorResultIngestionError("worker_result_part_shape_invalid")
    try:
        raw = json.loads(part["text"], object_pairs_hook=_unique_json_object)
    except (json.JSONDecodeError, TypeError, _DuplicateJsonKey):
        raise OrchestratorResultIngestionError("worker_result_json_invalid") from None
    if not isinstance(raw, dict):
        raise OrchestratorResultIngestionError("worker_result_json_not_object")
    try:
        result = WorkerExecutionResult.model_validate(raw, strict=True)
    except ValidationError:
        raise OrchestratorResultIngestionError("worker_result_schema_invalid") from None
    if _contains_unsafe_value(result.model_dump(mode="json")):
        raise OrchestratorResultIngestionError("worker_result_privacy_invalid")

    task_state = getattr(getattr(task, "status", None), "state", None)
    if task_state == TaskState.COMPLETED:
        valid = (
            result.execution_status == "completed"
            and result.result_status in _PRODUCTIVE_STATUSES
        )
    elif task_state == TaskState.FAILED:
        valid = (
            result.execution_status == "failed"
            and result.result_status in _TERMINAL_FAILURE_STATUSES
        )
    else:
        valid = False
    if not valid:
        raise OrchestratorResultIngestionError("worker_task_state_result_mismatch")
    return result


def _validate_result_identity(
    *,
    run_id: str,
    state: OrchestratorExecutionState,
    task_id: str,
    result: WorkerExecutionResult,
) -> None:
    task = state.worker_tasks[task_id]
    if (
        result.run_id != run_id
        or result.routing_plan_id != state.routing.routing_plan_id
        or result.routing_decision_id != task.routing_decision_id
        or result.task_id != task_id
        or result.agent_id != task.agent_id
        or result.capability_id != task.capability_id
        or result.retry_of_task_id != task.retry_of_task_id
    ):
        raise OrchestratorResultIngestionError("worker_result_identity_mismatch")


def _apply_ingestion_failure(
    state: OrchestratorExecutionState, task_id: str, *, error_code: str
) -> OrchestratorExecutionState:
    if not _COMPACT_CODE.fullmatch(error_code):
        error_code = "worker_result_invalid"
    try:
        return mark_task_result(
            state,
            task_id,
            result_status="tool_failed",
            error_code=error_code,
            output_artifact_refs={},
        )
    except OrchestratorExecutionStateError:
        raise OrchestratorResultIngestionError(
            "worker_result_state_transition_invalid"
        ) from None


def _failure_receipt(
    *, task_state: Any, error_code: str, result: WorkerExecutionResult | None = None
) -> ResultIngestionReceipt:
    return ResultIngestionReceipt(
        task_id=task_state.task_id,
        routing_decision_id=task_state.routing_decision_id,
        agent_id=task_state.agent_id,
        capability_id=task_state.capability_id,
        ingestion_status="failed",
        execution_status="failed",
        result_status="tool_failed",
        error_code=(
            error_code
            if _COMPACT_CODE.fullmatch(error_code)
            else "worker_result_invalid"
        ),
        tool_call_summary=(
            result.tool_call_summary if result is not None else ToolCallSummary()
        ),
        skipped_or_failed_tool_count=(
            len(result.skipped_or_failed_tools) if result is not None else 0
        ),
        warning_count=len(result.warnings) if result is not None else 0,
    )


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateJsonKey
        result[key] = value
    return result


def _contains_unsafe_value(value: Any) -> bool:
    if isinstance(value, str):
        return contains_unsafe_routing_text(value)
    if isinstance(value, dict):
        return any(
            _contains_unsafe_value(key) or _contains_unsafe_value(nested)
            for key, nested in value.items()
        )
    if isinstance(value, (list, tuple)):
        return any(_contains_unsafe_value(item) for item in value)
    return False


async def _checkpoint_state(
    execution_graph: Any,
    state: OrchestratorExecutionState,
    checkpoint_config: Any,
) -> OrchestratorExecutionState:
    try:
        payload = await execution_graph.ainvoke(state, config=checkpoint_config)
        return OrchestratorExecutionState.model_validate(payload)
    except Exception:  # noqa: BLE001 - fixed compact recovery boundary
        raise OrchestratorResultIngestionError(
            "result_ingestion_post_checkpoint_failed"
        ) from None


__all__ = [
    "OrchestratorResultIngestionError",
    "OrchestratorResultIngestionResult",
    "ResultIngestionPostCheckpointError",
    "ResultIngestionReceipt",
    "ingest_orchestrator_worker_results",
    "validate_reconciled_worker_response",
]
