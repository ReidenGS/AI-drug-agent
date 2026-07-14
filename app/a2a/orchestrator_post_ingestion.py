"""Generic post-ingestion DAG revalidation and compact state reconciliation."""

from __future__ import annotations

import copy
import json
from types import MappingProxyType
from typing import Any, Mapping

from pydantic import BaseModel, ConfigDict, PrivateAttr, ValidationError

from app.schemas.orchestrator_execution_state import OrchestratorExecutionState

from .contracts import WorkerExecutionResult
from .orchestrator_context_projection import contains_unsafe_routing_text
from .orchestrator_execution_state import (
    OrchestratorExecutionStateError,
    reconcile_execution_state_after_revalidation,
)
from .orchestrator_result_ingestion import OrchestratorResultIngestionResult
from .orchestrator_routing_service import (
    OrchestratorRoutingService,
    OrchestratorRoutingServiceError,
)
from .orchestrator_task_builder import PreparedA2ATask


class OrchestratorPostIngestionError(RuntimeError):
    """Fixed compact post-ingestion error without proof or task payloads."""


class OrchestratorPostIngestionResult(BaseModel):
    """Checkpointed state plus private current-process continuation authority."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    state: OrchestratorExecutionState
    checkpoint_written: bool
    _prepared_tasks: tuple[PreparedA2ATask, ...] = PrivateAttr(default=())
    _completion_proofs: dict[str, WorkerExecutionResult] = PrivateAttr(
        default_factory=dict
    )

    def __init__(self, **data: Any) -> None:
        prepared_tasks = data.pop("prepared_tasks", ())
        completion_proofs = data.pop("completion_proofs", {})
        super().__init__(**data)
        self._prepared_tasks = tuple(copy.deepcopy(tuple(prepared_tasks)))
        self._completion_proofs = {
            str(task_id): proof.model_copy(deep=True)
            for task_id, proof in dict(completion_proofs).items()
        }

    @property
    def prepared_tasks(self) -> tuple[PreparedA2ATask, ...]:
        return tuple(copy.deepcopy(self._prepared_tasks))

    @property
    def completion_proofs(self) -> Mapping[str, WorkerExecutionResult]:
        return MappingProxyType(
            {
                task_id: proof.model_copy(deep=True)
                for task_id, proof in self._completion_proofs.items()
            }
        )

    def __reduce_ex__(self, protocol: int) -> Any:
        raise TypeError("orchestrator_post_ingestion_result_pickle_unsupported")


class PostIngestionCheckpointError(OrchestratorPostIngestionError):
    """Checkpoint failure with repr-safe in-process recovery authority."""

    __slots__ = ("_recovery_result",)

    def __init__(self, recovery_result: OrchestratorPostIngestionResult) -> None:
        super().__init__("post_ingestion_checkpoint_failed")
        self._recovery_result = recovery_result

    @property
    def recovery_result(self) -> OrchestratorPostIngestionResult:
        return self._recovery_result

    def __repr__(self) -> str:
        return "PostIngestionCheckpointError('post_ingestion_checkpoint_failed')"

    def __reduce_ex__(self, protocol: int) -> Any:
        raise TypeError("post_ingestion_checkpoint_error_pickle_unsupported")


async def revalidate_orchestrator_after_ingestion(
    *,
    run_id: str,
    ingestion_result: OrchestratorResultIngestionResult,
    previous_completion_proofs: Mapping[str, WorkerExecutionResult],
    routing_service: OrchestratorRoutingService,
    execution_graph: Any,
    checkpoint_config: Any,
) -> OrchestratorPostIngestionResult:
    """Revalidate using cumulative terminal proofs; release productive DAG only."""
    state = _validated_ingestion_state(run_id, ingestion_result)
    cumulative = _merge_completion_proofs(
        state=state,
        previous=previous_completion_proofs,
        current=ingestion_result.completion_proofs,
    )
    latest = _latest_completion_proofs(state, cumulative)
    try:
        routing = routing_service.revalidate_for_run(
            run_id,
            completed_results=[latest[key] for key in sorted(latest)],
            expected_routing_plan_id=state.routing.routing_plan_id,
            execution_state=state,
        )
    except OrchestratorRoutingServiceError as exc:
        raise OrchestratorPostIngestionError(str(exc)) from None
    except Exception:  # noqa: BLE001 - fixed compact orchestration boundary
        raise OrchestratorPostIngestionError("routing_revalidation_failed") from None
    try:
        reconciled, prepared = reconcile_execution_state_after_revalidation(
            state, routing
        )
    except OrchestratorExecutionStateError as exc:
        raise OrchestratorPostIngestionError(str(exc)) from None

    recovery = OrchestratorPostIngestionResult(
        state=reconciled,
        checkpoint_written=False,
        prepared_tasks=prepared,
        completion_proofs=cumulative,
    )
    if reconciled == state:
        return recovery
    try:
        payload = await execution_graph.ainvoke(
            reconciled, config=checkpoint_config
        )
        checkpointed = OrchestratorExecutionState.model_validate(payload)
        if checkpointed != reconciled:
            raise ValueError("checkpoint_state_mismatch")
    except Exception:  # noqa: BLE001 - fixed compact recovery boundary
        raise PostIngestionCheckpointError(recovery) from None
    return OrchestratorPostIngestionResult(
        state=checkpointed,
        checkpoint_written=True,
        prepared_tasks=prepared,
        completion_proofs=cumulative,
    )


def _validated_ingestion_state(
    run_id: str, ingestion_result: OrchestratorResultIngestionResult
) -> OrchestratorExecutionState:
    if not isinstance(ingestion_result, OrchestratorResultIngestionResult):
        raise OrchestratorPostIngestionError("ingestion_result_type_invalid")
    try:
        state = OrchestratorExecutionState.model_validate(
            ingestion_result.state.model_dump()
        )
    except (AttributeError, ValidationError):
        raise OrchestratorPostIngestionError("ingestion_state_invalid") from None
    if state.run_id != run_id:
        raise OrchestratorPostIngestionError("ingestion_run_identity_mismatch")
    return state


def _merge_completion_proofs(
    *,
    state: OrchestratorExecutionState,
    previous: Mapping[str, WorkerExecutionResult],
    current: Mapping[str, WorkerExecutionResult],
) -> dict[str, WorkerExecutionResult]:
    if not isinstance(previous, Mapping) or not isinstance(current, Mapping):
        raise OrchestratorPostIngestionError("completion_proofs_type_invalid")
    merged: dict[str, WorkerExecutionResult] = {}
    canonical: dict[str, bytes] = {}
    # The current ingestion state is the runtime authority.  Validate its proof
    # first so a conflicting historical proof is classified deterministically
    # as a replay conflict rather than as an incidental state mismatch.
    for source in (current, previous):
        for task_id, supplied in source.items():
            if not isinstance(task_id, str) or not isinstance(
                supplied, WorkerExecutionResult
            ):
                raise OrchestratorPostIngestionError("completion_proof_type_invalid")
            try:
                proof = WorkerExecutionResult.model_validate(
                    supplied.model_dump(mode="python", warnings=False),
                    strict=True,
                )
            except (AttributeError, ValidationError):
                raise OrchestratorPostIngestionError(
                    "completion_proof_schema_invalid"
                ) from None
            if task_id != proof.task_id:
                raise OrchestratorPostIngestionError("completion_proof_key_mismatch")
            encoded = json.dumps(
                proof.model_dump(mode="json"),
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            if task_id in canonical and canonical[task_id] != encoded:
                raise OrchestratorPostIngestionError("completion_proof_conflict")
            _validate_proof_against_state(state, proof)
            canonical[task_id] = encoded
            merged[task_id] = proof
    return merged


def _validate_proof_against_state(
    state: OrchestratorExecutionState, proof: WorkerExecutionResult
) -> None:
    if contains_unsafe_routing_text(proof.model_dump(mode="json")):
        raise OrchestratorPostIngestionError("completion_proof_privacy_invalid")
    task = state.worker_tasks.get(proof.task_id)
    if task is None:
        raise OrchestratorPostIngestionError("completion_proof_task_unknown")
    decision = state.routing.decisions.get(task.routing_decision_id)
    if decision is None:
        raise OrchestratorPostIngestionError("completion_proof_identity_mismatch")
    productive = proof.result_status in {"success", "partial"}
    task_productive = task.result_status in {"success", "partial"}
    if (
        proof.run_id != state.run_id
        or proof.routing_plan_id != state.routing.routing_plan_id
        or proof.routing_decision_id != task.routing_decision_id
        or proof.agent_id != task.agent_id
        or proof.capability_id != task.capability_id
        or productive != task_productive
        or proof.execution_status != task.execution_status
        or proof.retry_of_task_id != task.retry_of_task_id
        or task.execution_status not in {"completed", "failed"}
        or task.result_status != proof.result_status
    ):
        raise OrchestratorPostIngestionError("completion_proof_identity_mismatch")
    expected = set(decision.expected_output_artifact_names)
    actual = set(proof.output_artifact_refs)
    if actual - expected or (productive and actual != expected):
        raise OrchestratorPostIngestionError("completion_proof_output_mismatch")
    compact_ids = {
        name: ref.artifact_id for name, ref in proof.output_artifact_refs.items()
    }
    if dict(task.output_artifact_refs) != compact_ids:
        raise OrchestratorPostIngestionError("completion_proof_output_mismatch")
    is_latest_attempt = bool(decision.task_ids) and decision.task_ids[-1] == proof.task_id
    for name, artifact_id in compact_ids.items():
        ref = proof.output_artifact_refs[name]
        if (
            ref.run_id != state.run_id
            or ref.artifact_type != name
        ):
            raise OrchestratorPostIngestionError(
                "completion_proof_output_mismatch"
            )
        if is_latest_attempt:
            artifact = state.artifacts.get(name)
            if (
                artifact is None
                or artifact.status != ("available" if productive else "invalid")
                or artifact.artifact_id != artifact_id
                or artifact.producer_task_id != proof.task_id
            ):
                raise OrchestratorPostIngestionError(
                    "completion_proof_output_mismatch"
                )


def _latest_completion_proofs(
    state: OrchestratorExecutionState,
    cumulative: Mapping[str, WorkerExecutionResult],
) -> dict[str, WorkerExecutionResult]:
    """Select one current terminal attestation per decision for DAG validation."""
    latest: dict[str, WorkerExecutionResult] = {}
    for decision_id, decision in state.routing.decisions.items():
        if not decision.task_ids:
            continue
        task_id = decision.task_ids[-1]
        proof = cumulative.get(task_id)
        task = state.worker_tasks[task_id]
        if task.execution_status in {"completed", "failed"} and proof is None:
            raise OrchestratorPostIngestionError("completion_proof_required")
        if proof is not None:
            latest[decision_id] = proof
    return latest


__all__ = [
    "OrchestratorPostIngestionError",
    "OrchestratorPostIngestionResult",
    "PostIngestionCheckpointError",
    "revalidate_orchestrator_after_ingestion",
]
