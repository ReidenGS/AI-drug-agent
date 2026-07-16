"""Fail-closed authority and reconstruction tests for Turn F2-B2B2 retries."""

from __future__ import annotations

from dataclasses import replace

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from python_a2a import Message, MessageRole, Task, TextContent

from app.a2a.contracts import (
    InputArtifactRef,
    WorkerArtifactRef,
    WorkerExecutionRequest,
    WorkerExecutionResult,
)
from app.a2a.orchestrator_discovery import DispatchTarget
from app.a2a.orchestrator_dispatch import (
    OrchestratorDispatchError,
    dispatch_orchestrator_tasks,
)
from app.a2a.orchestrator_execution_state import (
    mark_task_dispatched,
    mark_task_dispatching,
    mark_task_result,
)
from app.a2a.orchestrator_post_ingestion import (
    OrchestratorPostIngestionError,
    revalidate_orchestrator_after_ingestion,
)
from app.a2a.orchestrator_result_ingestion import OrchestratorResultIngestionResult
from app.a2a.orchestrator_retry import (
    OrchestratorRetryError,
    prepare_orchestrator_retries,
)
from app.a2a.orchestrator_routing_service import OrchestratorRoutingServiceError
from app.graph.orchestrator_execution_graph import (
    build_orchestrator_execution_graph,
    execution_graph_config,
)
from tests.a2a.test_orchestrator_dispatch import _serve_task_handler
from tests.a2a.test_orchestrator_post_ingestion import (
    ARTIFACTS,
    RUN_ID,
    _complete,
    _contract,
    _environment,
    _persist,
)
from tests.a2a.test_orchestrator_retry_loop import _card


def _pending_retry(storage, registry):
    service, _llm, discovery, initial, state = _environment(
        storage, registry, [_contract("agent_alpha")], ("agent_alpha",)
    )
    ingestion, proof = _complete(
        storage,
        registry,
        state,
        initial.prepared_tasks[0],
        status="tool_failed",
        persist_failure_output=True,
    )
    retry = prepare_orchestrator_retries(
        state=ingestion.state,
        completion_proofs={proof.task_id: proof},
        max_worker_retries=3,
    )
    return service, discovery, initial, ingestion, proof, retry


def _plan_snapshot(storage):
    return storage.read_json(storage.run_key(RUN_ID, "inputs/worker_routing_plan.json"))


def _terminal_retry(storage, registry):
    service, discovery, initial, ingestion, root_proof, retry = _pending_retry(
        storage, registry
    )
    task_id = retry.retry_task_ids[0]
    task = retry.state.worker_tasks[task_id]
    name, path = ARTIFACTS["agent_alpha"]
    artifact_id = _persist(
        storage,
        registry,
        name,
        path,
        {"result_state": "failed", "payload_count": 1},
    )
    state = mark_task_dispatched(
        mark_task_dispatching(retry.state, task_id), task_id
    )
    state = mark_task_result(
        state,
        task_id,
        result_status="tool_failed",
        error_code="synthetic_tool_failed",
        output_artifact_refs={name: artifact_id},
    )
    proof = WorkerExecutionResult(
        payload_type="worker_execution_result",
        payload_version="v1",
        run_id=RUN_ID,
        task_id=task_id,
        routing_plan_id=state.routing.routing_plan_id,
        routing_decision_id=task.routing_decision_id,
        agent_id=task.agent_id,
        capability_id=task.capability_id,
        execution_status="failed",
        result_status="tool_failed",
        error_code="synthetic_tool_failed",
        retry_of_task_id=task.retry_of_task_id,
        output_artifact_refs={
            name: WorkerArtifactRef(
                artifact_id=artifact_id,
                artifact_type=name,
                storage_key=path,
                run_id=RUN_ID,
            )
        },
    )
    return service, discovery, initial, state, root_proof, proof


def test_forged_task_id_cannot_authorize_completion_or_write_plan(
    local_storage, registry_service
):
    service, _discovery, _initial, ingestion, proof, _retry = _pending_retry(
        local_storage, registry_service
    )
    state = ingestion.state
    old_task_id = proof.task_id
    forged_task_id = "task_deadbeefdeadbeef"
    old_task = state.worker_tasks[old_task_id]
    decision_id = old_task.routing_decision_id
    forged_task = old_task.model_copy(update={"task_id": forged_task_id})
    decisions = dict(state.routing.decisions)
    decisions[decision_id] = decisions[decision_id].model_copy(
        update={"task_ids": [forged_task_id]}
    )
    artifacts = {
        name: artifact.model_copy(
            update={"producer_task_id": forged_task_id}
        )
        for name, artifact in state.artifacts.items()
    }
    forged_state = state.model_copy(
        update={
            "worker_tasks": {forged_task_id: forged_task},
            "routing": state.routing.model_copy(update={"decisions": decisions}),
            "artifacts": artifacts,
        }
    )
    forged_proof = proof.model_copy(update={"task_id": forged_task_id})
    plan_before = _plan_snapshot(local_storage)
    registry_before = registry_service.get(RUN_ID).model_dump()
    saver = InMemorySaver()
    with pytest.raises(
        OrchestratorRoutingServiceError, match="^completion_identity_mismatch$"
    ):
        service.revalidate_for_run(
            RUN_ID,
            completed_results=[forged_proof],
            execution_state=forged_state,
        )
    assert _plan_snapshot(local_storage) == plan_before
    assert registry_service.get(RUN_ID).model_dump() == registry_before
    assert len(list(saver.list(None))) == 0


@pytest.mark.parametrize("layer", ["post", "retry", "routing"])
@pytest.mark.asyncio
async def test_bad_retry_parent_proof_is_rejected_at_every_public_boundary(
    local_storage, registry_service, layer
):
    service, _discovery, _initial, state, root_proof, proof = _terminal_retry(
        local_storage, registry_service
    )
    bad = proof.model_copy(update={"retry_of_task_id": "task_deadbeefdeadbeef"})
    plan_before = _plan_snapshot(local_storage)
    saver = InMemorySaver()
    graph = build_orchestrator_execution_graph(checkpointer=saver)
    if layer == "retry":
        with pytest.raises(
            OrchestratorRetryError, match="^worker_retry_proof_identity_mismatch$"
        ):
            prepare_orchestrator_retries(
                state=state,
                completion_proofs={root_proof.task_id: root_proof, bad.task_id: bad},
                max_worker_retries=3,
            )
    elif layer == "routing":
        with pytest.raises(
            OrchestratorRoutingServiceError, match="^completion_identity_mismatch$"
        ):
            service.revalidate_for_run(
                RUN_ID,
                completed_results=[bad],
                execution_state=state,
            )
    else:
        bad_ingestion = OrchestratorResultIngestionResult(
            state=state,
            receipts=(),
            checkpoint_written=False,
            completion_proofs={bad.task_id: bad},
        )
        with pytest.raises(
            OrchestratorPostIngestionError,
            match="^completion_proof_identity_mismatch$",
        ):
            await revalidate_orchestrator_after_ingestion(
                run_id=RUN_ID,
                ingestion_result=bad_ingestion,
                previous_completion_proofs={root_proof.task_id: root_proof},
                routing_service=service,
                execution_graph=graph,
                checkpoint_config=execution_graph_config(RUN_ID),
            )
    assert _plan_snapshot(local_storage) == plan_before
    assert len(list(saver.list(None))) == 0


@pytest.mark.parametrize("tampering", ["objective", "reason", "priority", "input_ref"])
@pytest.mark.asyncio
async def test_safe_retry_payload_tampering_has_zero_http_or_checkpoint_writes(
    local_storage, registry_service, tampering
):
    service, discovery, _initial, _ingestion, _proof, retry = _pending_retry(
        local_storage, registry_service
    )
    handle = _serve_task_handler(
        _card("agent_alpha"),
        lambda task: Task(id=task.id, message=task.message, metadata=task.metadata),
    )
    try:
        discovery.resolve_dispatch_target = lambda run_id, **kwargs: DispatchTarget(
            agent_id=kwargs["agent_id"],
            capability_id=kwargs["capability_id"],
            dispatch_url=handle.url,
            dispatch_mode=kwargs["dispatch_mode"],
        )
        # Rebuild after freezing the localhost target.
        prepared = service.rebuild_retry_task(
            run_id=RUN_ID,
            execution_state=retry.state,
            task_id=retry.retry_task_ids[0],
        )
        request = WorkerExecutionRequest.model_validate_json(
            prepared.task.message["content"]["text"]
        )
        decision = prepared.decision
        refs = dict(prepared.input_artifact_refs)
        if tampering == "objective":
            decision = decision.model_copy(update={"objective": "safe changed objective"})
            request = request.model_copy(
                update={
                    "worker_request": request.worker_request.model_copy(
                        update={"objective": "safe changed objective"}
                    )
                }
            )
        elif tampering == "reason":
            decision = decision.model_copy(
                update={"selection_reason": "safe changed reason"}
            )
            request = request.model_copy(
                update={
                    "worker_request": request.worker_request.model_copy(
                        update={"reason": "safe changed reason"}
                    )
                }
            )
        elif tampering == "priority":
            decision = decision.model_copy(update={"priority": "high"})
            request = request.model_copy(
                update={
                    "worker_request": request.worker_request.model_copy(
                        update={"priority": "high"}
                    )
                }
            )
        else:
            refs["scoring_handoff"] = InputArtifactRef(
                artifact_id="scoring_handoff_aaaaaaaaaaaa",
                run_id=RUN_ID,
                artifact_type="scoring_handoff",
                artifact_role="scoring_handoff",
                can_read_from_db=True,
            )
            request = request.model_copy(
                update={
                    "input_projection": request.input_projection.model_copy(
                        update={"input_artifact_refs": refs}
                    )
                }
            )
        message = Message(
            content=TextContent(text=request.model_dump_json()),
            role=MessageRole.USER,
        )
        tampered = replace(
            prepared,
            decision=decision,
            input_artifact_refs=refs,
            task=Task(
                id=prepared.task.id,
                message=message.to_dict(),
                metadata=prepared.task.metadata,
            ),
        )
        saver = InMemorySaver()
        with pytest.raises(
            OrchestratorDispatchError,
            match="^prepared_task_payload_contract_mismatch$",
        ):
            await dispatch_orchestrator_tasks(
                run_id=RUN_ID,
                state=retry.state,
                prepared_tasks=(tampered,),
                discovery=discovery,
                routing_service=service,
                execution_graph=build_orchestrator_execution_graph(
                    checkpointer=saver
                ),
                checkpoint_config=execution_graph_config(RUN_ID),
                timeout_seconds=1,
            )
        assert handle.hits["card"] == 0
        assert handle.hits["task"] == 0
        assert len(list(saver.list(None))) == 0
    finally:
        handle.close()
