"""Turn F2-B2A generic worker-result ingestion tests.

Synthetic localhost workers exercise real A2A transport. They are test-only
fixtures, not live LLM, MCP, ToolUniverse, or biomedical tool executions.
"""

from __future__ import annotations

import json
import pickle
from collections import Counter
from dataclasses import replace
from types import SimpleNamespace

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import ValidationError
from python_a2a import Message, MessageRole, Task, TaskState, TaskStatus, TextContent

from app.a2a.agent_cards import (
    AdcAgentContract,
    AgentCapabilityContract,
    ArtifactFieldRequirement,
    ContractArtifactRef,
)
from app.a2a.contracts import A2ATaskMetadata, ToolCallSummary, WorkerArtifactRef
from app.a2a.contracts import WorkerExecutionResult
from app.a2a.orchestrator_completion_validation import artifact_id_fingerprint
from app.a2a.orchestrator_dispatch import (
    DispatchReceipt,
    OrchestratorDispatchResult,
    dispatch_orchestrator_tasks,
)
from app.a2a.orchestrator_execution_state import (
    execution_state_from_routing_result,
    mark_task_dispatched,
    mark_task_dispatch_failed,
    mark_task_dispatching,
    mark_task_running,
)
from app.a2a.orchestrator_result_ingestion import (
    OrchestratorResultIngestionError,
    ResultIngestionPostCheckpointError,
    ingest_orchestrator_worker_results,
)
from app.a2a.orchestrator_task_builder import (
    PreparedA2ATask,
    build_canonical_worker_execution_request,
)
from app.graph.orchestrator_execution_graph import (
    build_orchestrator_execution_graph,
    execution_graph_config,
)
from tests.a2a.test_orchestrator_dispatch import (
    _CountingGraph,
    _FrozenTargets,
    _serve_task_handler,
    _synthetic_dispatch_result,
)

RUN_ID = "run_20260714_1234abcd"
PLAN_ID = "wrp_abcdef0123456789"
OUTPUTS = {
    "agent_alpha": ("scoring_handoff", "alpha_output.json"),
    "agent_beta": ("ranking_table", "beta_output.json"),
}
ARTIFACT_IDS = {
    "scoring_handoff": "scoring_handoff_aaaaaaaaaaaa",
    "ranking_table": "ranking_table_bbbbbbbbbbbb",
}


@pytest.fixture(autouse=True)
def _localhost_proxy_isolation(monkeypatch):
    """Test-only localhost transport isolation from host proxy settings."""
    for name in ("NO_PROXY", "no_proxy"):
        monkeypatch.setenv(name, "127.0.0.1,localhost")
    for name in (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "http_proxy",
        "https_proxy",
        "ALL_PROXY",
        "all_proxy",
    ):
        monkeypatch.delenv(name, raising=False)


class _Discovery:
    def __init__(self, contracts):
        self._cache = SimpleNamespace(
            workers={
                contract.agent_id: SimpleNamespace(contract=contract)
                for contract in contracts
            }
        )

    def get_full_card_cache(self, run_id):
        assert run_id == RUN_ID
        return self._cache


class _FailCheckpoint:
    def __init__(self):
        self.calls = 0

    async def ainvoke(self, state, *, config):
        self.calls += 1
        raise RuntimeError("sk-live-PRIVATE-CHECKPOINT-FAILURE")


def _contracts(*, ready_values=("ready",)):
    contracts = []
    for agent_id, (artifact_name, storage_path) in OUTPUTS.items():
        capability_id = agent_id.replace("agent", "capability")
        contracts.append(
            AdcAgentContract(
                agent_id=agent_id,
                agent_role="worker",
                display_name=agent_id,
                description="Synthetic generic result worker.",
                capabilities=[
                    AgentCapabilityContract(
                        capability_id=capability_id,
                        skill_name=capability_id,
                        capability_summary="Persist one generic output.",
                        output_artifacts=[
                            ContractArtifactRef(
                                artifact_name=artifact_name,
                                storage_path=storage_path,
                                readiness_status_field="result_state",
                                ready_status_values=list(ready_values),
                            )
                        ],
                        required_artifact_fields={
                            artifact_name: ArtifactFieldRequirement(
                                required_field_keys=["payload_count"]
                            )
                        },
                        uses_llm=False,
                        uses_mcp=False,
                    )
                ],
                dispatch_modes=["python_a2a"],
                routable=True,
                uses_llm=False,
                uses_mcp=False,
            )
        )
    return contracts


def _routing(targets):
    base = _synthetic_dispatch_result(targets)
    decisions = []
    prepared = []
    for original in base.plan.validated_decisions:
        artifact_name = OUTPUTS[original.agent_id][0]
        decision = original.model_copy(
            update={"expected_output_artifact_names": [artifact_name]}
        )
        decisions.append(decision)
        request = build_canonical_worker_execution_request(
            run_id=RUN_ID,
            routing_plan_id=PLAN_ID,
            decision=decision,
            input_artifact_refs={},
        )
        metadata = A2ATaskMetadata(
            adc_payload_type="worker_execution_request",
            adc_payload_version="v1",
            run_id=RUN_ID,
            task_id=decision.task_id,
            routing_plan_id=PLAN_ID,
            routing_decision_id=decision.routing_decision_id,
            agent_id=decision.agent_id,
            capability_id=decision.capability_id,
            created_by=request.created_by,
        )
        original_prepared = next(
            item
            for item in base.prepared_tasks
            if item.decision.agent_id == decision.agent_id
        )
        prepared.append(
            PreparedA2ATask(
                decision=decision,
                task=Task(
                    id=decision.task_id,
                    message=Message(
                        content=TextContent(text=request.model_dump_json()),
                        role=MessageRole.USER,
                    ).to_dict(),
                    metadata=metadata.model_dump(),
                ),
                dispatch_target=replace(
                    original_prepared.dispatch_target,
                    dispatch_url=targets[decision.agent_id],
                ),
                input_artifact_refs={},
            )
        )
    return replace(
        base,
        plan=base.plan.model_copy(update={"validated_decisions": decisions}),
        prepared_tasks=tuple(prepared),
    )


def _persist_outputs(storage, registry, *, readiness="ready"):
    registry.init_registry(RUN_ID)
    for artifact_name, artifact_id in ARTIFACT_IDS.items():
        path = next(path for name, path in OUTPUTS.values() if name == artifact_name)
        storage.write_json(
            storage.run_key(RUN_ID, path),
            {
                "artifact_id": artifact_id,
                "run_id": RUN_ID,
                "result_state": readiness,
                "payload_count": 1,
            },
        )
    registry.update_active(
        RUN_ID,
        scoring_handoff_id=ARTIFACT_IDS["scoring_handoff"],
        ranking_table_id=ARTIFACT_IDS["ranking_table"],
        worker_routing_plan_output_baselines={
            name: artifact_id_fingerprint(None) for name in ARTIFACT_IDS
        },
    )


def _result(decision, *, status="success", outputs=True, **updates):
    productive = status in {"success", "partial"}
    artifact_name = OUTPUTS[decision.agent_id][0]
    payload = WorkerExecutionResult(
        payload_type="worker_execution_result",
        payload_version="v1",
        run_id=RUN_ID,
        task_id=decision.task_id,
        routing_plan_id=PLAN_ID,
        routing_decision_id=decision.routing_decision_id,
        agent_id=decision.agent_id,
        capability_id=decision.capability_id,
        execution_status="completed" if productive else "failed",
        result_status=status,
        error_code=None if productive else "synthetic_worker_failure",
        output_artifact_refs=(
            {
                artifact_name: WorkerArtifactRef(
                    artifact_id=ARTIFACT_IDS[artifact_name],
                    artifact_type=artifact_name,
                    run_id=RUN_ID,
                )
            }
            if outputs
            else {}
        ),
        compact_summary={"record_count": 1},
        tool_call_summary=ToolCallSummary(attempted=1, success=1),
    )
    return payload.model_copy(update=updates)


def _response(decision, result, *, task_state=None):
    state = task_state or (
        TaskState.COMPLETED
        if result.result_status in {"success", "partial"}
        else TaskState.FAILED
    )
    task = Task(
        id=decision.task_id,
        status=TaskStatus(state=state),
        message=Message(
            content=TextContent(text="result"), role=MessageRole.AGENT
        ).to_dict(),
    )
    task.artifacts = [
        {"parts": [{"type": "text", "text": result.model_dump_json()}]}
    ]
    return task


def _dispatched_context(storage, registry, *, statuses=None, readiness="ready"):
    _persist_outputs(storage, registry, readiness=readiness)
    routing = _routing({"agent_alpha": "http://alpha", "agent_beta": "http://beta"})
    state = execution_state_from_routing_result(routing)
    for task_id in state.worker_tasks:
        state = mark_task_dispatched(mark_task_dispatching(state, task_id), task_id)
    statuses = statuses or {"agent_alpha": "success", "agent_beta": "success"}
    responses = {}
    receipts = []
    for prepared in routing.prepared_tasks:
        result = _result(prepared.decision, status=statuses[prepared.decision.agent_id])
        responses[prepared.decision.task_id] = _response(prepared.decision, result)
        receipts.append(
            DispatchReceipt(
                task_id=prepared.decision.task_id,
                routing_decision_id=prepared.decision.routing_decision_id,
                agent_id=prepared.decision.agent_id,
                capability_id=prepared.decision.capability_id,
                dispatch_status="dispatched",
                agent_failure_reason="none",
            )
        )
    return (
        routing,
        OrchestratorDispatchResult(
            state=state,
            receipts=tuple(receipts),
            response_tasks=responses,
        ),
        _Discovery(_contracts()),
    )


def _graph(run_id=RUN_ID):
    saver = InMemorySaver()
    graph = _CountingGraph(build_orchestrator_execution_graph(checkpointer=saver))
    return graph, saver, execution_graph_config(run_id)


@pytest.mark.asyncio
async def test_two_real_http_synthetic_workers_dispatch_then_ingest_generically(
    local_storage, registry_service
):
    _persist_outputs(local_storage, registry_service)
    routing_seed = _routing({"agent_alpha": "http://alpha", "agent_beta": "http://beta"})
    decisions = {item.agent_id: item for item in routing_seed.plan.validated_decisions}

    def card(agent_id):
        from python_a2a import AgentCard

        return lambda url: AgentCard(
            name=agent_id, description="Synthetic HTTP worker.", url=url
        )

    def handler(agent_id):
        def execute(task):
            decision = decisions[agent_id]
            return _response(decision, _result(decision))

        return execute

    alpha = _serve_task_handler(card("agent_alpha"), handler("agent_alpha"))
    beta = _serve_task_handler(card("agent_beta"), handler("agent_beta"))
    try:
        routing = _routing({"agent_alpha": alpha.url, "agent_beta": beta.url})
        state = execution_state_from_routing_result(routing)
        graph, saver, config = _graph()
        dispatched = await dispatch_orchestrator_tasks(
            run_id=RUN_ID,
            state=state,
            prepared_tasks=routing.prepared_tasks,
            discovery=_FrozenTargets(routing.prepared_tasks),
            execution_graph=graph,
            checkpoint_config=config,
            timeout_seconds=2,
        )
        ingested = await ingest_orchestrator_worker_results(
            run_id=RUN_ID,
            dispatch_result=dispatched,
            discovery=_Discovery(_contracts()),
            registry=registry_service,
            storage=local_storage,
            execution_graph=graph,
            checkpoint_config=config,
        )
        assert alpha.hits["task"] == beta.hits["task"] == 1
        assert len(graph.inputs) == 3  # dispatch pre/post + one ingestion merge
        assert ingested.checkpoint_written is True
        assert len(ingested.completion_proofs) == 2
        assert {item.result_status for item in ingested.receipts} == {"success"}
        assert all(
            task.execution_status == "completed"
            for task in ingested.state.worker_tasks.values()
        )
        assert {
            name: artifact.status for name, artifact in ingested.state.artifacts.items()
        } == {"scoring_handoff": "available", "ranking_table": "available"}
        assert ingested.state.orchestrator.status == "evaluating_results"
        assert ingested.state.next_wakeup.reason == "worker_result_received"
        # One graph invoke per phase; StateGraph persists its internal
        # input/node/output checkpoints for each invoke.
        assert len(list(saver.list(None))) == 9
    finally:
        alpha.close()
        beta.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "task_execution", "decision_status", "artifact_status", "proofs"),
    [
        ("success", "completed", "completed", "available", 2),
        ("partial", "completed", "completed", "available", 2),
        ("tool_failed", "failed", "failed", "invalid", 1),
        ("validation_failed", "failed", "failed", "invalid", 1),
        ("blocked", "failed", "blocked", "invalid", 1),
        ("needs_user_input", "failed", "blocked", "invalid", 1),
    ],
)
async def test_result_status_mapping_is_compact_and_deterministic(
    local_storage,
    registry_service,
    status,
    task_execution,
    decision_status,
    artifact_status,
    proofs,
):
    routing, dispatched, discovery = _dispatched_context(
        local_storage,
        registry_service,
        statuses={"agent_alpha": status, "agent_beta": "success"},
    )
    graph, _, config = _graph()
    result = await ingest_orchestrator_worker_results(
        run_id=RUN_ID,
        dispatch_result=dispatched,
        discovery=discovery,
        registry=registry_service,
        storage=local_storage,
        execution_graph=graph,
        checkpoint_config=config,
    )
    decision = next(item for item in routing.plan.validated_decisions if item.agent_id == "agent_alpha")
    assert result.state.worker_tasks[decision.task_id].execution_status == task_execution
    assert result.state.routing.decisions[decision.routing_decision_id].status == decision_status
    assert result.state.artifacts[OUTPUTS["agent_alpha"][0]].status == artifact_status
    assert len(result.completion_proofs) == proofs
    assert result.state.orchestrator.status == "evaluating_results"


@pytest.mark.asyncio
async def test_malformed_sibling_does_not_cancel_valid_sibling(
    local_storage, registry_service
):
    _, dispatched, discovery = _dispatched_context(local_storage, registry_service)
    responses = dict(dispatched.response_tasks)
    bad_id = next(
        task_id
        for task_id, task in dispatched.state.worker_tasks.items()
        if task.agent_id == "agent_alpha"
    )
    responses[bad_id].artifacts = [
        {"parts": [{"type": "text", "text": "not-json"}]}
    ]
    dispatched = OrchestratorDispatchResult(
        state=dispatched.state,
        receipts=dispatched.receipts,
        response_tasks=responses,
    )
    graph, _, config = _graph()
    result = await ingest_orchestrator_worker_results(
        run_id=RUN_ID,
        dispatch_result=dispatched,
        discovery=discovery,
        registry=registry_service,
        storage=local_storage,
        execution_graph=graph,
        checkpoint_config=config,
    )
    by_task = {item.task_id: item for item in result.receipts}
    assert by_task[bad_id].error_code == "worker_result_json_invalid"
    assert by_task[bad_id].result_status == "tool_failed"
    assert len(result.completion_proofs) == 1
    assert Counter(
        task.execution_status for task in result.state.worker_tasks.values()
    ) == Counter({"failed": 1, "completed": 1})


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("shape", "error_code"),
    [
        ("no_artifact", "worker_result_artifact_shape_invalid"),
        ("extra_artifact", "worker_result_artifact_shape_invalid"),
        ("no_part", "worker_result_part_shape_invalid"),
        ("extra_part", "worker_result_part_shape_invalid"),
        ("duplicate_json_key", "worker_result_json_invalid"),
    ],
)
async def test_strict_official_artifact_shape_rejects_ambiguous_result(
    local_storage, registry_service, shape, error_code
):
    _, dispatched, discovery = _dispatched_context(local_storage, registry_service)
    responses = dict(dispatched.response_tasks)
    task_id = next(iter(responses))
    response = responses[task_id]
    original = response.artifacts[0]
    if shape == "no_artifact":
        response.artifacts = []
    elif shape == "extra_artifact":
        response.artifacts = [original, original]
    elif shape == "no_part":
        response.artifacts = [{"parts": []}]
    elif shape == "extra_part":
        response.artifacts = [
            {"parts": [original["parts"][0], original["parts"][0]]}
        ]
    else:
        text = original["parts"][0]["text"]
        response.artifacts = [
            {
                "parts": [
                    {
                        "type": "text",
                        "text": text[:-1] + ',"warnings":[]}',
                    }
                ]
            }
        ]
    tampered = OrchestratorDispatchResult(
        state=dispatched.state,
        receipts=dispatched.receipts,
        response_tasks=responses,
    )
    graph, _, config = _graph()
    result = await ingest_orchestrator_worker_results(
        run_id=RUN_ID,
        dispatch_result=tampered,
        discovery=discovery,
        registry=registry_service,
        storage=local_storage,
        execution_graph=graph,
        checkpoint_config=config,
    )
    receipt = next(item for item in result.receipts if item.task_id == task_id)
    assert receipt.error_code == error_code
    assert result.state.worker_tasks[task_id].result_status == "tool_failed"
    assert len(result.completion_proofs) == 1


@pytest.mark.asyncio
async def test_running_task_accepts_terminal_result_and_failure_may_omit_outputs(
    local_storage, registry_service
):
    _, dispatched, discovery = _dispatched_context(local_storage, registry_service)
    alpha_id = next(
        task_id
        for task_id, task in dispatched.state.worker_tasks.items()
        if task.agent_id == "agent_alpha"
    )
    running_state = mark_task_running(dispatched.state, alpha_id)
    responses = dict(dispatched.response_tasks)
    alpha_decision = SimpleNamespace(
        task_id=alpha_id,
        routing_decision_id=running_state.worker_tasks[alpha_id].routing_decision_id,
        agent_id="agent_alpha",
        capability_id="capability_alpha",
    )
    responses[alpha_id] = _response(
        alpha_decision,
        _result(alpha_decision, status="tool_failed", outputs=False),
    )
    dispatched = OrchestratorDispatchResult(
        state=running_state,
        receipts=dispatched.receipts,
        response_tasks=responses,
    )
    graph, _, config = _graph()
    result = await ingest_orchestrator_worker_results(
        run_id=RUN_ID,
        dispatch_result=dispatched,
        discovery=discovery,
        registry=registry_service,
        storage=local_storage,
        execution_graph=graph,
        checkpoint_config=config,
    )
    assert result.state.worker_tasks[alpha_id].execution_status == "failed"
    assert result.state.worker_tasks[alpha_id].output_artifact_refs == {}
    assert result.state.artifacts["scoring_handoff"].status == "invalid"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mutation",
    ["missing", "extra", "unknown", "task_id", "receipt_duplicate"],
)
async def test_global_batch_identity_corruption_is_rejected_before_checkpoint(
    local_storage, registry_service, mutation
):
    _, dispatched, discovery = _dispatched_context(local_storage, registry_service)
    responses = dict(dispatched.response_tasks)
    receipts = list(dispatched.receipts)
    first_id = receipts[0].task_id
    if mutation == "missing":
        responses.pop(first_id)
    elif mutation == "extra":
        extra = next(iter(responses.values()))
        responses["task_cccccccccccccccc"] = extra
    elif mutation == "unknown":
        responses["task_cccccccccccccccc"] = Task(
            id="task_cccccccccccccccc", status=TaskStatus(state=TaskState.FAILED)
        )
    elif mutation == "task_id":
        response = responses[first_id]
        response.id = "task_cccccccccccccccc"
    else:
        receipts.append(receipts[0])
    tampered = OrchestratorDispatchResult(
        state=dispatched.state,
        receipts=tuple(receipts),
        response_tasks=responses,
    )
    graph, saver, config = _graph()
    with pytest.raises(OrchestratorResultIngestionError):
        await ingest_orchestrator_worker_results(
            run_id=RUN_ID,
            dispatch_result=tampered,
            discovery=discovery,
            registry=registry_service,
            storage=local_storage,
            execution_graph=graph,
            checkpoint_config=config,
        )
    assert graph.inputs == []
    assert list(saver.list(None)) == []


@pytest.mark.asyncio
async def test_dispatch_failed_receipt_has_no_worker_result_and_cannot_masquerade(
    local_storage, registry_service
):
    _persist_outputs(local_storage, registry_service)
    routing = _routing({"agent_alpha": "http://alpha", "agent_beta": "http://beta"})
    state = execution_state_from_routing_result(routing)
    alpha_id = next(
        task_id
        for task_id, task in state.worker_tasks.items()
        if task.agent_id == "agent_alpha"
    )
    beta_id = next(task_id for task_id in state.worker_tasks if task_id != alpha_id)
    state = mark_task_dispatch_failed(
        mark_task_dispatching(state, alpha_id),
        alpha_id,
        "dispatch_connection_failed",
    )
    state = mark_task_dispatched(mark_task_dispatching(state, beta_id), beta_id)
    decisions = {item.task_id: item for item in routing.plan.validated_decisions}
    receipts = tuple(
        DispatchReceipt(
            task_id=task_id,
            routing_decision_id=task.routing_decision_id,
            agent_id=task.agent_id,
            capability_id=task.capability_id,
            dispatch_status=task.dispatch_status,
            agent_failure_reason=task.agent_failure_reason,
        )
        for task_id, task in state.worker_tasks.items()
    )
    beta_response = _response(decisions[beta_id], _result(decisions[beta_id]))
    dispatch_result = OrchestratorDispatchResult(
        state=state,
        receipts=receipts,
        response_tasks={beta_id: beta_response},
    )
    graph, _, config = _graph()
    ingested = await ingest_orchestrator_worker_results(
        run_id=RUN_ID,
        dispatch_result=dispatch_result,
        discovery=_Discovery(_contracts()),
        registry=registry_service,
        storage=local_storage,
        execution_graph=graph,
        checkpoint_config=config,
    )
    by_task = {item.task_id: item for item in ingested.receipts}
    assert by_task[alpha_id].ingestion_status == "not_received"
    assert by_task[alpha_id].error_code == "dispatch_failed_no_worker_result"
    assert alpha_id not in ingested.completion_proofs
    assert beta_id in ingested.completion_proofs

    masquerade = OrchestratorDispatchResult(
        state=state,
        receipts=receipts,
        response_tasks={alpha_id: beta_response, beta_id: beta_response},
    )
    clean_graph, clean_saver, clean_config = _graph()
    with pytest.raises(
        OrchestratorResultIngestionError,
        match="^worker_response_identity_mismatch$|^dispatch_failed_response_unexpected$",
    ):
        await ingest_orchestrator_worker_results(
            run_id=RUN_ID,
            dispatch_result=masquerade,
            discovery=_Discovery(_contracts()),
            registry=registry_service,
            storage=local_storage,
            execution_graph=clean_graph,
            checkpoint_config=clean_config,
        )
    assert clean_graph.inputs == []
    assert list(clean_saver.list(None)) == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "field",
    [
        "run_id",
        "routing_plan_id",
        "routing_decision_id",
        "task_id",
        "agent_id",
        "capability_id",
        "retry_of_task_id",
    ],
)
async def test_result_identity_mismatch_rejects_entire_batch_without_checkpoint(
    local_storage, registry_service, field
):
    _, dispatched, discovery = _dispatched_context(local_storage, registry_service)
    responses = dict(dispatched.response_tasks)
    task_id = next(iter(responses))
    response = responses[task_id]
    raw = json.loads(response.artifacts[0]["parts"][0]["text"])
    raw[field] = (
        "task_cccccccccccccccc"
        if field in {"task_id", "retry_of_task_id"}
        else "wrong_identity"
    )
    response.artifacts[0]["parts"][0]["text"] = json.dumps(raw)
    tampered = OrchestratorDispatchResult(
        state=dispatched.state,
        receipts=dispatched.receipts,
        response_tasks=responses,
    )
    graph, saver, config = _graph()
    with pytest.raises(
        OrchestratorResultIngestionError,
        match="^worker_result_identity_mismatch$",
    ):
        await ingest_orchestrator_worker_results(
            run_id=RUN_ID,
            dispatch_result=tampered,
            discovery=discovery,
            registry=registry_service,
            storage=local_storage,
            execution_graph=graph,
            checkpoint_config=config,
        )
    assert graph.inputs == []
    assert list(saver.list(None)) == []


@pytest.mark.asyncio
async def test_task_state_result_mismatch_becomes_compact_failed_outcome(
    local_storage, registry_service
):
    _, dispatched, discovery = _dispatched_context(local_storage, registry_service)
    responses = dict(dispatched.response_tasks)
    task_id = next(iter(responses))
    responses[task_id].status = TaskStatus(state=TaskState.FAILED)
    dispatched = OrchestratorDispatchResult(
        state=dispatched.state,
        receipts=dispatched.receipts,
        response_tasks=responses,
    )
    graph, _, config = _graph()
    result = await ingest_orchestrator_worker_results(
        run_id=RUN_ID,
        dispatch_result=dispatched,
        discovery=discovery,
        registry=registry_service,
        storage=local_storage,
        execution_graph=graph,
        checkpoint_config=config,
    )
    receipt = next(item for item in result.receipts if item.task_id == task_id)
    assert receipt.error_code == "worker_task_state_result_mismatch"
    assert result.state.worker_tasks[task_id].execution_status == "failed"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "corruption",
    [
        "missing_output",
        "unexpected_output",
        "registry_id",
        "planning_baseline",
        "storage_missing",
        "persisted_artifact_id",
        "persisted_run_id",
        "required_field",
        "readiness",
    ],
)
async def test_output_contract_and_persisted_corruption_fail_closed_per_task(
    local_storage, registry_service, corruption
):
    _, dispatched, discovery = _dispatched_context(local_storage, registry_service)
    responses = dict(dispatched.response_tasks)
    task_id = next(
        key
        for key, task in dispatched.state.worker_tasks.items()
        if task.agent_id == "agent_alpha"
    )
    response = responses[task_id]
    raw = json.loads(response.artifacts[0]["parts"][0]["text"])
    artifact_name = "scoring_handoff"
    path = OUTPUTS["agent_alpha"][1]
    if corruption == "missing_output":
        raw["output_artifact_refs"] = {}
    elif corruption == "unexpected_output":
        raw["output_artifact_refs"]["extra_output"] = dict(
            raw["output_artifact_refs"][artifact_name]
        )
    elif corruption == "registry_id":
        registry_service.update_active(
            RUN_ID, scoring_handoff_id="scoring_handoff_cccccccccccc"
        )
    elif corruption == "planning_baseline":
        registry_service.update_active(
            RUN_ID,
            worker_routing_plan_output_baselines={
                "scoring_handoff": artifact_id_fingerprint(
                    ARTIFACT_IDS["scoring_handoff"]
                ),
                "ranking_table": artifact_id_fingerprint(None),
            },
        )
    elif corruption == "storage_missing":
        local_storage.delete(local_storage.run_key(RUN_ID, path))
    else:
        body = local_storage.read_json(local_storage.run_key(RUN_ID, path))
        if corruption == "persisted_artifact_id":
            body["artifact_id"] = "scoring_handoff_cccccccccccc"
        elif corruption == "persisted_run_id":
            body["run_id"] = "run_20260714_deadbeef"
        elif corruption == "required_field":
            body.pop("payload_count")
        else:
            body["result_state"] = "failed"
        local_storage.write_json(local_storage.run_key(RUN_ID, path), body)
    response.artifacts[0]["parts"][0]["text"] = json.dumps(raw)
    dispatched = OrchestratorDispatchResult(
        state=dispatched.state,
        receipts=dispatched.receipts,
        response_tasks=responses,
    )
    graph, _, config = _graph()
    result = await ingest_orchestrator_worker_results(
        run_id=RUN_ID,
        dispatch_result=dispatched,
        discovery=discovery,
        registry=registry_service,
        storage=local_storage,
        execution_graph=graph,
        checkpoint_config=config,
    )
    receipt = next(item for item in result.receipts if item.task_id == task_id)
    assert receipt.ingestion_status == "failed"
    assert receipt.error_code.startswith("completion_output_")
    assert result.state.worker_tasks[task_id].result_status == "tool_failed"
    assert result.state.artifacts[artifact_name].status == "invalid"
    assert task_id not in result.completion_proofs


@pytest.mark.asyncio
async def test_non_ready_failure_audit_artifact_is_retained_invalid_without_proof(
    local_storage, registry_service
):
    _, dispatched, discovery = _dispatched_context(
        local_storage,
        registry_service,
        statuses={"agent_alpha": "tool_failed", "agent_beta": "success"},
        readiness="failed",
    )
    # Keep beta productive by restoring only its persisted readiness.
    beta_path = OUTPUTS["agent_beta"][1]
    beta_body = local_storage.read_json(local_storage.run_key(RUN_ID, beta_path))
    beta_body["result_state"] = "ready"
    local_storage.write_json(local_storage.run_key(RUN_ID, beta_path), beta_body)
    graph, _, config = _graph()
    result = await ingest_orchestrator_worker_results(
        run_id=RUN_ID,
        dispatch_result=dispatched,
        discovery=discovery,
        registry=registry_service,
        storage=local_storage,
        execution_graph=graph,
        checkpoint_config=config,
    )
    alpha_id = next(
        task_id
        for task_id, task in result.state.worker_tasks.items()
        if task.agent_id == "agent_alpha"
    )
    assert result.state.artifacts["scoring_handoff"].status == "invalid"
    assert result.state.worker_tasks[alpha_id].output_artifact_refs == {
        "scoring_handoff": ARTIFACT_IDS["scoring_handoff"]
    }
    assert alpha_id not in result.completion_proofs


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("storage_key", "accepted"),
    [
        (None, True),
        ("alpha_output.json", True),
        ("untrusted_output.json", False),
    ],
)
async def test_output_storage_key_must_match_agent_card_when_present(
    local_storage, registry_service, storage_key, accepted
):
    _, dispatched, discovery = _dispatched_context(local_storage, registry_service)
    responses = dict(dispatched.response_tasks)
    task_id = next(
        key
        for key, task in dispatched.state.worker_tasks.items()
        if task.agent_id == "agent_alpha"
    )
    raw = json.loads(responses[task_id].artifacts[0]["parts"][0]["text"])
    raw["output_artifact_refs"]["scoring_handoff"]["storage_key"] = storage_key
    responses[task_id].artifacts[0]["parts"][0]["text"] = json.dumps(raw)
    dispatched = OrchestratorDispatchResult(
        state=dispatched.state,
        receipts=dispatched.receipts,
        response_tasks=responses,
    )
    graph, _, config = _graph()
    result = await ingest_orchestrator_worker_results(
        run_id=RUN_ID,
        dispatch_result=dispatched,
        discovery=discovery,
        registry=registry_service,
        storage=local_storage,
        execution_graph=graph,
        checkpoint_config=config,
    )
    receipt = next(item for item in result.receipts if item.task_id == task_id)
    if accepted:
        assert receipt.result_status == "success"
        assert task_id in result.completion_proofs
        assert result.state.artifacts["scoring_handoff"].status == "available"
    else:
        assert (
            receipt.error_code
            == "completion_output_artifact_storage_key_mismatch"
        )
        assert receipt.result_status == "tool_failed"
        assert task_id not in result.completion_proofs
        assert result.state.artifacts["scoring_handoff"].status == "invalid"


@pytest.mark.asyncio
async def test_completion_proofs_and_receipt_summary_are_defensively_immutable(
    local_storage, registry_service
):
    _, dispatched, discovery = _dispatched_context(local_storage, registry_service)
    graph, _, config = _graph()
    result = await ingest_orchestrator_worker_results(
        run_id=RUN_ID,
        dispatch_result=dispatched,
        discovery=discovery,
        registry=registry_service,
        storage=local_storage,
        execution_graph=graph,
        checkpoint_config=config,
    )
    task_id = next(iter(result.completion_proofs))
    exposed = result.completion_proofs
    proof = exposed[task_id]
    original = proof.model_dump()
    proof.result_status = "tool_failed"
    proof.output_artifact_refs["scoring_handoff"].artifact_id = (
        "scoring_handoff_cccccccccccc"
    )
    proof.warnings.append("caller_mutation")
    proof.tool_call_summary = ToolCallSummary(attempted=99, success=99)
    assert result.completion_proofs[task_id].model_dump() == original
    assert result.completion_proofs[task_id] is not proof
    with pytest.raises(TypeError):
        exposed["task_cccccccccccccccc"] = proof

    summary = result.receipts[0].tool_call_summary
    with pytest.raises(ValidationError):
        summary.success = 99
    assert result.receipts[0].tool_call_summary.success == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mutation", "raw_value"),
    [
        ("success_error", "unexpected_success_error"),
        ("failure_missing_error", "missing"),
        ("invalid_error", "Private Invalid Error Code"),
        ("negative_attempted", "-1"),
        ("negative_success", "-1"),
        ("negative_failed", "-1"),
        ("negative_dependency", "-1"),
        ("negative_skipped", "-1"),
    ],
)
async def test_strict_result_semantics_reject_invalid_error_or_negative_counts(
    local_storage, registry_service, mutation, raw_value
):
    _, dispatched, discovery = _dispatched_context(local_storage, registry_service)
    responses = dict(dispatched.response_tasks)
    task_id = next(iter(responses))
    raw = json.loads(responses[task_id].artifacts[0]["parts"][0]["text"])
    if mutation == "success_error":
        raw["error_code"] = raw_value
    elif mutation == "failure_missing_error":
        raw["result_status"] = "tool_failed"
        raw["execution_status"] = "failed"
        raw["error_code"] = None
    elif mutation == "invalid_error":
        raw["result_status"] = "tool_failed"
        raw["execution_status"] = "failed"
        raw["error_code"] = raw_value
    else:
        field = {
            "negative_attempted": "attempted",
            "negative_success": "success",
            "negative_failed": "failed",
            "negative_dependency": "dependency_unavailable",
            "negative_skipped": "skipped",
        }[mutation]
        raw["tool_call_summary"][field] = -1
    responses[task_id].artifacts[0]["parts"][0]["text"] = json.dumps(raw)
    dispatched = OrchestratorDispatchResult(
        state=dispatched.state,
        receipts=dispatched.receipts,
        response_tasks=responses,
    )
    graph, saver, config = _graph()
    result = await ingest_orchestrator_worker_results(
        run_id=RUN_ID,
        dispatch_result=dispatched,
        discovery=discovery,
        registry=registry_service,
        storage=local_storage,
        execution_graph=graph,
        checkpoint_config=config,
    )
    receipt = next(item for item in result.receipts if item.task_id == task_id)
    assert receipt.error_code == "worker_result_schema_invalid"
    assert receipt.result_status == "tool_failed"
    assert task_id not in result.completion_proofs
    surface = " ".join(
        [repr(result), result.model_dump_json(), repr(list(saver.list(None)))]
    )
    if mutation in {"success_error", "invalid_error"}:
        assert raw_value not in surface


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "sentinel",
    [
        "ACDEFGHIKLMNPQRSTVWYACDEFGHIK",
        ">private_fasta\nACDEFGHIKLMNPQRSTVWY",
        ">private_a3m\nACDEFGHIKLMNPQRSTVWY",
        "ATOM  1 PRIVATE PDB BODY",
        "data_private_mmcif",
        "sk-live-PRIVATE-API-KEY",
        "bearer PRIVATE-AUTHORIZATION",
        "full prompt private body",
        "raw LLM response private body",
        "raw ToolUniverse payload private body",
    ],
)
async def test_result_privacy_sentinel_is_rejected_without_leak(
    local_storage, registry_service, sentinel
):
    _, dispatched, discovery = _dispatched_context(local_storage, registry_service)
    responses = dict(dispatched.response_tasks)
    task_id = next(iter(responses))
    raw = json.loads(responses[task_id].artifacts[0]["parts"][0]["text"])
    raw["warnings"] = [sentinel]
    responses[task_id].artifacts[0]["parts"][0]["text"] = json.dumps(raw)
    dispatched = OrchestratorDispatchResult(
        state=dispatched.state,
        receipts=dispatched.receipts,
        response_tasks=responses,
    )
    graph, saver, config = _graph()
    result = await ingest_orchestrator_worker_results(
        run_id=RUN_ID,
        dispatch_result=dispatched,
        discovery=discovery,
        registry=registry_service,
        storage=local_storage,
        execution_graph=graph,
        checkpoint_config=config,
    )
    receipt = next(item for item in result.receipts if item.task_id == task_id)
    assert receipt.error_code == "worker_result_privacy_invalid"
    surface = " ".join(
        [repr(result), result.model_dump_json(), repr(list(saver.list(None)))]
    )
    assert sentinel not in surface


@pytest.mark.asyncio
async def test_idempotent_replay_writes_no_second_checkpoint(
    local_storage, registry_service
):
    _, dispatched, discovery = _dispatched_context(local_storage, registry_service)
    graph, _, config = _graph()
    first = await ingest_orchestrator_worker_results(
        run_id=RUN_ID,
        dispatch_result=dispatched,
        discovery=discovery,
        registry=registry_service,
        storage=local_storage,
        execution_graph=graph,
        checkpoint_config=config,
    )
    replay = OrchestratorDispatchResult(
        state=first.state,
        receipts=dispatched.receipts,
        response_tasks=dispatched.response_tasks,
    )
    second = await ingest_orchestrator_worker_results(
        run_id=RUN_ID,
        dispatch_result=replay,
        discovery=discovery,
        registry=registry_service,
        storage=local_storage,
        execution_graph=graph,
        checkpoint_config=config,
    )
    assert len(graph.inputs) == 1
    assert second.checkpoint_written is False
    assert second.state == first.state
    assert set(second.completion_proofs) == set(first.completion_proofs)


@pytest.mark.asyncio
async def test_post_checkpoint_failure_preserves_private_recovery_and_rejects_pickle(
    local_storage, registry_service
):
    _, dispatched, discovery = _dispatched_context(local_storage, registry_service)
    saver = InMemorySaver()
    persisted_graph = build_orchestrator_execution_graph(checkpointer=saver)
    config = execution_graph_config(RUN_ID)
    await persisted_graph.ainvoke(dispatched.state, config=config)
    checkpoint_before = list(saver.list(None))
    failing_graph = _FailCheckpoint()
    with pytest.raises(ResultIngestionPostCheckpointError) as caught:
        await ingest_orchestrator_worker_results(
            run_id=RUN_ID,
            dispatch_result=dispatched,
            discovery=discovery,
            registry=registry_service,
            storage=local_storage,
            execution_graph=failing_graph,
            checkpoint_config=config,
        )
    exc = caught.value
    recovery = exc.recovery_result
    assert str(exc) == "result_ingestion_post_checkpoint_failed"
    assert len(recovery.receipts) == 2
    assert len(recovery.completion_proofs) == 2
    assert recovery.checkpoint_written is False
    recovery_task_id = next(iter(recovery.completion_proofs))
    exposed_recovery_proof = recovery.completion_proofs[recovery_task_id]
    canonical_recovery_proof = exposed_recovery_proof.model_dump()
    exposed_recovery_proof.warnings.append("caller_mutation")
    exposed_recovery_proof.result_status = "tool_failed"
    assert (
        recovery.completion_proofs[recovery_task_id].model_dump()
        == canonical_recovery_proof
    )
    assert failing_graph.calls == 1
    assert list(saver.list(None)) == checkpoint_before
    persisted = persisted_graph.get_state(config).values
    assert all(
        task["execution_status"] == "not_started"
        for task in persisted["worker_tasks"].values()
    )
    for surface in (
        repr(recovery),
        recovery.model_dump_json(),
        repr(dict(recovery)),
        repr(list(recovery)),
        repr(exc),
        repr(exc.args),
    ):
        assert "completion_proofs" not in surface
        assert "sk-live" not in surface
    with pytest.raises(
        TypeError, match="^orchestrator_result_ingestion_pickle_unsupported$"
    ):
        pickle.dumps(recovery)
    with pytest.raises(
        TypeError,
        match="^result_ingestion_post_checkpoint_error_pickle_unsupported$",
    ):
        pickle.dumps(exc)


def test_execution_state_schema_rejects_terminal_status_conflicts():
    routing = _routing({"agent_alpha": "http://alpha", "agent_beta": "http://beta"})
    state = execution_state_from_routing_result(routing)
    task_id = next(iter(state.worker_tasks))
    payload = state.model_dump()
    payload["worker_tasks"][task_id].update(
        {"execution_status": "completed", "result_status": "tool_failed"}
    )
    from pydantic import ValidationError
    from app.schemas.orchestrator_execution_state import OrchestratorExecutionState

    with pytest.raises(ValidationError):
        OrchestratorExecutionState.model_validate(payload)
