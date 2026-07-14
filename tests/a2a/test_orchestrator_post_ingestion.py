"""Turn F2-B2B1 generic post-ingestion DAG reconciliation tests.

All workers, capabilities, and artifacts are synthetic.  The deterministic LLM
and frozen discovery fixtures are test-only: no live LLM, A2A task transport,
worker, MCP, ToolUniverse, or biomedical tool is invoked.
"""

from __future__ import annotations

import copy
import pickle
from types import SimpleNamespace

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from python_a2a import Message, MessageRole, Task, TaskState, TaskStatus, TextContent

from app.a2a.agent_cards import (
    AdcAgentContract,
    AgentCapabilityContract,
    ArtifactFieldRequirement,
    ContractArtifactRef,
)
from app.a2a.contracts import (
    ToolCallSummary,
    WorkerArtifactRef,
    WorkerExecutionResult,
)
from app.a2a.orchestrator_completion_validation import artifact_id_fingerprint
from app.a2a.orchestrator_discovery import DispatchTarget
from app.a2a.orchestrator_dispatch import (
    DispatchReceipt,
    OrchestratorDispatchResult,
)
from app.a2a.orchestrator_execution_state import (
    dispatch_eligible_task_ids,
    execution_state_from_routing_result,
    mark_task_dispatched,
    mark_task_dispatching,
    mark_task_result,
    mark_task_running,
)
from app.a2a.orchestrator_post_ingestion import (
    OrchestratorPostIngestionError,
    PostIngestionCheckpointError,
    revalidate_orchestrator_after_ingestion,
)
from app.a2a.orchestrator_result_ingestion import (
    OrchestratorResultIngestionResult,
    ingest_orchestrator_worker_results,
)
from app.a2a.orchestrator_routing_service import OrchestratorRoutingService
from app.graph.orchestrator_execution_graph import (
    build_orchestrator_execution_graph,
    execution_graph_config,
)
from app.utils.ids import new_artifact_id

RUN_ID = "run_20260714_c0ffee12"
ARTIFACTS = {
    "agent_alpha": ("scoring_handoff", "artifact_alpha.json"),
    "agent_beta": ("ranking_table", "artifact_beta.json"),
    "agent_gamma": ("scientific_evidence_table", "artifact_gamma.json"),
    "agent_delta": ("patent_prior_art_table", "artifact_delta.json"),
}


class _DeterministicLLM:
    """Test-only proposal fixture; never a production-provider fallback."""

    name = "deterministic_test"
    model = "deterministic-test-v1"

    def __init__(self, decisions):
        self.decisions = decisions
        self.call_count = 0

    def generate_json(self, prompt, *, schema, system=None):
        self.call_count += 1
        return {
            "loop_decision": "dispatch_next_workers",
            "decisions": copy.deepcopy(self.decisions),
            "decision_summary": "Validate a synthetic generic dependency DAG.",
        }


class _FrozenDiscovery:
    """Test-only frozen AgentCard authority; no HTTP or business task calls."""

    def __init__(self, contracts):
        self.contracts = contracts
        self.workers = {
            contract.agent_id: SimpleNamespace(
                is_available=True,
                contract=contract,
            )
            for contract in contracts
        }
        self.discover_count = 0

    def discover_for_run(self, run_id):
        self.discover_count += 1
        return SimpleNamespace(
            available_agent_ids=sorted(self.workers),
            unavailable_agent_ids=[],
        )

    def get_compact_card_catalog(self, run_id):
        return [
            {
                "agent_id": contract.agent_id,
                "capabilities": [
                    {"capability_id": item.capability_id}
                    for item in contract.capabilities
                ],
            }
            for contract in self.contracts
        ]

    def get_full_card_cache(self, run_id):
        return SimpleNamespace(workers=self.workers)

    def resolve_dispatch_target(
        self, run_id, *, agent_id, capability_id, dispatch_mode="python_a2a"
    ):
        return DispatchTarget(
            agent_id=agent_id,
            capability_id=capability_id,
            dispatch_url=f"http://{agent_id}.invalid",
            dispatch_mode=dispatch_mode,
        )


class _CountingGraph:
    def __init__(self, graph):
        self.graph = graph
        self.calls = 0

    async def ainvoke(self, state, *, config):
        self.calls += 1
        return await self.graph.ainvoke(state, config=config)


class _FailCheckpoint:
    def __init__(self):
        self.calls = 0

    async def ainvoke(self, state, *, config):
        self.calls += 1
        raise RuntimeError("sk-live-PRIVATE-CHECKPOINT-ERROR")


def _artifact_ref(agent_id):
    name, path = ARTIFACTS[agent_id]
    return ContractArtifactRef(
        artifact_name=name,
        storage_path=path,
        readiness_status_field="result_state",
        ready_status_values=["ready"],
    )


def _contract(agent_id, *, requires=()):
    required = [_artifact_ref(item) for item in requires]
    output = _artifact_ref(agent_id)
    fields = {
        ref.artifact_name: ArtifactFieldRequirement(
            required_field_keys=["payload_count"]
        )
        for ref in [*required, output]
    }
    return AdcAgentContract(
        agent_id=agent_id,
        agent_role="worker",
        display_name=agent_id,
        description="Synthetic generic DAG worker.",
        capabilities=[
            AgentCapabilityContract(
                capability_id=agent_id.replace("agent", "capability"),
                skill_name=f"{agent_id}_skill",
                capability_summary="Produce one declared synthetic artifact.",
                required_input_artifacts=required,
                required_artifact_fields=fields,
                output_artifacts=[output],
                uses_llm=False,
                uses_mcp=False,
            )
        ],
        dispatch_modes=["python_a2a"],
        routable=True,
        uses_llm=False,
        uses_mcp=False,
    )


def _proposal(*agent_ids):
    return [
        {
            "agent_id": agent_id,
            "capability_id": agent_id.replace("agent", "capability"),
            "objective": f"Produce the declared output for {agent_id}.",
            "selection_reason": "The validated synthetic DAG requires it.",
            "priority": "normal",
        }
        for agent_id in agent_ids
    ]


def _seed_inputs(storage, registry):
    registry.init_registry(RUN_ID)
    raw_id = _persist(
        storage,
        registry,
        "raw_request_record",
        "inputs/raw_request_record.json",
        {"raw_user_query": "Run generic DAG.", "uploaded_files": []},
    )
    query_id = _persist(
        storage,
        registry,
        "structured_query",
        "inputs/structured_query.json",
        {
            "task_intent": {"primary_intent": "generic_analysis", "secondary_intents": []},
            "canonical_query": "Run generic DAG.",
            "requested_outputs": ["generic_result"],
            "missing_slots": [],
            "mentioned_entities": {},
            "referenced_inputs": [],
            "normalized_entities": [],
            "entity_decompositions": [],
        },
    )
    _persist(
        storage,
        registry,
        "input_readiness_status",
        "inputs/input_readiness_status.json",
        {
            "checked_at": "2026-07-14T00:00:00Z",
            "source_refs": {
                "raw_request_record_id": raw_id,
                "structured_query_id": query_id,
            },
            "input_readiness_status": "ready",
            "missing_input_checklist": [],
            "blocking_reasons": [],
        },
    )


def _persist(storage, registry, name, path, body):
    artifact_id = new_artifact_id(name)
    storage.write_json(
        storage.run_key(RUN_ID, path),
        {"artifact_id": artifact_id, "run_id": RUN_ID, **body},
    )
    registry.update_active(RUN_ID, **{f"{name}_id": artifact_id})
    return artifact_id


def _environment(storage, registry, contracts, selected):
    _seed_inputs(storage, registry)
    discovery = _FrozenDiscovery(contracts)
    llm = _DeterministicLLM(_proposal(*selected))
    service = OrchestratorRoutingService(
        discovery=discovery,
        storage=storage,
        registry=registry,
        llm=llm,
    )
    initial = service.plan_for_run(RUN_ID)
    state = execution_state_from_routing_result(initial)
    return service, llm, discovery, initial, state


def _prepared_for_agent(prepared_tasks, agent_id):
    return next(item for item in prepared_tasks if item.decision.agent_id == agent_id)


def _complete(
    storage,
    registry,
    state,
    prepared,
    *,
    status="success",
    persist_failure_output=False,
):
    name, path = ARTIFACTS[prepared.decision.agent_id]
    productive = status in {"success", "partial"}
    artifact_id = None
    refs = {}
    if productive or persist_failure_output:
        artifact_id = _persist(
            storage,
            registry,
            name,
            path,
            {
                "result_state": "ready" if productive else "failed",
                "payload_count": 1,
            },
        )
        refs[name] = WorkerArtifactRef(
            artifact_id=artifact_id,
            artifact_type=name,
            storage_key=path,
            run_id=RUN_ID,
        )
    task_id = prepared.decision.task_id
    dispatched = mark_task_dispatched(mark_task_dispatching(state, task_id), task_id)
    completed = mark_task_result(
        dispatched,
        task_id,
        result_status=status,
        output_artifact_refs=(
            {name: artifact_id} if artifact_id is not None else {}
        ),
        available_output_artifact_names=(
            frozenset({name}) if productive and artifact_id is not None else frozenset()
        ),
    )
    proof = WorkerExecutionResult(
        payload_type="worker_execution_result",
        payload_version="v1",
        run_id=RUN_ID,
        task_id=task_id,
        routing_plan_id=completed.routing.routing_plan_id,
        routing_decision_id=prepared.decision.routing_decision_id,
        agent_id=prepared.decision.agent_id,
        capability_id=prepared.decision.capability_id,
        execution_status="completed" if productive else "failed",
        result_status=status,
        error_code=None if productive else "synthetic_worker_failure",
        output_artifact_refs=refs,
    )
    ingestion = OrchestratorResultIngestionResult(
        state=completed,
        receipts=(),
        checkpoint_written=True,
        completion_proofs={task_id: proof},
    )
    return ingestion, proof


@pytest.mark.asyncio
async def test_generic_chain_releases_one_dependency_at_a_time(
    local_storage, registry_service
):
    contracts = [
        _contract("agent_alpha"),
        _contract("agent_beta", requires=("agent_alpha",)),
        _contract("agent_gamma", requires=("agent_beta",)),
    ]
    service, llm, _discovery, initial, state = _environment(
        local_storage,
        registry_service,
        contracts,
        ("agent_alpha", "agent_beta", "agent_gamma"),
    )
    assert dispatch_eligible_task_ids(state) == (
        _prepared_for_agent(initial.prepared_tasks, "agent_alpha").decision.task_id,
    )
    assert {
        item.agent_id: item.validation_status
        for item in initial.plan.validated_decisions
    } == {
        "agent_alpha": "ready",
        "agent_beta": "waiting_for_dependencies",
        "agent_gamma": "waiting_for_dependencies",
    }

    alpha_prepared = _prepared_for_agent(initial.prepared_tasks, "agent_alpha")
    alpha_ingestion, alpha_proof = _complete(
        local_storage, registry_service, state, alpha_prepared
    )
    saver = InMemorySaver()
    graph = _CountingGraph(build_orchestrator_execution_graph(checkpointer=saver))
    after_alpha = await revalidate_orchestrator_after_ingestion(
        run_id=RUN_ID,
        ingestion_result=alpha_ingestion,
        previous_completion_proofs={},
        routing_service=service,
        execution_graph=graph,
        checkpoint_config=execution_graph_config(RUN_ID),
    )
    assert llm.call_count == 1
    assert after_alpha.state.routing.decisions[
        alpha_prepared.decision.routing_decision_id
    ].status == "completed"
    assert {item.decision.agent_id for item in after_alpha.prepared_tasks} == {
        "agent_beta"
    }
    beta_prepared = _prepared_for_agent(after_alpha.prepared_tasks, "agent_beta")
    assert dispatch_eligible_task_ids(after_alpha.state) == (
        beta_prepared.decision.task_id,
    )
    assert next(
        item
        for item in after_alpha.state.routing.decisions.values()
        if item.agent_id == "agent_gamma"
    ).status == "pending_dependency"
    assert after_alpha.state.artifacts[ARTIFACTS["agent_alpha"][0]].status == (
        "available"
    )

    beta_ingestion, beta_proof = _complete(
        local_storage, registry_service, after_alpha.state, beta_prepared
    )
    after_beta = await revalidate_orchestrator_after_ingestion(
        run_id=RUN_ID,
        ingestion_result=beta_ingestion,
        previous_completion_proofs={alpha_proof.task_id: alpha_proof},
        routing_service=service,
        execution_graph=graph,
        checkpoint_config=execution_graph_config(RUN_ID),
    )
    gamma_prepared = _prepared_for_agent(after_beta.prepared_tasks, "agent_gamma")
    assert dispatch_eligible_task_ids(after_beta.state) == (
        gamma_prepared.decision.task_id,
    )
    assert set(after_beta.completion_proofs) == {
        alpha_proof.task_id,
        beta_proof.task_id,
    }
    assert len(after_beta.state.worker_tasks) == 3
    assert graph.calls == 2

    before_replay = len(list(saver.list(None)))
    replay = await revalidate_orchestrator_after_ingestion(
        run_id=RUN_ID,
        ingestion_result=OrchestratorResultIngestionResult(
            state=after_beta.state,
            receipts=(),
            checkpoint_written=True,
            completion_proofs={beta_proof.task_id: beta_proof},
        ),
        previous_completion_proofs={alpha_proof.task_id: alpha_proof},
        routing_service=service,
        execution_graph=graph,
        checkpoint_config=execution_graph_config(RUN_ID),
    )
    assert len(replay.state.worker_tasks) == 3
    assert _prepared_for_agent(replay.prepared_tasks, "agent_gamma").decision.task_id == (
        gamma_prepared.decision.task_id
    )
    assert replay.checkpoint_written is False
    assert len(list(saver.list(None))) == before_replay


@pytest.mark.asyncio
async def test_fanout_and_independent_ready_tasks_remain_concurrent(
    local_storage, registry_service
):
    contracts = [
        _contract("agent_alpha"),
        _contract("agent_beta", requires=("agent_alpha",)),
        _contract("agent_gamma", requires=("agent_alpha",)),
        _contract("agent_delta"),
    ]
    service, _llm, _discovery, initial, state = _environment(
        local_storage,
        registry_service,
        contracts,
        ("agent_alpha", "agent_beta", "agent_gamma", "agent_delta"),
    )
    alpha = _prepared_for_agent(initial.prepared_tasks, "agent_alpha")
    delta = _prepared_for_agent(initial.prepared_tasks, "agent_delta")
    assert set(dispatch_eligible_task_ids(state)) == {
        alpha.decision.task_id,
        delta.decision.task_id,
    }
    state = mark_task_running(
        mark_task_dispatched(
            mark_task_dispatching(state, delta.decision.task_id),
            delta.decision.task_id,
        ),
        delta.decision.task_id,
    )
    ingestion, _proof = _complete(
        local_storage, registry_service, state, alpha
    )
    graph = _CountingGraph(
        build_orchestrator_execution_graph(checkpointer=InMemorySaver())
    )
    result = await revalidate_orchestrator_after_ingestion(
        run_id=RUN_ID,
        ingestion_result=ingestion,
        previous_completion_proofs={},
        routing_service=service,
        execution_graph=graph,
        checkpoint_config=execution_graph_config(RUN_ID),
    )
    assert {item.decision.agent_id for item in result.prepared_tasks} == {
        "agent_beta",
        "agent_gamma",
    }
    assert set(dispatch_eligible_task_ids(result.state)) == {
        item.decision.task_id for item in result.prepared_tasks
    }
    assert result.state.orchestrator.status == "dispatching"
    assert result.state.next_wakeup.reason == "ready_tasks_available"
    assert result.state.worker_tasks[delta.decision.task_id].execution_status == (
        "running"
    )


@pytest.mark.asyncio
async def test_failed_producer_does_not_release_consumer(
    local_storage, registry_service
):
    contracts = [
        _contract("agent_alpha"),
        _contract("agent_beta", requires=("agent_alpha",)),
    ]
    service, _llm, discovery, initial, state = _environment(
        local_storage,
        registry_service,
        contracts,
        ("agent_alpha", "agent_beta"),
    )
    alpha = _prepared_for_agent(initial.prepared_tasks, "agent_alpha")
    active_before = registry_service.get(RUN_ID).active_artifacts
    baseline = active_before.worker_routing_plan_output_baselines[
        ARTIFACTS["agent_alpha"][0]
    ]
    assert active_before.scoring_handoff_id is None
    _manual_ingestion, proof = _complete(
        local_storage,
        registry_service,
        state,
        alpha,
        status="tool_failed",
        persist_failure_output=True,
    )
    dispatched_state = mark_task_dispatched(
        mark_task_dispatching(state, alpha.decision.task_id),
        alpha.decision.task_id,
    )
    response = Task(
        id=alpha.decision.task_id,
        status=TaskStatus(state=TaskState.FAILED),
        message=Message(
            content=TextContent(text="result"), role=MessageRole.AGENT
        ).to_dict(),
    )
    response.artifacts = [
        {"parts": [{"type": "text", "text": proof.model_dump_json()}]}
    ]
    dispatch_result = OrchestratorDispatchResult(
        state=dispatched_state,
        receipts=(
            DispatchReceipt(
                task_id=alpha.decision.task_id,
                routing_decision_id=alpha.decision.routing_decision_id,
                agent_id=alpha.decision.agent_id,
                capability_id=alpha.decision.capability_id,
                dispatch_status="dispatched",
                agent_failure_reason="none",
            ),
        ),
        response_tasks={alpha.decision.task_id: response},
    )
    ingestion = await ingest_orchestrator_worker_results(
        run_id=RUN_ID,
        dispatch_result=dispatch_result,
        discovery=discovery,
        registry=registry_service,
        storage=local_storage,
        execution_graph=_CountingGraph(
            build_orchestrator_execution_graph(checkpointer=InMemorySaver())
        ),
        checkpoint_config=execution_graph_config(RUN_ID),
    )
    active_after = registry_service.get(RUN_ID).active_artifacts
    assert active_after.scoring_handoff_id == (
        proof.output_artifact_refs["scoring_handoff"].artifact_id
    )
    assert artifact_id_fingerprint(active_after.scoring_handoff_id) != baseline
    assert proof.task_id in ingestion.completion_proofs
    graph = _CountingGraph(
        build_orchestrator_execution_graph(checkpointer=InMemorySaver())
    )
    plan_key = local_storage.run_key(RUN_ID, "inputs/worker_routing_plan.json")
    plan_before_missing_proof = local_storage.read_bytes(plan_key)
    with pytest.raises(
        OrchestratorPostIngestionError, match="^completion_proof_required$"
    ):
        await revalidate_orchestrator_after_ingestion(
            run_id=RUN_ID,
            ingestion_result=OrchestratorResultIngestionResult(
                state=ingestion.state,
                receipts=(),
                checkpoint_written=True,
                completion_proofs={},
            ),
            previous_completion_proofs={},
            routing_service=service,
            execution_graph=graph,
            checkpoint_config=execution_graph_config(RUN_ID),
        )
    assert graph.calls == 0
    assert local_storage.read_bytes(plan_key) == plan_before_missing_proof

    result = await revalidate_orchestrator_after_ingestion(
        run_id=RUN_ID,
        ingestion_result=ingestion,
        previous_completion_proofs={},
        routing_service=service,
        execution_graph=graph,
        checkpoint_config=execution_graph_config(RUN_ID),
    )
    assert graph.calls == 1
    assert set(result.completion_proofs) == {proof.task_id}
    assert result.completion_proofs[proof.task_id].result_status == "tool_failed"
    assert result.prepared_tasks == ()
    assert dispatch_eligible_task_ids(result.state) == ()
    assert next(
        item
        for item in result.state.routing.decisions.values()
        if item.agent_id == "agent_beta"
    ).status == "pending_dependency"
    assert next(
        item
        for item in result.state.routing.decisions.values()
        if item.agent_id == "agent_alpha"
    ).status == "failed"
    assert result.state.artifacts["scoring_handoff"].status == "invalid"
    persisted_plan = local_storage.read_json(plan_key)
    assert persisted_plan["routing_status"] == "waiting"
    assert persisted_plan["ready_task_count"] == 0


@pytest.mark.asyncio
async def test_conflicting_proof_fails_before_routing_and_checkpoint(
    local_storage, registry_service
):
    contracts = [_contract("agent_alpha")]
    service, llm, _discovery, initial, state = _environment(
        local_storage, registry_service, contracts, ("agent_alpha",)
    )
    prepared = initial.prepared_tasks[0]
    ingestion, proof = _complete(
        local_storage, registry_service, state, prepared
    )
    conflict = proof.model_copy(update={"result_status": "partial"})
    graph = _FailCheckpoint()
    with pytest.raises(
        OrchestratorPostIngestionError, match="^completion_proof_conflict$"
    ):
        await revalidate_orchestrator_after_ingestion(
            run_id=RUN_ID,
            ingestion_result=ingestion,
            previous_completion_proofs={proof.task_id: conflict},
            routing_service=service,
            execution_graph=graph,
            checkpoint_config=execution_graph_config(RUN_ID),
        )
    assert llm.call_count == 1
    assert graph.calls == 0


@pytest.mark.asyncio
async def test_private_authority_and_post_checkpoint_recovery_are_compact(
    local_storage, registry_service
):
    contracts = [
        _contract("agent_alpha"),
        _contract("agent_beta", requires=("agent_alpha",)),
    ]
    service, _llm, _discovery, initial, state = _environment(
        local_storage,
        registry_service,
        contracts,
        ("agent_alpha", "agent_beta"),
    )
    ingestion, proof = _complete(
        local_storage,
        registry_service,
        state,
        initial.prepared_tasks[0],
    )
    graph = _FailCheckpoint()
    with pytest.raises(PostIngestionCheckpointError) as caught:
        await revalidate_orchestrator_after_ingestion(
            run_id=RUN_ID,
            ingestion_result=ingestion,
            previous_completion_proofs={},
            routing_service=service,
            execution_graph=graph,
            checkpoint_config=execution_graph_config(RUN_ID),
        )
    error = caught.value
    recovery = error.recovery_result
    assert graph.calls == 1
    assert recovery.checkpoint_written is False
    assert len(recovery.prepared_tasks) == 1
    assert set(recovery.completion_proofs) == {proof.task_id}
    dumped = recovery.model_dump_json()
    assert "prepared_tasks" not in dict(recovery)
    assert "completion_proofs" not in dict(recovery)
    assert "prepared_tasks" not in dict(list(recovery))
    assert "completion_proofs" not in dict(list(recovery))
    for forbidden in (
        "prepared_tasks",
        "completion_proofs",
        "WorkerExecutionResult",
        "PreparedA2ATask",
        "python_a2a.Task",
        "artifact_alpha.json",
        "sk-live-PRIVATE-CHECKPOINT-ERROR",
    ):
        assert forbidden not in dumped
        assert forbidden not in repr(recovery)
        assert forbidden not in repr(error)
    with pytest.raises(
        TypeError, match="^orchestrator_post_ingestion_result_pickle_unsupported$"
    ):
        pickle.dumps(recovery)
    with pytest.raises(
        TypeError, match="^post_ingestion_checkpoint_error_pickle_unsupported$"
    ):
        pickle.dumps(error)

    exposed = recovery.completion_proofs[proof.task_id]
    exposed.warnings.append("caller_mutation")
    assert recovery.completion_proofs[proof.task_id].warnings == []
    exposed_task = recovery.prepared_tasks[0]
    exposed_task.task.metadata["caller_mutation"] = True
    assert "caller_mutation" not in recovery.prepared_tasks[0].task.metadata


@pytest.mark.asyncio
async def test_missing_cumulative_proof_fails_closed_and_full_chain_completes(
    local_storage, registry_service
):
    contracts = [
        _contract("agent_alpha"),
        _contract("agent_beta", requires=("agent_alpha",)),
    ]
    service, _llm, _discovery, initial, state = _environment(
        local_storage,
        registry_service,
        contracts,
        ("agent_alpha", "agent_beta"),
    )
    graph = _CountingGraph(
        build_orchestrator_execution_graph(checkpointer=InMemorySaver())
    )
    alpha_ingestion, alpha_proof = _complete(
        local_storage, registry_service, state, initial.prepared_tasks[0]
    )
    after_alpha = await revalidate_orchestrator_after_ingestion(
        run_id=RUN_ID,
        ingestion_result=alpha_ingestion,
        previous_completion_proofs={},
        routing_service=service,
        execution_graph=graph,
        checkpoint_config=execution_graph_config(RUN_ID),
    )
    beta = after_alpha.prepared_tasks[0]
    beta_ingestion, beta_proof = _complete(
        local_storage, registry_service, after_alpha.state, beta
    )
    plan_key = local_storage.run_key(RUN_ID, "inputs/worker_routing_plan.json")
    plan_before = local_storage.read_bytes(plan_key)
    calls_before = graph.calls
    with pytest.raises(
        OrchestratorPostIngestionError, match="^completion_proof_required$"
    ):
        await revalidate_orchestrator_after_ingestion(
            run_id=RUN_ID,
            ingestion_result=beta_ingestion,
            previous_completion_proofs={},
            routing_service=service,
            execution_graph=graph,
            checkpoint_config=execution_graph_config(RUN_ID),
        )
    assert graph.calls == calls_before
    assert local_storage.read_bytes(plan_key) == plan_before

    completed = await revalidate_orchestrator_after_ingestion(
        run_id=RUN_ID,
        ingestion_result=beta_ingestion,
        previous_completion_proofs={alpha_proof.task_id: alpha_proof},
        routing_service=service,
        execution_graph=graph,
        checkpoint_config=execution_graph_config(RUN_ID),
    )
    assert set(completed.completion_proofs) == {
        alpha_proof.task_id,
        beta_proof.task_id,
    }
    assert completed.prepared_tasks == ()
    assert completed.state.run_status == "completed"
    assert completed.state.orchestrator.status == "completed"
    assert completed.state.next_wakeup.reason == "routing_completed"
    assert dispatch_eligible_task_ids(completed.state) == ()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("field", "value", "code"),
    [
        ("run_id", "run_20260714_deadbeef", "completion_proof_identity_mismatch"),
        ("routing_plan_id", "wrp_deadbeefdeadbeef", "completion_proof_identity_mismatch"),
        ("routing_decision_id", "route_deadbeefdeadbeef", "completion_proof_identity_mismatch"),
        ("task_id", "task_deadbeefdeadbeef", "completion_proof_key_mismatch"),
        ("agent_id", "agent_intruder", "completion_proof_identity_mismatch"),
        ("capability_id", "capability_intruder", "completion_proof_identity_mismatch"),
    ],
)
async def test_completion_proof_identity_attacks_have_zero_checkpoint_side_effects(
    local_storage, registry_service, field, value, code
):
    contracts = [_contract("agent_alpha")]
    service, _llm, _discovery, initial, state = _environment(
        local_storage, registry_service, contracts, ("agent_alpha",)
    )
    ingestion, proof = _complete(
        local_storage, registry_service, state, initial.prepared_tasks[0]
    )
    attacked = proof.model_copy(update={field: value})
    attacked_ingestion = OrchestratorResultIngestionResult(
        state=ingestion.state,
        receipts=(),
        checkpoint_written=True,
        completion_proofs={proof.task_id: attacked},
    )
    graph = _FailCheckpoint()
    with pytest.raises(OrchestratorPostIngestionError, match=f"^{code}$") as caught:
        await revalidate_orchestrator_after_ingestion(
            run_id=RUN_ID,
            ingestion_result=attacked_ingestion,
            previous_completion_proofs={},
            routing_service=service,
            execution_graph=graph,
            checkpoint_config=execution_graph_config(RUN_ID),
        )
    assert graph.calls == 0
    assert value not in str(caught.value)
    assert value not in repr(caught.value)


@pytest.mark.asyncio
async def test_unsafe_proof_text_never_reaches_checkpoint_or_exception(
    local_storage, registry_service
):
    sentinel = "sk-live-RAW-PROOF-SECRET"
    contracts = [_contract("agent_alpha")]
    service, _llm, _discovery, initial, state = _environment(
        local_storage, registry_service, contracts, ("agent_alpha",)
    )
    ingestion, proof = _complete(
        local_storage, registry_service, state, initial.prepared_tasks[0]
    )
    unsafe = proof.model_copy(update={"warnings": [sentinel]})
    attacked = OrchestratorResultIngestionResult(
        state=ingestion.state,
        receipts=(),
        checkpoint_written=True,
        completion_proofs={proof.task_id: unsafe},
    )
    saver = InMemorySaver()
    graph = _CountingGraph(build_orchestrator_execution_graph(checkpointer=saver))
    with pytest.raises(
        OrchestratorPostIngestionError,
        match="^completion_proof_privacy_invalid$",
    ) as caught:
        await revalidate_orchestrator_after_ingestion(
            run_id=RUN_ID,
            ingestion_result=attacked,
            previous_completion_proofs={},
            routing_service=service,
            execution_graph=graph,
            checkpoint_config=execution_graph_config(RUN_ID),
        )
    assert graph.calls == 0
    assert len(list(saver.list(None))) == 0
    assert sentinel not in str(caught.value)
    assert sentinel not in repr(caught.value)
    assert sentinel not in repr(list(saver.list(None)))


@pytest.mark.asyncio
async def test_state_plan_identity_mismatch_fails_before_plan_write_or_checkpoint(
    local_storage, registry_service
):
    contracts = [_contract("agent_alpha")]
    service, _llm, _discovery, _initial, state = _environment(
        local_storage, registry_service, contracts, ("agent_alpha",)
    )
    payload = state.model_dump()
    wrong_plan_id = "wrp_deadbeefdeadbeef"
    payload["routing"]["routing_plan_id"] = wrong_plan_id
    for task in payload["worker_tasks"].values():
        task["routing_plan_id"] = wrong_plan_id
    attacked_state = type(state).model_validate(payload)
    ingestion = OrchestratorResultIngestionResult(
        state=attacked_state,
        receipts=(),
        checkpoint_written=True,
        completion_proofs={},
    )
    plan_key = local_storage.run_key(RUN_ID, "inputs/worker_routing_plan.json")
    before = local_storage.read_bytes(plan_key)
    graph = _FailCheckpoint()
    with pytest.raises(
        OrchestratorPostIngestionError,
        match="^worker_routing_plan_identity_mismatch$",
    ):
        await revalidate_orchestrator_after_ingestion(
            run_id=RUN_ID,
            ingestion_result=ingestion,
            previous_completion_proofs={},
            routing_service=service,
            execution_graph=graph,
            checkpoint_config=execution_graph_config(RUN_ID),
        )
    assert local_storage.read_bytes(plan_key) == before
    assert graph.calls == 0


def _schema_bypassed_proof(proof, mutation):
    if mutation == "productive_error_code":
        return proof.model_copy(update={"error_code": "unexpected_error"})
    if mutation == "failure_missing_error_code":
        return proof.model_copy(
            update={
                "execution_status": "failed",
                "result_status": "tool_failed",
                "error_code": None,
            }
        )
    if mutation == "failure_non_snake_error_code":
        return proof.model_copy(
            update={
                "execution_status": "failed",
                "result_status": "tool_failed",
                "error_code": "Not-Snake-sk-live-SECRET",
            }
        )
    if mutation == "productive_failed_execution":
        return proof.model_copy(update={"execution_status": "failed"})
    if mutation == "failure_completed_execution":
        return proof.model_copy(
            update={
                "execution_status": "completed",
                "result_status": "tool_failed",
                "error_code": "synthetic_worker_failure",
            }
        )
    if mutation == "negative_tool_count":
        return proof.model_copy(
            update={
                "tool_call_summary": ToolCallSummary.model_construct(attempted=-1)
            }
        )
    if mutation == "malformed_output_ref":
        name, ref = next(iter(proof.output_artifact_refs.items()))
        malformed = WorkerArtifactRef.model_construct(
            artifact_id=123,
            artifact_type=ref.artifact_type,
            run_id=ref.run_id,
            storage_key=ref.storage_key,
        )
        return proof.model_copy(update={"output_artifact_refs": {name: malformed}})
    raise AssertionError(mutation)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mutation",
    [
        "productive_error_code",
        "failure_missing_error_code",
        "failure_non_snake_error_code",
        "productive_failed_execution",
        "failure_completed_execution",
        "negative_tool_count",
        "malformed_output_ref",
    ],
)
async def test_schema_bypassed_typed_proofs_fail_before_plan_or_checkpoint(
    local_storage, registry_service, mutation
):
    contracts = [_contract("agent_alpha")]
    service, _llm, _discovery, initial, state = _environment(
        local_storage, registry_service, contracts, ("agent_alpha",)
    )
    ingestion, proof = _complete(
        local_storage, registry_service, state, initial.prepared_tasks[0]
    )
    invalid = _schema_bypassed_proof(proof, mutation)
    attacked = OrchestratorResultIngestionResult(
        state=ingestion.state,
        receipts=(),
        checkpoint_written=True,
        completion_proofs={proof.task_id: invalid},
    )
    plan_key = local_storage.run_key(RUN_ID, "inputs/worker_routing_plan.json")
    plan_before = local_storage.read_bytes(plan_key)
    graph = _FailCheckpoint()
    with pytest.raises(
        OrchestratorPostIngestionError,
        match="^completion_proof_schema_invalid$",
    ) as caught:
        await revalidate_orchestrator_after_ingestion(
            run_id=RUN_ID,
            ingestion_result=attacked,
            previous_completion_proofs={},
            routing_service=service,
            execution_graph=graph,
            checkpoint_config=execution_graph_config(RUN_ID),
        )
    assert graph.calls == 0
    assert local_storage.read_bytes(plan_key) == plan_before
    for forbidden in ("sk-live", "Not-Snake", mutation):
        assert forbidden not in str(caught.value)
        assert forbidden not in repr(caught.value)
