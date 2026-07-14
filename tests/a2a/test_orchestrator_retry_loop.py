"""Turn F2-B2B2 generic deterministic retry lifecycle integration tests.

The localhost A2AServer instances are test-only transport fixtures.  They
exercise the real A2AClient HTTP path and strict persisted-artifact contract;
they are not live LLM, MCP, ToolUniverse, or biomedical tool executions and do
not install a production mock-success fallback.
"""

from __future__ import annotations

import threading
import time
import pickle
from collections import Counter
from dataclasses import replace
from types import SimpleNamespace

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from python_a2a import AgentCard, Task, TaskState, TaskStatus

from app.a2a.contracts import (
    WorkerArtifactRef,
    WorkerExecutionRequest,
    WorkerExecutionResult,
)
from app.a2a.orchestrator_discovery import DispatchTarget
from app.a2a.orchestrator_execution_loop import (
    ExecutionLoopCheckpointError,
    OrchestratorExecutionLoopError,
    execute_orchestrator_worker_loop,
)
from app.a2a.orchestrator_execution_state import dispatch_eligible_task_ids
from app.a2a.orchestrator_result_ingestion import (
    ResultIngestionPostCheckpointError,
)
from app.a2a.orchestrator_routing_service import OrchestratorRoutingService
from app.schemas.orchestrator_execution_state import OrchestratorExecutionState
from app.graph.orchestrator_execution_graph import (
    build_orchestrator_execution_graph,
    execution_graph_config,
)
from tests.a2a.test_orchestrator_dispatch import _free_port, _serve_task_handler
from tests.a2a.test_orchestrator_post_ingestion import (
    ARTIFACTS,
    RUN_ID,
    _contract,
    _DeterministicLLM,
    _environment,
    _FrozenDiscovery,
    _persist,
    _proposal,
)

_PERSIST_LOCK = threading.Lock()


@pytest.fixture(autouse=True)
def _localhost_proxy_isolation(monkeypatch):
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


class _SyntheticWorker:
    def __init__(self, *, agent_id, storage, registry, fail_count=0, status="tool_failed"):
        self.agent_id = agent_id
        self.storage = storage
        self.registry = registry
        self.fail_count = fail_count
        self.failure_status = status
        self.requests: list[WorkerExecutionRequest] = []
        self.windows: list[tuple[float, float]] = []
        self.barriers: dict[int, threading.Barrier] = {}

    def handle(self, task: Task) -> Task:
        started = time.monotonic()
        request = WorkerExecutionRequest.model_validate_json(
            task.message["content"]["text"]
        )
        self.requests.append(request)
        attempt = len(self.requests) - 1
        barrier = self.barriers.get(attempt)
        if barrier is not None:
            barrier.wait(timeout=2)
            time.sleep(0.04)
        productive = len(self.requests) > self.fail_count
        status = "success" if productive else self.failure_status
        name, path = ARTIFACTS[self.agent_id]
        # The production registry service is called serially by each worker
        # process. This in-process multi-server fixture shares one service
        # object, so serialize only that fixture boundary while retaining
        # overlapping HTTP handler windows.
        with _PERSIST_LOCK:
            artifact_id = _persist(
                self.storage,
                self.registry,
                name,
                path,
                {
                    "result_state": "ready" if productive else "failed",
                    "payload_count": 1,
                },
            )
        result = WorkerExecutionResult(
            payload_type="worker_execution_result",
            payload_version="v1",
            run_id=request.run_id,
            task_id=request.task_id,
            routing_plan_id=request.routing_plan_id,
            routing_decision_id=request.routing_decision_id,
            agent_id=request.agent_id,
            capability_id=request.capability_id,
            execution_status="completed" if productive else "failed",
            result_status=status,
            error_code=None if productive else "synthetic_tool_failed",
            retry_of_task_id=(
                request.retry_context.retry_of_task_id
                if request.retry_context is not None
                else None
            ),
            output_artifact_refs={
                name: WorkerArtifactRef(
                    artifact_id=artifact_id,
                    artifact_type=name,
                    storage_key=path,
                    run_id=request.run_id,
                )
            },
            compact_summary={"record_count": 1},
        )
        response = Task(
            id=request.task_id,
            status=TaskStatus(
                state=TaskState.COMPLETED if productive else TaskState.FAILED
            ),
            message=task.message,
            metadata=task.metadata,
        )
        response.artifacts = [
            {"parts": [{"type": "text", "text": result.model_dump_json()}]}
        ]
        self.windows.append((started, time.monotonic()))
        return response


class _FailPendingRetryCheckpoint:
    """Test-only fault at the pending-retry checkpoint, before retry HTTP."""

    def __init__(self, graph):
        self.graph = graph
        self.failed = False

    async def ainvoke(self, state, *, config):
        if not self.failed and any(
            task.retry_attempt > 0 and task.dispatch_status == "not_dispatched"
            for task in state.worker_tasks.values()
        ):
            self.failed = True
            raise RuntimeError("private checkpoint fault")
        return await self.graph.ainvoke(state, config=config)


class _FailTerminalCheckpoint:
    """Test-only fault after final disposition is computed, before persistence."""

    def __init__(self, graph):
        self.graph = graph

    async def ainvoke(self, state, *, config):
        if state.orchestrator.status == "routing_to_final":
            raise RuntimeError("private terminal checkpoint fault")
        return await self.graph.ainvoke(state, config=config)


class _FailMixedIngestionCheckpoint:
    """Test-only fault after both HTTP results, at ingestion persistence."""

    def __init__(self, graph):
        self.graph = graph
        self.calls = 0

    async def ainvoke(self, state, *, config):
        self.calls += 1
        # Dispatch writes dispatching and its merged transport state first.
        if self.calls == 3:
            raise RuntimeError("sk-live-PRIVATE-MIXED-CHECKPOINT")
        return await self.graph.ainvoke(state, config=config)


def _card(agent_id):
    return lambda url: AgentCard(
        name=agent_id,
        description="Synthetic retry transport fixture.",
        url=url,
    )


def _bind_http_targets(initial, discovery, handles):
    urls = {agent_id: handle.url for agent_id, handle in handles.items()}
    prepared = tuple(
        replace(
            item,
            dispatch_target=replace(
                item.dispatch_target,
                dispatch_url=urls[item.decision.agent_id],
            ),
        )
        for item in initial.prepared_tasks
    )

    def resolve(run_id, *, agent_id, capability_id, dispatch_mode="python_a2a"):
        assert run_id == RUN_ID
        return DispatchTarget(
            agent_id=agent_id,
            capability_id=capability_id,
            dispatch_url=urls[agent_id],
            dispatch_mode=dispatch_mode,
        )

    discovery.resolve_dispatch_target = resolve
    return prepared


def _independent_ready_environment(storage, registry):
    contracts = [_contract("agent_alpha"), _contract("agent_delta")]
    return (
        contracts,
        *_environment(
            storage,
            registry,
            contracts,
            ("agent_alpha", "agent_delta"),
        ),
    )


def _task_by_agent(state, agent_id):
    return next(
        task for task in state.worker_tasks.values() if task.agent_id == agent_id
    )


def _assert_reconciliation_state(state):
    assert state.run_status == "running"
    assert state.orchestrator.status == "evaluating_results"
    assert state.orchestrator.next_wakeup_reason == (
        "worker_result_reconciliation_required"
    )
    assert state.next_wakeup.target == "orchestrator_loop"
    assert state.next_wakeup.reason == "worker_result_reconciliation_required"


async def _run_chain(storage, registry, *, fail_count, failure_status="tool_failed"):
    contracts = [
        _contract("agent_alpha"),
        _contract("agent_beta", requires=("agent_alpha",)),
    ]
    service, _llm, discovery, initial, state = _environment(
        storage, registry, contracts, ("agent_alpha", "agent_beta")
    )
    alpha_worker = _SyntheticWorker(
        agent_id="agent_alpha",
        storage=storage,
        registry=registry,
        fail_count=fail_count,
        status=failure_status,
    )
    beta_worker = _SyntheticWorker(
        agent_id="agent_beta", storage=storage, registry=registry
    )
    handles = {
        "agent_alpha": _serve_task_handler(_card("agent_alpha"), alpha_worker.handle),
        "agent_beta": _serve_task_handler(_card("agent_beta"), beta_worker.handle),
    }
    try:
        prepared = _bind_http_targets(initial, discovery, handles)
        saver = InMemorySaver()
        graph = build_orchestrator_execution_graph(checkpointer=saver)
        result = await execute_orchestrator_worker_loop(
            run_id=RUN_ID,
            state=state,
            prepared_tasks=prepared,
            routing_service=service,
            discovery=discovery,
            registry=registry,
            storage=storage,
            execution_graph=graph,
            checkpoint_config=execution_graph_config(RUN_ID),
            timeout_seconds=2,
            max_worker_retries=3,
        )
        return result, alpha_worker, beta_worker, handles, saver
    finally:
        for handle in handles.values():
            handle.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("failed_attempts", [1, 2, 3])
async def test_retry_success_stops_immediately_then_releases_consumer(
    local_storage, registry_service, failed_attempts
):
    result, alpha, beta, handles, saver = await _run_chain(
        local_storage, registry_service, fail_count=failed_attempts
    )
    assert result.outcome == "completed"
    assert handles["agent_alpha"].hits["task"] == failed_attempts + 1
    assert handles["agent_beta"].hits["task"] == 1
    alpha_decision = next(
        item
        for item in result.state.routing.decisions.values()
        if item.agent_id == "agent_alpha"
    )
    assert len(alpha_decision.task_ids) == failed_attempts + 1
    for attempt, task_id in enumerate(alpha_decision.task_ids):
        task = result.state.worker_tasks[task_id]
        assert task.retry_attempt == attempt
        assert task.retry_of_task_id == (
            alpha_decision.task_ids[attempt - 1] if attempt else None
        )
    assert [request.retry_context is None for request in alpha.requests] == [
        True,
        *([False] * failed_attempts),
    ]
    assert all(
        request.retry_context.retry_attempt == attempt
        and request.retry_context.max_retry_attempts == 3
        and request.retry_context.retry_reason == "synthetic_tool_failed"
        for attempt, request in enumerate(alpha.requests[1:], start=1)
    )
    assert result.state.routing.decisions[alpha_decision.routing_decision_id].status == "completed"
    assert beta.requests[0].input_projection.input_artifact_refs
    serialized = repr(list(saver.list(None)))
    assert "WorkerExecutionRequest" not in serialized
    assert "PreparedA2ATask" not in serialized


@pytest.mark.asyncio
async def test_retry_exhaustion_posts_four_times_blocks_dependency_and_routes_final(
    local_storage, registry_service
):
    result, alpha, _beta, handles, _saver = await _run_chain(
        local_storage, registry_service, fail_count=4
    )
    assert result.outcome == "retry_exhausted"
    assert handles["agent_alpha"].hits["task"] == 4
    assert handles["agent_beta"].hits["task"] == 0
    assert len(alpha.requests) == 4
    by_agent = {item.agent_id: item for item in result.state.routing.decisions.values()}
    assert by_agent["agent_alpha"].status == "failed"
    assert by_agent["agent_beta"].status == "blocked"
    assert by_agent["agent_beta"].blocking_reason == "dependency_failed"
    assert result.state.run_status == "failed"
    assert result.state.orchestrator.status == "routing_to_final"
    assert result.state.next_wakeup.target == "final_response"
    assert result.state.next_wakeup.reason == "worker_retry_exhausted"
    assert result.state.artifacts[ARTIFACTS["agent_alpha"][0]].status == "invalid"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "outcome", "run_status", "orchestrator_status", "target", "reason"),
    [
        (
            "validation_failed",
            "non_retryable_failure",
            "failed",
            "routing_to_final",
            "final_response",
            "worker_non_retryable_failure",
        ),
        (
            "blocked",
            "non_retryable_failure",
            "failed",
            "routing_to_final",
            "final_response",
            "worker_non_retryable_failure",
        ),
        (
            "needs_user_input",
            "needs_user_input",
            "waiting_for_input",
            "planning",
            "user_input",
            "needs_user_input",
        ),
    ],
)
async def test_non_retryable_worker_result_has_deterministic_terminal_disposition(
    local_storage,
    registry_service,
    status,
    outcome,
    run_status,
    orchestrator_status,
    target,
    reason,
):
    result, alpha, _beta, handles, _saver = await _run_chain(
        local_storage,
        registry_service,
        fail_count=1,
        failure_status=status,
    )
    assert result.outcome == outcome
    assert handles["agent_alpha"].hits["task"] == 1
    assert handles["agent_beta"].hits["task"] == 0
    assert len(alpha.requests) == 1
    decision = next(
        item for item in result.state.routing.decisions.values() if item.agent_id == "agent_alpha"
    )
    assert len(decision.task_ids) == 1
    assert result.state.run_status == run_status
    assert result.state.orchestrator.status == orchestrator_status
    assert result.state.next_wakeup.target == target
    assert result.state.next_wakeup.reason == reason
    consumer = next(
        item for item in result.state.routing.decisions.values() if item.agent_id == "agent_beta"
    )
    assert consumer.status == "blocked"
    assert consumer.blocking_reason == "dependency_failed"


@pytest.mark.asyncio
async def test_two_independent_failures_are_retried_in_the_same_concurrent_round(
    local_storage, registry_service
):
    contracts = [_contract("agent_alpha"), _contract("agent_delta")]
    service, _llm, discovery, initial, state = _environment(
        local_storage,
        registry_service,
        contracts,
        ("agent_alpha", "agent_delta"),
    )
    barrier0 = threading.Barrier(2)
    barrier1 = threading.Barrier(2)
    workers = {
        agent_id: _SyntheticWorker(
            agent_id=agent_id,
            storage=local_storage,
            registry=registry_service,
            fail_count=1,
        )
        for agent_id in ("agent_alpha", "agent_delta")
    }
    for worker in workers.values():
        worker.barriers = {0: barrier0, 1: barrier1}
    handles = {
        agent_id: _serve_task_handler(_card(agent_id), worker.handle)
        for agent_id, worker in workers.items()
    }
    try:
        prepared = _bind_http_targets(initial, discovery, handles)
        graph = build_orchestrator_execution_graph(checkpointer=InMemorySaver())
        result = await execute_orchestrator_worker_loop(
            run_id=RUN_ID,
            state=state,
            prepared_tasks=prepared,
            routing_service=service,
            discovery=discovery,
            registry=registry_service,
            storage=local_storage,
            execution_graph=graph,
            checkpoint_config=execution_graph_config(RUN_ID),
            timeout_seconds=2,
            max_worker_retries=3,
        )
        assert result.outcome == "completed"
        assert Counter(handle.hits["task"] for handle in handles.values()) == Counter({2: 2})
        for attempt in (0, 1):
            left = workers["agent_alpha"].windows[attempt]
            right = workers["agent_delta"].windows[attempt]
            assert max(left[0], right[0]) < min(left[1], right[1])
    finally:
        for handle in handles.values():
            handle.close()


@pytest.mark.asyncio
async def test_mixed_success_and_connection_failure_ingests_success_then_replays_safely(
    local_storage, registry_service
):
    (
        _contracts,
        service,
        _llm,
        discovery,
        initial,
        state,
    ) = _independent_ready_environment(local_storage, registry_service)
    alpha_worker = _SyntheticWorker(
        agent_id="agent_alpha",
        storage=local_storage,
        registry=registry_service,
    )
    alpha = _serve_task_handler(_card("agent_alpha"), alpha_worker.handle)
    closed = SimpleNamespace(
        url=f"http://127.0.0.1:{_free_port()}", hits=Counter()
    )
    saver = InMemorySaver()
    graph = build_orchestrator_execution_graph(checkpointer=saver)
    config = execution_graph_config(RUN_ID)
    try:
        prepared = _bind_http_targets(
            initial,
            discovery,
            {"agent_alpha": alpha, "agent_delta": closed},
        )
        result = await execute_orchestrator_worker_loop(
            run_id=RUN_ID,
            state=state,
            prepared_tasks=prepared,
            routing_service=service,
            discovery=discovery,
            registry=registry_service,
            storage=local_storage,
            execution_graph=graph,
            checkpoint_config=config,
            timeout_seconds=1,
            max_worker_retries=3,
        )
        alpha_task = _task_by_agent(result.state, "agent_alpha")
        delta_task = _task_by_agent(result.state, "agent_delta")
        artifact_name, _ = ARTIFACTS["agent_alpha"]
        artifact = result.state.artifacts[artifact_name]
        active_id = getattr(
            registry_service.get(RUN_ID).active_artifacts,
            f"{artifact_name}_id",
        )

        assert result.outcome == "reconciliation_required"
        assert result.dispatch_attempt_count == 2
        _assert_reconciliation_state(result.state)
        assert alpha.hits == Counter({"card": 1, "task": 1})
        # No server listens on the failed target, so no remote GET or task POST
        # can be observed there; the connection fails during AgentCard fetch.
        assert closed.hits == Counter()
        assert alpha_task.dispatch_status == "dispatched"
        assert alpha_task.execution_status == "completed"
        assert alpha_task.result_status == "success"
        assert delta_task.dispatch_status == "dispatch_failed"
        assert delta_task.execution_status == "not_started"
        assert delta_task.result_status is None
        assert set(result.completion_proofs) == {alpha_task.task_id}
        assert artifact.status == "available"
        assert artifact.artifact_id == active_id
        assert artifact.producer_task_id == alpha_task.task_id
        assert all(
            len(decision.task_ids) == 1
            for decision in result.state.routing.decisions.values()
        )

        checkpoint_count = len(list(saver.list(None)))
        before_hits = Counter(alpha.hits)
        reconstructed = build_orchestrator_execution_graph(checkpointer=saver)
        restored = OrchestratorExecutionState.model_validate(
            reconstructed.get_state(config).values
        )
        replay = await execute_orchestrator_worker_loop(
            run_id=RUN_ID,
            state=restored,
            prepared_tasks=(),
            routing_service=None,
            discovery=None,
            registry=None,
            storage=None,
            execution_graph=reconstructed,
            checkpoint_config=config,
            timeout_seconds=1,
            max_worker_retries=3,
            completion_proofs=result.completion_proofs,
        )
        assert replay.outcome == "reconciliation_required"
        _assert_reconciliation_state(replay.state)
        assert Counter(alpha.hits) == before_hits
        assert replay.dispatch_attempt_count == 0
        assert len(list(saver.list(None))) == checkpoint_count
        assert replay.state.artifacts[artifact_name] == artifact
        assert set(replay.completion_proofs) == {alpha_task.task_id}

        compact_blob = "\n".join(
            (
                result.state.model_dump_json(),
                replay.state.model_dump_json(),
                repr(list(saver.list(None))),
            )
        ).lower()
        for forbidden in (
            "workerexecutionrequest",
            "prepareda2atask",
            "synthetic retry transport fixture",
            "authorization",
            "api_key",
        ):
            assert forbidden not in compact_blob
    finally:
        alpha.close()


@pytest.mark.asyncio
async def test_mixed_success_and_timeout_ingests_success_without_retrying_timeout(
    local_storage, registry_service
):
    (
        _contracts,
        service,
        _llm,
        discovery,
        initial,
        state,
    ) = _independent_ready_environment(local_storage, registry_service)
    alpha_worker = _SyntheticWorker(
        agent_id="agent_alpha",
        storage=local_storage,
        registry=registry_service,
    )
    delta_worker = _SyntheticWorker(
        agent_id="agent_delta",
        storage=local_storage,
        registry=registry_service,
    )
    alpha = _serve_task_handler(_card("agent_alpha"), alpha_worker.handle)

    def delayed_delta(task):
        time.sleep(0.15)
        return delta_worker.handle(task)

    delta = _serve_task_handler(_card("agent_delta"), delayed_delta)
    try:
        prepared = _bind_http_targets(
            initial,
            discovery,
            {"agent_alpha": alpha, "agent_delta": delta},
        )
        result = await execute_orchestrator_worker_loop(
            run_id=RUN_ID,
            state=state,
            prepared_tasks=prepared,
            routing_service=service,
            discovery=discovery,
            registry=registry_service,
            storage=local_storage,
            execution_graph=build_orchestrator_execution_graph(
                checkpointer=InMemorySaver()
            ),
            checkpoint_config=execution_graph_config(RUN_ID),
            timeout_seconds=0.03,
            max_worker_retries=3,
        )
        alpha_task = _task_by_agent(result.state, "agent_alpha")
        delta_task = _task_by_agent(result.state, "agent_delta")
        assert result.outcome == "reconciliation_required"
        assert result.dispatch_attempt_count == 2
        _assert_reconciliation_state(result.state)
        assert alpha_task.execution_status == "completed"
        assert set(result.completion_proofs) == {alpha_task.task_id}
        assert delta_task.dispatch_status == "dispatch_failed"
        assert delta_task.agent_failure_reason == "dispatch_timeout"
        assert delta_task.execution_status == "not_started"
        assert alpha.hits == Counter({"card": 1, "task": 1})
        assert delta.hits == Counter({"card": 1, "task": 1})
        time.sleep(0.2)
        assert delta.hits["task"] == 1
        assert len(delta_worker.requests) == 1
        assert all(
            len(decision.task_ids) == 1
            for decision in result.state.routing.decisions.values()
        )
    finally:
        alpha.close()
        delta.close()


@pytest.mark.asyncio
async def test_mixed_tool_failure_and_transport_failure_keeps_terminal_proof_without_retry(
    local_storage, registry_service
):
    (
        _contracts,
        service,
        _llm,
        discovery,
        initial,
        state,
    ) = _independent_ready_environment(local_storage, registry_service)
    alpha_worker = _SyntheticWorker(
        agent_id="agent_alpha",
        storage=local_storage,
        registry=registry_service,
        fail_count=1,
    )
    alpha = _serve_task_handler(_card("agent_alpha"), alpha_worker.handle)
    closed = SimpleNamespace(url=f"http://127.0.0.1:{_free_port()}")
    try:
        prepared = _bind_http_targets(
            initial,
            discovery,
            {"agent_alpha": alpha, "agent_delta": closed},
        )
        result = await execute_orchestrator_worker_loop(
            run_id=RUN_ID,
            state=state,
            prepared_tasks=prepared,
            routing_service=service,
            discovery=discovery,
            registry=registry_service,
            storage=local_storage,
            execution_graph=build_orchestrator_execution_graph(
                checkpointer=InMemorySaver()
            ),
            checkpoint_config=execution_graph_config(RUN_ID),
            timeout_seconds=1,
            max_worker_retries=3,
        )
        alpha_task = _task_by_agent(result.state, "agent_alpha")
        assert result.outcome == "reconciliation_required"
        _assert_reconciliation_state(result.state)
        assert alpha_task.execution_status == "failed"
        assert alpha_task.result_status == "tool_failed"
        assert set(result.completion_proofs) == {alpha_task.task_id}
        assert len(
            next(
                decision
                for decision in result.state.routing.decisions.values()
                if decision.agent_id == "agent_alpha"
            ).task_ids
        ) == 1
        assert alpha.hits == Counter({"card": 1, "task": 1})
        assert result.state.artifacts[ARTIFACTS["agent_alpha"][0]].status == (
            "invalid"
        )
    finally:
        alpha.close()


@pytest.mark.asyncio
async def test_mixed_ingestion_checkpoint_failure_exposes_success_recovery_without_resend(
    local_storage, registry_service
):
    (
        _contracts,
        service,
        _llm,
        discovery,
        initial,
        state,
    ) = _independent_ready_environment(local_storage, registry_service)
    alpha_worker = _SyntheticWorker(
        agent_id="agent_alpha",
        storage=local_storage,
        registry=registry_service,
    )
    alpha = _serve_task_handler(_card("agent_alpha"), alpha_worker.handle)
    closed = SimpleNamespace(url=f"http://127.0.0.1:{_free_port()}")
    saver = InMemorySaver()
    failing = _FailMixedIngestionCheckpoint(
        build_orchestrator_execution_graph(checkpointer=saver)
    )
    try:
        prepared = _bind_http_targets(
            initial,
            discovery,
            {"agent_alpha": alpha, "agent_delta": closed},
        )
        with pytest.raises(
            ResultIngestionPostCheckpointError,
            match="^result_ingestion_post_checkpoint_failed$",
        ) as caught:
            await execute_orchestrator_worker_loop(
                run_id=RUN_ID,
                state=state,
                prepared_tasks=prepared,
                routing_service=service,
                discovery=discovery,
                registry=registry_service,
                storage=local_storage,
                execution_graph=failing,
                checkpoint_config=execution_graph_config(RUN_ID),
                timeout_seconds=1,
                max_worker_retries=3,
            )
        recovery = caught.value.recovery_result
        alpha_task = _task_by_agent(recovery.state, "agent_alpha")
        delta_task = _task_by_agent(recovery.state, "agent_delta")
        assert alpha_task.execution_status == "completed"
        assert alpha_task.result_status == "success"
        assert delta_task.dispatch_status == "dispatch_failed"
        assert delta_task.execution_status == "not_started"
        assert set(recovery.completion_proofs) == {alpha_task.task_id}
        assert alpha.hits == Counter({"card": 1, "task": 1})
        assert failing.calls == 3
        checkpoint_count = len(list(saver.list(None)))
        assert checkpoint_count > 0
        persisted = OrchestratorExecutionState.model_validate(
            failing.graph.get_state(execution_graph_config(RUN_ID)).values
        )
        assert _task_by_agent(persisted, "agent_alpha").execution_status == (
            "not_started"
        )
        assert _task_by_agent(persisted, "agent_delta").dispatch_status == (
            "dispatch_failed"
        )
        for exposed in (str(caught.value), repr(caught.value), caught.value.args):
            text = repr(exposed).lower()
            assert "private-mixed-checkpoint" not in text
            assert "http://" not in text
            assert alpha_task.task_id.lower() not in text
    finally:
        alpha.close()


def test_retry_production_modules_have_no_business_worker_name_branches():
    from pathlib import Path

    root = Path(__file__).parents[2]
    text = "\n".join(
        (root / "app/a2a" / name).read_text()
        for name in ("orchestrator_retry.py", "orchestrator_execution_loop.py")
    ).lower()
    for forbidden in (
        "step5",
        "step6",
        "structure",
        "candidate_context",
        "developability",
    ):
        assert forbidden not in text


@pytest.mark.asyncio
async def test_dispatch_timeout_is_uncertain_and_never_auto_retried(
    local_storage, registry_service
):
    contracts = [_contract("agent_alpha")]
    service, _llm, discovery, initial, state = _environment(
        local_storage, registry_service, contracts, ("agent_alpha",)
    )
    worker = _SyntheticWorker(
        agent_id="agent_alpha",
        storage=local_storage,
        registry=registry_service,
    )

    def delayed(task):
        time.sleep(0.15)
        return worker.handle(task)

    handle = _serve_task_handler(_card("agent_alpha"), delayed)
    try:
        prepared = _bind_http_targets(
            initial, discovery, {"agent_alpha": handle}
        )
        saver = InMemorySaver()
        result = await execute_orchestrator_worker_loop(
            run_id=RUN_ID,
            state=state,
            prepared_tasks=prepared,
            routing_service=service,
            discovery=discovery,
            registry=registry_service,
            storage=local_storage,
            execution_graph=build_orchestrator_execution_graph(checkpointer=saver),
            checkpoint_config=execution_graph_config(RUN_ID),
            timeout_seconds=0.02,
            max_worker_retries=3,
        )
        assert result.outcome == "reconciliation_required"
        assert result.dispatch_attempt_count == 1
        assert handle.hits == Counter({"card": 1, "task": 1})
        task = next(iter(result.state.worker_tasks.values()))
        assert task.dispatch_status == "dispatch_failed"
        assert task.agent_failure_reason == "dispatch_timeout"
        assert task.retry_attempt == 0
        assert len(result.prepared_task_history) == 1
        time.sleep(0.2)
        assert len(worker.requests) == 1  # handler may finish after client timeout
    finally:
        handle.close()


@pytest.mark.asyncio
async def test_ingestion_invalid_without_proof_requires_reconciliation(
    local_storage, registry_service
):
    service, _llm, discovery, initial, state = _environment(
        local_storage,
        registry_service,
        [_contract("agent_alpha")],
        ("agent_alpha",),
    )

    def malformed(task):
        response = Task(
            id=task.id,
            status=TaskStatus(state=TaskState.FAILED),
            message=task.message,
            metadata=task.metadata,
        )
        response.artifacts = [{"parts": [{"type": "text", "text": "not-json"}]}]
        return response

    handle = _serve_task_handler(_card("agent_alpha"), malformed)
    try:
        prepared = _bind_http_targets(
            initial, discovery, {"agent_alpha": handle}
        )
        result = await execute_orchestrator_worker_loop(
            run_id=RUN_ID,
            state=state,
            prepared_tasks=prepared,
            routing_service=service,
            discovery=discovery,
            registry=registry_service,
            storage=local_storage,
            execution_graph=build_orchestrator_execution_graph(
                checkpointer=InMemorySaver()
            ),
            checkpoint_config=execution_graph_config(RUN_ID),
            timeout_seconds=1,
            max_worker_retries=3,
        )
        assert handle.hits["task"] == 1
        assert result.outcome == "reconciliation_required"
        assert result.state.run_status == "running"
        assert result.state.orchestrator.status == "evaluating_results"
        assert result.state.next_wakeup.reason == (
            "worker_result_reconciliation_required"
        )
        assert result.completion_proofs == {}
        assert len(result.state.routing.decisions[next(iter(result.state.routing.decisions))].task_ids) == 1
    finally:
        handle.close()


@pytest.mark.asyncio
async def test_pending_retry_checkpoint_recovery_and_fresh_process_reconstruction(
    local_storage, registry_service
):
    contracts = [_contract("agent_alpha")]
    service, _llm, discovery, initial, state = _environment(
        local_storage, registry_service, contracts, ("agent_alpha",)
    )
    worker = _SyntheticWorker(
        agent_id="agent_alpha",
        storage=local_storage,
        registry=registry_service,
        fail_count=1,
    )
    handle = _serve_task_handler(_card("agent_alpha"), worker.handle)
    saver = InMemorySaver()
    base_graph = build_orchestrator_execution_graph(checkpointer=saver)
    failing_graph = _FailPendingRetryCheckpoint(base_graph)
    try:
        prepared = _bind_http_targets(
            initial, discovery, {"agent_alpha": handle}
        )
        with pytest.raises(
            ExecutionLoopCheckpointError,
            match="^execution_loop_checkpoint_failed$",
        ) as caught:
            await execute_orchestrator_worker_loop(
                run_id=RUN_ID,
                state=state,
                prepared_tasks=prepared,
                routing_service=service,
                discovery=discovery,
                registry=registry_service,
                storage=local_storage,
                execution_graph=failing_graph,
                checkpoint_config=execution_graph_config(RUN_ID),
                timeout_seconds=1,
                max_worker_retries=3,
            )
        recovery = caught.value.recovery_result
        retry_ids = dispatch_eligible_task_ids(recovery.state)
        assert len(retry_ids) == 1
        assert handle.hits["task"] == 1
        assert failing_graph.failed is True

        # Persist only the compact pending state, then reconstruct all ephemeral
        # Task authority from a new service/discovery instance.
        await base_graph.ainvoke(
            recovery.state, config=execution_graph_config(RUN_ID)
        )
        fresh_discovery = _FrozenDiscovery(contracts)
        fresh_discovery.resolve_dispatch_target = (
            lambda run_id, **kwargs: DispatchTarget(
                agent_id=kwargs["agent_id"],
                capability_id=kwargs["capability_id"],
                dispatch_url=handle.url,
                dispatch_mode=kwargs["dispatch_mode"],
            )
        )
        fresh_llm = _DeterministicLLM(_proposal("agent_alpha"))
        fresh_service = OrchestratorRoutingService(
            discovery=fresh_discovery,
            storage=local_storage,
            registry=registry_service,
            llm=fresh_llm,
        )
        reconstructed_graph = build_orchestrator_execution_graph(
            checkpointer=saver
        )
        restored = OrchestratorExecutionState.model_validate(
            reconstructed_graph.get_state(execution_graph_config(RUN_ID)).values
        )
        resumed = await execute_orchestrator_worker_loop(
            run_id=RUN_ID,
            state=restored,
            prepared_tasks=(),
            routing_service=fresh_service,
            discovery=fresh_discovery,
            registry=registry_service,
            storage=local_storage,
            execution_graph=reconstructed_graph,
            checkpoint_config=execution_graph_config(RUN_ID),
            timeout_seconds=1,
            max_worker_retries=3,
            completion_proofs=recovery.completion_proofs,
        )
        assert resumed.outcome == "completed"
        assert handle.hits["task"] == 2
        assert retry_ids[0] in resumed.state.worker_tasks
        assert resumed.state.worker_tasks[retry_ids[0]].retry_attempt == 1
        assert fresh_llm.call_count == 0
    finally:
        handle.close()


@pytest.mark.asyncio
async def test_final_disposition_checkpoint_failure_returns_typed_recovery(
    local_storage, registry_service
):
    contracts = [_contract("agent_alpha")]
    service, _llm, discovery, initial, state = _environment(
        local_storage, registry_service, contracts, ("agent_alpha",)
    )
    worker = _SyntheticWorker(
        agent_id="agent_alpha",
        storage=local_storage,
        registry=registry_service,
        fail_count=1,
        status="validation_failed",
    )
    handle = _serve_task_handler(_card("agent_alpha"), worker.handle)
    saver = InMemorySaver()
    try:
        prepared = _bind_http_targets(
            initial, discovery, {"agent_alpha": handle}
        )
        with pytest.raises(
            ExecutionLoopCheckpointError,
            match="^execution_loop_checkpoint_failed$",
        ) as caught:
            await execute_orchestrator_worker_loop(
                run_id=RUN_ID,
                state=state,
                prepared_tasks=prepared,
                routing_service=service,
                discovery=discovery,
                registry=registry_service,
                storage=local_storage,
                execution_graph=_FailTerminalCheckpoint(
                    build_orchestrator_execution_graph(checkpointer=saver)
                ),
                checkpoint_config=execution_graph_config(RUN_ID),
                timeout_seconds=1,
                max_worker_retries=3,
            )
        recovery = caught.value.recovery_result
        assert handle.hits["task"] == 1
        assert recovery.outcome == "non_retryable_failure"
        assert recovery.state.run_status == "failed"
        assert recovery.state.orchestrator.status == "routing_to_final"
        assert recovery.state.next_wakeup.reason == "worker_non_retryable_failure"
        assert dispatch_eligible_task_ids(recovery.state) == ()
        assert "private terminal" not in str(caught.value)
        assert "private terminal" not in repr(caught.value)
    finally:
        handle.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "corruption",
    [
        "missing_parent",
        "attempt_gap",
        "lineage_cycle",
        "attempt_over_limit",
        "cross_agent",
        "cross_capability",
        "lineage_fork",
        "duplicate_task",
    ],
)
async def test_lineage_corruption_fails_before_checkpoint_or_network(
    local_storage, registry_service, corruption
):
    contracts = [_contract("agent_alpha")]
    service, _llm, discovery, initial, state = _environment(
        local_storage, registry_service, contracts, ("agent_alpha",)
    )
    task_id = next(iter(state.worker_tasks))
    task = state.worker_tasks[task_id]
    decision_id = task.routing_decision_id
    tasks = dict(state.worker_tasks)
    decisions = dict(state.routing.decisions)
    update = {}
    if corruption == "missing_parent":
        update = {"retry_attempt": 1, "retry_of_task_id": None}
    elif corruption == "attempt_gap":
        update = {"retry_attempt": 2, "retry_of_task_id": task_id}
    elif corruption == "lineage_cycle":
        update = {"retry_of_task_id": task_id}
    elif corruption == "attempt_over_limit":
        update = {"max_retry_attempts": 4}
    elif corruption == "cross_agent":
        update = {"agent_id": "agent_intruder"}
    elif corruption == "cross_capability":
        update = {"capability_id": "capability_intruder"}
    elif corruption in {"lineage_fork", "duplicate_task"}:
        retry_id = "task_aaaaaaaaaaaaaaaa"
        tasks[retry_id] = task.model_copy(
            update={
                "task_id": retry_id,
                "retry_attempt": 1,
                "retry_of_task_id": task_id,
            }
        )
        task_ids = (
            [task_id, retry_id, retry_id]
            if corruption == "duplicate_task"
            else [task_id, retry_id, "task_bbbbbbbbbbbbbbbb"]
        )
        if corruption == "lineage_fork":
            tasks["task_bbbbbbbbbbbbbbbb"] = task.model_copy(
                update={
                    "task_id": "task_bbbbbbbbbbbbbbbb",
                    "retry_attempt": 1,
                    "retry_of_task_id": task_id,
                }
            )
        decisions[decision_id] = decisions[decision_id].model_copy(
            update={"task_ids": task_ids}
        )
    if update:
        tasks[task_id] = task.model_copy(update=update)
    corrupt = state.model_copy(
        update={
            "worker_tasks": tasks,
            "routing": state.routing.model_copy(update={"decisions": decisions}),
        }
    )
    saver = InMemorySaver()
    with pytest.raises(
        OrchestratorExecutionLoopError, match="^execution_loop_state_invalid$"
    ) as caught:
        await execute_orchestrator_worker_loop(
            run_id=RUN_ID,
            state=corrupt,
            prepared_tasks=initial.prepared_tasks,
            routing_service=service,
            discovery=discovery,
            registry=registry_service,
            storage=local_storage,
            execution_graph=build_orchestrator_execution_graph(checkpointer=saver),
            checkpoint_config=execution_graph_config(RUN_ID),
            timeout_seconds=1,
            max_worker_retries=3,
        )
    assert len(list(saver.list(None))) == 0
    assert "task_" not in str(caught.value)


def test_required_retry_configuration_is_explicit(monkeypatch):
    from app.settings import Settings

    monkeypatch.setenv("ORCHESTRATOR_MAX_WORKER_RETRIES", "3")
    assert Settings().orchestrator_max_worker_retries == 3


@pytest.mark.asyncio
async def test_completed_checkpoint_reconstruction_and_replay_send_zero_posts(
    local_storage, registry_service
):
    first, _alpha, _beta, _handles, saver = await _run_chain(
        local_storage, registry_service, fail_count=1
    )
    graph = build_orchestrator_execution_graph(checkpointer=saver)
    config = execution_graph_config(RUN_ID)
    restored = OrchestratorExecutionState.model_validate(graph.get_state(config).values)
    before_task_ids = tuple(sorted(first.state.worker_tasks))
    replay = await execute_orchestrator_worker_loop(
        run_id=RUN_ID,
        state=restored,
        prepared_tasks=(),
        routing_service=None,
        discovery=None,
        registry=None,
        storage=None,
        execution_graph=graph,
        checkpoint_config=config,
        timeout_seconds=1,
        max_worker_retries=3,
        completion_proofs=first.completion_proofs,
    )
    assert replay.outcome == "completed"
    assert replay.dispatch_round_count == 0
    assert replay.dispatch_attempt_count == 0
    assert tuple(sorted(replay.state.worker_tasks)) == before_task_ids
    serialized = "\n".join(
        [repr(first), first.model_dump_json(), repr(dict(first)), repr(list(first))]
    )
    assert "PreparedA2ATask" not in serialized
    assert "WorkerExecutionRequest" not in serialized
    assert "WorkerExecutionResult" not in serialized
    assert "Synthetic retry transport fixture" not in serialized
    with pytest.raises(
        TypeError, match="^orchestrator_execution_loop_result_pickle_unsupported$"
    ):
        pickle.dumps(first)
