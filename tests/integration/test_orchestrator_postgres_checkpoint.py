"""Real AsyncPostgresSaver cross-runtime persistence and privacy smoke."""

from __future__ import annotations

import asyncio
import os
import socket
import threading
import time
from collections import Counter

import psycopg
import pytest
from flask import jsonify, request
from python_a2a import A2AServer
from python_a2a.server.http import create_flask_app
from psycopg.rows import dict_row
from werkzeug.serving import make_server

from app.a2a.agent_cards import _build_agent_card
from app.a2a.orchestrator_discovery import (
    ExpectedWorkerEndpoint,
    WorkerDiscoveryService,
)
from app.a2a.orchestrator_execution_state import (
    dispatch_eligible_task_ids,
    execution_state_from_routing_result,
    mark_task_dispatched,
    mark_task_dispatching,
    mark_task_running,
)
from app.a2a.orchestrator_dispatch import (
    DispatchPostCheckpointError,
    dispatch_orchestrator_tasks,
)
from app.a2a.orchestrator_execution_loop import (
    ExecutionLoopCheckpointError,
    execute_orchestrator_worker_loop,
)
from app.a2a.orchestrator_discovery import DispatchTarget
from app.a2a.orchestrator_resume import resume_orchestrator_run
from app.a2a.orchestrator_routing_service import OrchestratorRoutingService
from app.graph.orchestrator_checkpoint_runtime import (
    OrchestratorCheckpointRuntimeError,
    OrchestratorPostgresCheckpointRuntime,
)
from app.graph.orchestrator_execution_graph import execution_graph_config
from app.schemas.orchestrator_execution_state import OrchestratorExecutionState
from tests.a2a.test_orchestrator_execution_state import routing_result_fixture
from tests.a2a.test_orchestrator_dispatch import (
    _FailSecondCheckpointGraph,
    _serve_task_handler,
)
from tests.a2a.test_orchestrator_post_ingestion import (
    RUN_ID,
    _contract,
    _DeterministicLLM,
    _environment,
    _FrozenDiscovery,
    _proposal,
)
from tests.a2a.test_orchestrator_retry_loop import (
    _bind_http_targets,
    _card,
    _FailPendingRetryCheckpoint,
    _SyntheticWorker,
)


def _database_url() -> str:
    value = os.environ.get("LANGGRAPH_CHECKPOINT_DATABASE_URL", "")
    if not value:
        pytest.skip("real Postgres integration requires explicit test DSN")
    return value


def _serve_discoverable_worker(contract, handler):
    """Real AgentCard + health + task transport for synthetic integration."""
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    url = f"http://127.0.0.1:{port}"
    card = _build_agent_card(contract=contract, url=url)

    class _Server(A2AServer):
        def handle_task(self, task):
            return handler(task)

    app = create_flask_app(
        _Server(agent_card=card, google_a2a_compatible=False)
    )
    hits = Counter()

    @app.before_request
    def _count():
        if "agent.json" in request.path:
            hits["card"] += 1
        elif request.path == "/health":
            hits["health"] += 1
        elif request.path in {"/tasks/send", "/a2a/tasks/send"}:
            hits["task"] += 1
        elif request.path in {"/tasks/get", "/a2a/tasks/get"}:
            hits["get_task"] += 1

    @app.route("/health")
    def _health():
        return jsonify(
            {
                "status": "ok",
                "agent_id": contract.agent_id,
                "capabilities": [
                    item.capability_id for item in contract.capabilities
                ],
            }
        )

    server = make_server("127.0.0.1", port, app, threaded=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    class _Handle:
        def close(self):
            server.shutdown()
            thread.join(timeout=5)

    handle = _Handle()
    handle.url = url
    handle.hits = hits
    return handle


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["wrong_credentials", "unreachable"])
async def test_postgres_startup_failure_is_compact_without_fallback(mode):
    dsn = _database_url()
    if mode == "wrong_credentials":
        invalid = dsn.replace("checkpoint_test_only", "wrong_password")
    else:
        invalid = "postgresql://checkpoint_test:private@127.0.0.1:1/unavailable"
    runtime = OrchestratorPostgresCheckpointRuntime(invalid)
    with pytest.raises(
        OrchestratorCheckpointRuntimeError,
        match="^checkpoint_runtime_startup_failed$",
    ) as caught:
        await runtime.startup()
    assert "wrong_password" not in str(caught.value)
    assert "private" not in repr(caught.value)
    with pytest.raises(
        OrchestratorCheckpointRuntimeError,
        match="^checkpoint_runtime_not_started$",
    ):
        _ = runtime.graph


@pytest.mark.asyncio
async def test_runtime_a_to_b_persists_isolated_compact_state_and_migrations():
    dsn = _database_url()
    initial = execution_state_from_routing_result(routing_result_fixture())
    task_id = "task_1111111111111111"
    dispatched = mark_task_dispatched(
        mark_task_dispatching(initial, task_id), task_id
    )
    config = execution_graph_config(dispatched.run_id)

    runtime_a = OrchestratorPostgresCheckpointRuntime(dsn)
    parallel_runtime = OrchestratorPostgresCheckpointRuntime(dsn)
    await asyncio.gather(runtime_a.startup(), parallel_runtime.startup())
    await parallel_runtime.shutdown()
    await runtime_a.graph.ainvoke(dispatched, config=config)
    await runtime_a.shutdown()

    runtime_b = OrchestratorPostgresCheckpointRuntime(dsn)
    await runtime_b.startup()  # second real setup/migration call is idempotent
    snapshot = await runtime_b.graph.aget_state(config)
    restored = OrchestratorExecutionState.model_validate(snapshot.values)
    assert restored == dispatched
    assert restored.worker_tasks[task_id].dispatch_status == "dispatched"
    assert set(snapshot.values) == {
        "run_id",
        "run_status",
        "orchestrator",
        "routing",
        "worker_tasks",
        "artifacts",
        "memory_refs",
        "next_wakeup",
    }

    other = dispatched.model_copy(update={"run_id": "run_20260714_deadbeef"})
    other_config = execution_graph_config(other.run_id)
    await runtime_b.graph.ainvoke(other, config=other_config)
    assert OrchestratorExecutionState.model_validate(
        (await runtime_b.graph.aget_state(config)).values
    ).run_id == dispatched.run_id
    assert OrchestratorExecutionState.model_validate(
        (await runtime_b.graph.aget_state(other_config)).values
    ).run_id == other.run_id
    await runtime_b.shutdown()

    async with await psycopg.AsyncConnection.connect(
        dsn, autocommit=True, row_factory=dict_row
    ) as conn:
        rows = await (
            await conn.execute(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = current_schema() "
                "AND table_name LIKE 'checkpoint%' ORDER BY table_name"
            )
        ).fetchall()
        table_names = {row["table_name"] for row in rows}
        assert {
            "checkpoint_blobs",
            "checkpoint_migrations",
            "checkpoint_writes",
            "checkpoints",
        } <= table_names
        checkpoint_count = (
            await (await conn.execute("SELECT count(*) AS n FROM checkpoints")).fetchone()
        )["n"]
        assert checkpoint_count > 0
        serialized_rows = await (
            await conn.execute(
                "SELECT thread_id, checkpoint::text AS checkpoint_text, "
                "metadata::text AS metadata_text FROM checkpoints"
            )
        ).fetchall()
        blob_rows = await (
            await conn.execute(
                "SELECT thread_id, channel, type, blob "
                "FROM checkpoint_blobs"
            )
        ).fetchall()
        write_rows = await (
            await conn.execute(
                "SELECT thread_id, task_id, channel, type, blob "
                "FROM checkpoint_writes"
            )
        ).fetchall()

    blob = repr((serialized_rows, blob_rows, write_rows)).lower()
    assert dispatched.run_id in blob
    assert other.run_id in blob
    for forbidden in (
        "sk-live-checkpoint-secret",
        "authorization",
        "workerexecutionresult",
        "workerexecutionrequest",
        "prepareda2atask",
        "http://",
        "postgresql://",
        "checkpoint_test_only",
        "raw_tooluniverse_payload",
        "full_prompt",
        "raw_llm_response",
        "atomprivatepdb",
        ">private_fasta",
    ):
        assert forbidden not in blob


@pytest.mark.asyncio
async def test_pending_retry_survives_runtime_reconstruction_and_replays_terminal(
    local_storage, registry_service
):
    """Synthetic HTTP worker only; no live LLM/MCP/ToolUniverse call."""
    dsn = _database_url()
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
    runtime_a = OrchestratorPostgresCheckpointRuntime(dsn)
    runtime_b = OrchestratorPostgresCheckpointRuntime(dsn)
    runtime_c = OrchestratorPostgresCheckpointRuntime(dsn)
    try:
        await runtime_a.startup()
        prepared = _bind_http_targets(
            initial, discovery, {"agent_alpha": handle}
        )
        failing_graph = _FailPendingRetryCheckpoint(runtime_a.graph)
        with pytest.raises(ExecutionLoopCheckpointError) as caught:
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
        pending = caught.value.recovery_result
        retry_ids = dispatch_eligible_task_ids(pending.state)
        assert len(retry_ids) == 1
        retry_id = retry_ids[0]
        root_id = pending.state.worker_tasks[retry_id].retry_of_task_id
        await runtime_a.graph.ainvoke(
            pending.state, config=execution_graph_config(RUN_ID)
        )
        await runtime_a.shutdown()

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
        await runtime_b.startup()
        resumed = await resume_orchestrator_run(
            run_id=RUN_ID,
            checkpoint_runtime=runtime_b,
            routing_service=fresh_service,
            discovery=fresh_discovery,
            registry=registry_service,
            storage=local_storage,
            timeout_seconds=1,
            max_worker_retries=3,
        )
        assert resumed.outcome == "completed"
        assert resumed.dispatch_attempt_count == 1
        assert fresh_llm.call_count == 0
        assert set(resumed.state.worker_tasks) == {root_id, retry_id}
        retry = resumed.state.worker_tasks[retry_id]
        assert retry.retry_attempt == 1
        assert retry.retry_of_task_id == root_id
        assert handle.hits["task"] == 2
        assert handle.hits["get_task"] == 1
        assert handle.hits["card"] == 3
        assert all(
            artifact.status == "available"
            for artifact in resumed.state.artifacts.values()
            if artifact.producer_task_id is not None
        )
        await runtime_b.shutdown()

        # A third process sees the terminal compact checkpoint and performs no
        # discovery, reconciliation, routing LLM call, or task POST.
        await runtime_c.startup()
        before = dict(handle.hits)
        fresh_discovery.discover_for_run = lambda _run_id: (_ for _ in ()).throw(
            AssertionError("terminal resume must not discover")
        )
        replay = await resume_orchestrator_run(
            run_id=RUN_ID,
            checkpoint_runtime=runtime_c,
            routing_service=fresh_service,
            discovery=fresh_discovery,
            registry=registry_service,
            storage=local_storage,
            timeout_seconds=1,
            max_worker_retries=3,
        )
        assert replay.outcome == "completed"
        assert replay.dispatch_attempt_count == 0
        assert dict(handle.hits) == before
    finally:
        await runtime_a.shutdown()
        await runtime_b.shutdown()
        await runtime_c.shutdown()
        handle.close()


@pytest.mark.asyncio
async def test_mixed_success_timeout_is_durable_then_reconciled_without_resend(
    local_storage, registry_service
):
    """Real HTTP timeout/get_task with synthetic deterministic workers."""
    dsn = _database_url()
    contracts = [_contract("agent_alpha"), _contract("agent_beta")]
    service, _llm, discovery, initial, state = _environment(
        local_storage,
        registry_service,
        contracts,
        ("agent_alpha", "agent_beta"),
    )
    alpha_worker = _SyntheticWorker(
        agent_id="agent_alpha",
        storage=local_storage,
        registry=registry_service,
    )
    beta_worker = _SyntheticWorker(
        agent_id="agent_beta",
        storage=local_storage,
        registry=registry_service,
    )

    def delayed_beta(task):
        time.sleep(0.15)
        return beta_worker.handle(task)

    alpha = _serve_task_handler(_card("agent_alpha"), alpha_worker.handle)
    beta = _serve_task_handler(_card("agent_beta"), delayed_beta)
    runtime_a = OrchestratorPostgresCheckpointRuntime(dsn)
    runtime_b = OrchestratorPostgresCheckpointRuntime(dsn)
    try:
        await runtime_a.startup()
        prepared = _bind_http_targets(
            initial,
            discovery,
            {"agent_alpha": alpha, "agent_beta": beta},
        )
        uncertain = await execute_orchestrator_worker_loop(
            run_id=RUN_ID,
            state=state,
            prepared_tasks=prepared,
            routing_service=service,
            discovery=discovery,
            registry=registry_service,
            storage=local_storage,
            execution_graph=runtime_a.graph,
            checkpoint_config=execution_graph_config(RUN_ID),
            timeout_seconds=0.05,
            max_worker_retries=3,
        )
        assert uncertain.outcome == "reconciliation_required"
        assert len(uncertain.completion_proofs) == 1
        assert alpha.hits["task"] == 1
        assert beta.hits["task"] == 1
        await runtime_a.shutdown()
        time.sleep(0.2)

        fresh_discovery = _FrozenDiscovery(contracts)
        targets = {"agent_alpha": alpha.url, "agent_beta": beta.url}
        fresh_discovery.resolve_dispatch_target = (
            lambda run_id, **kwargs: DispatchTarget(
                agent_id=kwargs["agent_id"],
                capability_id=kwargs["capability_id"],
                dispatch_url=targets[kwargs["agent_id"]],
                dispatch_mode=kwargs["dispatch_mode"],
            )
        )
        fresh_llm = _DeterministicLLM(
            _proposal("agent_alpha", "agent_beta")
        )
        fresh_service = OrchestratorRoutingService(
            discovery=fresh_discovery,
            storage=local_storage,
            registry=registry_service,
            llm=fresh_llm,
        )
        await runtime_b.startup()
        resumed = await resume_orchestrator_run(
            run_id=RUN_ID,
            checkpoint_runtime=runtime_b,
            routing_service=fresh_service,
            discovery=fresh_discovery,
            registry=registry_service,
            storage=local_storage,
            timeout_seconds=1,
            max_worker_retries=3,
        )
        assert resumed.outcome == "completed"
        assert resumed.dispatch_attempt_count == 0
        assert fresh_llm.call_count == 0
        assert alpha.hits["task"] == beta.hits["task"] == 1
        assert alpha.hits["get_task"] == beta.hits["get_task"] == 1
        assert alpha.hits["card"] == beta.hits["card"] == 2
        assert all(
            task.execution_status == "completed"
            for task in resumed.state.worker_tasks.values()
        )
        assert all(
            artifact.status == "available"
            for artifact in resumed.state.artifacts.values()
            if artifact.producer_task_id is not None
        )
    finally:
        await runtime_a.shutdown()
        await runtime_b.shutdown()
        alpha.close()
        beta.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "crash_state", ["dispatching", "dispatched_not_started", "running"]
)
async def test_real_postgres_crash_windows_reconcile_by_get_without_resend(
    local_storage,
    registry_service,
    crash_state,
):
    """Synthetic localhost transport; no live LLM/MCP/ToolUniverse call."""
    dsn = _database_url()
    contracts = [_contract("agent_alpha")]
    service, _llm, discovery, initial, state = _environment(
        local_storage, registry_service, contracts, ("agent_alpha",)
    )
    worker = _SyntheticWorker(
        agent_id="agent_alpha",
        storage=local_storage,
        registry=registry_service,
    )
    handle = _serve_task_handler(_card("agent_alpha"), worker.handle)
    runtime_a = OrchestratorPostgresCheckpointRuntime(dsn)
    runtime_b = OrchestratorPostgresCheckpointRuntime(dsn)
    config = execution_graph_config(RUN_ID)
    try:
        await runtime_a.startup()
        prepared = _bind_http_targets(
            initial, discovery, {"agent_alpha": handle}
        )
        if crash_state == "dispatching":
            failing_graph = _FailSecondCheckpointGraph(
                runtime_a.graph, raw_failure="private post checkpoint fault"
            )
            with pytest.raises(DispatchPostCheckpointError):
                await dispatch_orchestrator_tasks(
                    run_id=RUN_ID,
                    state=state,
                    prepared_tasks=prepared,
                    discovery=discovery,
                    routing_service=service,
                    execution_graph=failing_graph,
                    checkpoint_config=config,
                    timeout_seconds=1,
                )
        else:
            dispatched = await dispatch_orchestrator_tasks(
                run_id=RUN_ID,
                state=state,
                prepared_tasks=prepared,
                discovery=discovery,
                routing_service=service,
                execution_graph=runtime_a.graph,
                checkpoint_config=config,
                timeout_seconds=1,
            )
            if crash_state == "running":
                task_id = next(iter(dispatched.state.worker_tasks))
                running = mark_task_running(dispatched.state, task_id)
                await runtime_a.graph.ainvoke(running, config=config)
        assert handle.hits["task"] == 1
        persisted = OrchestratorExecutionState.model_validate(
            (await runtime_a.graph.aget_state(config)).values
        )
        task_before = next(iter(persisted.worker_tasks.values()))
        expected_dispatch = (
            "dispatching" if crash_state == "dispatching" else "dispatched"
        )
        expected_execution = "running" if crash_state == "running" else "not_started"
        assert task_before.dispatch_status == expected_dispatch
        assert task_before.execution_status == expected_execution
        await runtime_a.shutdown()

        fresh_discovery = _FrozenDiscovery(contracts)
        fresh_discovery.resolve_dispatch_target = (
            lambda run_id, **kwargs: DispatchTarget(
                agent_id=kwargs["agent_id"],
                capability_id=kwargs["capability_id"],
                dispatch_url=handle.url,
                dispatch_mode=kwargs["dispatch_mode"],
            )
        )
        fresh_service = OrchestratorRoutingService(
            discovery=fresh_discovery,
            storage=local_storage,
            registry=registry_service,
            llm=_DeterministicLLM(_proposal("agent_alpha")),
        )
        await runtime_b.startup()
        resumed = await resume_orchestrator_run(
            run_id=RUN_ID,
            checkpoint_runtime=runtime_b,
            routing_service=fresh_service,
            discovery=fresh_discovery,
            registry=registry_service,
            storage=local_storage,
            timeout_seconds=1,
            max_worker_retries=3,
        )
        task_after = next(iter(resumed.state.worker_tasks.values()))
        assert resumed.outcome == "completed"
        assert resumed.dispatch_attempt_count == 0
        assert task_after.execution_status == "completed"
        assert task_after.result_status == "success"
        assert handle.hits["task"] == 1
        assert handle.hits["get_task"] == 1
        assert handle.hits["card"] == 2
    finally:
        await runtime_a.shutdown()
        await runtime_b.shutdown()
        handle.close()


@pytest.mark.asyncio
async def test_real_postgres_unknown_task_stays_reconciliation_without_post(
    local_storage, registry_service
):
    dsn = _database_url()
    contracts = [_contract("agent_alpha")]
    service, _llm, discovery, initial, state = _environment(
        local_storage, registry_service, contracts, ("agent_alpha",)
    )
    worker = _SyntheticWorker(
        agent_id="agent_alpha",
        storage=local_storage,
        registry=registry_service,
    )
    handle = _serve_task_handler(_card("agent_alpha"), worker.handle)
    runtime_a = OrchestratorPostgresCheckpointRuntime(dsn)
    runtime_b = OrchestratorPostgresCheckpointRuntime(dsn)
    config = execution_graph_config(RUN_ID)
    try:
        await runtime_a.startup()
        _bind_http_targets(initial, discovery, {"agent_alpha": handle})
        task_id = next(iter(state.worker_tasks))
        uncertain = mark_task_dispatched(mark_task_dispatching(state, task_id), task_id)
        await runtime_a.graph.ainvoke(uncertain, config=config)
        await runtime_a.shutdown()

        fresh_discovery = _FrozenDiscovery(contracts)
        fresh_discovery.resolve_dispatch_target = (
            lambda run_id, **kwargs: DispatchTarget(
                agent_id=kwargs["agent_id"],
                capability_id=kwargs["capability_id"],
                dispatch_url=handle.url,
                dispatch_mode=kwargs["dispatch_mode"],
            )
        )
        fresh_service = OrchestratorRoutingService(
            discovery=fresh_discovery,
            storage=local_storage,
            registry=registry_service,
            llm=_DeterministicLLM(_proposal("agent_alpha")),
        )
        await runtime_b.startup()
        resumed = await resume_orchestrator_run(
            run_id=RUN_ID,
            checkpoint_runtime=runtime_b,
            routing_service=fresh_service,
            discovery=fresh_discovery,
            registry=registry_service,
            storage=local_storage,
            timeout_seconds=1,
            max_worker_retries=3,
        )
        assert resumed.outcome == "reconciliation_required"
        assert resumed.dispatch_attempt_count == 0
        assert resumed.completion_proofs == {}
        assert handle.hits["task"] == 0
        assert handle.hits["get_task"] == 2
        assert handle.hits["card"] == 1
    finally:
        await runtime_a.shutdown()
        await runtime_b.shutdown()
        handle.close()


@pytest.mark.asyncio
async def test_real_discovery_cache_authorizes_resume_and_http_get_task(
    local_storage, registry_service
):
    """Production discovery/health authority with synthetic task execution."""
    dsn = _database_url()
    synthetic = _contract("agent_alpha")
    contract = synthetic.model_copy(
        update={
            "capabilities": [
                synthetic.capabilities[0].model_copy(
                    update={"required_artifact_fields": {}}
                )
            ]
        }
    )
    _seed_service, _llm, _seed_discovery, _initial, state = _environment(
        local_storage, registry_service, [contract], ("agent_alpha",)
    )
    worker = _SyntheticWorker(
        agent_id="agent_alpha",
        storage=local_storage,
        registry=registry_service,
    )
    handle = _serve_discoverable_worker(contract, worker.handle)
    discovery = WorkerDiscoveryService(
        expected_workers=[
            ExpectedWorkerEndpoint(
                contract.agent_id,
                (contract.capabilities[0].capability_id,),
                handle.url,
            )
        ],
        storage=local_storage,
        registry=registry_service,
        discovery_timeout_seconds=1,
        health_timeout_seconds=1,
    )
    service = OrchestratorRoutingService(
        discovery=discovery,
        storage=local_storage,
        registry=registry_service,
        llm=_DeterministicLLM(_proposal("agent_alpha")),
    )
    runtime_a = OrchestratorPostgresCheckpointRuntime(dsn)
    runtime_b = OrchestratorPostgresCheckpointRuntime(dsn)
    config = execution_graph_config(RUN_ID)
    try:
        snapshot = discovery.discover_for_run(RUN_ID)
        assert snapshot.discovery_status == "all_available"
        assert snapshot.available_agent_ids == ["agent_alpha"]
        await runtime_a.startup()
        prepared = tuple(
            service.rebuild_task_from_execution_state(
                run_id=RUN_ID,
                execution_state=state,
                task_id=task_id,
            )
            for task_id in dispatch_eligible_task_ids(state)
        )
        dispatched = await dispatch_orchestrator_tasks(
            run_id=RUN_ID,
            state=state,
            prepared_tasks=prepared,
            discovery=discovery,
            routing_service=service,
            execution_graph=runtime_a.graph,
            checkpoint_config=config,
            timeout_seconds=1,
        )
        assert next(iter(dispatched.state.worker_tasks.values())).dispatch_status == (
            "dispatched"
        )
        assert handle.hits == Counter({"card": 3, "health": 1, "task": 1})
        await runtime_a.shutdown()

        await runtime_b.startup()
        resumed = await resume_orchestrator_run(
            run_id=RUN_ID,
            checkpoint_runtime=runtime_b,
            routing_service=service,
            discovery=discovery,
            registry=registry_service,
            storage=local_storage,
            timeout_seconds=1,
            max_worker_retries=3,
        )
        assert resumed.outcome == "completed"
        assert resumed.dispatch_attempt_count == 0
        assert next(iter(resumed.state.worker_tasks.values())).result_status == (
            "success"
        )
        assert handle.hits == Counter(
            {"card": 4, "health": 1, "task": 1, "get_task": 1}
        )
        # The same-run discovery call inside resume used the frozen catalog:
        # only the get_task client's AgentCard fetch increased card traffic.
        assert discovery.discover_for_run(RUN_ID) == snapshot
        assert handle.hits["health"] == 1
    finally:
        await runtime_a.shutdown()
        await runtime_b.shutdown()
        handle.close()
