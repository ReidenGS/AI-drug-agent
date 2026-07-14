"""Turn F2-B1 concurrent transport tests.

Localhost servers are test-only transport fixtures.  They exercise the real
``A2AClient.send_task_async -> TCP -> A2AServer`` path and never substitute a
production direct-call fallback.  The final Step 5 case additionally runs the
real request-based worker production path with deterministic local MCP bindings
and MockLLMProvider; it is not a live LLM/MCP/ToolUniverse smoke.
"""

from __future__ import annotations

import asyncio
import json
import pickle
import socket
import threading
import time
from collections import Counter
from dataclasses import replace
from types import SimpleNamespace

import pytest
from flask import Flask, jsonify
from langgraph.checkpoint.memory import InMemorySaver
from python_a2a import (
    A2AServer,
    AgentCard,
    Message,
    MessageRole,
    Task,
    TaskState,
    TaskStatus,
    TextContent,
)
from python_a2a.server.http import create_flask_app
from werkzeug.serving import make_server

from app.a2a.agent_cards import (
    AGENT_ID_STEP5,
    CAP_STEP5_CANDIDATE_CONTEXT,
    build_step5_agent_card,
    build_step6_agent_card,
)
from app.a2a.contracts import (
    A2ATaskMetadata,
    InputProjection,
    OrchestratorRoutingDecisionRef,
    PrivacyConstraints,
    RetryContext,
    RuntimeRef,
    WorkerExecutionRequest,
    WorkerRequestSpec,
)
from app.a2a.orchestrator_discovery import DispatchTarget
from app.a2a.orchestrator_discovery import (
    ExpectedWorkerEndpoint,
    WorkerDiscoveryService,
)
from app.a2a.orchestrator_dispatch import (
    DispatchPostCheckpointError,
    OrchestratorDispatchError,
    OrchestratorDispatchResult,
    dispatch_orchestrator_tasks,
)
from app.a2a.orchestrator_execution_state import (
    dispatch_eligible_task_ids,
    execution_state_from_routing_result,
)
from app.a2a.orchestrator_routing_service import OrchestratorRoutingServiceResult
from app.a2a.orchestrator_result_ingestion import (
    ingest_orchestrator_worker_results,
)
from app.a2a.orchestrator_task_builder import (
    PreparedA2ATask,
    build_canonical_worker_execution_request,
)
from app.a2a.step5_worker import Step5A2AWorker, create_step5_flask_app
from app.agents.candidate_context_agent import CandidateContextAgent
from app.agents.supervisor_agent import SupervisorAgent
from app.graph.orchestrator_execution_graph import (
    build_orchestrator_execution_graph,
    execution_graph_config,
)
from app.llm.provider import MockLLMProvider
from app.mcp.client import LocalMCPClient
from app.services.input_readiness_service import InputReadinessService
from app.services.intake_service import IntakeService
from app.services.structured_query_service import StructuredQueryService
from app.utils.ids import new_run_id
from app.schemas.worker_routing_plan import (
    OrchestratorRouteDecision,
    ValidatedRoutingDecision,
    WorkerRoutingPlan,
)
from tests.a2a.test_orchestrator_execution_state import routing_result_fixture


@pytest.fixture(autouse=True)
def _localhost_proxy_isolation(monkeypatch):
    """Test environment only: bypass the host proxy for localhost sockets."""
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


def _request_for(prepared: PreparedA2ATask) -> WorkerExecutionRequest:
    decision = prepared.decision
    assert decision.task_id is not None
    return WorkerExecutionRequest(
        payload_type="worker_execution_request",
        payload_version="v1",
        run_id="run_20260714_abcdef12",
        task_id=decision.task_id,
        routing_plan_id="wrp_0123456789abcdef",
        routing_decision_id=decision.routing_decision_id,
        agent_id=decision.agent_id,
        capability_id=decision.capability_id,
        created_by="step_04_orchestrator",
        worker_request=WorkerRequestSpec(
            objective=decision.objective,
            reason=decision.selection_reason,
            priority=decision.priority,
        ),
        orchestrator_routing_decision=OrchestratorRoutingDecisionRef(
            planned_status="run",
            dispatch_mode="python_a2a",
            deterministic_gate_status="passed",
            expected_outputs=decision.expected_output_artifact_names,
        ),
        input_projection=InputProjection(
            input_artifact_refs=prepared.input_artifact_refs
        ),
        privacy_constraints=PrivacyConstraints(),
    )


def _replace_request(
    prepared: PreparedA2ATask,
    request: WorkerExecutionRequest,
    *,
    metadata: dict | None = None,
) -> PreparedA2ATask:
    message = Message(
        content=TextContent(text=request.model_dump_json()),
        role=MessageRole.USER,
    )
    return replace(
        prepared,
        task=Task(
            id=prepared.task.id,
            message=message.to_dict(),
            metadata=metadata or prepared.task.metadata,
        ),
    )


def _transport_ready_result(*, targets: dict[str, str] | None = None):
    result = routing_result_fixture(independent_ready=True)
    safe_decisions = [
        decision.model_copy(
            update={
                "objective": f"Execute {decision.capability_id}.",
                "selection_reason": "Validated routing selection.",
            }
        )
        for decision in result.plan.validated_decisions
    ]
    safe_by_id = {
        decision.routing_decision_id: decision for decision in safe_decisions
    }
    safe_proposals = [
        proposal.model_copy(
            update={
                "objective": f"Execute {proposal.capability_id}.",
                "selection_reason": "Validated routing selection.",
            }
        )
        for proposal in result.plan.proposed_decisions
    ]
    result = replace(
        result,
        plan=result.plan.model_copy(
            update={
                "validated_decisions": safe_decisions,
                "proposed_decisions": safe_proposals,
            }
        ),
    )
    updated: list[PreparedA2ATask] = []
    for prepared in result.prepared_tasks:
        prepared = replace(
            prepared,
            decision=safe_by_id[prepared.decision.routing_decision_id],
        )
        request = _request_for(prepared)
        metadata = A2ATaskMetadata(
            adc_payload_type="worker_execution_request",
            adc_payload_version="v1",
            run_id=request.run_id,
            task_id=request.task_id,
            routing_plan_id=request.routing_plan_id,
            routing_decision_id=request.routing_decision_id,
            agent_id=request.agent_id,
            capability_id=request.capability_id,
            created_by=request.created_by,
        )
        message = Message(
            content=TextContent(text=request.model_dump_json()),
            role=MessageRole.USER,
        )
        url = (
            targets[prepared.decision.agent_id]
            if targets is not None
            else prepared.dispatch_target.dispatch_url
        )
        updated.append(
            PreparedA2ATask(
                decision=prepared.decision,
                task=Task(
                    id=request.task_id,
                    message=message.to_dict(),
                    metadata=metadata.model_dump(),
                ),
                dispatch_target=DispatchTarget(
                    agent_id=prepared.decision.agent_id,
                    capability_id=prepared.decision.capability_id,
                    dispatch_url=url,
                    dispatch_mode="python_a2a",
                ),
                input_artifact_refs=prepared.input_artifact_refs,
            )
        )
    return replace(result, prepared_tasks=tuple(updated))


def _single_prepared_result(base, *, index: int = 0, url: str | None = None):
    prepared = base.prepared_tasks[index]
    if url is not None:
        prepared = replace(
            prepared,
            dispatch_target=replace(prepared.dispatch_target, dispatch_url=url),
        )
    plan = base.plan.model_copy(
        update={
            "validated_decisions": [base.plan.validated_decisions[index]],
            "proposed_decisions": [base.plan.proposed_decisions[index]],
            "ready_task_count": 1,
            "dependency_edges": [],
        }
    )
    return replace(base, plan=plan, prepared_tasks=(prepared,))


def _synthetic_dispatch_result(targets: dict[str, str]):
    run_id = "run_20260714_1234abcd"
    routing_plan_id = "wrp_abcdef0123456789"
    specs = (
        (
            "route_aaaaaaaaaaaaaaaa",
            "task_aaaaaaaaaaaaaaaa",
            "agent_alpha",
            "capability_alpha",
            "artifact_alpha_output",
        ),
        (
            "route_bbbbbbbbbbbbbbbb",
            "task_bbbbbbbbbbbbbbbb",
            "agent_beta",
            "capability_beta",
            "artifact_beta_output",
        ),
    )
    decisions = [
        ValidatedRoutingDecision(
            routing_decision_id=route_id,
            agent_id=agent_id,
            capability_id=capability_id,
            objective=f"Execute {capability_id}.",
            selection_reason="Independent synthetic dispatch contract.",
            priority="normal",
            validation_status="ready",
            expected_output_artifact_names=[output_name],
            task_id=task_id,
        )
        for route_id, task_id, agent_id, capability_id, output_name in specs
    ]
    prepared = []
    for decision in decisions:
        request = build_canonical_worker_execution_request(
            run_id=run_id,
            routing_plan_id=routing_plan_id,
            decision=decision,
            input_artifact_refs={},
        )
        metadata = A2ATaskMetadata(
            adc_payload_type="worker_execution_request",
            adc_payload_version="v1",
            run_id=run_id,
            task_id=decision.task_id,
            routing_plan_id=routing_plan_id,
            routing_decision_id=decision.routing_decision_id,
            agent_id=decision.agent_id,
            capability_id=decision.capability_id,
            created_by=request.created_by,
        )
        message = Message(
            content=TextContent(text=request.model_dump_json()),
            role=MessageRole.USER,
        )
        prepared.append(
            PreparedA2ATask(
                decision=decision,
                task=Task(
                    id=decision.task_id,
                    message=message.to_dict(),
                    metadata=metadata.model_dump(),
                ),
                dispatch_target=DispatchTarget(
                    agent_id=decision.agent_id,
                    capability_id=decision.capability_id,
                    dispatch_url=targets[decision.agent_id],
                    dispatch_mode="python_a2a",
                ),
                input_artifact_refs={},
            )
        )
    plan = WorkerRoutingPlan(
        run_id=run_id,
        routing_plan_id=routing_plan_id,
        planned_at="2026-07-14T00:00:00Z",
        loop_decision="dispatch_next_workers",
        routing_status="ready",
        llm_selection_source="llm_primary_validated",
        proposed_decisions=[
            OrchestratorRouteDecision(
                agent_id=item.agent_id,
                capability_id=item.capability_id,
                objective=item.objective,
                selection_reason=item.selection_reason,
                priority=item.priority,
            )
            for item in decisions
        ],
        validated_decisions=decisions,
        ready_task_count=2,
        waiting_decision_count=0,
        rejected_decision_count=0,
    )
    return OrchestratorRoutingServiceResult(
        plan=plan,
        plan_artifact_id="worker_routing_plan_123456789abc",
        prepared_tasks=tuple(prepared),
        reused_existing_plan=False,
        llm_called=True,
        discovery_performed=True,
    )


class _FrozenTargets:
    def __init__(self, prepared_tasks):
        self.targets = {
            item.decision.agent_id: item.dispatch_target
            for item in prepared_tasks
        }
        self.resolve_count = 0

    def resolve_dispatch_target(
        self, run_id, *, agent_id, capability_id, dispatch_mode="python_a2a"
    ):
        self.resolve_count += 1
        target = self.targets[agent_id]
        assert target.capability_id == capability_id
        assert target.dispatch_mode == dispatch_mode
        return target


class _CountingGraph:
    def __init__(self, graph):
        self.graph = graph
        self.inputs: list[dict] = []

    async def ainvoke(self, state, *, config):
        self.inputs.append(state.model_dump())
        return await self.graph.ainvoke(state, config=config)


class _FailSecondCheckpointGraph(_CountingGraph):
    """Test-only checkpoint fault after network; no production fallback."""

    def __init__(self, graph, *, raw_failure: str):
        super().__init__(graph)
        self.raw_failure = raw_failure

    async def ainvoke(self, state, *, config):
        self.inputs.append(state.model_dump())
        if len(self.inputs) == 2:
            raise RuntimeError(self.raw_failure)
        return await self.graph.ainvoke(state, config=config)


class _Handle:
    def __init__(self, url, server, thread, hits):
        self.url = url
        self.server = server
        self.thread = thread
        self.hits = hits

    def close(self):
        self.server.shutdown()
        self.thread.join(timeout=5)


def _free_port() -> int:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def _serve_task_handler(card_builder, handler) -> _Handle:
    port = _free_port()
    url = f"http://127.0.0.1:{port}"

    class _Server(A2AServer):
        def handle_task(self, task):
            return handler(task)

    app = create_flask_app(
        _Server(agent_card=card_builder(url), google_a2a_compatible=False)
    )
    hits = Counter()

    @app.before_request
    def _count():
        from flask import request

        if "agent.json" in request.path:
            hits["card"] += 1
        elif request.path in {"/tasks/send", "/a2a/tasks/send"}:
            hits["task"] += 1
        elif request.path in {"/tasks/get", "/a2a/tasks/get"}:
            hits["get_task"] += 1

    server = make_server("127.0.0.1", port, app, threaded=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return _Handle(url, server, thread, hits)


def _successful_task(task: Task, *, state: TaskState = TaskState.COMPLETED) -> Task:
    return Task(
        id=str(task.id),
        status=TaskStatus(state=state),
        message=task.message,
        metadata=task.metadata,
    )


def _graph_for(state):
    saver = InMemorySaver()
    graph = _CountingGraph(build_orchestrator_execution_graph(checkpointer=saver))
    return graph, saver, execution_graph_config(state.run_id)


async def test_two_real_http_workers_overlap_and_checkpoint_batch_once_per_phase():
    barrier = threading.Barrier(2)
    windows: dict[str, tuple[float, float]] = {}

    def handler(name):
        def _handle(task):
            start = time.monotonic()
            barrier.wait(timeout=2)
            time.sleep(0.05)
            windows[name] = (start, time.monotonic())
            return _successful_task(task)

        return _handle

    step5 = _serve_task_handler(build_step5_agent_card, handler("step5"))
    step6 = _serve_task_handler(build_step6_agent_card, handler("step6"))
    try:
        seed = routing_result_fixture(independent_ready=True)
        targets = {
            seed.prepared_tasks[0].decision.agent_id: step5.url,
            seed.prepared_tasks[1].decision.agent_id: step6.url,
        }
        routing = _transport_ready_result(targets=targets)
        state = execution_state_from_routing_result(routing)
        discovery = _FrozenTargets(routing.prepared_tasks)
        graph, saver, config = _graph_for(state)

        result = await dispatch_orchestrator_tasks(
            run_id=state.run_id,
            state=state,
            prepared_tasks=routing.prepared_tasks,
            discovery=discovery,
            execution_graph=graph,
            checkpoint_config=config,
            timeout_seconds=2,
        )

        assert step5.hits["task"] == step6.hits["task"] == 1
        assert step5.hits["card"] == step6.hits["card"] == 1
        assert max(start for start, _ in windows.values()) < min(
            end for _, end in windows.values()
        )
        assert len(graph.inputs) == 2
        assert {
            task["dispatch_status"]
            for task in graph.inputs[0]["worker_tasks"].values()
        } == {"dispatching"}
        assert {
            task["dispatch_status"]
            for task in graph.inputs[1]["worker_tasks"].values()
        } == {"dispatched"}
        assert [receipt.dispatch_status for receipt in result.receipts] == [
            "dispatched",
            "dispatched",
        ]
        assert set(result.response_tasks) == set(state.worker_tasks)
        assert dispatch_eligible_task_ids(result.state) == ()
        assert result.state.orchestrator.status == "waiting_for_workers"
        assert all(
            artifact.status == "producing"
            for artifact in result.state.artifacts.values()
            if artifact.producer_task_id
        )
        checkpoint_blob = repr(list(saver.list(None))).lower()
        for forbidden in (
            "workerexecutionrequest",
            "dispatch_url",
            "task.message",
            step5.url,
            step6.url,
            "authorization",
        ):
            assert forbidden.lower() not in checkpoint_blob
    finally:
        step5.close()
        step6.close()


async def test_synthetic_alpha_beta_ready_tasks_dispatch_concurrently():
    barrier = threading.Barrier(2)
    windows: dict[str, tuple[float, float]] = {}

    def _handler(name):
        def _handle(task):
            started = time.monotonic()
            barrier.wait(timeout=2)
            time.sleep(0.05)
            windows[name] = (started, time.monotonic())
            return _successful_task(task)

        return _handle

    def _card(agent_id):
        return lambda url: AgentCard(
            name=agent_id,
            description="Synthetic independent worker.",
            url=url,
        )

    alpha = _serve_task_handler(_card("agent_alpha"), _handler("alpha"))
    beta = _serve_task_handler(_card("agent_beta"), _handler("beta"))
    try:
        routing = _synthetic_dispatch_result(
            {"agent_alpha": alpha.url, "agent_beta": beta.url}
        )
        state = execution_state_from_routing_result(routing)
        assert dispatch_eligible_task_ids(state) == (
            "task_aaaaaaaaaaaaaaaa",
            "task_bbbbbbbbbbbbbbbb",
        )
        graph, _, config = _graph_for(state)
        result = await dispatch_orchestrator_tasks(
            run_id=state.run_id,
            state=state,
            prepared_tasks=routing.prepared_tasks,
            discovery=_FrozenTargets(routing.prepared_tasks),
            execution_graph=graph,
            checkpoint_config=config,
            timeout_seconds=2,
        )
        assert alpha.hits["task"] == beta.hits["task"] == 1
        assert max(start for start, _ in windows.values()) < min(
            end for _, end in windows.values()
        )
        assert {item.agent_id for item in result.receipts} == {
            "agent_alpha",
            "agent_beta",
        }
        assert {item.capability_id for item in result.receipts} == {
            "capability_alpha",
            "capability_beta",
        }
        assert set(result.state.artifacts) == {
            "artifact_alpha_output",
            "artifact_beta_output",
        }
        assert {item.dispatch_status for item in result.receipts} == {
            "dispatched"
        }
    finally:
        alpha.close()
        beta.close()


async def test_one_success_and_one_connection_failure_are_merged_independently():
    success = _serve_task_handler(build_step5_agent_card, _successful_task)
    closed_port = _free_port()
    try:
        seed = routing_result_fixture(independent_ready=True)
        targets = {
            seed.prepared_tasks[0].decision.agent_id: success.url,
            seed.prepared_tasks[1].decision.agent_id: f"http://127.0.0.1:{closed_port}",
        }
        routing = _transport_ready_result(targets=targets)
        state = execution_state_from_routing_result(routing)
        graph, _, config = _graph_for(state)
        result = await dispatch_orchestrator_tasks(
            run_id=state.run_id,
            state=state,
            prepared_tasks=routing.prepared_tasks,
            discovery=_FrozenTargets(routing.prepared_tasks),
            execution_graph=graph,
            checkpoint_config=config,
            timeout_seconds=1,
        )
        receipts = {item.task_id: item for item in result.receipts}
        success_id = routing.prepared_tasks[0].decision.task_id
        failed_id = routing.prepared_tasks[1].decision.task_id
        assert receipts[success_id].dispatch_status == "dispatched"
        assert receipts[success_id].agent_failure_reason == "none"
        assert receipts[failed_id].dispatch_status == "dispatch_failed"
        assert receipts[failed_id].agent_failure_reason == (
            "dispatch_connection_failed"
        )
        assert set(result.response_tasks) == {success_id}
        assert len(graph.inputs) == 2
        assert {
            item["dispatch_status"]
            for item in graph.inputs[0]["worker_tasks"].values()
        } == {"dispatching"}
        assert sorted(
            item["dispatch_status"]
            for item in graph.inputs[1]["worker_tasks"].values()
        ) == ["dispatch_failed", "dispatched"]
        assert result.state.artifacts["candidate_context_table"].status == (
            "producing"
        )
        assert result.state.artifacts["structured_liability_summary"].status == (
            "invalid"
        )
    finally:
        success.close()


async def test_pre_checkpoint_failure_has_zero_http_side_effects():
    servers = (
        _serve_task_handler(build_step5_agent_card, _successful_task),
        _serve_task_handler(build_step6_agent_card, _successful_task),
    )

    class _FailFirstCheckpoint:
        async def ainvoke(self, state, *, config):
            raise RuntimeError("raw pre-checkpoint exception private")

    try:
        seed = routing_result_fixture(independent_ready=True)
        targets = {
            seed.prepared_tasks[0].decision.agent_id: servers[0].url,
            seed.prepared_tasks[1].decision.agent_id: servers[1].url,
        }
        routing = _transport_ready_result(targets=targets)
        state = execution_state_from_routing_result(routing)
        with pytest.raises(
            OrchestratorDispatchError, match="^dispatch_pre_checkpoint_failed$"
        ) as caught:
            await dispatch_orchestrator_tasks(
                run_id=state.run_id,
                state=state,
                prepared_tasks=routing.prepared_tasks,
                discovery=_FrozenTargets(routing.prepared_tasks),
                execution_graph=_FailFirstCheckpoint(),
                checkpoint_config=execution_graph_config(state.run_id),
                timeout_seconds=2,
            )
        assert "raw pre-checkpoint exception private" not in repr(caught.value)
        assert all(server.hits["card"] == 0 for server in servers)
        assert all(server.hits["task"] == 0 for server in servers)
    finally:
        for server in servers:
            server.close()


async def test_post_checkpoint_failure_preserves_repr_safe_recovery_without_resend():
    sentinel = "sk-live-RESPONSE-PRIVATE-SENTINEL"

    def _response(task):
        response = _successful_task(task)
        response.artifacts = [
            {"parts": [{"type": "text", "text": sentinel}]}
        ]
        return response

    servers = (
        _serve_task_handler(build_step5_agent_card, _response),
        _serve_task_handler(build_step6_agent_card, _response),
    )
    try:
        seed = routing_result_fixture(independent_ready=True)
        targets = {
            seed.prepared_tasks[0].decision.agent_id: servers[0].url,
            seed.prepared_tasks[1].decision.agent_id: servers[1].url,
        }
        routing = _transport_ready_result(targets=targets)
        state = execution_state_from_routing_result(routing)
        saver = InMemorySaver()
        compiled = build_orchestrator_execution_graph(checkpointer=saver)
        graph = _FailSecondCheckpointGraph(
            compiled,
            raw_failure="raw post-checkpoint exception private",
        )
        config = execution_graph_config(state.run_id)

        with pytest.raises(DispatchPostCheckpointError) as caught:
            await dispatch_orchestrator_tasks(
                run_id=state.run_id,
                state=state,
                prepared_tasks=routing.prepared_tasks,
                discovery=_FrozenTargets(routing.prepared_tasks),
                execution_graph=graph,
                checkpoint_config=config,
                timeout_seconds=2,
            )

        error = caught.value
        assert str(error) == "dispatch_post_checkpoint_failed"
        assert error.args == ("dispatch_post_checkpoint_failed",)
        recovery = error.recovery_result
        assert len(recovery.receipts) == 2
        assert {item.dispatch_status for item in recovery.receipts} == {
            "dispatched"
        }
        assert set(recovery.response_tasks) == set(state.worker_tasks)
        assert all(
            task.status.state == TaskState.COMPLETED
            for task in recovery.response_tasks.values()
        )
        assert sentinel not in repr(recovery)
        assert sentinel not in recovery.model_dump_json()
        assert sentinel not in repr(dict(recovery))
        with pytest.raises(TypeError) as result_pickle:
            pickle.dumps(recovery)
        assert str(result_pickle.value) == (
            "orchestrator_dispatch_result_pickle_unsupported"
        )
        with pytest.raises(TypeError) as error_pickle:
            pickle.dumps(error)
        assert str(error_pickle.value) == (
            "dispatch_post_checkpoint_error_pickle_unsupported"
        )
        persisted = compiled.get_state(config).values
        assert {
            item["dispatch_status"]
            for item in persisted["worker_tasks"].values()
        } == {"dispatching"}
        assert len(graph.inputs) == 2
        assert all(server.hits["task"] == 1 for server in servers)
        await asyncio.sleep(0.05)
        assert all(server.hits["task"] == 1 for server in servers)

        compact_error = " ".join(
            (str(error), repr(error), repr(error.args))
        ).lower()
        checkpoint_blob = repr(list(saver.list(None))).lower()
        for forbidden in (
            sentinel,
            "raw post-checkpoint exception private",
            servers[0].url,
            servers[1].url,
        ):
            assert forbidden.lower() not in compact_error
            assert forbidden.lower() not in checkpoint_blob
        for task_id in state.worker_tasks:
            assert task_id.lower() not in compact_error
    finally:
        for server in servers:
            server.close()


async def test_timeout_server_error_response_mismatch_and_failed_task_semantics():
    slow = _serve_task_handler(
        build_step5_agent_card,
        lambda task: (time.sleep(0.2), _successful_task(task))[1],
    )
    mismatch = _serve_task_handler(
        build_step5_agent_card,
        lambda task: Task(
            id="task_ffffffffffffffff",
            status=TaskStatus(state=TaskState.COMPLETED),
        ),
    )
    failed = _serve_task_handler(
        build_step5_agent_card,
        lambda task: _successful_task(task, state=TaskState.FAILED),
    )
    server_error_app = Flask(__name__)

    @server_error_app.route("/.well-known/agent.json")
    def _card():
        return jsonify(build_step5_agent_card("http://server-error").to_dict())

    @server_error_app.route("/tasks/send", methods=["POST"])
    @server_error_app.route("/a2a/tasks/send", methods=["POST"])
    def _error():
        return jsonify({"error": "test-only server failure"}), 503

    port = _free_port()
    httpd = make_server("127.0.0.1", port, server_error_app, threaded=True)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    server_error = _Handle(f"http://127.0.0.1:{port}", httpd, thread, Counter())
    cases = [
        (slow.url, 0.05, "dispatch_timeout", "dispatch_failed"),
        (server_error.url, 1, "server_error", "dispatch_failed"),
        (mismatch.url, 1, "dispatch_transport_error", "dispatch_failed"),
        (failed.url, 1, "none", "dispatched"),
    ]
    try:
        for url, timeout, reason, status in cases:
            base = _transport_ready_result()
            prepared = base.prepared_tasks[0]
            one_plan = base.plan.model_copy(
                update={
                    "validated_decisions": [base.plan.validated_decisions[0]],
                    "proposed_decisions": [base.plan.proposed_decisions[0]],
                    "ready_task_count": 1,
                    "dependency_edges": [],
                }
            )
            target = replace(
                prepared.dispatch_target,
                dispatch_url=url,
            )
            one = replace(
                base,
                plan=one_plan,
                prepared_tasks=(replace(prepared, dispatch_target=target),),
            )
            state = execution_state_from_routing_result(one)
            graph, _, config = _graph_for(state)
            result = await dispatch_orchestrator_tasks(
                run_id=state.run_id,
                state=state,
                prepared_tasks=one.prepared_tasks,
                discovery=_FrozenTargets(one.prepared_tasks),
                execution_graph=graph,
                checkpoint_config=config,
                timeout_seconds=timeout,
            )
            assert result.receipts[0].agent_failure_reason == reason
            assert result.receipts[0].dispatch_status == status
            if url == failed.url:
                response = next(iter(result.response_tasks.values()))
                assert response.status.state == TaskState.FAILED
    finally:
        time.sleep(0.25)
        slow.close()
        mismatch.close()
        failed.close()
        server_error.close()


async def test_timeout_after_post_is_delivery_uncertain_and_never_auto_retries():
    """Test-only short timeout: the handler may complete after B1 times out."""
    received = threading.Event()
    completed = threading.Event()

    def _slow_after_receive(task):
        received.set()
        time.sleep(0.7)
        completed.set()
        return _successful_task(task)

    server = _serve_task_handler(build_step5_agent_card, _slow_after_receive)
    try:
        one = _single_prepared_result(
            _transport_ready_result(), url=server.url
        )
        state = execution_state_from_routing_result(one)
        graph, _, config = _graph_for(state)
        result = await dispatch_orchestrator_tasks(
            run_id=state.run_id,
            state=state,
            prepared_tasks=one.prepared_tasks,
            discovery=_FrozenTargets(one.prepared_tasks),
            execution_graph=graph,
            checkpoint_config=config,
            timeout_seconds=0.3,
        )
        assert received.is_set()
        assert result.receipts[0].dispatch_status == "dispatch_failed"
        assert result.receipts[0].agent_failure_reason == "dispatch_timeout"
        assert result.response_tasks == {}
        assert server.hits["task"] == 1
        assert await asyncio.to_thread(completed.wait, 2)
        assert server.hits["task"] == 1
    finally:
        server.close()


@pytest.mark.parametrize(
    "mode", ["missing", "extra", "duplicate", "tampered_target"]
)
async def test_batch_or_target_tampering_fails_before_checkpoint_or_network(mode):
    routing = _transport_ready_result()
    state = execution_state_from_routing_result(routing)
    discovery = _FrozenTargets(routing.prepared_tasks)
    graph, saver, config = _graph_for(state)
    network_calls = 0

    class _NeverClient:
        def __init__(self, *args, **kwargs):
            nonlocal network_calls
            network_calls += 1

    prepared = list(routing.prepared_tasks)
    if mode == "missing":
        prepared.pop()
        code = "prepared_task_set_mismatch"
    elif mode == "extra":
        prepared.append(
            replace(
                prepared[0],
                task=Task(
                    id="task_9999999999999999",
                    message=prepared[0].task.message,
                    metadata=prepared[0].task.metadata,
                ),
            )
        )
        code = "prepared_task_set_mismatch"
    elif mode == "duplicate":
        prepared.append(prepared[0])
        code = "prepared_task_identity_duplicate"
    else:
        sentinel = "sk-live-TAMPERED-TARGET-SECRET"
        prepared[0] = replace(
            prepared[0],
            dispatch_target=replace(
                prepared[0].dispatch_target,
                dispatch_url=f"http://tampered.invalid/{sentinel}",
            ),
        )
        code = "dispatch_target_mismatch"

    with pytest.raises(OrchestratorDispatchError, match=f"^{code}$") as caught:
        await dispatch_orchestrator_tasks(
            run_id=state.run_id,
            state=state,
            prepared_tasks=prepared,
            discovery=discovery,
            execution_graph=graph,
            checkpoint_config=config,
            timeout_seconds=1,
            client_factory=_NeverClient,
        )
    assert graph.inputs == []
    assert list(saver.list(None)) == []
    assert network_calls == 0
    if mode == "tampered_target":
        assert sentinel not in str(caught.value)
        assert sentinel not in repr(caught.value)


@pytest.mark.parametrize("tamper", ["metadata_decision", "prepared_agent"])
async def test_task_routing_identity_tampering_fails_before_side_effects(tamper):
    routing = _transport_ready_result()
    state = execution_state_from_routing_result(routing)
    prepared = list(routing.prepared_tasks)
    if tamper == "metadata_decision":
        metadata = dict(prepared[0].task.metadata)
        metadata["routing_decision_id"] = "route_ffffffffffffffff"
        prepared[0] = replace(
            prepared[0],
            task=Task(
                id=prepared[0].task.id,
                message=prepared[0].task.message,
                metadata=metadata,
            ),
        )
        code = "prepared_task_payload_identity_mismatch"
    else:
        prepared[0] = replace(
            prepared[0],
            decision=prepared[0].decision.model_copy(
                update={"agent_id": "tampered_agent"}
            ),
        )
        code = "prepared_task_identity_mismatch"
    graph, saver, config = _graph_for(state)
    with pytest.raises(OrchestratorDispatchError, match=f"^{code}$"):
        await dispatch_orchestrator_tasks(
            run_id=state.run_id,
            state=state,
            prepared_tasks=prepared,
            discovery=_FrozenTargets(routing.prepared_tasks),
            execution_graph=graph,
            checkpoint_config=config,
            timeout_seconds=1,
        )
    assert graph.inputs == []
    assert list(saver.list(None)) == []


@pytest.mark.parametrize(
    "field",
    [
        "objective",
        "reason",
        "priority",
        "expected_outputs",
        "created_by",
        "planned_status",
        "deterministic_gate_status",
    ],
)
async def test_task_builder_business_contract_tampering_is_rejected_pre_network(
    field,
):
    routing = _transport_ready_result()
    state = execution_state_from_routing_result(routing)
    prepared = list(routing.prepared_tasks)
    request = WorkerExecutionRequest.model_validate_json(
        prepared[0].task.message["content"]["text"]
    )
    worker_request = request.worker_request
    routing_ref = request.orchestrator_routing_decision
    update = {}
    if field == "objective":
        worker_request = worker_request.model_copy(update={"objective": "Changed"})
    elif field == "reason":
        worker_request = worker_request.model_copy(update={"reason": "Changed"})
    elif field == "priority":
        worker_request = worker_request.model_copy(update={"priority": "low"})
    elif field == "expected_outputs":
        routing_ref = routing_ref.model_copy(
            update={"expected_outputs": ["unexpected_output"]}
        )
    elif field == "created_by":
        update["created_by"] = "different_orchestrator"
    elif field == "planned_status":
        routing_ref = routing_ref.model_copy(update={"planned_status": "skip"})
    else:
        routing_ref = routing_ref.model_copy(
            update={"deterministic_gate_status": "unchecked"}
        )
    request = request.model_copy(
        update={
            **update,
            "worker_request": worker_request,
            "orchestrator_routing_decision": routing_ref,
        }
    )
    prepared[0] = _replace_request(prepared[0], request)
    graph, saver, config = _graph_for(state)
    client_count = 0

    class _NeverClient:
        def __init__(self, *args, **kwargs):
            nonlocal client_count
            client_count += 1

    with pytest.raises(
        OrchestratorDispatchError,
        match="^prepared_task_payload_contract_mismatch$",
    ):
        await dispatch_orchestrator_tasks(
            run_id=state.run_id,
            state=state,
            prepared_tasks=prepared,
            discovery=_FrozenTargets(routing.prepared_tasks),
            execution_graph=graph,
            checkpoint_config=config,
            timeout_seconds=1,
            client_factory=_NeverClient,
        )
    assert graph.inputs == []
    assert list(saver.list(None)) == []
    assert client_count == 0


@pytest.mark.parametrize(
    "drift",
    [
        "compact_inputs",
        "runtime_refs",
        "artifact_id",
        "field_keys",
        "privacy_constraints",
        "session_id",
        "retry_context",
        "routing_context",
    ],
)
async def test_full_request_contract_drift_is_rejected_with_zero_side_effects(
    drift,
):
    servers = (
        _serve_task_handler(build_step5_agent_card, _successful_task),
        _serve_task_handler(build_step6_agent_card, _successful_task),
    )
    injected = "safe_injected_value"
    try:
        seed = routing_result_fixture(independent_ready=True)
        targets = {
            seed.prepared_tasks[0].decision.agent_id: servers[0].url,
            seed.prepared_tasks[1].decision.agent_id: servers[1].url,
        }
        routing = _transport_ready_result(targets=targets)
        prepared = list(routing.prepared_tasks)
        request = WorkerExecutionRequest.model_validate_json(
            prepared[0].task.message["content"]["text"]
        )
        projection = request.input_projection
        update = {}
        if drift == "compact_inputs":
            projection = projection.model_copy(
                update={"compact_inputs": {"safe_note": injected}}
            )
        elif drift == "runtime_refs":
            projection = projection.model_copy(
                update={
                    "runtime_refs": {
                        "safe_runtime_ref": RuntimeRef(
                            ref=injected,
                            runtime_type="safe_material_ref",
                        )
                    }
                }
            )
        elif drift in {"artifact_id", "field_keys"}:
            refs = dict(projection.input_artifact_refs)
            name = next(iter(refs))
            ref_update = (
                {"artifact_id": f"artifact_{injected}"}
                if drift == "artifact_id"
                else {"field_keys": [injected]}
            )
            refs[name] = refs[name].model_copy(update=ref_update)
            projection = projection.model_copy(update={"input_artifact_refs": refs})
        elif drift == "privacy_constraints":
            update["privacy_constraints"] = request.privacy_constraints.model_copy(
                update={"no_api_keys": False}
            )
        elif drift == "session_id":
            update["session_id"] = injected
        elif drift == "retry_context":
            update["retry_context"] = RetryContext(
                retry_of_task_id="task_aaaaaaaaaaaaaaaa",
                retry_attempt=1,
                max_retry_attempts=3,
                retry_reason="safe_injected_value",
            )
        else:
            update["orchestrator_routing_decision"] = (
                request.orchestrator_routing_decision.model_copy(
                    update={"reason": injected, "routing_phase": "repair"}
                )
            )
        request = request.model_copy(
            update={**update, "input_projection": projection}
        )
        prepared[0] = _replace_request(prepared[0], request)
        graph, saver, config = _graph_for(
            execution_state_from_routing_result(routing)
        )
        state = execution_state_from_routing_result(routing)
        with pytest.raises(
            OrchestratorDispatchError,
            match="^prepared_task_payload_contract_mismatch$",
        ) as caught:
            await dispatch_orchestrator_tasks(
                run_id=state.run_id,
                state=state,
                prepared_tasks=prepared,
                discovery=_FrozenTargets(routing.prepared_tasks),
                execution_graph=graph,
                checkpoint_config=config,
                timeout_seconds=1,
            )
        assert injected not in str(caught.value)
        assert injected not in repr(caught.value)
        assert graph.inputs == []
        assert list(saver.list(None)) == []
        assert all(server.hits["card"] == 0 for server in servers)
        assert all(server.hits["task"] == 0 for server in servers)
    finally:
        for server in servers:
            server.close()


@pytest.mark.parametrize(
    "drift", ["run_id", "artifact_type", "artifact_id", "not_available"]
)
async def test_prepared_artifact_refs_are_reconciled_against_execution_state(drift):
    routing = _transport_ready_result()
    state = execution_state_from_routing_result(routing)
    prepared = list(routing.prepared_tasks)
    refs = dict(prepared[0].input_artifact_refs)
    name = next(iter(refs))
    ref = refs[name]
    if drift == "run_id":
        refs[name] = ref.model_copy(update={"run_id": "run_20260714_deadbeef"})
    elif drift == "artifact_type":
        refs[name] = ref.model_copy(update={"artifact_type": "different_artifact"})
    elif drift == "artifact_id":
        refs[name] = ref.model_copy(
            update={"artifact_id": "safe_artifact_222222222222"}
        )
    else:
        artifact = state.artifacts[name].model_copy(update={"status": "invalid"})
        state = state.model_copy(
            update={"artifacts": {**state.artifacts, name: artifact}}
        )
    if drift != "not_available":
        request = WorkerExecutionRequest.model_validate_json(
            prepared[0].task.message["content"]["text"]
        )
        request = request.model_copy(
            update={
                "input_projection": request.input_projection.model_copy(
                    update={"input_artifact_refs": refs}
                )
            }
        )
        prepared[0] = replace(
            _replace_request(prepared[0], request), input_artifact_refs=refs
        )
    graph, saver, config = _graph_for(state)
    with pytest.raises(
        OrchestratorDispatchError,
        match="^prepared_task_payload_contract_mismatch$",
    ):
        await dispatch_orchestrator_tasks(
            run_id=state.run_id,
            state=state,
            prepared_tasks=prepared,
            discovery=_FrozenTargets(routing.prepared_tasks),
            execution_graph=graph,
            checkpoint_config=config,
            timeout_seconds=1,
        )
    assert graph.inputs == []
    assert list(saver.list(None)) == []


@pytest.mark.parametrize(
    "sentinel",
    [
        "ACDEFGHIKLMNPQRSTVWYACDEFGHIK",
        ">private_fasta\nACDEFGHIKLMNPQRSTVWY",
        ">private_a3m\nACDEFGHIKLMNPQRSTVWY",
        "ATOM  1 PRIVATE PDB BODY",
        "data_private_mmcif",
        "sk-live-PRIVATE-API-KEY",
        "full prompt private body",
        "raw LLM response private body",
        "raw ToolUniverse payload private body",
    ],
)
async def test_raw_task_sentinels_are_rejected_before_http_or_checkpoint(
    sentinel,
):
    servers = (
        _serve_task_handler(build_step5_agent_card, _successful_task),
        _serve_task_handler(build_step6_agent_card, _successful_task),
    )
    try:
        seed = routing_result_fixture(independent_ready=True)
        targets = {
            seed.prepared_tasks[0].decision.agent_id: servers[0].url,
            seed.prepared_tasks[1].decision.agent_id: servers[1].url,
        }
        routing = _transport_ready_result(targets=targets)
        prepared = list(routing.prepared_tasks)
        unsafe_objective = sentinel
        prepared[0] = replace(
            prepared[0],
            decision=prepared[0].decision.model_copy(
                update={"objective": unsafe_objective}
            ),
        )
        request = WorkerExecutionRequest.model_validate_json(
            prepared[0].task.message["content"]["text"]
        ).model_copy(
            update={
                "worker_request": WorkerRequestSpec(
                    objective=unsafe_objective,
                    reason=prepared[0].decision.selection_reason,
                    priority=prepared[0].decision.priority,
                )
            }
        )
        message = Message(
            content=TextContent(text=request.model_dump_json()),
            role=MessageRole.USER,
        )
        prepared[0] = replace(
            prepared[0],
            task=Task(
                id=prepared[0].task.id,
                message=message.to_dict(),
                metadata=prepared[0].task.metadata,
            ),
        )
        routing = replace(routing, prepared_tasks=tuple(prepared))
        state = execution_state_from_routing_result(routing)
        graph, saver, config = _graph_for(state)
        with pytest.raises(
            OrchestratorDispatchError, match="^prepared_task_privacy_invalid$"
        ) as caught:
            await dispatch_orchestrator_tasks(
                run_id=state.run_id,
                state=state,
                prepared_tasks=routing.prepared_tasks,
                discovery=_FrozenTargets(routing.prepared_tasks),
                execution_graph=graph,
                checkpoint_config=config,
                timeout_seconds=2,
            )
        compact_surfaces = " ".join(
            (
                str(caught.value),
                repr(caught.value),
                repr(caught.value.args),
                repr(list(saver.list(None))),
            )
        ).lower()
        assert sentinel.lower() not in compact_surfaces
        assert graph.inputs == []
        assert list(saver.list(None)) == []
        assert servers[0].hits["card"] == servers[1].hits["card"] == 0
        assert servers[0].hits["task"] == servers[1].hits["task"] == 0
    finally:
        for server in servers:
            server.close()


async def test_empty_eligible_and_empty_prepared_is_noop_without_checkpoint():
    routing = _transport_ready_result()
    state = execution_state_from_routing_result(routing)
    for task_id in tuple(state.worker_tasks):
        from app.a2a.orchestrator_execution_state import (
            mark_task_dispatched,
            mark_task_dispatching,
        )

        state = mark_task_dispatched(mark_task_dispatching(state, task_id), task_id)
    graph, saver, config = _graph_for(state)
    result = await dispatch_orchestrator_tasks(
        run_id=state.run_id,
        state=state,
        prepared_tasks=(),
        discovery=SimpleNamespace(),
        execution_graph=graph,
        checkpoint_config=config,
        timeout_seconds=1,
    )
    assert result.state == state
    assert result.receipts == ()
    assert result.response_tasks == {}
    assert graph.inputs == []
    assert list(saver.list(None)) == []


# -- production Step 5 integration ------------------------------------------
def _local_mcp() -> LocalMCPClient:
    """Test-only deterministic local bindings; no live MCP or ToolUniverse."""

    def binding(payload):
        return lambda **_kwargs: payload

    return LocalMCPClient(
        bindings={
            "SAbDab_search_structures": binding({"hits": [{"pdb_id": "1n8z"}]}),
            "ChEMBL_search_molecules": binding(
                {"hits": [{"chembl_id": "CHEMBL1201585"}]}
            ),
            "ChEMBL_search_substructure": binding(
                {"hits": [{"chembl_id": "CHEMBL_linker"}]}
            ),
        }
    )


class _RecordingStep5(Step5A2AWorker):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.agent_runs = 0
        self.threads: list[str] = []

    def execute_request(self, request):
        self.threads.append(threading.current_thread().name)
        return super().execute_request(request)

    def _default_agent_factory(self):
        outer = self
        real = CandidateContextAgent(
            storage=self._storage,
            registry=self._registry,
            workflow_state=self._workflow_state,
            mcp_client=self._mcp_client,
            llm=self._llm,
        )

        class _Agent:
            def run_from_artifacts(self, run_id, **kwargs):
                outer.agent_runs += 1
                return real.run_from_artifacts(run_id, **kwargs)

        return _Agent()


def _setup_real_step5_inputs(storage, registry, workflow_state) -> str:
    run_id = new_run_id()
    record = IntakeService(storage, registry, workflow_state).submit(
        run_id=run_id,
        raw_user_query="Design ADC against HER2 with vc-MMAE payload",
        user_provided_context={
            "target_or_antigen_text": "HER2",
            "candidate_text": "Trastuzumab analog",
            "payload_linker_text": "vc-MMAE",
        },
    )
    supervisor = SupervisorAgent(llm=MockLLMProvider())
    StructuredQueryService(storage, registry, workflow_state, supervisor).parse(
        record.run_id
    )
    InputReadinessService(storage, registry, workflow_state).check(record.run_id)
    return record.run_id


def _step5_prepared_from_registry(run_id, storage, registry, url):
    from app.a2a.orchestrator_routing_service import OrchestratorRoutingService

    proposal = {
        "loop_decision": "dispatch_next_workers",
        "decisions": [
            {
                "agent_id": AGENT_ID_STEP5,
                "capability_id": CAP_STEP5_CANDIDATE_CONTEXT,
                "objective": "Build normalized candidate context.",
                "selection_reason": "Candidate context is required.",
                "priority": "high",
            }
        ],
        "decision_summary": "Build candidate context.",
    }

    class _DeterministicLLM:
        """Test-only routing proposal provider; not a live LLM or fallback."""

        name = "deterministic_test"
        model = "deterministic-test-v1"

        def generate_json(self, prompt, *, schema, system=None):
            return proposal

    discovery = WorkerDiscoveryService(
        expected_workers=[
            ExpectedWorkerEndpoint(
                AGENT_ID_STEP5,
                (CAP_STEP5_CANDIDATE_CONTEXT,),
                url,
            )
        ],
        storage=storage,
        registry=registry,
        discovery_timeout_seconds=2,
        health_timeout_seconds=2,
    )
    service = OrchestratorRoutingService(
        discovery=discovery,
        storage=storage,
        registry=registry,
        llm=_DeterministicLLM(),
    )
    result = service.plan_for_run(run_id)
    return result, discovery


async def test_real_routing_to_dispatch_to_step5_worker_and_compact_artifact_reconcile(
    local_storage, registry_service, workflow_state_service
):
    port = _free_port()
    url = f"http://127.0.0.1:{port}"
    worker = _RecordingStep5(
        url=url,
        storage=local_storage,
        registry=registry_service,
        workflow_state=workflow_state_service,
        mcp_client=_local_mcp(),
        llm=MockLLMProvider(),
    )
    app = create_step5_flask_app(worker)
    http_hits = Counter()

    @app.before_request
    def _count_step5_http():
        from flask import request

        if "agent.json" in request.path:
            http_hits["card"] += 1
        elif request.path == "/health":
            http_hits["health"] += 1
        elif request.path in {"/tasks/send", "/a2a/tasks/send"}:
            http_hits["task"] += 1

    server = make_server("127.0.0.1", port, app, threaded=False)
    thread = threading.Thread(
        target=server.serve_forever, name="dispatch-step5-http", daemon=True
    )
    thread.start()
    try:
        run_id = _setup_real_step5_inputs(
            local_storage, registry_service, workflow_state_service
        )
        routing, discovery = _step5_prepared_from_registry(
            run_id, local_storage, registry_service, url
        )
        state = execution_state_from_routing_result(routing)
        graph, saver, config = _graph_for(state)
        result = await dispatch_orchestrator_tasks(
            run_id=run_id,
            state=state,
            prepared_tasks=routing.prepared_tasks,
            discovery=discovery,
            execution_graph=graph,
            checkpoint_config=config,
            timeout_seconds=5,
        )

        task_id = routing.prepared_tasks[0].decision.task_id
        response = result.response_tasks[task_id]
        assert response.status.state == TaskState.COMPLETED
        compact = json.loads(response.artifacts[0]["parts"][0]["text"])
        persisted = local_storage.read_json(
            local_storage.run_key(run_id, "candidate_context_table.json")
        )
        artifact_id = registry_service.get(
            run_id
        ).active_artifacts.candidate_context_table_id
        ref = compact["output_artifact_refs"]["candidate_context_table"]
        assert ref["artifact_id"] == persisted["artifact_id"] == artifact_id
        assert ref["run_id"] == persisted["run_id"] == run_id
        assert compact["compact_summary"]["candidate_count"] == len(
            persisted["candidate_records"]
        )
        records = persisted["tool_call_records"]
        counts = Counter(
            "skipped"
            if record["run_status"] in {"skipped", "not_run"}
            else record["run_status"]
            if record["run_status"] in {"success", "dependency_unavailable"}
            else "failed"
            for record in records
        )
        expected = {
            "attempted": len(records) - counts["skipped"],
            "success": counts["success"],
            "failed": counts["failed"],
            "dependency_unavailable": counts["dependency_unavailable"],
            "skipped": counts["skipped"],
        }
        assert compact["tool_call_summary"] == expected
        assert compact["skipped_or_failed_tools"] == sorted(
            {
                record["tool_name"]
                for record in records
                if record["run_status"] != "success"
            }
        )
        assert worker.agent_runs == 1
        assert worker.threads == ["dispatch-step5-http"]
        assert http_hits["card"] == 3
        assert http_hits["health"] == 1
        assert http_hits["task"] == 1
        assert result.state.worker_tasks[task_id].dispatch_status == "dispatched"
        assert result.state.worker_tasks[task_id].execution_status == "not_started"
        assert "candidate_records" not in result.state.model_dump_json()
        ingested = await ingest_orchestrator_worker_results(
            run_id=run_id,
            dispatch_result=result,
            discovery=discovery,
            registry=registry_service,
            storage=local_storage,
            execution_graph=graph,
            checkpoint_config=config,
        )
        assert ingested.state.worker_tasks[task_id].execution_status == "completed"
        assert ingested.state.worker_tasks[task_id].result_status == "success"
        assert ingested.state.artifacts["candidate_context_table"].status == (
            "available"
        )
        assert set(ingested.completion_proofs) == {task_id}
        assert ingested.receipts[0].tool_call_summary.model_dump() == expected
        assert ingested.receipts[0].skipped_or_failed_tool_count == len(
            compact["skipped_or_failed_tools"]
        )
        assert "candidate_records" not in ingested.state.model_dump_json()
        assert len(graph.inputs) == 3
        assert list(saver.list(None))
    finally:
        server.shutdown()
        thread.join(timeout=5)


def test_dispatch_result_private_responses_are_readable_but_never_publicly_serialized():
    routing = _transport_ready_result()
    state = execution_state_from_routing_result(routing)
    response = Task(id="task_9999999999999999")

    result = OrchestratorDispatchResult(
        state=state,
        receipts=(),
        response_tasks={response.id: response},
    )
    assert result.response_tasks[response.id] is response
    assert "response_tasks" not in type(result).model_fields
    with pytest.raises(TypeError):
        result.response_tasks["task_8888888888888888"] = response
    with pytest.raises(TypeError) as caught:
        pickle.dumps(result)
    assert str(caught.value) == "orchestrator_dispatch_result_pickle_unsupported"
    assert response.id not in str(caught.value)

    saver = InMemorySaver()
    graph = build_orchestrator_execution_graph(checkpointer=saver)
    graph.invoke(state, config=execution_graph_config(state.run_id))
    public_surfaces = (
        repr(result),
        repr(result.model_dump()),
        result.model_dump_json(),
        repr(dict(result)),
        repr(list(result)),
        repr(list(saver.list(None))),
    )
    for surface in public_surfaces:
        assert "response_tasks" not in surface
        assert response.id not in surface


@pytest.mark.parametrize("value", [0, -1, float("inf"), float("nan")])
async def test_explicit_dispatch_timeout_must_be_finite_and_positive(value):
    routing = _transport_ready_result()
    state = execution_state_from_routing_result(routing)
    graph, saver, config = _graph_for(state)
    with pytest.raises(
        OrchestratorDispatchError, match="^dispatch_timeout_config_invalid$"
    ):
        await dispatch_orchestrator_tasks(
            run_id=state.run_id,
            state=state,
            prepared_tasks=routing.prepared_tasks,
            discovery=_FrozenTargets(routing.prepared_tasks),
            execution_graph=graph,
            checkpoint_config=config,
            timeout_seconds=value,
        )
    assert graph.inputs == []
    assert list(saver.list(None)) == []
