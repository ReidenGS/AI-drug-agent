"""Turn G application-service terminal and uncertainty behavior.

Local synthetic A2AServers exercise transport/control flow only. They are not
live LLM, MCP, ToolUniverse, or biomedical worker evidence.
"""

from __future__ import annotations

import asyncio
import socket
import threading
from contextlib import asynccontextmanager

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from app.a2a.agent_cards import AGENT_ID_STEP5
from app.a2a.orchestrator_application_service import (
    OrchestratorApplicationService,
    OrchestratorApplicationServiceError,
)
from app.a2a.orchestrator_discovery import DispatchTarget
from app.a2a.orchestrator_routing_service import OrchestratorRoutingService
from app.graph.orchestrator_execution_graph import (
    build_orchestrator_execution_graph,
)
from app.utils.ids import new_artifact_id
from tests.a2a.test_orchestrator_dispatch import _serve_task_handler
from tests.a2a.test_orchestrator_post_ingestion import (
    RUN_ID,
    _contract,
    _DeterministicLLM,
    _FrozenDiscovery,
    _proposal,
    _seed_inputs,
)
from tests.a2a.test_orchestrator_retry_loop import _card, _SyntheticWorker
from tests.a2a.test_orchestrator_routing_service import (
    _DeterministicRoutingLLM,
    _FrozenDiscovery as _StepDiscovery,
    _seed_inputs as _seed_step_inputs,
)


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


class _Runtime:
    def __init__(self):
        self.saver = InMemorySaver()
        self.graph = build_orchestrator_execution_graph(checkpointer=self.saver)

    @asynccontextmanager
    async def run_lock(self, _run_id):
        yield


class _BusyRuntime(_Runtime):
    @asynccontextmanager
    async def run_lock(self, _run_id):
        raise RuntimeError("checkpoint_run_lock_unavailable")
        yield


class _ObservableRunLockRuntime(_Runtime):
    def __init__(self):
        super().__init__()
        self._guard = asyncio.Lock()
        self.entered = asyncio.Event()
        self.exited = asyncio.Event()

    @asynccontextmanager
    async def run_lock(self, _run_id):
        if self._guard.locked():
            raise RuntimeError("checkpoint_run_lock_unavailable")
        await self._guard.acquire()
        self.entered.set()
        try:
            yield
        finally:
            self._guard.release()
            self.exited.set()


class _BlockingRoutingService:
    def __init__(self, delegate):
        self._delegate = delegate
        self.started = threading.Event()
        self.release = threading.Event()
        self.call_count = 0

    def plan_for_run(self, run_id):
        self.call_count += 1
        self.started.set()
        self.release.wait(timeout=5)
        return self._delegate.plan_for_run(run_id)

    def __getattr__(self, name):
        return getattr(self._delegate, name)


def _application(
    storage,
    registry,
    routing,
    discovery,
    *,
    runtime=None,
    client_factory=None,
):
    return OrchestratorApplicationService(
        checkpoint_runtime=runtime or _Runtime(),
        routing_service=routing,
        discovery=discovery,
        registry=registry,
        storage=storage,
        worker_timeout_seconds=0.3,
        max_worker_retries=3,
        client_factory=client_factory,
    )


def _generic_environment(storage, registry, *, failure_status, fail_count):
    _seed_inputs(storage, registry)
    contract = _contract("agent_alpha")
    discovery = _FrozenDiscovery([contract])
    llm = _DeterministicLLM(_proposal("agent_alpha"))
    routing = OrchestratorRoutingService(
        discovery=discovery,
        storage=storage,
        registry=registry,
        llm=llm,
    )
    worker = _SyntheticWorker(
        agent_id="agent_alpha",
        storage=storage,
        registry=registry,
        fail_count=fail_count,
        status=failure_status,
    )
    handle = _serve_task_handler(_card("agent_alpha"), worker.handle)
    discovery.resolve_dispatch_target = lambda run_id, **kwargs: DispatchTarget(
        agent_id=kwargs["agent_id"],
        capability_id=kwargs["capability_id"],
        dispatch_url=handle.url,
        dispatch_mode=kwargs.get("dispatch_mode", "python_a2a"),
    )
    return _application(storage, registry, routing, discovery), llm, worker, handle


@pytest.mark.asyncio
async def test_step3_needs_user_input_stops_before_checkpoint_llm_or_http(
    local_storage, registry_service
):
    _seed_step_inputs(
        local_storage,
        registry_service,
        run_id=RUN_ID,
        readiness="needs_user_input",
    )
    discovery = _StepDiscovery()
    llm = _DeterministicRoutingLLM()
    routing = OrchestratorRoutingService(
        discovery=discovery,
        storage=local_storage,
        registry=registry_service,
        llm=llm,
    )

    runtime = _Runtime()
    with pytest.raises(
        OrchestratorApplicationServiceError,
        match="^input_readiness_not_ready$",
    ):
        await _application(
            local_storage,
            registry_service,
            routing,
            discovery,
            runtime=runtime,
        ).execute(RUN_ID)

    assert llm.call_count == 0
    assert discovery.discover_count == 0
    assert list(runtime.saver.list(None)) == []
    active = registry_service.get(RUN_ID).active_artifacts
    assert active.worker_discovery_snapshot_id is None
    assert active.worker_routing_plan_id is None


@pytest.mark.parametrize(
    ("slot_name", "category"),
    [
        ("prompt_sequence", "structure_or_sequence"),
        ("structure_or_sequence", "structure_or_sequence"),
    ],
)
@pytest.mark.asyncio
async def test_semantically_inconsistent_ready_stops_with_zero_side_effects(
    local_storage, registry_service, slot_name, category
):
    _seed_step_inputs(local_storage, registry_service, run_id=RUN_ID)
    key = local_storage.run_key(
        RUN_ID, "inputs/input_readiness_status.json"
    )
    body = local_storage.read_json(key)
    body.update(
        {
            "input_readiness_status": "ready",
            "missing_input_checklist": [
                {
                    "field": f"structured_query.missing_slots.{slot_name}",
                    "severity": "blocking",
                    "message": "test-only recoverable gap",
                    "category": category,
                    "recoverable": True,
                }
            ],
            "blocking_reasons": [],
            "clarification_requests": [
                {
                    "request_id": f"clr_{slot_name}_test",
                    "slot_name": slot_name,
                    "slot_category": "structure",
                    "severity": "blocking",
                    "question": "Provide the missing input.",
                    "resolved": False,
                }
            ],
        }
    )
    local_storage.write_json(key, body)
    discovery = _StepDiscovery()
    llm = _DeterministicRoutingLLM()
    routing = OrchestratorRoutingService(
        discovery=discovery,
        storage=local_storage,
        registry=registry_service,
        llm=llm,
    )
    runtime = _Runtime()
    client_constructions = 0

    def _never_client(*_args, **_kwargs):
        nonlocal client_constructions
        client_constructions += 1
        raise AssertionError("semantic readiness must stop before A2A client")

    with pytest.raises(
        OrchestratorApplicationServiceError,
        match="^input_readiness_status_semantic_invalid$",
    ):
        await _application(
            local_storage,
            registry_service,
            routing,
            discovery,
            runtime=runtime,
            client_factory=_never_client,
        ).execute(RUN_ID)

    active = registry_service.get(RUN_ID).active_artifacts
    assert llm.call_count == discovery.discover_count == 0
    assert client_constructions == 0
    assert list(runtime.saver.list(None)) == []
    assert active.worker_discovery_snapshot_id is None
    assert active.worker_routing_plan_id is None
    assert active.worker_routing_plan_control_id is None
    assert active.candidate_context_table_id is None
    assert active.structured_liability_summary_id is None


@pytest.mark.asyncio
async def test_stale_ready_source_chain_stops_before_all_step4_side_effects(
    local_storage, registry_service
):
    _seed_step_inputs(local_storage, registry_service, run_id=RUN_ID)
    structured_key = local_storage.run_key(
        RUN_ID, "inputs/structured_query.json"
    )
    replacement = local_storage.read_json(structured_key)
    replacement_id = new_artifact_id("structured_query")
    replacement["artifact_id"] = replacement_id
    local_storage.write_json(structured_key, replacement)
    registry_service.update_active(
        RUN_ID, structured_query_id=replacement_id
    )
    discovery = _StepDiscovery()
    llm = _DeterministicRoutingLLM()
    routing = OrchestratorRoutingService(
        discovery=discovery,
        storage=local_storage,
        registry=registry_service,
        llm=llm,
    )
    runtime = _Runtime()
    client_constructions = 0

    def _never_client(*_args, **_kwargs):
        nonlocal client_constructions
        client_constructions += 1
        raise AssertionError("stale readiness must stop before A2A")

    with pytest.raises(
        OrchestratorApplicationServiceError,
        match="^input_readiness_status_source_mismatch$",
    ):
        await _application(
            local_storage,
            registry_service,
            routing,
            discovery,
            runtime=runtime,
            client_factory=_never_client,
        ).execute(RUN_ID)

    active = registry_service.get(RUN_ID).active_artifacts
    assert llm.call_count == discovery.discover_count == 0
    assert client_constructions == 0
    assert list(runtime.saver.list(None)) == []
    assert active.worker_discovery_snapshot_id is None
    assert active.worker_routing_plan_id is None
    assert active.worker_routing_plan_control_id is None
    assert active.candidate_context_table_id is None


@pytest.mark.asyncio
async def test_competing_backend_run_lock_fails_before_planning_or_checkpoint(
    local_storage, registry_service
):
    _seed_inputs(local_storage, registry_service)
    contract = _contract("agent_alpha")
    discovery = _FrozenDiscovery([contract])
    llm = _DeterministicLLM(_proposal("agent_alpha"))
    routing = OrchestratorRoutingService(
        discovery=discovery,
        storage=local_storage,
        registry=registry_service,
        llm=llm,
    )
    runtime = _BusyRuntime()

    with pytest.raises(
        OrchestratorApplicationServiceError,
        match="^orchestrator_run_busy$",
    ):
        await _application(
            local_storage,
            registry_service,
            routing,
            discovery,
            runtime=runtime,
        ).execute(RUN_ID)

    assert llm.call_count == 0
    assert registry_service.get(
        RUN_ID
    ).active_artifacts.worker_routing_plan_id is None
    assert list(runtime.saver.list(None)) == []


@pytest.mark.asyncio
async def test_cancelled_planning_quiesces_before_local_and_database_lock_release(
    local_storage, registry_service
):
    _seed_inputs(local_storage, registry_service)
    contract = _contract("agent_alpha")
    discovery = _FrozenDiscovery([contract])
    llm = _DeterministicLLM(_proposal("agent_alpha"))
    delegate = OrchestratorRoutingService(
        discovery=discovery,
        storage=local_storage,
        registry=registry_service,
        llm=llm,
    )
    routing = _BlockingRoutingService(delegate)
    runtime = _ObservableRunLockRuntime()
    first = _application(
        local_storage, registry_service, routing, discovery, runtime=runtime
    )
    competing = _application(
        local_storage, registry_service, routing, discovery, runtime=runtime
    )

    execute_task = asyncio.create_task(first.execute(RUN_ID))
    assert await asyncio.to_thread(routing.started.wait, 2)
    await runtime.entered.wait()
    execute_task.cancel()
    await asyncio.sleep(0.02)

    assert not execute_task.done()
    assert first._run_lock(RUN_ID).locked()
    assert runtime._guard.locked()
    assert not runtime.exited.is_set()
    with pytest.raises(
        OrchestratorApplicationServiceError,
        match="^orchestrator_run_busy$",
    ):
        await competing.execute(RUN_ID)
    assert routing.call_count == 1
    assert llm.call_count == 0
    assert list(runtime.saver.list(None)) == []

    routing.release.set()
    with pytest.raises(asyncio.CancelledError):
        await execute_task

    assert runtime.exited.is_set()
    assert not first._run_lock(RUN_ID).locked()
    assert not runtime._guard.locked()
    assert routing.call_count == 1
    assert llm.call_count == 1
    authority = registry_service.get(RUN_ID).active_artifacts
    assert authority.worker_routing_plan_id is not None
    assert authority.worker_routing_plan_control_id is not None
    assert list(runtime.saver.list(None)) == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("failure_status", "fail_count", "expected_posts", "expected_retries"),
    [
        ("validation_failed", 1, 1, 0),
        ("blocked", 1, 1, 0),
        ("tool_failed", 4, 4, 3),
    ],
)
async def test_worker_terminal_failure_maps_to_compact_failed_outcome(
    local_storage,
    registry_service,
    failure_status,
    fail_count,
    expected_posts,
    expected_retries,
):
    app_service, llm, worker, handle = _generic_environment(
        local_storage,
        registry_service,
        failure_status=failure_status,
        fail_count=fail_count,
    )
    try:
        result = await app_service.execute(RUN_ID)
    finally:
        handle.close()

    assert result.outcome == "failed"
    assert result.run_status == "failed"
    assert result.action_code == "inspect_compact_failure"
    assert result.task_counts.failed == expected_posts
    assert result.task_counts.retry_tasks == expected_retries
    assert result.dispatch_attempt_count == expected_posts
    assert handle.hits["task"] == expected_posts
    assert len(worker.requests) == expected_posts
    assert llm.call_count == 1


@pytest.mark.asyncio
async def test_transport_connection_uncertainty_never_auto_retries(
    local_storage, registry_service
):
    _seed_inputs(local_storage, registry_service)
    contract = _contract("agent_alpha")
    discovery = _FrozenDiscovery([contract])
    llm = _DeterministicLLM(_proposal("agent_alpha"))
    routing = OrchestratorRoutingService(
        discovery=discovery,
        storage=local_storage,
        registry=registry_service,
        llm=llm,
    )
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    dead_port = sock.getsockname()[1]
    sock.close()
    discovery.resolve_dispatch_target = lambda run_id, **kwargs: DispatchTarget(
        agent_id=kwargs["agent_id"],
        capability_id=kwargs["capability_id"],
        dispatch_url=f"http://127.0.0.1:{dead_port}",
        dispatch_mode=kwargs.get("dispatch_mode", "python_a2a"),
    )

    result = await _application(
        local_storage, registry_service, routing, discovery
    ).execute(RUN_ID)

    assert result.outcome == "reconciliation_required"
    assert result.action_code == "reconcile_worker_result"
    assert result.dispatch_attempt_count == 1
    assert result.task_counts.total == 1
    assert result.task_counts.dispatch_failed == 1
    assert result.task_counts.retry_tasks == 0
    assert result.next_wakeup.reason == "worker_result_reconciliation_required"


@pytest.mark.asyncio
async def test_discovery_unavailable_worker_produces_no_task_or_dispatch(
    local_storage, registry_service
):
    _seed_step_inputs(local_storage, registry_service, run_id=RUN_ID)
    discovery = _StepDiscovery(unavailable={AGENT_ID_STEP5})
    llm = _DeterministicRoutingLLM(
        {
            "loop_decision": "dispatch_next_workers",
            "decisions": [
                {
                    "agent_id": AGENT_ID_STEP5,
                    "capability_id": "step_05_candidate_context",
                    "objective": "Build candidate context.",
                    "selection_reason": "The requested output requires it.",
                    "priority": "normal",
                }
            ],
            "decision_summary": "Attempt one unavailable worker.",
        }
    )
    routing = OrchestratorRoutingService(
        discovery=discovery,
        storage=local_storage,
        registry=registry_service,
        llm=llm,
    )

    result = await _application(
        local_storage, registry_service, routing, discovery
    ).execute(RUN_ID)

    assert result.outcome == "failed"
    assert result.dispatch_attempt_count == 0
    assert result.task_counts.total == 0
    assert result.decision_counts.total == 0
    assert llm.call_count == 1
