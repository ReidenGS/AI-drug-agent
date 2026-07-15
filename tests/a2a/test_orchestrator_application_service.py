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


def _application(storage, registry, routing, discovery, *, runtime=None):
    return OrchestratorApplicationService(
        checkpoint_runtime=runtime or _Runtime(),
        routing_service=routing,
        discovery=discovery,
        registry=registry,
        storage=storage,
        worker_timeout_seconds=0.3,
        max_worker_retries=3,
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
async def test_step3_needs_user_input_checkpoints_without_llm_or_http(
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

    result = await _application(
        local_storage, registry_service, routing, discovery
    ).execute(RUN_ID)

    assert result.outcome == "waiting_for_input"
    assert result.run_status == "waiting_for_input"
    assert result.action_code == "provide_required_input"
    assert result.task_counts.total == 0
    assert result.dispatch_attempt_count == 0
    assert llm.call_count == 0


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
