"""Turn G production Step 4 application/lifespan API integration tests.

The synthetic localhost A2AServer is a transport fixture only. It is not a
live LLM, MCP, ToolUniverse, or biomedical-worker success claim.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

import pytest
from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import InMemorySaver

from app.a2a.orchestrator_application_service import (
    OrchestratorApplicationService,
)
from app.a2a.orchestrator_discovery import DispatchTarget
from app.a2a.orchestrator_routing_service import OrchestratorRoutingService
from app.graph.orchestrator_execution_graph import (
    build_orchestrator_execution_graph,
)
from app.graph.orchestrator_checkpoint_runtime import (
    OrchestratorCheckpointRuntimeError,
)
from app.main import create_app
from app.settings import get_settings
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


class _MemoryRuntime:
    def __init__(self):
        self.saver = InMemorySaver()
        self.graph = build_orchestrator_execution_graph(checkpointer=self.saver)
        self.startup_count = 0
        self.shutdown_count = 0

    async def startup(self):
        self.startup_count += 1
        return self

    async def shutdown(self):
        self.shutdown_count += 1

    @asynccontextmanager
    async def run_lock(self, _run_id):
        yield


class _StartupFailureRuntime:
    async def startup(self):
        raise RuntimeError("postgresql://private:secret@unavailable/raw")

    async def shutdown(self):
        raise AssertionError("never started")


def _configure_lifespan(monkeypatch):
    monkeypatch.setenv(
        "LANGGRAPH_CHECKPOINT_DATABASE_URL",
        "postgresql://test_only:private@checkpoint.invalid/adc",
    )
    monkeypatch.setenv("ORCHESTRATOR_WORKER_TIMEOUT_SECONDS", "2")
    get_settings.cache_clear()


def test_missing_dsn_keeps_health_but_step4_fails_without_memory_fallback(
    monkeypatch,
):
    monkeypatch.delenv("LANGGRAPH_CHECKPOINT_DATABASE_URL", raising=False)
    monkeypatch.setenv("ORCHESTRATOR_WORKER_TIMEOUT_SECONDS", "2")
    get_settings.cache_clear()
    runtime_factory_calls = []
    app = create_app(
        checkpoint_runtime_factory=lambda settings: runtime_factory_calls.append(
            settings
        ),
    )

    with TestClient(app) as client:
        assert client.get("/healthz").status_code == 200
        response = client.post(f"/runs/{RUN_ID}/steps/4/execute")
        invalid = client.get(
            "/runs/sk-live-RAW-PATH-SECRET/steps/4/status"
        )

    assert response.status_code == 503
    assert response.json()["error_code"] == (
        "orchestrator_checkpoint_database_url_required"
    )
    assert response.json()["outcome"] == "unavailable"
    assert runtime_factory_calls == []
    assert invalid.status_code == 422
    assert invalid.json()["run_id"] is None
    assert invalid.json()["error_code"] == "orchestrator_request_invalid"
    assert "sk-live" not in invalid.text
    assert "RAW-PATH-SECRET" not in invalid.text


def test_configured_postgres_startup_failure_aborts_lifespan_without_fallback(
    monkeypatch,
):
    _configure_lifespan(monkeypatch)
    app = create_app(
        checkpoint_runtime_factory=lambda _settings: _StartupFailureRuntime()
    )
    with pytest.raises(
        OrchestratorCheckpointRuntimeError,
        match="^checkpoint_runtime_startup_failed$",
    ) as caught:
        with TestClient(app):
            pass
    assert "private" not in repr(caught.value)
    assert "secret" not in str(caught.value)


def test_fresh_concurrent_execute_status_and_resume_are_idempotent_over_http(
    monkeypatch, local_storage, registry_service
):
    _configure_lifespan(monkeypatch)
    _seed_inputs(local_storage, registry_service)
    contract = _contract("agent_alpha")
    discovery = _FrozenDiscovery([contract])
    llm = _DeterministicLLM(_proposal("agent_alpha"))
    routing_service = OrchestratorRoutingService(
        discovery=discovery,
        storage=local_storage,
        registry=registry_service,
        llm=llm,
    )
    worker = _SyntheticWorker(
        agent_id="agent_alpha",
        storage=local_storage,
        registry=registry_service,
    )
    handle = _serve_task_handler(_card("agent_alpha"), worker.handle)

    def resolve(
        run_id, *, agent_id, capability_id, dispatch_mode="python_a2a"
    ):
        assert run_id == RUN_ID
        return DispatchTarget(
            agent_id=agent_id,
            capability_id=capability_id,
            dispatch_url=handle.url,
            dispatch_mode=dispatch_mode,
        )

    discovery.resolve_dispatch_target = resolve
    runtime = _MemoryRuntime()

    def service_factory(started_runtime):
        assert started_runtime is runtime
        return OrchestratorApplicationService(
            checkpoint_runtime=runtime,
            routing_service=routing_service,
            discovery=discovery,
            registry=registry_service,
            storage=local_storage,
            worker_timeout_seconds=2,
            max_worker_retries=3,
        )

    app = create_app(
        checkpoint_runtime_factory=lambda _settings: runtime,
        orchestrator_service_factory=service_factory,
    )
    try:
        with TestClient(app) as client:
            missing_status = client.get(
                f"/runs/{RUN_ID}/steps/4/status"
            )
            missing_resume = client.post(
                f"/runs/{RUN_ID}/steps/4/resume"
            )
            with ThreadPoolExecutor(max_workers=2) as pool:
                responses = list(
                    pool.map(
                        lambda _index: client.post(
                            f"/runs/{RUN_ID}/steps/4/execute"
                        ),
                        range(2),
                    )
                )
            status = client.get(f"/runs/{RUN_ID}/steps/4/status")
            resumed = client.post(f"/runs/{RUN_ID}/steps/4/resume")
    finally:
        handle.close()
        get_settings.cache_clear()

    assert runtime.startup_count == runtime.shutdown_count == 1
    assert missing_status.status_code == missing_resume.status_code == 404
    assert missing_status.json()["error_code"] == (
        "orchestrator_checkpoint_not_found"
    )
    assert missing_resume.json()["error_code"] == (
        "orchestrator_checkpoint_not_found"
    )
    assert all(item.status_code == 200 for item in responses)
    assert {item.json()["outcome"] for item in responses} == {"completed"}
    assert sum(item.json()["llm_routing_called"] for item in responses) == 1
    assert sorted(item.json()["checkpoint_reused"] for item in responses) == [
        False,
        True,
    ]
    assert llm.call_count == 1
    assert handle.hits["task"] == 1
    assert len(worker.requests) == 1
    assert status.status_code == resumed.status_code == 200
    assert status.json()["outcome"] == resumed.json()["outcome"] == "completed"
    assert status.json()["task_counts"]["completed"] == 1
    assert status.json()["artifact_counts"]["available"] == 1
    assert resumed.json()["dispatch_attempt_count"] == 0
    assert set(status.json()) == {
        "run_id",
        "routing_plan_id",
        "outcome",
        "run_status",
        "orchestrator_status",
        "next_wakeup",
        "checkpoint_reused",
        "llm_routing_called",
        "dispatch_attempt_count",
        "decision_counts",
        "task_counts",
        "artifact_counts",
        "artifact_refs",
        "action_code",
        "error_code",
    }
    assert set(status.json()["artifact_refs"][0]) == {
        "artifact_name",
        "status",
        "artifact_id",
        "producer_task_id",
        "safe_summary_ref",
    }

    serialized = repr([item.json() for item in [*responses, status, resumed]])
    for forbidden in (
        "WorkerExecutionRequest",
        "WorkerExecutionResult",
        "PreparedA2ATask",
        "http://127.0.0.1",
        "storage_key",
        "raw_tooluniverse_payload",
        "full_prompt",
        "api_key",
    ):
        assert forbidden not in serialized
