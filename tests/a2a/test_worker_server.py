"""Turn B — generic A2A worker transport (app/a2a/worker_server.py).

These tests validate the GENERIC adapter contract with a minimal fake worker
core. The fake core exists ONLY for adapter-unit validation (result-identity
enforcement, port/URL guards); it is NOT a production worker and NOT a
production workflow smoke. The real Step 5 production path is covered by
``test_step5_worker_a2a.py``.

Test-env isolation (disclosed): the ``_no_proxy`` fixture bypasses the macOS
system proxy for localhost so requests/A2AClient hit the local TCP server. Not a
production change.
"""

from __future__ import annotations

import json
import socket
import threading

import pytest
from werkzeug.serving import make_server

from python_a2a import (
    A2AClient,
    AgentCard,
    AgentSkill,
    Message,
    MessageRole,
    Task,
    TaskState,
    TextContent,
)

from app.a2a.agent_cards import (
    AdcAgentContract,
    AgentCapabilityContract,
    ContractArtifactRef,
)
from app.a2a.contracts import (
    A2ATaskMetadata,
    InputProjection,
    OrchestratorRoutingDecisionRef,
    PrivacyConstraints,
    ToolCallSummary,
    WorkerExecutionRequest,
    WorkerExecutionResult,
    WorkerRequestSpec,
)
from app.a2a.worker_server import (
    A2AWorkerAdapter,
    WorkerPortInUseError,
    WorkerServerConfigurationError,
    assert_advertised_url_matches_port,
    create_worker_flask_app,
    effective_url_port,
    serve_worker_http,
)

_FAKE_AGENT_ID = "fake_worker"
_FAKE_CAPABILITY = "fake_capability"
_UNSET = object()


def _fake_agent_card(
    *,
    agent_id: str = _FAKE_AGENT_ID,
    agent_role: str = "worker",
    capability_id: str = _FAKE_CAPABILITY,
    routable: bool = True,
    status: str = "active",
    url: str = "http://fake-worker:9999",
) -> AgentCard:
    """A complete, minimal, LEGAL worker AgentCard for adapter-unit tests."""
    contract = AdcAgentContract(
        agent_id=agent_id,
        agent_role=agent_role,  # type: ignore[arg-type]
        display_name="Fake Worker",
        description="adapter unit stub worker",
        capabilities=[
            AgentCapabilityContract(
                capability_id=capability_id,
                skill_name="Fake capability",
                capability_summary="adapter unit stub capability",
                output_artifacts=[
                    ContractArtifactRef(artifact_name="fake_output", storage_path="fake_output.json")
                ],
                uses_llm=False,
                uses_mcp=False,
            )
        ],
        dispatch_modes=["python_a2a"],
        routable=routable,
        status=status,  # type: ignore[arg-type]
        uses_llm=False,
        uses_mcp=False,
    )
    return AgentCard(
        name=contract.display_name,
        description=contract.description,
        url=url,
        version="1.0.0",
        capabilities={"adc_agent_contract": contract.model_dump()},
        skills=[
            AgentSkill(
                id=capability_id,
                name="Fake capability",
                description="adapter unit stub capability",
                tags=[],
                examples=[],
            )
        ],
    )


@pytest.fixture(autouse=True)
def _no_proxy(monkeypatch):
    for var in ("NO_PROXY", "no_proxy"):
        monkeypatch.setenv(var, "127.0.0.1,localhost")
    for var in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy"):
        monkeypatch.delenv(var, raising=False)


class _FakeWorkerCore:
    """ADAPTER-UNIT STUB — not a production worker, not a workflow smoke.

    Serves a complete, legal worker AgentCard (by default) and returns whatever
    result override it is given, so the generic adapter's startup validation and
    post-execution contract (result type + identity enforcement) can be exercised.
    """

    agent_id = _FAKE_AGENT_ID
    capability_ids = frozenset({_FAKE_CAPABILITY})

    def __init__(self, *, result_overrides: dict | None = None, card=None, raw_return=_UNSET):
        self._result_overrides = result_overrides or {}
        self._card = card if card is not None else _fake_agent_card()
        self._raw_return = raw_return
        self.execute_count = 0

    @property
    def agent_card(self) -> AgentCard:
        return self._card

    def health(self) -> dict:
        return {"status": "ok", "agent_id": self.agent_id, "capabilities": sorted(self.capability_ids)}

    def execute_request(self, request: WorkerExecutionRequest):
        self.execute_count += 1
        # Deliberately return a non-WorkerExecutionResult when raw_return is set,
        # to exercise the adapter's result-type guard.
        if self._raw_return is not _UNSET:
            return self._raw_return
        base = dict(
            payload_type="worker_execution_result",
            payload_version="v1",
            run_id=request.run_id,
            task_id=request.task_id,
            routing_plan_id=request.routing_plan_id,
            routing_decision_id=request.routing_decision_id,
            agent_id=request.agent_id,
            capability_id=request.capability_id,
            execution_status="completed",
            result_status="success",
            compact_summary={"ok": True},
            tool_call_summary=ToolCallSummary(),
        )
        base.update(self._result_overrides)
        return WorkerExecutionResult(**base)


class _ServerHandle:
    def __init__(self, base_url, server, thread):
        self.base_url = base_url
        self._server = server
        self._thread = thread

    def close(self):
        self._server.shutdown()
        self._thread.join(timeout=5)


def _serve(core) -> _ServerHandle:
    app = create_worker_flask_app(core)
    server = make_server("127.0.0.1", 0, app, threaded=False)
    thread = threading.Thread(target=server.serve_forever, name="fake-worker-http", daemon=True)
    thread.start()
    return _ServerHandle(f"http://127.0.0.1:{server.server_port}", server, thread)


def _request() -> WorkerExecutionRequest:
    return WorkerExecutionRequest(
        payload_type="worker_execution_request",
        payload_version="v1",
        run_id="run_1",
        task_id="task_1",
        routing_plan_id="wrp_1",
        routing_decision_id="route_1",
        agent_id=_FAKE_AGENT_ID,
        capability_id=_FAKE_CAPABILITY,
        created_by="step_04_orchestrator_planner",
        worker_request=WorkerRequestSpec(objective="x"),
        orchestrator_routing_decision=OrchestratorRoutingDecisionRef(
            planned_status="run", dispatch_mode="python_a2a"
        ),
        input_projection=InputProjection(),
        privacy_constraints=PrivacyConstraints(),
    )


def _valid_core_result(**overrides) -> WorkerExecutionResult:
    base = dict(
        payload_type="worker_execution_result",
        payload_version="v1",
        run_id="run_1",
        task_id="task_1",
        routing_plan_id="wrp_1",
        routing_decision_id="route_1",
        agent_id=_FAKE_AGENT_ID,
        capability_id=_FAKE_CAPABILITY,
        execution_status="completed",
        result_status="success",
        compact_summary={"ok": True},
        tool_call_summary=ToolCallSummary(),
    )
    base.update(overrides)
    return WorkerExecutionResult(**base)


def _task(
    request: WorkerExecutionRequest, *, transport_task_id: str | None = None
) -> Task:
    msg = Message(content=TextContent(text=request.model_dump_json()), role=MessageRole.USER)
    meta = A2ATaskMetadata(
        adc_payload_type="worker_execution_request",
        adc_payload_version="v1",
        run_id=request.run_id,
        task_id=request.task_id,
        routing_plan_id=request.routing_plan_id,
        routing_decision_id=request.routing_decision_id,
        agent_id=request.agent_id,
        capability_id=request.capability_id,
        created_by=request.created_by,
    ).model_dump()
    return Task(
        id=transport_task_id or request.task_id,
        message=msg.to_dict(),
        metadata=meta,
    )


def _result(result_task: Task) -> dict:
    return json.loads((result_task.artifacts or [{}])[0]["parts"][0]["text"])


async def _send(base_url, task):
    return await A2AClient(base_url).send_task_async(task)


# ── result identity enforcement ───────────────────────────────────────────────
async def test_matching_identity_result_passes_through():
    handle = _serve(_FakeWorkerCore())
    try:
        result = _result(await _send(handle.base_url, _task(_request())))
    finally:
        handle.close()
    assert result["result_status"] == "success"
    assert result["task_id"] == "task_1"


async def test_transport_task_id_mismatch_rejected_before_core_execution():
    core = _FakeWorkerCore()
    handle = _serve(core)
    request = _request()
    try:
        result_task = await _send(
            handle.base_url,
            _task(request, transport_task_id="wrong_transport_task_id"),
        )
    finally:
        handle.close()
    result = _result(result_task)

    assert result_task.status.state == TaskState.FAILED
    assert result["result_status"] == "validation_failed"
    assert result["execution_status"] == "failed"
    assert result["error_code"] == "task_transport_id_mismatch"
    assert core.execute_count == 0
    # Correlation remains the validated ADC request identity; the wrong outer
    # python-a2a transport id is never copied into the compact result.
    assert result["run_id"] == request.run_id
    assert result["task_id"] == request.task_id
    assert result["routing_plan_id"] == request.routing_plan_id
    assert result["routing_decision_id"] == request.routing_decision_id
    assert result["agent_id"] == request.agent_id
    assert result["capability_id"] == request.capability_id


@pytest.mark.parametrize(
    "override",
    [
        {"run_id": "WRONG_RUN"},
        {"task_id": "WRONG_TASK"},
        {"routing_plan_id": "WRONG_PLAN"},
        {"routing_decision_id": "WRONG_DECISION"},
        {"agent_id": "WRONG_AGENT"},
        {"capability_id": "WRONG_CAP"},
    ],
)
async def test_core_result_identity_mismatch_rejected(override):
    handle = _serve(_FakeWorkerCore(result_overrides=override))
    try:
        result = _result(await _send(handle.base_url, _task(_request())))
    finally:
        handle.close()
    assert result["result_status"] == "tool_failed"
    assert result["execution_status"] == "failed"
    assert result["error_code"] == "worker_result_identity_mismatch"
    # The mis-identified core result is NOT forwarded; the adapter re-stamps the
    # failure with the REQUEST identity.
    assert result["run_id"] == "run_1"
    assert result["task_id"] == "task_1"


# ── core returns a non-WorkerExecutionResult -> compact failure, no HTTP 500 ─
@pytest.mark.parametrize("bad", [{"foo": "bar"}, None, "not-a-result", 123])
async def test_core_result_wrong_type_rejected(bad):
    handle = _serve(_FakeWorkerCore(raw_return=bad))
    try:
        result_task = await _send(handle.base_url, _task(_request()))
    finally:
        handle.close()
    # The A2A response is a valid task carrying a compact failure — never a 500.
    result = _result(result_task)
    assert result["result_status"] == "tool_failed"
    assert result["execution_status"] == "failed"
    assert result["error_code"] == "worker_result_schema_invalid"
    assert result["run_id"] == "run_1"
    assert result["task_id"] == "task_1"


async def test_core_result_model_copy_with_v2_is_revalidated_and_rejected():
    # model_copy deliberately bypasses Pydantic validation. The adapter must
    # still reject this typed-but-invalid result and must never emit v2.
    bad = _valid_core_result().model_copy(update={"payload_version": "v2"})
    handle = _serve(_FakeWorkerCore(raw_return=bad))
    try:
        result_task = await _send(handle.base_url, _task(_request()))
    finally:
        handle.close()
    result = _result(result_task)
    assert result_task.status.state == TaskState.FAILED
    assert result["payload_version"] == "v1"
    assert result["result_status"] == "tool_failed"
    assert result["execution_status"] == "failed"
    assert result["error_code"] == "worker_result_schema_invalid"


async def test_core_result_model_construct_missing_identity_is_compact_failure():
    # model_construct can create an instance without required fields. Revalidation
    # must prevent getattr/AttributeError from escaping as an HTTP 500.
    bad = WorkerExecutionResult.model_construct(
        payload_type="worker_execution_result",
        payload_version="v1",
        execution_status="completed",
        result_status="success",
    )
    handle = _serve(_FakeWorkerCore(raw_return=bad))
    try:
        result_task = await _send(handle.base_url, _task(_request()))
    finally:
        handle.close()
    result = _result(result_task)
    assert result_task.status.state == TaskState.FAILED
    assert result["result_status"] == "tool_failed"
    assert result["execution_status"] == "failed"
    assert result["error_code"] == "worker_result_schema_invalid"
    assert result["run_id"] == "run_1"
    assert result["task_id"] == "task_1"
    assert "AttributeError" not in json.dumps(result)


async def test_core_success_result_with_failed_execution_status_is_rejected():
    # Deliberately bypass the now-strict shared result schema to prove the
    # adapter still revalidates a malicious/buggy core object before emission.
    bad = _valid_core_result().model_copy(update={"execution_status": "failed"})
    handle = _serve(_FakeWorkerCore(raw_return=bad))
    try:
        result = _result(await _send(handle.base_url, _task(_request())))
    finally:
        handle.close()
    assert result["result_status"] == "tool_failed"
    assert result["execution_status"] == "failed"
    assert result["error_code"] == "worker_result_schema_invalid"


# ── startup core/card consistency validation (fail fast at construction) ─────
async def test_valid_core_card_constructs_and_serves():
    # A legal core/card constructs the adapter and serves over real HTTP.
    handle = _serve(_FakeWorkerCore())
    try:
        result = _result(await _send(handle.base_url, _task(_request())))
    finally:
        handle.close()
    assert result["result_status"] == "success"


def test_missing_adc_contract_fails_fast():
    bare = AgentCard(name="x", description="d", url="http://x:1", version="1.0.0")
    with pytest.raises(WorkerServerConfigurationError):
        A2AWorkerAdapter(_FakeWorkerCore(card=bare))


def test_malformed_adc_contract_fails_as_server_configuration_error():
    card = _fake_agent_card()
    card.capabilities["adc_agent_contract"]["unknown_field"] = True
    with pytest.raises(WorkerServerConfigurationError):
        A2AWorkerAdapter(_FakeWorkerCore(card=card))


def test_non_worker_agent_role_fails_fast():
    card = _fake_agent_card(agent_role="orchestrator")
    with pytest.raises(WorkerServerConfigurationError):
        A2AWorkerAdapter(_FakeWorkerCore(card=card))


def test_agent_id_mismatch_fails_fast():
    card = _fake_agent_card(agent_id="a_different_worker")
    with pytest.raises(WorkerServerConfigurationError):
        A2AWorkerAdapter(_FakeWorkerCore(card=card))


def test_capability_ids_mismatch_fails_fast():
    card = _fake_agent_card(capability_id="a_different_capability")
    with pytest.raises(WorkerServerConfigurationError):
        A2AWorkerAdapter(_FakeWorkerCore(card=card))


@pytest.mark.parametrize("status", ["disabled", "planned"])
def test_non_active_status_fails_fast(status):
    card = _fake_agent_card(status=status)
    with pytest.raises(WorkerServerConfigurationError):
        A2AWorkerAdapter(_FakeWorkerCore(card=card))


def test_not_routable_fails_fast():
    card = _fake_agent_card(routable=False)
    with pytest.raises(WorkerServerConfigurationError):
        A2AWorkerAdapter(_FakeWorkerCore(card=card))


@pytest.mark.parametrize(
    "url",
    [
        "step5-worker",
        "ftp://step5-worker:8005",
        "http://:8005",
        "http://step5-worker:not-a-port",
    ],
)
def test_non_http_or_undeterminable_agent_card_url_fails_fast(url):
    card = _fake_agent_card(url=url)
    with pytest.raises(WorkerServerConfigurationError):
        A2AWorkerAdapter(_FakeWorkerCore(card=card))


# ── effective URL port + advertised-URL/port guard ───────────────────────────
def test_effective_url_port():
    assert effective_url_port("http://step5-worker:8005") == 8005
    assert effective_url_port("http://step5-worker") == 80
    assert effective_url_port("https://step5-worker") == 443
    assert effective_url_port("https://step5-worker:8443") == 8443
    assert effective_url_port("step5-worker") is None  # no scheme -> undeterminable
    assert effective_url_port("ftp://step5-worker:8005") is None
    assert effective_url_port("http://:8005") is None
    assert effective_url_port("http://step5-worker:not-a-port") is None


def test_assert_advertised_url_matches_port_ok():
    # Docker internal URL with explicit port matching the bind port.
    assert_advertised_url_matches_port("http://step5-worker:8005", 8005, agent_id="step5")


def test_assert_advertised_url_matches_port_default_http_mismatch():
    # No explicit port -> effective 80, which does not match bind 9000.
    with pytest.raises(ValueError):
        assert_advertised_url_matches_port("http://step5-worker", 9000, agent_id="step5")


def test_assert_advertised_url_matches_port_default_https():
    assert_advertised_url_matches_port("https://step5-worker", 443, agent_id="step5")
    with pytest.raises(ValueError):
        assert_advertised_url_matches_port("https://step5-worker", 8443, agent_id="step5")


@pytest.mark.parametrize(
    "url",
    [
        "step5-worker",
        "ftp://step5-worker:8005",
        "http://step5-worker:not-a-port",
    ],
)
def test_assert_advertised_url_rejects_undeterminable_http_port(url):
    with pytest.raises(ValueError):
        assert_advertised_url_matches_port(url, 8005, agent_id="step5")


def test_serve_worker_http_enforces_advertised_url_port():
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    probe.bind(("127.0.0.1", 0))
    free_port = probe.getsockname()[1]
    probe.close()
    card = _fake_agent_card(url="http://fake-worker:1")
    with pytest.raises(ValueError):
        serve_worker_http(
            _FakeWorkerCore(card=card), host="127.0.0.1", port=free_port
        )


def test_serve_worker_http_accepts_matching_advertised_port(monkeypatch):
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    probe.bind(("127.0.0.1", 0))
    free_port = probe.getsockname()[1]
    probe.close()
    card = _fake_agent_card(url=f"http://fake-worker:{free_port}")
    served = []

    class _NonBlockingServer:
        def serve_forever(self):
            served.append(True)

    def _make_server(host, port, app):  # noqa: ARG001
        return _NonBlockingServer()

    monkeypatch.setattr("werkzeug.serving.make_server", _make_server)
    serve_worker_http(
        _FakeWorkerCore(card=card), host="127.0.0.1", port=free_port
    )
    assert served == [True]


# ── occupied port -> WorkerPortInUseError (not SystemExit) ───────────────────
def test_serve_worker_http_fails_fast_on_occupied_port():
    occupant = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    occupant.bind(("127.0.0.1", 0))
    occupant.listen(1)
    occupied_port = occupant.getsockname()[1]
    try:
        with pytest.raises(WorkerPortInUseError):
            serve_worker_http(_FakeWorkerCore(), host="127.0.0.1", port=occupied_port)
        # Explicitly confirm no SystemExit leaked (pytest.raises above already
        # excludes it since SystemExit is not a WorkerPortInUseError).
        raised_system_exit = False
        try:
            serve_worker_http(_FakeWorkerCore(), host="127.0.0.1", port=occupied_port)
        except WorkerPortInUseError:
            pass
        except SystemExit:  # pragma: no cover - would be a regression
            raised_system_exit = True
        assert raised_system_exit is False
    finally:
        occupant.close()
