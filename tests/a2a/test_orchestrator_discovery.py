"""Turn D — Orchestrator worker discovery + deterministic target validation.

Every discovery case runs over REAL localhost HTTP: the service always obtains
the AgentCard via ``python_a2a.A2AClient(endpoint).get_agent_card()`` and probes
``/health`` with real ``requests`` calls. AgentCards are never handed to the
service directly.

TEST STUBS (disclosed, test-only — NOT business worker execution smokes):

- ``_agentserver_stub``: a genuine ``python_a2a.A2AServer`` discovery endpoint
  (via ``create_flask_app``) serving a real (optionally mutated) AgentCard, plus
  a controllable ``/health`` route. Used for available / identity / health cases.
- ``_raw_stub``: a bare-Flask stub serving the A2A-standard discovery URL
  (``/.well-known/agent.json``) and ``/health`` with artificial DELAYS, used only
  to exercise network-timeout classification. It serves the real discovery URL
  over real HTTP.
- Connection-refused is simulated by pointing at a closed ephemeral port.

None of these send an A2A task or call any worker business method. The proxy
isolation fixture bypasses the macOS system proxy for localhost (test-env only).
"""

from __future__ import annotations

import json
import socket
import threading
import time
import uuid
from collections import Counter
from types import SimpleNamespace

import pytest
from werkzeug.serving import make_server

from python_a2a import A2AClient, A2AServer
from python_a2a.server.http import create_flask_app

from app.a2a.agent_cards import (
    AGENT_ID_PATENT_EVIDENCE,
    AGENT_ID_STEP5,
    AGENT_ID_STEP6,
    AGENT_ID_STRUCTURE,
    CAP_PATENT_EVIDENCE_WORKFLOW,
    CAP_STEP5_CANDIDATE_CONTEXT,
    CAP_STEP6_DEVELOPABILITY,
    CAP_STRUCTURE_DESIGN_WORKFLOW,
    build_patent_evidence_agent_card,
    build_step5_agent_card,
    build_step6_agent_card,
    build_structure_agent_card,
)
from app.a2a.orchestrator_discovery import (
    MAX_DISCOVERY_ERROR_LEN,
    DispatchTargetValidationError,
    ExpectedWorkerEndpoint,
    WorkerDiscoveryService,
    WorkerUnavailableError,
    default_expected_workers,
)

_CONTRACT_KEY = "adc_agent_contract"


@pytest.fixture(autouse=True)
def _no_proxy(monkeypatch):
    for var in ("NO_PROXY", "no_proxy"):
        monkeypatch.setenv(var, "127.0.0.1,localhost")
    for var in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy"):
        monkeypatch.delenv(var, raising=False)


# ── stub infrastructure ──────────────────────────────────────────────────────
def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


class _StubHandle:
    def __init__(self, url, httpd, hits: Counter):
        self.url = url
        self.httpd = httpd
        self.hits = hits

    def close(self):
        self.httpd.shutdown()


def _start(app, port) -> tuple:
    httpd = make_server("127.0.0.1", port, app, threaded=True)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    return httpd, thread


def _agentserver_stub(
    *,
    base_builder=build_step5_agent_card,
    mutate=None,
    health_body="auto",
    health_status=200,
    health_delay=0.0,
) -> _StubHandle:
    """Real python_a2a A2AServer discovery endpoint + controllable /health."""
    from flask import Response, request

    port = _free_port()
    url = f"http://127.0.0.1:{port}"
    card = base_builder(url)
    if mutate is not None:
        mutate(card)  # mutate card.url or card.capabilities['adc_agent_contract']

    server = A2AServer(agent_card=card, google_a2a_compatible=False)
    app = create_flask_app(server)
    hits: Counter = Counter()

    @app.before_request
    def _count():
        if "agent.json" in request.path:
            hits["card"] += 1
        if request.path == "/health":
            hits["health"] += 1

    @app.route("/health")
    def _health():
        if health_delay:
            time.sleep(health_delay)
        if health_body == "auto":
            contract = card.capabilities[_CONTRACT_KEY]
            body = {
                "status": "ok",
                "agent_id": contract["agent_id"],
                "capabilities": [c["capability_id"] for c in contract["capabilities"]],
            }
        else:
            body = health_body
        return Response(json.dumps(body), status=health_status, mimetype="application/json")

    httpd, _ = _start(app, port)
    return _StubHandle(url, httpd, hits)


def _raw_stub(*, card_delay=0.0, card_body_builder=build_step5_agent_card, mutate=None) -> _StubHandle:
    """Bare-Flask discovery stub serving the A2A-standard discovery URL VERBATIM
    (no host rewrite), with an optional DELAY. Test-only: used to exercise
    discovery-timeout classification and a served card whose ``url`` is not
    rewritten to the serving host (URL-mismatch case)."""
    from flask import Flask, Response, jsonify, request

    port = _free_port()
    url = f"http://127.0.0.1:{port}"
    card = card_body_builder(url)
    if mutate is not None:
        mutate(card)
    app = Flask(__name__)
    hits: Counter = Counter()

    @app.before_request
    def _count():
        if "agent.json" in request.path:
            hits["card"] += 1
        if request.path == "/health":
            hits["health"] += 1

    @app.route("/.well-known/agent.json")
    def _card():
        if card_delay:
            time.sleep(card_delay)
        return jsonify(card.to_dict())

    # The other two A2A card endpoints A2AClient tries return 404 quickly so the
    # client falls back fast after the single timed-out standard endpoint.
    @app.route("/agent.json")
    @app.route("/a2a/agent.json")
    def _not_here():
        return Response("no", status=404)

    @app.route("/health")
    def _health():
        contract = card.capabilities[_CONTRACT_KEY]
        return jsonify(
            {
                "status": "ok",
                "agent_id": contract["agent_id"],
                "capabilities": [c["capability_id"] for c in contract["capabilities"]],
            }
        )

    httpd, _ = _start(app, port)
    return _StubHandle(url, httpd, hits)


# ── service helpers ──────────────────────────────────────────────────────────
def _run(registry_service) -> str:
    run_id = f"run_disc_{uuid.uuid4().hex[:12]}"
    registry_service.init_registry(run_id)
    return run_id


def _service(expected, local_storage, registry_service, *, discovery_timeout=3.0, health_timeout=3.0):
    return WorkerDiscoveryService(
        expected_workers=expected,
        storage=local_storage,
        registry=registry_service,
        discovery_timeout_seconds=discovery_timeout,
        health_timeout_seconds=health_timeout,
    )


def _one_step5(stub_url) -> list:
    return [ExpectedWorkerEndpoint(AGENT_ID_STEP5, (CAP_STEP5_CANDIDATE_CONTEXT,), stub_url)]


def _status_by_agent(snapshot):
    return {w.agent_id: w for w in snapshot.worker_statuses}


def test_default_expected_workers_are_four_in_stable_order():
    settings = SimpleNamespace(
        step5_worker_url="http://step5:8005",
        step6_worker_url="http://step6:8006",
        structure_worker_url="http://structure:8009",
        patent_evidence_worker_url="http://patent:8014",
    )
    expected = default_expected_workers(settings)
    assert [item.agent_id for item in expected] == [
        AGENT_ID_STEP5,
        AGENT_ID_STEP6,
        AGENT_ID_STRUCTURE,
        AGENT_ID_PATENT_EVIDENCE,
    ]
    assert [item.capability_ids for item in expected] == [
        (CAP_STEP5_CANDIDATE_CONTEXT,),
        (CAP_STEP6_DEVELOPABILITY,),
        (CAP_STRUCTURE_DESIGN_WORKFLOW,),
        (CAP_PATENT_EVIDENCE_WORKFLOW,),
    ]


def test_all_four_real_http_discovery_and_frozen_cache_counts(
    local_storage, registry_service
):
    stubs = [
        _agentserver_stub(base_builder=build_step5_agent_card),
        _agentserver_stub(base_builder=build_step6_agent_card),
        _agentserver_stub(base_builder=build_structure_agent_card),
        _agentserver_stub(base_builder=build_patent_evidence_agent_card),
    ]
    try:
        expected = [
            ExpectedWorkerEndpoint(
                AGENT_ID_STEP5, (CAP_STEP5_CANDIDATE_CONTEXT,), stubs[0].url
            ),
            ExpectedWorkerEndpoint(
                AGENT_ID_STEP6, (CAP_STEP6_DEVELOPABILITY,), stubs[1].url
            ),
            ExpectedWorkerEndpoint(
                AGENT_ID_STRUCTURE, (CAP_STRUCTURE_DESIGN_WORKFLOW,), stubs[2].url
            ),
            ExpectedWorkerEndpoint(
                AGENT_ID_PATENT_EVIDENCE,
                (CAP_PATENT_EVIDENCE_WORKFLOW,),
                stubs[3].url,
            ),
        ]
        run_id = _run(registry_service)
        service = _service(expected, local_storage, registry_service)
        first = service.discover_for_run(run_id)
        first_counts = [dict(stub.hits) for stub in stubs]
        second = service.discover_for_run(run_id)
        assert first == second
        assert first.available_agent_ids == [item.agent_id for item in expected]
        assert first_counts == [{"card": 2, "health": 1}] * 4
        assert [dict(stub.hits) for stub in stubs] == first_counts
    finally:
        for stub in stubs:
            stub.close()


# ══════════════════════════════════════════════════════════════════════════════
# 1. all three workers available
# ══════════════════════════════════════════════════════════════════════════════
def test_all_three_available(local_storage, registry_service):
    s5 = _agentserver_stub(base_builder=build_step5_agent_card)
    s6 = _agentserver_stub(base_builder=build_step6_agent_card)
    ss = _agentserver_stub(base_builder=build_structure_agent_card)
    try:
        expected = [
            ExpectedWorkerEndpoint(AGENT_ID_STEP5, (CAP_STEP5_CANDIDATE_CONTEXT,), s5.url),
            ExpectedWorkerEndpoint(AGENT_ID_STEP6, (CAP_STEP6_DEVELOPABILITY,), s6.url),
            ExpectedWorkerEndpoint(AGENT_ID_STRUCTURE, (CAP_STRUCTURE_DESIGN_WORKFLOW,), ss.url),
        ]
        run_id = _run(registry_service)
        snap = _service(expected, local_storage, registry_service).discover_for_run(run_id)
        assert snap.discovery_status == "all_available"
        assert snap.available_agent_ids == [AGENT_ID_STEP5, AGENT_ID_STEP6, AGENT_ID_STRUCTURE]
        assert snap.unavailable_agent_ids == []
        for st in snap.worker_statuses:
            assert st.availability == "available"
            assert st.agent_failure_reason == "none"
            assert st.discovery_error is None
    finally:
        s5.close()
        s6.close()
        ss.close()


# ══════════════════════════════════════════════════════════════════════════════
# 2. one connection-unavailable, other two available
# ══════════════════════════════════════════════════════════════════════════════
def test_one_unavailable_others_available(local_storage, registry_service):
    s5 = _agentserver_stub(base_builder=build_step5_agent_card)
    ss = _agentserver_stub(base_builder=build_structure_agent_card)
    dead_port = _free_port()  # nothing listens here
    try:
        expected = [
            ExpectedWorkerEndpoint(AGENT_ID_STEP5, (CAP_STEP5_CANDIDATE_CONTEXT,), s5.url),
            ExpectedWorkerEndpoint(
                AGENT_ID_STEP6, (CAP_STEP6_DEVELOPABILITY,), f"http://127.0.0.1:{dead_port}"
            ),
            ExpectedWorkerEndpoint(AGENT_ID_STRUCTURE, (CAP_STRUCTURE_DESIGN_WORKFLOW,), ss.url),
        ]
        run_id = _run(registry_service)
        snap = _service(expected, local_storage, registry_service).discover_for_run(run_id)
        assert snap.discovery_status == "partially_available"
        assert snap.available_agent_ids == [AGENT_ID_STEP5, AGENT_ID_STRUCTURE]
        assert snap.unavailable_agent_ids == [AGENT_ID_STEP6]
        step6 = _status_by_agent(snap)[AGENT_ID_STEP6]
        assert step6.availability == "unavailable"
        assert step6.agent_failure_reason == "discovery_connection_failed"
    finally:
        s5.close()
        ss.close()


# ══════════════════════════════════════════════════════════════════════════════
# 3. discovery timeout
# ══════════════════════════════════════════════════════════════════════════════
def test_discovery_timeout(local_storage, registry_service):
    stub = _raw_stub(card_delay=3.0)
    try:
        run_id = _run(registry_service)
        svc = _service(
            _one_step5(stub.url), local_storage, registry_service,
            discovery_timeout=1.0, health_timeout=1.0,
        )
        snap = svc.discover_for_run(run_id)
        st = _status_by_agent(snap)[AGENT_ID_STEP5]
        assert st.availability == "unavailable"
        assert st.agent_failure_reason == "discovery_timeout"
        assert st.discovery_error == "discovery_timeout"
    finally:
        stub.close()


# ══════════════════════════════════════════════════════════════════════════════
# 4. malformed adc_agent_contract
# ══════════════════════════════════════════════════════════════════════════════
def test_malformed_adc_agent_contract(local_storage, registry_service):
    def _break(card):
        # Remove a required contract field -> parse_adc_agent_contract fails.
        card.capabilities[_CONTRACT_KEY].pop("agent_role", None)

    stub = _agentserver_stub(mutate=_break)
    try:
        run_id = _run(registry_service)
        snap = _service(_one_step5(stub.url), local_storage, registry_service).discover_for_run(run_id)
        st = _status_by_agent(snap)[AGENT_ID_STEP5]
        assert st.availability == "unavailable"
        assert st.agent_failure_reason == "card_invalid"
        assert st.discovery_error == "adc_agent_contract_invalid"
    finally:
        stub.close()


# ══════════════════════════════════════════════════════════════════════════════
# 5. wrong agent_id
# ══════════════════════════════════════════════════════════════════════════════
def test_wrong_agent_id(local_storage, registry_service):
    def _wrong(card):
        card.capabilities[_CONTRACT_KEY]["agent_id"] = "some_other_agent"

    stub = _agentserver_stub(mutate=_wrong)
    try:
        run_id = _run(registry_service)
        snap = _service(_one_step5(stub.url), local_storage, registry_service).discover_for_run(run_id)
        st = _status_by_agent(snap)[AGENT_ID_STEP5]
        assert st.agent_failure_reason == "card_invalid"
        assert st.discovery_error == "agent_id_mismatch"
    finally:
        stub.close()


# ══════════════════════════════════════════════════════════════════════════════
# 6. wrong capability ids
# ══════════════════════════════════════════════════════════════════════════════
def test_wrong_capability_ids(local_storage, registry_service):
    # Expect a capability the served card does not publish.
    stub = _agentserver_stub(base_builder=build_step5_agent_card)
    try:
        expected = [ExpectedWorkerEndpoint(AGENT_ID_STEP5, ("some_other_capability",), stub.url)]
        run_id = _run(registry_service)
        snap = _service(expected, local_storage, registry_service).discover_for_run(run_id)
        st = _status_by_agent(snap)[AGENT_ID_STEP5]
        assert st.agent_failure_reason == "card_invalid"
        assert st.discovery_error == "capability_ids_mismatch"
    finally:
        stub.close()


# ══════════════════════════════════════════════════════════════════════════════
# 7. card status planned / disabled
# ══════════════════════════════════════════════════════════════════════════════
@pytest.mark.parametrize("bad_status", ["planned", "disabled"])
def test_card_status_not_active(local_storage, registry_service, bad_status):
    def _status(card):
        card.capabilities[_CONTRACT_KEY]["status"] = bad_status

    stub = _agentserver_stub(mutate=_status)
    try:
        run_id = _run(registry_service)
        snap = _service(_one_step5(stub.url), local_storage, registry_service).discover_for_run(run_id)
        st = _status_by_agent(snap)[AGENT_ID_STEP5]
        assert st.agent_failure_reason == "card_invalid"
        assert st.discovery_error == "status_not_active"
    finally:
        stub.close()


# ══════════════════════════════════════════════════════════════════════════════
# 8. routable=false
# ══════════════════════════════════════════════════════════════════════════════
def test_card_not_routable(local_storage, registry_service):
    def _unroutable(card):
        card.capabilities[_CONTRACT_KEY]["routable"] = False

    stub = _agentserver_stub(mutate=_unroutable)
    try:
        run_id = _run(registry_service)
        snap = _service(_one_step5(stub.url), local_storage, registry_service).discover_for_run(run_id)
        st = _status_by_agent(snap)[AGENT_ID_STEP5]
        assert st.agent_failure_reason == "card_invalid"
        assert st.discovery_error == "not_routable"
    finally:
        stub.close()


# ══════════════════════════════════════════════════════════════════════════════
# 9. dispatch mode not python_a2a
# ══════════════════════════════════════════════════════════════════════════════
def test_dispatch_mode_not_python_a2a(local_storage, registry_service):
    def _mode(card):
        card.capabilities[_CONTRACT_KEY]["dispatch_modes"] = ["http_a2a"]

    stub = _agentserver_stub(mutate=_mode)
    try:
        run_id = _run(registry_service)
        snap = _service(_one_step5(stub.url), local_storage, registry_service).discover_for_run(run_id)
        st = _status_by_agent(snap)[AGENT_ID_STEP5]
        # parse_adc_agent_contract enforces dispatch_modes == ['python_a2a'].
        assert st.agent_failure_reason == "card_invalid"
        assert st.discovery_error == "adc_agent_contract_invalid"
    finally:
        stub.close()


# ══════════════════════════════════════════════════════════════════════════════
# 10. AgentCard.url does not match configured endpoint
# ══════════════════════════════════════════════════════════════════════════════
def test_card_url_mismatch(local_storage, registry_service):
    # Use a verbatim raw stub: A2AServer rewrites the served url to the request
    # host, which would mask a mismatch, so this case needs an un-rewritten card.
    def _redirect(card):
        card.url = "http://evil-host:9999"

    stub = _raw_stub(mutate=_redirect)
    try:
        run_id = _run(registry_service)
        snap = _service(_one_step5(stub.url), local_storage, registry_service).discover_for_run(run_id)
        st = _status_by_agent(snap)[AGENT_ID_STEP5]
        assert st.agent_failure_reason == "card_invalid"
        assert st.discovery_error == "card_url_mismatch"
    finally:
        stub.close()


# ══════════════════════════════════════════════════════════════════════════════
# 11-14. health failures
# ══════════════════════════════════════════════════════════════════════════════
def test_health_non_200(local_storage, registry_service):
    stub = _agentserver_stub(health_status=503)
    try:
        run_id = _run(registry_service)
        snap = _service(_one_step5(stub.url), local_storage, registry_service).discover_for_run(run_id)
        st = _status_by_agent(snap)[AGENT_ID_STEP5]
        assert st.agent_failure_reason == "health_failed"
        assert st.discovery_error == "health_non_200"
    finally:
        stub.close()


def test_health_status_not_ok(local_storage, registry_service):
    stub = _agentserver_stub(
        health_body={"status": "degraded", "agent_id": AGENT_ID_STEP5, "capabilities": [CAP_STEP5_CANDIDATE_CONTEXT]}
    )
    try:
        run_id = _run(registry_service)
        snap = _service(_one_step5(stub.url), local_storage, registry_service).discover_for_run(run_id)
        st = _status_by_agent(snap)[AGENT_ID_STEP5]
        assert st.agent_failure_reason == "health_failed"
        assert st.discovery_error == "health_status_not_ok"
    finally:
        stub.close()


def test_health_agent_id_mismatch(local_storage, registry_service):
    stub = _agentserver_stub(
        health_body={"status": "ok", "agent_id": "wrong_agent", "capabilities": [CAP_STEP5_CANDIDATE_CONTEXT]}
    )
    try:
        run_id = _run(registry_service)
        snap = _service(_one_step5(stub.url), local_storage, registry_service).discover_for_run(run_id)
        st = _status_by_agent(snap)[AGENT_ID_STEP5]
        assert st.agent_failure_reason == "health_failed"
        assert st.discovery_error == "health_agent_id_mismatch"
    finally:
        stub.close()


def test_health_capabilities_mismatch(local_storage, registry_service):
    stub = _agentserver_stub(
        health_body={"status": "ok", "agent_id": AGENT_ID_STEP5, "capabilities": ["something_else"]}
    )
    try:
        run_id = _run(registry_service)
        snap = _service(_one_step5(stub.url), local_storage, registry_service).discover_for_run(run_id)
        st = _status_by_agent(snap)[AGENT_ID_STEP5]
        assert st.agent_failure_reason == "health_failed"
        assert st.discovery_error == "health_capabilities_mismatch"
    finally:
        stub.close()


# ══════════════════════════════════════════════════════════════════════════════
# 15. compact catalog fixed order
# ══════════════════════════════════════════════════════════════════════════════
def test_compact_catalog_fixed_order(local_storage, registry_service):
    s5 = _agentserver_stub(base_builder=build_step5_agent_card)
    s6 = _agentserver_stub(base_builder=build_step6_agent_card)
    ss = _agentserver_stub(base_builder=build_structure_agent_card)
    try:
        # Provide expected in a shuffled order; snapshot must still be Step5/6/Structure.
        expected = [
            ExpectedWorkerEndpoint(AGENT_ID_STEP5, (CAP_STEP5_CANDIDATE_CONTEXT,), s5.url),
            ExpectedWorkerEndpoint(AGENT_ID_STEP6, (CAP_STEP6_DEVELOPABILITY,), s6.url),
            ExpectedWorkerEndpoint(AGENT_ID_STRUCTURE, (CAP_STRUCTURE_DESIGN_WORKFLOW,), ss.url),
        ]
        run_id = _run(registry_service)
        catalog = _service(expected, local_storage, registry_service).discover_for_run(run_id).compact_card_catalog
        assert [c["agent_id"] for c in catalog] == [
            AGENT_ID_STEP5, AGENT_ID_STEP6, AGENT_ID_STRUCTURE
        ]
    finally:
        s5.close()
        s6.close()
        ss.close()


# ══════════════════════════════════════════════════════════════════════════════
# 16 & 17. privacy: compact catalog + persisted snapshot carry no secrets;
#          full card cache lives only in memory.
# ══════════════════════════════════════════════════════════════════════════════
def test_snapshot_and_catalog_have_no_secrets_and_no_full_card(local_storage, registry_service):
    s5 = _agentserver_stub(base_builder=build_step5_agent_card)
    try:
        run_id = _run(registry_service)
        svc = _service(_one_step5(s5.url), local_storage, registry_service)
        svc.discover_for_run(run_id)

        persisted = local_storage.read_json(
            local_storage.run_key(run_id, "inputs/worker_discovery_snapshot.json")
        )
        blob = json.dumps(persisted).lower()

        # No endpoint / host / port / scheme.
        host_port = s5.url.split("//", 1)[1]
        for needle in ("http://", "https://", "127.0.0.1", host_port, host_port.split(":")[1]):
            assert needle not in blob, f"persisted snapshot leaked endpoint token: {needle}"
        # No auth / raw payload / raw biological data / prompt / llm response.
        for needle in (
            "authorization", "api_key", "apikey", "bearer", "raw_sequence", "fasta",
            "pdb", "cif", "a3m", "tooluniverse", "full_prompt", "raw_llm_response",
        ):
            assert needle not in blob, f"persisted snapshot leaked forbidden token: {needle}"
        # Full AgentCard internals are NOT persisted.
        assert "adc_agent_contract" not in blob
        assert '"skills"' not in blob
        assert "well-known" not in blob

        # But the full card cache IS available in memory for the validator.
        cache = svc.get_full_card_cache(run_id)
        worker = cache.workers[AGENT_ID_STEP5]
        assert worker.agent_card is not None
        assert worker.contract is not None
        assert worker.dispatch_url == s5.url
    finally:
        s5.close()


# ══════════════════════════════════════════════════════════════════════════════
# 18. same run second discover: no new HTTP (card/health counters unchanged)
# ══════════════════════════════════════════════════════════════════════════════
def test_second_discover_same_run_no_new_http(local_storage, registry_service):
    s5 = _agentserver_stub(base_builder=build_step5_agent_card)
    try:
        run_id = _run(registry_service)
        svc = _service(_one_step5(s5.url), local_storage, registry_service)
        snap1 = svc.discover_for_run(run_id)
        card_hits = s5.hits["card"]
        health_hits = s5.hits["health"]
        assert card_hits >= 1 and health_hits >= 1

        snap2 = svc.discover_for_run(run_id)
        # No additional card/health HTTP on the cached path.
        assert s5.hits["card"] == card_hits
        assert s5.hits["health"] == health_hits
        # "Frozen" == same CONTENT, not the same Python object. Each call returns
        # an independent defensive deep copy.
        assert snap2 == snap1
        assert snap2 is not snap1
    finally:
        s5.close()


# ══════════════════════════════════════════════════════════════════════════════
# 19. different run re-discovers
# ══════════════════════════════════════════════════════════════════════════════
def test_different_run_rediscovers(local_storage, registry_service):
    s5 = _agentserver_stub(base_builder=build_step5_agent_card)
    try:
        svc = _service(_one_step5(s5.url), local_storage, registry_service)
        run_a = _run(registry_service)
        svc.discover_for_run(run_a)
        after_a = s5.hits["card"]
        run_b = _run(registry_service)
        svc.discover_for_run(run_b)
        assert s5.hits["card"] > after_a  # a fresh discovery happened for run_b
    finally:
        s5.close()


# ══════════════════════════════════════════════════════════════════════════════
# 20-24. resolve_dispatch_target
# ══════════════════════════════════════════════════════════════════════════════
def test_resolve_dispatch_target_valid(local_storage, registry_service):
    s5 = _agentserver_stub(base_builder=build_step5_agent_card)
    try:
        run_id = _run(registry_service)
        svc = _service(_one_step5(s5.url), local_storage, registry_service)
        svc.discover_for_run(run_id)
        target = svc.resolve_dispatch_target(
            run_id, agent_id=AGENT_ID_STEP5, capability_id=CAP_STEP5_CANDIDATE_CONTEXT
        )
        assert target.agent_id == AGENT_ID_STEP5
        assert target.capability_id == CAP_STEP5_CANDIDATE_CONTEXT
        assert target.dispatch_url == s5.url
        assert target.dispatch_mode == "python_a2a"
    finally:
        s5.close()


def test_resolve_unknown_agent_rejected(local_storage, registry_service):
    s5 = _agentserver_stub(base_builder=build_step5_agent_card)
    try:
        run_id = _run(registry_service)
        svc = _service(_one_step5(s5.url), local_storage, registry_service)
        svc.discover_for_run(run_id)
        with pytest.raises(DispatchTargetValidationError):
            svc.resolve_dispatch_target(run_id, agent_id="nope", capability_id="x")
    finally:
        s5.close()


def test_resolve_unavailable_worker_rejected(local_storage, registry_service):
    dead_port = _free_port()
    run_id = _run(registry_service)
    expected = _one_step5(f"http://127.0.0.1:{dead_port}")
    svc = _service(expected, local_storage, registry_service)
    svc.discover_for_run(run_id)
    with pytest.raises(WorkerUnavailableError):
        svc.resolve_dispatch_target(
            run_id, agent_id=AGENT_ID_STEP5, capability_id=CAP_STEP5_CANDIDATE_CONTEXT
        )


def test_resolve_wrong_capability_rejected(local_storage, registry_service):
    s5 = _agentserver_stub(base_builder=build_step5_agent_card)
    try:
        run_id = _run(registry_service)
        svc = _service(_one_step5(s5.url), local_storage, registry_service)
        svc.discover_for_run(run_id)
        with pytest.raises(DispatchTargetValidationError):
            svc.resolve_dispatch_target(
                run_id, agent_id=AGENT_ID_STEP5, capability_id="step_06_developability"
            )
    finally:
        s5.close()


def test_resolve_wrong_dispatch_mode_rejected(local_storage, registry_service):
    s5 = _agentserver_stub(base_builder=build_step5_agent_card)
    try:
        run_id = _run(registry_service)
        svc = _service(_one_step5(s5.url), local_storage, registry_service)
        svc.discover_for_run(run_id)
        with pytest.raises(DispatchTargetValidationError):
            svc.resolve_dispatch_target(
                run_id,
                agent_id=AGENT_ID_STEP5,
                capability_id=CAP_STEP5_CANDIDATE_CONTEXT,
                dispatch_mode="http_a2a",
            )
    finally:
        s5.close()


def test_resolve_before_discovery_rejected(local_storage, registry_service):
    s5 = _agentserver_stub(base_builder=build_step5_agent_card)
    try:
        svc = _service(_one_step5(s5.url), local_storage, registry_service)
        with pytest.raises(DispatchTargetValidationError):
            svc.resolve_dispatch_target(
                "run_never_discovered", agent_id=AGENT_ID_STEP5, capability_id=CAP_STEP5_CANDIDATE_CONTEXT
            )
    finally:
        s5.close()


# ══════════════════════════════════════════════════════════════════════════════
# 25. unavailable worker: no worker business method / no A2A dispatch / no MCP / no LLM
# ══════════════════════════════════════════════════════════════════════════════
def test_no_worker_calls_no_dispatch_no_llm_no_mcp(local_storage, registry_service, monkeypatch):
    import app.agents.candidate_context_agent as cca
    import app.agents.developability_agent as dev
    import app.agents.structure_and_design_agent as sad

    calls = Counter()

    def _spy(name):
        def _raise(*a, **k):
            calls[name] += 1
            raise AssertionError(f"worker business method {name} was called during discovery")
        return _raise

    monkeypatch.setattr(cca.CandidateContextAgent, "run", _spy("cca.run"), raising=False)
    monkeypatch.setattr(cca.CandidateContextAgent, "run_from_artifacts", _spy("cca.run_from_artifacts"), raising=False)
    monkeypatch.setattr(dev.DevelopabilityAgent, "run", _spy("dev.run"), raising=False)
    monkeypatch.setattr(sad.StructureAndDesignAgent, "run_step_7", _spy("sad.run_step_7"), raising=False)

    dispatched = Counter()

    async def _no_dispatch(self, task):
        dispatched["send_task_async"] += 1
        raise AssertionError("A2A task dispatch attempted during discovery")

    monkeypatch.setattr(A2AClient, "send_task_async", _no_dispatch, raising=False)

    dead_port = _free_port()
    run_id = _run(registry_service)
    svc = _service(_one_step5(f"http://127.0.0.1:{dead_port}"), local_storage, registry_service)
    snap = svc.discover_for_run(run_id)  # worker unavailable
    assert snap.discovery_status == "unavailable"
    with pytest.raises(WorkerUnavailableError):
        svc.resolve_dispatch_target(
            run_id, agent_id=AGENT_ID_STEP5, capability_id=CAP_STEP5_CANDIDATE_CONTEXT
        )
    assert calls.total() == 0
    assert dispatched.total() == 0


# ══════════════════════════════════════════════════════════════════════════════
# 26. persisted artifact id / run id reconcile with registry
# ══════════════════════════════════════════════════════════════════════════════
def test_persisted_snapshot_reconciles_with_registry(local_storage, registry_service):
    s5 = _agentserver_stub(base_builder=build_step5_agent_card)
    try:
        run_id = _run(registry_service)
        svc = _service(_one_step5(s5.url), local_storage, registry_service)
        svc.discover_for_run(run_id)
        persisted = local_storage.read_json(
            local_storage.run_key(run_id, "inputs/worker_discovery_snapshot.json")
        )
        reg = registry_service.get(run_id)
        assert persisted["run_id"] == run_id
        assert persisted["artifact_id"] == reg.active_artifacts.worker_discovery_snapshot_id
        assert persisted["cache_frozen"] is True
        # run_step_plan is untouched by discovery.
        assert reg.active_artifacts.run_step_plan_id is None
    finally:
        s5.close()


# ══════════════════════════════════════════════════════════════════════════════
# 27. discovery_error is compact and leaks no endpoint / raw exception
# ══════════════════════════════════════════════════════════════════════════════
def test_discovery_error_is_compact_and_safe(local_storage, registry_service):
    dead_port = _free_port()
    endpoint = f"http://127.0.0.1:{dead_port}"
    run_id = _run(registry_service)
    snap = _service(_one_step5(endpoint), local_storage, registry_service).discover_for_run(run_id)
    st = _status_by_agent(snap)[AGENT_ID_STEP5]
    assert st.discovery_error is not None
    assert len(st.discovery_error) <= MAX_DISCOVERY_ERROR_LEN
    assert "127.0.0.1" not in st.discovery_error
    assert str(dead_port) not in st.discovery_error
    assert "http" not in st.discovery_error
    assert " " not in st.discovery_error  # code-shaped, not a sentence/exception


# ══════════════════════════════════════════════════════════════════════════════
# Settings validator: timeouts must be > 0
# ══════════════════════════════════════════════════════════════════════════════
def test_settings_reject_non_positive_timeouts():
    from pydantic import ValidationError

    from app.settings import Settings

    with pytest.raises(ValidationError):
        Settings(a2a_discovery_timeout_seconds=0)
    with pytest.raises(ValidationError):
        Settings(a2a_health_timeout_seconds=-1)


# ══════════════════════════════════════════════════════════════════════════════
# Frozen-cache isolation: external returns are defensive copies of a canonical
# per-run cache; caller mutation cannot change routing or re-trigger discovery.
# ══════════════════════════════════════════════════════════════════════════════
def _two_worker_service(local_storage, registry_service):
    s5 = _agentserver_stub(base_builder=build_step5_agent_card)
    s6 = _agentserver_stub(base_builder=build_step6_agent_card)
    expected = [
        ExpectedWorkerEndpoint(AGENT_ID_STEP5, (CAP_STEP5_CANDIDATE_CONTEXT,), s5.url),
        ExpectedWorkerEndpoint(AGENT_ID_STEP6, (CAP_STEP6_DEVELOPABILITY,), s6.url),
    ]
    svc = _service(expected, local_storage, registry_service)
    return svc, s5, s6


# A. mutating the returned snapshot does not corrupt the frozen content, add HTTP,
#    or change the persisted snapshot.
def test_mutating_returned_snapshot_does_not_affect_frozen_state(local_storage, registry_service):
    svc, s5, s6 = _two_worker_service(local_storage, registry_service)
    try:
        run_id = _run(registry_service)
        snap1 = svc.discover_for_run(run_id)
        card_hits = (s5.hits["card"], s6.hits["card"])
        health_hits = (s5.hits["health"], s6.hits["health"])
        persisted_before = local_storage.read_json(
            local_storage.run_key(run_id, "inputs/worker_discovery_snapshot.json")
        )

        # Caller vandalises its copy.
        snap1.available_agent_ids.append("ATTACKER_AGENT")
        snap1.available_agent_ids.clear()
        step5_entry = next(c for c in snap1.compact_card_catalog if c["agent_id"] == AGENT_ID_STEP5)
        step5_entry["capabilities"][0]["capability_id"] = "HACKED_CAPABILITY"
        step5_entry["availability"] = "unavailable"

        snap2 = svc.discover_for_run(run_id)

        # Frozen content intact.
        assert snap2.available_agent_ids == [AGENT_ID_STEP5, AGENT_ID_STEP6]
        step5_entry2 = next(c for c in snap2.compact_card_catalog if c["agent_id"] == AGENT_ID_STEP5)
        assert step5_entry2["capabilities"][0]["capability_id"] == CAP_STEP5_CANDIDATE_CONTEXT
        assert step5_entry2["availability"] == "available"
        # No new HTTP.
        assert (s5.hits["card"], s6.hits["card"]) == card_hits
        assert (s5.hits["health"], s6.hits["health"]) == health_hits
        # Persisted snapshot unchanged.
        persisted_after = local_storage.read_json(
            local_storage.run_key(run_id, "inputs/worker_discovery_snapshot.json")
        )
        assert persisted_after == persisted_before
    finally:
        s5.close()
        s6.close()


# B. mutating get_compact_card_catalog() nested dict/list does not affect the
#    catalog on a subsequent read.
def test_mutating_compact_catalog_return_does_not_persist(local_storage, registry_service):
    svc, s5, s6 = _two_worker_service(local_storage, registry_service)
    try:
        run_id = _run(registry_service)
        svc.discover_for_run(run_id)
        cat1 = svc.get_compact_card_catalog(run_id)
        cat1[0]["capabilities"].append({"capability_id": "INJECTED"})
        cat1[0]["availability"] = "unavailable"
        cat1.append({"agent_id": "ROGUE"})

        cat2 = svc.get_compact_card_catalog(run_id)
        assert [c["agent_id"] for c in cat2] == [AGENT_ID_STEP5, AGENT_ID_STEP6]
        assert cat2[0]["availability"] == "available"
        assert all(
            cap["capability_id"] != "INJECTED" for cap in cat2[0]["capabilities"]
        )
    finally:
        s5.close()
        s6.close()


# C. mutating get_full_card_cache() objects does not affect resolve_dispatch_target.
def test_mutating_full_cache_copy_does_not_change_routing(local_storage, registry_service):
    s5 = _agentserver_stub(base_builder=build_step5_agent_card)
    try:
        run_id = _run(registry_service)
        svc = _service(_one_step5(s5.url), local_storage, registry_service)
        svc.discover_for_run(run_id)

        # Baseline: routing works.
        target = svc.resolve_dispatch_target(
            run_id, agent_id=AGENT_ID_STEP5, capability_id=CAP_STEP5_CANDIDATE_CONTEXT
        )
        assert target.dispatch_url == s5.url

        # Vandalise the DEFENSIVE COPY of the full cache in every way that would
        # break routing if it leaked into the canonical cache.
        copy_cache = svc.get_full_card_cache(run_id)
        worker = copy_cache.workers[AGENT_ID_STEP5]
        worker.availability = "unavailable"
        worker.dispatch_url = "http://attacker:1"
        worker.contract.routable = False
        worker.contract.status = "disabled"
        worker.contract.capabilities.clear()
        worker.contract.dispatch_modes = ["http_a2a"]
        copy_cache.workers.pop(AGENT_ID_STEP5, None)

        # Canonical routing is unaffected.
        target2 = svc.resolve_dispatch_target(
            run_id, agent_id=AGENT_ID_STEP5, capability_id=CAP_STEP5_CANDIDATE_CONTEXT
        )
        assert target2.agent_id == AGENT_ID_STEP5
        assert target2.capability_id == CAP_STEP5_CANDIDATE_CONTEXT
        assert target2.dispatch_url == s5.url
        assert target2.dispatch_mode == "python_a2a"

        # A second fetched copy is pristine too (proves the canonical was untouched).
        fresh = svc.get_full_card_cache(run_id)
        assert fresh.workers[AGENT_ID_STEP5].availability == "available"
        assert fresh.workers[AGENT_ID_STEP5].contract.routable is True
    finally:
        s5.close()


# D. after external copies are mutated, existing deterministic validation still
#    rejects unknown capability / unavailable worker per the canonical cache.
def test_deterministic_validation_uses_canonical_after_external_mutation(local_storage, registry_service):
    s5 = _agentserver_stub(base_builder=build_step5_agent_card)
    dead = _free_port()
    try:
        expected = [
            ExpectedWorkerEndpoint(AGENT_ID_STEP5, (CAP_STEP5_CANDIDATE_CONTEXT,), s5.url),
            ExpectedWorkerEndpoint(AGENT_ID_STEP6, (CAP_STEP6_DEVELOPABILITY,), f"http://127.0.0.1:{dead}"),
        ]
        run_id = _run(registry_service)
        svc = _service(expected, local_storage, registry_service)
        svc.discover_for_run(run_id)

        # Attacker tries to "enable" the unavailable Step6 and grant Step5 a new
        # capability by editing external copies.
        copy_cache = svc.get_full_card_cache(run_id)
        copy_cache.workers[AGENT_ID_STEP6].availability = "available"
        copy_cache.workers[AGENT_ID_STEP6].dispatch_url = "http://127.0.0.1:1"
        copy_cache.workers[AGENT_ID_STEP5].contract.capabilities.append(
            copy_cache.workers[AGENT_ID_STEP5].contract.capabilities[0]
        )
        cat = svc.get_compact_card_catalog(run_id)
        cat[0]["capabilities"].append({"capability_id": "step_06_developability"})

        # Canonical validation is unchanged: unavailable stays rejected, and an
        # unknown capability stays rejected.
        with pytest.raises(WorkerUnavailableError):
            svc.resolve_dispatch_target(
                run_id, agent_id=AGENT_ID_STEP6, capability_id=CAP_STEP6_DEVELOPABILITY
            )
        with pytest.raises(DispatchTargetValidationError):
            svc.resolve_dispatch_target(
                run_id, agent_id=AGENT_ID_STEP5, capability_id="step_06_developability"
            )
    finally:
        s5.close()
