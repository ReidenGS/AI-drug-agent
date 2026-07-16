"""Internal resume API regression tests; no public Step4 API wiring."""

from __future__ import annotations

import pickle

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from app.a2a.orchestrator_discovery import DispatchTarget
from app.a2a.orchestrator_execution_state import dispatch_eligible_task_ids
from app.a2a.orchestrator_resume import resume_orchestrator_run
from app.a2a.orchestrator_routing_service import OrchestratorRoutingService
from app.graph.orchestrator_execution_graph import (
    build_orchestrator_execution_graph,
    execution_graph_config,
)
from tests.a2a.test_orchestrator_dispatch import _serve_task_handler
from tests.a2a.test_orchestrator_post_ingestion import (
    RUN_ID,
    _contract,
    _DeterministicLLM,
    _environment,
    _FrozenDiscovery,
    _proposal,
)
from tests.a2a.test_orchestrator_retry_loop import (
    _SyntheticWorker,
    _bind_http_targets,
    _card,
    _run_chain,
)


class _Runtime:
    def __init__(self, graph):
        self.graph = graph


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


@pytest.mark.asyncio
async def test_completed_resume_returns_without_discovery_llm_or_post(
    local_storage, registry_service
):
    completed, _alpha, _beta, _handles, saver = await _run_chain(
        local_storage, registry_service, fail_count=0
    )
    graph = build_orchestrator_execution_graph(checkpointer=saver)
    contracts = [
        _contract("agent_alpha"),
        _contract("agent_beta", requires=("agent_alpha",)),
    ]
    discovery = _FrozenDiscovery(contracts)
    llm = _DeterministicLLM(_proposal("agent_alpha", "agent_beta"))
    service = OrchestratorRoutingService(
        discovery=discovery,
        storage=local_storage,
        registry=registry_service,
        llm=llm,
    )
    llm_before = llm.call_count
    discovery.discover_for_run = lambda _run_id: (_ for _ in ()).throw(
        AssertionError("completed resume must not discover")
    )
    result = await resume_orchestrator_run(
        run_id=RUN_ID,
        checkpoint_runtime=_Runtime(graph),
        routing_service=service,
        discovery=discovery,
        registry=registry_service,
        storage=local_storage,
        timeout_seconds=1,
        max_worker_retries=3,
    )
    assert result.outcome == "completed"
    assert result.state == completed.state
    assert result.dispatch_attempt_count == 0
    assert result.completion_proofs == {}
    assert llm.call_count == llm_before
    with pytest.raises(
        TypeError, match="^orchestrator_resume_result_pickle_unsupported$"
    ) as caught:
        pickle.dumps(result)
    assert RUN_ID not in str(caught.value)
    assert "completion_proofs" not in repr(result)
    assert "completion_proofs" not in result.model_dump()


@pytest.mark.asyncio
async def test_fresh_resume_rebuilds_pending_task_without_old_prepared_object(
    local_storage, registry_service
):
    contracts = [_contract("agent_alpha")]
    _service, _llm, discovery, initial, state = _environment(
        local_storage, registry_service, contracts, ("agent_alpha",)
    )
    worker = _SyntheticWorker(
        agent_id="agent_alpha",
        storage=local_storage,
        registry=registry_service,
    )
    handle = _serve_task_handler(_card("agent_alpha"), worker.handle)
    saver = InMemorySaver()
    graph = build_orchestrator_execution_graph(checkpointer=saver)
    try:
        _bind_http_targets(initial, discovery, {"agent_alpha": handle})
        await graph.ainvoke(state, config=execution_graph_config(RUN_ID))

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
        task_id = next(iter(state.worker_tasks))
        result = await resume_orchestrator_run(
            run_id=RUN_ID,
            checkpoint_runtime=_Runtime(
                build_orchestrator_execution_graph(checkpointer=saver)
            ),
            routing_service=fresh_service,
            discovery=fresh_discovery,
            registry=registry_service,
            storage=local_storage,
            timeout_seconds=1,
            max_worker_retries=3,
        )
        assert result.outcome == "completed"
        assert result.dispatch_attempt_count == 1
        assert next(iter(result.state.worker_tasks)) == task_id
        assert set(result.completion_proofs) == {task_id}
        assert "completion_proofs" not in repr(result)
        assert "completion_proofs" not in result.model_dump()
        assert "completion_proofs" not in dict(result)
        assert handle.hits["task"] == 1
        assert fresh_llm.call_count == 0
    finally:
        handle.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("failure_status", "fail_count", "expected_outcome", "expected_posts"),
    [
        ("tool_failed", 4, "failed", 4),
        ("validation_failed", 1, "failed", 1),
        ("needs_user_input", 1, "waiting_for_input", 1),
    ],
)
async def test_resumed_terminal_outcomes_are_not_reported_as_dispatched(
    local_storage,
    registry_service,
    failure_status,
    fail_count,
    expected_outcome,
    expected_posts,
):
    contracts = [_contract("agent_alpha")]
    service, _llm, discovery, initial, state = _environment(
        local_storage, registry_service, contracts, ("agent_alpha",)
    )
    worker = _SyntheticWorker(
        agent_id="agent_alpha",
        storage=local_storage,
        registry=registry_service,
        fail_count=fail_count,
        status=failure_status,
    )
    handle = _serve_task_handler(_card("agent_alpha"), worker.handle)
    saver = InMemorySaver()
    graph = build_orchestrator_execution_graph(checkpointer=saver)
    try:
        _bind_http_targets(initial, discovery, {"agent_alpha": handle})
        await graph.ainvoke(state, config=execution_graph_config(RUN_ID))
        result = await resume_orchestrator_run(
            run_id=RUN_ID,
            checkpoint_runtime=_Runtime(graph),
            routing_service=service,
            discovery=discovery,
            registry=registry_service,
            storage=local_storage,
            timeout_seconds=1,
            max_worker_retries=3,
        )
        task = list(result.state.worker_tasks.values())[-1]
        assert result.outcome == expected_outcome
        assert result.dispatch_attempt_count == expected_posts
        assert handle.hits["task"] == expected_posts
        assert task.execution_status == "failed"
        assert task.result_status == failure_status
        assert dispatch_eligible_task_ids(result.state) == ()
        if expected_outcome == "waiting_for_input":
            assert result.state.run_status == "waiting_for_input"
            assert result.state.next_wakeup.target == "user_input"
            assert result.state.next_wakeup.reason == "needs_user_input"
        else:
            assert result.state.run_status == "failed"
            assert result.state.next_wakeup.target == "final_response"
    finally:
        handle.close()
