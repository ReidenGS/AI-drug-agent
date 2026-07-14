"""Real HTTP get_task reconciliation tests with synthetic transport workers."""

from __future__ import annotations

import json
import pickle
import time
from collections import Counter

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from app.a2a.orchestrator_execution_loop import execute_orchestrator_worker_loop
from app.a2a.orchestrator_reconciliation import (
    OrchestratorReconciliationError,
    reconcile_orchestrator_tasks,
)
from app.graph.orchestrator_execution_graph import (
    build_orchestrator_execution_graph,
    execution_graph_config,
)
from app.schemas.orchestrator_execution_state import (
    OrchestratorExecutionState,
)
from tests.a2a.test_orchestrator_dispatch import _serve_task_handler
from tests.a2a.test_orchestrator_post_ingestion import (
    ARTIFACTS,
    RUN_ID,
    _contract,
    _environment,
)
from tests.a2a.test_orchestrator_retry_loop import (
    _SyntheticWorker,
    _bind_http_targets,
    _card,
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


@pytest.mark.asyncio
async def test_timeout_terminal_task_is_recovered_by_real_http_get_task(
    local_storage, registry_service
):
    service, _llm, discovery, initial, state = _environment(
        local_storage,
        registry_service,
        [_contract("agent_alpha")],
        ("agent_alpha",),
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
    saver = InMemorySaver()
    graph = build_orchestrator_execution_graph(checkpointer=saver)
    config = execution_graph_config(RUN_ID)
    try:
        prepared = _bind_http_targets(
            initial, discovery, {"agent_alpha": handle}
        )
        uncertain = await execute_orchestrator_worker_loop(
            run_id=RUN_ID,
            state=state,
            prepared_tasks=prepared,
            routing_service=service,
            discovery=discovery,
            registry=registry_service,
            storage=local_storage,
            execution_graph=graph,
            checkpoint_config=config,
            timeout_seconds=0.02,
            max_worker_retries=3,
        )
        task_id = next(iter(uncertain.state.worker_tasks))
        assert uncertain.outcome == "reconciliation_required"
        assert uncertain.completion_proofs == {}
        assert handle.hits == Counter({"card": 1, "task": 1})
        time.sleep(0.2)

        reconciled = await reconcile_orchestrator_tasks(
            run_id=RUN_ID,
            state=uncertain.state,
            task_ids=(task_id,),
            routing_service=service,
            discovery=discovery,
            registry=registry_service,
            storage=local_storage,
            execution_graph=graph,
            checkpoint_config=config,
            timeout_seconds=1,
        )
        task = reconciled.state.worker_tasks[task_id]
        artifact = reconciled.state.artifacts[ARTIFACTS["agent_alpha"][0]]
        assert reconciled.status == "resolved"
        assert reconciled.resolved_task_ids == (task_id,)
        assert task.dispatch_status == "dispatched"
        assert task.execution_status == "completed"
        assert task.result_status == "success"
        assert set(reconciled.completion_proofs) == {task_id}
        assert artifact.status == "available"
        assert artifact.artifact_id == task.output_artifact_refs[artifact.artifact_name]
        assert artifact.producer_task_id == task_id
        assert handle.hits == Counter({"card": 2, "task": 1, "get_task": 1})
        with pytest.raises(
            TypeError,
            match="^orchestrator_reconciliation_result_pickle_unsupported$",
        ) as caught:
            pickle.dumps(reconciled)
        assert task_id not in str(caught.value)
        assert "completion_proofs" not in repr(reconciled)
        assert "completion_proofs" not in reconciled.model_dump()
    finally:
        handle.close()


@pytest.mark.asyncio
async def test_get_task_unavailable_stays_reconciliation_without_checkpoint_write(
    local_storage, registry_service
):
    service, _llm, discovery, initial, state = _environment(
        local_storage,
        registry_service,
        [_contract("agent_alpha")],
        ("agent_alpha",),
    )
    worker = _SyntheticWorker(
        agent_id="agent_alpha",
        storage=local_storage,
        registry=registry_service,
    )
    handle = _serve_task_handler(_card("agent_alpha"), worker.handle)
    try:
        prepared = _bind_http_targets(
            initial, discovery, {"agent_alpha": handle}
        )
        task_id = str(prepared[0].task.id)
        payload = state.model_dump()
        payload["worker_tasks"][task_id].update(
            {
                "dispatch_status": "dispatch_failed",
                "agent_failure_reason": "dispatch_timeout",
            }
        )
        payload["routing"]["decisions"][
            state.worker_tasks[task_id].routing_decision_id
        ]["status"] = "failed"
        payload["artifacts"][ARTIFACTS["agent_alpha"][0]]["status"] = "invalid"
        payload["run_status"] = "running"
        payload["orchestrator"].update(
            {
                "status": "evaluating_results",
                "next_wakeup_reason": "worker_result_reconciliation_required",
            }
        )
        payload["next_wakeup"] = {
            "target": "orchestrator_loop",
            "reason": "worker_result_reconciliation_required",
        }
        from app.schemas.orchestrator_execution_state import (
            OrchestratorExecutionState,
        )

        uncertain = OrchestratorExecutionState.model_validate(payload)
        saver = InMemorySaver()
        result = await reconcile_orchestrator_tasks(
            run_id=RUN_ID,
            state=uncertain,
            task_ids=(task_id,),
            routing_service=service,
            discovery=discovery,
            registry=registry_service,
            storage=local_storage,
            execution_graph=build_orchestrator_execution_graph(
                checkpointer=saver
            ),
            checkpoint_config=execution_graph_config(RUN_ID),
            timeout_seconds=1,
        )
        assert result.status == "reconciliation_required"
        assert result.unresolved_task_ids == (task_id,)
        assert result.completion_proofs == {}
        assert list(saver.list(None)) == []
        assert handle.hits == Counter({"card": 1, "get_task": 2})
    finally:
        handle.close()


def test_reconciliation_module_has_no_business_worker_special_cases():
    from pathlib import Path

    text = (
        Path(__file__).parents[2]
        / "app/a2a/orchestrator_reconciliation.py"
    ).read_text().lower()
    for forbidden in (
        "step5",
        "step6",
        "structure",
        "candidate_context",
        "developability",
    ):
        assert forbidden not in text


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("corruption", "expected_code"),
    [
        ("task_id", "reconciliation_task_identity_mismatch"),
        ("request_identity", "reconciliation_request_identity_mismatch"),
        ("wrong_retry_attempt", "reconciliation_request_identity_mismatch"),
        ("wrong_retry_parent", "reconciliation_request_identity_mismatch"),
        ("malformed_result", "worker_result_json_invalid"),
        ("missing_output", "completion_output_artifacts_missing"),
        ("unexpected_output", "completion_output_artifacts_unexpected"),
        ("persisted_identity", "completion_output_artifact_invalid"),
    ],
)
async def test_malformed_reconciliation_fails_before_checkpoint(
    local_storage,
    registry_service,
    corruption,
    expected_code,
):
    """Fake get_task response isolates malformed transport payloads.

    Real HTTP get_task coverage is provided above; this fixture calls the
    production reconciliation/parser/artifact validators without a live LLM,
    MCP, ToolUniverse, or biomedical worker.
    """
    service, _llm, discovery, initial, state = _environment(
        local_storage,
        registry_service,
        [_contract("agent_alpha")],
        ("agent_alpha",),
    )
    prepared = initial.prepared_tasks[0]
    worker = _SyntheticWorker(
        agent_id="agent_alpha",
        storage=local_storage,
        registry=registry_service,
    )
    response = worker.handle(prepared.task)
    task_id = str(prepared.task.id)

    payload = state.model_dump()
    payload["worker_tasks"][task_id].update(
        {
            "dispatch_status": "dispatch_failed",
            "agent_failure_reason": "dispatch_timeout",
        }
    )
    decision_id = state.worker_tasks[task_id].routing_decision_id
    payload["routing"]["decisions"][decision_id].update(
        {"status": "failed", "blocking_reason": "dispatch_failed"}
    )
    artifact_name, artifact_path = ARTIFACTS["agent_alpha"]
    payload["artifacts"][artifact_name]["status"] = "invalid"
    payload["run_status"] = "running"
    payload["orchestrator"].update(
        {
            "status": "evaluating_results",
            "next_wakeup_reason": "worker_result_reconciliation_required",
        }
    )
    payload["next_wakeup"] = {
        "target": "orchestrator_loop",
        "reason": "worker_result_reconciliation_required",
    }
    uncertain = OrchestratorExecutionState.model_validate(payload)

    if corruption == "task_id":
        response.id = "task_ffffffffffffffff"
    elif corruption == "request_identity":
        request = json.loads(response.message["content"]["text"])
        request["capability_id"] = "capability_intruder"
        response.message["content"]["text"] = json.dumps(request)
    elif corruption in {"wrong_retry_attempt", "wrong_retry_parent"}:
        request = json.loads(response.message["content"]["text"])
        request["retry_context"] = {
            "retry_attempt": 1,
            "max_retry_attempts": 3,
            "retry_of_task_id": (
                "task_ffffffffffffffff"
                if corruption == "wrong_retry_parent"
                else task_id
            ),
            "retry_reason": "synthetic_tool_failed",
        }
        if corruption == "wrong_retry_attempt":
            request["retry_context"]["retry_attempt"] = 2
        response.message["content"]["text"] = json.dumps(request)
    elif corruption == "malformed_result":
        response.artifacts[0]["parts"][0]["text"] = "{not-json"
    elif corruption in {"missing_output", "unexpected_output"}:
        result = json.loads(response.artifacts[0]["parts"][0]["text"])
        if corruption == "missing_output":
            result["output_artifact_refs"] = {}
        else:
            result["output_artifact_refs"]["unexpected_artifact"] = dict(
                next(iter(result["output_artifact_refs"].values()))
            )
        response.artifacts[0]["parts"][0]["text"] = json.dumps(result)
    else:
        key = local_storage.run_key(RUN_ID, artifact_path)
        body = local_storage.read_json(key)
        body["artifact_id"] = "artifact_ffffffffffffffff"
        local_storage.write_json(key, body)

    class _Client:
        def __init__(self, *_args, **_kwargs):
            pass

        def get_task(self, _task_id):
            return response

    saver = InMemorySaver()
    plan_key = local_storage.run_key(RUN_ID, "inputs/worker_routing_plan.json")
    plan_before = local_storage.read_json(plan_key)
    with pytest.raises(
        OrchestratorReconciliationError,
        match=f"^{expected_code}$",
    ) as caught:
        await reconcile_orchestrator_tasks(
            run_id=RUN_ID,
            state=uncertain,
            task_ids=(task_id,),
            routing_service=service,
            discovery=discovery,
            registry=registry_service,
            storage=local_storage,
            execution_graph=build_orchestrator_execution_graph(
                checkpointer=saver
            ),
            checkpoint_config=execution_graph_config(RUN_ID),
            timeout_seconds=1,
            client_factory=_Client,
        )
    assert list(saver.list(None)) == []
    assert local_storage.read_json(plan_key) == plan_before
    assert "ffffffff" not in str(caught.value)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mode", "expected_code"),
    [
        ("non_task", "reconciliation_task_protocol_invalid"),
        ("missing_status", "reconciliation_task_protocol_invalid"),
        ("invalid_status", "reconciliation_task_protocol_invalid"),
        ("sdk_exception", "reconciliation_get_task_protocol_error"),
    ],
)
async def test_get_task_protocol_failures_are_compact_and_write_no_checkpoint(
    local_storage,
    registry_service,
    mode,
    expected_code,
):
    service, _llm, discovery, initial, state = _environment(
        local_storage,
        registry_service,
        [_contract("agent_alpha")],
        ("agent_alpha",),
    )
    prepared = initial.prepared_tasks[0]
    task_id = str(prepared.task.id)
    response = _SyntheticWorker(
        agent_id="agent_alpha",
        storage=local_storage,
        registry=registry_service,
    ).handle(prepared.task)
    payload = state.model_dump()
    payload["worker_tasks"][task_id].update(
        {
            "dispatch_status": "dispatch_failed",
            "agent_failure_reason": "dispatch_timeout",
        }
    )
    decision_id = state.worker_tasks[task_id].routing_decision_id
    payload["routing"]["decisions"][decision_id].update(
        {"status": "failed", "blocking_reason": "dispatch_failed"}
    )
    payload["artifacts"][ARTIFACTS["agent_alpha"][0]]["status"] = "invalid"
    payload["orchestrator"].update(
        {
            "status": "evaluating_results",
            "next_wakeup_reason": "worker_result_reconciliation_required",
        }
    )
    payload["next_wakeup"] = {
        "target": "orchestrator_loop",
        "reason": "worker_result_reconciliation_required",
    }
    uncertain = OrchestratorExecutionState.model_validate(payload)
    sentinel = "sk-live-GET-TASK-PROTOCOL-SECRET"
    if mode == "non_task":
        returned = {"payload": sentinel}
    else:
        returned = response
        if mode == "missing_status":
            returned.status = None
        elif mode == "invalid_status":
            returned.status.state = "private_invalid_status"

    class _Client:
        def __init__(self, *_args, **_kwargs):
            pass

        def get_task(self, _task_id):
            if mode == "sdk_exception":
                raise RuntimeError(sentinel)
            return returned

    saver = InMemorySaver()
    plan_key = local_storage.run_key(RUN_ID, "inputs/worker_routing_plan.json")
    plan_before = local_storage.read_json(plan_key)
    with pytest.raises(
        OrchestratorReconciliationError,
        match=f"^{expected_code}$",
    ) as caught:
        await reconcile_orchestrator_tasks(
            run_id=RUN_ID,
            state=uncertain,
            task_ids=(task_id,),
            routing_service=service,
            discovery=discovery,
            registry=registry_service,
            storage=local_storage,
            execution_graph=build_orchestrator_execution_graph(
                checkpointer=saver
            ),
            checkpoint_config=execution_graph_config(RUN_ID),
            timeout_seconds=1,
            client_factory=_Client,
        )
    assert list(saver.list(None)) == []
    assert local_storage.read_json(plan_key) == plan_before
    assert sentinel not in str(caught.value)
    assert sentinel not in repr(caught.value)
    assert sentinel not in repr(caught.value.args)
