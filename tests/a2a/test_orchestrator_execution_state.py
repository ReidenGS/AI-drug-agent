"""Compact execution-state projection/reducer tests; no task is dispatched."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError
from python_a2a import Task

from app.a2a.contracts import InputArtifactRef
from app.a2a.orchestrator_discovery import DispatchTarget
from app.a2a.orchestrator_execution_state import (
    OrchestratorExecutionStateError,
    dispatch_eligible_task_ids,
    execution_state_from_routing_result,
    mark_task_dispatch_failed,
    mark_task_dispatched,
    mark_task_dispatching,
    mark_task_result,
    mark_task_running,
    recompute_aggregate_state,
    transition_task_lifecycle,
)
from app.a2a.orchestrator_routing_service import OrchestratorRoutingServiceResult
from app.a2a.orchestrator_task_builder import PreparedA2ATask
from app.schemas.orchestrator_execution_state import (
    NextWakeupState,
    OrchestratorExecutionState,
)
from app.schemas.worker_routing_plan import (
    DependencyEdge,
    OrchestratorRouteDecision,
    ValidatedRoutingDecision,
    WorkerRoutingPlan,
)

_SENSITIVE = {
    "sequence": "ACDEFGHIKLMNPQRSTVWYACDEFGHIK",
    "fasta": ">private_fasta",
    "a3m": ">private_a3m",
    "pdb": "ATOM      1 PRIVATE_PDB",
    "cif": "data_private_mmcif",
    "api_key": "sk-private-api-key",
    "authorization": "Authorization: Bearer private-token",
    "tool_payload": "raw ToolUniverse payload private",
    "prompt": "full prompt private",
    "response": "raw LLM response private",
    "url": "http://private-worker.internal:8005",
    "path": "inputs/private-storage-path.json",
    "task_message": "WorkerExecutionRequest body private",
}


def _decision(
    *,
    decision_id: str,
    agent_id: str,
    capability_id: str,
    status: str,
    required: list[str],
    outputs: list[str],
    task_id: str | None,
) -> ValidatedRoutingDecision:
    return ValidatedRoutingDecision(
        routing_decision_id=decision_id,
        agent_id=agent_id,
        capability_id=capability_id,
        objective=f"compact objective {_SENSITIVE['sequence']}",
        selection_reason=f"compact reason {_SENSITIVE['api_key']}",
        priority="normal",
        validation_status=status,
        required_artifact_names=required,
        dependency_artifact_names=(required if status == "waiting_for_dependencies" else []),
        dependency_producers=(
            ["route_1111111111111111"]
            if status == "waiting_for_dependencies"
            else []
        ),
        expected_output_artifact_names=outputs,
        task_id=task_id,
        reason=("waiting_for_dependencies" if status == "waiting_for_dependencies" else None),
    )


def _prepared(
    decision: ValidatedRoutingDecision,
    *,
    input_refs: dict[str, InputArtifactRef],
) -> PreparedA2ATask:
    assert decision.task_id is not None
    return PreparedA2ATask(
        decision=decision,
        task=Task(
            id=decision.task_id,
            message={"content": {"text": json.dumps(_SENSITIVE)}},
            metadata={"authorization": _SENSITIVE["authorization"]},
        ),
        dispatch_target=DispatchTarget(
            agent_id=decision.agent_id,
            capability_id=decision.capability_id,
            dispatch_url=_SENSITIVE["url"],
            dispatch_mode="python_a2a",
        ),
        input_artifact_refs=input_refs,
    )


def _ref(run_id: str, artifact_name: str) -> InputArtifactRef:
    return InputArtifactRef(
        artifact_id=f"{artifact_name}_0123456789ab",
        run_id=run_id,
        artifact_type=artifact_name,
        artifact_role=artifact_name,
        field_keys=["safe_declared_field"],
        can_read_from_db=True,
    )


def routing_result_fixture(*, independent_ready: bool = False):
    """Deterministic F1 result fixture using real production contract types."""
    run_id = "run_20260714_abcdef12"
    step5 = _decision(
        decision_id="route_1111111111111111",
        agent_id="step_05_candidate_context_agent",
        capability_id="step_05_candidate_context",
        status="ready",
        required=["raw_request_record", "structured_query"],
        outputs=["candidate_context_table"],
        task_id="task_1111111111111111",
    )
    second = _decision(
        decision_id="route_2222222222222222",
        agent_id="step_06_developability_agent",
        capability_id="step_06_developability",
        status="ready" if independent_ready else "waiting_for_dependencies",
        required=(
            ["independent_input"] if independent_ready else ["candidate_context_table"]
        ),
        outputs=["structured_liability_summary"],
        task_id="task_2222222222222222" if independent_ready else None,
    )
    prepared = [
        _prepared(
            step5,
            input_refs={
                "raw_request_record": _ref(run_id, "raw_request_record"),
                "structured_query": _ref(run_id, "structured_query"),
            },
        )
    ]
    if independent_ready:
        prepared.append(
            _prepared(
                second,
                input_refs={"independent_input": _ref(run_id, "independent_input")},
            )
        )
    plan = WorkerRoutingPlan(
        run_id=run_id,
        session_id="sess_0123456789abcdef",
        routing_plan_id="wrp_0123456789abcdef",
        planned_at="2026-07-13T00:00:00Z",
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
            for item in (step5, second)
        ],
        validated_decisions=[step5, second],
        dependency_edges=(
            []
            if independent_ready
            else [
                DependencyEdge(
                    artifact_name="candidate_context_table",
                    producer_agent_id=step5.agent_id,
                    producer_capability_id=step5.capability_id,
                    consumer_agent_id=second.agent_id,
                    consumer_capability_id=second.capability_id,
                )
            ]
        ),
        ready_task_count=len(prepared),
        waiting_decision_count=0 if independent_ready else 1,
        rejected_decision_count=0,
    )
    return OrchestratorRoutingServiceResult(
        plan=plan,
        plan_artifact_id="worker_routing_plan_0123456789ab",
        prepared_tasks=tuple(prepared),
        reused_existing_plan=False,
        llm_called=True,
        discovery_performed=True,
    )


def blocked_routing_result_fixture():
    """Production-shaped blocked routing plan with no dispatchable worker."""
    decision = _decision(
        decision_id="route_3333333333333333",
        agent_id="step_05_candidate_context_agent",
        capability_id="step_05_candidate_context",
        status="blocked_missing_dependency",
        required=["raw_request_record"],
        outputs=["candidate_context_table"],
        task_id=None,
    )
    decision.reason = "missing_required_artifact"
    plan = WorkerRoutingPlan(
        run_id="run_20260714_deadbeef",
        routing_plan_id="wrp_3333333333333333",
        planned_at="2026-07-14T00:00:00Z",
        loop_decision="dispatch_next_workers",
        routing_status="blocked",
        llm_selection_source="llm_primary_validated",
        proposed_decisions=[
            OrchestratorRouteDecision(
                agent_id=decision.agent_id,
                capability_id=decision.capability_id,
                objective=decision.objective,
                selection_reason=decision.selection_reason,
                priority=decision.priority,
            )
        ],
        validated_decisions=[decision],
        ready_task_count=0,
        waiting_decision_count=0,
        rejected_decision_count=0,
    )
    return OrchestratorRoutingServiceResult(
        plan=plan,
        plan_artifact_id="worker_routing_plan_333333333333",
        prepared_tasks=(),
        reused_existing_plan=False,
        llm_called=True,
        discovery_performed=True,
    )


def test_initial_ready_waiting_projection_is_compact_and_generic():
    state = execution_state_from_routing_result(routing_result_fixture())

    assert state.routing.decisions["route_1111111111111111"].status == "ready"
    assert state.routing.decisions["route_2222222222222222"].status == (
        "pending_dependency"
    )
    assert state.routing.decisions["route_2222222222222222"].blocking_reason == (
        "missing_required_artifact"
    )
    assert state.worker_tasks["task_1111111111111111"].model_dump() == {
        "task_id": "task_1111111111111111",
        "routing_plan_id": "wrp_0123456789abcdef",
        "routing_decision_id": "route_1111111111111111",
        "agent_id": "step_05_candidate_context_agent",
        "capability_id": "step_05_candidate_context",
        "dispatch_status": "not_dispatched",
        "execution_status": "not_started",
        "result_status": None,
            "agent_failure_reason": "none",
            "retry_of_task_id": None,
            "retry_attempt": 0,
            "max_retry_attempts": 3,
            "terminal_error_code": None,
            "output_artifact_refs": {},
        }
    assert dispatch_eligible_task_ids(state) == ("task_1111111111111111",)
    assert state.artifacts["candidate_context_table"].status == "planned"
    assert state.artifacts["structured_liability_summary"].status == "planned"
    assert state.artifacts["raw_request_record"].status == "available"
    assert state.run_status == "running"
    assert state.orchestrator.status == "dispatching"
    assert state.next_wakeup.model_dump() == {
        "target": "worker_dispatch",
        "reason": "ready_tasks_available",
    }
    serialized = state.model_dump_json()
    for value in _SENSITIVE.values():
        assert value.lower() not in serialized.lower()
    for forbidden_key in (
        "message",
        "metadata",
        "storage_path",
        "worker_execution_request",
        "dispatch_url",
    ):
        assert forbidden_key not in serialized.lower()


def test_all_independent_ready_tasks_remain_eligible_without_cap():
    state = execution_state_from_routing_result(
        routing_result_fixture(independent_ready=True)
    )
    assert dispatch_eligible_task_ids(state) == (
        "task_1111111111111111",
        "task_2222222222222222",
    )
    assert len(state.worker_tasks) == 2

    one_dispatched = mark_task_dispatched(
        mark_task_dispatching(state, "task_1111111111111111"),
        "task_1111111111111111",
    )
    assert dispatch_eligible_task_ids(one_dispatched) == ("task_2222222222222222",)
    assert one_dispatched.orchestrator.status == "dispatching"
    assert one_dispatched.next_wakeup.model_dump() == {
        "target": "worker_dispatch",
        "reason": "ready_tasks_available",
    }
    assert one_dispatched.artifacts["candidate_context_table"].status == "producing"
    assert one_dispatched.artifacts["structured_liability_summary"].status == (
        "planned"
    )


def test_dispatch_lifecycle_is_identity_stable_and_idempotent():
    initial = execution_state_from_routing_result(routing_result_fixture())
    dispatching = mark_task_dispatching(initial, "task_1111111111111111")
    assert dispatch_eligible_task_ids(dispatching) == ()
    assert dispatching.orchestrator.status == "dispatching"
    assert dispatching.next_wakeup.model_dump() == {
        "target": "worker_dispatch",
        "reason": "dispatch_in_progress",
    }
    assert dispatching.artifacts["candidate_context_table"].status == "planned"
    assert mark_task_dispatching(dispatching, "task_1111111111111111") == dispatching

    dispatched = mark_task_dispatched(dispatching, "task_1111111111111111")
    running = mark_task_running(dispatched, "task_1111111111111111")
    assert mark_task_dispatched(dispatched, "task_1111111111111111") == dispatched
    assert mark_task_running(running, "task_1111111111111111") == running
    assert dispatch_eligible_task_ids(dispatched) == ()
    assert dispatch_eligible_task_ids(running) == ()
    for state in (dispatched, running):
        assert state.run_status == "running"
        assert state.orchestrator.status == "waiting_for_workers"
        assert state.next_wakeup.model_dump() == {
            "target": "orchestrator_loop",
            "reason": "worker_result_received",
        }
        assert state.orchestrator.next_wakeup_reason == "worker_result_received"
        assert state.routing.decisions["route_1111111111111111"].status == (
            "dispatched"
        )
        assert state.artifacts["candidate_context_table"].status == "producing"
    assert running.worker_tasks["task_1111111111111111"].task_id == (
        initial.worker_tasks["task_1111111111111111"].task_id
    )
    assert running.routing.routing_plan_id == initial.routing.routing_plan_id
    assert running.worker_tasks["task_1111111111111111"].routing_decision_id == (
        "route_1111111111111111"
    )


def _received_productive_result(state, task_id, artifact_name, artifact_id):
    dispatched = mark_task_dispatched(mark_task_dispatching(state, task_id), task_id)
    return mark_task_result(
        dispatched,
        task_id,
        result_status="success",
        output_artifact_refs={artifact_name: artifact_id},
        available_output_artifact_names=frozenset({artifact_name}),
    )


def test_new_result_explicitly_enters_evaluating_but_completed_recompute_is_stable():
    received = _received_productive_result(
        execution_state_from_routing_result(routing_result_fixture()),
        "task_1111111111111111",
        "candidate_context_table",
        "candidate_context_table_aaaaaaaaaaaa",
    )
    assert received.run_status == "running"
    assert received.orchestrator.status == "evaluating_results"
    assert received.next_wakeup.model_dump() == {
        "target": "orchestrator_loop",
        "reason": "worker_result_received",
    }

    completed = received.model_copy(
        update={
            "run_status": "completed",
            "orchestrator": received.orchestrator.model_copy(
                update={
                    "status": "completed",
                    "next_wakeup_reason": "routing_completed",
                }
            ),
            "next_wakeup": NextWakeupState(
                target="final_response", reason="routing_completed"
            ),
        }
    )
    completed = OrchestratorExecutionState.model_validate(completed.model_dump())
    assert recompute_aggregate_state(completed) == completed

    replayed = mark_task_result(
        completed,
        "task_1111111111111111",
        result_status="success",
        output_artifact_refs={
            "candidate_context_table": "candidate_context_table_aaaaaaaaaaaa"
        },
        available_output_artifact_names=frozenset({"candidate_context_table"}),
    )
    assert replayed == completed


def test_historical_terminal_tasks_do_not_mask_future_ready_or_waiting_state():
    independent = execution_state_from_routing_result(
        routing_result_fixture(independent_ready=True)
    )
    received = _received_productive_result(
        independent,
        "task_1111111111111111",
        "candidate_context_table",
        "candidate_context_table_aaaaaaaaaaaa",
    )
    future_ready = recompute_aggregate_state(received)
    assert future_ready.orchestrator.status == "dispatching"
    assert future_ready.next_wakeup.reason == "ready_tasks_available"
    assert dispatch_eligible_task_ids(future_ready) == (
        "task_2222222222222222",
    )

    waiting = _received_productive_result(
        execution_state_from_routing_result(routing_result_fixture()),
        "task_1111111111111111",
        "candidate_context_table",
        "candidate_context_table_bbbbbbbbbbbb",
    )
    advanced_waiting = waiting.model_copy(
        update={
            "orchestrator": waiting.orchestrator.model_copy(
                update={
                    "status": "validating",
                    "next_wakeup_reason": "dependencies_pending",
                }
            ),
            "next_wakeup": NextWakeupState(
                target="orchestrator_loop", reason="dependencies_pending"
            ),
        }
    )
    future_waiting = recompute_aggregate_state(advanced_waiting)
    assert future_waiting.run_status == "running"
    assert future_waiting.orchestrator.status == "validating"
    assert future_waiting.next_wakeup.reason == "dependencies_pending"


def test_dispatch_failed_requires_compact_transport_reason():
    initial = execution_state_from_routing_result(routing_result_fixture())
    dispatching = mark_task_dispatching(initial, "task_1111111111111111")
    failed = mark_task_dispatch_failed(
        dispatching, "task_1111111111111111", "dispatch_timeout"
    )
    assert failed.worker_tasks["task_1111111111111111"].dispatch_status == (
        "dispatch_failed"
    )
    assert failed.worker_tasks["task_1111111111111111"].execution_status == (
        "not_started"
    )
    assert failed.routing.decisions["route_1111111111111111"].status == "failed"
    assert failed.run_status == "running"
    assert failed.orchestrator.status == "evaluating_results"
    assert failed.next_wakeup.model_dump() == {
        "target": "orchestrator_loop",
        "reason": "dispatch_failed",
    }
    assert failed.orchestrator.next_wakeup_reason == "dispatch_failed"
    assert failed.artifacts["candidate_context_table"].status == "invalid"
    assert failed.artifacts["candidate_context_table"].producer_task_id == (
        "task_1111111111111111"
    )
    with pytest.raises(OrchestratorExecutionStateError, match="reason_invalid"):
        mark_task_dispatch_failed(dispatching, "task_1111111111111111", "raw timeout")


@pytest.mark.parametrize(
    ("source_builder", "target"),
    [
        (
            lambda state: mark_task_dispatched(
                mark_task_dispatching(state, "task_1111111111111111"),
                "task_1111111111111111",
            ),
            ("not_dispatched", "not_started"),
        ),
        (
            lambda state: mark_task_running(
                mark_task_dispatched(
                    mark_task_dispatching(state, "task_1111111111111111"),
                    "task_1111111111111111",
                ),
                "task_1111111111111111",
            ),
            ("dispatching", "not_started"),
        ),
    ],
)
def test_invalid_transitions_fail_without_partial_mutation(source_builder, target):
    source = source_builder(
        execution_state_from_routing_result(routing_result_fixture())
    )
    before = source.model_dump_json()
    with pytest.raises(OrchestratorExecutionStateError):
        transition_task_lifecycle(
            source,
            "task_1111111111111111",
            dispatch_status=target[0],
            execution_status=target[1],
        )
    assert source.model_dump_json() == before


def test_unknown_task_and_conflicting_identity_fail_without_mutation():
    state = execution_state_from_routing_result(routing_result_fixture())
    before = state.model_dump_json()
    with pytest.raises(OrchestratorExecutionStateError, match="task_id_unknown"):
        mark_task_dispatching(state, "task_unknown")
    assert state.model_dump_json() == before

    corrupt = state.model_copy(deep=True)
    corrupt.worker_tasks["task_1111111111111111"].agent_id = "wrong_agent"
    corrupt_before = corrupt.model_dump_json()
    with pytest.raises(OrchestratorExecutionStateError, match="identity_invalid"):
        mark_task_dispatching(corrupt, "task_1111111111111111")
    assert corrupt.model_dump_json() == corrupt_before


def test_schema_forbids_extra_fields_and_invalid_enums():
    state = execution_state_from_routing_result(routing_result_fixture())
    payload = state.model_dump()
    payload["worker_tasks"]["task_1111111111111111"]["queued_payload"] = (
        "forbidden"
    )
    with pytest.raises(ValidationError):
        OrchestratorExecutionState.model_validate(payload)

    payload = state.model_dump()
    payload["routing"]["decisions"]["route_1111111111111111"]["status"] = (
        "superseded"
    )
    with pytest.raises(ValidationError):
        OrchestratorExecutionState.model_validate(payload)


def test_blocked_plan_without_worker_fails_instead_of_waiting_for_worker():
    state = execution_state_from_routing_result(blocked_routing_result_fixture())
    assert state.run_status == "failed"
    assert state.orchestrator.status == "failed"
    assert state.next_wakeup.model_dump() == {
        "target": "orchestrator_loop",
        "reason": "routing_blocked",
    }
    assert state.orchestrator.next_wakeup_reason == "routing_blocked"
    assert state.worker_tasks == {}
    assert dispatch_eligible_task_ids(state) == ()
    assert state.artifacts["candidate_context_table"].status == "planned"


@pytest.mark.parametrize(
    "unsafe_ref",
    [
        "sk-live-RAW-SECRET",
        "mem_sk-live-RAW-SECRET",
        "mem_http://private.example",
        "mem_inputs/private.json",
        "mem_raw_sequence_ACDEFGHIKLMNPQRSTVWY",
        "mem_ACDEFGHIKLMNPQRSTVWY",
        "mem_full_prompt_private",
        "mem_raw_ToolUniverse_payload",
        "summary_worker_ACDEFGHIKLMNPQRSTVWY_0123456789ab",
        "mem_orchestrator_ATOMPRIVATEPDB_0123456789ab",
    ],
)
def test_typed_memory_refs_reject_raw_or_sensitive_values(unsafe_ref):
    state = execution_state_from_routing_result(routing_result_fixture())
    payload = state.model_dump()
    payload["memory_refs"] = {"completed_worker_summaries": [unsafe_ref]}
    with pytest.raises(ValidationError):
        OrchestratorExecutionState.model_validate(payload)


def test_typed_memory_refs_accept_only_declared_safe_ref_fields():
    state = execution_state_from_routing_result(routing_result_fixture())
    payload = state.model_dump()
    payload["memory_refs"] = {
        "orchestrator_run_summary": "mem_orchestrator_0123456789ab",
        "completed_worker_summaries": ["summary_worker_0123456789ab"],
        "final_response_context": "mem_final_response_0123456789ab",
    }
    payload["artifacts"]["candidate_context_table"]["safe_summary_ref"] = (
        "summary_artifact_0123456789ab"
    )
    validated = OrchestratorExecutionState.model_validate(payload)
    assert validated.memory_refs.model_dump() == payload["memory_refs"]
    assert validated.artifacts["candidate_context_table"].safe_summary_ref == (
        "summary_artifact_0123456789ab"
    )

    payload["memory_refs"]["raw_memory_body"] = "mem_forbidden_0123456789ab"
    with pytest.raises(ValidationError):
        OrchestratorExecutionState.model_validate(payload)


def test_identity_refs_wakeup_and_routing_source_are_strict():
    state = execution_state_from_routing_result(routing_result_fixture())
    mutations = [
        (("run_id",), "run_not_production_format"),
        (("routing", "routing_plan_id"), "wrp_sk-live-secret"),
        (
            ("routing", "decisions", "route_1111111111111111", "agent_id"),
            "Step_05_Invalid",
        ),
        (
            (
                "routing",
                "decisions",
                "route_1111111111111111",
                "capability_id",
            ),
            "step/05/private",
        ),
        (("routing", "routing_source"), "mock"),
        (
            ("artifacts", "raw_request_record", "artifact_id"),
            "raw_request_record_sk_live_secret",
        ),
        (("next_wakeup", "target"), "http://private-worker"),
        (("next_wakeup", "reason"), "arbitrary_raw_reason"),
    ]
    for path, invalid_value in mutations:
        payload = state.model_dump()
        current = payload
        for key in path[:-1]:
            current = current[key]
        current[path[-1]] = invalid_value
        with pytest.raises(ValidationError):
            OrchestratorExecutionState.model_validate(payload)


def test_next_wakeup_reason_must_be_synchronized_including_none():
    state = execution_state_from_routing_result(routing_result_fixture())
    payload = state.model_dump()
    payload["next_wakeup"] = None
    payload["orchestrator"]["next_wakeup_reason"] = None
    validated = OrchestratorExecutionState.model_validate(payload)
    assert validated.next_wakeup is None
    assert validated.orchestrator.next_wakeup_reason is None

    payload["orchestrator"]["next_wakeup_reason"] = "ready_tasks_available"
    with pytest.raises(ValidationError, match="next_wakeup_reason_mismatch"):
        OrchestratorExecutionState.model_validate(payload)
