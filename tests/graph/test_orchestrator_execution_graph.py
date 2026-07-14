"""Real StateGraph + InMemorySaver tests for compact local reconstruction."""

from __future__ import annotations

import copy
import inspect
import json

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from app.a2a.orchestrator_execution_state import (
    dispatch_eligible_task_ids,
    execution_state_from_routing_result,
    mark_task_dispatched,
    mark_task_dispatching,
    mark_task_running,
)
from app.graph.orchestrator_execution_graph import (
    OrchestratorExecutionGraphError,
    build_orchestrator_execution_graph,
    execution_graph_config,
)
from app.schemas.orchestrator_execution_state import OrchestratorExecutionState
from tests.a2a.test_orchestrator_execution_state import (
    _SENSITIVE,
    routing_result_fixture,
)

_CONFIG_ATTACKS = (
    "configurable_api_key",
    "metadata_authorization",
    "metadata_raw_sequence",
    "metadata_full_prompt",
    "configurable_extra",
)


def _assert_compact_error(exc, *, code, sentinel):
    assert str(exc) == code
    assert exc.args == (code,)
    for rendered in (str(exc), repr(exc), repr(exc.args)):
        assert sentinel.lower() not in rendered.lower()


def _malicious_config(attack, run_id, sentinel):
    minimal = execution_graph_config(run_id)
    if attack == "configurable_api_key":
        minimal["configurable"]["api_key"] = sentinel
    elif attack == "metadata_authorization":
        minimal["metadata"] = {"authorization": sentinel}
    elif attack == "metadata_raw_sequence":
        minimal["metadata"] = {"raw_sequence": sentinel}
    elif attack == "metadata_full_prompt":
        minimal["metadata"] = {"full_prompt": sentinel}
    elif attack == "configurable_extra":
        minimal["configurable"]["unexpected_ref"] = sentinel
    else:  # pragma: no cover - parametrization is closed above
        raise AssertionError("unknown_config_attack")
    return minimal


def test_execution_graph_config_is_exact_minimal_contract():
    assert execution_graph_config("run_20260714_abcdef12") == {
        "configurable": {"thread_id": "run_20260714_abcdef12"}
    }


@pytest.mark.parametrize("execution_phase", ["dispatched", "running"])
def test_real_in_memory_checkpoint_reconstructs_compact_dispatch_state(
    execution_phase,
):
    initial = execution_state_from_routing_result(routing_result_fixture())
    dispatched = mark_task_dispatched(
        mark_task_dispatching(initial, "task_1111111111111111"),
        "task_1111111111111111",
    )
    checkpointed = (
        mark_task_running(dispatched, "task_1111111111111111")
        if execution_phase == "running"
        else dispatched
    )
    saver = InMemorySaver()
    config = execution_graph_config(checkpointed.run_id)
    graph = build_orchestrator_execution_graph(checkpointer=saver)

    graph.invoke(checkpointed.model_dump(), config=config)
    snapshot = graph.get_state(config)
    restored = OrchestratorExecutionState.model_validate(snapshot.values)
    assert restored.worker_tasks["task_1111111111111111"].dispatch_status == (
        "dispatched"
    )
    assert (
        restored.worker_tasks["task_1111111111111111"].execution_status
        == ("running" if execution_phase == "running" else "not_started")
    )
    assert dispatch_eligible_task_ids(restored) == ()
    assert restored.orchestrator.status == "waiting_for_workers"
    assert restored.next_wakeup.model_dump() == {
        "target": "orchestrator_loop",
        "reason": "worker_result_received",
    }
    assert restored.artifacts["candidate_context_table"].status == "producing"

    reconstructed_graph = build_orchestrator_execution_graph(checkpointer=saver)
    reconstructed_values = reconstructed_graph.invoke(None, config=config)
    reconstructed = OrchestratorExecutionState.model_validate(reconstructed_values)
    assert reconstructed == restored
    assert reconstructed.routing.routing_plan_id == "wrp_0123456789abcdef"
    assert reconstructed.worker_tasks["task_1111111111111111"].routing_decision_id == (
        "route_1111111111111111"
    )


def test_checkpoint_values_have_only_compact_schema_keys_and_no_sensitive_body():
    state = execution_state_from_routing_result(routing_result_fixture())
    saver = InMemorySaver()
    config = execution_graph_config(state.run_id)
    graph = build_orchestrator_execution_graph(checkpointer=saver)
    graph.invoke(state, config=config)
    values = graph.get_state(config).values

    assert sorted(values) == [
        "artifacts",
        "memory_refs",
        "next_wakeup",
        "orchestrator",
        "routing",
        "run_id",
        "run_status",
        "worker_tasks",
    ]
    serialized = json.dumps(values, sort_keys=True)
    for value in _SENSITIVE.values():
        assert value.lower() not in serialized.lower()
    for forbidden in (
        "prepareda2atask",
        "workerexecutionresult",
        "workerexecutionrequest",
        "task.message",
        "dispatch_url",
        "storage_path",
    ):
        assert forbidden not in serialized.lower()


def test_execution_graph_has_no_dispatch_or_worker_call_path():
    source = inspect.getsource(build_orchestrator_execution_graph)
    for forbidden in (
        "A2AClient",
        "send_task",
        "send_task_async",
        "execute_request",
        "generate_json",
        "call_tool",
    ):
        assert forbidden not in source


def test_malicious_memory_ref_is_rejected_before_any_checkpoint_write():
    state = execution_state_from_routing_result(routing_result_fixture())
    saver = InMemorySaver()
    config = execution_graph_config(state.run_id)
    graph = build_orchestrator_execution_graph(checkpointer=saver)
    graph.invoke(state.model_dump(), config=config)
    before = list(saver.list(None))

    sentinel = "sk-live-RAW-SECRET"
    malicious = state.model_dump()
    malicious["memory_refs"] = {"completed_worker_summaries": [sentinel]}
    with pytest.raises(OrchestratorExecutionGraphError) as caught:
        graph.invoke(malicious, config=config)
    _assert_compact_error(
        caught.value, code="checkpoint_state_input_invalid", sentinel=sentinel
    )

    after = list(saver.list(None))
    assert len(after) == len(before)
    assert sentinel.lower() not in repr(after).lower()
    for checkpoint_tuple in after:
        assert sentinel.lower() not in repr(checkpoint_tuple.checkpoint).lower()


@pytest.mark.parametrize("entrypoint", ["invoke", "stream"])
def test_sync_public_entrypoints_reject_malicious_input_without_checkpoint_write(
    entrypoint,
):
    state = execution_state_from_routing_result(routing_result_fixture())
    saver = InMemorySaver()
    config = execution_graph_config(state.run_id)
    graph = build_orchestrator_execution_graph(checkpointer=saver)
    sentinel = "sk-live-RAW-SECRET"
    malicious = state.model_dump()
    malicious["memory_refs"] = {"completed_worker_summaries": [sentinel]}
    before = list(saver.list(None))

    with pytest.raises(OrchestratorExecutionGraphError) as caught:
        result = getattr(graph, entrypoint)(malicious, config=config)
        if entrypoint == "stream":
            list(result)
    _assert_compact_error(
        caught.value, code="checkpoint_state_input_invalid", sentinel=sentinel
    )

    after = list(saver.list(None))
    assert len(after) == len(before) == 0
    assert sentinel.lower() not in repr(after).lower()


@pytest.mark.parametrize("entrypoint", ["ainvoke", "astream"])
async def test_async_public_entrypoints_reject_malicious_input_without_checkpoint_write(
    entrypoint,
):
    state = execution_state_from_routing_result(routing_result_fixture())
    saver = InMemorySaver()
    config = execution_graph_config(state.run_id)
    graph = build_orchestrator_execution_graph(checkpointer=saver)
    sentinel = "sk-live-RAW-SECRET"
    malicious = state.model_dump()
    malicious["memory_refs"] = {"completed_worker_summaries": [sentinel]}
    before = list(saver.list(None))

    with pytest.raises(OrchestratorExecutionGraphError) as caught:
        if entrypoint == "ainvoke":
            await graph.ainvoke(malicious, config=config)
        else:
            async for _ in graph.astream(malicious, config=config):
                pass
    _assert_compact_error(
        caught.value, code="checkpoint_state_input_invalid", sentinel=sentinel
    )

    after = list(saver.list(None))
    assert len(after) == len(before) == 0
    assert sentinel.lower() not in repr(after).lower()


@pytest.mark.parametrize("entrypoint", ["invoke", "stream"])
@pytest.mark.parametrize("attack", _CONFIG_ATTACKS)
def test_sync_config_attacks_are_compact_and_never_checkpointed(entrypoint, attack):
    state = execution_state_from_routing_result(routing_result_fixture())
    saver = InMemorySaver()
    graph = build_orchestrator_execution_graph(checkpointer=saver)
    sentinel = f"sk-live-CONFIG-SECRET-{attack}"
    config = _malicious_config(attack, state.run_id, sentinel)

    with pytest.raises(OrchestratorExecutionGraphError) as caught:
        result = getattr(graph, entrypoint)(state, config=config)
        if entrypoint == "stream":
            list(result)

    _assert_compact_error(
        caught.value, code="checkpoint_config_invalid", sentinel=sentinel
    )
    checkpoints = list(saver.list(None))
    assert checkpoints == []
    assert sentinel.lower() not in repr(checkpoints).lower()


@pytest.mark.parametrize("entrypoint", ["ainvoke", "astream"])
@pytest.mark.parametrize("attack", _CONFIG_ATTACKS)
async def test_async_config_attacks_are_compact_and_never_checkpointed(
    entrypoint, attack
):
    state = execution_state_from_routing_result(routing_result_fixture())
    saver = InMemorySaver()
    graph = build_orchestrator_execution_graph(checkpointer=saver)
    sentinel = f"sk-live-CONFIG-SECRET-{attack}"
    config = _malicious_config(attack, state.run_id, sentinel)

    with pytest.raises(OrchestratorExecutionGraphError) as caught:
        if entrypoint == "ainvoke":
            await graph.ainvoke(state, config=config)
        else:
            async for _ in graph.astream(state, config=config):
                pass

    _assert_compact_error(
        caught.value, code="checkpoint_config_invalid", sentinel=sentinel
    )
    checkpoints = list(saver.list(None))
    assert checkpoints == []
    assert sentinel.lower() not in repr(checkpoints).lower()


async def test_read_only_state_methods_reject_unsanitized_config_without_writes():
    state = execution_state_from_routing_result(routing_result_fixture())
    saver = InMemorySaver()
    minimal = execution_graph_config(state.run_id)
    graph = build_orchestrator_execution_graph(checkpointer=saver)
    graph.invoke(state, config=minimal)
    before = list(saver.list(None))
    sentinel = "sk-live-CONFIG-SECRET-read-methods"
    malicious = _malicious_config("metadata_authorization", state.run_id, sentinel)

    for method in (graph.get_state, graph.get_state_history):
        with pytest.raises(OrchestratorExecutionGraphError) as caught:
            result = method(malicious)
            if method == graph.get_state_history:
                list(result)
        _assert_compact_error(
            caught.value, code="checkpoint_config_invalid", sentinel=sentinel
        )
    for method in (graph.aget_state, graph.aget_state_history):
        with pytest.raises(OrchestratorExecutionGraphError) as caught:
            if method == graph.aget_state:
                await method(malicious)
            else:
                async for _ in method(malicious):
                    pass
        _assert_compact_error(
            caught.value, code="checkpoint_config_invalid", sentinel=sentinel
        )

    after = list(saver.list(None))
    assert len(after) == len(before)
    assert sentinel.lower() not in repr(after).lower()


async def test_external_state_mutation_is_explicitly_unsupported_and_atomic():
    state = execution_state_from_routing_result(routing_result_fixture())
    saver = InMemorySaver()
    config = execution_graph_config(state.run_id)
    graph = build_orchestrator_execution_graph(checkpointer=saver)
    graph.invoke(state, config=config)
    before = list(saver.list(None))
    malicious = {"memory_refs": {"completed_worker_summaries": ["raw sentinel"]}}

    with pytest.raises(
        OrchestratorExecutionGraphError, match="external_state_mutation_unsupported"
    ):
        graph.update_state(config, malicious)
    with pytest.raises(
        OrchestratorExecutionGraphError, match="external_state_mutation_unsupported"
    ):
        await graph.aupdate_state(config, malicious)
    with pytest.raises(
        OrchestratorExecutionGraphError, match="external_state_mutation_unsupported"
    ):
        graph.bulk_update_state(config, [[malicious]])
    with pytest.raises(
        OrchestratorExecutionGraphError, match="external_state_mutation_unsupported"
    ):
        await graph.abulk_update_state(config, [[malicious]])

    after = list(saver.list(None))
    assert len(after) == len(before)
    assert "raw sentinel" not in repr(after).lower()
    assert not hasattr(graph, "batch")
    assert not hasattr(graph, "abatch")
    assert not hasattr(graph, "with_config")
    assert not hasattr(graph, "_compiled")


def test_cross_run_and_malformed_thread_ids_fail_before_checkpoint_write():
    state = execution_state_from_routing_result(routing_result_fixture())
    saver = InMemorySaver()
    graph = build_orchestrator_execution_graph(checkpointer=saver)
    other_config = execution_graph_config("run_20260714_beadfeed")

    for entrypoint in ("invoke", "stream"):
        with pytest.raises(
            OrchestratorExecutionGraphError,
            match="checkpoint_thread_identity_mismatch",
        ):
            result = getattr(graph, entrypoint)(state, config=other_config)
            if entrypoint == "stream":
                list(result)
    assert list(saver.list(other_config)) == []

    malformed = {"configurable": {"thread_id": "run_not_valid"}}
    with pytest.raises(
        OrchestratorExecutionGraphError, match="checkpoint_config_invalid"
    ):
        graph.invoke(state, config=malformed)
    with pytest.raises(
        OrchestratorExecutionGraphError, match="orchestrator_execution_run_id_invalid"
    ):
        execution_graph_config("run_not_valid")
    with pytest.raises(
        OrchestratorExecutionGraphError, match="checkpoint_config_invalid"
    ):
        graph.invoke(state)
    with pytest.raises(
        OrchestratorExecutionGraphError, match="checkpoint_config_invalid"
    ):
        graph.invoke(state, config={"configurable": {}})
    assert list(saver.list(None)) == []


@pytest.mark.parametrize("entrypoint", ["ainvoke", "astream"])
async def test_async_cross_run_identity_mismatch_never_writes_checkpoint(entrypoint):
    state = execution_state_from_routing_result(routing_result_fixture())
    saver = InMemorySaver()
    graph = build_orchestrator_execution_graph(checkpointer=saver)
    other_config = execution_graph_config("run_20260714_beadfeed")

    with pytest.raises(
        OrchestratorExecutionGraphError, match="checkpoint_thread_identity_mismatch"
    ):
        if entrypoint == "ainvoke":
            await graph.ainvoke(state, config=other_config)
        else:
            async for _ in graph.astream(state, config=other_config):
                pass
    assert list(saver.list(None)) == []


async def test_resume_rejects_checkpoint_whose_state_run_id_differs_from_namespace():
    state = execution_state_from_routing_result(routing_result_fixture())
    saver = InMemorySaver()
    source_config = execution_graph_config(state.run_id)
    graph = build_orchestrator_execution_graph(checkpointer=saver)
    graph.invoke(state, config=source_config)
    source_tuple = saver.get_tuple(source_config)
    assert source_tuple is not None

    wrong_config = execution_graph_config("run_20260714_beadfeed")
    saver_seed_config = {
        "configurable": {
            "thread_id": wrong_config["configurable"]["thread_id"],
            "checkpoint_ns": "",
        }
    }
    saver.put(
        saver_seed_config,
        source_tuple.checkpoint,
        source_tuple.metadata,
        source_tuple.checkpoint["channel_versions"],
    )
    before = list(saver.list(wrong_config))
    with pytest.raises(
        OrchestratorExecutionGraphError, match="checkpoint_thread_identity_mismatch"
    ):
        graph.invoke(None, config=wrong_config)
    with pytest.raises(
        OrchestratorExecutionGraphError, match="checkpoint_thread_identity_mismatch"
    ):
        await graph.ainvoke(None, config=wrong_config)
    after = list(saver.list(wrong_config))
    assert len(after) == len(before)
    assert all(
        state.run_id in repr(item.checkpoint)
        for item in after
        if item.checkpoint.get("channel_values")
    )


def test_invalid_stored_snapshot_uses_compact_error_without_leaking_value():
    state = execution_state_from_routing_result(routing_result_fixture())
    source_saver = InMemorySaver()
    config = execution_graph_config(state.run_id)
    source_graph = build_orchestrator_execution_graph(checkpointer=source_saver)
    source_graph.invoke(state, config=config)
    source_tuple = source_saver.get_tuple(config)
    assert source_tuple is not None

    sentinel = "sk-live-STORED-SNAPSHOT-SECRET"
    corrupted = copy.deepcopy(source_tuple.checkpoint)
    corrupted["channel_values"]["memory_refs"] = {
        "completed_worker_summaries": [sentinel]
    }
    corrupted_saver = InMemorySaver()
    seed_config = {
        "configurable": {"thread_id": state.run_id, "checkpoint_ns": ""}
    }
    corrupted_saver.put(
        seed_config,
        corrupted,
        source_tuple.metadata,
        corrupted["channel_versions"],
    )
    graph = build_orchestrator_execution_graph(checkpointer=corrupted_saver)
    before = list(corrupted_saver.list(None))

    with pytest.raises(OrchestratorExecutionGraphError) as caught:
        graph.invoke(None, config=config)
    _assert_compact_error(
        caught.value, code="checkpoint_state_invalid", sentinel=sentinel
    )
    after = list(corrupted_saver.list(None))
    assert len(after) == len(before)


async def test_safe_opaque_memory_refs_checkpoint_and_reconstruct_sync_and_async():
    state = execution_state_from_routing_result(routing_result_fixture())
    payload = state.model_dump()
    payload["memory_refs"] = {
        "orchestrator_run_summary": "mem_orchestrator_0123456789ab",
        "completed_worker_summaries": ["summary_worker_0123456789ab"],
        "final_response_context": "mem_final_response_0123456789ab",
    }
    with_memory = OrchestratorExecutionState.model_validate(payload)
    saver = InMemorySaver()
    config = execution_graph_config(with_memory.run_id)
    graph = build_orchestrator_execution_graph(checkpointer=saver)
    graph.invoke(with_memory, config=config)
    await graph.ainvoke(with_memory, config=config)
    assert OrchestratorExecutionState.model_validate(
        graph.get_state(config).values
    ).run_id == with_memory.run_id
    assert OrchestratorExecutionState.model_validate(
        (await graph.aget_state(config)).values
    ).run_id == with_memory.run_id
    assert list(graph.get_state_history(config))
    assert [item async for item in graph.aget_state_history(config)]

    reconstructed = build_orchestrator_execution_graph(checkpointer=saver)
    restored = OrchestratorExecutionState.model_validate(
        reconstructed.invoke(None, config=config)
    )
    assert restored.memory_refs == with_memory.memory_refs
    assert restored.orchestrator.next_wakeup_reason == restored.next_wakeup.reason
