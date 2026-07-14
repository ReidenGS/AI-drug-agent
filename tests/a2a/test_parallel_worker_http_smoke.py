"""Business-level parallel A2A smoke over the production worker cores.

MockLLMProvider and LocalMCPClient bindings are deterministic test/offline
fixtures. They do not prove live LLM, MCP, ToolUniverse, or biomedical-tool
success, and production contains no mock-success fallback for this test.
"""

from __future__ import annotations

import json
import socket
import threading
import time
from collections import Counter

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from werkzeug.serving import make_server

from app.a2a.agent_cards import (
    AGENT_ID_STEP5,
    AGENT_ID_STEP6,
    AGENT_ID_STRUCTURE,
    CAP_STEP5_CANDIDATE_CONTEXT,
    CAP_STEP6_DEVELOPABILITY,
    CAP_STRUCTURE_DESIGN_WORKFLOW,
)
from app.a2a.orchestrator_discovery import (
    ExpectedWorkerEndpoint,
    WorkerDiscoveryService,
)
from app.a2a.orchestrator_execution_loop import (
    execute_orchestrator_worker_loop,
)
from app.a2a.orchestrator_execution_state import (
    execution_state_from_routing_result,
)
from app.a2a.orchestrator_routing_service import OrchestratorRoutingService
from app.a2a.step5_worker import create_step5_flask_app
from app.a2a.step6_worker import create_step6_flask_app
from app.a2a.structure_worker import create_structure_flask_app
from app.agents.supervisor_agent import SupervisorAgent
from app.graph.orchestrator_execution_graph import (
    build_orchestrator_execution_graph,
    execution_graph_config,
)
from app.llm.provider import MockLLMProvider
from app.services.artifact_registry_service import ArtifactRegistryService
from app.services.input_readiness_service import InputReadinessService
from app.services.intake_service import IntakeService
from app.services.storage_local import LocalStorage
from app.services.structured_query_service import StructuredQueryService
from app.services.workflow_state_service import WorkflowStateService
from tests.a2a.test_orchestrator_dispatch import _RecordingStep5, _local_mcp
from tests.a2a.test_orchestrator_routing_intent import QUERY
from tests.a2a.test_step6_worker_a2a import (
    _RecordingStep6Worker,
    _success_mcp,
)
from tests.a2a.test_structure_worker_a2a import (
    _RecordingStructureWorker,
    _auditable_local_mcp,
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


class _TimedStep5(_RecordingStep5):
    def execute_request(self, request):
        started = time.monotonic()
        try:
            return super().execute_request(request)
        finally:
            self.window = (started, time.monotonic())


class _TimedStep6(_RecordingStep6Worker):
    def execute_request(self, request):
        started = time.monotonic()
        try:
            return super().execute_request(request)
        finally:
            self.window = (started, time.monotonic())


class _TimedStructure(_RecordingStructureWorker):
    def execute_request(self, request):
        started = time.monotonic()
        try:
            return super().execute_request(request)
        finally:
            self.window = (started, time.monotonic())


class _Handle:
    def __init__(self, url, worker, server, thread, hits):
        self.url = url
        self.worker = worker
        self.server = server
        self.thread = thread
        self.hits = hits

    def close(self):
        self.server.shutdown()
        self.thread.join(timeout=5)


def _serve(worker_type, app_factory, *, storage, registry, workflow, mcp):
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    url = f"http://127.0.0.1:{port}"
    worker = worker_type(
        url=url,
        storage=storage,
        registry=registry,
        workflow_state=workflow,
        mcp_client=mcp,
        llm=MockLLMProvider(),
    )
    app = app_factory(worker)
    hits = Counter()

    @app.before_request
    def _count():
        from flask import request

        if "agent.json" in request.path:
            hits["card"] += 1
        elif request.path == "/health":
            hits["health"] += 1
        elif request.path in {"/tasks/send", "/a2a/tasks/send"}:
            hits["task"] += 1

    httpd = make_server("127.0.0.1", port, app, threaded=True)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    return _Handle(url, worker, httpd, thread, hits)


def _independent_services(local_storage):
    storage = LocalStorage(str(local_storage.root), local_storage.prefix)
    return (
        storage,
        ArtifactRegistryService(storage),
        WorkflowStateService(storage),
    )


def _record_status(record):
    status = record.get("run_status")
    if status in {"success", "dependency_unavailable", "skipped", "not_run"}:
        return "skipped" if status in {"skipped", "not_run"} else status
    return "failed"


def _tool_records(artifacts):
    records = list(artifacts["candidate_context_table"].get("tool_call_records", []))
    for candidate in artifacts["structured_liability_summary"].get(
        "candidate_liability_results", []
    ):
        for lane in candidate.get("lane_results", []):
            records.extend(lane.get("tool_call_records", []))
    records.extend(
        artifacts["prepared_structure_input_package"].get(
            "structure_tool_call_records", []
        )
    )
    records.extend(
        artifacts["structure_prediction_and_interface_results"].get(
            "tool_call_records", []
        )
    )
    records.extend(
        artifacts["structure_variant_and_compound_screening"].get(
            "tool_call_records", []
        )
    )
    return records


@pytest.mark.asyncio
async def test_real_three_worker_http_parallel_smoke_without_retry(local_storage):
    base_storage, base_registry, base_workflow = _independent_services(local_storage)
    record = IntakeService(base_storage, base_registry, base_workflow).submit(
        raw_user_query=QUERY,
        user_provided_context={
            "target_or_antigen_text": "HER2",
            "candidate_text": "trastuzumab-like antibody",
            "payload_linker_text": "vc-MMAE",
        },
    )
    StructuredQueryService(
        base_storage,
        base_registry,
        base_workflow,
        SupervisorAgent(llm=MockLLMProvider()),
    ).parse(record.run_id)
    assert InputReadinessService(
        base_storage, base_registry, base_workflow
    ).check(record.run_id).input_readiness_status == "ready"

    step5_services = _independent_services(local_storage)
    step6_services = _independent_services(local_storage)
    structure_services = _independent_services(local_storage)
    step5 = _serve(
        _TimedStep5,
        create_step5_flask_app,
        storage=step5_services[0],
        registry=step5_services[1],
        workflow=step5_services[2],
        mcp=_local_mcp(),
    )
    step6 = _serve(
        _TimedStep6,
        create_step6_flask_app,
        storage=step6_services[0],
        registry=step6_services[1],
        workflow=step6_services[2],
        mcp=_success_mcp(),
    )
    structure = _serve(
        _TimedStructure,
        create_structure_flask_app,
        storage=structure_services[0],
        registry=structure_services[1],
        workflow=structure_services[2],
        mcp=_auditable_local_mcp(),
    )
    try:
        discovery = WorkerDiscoveryService(
            expected_workers=[
                ExpectedWorkerEndpoint(
                    AGENT_ID_STEP5,
                    (CAP_STEP5_CANDIDATE_CONTEXT,),
                    step5.url,
                ),
                ExpectedWorkerEndpoint(
                    AGENT_ID_STEP6,
                    (CAP_STEP6_DEVELOPABILITY,),
                    step6.url,
                ),
                ExpectedWorkerEndpoint(
                    AGENT_ID_STRUCTURE,
                    (CAP_STRUCTURE_DESIGN_WORKFLOW,),
                    structure.url,
                ),
            ],
            storage=base_storage,
            registry=base_registry,
            discovery_timeout_seconds=3,
            health_timeout_seconds=3,
        )
        routing_service = OrchestratorRoutingService(
            discovery=discovery,
            storage=base_storage,
            registry=base_registry,
            llm=MockLLMProvider(),
        )
        routing = routing_service.plan_for_run(record.run_id)
        initial_decisions = {
            item.agent_id: item.validation_status
            for item in routing.plan.validated_decisions
        }
        assert initial_decisions == {
            AGENT_ID_STEP5: "ready",
            AGENT_ID_STEP6: "waiting_for_dependencies",
            AGENT_ID_STRUCTURE: "waiting_for_dependencies",
        }
        assert [item.decision.agent_id for item in routing.prepared_tasks] == [
            AGENT_ID_STEP5
        ]

        state = execution_state_from_routing_result(routing)
        saver = InMemorySaver()
        graph = build_orchestrator_execution_graph(checkpointer=saver)
        loop = await execute_orchestrator_worker_loop(
            run_id=record.run_id,
            state=state,
            prepared_tasks=routing.prepared_tasks,
            routing_service=routing_service,
            discovery=discovery,
            registry=base_registry,
            storage=base_storage,
            execution_graph=graph,
            checkpoint_config=execution_graph_config(record.run_id),
            timeout_seconds=60,
            max_worker_retries=3,
        )
    finally:
        step5.close()
        step6.close()
        structure.close()

    assert loop.outcome == "completed"
    assert loop.dispatch_round_count == 2
    assert loop.dispatch_attempt_count == 3
    assert step5.hits["task"] == step6.hits["task"] == structure.hits["task"] == 1
    assert step5.hits["card"] == step6.hits["card"] == structure.hits["card"] == 3
    assert step5.hits["health"] == step6.hits["health"] == structure.hits["health"] == 1
    assert all(task.retry_attempt == 0 for task in loop.state.worker_tasks.values())
    assert len(loop.state.worker_tasks) == 3
    assert step6.worker.window[0] < structure.worker.window[1]
    assert structure.worker.window[0] < step6.worker.window[1]

    active = base_registry.get(record.run_id).active_artifacts
    artifact_specs = {
        "candidate_context_table": (
            "candidate_context_table.json",
            active.candidate_context_table_id,
        ),
        "structured_liability_summary": (
            "structured_liability_summary.json",
            active.structured_liability_summary_id,
        ),
        "prepared_structure_input_package": (
            "prepared_structure_input_package.json",
            active.prepared_structure_input_package_id,
        ),
        "structure_prediction_and_interface_results": (
            "structure_prediction_and_interface_results.json",
            active.structure_prediction_and_interface_results_id,
        ),
        "structure_variant_and_compound_screening": (
            "compound_screening_artifact.json",
            active.structure_variant_and_compound_screening_id,
        ),
    }
    artifacts = {}
    for name, (path, active_id) in artifact_specs.items():
        assert active_id is not None
        body = base_storage.read_json(base_storage.run_key(record.run_id, path))
        assert body["artifact_id"] == active_id
        assert body["run_id"] == record.run_id
        artifacts[name] = body

    workflow = base_workflow.get(record.run_id)
    assert {
        key: workflow["steps"][key]
        for key in ("step_05", "step_06", "step_07", "step_08", "step_09")
    } == {
        "step_05": "completed",
        "step_06": "completed",
        "step_07": "completed",
        "step_08": "completed",
        "step_09": "completed",
    }
    proofs = {
        task.agent_id: loop.completion_proofs[task.task_id]
        for task in loop.state.worker_tasks.values()
    }
    assert set(proofs) == {AGENT_ID_STEP5, AGENT_ID_STEP6, AGENT_ID_STRUCTURE}
    assert all(proof.execution_status == "completed" for proof in proofs.values())
    assert all(proof.result_status in {"success", "partial"} for proof in proofs.values())
    assert all(proof.error_code is None for proof in proofs.values())

    records = _tool_records(artifacts)
    distribution = Counter(_record_status(item) for item in records)
    success_tools = sorted(
        item.get("tool_name")
        for item in records
        if item.get("run_status") == "success"
    )
    non_success = [
        {
            "tool_name": item.get("tool_name"),
            "status": item.get("run_status"),
            "reason": item.get("error_code") or item.get("error_message"),
        }
        for item in records
        if item.get("run_status") != "success"
    ]
    step9 = artifacts["structure_variant_and_compound_screening"]
    audit = step9
    initial_task_ids = {
        str(item.task.id) for item in routing.prepared_tasks
    }
    second_round_agents = sorted(
        item.decision.agent_id
        for task_id, item in loop.prepared_task_history.items()
        if task_id not in initial_task_ids
    )
    assert second_round_agents == sorted([AGENT_ID_STEP6, AGENT_ID_STRUCTURE])
    inspection = {
        "initial_decisions": initial_decisions,
        "dependency_edges": [
            edge.model_dump() for edge in routing.plan.dependency_edges
        ],
        "prepared_rounds": [[AGENT_ID_STEP5], second_round_agents],
        "posts": {
            AGENT_ID_STEP5: step5.hits["task"],
            AGENT_ID_STEP6: step6.hits["task"],
            AGENT_ID_STRUCTURE: structure.hits["task"],
        },
        "windows": {
            AGENT_ID_STEP5: step5.worker.window,
            AGENT_ID_STEP6: step6.worker.window,
            AGENT_ID_STRUCTURE: structure.worker.window,
        },
        "proofs": {
            agent_id: {
                "task_id": proof.task_id,
                "attempt": loop.state.worker_tasks[proof.task_id].retry_attempt,
                "result_status": proof.result_status,
                "execution_status": proof.execution_status,
                "error_code": proof.error_code,
            }
            for agent_id, proof in proofs.items()
        },
        "artifact_status": {
            name: body.get("context_build_status")
            or body.get("prefilter_status")
            or body.get("structure_preparation_status")
            or body.get("structure_modeling_status")
            or {
                "design_status": body.get("design_status"),
                "screening_status": body.get("screening_status"),
            }
            for name, body in artifacts.items()
        },
        "tool_status_distribution": dict(distribution),
        "success_tools": success_tools,
        "non_success_tools": non_success,
        "step9_counts": {
            "selected": len(audit.get("step9_stage1_selected_tools", [])),
            "mapped": len(audit.get("step9_stage2_mapped_tools", [])),
            "uninvokable": len(
                audit.get("step9_stage2_uninvokable_tools", [])
            ),
            "executed": len(audit.get("step9_runtime_executed_tools", [])),
        },
        "step9_selected_tools": audit.get("step9_stage1_selected_tools", []),
        "step9_mapped_tools": audit.get("step9_stage2_mapped_tools", []),
        "step9_uninvokable_details": audit.get(
            "step9_stage2_uninvokable_tool_details", []
        ),
        "step9_executed_tools": audit.get("step9_runtime_executed_tools", []),
        "test_only_mocked_local_bindings": sorted(
            {
                "SAbDab_search_structures",
                "ChEMBL_search_molecules",
                "ChEMBL_search_substructure",
                "DrugProps_pains_filter",
                "RCSBData_get_entry",
                "RCSBData_get_assembly",
                "get_refinement_resolution_by_pdb_id",
                "PDBePISA_get_interfaces",
            }
        ),
    }
    serialized = json.dumps(inspection, sort_keys=True, default=str)
    print("PARALLEL_SMOKE_INSPECTION=" + serialized)
    checkpoint_blob = repr(list(saver.list(None)))
    for forbidden in (
        "WorkerExecutionResult",
        "WorkerExecutionRequest",
        "PreparedA2ATask",
        "raw_tooluniverse_payload",
        "full_prompt",
        "raw_llm_response",
        "Authorization",
        "sk-live-",
        QUERY,
    ):
        assert forbidden not in checkpoint_blob
        assert forbidden not in routing.plan.model_dump_json()
        assert forbidden not in loop.state.model_dump_json()
