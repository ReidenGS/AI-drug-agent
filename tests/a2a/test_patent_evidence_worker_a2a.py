"""Turn H2 real localhost HTTP A2A tests.

LLM and MCP fixtures are deterministic and local, not live evidence. The
production AgentCard, adapter, planner, validator, resolver, domain core,
persistence, registry, and HTTP transport paths remain unchanged.
"""

from __future__ import annotations

import contextlib
import json
import threading

import pytest
import requests
from python_a2a import A2AClient, Message, MessageRole, Task, TaskState, TextContent
from werkzeug.serving import make_server

from app.a2a.contracts import (
    A2ATaskMetadata,
    InputArtifactRef,
    InputProjection,
    OrchestratorRoutingDecisionRef,
    PrivacyConstraints,
    RetryContext,
    WorkerExecutionRequest,
    WorkerRequestSpec,
)
from app.a2a.patent_evidence_worker import (
    PatentEvidenceA2AWorker,
    create_patent_evidence_flask_app,
)
from app.agents.patent_evidence_agent import PatentEvidenceAgent
from app.mcp import tooluniverse_adapter
from app.mcp.client import LocalMCPClient
from app.schemas.step_02_structured_query import (
    NormalizedEntity,
    SourceRawRequestRef,
    StructuredQuery,
    TaskIntent,
)
from app.schemas.step_05_candidate_context_table import (
    CandidateContextTable,
    CandidateRecord,
    Identifier,
    Material,
)
from app.services.workflow_state_service import WorkflowStateService
from app.utils.ids import new_artifact_id


RUN_ID = "run_20260718_1234abcd"
_REAL_GET_UNIVERSE = tooluniverse_adapter._get_universe
_SQ_FIELDS = [
    "task_intent",
    "referenced_inputs",
    "normalized_entities",
    "entity_decompositions",
    "requested_outputs",
    "canonical_query",
]
_CCT_FIELDS = ["candidate_records", "downstream_query_hints"]


@pytest.fixture(autouse=True)
def _isolated_http_and_real_metadata(monkeypatch):
    monkeypatch.setattr(tooluniverse_adapter, "_get_universe", _REAL_GET_UNIVERSE)
    for variable in ("NO_PROXY", "no_proxy"):
        monkeypatch.setenv(variable, "127.0.0.1,localhost")
    for variable in (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "http_proxy",
        "https_proxy",
        "ALL_PROXY",
        "all_proxy",
    ):
        monkeypatch.delenv(variable, raising=False)


class _LLM:
    def __init__(self, *, fail_all=False):
        self.calls = 0
        self.fail_all = fail_all
        self.last_payload = None

    def generate_json(self, prompt, *, schema=None, system=None):
        self.calls += 1
        self.last_payload = schema
        refs = schema["input_refs"]
        query = next(
            (ref["ref_id"] for ref in refs if ref["role"] == "query"), None
        )
        cid = next(
            (ref["ref_id"] for ref in refs if ref["role"] == "pubchem_cid"),
            None,
        )
        lanes = schema["search_scope"]["requested_lanes"]
        plans = []
        planned_lanes = set()
        if "evidence" in lanes and query is not None:
            plans.append(
                {
                    "tool_name": "EuropePMC_search_articles",
                    "can_invoke": True,
                    "argument_mappings": [{"schema_arg": "query", "input_ref_id": query}],
                    "argument_literals": [],
                    "missing_required_args": [],
                    "selection_reason": "http fixture evidence",
                }
            )
            planned_lanes.add("evidence")
        if "patent" in lanes and cid is not None:
            plans.append(
                {
                    "tool_name": "PubChem_get_associated_patents_by_CID",
                    "can_invoke": True,
                    "argument_mappings": [
                        {"schema_arg": "cid", "input_ref_id": cid}
                    ],
                    "argument_literals": [],
                    "missing_required_args": [],
                    "selection_reason": "http fixture patent",
                }
            )
            planned_lanes.add("patent")
        return {
            "lane_assessments": [
                {
                    "search_lane": lane,
                    "status": "planned" if lane in planned_lanes else "missing_inputs",
                    "reason": "http fixture",
                }
                for lane in lanes
            ],
            "tool_plans": plans,
        }


class _RejectionLLM(_LLM):
    def generate_json(self, prompt, *, schema=None, system=None):
        response = super().generate_json(prompt, schema=schema, system=system)
        query_ref = next(
            ref["ref_id"] for ref in schema["input_refs"] if ref["role"] == "query"
        )
        response["tool_plans"].extend(
            [
                {
                    "tool_name": "MODEL_SECRET_UNKNOWN_TOOL",
                    "can_invoke": True,
                    "argument_mappings": [],
                    "argument_literals": [],
                    "missing_required_args": [],
                    "selection_reason": "fixture rejection",
                },
                {
                    "tool_name": "EuropePMC_search_articles",
                    "can_invoke": True,
                    "argument_mappings": [
                        {
                            "schema_arg": "MODEL_SECRET_SCHEMA_ARG",
                            "input_ref_id": query_ref,
                        }
                    ],
                    "argument_literals": [],
                    "missing_required_args": [],
                    "selection_reason": "fixture rejection",
                },
                {
                    "tool_name": "EuropePMC_search_articles",
                    "can_invoke": True,
                    "argument_mappings": [
                        {
                            "schema_arg": "query",
                            "input_ref_id": "MODEL_SECRET_REF",
                        }
                    ],
                    "argument_literals": [],
                    "missing_required_args": [],
                    "selection_reason": "fixture rejection",
                },
            ]
        )
        return response


class _AllMissingLLM(_LLM):
    def generate_json(self, prompt, *, schema=None, system=None):
        self.calls += 1
        self.last_payload = schema
        return {
            "lane_assessments": [
                {
                    "search_lane": lane,
                    "status": "missing_inputs",
                    "reason": "fixture contradiction",
                }
                for lane in schema["search_scope"]["requested_lanes"]
            ],
            "tool_plans": [],
        }


def _bindings(*, fail_all=False):
    def evidence(**_kwargs):
        if fail_all:
            return {"status": "upstream_error", "error_message": "fixture failure"}
        return {
            "status": "ok",
            "executor": "test_fixture",
            "payload": {"results": [{"title": "ADC fixture", "doi": "10.1/http"}]},
        }

    def patent(**_kwargs):
        if fail_all:
            return {"status": "failed", "error_message": "fixture failure"}
        return {
            "status": "ok",
            "executor": "test_fixture",
            "payload": {
                "data": {"Record": {"Reference": [{"SourceName": "fixture"}]}}
            },
        }

    return {
        "EuropePMC_search_articles": evidence,
        "PubChem_get_associated_patents_by_CID": patent,
    }


def _seed(
    local_storage,
    registry_service,
    *,
    lanes=("evidence", "patent"),
    include_pubchem=True,
    production_drug_shape=False,
    canonical_query="PRIVATE_QUERY_SENTINEL HER2 ADC",
):
    registry_service.init_registry(RUN_ID)
    WorkflowStateService(local_storage).init_run(RUN_ID)
    sq_id = new_artifact_id("structured_query")
    cct_id = new_artifact_id("candidate_context_table")
    requested_outputs = []
    if "evidence" in lanes:
        requested_outputs.append("literature_review_summary")
    if "patent" in lanes:
        requested_outputs.append("patent_or_ip_summary")
    sq = {
        "artifact_id": sq_id,
        **StructuredQuery(
            run_id=RUN_ID,
            parsed_at="2026-07-18T00:00:00Z",
            source_raw_request_ref=SourceRawRequestRef(
                raw_request_record_id="raw_request_record_fixture"
            ),
            task_intent=TaskIntent(
                task_type="patent_evidence_fixture",
                primary_intent="new_adc_design",
            ),
            referenced_inputs=[],
            requested_outputs=requested_outputs,
            normalized_entities=(
                [
                    NormalizedEntity(
                        original_text="T-DM1",
                        canonical_name="trastuzumab emtansine",
                        entity_type="drug",
                        explicit_or_inferred="explicit",
                        confidence=1.0,
                    )
                ]
                if production_drug_shape
                else []
            ),
            canonical_query=canonical_query,
        ).model_dump(),
    }
    cct = {
        "artifact_id": cct_id,
        **CandidateContextTable(
            run_id=RUN_ID,
            created_at="2026-07-18T00:00:00Z",
            context_build_status="ok",
            candidate_records=[
                CandidateRecord(
                    candidate_id="cand_payload",
                    candidate_label="Fixture payload candidate",
                    candidate_type="compound_component",
                    identifiers=(
                        [
                            Identifier(
                                id_type="pubchem_cid",
                                id_value="12345",
                                confidence=1.0,
                            )
                        ]
                        if include_pubchem
                        else []
                    ),
                    materials=[
                        Material(
                            material_id="mat_payload",
                            material_type="payload_name",
                            value="MMAE",
                            role="payload",
                        )
                    ],
                )
            ],
            downstream_query_hints=(
                [
                    {
                        "entity": "trastuzumab emtansine",
                        "role": "complete_adc",
                        "explicit_or_inferred": "explicit",
                        "source": "normalized_entity",
                    }
                ]
                if production_drug_shape
                else []
            ),
        ).model_dump(),
    }
    local_storage.write_json(local_storage.run_key(RUN_ID, "inputs", "structured_query.json"), sq)
    local_storage.write_json(local_storage.run_key(RUN_ID, "candidate_context_table.json"), cct)
    registry_service.update_active(
        RUN_ID,
        structured_query_id=sq_id,
        candidate_context_table_id=cct_id,
    )
    return sq, cct


def _refs(registry_service):
    active = registry_service.get(RUN_ID).active_artifacts
    return {
        "structured_query": InputArtifactRef(
            artifact_id=active.structured_query_id,
            run_id=RUN_ID,
            artifact_type="structured_query",
            field_keys=_SQ_FIELDS,
            can_read_from_db=True,
        ),
        "candidate_context_table": InputArtifactRef(
            artifact_id=active.candidate_context_table_id,
            run_id=RUN_ID,
            artifact_type="candidate_context_table",
            entity_type="candidate",
            selection_mode="all_in_artifact",
            field_keys=_CCT_FIELDS,
            can_read_from_db=True,
        ),
    }


def _request(registry_service):
    return WorkerExecutionRequest(
        payload_type="worker_execution_request",
        payload_version="v1",
        run_id=RUN_ID,
        task_id="task_patent_evidence_001",
        routing_plan_id="wrp_patent_evidence_001",
        routing_decision_id="route_patent_evidence_001",
        agent_id=PatentEvidenceA2AWorker.AGENT_ID,
        capability_id=PatentEvidenceA2AWorker.CAPABILITY_ID,
        created_by="step_04_orchestrator_planner",
        worker_request=WorkerRequestSpec(objective="Review evidence and patents"),
        orchestrator_routing_decision=OrchestratorRoutingDecisionRef(
            planned_status="run",
            dispatch_mode="python_a2a",
            expected_outputs=[
                "scientific_evidence_table",
                "patent_prior_art_table",
            ],
        ),
        input_projection=InputProjection(
            compact_inputs={},
            input_artifact_refs=_refs(registry_service),
        ),
        privacy_constraints=PrivacyConstraints(),
    )


def _task(request, *, task_id=None):
    metadata = A2ATaskMetadata(
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
        id=task_id or request.task_id,
        message=Message(
            content=TextContent(text=request.model_dump_json()), role=MessageRole.USER
        ).to_dict(),
        metadata=metadata,
    )


def _result(task):
    return json.loads(task.artifacts[0]["parts"][0]["text"])


class _RecordingWorker(PatentEvidenceA2AWorker):
    def __init__(self, *args, **kwargs):
        self.agent_run_count = 0
        self.execute_threads = []
        super().__init__(*args, **kwargs)

    def execute_request(self, request):
        self.execute_threads.append(threading.current_thread().name)
        return super().execute_request(request)

    def _default_agent_factory(self):
        outer = self
        real = PatentEvidenceAgent(
            storage=self._storage,
            registry=self._registry,
            workflow_state=self._workflow_state,
            mcp_client=self._mcp_client,
            llm=self._llm,
        )

        class Counting:
            def run_from_artifacts(self, *args, **kwargs):
                outer.agent_run_count += 1
                return real.run_from_artifacts(*args, **kwargs)

        return Counting()


class _TamperingWorker(_RecordingWorker):
    def __init__(self, *args, output_name, identity_field, **kwargs):
        self.output_name = output_name
        self.identity_field = identity_field
        super().__init__(*args, **kwargs)

    def _default_agent_factory(self):
        outer = self
        real = PatentEvidenceAgent(
            storage=self._storage,
            registry=self._registry,
            workflow_state=self._workflow_state,
            mcp_client=self._mcp_client,
            llm=self._llm,
        )

        class Tampering:
            def run_from_artifacts(self, *args, **kwargs):
                outer.agent_run_count += 1
                result = real.run_from_artifacts(*args, **kwargs)
                path = {
                    "scientific_evidence_table": "scientific_evidence_table.json",
                    "patent_prior_art_table": "patent_prior_art_table.json",
                }[outer.output_name]
                key = outer._storage.run_key(RUN_ID, path)
                body = outer._storage.read_json(key)
                body[outer.identity_field] = "tampered_test_only"
                outer._storage.write_json(key, body)
                return result

        return Tampering()


class _SchemaTamperingWorker(_RecordingWorker):
    def _default_agent_factory(self):
        outer = self
        real = PatentEvidenceAgent(
            storage=self._storage,
            registry=self._registry,
            workflow_state=self._workflow_state,
            mcp_client=self._mcp_client,
            llm=self._llm,
        )

        class Tampering:
            def run_from_artifacts(self, *args, **kwargs):
                outer.agent_run_count += 1
                result = real.run_from_artifacts(*args, **kwargs)
                key = outer._storage.run_key(
                    RUN_ID, "scientific_evidence_table.json"
                )
                body = outer._storage.read_json(key)
                body["review_status"] = 42
                outer._storage.write_json(key, body)
                return result

        return Tampering()


@contextlib.contextmanager
def _serve(worker):
    app = create_patent_evidence_flask_app(worker)
    hits = {"card": 0, "health": 0, "task": 0}

    @app.before_request
    def count_http():
        from flask import request

        if "agent" in request.path and "json" in request.path:
            hits["card"] += 1
        elif request.path == "/health":
            hits["health"] += 1
        elif request.path in {"/tasks/send", "/a2a/tasks/send"}:
            hits["task"] += 1

    server = make_server("127.0.0.1", 0, app, threaded=False)
    thread = threading.Thread(
        target=server.serve_forever,
        name="patent-evidence-worker-http",
        daemon=True,
    )
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", hits
    finally:
        server.shutdown()
        thread.join(timeout=5)


def _worker(local_storage, registry_service, workflow_state_service, *, fail_all=False, cls=_RecordingWorker, **kwargs):
    llm = kwargs.pop("llm", None) or _LLM(fail_all=fail_all)
    bindings = kwargs.pop("bindings", None) or _bindings(fail_all=fail_all)
    return cls(
        url="http://patent-evidence-worker:8014",
        storage=local_storage,
        registry=registry_service,
        workflow_state=workflow_state_service,
        mcp_client=LocalMCPClient(bindings=bindings),
        llm=llm,
        **kwargs,
    )


async def _send(url, task):
    return await A2AClient(url).send_task_async(task)


async def test_real_http_two_lane_success_counts_and_privacy(
    local_storage, registry_service, workflow_state_service
):
    _seed(local_storage, registry_service)
    worker = _worker(local_storage, registry_service, workflow_state_service)
    with _serve(worker) as (url, hits):
        assert requests.get(f"{url}/health", timeout=5).status_code == 200
        request = _request(registry_service)
        returned = await _send(url, _task(request))
        result = _result(returned)
    assert returned.status.state == TaskState.COMPLETED
    assert result["result_status"] == "success"
    assert result["tool_call_summary"] == {
        "attempted": 2,
        "success": 2,
        "failed": 0,
        "dependency_unavailable": 0,
        "skipped": 0,
    }
    assert set(result["output_artifact_refs"]) == {
        "scientific_evidence_table",
        "patent_prior_art_table",
    }
    assert worker.agent_run_count == 1
    assert worker.execute_threads == ["patent-evidence-worker-http"]
    assert hits == {"card": 1, "health": 1, "task": 1}
    state = workflow_state_service.get(RUN_ID)
    assert state["steps"]["step_13"] == "completed"
    assert state["steps"]["step_14"] == "completed"
    active = registry_service.get(RUN_ID).active_artifacts
    assert (
        result["output_artifact_refs"]["scientific_evidence_table"]["artifact_id"]
        == active.scientific_evidence_table_id
    )
    assert (
        result["output_artifact_refs"]["patent_prior_art_table"]["artifact_id"]
        == active.patent_prior_art_table_id
    )
    persisted_records = []
    for path in ("scientific_evidence_table.json", "patent_prior_art_table.json"):
        body = local_storage.read_json(local_storage.run_key(RUN_ID, path))
        persisted_records.extend(body["tool_call_records"])
    assert len(persisted_records) == result["tool_call_summary"]["attempted"]
    patent_body = local_storage.read_json(
        local_storage.run_key(RUN_ID, "patent_prior_art_table.json")
    )
    assert patent_body["patent_records"] == []
    assert patent_body["lookup_summaries"][0]["source_type"] == (
        "pubchem_associated_reference"
    )
    compact = json.dumps(result)
    for forbidden in (
        "PRIVATE_QUERY_SENTINEL",
        "MMAE",
        "raw_envelope",
        "scientific_evidence_table.json",
        "http://patent-evidence-worker",
        "sk-test-api-key",
    ):
        assert forbidden not in compact


async def test_http_evidence_only_and_no_requested_lane(
    local_storage, registry_service, workflow_state_service
):
    _seed(local_storage, registry_service, lanes=("evidence",))
    worker = _worker(local_storage, registry_service, workflow_state_service)
    request = _request(registry_service)
    request = request.model_copy(
        update={
            "input_projection": request.input_projection.model_copy(
                update={"compact_inputs": {"requested_lanes": ["patent"]}}
            )
        }
    )
    with _serve(worker) as (url, _hits):
        evidence = await _send(url, _task(request))
    evidence_result = _result(evidence)
    assert evidence.status.state == TaskState.COMPLETED
    assert evidence_result["compact_summary"]["lane_statuses"]["patent"] == "not_requested"
    assert evidence_result["tool_call_summary"]["attempted"] == 1
    state = workflow_state_service.get(RUN_ID)
    assert state["steps"]["step_13"] == "completed"
    assert state["steps"]["step_14"] == "skipped"
    _seed(local_storage, registry_service, lanes=())
    with _serve(worker) as (url, _hits):
        no_lane = await _send(url, _task(_request(registry_service)))
    failed = _result(no_lane)
    assert no_lane.status.state == TaskState.FAILED
    assert failed["error_code"] == "patent_evidence_no_requested_lane"
    assert failed["output_artifact_refs"] == {}
    assert worker.agent_run_count == 1


async def test_http_patent_only_is_derived_from_structured_query(
    local_storage, registry_service, workflow_state_service
):
    _seed(local_storage, registry_service, lanes=("patent",))
    worker = _worker(local_storage, registry_service, workflow_state_service)
    request = _request(registry_service)
    assert request.input_projection.compact_inputs == {}
    with _serve(worker) as (url, _hits):
        returned = await _send(url, _task(request))
    result = _result(returned)
    assert returned.status.state == TaskState.COMPLETED
    assert result["compact_summary"]["lane_statuses"] == {
        "evidence": "not_requested",
        "patent": "completed",
    }
    assert result["tool_call_summary"]["attempted"] == 1
    patent_body = local_storage.read_json(
        local_storage.run_key(RUN_ID, "patent_prior_art_table.json")
    )
    assert patent_body["patent_records"] == []
    assert patent_body["lookup_summaries"][0]["source_type"] == (
        "pubchem_associated_reference"
    )
    state = workflow_state_service.get(RUN_ID)
    assert state["steps"]["step_13"] == "skipped"
    assert state["steps"]["step_14"] == "completed"


async def test_production_drug_shape_has_evidence_refs_but_no_patent_identifier(
    local_storage, registry_service, workflow_state_service
):
    _seed(
        local_storage,
        registry_service,
        include_pubchem=False,
        production_drug_shape=True,
    )
    llm = _LLM()
    worker = _worker(
        local_storage,
        registry_service,
        workflow_state_service,
        llm=llm,
    )
    with _serve(worker) as (url, _hits):
        returned = await _send(url, _task(_request(registry_service)))
    result = _result(returned)
    assert returned.status.state == TaskState.COMPLETED
    assert result["result_status"] == "partial"
    assert result["compact_summary"]["lane_statuses"] == {
        "evidence": "ok",
        "patent": "failed",
    }
    refs = llm.last_payload["input_refs"]
    assert [(ref["role"], ref["supports_tool_args"]) for ref in refs] == [
        ("query", ["query", "research_topic"]),
        ("complete_adc", ["query", "research_topic"]),
    ]
    assert not {"brand_name", "application_number", "pubchem_cid"} & {
        ref["role"] for ref in refs
    }
    evidence = local_storage.read_json(
        local_storage.run_key(RUN_ID, "scientific_evidence_table.json")
    )
    audit = evidence["patent_evidence_planning_audit"]
    assert {
        key: audit[key]
        for key in (
            "catalog_visible_count",
            "eligible_count",
            "selected_count",
            "accepted_count",
            "executed_count",
        )
    } == {
        "catalog_visible_count": 11,
        "eligible_count": 4,
        "selected_count": 1,
        "accepted_count": 1,
        "executed_count": 1,
    }
    assert [
        (item["search_lane"], item["status"])
        for item in audit["lane_assessments"]
    ] == [("evidence", "planned"), ("patent", "missing_inputs")]
    assert result["tool_call_summary"] == {
        "attempted": 1,
        "success": 1,
        "failed": 0,
        "dependency_unavailable": 0,
        "skipped": 0,
    }


async def test_all_missing_inputs_is_blocked_and_executes_no_mcp(
    local_storage, registry_service, workflow_state_service
):
    _seed(
        local_storage,
        registry_service,
        include_pubchem=False,
        canonical_query=None,
    )
    calls = {"mcp": 0}

    def forbidden(**_kwargs):
        calls["mcp"] += 1
        raise AssertionError("blocked planning must not execute MCP")

    worker = _worker(
        local_storage,
        registry_service,
        workflow_state_service,
        bindings={
            "EuropePMC_search_articles": forbidden,
            "PubChem_get_associated_patents_by_CID": forbidden,
        },
    )
    with _serve(worker) as (url, _hits):
        returned = await _send(url, _task(_request(registry_service)))
    result = _result(returned)
    assert returned.status.state == TaskState.FAILED
    assert result["result_status"] == "blocked"
    assert result["execution_status"] == "failed"
    assert result["error_code"] == "patent_evidence_inputs_unavailable"
    assert result["output_artifact_refs"] == {}
    assert result["tool_call_summary"] == {
        "attempted": 0,
        "success": 0,
        "failed": 0,
        "dependency_unavailable": 0,
        "skipped": 0,
    }
    assert calls == {"mcp": 0}
    evidence = local_storage.read_json(
        local_storage.run_key(RUN_ID, "scientific_evidence_table.json")
    )
    audit = evidence["patent_evidence_planning_audit"]
    assert (
        audit["eligible_count"],
        audit["selected_count"],
        audit["accepted_count"],
        audit["executed_count"],
    ) == (0, 0, 0, 0)
    assert [item["status"] for item in audit["lane_assessments"]] == [
        "missing_inputs",
        "missing_inputs",
    ]


async def test_eligible_query_and_cid_cannot_be_mislabeled_blocked(
    local_storage, registry_service, workflow_state_service
):
    _seed(local_storage, registry_service)
    calls = {"mcp": 0}

    def forbidden(**_kwargs):
        calls["mcp"] += 1
        raise AssertionError("contradictory planning must fail before MCP")

    llm = _AllMissingLLM()
    worker = _worker(
        local_storage,
        registry_service,
        workflow_state_service,
        llm=llm,
        bindings={
            "EuropePMC_search_articles": forbidden,
            "PubChem_get_associated_patents_by_CID": forbidden,
        },
    )
    with _serve(worker) as (url, _hits):
        returned = await _send(url, _task(_request(registry_service)))
    result = _result(returned)
    assert returned.status.state == TaskState.FAILED
    assert result["result_status"] == "tool_failed"
    assert result["error_code"] == "worker_execution_error"
    assert result["error_code"] != "patent_evidence_inputs_unavailable"
    assert result["output_artifact_refs"] == {}
    assert calls == {"mcp": 0}
    assert llm.calls == 1
    roles = {ref["role"] for ref in llm.last_payload["input_refs"]}
    assert {"query", "pubchem_cid"} <= roles


async def test_rejected_llm_plans_are_sanitized_and_degrade_success_to_partial(
    local_storage, registry_service, workflow_state_service
):
    _seed(local_storage, registry_service, lanes=("evidence",))
    worker = _worker(
        local_storage,
        registry_service,
        workflow_state_service,
        llm=_RejectionLLM(),
    )
    with _serve(worker) as (url, _hits):
        returned = await _send(url, _task(_request(registry_service)))
    result = _result(returned)
    assert returned.status.state == TaskState.COMPLETED
    assert result["result_status"] == "partial"
    evidence = local_storage.read_json(
        local_storage.run_key(RUN_ID, "scientific_evidence_table.json")
    )
    audit = evidence["patent_evidence_planning_audit"]
    assert audit["selected_count"] == 4
    assert audit["accepted_count"] == 1
    assert audit["rejected_count"] == 3
    assert audit["executed_count"] == 1
    assert audit["rejections"] == [
        {"tool_name": "unknown_tool", "reason": "unknown_tool"},
        {
            "tool_name": "EuropePMC_search_articles",
            "reason": "unknown_schema_arg",
        },
        {
            "tool_name": "EuropePMC_search_articles",
            "reason": "unknown_input_ref_id",
        },
    ]
    serialized = json.dumps(audit)
    for forbidden in (
        "MODEL_SECRET_UNKNOWN_TOOL",
        "MODEL_SECRET_SCHEMA_ARG",
        "MODEL_SECRET_REF",
    ):
        assert forbidden not in serialized


async def test_http_all_tools_failed_maps_failed(
    local_storage, registry_service, workflow_state_service
):
    _seed(local_storage, registry_service)
    worker = _worker(
        local_storage, registry_service, workflow_state_service, fail_all=True
    )
    with _serve(worker) as (url, _hits):
        returned = await _send(url, _task(_request(registry_service)))
    result = _result(returned)
    assert returned.status.state == TaskState.FAILED
    assert result["result_status"] == "tool_failed"
    assert result["error_code"] == "patent_evidence_workflow_failed"
    assert result["output_artifact_refs"] == {}
    assert result["tool_call_summary"]["failed"] == 2
    state = workflow_state_service.get(RUN_ID)
    assert state["steps"]["step_13"] == "failed"
    assert state["steps"]["step_14"] == "failed"


async def test_http_partial_upstream_failure_is_completed_partial(
    local_storage, registry_service, workflow_state_service
):
    _seed(local_storage, registry_service)

    class PartialLLM(_LLM):
        def generate_json(self, prompt, *, schema=None, system=None):
            response = super().generate_json(prompt, schema=schema, system=system)
            query_ref = next(
                ref["ref_id"] for ref in schema["input_refs"] if ref["role"] == "query"
            )
            response["tool_plans"].append(
                {
                    "tool_name": "SemanticScholar_search_papers",
                    "can_invoke": True,
                    "argument_mappings": [
                        {"schema_arg": "query", "input_ref_id": query_ref}
                    ],
                    "argument_literals": [],
                    "missing_required_args": [],
                    "selection_reason": "partial fixture",
                }
            )
            return response

    bindings = _bindings()
    bindings["SemanticScholar_search_papers"] = lambda **_kwargs: {
        "status": "upstream_error",
        "error_message": "fixture maintenance",
    }
    worker = _worker(
        local_storage,
        registry_service,
        workflow_state_service,
        llm=PartialLLM(),
        bindings=bindings,
    )
    with _serve(worker) as (url, _hits):
        returned = await _send(url, _task(_request(registry_service)))
    result = _result(returned)
    assert returned.status.state == TaskState.COMPLETED
    assert result["result_status"] == "partial"
    assert result["compact_summary"]["lane_statuses"]["evidence"] == "partial"
    assert result["tool_call_summary"] == {
        "attempted": 3,
        "success": 2,
        "failed": 1,
        "dependency_unavailable": 0,
        "skipped": 0,
    }
    assert result["compact_summary"]["non_success_tools"] == [
        {"tool_name": "SemanticScholar_search_papers", "run_status": "failed"}
    ]


@pytest.mark.parametrize("case", ["missing_ref", "not_ready", "schema_invalid"])
async def test_required_artifact_contract_failure_runs_no_agent_or_mcp(
    local_storage, registry_service, workflow_state_service, case
):
    _seed(local_storage, registry_service)
    request = _request(registry_service)
    if case == "missing_ref":
        refs = dict(request.input_projection.input_artifact_refs)
        refs.pop("structured_query")
        request = request.model_copy(
            update={
                "input_projection": request.input_projection.model_copy(
                    update={"input_artifact_refs": refs}
                )
            }
        )
    elif case == "not_ready":
        key = local_storage.run_key(RUN_ID, "candidate_context_table.json")
        body = local_storage.read_json(key)
        body["context_build_status"] = "failed"
        local_storage.write_json(key, body)
    else:
        key = local_storage.run_key(RUN_ID, "inputs", "structured_query.json")
        body = local_storage.read_json(key)
        body["task_intent"]["secondary_intents"] = "not_a_list"
        local_storage.write_json(key, body)
    worker = _worker(local_storage, registry_service, workflow_state_service)
    with _serve(worker) as (url, _hits):
        returned = await _send(url, _task(request))
    result = _result(returned)
    assert returned.status.state == TaskState.FAILED
    assert result["result_status"] == "validation_failed"
    if case == "schema_invalid":
        assert result["error_code"] == "input_artifact_schema_invalid"
    assert result["output_artifact_refs"] == {}
    assert worker.agent_run_count == 0


@pytest.mark.parametrize("artifact_name", ["structured_query", "candidate_context_table"])
@pytest.mark.parametrize("identity_field", ["artifact_id", "run_id"])
async def test_input_identity_tampering_never_runs_agent(
    local_storage, registry_service, workflow_state_service, artifact_name, identity_field
):
    _seed(local_storage, registry_service)
    path = {
        "structured_query": ("inputs", "structured_query.json"),
        "candidate_context_table": ("candidate_context_table.json",),
    }[artifact_name]
    key = local_storage.run_key(RUN_ID, *path)
    body = local_storage.read_json(key)
    body[identity_field] = "tampered_test_only"
    local_storage.write_json(key, body)
    worker = _worker(local_storage, registry_service, workflow_state_service)
    with _serve(worker) as (url, _hits):
        returned = await _send(url, _task(_request(registry_service)))
    result = _result(returned)
    assert returned.status.state == TaskState.FAILED
    assert result["error_code"] == "input_artifact_identity_mismatch"
    assert result["output_artifact_refs"] == {}
    assert worker.agent_run_count == 0


@pytest.mark.parametrize(
    "output_name", ["scientific_evidence_table", "patent_prior_art_table"]
)
@pytest.mark.parametrize("identity_field", ["artifact_id", "run_id"])
async def test_output_identity_tampering_fails_without_refs(
    local_storage,
    registry_service,
    workflow_state_service,
    output_name,
    identity_field,
):
    _seed(local_storage, registry_service)
    worker = _worker(
        local_storage,
        registry_service,
        workflow_state_service,
        cls=_TamperingWorker,
        output_name=output_name,
        identity_field=identity_field,
    )
    with _serve(worker) as (url, _hits):
        returned = await _send(url, _task(_request(registry_service)))
    result = _result(returned)
    assert returned.status.state == TaskState.FAILED
    assert result["error_code"] == "patent_evidence_output_identity_mismatch"
    assert result["output_artifact_refs"] == {}


async def test_output_typed_schema_corruption_fails_without_refs(
    local_storage, registry_service, workflow_state_service
):
    _seed(local_storage, registry_service)
    worker = _worker(
        local_storage,
        registry_service,
        workflow_state_service,
        cls=_SchemaTamperingWorker,
    )
    with _serve(worker) as (url, _hits):
        returned = await _send(url, _task(_request(registry_service)))
    result = _result(returned)
    assert returned.status.state == TaskState.FAILED
    assert result["error_code"] == "patent_evidence_output_schema_invalid"
    assert result["output_artifact_refs"] == {}


async def test_transport_task_id_mismatch_never_runs_agent(
    local_storage, registry_service, workflow_state_service
):
    _seed(local_storage, registry_service)
    worker = _worker(local_storage, registry_service, workflow_state_service)
    with _serve(worker) as (url, _hits):
        returned = await _send(
            url, _task(_request(registry_service), task_id="task_transport_mismatch")
        )
    result = _result(returned)
    assert returned.status.state == TaskState.FAILED
    assert result["error_code"] == "task_transport_id_mismatch"
    assert worker.agent_run_count == 0


async def test_retry_parent_identity_round_trips(
    local_storage, registry_service, workflow_state_service
):
    _seed(local_storage, registry_service)
    request = _request(registry_service).model_copy(
        update={
            "retry_context": RetryContext(
                retry_of_task_id="task_patent_evidence_parent",
                retry_attempt=1,
                max_retry_attempts=3,
                retry_reason="transient_worker_failure",
            )
        }
    )
    worker = _worker(local_storage, registry_service, workflow_state_service)
    with _serve(worker) as (url, _hits):
        returned = await _send(url, _task(request))
    result = _result(returned)
    assert returned.status.state == TaskState.COMPLETED
    assert result["retry_of_task_id"] == "task_patent_evidence_parent"
