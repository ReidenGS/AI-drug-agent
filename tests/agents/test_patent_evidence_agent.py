"""Turn H2 unified domain-core tests; fixtures are not live evidence."""

from __future__ import annotations

from collections import Counter

import pytest
from pydantic import ValidationError

from app.agents.patent_evidence_agent import (
    PatentEvidenceAgent,
    _build_request,
    requested_lanes_from_structured_query,
)
from app.mcp import tooluniverse_adapter
from app.mcp.client import LocalMCPClient
from app.schemas.step_02_structured_query import (
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
from app.schemas.step_13_scientific_evidence_table import ScientificEvidenceTable


RUN_ID = "run_20260718_abcdef12"
_REAL_GET_UNIVERSE = tooluniverse_adapter._get_universe


@pytest.fixture(autouse=True)
def _use_real_metadata(monkeypatch):
    """Use installed ToolUniverse metadata; no biomedical API is called."""
    monkeypatch.setattr(tooluniverse_adapter, "_get_universe", _REAL_GET_UNIVERSE)


class _PlanningLLM:
    def __init__(self, *, evidence_tools=None, patent=True):
        self.calls = 0
        self.evidence_tools = evidence_tools or ["EuropePMC_search_articles"]
        self.patent = patent

    def generate_json(self, prompt, *, schema=None, system=None):
        self.calls += 1
        refs = schema["input_refs"]
        query_ref = next(ref["ref_id"] for ref in refs if ref["role"] == "query")
        cid_ref = next(
            (ref["ref_id"] for ref in refs if ref["role"] == "pubchem_cid"),
            None,
        )
        lanes = schema["search_scope"]["requested_lanes"]
        plans = []
        if "evidence" in lanes:
            for tool in self.evidence_tools:
                plans.append(
                    {
                        "tool_name": tool,
                        "can_invoke": True,
                        "argument_mappings": [
                            {"schema_arg": "query", "input_ref_id": query_ref}
                        ],
                        "argument_literals": [],
                        "missing_required_args": [],
                        "selection_reason": "test fixture evidence plan",
                    }
                )
        if "patent" in lanes and self.patent:
            plans.append(
                {
                    "tool_name": "PubChem_get_associated_patents_by_CID",
                    "can_invoke": True,
                    "argument_mappings": [
                        {"schema_arg": "cid", "input_ref_id": cid_ref}
                    ],
                    "argument_literals": [],
                    "missing_required_args": [],
                    "selection_reason": "test fixture regulatory plan",
                }
            )
        planned_lanes = {"evidence" if self.evidence_tools else None}
        if self.patent:
            planned_lanes.add("patent")
        assessments = [
            {
                "search_lane": lane,
                "status": "planned" if lane in planned_lanes else "missing_inputs",
                "reason": "test fixture assessment",
            }
            for lane in lanes
        ]
        return {"lane_assessments": assessments, "tool_plans": plans}


class _CapturingLLM(_PlanningLLM):
    def __init__(self):
        super().__init__()
        self.prompt = None
        self.payload = None
        self.system = None

    def generate_json(self, prompt, *, schema=None, system=None):
        self.prompt = prompt
        self.payload = schema
        self.system = system
        return super().generate_json(prompt, schema=schema, system=system)


def _artifacts(
    *,
    requested_outputs=("literature_review_summary", "patent_or_ip_summary"),
    primary_intent="new_adc_design",
    referenced_inputs=None,
    downstream_query_hints=None,
    materials=None,
    identifiers=None,
):
    structured = StructuredQuery(
        run_id=RUN_ID,
        parsed_at="2026-07-18T00:00:00Z",
        source_raw_request_ref=SourceRawRequestRef(
            raw_request_record_id="raw_request_record_fixture"
        ),
        task_intent=TaskIntent(
            task_type="patent_evidence_fixture",
            primary_intent=primary_intent,
        ),
        referenced_inputs=list(referenced_inputs or []),
        requested_outputs=list(requested_outputs),
        canonical_query="PRIVATE_CANONICAL_QUERY_SENTINEL",
    )
    candidate = CandidateContextTable(
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
                    if identifiers is None
                    else list(identifiers)
                ),
                materials=list(
                    materials
                    or [
                        Material(
                            material_id="mat_payload",
                            material_type="payload_name",
                            value="MMAE",
                            role="payload",
                        )
                    ]
                ),
            )
        ],
        downstream_query_hints=list(downstream_query_hints or []),
    )
    return {
        "structured_query": structured.model_dump(),
        "candidate_context_table": candidate.model_dump(),
    }


def _init(registry_service, workflow_state_service):
    registry_service.init_registry(RUN_ID)
    workflow_state_service.init_run(RUN_ID)


def _agent(
    local_storage, registry_service, workflow_state_service, bindings, llm
):
    return PatentEvidenceAgent(
        storage=local_storage,
        registry=registry_service,
        workflow_state=workflow_state_service,
        mcp_client=LocalMCPClient(bindings=bindings),
        llm=llm,
    )


def _ok_results(source):
    def call(**_kwargs):
        return {
            "status": "ok",
            "executor": "test_fixture",
            "payload": {
                "results": [
                    {"title": f"{source} ADC paper", "doi": f"10.1/{source}"}
                ]
            },
        }

    return call


def _pubchem(**_kwargs):
    return {
        "status": "ok",
        "executor": "test_fixture",
        "payload": {
            "data": {"Record": {"Reference": [{"SourceName": "fixture"}]}}
        },
    }


def test_two_lane_one_llm_call_and_persisted_identity(
    local_storage, registry_service, workflow_state_service
):
    _init(registry_service, workflow_state_service)
    llm = _PlanningLLM()
    result = _agent(
        local_storage,
        registry_service,
        workflow_state_service,
        {
            "EuropePMC_search_articles": _ok_results("europe"),
            "PubChem_get_associated_patents_by_CID": _pubchem,
        },
        llm,
    ).run_from_artifacts(RUN_ID, **_artifacts())
    assert llm.calls == 1
    assert result.evidence.review_status == "ok"
    assert result.patent.patent_review_status == "completed"
    assert len(result.evidence.evidence_records) == 1
    assert result.patent.patent_records == []
    assert result.patent.lookup_summaries[0].record_count == 1
    assert (
        result.patent.lookup_summaries[0].source_type
        == "pubchem_associated_reference"
    )
    assert result.planning_audit.catalog_visible_count == 11
    assert result.planning_audit.executed_count == 2
    state = workflow_state_service.get(RUN_ID)
    assert state["steps"]["step_13"] == "completed"
    assert state["steps"]["step_14"] == "completed"
    reg = registry_service.get(RUN_ID)
    for path, artifact_id in (
        ("scientific_evidence_table.json", reg.active_artifacts.scientific_evidence_table_id),
        ("patent_prior_art_table.json", reg.active_artifacts.patent_prior_art_table_id),
    ):
        body = local_storage.read_json(local_storage.run_key(RUN_ID, path))
        assert body["artifact_id"] == artifact_id
        assert body["run_id"] == RUN_ID
        assert "PRIVATE_CANONICAL_QUERY_SENTINEL" not in str(
            {
                "planning": body["patent_evidence_planning_audit"],
                "resolver": body["patent_evidence_resolver_audit"],
            }
        )


def test_lane_isolation_and_empty_does_not_fabricate_records(
    local_storage, registry_service, workflow_state_service
):
    _init(registry_service, workflow_state_service)
    calls = Counter()

    def empty(**_kwargs):
        calls["evidence"] += 1
        return {"status": "empty", "executor": "test_fixture", "payload": {"results": []}}

    def forbidden(**_kwargs):
        calls["patent"] += 1
        raise AssertionError("patent lane must not execute")

    result = _agent(
        local_storage,
        registry_service,
        workflow_state_service,
        {
            "EuropePMC_search_articles": empty,
            "PubChem_get_associated_patents_by_CID": forbidden,
        },
        _PlanningLLM(),
    ).run_from_artifacts(
        RUN_ID,
        **_artifacts(requested_outputs=("literature_review_summary",)),
    )
    assert calls == {"evidence": 1}
    assert result.evidence.review_status == "ok"
    assert result.evidence.evidence_records == []
    assert result.patent.patent_review_status == "not_requested"
    assert result.patent.tool_call_records == []
    state = workflow_state_service.get(RUN_ID)
    assert state["steps"]["step_13"] == "completed"
    assert state["steps"]["step_14"] == "skipped"


def test_patent_only_never_executes_evidence(
    local_storage, registry_service, workflow_state_service
):
    _init(registry_service, workflow_state_service)
    calls = Counter()

    def pubchem(**_kwargs):
        calls["patent"] += 1
        return _pubchem()

    result = _agent(
        local_storage,
        registry_service,
        workflow_state_service,
        {"PubChem_get_associated_patents_by_CID": pubchem},
        _PlanningLLM(),
    ).run_from_artifacts(
        RUN_ID,
        **_artifacts(requested_outputs=("patent_or_ip_summary",)),
    )
    assert calls == {"patent": 1}
    assert result.evidence.review_status == "not_requested"
    assert result.evidence.tool_call_records == []
    assert result.patent.patent_review_status == "completed"
    assert result.patent.patent_records == []
    assert result.patent.lookup_summaries[0].source_type == (
        "pubchem_associated_reference"
    )
    state = workflow_state_service.get(RUN_ID)
    assert state["steps"]["step_13"] == "skipped"
    assert state["steps"]["step_14"] == "completed"


def test_partial_and_all_failed_statuses_are_honest(
    local_storage, registry_service, workflow_state_service
):
    _init(registry_service, workflow_state_service)

    def failed(**_kwargs):
        return {"status": "upstream_error", "error_message": "maintenance"}

    partial = _agent(
        local_storage,
        registry_service,
        workflow_state_service,
        {
            "EuropePMC_search_articles": _ok_results("europe"),
            "SemanticScholar_search_papers": failed,
        },
        _PlanningLLM(evidence_tools=["EuropePMC_search_articles", "SemanticScholar_search_papers"], patent=False),
    ).run_from_artifacts(
        RUN_ID,
        **_artifacts(requested_outputs=("literature_review_summary",)),
    )
    assert partial.evidence.review_status == "partial"
    assert [r.run_status for r in partial.evidence.tool_call_records] == ["success", "failed"]

    failed_result = _agent(
        local_storage,
        registry_service,
        workflow_state_service,
        {"EuropePMC_search_articles": failed},
        _PlanningLLM(patent=False),
    ).run_from_artifacts(
        RUN_ID,
        **_artifacts(requested_outputs=("literature_review_summary",)),
    )
    assert failed_result.evidence.review_status == "failed"
    assert failed_result.evidence.evidence_records == []
    assert workflow_state_service.get(RUN_ID)["steps"]["step_13"] == "failed"


def test_no_requested_lane_fails_before_llm_or_mcp(
    local_storage, registry_service, workflow_state_service
):
    _init(registry_service, workflow_state_service)
    llm = _PlanningLLM()
    agent = _agent(local_storage, registry_service, workflow_state_service, {}, llm)
    try:
        agent.run_from_artifacts(RUN_ID, **_artifacts(requested_outputs=()))
    except ValueError as exc:
        assert str(exc) == "patent_evidence_no_requested_lane"
    else:
        raise AssertionError("empty lanes must fail")
    assert llm.calls == 0
    state = workflow_state_service.get(RUN_ID)
    assert state["steps"]["step_13"] == "pending"
    assert state["steps"]["step_14"] == "pending"


def test_canonical_query_enters_only_internal_dynamic_payload(
    local_storage, registry_service, workflow_state_service
):
    _init(registry_service, workflow_state_service)
    llm = _CapturingLLM()
    _agent(
        local_storage,
        registry_service,
        workflow_state_service,
        {
            "EuropePMC_search_articles": _ok_results("europe"),
                "PubChem_get_associated_patents_by_CID": _pubchem,
        },
        llm,
    ).run_from_artifacts(RUN_ID, **_artifacts())
    assert llm.payload["user_query"] == "PRIVATE_CANONICAL_QUERY_SENTINEL"
    assert llm.payload["user_query"]
    dynamic_payload = str(llm.payload)
    for forbidden in (
        "MMAE",
        "inputs/structured_query.json",
        "raw_envelope",
        "sk-test-api-key",
    ):
        assert forbidden not in dynamic_payload
    fixed_prompt = f"{llm.prompt}\n{llm.system}"
    for forbidden in (
        "PRIVATE_CANONICAL_QUERY_SENTINEL",
        "raw_request_record",
        "candidate_context_table.json",
        "raw_envelope",
        "sk-test-api-key",
    ):
        assert forbidden not in fixed_prompt


def test_typed_ref_authority_never_promotes_material_content():
    artifacts = _artifacts(
        referenced_inputs=[
            {"id_type": "application_number", "value": "NDA123"},
            {"id_type": "pubchem_cid", "value": "123"},
            {"id_type": "drugbank_id", "value": "DB00001"},
        ],
        identifiers=[],
        materials=[
            Material(
                material_id="mat_target_sequence",
                material_type="target_sequence",
                value="SEQUENCE_SENTINEL",
                role="target",
            ),
            Material(
                material_id="mat_pdb_ref",
                material_type="pdb_file",
                value="inputs/structure.pdb",
                role="structure",
            ),
            Material(
                material_id="mat_smiles",
                material_type="payload_smiles",
                value="CCO",
                role="payload",
            ),
        ],
    )
    request = _build_request(run_id=RUN_ID, artifacts=artifacts, lanes=["patent"])
    roles = [ref.role for ref in request.input_refs]
    assert roles == [
        "query",
        "application_number",
        "pubchem_cid",
        "drugbank_id",
    ]
    serialized = str([ref.model_dump() for ref in request.input_refs])
    for forbidden in ("SEQUENCE_SENTINEL", "structure.pdb", "CCO"):
        assert forbidden not in serialized


@pytest.mark.parametrize(
    ("explicit_or_inferred", "allowed"),
    [("explicit", True), ("inferred", False)],
)
def test_only_explicit_antibody_hint_opens_antibody_gate(
    explicit_or_inferred, allowed
):
    artifacts = _artifacts(
        downstream_query_hints=[
            {
                "entity": "antibody_ref",
                "role": "antibody",
                "explicit_or_inferred": explicit_or_inferred,
                "source": "fixture",
            }
        ]
    )
    request = _build_request(run_id=RUN_ID, artifacts=artifacts, lanes=["evidence"])
    assert request.search_scope.antibody_search_allowed is allowed


def test_typed_compact_audit_is_additive_and_forbids_unknown_or_negative_fields():
    legacy = ScientificEvidenceTable.model_validate(
        {"run_id": RUN_ID, "created_at": "2026-07-18T00:00:00Z"}
    )
    assert legacy.patent_evidence_planning_audit.executed_count == 0
    assert legacy.patent_evidence_resolver_audit == []
    with pytest.raises(ValidationError):
        ScientificEvidenceTable.model_validate(
            {
                "run_id": RUN_ID,
                "created_at": "2026-07-18T00:00:00Z",
                "patent_evidence_planning_audit": {
                    "executed_count": -1,
                    "raw_query": "forbidden",
                },
            }
        )


@pytest.mark.parametrize(
    ("primary", "secondary", "expected"),
    [
        ("literature_review", [], ["evidence"]),
        ("patent_ip_review", [], ["patent"]),
        (
            "new_adc_design",
            ["literature_review", "patent_ip_review"],
            ["evidence", "patent"],
        ),
        ("optimization", [], []),
    ],
)
def test_lane_authority_uses_only_typed_intents_or_requested_outputs(
    primary, secondary, expected
):
    body = _artifacts(requested_outputs=(), primary_intent=primary)[
        "structured_query"
    ]
    body["task_intent"]["secondary_intents"] = secondary
    structured = StructuredQuery.model_validate(body, strict=True)
    assert requested_lanes_from_structured_query(structured) == expected
