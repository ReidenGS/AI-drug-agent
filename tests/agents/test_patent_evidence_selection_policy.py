from __future__ import annotations

import pytest

from app.mcp import tooluniverse_adapter
from app.agents.patent_evidence_selection_policy import (
    OfficialMetadataUnavailableError,
    PATENT_EVIDENCE_SELECTION_SYSTEM_PROMPT,
    PatentEvidenceSelectionValidationError,
    _json_schema_literal_error,
    build_patent_evidence_catalog,
    plan_patent_evidence_tool_calls,
)
from app.llm.provider import MockLLMProvider
from app.llm.json_task_validation import build_json_prompt_sections
from app.llm.openai_provider import _cache_layout_for_schema, _prompt_cache_key
from app.schemas.patent_evidence_request import (
    PatentEvidenceInputRef,
    PatentEvidenceRequest,
    PatentEvidenceSearchScope,
)


_REAL_GET_UNIVERSE = tooluniverse_adapter._get_universe


@pytest.fixture(autouse=True)
def _use_real_official_metadata(monkeypatch):
    """This module is the explicit real-metadata exception to agents/conftest."""
    monkeypatch.setattr(tooluniverse_adapter, "_get_universe", _REAL_GET_UNIVERSE)


def _ref(ref_id: str, role: str, *supports: str) -> PatentEvidenceInputRef:
    return PatentEvidenceInputRef(
        ref_id=ref_id,
        source_artifact="candidate_context_table",
        source_path=f"refs.{ref_id}",
        role=role,
        supports_tool_args=list(supports),
    )


def _request(*refs: PatentEvidenceInputRef, lanes=None, antibody=False):
    return PatentEvidenceRequest(
        run_id="run_20260717_abcdef12",
        user_query="Assess scientific evidence and patent prior art",
        input_refs=list(refs),
        search_scope=PatentEvidenceSearchScope(
            requested_lanes=lanes or ["evidence", "patent"],
            antibody_search_allowed=antibody,
        ),
    )


def test_real_official_catalog_is_exact_and_complete():
    catalog = build_patent_evidence_catalog()
    assert len(catalog) == 11
    assert [entry["tool_name"] for entry in catalog] == sorted(
        entry["tool_name"] for entry in catalog
    )
    assert (
        sum(entry["tool_name"] == "EuropePMC_search_articles" for entry in catalog) == 1
    )
    assert sum(entry["search_lane"] == "evidence" for entry in catalog) == 8
    assert sum(entry["search_lane"] == "patent" for entry in catalog) == 3
    assert all(entry["description"] for entry in catalog)
    assert all(entry["full_schema"]["properties"] for entry in catalog)
    assert all(
        "required" not in prop
        for entry in catalog
        for prop in entry["full_schema"]["properties"].values()
    )
    assert all(
        entry["metadata_authority"] == "tooluniverse_official_spec" for entry in catalog
    )
    assert all(entry["wrapper_identity_parity"] for entry in catalog)
    assert all(entry["wrapper_full_schema_acceptance"] for entry in catalog)
    drugbank = next(
        entry
        for entry in catalog
        if entry["tool_name"] == "drugbank_get_drug_references_by_drug_name_or_id"
    )
    assert drugbank["runtime_availability"] == {
        "status": "license_gated",
        "can_execute": False,
        "reason_code": "drugbank_license_required",
    }
    assert drugbank["wrapper_identity_parity"] is True
    assert drugbank["wrapper_full_executable_schema_parity"] is False
    assert (
        drugbank["wrapper_full_executable_schema_parity_status"]
        == "not_executable_license_gated"
    )
    executable = [
        entry for entry in catalog if entry["runtime_availability"]["can_execute"]
    ]
    assert len(executable) == 8
    assert all(entry["wrapper_full_executable_schema_parity"] for entry in executable)
    multi = next(
        entry for entry in catalog if entry["tool_name"] == "MultiAgentLiteratureSearch"
    )
    assert multi["runtime_constraints"] == [
        {
            "schema_arg": "max_iterations",
            "constraint_schema": {"const": 1},
            "reason_code": "max_iterations_runtime_cap_1",
        }
    ]
    assert multi["runtime_effective_defaults"] == {
        "max_iterations": 1,
        "quality_threshold": 0.7,
    }
    assert multi["full_schema"]["properties"]["quality_threshold"]["default"] == 0.7
    assert multi["runtime_availability"] == {
        "status": "scope_blocked",
        "can_execute": False,
        "reason_code": "uncontained_tooluniverse_full_discovery",
    }
    literature = next(
        entry for entry in catalog if entry["tool_name"] == "LiteratureSearchTool"
    )
    assert literature["runtime_availability"] == {
        "status": "dependency_unavailable",
        "can_execute": False,
        "reason_code": "medical_literature_reviewer_outside_approved_inventory",
    }
    europe = next(
        entry
        for entry in catalog
        if entry["tool_name"] == "EuropePMC_search_articles"
    )
    assert europe["mutually_exclusive_schema_arg_groups"] == [
        ["limit", "page_size"]
    ]
    openalex = next(
        entry for entry in catalog if entry["tool_name"] == "openalex_search_works"
    )
    assert openalex["mutually_exclusive_schema_arg_groups"] == [
        ["query", "search"],
        ["per_page", "limit"],
    ]
    assert openalex["full_schema"]["properties"]["per_page"]["default"] == 10
    pubchem = next(
        entry
        for entry in catalog
        if entry["tool_name"] == "PubChem_get_associated_patents_by_CID"
    )
    assert pubchem["schema_arg_allowed_ref_roles"]["cid"] == ["pubchem_cid"]
    fda = next(
        entry
        for entry in catalog
        if entry["tool_name"] == "FDA_OrangeBook_get_patent_info"
    )
    assert fda["schema_arg_allowed_ref_roles"] == {
        "application_number": ["application_number"],
        "brand_name": ["brand_name"],
    }


def test_official_metadata_missing_fails_closed(monkeypatch):
    monkeypatch.setattr(
        "app.agents.patent_evidence_selection_policy.tooluniverse_adapter.get_tool_specifications",
        lambda _names: {},
    )
    with pytest.raises(
        OfficialMetadataUnavailableError, match="official_metadata_unavailable"
    ):
        build_patent_evidence_catalog()


def test_mock_one_call_returns_both_lanes_from_supplied_catalog():
    class CountingMock(MockLLMProvider):
        calls = 0

        def generate_json(self, prompt, *, schema, system=None):
            self.calls += 1
            return super().generate_json(prompt, schema=schema, system=system)

    llm = CountingMock()
    result = plan_patent_evidence_tool_calls(
        llm=llm,
        request=_request(
            _ref("r_payload", "payload", "query"),
            _ref("r_cid", "pubchem_cid", "cid"),
        ),
    )
    assert llm.calls == 1
    assert {plan.search_lane for plan in result.tool_plans if plan.can_invoke} == {
        "evidence",
        "patent",
    }
    assert all(
        plan.execution_step_id
        == ("step_13" if plan.search_lane == "evidence" else "step_14")
        for plan in result.tool_plans
    )


def test_canonical_query_ref_drives_mock_evidence_planning():
    result = plan_patent_evidence_tool_calls(
        llm=MockLLMProvider(),
        request=_request(_ref("r_query", "query", "query"), lanes=["evidence"]),
    )
    assert result.lane_assessments[0].status == "planned"
    assert result.tool_plans
    assert all(plan.search_lane == "evidence" for plan in result.tool_plans)
    assert all(plan.can_invoke for plan in result.tool_plans)


def test_canonical_query_ref_europepmc_mapping_is_accepted_in_production_validator():
    result = plan_patent_evidence_tool_calls(
        llm=_FakeLLM(
            [_plan("EuropePMC_search_articles", "query", "r_query")],
            assessments=[
                {
                    "search_lane": "evidence",
                    "status": "planned",
                    "reason": "canonical query ref",
                }
            ],
        ),
        request=_request(_ref("r_query", "query", "query"), lanes=["evidence"]),
    )
    assert [plan.tool_name for plan in result.tool_plans] == [
        "EuropePMC_search_articles"
    ]
    assert result.tool_plans[0].argument_mappings[0].input_ref_id == "r_query"
    assert result.rejected_tool_plans == []


def test_catalog_query_role_is_limited_to_generic_evidence_and_drugbank_query():
    by_tool = {
        entry["tool_name"]: entry for entry in build_patent_evidence_catalog()
    }
    expected_evidence_args = {
        "LiteratureSearchTool": ["research_topic"],
        "EuropePMC_search_articles": ["query"],
        "openalex_search_works": ["query", "search"],
        "PubTator3_LiteratureSearch": ["query"],
        "SemanticScholar_search_papers": ["query"],
        "MultiAgentLiteratureSearch": ["query"],
    }
    for tool_name, schema_args in expected_evidence_args.items():
        for schema_arg in schema_args:
            assert "query" in by_tool[tool_name]["schema_arg_allowed_ref_roles"][
                schema_arg
            ]

    forbidden_typed_args = {
        "PubTator3_get_annotations": ["pmids"],
        "ChEMBL_search_documents": ["document_id", "title__contains"],
        "PubChem_get_associated_patents_by_CID": ["cid"],
        "FDA_OrangeBook_get_patent_info": ["brand_name", "application_number"],
    }
    for tool_name, schema_args in forbidden_typed_args.items():
        for schema_arg in schema_args:
            assert "query" not in by_tool[tool_name][
                "schema_arg_allowed_ref_roles"
            ][schema_arg]

    drugbank = by_tool["drugbank_get_drug_references_by_drug_name_or_id"]
    assert drugbank["schema_arg_allowed_ref_roles"]["query"] == [
        "drug_name",
        "query",
    ]
    assert drugbank["runtime_availability"] == {
        "status": "license_gated",
        "can_execute": False,
        "reason_code": "drugbank_license_required",
    }


def test_query_ref_dual_lane_plans_evidence_but_not_license_gated_drugbank():
    result = plan_patent_evidence_tool_calls(
        llm=MockLLMProvider(),
        request=_request(_ref("r_query", "query", "query")),
    )
    assert [(item.search_lane, item.status) for item in result.lane_assessments] == [
        ("evidence", "planned"),
        ("patent", "missing_inputs"),
    ]
    assert result.tool_plans
    assert all(plan.search_lane == "evidence" for plan in result.tool_plans)
    assert all(
        plan.tool_name != "drugbank_get_drug_references_by_drug_name_or_id"
        for plan in result.tool_plans
    )


class _FakeLLM:
    def __init__(self, plans, assessments=None):
        self.plans = plans
        self.assessments = assessments
        self.calls = 0

    def generate_json(self, _prompt, *, schema, system=None):
        self.calls += 1
        assert len(schema["tool_catalog"]) == 11
        assessments = self.assessments
        if assessments is None:
            tool_lane = {
                entry["tool_name"]: entry["search_lane"]
                for entry in schema["tool_catalog"]
            }
            raw_lanes = {
                tool_lane.get(plan.get("tool_name"))
                for plan in self.plans
                if isinstance(plan, dict)
            }
            assessments = [
                {
                    "search_lane": lane,
                    "status": "planned" if lane in raw_lanes else "missing_inputs",
                    "reason": "test assessment",
                }
                for lane in schema["search_scope"]["requested_lanes"]
            ]
        return {"lane_assessments": assessments, "tool_plans": self.plans}


def _plan(tool, arg, ref, *, can_invoke=True):
    return {
        "tool_name": tool,
        "can_invoke": can_invoke,
        "argument_mappings": [{"schema_arg": arg, "input_ref_id": ref}],
        "argument_literals": [],
        "missing_required_args": [],
        "selection_reason": "test",
    }


def test_validator_rejects_hallucinations_gate_missing_pmid_and_duplicates():
    valid = _plan("EuropePMC_search_articles", "query", "r_query")
    plans = [
        valid,
        valid,
        _plan("invented_tool", "query", "r_query"),
        _plan("EuropePMC_search_articles", "made_up", "r_query"),
        _plan("EuropePMC_search_articles", "query", "missing_ref"),
        _plan("EuropePMC_search_articles", "query", "r_antibody"),
        _plan("PubTator3_get_annotations", "pmids", "r_query"),
    ]
    result = plan_patent_evidence_tool_calls(
        llm=_FakeLLM(plans),
        request=_request(
            _ref("r_query", "payload", "query"),
            _ref("r_antibody", "antibody", "query"),
        ),
    )
    assert [p.tool_name for p in result.tool_plans] == ["EuropePMC_search_articles"]
    reasons = [p.reason for p in result.rejected_tool_plans]
    assert "duplicate_plan" in reasons
    assert "unknown_tool" in reasons
    assert any("unknown_schema_arg" in reason for reason in reasons)
    assert any("unknown_input_ref_id" in reason for reason in reasons)
    assert any("antibody_search_not_allowed" in reason for reason in reasons)
    assert any("input_ref_cannot_satisfy_schema_arg" in reason for reason in reasons)


def test_same_tool_different_refs_is_not_duplicate():
    llm = _FakeLLM(
        [
            _plan("EuropePMC_search_articles", "query", "r1"),
            _plan("EuropePMC_search_articles", "query", "r2"),
        ]
    )
    result = plan_patent_evidence_tool_calls(
        llm=llm,
        request=_request(_ref("r1", "payload", "query"), _ref("r2", "linker", "query")),
    )
    assert len(result.tool_plans) == 2


def test_requested_lane_is_enforced():
    result = plan_patent_evidence_tool_calls(
        llm=_FakeLLM([_plan("PubChem_get_associated_patents_by_CID", "cid", "r_cid")]),
        request=_request(_ref("r_cid", "pubchem_cid", "cid"), lanes=["evidence"]),
    )
    assert result.tool_plans == []
    assert result.rejected_tool_plans[0].reason == "tool_not_in_requested_lane"


def test_allowed_roles_has_same_deny_semantics_in_mock_and_production_validator():
    request = PatentEvidenceRequest(
        run_id="run_20260717_abcdef12",
        input_refs=[_ref("r_payload", "payload", "query")],
        search_scope=PatentEvidenceSearchScope(
            requested_lanes=["evidence"], allowed_roles=["target"]
        ),
    )
    mock_result = plan_patent_evidence_tool_calls(
        llm=MockLLMProvider(), request=request
    )
    fake_result = plan_patent_evidence_tool_calls(
        llm=_FakeLLM(
            [],
            assessments=[
                {
                    "search_lane": "evidence",
                    "status": "missing_inputs",
                    "reason": "role is not allowed",
                }
            ],
        ),
        request=request,
    )
    assert mock_result.tool_plans == fake_result.tool_plans == []
    assert mock_result.lane_assessments[0].status == "missing_inputs"


def test_antibody_gate_is_consistent_for_default_scope_mock_and_production():
    antibody_ref = _ref("r_antibody", "antibody", "query")
    payload_ref = _ref("r_payload", "payload", "query")

    mock_denied = plan_patent_evidence_tool_calls(
        llm=MockLLMProvider(),
        request=_request(antibody_ref, lanes=["evidence"], antibody=False),
    )
    mock_allowed = plan_patent_evidence_tool_calls(
        llm=MockLLMProvider(),
        request=_request(antibody_ref, lanes=["evidence"], antibody=True),
    )
    assert mock_denied.tool_plans == []
    assert mock_denied.lane_assessments[0].status == "missing_inputs"
    assert mock_allowed.tool_plans
    assert all(plan.can_invoke for plan in mock_allowed.tool_plans)

    plans = [
        _plan("EuropePMC_search_articles", "query", "r_antibody"),
        _plan("openalex_search_works", "query", "r_payload"),
    ]
    denied = plan_patent_evidence_tool_calls(
        llm=_FakeLLM(plans),
        request=_request(
            antibody_ref, payload_ref, lanes=["evidence"], antibody=False
        ),
    )
    allowed = plan_patent_evidence_tool_calls(
        llm=_FakeLLM(plans),
        request=_request(
            antibody_ref, payload_ref, lanes=["evidence"], antibody=True
        ),
    )
    assert any(
        rejection.reason == "antibody_search_not_allowed"
        for rejection in denied.rejected_tool_plans
    )
    assert [plan.tool_name for plan in denied.tool_plans] == [
        "openalex_search_works"
    ]
    assert {plan.tool_name for plan in allowed.tool_plans} == {
        "EuropePMC_search_articles",
        "openalex_search_works",
    }


def test_not_applicable_lane_is_supported_without_tool_plan():
    result = plan_patent_evidence_tool_calls(
        llm=_FakeLLM(
            [],
            assessments=[
                {
                    "search_lane": "patent",
                    "status": "not_applicable",
                    "reason": "request is explicitly evidence-only in substance",
                }
            ],
        ),
        request=_request(lanes=["patent"]),
    )
    assert result.tool_plans == []
    assert result.lane_assessments[0].status == "not_applicable"


def test_prompt_split_and_cache_namespace_reconcile():
    catalog = build_patent_evidence_catalog()
    request1 = _request(_ref("r1", "payload", "query"))
    request2 = PatentEvidenceRequest(
        run_id="run_20260717_deadbeef",
        user_query="different query",
        input_refs=[_ref("r2", "linker", "query")],
    )
    from app.agents.patent_evidence_selection_policy import (
        PATENT_EVIDENCE_SELECTION_SYSTEM_PROMPT,
        PATENT_EVIDENCE_SELECTION_USER_PROMPT,
        build_patent_evidence_selection_payload,
    )

    payload1 = build_patent_evidence_selection_payload(
        request=request1, catalog=catalog
    )
    payload2 = build_patent_evidence_selection_payload(
        request=request2, catalog=catalog
    )
    stable1, dynamic1 = build_json_prompt_sections(
        prompt=PATENT_EVIDENCE_SELECTION_USER_PROMPT,
        schema=payload1,
        system=PATENT_EVIDENCE_SELECTION_SYSTEM_PROMPT,
    )
    stable2, dynamic2 = build_json_prompt_sections(
        prompt=PATENT_EVIDENCE_SELECTION_USER_PROMPT,
        schema=payload2,
        system=PATENT_EVIDENCE_SELECTION_SYSTEM_PROMPT,
    )
    assert stable1 == stable2
    assert dynamic1 != dynamic2
    layout1 = _cache_layout_for_schema(payload1, "patent_evidence_tool_selection")
    layout2 = _cache_layout_for_schema(payload2, "patent_evidence_tool_selection")
    assert layout1 == layout2
    key1 = _prompt_cache_key(
        model="gpt-5.5", task="patent_evidence_tool_selection", layout_version=layout1
    )
    key2 = _prompt_cache_key(
        model="gpt-5.5", task="patent_evidence_tool_selection", layout_version=layout2
    )
    assert key1 == key2

    changed_catalog = [dict(entry) for entry in catalog]
    changed_catalog[0] = {**changed_catalog[0], "description": "catalog changed"}
    changed_payload = dict(payload1, tool_catalog=changed_catalog)
    changed_layout = _cache_layout_for_schema(
        changed_payload, "patent_evidence_tool_selection"
    )
    assert changed_layout != layout1
    changed_version_payload = dict(payload1, prompt_cache_layout_version="v_next")
    assert (
        _cache_layout_for_schema(
            changed_version_payload, "patent_evidence_tool_selection"
        )
        != layout1
    )


def test_prompt_requires_nonplanned_lanes_to_emit_no_tool_plan():
    assert "can_invoke=false" not in PATENT_EVIDENCE_SELECTION_SYSTEM_PROMPT
    assert "missing_inputs and not_applicable must emit no tool plan" in (
        PATENT_EVIDENCE_SELECTION_SYSTEM_PROMPT
    )


@pytest.mark.parametrize(
    ("tool_name", "mappings", "literals", "expected_reason"),
    [
        (
            "EuropePMC_search_articles",
            [{"schema_arg": "query", "input_ref_id": "r_query"}],
            [
                {"schema_arg": "limit", "literal_value_json": "5"},
                {"schema_arg": "page_size", "literal_value_json": "5"},
            ],
            "mutually_exclusive_schema_args:limit|page_size",
        ),
        (
            "openalex_search_works",
            [
                {"schema_arg": "query", "input_ref_id": "r_query"},
                {"schema_arg": "search", "input_ref_id": "r_search"},
            ],
            [],
            "mutually_exclusive_schema_args:query|search",
        ),
        (
            "openalex_search_works",
            [{"schema_arg": "query", "input_ref_id": "r_query"}],
            [
                {"schema_arg": "per_page", "literal_value_json": "10"},
                {"schema_arg": "limit", "literal_value_json": "10"},
            ],
            "mutually_exclusive_schema_args:per_page|limit",
        ),
    ],
)
def test_deterministic_validator_rejects_alias_groups(
    tool_name, mappings, literals, expected_reason
):
    conflicting = {
        "tool_name": tool_name,
        "can_invoke": True,
        "argument_mappings": mappings,
        "argument_literals": literals,
        "missing_required_args": [],
        "selection_reason": "conflicting aliases",
    }
    result = plan_patent_evidence_tool_calls(
        llm=_FakeLLM(
            [
                conflicting,
                _plan("SemanticScholar_search_papers", "query", "r_query"),
            ],
            assessments=[
                {"search_lane": "evidence", "status": "planned", "reason": "test"}
            ],
        ),
        request=_request(
            _ref("r_query", "payload", "query"),
            _ref("r_search", "linker", "search"),
            lanes=["evidence"],
        ),
    )
    rejection = next(
        item for item in result.rejected_tool_plans if item.tool_name == tool_name
    )
    assert rejection.reason == expected_reason


def test_tool_specific_role_contract_rejects_globally_valid_support_mapping():
    result = plan_patent_evidence_tool_calls(
        llm=_FakeLLM(
            [
                _plan("EuropePMC_search_articles", "query", "r_drug"),
                _plan("EuropePMC_search_articles", "query", "r_payload"),
            ]
        ),
        request=_request(
            _ref("r_drug", "drug_name", "query"),
            _ref("r_payload", "payload", "query"),
            lanes=["evidence"],
        ),
    )
    assert [plan.argument_mappings[0].input_ref_id for plan in result.tool_plans] == [
        "r_payload"
    ]
    assert result.rejected_tool_plans[0].reason == (
        "ref_role_not_allowed_for_schema_arg:query"
    )


@pytest.mark.parametrize(
    ("schema", "valid", "invalid", "keyword"),
    [
        ({"type": "integer", "minimum": 1}, 1, 0, "minimum"),
        ({"type": "integer", "maximum": 2}, 2, 3, "maximum"),
        ({"type": "string", "minLength": 2}, "ab", "a", "minLength"),
        ({"enum": ["a", "b"]}, "a", "c", "enum"),
        ({"const": 1}, 1, 2, "const"),
        ({"type": "array", "items": {"type": "string"}}, ["a"], [1], "type"),
        ({"type": "boolean"}, True, 1, "type"),
        ({"type": "integer"}, 1, True, "type"),
        ({"type": "number"}, 1.5, True, "type"),
    ],
)
def test_complete_json_schema_literal_validation(schema, valid, invalid, keyword):
    assert _json_schema_literal_error(schema, valid) is None
    assert _json_schema_literal_error(schema, invalid) == f"json_schema_{keyword}"


def test_multiagent_runtime_constraint_rejects_literal_without_value_leak():
    plan = _plan("MultiAgentLiteratureSearch", "query", "r_query")
    plan["argument_literals"] = [
        {"schema_arg": "max_iterations", "literal_value_json": "2"}
    ]
    result = plan_patent_evidence_tool_calls(
        llm=_FakeLLM(
            [plan, _plan("EuropePMC_search_articles", "query", "r_query")],
            assessments=[
                {"search_lane": "evidence", "status": "planned", "reason": "test"}
            ],
        ),
        request=_request(_ref("r_query", "payload", "query"), lanes=["evidence"]),
    )
    rejection = next(
        item
        for item in result.rejected_tool_plans
        if item.tool_name == "MultiAgentLiteratureSearch"
    )
    assert rejection.reason == "runtime_unavailable:scope_blocked"
    assert "2" not in rejection.reason


def test_drugbank_license_gate_prevents_invocable_plan():
    drugbank = _plan(
        "drugbank_get_drug_references_by_drug_name_or_id", "query", "r_drug"
    )
    pubchem = _plan("PubChem_get_associated_patents_by_CID", "cid", "r_cid")
    result = plan_patent_evidence_tool_calls(
        llm=_FakeLLM(
            [drugbank, pubchem],
            assessments=[
                {"search_lane": "patent", "status": "planned", "reason": "CID ref"}
            ],
        ),
        request=_request(
            _ref("r_drug", "drug_name", "query"),
            _ref("r_cid", "pubchem_cid", "cid"),
            lanes=["patent"],
        ),
    )
    assert [plan.tool_name for plan in result.tool_plans if plan.can_invoke] == [
        "PubChem_get_associated_patents_by_CID"
    ]
    rejection = next(
        item
        for item in result.rejected_tool_plans
        if item.tool_name == "drugbank_get_drug_references_by_drug_name_or_id"
    )
    assert rejection.reason == "runtime_unavailable:license_gated"
    mock_only = plan_patent_evidence_tool_calls(
        llm=MockLLMProvider(),
        request=_request(_ref("r_drug", "drug_name", "query"), lanes=["patent"]),
    )
    assert mock_only.tool_plans == []
    assert mock_only.lane_assessments[0].status == "missing_inputs"


def test_planned_lane_without_any_accepted_plan_fails_closed():
    plan = _plan("MultiAgentLiteratureSearch", "query", "r_query")
    plan["argument_literals"] = [
        {"schema_arg": "max_iterations", "literal_value_json": "2"}
    ]
    llm = _FakeLLM(
        [plan],
        assessments=[
            {"search_lane": "evidence", "status": "planned", "reason": "test"}
        ],
    )
    with pytest.raises(
        PatentEvidenceSelectionValidationError,
        match="lane_assessment_planned_without_accepted_plan",
    ):
        plan_patent_evidence_tool_calls(
            llm=llm,
            request=_request(_ref("r_query", "payload", "query"), lanes=["evidence"]),
        )


@pytest.mark.parametrize(
    ("assessments", "error"),
    [
        ([], "lane_assessment_requested_set_mismatch"),
        (
            [
                {"search_lane": "evidence", "status": "planned", "reason": "a"},
                {"search_lane": "evidence", "status": "planned", "reason": "b"},
            ],
            "lane_assessment_duplicate_lane",
        ),
        (
            [{"search_lane": "patent", "status": "missing_inputs", "reason": "x"}],
            "lane_assessment_unrequested_lane",
        ),
        (
            [{"search_lane": "evidence", "status": "missing_inputs", "reason": "x"}],
            "lane_assessment_nonplanned_with_tool_plan",
        ),
    ],
)
def test_lane_assessments_fail_closed(assessments, error):
    with pytest.raises(PatentEvidenceSelectionValidationError, match=error):
        plan_patent_evidence_tool_calls(
            llm=_FakeLLM(
                [_plan("EuropePMC_search_articles", "query", "r_query")],
                assessments=assessments,
            ),
            request=_request(_ref("r_query", "payload", "query"), lanes=["evidence"]),
        )
