"""Step 14 request-based PatentIPAgent (`run_from_request`) tests."""

from __future__ import annotations

import json

from app.agents.candidate_context_agent import CandidateContextAgent
from app.agents.patent_ip_agent import PatentIPAgent
from app.agents.supervisor_agent import SupervisorAgent
from app.llm.provider import MockLLMProvider
from app.mcp.client import LocalMCPClient
from app.schemas.step_14_patent_request import (
    Step14InputRef,
    Step14PatentRequest,
    Step14PatentScope,
)
from app.services.input_readiness_service import InputReadinessService
from app.services.intake_service import IntakeService
from app.services.structured_query_service import StructuredQueryService
from app.services.workflow_setup_service import WorkflowSetupService


def _seed_through_step_5(
    local_storage, registry_service, workflow_state_service,
    *, referenced_inputs=None, user_ctx=None,
):
    intake = IntakeService(local_storage, registry_service, workflow_state_service)
    rec = intake.submit(
        raw_user_query="HER2 ADC vc-MMAE",
        user_provided_context=user_ctx or {
            "target_or_antigen_text": "HER2",
            "candidate_text": "Trastuzumab",
            "payload_linker_text": "vc-MMAE",
        },
    )
    run_id = rec.run_id
    StructuredQueryService(
        local_storage, registry_service, workflow_state_service,
        SupervisorAgent(llm=MockLLMProvider()),
    ).parse(run_id)
    if referenced_inputs:
        path = local_storage.run_key(run_id, "inputs/structured_query.json")
        sq = local_storage.read_json(path)
        sq.setdefault("referenced_inputs", []).extend(referenced_inputs)
        local_storage.write_json(path, sq)
    InputReadinessService(local_storage, registry_service, workflow_state_service).check(run_id)
    WorkflowSetupService(local_storage, registry_service, workflow_state_service).plan(run_id)
    CandidateContextAgent(
        storage=local_storage, registry=registry_service,
        workflow_state=workflow_state_service, mcp_client=LocalMCPClient(),
    ).run(run_id)
    return run_id


def _bind_fixed(canned: dict[str, dict]) -> dict:
    def make(payload):
        def _fn(**_kw):
            return payload
        return _fn
    return {name: make(p) for name, p in canned.items()}


def _agent(local_storage, registry_service, workflow_state_service, *, bindings=None, llm=None):
    mcp = LocalMCPClient(bindings=bindings) if bindings else LocalMCPClient()
    return PatentIPAgent(
        storage=local_storage, registry=registry_service,
        workflow_state=workflow_state_service, mcp_client=mcp, llm=llm,
    )


class _StubPlannerLLM:
    """Returns a fixed step14 planner response (proves runtime uses ONLY the
    LLM-provided argument_mappings, never re-derived ones)."""

    name = "stub"
    model = "stub"

    def __init__(self, response: dict):
        self.response = response

    def generate(self, prompt: str, *, system=None, **kwargs):
        raise NotImplementedError

    def generate_json(self, prompt: str, *, schema: dict, system=None) -> dict:
        return self.response


def _cct(local_storage, run_id):
    return local_storage.read_json(
        local_storage.run_key(run_id, "candidate_context_table.json")
    )


def _pubchem_candidate(cct):
    for rec in cct.get("candidate_records") or []:
        if rec.get("candidate_type") != "compound_component":
            continue
        for ident in rec.get("identifiers") or []:
            if ident.get("id_type") == "pubchem_cid":
                return rec.get("candidate_id"), ident.get("id_value")
    return None, None


def _request(run_id, refs, *, antibody_allowed=False):
    return Step14PatentRequest(
        run_id=run_id,
        user_query="patent prior-art search",
        source_artifact_refs={"candidate_context_table": "cct", "structured_query": "sq"},
        input_refs=refs,
        patent_scope=Step14PatentScope(antibody_search_allowed=antibody_allowed),
    )


# ── test 10: pubchem cid resolved from cct → PubChem called ─────────────────


def test_run_from_request_calls_pubchem_from_cid_ref(
    local_storage, registry_service, workflow_state_service
):
    run_id = _seed_through_step_5(
        local_storage, registry_service, workflow_state_service,
        referenced_inputs=[{"id_type": "pubchem_cid", "value": "123456", "source": "raw_request_text"}],
    )
    cand_id, cid = _pubchem_candidate(_cct(local_storage, run_id))
    assert cid, "expected a compound candidate carrying a pubchem_cid identifier"

    ref = Step14InputRef(
        ref_id="r_cid", source_artifact="candidate_context_table",
        source_path="candidate_records[].identifiers[].id_value",
        role="pubchem_cid", candidate_id=cand_id,
        supports_tool_args=["cid", "pubchem_cid"],
    )
    canned = _bind_fixed({
        "PubChem_get_associated_patents_by_CID": {
            "status": "ok", "source": "PubChem_get_associated_patents_by_CID",
            "patents": [{"patent_number": "US-CID-1", "title": "Compound patent"}],
        },
    })
    table = _agent(
        local_storage, registry_service, workflow_state_service, bindings=canned,
    ).run_from_request(_request(run_id, [ref]))

    calls = [tc for tc in table.tool_call_records if tc.tool_name == "PubChem_get_associated_patents_by_CID"]
    assert calls, "PubChem tool call expected"
    assert calls[0].run_status == "success"
    assert (calls[0].tool_input_summary or {}).get("cid") == str(cid)
    assert table.step14_request_source == "request"


# ── test 11: brand_name / application_number → FDA OrangeBook called ─────────


def test_run_from_request_calls_fda_from_application_number_ref(
    local_storage, registry_service, workflow_state_service
):
    run_id = _seed_through_step_5(
        local_storage, registry_service, workflow_state_service,
        referenced_inputs=[{"id_type": "application_number", "value": "NDA761139", "source": "user"}],
    )
    ref = Step14InputRef(
        ref_id="r_app", source_artifact="structured_query",
        source_path="referenced_inputs[].value",
        role="application_number", supports_tool_args=["application_number"],
    )
    canned = _bind_fixed({
        "FDA_OrangeBook_get_patent_info": {
            "status": "ok", "source": "FDA_OrangeBook_get_patent_info",
            "records": [{"patent_number": "US-OB-1", "title": "Orange Book patent"}],
        },
    })
    table = _agent(
        local_storage, registry_service, workflow_state_service, bindings=canned,
    ).run_from_request(_request(run_id, [ref]))
    calls = [tc for tc in table.tool_call_records if tc.tool_name == "FDA_OrangeBook_get_patent_info"]
    assert calls and calls[0].run_status == "success"
    assert (calls[0].tool_input_summary or {}).get("application_number") == "NDA761139"


def test_run_from_request_fda_upstream_error_is_not_success(
    local_storage, registry_service, workflow_state_service
):
    # Envelope-aware: a wrapper envelope carrying status="upstream_error" must
    # NOT be recorded as success even though the outer MCP call returned ok.
    run_id = _seed_through_step_5(
        local_storage, registry_service, workflow_state_service,
        referenced_inputs=[{"id_type": "application_number", "value": "NDA761139", "source": "user"}],
    )
    ref = Step14InputRef(
        ref_id="r_app", source_artifact="structured_query",
        source_path="referenced_inputs[].value",
        role="application_number", supports_tool_args=["application_number"],
    )
    canned = _bind_fixed({
        "FDA_OrangeBook_get_patent_info": {
            "status": "upstream_error",
            "source": "FDA_OrangeBook_get_patent_info",
            "error_message": "compact failure",
            "records": [{"patent_number": "US-SHOULD-NOT-EXTRACT", "title": "leaked"}],
            "raw_debug": "RAW_ENVELOPE_BODY_MUST_NOT_LEAK",
        },
    })
    table = _agent(
        local_storage, registry_service, workflow_state_service, bindings=canned,
    ).run_from_request(_request(run_id, [ref]))

    calls = [tc for tc in table.tool_call_records if tc.tool_name == "FDA_OrangeBook_get_patent_info"]
    assert calls, "expected an FDA tool call record"
    tc = calls[0]
    # 1) status is not success (upstream_error → failed).
    assert tc.run_status != "success"
    assert tc.run_status == "failed"
    # 2) overall review status is not completed.
    assert table.patent_review_status != "completed"
    # 3) no patent hit extracted from a failed envelope.
    assert not any(r.patent_number == "US-SHOULD-NOT-EXTRACT" for r in table.patent_records)
    # 4) raw envelope persisted under tool_output_ref for traceability.
    assert tc.tool_output_ref and local_storage.exists(tc.tool_output_ref)
    raw = local_storage.read_json(tc.tool_output_ref)
    assert "RAW_ENVELOPE_BODY_MUST_NOT_LEAK" in json.dumps(raw)
    # 5) normalized artifact + summary only carry a compact error, no raw body.
    blob = json.dumps(table.model_dump())
    assert "RAW_ENVELOPE_BODY_MUST_NOT_LEAK" not in blob
    assert "US-SHOULD-NOT-EXTRACT" not in blob
    assert (tc.error_message or "") and "compact failure" in (tc.error_message or "")
    assert (tc.tool_input_summary or {}).get("output_envelope_status") == "upstream_error"


def test_run_from_request_calls_fda_from_brand_name_ref(
    local_storage, registry_service, workflow_state_service
):
    run_id = _seed_through_step_5(local_storage, registry_service, workflow_state_service)
    cand_id = (_cct(local_storage, run_id).get("candidate_records") or [{}])
    # Use a payload-name compound material as the brand-name value source.
    compound_id = None
    for rec in _cct(local_storage, run_id).get("candidate_records") or []:
        if rec.get("candidate_type") == "compound_component":
            compound_id = rec.get("candidate_id")
            break
    ref = Step14InputRef(
        ref_id="r_brand", source_artifact="candidate_context_table",
        source_path="candidate_records[].materials[].value",
        role="brand_name", candidate_id=compound_id,
        supports_tool_args=["brand_name"],
    )
    canned = _bind_fixed({
        "FDA_OrangeBook_get_patent_info": {
            "status": "ok", "source": "FDA_OrangeBook_get_patent_info",
            "records": [{"patent_number": "US-OB-2", "title": "OB patent"}],
        },
    })
    table = _agent(
        local_storage, registry_service, workflow_state_service, bindings=canned,
    ).run_from_request(_request(run_id, [ref]))
    calls = [tc for tc in table.tool_call_records if tc.tool_name == "FDA_OrangeBook_get_patent_info"]
    assert calls and calls[0].run_status == "success"
    assert (calls[0].tool_input_summary or {}).get("brand_name")


# ── EuropePMC literature / prior-art evidence tool ──────────────────────────


def test_run_from_request_calls_europepmc_from_query_ref(
    local_storage, registry_service, workflow_state_service
):
    run_id = _seed_through_step_5(local_storage, registry_service, workflow_state_service)
    ref = Step14InputRef(
        ref_id="r_lit", source_artifact="structured_query",
        source_path="mentioned_entities.payload_text",
        role="payload", supports_tool_args=["query"],
    )
    # Fake EuropePMC binding: ok envelope with literature results (title +
    # a big abstract that must NOT land in the normalized artifact).
    canned = {
        "EuropePMC_search_articles": lambda **kw: {
            "status": "ok", "source": "EuropePMC_search_articles",
            "query": kw.get("query"),
            "results": [{
                "title": "MMAE ADC prior-art article",
                "abstract": "RAW_ABSTRACT_MUST_NOT_LEAK " * 40,
            }],
        },
    }
    table = _agent(
        local_storage, registry_service, workflow_state_service, bindings=canned,
    ).run_from_request(_request(run_id, [ref]))

    calls = [tc for tc in table.tool_call_records if tc.tool_name == "EuropePMC_search_articles"]
    assert calls, "expected an EuropePMC tool call"
    tc = calls[0]
    assert tc.run_status == "success"
    s = tc.tool_input_summary or {}
    # runtime resolved the real value into the LLM-mapped `query` arg.
    assert s.get("query") == "MMAE"
    assert s.get("selected_by") == "llm_step14"
    assert s.get("argument_mappings") == [{"schema_arg": "query", "input_ref_id": "r_lit"}]
    # raw output persisted only under tool_output_ref.
    assert tc.tool_output_ref and local_storage.exists(tc.tool_output_ref)
    assert "RAW_ABSTRACT_MUST_NOT_LEAK" in json.dumps(local_storage.read_json(tc.tool_output_ref))
    # normalized artifact carries compact provenance (EuropePMC source), no raw abstract.
    blob = json.dumps(table.model_dump())
    assert "RAW_ABSTRACT_MUST_NOT_LEAK" not in blob
    lit_records = [r for r in table.patent_records if "EuropePMC" in (r.sources or [])]
    assert lit_records, "expected a record tagged with EuropePMC provenance"
    # Literature record must NOT claim a fabricated patent_number.
    assert lit_records[0].patent_number is None
    # source_database Literal is not mislabeled as a patent DB.
    assert lit_records[0].source_database == "other"


def test_run_from_request_europepmc_upstream_error_not_success(
    local_storage, registry_service, workflow_state_service
):
    run_id = _seed_through_step_5(local_storage, registry_service, workflow_state_service)
    ref = Step14InputRef(
        ref_id="r_lit", source_artifact="structured_query",
        source_path="mentioned_entities.payload_text",
        role="payload", supports_tool_args=["query"],
    )
    canned = _bind_fixed({
        "EuropePMC_search_articles": {
            "status": "upstream_error", "source": "EuropePMC_search_articles",
            "error_message": "europepmc timeout",
        },
    })
    table = _agent(
        local_storage, registry_service, workflow_state_service, bindings=canned,
    ).run_from_request(_request(run_id, [ref]))
    calls = [tc for tc in table.tool_call_records if tc.tool_name == "EuropePMC_search_articles"]
    assert calls and calls[0].run_status == "failed"
    assert table.patent_review_status != "completed"
    assert (calls[0].tool_input_summary or {}).get("output_envelope_status") == "upstream_error"


# ── test 12: unresolved ref → skipped / input_missing ───────────────────────


def test_run_from_request_unresolved_ref_is_skipped(
    local_storage, registry_service, workflow_state_service
):
    run_id = _seed_through_step_5(local_storage, registry_service, workflow_state_service)
    # No pubchem_cid identifier exists in the cct for this run → unresolved.
    ref = Step14InputRef(
        ref_id="r_cid", source_artifact="candidate_context_table",
        source_path="candidate_records[].identifiers[].id_value",
        role="pubchem_cid", supports_tool_args=["cid", "pubchem_cid"],
    )
    table = _agent(
        local_storage, registry_service, workflow_state_service,
    ).run_from_request(_request(run_id, [ref]))
    skipped = [tc for tc in table.tool_call_records if tc.run_status == "skipped"]
    assert skipped, "expected a skipped input_missing tool call"
    assert (skipped[0].tool_input_summary or {}).get("skip_reason") == "input_missing"
    # No fake success / no patent record fabricated from the skipped call.
    assert not any(
        (tc.tool_input_summary or {}).get("skip_reason") == "input_missing"
        and tc.run_status == "success"
        for tc in table.tool_call_records
    )


# ── test 13: only request-declared input refs are used ──────────────────────


def test_run_from_request_only_uses_declared_input_refs(
    local_storage, registry_service, workflow_state_service
):
    run_id = _seed_through_step_5(local_storage, registry_service, workflow_state_service)
    # Step 5 built many downstream hints (payload/linker/target/antibody). We
    # pass only ONE payload ref; no antibody/target calls may appear.
    ref = Step14InputRef(
        ref_id="r_payload", source_artifact="structured_query",
        source_path="mentioned_entities.payload_text",
        role="payload", supports_tool_args=["query"],
    )
    table = _agent(
        local_storage, registry_service, workflow_state_service,
    ).run_from_request(_request(run_id, [ref]))
    roles = {(tc.tool_input_summary or {}).get("query_role") for tc in table.tool_call_records}
    assert roles <= {"payload"}
    assert "antibody" not in roles and "target" not in roles
    # Audit only mentions the single declared ref.
    audited = {a.get("ref_id") for a in table.step14_runtime_resolver_audit}
    assert audited <= {"r_payload"}


# ── test 14/15: raw payload isolation + no leakage ──────────────────────────


def test_run_from_request_raw_payload_only_in_tool_output_ref(
    local_storage, registry_service, workflow_state_service
):
    run_id = _seed_through_step_5(local_storage, registry_service, workflow_state_service)
    ref = Step14InputRef(
        ref_id="r_payload", source_artifact="structured_query",
        source_path="mentioned_entities.payload_text",
        role="payload", supports_tool_args=["query", "drug_name_or_id"],
    )
    canned = _bind_fixed({
        "drugbank_get_drug_references_by_drug_name_or_id": {
            "status": "ok", "source": "drugbank_get_drug_references_by_drug_name_or_id",
            "references": [{
                "patent_number": "US-SECRET", "title": "ADC payload",
                "description": "SECRET_FULL_DESCRIPTION_MUST_NOT_LEAK",
                "raw_sequence": "EVQLVESGGGLVQPGGSLRLSCAAS",
            }],
        },
    })
    table = _agent(
        local_storage, registry_service, workflow_state_service, bindings=canned,
    ).run_from_request(_request(run_id, [ref]))

    dumped = json.dumps(table.model_dump())
    assert "SECRET_FULL_DESCRIPTION_MUST_NOT_LEAK" not in dumped
    assert "EVQLVESGGGLVQPGGSLRLSCAAS" not in dumped
    assert "mocked" not in dumped
    # But raw payload IS persisted under a tool_output_ref.
    refs = [
        tc.tool_output_ref for tc in table.tool_call_records
        if tc.run_status == "success" and tc.tool_output_ref
    ]
    assert refs
    assert any(
        "SECRET_FULL_DESCRIPTION_MUST_NOT_LEAK" in json.dumps(local_storage.read_json(r))
        for r in refs
    )
    # Extracted patent row exists but only holds compact fields.
    assert any(r.patent_number == "US-SECRET" for r in table.patent_records)


# ── DrugBank runtime arg uses the OFFICIAL schema arg name (`query`) ────────


def test_run_from_request_drugbank_uses_official_query_arg(
    local_storage, registry_service, workflow_state_service
):
    run_id = _seed_through_step_5(local_storage, registry_service, workflow_state_service)
    captured: dict = {}

    def _drugbank(**kwargs):
        captured.update(kwargs)
        return {
            "status": "ok", "source": "drugbank_get_drug_references_by_drug_name_or_id",
            "references": [{"patent_number": "US-DB", "title": "ADC ref"}],
        }

    ref = Step14InputRef(
        ref_id="r_payload", source_artifact="structured_query",
        source_path="mentioned_entities.payload_text",
        role="payload", supports_tool_args=["query", "drug_name_or_id"],
    )
    table = _agent(
        local_storage, registry_service, workflow_state_service,
        bindings={"drugbank_get_drug_references_by_drug_name_or_id": _drugbank},
    ).run_from_request(_request(run_id, [ref]))

    calls = [tc for tc in table.tool_call_records
             if tc.tool_name == "drugbank_get_drug_references_by_drug_name_or_id"]
    assert calls and calls[0].run_status == "success"
    # Runtime call + tool_input_summary use the official arg `query`, not the
    # wrapper alias `drug_name_or_id`.
    assert "query" in captured and captured["query"]
    assert "drug_name_or_id" not in captured
    summary = calls[0].tool_input_summary or {}
    assert "query" in summary
    assert "drug_name_or_id" not in summary


# ── runtime kwargs come ONLY from LLM mappings ──────────────────────────────


def test_run_from_request_kwargs_come_only_from_llm_mappings(
    local_storage, registry_service, workflow_state_service
):
    run_id = _seed_through_step_5(
        local_storage, registry_service, workflow_state_service,
        referenced_inputs=[{"id_type": "pubchem_cid", "value": "123456", "source": "raw_request_text"}],
    )
    cand_id, cid = _pubchem_candidate(_cct(local_storage, run_id))
    assert cid

    captured: dict = {}

    def _pubchem(**kwargs):
        captured.update(kwargs)
        return {"status": "ok", "source": "PubChem_get_associated_patents_by_CID",
                "patents": [{"patent_number": "US-CID", "title": "Compound"}]}

    ref = Step14InputRef(
        ref_id="r_cid", source_artifact="candidate_context_table",
        source_path="candidate_records[].identifiers[].id_value",
        role="pubchem_cid", candidate_id=cand_id,
        supports_tool_args=["cid", "pubchem_cid"],
    )
    # The planner LLM returns exactly one mapping cid -> r_cid.
    stub = _StubPlannerLLM({"tool_plans": [
        {"tool_name": "PubChem_get_associated_patents_by_CID", "can_invoke": True,
         "argument_mappings": [{"schema_arg": "cid", "input_ref_id": "r_cid"}],
         "argument_literals": [], "missing_required_args": [],
         "selection_reason": "cid ref"},
    ]})
    table = _agent(
        local_storage, registry_service, workflow_state_service,
        bindings={"PubChem_get_associated_patents_by_CID": _pubchem}, llm=stub,
    ).run_from_request(_request(run_id, [ref]))

    # kwargs contain ONLY the LLM-mapped schema_arg → resolved value. No extra
    # invented args.
    assert captured == {"cid": str(cid)}
    calls = [tc for tc in table.tool_call_records
             if tc.tool_name == "PubChem_get_associated_patents_by_CID"]
    assert calls and calls[0].run_status == "success"


def test_run_from_request_uninvokable_plan_is_not_called(
    local_storage, registry_service, workflow_state_service
):
    run_id = _seed_through_step_5(local_storage, registry_service, workflow_state_service)
    called: dict = {"n": 0}

    def _pubchem(**kwargs):
        called["n"] += 1
        return {"status": "ok", "source": "PubChem_get_associated_patents_by_CID", "patents": []}

    ref = Step14InputRef(
        ref_id="r_cid", source_artifact="candidate_context_table",
        source_path="candidate_records[].identifiers[].id_value",
        role="pubchem_cid", supports_tool_args=["cid"],
    )
    # Planner returns a plan with NO mapping → identity unsatisfied →
    # can_invoke=false → runtime must not call the tool.
    stub = _StubPlannerLLM({"tool_plans": [
        {"tool_name": "PubChem_get_associated_patents_by_CID", "can_invoke": True,
         "argument_mappings": [], "argument_literals": [], "missing_required_args": []},
    ]})
    table = _agent(
        local_storage, registry_service, workflow_state_service,
        bindings={"PubChem_get_associated_patents_by_CID": _pubchem}, llm=stub,
    ).run_from_request(_request(run_id, [ref]))

    assert called["n"] == 0, "uninvokable plan must not call the tool"
    skipped = [tc for tc in table.tool_call_records if tc.run_status == "skipped"]
    assert skipped and (skipped[0].tool_input_summary or {}).get("skip_reason") == "uninvokable"
    assert (skipped[0].tool_input_summary or {}).get("missing_required_args") == ["cid"]


# ── test 16: Step 14 never writes ranking_table ─────────────────────────────


def test_run_from_request_does_not_write_ranking_table(
    local_storage, registry_service, workflow_state_service
):
    run_id = _seed_through_step_5(local_storage, registry_service, workflow_state_service)
    ranking_key = local_storage.run_key(run_id, "ranking_table.json")
    assert not local_storage.exists(ranking_key)
    ref = Step14InputRef(
        ref_id="r_payload", source_artifact="structured_query",
        source_path="mentioned_entities.payload_text",
        role="payload", supports_tool_args=["query"],
    )
    _agent(
        local_storage, registry_service, workflow_state_service,
    ).run_from_request(_request(run_id, [ref]))
    assert not local_storage.exists(ranking_key)
    assert registry_service.get(run_id).active_artifacts.ranking_table_id is None


# ── audit fields populated ──────────────────────────────────────────────────


def test_run_from_request_records_selected_and_scope(
    local_storage, registry_service, workflow_state_service
):
    run_id = _seed_through_step_5(local_storage, registry_service, workflow_state_service)
    ref = Step14InputRef(
        ref_id="r_payload", source_artifact="structured_query",
        source_path="mentioned_entities.payload_text",
        role="payload", supports_tool_args=["query"],
    )
    table = _agent(
        local_storage, registry_service, workflow_state_service,
    ).run_from_request(_request(run_id, [ref]))
    # Single-stage planner audit fields.
    assert table.step14_llm_tool_plans
    assert table.step14_prompt_cache_layout_version == "step14_selection_v3"
    # DrugBank plan maps official `query` arg to the payload ref.
    db_plans = [
        p for p in table.step14_llm_tool_plans
        if p["tool_name"] == "drugbank_get_drug_references_by_drug_name_or_id"
    ]
    assert db_plans and db_plans[0]["argument_mappings"] == [
        {"schema_arg": "query", "input_ref_id": "r_payload"}
    ]
    # argument_mapping_audit records the accepted mapping + support token.
    assert any(
        a["schema_arg"] == "query" and a["input_ref_id"] == "r_payload"
        and a["satisfied_by_support"] in {"query", "drug_name_or_id"}
        for a in table.step14_argument_mapping_audit
    )
    # Turn A BC field is left empty by the single-stage planner.
    assert table.step14_llm_selected_tool_plans == []
    assert table.step14_patent_scope.get("antibody_search_allowed") is False
    assert [r["ref_id"] for r in table.step14_request_refs] == ["r_payload"]
