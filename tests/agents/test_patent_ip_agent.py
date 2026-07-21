"""Step 14 PatentIPAgent tests."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from app.agents.candidate_context_agent import CandidateContextAgent
from app.agents.developability_agent import DevelopabilityAgent
from app.agents.patent_ip_agent import PatentIPAgent
from app.agents.structure_and_design_agent import StructureAndDesignAgent
from app.agents.supervisor_agent import SupervisorAgent
from app.llm.provider import MockLLMProvider
from app.mcp.client import LocalMCPClient
from app.services.input_readiness_service import InputReadinessService
from app.services.intake_service import IntakeService
from app.services.structured_query_service import StructuredQueryService
from app.services.tool_inventory_service import ToolInventoryService
from app.services.workflow_setup_service import WorkflowSetupService


PROJECT_ROOT = Path(__file__).resolve().parents[2].parent
DEFAULT_XLSX = PROJECT_ROOT / "\u9879\u76ee\u6587\u4ef6" / "ToolUniversity_inventory_v0.2.xlsx"


def _inventory():
    xlsx = os.environ.get("TOOL_INVENTORY_XLSX", str(DEFAULT_XLSX))
    if not Path(xlsx).exists():
        pytest.skip(f"Inventory xlsx not at {xlsx}")
    return ToolInventoryService(xlsx)


def _fda_fixture_mcp() -> LocalMCPClient:
    """Observed-shape test envelope; not live MCP/ToolUniverse evidence."""

    def _fda(**_kwargs):
        return {
            "status": "ok",
            "executor": "test_fixture",
            "payload": {"data": {"drugs": []}},
        }

    return LocalMCPClient(
        inventory=_inventory(), bindings={"FDA_OrangeBook_get_patent_info": _fda}
    )


def _seed_through_step_9(local_storage, registry_service, workflow_state_service, *, referenced_inputs=None):
    intake = IntakeService(local_storage, registry_service, workflow_state_service)
    rec = intake.submit(
        raw_user_query="HER2 ADC vc-MMAE",
        user_provided_context={
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
    DevelopabilityAgent(
        storage=local_storage, registry=registry_service,
        workflow_state=workflow_state_service, mcp_client=LocalMCPClient(),
    ).run(run_id)
    sd = StructureAndDesignAgent(
        storage=local_storage, registry=registry_service,
        workflow_state=workflow_state_service,
        mcp_client=LocalMCPClient(inventory=_inventory()),
    )
    sd.run_step_7(run_id)
    sd.run_step_8(run_id)
    sd.run_step_9(run_id)
    return run_id


def test_step14_routes_drugbank_and_orangebook_for_payload(
    local_storage, registry_service, workflow_state_service
):
    run_id = _seed_through_step_9(local_storage, registry_service, workflow_state_service)
    table = PatentIPAgent(
        storage=local_storage, registry=registry_service,
        workflow_state=workflow_state_service,
        mcp_client=LocalMCPClient(inventory=_inventory()),
    ).run(run_id)

    tools = {tc.tool_name for tc in table.tool_call_records}
    assert "drugbank_get_drug_references_by_drug_name_or_id" in tools
    assert "FDA_OrangeBook_get_patent_info" in tools


def test_step14_routes_pubchem_when_compound_has_pubchem_cid(
    local_storage, registry_service, workflow_state_service
):
    run_id = _seed_through_step_9(
        local_storage, registry_service, workflow_state_service,
        referenced_inputs=[
            {"id_type": "pubchem_cid", "value": "123456", "source": "raw_request_text"},
        ],
    )
    table = PatentIPAgent(
        storage=local_storage, registry=registry_service,
        workflow_state=workflow_state_service,
        mcp_client=LocalMCPClient(inventory=_inventory()),
    ).run(run_id)
    tools = {tc.tool_name for tc in table.tool_call_records}
    assert "PubChem_get_associated_patents_by_CID" in tools


def test_step14_orangebook_uses_canonical_normalized_fields(
    local_storage, registry_service, workflow_state_service
):
    run_id = _seed_through_step_9(local_storage, registry_service, workflow_state_service)
    table = PatentIPAgent(
        storage=local_storage, registry=registry_service,
        workflow_state=workflow_state_service,
        mcp_client=_fda_fixture_mcp(),
    ).run(run_id)
    ob = [r for r in table.patent_records if r.source_database == "FDA_OrangeBook"]
    assert ob, "expected at least one Orange Book row"
    assert all(
        r.matched_entity_type == "drug_application_or_regulatory_reference" for r in ob
    )
    # legal_disclaimer present
    assert "demonstration purposes only" in table.legal_disclaimer.lower()


def test_step14_orangebook_raw_payload_not_in_normalized_record(
    local_storage, registry_service, workflow_state_service
):
    run_id = _seed_through_step_9(local_storage, registry_service, workflow_state_service)
    table = PatentIPAgent(
        storage=local_storage, registry=registry_service,
        workflow_state=workflow_state_service,
        mcp_client=_fda_fixture_mcp(),
    ).run(run_id)
    blob = json.dumps([r.model_dump() for r in table.patent_records])
    # The observed-shape test fixture carries nested `data.drugs`. Normalized
    # rows must NOT carry that raw
    # envelope — the storage_ref is the only allowed escape hatch.
    assert "mocked" not in blob
    assert '"records":' not in blob.replace(" ", "")  # raw OB list field
    # And the raw payload IS persisted under the tool_output_ref so we
    # haven't simply dropped it on the floor.
    refs = [
        tc.tool_output_ref for tc in table.tool_call_records
        if tc.run_status == "success" and tc.tool_name == "FDA_OrangeBook_get_patent_info"
    ]
    assert refs
    for ref in refs:
        raw = local_storage.read_json(ref)
        assert raw["output"]["executor"] == "test_fixture"
        assert raw["output"]["payload"] == {"data": {"drugs": []}}


def test_step14_partial_when_wrappers_unwired(
    local_storage, registry_service, workflow_state_service
):
    from app.mcp.tools._registry import _all_bindings

    def _ni(**_):
        raise NotImplementedError

    bindings = dict(_all_bindings())
    for name in (
        "PubChem_get_associated_patents_by_CID",
        "drugbank_get_drug_references_by_drug_name_or_id",
        "FDA_OrangeBook_get_patent_info",
    ):
        bindings[name] = _ni
    mcp = LocalMCPClient(inventory=_inventory(), bindings=bindings)

    run_id = _seed_through_step_9(local_storage, registry_service, workflow_state_service)
    table = PatentIPAgent(
        storage=local_storage, registry=registry_service,
        workflow_state=workflow_state_service, mcp_client=mcp,
    ).run(run_id)
    assert table.patent_review_status in {"partial", "failed"}
    statuses = [tc.run_status for tc in table.tool_call_records]
    assert "dependency_unavailable" in statuses


# ── Systematic prior-art normalization tests ────────────────────────────────


def _seed_through_step_5_only(
    local_storage, registry_service, workflow_state_service,
    *, user_ctx: dict | None = None,
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


def test_step14_uses_downstream_query_hints_for_query_construction(
    local_storage, registry_service, workflow_state_service
):
    run_id = _seed_through_step_5_only(local_storage, registry_service, workflow_state_service)
    mcp = LocalMCPClient(inventory=_inventory())
    table = PatentIPAgent(
        storage=local_storage, registry=registry_service,
        workflow_state=workflow_state_service, mcp_client=mcp,
    ).run(run_id)
    roles = {(tc.tool_input_summary or {}).get("query_role") for tc in table.tool_call_records}
    assert roles - {None}, f"expected query_role set on at least some tool calls; got {roles}"
    # Step 5 downstream hint roles should be represented.
    assert roles & {"payload", "linker_payload", "linker"}, (
        f"payload/linker_payload/linker role missing; got {roles}"
    )
    # Each tool call has a query_term referencing a Step 5 hint.
    terms = {((tc.tool_input_summary or {}).get("query_term") or "").lower() for tc in table.tool_call_records}
    assert any("mmae" in t for t in terms)


def test_step14_no_antibody_query_when_no_antibody_hint(
    local_storage, registry_service, workflow_state_service
):
    run_id = _seed_through_step_5_only(
        local_storage, registry_service, workflow_state_service,
        user_ctx={
            "target_or_antigen_text": "TROP2",
            "payload_linker_text": "MMAE",
        },
    )
    cct = local_storage.read_json(
        local_storage.run_key(run_id, "candidate_context_table.json")
    )
    hint_roles = {h["role"] for h in cct.get("downstream_query_hints") or []}
    assert "antibody" not in hint_roles
    table = PatentIPAgent(
        storage=local_storage, registry=registry_service,
        workflow_state=workflow_state_service,
        mcp_client=LocalMCPClient(inventory=_inventory()),
    ).run(run_id)
    for tc in table.tool_call_records:
        assert (tc.tool_input_summary or {}).get("query_role") != "antibody"
        qt = ((tc.tool_input_summary or {}).get("query_term") or "").lower()
        assert "trastuzumab" not in qt


def test_step14_antibody_query_only_when_antibody_hint_present(
    local_storage, registry_service, workflow_state_service
):
    run_id = _seed_through_step_5_only(local_storage, registry_service, workflow_state_service)
    cct = local_storage.read_json(
        local_storage.run_key(run_id, "candidate_context_table.json")
    )
    hint_roles = {h["role"] for h in cct.get("downstream_query_hints") or []}
    assert "antibody" in hint_roles
    table = PatentIPAgent(
        storage=local_storage, registry=registry_service,
        workflow_state=workflow_state_service,
        mcp_client=LocalMCPClient(inventory=_inventory()),
    ).run(run_id)
    roles = {(tc.tool_input_summary or {}).get("query_role") for tc in table.tool_call_records}
    assert "antibody" in roles


def test_step14_dedup_by_patent_number_across_tools(
    local_storage, registry_service, workflow_state_service
):
    run_id = _seed_through_step_5_only(local_storage, registry_service, workflow_state_service)
    bindings = _bind_fixed({
        "drugbank_get_drug_references_by_drug_name_or_id": {
            "status": "ok", "source": "drugbank_get_drug_references_by_drug_name_or_id",
            "references": [
                {"patent_number": "US-1234567-B2", "title": "ADC with vc-MMAE",
                 "assignee": "AcmeBio", "publication_date": "2020-05-01"},
                {"patent_number": "EP-OTHER", "title": "Other"},
            ],
        },
        "FDA_OrangeBook_get_patent_info": {
            "status": "ok", "source": "FDA_OrangeBook_get_patent_info",
            "records": [
                {"patent_number": "us 1234567 b2",
                 "title": "Variant title", "assignee": "Acme Bio"},
            ],
        },
    })
    mcp = LocalMCPClient(inventory=_inventory(), bindings=bindings)
    table = PatentIPAgent(
        storage=local_storage, registry=registry_service,
        workflow_state=workflow_state_service, mcp_client=mcp,
    ).run(run_id)
    same = [
        r for r in table.patent_records
        if (r.patent_number or "").lower().replace(" ", "").replace("-", "") == "us1234567b2"
    ]
    assert len(same) == 1, f"patent_number dedup failed; got {len(same)}"
    assert {"DrugBank", "FDA_OrangeBook"} <= set(same[0].sources)
    assert len(same[0].source_refs) >= 2


def test_step14_dedup_by_title_assignee_when_no_patent_number(
    local_storage, registry_service, workflow_state_service
):
    run_id = _seed_through_step_5_only(local_storage, registry_service, workflow_state_service)
    bindings = _bind_fixed({
        "drugbank_get_drug_references_by_drug_name_or_id": {
            "status": "ok", "source": "drugbank_get_drug_references_by_drug_name_or_id",
            "references": [{"title": "MMAE Conjugation Method", "assignee": "AcmeBio"}],
        },
        "FDA_OrangeBook_get_patent_info": {
            "status": "ok", "source": "FDA_OrangeBook_get_patent_info",
            "records": [{"title": "  mmae   conjugation method  ", "assignee": "acmebio"}],
        },
    })
    mcp = LocalMCPClient(inventory=_inventory(), bindings=bindings)
    table = PatentIPAgent(
        storage=local_storage, registry=registry_service,
        workflow_state=workflow_state_service, mcp_client=mcp,
    ).run(run_id)
    same = [
        r for r in table.patent_records
        if (r.patent_title or "").lower().startswith("mmae conjugation method")
    ]
    assert len(same) == 1, f"title+assignee dedup failed; got {len(same)}"
    assert {"DrugBank", "FDA_OrangeBook"} <= set(same[0].sources)


def test_step14_relevance_scoring_prefers_payload_conjugation_hits(
    local_storage, registry_service, workflow_state_service
):
    run_id = _seed_through_step_5_only(local_storage, registry_service, workflow_state_service)
    bindings = _bind_fixed({
        "drugbank_get_drug_references_by_drug_name_or_id": {
            "status": "ok", "source": "drugbank_get_drug_references_by_drug_name_or_id",
            "references": [
                {"patent_number": "US-AAA",
                 "title": "Generic antibody monoclonal mAb description",
                 "publication_date": "2008-01-01"},
                {"patent_number": "US-BBB",
                 "title": "Antibody-drug conjugate with vc-MMAE cleavable linker",
                 "publication_date": "2022-03-01"},
            ],
        },
    })
    mcp = LocalMCPClient(inventory=_inventory(), bindings=bindings)
    table = PatentIPAgent(
        storage=local_storage, registry=registry_service,
        workflow_state=workflow_state_service, mcp_client=mcp,
    ).run(run_id)
    indexed = {
        (r.patent_number or "").upper(): (i, r)
        for i, r in enumerate(table.patent_records) if r.patent_number
    }
    assert "US-BBB" in indexed and "US-AAA" in indexed
    i_b, rec_b = indexed["US-BBB"]
    i_a, rec_a = indexed["US-AAA"]
    assert (rec_b.ip_relevance_score or 0) > (rec_a.ip_relevance_score or 0)
    assert i_b < i_a, "ADC/conjugation hit should appear before generic mAb hit"


def test_step14_raw_payload_isolation_for_extracted_records(
    local_storage, registry_service, workflow_state_service
):
    run_id = _seed_through_step_5_only(local_storage, registry_service, workflow_state_service)
    bindings = _bind_fixed({
        "drugbank_get_drug_references_by_drug_name_or_id": {
            "status": "ok", "source": "drugbank_get_drug_references_by_drug_name_or_id",
            "references": [{
                "patent_number": "US-DESCR",
                "title": "ADC payload",
                "description": "SECRET_FULL_DESCRIPTION_THAT_MUST_NOT_LEAK",
                "claims": ["SECRET_FULL_CLAIMS_LIST_DO_NOT_LEAK"],
                "abstract": "SECRET_FULL_ABSTRACT_HERE",
            }],
        },
    })
    mcp = LocalMCPClient(inventory=_inventory(), bindings=bindings)
    table = PatentIPAgent(
        storage=local_storage, registry=registry_service,
        workflow_state=workflow_state_service, mcp_client=mcp,
    ).run(run_id)
    blob = json.dumps([r.model_dump() for r in table.patent_records])
    assert "SECRET_FULL_DESCRIPTION_THAT_MUST_NOT_LEAK" not in blob
    assert "SECRET_FULL_CLAIMS_LIST_DO_NOT_LEAK" not in blob
    assert "SECRET_FULL_ABSTRACT_HERE" not in blob
    refs = [
        tc.tool_output_ref for tc in table.tool_call_records
        if tc.run_status == "success" and tc.tool_output_ref
    ]
    assert any(
        "SECRET_FULL_DESCRIPTION" in json.dumps(local_storage.read_json(r))
        for r in refs
    )


def test_step14_each_extracted_record_has_source_refs(
    local_storage, registry_service, workflow_state_service
):
    run_id = _seed_through_step_5_only(local_storage, registry_service, workflow_state_service)
    bindings = _bind_fixed({
        "drugbank_get_drug_references_by_drug_name_or_id": {
            "status": "ok", "source": "drugbank_get_drug_references_by_drug_name_or_id",
            "references": [{"patent_number": "US-A", "title": "ADC linker patent"}],
        },
    })
    mcp = LocalMCPClient(inventory=_inventory(), bindings=bindings)
    table = PatentIPAgent(
        storage=local_storage, registry=registry_service,
        workflow_state=workflow_state_service, mcp_client=mcp,
    ).run(run_id)
    extracted = [r for r in table.patent_records if r.patent_number]
    assert extracted
    for r in extracted:
        assert r.source_refs
        for ref in r.source_refs:
            assert local_storage.exists(ref)


def test_step14_does_not_write_step12_ranking_table(
    local_storage, registry_service, workflow_state_service
):
    run_id = _seed_through_step_5_only(local_storage, registry_service, workflow_state_service)
    ranking_key = local_storage.run_key(run_id, "ranking_table.json")
    assert not local_storage.exists(ranking_key)
    PatentIPAgent(
        storage=local_storage, registry=registry_service,
        workflow_state=workflow_state_service,
        mcp_client=LocalMCPClient(inventory=_inventory()),
    ).run(run_id)
    assert not local_storage.exists(ranking_key)
    reg = registry_service.get(run_id)
    assert reg.active_artifacts.ranking_table_id is None


def test_step14_limit_constants_support_large_n():
    from app.agents.patent_ip_agent import DEFAULT_TOTAL_LIMIT, MAX_TOTAL_LIMIT
    assert MAX_TOTAL_LIMIT >= 1000
    assert DEFAULT_TOTAL_LIMIT >= 1
