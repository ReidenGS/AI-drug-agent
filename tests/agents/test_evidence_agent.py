"""Step 13 EvidenceAgent tests."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from app.agents.candidate_context_agent import CandidateContextAgent
from app.agents.developability_agent import DevelopabilityAgent
from app.agents.evidence_agent import EvidenceAgent
from app.agents.structure_and_design_agent import StructureAndDesignAgent
from app.agents.supervisor_agent import SupervisorAgent
from app.llm.provider import MockLLMProvider
from app.mcp.client import LocalMCPClient
from app.services.input_readiness_service import InputReadinessService
from app.services.intake_service import IntakeService
from app.services.scoring_handoff_service import ScoringHandoffService
from app.services.scoring_validation_service import ScoringValidationService
from app.services.ranking_service import RankingService
from app.services.structured_query_service import StructuredQueryService
from app.services.tool_inventory_service import ToolInventoryService
from app.services.workflow_setup_service import WorkflowSetupService
from app.utils.errors import WorkflowStateError


PROJECT_ROOT = Path(__file__).resolve().parents[2].parent
DEFAULT_XLSX = PROJECT_ROOT / "\u9879\u76ee\u6587\u4ef6" / "ToolUniversity_inventory_v0.2.xlsx"


def _inventory():
    xlsx = os.environ.get("TOOL_INVENTORY_XLSX", str(DEFAULT_XLSX))
    if not Path(xlsx).exists():
        pytest.skip(f"Inventory xlsx not at {xlsx}")
    return ToolInventoryService(xlsx)


def _mcp() -> LocalMCPClient:
    return LocalMCPClient(inventory=_inventory())


def _seed_through_step_12(local_storage, registry_service, workflow_state_service):
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
        workflow_state=workflow_state_service, mcp_client=_mcp(),
    )
    sd.run_step_7(run_id)
    sd.run_step_8(run_id)
    sd.run_step_9(run_id)
    ScoringHandoffService(local_storage, registry_service, workflow_state_service).prepare(run_id)
    ScoringValidationService(local_storage, registry_service, workflow_state_service).validate(run_id)
    RankingService(local_storage, registry_service, workflow_state_service).build_ranking_table(run_id)
    return run_id


def test_step13_builds_evidence_records_from_target_payload_candidates(
    local_storage, registry_service, workflow_state_service
):
    run_id = _seed_through_step_12(local_storage, registry_service, workflow_state_service)
    table = EvidenceAgent(
        storage=local_storage, registry=registry_service,
        workflow_state=workflow_state_service, mcp_client=_mcp(),
    ).run(run_id)

    # Tool routing fired for target, payload, and candidates.
    tool_names = {tc.tool_name for tc in table.tool_call_records}
    assert "EuropePMC_search_articles" in tool_names
    assert "LiteratureSearchTool" in tool_names
    assert "PubTator3_LiteratureSearch" in tool_names

    # Records exist and carry source attribution.
    assert table.evidence_records
    sources = {r.source for r in table.evidence_records}
    assert sources & {"EuropePMC_search_articles", "LiteratureSearchTool", "PubTator3_LiteratureSearch"}


def test_step13_raw_payload_not_in_normalized_artifact(
    local_storage, registry_service, workflow_state_service
):
    """Mock wrappers stamp `"mocked"` into envelopes; that string must not
    leak into evidence_records / table top-level fields."""
    run_id = _seed_through_step_12(local_storage, registry_service, workflow_state_service)
    table = EvidenceAgent(
        storage=local_storage, registry=registry_service,
        workflow_state=workflow_state_service, mcp_client=_mcp(),
    ).run(run_id)

    blob = json.dumps([r.model_dump() for r in table.evidence_records])
    assert "mocked" not in blob
    # And the raw tool_output_ref files exist on disk.
    for tc in table.tool_call_records:
        if tc.run_status == "success":
            assert tc.tool_output_ref
            raw = local_storage.read_json(tc.tool_output_ref)
            assert "output" in raw


def test_step13_partial_when_wrappers_unwired(
    local_storage, registry_service, workflow_state_service
):
    from app.mcp.tools._registry import _all_bindings

    def _ni(**_):
        raise NotImplementedError

    bindings = dict(_all_bindings())
    for name in (
        "LiteratureSearchTool", "EuropePMC_search_articles", "openalex_search_works",
        "PubTator3_LiteratureSearch", "PubTator3_get_annotations",
        "SemanticScholar_search_papers", "MultiAgentLiteratureSearch",
    ):
        bindings[name] = _ni
    mcp = LocalMCPClient(inventory=_inventory(), bindings=bindings)

    run_id = _seed_through_step_12(local_storage, registry_service, workflow_state_service)
    table = EvidenceAgent(
        storage=local_storage, registry=registry_service,
        workflow_state=workflow_state_service, mcp_client=mcp,
    ).run(run_id)
    assert table.review_status in {"partial", "failed"}
    statuses = [tc.run_status for tc in table.tool_call_records]
    assert "dependency_unavailable" in statuses


def test_step13_requires_step5_artifact(
    local_storage, registry_service, workflow_state_service
):
    intake = IntakeService(local_storage, registry_service, workflow_state_service)
    rec = intake.submit(
        raw_user_query="x", user_provided_context={"target_or_antigen_text": "HER2"}
    )
    with pytest.raises(WorkflowStateError):
        EvidenceAgent(
            storage=local_storage, registry=registry_service,
            workflow_state=workflow_state_service, mcp_client=LocalMCPClient(),
        ).run(rec.run_id)


# ── Systematic review hardening helpers ─────────────────────────────────────

def _seed_through_step_5(
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
    """Bindings that return a fixed payload regardless of query args."""

    def make(payload):
        def _fn(**_kw):
            return payload
        return _fn
    return {name: make(p) for name, p in canned.items()}


# ── 1. downstream hints drive query construction ─────────────────────────────

def test_step13_uses_downstream_query_hints(
    local_storage, registry_service, workflow_state_service
):
    run_id = _seed_through_step_5(local_storage, registry_service, workflow_state_service)
    # canned hit on the payload search
    bindings = _bind_fixed({
        "LiteratureSearchTool": {
            "status": "mocked",
            "results": [
                {"title": "vc-MMAE ADC payload efficacy",
                 "doi": "10.1234/AAA", "year": 2021,
                 "abstract": "Antibody-drug conjugate study."},
            ],
        },
        "EuropePMC_search_articles": {
            "status": "mocked",
            "results": [
                {"title": "HER2 overexpression in breast cancer",
                 "doi": "10.5678/BBB", "year": 2019,
                 "abstract": "Target context."},
            ],
        },
    })
    mcp = LocalMCPClient(inventory=_inventory(), bindings=bindings)
    table = EvidenceAgent(
        storage=local_storage, registry=registry_service,
        workflow_state=workflow_state_service, mcp_client=mcp,
    ).run(run_id)

    # Each evidence record must carry the role/term it came from.
    roles = {r.query_role for r in table.evidence_records if r.query_role}
    assert roles & {"payload", "target", "linker_payload"}, (
        f"expected payload/target/linker_payload roles, got: {roles}"
    )
    # Recorded query_term must reference Step 5 downstream hints.
    terms = {(r.query_term or "").lower() for r in table.evidence_records}
    assert any("her2" in t for t in terms)
    assert any("mmae" in t for t in terms)


# ── 2. DOI dedup across multiple search calls ────────────────────────────────

def test_step13_dedup_by_doi_across_tools(
    local_storage, registry_service, workflow_state_service
):
    run_id = _seed_through_step_5(local_storage, registry_service, workflow_state_service)
    shared_hit = {
        "title": "Same paper variant A",
        "doi": "10.9999/SAME-DOI",
        "year": 2020,
        "abstract": "ADC review.",
    }
    bindings = _bind_fixed({
        "LiteratureSearchTool": {
            "status": "mocked",
            "results": [shared_hit, {"title": "Other", "doi": "10.1/UNIQ", "year": 2022}],
        },
        "EuropePMC_search_articles": {
            "status": "mocked",
            "results": [
                {"title": "Same paper variant B (formatting)",
                 "doi": "https://doi.org/10.9999/SAME-DOI",
                 "year": 2020, "abstract": "ADC."},
            ],
        },
    })
    mcp = LocalMCPClient(inventory=_inventory(), bindings=bindings)
    table = EvidenceAgent(
        storage=local_storage, registry=registry_service,
        workflow_state=workflow_state_service, mcp_client=mcp,
    ).run(run_id)

    # The shared DOI must appear exactly once after dedup, with both sources.
    same_doi_records = [r for r in table.evidence_records if (r.doi or "").lower().endswith("same-doi")]
    assert len(same_doi_records) == 1, (
        f"DOI dedup failed; got {len(same_doi_records)} records for shared DOI"
    )
    rec = same_doi_records[0]
    assert {"LiteratureSearchTool", "EuropePMC_search_articles"} <= set(rec.sources), (
        f"sources should merge across tools; got {rec.sources}"
    )
    # source_refs should also be populated with one ref per source.
    assert len(rec.source_refs) >= 2


# ── 3. Title dedup when DOI missing ──────────────────────────────────────────

def test_step13_dedup_by_title_when_no_doi(
    local_storage, registry_service, workflow_state_service
):
    run_id = _seed_through_step_5(local_storage, registry_service, workflow_state_service)
    bindings = _bind_fixed({
        "LiteratureSearchTool": {
            "status": "mocked",
            "results": [
                {"title": "Novel ADC payload screen", "year": 2021},
            ],
        },
        "EuropePMC_search_articles": {
            "status": "mocked",
            "results": [
                {"title": "  novel adc payload   screen  ", "year": 2021},
            ],
        },
    })
    mcp = LocalMCPClient(inventory=_inventory(), bindings=bindings)
    table = EvidenceAgent(
        storage=local_storage, registry=registry_service,
        workflow_state=workflow_state_service, mcp_client=mcp,
    ).run(run_id)

    matching = [
        r for r in table.evidence_records
        if (r.title or "").lower().replace(" ", "").startswith("noveladcpayloadscreen")
    ]
    assert len(matching) == 1
    assert {"LiteratureSearchTool", "EuropePMC_search_articles"} <= set(matching[0].sources)


# ── 4. Deterministic relevance ranking on hits ──────────────────────────────

def test_step13_relevance_ranking_prefers_adc_payload_hits(
    local_storage, registry_service, workflow_state_service
):
    run_id = _seed_through_step_5(local_storage, registry_service, workflow_state_service)
    bindings = _bind_fixed({
        "LiteratureSearchTool": {
            "status": "mocked",
            "results": [
                {"title": "Generic monoclonal antibody crystal structure",
                 "doi": "10.111/A", "year": 2010, "abstract": "Mab."},
                {"title": "vc-MMAE ADC payload efficacy in HER2+ tumors",
                 "doi": "10.222/B", "year": 2022,
                 "abstract": "Antibody-drug conjugate trial."},
            ],
        },
    })
    mcp = LocalMCPClient(inventory=_inventory(), bindings=bindings)
    table = EvidenceAgent(
        storage=local_storage, registry=registry_service,
        workflow_state=workflow_state_service, mcp_client=mcp,
    ).run(run_id)

    indexed = {(r.doi or "").lower(): (i, r) for i, r in enumerate(table.evidence_records)}
    assert "10.222/b" in indexed and "10.111/a" in indexed
    i_adc, rec_adc = indexed["10.222/b"]
    i_other, rec_other = indexed["10.111/a"]
    assert (rec_adc.relevance_score or 0) > (rec_other.relevance_score or 0), (
        f"ADC payload hit should outrank generic hit; scores: {rec_adc.relevance_score} vs {rec_other.relevance_score}"
    )
    assert i_adc < i_other, "ADC payload record should appear before generic record"


# ── 5. Raw payload isolation: abstract not embedded in normalized records ────

def test_step13_systematic_review_raw_payload_isolated(
    local_storage, registry_service, workflow_state_service
):
    run_id = _seed_through_step_5(local_storage, registry_service, workflow_state_service)
    bindings = _bind_fixed({
        "LiteratureSearchTool": {
            "status": "mocked",
            "results": [
                {"title": "ADC trial", "doi": "10.1/X", "year": 2020,
                 "abstract": "SECRET_FULL_ABSTRACT_BODY_TEXT_TO_NOT_LEAK"},
            ],
        },
    })
    mcp = LocalMCPClient(inventory=_inventory(), bindings=bindings)
    table = EvidenceAgent(
        storage=local_storage, registry=registry_service,
        workflow_state=workflow_state_service, mcp_client=mcp,
    ).run(run_id)
    blob = json.dumps([r.model_dump() for r in table.evidence_records])
    assert "SECRET_FULL_ABSTRACT_BODY_TEXT_TO_NOT_LEAK" not in blob
    # tool_output_ref still has it.
    persisted_refs = [
        tc.tool_output_ref for tc in table.tool_call_records
        if tc.run_status == "success" and tc.tool_output_ref
    ]
    assert any(
        "SECRET_FULL_ABSTRACT_BODY_TEXT_TO_NOT_LEAK"
        in json.dumps(local_storage.read_json(ref))
        for ref in persisted_refs
    )


# ── 6. Antibody-centered queries only when Step 5 hint exists ───────────────

def test_step13_does_not_query_antibody_when_no_antibody_hint(
    local_storage, registry_service, workflow_state_service
):
    run_id = _seed_through_step_5(
        local_storage, registry_service, workflow_state_service,
        user_ctx={
            "target_or_antigen_text": "TROP2",
            "payload_linker_text": "MMAE",
            # No `candidate_text` and no antibody_candidate_text → no
            # antibody hint in Step 5 downstream hints.
        },
    )
    cct = local_storage.read_json(
        local_storage.run_key(run_id, "candidate_context_table.json")
    )
    hint_roles = {h["role"] for h in cct.get("downstream_query_hints") or []}
    assert "antibody" not in hint_roles, "fixture seeded wrong; antibody hint should be absent"

    bindings = _bind_fixed({
        "LiteratureSearchTool": {"status": "mocked", "results": []},
        "EuropePMC_search_articles": {"status": "mocked", "results": []},
    })
    mcp = LocalMCPClient(inventory=_inventory(), bindings=bindings)
    table = EvidenceAgent(
        storage=local_storage, registry=registry_service,
        workflow_state=workflow_state_service, mcp_client=mcp,
    ).run(run_id)

    # No tool_call should carry an antibody-role query term (e.g. Trastuzumab).
    query_roles = {
        (tc.tool_input_summary or {}).get("query_role")
        for tc in table.tool_call_records
    }
    assert "antibody" not in query_roles
    # And no antibody name should appear as a query term in any tool call.
    for tc in table.tool_call_records:
        qt = ((tc.tool_input_summary or {}).get("query_term") or "").lower()
        assert "trastuzumab" not in qt


# ── 7. Source refs / traceability ───────────────────────────────────────────

def test_step13_every_hit_record_has_source_refs(
    local_storage, registry_service, workflow_state_service
):
    run_id = _seed_through_step_5(local_storage, registry_service, workflow_state_service)
    bindings = _bind_fixed({
        "LiteratureSearchTool": {
            "status": "mocked",
            "results": [
                {"title": "ADC paper", "doi": "10.1/Y", "year": 2021},
            ],
        },
    })
    mcp = LocalMCPClient(inventory=_inventory(), bindings=bindings)
    table = EvidenceAgent(
        storage=local_storage, registry=registry_service,
        workflow_state=workflow_state_service, mcp_client=mcp,
    ).run(run_id)
    for r in table.evidence_records:
        # search-executed receipts may lack hit-level title; hit-derived
        # records must carry source_refs pointing back to tool outputs.
        if r.title:
            assert r.source_refs, f"hit-derived record missing source_refs: {r}"
            for ref in r.source_refs:
                assert local_storage.exists(ref)


# ── 8. Per-query / total limit is configurable, not hardcoded to a tiny n ────

def test_step13_total_limit_supports_large_n_by_design(
    local_storage, registry_service, workflow_state_service
):
    """Demo limit can be small, but the public limit parameter must allow
    a much larger pool (≥1000) for production usage."""
    from app.agents.evidence_agent import DEFAULT_TOTAL_LIMIT, MAX_TOTAL_LIMIT
    assert MAX_TOTAL_LIMIT >= 1000
    assert DEFAULT_TOTAL_LIMIT >= 1
