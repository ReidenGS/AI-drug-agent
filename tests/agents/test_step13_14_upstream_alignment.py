"""Upstream context alignment for Step 13 EvidenceAgent and Step 14 PatentIPAgent.

These tests pin the fallback order documented in the agent docstrings:
- Step 13: Step 12 ranking → Step 10 handoff → Step 5 candidates.
- Step 14: Step 12 ranking → Step 10 handoff → Step 5/9 → Step 2 payload text.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from app.agents.candidate_context_agent import CandidateContextAgent
from app.agents.developability_agent import DevelopabilityAgent
from app.agents.evidence_agent import EvidenceAgent, _resolve_shortlist
from app.agents.patent_ip_agent import PatentIPAgent, _resolve_scope
from app.agents.structure_and_design_agent import StructureAndDesignAgent
from app.agents.supervisor_agent import SupervisorAgent
from app.llm.provider import MockLLMProvider
from app.mcp.client import LocalMCPClient
from app.services.input_readiness_service import InputReadinessService
from app.services.intake_service import IntakeService
from app.services.ranking_service import RankingService
from app.services.scoring_handoff_service import ScoringHandoffService
from app.services.scoring_validation_service import ScoringValidationService
from app.services.structured_query_service import StructuredQueryService
from app.services.tool_inventory_service import ToolInventoryService
from app.services.workflow_setup_service import WorkflowSetupService


PROJECT_ROOT = Path(__file__).resolve().parents[2].parent
DEFAULT_XLSX = PROJECT_ROOT / "\u9879\u76ee\u6587\u4ef6" / "ToolUniversity_inventory_v0.2.xlsx"


def _inventory() -> ToolInventoryService:
    xlsx = os.environ.get("TOOL_INVENTORY_XLSX", str(DEFAULT_XLSX))
    if not Path(xlsx).exists():
        pytest.skip(f"Inventory xlsx not at {xlsx}")
    return ToolInventoryService(xlsx)


def _mcp() -> LocalMCPClient:
    return LocalMCPClient(inventory=_inventory())


def _seed_to_step_12(local_storage, registry_service, workflow_state_service):
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


def _drop_artifact(local_storage, registry_service, run_id, *, registry_field: str, file_key: str | None = None):
    """Make the upstream artifact look absent: clear the registry pointer and
    optionally delete the on-disk JSON."""
    reg = registry_service.get(run_id)
    active = reg.active_artifacts.model_dump()
    active[registry_field] = None
    from app.schemas.registry import ActiveArtifacts
    new_reg = reg.model_copy(update={"active_artifacts": ActiveArtifacts(**active)})
    local_storage.write_json(
        local_storage.run_key(run_id, "registry/current.json"),
        new_reg.model_dump(),
    )
    if file_key and local_storage.exists(local_storage.run_key(run_id, file_key)):
        local_storage.delete(local_storage.run_key(run_id, file_key))


# ── _resolve_shortlist unit ─────────────────────────────────────────────────

def test_resolve_shortlist_prefers_completed_ranking():
    ranking = {
        "ranking_status": "completed",
        "ranked_candidates": [{"candidate_id": "cand_A"}, {"candidate_id": "cand_B"}],
    }
    handoff = {"candidate_ids": ["cand_X"]}
    out, src = _resolve_shortlist(ranking, handoff, [{"candidate_id": "cand_Y"}])
    assert out == ["cand_A", "cand_B"]
    assert src == "step_12_ranking"


def test_resolve_shortlist_falls_back_to_handoff_when_ranking_awaiting():
    ranking = {"ranking_status": "awaiting_external_scoring", "ranked_candidates": []}
    handoff = {"candidate_ids": ["cand_X", "cand_Y"]}
    out, src = _resolve_shortlist(ranking, handoff, [{"candidate_id": "cand_Z"}])
    assert out == ["cand_X", "cand_Y"]
    assert src == "step_10_handoff"


def test_resolve_shortlist_falls_back_to_candidates_when_no_handoff():
    out, src = _resolve_shortlist(None, None, [{"candidate_id": "cand_Z"}])
    assert out == ["cand_Z"]
    assert src == "step_05_candidates"


# ── Evidence: handoff shortlist when ranking awaiting ──────────────────────

def test_step13_uses_step10_handoff_when_ranking_awaiting(
    local_storage, registry_service, workflow_state_service
):
    """No external scoring file → Step 12 is `awaiting_external_scoring` with
    zero ranked_candidates. EvidenceAgent must drop to Step 10 handoff scope,
    not silently fall through to Step 5 (which would hide the Step 10
    integration)."""
    run_id = _seed_to_step_12(local_storage, registry_service, workflow_state_service)
    handoff = local_storage.read_json(
        local_storage.run_key(run_id, "scoring_handoff_package.json")
    )
    assert handoff["candidate_ids"]

    table = EvidenceAgent(
        storage=local_storage, registry=registry_service,
        workflow_state=workflow_state_service, mcp_client=_mcp(),
    ).run(run_id)

    multi = [tc for tc in table.tool_call_records if tc.tool_name == "MultiAgentLiteratureSearch"]
    assert multi, "MultiAgentLiteratureSearch should have been triggered for the shortlist"
    assert multi[0].tool_input_summary["shortlist_source"] == "step_10_handoff"
    # And the resolved shortlist query carries the Step 10 candidate_ids.
    payload_query = multi[0].tool_input_summary["query"]
    assert any(cid in payload_query for cid in handoff["candidate_ids"])


# ── Evidence: ranking-driven shortlist when valid scoring present ──────────

def test_step13_uses_step12_ranking_shortlist_when_completed(
    local_storage, registry_service, workflow_state_service
):
    run_id = _seed_to_step_12(local_storage, registry_service, workflow_state_service)
    handoff = local_storage.read_json(
        local_storage.run_key(run_id, "scoring_handoff_package.json")
    )
    cand_ids = handoff["candidate_ids"][:2]
    external = {
        "candidates": [
            {"candidate_id": cand_ids[0], "total_score": 7.0},
            {"candidate_id": cand_ids[1], "total_score": 9.0},
        ],
    }
    local_storage.write_json(
        local_storage.run_key(run_id, "inputs/external_scoring_result.json"), external
    )
    ScoringValidationService(local_storage, registry_service, workflow_state_service).validate(run_id)
    RankingService(local_storage, registry_service, workflow_state_service).build_ranking_table(run_id)

    table = EvidenceAgent(
        storage=local_storage, registry=registry_service,
        workflow_state=workflow_state_service, mcp_client=_mcp(),
    ).run(run_id)
    multi = [tc for tc in table.tool_call_records if tc.tool_name == "MultiAgentLiteratureSearch"]
    assert multi
    assert multi[0].tool_input_summary["shortlist_source"] == "step_12_ranking"


# ── _resolve_scope unit ────────────────────────────────────────────────────

def test_resolve_scope_prefers_ranking():
    ranking = {
        "ranking_status": "completed",
        "ranked_candidates": [{"candidate_id": "cand_top"}],
    }
    ids, src = _resolve_scope(ranking=ranking, handoff={"candidate_ids": ["cand_x"]}, cct_candidate_ids=["cand_y"])
    assert ids == {"cand_top"} and src == "step_12_ranking"


def test_resolve_scope_falls_back_to_handoff():
    ids, src = _resolve_scope(
        ranking={"ranking_status": "awaiting_external_scoring"},
        handoff={"candidate_ids": ["cand_h1", "cand_h2"]},
        cct_candidate_ids=["cand_other"],
    )
    assert ids == {"cand_h1", "cand_h2"} and src == "step_10_handoff"


def test_resolve_scope_falls_back_to_step5():
    ids, src = _resolve_scope(ranking=None, handoff=None, cct_candidate_ids=["cand_a"])
    assert ids == {"cand_a"} and src == "step_05_candidates"


# ── Patent: ranking shortlist filters tool calls ───────────────────────────

def test_step14_uses_step12_ranking_shortlist_for_compound_targets(
    local_storage, registry_service, workflow_state_service
):
    run_id = _seed_to_step_12(local_storage, registry_service, workflow_state_service)
    # Pick just the FIRST compound candidate as the "ranked" one; the agent
    # should only patent-search that candidate.
    cct = local_storage.read_json(local_storage.run_key(run_id, "candidate_context_table.json"))
    compound_ids = [
        c["candidate_id"] for c in cct["candidate_records"]
        if c["candidate_type"] == "compound_component"
    ]
    assert compound_ids, "fixture needs at least one compound_component candidate"
    ranked_id = compound_ids[0]

    external = {
        "candidates": [{"candidate_id": ranked_id, "total_score": 8.0}],
    }
    local_storage.write_json(
        local_storage.run_key(run_id, "inputs/external_scoring_result.json"), external
    )
    ScoringValidationService(local_storage, registry_service, workflow_state_service).validate(run_id)
    RankingService(local_storage, registry_service, workflow_state_service).build_ranking_table(run_id)

    table = PatentIPAgent(
        storage=local_storage, registry=registry_service,
        workflow_state=workflow_state_service, mcp_client=_mcp(),
    ).run(run_id)

    # Every compound-search call should reference the ranked candidate.
    compound_calls = [
        tc for tc in table.tool_call_records
        if tc.tool_name in {
            "drugbank_get_drug_references_by_drug_name_or_id",
            "FDA_OrangeBook_get_patent_info",
            "PubChem_get_associated_patents_by_CID",
        }
    ]
    assert compound_calls
    for tc in compound_calls:
        assert tc.tool_input_summary["candidate_id"] == ranked_id
        assert tc.tool_input_summary["shortlist_source"] == "step_12_ranking"


# ── Patent: structured_query payload fallback ──────────────────────────────

def test_step14_uses_structured_query_payload_text_when_no_step5_compound(
    local_storage, registry_service, workflow_state_service
):
    """If Step 5 produced no compound_component candidate (we wipe them
    here), but structured_query still mentions a payload, Step 14 must still
    issue DrugBank + Orange Book queries from that text."""
    run_id = _seed_to_step_12(local_storage, registry_service, workflow_state_service)

    # Wipe compound candidates from Step 5 (and Step 9 compound_hits) so we
    # exercise the structured_query fallback path.
    cct_path = local_storage.run_key(run_id, "candidate_context_table.json")
    cct = local_storage.read_json(cct_path)
    cct["candidate_records"] = [
        c for c in cct["candidate_records"] if c["candidate_type"] != "compound_component"
    ]
    local_storage.write_json(cct_path, cct)

    cs_path = local_storage.run_key(run_id, "compound_screening_artifact.json")
    cs = local_storage.read_json(cs_path)
    cs["compound_hits"] = []
    local_storage.write_json(cs_path, cs)

    table = PatentIPAgent(
        storage=local_storage, registry=registry_service,
        workflow_state=workflow_state_service, mcp_client=_mcp(),
    ).run(run_id)

    sq_fallback_calls = [
        tc for tc in table.tool_call_records
        if tc.tool_input_summary.get("shortlist_source") == "step_02_structured_query"
    ]
    assert sq_fallback_calls, (
        "expected DrugBank/Orange Book calls derived from structured_query payload text"
    )
    tool_names = {tc.tool_name for tc in sq_fallback_calls}
    assert "drugbank_get_drug_references_by_drug_name_or_id" in tool_names
    assert "FDA_OrangeBook_get_patent_info" in tool_names


# ── raw isolation still holds ──────────────────────────────────────────────

def test_step13_step14_raw_payload_still_does_not_leak_after_upgrade(
    local_storage, registry_service, workflow_state_service
):
    import json

    run_id = _seed_to_step_12(local_storage, registry_service, workflow_state_service)
    evidence = EvidenceAgent(
        storage=local_storage, registry=registry_service,
        workflow_state=workflow_state_service, mcp_client=_mcp(),
    ).run(run_id)
    patent = PatentIPAgent(
        storage=local_storage, registry=registry_service,
        workflow_state=workflow_state_service, mcp_client=_mcp(),
    ).run(run_id)

    e_blob = json.dumps([r.model_dump() for r in evidence.evidence_records])
    p_blob = json.dumps([r.model_dump() for r in patent.patent_records])
    assert "mocked" not in e_blob
    assert "mocked" not in p_blob
