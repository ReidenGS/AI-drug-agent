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
