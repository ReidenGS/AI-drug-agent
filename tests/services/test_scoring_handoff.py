"""Step 10 — ScoringHandoffService."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from app.agents.candidate_context_agent import CandidateContextAgent
from app.agents.developability_agent import DevelopabilityAgent
from app.agents.structure_and_design_agent import StructureAndDesignAgent
from app.agents.supervisor_agent import SupervisorAgent
from app.llm.provider import MockLLMProvider
from app.mcp.client import LocalMCPClient
from app.services.input_readiness_service import InputReadinessService
from app.services.intake_service import IntakeService
from app.services.scoring_handoff_service import ScoringHandoffService
from app.services.structured_query_service import StructuredQueryService
from app.services.tool_inventory_service import ToolInventoryService
from app.services.workflow_setup_service import WorkflowSetupService


PROJECT_ROOT = Path(__file__).resolve().parents[2].parent
DEFAULT_XLSX = PROJECT_ROOT / "\u9879\u76ee\u6587\u4ef6" / "ToolUniversity_inventory_v0.2.xlsx"


def _mcp() -> LocalMCPClient:
    xlsx = os.environ.get("TOOL_INVENTORY_XLSX", str(DEFAULT_XLSX))
    if not Path(xlsx).exists():
        pytest.skip(f"Inventory xlsx not at {xlsx}")
    return LocalMCPClient(inventory=ToolInventoryService(xlsx))


def _seed_through_step_9(
    local_storage, registry_service, workflow_state_service,
    *, with_smiles=True,
):
    intake = IntakeService(local_storage, registry_service, workflow_state_service)
    rec = intake.submit(
        raw_user_query="HER2 ADC vc-MMAE",
        user_provided_context={
            "target_or_antigen_text": "HER2",
            "candidate_text": "Trastuzumab",
            "payload_linker_text": "vc-MMAE",
        },
    )
    StructuredQueryService(
        local_storage, registry_service, workflow_state_service,
        SupervisorAgent(llm=MockLLMProvider()),
    ).parse(rec.run_id)
    if with_smiles:
        sq_path = local_storage.run_key(rec.run_id, "inputs/structured_query.json")
        sq = local_storage.read_json(sq_path)
        sq.setdefault("referenced_inputs", []).append(
            {"id_type": "smiles", "value": "CC(=O)NCC1=CN(C2=CC=CC=C2)C(=O)C1",
             "source": "raw_request_text"}
        )
        local_storage.write_json(sq_path, sq)
    InputReadinessService(local_storage, registry_service, workflow_state_service).check(rec.run_id)
    WorkflowSetupService(local_storage, registry_service, workflow_state_service).plan(rec.run_id)
    CandidateContextAgent(
        storage=local_storage, registry=registry_service,
        workflow_state=workflow_state_service, mcp_client=LocalMCPClient(),
    ).run(rec.run_id)
    DevelopabilityAgent(
        storage=local_storage, registry=registry_service,
        workflow_state=workflow_state_service, mcp_client=LocalMCPClient(),
    ).run(rec.run_id)
    sd = StructureAndDesignAgent(
        storage=local_storage, registry=registry_service,
        workflow_state=workflow_state_service, mcp_client=_mcp(),
    )
    sd.run_step_7(rec.run_id)
    sd.run_step_8(rec.run_id)
    sd.run_step_9(rec.run_id)
    return rec.run_id


def test_step10_aggregates_step5_through_9_into_handoff(
    local_storage, registry_service, workflow_state_service
):
    run_id = _seed_through_step_9(
        local_storage, registry_service, workflow_state_service, with_smiles=True
    )
    pkg = ScoringHandoffService(
        local_storage, registry_service, workflow_state_service
    ).prepare(run_id)
    assert pkg.handoff_status == "awaiting_external_scoring"
    assert pkg.external_module == "yufei_aee"
    assert pkg.candidate_summaries
    # Source artifact refs point at upstream artifact ids — never raw payloads.
    refs = pkg.candidate_summaries[0].source_artifact_refs
    assert refs.candidate_context_table_id
    assert refs.structure_prediction_and_interface_results_id
    assert refs.structure_variant_and_compound_screening_id


def test_step10_does_not_require_legacy_compound_hits(
    local_storage, registry_service, workflow_state_service
):
    """Step 9 compound hits are legacy-compatible fields, not required scoring inputs."""
    run_id = _seed_through_step_9(
        local_storage, registry_service, workflow_state_service, with_smiles=False
    )

    pkg = ScoringHandoffService(
        local_storage, registry_service, workflow_state_service
    ).prepare(run_id)
    assert pkg.handoff_status == "awaiting_external_scoring"
    assert not any("compound" in flag for flag in pkg.missing_inputs)


def test_step10_does_not_embed_raw_mcp_payloads(
    local_storage, registry_service, workflow_state_service
):
    run_id = _seed_through_step_9(
        local_storage, registry_service, workflow_state_service, with_smiles=True
    )
    pkg = ScoringHandoffService(
        local_storage, registry_service, workflow_state_service
    ).prepare(run_id)
    # Mock wrappers stamp `"mocked"` into payloads; that string must never
    # leak into the handoff package.
    blob = json.dumps(pkg.model_dump())
    assert "mocked" not in blob
    assert "hits" not in blob.lower() or "compound_hits" in blob.lower()
