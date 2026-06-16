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
DEFAULT_XLSX = PROJECT_ROOT / "项目文件" / "ToolUniversity_inventory_v0.2.xlsx"


def _inventory():
    xlsx = os.environ.get("TOOL_INVENTORY_XLSX", str(DEFAULT_XLSX))
    if not Path(xlsx).exists():
        pytest.skip(f"Inventory xlsx not at {xlsx}")
    return ToolInventoryService(xlsx)


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
        mcp_client=LocalMCPClient(inventory=_inventory()),
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
        mcp_client=LocalMCPClient(inventory=_inventory()),
    ).run(run_id)
    blob = json.dumps([r.model_dump() for r in table.patent_records])
    # Mock OB wrapper stamps `"status": "mocked"` plus echoes the inputs
    # into a `records: []` field. Normalized rows must NOT carry that raw
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
        assert raw["output"]["source"] == "FDA_OrangeBook_get_patent_info"


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
