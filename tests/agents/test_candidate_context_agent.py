from __future__ import annotations

import json

from app.agents.candidate_context_agent import CandidateContextAgent
from app.agents.supervisor_agent import SupervisorAgent
from app.llm.provider import MockLLMProvider
from app.mcp.client import LocalMCPClient
from app.services.intake_service import IntakeService
from app.services.input_readiness_service import InputReadinessService
from app.services.structured_query_service import StructuredQueryService
from app.services.workflow_setup_service import WorkflowSetupService


def _setup_run(local_storage, registry_service, workflow_state_service) -> str:
    intake = IntakeService(local_storage, registry_service, workflow_state_service)
    rec = intake.submit(
        raw_user_query="Design ADC against HER2 with vc-MMAE payload",
        user_provided_context={
            "target_or_antigen_text": "HER2",
            "candidate_text": "Trastuzumab analog",
            "payload_linker_text": "vc-MMAE",
        },
    )
    supervisor = SupervisorAgent(llm=MockLLMProvider())
    StructuredQueryService(
        local_storage, registry_service, workflow_state_service, supervisor
    ).parse(rec.run_id)
    InputReadinessService(local_storage, registry_service, workflow_state_service).check(rec.run_id)
    WorkflowSetupService(local_storage, registry_service, workflow_state_service).plan(rec.run_id)
    return rec.run_id


def _bindings(canned: dict[str, dict]) -> dict:
    def make(payload):
        def _fn(**_kwargs):
            return payload
        return _fn
    return {name: make(p) for name, p in canned.items()}


def test_candidate_context_agent_produces_table_with_target_material(
    local_storage, registry_service, workflow_state_service
):
    run_id = _setup_run(local_storage, registry_service, workflow_state_service)
    mcp = LocalMCPClient(
        bindings=_bindings(
            {
                "SAbDab_search_structures": {"hits": [{"pdb_id": "1n8z"}]},
                "ChEMBL_search_molecules": {"hits": [{"chembl_id": "CHEMBL1201585"}]},
                "ChEMBL_search_substructure": {"hits": [{"chembl_id": "CHEMBL_linker"}]},
            }
        )
    )
    agent = CandidateContextAgent(
        storage=local_storage,
        registry=registry_service,
        workflow_state=workflow_state_service,
        mcp_client=mcp,
    )
    table = agent.run(run_id)

    material_types = {m.material_type for c in table.candidate_records for m in c.materials}
    assert "target_antigen_name" in material_types

    # Read back the persisted table to inspect tool_call_records (table object
    # itself doesn't carry them in MVP shape).
    persisted = local_storage.read_json(
        local_storage.run_key(run_id, "candidate_context_table.json")
    )
    tcs = persisted["tool_call_records"]
    assert tcs and all(t["tool_output_ref"] for t in tcs)

    # tool_output_ref points at an existing key — and that key is NOT inside
    # the normalized candidate_records.
    for t in tcs:
        assert local_storage.exists(t["tool_output_ref"])
        raw = local_storage.read_json(t["tool_output_ref"])
        assert "output" in raw

    cand_blob = json.dumps(persisted["candidate_records"])
    assert "hits" not in cand_blob, "Raw payload leaked into candidate_records"


def test_candidate_context_agent_blocks_skipped_tools(
    local_storage, registry_service, workflow_state_service
):
    run_id = _setup_run(local_storage, registry_service, workflow_state_service)
    # Use real (unwired) ChEMBL bindings → dependency_unavailable
    agent = CandidateContextAgent(
        storage=local_storage,
        registry=registry_service,
        workflow_state=workflow_state_service,
        mcp_client=LocalMCPClient(),
    )
    table = agent.run(run_id)
    persisted = local_storage.read_json(
        local_storage.run_key(run_id, "candidate_context_table.json")
    )
    tcs = persisted["tool_call_records"]
    assert any(t["run_status"] == "dependency_unavailable" for t in tcs)
    # context_build_status reflects partial enrichment, not failed
    assert table.context_build_status in {"partial", "ok"}
