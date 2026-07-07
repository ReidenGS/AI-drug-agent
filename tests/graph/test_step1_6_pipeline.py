"""End-to-end: LangGraph Step 1→6 against LocalStorage + inventory-scoped MCP."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from app.graph.adc_graph import build_pipeline_graph
from app.mcp.client import LocalMCPClient
from app.services.tool_inventory_service import ToolInventoryService


PROJECT_ROOT = Path(__file__).resolve().parents[2].parent
DEFAULT_XLSX = PROJECT_ROOT / "\u9879\u76ee\u6587\u4ef6" / "ToolUniversity_inventory_v0.2.xlsx"


@pytest.fixture
def inventory():
    xlsx = os.environ.get("TOOL_INVENTORY_XLSX", str(DEFAULT_XLSX))
    if not Path(xlsx).exists():
        pytest.skip(f"Inventory xlsx not at {xlsx}")
    return ToolInventoryService(xlsx)


def _intake_request() -> dict:
    return {
        "intake_request": {
            "raw_user_query": (
                "HER2 ADC, target UniProt P04626, vc-MMAE payload, "
                "payload SMILES CCO, DAR 4"
            ),
            "user_provided_context": {
                "target_or_antigen_text": "HER2 (UniProt P04626)",
                "candidate_text": "Trastuzumab analog",
                "payload_linker_text": "vc-MMAE; payload SMILES CCO",
            },
        }
    }


# ── 1. happy path: Step 1→6 completes end-to-end ─────────────────────────────

def test_pipeline_graph_runs_steps_1_to_6(
    local_storage, registry_service, workflow_state_service, inventory
):
    graph = build_pipeline_graph(
        storage=local_storage,
        registry=registry_service,
        workflow_state=workflow_state_service,
        mcp_client=LocalMCPClient(inventory=inventory),
    )
    final = graph.invoke(_intake_request())

    run_id = final["run_id"]
    artifacts = final["artifacts"]
    for key in (
        "raw_request_record",
        "structured_query",
        "input_readiness_status",
        "run_step_plan",
        "candidate_context_table",
        "structured_liability_summary",
    ):
        assert artifacts.get(key), f"missing artifact id for {key}"

    # workflow_state has all six steps completed
    state = workflow_state_service.get(run_id)
    for s in ("step_01", "step_02", "step_03", "step_04", "step_05", "step_06"):
        assert state["steps"][s] == "completed", f"{s} not completed: {state['steps'][s]}"

    # registry carries the new artifact ids
    reg = registry_service.get(run_id)
    assert reg.active_artifacts.candidate_context_table_id
    assert reg.active_artifacts.structured_liability_summary_id


def test_pipeline_graph_runs_steps_1_to_6_for_protein_variant_query(
    local_storage, registry_service, workflow_state_service, inventory
):
    """Regression for the reported protein_variant crash: the Step 2 LLM
    labels V777L as entity_type="protein_variant"; the pipeline must run
    Step 1-6 to completion and structure the variant + UniProt so Step 9's
    variant tools can consume them (identifier:uniprot_id / identifier:variant).
    """
    from app.agents.step_09_input_projection import project_step9_inputs

    graph = build_pipeline_graph(
        storage=local_storage,
        registry=registry_service,
        workflow_state=workflow_state_service,
        mcp_client=LocalMCPClient(inventory=inventory),
    )
    final = graph.invoke(
        {
            "intake_request": {
                "raw_user_query": (
                    "Evaluate the HER2 variant V777L using UniProt P04626. "
                    "Use variant scoring only; do not generate protein sequences."
                ),
                "user_provided_context": {},
            }
        }
    )
    run_id = final["run_id"]

    state = workflow_state_service.get(run_id)
    for s in ("step_01", "step_02", "step_03", "step_04", "step_05", "step_06"):
        assert state["steps"][s] == "completed", f"{s} not completed: {state['steps'][s]}"

    # Step 2 structured both the UniProt accession and the variant.
    sq = local_storage.read_json(local_storage.run_key(run_id, "inputs/structured_query.json"))
    refs = {r["id_type"]: r for r in sq["referenced_inputs"] if isinstance(r, dict)}
    assert refs["uniprot_id"]["value"] == "P04626"
    assert refs["variant"]["value"] == "V777L"

    # Step 5 -> Step 9 projection surfaces both identifier fields.
    cct = local_storage.read_json(local_storage.run_key(run_id, "candidate_context_table.json"))
    projection = project_step9_inputs(
        candidate_context_table=cct,
        prepared_structure_input_package=None,
        structure_prediction_and_interface_results=None,
    )
    field_refs = {f.field_ref for f in projection["input_fields"]}
    assert "identifier:uniprot_id:P04626" in field_refs
    assert "identifier:variant:V777L" in field_refs


# ── 2. inventory scope guard ─────────────────────────────────────────────────

def test_pipeline_graph_refuses_non_inventory_scoped_client(
    local_storage, registry_service, workflow_state_service
):
    """Passing a bare LocalMCPClient (no inventory) must be rejected at build
    time so Steps 5/6 cannot see a wider tool surface than v0.2 allows."""
    with pytest.raises(ValueError, match="inventory-scoped"):
        build_pipeline_graph(
            storage=local_storage,
            registry=registry_service,
            workflow_state=workflow_state_service,
            mcp_client=LocalMCPClient(),  # no inventory
        )


# ── 3. raw tool payloads stay outside normalized records ─────────────────────

def test_pipeline_graph_does_not_leak_raw_payloads_into_artifacts(
    local_storage, registry_service, workflow_state_service, inventory
):
    """Run Step 1-6 with mock bindings for Step 5 + Step 6 tools that return a
    distinctive `"hits"` marker; confirm `"hits"` only appears in the raw
    tool_output_ref files, never in the normalized step artifacts."""
    bindings = {
        # Step 5 tools
        "SAbDab_search_structures": lambda **kw: {"hits_step5_sabdab": [kw.get("query")]},
        "ChEMBL_search_molecules": lambda **kw: {"hits_step5_chembl_mol": [kw.get("query")]},
        "ChEMBL_search_substructure": lambda **kw: {"hits_step5_chembl_sub": [kw.get("query")]},
        # Step 6 tools
        "DrugProps_pains_filter": lambda **kw: {"hits_step6_drugprops": [kw.get("query")]},
        "PROSITE_scan_sequence": lambda **kw: {"hits_step6_prosite": [kw.get("query")]},
        "EBIProteins_get_features": lambda **kw: {"hits_step6_ebi": [kw.get("query")]},
        "ProteinsPlus_profile_structure_quality": lambda **kw: {"hits_step6_pp": [kw.get("query")]},
        "ChEMBL_search_activities": lambda **kw: {"hits_step6_chembl_act": [kw.get("query")]},
    }
    mcp = LocalMCPClient(inventory=inventory, bindings=bindings)
    graph = build_pipeline_graph(
        storage=local_storage,
        registry=registry_service,
        workflow_state=workflow_state_service,
        mcp_client=mcp,
    )
    final = graph.invoke(_intake_request())
    run_id = final["run_id"]

    # Step 5 normalized artifact
    step5 = local_storage.read_json(
        local_storage.run_key(run_id, "candidate_context_table.json")
    )
    cand_blob = json.dumps(step5["candidate_records"])
    assert "hits_step5_" not in cand_blob, "Step 5 raw payload leaked into candidate_records"

    # Step 5 raw payloads live where tool_output_ref points
    for tc in step5["tool_call_records"]:
        if tc.get("tool_output_ref"):
            raw = local_storage.read_json(tc["tool_output_ref"])
            assert any(k.startswith("hits_step5_") for k in raw.get("output", {}))

    # Step 6 normalized artifact
    step6 = local_storage.read_json(
        local_storage.run_key(run_id, "structured_liability_summary.json")
    )
    for cand in step6["candidate_liability_results"]:
        for lane in cand["lane_results"]:
            assert "hits_step6_" not in json.dumps(lane["liability_flags"])
            assert "hits_step6_" not in (lane.get("lane_summary") or "")
            for tc in lane["tool_call_records"]:
                if tc.get("run_status") == "success":
                    assert tc["tool_output_ref"]
                    raw = local_storage.read_json(tc["tool_output_ref"])
                    assert any(k.startswith("hits_step6_") for k in raw.get("output", {}))


# ── 4. graph tolerates dependency_unavailable in Step 6 ─────────────────────

def test_pipeline_graph_tolerates_dep_unavailable_in_step6(
    local_storage, registry_service, workflow_state_service, inventory
):
    """Step 5 succeeds (canned), Step 6 wrappers remain unwired → every Step
    6 tool returns `dependency_unavailable`. The graph must still finish; the
    step artifact ends in `partial` (or `completed_with_missing_lanes`)
    instead of crashing the graph.

    Note: we merge custom Step 5 bindings on top of the default `_all_bindings`
    table so Step 6 wrappers stay at their default (NotImplementedError →
    dependency_unavailable). Replacing the dict outright would make Step 6
    tools look out-of-scope instead of unwired.
    """
    from app.mcp.tools._registry import _all_bindings

    def _force_ni(*_a, **_kw):
        raise NotImplementedError

    bindings = dict(_all_bindings())
    bindings.update(
        {
            "SAbDab_search_structures": lambda **kw: {"hits": []},
            "ChEMBL_search_molecules": lambda **kw: {"hits": []},
            "ChEMBL_search_substructure": lambda **kw: {"hits": []},
        }
    )
    # Force every Step 6 wrapper back to `_ni` so the agent's lane plan
    # exercises the `dependency_unavailable` path even after the recent
    # batch of Step 6 ToolUniverseAdapter wire-ups.
    for name in (
        "DrugProps_pains_filter",
        "DrugProps_lipinski_filter",
        "DrugProps_calculate_qed",
        "BindingDB_get_targets_by_compound",
        "PROSITE_scan_sequence",
        "EBIProteins_get_epitopes",
        "EBIProteins_get_antigen",
        "EBIProteins_get_features",
        "ProteinsPlus_profile_structure_quality",
    ):
        bindings[name] = _force_ni
    mcp = LocalMCPClient(inventory=inventory, bindings=bindings)
    graph = build_pipeline_graph(
        storage=local_storage,
        registry=registry_service,
        workflow_state=workflow_state_service,
        mcp_client=mcp,
    )
    final = graph.invoke(_intake_request())
    run_id = final["run_id"]

    state = workflow_state_service.get(run_id)
    assert state["steps"]["step_06"] == "completed"

    step6 = local_storage.read_json(
        local_storage.run_key(run_id, "structured_liability_summary.json")
    )
    assert step6["prefilter_status"] in {"partial", "completed_with_missing_lanes"}
    # at least one dependency_unavailable recorded
    dep_unavail = [
        tc
        for cand in step6["candidate_liability_results"]
        for lane in cand["lane_results"]
        for tc in lane["tool_call_records"]
        if tc["run_status"] == "dependency_unavailable"
    ]
    assert dep_unavail
