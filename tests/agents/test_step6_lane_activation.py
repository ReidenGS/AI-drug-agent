"""Step 6 lane-activation guards.

The professor's requirement: Step 6 must be lane-based, with each lane
activated only when its specific input material is available. Antibody
sequence, target/antigen metadata, PDB structure, and payload/linker
SMILES MUST NOT be simultaneously required. Unavailable lanes are
marked skipped / missing input rather than causing the step to fail.

These tests exercise that directly by writing a synthetic
`candidate_context_table.json` that contains exactly ONE material type
per case, then asserting which lanes activate and which stay skipped.

Also pins:

- The LLM cannot select Step 13 / Step 14 tools from a Step 6 catalog
  (out-of-scope selections are dropped before any MCP call).
- Stage 1 payload contains only the compact catalog; Stage 2 schema
  never exposes `_live` to the LLM.
- Partial inputs do not flip the summary into `failed`.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from app.agents.developability_agent import DevelopabilityAgent
from app.agents.step_06_capability_registry import STEP_06_CAPABILITY_REGISTRY
from app.mcp.client import LocalMCPClient
from app.services.intake_service import IntakeService
from app.utils.ids import new_artifact_id
from app.utils.time import now_iso


# ── helpers ─────────────────────────────────────────────────────────────────

def _bindings(canned: dict[str, dict]) -> dict:
    def make(payload):
        def _fn(**_kw):
            return payload
        return _fn
    return {name: make(p) for name, p in canned.items()}


_DEFAULT_OK_BINDINGS = {
    # small-molecule lane fallback + bioactivity
    "DrugProps_pains_filter": {"status": "mocked", "alerts": []},
    "DrugProps_lipinski_filter": {"status": "mocked", "violations": []},
    "DrugProps_calculate_qed": {"status": "mocked", "qed": 0.7},
    "SwissADME_calculate_adme": {"status": "mocked", "warnings": []},
    "SwissADME_check_druglikeness": {"status": "mocked", "warnings": []},
    "ADMETAI_predict_toxicity": {"status": "mocked", "predictions": {}},
    "ADMETAI_predict_physicochemical_properties": {"status": "mocked", "warnings": []},
    "ChEMBL_search_activities": {"status": "mocked", "results": []},
    "ChEMBL_search_compound_structural_alerts": {"status": "mocked", "structural_alerts": []},
    "ChEMBL_get_molecule_targets": {"status": "mocked", "targets": []},
    "BindingDB_get_targets_by_compound": {"status": "mocked", "targets": []},
    # sequence lane
    "PROSITE_scan_sequence": {"status": "mocked", "motifs": []},
    "IEDB_predict_mhci_binding": {"status": "mocked", "predictions": []},
    # antigen lane
    "EBIProteins_get_features": {"status": "mocked", "features": []},
    "EBIProteins_get_epitopes": {"status": "mocked", "epitopes": []},
    "EBIProteins_get_antigen": {"status": "mocked", "antigens": []},
    "GlyGen_get_glycoprotein": {"status": "mocked", "glycosylation_sites": []},
    "iPTMnet_get_ptm_sites": {"status": "mocked", "ptm_sites": []},
    "PDBe_KB_get_interface_residues": {"status": "mocked", "interface_residues": []},
    # structure lane
    "ProteinsPlus_profile_structure_quality": {"status": "mocked", "quality": "ok"},
    "PDBePISA_get_interfaces": {"status": "mocked", "interfaces": []},
    "PDBePISA_get_monomer_analysis": {"status": "mocked", "monomers": []},
}

_FASTA_PATH = "adc_pilot/runs/run_x/inputs/heavy_chain.fasta"
_FASTA_CONTENT = ">chainA\nEVQLVESGGGLVQPGGSLRLSCAASGFNI\n"
_S1_PDB_PATH = "adc_pilot/runs/run_x/inputs/S1.pdb"


def _seed_synthetic_cct(
    local_storage,
    registry_service,
    workflow_state_service,
    *,
    materials: list[dict],
    identifiers: list[dict] | None = None,
    candidate_type: str = "compound_component",
) -> str:
    """Submit intake, then overwrite Step 5 cct with a minimal synthetic
    record carrying only the requested material(s)."""
    intake = IntakeService(local_storage, registry_service, workflow_state_service)
    rec = intake.submit(
        raw_user_query="step6 lane activation fixture",
        user_provided_context={"target_or_antigen_text": "synthetic"},
    )
    run_id = rec.run_id
    artifact_id = new_artifact_id("candidate_context_table")
    cct = {
        "artifact_id": artifact_id,
        "run_id": run_id,
        "step_id": "step_05_candidate_context",
        "created_at": now_iso(),
        "context_build_status": "ok",
        "candidate_records": [
            {
                "candidate_id": "cand_synthetic_1",
                "candidate_label": "synthetic",
                "candidate_type": candidate_type,
                "source_records": [],
                "identifiers": identifiers or [],
                "materials": materials,
                "adc_links": {
                    "target_material_ids": [],
                    "antibody_material_ids": [],
                    "payload_material_ids": [],
                    "linker_material_ids": [],
                    "dar_material_ids": [],
                },
                "candidate_status": "partially_ready_for_step6",
                "candidate_notes": None,
                "candidate_role": "user_provided_candidate",
                "is_generated_candidate": False,
                "context_status": "partial",
                "data_gaps": [],
                "missing_material_roles": [],
                "context_notes": [],
            }
        ],
        "missing_context_flags": [],
        "tool_call_records": [],
        "downstream_query_hints": [],
    }
    local_storage.write_json(
        local_storage.run_key(run_id, "candidate_context_table.json"), cct
    )
    registry_service.update_active(run_id, candidate_context_table_id=artifact_id)
    return run_id


def _material(
    mat_type: str,
    value: str,
    value_format: str | None = None,
) -> dict:
    return {
        "material_id": f"mat_{mat_type}",
        "material_type": mat_type,
        "value": value,
        "value_format": value_format,
        "extraction_status": "extracted",
        "validation_status": "unknown",
        "role": None,
        "role_status": "unknown",
    }


def _lane_results(persisted: dict) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for cand in persisted["candidate_liability_results"]:
        for lane in cand["lane_results"]:
            out[lane["lane_type"]] = lane
    return out


# ── ChEMBL-id-gated bioactivity lane ─────────────────────────────────────────


def test_step6_bioactivity_lane_runs_with_chembl_id_identifier_only(
    local_storage, registry_service, workflow_state_service
):
    """A candidate with a typed ``chembl_id`` identifier (but no SMILES)
    must activate ``compound_bioactivity_prior_context`` and dispatch
    ``ChEMBL_search_activities``."""
    canned = {"ChEMBL_search_activities": {"status": "mocked", "activities": []}}
    run_id = _seed_synthetic_cct(
        local_storage, registry_service, workflow_state_service,
        materials=[],
        identifiers=[{"id_type": "chembl_id", "id_value": "CHEMBL2107839",
                      "source_ids": ["tc1"], "confidence": 0.8}],
    )
    DevelopabilityAgent(
        storage=local_storage, registry=registry_service,
        workflow_state=workflow_state_service,
        mcp_client=LocalMCPClient(bindings=_bindings(canned)),
    ).run(run_id)
    persisted = local_storage.read_json(
        local_storage.run_key(run_id, "structured_liability_summary.json")
    )
    lanes = _lane_results(persisted)
    bio = lanes["compound_bioactivity_prior_context"]
    assert bio["run_status"] in {"ok", "partial"}, bio
    tool_names = {tc["tool_name"] for tc in bio["tool_call_records"]}
    assert "ChEMBL_search_activities" in tool_names


def test_step6_smiles_runs_bindingdb_prior_without_chembl_id(
    local_storage, registry_service, workflow_state_service
):
    """SMILES activates BindingDB prior context, not ChEMBL-ID tools."""
    run_id = _seed_synthetic_cct(
        local_storage, registry_service, workflow_state_service,
        materials=[_material("payload_smiles", "CCO")],
    )
    DevelopabilityAgent(
        storage=local_storage, registry=registry_service,
        workflow_state=workflow_state_service,
        mcp_client=LocalMCPClient(bindings=_bindings(_DEFAULT_OK_BINDINGS)),
    ).run(run_id)
    persisted = local_storage.read_json(
        local_storage.run_key(run_id, "structured_liability_summary.json")
    )
    lanes = _lane_results(persisted)
    bio = lanes["compound_bioactivity_prior_context"]
    assert bio["run_status"] in {"ok", "partial"}
    assert bio["input_status"] == "sufficient"
    names = {tc["tool_name"] for tc in bio["tool_call_records"]}
    assert "BindingDB_get_targets_by_compound" in names
    successful_names = {
        tc["tool_name"] for tc in bio["tool_call_records"]
        if tc["run_status"] == "success"
    }
    assert "ChEMBL_search_activities" not in successful_names


# ── 1. payload SMILES only → compound liability + bioactivity lanes only ────

def test_step6_payload_smiles_only_runs_compound_lane_and_marks_others_missing(
    local_storage, registry_service, workflow_state_service
):
    run_id = _seed_synthetic_cct(
        local_storage, registry_service, workflow_state_service,
        materials=[_material("payload_smiles", "CCO")],
    )
    DevelopabilityAgent(
        storage=local_storage, registry=registry_service,
        workflow_state=workflow_state_service,
        mcp_client=LocalMCPClient(bindings=_bindings(_DEFAULT_OK_BINDINGS)),
    ).run(run_id)
    persisted = local_storage.read_json(
        local_storage.run_key(run_id, "structured_liability_summary.json")
    )
    lanes = _lane_results(persisted)

    # payload SMILES activates compound liability lane.
    assert lanes["payload_linker_compound_liability"]["run_status"] in {"ok", "partial"}
    assert lanes["payload_linker_compound_liability"]["input_status"] == "sufficient"

    # Sequence / antigen-feature / structure lanes all marked missing.
    for lane_type in (
        "antibody_protein_sequence_liability",
        "antigen_protein_feature_context",
        "structure_interface_quality",
    ):
        lane = lanes[lane_type]
        assert lane["run_status"] == "skipped"
        assert lane["input_status"] == "missing"
        assert lane["tool_call_records"] == []
        # Either family — materials (sequence/structure) or identifiers
        # (uniprot_id/chembl_id) — is acceptable; the wording is family-aware.
        summary = lane["lane_summary"] or ""
        assert "no candidate" in summary and "family" in summary

    # Summary status stays in the "completed-ish" band, not failed.
    assert persisted["prefilter_status"] in {"completed", "partial"}
    audit = persisted["selection_audit"]
    assert set(audit["step_06_stage1_catalog_tool_names"]) == set(
        audit["step_06_stage1_disclosed_tool_names"]
    )
    assert set(audit["step_06_stage1_catalog_tool_names"]) < set(
        audit["step_06_stage1_scope_tool_names"]
    )
    assert "PROSITE_scan_sequence" in audit["step_06_stage1_scope_tool_names"]
    assert "PROSITE_scan_sequence" not in audit["step_06_stage1_catalog_tool_names"]


def test_step6_payload_name_does_not_run_smiles_liability_tools(
    local_storage, registry_service, workflow_state_service
):
    """A payload/linker name is not a SMILES string. Step 6 must not pass
    names such as vc-MMAE into DrugProps/ADMET/SwissADME `smiles` args."""
    run_id = _seed_synthetic_cct(
        local_storage, registry_service, workflow_state_service,
        materials=[_material("payload_name", "vc-MMAE")],
    )
    DevelopabilityAgent(
        storage=local_storage, registry=registry_service,
        workflow_state=workflow_state_service,
        mcp_client=LocalMCPClient(bindings=_bindings(_DEFAULT_OK_BINDINGS)),
    ).run(run_id)
    persisted = local_storage.read_json(
        local_storage.run_key(run_id, "structured_liability_summary.json")
    )
    lanes = _lane_results(persisted)

    assert lanes["payload_linker_compound_liability"]["run_status"] == "skipped"
    assert lanes["payload_linker_compound_liability"]["input_status"] == "insufficient"
    successful = {
        tc["tool_name"]
        for cand in persisted["candidate_liability_results"]
        for lane in cand["lane_results"]
        for tc in lane["tool_call_records"]
        if tc["run_status"] == "success"
    }
    assert not (successful & {
        "DrugProps_pains_filter",
        "DrugProps_lipinski_filter",
        "DrugProps_calculate_qed",
        "SwissADME_calculate_adme",
        "SwissADME_check_druglikeness",
        "ADMETAI_predict_toxicity",
        "ADMETAI_predict_physicochemical_properties",
    })
    assert "ChEMBL_search_activities" not in successful
    audit = persisted["selection_audit"]
    assert "ambiguous_modality_fail_open" in (
        audit["step_06_stage1_disclosure_summary"]["cand_synthetic_1"]["disclosure_tags"]
    )


def test_step6_antibody_name_does_not_run_sequence_tools(
    local_storage, registry_service, workflow_state_service
):
    """An antibody display name is not an amino-acid sequence."""
    run_id = _seed_synthetic_cct(
        local_storage, registry_service, workflow_state_service,
        materials=[_material("antibody_name", "trastuzumab")],
        candidate_type="antibody",
    )
    DevelopabilityAgent(
        storage=local_storage, registry=registry_service,
        workflow_state=workflow_state_service,
        mcp_client=LocalMCPClient(bindings=_bindings(_DEFAULT_OK_BINDINGS)),
    ).run(run_id)
    persisted = local_storage.read_json(
        local_storage.run_key(run_id, "structured_liability_summary.json")
    )
    lanes = _lane_results(persisted)

    assert lanes["antibody_protein_sequence_liability"]["run_status"] == "skipped"
    assert lanes["antibody_protein_sequence_liability"]["input_status"] == "insufficient"
    successful = {
        tc["tool_name"]
        for cand in persisted["candidate_liability_results"]
        for lane in cand["lane_results"]
        for tc in lane["tool_call_records"]
        if tc["run_status"] == "success"
    }
    assert "PROSITE_scan_sequence" not in successful


def test_step6_target_name_without_accession_does_not_run_accession_tools(
    local_storage, registry_service, workflow_state_service
):
    """A target name alone is not a UniProt accession."""
    run_id = _seed_synthetic_cct(
        local_storage, registry_service, workflow_state_service,
        materials=[_material("target_antigen_name", "HER2")],
        identifiers=[],
        candidate_type="target_antigen",
    )
    DevelopabilityAgent(
        storage=local_storage, registry=registry_service,
        workflow_state=workflow_state_service,
        mcp_client=LocalMCPClient(bindings=_bindings(_DEFAULT_OK_BINDINGS)),
    ).run(run_id)
    persisted = local_storage.read_json(
        local_storage.run_key(run_id, "structured_liability_summary.json")
    )
    lanes = _lane_results(persisted)

    assert lanes["antigen_protein_feature_context"]["run_status"] == "skipped"
    assert lanes["antigen_protein_feature_context"]["input_status"] == "insufficient"
    successful = {
        tc["tool_name"]
        for cand in persisted["candidate_liability_results"]
        for lane in cand["lane_results"]
        for tc in lane["tool_call_records"]
        if tc["run_status"] == "success"
    }
    assert "EBIProteins_get_features" not in successful
    assert "EBIProteins_get_epitopes" not in successful


# ── 2. antibody sequence only → sequence lane only ──────────────────────────

def test_step6_sequence_only_runs_sequence_lane(
    local_storage, registry_service, workflow_state_service
):
    run_id = _seed_synthetic_cct(
        local_storage, registry_service, workflow_state_service,
        materials=[_material("antibody_heavy_chain_sequence", "EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGK")],
        candidate_type="antibody",
    )
    DevelopabilityAgent(
        storage=local_storage, registry=registry_service,
        workflow_state=workflow_state_service,
        mcp_client=LocalMCPClient(bindings=_bindings(_DEFAULT_OK_BINDINGS)),
    ).run(run_id)
    persisted = local_storage.read_json(
        local_storage.run_key(run_id, "structured_liability_summary.json")
    )
    lanes = _lane_results(persisted)

    assert lanes["antibody_protein_sequence_liability"]["run_status"] in {"ok", "partial"}
    assert lanes["antibody_protein_sequence_liability"]["input_status"] == "sufficient"
    for lane_type in (
        "payload_linker_compound_liability",
        "antigen_protein_feature_context",
        "structure_interface_quality",
        "compound_bioactivity_prior_context",
    ):
        assert lanes[lane_type]["run_status"] == "skipped"
        assert lanes[lane_type]["input_status"] == "missing"
        assert lanes[lane_type]["tool_call_records"] == []


# ── 3. target_antigen_name only → antigen feature lane only ─────────────────

def test_step6_target_antigen_only_runs_antigen_feature_lane(
    local_storage, registry_service, workflow_state_service
):
    run_id = _seed_synthetic_cct(
        local_storage, registry_service, workflow_state_service,
        materials=[_material("target_antigen_name", "HER2")],
        identifiers=[{"id_type": "uniprot_id", "id_value": "P04626", "source_ids": [], "confidence": 0.9}],
        candidate_type="target_antigen",
    )
    DevelopabilityAgent(
        storage=local_storage, registry=registry_service,
        workflow_state=workflow_state_service,
        mcp_client=LocalMCPClient(bindings=_bindings(_DEFAULT_OK_BINDINGS)),
    ).run(run_id)
    persisted = local_storage.read_json(
        local_storage.run_key(run_id, "structured_liability_summary.json")
    )
    lanes = _lane_results(persisted)

    assert lanes["antigen_protein_feature_context"]["run_status"] in {"ok", "partial"}
    assert lanes["antigen_protein_feature_context"]["input_status"] == "sufficient"
    for lane_type in (
        "payload_linker_compound_liability",
        "antibody_protein_sequence_liability",
        "structure_interface_quality",
        "compound_bioactivity_prior_context",
    ):
        assert lanes[lane_type]["run_status"] == "skipped"
        assert lanes[lane_type]["input_status"] == "missing"


# ── 4. structure-only → structure lane only ─────────────────────────────────

def test_step6_pdb_only_runs_structure_lane(
    local_storage, registry_service, workflow_state_service
):
    run_id = _seed_synthetic_cct(
        local_storage, registry_service, workflow_state_service,
        materials=[_material("structure_ref", "pdb:1N8Z")],
        identifiers=[{"id_type": "pdb_id", "id_value": "1N8Z", "source_ids": [], "confidence": 0.9}],
        candidate_type="adc_construct",
    )
    DevelopabilityAgent(
        storage=local_storage, registry=registry_service,
        workflow_state=workflow_state_service,
        mcp_client=LocalMCPClient(bindings=_bindings(_DEFAULT_OK_BINDINGS)),
    ).run(run_id)
    persisted = local_storage.read_json(
        local_storage.run_key(run_id, "structured_liability_summary.json")
    )
    lanes = _lane_results(persisted)

    assert lanes["structure_interface_quality"]["run_status"] in {"ok", "partial"}
    assert lanes["structure_interface_quality"]["input_status"] == "sufficient"
    for lane_type in (
        "payload_linker_compound_liability",
        "antibody_protein_sequence_liability",
        "antigen_protein_feature_context",
        "compound_bioactivity_prior_context",
    ):
        assert lanes[lane_type]["run_status"] == "skipped"
        assert lanes[lane_type]["input_status"] == "missing"


def test_step6_uploaded_fasta_is_resolved_at_runtime_and_injected(
    local_storage, registry_service, workflow_state_service
):
    local_storage.write_bytes(_FASTA_PATH, _FASTA_CONTENT.encode("utf-8"))

    class _FastaLLM:
        name = "fasta_stage2"
        model = "test"
        def generate(self, *_a, **_kw):
            raise NotImplementedError
        def generate_json(self, prompt: str, *, schema: dict, system: str | None = None):
            task = (schema or {}).get("task")
            if task == "step6_schema_mapping_stage_1":
                return {
                    "selections": [
                        {"tool_name": "PROSITE_scan_sequence", "selection_reason": "sequence lane"}
                    ]
                }
            if task == "step6_schema_mapping_stage_2":
                field_ref = next(
                    f["field_ref"] for f in schema.get("candidate_available_fields", [])
                    if f.get("value_kind") == "uploaded_fasta_ref"
                )
                return {
                    "tools": [{
                        "tool_name": "PROSITE_scan_sequence",
                        "can_invoke": True,
                        "argument_mapping": {"sequence": field_ref},
                        "missing_required_fields": [],
                        "argument_mapping_reason": "resolve uploaded FASTA",
                    }]
                }
            return {}

    observed: dict[str, Any] = {}
    def _prosite(**kw):
        observed["prosite_kwargs"] = dict(kw)
        return {"status": "mocked", "motifs": []}

    run_id = _seed_synthetic_cct(
        local_storage, registry_service, workflow_state_service,
        materials=[_material("antibody_heavy_chain_sequence", _FASTA_PATH, value_format="fasta")],
    )
    DevelopabilityAgent(
        storage=local_storage, registry=registry_service,
        workflow_state=workflow_state_service,
        mcp_client=LocalMCPClient(bindings={"PROSITE_scan_sequence": _prosite}),
        llm=_FastaLLM(),
    ).run(run_id)

    persisted = local_storage.read_json(local_storage.run_key(run_id, "structured_liability_summary.json"))
    lane = _lane_results(persisted)["antibody_protein_sequence_liability"]
    calls = [
        tc for tc in lane["tool_call_records"]
        if tc["tool_name"] == "PROSITE_scan_sequence"
    ]
    assert calls, "PROSITE should be selected and attempted"
    assert calls[0]["run_status"] == "success"
    assert observed["prosite_kwargs"]["sequence"] == _FASTA_CONTENT.splitlines()[1]
    assert _FASTA_PATH not in str(observed["prosite_kwargs"]["sequence"])
    assert "EVQLVESGGGLVQPGGSLRLSCAASGFNI" not in str(calls[0]["tool_input_summary"])
    assert _FASTA_PATH not in str(calls[0]["tool_input_summary"])
    summary = persisted["selection_audit"]
    assert "PROSITE_scan_sequence" in summary["step_06_stage1_selected_tools"]
    assert "PROSITE_scan_sequence" in summary["step_06_stage2_mapped_tools"]
    assert "PROSITE_scan_sequence" in summary["step_06_executed_tools"]
    assert "PROSITE_scan_sequence" in summary["step_06_runtime_resolved_tools"]
    assert "PROSITE_scan_sequence" not in summary["step_06_resolver_failed_tools"]


def test_step6_uploaded_structure_ref_cannot_satisfy_pdb_id_only_structure_tools(
    local_storage, registry_service, workflow_state_service, monkeypatch
):
    captured: dict[str, dict[str, str]] = {}

    class _StructureLLM:
        name = "structure_schema"
        model = "test"
        def generate(self, *_a, **_kw):
            raise NotImplementedError
        def generate_json(self, prompt: str, *, schema: dict, system: str | None = None):
            task = (schema or {}).get("task")
            if task == "step6_schema_mapping_stage_1":
                return {"selections": [
                    {"tool_name": "PDBePISA_get_interfaces", "selection_reason": "structure"},
                    {"tool_name": "ProteinsPlus_profile_structure_quality", "selection_reason": "structure"},
                ]}
            if task == "step6_schema_mapping_stage_2":
                return {
                    "tools": [
                        {
                            "tool_name": "PDBePISA_get_interfaces",
                            "can_invoke": True,
                            "argument_mapping": {"pdb_id": next(
                                f["field_ref"] for f in schema.get("candidate_available_fields", [])
                                if f.get("value_kind") == "structure_ref"
                            )},
                            "missing_required_fields": [],
                            "argument_mapping_reason": "should fail for path-only ref",
                        },
                        {
                            "tool_name": "ProteinsPlus_profile_structure_quality",
                            "can_invoke": True,
                            "argument_mapping": {"pdb_id": next(
                                f["field_ref"] for f in schema.get("candidate_available_fields", [])
                                if f.get("value_kind") == "structure_ref"
                            )},
                            "missing_required_fields": [],
                            "argument_mapping_reason": "should fail for path-only ref",
                        },
                    ]
                }
            return {}

    def _proteinsplus(**kw):
        captured["proteinsplus"] = dict(kw)
        return {"status": "mocked", "quality": "ok"}

    from app.agents import step_06_schema_mapping_selector as selector
    def _signature_schema_for(tool_name: str) -> dict:
        return {
            "type": "object",
            "properties": {"pdb_id": {"type": "string"}},
            "required": ["pdb_id"],
        }

    monkeypatch.setattr(selector, "signature_schema_for", _signature_schema_for)
    run_id = _seed_synthetic_cct(
        local_storage, registry_service, workflow_state_service,
        materials=[_material("structure_file", _S1_PDB_PATH, value_format="pdb")],
        candidate_type="adc_construct",
    )
    DevelopabilityAgent(
        storage=local_storage, registry=registry_service,
        workflow_state=workflow_state_service,
        mcp_client=LocalMCPClient(bindings={
            "PDBePISA_get_interfaces": lambda **_kw: {"status": "mocked", "interfaces": []},
            "ProteinsPlus_profile_structure_quality": _proteinsplus,
        }),
        llm=_StructureLLM(),
    ).run(run_id)

    persisted = local_storage.read_json(local_storage.run_key(run_id, "structured_liability_summary.json"))
    lane = _lane_results(persisted)["structure_interface_quality"]
    assert lane["run_status"] == "skipped"
    call_names = {tc["tool_name"] for tc in lane["tool_call_records"]}
    assert call_names == {"PDBePISA_get_interfaces", "ProteinsPlus_profile_structure_quality"}
    tool_status = {tc["tool_name"]: tc["run_status"] for tc in lane["tool_call_records"]}
    assert tool_status["PDBePISA_get_interfaces"] == "skipped"
    assert tool_status["ProteinsPlus_profile_structure_quality"] == "skipped"
    assert "proteinsplus" not in captured
    summary = persisted["selection_audit"]
    assert "PDBePISA_get_interfaces" in summary["step_06_stage2_uninvokable_tools"]
    assert "ProteinsPlus_profile_structure_quality" in summary["step_06_stage2_uninvokable_tools"]
    assert any(
        entry.get("tool_name") == "PDBePISA_get_interfaces"
        and entry.get("candidate_id") == "cand_synthetic_1"
        and entry.get("lane_type") == "structure_interface_quality"
        and "pdb_id" in entry.get("missing_required_fields", [])
        for entry in summary.get("step_06_stage2_uninvokable_tool_details", [])
        if isinstance(entry, dict)
    )
    assert any(
        entry.get("tool_name") == "ProteinsPlus_profile_structure_quality"
        and entry.get("candidate_id") == "cand_synthetic_1"
        and entry.get("lane_type") == "structure_interface_quality"
        and "pdb_id" in entry.get("missing_required_fields", [])
        for entry in summary.get("step_06_stage2_uninvokable_tool_details", [])
        if isinstance(entry, dict)
    )
    assert "PDBePISA_get_interfaces" in summary["step_06_stage1_selected_tools"]
    assert "ProteinsPlus_profile_structure_quality" not in summary["step_06_stage2_mapped_tools"]
    assert "ProteinsPlus_profile_structure_quality" not in summary["step_06_runtime_resolved_tools"]
    assert _FASTA_PATH not in str(persisted["selection_audit"])


# ── 5. partial inputs do NOT flip the summary into failed ───────────────────

def test_step6_partial_inputs_do_not_fail_prefilter(
    local_storage, registry_service, workflow_state_service
):
    # Two candidates: one structure-only, one antibody-sequence-only. No
    # candidate carries a SMILES; that's a partial-input scenario per
    # professor wording and Step 6 must NOT fail.
    run_id = _seed_synthetic_cct(
        local_storage, registry_service, workflow_state_service,
        materials=[_material("structure_ref", "pdb:1N8Z")],
        candidate_type="adc_construct",
    )
    DevelopabilityAgent(
        storage=local_storage, registry=registry_service,
        workflow_state=workflow_state_service,
        mcp_client=LocalMCPClient(bindings=_bindings(_DEFAULT_OK_BINDINGS)),
    ).run(run_id)
    persisted = local_storage.read_json(
        local_storage.run_key(run_id, "structured_liability_summary.json")
    )
    assert persisted["prefilter_status"] in {
        "completed", "partial", "completed_with_missing_lanes"
    }
    assert persisted["prefilter_status"] != "failed"


# ── 6. fake LLM picks an out-of-scope (Step 13/14) tool — never executed ────

class _OutOfScopeLLM:
    name = "out_of_scope_test"
    model = "test"

    def generate(self, prompt: str, *, system: str | None = None, **kw: Any) -> str:
        raise NotImplementedError

    def generate_json(self, prompt: str, *, schema: dict, system: str | None = None) -> dict:
        task = (schema or {}).get("task")
        if task in {"tool_selection_stage_1", "tool_selection_stage_1_multi_lane", "step6_schema_mapping_stage_1"}:
            # Try to pick clearly out-of-scope Step 13 / Step 14 tool names
            # plus a hallucinated one. None is in the Step 6 catalog, so the
            # per-candidate selector must drop all and fall back deterministically.
            return {
                "selections": [
                    {"tool_name": "MultiAgentLiteratureSearch", "selection_reason": "step13"},
                    {"tool_name": "FDA_OrangeBook_get_patent_info", "selection_reason": "step14"},
                    {"tool_name": "TotallyHallucinatedTool", "selection_reason": "halluc"},
                ]
            }
        if task in {"tool_selection_stage_2", "tool_selection_stage_2_multi_tool", "step6_schema_mapping_stage_2"}:
            return {"arguments": {}, "argument_construction_reason": "", "tools": []}
        return {}


def test_step6_fake_llm_out_of_scope_tool_is_not_executed(
    local_storage, registry_service, workflow_state_service
):
    """If the LLM proposes Step 13/14 / hallucinated tools, Step 6 must drop
    them and fall back to the deterministic in-lane tool. The forbidden
    tool names must never appear in tool_call_records and must never be
    invoked through the MCP client."""

    # Bind ONLY the in-lane fallback so any out-of-scope call would raise.
    bindings = _bindings({
        "DrugProps_pains_filter": {"status": "mocked", "alerts": []},
        "ChEMBL_search_activities": {"status": "mocked", "results": []},
    })

    run_id = _seed_synthetic_cct(
        local_storage, registry_service, workflow_state_service,
        materials=[_material("payload_smiles", "CCO")],
    )
    DevelopabilityAgent(
        storage=local_storage, registry=registry_service,
        workflow_state=workflow_state_service,
        mcp_client=LocalMCPClient(bindings=bindings),
        llm=_OutOfScopeLLM(),
    ).run(run_id)
    persisted = local_storage.read_json(
        local_storage.run_key(run_id, "structured_liability_summary.json")
    )

    called = {
        tc["tool_name"]
        for cand in persisted["candidate_liability_results"]
        for lane in cand["lane_results"]
        for tc in lane["tool_call_records"]
    }
    for forbidden in (
        "MultiAgentLiteratureSearch",
        "FDA_OrangeBook_get_patent_info",
        "PubChem_get_associated_patents_by_CID",
        "drugbank_get_drug_references_by_drug_name_or_id",
        "EuropePMC_search_articles",
        "TotallyHallucinatedTool",
    ):
        assert forbidden not in called, (
            f"Step 6 must never call {forbidden}; got tool call set {called}"
        )
    # Turn B respects cleaned-empty selections instead of silently falling
    # back to a deterministic tool execution.
    assert "DrugProps_pains_filter" not in called


# ── 7. Stage 1 / Stage 2 LLM payload boundary (catalog only, no _live) ──────

_STEP6_VALID_TOOLS = {
    capability.tool_name for capability in STEP_06_CAPABILITY_REGISTRY
    if capability.lane_type is not None
}


class _RecordingLLM:
    name = "recording"
    model = "rec"

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def generate(self, prompt: str, *, system: str | None = None, **kw: Any) -> str:
        raise NotImplementedError

    def generate_json(self, prompt: str, *, schema: dict, system: str | None = None) -> dict:
        self.calls.append({"system": system, "schema": schema})
        task = (schema or {}).get("task")
        if task == "step6_schema_mapping_stage_1":
            return {
                "selections": [
                    {"tool_name": "DrugProps_pains_filter",
                     "selection_reason": "test"}
                ]
            }
        if task == "step6_schema_mapping_stage_2":
            tools = (schema or {}).get("tools") or []
            field_ref = next(
                f["field_ref"] for f in schema.get("candidate_available_fields", [])
                if f.get("value_kind") == "smiles"
            )
            out = []
            for t in tools:
                out.append({
                    "tool_name": t.get("tool_name"),
                    "can_invoke": True,
                    "argument_mapping": {"smiles": field_ref},
                    "missing_required_fields": [],
                    "argument_mapping_reason": "ok",
                })
            return {"tools": out}
        return {}


class _SelectAllEligibleLLM:
    name = "select_all_eligible"
    model = "test"

    def generate(self, prompt: str, *, system: str | None = None, **kw: Any) -> str:
        raise NotImplementedError

    def generate_json(self, prompt: str, *, schema: dict, system: str | None = None) -> dict:
        if schema.get("task") == "step6_schema_mapping_stage_1":
            return {
                "selections": [
                    {
                        "tool_name": entry["tool_name"],
                        "selection_reason": "complementary production coverage",
                    }
                    for entry in schema.get("compact_catalog") or []
                ]
            }
        if schema.get("task") == "step6_schema_mapping_stage_2":
            field_ref = next(
                f["field_ref"] for f in schema.get("candidate_available_fields", [])
                if f.get("value_kind") == "smiles"
            )
            return {
                "tools": [
                    {
                        "tool_name": tool["tool_name"],
                        "can_invoke": True,
                        "argument_mapping": {"smiles": field_ref},
                        "missing_required_fields": [],
                        "argument_mapping_reason": "test",
                    }
                    for tool in schema.get("tools") or []
                ]
            }
        return {"tools": []}


def test_step6_five_complementary_smiles_tools_are_not_truncated(
    local_storage, registry_service, workflow_state_service
):
    run_id = _seed_synthetic_cct(
        local_storage, registry_service, workflow_state_service,
        materials=[_material("payload_smiles", "CCO")],
    )
    DevelopabilityAgent(
        storage=local_storage, registry=registry_service,
        workflow_state=workflow_state_service,
        mcp_client=LocalMCPClient(bindings=_bindings(_DEFAULT_OK_BINDINGS)),
        llm=_SelectAllEligibleLLM(),
    ).run(run_id)
    persisted = local_storage.read_json(
        local_storage.run_key(run_id, "structured_liability_summary.json")
    )
    lane = _lane_results(persisted)["payload_linker_compound_liability"]
    selected = set(lane["selected_tools"])
    assert {
        "DrugProps_pains_filter",
        "DrugProps_lipinski_filter",
        "DrugProps_calculate_qed",
        "SwissADME_calculate_adme",
        "SwissADME_check_druglikeness",
    } <= selected
    assert "ADMETAI_predict_toxicity" in selected


def test_step6_swissadme_executes_with_official_operation_literal(
    local_storage, registry_service, workflow_state_service
):
    run_id = _seed_synthetic_cct(
        local_storage, registry_service, workflow_state_service,
        materials=[_material("payload_smiles", "CCO")],
    )
    captured: list[dict] = []

    def _swiss(**kwargs):
        captured.append(dict(kwargs))
        return {"status": "mocked", "adme": {}}

    DevelopabilityAgent(
        storage=local_storage,
        registry=registry_service,
        workflow_state=workflow_state_service,
        mcp_client=LocalMCPClient(bindings={"SwissADME_calculate_adme": _swiss}),
        llm=_SelectAllEligibleLLM(),
    ).run(run_id)

    assert captured
    assert captured[0]["operation"] == "calculate_adme"
    assert captured[0]["smiles"] == "CCO"
    persisted = local_storage.read_json(
        local_storage.run_key(run_id, "structured_liability_summary.json")
    )
    blob = json.dumps(persisted)
    assert "CCO" not in blob
    audit_entries = [
        entry
        for cand in persisted["candidate_liability_results"]
        for lane in cand["lane_results"]
        for entry in lane.get("argument_mapping_audit", [])
        if entry.get("tool_name") == "SwissADME_calculate_adme"
    ]
    assert any(
        entry.get("schema_arg") == "operation"
        and entry.get("argument_value_source") == "mapped_from_official_schema_literal"
        for entry in audit_entries
    )


def test_step6_tool_selection_prompt_hides_live_and_uses_progressive_disclosure(
    local_storage, registry_service, workflow_state_service
):
    llm = _RecordingLLM()
    run_id = _seed_synthetic_cct(
        local_storage, registry_service, workflow_state_service,
        materials=[_material("payload_smiles", "CCO")],
    )
    DevelopabilityAgent(
        storage=local_storage, registry=registry_service,
        workflow_state=workflow_state_service,
        mcp_client=LocalMCPClient(bindings=_bindings(_DEFAULT_OK_BINDINGS)),
        llm=llm,
    ).run(run_id)

    stage1 = [c for c in llm.calls if c["schema"].get("task") == "step6_schema_mapping_stage_1"]
    stage2 = [c for c in llm.calls if c["schema"].get("task") == "step6_schema_mapping_stage_2"]
    # Stage 1 must fire exactly once per candidate. Stage 2 maps selected
    # tool schemas to candidate_available_fields refs.
    assert stage1

    # Stage 1 payload: disclosed compact catalog + safe candidate fields; no full_schema, no `_live`.
    for c in stage1:
        sc = c["schema"]
        assert "compact_catalog" in sc and "candidate_available_fields" in sc
        assert "full_schema" not in sc
        blob = str(sc)
        assert "_live" not in blob
        for entry in sc["compact_catalog"]:
            assert set(entry).issuperset({
                "tool_name", "short_description", "capability_tags",
                "coarse_input_requirements", "step_id", "agent_name",
            })
            assert set(entry).issubset({
                "tool_name", "short_description", "capability_tags",
                "coarse_input_requirements", "step_id", "agent_name",
                "runtime_policy", "runtime_status",
            })
        for entry in sc["compact_catalog"]:
            assert entry["tool_name"] in _STEP6_VALID_TOOLS

    # Stage 2 payload (if fired): per-tool schema; `_live` never exposed.
    for c in stage2:
        sc = c["schema"]
        tools = sc.get("tools") or []
        assert tools
        for t in tools:
            assert t.get("tool_name") in _STEP6_VALID_TOOLS
            full_schema = t.get("full_schema") or {}
            assert "_live" not in (full_schema.get("properties") or {})


# ── Stage 2 maps schema args to field refs ─────────────────────────────────


class _DeterministicProbeLLM:
    """Stage 1 picks one tool; Stage 2 maps its arg to a field_ref."""
    name = "det_probe"
    model = "test"

    def __init__(self) -> None:
        self.tasks: list[str] = []

    def generate(self, prompt, *, system=None, **kw):
        raise NotImplementedError

    def generate_json(self, prompt, *, schema, system=None):
        task = (schema or {}).get("task") or ""
        self.tasks.append(task)
        if task == "step6_schema_mapping_stage_1":
            return {
                "selections": [
                    {"tool_name": "DrugProps_pains_filter",
                     "selection_reason": "smiles available"}
                ]
            }
        if task == "step6_schema_mapping_stage_2":
            field_ref = next(
                f["field_ref"] for f in schema.get("candidate_available_fields", [])
                if f.get("value_kind") == "smiles"
            )
            return {"tools": [{
                "tool_name": "DrugProps_pains_filter",
                "can_invoke": True,
                "argument_mapping": {"smiles": field_ref},
                "missing_required_fields": [],
                "argument_mapping_reason": "test mapping",
            }]}
        return {}


def test_step6_stage2_maps_required_args_to_field_refs(
    local_storage, registry_service, workflow_state_service
):
    """Stage 2 now emits schema-arg → field_ref mapping; raw values are
    resolved only at runtime."""
    llm = _DeterministicProbeLLM()
    run_id = _seed_synthetic_cct(
        local_storage, registry_service, workflow_state_service,
        materials=[_material("payload_smiles", "CCO")],
    )
    DevelopabilityAgent(
        storage=local_storage, registry=registry_service,
        workflow_state=workflow_state_service,
        mcp_client=LocalMCPClient(bindings=_bindings(_DEFAULT_OK_BINDINGS)),
        llm=llm,
    ).run(run_id)
    assert llm.tasks.count("step6_schema_mapping_stage_1") == 1
    assert llm.tasks.count("step6_schema_mapping_stage_2") == 1

    # And the plan still ran via deterministic args, not skipped.
    persisted = local_storage.read_json(
        local_storage.run_key(run_id, "structured_liability_summary.json")
    )
    provenance = {
        (
            tc["tool_input_summary"].get("tool_selection_source"),
            tc["tool_input_summary"].get("argument_construction_source"),
        )
        for cand in persisted["candidate_liability_results"]
        for lane in cand["lane_results"]
        for tc in lane["tool_call_records"]
        if tc.get("tool_name") == "DrugProps_pains_filter"
    }
    assert ("llm_stage1", "llm_stage2") in provenance


def _single_sequence_tool_llm(tool_name: str):
    class _LLM:
        name = "single_seq"
        model = "test"

        def __init__(self) -> None:
            self.calls: list[str] = []

        def generate(self, prompt: str, *, system: str | None = None, **kw: Any) -> str:
            raise NotImplementedError

        def generate_json(self, prompt: str, *, schema: dict, system: str | None = None) -> dict:
            self.calls.append((schema or {}).get("task", ""))
            task = (schema or {}).get("task")
            if task == "step6_schema_mapping_stage_1":
                return {"selections": [{"tool_name": tool_name, "selection_reason": "sequence lane"}]}
            if task == "step6_schema_mapping_stage_2":
                field_ref = next(
                    f["field_ref"] for f in schema.get("candidate_available_fields", [])
                    if f.get("material_type") == "antibody_heavy_chain_sequence"
                )
                return {
                    "tools": [{
                        "tool_name": tool_name,
                        "can_invoke": True,
                        "argument_mapping": {"sequence": field_ref},
                        "missing_required_fields": [],
                        "argument_mapping_reason": "force heavy sequence mapping",
                    }]
                }
            return {}

    return _LLM()


def _mock_sequence_tool(tool_name: str, calls: list[dict[str, Any]]):
    def _tool(**kw: Any) -> dict[str, Any]:
        calls.append(dict(kw))
        if tool_name == "PROSITE_scan_sequence":
            return {"status": "mocked", "motifs": []}
        return {"status": "mocked", "predictions": []}
    return _tool


@pytest.mark.parametrize("tool_name", ["PROSITE_scan_sequence", "IEDB_predict_mhci_binding"])
def test_step6_single_sequence_tool_expands_to_both_antibody_chains(
    local_storage, registry_service, workflow_state_service, tool_name: str
):
    heavy_seq = "EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGK"
    light_seq = "DIQMTQSPSSLSASVGDRVTITCQASQDIQLLNGRT"
    calls: list[dict[str, Any]] = []
    llm = _single_sequence_tool_llm(tool_name)
    run_id = _seed_synthetic_cct(
        local_storage,
        registry_service,
        workflow_state_service,
        materials=[
            _material("antibody_heavy_chain_sequence", heavy_seq),
            _material("antibody_light_chain_sequence", light_seq),
        ],
        candidate_type="antibody",
    )
    DevelopabilityAgent(
        storage=local_storage,
        registry=registry_service,
        workflow_state=workflow_state_service,
        mcp_client=LocalMCPClient(bindings={tool_name: _mock_sequence_tool(tool_name, calls)}),
        llm=llm,
    ).run(run_id)

    persisted = local_storage.read_json(local_storage.run_key(run_id, "structured_liability_summary.json"))
    lane = _lane_results(persisted)["antibody_protein_sequence_liability"]
    tool_calls = [
        tc for tc in lane["tool_call_records"] if tc["tool_name"] == tool_name
    ]
    assert len(tool_calls) == 2
    assert all(tc["run_status"] == "success" for tc in tool_calls)
    assert {tc["tool_input_summary"].get("chain_role") for tc in tool_calls} == {"heavy", "light"}
    expanded_refs = [
        tc["tool_input_summary"].get("expanded_field_ref")
        for tc in tool_calls
        if tc["tool_input_summary"].get("runtime_chain_expansion")
    ]
    assert len(expanded_refs) == 2
    assert len(set(expanded_refs)) == 2
    assert all(tc["tool_input_summary"].get("runtime_chain_expansion") is True for tc in tool_calls)
    for tc in tool_calls:
        summary = tc["tool_input_summary"]
        expanded_ref = summary.get("expanded_field_ref")
        assert summary["argument_field_refs"].get("sequence") == expanded_ref
        mapping_entries = [
            e for e in summary.get("argument_mapping_audit", [])
            if isinstance(e, dict) and e.get("schema_arg") == "sequence"
        ]
        assert mapping_entries
        assert all(e.get("field_ref") == expanded_ref for e in mapping_entries)
        resolver_entries = [
            e for e in summary.get("runtime_resolver_audit", [])
            if isinstance(e, dict) and e.get("schema_arg") == "sequence"
        ]
        assert resolver_entries
        assert all(e.get("field_ref") == expanded_ref for e in resolver_entries)
        assert heavy_seq not in json.dumps(tc["tool_input_summary"])
        assert light_seq not in json.dumps(tc["tool_input_summary"])
        assert tc["tool_input_summary"].get("expansion_reason") is not None
        if tc["tool_output_ref"]:
            out = local_storage.read_json(tc["tool_output_ref"])
            assert heavy_seq not in json.dumps(out)
            assert light_seq not in json.dumps(out)

    summary = persisted["selection_audit"]
    assert summary["step_06_runtime_chain_expanded_tools"].count(tool_name) >= 1
    assert any(
        entry.get("tool_name") == tool_name
        for entry in summary.get("step_06_runtime_chain_expansion_details", [])
    )


def test_step6_single_sequence_tool_runs_once_when_only_one_chain(
    local_storage, registry_service, workflow_state_service
):
    heavy_seq = "EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGK"
    calls: list[dict[str, Any]] = []
    tool_name = "PROSITE_scan_sequence"
    llm = _single_sequence_tool_llm(tool_name)
    run_id = _seed_synthetic_cct(
        local_storage,
        registry_service,
        workflow_state_service,
        materials=[_material("antibody_heavy_chain_sequence", heavy_seq)],
        candidate_type="antibody",
    )
    DevelopabilityAgent(
        storage=local_storage,
        registry=registry_service,
        workflow_state=workflow_state_service,
        mcp_client=LocalMCPClient(bindings={tool_name: _mock_sequence_tool(tool_name, calls)}),
        llm=llm,
    ).run(run_id)

    persisted = local_storage.read_json(local_storage.run_key(run_id, "structured_liability_summary.json"))
    lane = _lane_results(persisted)["antibody_protein_sequence_liability"]
    tool_calls = [tc for tc in lane["tool_call_records"] if tc["tool_name"] == tool_name]
    assert len(tool_calls) == 1
    assert tool_calls[0]["tool_input_summary"].get("runtime_chain_expansion") is not True
    summary = persisted["selection_audit"]
    assert summary["step_06_runtime_chain_expansion_details"] == []


def test_step6_non_antibody_sequence_ref_does_not_expand(
    local_storage, registry_service, workflow_state_service
):
    heavy_seq = "EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGK"
    light_seq = "DIQMTQSPSSLSASVGDRVTITCQASQDIQLLNGRT"
    target_seq = "MKTAYIAKQRQISFVKSHFSRQDILDLWIY"

    class _LLM:
        name = "target_sequence_mapping"
        model = "test"

        def __init__(self) -> None:
            self.calls: list[str] = []

        def generate(self, prompt, *, system=None, **kw):
            raise NotImplementedError

        def generate_json(self, prompt, *, schema, system=None):
            task = (schema or {}).get("task")
            if task == "step6_schema_mapping_stage_1":
                return {"selections": [{"tool_name": "IEDB_predict_mhci_binding"}]}
            if task == "step6_schema_mapping_stage_2":
                field_ref = next(
                    f["field_ref"] for f in schema.get("candidate_available_fields", [])
                    if f.get("material_type") == "target_sequence"
                )
                return {
                    "tools": [{
                        "tool_name": "IEDB_predict_mhci_binding",
                        "can_invoke": True,
                        "argument_mapping": {"sequence": field_ref},
                        "missing_required_fields": [],
                        "argument_mapping_reason": "non-antibody sequence",
                    }]
                }
            return {}

    llm = _LLM()
    calls: list[dict[str, Any]] = []
    run_id = _seed_synthetic_cct(
        local_storage,
        registry_service,
        workflow_state_service,
        materials=[
            _material("antibody_heavy_chain_sequence", heavy_seq),
            _material("antibody_light_chain_sequence", light_seq),
            _material("target_sequence", target_seq),
        ],
        candidate_type="antibody",
    )
    DevelopabilityAgent(
        storage=local_storage,
        registry=registry_service,
        workflow_state=workflow_state_service,
        mcp_client=LocalMCPClient(bindings={"IEDB_predict_mhci_binding": _mock_sequence_tool("IEDB_predict_mhci_binding", calls)}),
        llm=llm,
    ).run(run_id)

    persisted = local_storage.read_json(
        local_storage.run_key(run_id, "structured_liability_summary.json")
    )
    lane = _lane_results(persisted)["antibody_protein_sequence_liability"]
    tool_calls = [tc for tc in lane["tool_call_records"] if tc["tool_name"] == "IEDB_predict_mhci_binding"]
    assert len(tool_calls) == 1
    assert tool_calls[0]["tool_input_summary"].get("runtime_chain_expansion") is not True
    assert "chain_role" not in (tool_calls[0]["tool_input_summary"] or {})
    assert calls and calls[0].get("sequence") == target_seq


def test_step6_lane_summary_surfaces_upstream_error_envelope(
    local_storage, registry_service, workflow_state_service
):
    heavy_seq = "EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGK"
    light_seq = "DIQMTQSPSSLSASVGDRVTITCQASQDIQLLNGRT"
    llm = _single_sequence_tool_llm("IEDB_predict_mhci_binding")
    run_id = _seed_synthetic_cct(
        local_storage,
        registry_service,
        workflow_state_service,
        materials=[
            _material("antibody_heavy_chain_sequence", heavy_seq),
            _material("antibody_light_chain_sequence", light_seq),
        ],
        candidate_type="antibody",
    )

    def _iedb_upstream_error(**_kw: Any) -> dict[str, Any]:
        return {
            "status": "upstream_error",
            "source": "IEDB_predict_mhci_binding",
            "executor": "tooluniverse",
            "error_message": "IEDB HTTP 500",
        }

    DevelopabilityAgent(
        storage=local_storage,
        registry=registry_service,
        workflow_state=workflow_state_service,
        mcp_client=LocalMCPClient(bindings={"IEDB_predict_mhci_binding": _iedb_upstream_error}),
        llm=llm,
    ).run(run_id)

    persisted = local_storage.read_json(local_storage.run_key(run_id, "structured_liability_summary.json"))
    lane = _lane_results(persisted)["antibody_protein_sequence_liability"]
    assert lane["run_status"] == "partial"
    summary = lane["lane_summary"].lower()
    assert "upstream_error" in summary
    assert "outer run_statuses [success]" in summary
    calls = [
        tc for tc in lane["tool_call_records"]
        if tc["tool_name"] == "IEDB_predict_mhci_binding"
    ]
    assert calls
    assert all(tc["run_status"] == "success" for tc in calls)
    assert all(tc["tool_output_ref"] for tc in calls)
    assert all(
        tc["tool_input_summary"].get("output_envelope_status") == "upstream_error"
        for tc in calls
    )
    audit = persisted["selection_audit"]
    assert "IEDB_predict_mhci_binding" in audit["step_06_upstream_error_tools"]
    blob = json.dumps(persisted)
    assert heavy_seq not in blob
    assert light_seq not in blob

# ── LLM call-count budget (per-candidate Stage1 + Stage2) ───────────────────


class _CountingLLM:
    """Records every LLM call task name; returns the Mock outputs verbatim."""
    name = "counting"
    model = "cnt"

    def __init__(self) -> None:
        self.tasks: list[str] = []
        from app.llm.provider import (  # noqa: PLC0415
            _mock_step6_schema_mapping_stage1,
            _mock_step6_schema_mapping_stage2,
            _mock_stage1_multi_lane,
            _mock_stage2_multi_tool,
            _mock_stage1_selection,
            _mock_stage2_arguments,
        )
        self._dispatch = {
            "step6_schema_mapping_stage_1": _mock_step6_schema_mapping_stage1,
            "step6_schema_mapping_stage_2": _mock_step6_schema_mapping_stage2,
            "tool_selection_stage_1_multi_lane": _mock_stage1_multi_lane,
            "tool_selection_stage_2_multi_tool": _mock_stage2_multi_tool,
            "tool_selection_stage_1": _mock_stage1_selection,
            "tool_selection_stage_2": _mock_stage2_arguments,
        }

    def generate(self, prompt: str, *, system: str | None = None, **kw: Any) -> str:
        raise NotImplementedError

    def generate_json(self, prompt: str, *, schema: dict, system: str | None = None) -> dict:
        task = (schema or {}).get("task") or ""
        self.tasks.append(task)
        fn = self._dispatch.get(task)
        return fn(schema) if fn else {}


def _multi_lane_candidate_materials() -> list[dict]:
    """A single candidate with materials triggering 4 lanes simultaneously."""
    return [
        _material("payload_smiles", "CCO"),
        _material("antibody_heavy_chain_sequence", "EVQLVESGGGLVQPGGSLRLSCAASGFNI"),
        _material("target_antigen_name", "HER2"),
        _material("structure_ref", "pdb:1N8Z"),
    ]


def test_step6_one_candidate_multi_lane_makes_at_most_one_stage1_and_one_stage2(
    local_storage, registry_service, workflow_state_service
):
    """Per-candidate budget: one candidate with N active lanes calls LLM
    at most twice (Stage 1 + Stage 2), not N or N*tools times."""
    llm = _CountingLLM()
    run_id = _seed_synthetic_cct(
        local_storage, registry_service, workflow_state_service,
        materials=_multi_lane_candidate_materials(),
        identifiers=[{"id_type": "uniprot_id", "id_value": "P04626", "source_ids": [], "confidence": 0.9},
                     {"id_type": "pdb_id", "id_value": "1N8Z", "source_ids": [], "confidence": 0.9}],
        candidate_type="adc_construct",
    )
    DevelopabilityAgent(
        storage=local_storage, registry=registry_service,
        workflow_state=workflow_state_service,
        mcp_client=LocalMCPClient(bindings=_bindings(_DEFAULT_OK_BINDINGS)),
        llm=llm,
    ).run(run_id)

    stage1 = sum(1 for t in llm.tasks if t == "step6_schema_mapping_stage_1")
    stage2 = sum(1 for t in llm.tasks if t == "step6_schema_mapping_stage_2")
    # Single candidate ⇒ exactly one Stage 1, at most one Stage 2.
    assert stage1 == 1, f"expected 1 Stage 1 call, got {stage1} (tasks={llm.tasks})"
    assert stage2 <= 1, f"expected ≤1 Stage 2 call, got {stage2} (tasks={llm.tasks})"
    # And no legacy per-lane / per-tool task names.
    assert "tool_selection_stage_1" not in llm.tasks
    assert "tool_selection_stage_2" not in llm.tasks
    assert "tool_selection_stage_1_multi_lane" not in llm.tasks
    assert "tool_selection_stage_2_multi_tool" not in llm.tasks


def _seed_n_candidates(
    local_storage, registry_service, workflow_state_service, *, n: int
) -> str:
    """Stamp `n` synthetic compound-component candidates into one cct."""
    from app.services.intake_service import IntakeService  # noqa: PLC0415

    intake = IntakeService(local_storage, registry_service, workflow_state_service)
    rec = intake.submit(
        raw_user_query="multi-candidate fixture",
        user_provided_context={"target_or_antigen_text": "synthetic"},
    )
    run_id = rec.run_id
    artifact_id = new_artifact_id("candidate_context_table")
    cct = {
        "artifact_id": artifact_id,
        "run_id": run_id,
        "step_id": "step_05_candidate_context",
        "created_at": now_iso(),
        "context_build_status": "ok",
        "candidate_records": [
            {
                "candidate_id": f"cand_{i}",
                "candidate_label": f"synthetic_{i}",
                "candidate_type": "compound_component",
                "source_records": [],
                "identifiers": [],
                "materials": [_material("payload_smiles", f"CCO{i}")],
                "adc_links": {
                    "target_material_ids": [], "antibody_material_ids": [],
                    "payload_material_ids": [], "linker_material_ids": [],
                    "dar_material_ids": [],
                },
                "candidate_status": "partially_ready_for_step6",
                "candidate_role": "user_provided_candidate",
                "is_generated_candidate": False,
                "context_status": "partial",
                "data_gaps": [], "missing_material_roles": [], "context_notes": [],
            }
            for i in range(n)
        ],
        "missing_context_flags": [],
        "tool_call_records": [],
        "downstream_query_hints": [],
    }
    local_storage.write_json(
        local_storage.run_key(run_id, "candidate_context_table.json"), cct
    )
    registry_service.update_active(run_id, candidate_context_table_id=artifact_id)
    return run_id


def test_step6_five_candidates_stay_under_ten_llm_calls(
    local_storage, registry_service, workflow_state_service
):
    """5 candidates × (1 Stage 1 + 1 Stage 2) ≤ 10 LLM calls."""
    llm = _CountingLLM()
    run_id = _seed_n_candidates(
        local_storage, registry_service, workflow_state_service, n=5
    )
    DevelopabilityAgent(
        storage=local_storage, registry=registry_service,
        workflow_state=workflow_state_service,
        mcp_client=LocalMCPClient(bindings=_bindings(_DEFAULT_OK_BINDINGS)),
        llm=llm,
    ).run(run_id)
    stage1 = sum(1 for t in llm.tasks if t == "step6_schema_mapping_stage_1")
    stage2 = sum(1 for t in llm.tasks if t == "step6_schema_mapping_stage_2")
    assert stage1 == 5
    assert stage2 <= 5
    assert stage1 + stage2 <= 10


# ── Stage 2 contains schemas only for the Stage 1 survivors ─────────────────


def test_step6_stage2_payload_isolation_via_direct_policy_call():
    """Direct policy-level guard for Stage 2 payload structure when it
    fires. We bypass the agent (and TU schema lookup) and exercise the
    multi-lane Stage 2 path with a synthetic schema that has a required
    field deterministic mapping cannot satisfy — guaranteeing Stage 2
    fires in any environment.
    """
    from app.agents.tool_selection_policy import (  # noqa: PLC0415
        LaneSelectionRequest,
        ToolInvocationPlan,
        select_and_build_per_candidate_invocations,
    )

    class _Mcp:
        def list_tools(self, *, agent_name, step_id):
            return ["DrugProps_pains_filter"]
        def call_tool(self, **_kw):
            return {"run_status": "success", "payload": {}}

    class _LLM:
        name = "x"; model = "x"
        def __init__(self) -> None:
            self.calls: list[dict] = []
        def generate(self, *_a, **_kw):
            raise NotImplementedError
        def generate_json(self, prompt, *, schema, system=None):
            self.calls.append({"system": system, "schema": schema})
            task = (schema or {}).get("task")
            if task == "tool_selection_stage_1_multi_lane":
                return {"selections": [
                    {"lane_type": "payload_linker_compound_liability",
                     "tool_name": "DrugProps_pains_filter",
                     "selection_reason": "ok"}
                ]}
            if task == "tool_selection_stage_2_multi_tool":
                return {"tools": [
                    {"lane_type": "payload_linker_compound_liability",
                     "tool_name": "DrugProps_pains_filter",
                     "arguments": {"smiles": "CCO", "operation": "x"}}
                ]}
            return {}

    # Force Stage 2 by patching signature_schema_for so the survivor's
    # schema has a required field (`operation`) deterministic cannot fill.
    import app.agents.tool_selection_policy as policy

    forced_schema = {
        "type": "object",
        "properties": {
            "smiles": {"type": "string"},
            "operation": {"type": "string"},
        },
        "required": ["smiles", "operation"],
    }
    orig = policy.signature_schema_for
    policy.signature_schema_for = lambda name: forced_schema  # type: ignore[assignment]
    try:
        llm = _LLM()
        plans = select_and_build_per_candidate_invocations(
            agent_name="developability_agent",
            step_id="step_06",
            mcp_client=_Mcp(),
            llm=llm,
            candidate_id="cand_x",
            lanes=[LaneSelectionRequest(
                lane_type="payload_linker_compound_liability",
                allowed_tools=["DrugProps_pains_filter"],
                signals={"smiles": True},
                arg_hints={"smiles": "CCO"},
            )],
            deterministic_fallback=lambda lt: [ToolInvocationPlan(
                tool_name="DrugProps_pains_filter", selection_reason="fb",
                selected_by="deterministic_fallback",
            )],
            deterministic_argument_mapping=lambda tn, hints: {"smiles": hints.get("smiles", "")},
        )
    finally:
        policy.signature_schema_for = orig

    stage2 = [c for c in llm.calls if c["schema"].get("task") == "tool_selection_stage_2_multi_tool"]
    assert stage2, "Stage 2 must fire when deterministic mapping cannot satisfy required args"
    for c in stage2:
        tools_in_payload = c["schema"]["tools"]
        names = {t["tool_name"] for t in tools_in_payload}
        # Only the single Stage 1 survivor is in Stage 2; no Step 13/14 tools.
        assert names == {"DrugProps_pains_filter"}
        forbidden = {
            "EuropePMC_search_articles", "LiteratureSearchTool",
            "MultiAgentLiteratureSearch", "PubTator3_LiteratureSearch",
            "PubChem_get_associated_patents_by_CID",
            "drugbank_get_drug_references_by_drug_name_or_id",
            "FDA_OrangeBook_get_patent_info",
        }
        assert not (names & forbidden)
        # `_live` never exposed.
        for t in tools_in_payload:
            assert "_live" not in (t.get("full_schema", {}).get("properties") or {})
    # And the resulting plan has the LLM-supplied args, not the deterministic fallback.
    assert plans["payload_linker_compound_liability"][0].selected_by == "llm"


def test_step6_stage2_missing_required_args_does_not_crash(
    local_storage, registry_service, workflow_state_service
):
    """Stage 2 returning {} for every tool → policy falls back to
    deterministic mapping; if that also fails the plan is skipped, never
    raising."""

    class _EmptyStage2LLM:
        name = "empty_stage2"
        model = "test"

        def generate(self, prompt, *, system=None, **kw):
            raise NotImplementedError

        def generate_json(self, prompt, *, schema, system=None):
            task = (schema or {}).get("task")
            if task == "step6_schema_mapping_stage_1":
                return {
                    "selections": [
                        {"tool_name": "DrugProps_pains_filter",
                         "selection_reason": "ok"}
                    ]
                }
            if task == "step6_schema_mapping_stage_2":
                # Refuse to provide any arguments.
                return {"tools": []}
            return {}

    run_id = _seed_synthetic_cct(
        local_storage, registry_service, workflow_state_service,
        materials=[_material("payload_smiles", "CCO")],
    )
    DevelopabilityAgent(
        storage=local_storage, registry=registry_service,
        workflow_state=workflow_state_service,
        mcp_client=LocalMCPClient(bindings=_bindings(_DEFAULT_OK_BINDINGS)),
        llm=_EmptyStage2LLM(),
    ).run(run_id)
    # No crash — valid Stage 2 can_invoke=false / omitted mapping is
    # respected, so the step may finish with only missing lanes.
    persisted = local_storage.read_json(
        local_storage.run_key(run_id, "structured_liability_summary.json")
    )
    assert persisted["prefilter_status"] in {"completed", "partial", "completed_with_missing_lanes"}
