"""Step 9 readiness projection hard-gate behavior.

These tests validate that official-tool required argument metadata is treated as
the primary gate signal (with legacy fallback only for missing metadata).
"""

from __future__ import annotations

from app.agents import step_09_available_fields as step9


def _seed_candidate_context_table() -> dict:
    return {
        "candidate_records": [
            {
                "candidate_id": "cand_t1",
                "candidate_type": "target_antigen",
                "materials": [
                    {
                        "material_id": "tgt_seq",
                        "material_type": "target_sequence",
                        "value": "MKTAYIAKQNNVG",
                    }
                ],
                "identifiers": [
                    {
                        "id_type": "uniprot_id",
                        "id_value": "P00533",
                    },
                ],
            },
            {
                "candidate_id": "cand_c1",
                "candidate_type": "compound_component",
                "materials": [
                    {
                        "material_id": "cmp_smiles",
                        "material_type": "payload_smiles",
                        "value": "CCO",
                    }
                ],
            },
        ]
    }


def _seed_ready_step8_structure_reference() -> dict:
    return {
        "candidate_structure_results": [
            {
                "candidate_id": "cand_t1",
                "downstream_handoff": {
                    "validated_structure_ref": "s3://tests/structure.pdb",
                },
            }
        ]
    }


_STEP9_TOOL_SIGNATURE_CONTRACT_REQUIRED: dict[str, list[str]] = {
    "AlphaMissense_get_variant_score": ["uniprot_id", "variant"],
    "DynaMut2_predict_stability": ["operation", "pdb_id", "chain", "mutation"],
    "ESM_generate_protein_sequence": ["prompt_sequence"],
    "ESM_score_variant_sae_batch": ["sequence", "variants"],
    "NvidiaNIM_proteinmpnn": ["input_pdb"],
    "NvidiaNIM_rfdiffusion": ["contigs", "input_pdb"],
    "ChEMBL_search_molecules": [],
    "ChEMBL_search_similarity": ["smiles", "threshold"],
    "ChEMBL_search_substructure": ["smiles"],
    "ZINC_get_compound": ["operation", "zinc_id"],
    "ZINC_get_purchasable": ["operation", "tier"],
    "ZINC_search_by_properties": ["operation"],
    "ZINC_search_by_smiles": ["operation", "smiles"],
    "ZINC_search_compounds": ["operation", "query"],
}


def _schema_from_required_fields(required: list[str]) -> dict:
    properties = {}
    for arg in required:
        prop: dict[str, object] = {"type": "string"}
        if arg == "operation":
            prop["enum"] = ["query"]
        properties[arg] = prop
    return {
        "type": "object",
        "properties": properties,
        "required": required,
    }


def _configure_contract_schemas(
    monkeypatch,
    *,
    unavailable: set[str] | None = None,
) -> None:
    unavailable = unavailable or set()
    monkeypatch.setattr(step9, "_TOOL_REQUIRED_ARGS_CACHE", {}, raising=False)

    def _schema_for(name: str):
        if name in unavailable:
            return None
        required = _STEP9_TOOL_SIGNATURE_CONTRACT_REQUIRED.get(name)
        if required is None:
            return {"type": "object", "properties": {}, "required": []}
        return _schema_from_required_fields(required)

    monkeypatch.setattr(step9, "signature_schema_for", _schema_for)


def test_step9_hard_gate_schema_required_args_override_legacy_for_dynamut(monkeypatch):
    candidate_context = _seed_candidate_context_table()
    _configure_contract_schemas(monkeypatch)

    base_required = dict(_STEP9_TOOL_SIGNATURE_CONTRACT_REQUIRED)
    base_required["DynaMut2_predict_stability"] = ["structure_ref"]

    def _schema_for(name: str):
        required = base_required.get(name)
        if required is None:
            return {"type": "object", "properties": {}, "required": []}
        return _schema_from_required_fields(required)

    monkeypatch.setattr(step9, "signature_schema_for", _schema_for)

    readiness = step9.project_step9_readiness(
        candidate_context_table=candidate_context,
        prepared_structure_input_package={},
        structure_prediction_and_interface_results={},
        compound_context_text="",
    )
    blocked = {
        entry.tool_name: entry.reason
        for entry in readiness["step9_hard_gate_blocked_tools_with_reason"]
    }
    assert blocked["DynaMut2_predict_stability"].startswith("schema_required:")
    assert "structure_ref" in blocked["DynaMut2_predict_stability"]


def test_step9_hard_gate_schema_required_args_satisfied_allows_tool(monkeypatch):
    candidate_context = _seed_candidate_context_table()
    # add explicit variant to satisfy mutation requirement.
    candidate_context["candidate_records"][0]["identifiers"].append(
        {"id_type": "mutation", "id_value": "p.V600E"}
    )
    _configure_contract_schemas(monkeypatch)

    base_required = dict(_STEP9_TOOL_SIGNATURE_CONTRACT_REQUIRED)
    base_required["DynaMut2_predict_stability"] = ["pdb_id", "mutation"]

    def _schema_for(name: str):
        required = base_required.get(name)
        if required is None:
            return {"type": "object", "properties": {}, "required": []}
        return _schema_from_required_fields(required)

    monkeypatch.setattr(step9, "signature_schema_for", _schema_for)

    readiness = step9.project_step9_readiness(
        candidate_context_table=candidate_context,
        prepared_structure_input_package={},
        structure_prediction_and_interface_results=_seed_step8_complex_result(),
        compound_context_text="",
    )
    allowed = {entry.tool_name for entry in readiness["step9_hard_gate_allowed_tools"]}

    assert "DynaMut2_predict_stability" in allowed


def test_step9_tool_schema_required_args_summary_matches_signature_contract(monkeypatch):
    candidate_context = _seed_candidate_context_table()
    _configure_contract_schemas(monkeypatch)

    readiness = step9.project_step9_readiness(
        candidate_context_table=candidate_context,
        prepared_structure_input_package={},
        structure_prediction_and_interface_results=_seed_step8_complex_result(),
        compound_context_text="",
    )
    assert readiness["step9_tool_schema_requirements"]

    recorded = {
        entry.tool_name: set(entry.required_fields)
        for entry in readiness["step9_tool_schema_requirements"]
    }

    for tool_name, expected in _STEP9_TOOL_SIGNATURE_CONTRACT_REQUIRED.items():
        assert recorded[tool_name] == set(expected)


def _seed_target_candidate_with_antigen_context() -> dict:
    return {
        "candidate_records": [
            {
                "candidate_id": "cand_t1",
                "candidate_type": "target_antigen",
                "materials": [
                    {
                        "material_id": "tgt_seq",
                        "material_type": "target_sequence",
                        "value": "MKTAYIAKQNNVG",
                    },
                    {
                        "material_id": "antibody_name",
                        "material_type": "payload_name",
                        "value": "abc small molecule",
                    },
                ],
                "identifiers": [
                    {"id_type": "uniprot_id", "id_value": "P00533"},
                ],
            }
        ]
    }


def _seed_step8_complex_result() -> dict:
    return {
        "candidate_structure_results": [
            {
                "candidate_id": "cand_t1",
                "complex_structure_refs": [
                    {
                        "source_kind": "existing_pdb_complex",
                        "pdb_id": "1ABC",
                        "source_ref": "1ABC",
                    }
                ],
            }
        ]
    }


def _seed_step8_design_results_without_complex() -> dict:
    return {
        "candidate_structure_results": [
            {
                "candidate_id": "cand_t1",
                "complex_structure_refs": [
                    {
                        "source_kind": "uploaded_local_structure",
                        "source_ref": "s3://tests/validation.pdb",
                    }
                ],
                "downstream_handoff": {
                    "validated_structure_ref": "s3://tests/validation.pdb",
                },
            }
        ]
    }


def test_step9_esm_generate_requires_sequence_generation_intent(monkeypatch):
    candidate_context = _seed_target_candidate_with_antigen_context()
    _configure_contract_schemas(monkeypatch)
    readiness = step9.project_step9_readiness(
        candidate_context_table=candidate_context,
        prepared_structure_input_package={},
        structure_prediction_and_interface_results=_seed_step8_complex_result(),
        compound_context_text="",
    )
    blocked = {entry.tool_name: entry.reason for entry in readiness["step9_hard_gate_blocked_tools_with_reason"]}
    assert blocked["ESM_generate_protein_sequence"] == "intent_not_sequence_generation"


def test_step9_esm_score_requires_structured_variants(monkeypatch):
    candidate_context = _seed_target_candidate_with_antigen_context()
    _configure_contract_schemas(monkeypatch)
    readiness = step9.project_step9_readiness(
        candidate_context_table=candidate_context,
        prepared_structure_input_package={},
        structure_prediction_and_interface_results=_seed_step8_complex_result(),
        compound_context_text="",
    )
    blocked = {entry.tool_name: entry.reason for entry in readiness["step9_hard_gate_blocked_tools_with_reason"]}
    assert blocked["ESM_score_variant_sae_batch"] == "schema_required:variants"


def test_step9_alphamissense_requires_variant_and_uniprot_only(monkeypatch):
    candidate_context = _seed_target_candidate_with_antigen_context()
    _configure_contract_schemas(monkeypatch)
    readiness = step9.project_step9_readiness(
        candidate_context_table=candidate_context,
        prepared_structure_input_package={},
        structure_prediction_and_interface_results=_seed_step8_complex_result(),
        compound_context_text="",
    )
    blocked = {entry.tool_name: entry.reason for entry in readiness["step9_hard_gate_blocked_tools_with_reason"]}
    assert blocked["AlphaMissense_get_variant_score"] == "schema_required:variant"


def test_step9_alphamissense_uniprot_and_variant_allows(monkeypatch):
    candidate_context = _seed_target_candidate_with_antigen_context()
    _configure_contract_schemas(monkeypatch)
    candidate_context["candidate_records"][0]["identifiers"].append(
        {"id_type": "variant", "id_value": "p.V600E"}
    )
    readiness = step9.project_step9_readiness(
        candidate_context_table=candidate_context,
        prepared_structure_input_package={},
        structure_prediction_and_interface_results=_seed_step8_complex_result(),
        compound_context_text="",
    )
    allowed = {entry.tool_name for entry in readiness["step9_hard_gate_allowed_tools"]}
    assert "AlphaMissense_get_variant_score" in allowed


def test_step9_dynamut_requires_chain_even_with_other_inputs(monkeypatch):
    candidate_context = _seed_target_candidate_with_antigen_context()
    _configure_contract_schemas(monkeypatch)
    candidate_context["candidate_records"][0]["identifiers"].append(
        {"id_type": "mutation", "id_value": "p.V600E"}
    )
    readiness = step9.project_step9_readiness(
        candidate_context_table=candidate_context,
        prepared_structure_input_package={},
        structure_prediction_and_interface_results=_seed_step8_complex_result(),
        compound_context_text="",
    )
    blocked = {entry.tool_name: entry.reason for entry in readiness["step9_hard_gate_blocked_tools_with_reason"]}
    assert blocked["DynaMut2_predict_stability"] == "schema_required:chain"


def test_step9_rfdiffusion_requires_complex_and_contigs(monkeypatch):
    candidate_context = _seed_target_candidate_with_antigen_context()
    _configure_contract_schemas(monkeypatch)
    # has variant/uniprot to satisfy other schema args
    candidate_context["candidate_records"][0]["identifiers"].extend(
        [
            {"id_type": "variant", "id_value": "p.V600E"},
            {"id_type": "mutation", "id_value": "p.V600E"},
        ]
    )
    readiness = step9.project_step9_readiness(
        candidate_context_table=candidate_context,
        prepared_structure_input_package={},
        structure_prediction_and_interface_results=_seed_step8_complex_result(),
        compound_context_text="",
    )
    blocked = {entry.tool_name: entry.reason for entry in readiness["step9_hard_gate_blocked_tools_with_reason"]}
    assert blocked["NvidiaNIM_rfdiffusion"] == "schema_required:contigs"


def test_step9_rfdiffusion_stays_not_ready_without_true_complex(monkeypatch):
    candidate_context = _seed_target_candidate_with_antigen_context()
    _configure_contract_schemas(monkeypatch)
    readiness = step9.project_step9_readiness(
        candidate_context_table=candidate_context,
        prepared_structure_input_package={},
        structure_prediction_and_interface_results=_seed_step8_design_results_without_complex(),
        compound_context_text="",
    )
    blocked = {entry.tool_name: entry.reason for entry in readiness["step9_hard_gate_blocked_tools_with_reason"]}
    assert blocked["NvidiaNIM_rfdiffusion"] == "complex_structure_missing"


def test_step9_compound_gate_schema_driven_gaps_and_allow(monkeypatch):
    candidate_context = {
        "candidate_records": [
            {
                "candidate_id": "cand_c1",
                "candidate_type": "compound_component",
                "materials": [
                    {"material_id": "cmp_smiles", "material_type": "payload_smiles", "value": "CCO"},
                ],
                "identifiers": [
                    {"id_type": "zinc_id", "id_value": "ZINC000123"},
                ],
            }
        ]
    }
    _configure_contract_schemas(monkeypatch)
    readiness = step9.project_step9_readiness(
        candidate_context_table=candidate_context,
        prepared_structure_input_package={},
        structure_prediction_and_interface_results={},
        compound_context_text="",
    )
    blocked = {entry.tool_name: entry.reason for entry in readiness["step9_hard_gate_blocked_tools_with_reason"]}
    allowed = {entry.tool_name for entry in readiness["step9_hard_gate_allowed_tools"]}
    assert "ZINC_search_by_smiles" in allowed
    assert "ZINC_get_compound" in allowed
    assert "ChEMBL_search_substructure" in allowed
    assert blocked["ChEMBL_search_similarity"] == "schema_required:threshold"
    assert blocked["ZINC_get_purchasable"] == "schema_required:tier"


def test_step9_schema_unavailable_blocked_reason(monkeypatch):
    candidate_context = _seed_candidate_context_table()
    _configure_contract_schemas(monkeypatch, unavailable={"ChEMBL_search_similarity"})

    readiness = step9.project_step9_readiness(
        candidate_context_table=candidate_context,
        prepared_structure_input_package={},
        structure_prediction_and_interface_results=_seed_step8_complex_result(),
        compound_context_text="",
    )
    blocked = {entry.tool_name: entry.reason for entry in readiness["step9_hard_gate_blocked_tools_with_reason"]}
    assert blocked["ChEMBL_search_similarity"] == "tool_schema_unavailable"
