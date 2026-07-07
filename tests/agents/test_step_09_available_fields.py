"""Step 9 readiness projection hard-gate behavior.

These tests validate that official-tool required metadata is the authoritative gate
for Step 9, and that protein design / variant evaluation lanes are scoped.
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


def _seed_target_candidate_with_antigen_context() -> dict:
    base = _seed_candidate_context_table()["candidate_records"][0]
    candidate = dict(base)
    candidate["identifiers"] = candidate["identifiers"] + [
        {"id_type": "mutation", "id_value": "p.V600E"},
        {"id_type": "variant", "id_value": "p.V600E"},
    ]
    return {
        "candidate_records": [
            candidate,
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
                "downstream_handoff": {
                    "structure_for_variant_generation_ref": "1ABC",
                },
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


def _seed_step8_predicted_complex_without_pdb_id() -> dict:
    return {
        "candidate_structure_results": [
            {
                "candidate_id": "cand_t1",
                "complex_structure_refs": [
                    {
                        "source_kind": "predicted_complex",
                        "storage_ref": "s3://tests/predicted.pdb",
                    }
                ],
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
    candidate_context = _seed_target_candidate_with_antigen_context()
    _configure_contract_schemas(monkeypatch)

    base_required = dict(_STEP9_TOOL_SIGNATURE_CONTRACT_REQUIRED)
    base_required["DynaMut2_predict_stability"] = ["structure_ref", "operation"]

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
    blocked = {
        entry.tool_name: entry.reason
        for entry in readiness["step9_hard_gate_blocked_tools_with_reason"]
    }
    assert blocked.get("DynaMut2_predict_stability") in {None, ""}
    allowed = {entry.tool_name for entry in readiness["step9_hard_gate_allowed_tools"]}
    assert "DynaMut2_predict_stability" in allowed


def test_step9_hard_gate_schema_required_args_satisfied_allows_tool(monkeypatch):
    candidate_context = _seed_target_candidate_with_antigen_context()
    _configure_contract_schemas(monkeypatch)

    readiness = step9.project_step9_readiness(
        candidate_context_table=candidate_context,
        prepared_structure_input_package={},
        structure_prediction_and_interface_results=_seed_step8_complex_result(),
        compound_context_text="",
    )
    allowed = {
        entry.tool_name: entry for entry in readiness["step9_hard_gate_allowed_tools"]
    }
    assert "AlphaMissense_get_variant_score" in allowed
    assert "DynaMut2_predict_stability" not in allowed


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
        assert expected == sorted(recorded[tool_name]) or set(recorded[tool_name]) == set(expected)


def test_step9_alphamissense_requires_uniprot_and_variant(monkeypatch):
    candidate_context = _seed_candidate_context_table()
    _configure_contract_schemas(monkeypatch)
    readiness = step9.project_step9_readiness(
        candidate_context_table=candidate_context,
        prepared_structure_input_package={},
        structure_prediction_and_interface_results={"candidate_structure_results": []},
        compound_context_text="",
    )
    blocked = {
        entry.tool_name: entry.reason
        for entry in readiness["step9_hard_gate_blocked_tools_with_reason"]
    }
    assert blocked["AlphaMissense_get_variant_score"] == "variant_missing"


def test_step9_alphamissense_with_uniprot_and_variant_allows_without_complex(monkeypatch):
    candidate_context = _seed_target_candidate_with_antigen_context()
    _configure_contract_schemas(monkeypatch)
    readiness = step9.project_step9_readiness(
        candidate_context_table=candidate_context,
        prepared_structure_input_package={},
        structure_prediction_and_interface_results={"candidate_structure_results": []},
        compound_context_text="",
    )
    allowed = {
        entry.tool_name for entry in readiness["step9_hard_gate_allowed_tools"]
    }
    assert "AlphaMissense_get_variant_score" in allowed
    assert readiness["variant_evaluation_readiness"].status == "ready"


def test_step9_dynamut_requires_real_pdb_id_not_predicted_storage_ref(monkeypatch):
    candidate_context = _seed_target_candidate_with_antigen_context()
    candidate_context["candidate_records"][0]["identifiers"].append(
        {"id_type": "chain", "id_value": "A"}
    )
    _configure_contract_schemas(monkeypatch)

    readiness = step9.project_step9_readiness(
        candidate_context_table=candidate_context,
        prepared_structure_input_package={},
        structure_prediction_and_interface_results=_seed_step8_predicted_complex_without_pdb_id(),
        compound_context_text="",
    )
    blocked = {
        entry.tool_name: entry.reason
        for entry in readiness["step9_hard_gate_blocked_tools_with_reason"]
    }
    assert blocked["DynaMut2_predict_stability"] == "pdb_id_missing"
    fields = [
        field.model_dump()
        for field in readiness["step9_available_fields"]
        if field.value_kind == "pdb_id"
    ]
    assert fields == []


def test_step9_uploaded_or_storage_path_does_not_project_pdb_id(monkeypatch):
    candidate_context = _seed_target_candidate_with_antigen_context()
    candidate_context["candidate_records"][0]["identifiers"].append(
        {"id_type": "chain", "id_value": "A"}
    )
    _configure_contract_schemas(monkeypatch)

    readiness = step9.project_step9_readiness(
        candidate_context_table=candidate_context,
        prepared_structure_input_package={},
        structure_prediction_and_interface_results=_seed_step8_design_results_without_complex(),
        compound_context_text="",
    )
    assert not any(
        field.value_kind == "pdb_id"
        for field in readiness["step9_available_fields"]
    )
    blocked = {
        entry.tool_name: entry.reason
        for entry in readiness["step9_hard_gate_blocked_tools_with_reason"]
    }
    assert blocked["DynaMut2_predict_stability"] == "pdb_id_missing"


def test_step9_dynamut_with_existing_pdb_complex_is_allowed(monkeypatch):
    candidate_context = _seed_target_candidate_with_antigen_context()
    candidate_context["candidate_records"][0]["identifiers"].append(
        {"id_type": "chain", "id_value": "A"}
    )
    _configure_contract_schemas(monkeypatch)

    readiness = step9.project_step9_readiness(
        candidate_context_table=candidate_context,
        prepared_structure_input_package={},
        structure_prediction_and_interface_results=_seed_step8_complex_result(),
        compound_context_text="",
    )
    allowed = {
        entry.tool_name for entry in readiness["step9_hard_gate_allowed_tools"]
    }
    assert "DynaMut2_predict_stability" in allowed
    pdb_fields = [
        field
        for field in readiness["step9_available_fields"]
        if field.field_ref == "identifier:pdb_id:1ABC"
    ]
    assert len(pdb_fields) == 1
    assert pdb_fields[0].provider == "step_08"
    assert pdb_fields[0].field_type == "structure_identifier"
    assert pdb_fields[0].value_kind == "pdb_id"
    assert pdb_fields[0].source_ref == "1ABC"


def test_step9_esm_generate_requires_sequence_generation_intent(monkeypatch):
    candidate_context = _seed_target_candidate_with_antigen_context()
    _configure_contract_schemas(monkeypatch)

    readiness_missing = step9.project_step9_readiness(
        candidate_context_table=candidate_context,
        prepared_structure_input_package={},
        structure_prediction_and_interface_results=_seed_step8_complex_result(),
        compound_context_text="design an antibody region",
    )
    blocked_missing = {
        entry.tool_name: entry.reason
        for entry in readiness_missing["step9_hard_gate_blocked_tools_with_reason"]
    }
    assert blocked_missing["ESM_generate_protein_sequence"] in {
        "sequence_generation_intent_missing",
    }

    readiness_allowed = step9.project_step9_readiness(
        candidate_context_table=candidate_context,
        prepared_structure_input_package={},
        structure_prediction_and_interface_results=_seed_step8_complex_result(),
        compound_context_text="Please generate protein sequence from given sequence",
    )
    allowed = {
        entry.tool_name for entry in readiness_allowed["step9_hard_gate_allowed_tools"]
    }
    assert "ESM_generate_protein_sequence" in allowed


def test_step9_esm_score_requires_structured_variants(monkeypatch):
    candidate_context = _seed_target_candidate_with_antigen_context()
    candidate_context["candidate_records"][0]["identifiers"] = [
        ident
        for ident in candidate_context["candidate_records"][0]["identifiers"]
        if ident.get("id_type") == "uniprot_id"
    ]
    _configure_contract_schemas(monkeypatch)

    readiness = step9.project_step9_readiness(
        candidate_context_table=candidate_context,
        prepared_structure_input_package={},
        structure_prediction_and_interface_results=_seed_step8_complex_result(),
        compound_context_text="",
    )
    blocked = {
        entry.tool_name: entry.reason
        for entry in readiness["step9_hard_gate_blocked_tools_with_reason"]
    }
    assert blocked["ESM_score_variant_sae_batch"] == "variant_missing"


def test_step9_rfdiffusion_requires_complex_and_contigs(monkeypatch):
    candidate_context = _seed_target_candidate_with_antigen_context()
    _configure_contract_schemas(monkeypatch)
    # has variant/uniprot to satisfy non-structure args where present

    readiness = step9.project_step9_readiness(
        candidate_context_table=candidate_context,
        prepared_structure_input_package={},
        structure_prediction_and_interface_results=_seed_step8_complex_result(),
        compound_context_text="design sequence",
    )
    blocked = {
        entry.tool_name: entry.reason
        for entry in readiness["step9_hard_gate_blocked_tools_with_reason"]
    }
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
    blocked = {
        entry.tool_name: entry.reason
        for entry in readiness["step9_hard_gate_blocked_tools_with_reason"]
    }
    assert blocked["NvidiaNIM_rfdiffusion"] in {
        "complex_structure_missing",
        "schema_required:contigs,input_pdb",
    }
    assert "NvidiaNIM_proteinmpnn" in {
        entry.tool_name for entry in readiness["step9_hard_gate_allowed_tools"]
    }
    assert any(
        field.field_ref == "step8_validated_structure_ref:cand_t1"
        and field.provider == "step_08"
        and field.field_type == "structure"
        and field.value_kind == "structure_ref"
        for field in readiness["step9_available_fields"]
    )


def test_step9_proteinmpnn_allows_true_complex_for_input_pdb(monkeypatch):
    candidate_context = _seed_target_candidate_with_antigen_context()
    _configure_contract_schemas(monkeypatch)

    readiness = step9.project_step9_readiness(
        candidate_context_table=candidate_context,
        prepared_structure_input_package={},
        structure_prediction_and_interface_results=_seed_step8_complex_result(),
        compound_context_text="",
    )
    blocked = {
        entry.tool_name: entry.reason
        for entry in readiness["step9_hard_gate_blocked_tools_with_reason"]
    }
    assert "NvidiaNIM_proteinmpnn" not in blocked
    assert "NvidiaNIM_proteinmpnn" in {entry.tool_name for entry in readiness["step9_hard_gate_allowed_tools"]}


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
    allowed = {entry.tool_name for entry in readiness["step9_hard_gate_allowed_tools"]}
    blocked = {entry.tool_name for entry in readiness["step9_hard_gate_blocked_tools_with_reason"]}
    assert not any(name.startswith(("ZINC_", "ChEMBL_")) for name in allowed)
    assert not any(name.startswith(("ZINC_", "ChEMBL_")) for name in blocked)
    assert readiness["compound_screening_readiness"].status == "not_applicable"
    assert readiness["compound_screening_readiness"].ready_tool_count == 0
    assert readiness["compound_screening_readiness"].blocked_tool_count == 0


def test_step9_schema_unavailable_blocked_reason(monkeypatch):
    candidate_context = _seed_candidate_context_table()
    _configure_contract_schemas(monkeypatch, unavailable={"ChEMBL_search_similarity"})

    readiness = step9.project_step9_readiness(
        candidate_context_table=candidate_context,
        prepared_structure_input_package={},
        structure_prediction_and_interface_results=_seed_step8_complex_result(),
        compound_context_text="",
    )
    names = {
        entry.tool_name
        for entry in readiness["step9_hard_gate_blocked_tools_with_reason"]
    }
    assert "ChEMBL_search_similarity" not in names
    assert readiness["compound_screening_readiness"].status == "not_applicable"


def test_step9_variant_lane_readiness_profile_is_not_not_applicable(monkeypatch):
    candidate_context = _seed_target_candidate_with_antigen_context()
    _configure_contract_schemas(monkeypatch)
    readiness = step9.project_step9_readiness(
        candidate_context_table=candidate_context,
        prepared_structure_input_package={},
        structure_prediction_and_interface_results={},
        compound_context_text="generate protein sequence",
    )
    summary = readiness["step9_readiness_summary"]
    assert summary.variant_evaluation_ready_candidates >= 1
    assert readiness["variant_evaluation_readiness"].status == "ready"


def test_step9_schema_requirements_are_compact_no_raw_payload_leak(monkeypatch):
    candidate_context = _seed_candidate_context_table()
    # Include representative raw-looking material/token forms; schema audit must stay compact.
    candidate_context["candidate_records"][0]["materials"].append(
        {"material_id": "raw_payload_seq", "material_type": "target_sequence", "value": "MKTAYIAKQNNVGX9A"}
    )
    candidate_context["candidate_records"][1]["materials"].append(
        {
            "material_id": "raw_payload_smiles",
            "material_type": "payload_smiles",
            "value": "CCOCH3",
        }
    )

    _configure_contract_schemas(monkeypatch)
    readiness = step9.project_step9_readiness(
        candidate_context_table=candidate_context,
        prepared_structure_input_package={},
        structure_prediction_and_interface_results=_seed_step8_complex_result(),
        compound_context_text="",
    )
    requirements = readiness["step9_tool_schema_requirements"]
    assert requirements

    for entry in requirements:
        payload = entry.model_dump()
        assert set(payload.keys()) == {
            "candidate_id",
            "tool_name",
            "lane_type",
            "required_fields",
            "schema_source",
            "satisfiable_required_fields",
            "missing_required_fields",
            "hard_gate_decision",
            "reason",
        }
        flattened = str(payload)
        assert "MKTAYIAKQNNVGX9A" not in flattened
        assert "s3://tests/structure.pdb" not in flattened
        assert "a3m" not in flattened.lower()
        assert "fasta" not in flattened.lower()


def test_step9_aggregate_readiness_profile_counts_tools_in_protein_design_lane(monkeypatch):
    candidate_context = _seed_target_candidate_with_antigen_context()
    _configure_contract_schemas(monkeypatch)

    readiness = step9.project_step9_readiness(
        candidate_context_table=candidate_context,
        prepared_structure_input_package={},
        structure_prediction_and_interface_results=_seed_step8_complex_result(),
        compound_context_text="Please generate protein sequence",
    )

    profile = readiness["protein_design_readiness"]
    assert profile.ready_tool_count == 2
    assert profile.blocked_tool_count == 1
    assert profile.ready_tool_count == len(profile.allowed_tools)
    assert profile.blocked_tool_count == len(profile.blocked_tools)
    assert set(profile.allowed_tools) == {
        "NvidiaNIM_proteinmpnn",
        "ESM_generate_protein_sequence",
    }
    assert set(profile.blocked_tools) == {"NvidiaNIM_rfdiffusion"}

    requirement_tools = {entry.tool_name for entry in readiness["step9_tool_schema_requirements"]}
    assert requirement_tools >= {
        "NvidiaNIM_proteinmpnn",
        "NvidiaNIM_rfdiffusion",
        "ESM_generate_protein_sequence",
    }


def test_step9_aggregate_profile_does_not_show_allowed_tool_as_blocked(monkeypatch):
    candidate_a = _seed_target_candidate_with_antigen_context()["candidate_records"][0]
    candidate_a = {**candidate_a, "candidate_id": "cand_a"}
    candidate_b = _seed_target_candidate_with_antigen_context()["candidate_records"][0]
    candidate_b = {**candidate_b, "candidate_id": "cand_b"}
    candidate_context = {"candidate_records": [candidate_a, candidate_b]}
    step8_result = {
        "candidate_structure_results": [
            {
                "candidate_id": "cand_a",
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
    _configure_contract_schemas(monkeypatch)

    readiness = step9.project_step9_readiness(
        candidate_context_table=candidate_context,
        prepared_structure_input_package={},
        structure_prediction_and_interface_results=step8_result,
        compound_context_text="",
    )

    profile = readiness["protein_design_readiness"]
    assert "NvidiaNIM_proteinmpnn" in profile.allowed_tools
    assert "NvidiaNIM_proteinmpnn" not in profile.blocked_tools
    assert profile.ready_tool_count == len(profile.allowed_tools)
    assert profile.blocked_tool_count == len(profile.blocked_tools)

    blocked_records = [
        entry
        for entry in readiness["step9_hard_gate_blocked_tools_with_reason"]
        if entry.candidate_id == "cand_b" and entry.tool_name == "NvidiaNIM_proteinmpnn"
    ]
    assert blocked_records
    assert blocked_records[0].reason == "complex_structure_missing"


def test_step9_aggregate_readiness_profile_counts_tools_in_variant_evaluation_lane(monkeypatch):
    candidate_context = _seed_target_candidate_with_antigen_context()
    candidate_context["candidate_records"][0]["identifiers"].extend(
        [
            {"id_type": "pdb_id", "id_value": "1ABC"},
            {"id_type": "chain", "id_value": "A"},
        ]
    )
    _configure_contract_schemas(monkeypatch)

    readiness = step9.project_step9_readiness(
        candidate_context_table=candidate_context,
        prepared_structure_input_package={},
        structure_prediction_and_interface_results={},
        compound_context_text="generate protein sequence",
    )
    profile = readiness["variant_evaluation_readiness"]
    assert profile.ready_tool_count == 3
    assert profile.blocked_tool_count == 0
    assert profile.ready_tool_count == len(profile.allowed_tools)
    assert profile.blocked_tool_count == len(profile.blocked_tools)
    assert set(profile.allowed_tools) == {
        "AlphaMissense_get_variant_score",
        "DynaMut2_predict_stability",
        "ESM_score_variant_sae_batch",
    }


def test_step9_aggregate_readiness_profile_counts_tools_in_compound_screening_lane(monkeypatch):
    candidate_context = {
        "candidate_records": [
            {
                "candidate_id": "cand_c1",
                "candidate_type": "compound_component",
                "materials": [
                    {
                        "material_id": "cmp_smiles",
                        "material_type": "payload_smiles",
                        "value": "CCO",
                    },
                    {
                        "material_id": "cmp_name",
                        "material_type": "payload_name",
                        "value": "aspirin",
                    },
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
        compound_context_text="find compound for screening",
    )

    profile = readiness["compound_screening_readiness"]
    assert profile.status == "not_applicable"
    assert profile.allowed_tools == []
    assert profile.blocked_tools == []
    assert profile.ready_tool_count == 0
    assert profile.blocked_tool_count == 0

    profile_entries = readiness["step9_lane_statuses"]
    compound_lanes = [entry for entry in profile_entries if entry.lane_type == "compound_screening"]
    assert len(compound_lanes) == 1
    lane = compound_lanes[0]
    assert lane.status == "not_applicable"
    assert lane.allowed_tools == []
    assert lane.blocked_tools == []
