"""Step 9 runtime resolver (Turn C) unit tests.

`step_09_runtime_execution.resolve_step9_field_value` / `resolve_step9_execution_requests`
are the ONLY places (besides `step_09_input_projection`) allowed to read raw
Step 5/7/8 artifacts, and only to resolve an already-projected field's real
value via its `runtime_lookup` breadcrumb — never to re-derive Step 9 field
semantics. These tests exercise every supported field_ref/runtime_lookup
shape and prove no raw value ever leaks into a redacted audit summary.
"""

from __future__ import annotations

import json

from app.agents.step_09_runtime_execution import (
    resolve_step9_execution_requests,
    resolve_step9_field_value,
)


RAW_SEQ = "EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGK"
RAW_PDB_BODY = "HEADER TEST PDB\nATOM      1  N   GLY A   1"


def _field(**overrides):
    base = {
        "field_ref": "identifier:pdb_id:1N8Z",
        "candidate_id": "cand_a",
        "source_step": "step_05",
        "source_artifact": "candidate_context_table",
        "source_path": "candidate_records[].identifiers[]",
        "field_name": "pdb_id",
        "field_type": "structure_identifier",
        "value_kind": "pdb_id",
        "supports_tool_args": ["pdb_id"],
        "can_resolve_at_runtime": True,
        "llm_safe_metadata": {},
        "runtime_lookup": {},
        "status": "available",
        "missing_reason": None,
    }
    base.update(overrides)
    return base


# ── identifiers embedded directly in field_ref ──────────────────────────────

def test_resolve_identifier_pdb_id_true_four_char():
    value, error = resolve_step9_field_value(
        _field(field_ref="identifier:pdb_id:1N8Z", value_kind="pdb_id"),
        candidate_context_table={},
        prepared_inputs=[],
        step8_result={},
        storage=None,
    )
    assert value == "1N8Z"
    assert error is None


def test_resolve_identifier_pdb_id_rejects_non_four_char_defense_in_depth():
    """The projection layer never emits this shape, but the resolver still
    refuses to treat a non-4-char value as a real PDB id (defense in depth)."""
    value, error = resolve_step9_field_value(
        _field(field_ref="identifier:pdb_id:abcde", value_kind="pdb_id"),
        candidate_context_table={},
        prepared_inputs=[],
        step8_result={},
        storage=None,
    )
    assert value is None
    assert error == "not_a_true_pdb_id"


def test_resolve_identifier_uniprot_id():
    value, error = resolve_step9_field_value(
        _field(field_ref="identifier:uniprot_id:P04626", value_kind="uniprot_id", field_type="identifier"),
        candidate_context_table={},
        prepared_inputs=[],
        step8_result={},
        storage=None,
    )
    assert value == "P04626"
    assert error is None


def test_resolve_identifier_variant_and_chain():
    variant, _ = resolve_step9_field_value(
        _field(field_ref="identifier:variant:V777L", value_kind="variant", field_type="variant"),
        candidate_context_table={}, prepared_inputs=[], step8_result={}, storage=None,
    )
    assert variant == "V777L"
    chain, _ = resolve_step9_field_value(
        _field(field_ref="identifier:chain:A", value_kind="chain_id", field_type="chain"),
        candidate_context_table={}, prepared_inputs=[], step8_result={}, storage=None,
    )
    assert chain == "A"


# ── material: fields (variant / contigs / inline protein sequence) ─────────

def _cct_with_material(candidate_id: str, material_id: str, material_type: str, value: str):
    return {
        "candidate_records": [
            {
                "candidate_id": candidate_id,
                "materials": [
                    {"material_id": material_id, "material_type": material_type, "value": value}
                ],
            }
        ]
    }


def test_resolve_material_variant_value():
    cct = _cct_with_material("cand_a", "mat_variant", "mutation", "V600E")
    value, error = resolve_step9_field_value(
        _field(
            field_ref="material:mat_variant",
            field_type="variant",
            value_kind="mutation",
            runtime_lookup={"candidate_id": "cand_a", "material_id": "mat_variant"},
        ),
        candidate_context_table=cct,
        prepared_inputs=[],
        step8_result={},
        storage=None,
    )
    assert value == "V600E"
    assert error is None


def test_resolve_material_contigs_value():
    cct = _cct_with_material("cand_a", "mat_contigs", "contigs", "A:1-10;B:1-10")
    value, error = resolve_step9_field_value(
        _field(
            field_ref="material:mat_contigs",
            field_type="design_constraint",
            value_kind="contigs",
            runtime_lookup={"candidate_id": "cand_a", "material_id": "mat_contigs"},
        ),
        candidate_context_table=cct,
        prepared_inputs=[],
        step8_result={},
        storage=None,
    )
    assert value == "A:1-10;B:1-10"
    assert error is None


def test_resolve_material_protein_sequence_value():
    cct = _cct_with_material("cand_a", "mat_seq", "target_sequence", RAW_SEQ)
    value, error = resolve_step9_field_value(
        _field(
            field_ref="material:mat_seq",
            field_type="protein_sequence",
            value_kind="sequence_ref",
            runtime_lookup={"candidate_id": "cand_a", "material_id": "mat_seq"},
        ),
        candidate_context_table=cct,
        prepared_inputs=[],
        step8_result={},
        storage=None,
    )
    assert value == RAW_SEQ
    assert error is None


def test_resolve_material_not_found_is_unresolved():
    value, error = resolve_step9_field_value(
        _field(
            field_ref="material:missing",
            field_type="variant",
            value_kind="mutation",
            runtime_lookup={"candidate_id": "cand_a", "material_id": "missing"},
        ),
        candidate_context_table={"candidate_records": []},
        prepared_inputs=[],
        step8_result={},
        storage=None,
    )
    assert value is None
    assert error == "material_not_found"


# ── step7_structure_ref:<structure_input_id>:<index> ────────────────────────

def test_resolve_step7_structure_ref_to_real_path(local_storage):
    key = local_storage.run_key("run1", "uploads", "structure.pdb")
    local_storage.write_bytes(key, b"fake pdb bytes")
    prepared_inputs = [
        {
            "candidate_id": "cand_a",
            "structure_input_id": "si_1",
            "structure_refs": [{"storage_ref": key, "structure_format": "pdb"}],
        }
    ]
    field = _field(
        field_ref="step7_structure_ref:si_1:0",
        field_type="structure",
        value_kind="structure_ref",
        runtime_lookup={"candidate_id": "cand_a", "structure_input_id": "si_1", "index": 0},
    )
    value, error = resolve_step9_field_value(
        field,
        candidate_context_table={},
        prepared_inputs=prepared_inputs,
        step8_result={},
        storage=local_storage,
    )
    assert value == key
    assert error is None


def test_resolve_step7_structure_ref_missing_file_is_unresolved(local_storage):
    prepared_inputs = [
        {
            "candidate_id": "cand_a",
            "structure_input_id": "si_1",
            "structure_refs": [{"storage_ref": "does/not/exist.pdb"}],
        }
    ]
    field = _field(
        field_ref="step7_structure_ref:si_1:0",
        field_type="structure",
        value_kind="structure_ref",
        runtime_lookup={"candidate_id": "cand_a", "structure_input_id": "si_1", "index": 0},
    )
    value, error = resolve_step9_field_value(
        field, candidate_context_table={}, prepared_inputs=prepared_inputs,
        step8_result={}, storage=local_storage,
    )
    assert value is None
    assert error == "structure_ref_storage_ref_invalid"


# ── step8_complex_ref:<candidate_id>:<index> ────────────────────────────────

def test_resolve_step8_complex_ref_via_storage_ref(local_storage):
    key = local_storage.run_key("run1", "predicted", "complex.pdb")
    local_storage.write_bytes(key, b"fake predicted complex")
    step8_result = {
        "candidate_structure_results": [
            {
                "candidate_id": "cand_a",
                "complex_structure_refs": [{"storage_ref": key, "source_kind": "predicted_complex"}],
            }
        ]
    }
    field = _field(
        field_ref="step8_complex_ref:cand_a:0",
        field_type="complex_structure",
        value_kind="complex_structure_ref",
        runtime_lookup={"candidate_id": "cand_a", "index": 0},
    )
    value, error = resolve_step9_field_value(
        field, candidate_context_table={}, prepared_inputs=[],
        step8_result=step8_result, storage=local_storage,
    )
    assert value == key
    assert error is None


def test_resolve_step8_complex_ref_via_source_ref_chain_through_step7(local_storage):
    key = local_storage.run_key("run1", "uploads", "structure.pdb")
    local_storage.write_bytes(key, b"fake pdb bytes")
    step8_result = {
        "candidate_structure_results": [
            {
                "candidate_id": "cand_a",
                "complex_structure_refs": [
                    {"source_kind": "existing_pdb_complex", "source_ref": "mat_struct"}
                ],
            }
        ]
    }
    prepared_inputs = [
        {
            "candidate_id": "cand_a",
            "structure_input_id": "si_1",
            "structure_refs": [{"source_ref": "mat_struct", "storage_ref": key}],
        }
    ]
    field = _field(
        field_ref="step8_complex_ref:cand_a:0",
        field_type="complex_structure",
        value_kind="complex_structure_ref",
        runtime_lookup={"candidate_id": "cand_a", "index": 0},
    )
    value, error = resolve_step9_field_value(
        field, candidate_context_table={}, prepared_inputs=prepared_inputs,
        step8_result=step8_result, storage=local_storage,
    )
    assert value == key
    assert error is None


# ── step8_validated_structure_ref / step8_variant_structure_ref ────────────

def test_resolve_validated_structure_ref_through_step7_source_ref_chain(local_storage):
    key = local_storage.run_key("run1", "uploads", "validated.pdb")
    local_storage.write_bytes(key, b"fake validated pdb")
    step8_result = {
        "candidate_structure_results": [
            {
                "candidate_id": "cand_a",
                "downstream_handoff": {"validated_structure_ref": "mat_struct_id"},
            }
        ]
    }
    prepared_inputs = [
        {
            "candidate_id": "cand_a",
            "structure_input_id": "si_1",
            "structure_refs": [{"source_ref": "mat_struct_id", "storage_ref": key}],
        }
    ]
    field = _field(
        field_ref="step8_validated_structure_ref:cand_a",
        field_type="structure",
        value_kind="validated_structure_ref",
        runtime_lookup={"candidate_id": "cand_a"},
    )
    value, error = resolve_step9_field_value(
        field, candidate_context_table={}, prepared_inputs=prepared_inputs,
        step8_result=step8_result, storage=local_storage,
    )
    assert value == key
    assert error is None


def test_resolve_validated_structure_ref_through_step5_material_id_chain(local_storage):
    key = local_storage.run_key("run1", "uploads", "validated.pdb")
    local_storage.write_bytes(key, b"fake validated pdb")
    step8_result = {
        "candidate_structure_results": [
            {
                "candidate_id": "cand_a",
                "downstream_handoff": {"validated_structure_ref": "mat_id_9f8e"},
            }
        ]
    }
    cct = _cct_with_material("cand_a", "mat_id_9f8e", "structure_ref", key)
    field = _field(
        field_ref="step8_validated_structure_ref:cand_a",
        field_type="structure",
        value_kind="validated_structure_ref",
        runtime_lookup={"candidate_id": "cand_a"},
    )
    value, error = resolve_step9_field_value(
        field, candidate_context_table=cct, prepared_inputs=[],
        step8_result=step8_result, storage=local_storage,
    )
    assert value == key
    assert error is None


def test_resolve_validated_structure_ref_rejects_raw_body_defense_in_depth(local_storage):
    """The projection layer already filters raw bodies out at construction
    time, but the resolver still refuses one if it somehow reaches here."""
    step8_result = {
        "candidate_structure_results": [
            {
                "candidate_id": "cand_a",
                "downstream_handoff": {"validated_structure_ref": RAW_PDB_BODY},
            }
        ]
    }
    field = _field(
        field_ref="step8_validated_structure_ref:cand_a",
        field_type="structure",
        value_kind="validated_structure_ref",
        runtime_lookup={"candidate_id": "cand_a"},
    )
    value, error = resolve_step9_field_value(
        field, candidate_context_table={}, prepared_inputs=[],
        step8_result=step8_result, storage=local_storage,
    )
    assert value is None
    assert error == "structure_hint_is_raw_body_not_path"


def test_resolve_variant_structure_ref_unresolvable_when_no_chain_matches(local_storage):
    step8_result = {
        "candidate_structure_results": [
            {
                "candidate_id": "cand_a",
                "downstream_handoff": {"structure_for_variant_generation_ref": "nowhere"},
            }
        ]
    }
    field = _field(
        field_ref="step8_variant_structure_ref:cand_a",
        field_type="structure",
        value_kind="structure_ref",
        runtime_lookup={"candidate_id": "cand_a"},
    )
    value, error = resolve_step9_field_value(
        field, candidate_context_table={}, prepared_inputs=[],
        step8_result=step8_result, storage=local_storage,
    )
    assert value is None
    assert error == "structure_ref_not_resolvable"


# ── step7_sequence:<sequence_id> (raw sequence via Step5/FASTA fallback) ───

def test_resolve_step7_sequence_inline_falls_back_to_step5_material(local_storage):
    cct = _cct_with_material("cand_a", "mat_seq", "target_sequence", RAW_SEQ)
    prepared_inputs = [
        {
            "candidate_id": "cand_a",
            "sequence_refs_for_prediction": [
                {
                    "sequence_id": "seq_1",
                    "prediction_input_kind": "amino_acid_sequence",
                    "sequence_value_status": "inline",
                    "source_ref": "mat_seq",
                }
            ],
        }
    ]
    field = _field(
        field_ref="step7_sequence:seq_1",
        field_type="protein_sequence",
        value_kind="sequence_ref",
        runtime_lookup={"candidate_id": "cand_a", "sequence_id": "seq_1"},
    )
    value, error = resolve_step9_field_value(
        field, candidate_context_table=cct, prepared_inputs=prepared_inputs,
        step8_result={}, storage=local_storage,
    )
    assert value == RAW_SEQ
    assert error is None


def test_resolve_step7_sequence_fasta_ref_reads_storage(local_storage):
    fasta_key = local_storage.run_key("run1", "uploads", "seq.fasta")
    local_storage.write_bytes(fasta_key, f">chain_a\n{RAW_SEQ}\n".encode("utf-8"))
    prepared_inputs = [
        {
            "candidate_id": "cand_a",
            "sequence_refs_for_prediction": [
                {
                    "sequence_id": "seq_1",
                    "prediction_input_kind": "fasta_ref",
                    "sequence_value_status": "referenced",
                    "sequence_storage_ref": fasta_key,
                }
            ],
        }
    ]
    field = _field(
        field_ref="step7_sequence:seq_1",
        field_type="protein_sequence",
        value_kind="sequence_ref",
        runtime_lookup={"candidate_id": "cand_a", "sequence_id": "seq_1"},
    )
    value, error = resolve_step9_field_value(
        field, candidate_context_table={}, prepared_inputs=prepared_inputs,
        step8_result={}, storage=local_storage,
    )
    assert value == RAW_SEQ
    assert error is None


def test_resolve_step7_sequence_identifier_only_is_unresolvable(local_storage):
    prepared_inputs = [
        {
            "candidate_id": "cand_a",
            "sequence_refs_for_prediction": [
                {
                    "sequence_id": "seq_1",
                    "prediction_input_kind": "uniprot_id",
                    "sequence_value_status": "identifier_only",
                    "source_ref": "P04626",
                }
            ],
        }
    ]
    field = _field(
        field_ref="step7_sequence:seq_1",
        field_type="protein_sequence",
        value_kind="sequence_ref",
        runtime_lookup={"candidate_id": "cand_a", "sequence_id": "seq_1"},
    )
    value, error = resolve_step9_field_value(
        field, candidate_context_table={}, prepared_inputs=prepared_inputs,
        step8_result={}, storage=local_storage,
    )
    assert value is None
    assert error == "identifier_only_sequence_not_runtime_resolvable"


# ── generic unresolved-field guards ──────────────────────────────────────────

def test_field_not_marked_runtime_resolvable_is_rejected():
    value, error = resolve_step9_field_value(
        _field(can_resolve_at_runtime=False),
        candidate_context_table={}, prepared_inputs=[], step8_result={}, storage=None,
    )
    assert value is None
    assert error == "field_not_marked_runtime_resolvable"


def test_field_status_not_available_is_rejected():
    value, error = resolve_step9_field_value(
        _field(status="missing"),
        candidate_context_table={}, prepared_inputs=[], step8_result={}, storage=None,
    )
    assert value is None
    assert error == "field_status_not_available"


def test_unsupported_field_ref_shape_is_rejected():
    value, error = resolve_step9_field_value(
        _field(field_ref="query:summary", field_type="query_context", value_kind="query_summary"),
        candidate_context_table={}, prepared_inputs=[], step8_result={}, storage=None,
    )
    assert value is None
    assert error == "unsupported_field_ref_shape"


# ── resolve_step9_execution_requests: end-to-end contract resolution ───────

def _contract(tool_name, lane_type, kwargs_plan, *, can_build=True):
    return {
        "tool_name": tool_name,
        "lane_type": lane_type,
        "can_build_kwargs": can_build,
        "kwargs_plan": kwargs_plan,
        "unresolved_reasons": [],
    }


def test_execution_requests_resolves_real_kwargs_and_redacts_audit(local_storage):
    cct = _cct_with_material("cand_a", "mat_seq", "target_sequence", RAW_SEQ)
    contracts = [
        _contract(
            "ESM_generate_protein_sequence",
            "protein_design",
            [
                {
                    "runtime_arg": "prompt_sequence",
                    "source": "field_ref",
                    "schema_arg": "prompt_sequence",
                    "field_ref": "material:mat_seq",
                },
                {
                    "runtime_arg": "task",
                    "source": "official_schema_literal",
                    "schema_arg": "task",
                    "literal_value": "generate",
                },
            ],
        )
    ]
    input_fields = [
        _field(
            field_ref="material:mat_seq",
            field_type="protein_sequence",
            value_kind="sequence_ref",
            runtime_lookup={"candidate_id": "cand_a", "material_id": "mat_seq"},
        )
    ]
    requests = resolve_step9_execution_requests(
        kwargs_contracts=contracts,
        input_fields=input_fields,
        candidate_context_table=cct,
        prepared_structure_input_package=None,
        structure_prediction_and_interface_results=None,
        storage=local_storage,
    )
    assert len(requests) == 1
    request = requests[0]
    assert request["can_execute"] is True
    assert request["kwargs"]["prompt_sequence"] == RAW_SEQ
    assert request["kwargs"]["task"] == "generate"
    # Redacted summary carries only length/hash for the sequence, and the
    # literal (safe, fixed-vocabulary) value for the literal arg.
    assert RAW_SEQ not in json.dumps(request["kwargs_redacted_summary"])
    assert request["kwargs_redacted_summary"]["prompt_sequence"]["value_length"] == len(RAW_SEQ)
    assert request["kwargs_redacted_summary"]["task"] == {"source": "literal", "value": "generate"}


def test_execution_requests_unresolved_field_ref_skips_with_reason(local_storage):
    contracts = [
        _contract(
            "AlphaMissense_get_variant_score",
            "variant_evaluation",
            [
                {
                    "runtime_arg": "uniprot_id",
                    "source": "field_ref",
                    "schema_arg": "uniprot_id",
                    "field_ref": "identifier:uniprot_id:P04626",
                },
                {
                    "runtime_arg": "variant",
                    "source": "field_ref",
                    "schema_arg": "variant",
                    "field_ref": "identifier:variant:MISSING",
                },
            ],
        )
    ]
    input_fields = [
        _field(field_ref="identifier:uniprot_id:P04626", field_type="identifier", value_kind="uniprot_id"),
        # Note: no field for "identifier:variant:MISSING" in the projection.
    ]
    requests = resolve_step9_execution_requests(
        kwargs_contracts=contracts,
        input_fields=input_fields,
        candidate_context_table={},
        prepared_structure_input_package=None,
        structure_prediction_and_interface_results=None,
        storage=local_storage,
    )
    request = requests[0]
    assert request["can_execute"] is False
    assert request["kwargs"] is None
    assert request["skip_reason"] == "runtime_value_resolution_failed"
    assert any("field_ref_not_in_projection" in reason for reason in request["unresolved_reasons"])


def test_execution_requests_runtime_plan_unresolved_never_executes():
    contracts = [_contract("NvidiaNIM_rfdiffusion", "protein_design", [], can_build=False)]
    requests = resolve_step9_execution_requests(
        kwargs_contracts=contracts,
        input_fields=[],
        candidate_context_table={},
        prepared_structure_input_package=None,
        structure_prediction_and_interface_results=None,
        storage=None,
    )
    assert requests[0]["can_execute"] is False
    assert requests[0]["skip_reason"] == "runtime_plan_unresolved"


def test_execution_requests_multiple_tools_all_evaluated_independently(local_storage):
    cct = _cct_with_material("cand_a", "mat_seq", "target_sequence", RAW_SEQ)
    contracts = [
        _contract(
            "ESM_generate_protein_sequence",
            "protein_design",
            [{"runtime_arg": "prompt_sequence", "source": "field_ref", "schema_arg": "prompt_sequence", "field_ref": "material:mat_seq"}],
        ),
        _contract("NvidiaNIM_rfdiffusion", "protein_design", [], can_build=False),
    ]
    input_fields = [
        _field(
            field_ref="material:mat_seq",
            field_type="protein_sequence",
            value_kind="sequence_ref",
            runtime_lookup={"candidate_id": "cand_a", "material_id": "mat_seq"},
        )
    ]
    requests = resolve_step9_execution_requests(
        kwargs_contracts=contracts,
        input_fields=input_fields,
        candidate_context_table=cct,
        prepared_structure_input_package=None,
        structure_prediction_and_interface_results=None,
        storage=local_storage,
    )
    by_tool = {r["tool_name"]: r for r in requests}
    assert by_tool["ESM_generate_protein_sequence"]["can_execute"] is True
    assert by_tool["NvidiaNIM_rfdiffusion"]["can_execute"] is False
