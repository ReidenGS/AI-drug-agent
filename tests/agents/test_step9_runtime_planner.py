"""Step 9 runtime planning resolver tests.

The runtime planner consumes ONLY `Step9InputProjection.input_fields`
(passed as `input_fields=`) and Stage 2 mapped tools — never Step 5/7/8 raw
artifacts, and never the legacy `Step9AvailableField` hard-gate shape.
"""

from __future__ import annotations

import json

import pytest

from app.agents.step_09_input_projection import DuplicateStep9InputFieldError
from app.agents.step_09_runtime_planner import plan_step9_runtime_execution
from app.agents.step_09_selection_policy import (
    Step9Stage2ArgumentLiteral,
    Step9Stage2ArgumentMapping,
    Step9Stage2MappedTool,
)


RAW_SEQ = "EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGK"
RAW_PDB = "HEADER TEST PDB\nATOM      1  N   GLY A   1"


def _field(
    field_ref: str,
    *,
    candidate_id: str = "cand_a",
    source_step: str = "step_05",
    field_type: str = "identifier",
    value_kind: str = "uniprot_id",
    supports_tool_args: list[str] | None = None,
    status: str = "available",
    can_resolve_at_runtime: bool = True,
    runtime_lookup: dict | None = None,
) -> dict:
    return {
        "field_ref": field_ref,
        "candidate_id": candidate_id,
        "source_step": source_step,
        "source_artifact": "candidate_context_table",
        "source_path": "candidate_records[].identifiers[]",
        "field_name": "field",
        "field_type": field_type,
        "value_kind": value_kind,
        "supports_tool_args": supports_tool_args if supports_tool_args is not None else ["uniprot_id", "accession", "uniprot_accession"],
        "can_resolve_at_runtime": can_resolve_at_runtime,
        "status": status,
        "runtime_lookup": runtime_lookup or {},
    }


def _mapped(
    tool_name: str,
    lane_type: str,
    mappings: dict[str, str],
    *,
    literals: dict[str, object] | None = None,
    can_invoke: bool = True,
    missing: list[str] | None = None,
) -> Step9Stage2MappedTool:
    return Step9Stage2MappedTool(
        tool_name=tool_name,
        lane_type=lane_type,
        can_invoke=can_invoke,
        argument_mappings=[
            Step9Stage2ArgumentMapping(schema_arg=arg, field_ref=ref)
            for arg, ref in mappings.items()
        ],
        argument_literals=[
            Step9Stage2ArgumentLiteral(schema_arg=arg, literal_value=value)  # type: ignore[arg-type]
            for arg, value in (literals or {}).items()
        ],
        missing_required_fields=missing or [],
    )


def _plan(mapped, fields):
    return plan_step9_runtime_execution(mapped_tools=[mapped], input_fields=fields)


def test_alphamissense_uniprot_and_variant_resolves_planning_only():
    result = _plan(
        _mapped(
            "AlphaMissense_get_variant_score",
            "variant_evaluation",
            {
                "uniprot_id": "identifier:uniprot_id:P04626",
                "variant": "identifier:variant:V777L",
            },
        ),
        [
            _field("identifier:uniprot_id:P04626", value_kind="uniprot_id", supports_tool_args=["uniprot_id", "accession", "uniprot_accession"]),
            _field("identifier:variant:V777L", field_type="variant", value_kind="variant", supports_tool_args=["variant", "variants", "mutation", "mutations"]),
        ],
    )
    entry = result["step9_runtime_execution_plan"][0]
    assert entry["can_resolve"] is True
    assert entry["would_execute"] is False
    assert entry["execution_mode"] == "planning_only"
    assert set(entry["argument_keys"]) == {"uniprot_id", "variant"}
    contract = result["step9_runtime_kwargs_contracts"][0]
    assert contract["can_build_kwargs"] is True
    assert set(contract["kwargs_keys"]) == {"uniprot_id", "variant"}
    for item in contract["kwargs_plan"]:
        assert item["source"] == "field_ref"
        assert item["value_placeholder"] == "<resolved_at_execution_time>"
        assert "literal_value" not in item


def test_alphamissense_missing_variant_unresolved():
    result = _plan(
        _mapped(
            "AlphaMissense_get_variant_score",
            "variant_evaluation",
            {"uniprot_id": "identifier:uniprot_id:P04626"},
            can_invoke=False,
            missing=["variant"],
        ),
        [_field("identifier:uniprot_id:P04626", value_kind="uniprot_id", supports_tool_args=["uniprot_id", "accession", "uniprot_accession"])],
    )
    entry = result["step9_runtime_unresolved_tools"][0]
    assert entry["can_resolve"] is False
    assert "variant" in entry["missing_required_fields"]
    assert "variant_missing" in entry["unresolved_reasons"]


def test_dynamut_true_pdb_chain_mutation_resolves():
    result = _plan(
        _mapped(
            "DynaMut2_predict_stability",
            "variant_evaluation",
            {
                "pdb_id": "identifier:pdb_id:1N8Z",
                "chain": "identifier:chain:A",
                "mutation": "identifier:mutation:V777L",
            },
            literals={"operation": "predict_stability"},
        ),
        [
            _field("identifier:pdb_id:1N8Z", field_type="structure_identifier", value_kind="pdb_id", supports_tool_args=["pdb_id"]),
            _field("identifier:chain:A", field_type="chain", value_kind="chain_id", supports_tool_args=["chain", "chain_id"]),
            _field("identifier:mutation:V777L", field_type="variant", value_kind="mutation", supports_tool_args=["variant", "variants", "mutation", "mutations"]),
        ],
    )
    assert result["step9_runtime_execution_plan"][0]["can_resolve"] is True
    contract = result["step9_runtime_kwargs_contracts"][0]
    assert contract["can_build_kwargs"] is True
    assert set(contract["kwargs_keys"]) == {"operation", "pdb_id", "chain", "mutation"}
    operation = next(item for item in contract["kwargs_plan"] if item["runtime_arg"] == "operation")
    assert operation["source"] == "official_schema_literal"
    assert operation["literal_value"] == "predict_stability"
    placeholders = {
        item["runtime_arg"]: item
        for item in contract["kwargs_plan"]
        if item["source"] == "field_ref"
    }
    assert set(placeholders) == {"pdb_id", "chain", "mutation"}
    assert all(item["value_placeholder"] == "<resolved_at_execution_time>" for item in placeholders.values())


def test_dynamut_step8_true_pdb_id_field_resolves():
    result = _plan(
        _mapped(
            "DynaMut2_predict_stability",
            "variant_evaluation",
            {
                "pdb_id": "identifier:pdb_id:1ABC",
                "chain": "identifier:chain:A",
                "mutation": "identifier:mutation:V777L",
            },
            literals={"operation": "predict_stability"},
        ),
        [
            _field(
                "identifier:pdb_id:1ABC",
                source_step="step_08",
                field_type="structure_identifier",
                value_kind="pdb_id",
                supports_tool_args=["pdb_id"],
            ),
            _field("identifier:chain:A", field_type="chain", value_kind="chain_id", supports_tool_args=["chain", "chain_id"]),
            _field("identifier:mutation:V777L", field_type="variant", value_kind="mutation", supports_tool_args=["variant", "variants", "mutation", "mutations"]),
        ],
    )
    entry = result["step9_runtime_resolved_tools"][0]
    assert entry["can_resolve"] is True
    contract = result["step9_runtime_kwargs_contracts"][0]
    pdb_item = next(item for item in contract["kwargs_plan"] if item["runtime_arg"] == "pdb_id")
    assert pdb_item["field_ref"] == "identifier:pdb_id:1ABC"
    assert pdb_item["source_metadata"]["source_step"] == "step_08"


def test_dynamut_step8_complex_ref_without_true_pdb_id_unresolved():
    result = _plan(
        _mapped(
            "DynaMut2_predict_stability",
            "variant_evaluation",
            {
                "pdb_id": "step8_complex_ref:cand_a:0",
                "chain": "identifier:chain:A",
                "mutation": "identifier:mutation:V777L",
            },
        ),
        [
            _field(
                "step8_complex_ref:cand_a:0",
                source_step="step_08",
                field_type="complex_structure",
                value_kind="complex_structure_ref",
                supports_tool_args=["input_pdb", "pdb_file", "structure", "complex_structure", "backbone"],
            ),
            _field("identifier:chain:A", field_type="chain", value_kind="chain_id", supports_tool_args=["chain", "chain_id"]),
            _field("identifier:mutation:V777L", field_type="variant", value_kind="mutation", supports_tool_args=["variant", "variants", "mutation", "mutations"]),
        ],
    )
    entry = result["step9_runtime_unresolved_tools"][0]
    assert "true_pdb_id_field_ref_required" in entry["unresolved_reasons"]


def test_rfdiffusion_missing_contigs_unresolved():
    result = _plan(
        _mapped(
            "NvidiaNIM_rfdiffusion",
            "protein_design",
            {"input_pdb": "step8_complex_ref:cand_a:0"},
            can_invoke=False,
            missing=["contigs"],
        ),
        [
            _field(
                "step8_complex_ref:cand_a:0",
                source_step="step_08",
                field_type="complex_structure",
                value_kind="complex_structure_ref",
                supports_tool_args=["input_pdb", "pdb_file", "structure", "complex_structure", "backbone"],
            )
        ],
    )
    entry = result["step9_runtime_unresolved_tools"][0]
    assert "contigs_missing_or_not_validated" in entry["unresolved_reasons"]
    contract = result["step9_runtime_kwargs_contracts"][0]
    assert contract["can_build_kwargs"] is False
    assert "contigs" in contract["kwargs_keys"]
    assert any(item["runtime_arg"] == "contigs" and item["source"] == "unresolved" for item in contract["kwargs_plan"])


def test_proteinmpnn_true_complex_resolves_planning_only():
    result = _plan(
        _mapped(
            "NvidiaNIM_proteinmpnn",
            "protein_design",
            {"input_pdb": "step8_complex_ref:cand_a:0"},
        ),
        [
            _field(
                "step8_complex_ref:cand_a:0",
                source_step="step_08",
                field_type="complex_structure",
                value_kind="complex_structure_ref",
                supports_tool_args=["input_pdb", "pdb_file", "structure", "complex_structure", "backbone"],
            )
        ],
    )
    entry = result["step9_runtime_resolved_tools"][0]
    assert entry["tool_name"] == "NvidiaNIM_proteinmpnn"
    assert entry["would_execute"] is False
    contract = result["step9_runtime_kwargs_contracts"][0]
    assert contract["can_build_kwargs"] is True
    assert contract["kwargs_keys"] == ["input_pdb"]
    kwargs_item = contract["kwargs_plan"][0]
    assert kwargs_item["runtime_arg"] == "input_pdb"
    assert kwargs_item["field_ref"] == "step8_complex_ref:cand_a:0"
    assert kwargs_item["value_placeholder"] == "<resolved_at_execution_time>"


def test_proteinmpnn_validated_backbone_resolves_planning_only():
    result = _plan(
        _mapped(
            "NvidiaNIM_proteinmpnn",
            "protein_design",
            {"input_pdb": "step8_validated_structure_ref:cand_a"},
        ),
        [
            _field(
                "step8_validated_structure_ref:cand_a",
                source_step="step_08",
                field_type="structure",
                value_kind="validated_structure_ref",
                supports_tool_args=["input_pdb", "pdb_file", "structure", "backbone", "path"],
                runtime_lookup={
                    "resolution_path": [
                        "step_08.candidate_structure_results[].downstream_handoff.validated_structure_ref",
                        "step_07.prepared_structure_inputs[].structure_refs[].storage_ref",
                        "step_05.candidate_records[].materials[]",
                    ],
                    "candidate_id": "cand_a",
                },
            )
        ],
    )
    entry = result["step9_runtime_resolved_tools"][0]
    assert entry["tool_name"] == "NvidiaNIM_proteinmpnn"
    assert entry["can_resolve"] is True
    contract = result["step9_runtime_kwargs_contracts"][0]
    assert contract["can_build_kwargs"] is True
    assert contract["kwargs_plan"][0]["field_ref"] == "step8_validated_structure_ref:cand_a"
    # runtime_lookup carries the resolution chain, never a raw value.
    assert contract["kwargs_plan"][0]["source_metadata"]["runtime_lookup"]["candidate_id"] == "cand_a"


def test_esm_generate_does_not_use_raw_query_as_sequence():
    result = _plan(
        _mapped(
            "ESM_generate_protein_sequence",
            "protein_design",
            {"prompt_sequence": "raw_query"},
        ),
        [],
    )
    entry = result["step9_runtime_unresolved_tools"][0]
    assert "field_ref_not_available" in entry["unresolved_reasons"]
    assert result["step9_runtime_kwargs_contracts"][0]["can_build_kwargs"] is False
    assert RAW_SEQ not in json.dumps(result)


def test_compound_and_zinc_mapped_tools_ignored():
    result = plan_step9_runtime_execution(
        mapped_tools=[
            _mapped(
                "ZINC_search_by_smiles",
                "compound_screening",
                {"smiles": "material:smiles"},
            )
        ],
        input_fields=[
            _field("material:smiles", field_type="compound", value_kind="smiles", supports_tool_args=["smiles"])
        ],
    )
    blob = json.dumps(result)
    assert result["step9_runtime_execution_plan"] == []
    assert result["step9_runtime_kwargs_contracts"] == []
    assert result["step9_runtime_kwargs_contract_audit"] == []
    assert "ZINC" not in blob
    assert "ChEMBL" not in blob


def test_runtime_planner_result_has_no_raw_sequence_pdb_fasta_a3m():
    result = _plan(
        _mapped(
            "NvidiaNIM_proteinmpnn",
            "protein_design",
            {"input_pdb": "step8_complex_ref:cand_a:0"},
        ),
        [
            _field(
                "step8_complex_ref:cand_a:0",
                source_step="step_08",
                field_type="complex_structure",
                value_kind="complex_structure_ref",
                supports_tool_args=["input_pdb", "pdb_file", "structure", "complex_structure", "backbone"],
            ),
            _field(
                "sequence_ref:cand_a",
                source_step="step_07",
                field_type="protein_sequence",
                value_kind="sequence_ref",
                supports_tool_args=["sequence", "prompt_sequence"],
            ),
        ],
    )
    blob = json.dumps(result)
    assert RAW_SEQ not in blob
    assert "HEADER TEST PDB" not in blob
    assert "ATOM      1" not in blob
    assert "FASTA" not in blob.upper()
    assert "A3M" not in blob.upper()
    assert "sk-" not in blob.lower()


def test_field_not_runtime_resolvable_stays_unresolved():
    result = _plan(
        _mapped(
            "ESM_generate_protein_sequence",
            "protein_design",
            {"prompt_sequence": "material:seq_missing"},
        ),
        [
            _field(
                "material:seq_missing",
                field_type="protein_sequence",
                value_kind="sequence_ref",
                supports_tool_args=["sequence", "prompt_sequence"],
                can_resolve_at_runtime=False,
                status="missing",
            )
        ],
    )
    entry = result["step9_runtime_unresolved_tools"][0]
    assert "field_not_runtime_resolvable" in entry["unresolved_reasons"]


def test_kwargs_contract_audit_marks_no_candidate_value_persisted():
    result = _plan(
        _mapped(
            "AlphaMissense_get_variant_score",
            "variant_evaluation",
            {
                "uniprot_id": "identifier:uniprot_id:P04626",
                "variant": "identifier:variant:V777L",
            },
        ),
        [
            _field("identifier:uniprot_id:P04626", value_kind="uniprot_id", supports_tool_args=["uniprot_id", "accession", "uniprot_accession"]),
            _field("identifier:variant:V777L", field_type="variant", value_kind="variant", supports_tool_args=["variant", "variants", "mutation", "mutations"]),
        ],
    )
    audit = result["step9_runtime_kwargs_contract_audit"]
    assert audit
    assert {entry["runtime_arg"] for entry in audit} == {"uniprot_id", "variant"}
    assert all(entry["candidate_value_persisted"] is False for entry in audit)


def test_duplicate_field_ref_in_input_fields_raises_not_silent_overwrite():
    """A buggy/non-canonical `input_fields` list with two entries sharing the
    same field_ref must be rejected loudly, not resolved by whichever entry
    the dict-comprehension lookup happens to keep last."""
    duplicate_fields = [
        _field(
            "identifier:uniprot_id:P04626",
            candidate_id="cand_a",
            value_kind="uniprot_id",
            supports_tool_args=["uniprot_id", "accession", "uniprot_accession"],
        ),
        _field(
            "identifier:uniprot_id:P04626",
            candidate_id="cand_b",
            value_kind="uniprot_id",
            supports_tool_args=["uniprot_id", "accession", "uniprot_accession"],
        ),
    ]
    with pytest.raises(DuplicateStep9InputFieldError, match="identifier:uniprot_id:P04626"):
        _plan(
            _mapped(
                "AlphaMissense_get_variant_score",
                "variant_evaluation",
                {"uniprot_id": "identifier:uniprot_id:P04626", "variant": "identifier:variant:V777L"},
            ),
            duplicate_fields
            + [_field("identifier:variant:V777L", field_type="variant", value_kind="variant", supports_tool_args=["variant", "variants", "mutation", "mutations"])],
        )
