"""Step 9 dry-run runtime resolver tests."""

from __future__ import annotations

import json

from app.agents.step_09_runtime_resolver import build_step9_dry_run_execution_plan
from app.schemas.step_09_structure_variant_and_compound_screening import Step9AvailableField


RAW_SEQ = "EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGK"
RAW_PDB = "HEADER TEST\nATOM      1  N   GLY A   1"


def _field(field_ref: str, *, field_type: str, value_kind: str, provider: str = "step_05"):
    return Step9AvailableField(
        candidate_id="cand_a",
        field_ref=field_ref,
        provider=provider,  # type: ignore[arg-type]
        field_type=field_type,
        value_kind=value_kind,
        source_ref="1N8Z" if provider == "step_08" else None,
    )


def _tool(tool_name: str, lane_type: str, mappings, *, literals=None, missing=None, can_invoke=True):
    return {
        "tool_name": tool_name,
        "lane_type": lane_type,
        "can_invoke": can_invoke,
        "argument_mappings": mappings,
        "argument_literals": literals or [],
        "missing_required_fields": missing or [],
        "skip_reason": "" if not missing and can_invoke else "missing_required_fields",
    }


def test_alphamissense_uniprot_variant_would_execute_no_mcp_call_shape():
    result = build_step9_dry_run_execution_plan(
        mapped_tools=[
            _tool(
                "AlphaMissense_get_variant_score",
                "variant_evaluation",
                [
                    {"schema_arg": "uniprot_id", "field_ref": "identifier:uniprot_id:P04626"},
                    {"schema_arg": "variant", "field_ref": "identifier:variant:V777L"},
                ],
            )
        ],
        available_fields=[
            _field("identifier:uniprot_id:P04626", field_type="identifier", value_kind="uniprot_id"),
            _field("identifier:variant:V777L", field_type="variant", value_kind="variant"),
        ],
    )
    record = result.execution_plan[0]
    assert record["tool_name"] == "AlphaMissense_get_variant_score"
    assert record["would_execute"] is True
    assert result.resolved_tools == ["AlphaMissense_get_variant_score"]
    assert "raw_value" not in json.dumps(record)


def test_dynamut_missing_pdb_id_remains_unresolved():
    result = build_step9_dry_run_execution_plan(
        mapped_tools=[
            _tool(
                "DynaMut2_predict_stability",
                "variant_evaluation",
                [
                    {"schema_arg": "chain", "field_ref": "identifier:chain:A"},
                    {"schema_arg": "mutation", "field_ref": "identifier:mutation:V777L"},
                ],
                missing=["pdb_id"],
                can_invoke=False,
            )
        ],
        available_fields=[
            _field("identifier:chain:A", field_type="chain", value_kind="chain_id"),
            _field("identifier:mutation:V777L", field_type="variant", value_kind="mutation"),
        ],
    )
    record = result.execution_plan[0]
    assert record["would_execute"] is False
    assert record["missing_required_fields"] == ["pdb_id"]
    assert result.unresolved_tools == ["DynaMut2_predict_stability"]


def test_rfdiffusion_without_contigs_remains_unresolved():
    result = build_step9_dry_run_execution_plan(
        mapped_tools=[
            _tool(
                "NvidiaNIM_rfdiffusion",
                "protein_design",
                [{"schema_arg": "input_pdb", "field_ref": "step8_complex_ref:cand_a:0"}],
                missing=["contigs"],
                can_invoke=False,
            )
        ],
        available_fields=[
            _field(
                "step8_complex_ref:cand_a:0",
                field_type="structure",
                value_kind="complex_structure_ref",
                provider="step_08",
            )
        ],
    )
    record = result.execution_plan[0]
    assert record["would_execute"] is False
    assert record["missing_required_fields"] == ["contigs"]


def test_proteinmpnn_step8_complex_ref_resolves_dry_run_only():
    result = build_step9_dry_run_execution_plan(
        mapped_tools=[
            _tool(
                "NvidiaNIM_proteinmpnn",
                "protein_design",
                [{"schema_arg": "input_pdb", "field_ref": "step8_complex_ref:cand_a:0"}],
            )
        ],
        available_fields=[
            _field(
                "step8_complex_ref:cand_a:0",
                field_type="structure",
                value_kind="complex_structure_ref",
                provider="step_08",
            )
        ],
    )
    record = result.execution_plan[0]
    assert record["would_execute"] is True
    assert record["execution_mode"] == "dry_run_only"
    assert record["argument_plan"][0]["candidate_value_persisted"] is False


def test_compound_tool_appears_in_dry_run_audit():
    result = build_step9_dry_run_execution_plan(
        mapped_tools=[
            _tool(
                "ZINC_search_by_smiles",
                "compound_screening",
                [{"schema_arg": "smiles", "field_ref": "material:smiles"}],
                literals=[{"schema_arg": "operation", "literal_value": "search_by_smiles"}],
            )
        ],
        available_fields=[
            _field("material:smiles", field_type="compound", value_kind="smiles"),
        ],
    )
    assert result.execution_plan[0]["tool_name"] == "ZINC_search_by_smiles"
    assert result.execution_plan[0]["would_execute"] is True
    assert result.execution_mode == "dry_run_only"


def test_no_raw_sequence_pdb_a3m_fasta_or_api_key_in_dry_run_output():
    result = build_step9_dry_run_execution_plan(
        mapped_tools=[
            _tool(
                "ESM_generate_protein_sequence",
                "protein_design",
                [{"schema_arg": "prompt_sequence", "field_ref": "material:seq"}],
            )
        ],
        available_fields=[
            _field("material:seq", field_type="protein_sequence", value_kind="sequence_material"),
        ],
    )
    blob = json.dumps(result.model_dump())
    for forbidden in (RAW_SEQ, RAW_PDB, "A3M", "FASTA", "sk-secretvalue123"):
        assert forbidden not in blob
