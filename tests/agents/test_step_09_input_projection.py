"""Step9InputProjection tests.

`step_09_input_projection.project_step9_inputs` is the ONLY module that
understands raw Step 5 / Step 7 / Step 8 / query artifact shapes for Step 9.
These tests prove the projection rules from the workflow architecture spec
and the privacy invariant: no raw sequence / PDB body / storage path / A3M /
API key ever appears in the projection output.
"""

from __future__ import annotations

import json

import pytest

from app.agents.step_09_input_projection import (
    DuplicateStep9InputFieldError,
    assert_unique_input_field_refs,
    project_step9_inputs,
)


RAW_SEQ = "EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGK"
RAW_PDB_BODY = "HEADER TEST PDB\nATOM      1  N   GLY A   1"
STORAGE_PATH = "s3://bucket/runs/abc/prepared/validation.pdb"


def _cct(candidate: dict) -> dict:
    return {"candidate_records": [candidate]}


def _base_candidate(**overrides) -> dict:
    candidate = {
        "candidate_id": "cand_t1",
        "candidate_type": "target_antigen",
        "materials": [],
        "identifiers": [],
    }
    candidate.update(overrides)
    return candidate


def test_uniprot_identifier_creates_uniprot_field_not_raw_sequence_field():
    candidate = _base_candidate(identifiers=[{"id_type": "uniprot_id", "id_value": "P04626"}])
    projection = project_step9_inputs(
        candidate_context_table=_cct(candidate),
        prepared_structure_input_package=None,
        structure_prediction_and_interface_results=None,
    )
    fields = [f for f in projection["input_fields"] if f.candidate_id == "cand_t1"]
    assert len(fields) == 1
    field = fields[0]
    assert field.field_type == "identifier"
    assert field.value_kind == "uniprot_id"
    assert set(field.supports_tool_args) == {"uniprot_id", "accession", "uniprot_accession"}
    assert "sequence" not in field.supports_tool_args
    assert "prompt_sequence" not in field.supports_tool_args


def test_explicit_mutation_creates_variant_field():
    candidate = _base_candidate(identifiers=[{"id_type": "mutation", "id_value": "p.V600E"}])
    projection = project_step9_inputs(
        candidate_context_table=_cct(candidate),
        prepared_structure_input_package=None,
        structure_prediction_and_interface_results=None,
    )
    fields = [f for f in projection["input_fields"] if f.candidate_id == "cand_t1"]
    assert len(fields) == 1
    assert fields[0].field_type == "variant"
    assert fields[0].value_kind == "mutation"
    assert set(fields[0].supports_tool_args) == {"variant", "variants", "mutation", "mutations"}


def test_uniprot_and_variant_identifiers_project_both_fields():
    """The Step 2/5 protein-variant chain: a target candidate carrying both a
    UniProt accession and a `variant` identifier projects both an
    `identifier:uniprot_id:*` field and an `identifier:variant:*` field, so
    Step 9's AlphaMissense / DynaMut2 / ESM variant tools can be satisfied."""
    candidate = _base_candidate(
        identifiers=[
            {"id_type": "uniprot_id", "id_value": "P04626"},
            {"id_type": "variant", "id_value": "V777L"},
        ]
    )
    projection = project_step9_inputs(
        candidate_context_table=_cct(candidate),
        prepared_structure_input_package=None,
        structure_prediction_and_interface_results=None,
    )
    by_ref = {f.field_ref: f for f in projection["input_fields"] if f.candidate_id == "cand_t1"}
    assert "identifier:uniprot_id:P04626" in by_ref
    assert "identifier:variant:V777L" in by_ref
    uniprot = by_ref["identifier:uniprot_id:P04626"]
    variant = by_ref["identifier:variant:V777L"]
    assert uniprot.value_kind == "uniprot_id"
    assert "uniprot_id" in uniprot.supports_tool_args
    assert variant.field_type == "variant"
    assert variant.value_kind == "variant"
    assert set(variant.supports_tool_args) == {"variant", "variants", "mutation", "mutations"}


def test_chain_identifier_creates_chain_field():
    candidate = _base_candidate(identifiers=[{"id_type": "chain", "id_value": "A"}])
    projection = project_step9_inputs(
        candidate_context_table=_cct(candidate),
        prepared_structure_input_package=None,
        structure_prediction_and_interface_results=None,
    )
    fields = [f for f in projection["input_fields"] if f.candidate_id == "cand_t1"]
    assert fields[0].field_type == "chain"
    assert fields[0].value_kind == "chain_id"
    assert set(fields[0].supports_tool_args) == {"chain", "chain_id"}


def test_contigs_material_creates_design_constraint_field():
    candidate = _base_candidate(
        materials=[{"material_id": "mat_contigs", "material_type": "contigs", "value": "A:1-10;B:1-10"}]
    )
    projection = project_step9_inputs(
        candidate_context_table=_cct(candidate),
        prepared_structure_input_package=None,
        structure_prediction_and_interface_results=None,
    )
    fields = [f for f in projection["input_fields"] if f.candidate_id == "cand_t1"]
    assert fields[0].field_type == "design_constraint"
    assert fields[0].value_kind == "contigs"
    assert fields[0].supports_tool_args == ["contigs"]


def test_real_pdb_id_creates_pdb_id_field_uploaded_material_id_does_not():
    candidate = _base_candidate(identifiers=[{"id_type": "pdb_id", "id_value": "1N8Z"}])
    projection = project_step9_inputs(
        candidate_context_table=_cct(candidate),
        prepared_structure_input_package=None,
        structure_prediction_and_interface_results=None,
    )
    fields = [f for f in projection["input_fields"] if f.candidate_id == "cand_t1"]
    assert fields[0].field_type == "structure_identifier"
    assert fields[0].value_kind == "pdb_id"
    assert fields[0].field_ref == "identifier:pdb_id:1N8Z"

    step8 = {
        "candidate_structure_results": [
            {
                "candidate_id": "cand_t1",
                "downstream_handoff": {"validated_structure_ref": "mat_abc123"},
            }
        ]
    }
    projection2 = project_step9_inputs(
        candidate_context_table=_cct(_base_candidate()),
        prepared_structure_input_package=None,
        structure_prediction_and_interface_results=step8,
    )
    assert not any(f.value_kind == "pdb_id" for f in projection2["input_fields"] if f.candidate_id == "cand_t1")


def test_validated_structure_ref_produces_backbone_field_supporting_input_pdb_without_leaking_path():
    step8 = {
        "candidate_structure_results": [
            {
                "candidate_id": "cand_t1",
                "downstream_handoff": {"validated_structure_ref": STORAGE_PATH},
            }
        ]
    }
    projection = project_step9_inputs(
        candidate_context_table=_cct(_base_candidate()),
        prepared_structure_input_package=None,
        structure_prediction_and_interface_results=step8,
    )
    fields = [f for f in projection["input_fields"] if f.field_ref == "step8_validated_structure_ref:cand_t1"]
    assert len(fields) == 1
    field = fields[0]
    assert "input_pdb" in field.supports_tool_args
    assert field.can_resolve_at_runtime is True
    # The raw storage path / material id must never appear anywhere in the field.
    blob = json.dumps(field.model_dump())
    assert STORAGE_PATH not in blob
    # runtime_lookup preserves HOW to resolve it later, not the value.
    assert field.runtime_lookup["candidate_id"] == "cand_t1"
    assert "resolution_path" in field.runtime_lookup


def test_validated_structure_ref_material_id_does_not_leak_but_has_runtime_lookup():
    step8 = {
        "candidate_structure_results": [
            {
                "candidate_id": "cand_t1",
                "downstream_handoff": {"validated_structure_ref": "material_id_9f8e7d"},
            }
        ]
    }
    projection = project_step9_inputs(
        candidate_context_table=_cct(_base_candidate()),
        prepared_structure_input_package=None,
        structure_prediction_and_interface_results=step8,
    )
    field = next(f for f in projection["input_fields"] if f.field_ref == "step8_validated_structure_ref:cand_t1")
    blob = json.dumps(field.model_dump())
    assert "material_id_9f8e7d" not in blob
    assert field.runtime_lookup


def test_raw_pdb_body_is_never_projected_as_validated_structure_field():
    step8 = {
        "candidate_structure_results": [
            {
                "candidate_id": "cand_t1",
                "downstream_handoff": {"validated_structure_ref": RAW_PDB_BODY},
            }
        ]
    }
    projection = project_step9_inputs(
        candidate_context_table=_cct(_base_candidate()),
        prepared_structure_input_package=None,
        structure_prediction_and_interface_results=step8,
    )
    assert not any(f.field_ref == "step8_validated_structure_ref:cand_t1" for f in projection["input_fields"])
    blob = json.dumps([f.model_dump() for f in projection["input_fields"]])
    assert "HEADER TEST PDB" not in blob
    assert "ATOM      1" not in blob


def test_step7_structure_ref_with_storage_ref_supports_input_pdb():
    prepared = [
        {
            "candidate_id": "cand_t1",
            "structure_input_id": "si_1",
            "structure_refs": [
                {"storage_ref": STORAGE_PATH, "structure_format": "pdb", "source_kind": "uploaded_file"}
            ],
            "sequence_refs_for_prediction": [],
        }
    ]
    projection = project_step9_inputs(
        candidate_context_table=_cct(_base_candidate()),
        prepared_structure_input_package=prepared,
        structure_prediction_and_interface_results=None,
    )
    fields = [f for f in projection["input_fields"] if f.candidate_id == "cand_t1"]
    assert fields[0].field_type == "structure"
    assert "input_pdb" in fields[0].supports_tool_args
    blob = json.dumps(fields[0].model_dump())
    assert STORAGE_PATH not in blob


def test_step7_identifier_only_uniprot_sequence_ref_supports_uniprot_not_raw_sequence():
    prepared = [
        {
            "candidate_id": "cand_t1",
            "structure_input_id": "si_1",
            "structure_refs": [],
            "sequence_refs_for_prediction": [
                {
                    "sequence_id": "seq_1",
                    "sequence_value_status": "identifier_only",
                    "prediction_input_kind": "uniprot_id",
                    "source_ref": "P04626",
                }
            ],
        }
    ]
    projection = project_step9_inputs(
        candidate_context_table=_cct(_base_candidate()),
        prepared_structure_input_package=prepared,
        structure_prediction_and_interface_results=None,
    )
    fields = [f for f in projection["input_fields"] if f.candidate_id == "cand_t1"]
    assert fields[0].value_kind == "uniprot_id"
    assert "sequence" not in fields[0].supports_tool_args


def test_step7_inline_sequence_supports_sequence_only_not_prompt_sequence():
    """An ordinary complete protein sequence (e.g. a heavy/light chain) must
    only satisfy a plain `sequence` arg. ToolUniverse's
    ESM_generate_protein_sequence `prompt_sequence` expects a masked
    generation prompt, not a complete existing chain — mapping an ordinary
    sequence there causes a real ESM SDK error, not just an LLM mistake."""
    prepared = [
        {
            "candidate_id": "cand_t1",
            "structure_input_id": "si_1",
            "structure_refs": [],
            "sequence_refs_for_prediction": [
                {
                    "sequence_id": "seq_1",
                    "sequence_value_status": "inline",
                    "prediction_input_kind": "amino_acid_sequence",
                    "sequence_length": len(RAW_SEQ),
                }
            ],
        }
    ]
    projection = project_step9_inputs(
        candidate_context_table=_cct(_base_candidate()),
        prepared_structure_input_package=prepared,
        structure_prediction_and_interface_results=None,
    )
    fields = [f for f in projection["input_fields"] if f.candidate_id == "cand_t1"]
    assert set(fields[0].supports_tool_args) == {"sequence"}
    assert "prompt_sequence" not in fields[0].supports_tool_args
    blob = json.dumps(fields[0].model_dump())
    assert RAW_SEQ not in blob


def test_step5_material_heavy_light_target_sequence_supports_sequence_only():
    """Same contract for Step 5 candidate-material-sourced sequences
    (heavy/light chain, target antigen sequence)."""
    candidate = _base_candidate(
        materials=[
            {
                "material_id": "mat_heavy",
                "material_type": "antibody_heavy_chain_sequence",
                "value": RAW_SEQ,
            }
        ]
    )
    projection = project_step9_inputs(
        candidate_context_table=_cct(candidate),
        prepared_structure_input_package=None,
        structure_prediction_and_interface_results=None,
    )
    fields = [f for f in projection["input_fields"] if f.field_type == "protein_sequence"]
    assert len(fields) == 1
    assert set(fields[0].supports_tool_args) == {"sequence"}
    assert "prompt_sequence" not in fields[0].supports_tool_args


def test_step5_prompt_sequence_material_projects_masked_prompt_field():
    """A Step 5 `prompt_sequence` material (explicit masked generation prompt,
    value = storage ref, compact `content_descriptor`) is the ONLY Step 5
    material projected as a `masked_prompt_sequence` field that supports the
    `prompt_sequence` arg. The raw masked prompt is never in the material value
    (it's a storage ref) so nothing raw reaches the projection."""
    candidate = _base_candidate(
        materials=[
            {
                "material_id": "mat_prompt",
                "material_type": "prompt_sequence",
                "value": "adc_pilot/runs/r1/inputs/prompt_sequences/mat_prompt.txt",
                "value_format": "masked_amino_acid_sequence",
                "role": "protein_generation_prompt",
                "role_status": "explicit",
                "content_descriptor": {
                    "has_mask": True,
                    "mask_token_style": "underscore",
                    "prompt_length": 33,
                    "sha256_prefix": "deadbeef0001",
                    "source_kind": "inline",
                    "value_is_storage_ref": True,
                },
            }
        ]
    )
    projection = project_step9_inputs(
        candidate_context_table=_cct(candidate),
        prepared_structure_input_package=None,
        structure_prediction_and_interface_results=None,
    )
    fields = [f for f in projection["input_fields"] if f.candidate_id == "cand_t1"]
    assert len(fields) == 1
    field = fields[0]
    assert field.field_type == "protein_sequence"
    assert field.value_kind == "masked_prompt_sequence"
    assert field.supports_tool_args == ["prompt_sequence"]
    assert "sequence" not in field.supports_tool_args
    assert field.semantic_role == "protein_generation_prompt"
    assert field.can_resolve_at_runtime is True
    # compact fingerprint surfaced; raw prompt never present.
    assert field.llm_safe_metadata["has_mask"] is True
    assert field.llm_safe_metadata["sha256_prefix"] == "deadbeef0001"
    assert field.runtime_lookup["material_id"] == "mat_prompt"


def test_ordinary_sequence_and_masked_prompt_dont_cross_contaminate():
    """When both an ordinary heavy-chain sequence AND an explicit masked
    prompt are present, only the masked prompt supports `prompt_sequence`; the
    ordinary chain supports `sequence` only."""
    candidate = _base_candidate(
        candidate_type="antibody",
        materials=[
            {
                "material_id": "mat_heavy",
                "material_type": "antibody_heavy_chain_sequence",
                "value": RAW_SEQ,
            },
            {
                "material_id": "mat_prompt",
                "material_type": "prompt_sequence",
                "value": "adc_pilot/runs/r1/inputs/prompt_sequences/mat_prompt.txt",
                "content_descriptor": {"has_mask": True, "prompt_length": 20},
            },
        ],
    )
    projection = project_step9_inputs(
        candidate_context_table=_cct(candidate),
        prepared_structure_input_package=None,
        structure_prediction_and_interface_results=None,
    )
    by_material = {
        f.field_ref: f
        for f in projection["input_fields"]
        if f.field_ref.startswith("material:")
    }
    heavy = by_material["material:mat_heavy"]
    prompt = by_material["material:mat_prompt"]
    assert set(heavy.supports_tool_args) == {"sequence"}
    assert prompt.supports_tool_args == ["prompt_sequence"]
    assert prompt.value_kind == "masked_prompt_sequence"


def test_step8_complex_structure_ref_supports_complex_and_backbone_args():
    step8 = {
        "candidate_structure_results": [
            {
                "candidate_id": "cand_t1",
                "complex_structure_refs": [
                    {"source_kind": "existing_pdb_complex", "pdb_id": "1ABC", "source_ref": "1ABC"}
                ],
            }
        ]
    }
    projection = project_step9_inputs(
        candidate_context_table=_cct(_base_candidate()),
        prepared_structure_input_package=None,
        structure_prediction_and_interface_results=step8,
    )
    complex_fields = [f for f in projection["input_fields"] if f.field_type == "complex_structure"]
    assert len(complex_fields) == 1
    assert set(complex_fields[0].supports_tool_args) >= {"input_pdb", "structure", "complex_structure", "backbone"}
    pdb_fields = [f for f in projection["input_fields"] if f.field_type == "structure_identifier"]
    assert len(pdb_fields) == 1
    assert pdb_fields[0].field_ref == "identifier:pdb_id:1ABC"


def test_predicted_complex_without_true_pdb_id_creates_no_pdb_id_field():
    step8 = {
        "candidate_structure_results": [
            {
                "candidate_id": "cand_t1",
                "complex_structure_refs": [
                    {"source_kind": "predicted_complex", "storage_ref": STORAGE_PATH}
                ],
            }
        ]
    }
    projection = project_step9_inputs(
        candidate_context_table=_cct(_base_candidate()),
        prepared_structure_input_package=None,
        structure_prediction_and_interface_results=step8,
    )
    assert not any(f.field_type == "structure_identifier" for f in projection["input_fields"])
    complex_fields = [f for f in projection["input_fields"] if f.field_type == "complex_structure"]
    assert complex_fields
    blob = json.dumps(complex_fields[0].model_dump())
    assert STORAGE_PATH not in blob


def test_compound_candidate_produces_no_step9_input_fields():
    candidate = {
        "candidate_id": "cand_c1",
        "candidate_type": "compound_component",
        "materials": [{"material_id": "cmp_1", "material_type": "payload_smiles", "value": "CCO"}],
        "identifiers": [],
    }
    projection = project_step9_inputs(
        candidate_context_table=_cct(candidate),
        prepared_structure_input_package=None,
        structure_prediction_and_interface_results=None,
    )
    assert not any(f.candidate_id == "cand_c1" for f in projection["input_fields"])
    assert not any(s["candidate_id"] == "cand_c1" for s in projection["candidate_summaries"])


def test_query_summary_and_field_are_redacted():
    structured_query = {"canonical_query": f"design with {RAW_SEQ} and sk-secretvalue123"}
    raw_request = {"raw_user_query": f"{RAW_PDB_BODY}\n>seq\n{RAW_SEQ}\nA3M"}
    projection = project_step9_inputs(
        candidate_context_table=_cct(_base_candidate()),
        prepared_structure_input_package=None,
        structure_prediction_and_interface_results=None,
        structured_query=structured_query,
        raw_request=raw_request,
    )
    query_field = next(f for f in projection["input_fields"] if f.source_step == "query")
    assert query_field.field_type == "query_context"
    blob = json.dumps(query_field.model_dump())
    summary_blob = json.dumps(projection["query_summary"])
    for forbidden in (RAW_SEQ, "HEADER TEST PDB", "ATOM      1", "sk-secretvalue123"):
        assert forbidden not in blob
        assert forbidden not in summary_blob


def test_missing_inputs_flags_candidate_without_structure_or_sequence():
    candidate = _base_candidate(identifiers=[{"id_type": "chain", "id_value": "A"}])
    projection = project_step9_inputs(
        candidate_context_table=_cct(candidate),
        prepared_structure_input_package=None,
        structure_prediction_and_interface_results=None,
    )
    assert "cand_t1:missing_structure_or_sequence_input" in projection["missing_inputs"]


def test_no_raw_data_anywhere_in_full_projection_json():
    candidate = _base_candidate(
        materials=[
            {"material_id": "seq_1", "material_type": "target_sequence", "value": RAW_SEQ},
        ],
        identifiers=[{"id_type": "uniprot_id", "id_value": "P04626"}],
    )
    prepared = [
        {
            "candidate_id": "cand_t1",
            "structure_input_id": "si_1",
            "structure_refs": [{"storage_ref": STORAGE_PATH, "structure_format": "pdb", "source_kind": "uploaded_file"}],
            "sequence_refs_for_prediction": [],
        }
    ]
    step8 = {
        "candidate_structure_results": [
            {
                "candidate_id": "cand_t1",
                "complex_structure_refs": [{"source_kind": "existing_pdb_complex", "pdb_id": "1ABC"}],
                "downstream_handoff": {"validated_structure_ref": STORAGE_PATH},
            }
        ]
    }
    structured_query = {"canonical_query": f"design with {RAW_SEQ} sk-secretvalue123"}
    projection = project_step9_inputs(
        candidate_context_table=_cct(candidate),
        prepared_structure_input_package=prepared,
        structure_prediction_and_interface_results=step8,
        structured_query=structured_query,
        raw_request={},
    )
    blob = json.dumps(
        {
            "input_fields": [f.model_dump() for f in projection["input_fields"]],
            "candidate_summaries": projection["candidate_summaries"],
            "handoff_summary": projection["handoff_summary"],
            "missing_inputs": projection["missing_inputs"],
            "query_summary": projection["query_summary"],
        }
    )
    assert RAW_SEQ not in blob
    assert STORAGE_PATH not in blob
    assert "sk-secretvalue123" not in blob
    assert "A3M" not in blob.upper()


# ── field_ref uniqueness / duplicate merge ──────────────────────────────────

def test_step5_and_step7_same_uniprot_merges_into_single_field_ref():
    """A candidate with an explicit Step 5 UniProt identifier AND a Step 7
    identifier-only sequence ref for the SAME accession must project to
    exactly one field_ref, not two competing entries."""
    candidate = _base_candidate(identifiers=[{"id_type": "uniprot_id", "id_value": "P04626"}])
    prepared = [
        {
            "candidate_id": "cand_t1",
            "structure_input_id": "si_1",
            "structure_refs": [],
            "sequence_refs_for_prediction": [
                {
                    "sequence_id": "seq_1",
                    "sequence_value_status": "identifier_only",
                    "prediction_input_kind": "uniprot_id",
                    "source_ref": "P04626",
                    "chain_role": "antigen",
                }
            ],
        }
    ]
    projection = project_step9_inputs(
        candidate_context_table=_cct(candidate),
        prepared_structure_input_package=prepared,
        structure_prediction_and_interface_results=None,
    )
    refs = [f.field_ref for f in projection["input_fields"]]
    assert refs.count("identifier:uniprot_id:P04626") == 1
    assert len(refs) == len(set(refs))

    merged = next(f for f in projection["input_fields"] if f.field_ref == "identifier:uniprot_id:P04626")
    assert set(merged.supports_tool_args) == {"uniprot_id", "accession", "uniprot_accession"}
    # chain_role from the Step 7 contributor must not be lost.
    assert merged.chain_role == "antigen"
    assert merged.source_step == "step_05"
    assert merged.source_steps == ["step_05", "step_07"]
    assert merged.candidate_id == "cand_t1"
    assert merged.candidate_ids == ["cand_t1"]
    sources = merged.runtime_lookup.get("sources")
    assert sources is not None and len(sources) == 2
    assert {s["source_step"] for s in sources} == {"step_05", "step_07"}


def test_cross_candidate_same_pdb_id_merges_and_keeps_both_candidate_ids():
    """Two different candidates (e.g. antigen + antibody) whose Step 8
    complex results reference the same real PDB id must not silently drop
    one candidate's field — the merged field records both."""
    candidate_a = _base_candidate(candidate_id="cand_a")
    candidate_b = _base_candidate(candidate_id="cand_b")
    step8 = {
        "candidate_structure_results": [
            {
                "candidate_id": "cand_a",
                "complex_structure_refs": [
                    {"source_kind": "existing_pdb_complex", "pdb_id": "1N8Z", "source_ref": "1N8Z"}
                ],
            },
            {
                "candidate_id": "cand_b",
                "complex_structure_refs": [
                    {"source_kind": "existing_pdb_complex", "pdb_id": "1N8Z", "source_ref": "1N8Z"}
                ],
            },
        ]
    }
    projection = project_step9_inputs(
        candidate_context_table={"candidate_records": [candidate_a, candidate_b]},
        prepared_structure_input_package=None,
        structure_prediction_and_interface_results=step8,
    )
    refs = [f.field_ref for f in projection["input_fields"]]
    assert refs.count("identifier:pdb_id:1N8Z") == 1
    assert len(refs) == len(set(refs))

    merged = next(f for f in projection["input_fields"] if f.field_ref == "identifier:pdb_id:1N8Z")
    assert set(merged.candidate_ids) == {"cand_a", "cand_b"}
    assert merged.supports_tool_args == ["pdb_id"]


def test_all_projected_field_refs_are_globally_unique_in_real_pdb_like_fixture():
    """Realistic multi-candidate, multi-step fixture: every field_ref in the
    final projection must be unique, covering uniprot/pdb_id/structure/
    sequence duplication sources at once."""
    candidate_antigen = _base_candidate(
        candidate_id="cand_antigen",
        candidate_type="target_antigen",
        identifiers=[
            {"id_type": "uniprot_id", "id_value": "P04626"},
            {"id_type": "pdb_id", "id_value": "1N8Z"},
        ],
    )
    candidate_antibody = _base_candidate(
        candidate_id="cand_antibody",
        candidate_type="antibody",
        identifiers=[{"id_type": "chain", "id_value": "A"}],
    )
    prepared = [
        {
            "candidate_id": "cand_antigen",
            "structure_input_id": "si_antigen",
            "structure_refs": [{"pdb_id": "1N8Z", "structure_format": "pdb", "source_kind": "pdb_id"}],
            "sequence_refs_for_prediction": [
                {
                    "sequence_id": "seq_antigen",
                    "sequence_value_status": "identifier_only",
                    "prediction_input_kind": "uniprot_id",
                    "source_ref": "P04626",
                    "chain_role": "antigen",
                }
            ],
        }
    ]
    step8 = {
        "candidate_structure_results": [
            {
                "candidate_id": "cand_antigen",
                "complex_structure_refs": [
                    {"source_kind": "existing_pdb_complex", "pdb_id": "1N8Z", "source_ref": "1N8Z"}
                ],
            },
            {
                "candidate_id": "cand_antibody",
                "complex_structure_refs": [
                    {"source_kind": "existing_pdb_complex", "pdb_id": "1N8Z", "source_ref": "1N8Z"}
                ],
            },
        ]
    }
    projection = project_step9_inputs(
        candidate_context_table={"candidate_records": [candidate_antigen, candidate_antibody]},
        prepared_structure_input_package=prepared,
        structure_prediction_and_interface_results=step8,
    )
    refs = [f.field_ref for f in projection["input_fields"]]
    assert len(refs) == len(set(refs)), f"duplicate field_ref(s): {[r for r in refs if refs.count(r) > 1]}"

    pdb_field = next(f for f in projection["input_fields"] if f.field_ref == "identifier:pdb_id:1N8Z")
    assert set(pdb_field.candidate_ids) == {"cand_antigen", "cand_antibody"}
    assert pdb_field.source_steps == ["step_05", "step_07", "step_08"]


def test_assert_unique_input_field_refs_passes_for_unique_list():
    assert_unique_input_field_refs(
        [{"field_ref": "identifier:pdb_id:1ABC"}, {"field_ref": "identifier:uniprot_id:P04626"}]
    )


def test_assert_unique_input_field_refs_raises_for_duplicate_list():
    with pytest.raises(DuplicateStep9InputFieldError, match="identifier:uniprot_id:P04626"):
        assert_unique_input_field_refs(
            [
                {"field_ref": "identifier:uniprot_id:P04626", "candidate_id": "cand_a"},
                {"field_ref": "identifier:uniprot_id:P04626", "candidate_id": "cand_b"},
            ]
        )
