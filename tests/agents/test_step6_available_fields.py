from __future__ import annotations

import hashlib
import json

from app.agents.step_06_available_fields import (
    project_candidate_available_fields,
    project_candidate_context_available_fields,
)


HEAVY_SEQ = "EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGK"
LIGHT_SEQ = "DIQMTQSPSSLSASVGDRVTITCRASQ"
FASTA_PATH = "adc_pilot/runs/run_x/inputs/heavy_chain.fasta"
PDB_PATH = "adc_pilot/runs/run_x/inputs/her2_complex.pdb"
RAW_CDR3 = "ARDRGGYFDY"
SMILES = "CC(=O)NCCO"
UNIPROT = "P04626"
CHEMBL = "CHEMBL2107839"


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]


def _candidate(*, materials=None, identifiers=None):
    return {
        "candidate_id": "cand_step6_a",
        "candidate_label": "fixture",
        "candidate_type": "adc_construct",
        "materials": materials or [],
        "identifiers": identifiers or [],
        "context_notes": [],
        "data_gaps": [],
    }


def _material(material_id: str, material_type: str, value: str, **extra):
    return {
        "material_id": material_id,
        "material_type": material_type,
        "value": value,
        "value_format": extra.pop("value_format", None),
        "role": extra.pop("role", None),
        "role_status": extra.pop("role_status", "unknown"),
        **extra,
    }


def _identifier(id_type: str, value: str):
    return {
        "id_type": id_type,
        "id_value": value,
        "source_ids": ["sq1"],
        "confidence": 0.9,
    }


def _field(fields, *, material_type=None, id_type=None):
    for field in fields:
        if material_type and field.material_type == material_type:
            return field
        if id_type and field.id_type == id_type:
            return field
    raise AssertionError(f"field not found: material_type={material_type} id_type={id_type}")


def test_inline_heavy_and_light_sequences_are_digest_only():
    candidate = _candidate(
        materials=[
            _material(
                "mat_heavy",
                "antibody_heavy_chain_sequence",
                HEAVY_SEQ,
                role="antibody",
            ),
            _material(
                "mat_light",
                "antibody_light_chain_sequence",
                LIGHT_SEQ,
                role="antibody",
            ),
        ]
    )
    projection = project_candidate_available_fields(candidate)
    heavy = _field(projection.available_fields, material_type="antibody_heavy_chain_sequence")
    light = _field(projection.available_fields, material_type="antibody_light_chain_sequence")

    assert heavy.length == len(HEAVY_SEQ)
    assert heavy.sha256_prefix == _sha(HEAVY_SEQ)
    assert heavy.ref_length is None and heavy.ref_sha256_prefix is None
    assert heavy.chain_role == "heavy"
    assert "use_heavy_chain_sequence" in heavy.allowed_transforms

    assert light.length == len(LIGHT_SEQ)
    assert light.sha256_prefix == _sha(LIGHT_SEQ)
    assert light.ref_length is None and light.ref_sha256_prefix is None
    assert light.chain_role == "light"
    assert "use_light_chain_sequence" in light.allowed_transforms

    dumped = json.dumps([f.model_dump() for f in projection.available_fields])
    assert HEAVY_SEQ not in dumped
    assert LIGHT_SEQ not in dumped
    assert projection.modality_summary.has_antibody_heavy_sequence is True
    assert projection.modality_summary.has_antibody_light_sequence is True
    assert projection.modality_summary.has_antibody_sequence is True


def test_uploaded_fasta_and_pdb_refs_use_ref_metadata_without_path_leakage():
    candidate = _candidate(
        materials=[
            _material(
                "mat_fasta_ref",
                "antibody_heavy_chain_sequence",
                FASTA_PATH,
                value_format="fasta",
                role="antibody",
            ),
            _material(
                "mat_pdb_ref",
                "structure_ref",
                PDB_PATH,
                value_format="pdb",
                role="structure_reference",
            ),
        ]
    )
    projection = project_candidate_available_fields(candidate)
    fasta = _field(projection.available_fields, material_type="antibody_heavy_chain_sequence")
    pdb = _field(projection.available_fields, material_type="structure_ref")

    assert fasta.length is None and fasta.sha256_prefix is None
    assert fasta.ref_length == len(FASTA_PATH)
    assert fasta.ref_sha256_prefix == _sha(FASTA_PATH)
    assert fasta.value_kind == "uploaded_fasta_ref"

    assert pdb.length is None and pdb.sha256_prefix is None
    assert pdb.ref_length == len(PDB_PATH)
    assert pdb.ref_sha256_prefix == _sha(PDB_PATH)
    assert pdb.value_kind == "structure_ref"

    dumped = json.dumps([f.model_dump() for f in projection.available_fields])
    assert FASTA_PATH not in dumped
    assert PDB_PATH not in dumped
    assert projection.modality_summary.has_uploaded_fasta_ref is True
    assert projection.modality_summary.has_uploaded_structure_ref is True


def test_identifiers_and_smiles_have_typed_transforms_without_value_leakage():
    candidate = _candidate(
        materials=[
            _material("mat_payload_smiles", "payload_smiles", SMILES, role="payload"),
        ],
        identifiers=[
            _identifier("uniprot_id", UNIPROT),
            _identifier("chembl_id", CHEMBL),
            _identifier("pdb_id", "1N8Z"),
        ],
    )
    projection = project_candidate_available_fields(candidate)
    smiles = _field(projection.available_fields, material_type="payload_smiles")
    uniprot = _field(projection.available_fields, id_type="uniprot_id")
    chembl = _field(projection.available_fields, id_type="chembl_id")
    pdb = _field(projection.available_fields, id_type="pdb_id")

    assert smiles.value_kind == "smiles"
    assert "use_smiles" in smiles.allowed_transforms
    assert uniprot.value_kind == "uniprot_id"
    assert "use_accession" in uniprot.allowed_transforms
    assert chembl.value_kind == "chembl_id"
    assert "use_chembl_id" in chembl.allowed_transforms
    assert pdb.value_kind == "pdb_id"
    assert "use_pdb_id" in pdb.allowed_transforms

    dumped = json.dumps([f.model_dump() for f in projection.available_fields])
    assert SMILES not in dumped
    assert UNIPROT not in dumped
    assert CHEMBL not in dumped
    assert projection.modality_summary.has_payload_smiles is True
    assert projection.modality_summary.has_compound_smiles is True
    assert projection.modality_summary.has_uniprot_id is True
    assert projection.modality_summary.has_compound_identifier is True
    assert projection.modality_summary.has_pdb_id is True


def test_every_available_field_uses_value_or_ref_metadata_xor_and_stable_refs():
    candidate = _candidate(
        materials=[
            _material("mat_heavy", "antibody_heavy_chain_sequence", HEAVY_SEQ),
            _material("mat_fasta", "antibody_light_chain_sequence", FASTA_PATH, value_format="fasta"),
            _material("mat_linker", "linker_smiles", SMILES),
        ],
        identifiers=[_identifier("chembl_id", CHEMBL)],
    )
    first = project_candidate_available_fields(candidate).available_fields
    second = project_candidate_available_fields(candidate).available_fields
    assert [f.field_ref for f in first] == [f.field_ref for f in second]

    for field in first:
        has_value = field.length is not None and field.sha256_prefix is not None
        has_ref = field.ref_length is not None and field.ref_sha256_prefix is not None
        assert has_value ^ has_ref
        assert ":material:0:" not in field.field_ref
        assert ":identifier:0:" not in field.field_ref
        assert field.field_ref.startswith(f"candidate:{candidate['candidate_id']}:")


def test_no_raw_fasta_pdb_sequence_cdr3_or_path_substrings_in_projection_json():
    candidate = _candidate(
        materials=[
            _material("mat_heavy", "antibody_heavy_chain_sequence", HEAVY_SEQ),
            _material("mat_fasta", "antibody_light_chain_sequence", FASTA_PATH, value_format="fasta"),
            _material("mat_pdb", "structure_file", PDB_PATH, value_format="pdb"),
            _material("mat_cdr3", "antibody_heavy_cdr3_sequence", RAW_CDR3),
        ]
    )
    projection = project_candidate_available_fields(candidate)
    dumped = projection.model_dump_json()
    for forbidden in (HEAVY_SEQ, FASTA_PATH, PDB_PATH, RAW_CDR3, "heavy_chain.fasta", "her2_complex.pdb"):
        assert forbidden not in dumped
    assert projection.modality_summary.has_cdr3_ref_or_marker is True


def test_modality_summary_for_pdb_only_and_mixed_inputs():
    pdb_only = project_candidate_available_fields(
        _candidate(materials=[_material("mat_pdb", "structure_ref", PDB_PATH, value_format="pdb")])
    )
    assert pdb_only.modality_summary.has_uploaded_structure_ref is True
    assert pdb_only.modality_summary.has_protein_sequence is False

    mixed = project_candidate_available_fields(
        _candidate(
            materials=[
                _material("mat_payload", "payload_smiles", "CCO", role="payload"),
                _material("mat_linker", "linker_smiles", "NCCO", role="linker"),
                _material("mat_heavy", "antibody_heavy_chain_sequence", HEAVY_SEQ, role="antibody"),
                _material("mat_fasta", "antibody_light_chain_sequence", FASTA_PATH, value_format="fasta"),
            ],
            identifiers=[_identifier("uniprot_id", UNIPROT), _identifier("chembl_id", CHEMBL)],
        )
    )
    tags = set(mixed.modality_summary.modality_tags)
    assert {
        "payload_smiles",
        "linker_smiles",
        "compound_smiles",
        "antibody_sequence",
        "antibody_heavy_sequence",
        "fasta_ref",
        "uniprot_id",
        "compound_identifier",
    } <= tags


def test_ambiguous_sequence_reference_is_marked_unknown_not_false_certainty():
    projection = project_candidate_available_fields(
        _candidate(
            materials=[
                _material(
                    "mat_generic_ab_ref",
                    "antibody_sequence_reference",
                    FASTA_PATH,
                    value_format="fasta",
                    role="antibody_sequence_reference",
                )
            ]
        )
    )
    summary = projection.modality_summary
    assert summary.has_uploaded_fasta_ref is True
    assert summary.has_antibody_sequence is True
    assert summary.ambiguous_or_unknown is True
    assert summary.unknown_notes


def test_candidate_context_projection_keys_by_candidate_id():
    cct = {
        "candidate_records": [
            _candidate(materials=[_material("mat_payload", "payload_smiles", SMILES)]),
        ]
    }
    projection = project_candidate_context_available_fields(cct)
    assert set(projection) == {"cand_step6_a"}
