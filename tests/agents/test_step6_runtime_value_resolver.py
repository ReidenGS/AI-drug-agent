from __future__ import annotations

import json

from app.agents.step_06_available_fields import project_candidate_available_fields
from app.agents.step_06_runtime_value_resolver import resolve_runtime_value


HEAVY_SEQ = "EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGK"
SMILES = "CC(=O)NCCO"
UNIPROT = "P04626"
CHEMBL = "CHEMBL2107839"
FASTA_PATH = "adc_pilot/runs/run_x/inputs/heavy_chain.fasta"
FASTA_CONTENT = ">chainA\nEVQLVESGG\n"
FASTA_MULTI_CHAIN_CONTENT = (
    ">chainA\nEVQLVESGG\n>chainB\nDIQMTQSP\n"
)
PDB_PATH = "adc_pilot/runs/run_x/inputs/her2_complex.pdb"
RAW_CDR3 = "ARDRGGYFDY"


def _candidate():
    return {
        "candidate_id": "cand_resolver",
        "candidate_label": "fixture",
        "candidate_type": "adc_construct",
        "materials": [
            {
                "material_id": "mat_heavy",
                "material_type": "antibody_heavy_chain_sequence",
                "value": HEAVY_SEQ,
                "value_format": None,
                "role": "antibody",
            },
            {
                "material_id": "mat_smiles",
                "material_type": "payload_smiles",
                "value": SMILES,
                "value_format": None,
                "role": "payload",
            },
            {
                "material_id": "mat_fasta",
                "material_type": "antibody_light_chain_sequence",
                "value": FASTA_PATH,
                "value_format": "fasta",
                "role": "antibody",
            },
            {
                "material_id": "mat_pdb",
                "material_type": "structure_ref",
                "value": PDB_PATH,
                "value_format": "pdb",
                "role": "structure_reference",
            },
            {
                "material_id": "mat_cdr3",
                "material_type": "antibody_heavy_cdr3_sequence",
                "value": RAW_CDR3,
                "value_format": None,
                "role": "antibody",
            },
        ],
        "identifiers": [
            {"id_type": "uniprot_id", "id_value": UNIPROT, "source_ids": [], "confidence": 0.9},
            {"id_type": "chembl_id", "id_value": CHEMBL, "source_ids": [], "confidence": 0.9},
        ],
    }


def _field_ref(candidate, *, material_type=None, id_type=None):
    projection = project_candidate_available_fields(candidate)
    for field in projection.available_fields:
        if material_type and field.material_type == material_type:
            return field.field_ref
        if id_type and field.id_type == id_type:
            return field.field_ref
    raise AssertionError("field_ref not found")


def _assert_audit_clean(result):
    dumped = json.dumps(result.audit_metadata)
    model_dumped = result.model_dump_json()
    for forbidden in (
        HEAVY_SEQ,
        SMILES,
        UNIPROT,
        CHEMBL,
        FASTA_PATH,
        PDB_PATH,
        RAW_CDR3,
        "heavy_chain.fasta",
        "her2_complex.pdb",
    ):
        assert forbidden not in dumped
        assert forbidden not in model_dumped


def test_resolver_recovers_inline_sequence_but_audit_is_digest_only():
    candidate = _candidate()
    ref = _field_ref(candidate, material_type="antibody_heavy_chain_sequence")
    result = resolve_runtime_value(candidate=candidate, field_ref=ref)

    assert result.status == "resolved"
    assert result.raw_value == HEAVY_SEQ
    assert result.audit_metadata["field_type"] == "protein_sequence"
    assert result.audit_metadata["value_kind"] == "protein_sequence"
    assert result.audit_metadata["length"] == len(HEAVY_SEQ)
    _assert_audit_clean(result)


def test_resolver_recovers_smiles_uniprot_and_chembl_raw_values():
    candidate = _candidate()
    checks = [
        ("payload_smiles", None, SMILES, "smiles"),
        (None, "uniprot_id", UNIPROT, "uniprot_id"),
        (None, "chembl_id", CHEMBL, "chembl_id"),
    ]
    for material_type, id_type, expected, value_kind in checks:
        ref = _field_ref(candidate, material_type=material_type, id_type=id_type)
        result = resolve_runtime_value(candidate=candidate, field_ref=ref)
        assert result.status == "resolved"
        assert result.raw_value == expected
        assert result.audit_metadata["value_kind"] == value_kind
        _assert_audit_clean(result)


def test_resolver_uploaded_ref_materials_use_ref_or_runtime_resolution():
    candidate = _candidate()
    fasta_ref = _field_ref(candidate, material_type="antibody_light_chain_sequence")
    fasta_result = resolve_runtime_value(candidate=candidate, field_ref=fasta_ref)
    assert fasta_result.status in {"missing", "unresolved"}

    structure_ref = _field_ref(candidate, material_type="structure_ref")
    structure_result = resolve_runtime_value(candidate=candidate, field_ref=structure_ref)
    assert structure_result.status == "resolved"
    assert structure_result.raw_value == PDB_PATH
    assert "ref_length" in structure_result.audit_metadata
    assert "ref_sha256_prefix" in structure_result.audit_metadata
    _assert_audit_clean(structure_result)


def test_unresolved_and_missing_refs_return_explicit_status():
    candidate = _candidate()
    unresolved = resolve_runtime_value(
        candidate=candidate,
        field_ref="candidate:cand_resolver:material:does_not_exist:value",
    )
    assert unresolved.status == "unresolved"
    assert unresolved.raw_value is None
    assert unresolved.error_message

    missing_candidate = _candidate()
    missing_candidate["materials"][0]["value"] = ""
    ref = "candidate:cand_resolver:material:mat_heavy:value"
    missing = resolve_runtime_value(candidate=missing_candidate, field_ref=ref)
    assert missing.status in {"missing", "unresolved"}
    assert missing.raw_value is None
    assert missing.error_message


def test_resolver_resolves_uploaded_fasta_at_runtime(local_storage):
    candidate = _candidate()
    local_storage.write_bytes(FASTA_PATH, FASTA_CONTENT.encode("utf-8"))
    ref = _field_ref(candidate, material_type="antibody_light_chain_sequence")
    result = resolve_runtime_value(candidate=candidate, field_ref=ref, storage=local_storage)

    assert result.status == "resolved"
    assert result.raw_value == "EVQLVESGG"
    assert result.audit_metadata["field_type"] == "protein_sequence"
    assert result.audit_metadata["value_kind"] == "uploaded_fasta_ref"
    assert result.audit_metadata["chain_role"] == "light"
    _assert_audit_clean(result)


def test_resolver_rejects_missing_uploaded_fasta(local_storage):
    candidate = _candidate()
    ref = _field_ref(candidate, material_type="antibody_light_chain_sequence")
    result = resolve_runtime_value(
        candidate=candidate,
        field_ref=ref,
        storage=local_storage,
    )

    assert result.status in {"missing", "unresolved"}
    assert result.error_message
    _assert_audit_clean(result)


def test_resolver_picks_first_fasta_chain_only(local_storage):
    candidate = _candidate()
    local_storage.write_bytes(FASTA_PATH, FASTA_MULTI_CHAIN_CONTENT.encode("utf-8"))
    ref = _field_ref(candidate, material_type="antibody_light_chain_sequence")
    result = resolve_runtime_value(candidate=candidate, field_ref=ref, storage=local_storage)

    assert result.status == "resolved"
    assert result.raw_value == "EVQLVESGG"
    assert result.raw_value != "DIQMTQSP"
    assert result.raw_value == result.raw_value.strip()
    _assert_audit_clean(result)


def test_resolver_audit_does_not_leak_raw_cdr3():
    candidate = _candidate()
    ref = _field_ref(candidate, material_type="antibody_heavy_cdr3_sequence")
    result = resolve_runtime_value(candidate=candidate, field_ref=ref)
    assert result.status == "resolved"
    assert result.raw_value == RAW_CDR3
    assert result.audit_metadata["value_kind"] == "cdr3_marker"
    _assert_audit_clean(result)
