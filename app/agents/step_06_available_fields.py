"""Step 6 LLM-safe available-field projection.

This module prepares Turn-A inputs for later Step 6 LLM planning without
exposing raw candidate values. It does not read storage, inspect uploaded
files, call MCP, or decide tool eligibility.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, model_validator


SourceKind = Literal["material", "identifier", "candidate_metadata"]
PrivacyClass = Literal["llm_safe_digest", "llm_safe_ref_digest", "llm_safe_metadata"]

_SHA_PREFIX_LEN = 12


class AvailableField(BaseModel):
    field_ref: str
    candidate_id: str
    source_kind: SourceKind
    source_id: str
    field_name: str
    field_type: str
    value_kind: str
    semantic_role: Optional[str] = None
    material_type: Optional[str] = None
    id_type: Optional[str] = None
    chain_role: Optional[str] = None
    length: Optional[int] = None
    sha256_prefix: Optional[str] = None
    ref_length: Optional[int] = None
    ref_sha256_prefix: Optional[str] = None
    allowed_transforms: list[str] = Field(default_factory=list)
    privacy_class: PrivacyClass = "llm_safe_digest"

    @model_validator(mode="after")
    def _metadata_xor(self) -> "AvailableField":
        has_value_meta = self.length is not None or self.sha256_prefix is not None
        has_ref_meta = self.ref_length is not None or self.ref_sha256_prefix is not None
        if has_value_meta == has_ref_meta:
            raise ValueError(
                "AvailableField must carry exactly one of value metadata or ref metadata"
            )
        if has_value_meta and (self.length is None or self.sha256_prefix is None):
            raise ValueError("value metadata requires length and sha256_prefix")
        if has_ref_meta and (self.ref_length is None or self.ref_sha256_prefix is None):
            raise ValueError("ref metadata requires ref_length and ref_sha256_prefix")
        return self


class CandidateModalitySummary(BaseModel):
    candidate_id: str
    has_payload_smiles: bool = False
    has_linker_smiles: bool = False
    has_compound_smiles: bool = False
    has_compound_identifier: bool = False
    has_antibody_heavy_sequence: bool = False
    has_antibody_light_sequence: bool = False
    has_antibody_sequence: bool = False
    has_antigen_sequence: bool = False
    has_protein_sequence: bool = False
    has_uniprot_id: bool = False
    has_pdb_id: bool = False
    has_uploaded_structure_ref: bool = False
    has_uploaded_fasta_ref: bool = False
    has_cdr3_ref_or_marker: bool = False
    modality_tags: list[str] = Field(default_factory=list)
    ambiguous_or_unknown: bool = False
    unknown_notes: list[str] = Field(default_factory=list)


class CandidateAvailableFieldsProjection(BaseModel):
    candidate_id: str
    available_fields: list[AvailableField] = Field(default_factory=list)
    modality_summary: CandidateModalitySummary


def project_candidate_available_fields(candidate: dict[str, Any]) -> CandidateAvailableFieldsProjection:
    """Project one Step 5 candidate record into LLM-safe field refs."""
    candidate_id = str(candidate.get("candidate_id") or "unknown")
    fields: list[AvailableField] = []

    fields.extend(_metadata_fields(candidate_id, candidate))

    for material in candidate.get("materials") or []:
        if isinstance(material, dict):
            field = _field_from_material(candidate_id, material)
            if field is not None:
                fields.append(field)

    for identifier in candidate.get("identifiers") or []:
        if isinstance(identifier, dict):
            field = _field_from_identifier(candidate_id, identifier)
            if field is not None:
                fields.append(field)

    summary = summarize_candidate_modalities(candidate, fields)
    return CandidateAvailableFieldsProjection(
        candidate_id=candidate_id,
        available_fields=fields,
        modality_summary=summary,
    )


def project_candidate_context_available_fields(
    candidate_context_table: dict[str, Any],
) -> dict[str, CandidateAvailableFieldsProjection]:
    """Project every Step 5 candidate record, keyed by candidate_id."""
    out: dict[str, CandidateAvailableFieldsProjection] = {}
    for candidate in candidate_context_table.get("candidate_records") or []:
        if not isinstance(candidate, dict):
            continue
        projection = project_candidate_available_fields(candidate)
        out[projection.candidate_id] = projection
    return out


def _metadata_fields(candidate_id: str, candidate: dict[str, Any]) -> list[AvailableField]:
    out: list[AvailableField] = []
    for key in ("candidate_type", "candidate_role", "context_status"):
        value = candidate.get(key)
        if value in (None, ""):
            continue
        value_str = str(value)
        out.append(
            AvailableField(
                field_ref=f"candidate:{candidate_id}:candidate_metadata:{key}:value",
                candidate_id=candidate_id,
                source_kind="candidate_metadata",
                source_id=key,
                field_name="value",
                field_type="candidate_metadata",
                value_kind=key,
                semantic_role=key,
                length=len(value_str),
                sha256_prefix=_sha_prefix(value_str),
                allowed_transforms=["use_metadata"],
                privacy_class="llm_safe_metadata",
            )
        )
    return out


def summarize_candidate_modalities(
    candidate: dict[str, Any], fields: list[AvailableField] | None = None
) -> CandidateModalitySummary:
    candidate_id = str(candidate.get("candidate_id") or "unknown")
    fields = fields if fields is not None else project_candidate_available_fields(candidate).available_fields
    summary = CandidateModalitySummary(candidate_id=candidate_id)
    tags: set[str] = set()
    unknown_notes: list[str] = []

    for field in fields:
        mt = field.material_type or ""
        vk = field.value_kind
        role = field.semantic_role or ""

        if mt == "payload_smiles":
            summary.has_payload_smiles = True
            tags.add("payload_smiles")
        if mt == "linker_smiles":
            summary.has_linker_smiles = True
            tags.add("linker_smiles")
        if mt == "compound_smiles" or vk == "smiles":
            summary.has_compound_smiles = True
            tags.add("compound_smiles")
        if field.id_type in {"chembl_id", "pubchem_cid", "zinc_id", "drugbank_id"}:
            summary.has_compound_identifier = True
            tags.add("compound_identifier")
        if field.id_type == "uniprot_id":
            summary.has_uniprot_id = True
            tags.add("uniprot_id")
        if field.id_type == "pdb_id":
            summary.has_pdb_id = True
            tags.add("pdb_id")
        if field.value_kind in {"structure_ref", "structure_file", "uploaded_structure_ref"}:
            summary.has_uploaded_structure_ref = True
            tags.add("structure_ref")
        if field.value_kind == "uploaded_fasta_ref":
            summary.has_uploaded_fasta_ref = True
            tags.add("fasta_ref")
        if field.value_kind == "cdr3_marker":
            summary.has_cdr3_ref_or_marker = True
            tags.add("cdr3_marker")

        is_sequence = field.field_type == "protein_sequence"
        if is_sequence:
            summary.has_protein_sequence = True
            tags.add("protein_sequence")
            if field.chain_role == "heavy":
                summary.has_antibody_heavy_sequence = True
                tags.add("antibody_heavy_sequence")
            elif field.chain_role == "light":
                summary.has_antibody_light_sequence = True
                tags.add("antibody_light_sequence")
            elif role in {"target", "target_sequence_reference", "antigen"} or mt == "target_sequence":
                summary.has_antigen_sequence = True
                tags.add("antigen_sequence")
            else:
                unknown_notes.append(
                    f"sequence field {field.field_ref} has unknown antibody chain role"
                )
            if mt.startswith("antibody_") or role in {"antibody", "antibody_sequence_reference"}:
                summary.has_antibody_sequence = True
                tags.add("antibody_sequence")

    for material in candidate.get("materials") or []:
        if not isinstance(material, dict):
            continue
        mt = str(material.get("material_type") or "")
        if mt in {"antibody_sequence_reference"}:
            summary.has_uploaded_fasta_ref = summary.has_uploaded_fasta_ref or _is_ref_shaped_material(material)
            unknown_notes.extend(
                [
                    "antibody sequence reference has unknown chain role",
                    "generic_antibody_sequence_reference_not_executable",
                ]
            )
        elif mt in {"target_sequence"} and _is_ref_shaped_material(material):
            summary.has_antigen_sequence = True
            summary.has_protein_sequence = True
            summary.has_uploaded_fasta_ref = True
            tags.update({"antigen_sequence", "protein_sequence", "fasta_ref"})

    summary.unknown_notes = sorted(set(unknown_notes))
    summary.ambiguous_or_unknown = bool(summary.unknown_notes) or not fields
    summary.modality_tags = sorted(tags)
    return summary


def _field_from_material(candidate_id: str, material: dict[str, Any]) -> AvailableField | None:
    material_id = str(material.get("material_id") or "")
    material_type = str(material.get("material_type") or "")
    raw_value = material.get("value")
    if not material_id or raw_value in (None, ""):
        return None
    value = str(raw_value)
    role = material.get("role")
    if material_type == "antibody_sequence_reference":
        return None
    field_type, value_kind = _material_field_type_and_value_kind(material)
    if not field_type or not value_kind:
        return None

    field_ref = f"candidate:{candidate_id}:material:{material_id}:value"
    allowed = _allowed_transforms(value_kind, material_type)
    chain_role = _chain_role(material_type, role)
    common = {
        "field_ref": field_ref,
        "candidate_id": candidate_id,
        "source_kind": "material",
        "source_id": material_id,
        "field_name": "value",
        "field_type": field_type,
        "value_kind": value_kind,
        "semantic_role": str(role) if role else None,
        "material_type": material_type,
        "chain_role": chain_role,
        "allowed_transforms": allowed,
    }
    if _is_ref_shaped_material(material):
        return AvailableField(
            **common,
            ref_length=len(value),
            ref_sha256_prefix=_sha_prefix(value),
            privacy_class="llm_safe_ref_digest",
        )
    return AvailableField(
        **common,
        length=len(value),
        sha256_prefix=_sha_prefix(value),
        privacy_class="llm_safe_digest",
    )


def _field_from_identifier(candidate_id: str, identifier: dict[str, Any]) -> AvailableField | None:
    id_type = str(identifier.get("id_type") or "")
    raw_value = identifier.get("id_value")
    if not id_type or raw_value in (None, ""):
        return None
    value = str(raw_value)
    value_kind = _identifier_value_kind(id_type)
    field_type = _identifier_field_type(id_type)
    source_id = _identifier_source_id(id_type, value)
    return AvailableField(
        field_ref=f"candidate:{candidate_id}:identifier:{source_id}:value",
        candidate_id=candidate_id,
        source_kind="identifier",
        source_id=source_id,
        field_name="id_value",
        field_type=field_type,
        value_kind=value_kind,
        id_type=id_type,
        length=len(value),
        sha256_prefix=_sha_prefix(value),
        allowed_transforms=_allowed_transforms(value_kind, None),
        privacy_class="llm_safe_digest",
    )


def _material_field_type_and_value_kind(material: dict[str, Any]) -> tuple[str | None, str | None]:
    mt = str(material.get("material_type") or "")
    if mt in {"payload_smiles", "linker_smiles", "compound_smiles"}:
        return "small_molecule", "smiles"
    if mt in {
        "antibody_heavy_chain_sequence",
        "antibody_light_chain_sequence",
        "antibody_sequence_reference",
        "target_sequence",
    }:
        if _is_ref_shaped_material(material):
            return "protein_sequence", "uploaded_fasta_ref"
        return "protein_sequence", "protein_sequence"
    if mt in {"antibody_heavy_cdr3_sequence", "antibody_light_cdr3_sequence"}:
        return "protein_sequence", "cdr3_marker"
    if mt in {"structure_file", "structure_ref"}:
        return "structure", "structure_ref"
    if mt.startswith("compound_identifier_"):
        return "small_molecule_identifier", mt.removeprefix("compound_identifier_")
    if mt in {"target_antigen_name", "antibody_name", "payload_name", "linker_name", "compound_name", "linker_payload_name"}:
        return "candidate_metadata", "name"
    return None, None


def _identifier_field_type(id_type: str) -> str:
    if id_type == "uniprot_id":
        return "protein_identifier"
    if id_type == "pdb_id":
        return "structure_identifier"
    if id_type in {"chembl_id", "pubchem_cid", "zinc_id", "drugbank_id"}:
        return "small_molecule_identifier"
    return "identifier"


def _identifier_value_kind(id_type: str) -> str:
    if id_type == "uniprot_id":
        return "uniprot_id"
    if id_type == "pdb_id":
        return "pdb_id"
    if id_type == "chembl_id":
        return "chembl_id"
    return id_type or "identifier"


def _allowed_transforms(value_kind: str, material_type: str | None) -> list[str]:
    transforms = {
        "smiles": ["use_smiles"],
        "uniprot_id": ["use_accession"],
        "chembl_id": ["use_chembl_id"],
        "pdb_id": ["use_pdb_id"],
        "protein_sequence": ["use_sequence"],
        "uploaded_fasta_ref": ["resolve_ref_to_sequence"],
        "structure_ref": ["resolve_ref_to_structure_or_pdb_id"],
        "cdr3_marker": ["use_redacted_marker_metadata"],
        "pubchem_cid": ["use_pubchem_cid"],
        "zinc_id": ["use_zinc_id"],
        "drugbank_id": ["use_drugbank_id"],
    }.get(value_kind, [])
    if material_type == "antibody_heavy_chain_sequence":
        transforms = [*transforms, "use_heavy_chain_sequence"]
    if material_type == "antibody_light_chain_sequence":
        transforms = [*transforms, "use_light_chain_sequence"]
    return sorted(set(transforms))


def _chain_role(material_type: str, role: Any) -> str | None:
    if material_type == "antibody_heavy_chain_sequence" or material_type == "antibody_heavy_cdr3_sequence":
        return "heavy"
    if material_type == "antibody_light_chain_sequence" or material_type == "antibody_light_cdr3_sequence":
        return "light"
    role_str = str(role or "").lower()
    if "heavy" in role_str:
        return "heavy"
    if "light" in role_str:
        return "light"
    return None


def _is_ref_shaped_material(material: dict[str, Any]) -> bool:
    value = str(material.get("value") or "")
    value_format = str(material.get("value_format") or "").lower()
    mt = str(material.get("material_type") or "")
    if mt in {"payload_smiles", "linker_smiles", "compound_smiles"}:
        return False
    if mt in {"structure_file", "structure_ref"}:
        return True
    if value_format in {"fasta", "pdb", "cif", "mmcif", "file", "path", "storage_ref"}:
        return True
    if "/" in value or "\\" in value:
        return True
    if re.search(r"\.(?:fa|fasta|faa|seq|pdb|cif|mmcif|ent)$", value, flags=re.IGNORECASE):
        return True
    return False


def _identifier_source_id(id_type: str, value: str) -> str:
    return f"{id_type}-{_sha_prefix(f'{id_type}:{value}')}"


def _sha_prefix(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:_SHA_PREFIX_LEN]
