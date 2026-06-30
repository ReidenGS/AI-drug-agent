"""Step 7 — prepared_structure_input_package."""

from __future__ import annotations

from typing import Any, Literal, Optional
from pydantic import BaseModel, Field

from .common import ToolCallRecord


class StructureRef(BaseModel):
    pdb_id: Optional[str] = None
    file_id: Optional[str] = None
    structure_format: Optional[Literal["pdb", "cif", "mmcif"]] = None
    validation_status: Literal["valid", "invalid", "unknown"] = "unknown"
    source_kind: Literal[
        "uploaded_file", "pdb_id", "candidate_material", "predicted_needed", "unknown"
    ] = "unknown"
    source_ref: Optional[str] = None
    storage_ref: Optional[str] = None
    related_candidate_ids: list[str] = Field(default_factory=list)
    resource_binding_status: Literal["explicit", "inferred", "ambiguous", "unassigned"] = "unassigned"
    binding_confidence: float = 0.0


class SequenceRef(BaseModel):
    sequence_id: str
    chain_role: Optional[str] = None
    sequence: Optional[str] = None
    source_kind: Literal[
        "uploaded_fasta", "material_sequence", "uniprot_id", "unknown"
    ] = "unknown"
    source_ref: Optional[str] = None
    prediction_needed: bool = False
    sequence_storage_ref: Optional[str] = None
    sequence_value_status: Literal[
        "inline", "referenced", "identifier_only", "unavailable"
    ] = "unavailable"
    prediction_input_kind: Literal[
        "amino_acid_sequence", "fasta_ref", "uniprot_id", "unknown"
    ] = "unknown"
    related_candidate_ids: list[str] = Field(default_factory=list)
    resource_binding_status: Literal["explicit", "inferred", "ambiguous", "unassigned"] = "unassigned"
    binding_confidence: float = 0.0


class ChainMapping(BaseModel):
    chain_id: str
    chain_role: Literal["antigen", "antibody_heavy", "antibody_light", "payload", "linker", "other"]
    mapping_confidence: float = 0.0
    source: Literal["explicit", "inferred", "unknown"] = "unknown"
    source_ref: Optional[str] = None
    chain_id_kind: Literal["observed", "prediction_placeholder", "unknown"] = "unknown"


class CrystalMetadata(BaseModel):
    a: Optional[float] = None
    b: Optional[float] = None
    c: Optional[float] = None
    alpha: Optional[float] = None
    beta: Optional[float] = None
    gamma: Optional[float] = None
    space_group: Optional[str] = None
    z_value: Optional[int] = None
    source_kind: Optional[str] = None
    source_ref: Optional[str] = None
    parse_status: Literal["ok", "missing", "invalid", "unsupported"] = "missing"
    warnings: list[str] = Field(default_factory=list)


class MolecularWeightEstimate(BaseModel):
    value: Optional[float] = None
    unit: Literal["Da"] = "Da"
    method: Optional[str] = None
    status: Literal["estimated", "estimated_with_warnings", "missing", "unreliable"] = "missing"
    warnings: list[str] = Field(default_factory=list)
    source_kind: Optional[str] = None
    source_ref: Optional[str] = None


class StructureInputRecord(BaseModel):
    structure_input_id: str
    candidate_id: str
    input_case: Literal[
        "uploaded_structure_file",
        "known_pdb_id",
        "database_search_result",
        "sequence_only_input",
    ]
    structure_source: str
    assessment_intent: str
    structure_role: Literal["complex", "antigen_only", "antibody_only", "monomer"]
    structure_refs: list[StructureRef] = Field(default_factory=list)
    sequence_refs_for_prediction: list[SequenceRef] = Field(default_factory=list)
    chain_mapping: list[ChainMapping] = Field(default_factory=list)
    chain_pair_candidates: list[dict] = Field(default_factory=list)
    antigen_antibody_mapping: Optional[dict] = None
    residue_ranges: list[dict] = Field(default_factory=list)
    crystal_metadata: Optional[CrystalMetadata] = None
    molecular_weight_estimate: Optional[MolecularWeightEstimate] = None
    missing_metadata_flags: list[str] = Field(default_factory=list)
    preferred_input_rank: int = 0
    preferred_input_reason: Optional[str] = None
    prediction_required: bool = False
    source_priority_notes: list[str] = Field(default_factory=list)
    # Step 7 tool-call compact metadata for Step 8 consumer convenience.
    step7_tool_output_metadata: list[dict[str, Any]] = Field(default_factory=list)
    # Search tooling for ambiguous name-derived candidates should stay compact.
    database_search_candidates: list[dict[str, Any]] = Field(default_factory=list)


class PreparedStructureInputPackage(BaseModel):
    run_id: str
    step_id: str = "step_07_structure_input"
    created_at: str
    structure_preparation_status: Literal["ok", "partial", "failed"] = "partial"
    prepared_structure_inputs: list[StructureInputRecord] = Field(default_factory=list)
    structure_tool_call_records: list[ToolCallRecord] = Field(default_factory=list)
    structure_output_artifacts: list[str] = Field(default_factory=list)
    unresolved_resource_refs: list[dict] = Field(default_factory=list)
