"""Step 8 — structure_prediction_and_interface_results."""

from __future__ import annotations

from typing import Literal, Optional
from pydantic import BaseModel, Field

from .common import ToolCallRecord


class InterfaceMetrics(BaseModel):
    interface_area: Optional[float] = None
    solvation_energy: Optional[float] = None
    h_bond_count: Optional[int] = None
    salt_bridge_count: Optional[int] = None


class InterfaceFeature(BaseModel):
    chain_id_1: str
    chain_id_2: str
    interface_residues: list[str] = Field(default_factory=list)
    metrics: InterfaceMetrics = Field(default_factory=InterfaceMetrics)
    quality_flags: list[str] = Field(default_factory=list)


StructureConfidenceType = Literal[
    "prediction_confidence",
    "structure_quality",
    "interface_quality",
    "refinement_resolution",
    "crystal_density_validation",
    "unit_cell_consistency",
    "boltz_confidence_score",
    "ptm_score",
    "iptm_score",
    "complex_plddt",
    "complex_iplddt",
    "complex_pde",
    "complex_ipde",
    "chain_ptm",
    "pair_chain_iptm",
    "pae",
    "pde",
    "plddt",
    "iptm",
    "ptm",
    "unavailable",
    "other",
]


class StructureConfidenceRecord(BaseModel):
    confidence_type: StructureConfidenceType
    value: Optional[float] = None
    source: Optional[str] = None
    source_tool_call_id: Optional[str] = None


class StructureOutput(BaseModel):
    output_id: str
    storage_path: str
    structure_format: Literal["pdb", "cif"]
    source_tool_call_id: Optional[str] = None


StructureArtifactType = Literal[
    "predicted_complex_structure",
    "predicted_monomer_structure",
    "interface_analysis_raw_output",
    "normalized_interface_features",
    "structure_quality_report",
    "refinement_or_validation_report",
    "other",
]


class StructureOutputArtifact(BaseModel):
    """Step 8 `output_artifacts[]` entry — canonical (not just an id string)."""

    artifact_id: str
    related_candidate_id: Optional[str] = None
    related_structure_input_id: Optional[str] = None
    artifact_type: StructureArtifactType
    storage_ref: str
    storage_type: Literal["database_record", "s3_path", "local_run_storage", "other"] = "s3_path"
    content_type: Literal["json", "pdb", "cif", "mmcif", "text", "table", "other"] = "json"
    created_at: Optional[str] = None


class ComplexStructureRef(BaseModel):
    source_kind: Literal[
        "existing_pdb_complex",
        "uploaded_local_complex",
        "predicted_complex",
        "unknown",
    ]
    source_ref: Optional[str] = None
    storage_ref: Optional[str] = None
    pdb_id: Optional[str] = None
    structure_format: Literal["pdb", "cif", "mmcif", "unknown"] = "unknown"
    source_tool_call_id: Optional[str] = None
    confidence_summary: dict = Field(default_factory=dict)


class InterfaceAnalysisRecord(BaseModel):
    source_tool: Optional[str] = None
    source_tool_call_id: Optional[str] = None
    chain_pair: dict = Field(default_factory=dict)
    interface_residue_count: Optional[int] = None
    interface_area: Optional[float] = None
    h_bond_count: Optional[int] = None
    salt_bridge_count: Optional[int] = None
    quality_flags: list[str] = Field(default_factory=list)
    source_ref: Optional[str] = None


class Step8DownstreamHandoff(BaseModel):
    has_complex_structure: bool = False
    has_validated_structure: bool = False
    has_interface_features: bool = False
    structure_for_variant_generation_ref: Optional[str] = None
    validated_structure_ref: Optional[str] = None
    interface_quality_available: bool = False
    prediction_confidence_available: bool = False
    refinement_resolution_available: bool = False
    validation_available: bool = False
    missing_for_step9: list[str] = Field(default_factory=list)
    handoff_notes: list[str] = Field(default_factory=list)


class CandidateStructureResult(BaseModel):
    candidate_id: str
    structure_input_id: str
    run_case: Literal[
        "full_antigen_antibody_complex_prediction",
        "existing_complex_interface_evaluation",
        "monomer_or_partial_structure_preparation",
    ]
    run_status: Literal["ok", "partial", "failed"]
    partial_run_flag: bool = False
    structure_outputs: list[StructureOutput] = Field(default_factory=list)
    chain_mapping: list[dict] = Field(default_factory=list)
    interface_features: list[InterfaceFeature] = Field(default_factory=list)
    structure_confidence_records: list[StructureConfidenceRecord] = Field(default_factory=list)
    complex_structure_refs: list[ComplexStructureRef] = Field(default_factory=list)
    interface_analysis_records: list[InterfaceAnalysisRecord] = Field(default_factory=list)
    downstream_handoff: Step8DownstreamHandoff = Field(default_factory=Step8DownstreamHandoff)


class StructurePredictionAndInterfaceResults(BaseModel):
    run_id: str
    step_id: str = "step_08_structure_evaluation"
    created_at: str
    structure_modeling_status: Literal["ok", "partial", "failed"] = "partial"
    candidate_structure_results: list[CandidateStructureResult] = Field(default_factory=list)
    tool_call_records: list[ToolCallRecord] = Field(default_factory=list)
    output_artifacts: list[StructureOutputArtifact] = Field(default_factory=list)
    structure_modeling_notes: Optional[str] = None
