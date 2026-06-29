"""StructureAndDesignAgent — Step 7, Step 8, Step 9.

This agent owns its own functional scope; it does not behave like a generic
tool-calling agent. The three entry points are:

- `run_step_7(run_id)` — assemble `prepared_structure_input_package` from
  Step 1/2/5 artifacts (uploaded PDB/CIF + FASTA, structured_query refs,
  candidate materials). One optional `RCSBData_get_entry` enrichment call per
  PDB id, by reference only.
- `run_step_8(run_id)` — per `StructureInputRecord`, route to a small subset
  consumes Step 7 prepared structure inputs and routes only `step_08`
  scoped tools for existing complex interface evaluation, structure
  validation/refinement lookup, and deferred complex-prediction audit.
  Raw outputs are referenced via `output_artifacts[]` and stored under
  `tool_outputs/step_08/{tool_call_id}.json`.
- `run_step_9(run_id)` — compound library screening for compound-component
  candidates. Routes to ZINC tools when SMILES, ZINC id, or compound name is
  present. **No record claims `ZINC22` confirmation**; `source_library` stays
  `"ZINC"` and `source_database_version` stays `"unknown"`.

Hard constraints (architecture v0.1):
- RFdiffusion `contigs_dsl` is NOT generated freely by this agent. Step 9
  protein-design lane is not part of the MVP; see TODO at bottom.
- All MCP calls go through the inventory-scoped client. Raw payloads NEVER
  appear inside normalized records.
"""

from __future__ import annotations

import json
import hashlib
import re
from io import StringIO
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any, Iterable, Optional

from ..agents.tool_selection_policy import (
    SelectionContext,
    ToolInvocationPlan,
    select_and_build_invocations,
)
from ..llm.provider import LLMProvider, MockLLMProvider
from ..mcp.client import MCPClient
from ..schemas.common import ToolCallRecord
from ..schemas.step_07_prepared_structure_input_package import (
    ChainMapping,
    PreparedStructureInputPackage,
    SequenceRef,
    StructureInputRecord,
    StructureRef,
)
from ..schemas.step_08_structure_prediction_and_interface_results import (
    CandidateStructureResult,
    ComplexPredictionPlan,
    ComplexStructureRef,
    InterfaceAnalysisRecord,
    InterfaceFeature,
    InterfaceMetrics,
    Step8DownstreamHandoff,
    StructureConfidenceRecord,
    StructureOutput,
    StructureOutputArtifact,
    StructurePredictionAndInterfaceResults,
)
from ..schemas.step_09_structure_variant_and_compound_screening import (
    CompoundHit,
    CompoundScreeningArtifact,
)
from ..services.artifact_registry_service import ArtifactRegistryService
from ..services.storage_service import Storage
from ..services.workflow_state_service import WorkflowStateService
from ..utils.errors import WorkflowStateError
from ..utils.ids import new_artifact_id, new_tool_call_id
from ..utils.time import now_iso


_AGENT_NAME = "structure_and_design_agent"
_STEP_07 = "step_07"
_STEP_08 = "step_08"
_STEP_09 = "step_09"

_PDB_EXTS = {".pdb", ".cif", ".mmcif", ".ent"}
_FASTA_EXTS = {".fasta", ".fa", ".faa", ".seq"}
_RESIDUE_RANGE_RE = re.compile(
    r"\bresidues?\s+(?P<start>\d{1,6})\s*[-–]\s*(?P<end>\d{1,6})\b",
    re.IGNORECASE,
)
_STEP7_SCOPED_TOOLS = (
    "PDBeSearch_search_structures",
    "RCSBAdvSearch_search_structures",
    "RCSBData_get_assembly",
    "RCSBData_get_entry",
    "SAbDab_get_structure",
    "alphafold_get_prediction",
)
_STEP8_SCOPED_TOOL_POLICY = {
    "CrystalStructure_validate": "uploaded_structure_validation",
    "get_refinement_resolution_by_pdb_id": "known_pdb_refinement_lookup",
    "PDBePISA_get_interfaces": "existing_complex_interface_evaluation",
    "NvidiaNIM_alphafold2_multimer": "future_complex_prediction",
    "NvidiaNIM_openfold3": "future_complex_prediction",
    "NvidiaNIM_boltz2": "future_complex_prediction",
    "dynamic_package_discovery": "infrastructure_not_scientific_output",
}
_STEP8_NIM_COMPLEX_TOOLS = {
    "NvidiaNIM_alphafold2_multimer",
    "NvidiaNIM_openfold3",
    "NvidiaNIM_boltz2",
}
_NAME_LIKE_MATERIAL_TYPES = {
    "target_antigen_name",
    "antibody_name",
    "complete_adc_name",
    "antigen_name",
    "candidate_name",
    "antibody_candidate_name",
    "target_name",
    "antigen_candidate_name",
}


# ── shared utilities ────────────────────────────────────────────────────────

def _format_for_file(filename: str) -> str:
    ext = PurePosixPath(filename or "").suffix.lower()
    if ext in {".cif", ".mmcif"}:
        return "cif"
    return "pdb"


def _looks_like_pdb_id(value: Any) -> bool:
    return isinstance(value, str) and bool(re.fullmatch(r"[0-9][A-Za-z0-9]{3}", value.strip()))


def _looks_like_file_backed_sequence(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    return PurePosixPath(value).suffix.lower() in _FASTA_EXTS


def _materials_by_type(candidate: dict, types: Iterable[str]) -> list[dict]:
    types = set(types)
    return [m for m in (candidate.get("materials") or []) if m.get("material_type") in types]


def _identifiers_by_type(candidate: dict, types: Iterable[str]) -> list[dict]:
    types = set(types)
    return [
        i for i in (candidate.get("identifiers") or [])
        if i.get("id_type") in types
    ]


def _structure_storage_ref(storage: Storage, value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    value = value.strip()
    if not value:
        return None
    if Path(value).is_file() or storage.exists(value):
        return value
    return None


def _pdb_path_like(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    return PurePosixPath(value.strip()).suffix.lower() in _PDB_EXTS


def _is_concrete_pdb_path(storage: Storage, value: Any) -> bool:
    if not isinstance(value, str):
        return False
    if not _pdb_path_like(value):
        return False
    try:
        return Path(value).is_file() or storage.exists(value)
    except Exception:  # noqa: BLE001
        return False


# ── agent ───────────────────────────────────────────────────────────────────

class StructureAndDesignAgent:
    name = _AGENT_NAME

    def __init__(
        self,
        *,
        storage: Storage,
        registry: ArtifactRegistryService,
        workflow_state: WorkflowStateService,
        mcp_client: MCPClient,
        llm: LLMProvider | None = None,
    ) -> None:
        self.storage = storage
        self.registry = registry
        self.workflow_state = workflow_state
        self.mcp_client = mcp_client
        self.llm = llm or MockLLMProvider()

    # ── Step 7 ──────────────────────────────────────────────────────────────
    def run_step_7(self, run_id: str) -> PreparedStructureInputPackage:
        reg = self.registry.get(run_id)
        if not reg.active_artifacts.candidate_context_table_id:
            raise WorkflowStateError("Step 7 requires Step 5 candidate_context_table")
        if not reg.active_artifacts.run_step_plan_id:
            raise WorkflowStateError("Step 7 requires Step 4 run_step_plan")

        raw = self.storage.read_json(
            self.storage.run_key(run_id, "inputs/raw_request_record.json")
        )
        sq = self.storage.read_json(
            self.storage.run_key(run_id, "inputs/structured_query.json")
        )
        cct = self.storage.read_json(
            self.storage.run_key(run_id, "candidate_context_table.json")
        )

        # Group resources once.
        uploaded = raw.get("uploaded_files") or []
        structure_files = [
            f for f in uploaded
            if PurePosixPath(f.get("original_filename", "")).suffix.lower() in _PDB_EXTS
        ]
        sequence_files = [
            f for f in uploaded
            if PurePosixPath(f.get("original_filename", "")).suffix.lower() in _FASTA_EXTS
        ]
        refs_by_type: dict[str, list[dict]] = {}
        for r in sq.get("referenced_inputs") or []:
            if isinstance(r, dict):
                refs_by_type.setdefault(r.get("id_type", ""), []).append(r)

        tool_call_records: list[ToolCallRecord] = []
        prepared: list[StructureInputRecord] = []
        candidates = [
            c for c in (cct.get("candidate_records") or [])
            if c.get("candidate_type") in {"target_antigen", "antibody", "adc_construct"}
        ]
        bound_resources, unresolved_resource_refs = _bind_step7_resources(
            candidates=candidates,
            structure_files=structure_files,
            sequence_files=sequence_files,
            refs_by_type=refs_by_type,
        )

        for candidate in candidates:
            resources = bound_resources.get(candidate.get("candidate_id"), {})
            record = self._build_structure_input_record(
                candidate=candidate,
                structure_files=resources.get("structure_files") or [],
                sequence_files=resources.get("sequence_files") or [],
                refs_by_type=resources.get("refs_by_type") or {},
            )
            if record is None:
                continue
            calls = self._route_step7_scoped_tools(
                run_id=run_id,
                candidate=candidate,
                record=record,
            )
            tool_call_records.extend(calls)
            prepared.append(record)
        _attach_antigen_antibody_mapping(
            [r for r in prepared if r.input_case != "database_search_result"],
            candidates,
        )

        prep_status: str
        if not prepared:
            prep_status = "failed"
        elif unresolved_resource_refs or any(r.missing_metadata_flags for r in prepared):
            prep_status = "partial"
        else:
            prep_status = "ok"

        pkg = PreparedStructureInputPackage(
            run_id=run_id,
            created_at=now_iso(),
            structure_preparation_status=prep_status,  # type: ignore[arg-type]
            prepared_structure_inputs=prepared,
            structure_tool_call_records=tool_call_records,
            structure_output_artifacts=[
                tc.tool_output_artifact_id for tc in tool_call_records if tc.tool_output_artifact_id
            ],
            unresolved_resource_refs=unresolved_resource_refs,
        )

        artifact_id = new_artifact_id("prepared_structure_input_package")
        self.storage.write_json(
            self.storage.run_key(run_id, "prepared_structure_input_package.json"),
            {"artifact_id": artifact_id, **pkg.model_dump()},
        )
        self.registry.update_active(run_id, prepared_structure_input_package_id=artifact_id)
        self.workflow_state.mark(run_id, "step_07", "completed")
        return pkg

    def _build_structure_input_record(
        self,
        *,
        candidate: dict,
        structure_files: list[dict],
        sequence_files: list[dict],
        refs_by_type: dict[str, list[dict]],
    ) -> Optional[StructureInputRecord]:
        candidate_id = candidate.get("candidate_id", "unknown")
        ctype = candidate.get("candidate_type")
        materials = candidate.get("materials") or []

        # Structure signals on the candidate itself.
        cand_structure_mats = _materials_by_type(candidate, {"structure_file"})
        cand_structure_ref_mats = _materials_by_type(candidate, {"structure_ref"})
        cand_structure_file_refs = [
            m for m in cand_structure_ref_mats
            if not _looks_like_pdb_id(m.get("value"))
        ]
        cand_structure_mats.extend(cand_structure_file_refs)
        cand_pdb_material_refs = [
            m for m in cand_structure_ref_mats
            if _looks_like_pdb_id(m.get("value"))
        ]
        cand_pdb_ids = _identifiers_by_type(candidate, {"pdb_id"})
        cand_uniprot_ids = _identifiers_by_type(candidate, {"uniprot_id"})
        cand_sequence_mats = _materials_by_type(
            candidate,
            {"antibody_heavy_chain_sequence", "antibody_light_chain_sequence", "target_sequence"},
        )
        bound_sequence_paths = {
            f.get("storage_path") for f in sequence_files if f.get("storage_path")
        }
        cand_sequence_mats = [
            m for m in cand_sequence_mats
            if not _looks_like_file_backed_sequence(m.get("value"))
            or m.get("value") in bound_sequence_paths
        ]

        # Top-level uploads. For target/antibody we accept any uploaded structure
        # file as a candidate-attached structure ref; explicit pairing is the
        # Step 8 evaluator's concern.
        has_structure_file = bool(cand_structure_mats) or bool(structure_files)
        has_pdb_id = bool(cand_pdb_material_refs) or bool(cand_pdb_ids) or bool(refs_by_type.get("pdb_id"))
        has_sequence = bool(cand_sequence_mats) or bool(sequence_files) or bool(
            cand_uniprot_ids or refs_by_type.get("uniprot_id")
        )
        has_name_input = bool(_names_from_materials(materials))
        input_case = None
        structure_source = None
        preferred_rank = 0
        preferred_reason = "no structure signals available"
        if not (has_structure_file or has_pdb_id or has_sequence):
            if has_name_input:
                input_case = "database_search_result"
                structure_source = "name_only"
                preferred_rank = 5
                preferred_reason = "name-based structure search fallback when no structure/PDB/sequence is available"
            else:
                return None

        # Decide input_case in deterministic source-priority order.
        if input_case == "database_search_result":
            pass
        elif structure_files:
            input_case = "uploaded_structure_file"
            structure_source = "user_uploaded"
            preferred_rank = 1
            preferred_reason = "uploaded PDB/CIF structure file preferred over candidate/PDB ID/sequence inputs"
        elif cand_structure_mats:
            input_case = "uploaded_structure_file"
            structure_source = "candidate_material"
            preferred_rank = 2
            preferred_reason = "candidate material structure_file/structure_ref preferred over PDB ID/sequence inputs"
        elif cand_pdb_material_refs:
            input_case = "known_pdb_id"
            structure_source = "candidate_material"
            preferred_rank = 2
            preferred_reason = "candidate material structure_ref preferred over candidate/Step 2 PDB ID/sequence inputs"
        elif cand_pdb_ids:
            input_case = "known_pdb_id"
            structure_source = "candidate_identifier"
            preferred_rank = 3
            preferred_reason = "candidate pdb_id preferred over Step 2 PDB ID/sequence inputs"
        elif refs_by_type.get("pdb_id"):
            input_case = "known_pdb_id"
            structure_source = "structured_query.referenced_inputs"
            preferred_rank = 4
            preferred_reason = "Step 2 referenced pdb_id preferred over sequence-only inputs"
        else:
            input_case = "sequence_only_input"
            structure_source = "fasta_or_uniprot"
            preferred_rank = 5
            preferred_reason = "no structure file or PDB ID available; preparing sequence-only prediction input"

        structure_refs: list[StructureRef] = []
        def _has_pdb_id(value: Any) -> bool:
            if not value:
                return False
            needle = str(value).lower()
            return any((s.pdb_id or "").lower() == needle for s in structure_refs)

        for m in cand_structure_mats:
            source_ref = m.get("material_id") or m.get("value")
            storage_ref = _structure_storage_ref(self.storage, m.get("value"))
            structure_refs.append(
                StructureRef(
                    pdb_id=None,
                    file_id=None,
                    structure_format=_format_for_file(m.get("value_format") or ""),  # type: ignore[arg-type]
                    validation_status="unknown",
                    source_kind="candidate_material",
                    source_ref=source_ref,
                    storage_ref=storage_ref,
                    related_candidate_ids=[candidate_id],
                    resource_binding_status="explicit",
                    binding_confidence=1.0,
                )
            )
        for f in structure_files:
            storage_ref = f.get("storage_path") if _pdb_path_like(f.get("storage_path")) else None
            structure_refs.append(
                StructureRef(
                    pdb_id=None,
                    file_id=f.get("file_id"),
                    structure_format=_format_for_file(f.get("original_filename", "")),  # type: ignore[arg-type]
                    validation_status="unknown",
                    source_kind="uploaded_file",
                    source_ref=f.get("storage_path") or f.get("file_id"),
                    storage_ref=storage_ref,
                    related_candidate_ids=[candidate_id],
                    resource_binding_status=f.get("resource_binding_status", "inferred"),
                    binding_confidence=float(f.get("binding_confidence", 0.6)),
                )
            )
        for ident in cand_pdb_ids:
            value = ident.get("id_value")
            if not value or _has_pdb_id(value):
                continue
            structure_refs.append(
                StructureRef(
                    pdb_id=value,
                    structure_format="pdb",
                    validation_status="unknown",
                    source_kind="pdb_id",
                    source_ref=value,
                    related_candidate_ids=[candidate_id],
                    resource_binding_status="explicit",
                    binding_confidence=1.0,
                )
            )
        for material in cand_pdb_material_refs:
            value = material.get("value")
            if value and not _has_pdb_id(value):
                storage_ref = _structure_storage_ref(self.storage, value)
                structure_refs.append(StructureRef(
                    pdb_id=value,
                    structure_format="pdb",
                    validation_status="unknown",
                    source_kind="candidate_material",
                    source_ref=material.get("material_id") or value,
                    storage_ref=storage_ref,
                    related_candidate_ids=[candidate_id],
                    resource_binding_status="explicit",
                    binding_confidence=1.0,
                ))
        for ref in refs_by_type.get("pdb_id", []):
            value = ref.get("value")
            if value and not _has_pdb_id(value):
                structure_refs.append(
                    StructureRef(
                        pdb_id=value,
                        structure_format="pdb",
                        validation_status="unknown",
                        source_kind="pdb_id",
                        source_ref=value,
                        storage_ref=None,
                        related_candidate_ids=[candidate_id],
                        resource_binding_status=ref.get("resource_binding_status", "inferred"),
                        binding_confidence=float(ref.get("binding_confidence", 0.6)),
                    )
                )

        sequence_refs: list[SequenceRef] = []
        prediction_required = input_case == "sequence_only_input"
        for m in cand_sequence_mats:
            material_id = m.get("material_id", new_artifact_id("seq"))
            sequence_refs.append(
                SequenceRef(
                    sequence_id=material_id,
                    chain_role=_chain_role_from_material(m.get("material_type", "")),
                    sequence=str(m.get("value") or "") or None,
                    source_kind="material_sequence",
                    source_ref=material_id,
                    prediction_needed=prediction_required,
                    sequence_value_status="inline",
                    prediction_input_kind="amino_acid_sequence",
                    related_candidate_ids=[candidate_id],
                    resource_binding_status="explicit",
                    binding_confidence=1.0,
                )
            )
        for f in sequence_files:
            sequence_id = f.get("file_id", new_artifact_id("seq"))
            sequence_refs.append(
                SequenceRef(
                    sequence_id=sequence_id,
                    chain_role=_chain_role_from_fasta_file(f, ctype),
                    sequence=None,
                    source_kind="uploaded_fasta",
                    source_ref=f.get("storage_path") or sequence_id,
                    prediction_needed=prediction_required,
                    sequence_storage_ref=f.get("storage_path") or None,
                    sequence_value_status="referenced",
                    prediction_input_kind="fasta_ref",
                    related_candidate_ids=[candidate_id],
                    resource_binding_status=f.get("resource_binding_status", "inferred"),
                    binding_confidence=float(f.get("binding_confidence", 0.6)),
                )
            )
        for ident in cand_uniprot_ids:
            uniprot = ident.get("id_value", "uniprot")
            sequence_refs.append(
                SequenceRef(
                    sequence_id=uniprot,
                    chain_role="antigen" if ctype == "target_antigen" else None,
                    sequence=None,
                    source_kind="uniprot_id",
                    source_ref=uniprot,
                    prediction_needed=prediction_required,
                    sequence_value_status="identifier_only",
                    prediction_input_kind="uniprot_id",
                    related_candidate_ids=[candidate_id],
                    resource_binding_status="explicit",
                    binding_confidence=1.0,
                )
            )
        if ctype == "target_antigen":
            for ref in refs_by_type.get("uniprot_id", []):
                uniprot = ref.get("value")
                if uniprot and not any(s.sequence_id == uniprot for s in sequence_refs):
                    sequence_refs.append(
                        SequenceRef(
                            sequence_id=uniprot,
                            chain_role="antigen",
                            sequence=None,
                            source_kind="uniprot_id",
                            source_ref=uniprot,
                            prediction_needed=prediction_required,
                            sequence_value_status="identifier_only",
                            prediction_input_kind="uniprot_id",
                            related_candidate_ids=[candidate_id],
                            resource_binding_status=ref.get("resource_binding_status", "inferred"),
                            binding_confidence=float(ref.get("binding_confidence", 0.6)),
                        )
                    )

        structure_role = (
            "complex" if ctype == "adc_construct"
            else "antigen_only" if ctype == "target_antigen"
            else "antibody_only" if ctype == "antibody"
            else "monomer"
        )
        assessment_intent = (
            "antigen_antibody_interface_evaluation"
            if structure_role == "complex"
            else f"{structure_role}_structure_assessment"
        )

        missing_flags: list[str] = []
        source_priority_notes = [preferred_reason]
        if input_case == "uploaded_structure_file" and not structure_refs:
            missing_flags.append("uploaded_structure_file_present_but_no_ref")
        if input_case == "known_pdb_id" and not any(s.pdb_id for s in structure_refs):
            missing_flags.append("pdb_id_referenced_but_no_structure_ref")
        if input_case == "sequence_only_input" and not sequence_refs:
            missing_flags.append("sequence_only_input_but_no_sequence_ref")
        observed_chain_mapping, observed_ranges = _observed_structure_metadata(
            storage=self.storage,
            structure_files=structure_files,
            candidate_structure_materials=cand_structure_mats,
        )
        chain_mapping = observed_chain_mapping or _chain_mapping_from_sequence_refs(sequence_refs)
        if input_case in {"uploaded_structure_file", "known_pdb_id"} and not chain_mapping:
            missing_flags.append("chain_ids_missing")
        if observed_chain_mapping and all(cm.chain_role == "other" for cm in observed_chain_mapping):
            missing_flags.append("chain_roles_unknown")
        if input_case == "sequence_only_input" and sequence_refs and not chain_mapping:
            missing_flags.append("chain_mapping_incomplete")

        residue_ranges = _extract_residue_ranges(candidate, refs_by_type)
        for observed_range in observed_ranges:
            if observed_range not in residue_ranges:
                residue_ranges.append(observed_range)
        if _partial_input_hint_present(candidate, refs_by_type) and not residue_ranges:
            missing_flags.append("residue_range_missing_for_partial_input")

        return StructureInputRecord(
            structure_input_id=new_artifact_id("structure_input"),
            candidate_id=candidate_id,
            input_case=input_case,  # type: ignore[arg-type]
            structure_source=structure_source,
            assessment_intent=assessment_intent,
            structure_role=structure_role,  # type: ignore[arg-type]
            structure_refs=structure_refs,
            sequence_refs_for_prediction=sequence_refs,
            chain_mapping=chain_mapping,
            chain_pair_candidates=[],
            antigen_antibody_mapping=None,
            residue_ranges=residue_ranges,
            missing_metadata_flags=missing_flags,
            preferred_input_rank=preferred_rank,
            preferred_input_reason=preferred_reason,
            prediction_required=prediction_required,
            source_priority_notes=source_priority_notes,
        )

    # ── Step 8 ──────────────────────────────────────────────────────────────
    def run_step_8(self, run_id: str) -> StructurePredictionAndInterfaceResults:
        reg = self.registry.get(run_id)
        if not reg.active_artifacts.prepared_structure_input_package_id:
            raise WorkflowStateError("Step 8 requires Step 7 prepared_structure_input_package")

        pkg = self.storage.read_json(
            self.storage.run_key(run_id, "prepared_structure_input_package.json")
        )
        inputs = pkg.get("prepared_structure_inputs") or []

        tool_calls: list[ToolCallRecord] = []
        output_artifacts: list[StructureOutputArtifact] = []
        candidate_results: list[CandidateStructureResult] = []

        any_partial = False
        any_failed = False

        for sin in inputs:
            input_case = sin.get("input_case")
            structure_input_id = sin.get("structure_input_id")
            candidate_id = sin.get("candidate_id")

            confidence_records: list[StructureConfidenceRecord] = []
            structure_outputs: list[StructureOutput] = []
            interface_features: list[InterfaceFeature] = []
            interface_analysis_records: list[InterfaceAnalysisRecord] = []
            complex_structure_refs: list[ComplexStructureRef] = []
            complex_prediction_plans: list[ComplexPredictionPlan] = []
            run_status = "ok"
            partial = False

            routed_calls = self._route_step8_scoped_tools(run_id, sin)
            selected_calls = [
                tc for tc in routed_calls
                if tc.tool_input_summary
                and tc.tool_input_summary.get("routing_decision") == "selected"
            ]
            if not selected_calls:
                if input_case == "uploaded_structure_file":
                    if not any(
                        tc.tool_name == "structure_input_missing_ref"
                        for tc in routed_calls
                    ):
                        routed_calls.append(
                            _nonexecuted_tool_record(
                                tool_name="structure_input_missing_ref",
                                agent_name=_AGENT_NAME,
                                step_id=_STEP_08,
                                run_status="skipped",
                                summary={
                                    "label": f"step08:{structure_input_id}:structure_input",
                                    "candidate_id": candidate_id,
                                    "input_case": input_case,
                                    "routing_decision": "not_applicable",
                                    "reason": "no_usable_structure_reference",
                                    "structure_refs": _compact_structure_refs_for_audit(sin.get("structure_refs") or []),
                                },
                            )
                        )
                    run_status = "partial"
                    partial = True
                elif input_case == "known_pdb_id":
                    run_status = "partial"
                    partial = True

            for tc in routed_calls:
                tool_calls.append(tc)
                plan = _complex_prediction_plan_from_tool_call(tc)
                if plan:
                    complex_prediction_plans.append(plan)
                if tc.run_status == "success":
                    conf_type = _confidence_type_for_step8_tool(tc.tool_name)
                    confidence_records.append(
                        StructureConfidenceRecord(
                            confidence_type=conf_type,  # type: ignore[arg-type]
                            value=_extract_confidence_value(tc.tool_name, tc),
                            source=tc.tool_name,
                            source_tool_call_id=tc.tool_call_id,
                        )
                    )
                    interface_features.extend(
                        _extract_interface_features_for_step8(self.storage, tc)
                    )
                    interface_analysis_records.extend(
                        _extract_interface_analysis_records_for_step8(self.storage, tc)
                    )
                    complex_structure_refs.extend(
                        _extract_complex_structure_refs_for_step8(self.storage, sin, tc)
                    )
                    if tc.tool_output_artifact_id and tc.tool_output_ref:
                        output_artifacts.append(
                            StructureOutputArtifact(
                                artifact_id=tc.tool_output_artifact_id,
                                related_candidate_id=candidate_id,
                                related_structure_input_id=structure_input_id,
                                artifact_type=_artifact_type_for_tool(tc.tool_name),  # type: ignore[arg-type]
                                storage_ref=tc.tool_output_ref,
                                storage_type="local_run_storage",
                                content_type="json",
                                created_at=tc.finished_at,
                            )
                        )
                elif _step8_tool_call_affects_partial(tc):
                    partial = True
                    if tc.run_status == "failed":
                        any_failed = True

            if partial:
                run_status = "partial"
                any_partial = True

            candidate_results.append(
                CandidateStructureResult(
                    candidate_id=candidate_id,
                    structure_input_id=structure_input_id,
                    run_case=_run_case_from_input_case(input_case),  # type: ignore[arg-type]
                    run_status=run_status,  # type: ignore[arg-type]
                    partial_run_flag=partial,
                    structure_outputs=structure_outputs,
                    chain_mapping=sin.get("chain_mapping") or [],
                    interface_features=interface_features,
                    structure_confidence_records=confidence_records,
                    complex_structure_refs=_dedupe_complex_structure_refs(complex_structure_refs),
                    interface_analysis_records=interface_analysis_records,
                    downstream_handoff=_build_step8_downstream_handoff(
                        complex_structure_refs=_dedupe_complex_structure_refs(complex_structure_refs),
                        interface_features=interface_features,
                        interface_analysis_records=interface_analysis_records,
                        confidence_records=confidence_records,
                        tool_calls=routed_calls,
                        complex_prediction_plans=complex_prediction_plans,
                    ),
                    complex_prediction_plans=complex_prediction_plans,
                    complex_prediction_input_status=_summarize_prediction_input_status(complex_prediction_plans),
                    missing_prediction_inputs=_summarize_missing_prediction_inputs(complex_prediction_plans),
                    prediction_runtime_status=_summarize_prediction_runtime_status(complex_prediction_plans),
                    prediction_tool_contract_notes=_summarize_prediction_contract_notes(complex_prediction_plans),
                )
            )

        if not candidate_results:
            modeling_status = "failed"
        elif any_failed or any_partial:
            modeling_status = "partial"
        else:
            modeling_status = "ok"

        results = StructurePredictionAndInterfaceResults(
            run_id=run_id,
            created_at=now_iso(),
            structure_modeling_status=modeling_status,  # type: ignore[arg-type]
            candidate_structure_results=candidate_results,
            tool_call_records=tool_calls,
            output_artifacts=output_artifacts,
            structure_modeling_notes=(
                "Step 8 ran in MVP mode; tool wrappers may return mocked data "
                "(`status='mocked'`). Raw payloads are referenced via "
                "output_artifacts[].storage_ref."
            ),
        )

        artifact_id = new_artifact_id("structure_prediction_and_interface_results")
        self.storage.write_json(
            self.storage.run_key(run_id, "structure_prediction_and_interface_results.json"),
            {"artifact_id": artifact_id, **results.model_dump()},
        )
        self.registry.update_active(
            run_id, structure_prediction_and_interface_results_id=artifact_id
        )
        self.workflow_state.mark(run_id, "step_08", "completed")
        return results

    def _step8_scoped_tools(self) -> tuple[str, ...]:
        runtime_scoped = self.mcp_client.list_tools(
            agent_name=_AGENT_NAME, step_id=_STEP_08
        )
        return tuple(runtime_scoped)

    def _route_step8_scoped_tools(
        self, run_id: str, sin: dict
    ) -> list[ToolCallRecord]:
        input_case = sin.get("input_case")
        structure_input_id = sin.get("structure_input_id")
        candidate_id = sin.get("candidate_id")
        calls: list[ToolCallRecord] = []

        def _summary_base() -> dict[str, Any]:
            return {
                "label": f"step08:{structure_input_id}",
                "candidate_id": candidate_id,
                "input_case": input_case,
                "run_case": _run_case_from_input_case(input_case),
            }

        try:
            scoped_tools = set(self._step8_scoped_tools())
        except Exception as e:
            return [
                _nonexecuted_tool_record(
                    tool_name=tool_name,
                    agent_name=_AGENT_NAME,
                    step_id=_STEP_08,
                    run_status="skipped",
                    summary={
                        **_summary_base(),
                        "routing_decision": "scope_unavailable",
                        "reason": f"step_8 scope introspection failed: {str(e)}",
                    },
                )
                for tool_name in _STEP8_SCOPED_TOOL_POLICY
            ]

        for tool_name in _STEP8_SCOPED_TOOL_POLICY:
            if tool_name not in scoped_tools:
                calls.append(
                    _nonexecuted_tool_record(
                        tool_name=tool_name,
                        agent_name=_AGENT_NAME,
                        step_id=_STEP_08,
                        run_status="skipped",
                        summary={
                            **_summary_base(),
                            "routing_decision": "scope_unavailable",
                            "reason": "tool not available from MCP runtime scope",
                        },
                    )
                )

        def _skip(tool_name: str, reason: str, *, status: str = "skipped") -> None:
            calls.append(
                _nonexecuted_tool_record(
                    tool_name=tool_name,
                    agent_name=_AGENT_NAME,
                    step_id=_STEP_08,
                    run_status=status,
                    summary={
                        **_summary_base(),
                        "routing_decision": "not_applicable",
                        "reason": reason,
                    },
                )
            )

        def _run(tool_name: str, kwargs: dict[str, Any], reason: str) -> None:
            calls.append(
                self._call_tool(
                    run_id=run_id,
                    step_id=_STEP_08,
                    tool_name=tool_name,
                    kwargs=kwargs,
                    output_dir="step_08",
                    label=f"step08:{structure_input_id}:{tool_name}",
                    extra_input_summary={
                        **_summary_base(),
                        "routing_decision": "selected",
                        "routing_reason": reason,
                        "arguments": {k: _short(v) for k, v in kwargs.items()},
                    },
                )
            )

        pdb_id = _step8_pdb_id(sin.get("structure_refs") or [])
        local_ref = _step8_local_structure_ref(self.storage, sin.get("structure_refs") or [])
        if "CrystalStructure_validate" in scoped_tools:
            if input_case == "uploaded_structure_file" and local_ref:
                _run(
                    "CrystalStructure_validate",
                    {"pdb_id_or_path": local_ref["value"]},
                    "uploaded/local PDB or CIF validation by concrete storage_ref path",
                )
            else:
                _skip(
                    "CrystalStructure_validate",
                    "requires uploaded/local PDB or CIF storage_ref; Step 8 will not fabricate path/cell parameters",
                )

        if "get_refinement_resolution_by_pdb_id" in scoped_tools:
            if input_case == "known_pdb_id" and pdb_id:
                _run(
                    "get_refinement_resolution_by_pdb_id",
                    {"pdb_id": pdb_id},
                    "known PDB ID refinement metadata lookup",
                )
            else:
                _skip(
                    "get_refinement_resolution_by_pdb_id",
                    "requires real PDB ID; uploaded file paths are not valid pdb_id values",
                )

        if "PDBePISA_get_interfaces" in scoped_tools:
            if input_case == "known_pdb_id" and pdb_id:
                _run(
                    "PDBePISA_get_interfaces",
                    {"pdb_id": pdb_id},
                    "existing complex interface evaluation for real PDB ID",
                )
            else:
                _skip(
                    "PDBePISA_get_interfaces",
                    "requires real PDB ID for an existing complex; uploaded paths are not sent as pdb_id",
                )

        for tool_name in sorted(_STEP8_NIM_COMPLEX_TOOLS):
            if tool_name not in scoped_tools:
                continue
            plan = _plan_step8_nim_complex_prediction(tool_name, sin)
            if plan.input_status == "selected_but_deferred":
                calls.append(
                    _nonexecuted_tool_record(
                        tool_name=tool_name,
                        agent_name=_AGENT_NAME,
                        step_id=_STEP_08,
                        run_status="dependency_unavailable",
                        summary={
                            **_summary_base(),
                            "routing_decision": "selected_but_deferred",
                            "reason": "NvidiaNIM complex prediction input appears sufficient, but wrapper/runtime is deferred",
                            "complex_prediction_plan": plan.model_dump(),
                        },
                    )
                )
            else:
                calls.append(
                    _nonexecuted_tool_record(
                        tool_name=tool_name,
                        agent_name=_AGENT_NAME,
                        step_id=_STEP_08,
                        run_status="skipped",
                        summary={
                            **_summary_base(),
                            "routing_decision": plan.input_status,
                            "reason": "; ".join(plan.contract_notes) or plan.input_status,
                            "complex_prediction_plan": plan.model_dump(),
                        },
                    )
                )

        if "dynamic_package_discovery" in scoped_tools:
            _skip(
                "dynamic_package_discovery",
                "infrastructure discovery tool is scoped but not a Step 8 scientific prediction/evaluation output",
            )

        return calls

    # ── Step 7 scope / output compacting helpers ───────────────────────────
    def _step7_scoped_tools(self) -> tuple[str, ...]:
        """Return the effective scoped toolset for Step 7 from MCP runtime.

        The routing policy is intentionally defined centrally in
        `_STEP7_SCOPED_TOOLS`; we intersect it with the MCP-scoped tool list so
        runtime drift is surfaced via drift-fence tests.
        """
        runtime_scoped = self.mcp_client.list_tools(
            agent_name=_AGENT_NAME, step_id=_STEP_07
        )
        return tuple(runtime_scoped)

    # ── Step 7 scoped tool routing ───────────────────────────────────────
    def _route_step7_scoped_tools(
        self, run_id: str, candidate: dict, record: StructureInputRecord
    ) -> list[ToolCallRecord]:
        calls: list[ToolCallRecord] = []
        try:
            scoped_tools = set(self._step7_scoped_tools())
        except Exception as e:
            summary_base = {
                "label": f"step07:{record.structure_input_id}",
                "candidate_id": record.candidate_id,
                "candidate_type": candidate.get("candidate_type"),
                "input_case": record.input_case,
            }
            for tool_name in _STEP7_SCOPED_TOOLS:
                calls.append(
                    _skipped_tool_record(
                        tool_name=tool_name,
                        agent_name=_AGENT_NAME,
                        step_id=_STEP_07,
                        summary={
                            **summary_base,
                            "routing_decision": "scope_unavailable",
                            "reason": f"step_7 scope introspection failed: {str(e)}",
                        },
                    )
                )
            return calls
        input_case = record.input_case
        candidate_id = record.candidate_id
        structure_input_id = record.structure_input_id

        def _summary_base() -> dict[str, Any]:
            return {
                "label": f"step07:{structure_input_id}",
                "candidate_id": candidate_id,
                "candidate_type": candidate.get("candidate_type"),
                "input_case": input_case,
            }

        # Keep a deterministic coverage of the routing table; if runtime scope is
        # missing entries, make the omission explicit as skipped audit entries.
        for tool_name in _STEP7_SCOPED_TOOLS:
            if tool_name not in scoped_tools:
                calls.append(
                    _skipped_tool_record(
                        tool_name=tool_name,
                        agent_name=_AGENT_NAME,
                        step_id=_STEP_07,
                        summary={
                            **_summary_base(),
                            "routing_decision": "scope_unavailable",
                            "reason": "tool not available from MCP runtime scope",
                            "scope": {"step_7_scoped_tools": sorted(_STEP7_SCOPED_TOOLS)},
                        },
                    )
                )
        if not scoped_tools:
            return calls

        def _skip(tool_name: str, reason: str) -> None:
            calls.append(
                _skipped_tool_record(
                    tool_name=tool_name,
                    agent_name=_AGENT_NAME,
                    step_id=_STEP_07,
                    summary={
                        **_summary_base(),
                        "routing_decision": "not_applicable",
                        "reason": reason,
                    },
                )
            )

        def _run_and_record(tool_name: str, kwargs: dict[str, Any], decision: str) -> None:
            tc = self._call_tool(
                run_id=run_id,
                step_id=_STEP_07,
                tool_name=tool_name,
                kwargs=kwargs,
                output_dir="step_07",
                label=f"{input_case}:{structure_input_id}:{tool_name}",
                extra_input_summary={
                    **_summary_base(),
                    "routing_decision": decision,
                    "arguments": {k: _short(v) for k, v in kwargs.items()},
                },
            )
            calls.append(tc)
            _apply_step7_tool_output_metadata(
                storage=self.storage,
                record=record,
                tool_call=tc,
                tool_name=tool_name,
            )

        if input_case == "uploaded_structure_file":
            for name in _STEP7_SCOPED_TOOLS:
                if name not in scoped_tools:
                    continue
                if name in {"RCSBData_get_entry", "RCSBData_get_assembly"}:
                    _skip(name, "uploaded structure inputs use local parser and optional file-scoped metadata only")
                elif name in {"RCSBAdvSearch_search_structures", "PDBeSearch_search_structures"}:
                    _skip(name, "no local/known-PDB fallback path for fully uploaded structure input")
                elif name == "SAbDab_get_structure":
                    _skip(name, "SAbDab routing requires explicit antibody PDB-id path")
                else:
                    _skip(name, "sequence-only prediction route not used for uploaded structure input")
            return calls

        if input_case == "known_pdb_id":
            pdb_ref = next((s for s in record.structure_refs if s.pdb_id), None)
            if pdb_ref and pdb_ref.pdb_id:
                if "RCSBData_get_entry" in scoped_tools:
                    _run_and_record("RCSBData_get_entry", {"pdb_id": pdb_ref.pdb_id}, "selected")
                if "RCSBData_get_assembly" in scoped_tools:
                    _run_and_record(
                        "RCSBData_get_assembly",
                        {"pdb_id": pdb_ref.pdb_id, "assembly_id": "1"},
                        "selected",
                    )
                if record.structure_role == "antibody_only" and "SAbDab_get_structure" in scoped_tools:
                    _run_and_record(
                        "SAbDab_get_structure",
                        {"pdb_id": pdb_ref.pdb_id},
                        "selected",
                    )
                elif record.structure_role != "antibody_only":
                    _skip("SAbDab_get_structure", "only applies to antibody role inputs")
            else:
                if "RCSBData_get_entry" in scoped_tools:
                    _skip("RCSBData_get_entry", "known_pdb_id input case requires explicit PDB ID")
                if "RCSBData_get_assembly" in scoped_tools:
                    _skip("RCSBData_get_assembly", "known_pdb_id input case requires explicit PDB ID")
                if "SAbDab_get_structure" in scoped_tools:
                    _skip("SAbDab_get_structure", "known_pdb_id input case requires explicit PDB ID")
            if "RCSBAdvSearch_search_structures" in scoped_tools:
                _skip("RCSBAdvSearch_search_structures", "explicit PDB ID route has precedence over search")
            if "PDBeSearch_search_structures" in scoped_tools:
                _skip("PDBeSearch_search_structures", "explicit PDB ID route has precedence over search")
            return calls

        if input_case == "sequence_only_input":
            uniprot_sequence_ref = next(
                (
                    r for r in record.sequence_refs_for_prediction
                    if r.prediction_input_kind == "uniprot_id" and r.source_ref
                ),
                None,
            )
            for name in _STEP7_SCOPED_TOOLS:
                if name not in scoped_tools:
                    continue
                if name in {"RCSBData_get_entry", "RCSBData_get_assembly", "SAbDab_get_structure"}:
                    _skip(name, "sequence-only input uses UniProt prediction path, no PDB ID")
                elif name == "alphafold_get_prediction":
                    if uniprot_sequence_ref:
                        _run_and_record(
                            "alphafold_get_prediction",
                            {"uniprot": uniprot_sequence_ref.source_ref or uniprot_sequence_ref.sequence_id},
                            "selected",
                        )
                    else:
                        _skip(
                            name,
                            "sequence-only input has no UniProt accession for AlphaFold prediction lookup",
                        )
                else:
                    _skip(
                        name,
                        "sequence-only route uses local/sequence metadata unless explicit UniProt/ID is present",
                    )
            return calls

        if input_case == "database_search_result":
            query = _structure_query_from_candidate(candidate, record)
            if query:
                if "RCSBAdvSearch_search_structures" in scoped_tools:
                    _run_and_record("RCSBAdvSearch_search_structures", {"query": query}, "selected")
                if "PDBeSearch_search_structures" in scoped_tools:
                    _run_and_record("PDBeSearch_search_structures", {"query": query}, "selected")
            else:
                if "RCSBAdvSearch_search_structures" in scoped_tools:
                    _skip(
                        "RCSBAdvSearch_search_structures",
                        "name-only input has no usable query material",
                    )
                if "PDBeSearch_search_structures" in scoped_tools:
                    _skip(
                        "PDBeSearch_search_structures",
                        "name-only input has no usable query material",
                    )
            if "RCSBData_get_entry" in scoped_tools:
                _skip(
                    "RCSBData_get_entry",
                    "database search handles identity inference, not exact PDB ID path",
                )
            if "RCSBData_get_assembly" in scoped_tools:
                _skip(
                    "RCSBData_get_assembly",
                    "database search handles identity inference, not exact assembly path",
                )
            if "SAbDab_get_structure" in scoped_tools:
                _skip(
                    "SAbDab_get_structure",
                    "database search handles identity inference, not exact antibody-PDB path",
                )
            return calls

        for tool_name in _STEP7_SCOPED_TOOLS:
            if tool_name not in scoped_tools:
                continue
            _skip(tool_name, f"unhandled Step 7 input_case={input_case}")
        return calls
    # ── Step 9 ──────────────────────────────────────────────────────────────
    def run_step_9(self, run_id: str) -> CompoundScreeningArtifact:
        reg = self.registry.get(run_id)
        if not reg.active_artifacts.candidate_context_table_id:
            raise WorkflowStateError("Step 9 requires Step 5 candidate_context_table")

        cct = self.storage.read_json(
            self.storage.run_key(run_id, "candidate_context_table.json")
        )
        compound_candidates = [
            c for c in cct.get("candidate_records") or []
            if c.get("candidate_type") == "compound_component"
        ]

        tool_calls: list[ToolCallRecord] = []
        hits: list[CompoundHit] = []
        any_real_attempt = False
        any_partial = False

        for cand in compound_candidates:
            smiles_mats = _materials_by_type(cand, {"payload_smiles", "linker_smiles", "compound_smiles"})
            name_mats = _materials_by_type(cand, {"payload_name", "linker_name", "compound_name"})
            zinc_idents = _identifiers_by_type(cand, {"zinc_id"})
            chembl_idents = _identifiers_by_type(cand, {"chembl_id"})
            pubchem_idents = _identifiers_by_type(cand, {"pubchem_cid"})

            if not (smiles_mats or name_mats or zinc_idents or chembl_idents or pubchem_idents):
                continue

            context = _compound_selection_context(cand)
            plans = select_and_build_invocations(
                agent_name=_AGENT_NAME,
                step_id=_STEP_09,
                mcp_client=self.mcp_client,
                llm=self.llm,
                context=context,
                deterministic_fallback=lambda c=cand: _compound_fallback_plans(c),
                deterministic_argument_mapping=_compound_argument_mapping,
            )
            if not plans:
                plans = _compound_fallback_plans(cand)

            for plan in plans:
                if plan.validation_status == "skipped":
                    tc = _skipped_tool_record(
                        tool_name=plan.tool_name,
                        agent_name=_AGENT_NAME,
                        step_id=_STEP_09,
                        summary={
                            "label": f"compound:{cand.get('candidate_label')}",
                            **_selection_summary(plan),
                        },
                    )
                    tool_calls.append(tc)
                    any_partial = True
                    continue
                tc = self._call_tool(
                    run_id=run_id,
                    step_id=_STEP_09,
                    tool_name=plan.tool_name,
                    kwargs=plan.arguments,
                    output_dir="step_09",
                    label=f"compound:{cand.get('candidate_label')}",
                    extra_input_summary=_selection_summary(plan),
                )
                tool_calls.append(tc)
                any_real_attempt = True
                if tc.run_status == "success":
                    hits.append(_compound_hit_from_call(cand, tc, plan.arguments, plan.tool_name))
                else:
                    any_partial = True

        if not any_real_attempt:
            screening_status = "skipped"
        elif any_partial or not hits:
            screening_status = "partial"
        else:
            screening_status = "ok"

        artifact = CompoundScreeningArtifact(
            run_id=run_id,
            created_at=now_iso(),
            screening_status=screening_status,  # type: ignore[arg-type]
            compound_hits=hits,
            tool_call_records=tool_calls,
        )

        artifact_id = new_artifact_id("compound_screening_artifact")
        self.storage.write_json(
            self.storage.run_key(run_id, "compound_screening_artifact.json"),
            {"artifact_id": artifact_id, **artifact.model_dump()},
        )
        self.registry.update_active(
            run_id, structure_variant_and_compound_screening_id=artifact_id
        )
        self.workflow_state.mark(run_id, "step_09", "completed")
        return artifact

    # ── shared tool dispatch ────────────────────────────────────────────────
    def _call_tool(
        self,
        *,
        run_id: str,
        step_id: str,
        tool_name: str,
        kwargs: dict[str, Any],
        output_dir: str,
        label: str,
        extra_input_summary: Optional[dict[str, Any]] = None,
    ) -> ToolCallRecord:
        tc_id = new_tool_call_id()
        started = now_iso()
        result = self.mcp_client.call_tool(
            agent_name=_AGENT_NAME, step_id=step_id, tool_name=tool_name, **kwargs
        )
        finished = now_iso()

        output_ref = None
        output_artifact_id = None
        if "payload" in result:
            output_artifact_id = new_artifact_id("tool_output")
            output_key = self.storage.run_key(
                run_id, "tool_outputs", output_dir, f"{tc_id}.json"
            )
            self.storage.write_json(
                output_key,
                {
                    "tool_call_id": tc_id,
                    "tool_name": tool_name,
                    "label": label,
                    "input": kwargs,
                    "output": result["payload"],
                },
            )
            output_ref = output_key

        return ToolCallRecord(
            tool_call_id=tc_id,
            tool_name=tool_name,
            agent_name=_AGENT_NAME,
            step_id=step_id,
            run_status=result.get("run_status", "pending"),
            started_at=started,
            finished_at=finished,
            tool_input_summary={
                "label": label,
                **{k: _short(v) for k, v in kwargs.items()},
                **(extra_input_summary or {}),
            },
            tool_output_artifact_id=output_artifact_id,
            tool_output_ref=output_ref,
            error_message=result.get("error_message"),
        )


# ── module-level helpers ────────────────────────────────────────────────────

def _bind_step7_resources(
    *, candidates: list[dict], structure_files: list[dict],
    sequence_files: list[dict], refs_by_type: dict[str, list[dict]],
) -> tuple[dict[str, dict[str, Any]], list[dict]]:
    bound = {
        c["candidate_id"]: {"structure_files": [], "sequence_files": [], "refs_by_type": {}}
        for c in candidates if c.get("candidate_id")
    }
    by_type: dict[str, list[str]] = {}
    for candidate in candidates:
        if candidate.get("candidate_id"):
            by_type.setdefault(candidate.get("candidate_type", ""), []).append(candidate["candidate_id"])
    unresolved: list[dict] = []

    def _one_or_none(ids: list[str], confidence: float) -> tuple[list[str], str, float]:
        deduped = list(dict.fromkeys(ids))
        if len(deduped) == 1:
            return deduped, "inferred", confidence
        if deduped:
            return [], "ambiguous", 0.0
        return [], "unassigned", 0.0

    def compatible_ids(resource: dict, resource_kind: str) -> tuple[list[str], str, float]:
        explicit_ids = _explicit_resource_candidate_ids(resource, set(bound))
        if explicit_ids:
            return explicit_ids, "explicit", 1.0
        role = _resource_role(resource, resource_kind)
        if role in {"target", "antigen"}:
            return _one_or_none(by_type.get("target_antigen", []), 0.8)
        if role in {"antibody", "antibody_heavy", "antibody_light"}:
            return _one_or_none(by_type.get("antibody", []), 0.8)
        if role == "complex":
            return _one_or_none(by_type.get("adc_construct", []), 0.7)
        if resource_kind in {"sequence"}:
            return _one_or_none([*by_type.get("target_antigen", []), *by_type.get("antibody", [])], 0.5)
        if resource_kind in {"uniprot_id"}:
            return _one_or_none(by_type.get("target_antigen", []), 0.5)
        if resource_kind in {"structure", "pdb_id"}:
            ids = [
                *by_type.get("target_antigen", []),
                *by_type.get("antibody", []),
                *by_type.get("adc_construct", []),
            ]
            return _one_or_none(ids, 0.5)
        return [], "unassigned", 0.0

    def bind_file(resource: dict, kind: str) -> None:
        ids, status, confidence = compatible_ids(resource, kind)
        if not ids:
            unresolved.append(_unresolved_resource_summary(resource, kind, status))
            return
        for candidate_id in ids:
            item = dict(resource)
            item.update(resource_binding_status=status, binding_confidence=confidence, related_candidate_ids=ids)
            bound[candidate_id]["structure_files" if kind == "structure" else "sequence_files"].append(item)

    for resource in structure_files:
        bind_file(resource, "structure")
    for resource in sequence_files:
        bind_file(resource, "sequence")
    for id_type, refs in refs_by_type.items():
        for ref in refs:
            if id_type not in {"pdb_id", "uniprot_id"} and not _reference_has_range(ref):
                continue
            ids, status, confidence = compatible_ids(ref, id_type)
            if not ids:
                unresolved.append(_unresolved_resource_summary(ref, id_type, status))
                continue
            for candidate_id in ids:
                item = dict(ref)
                item.update(resource_binding_status=status, binding_confidence=confidence, related_candidate_ids=ids)
                bound[candidate_id]["refs_by_type"].setdefault(id_type, []).append(item)
    return bound, unresolved


def _names_from_materials(materials: list[dict]) -> list[str]:
    names: list[str] = []
    for material in materials:
        if not isinstance(material, dict):
            continue
        if material.get("material_type") not in _NAME_LIKE_MATERIAL_TYPES:
            continue
        value = (material.get("value") or "").strip()
        if value:
            names.append(value)
    return names


def _structure_query_from_candidate(candidate: dict, record: StructureInputRecord) -> str:
    names = _names_from_materials(candidate.get("materials") or [])
    if names:
        return names[0]
    candidate_type = record.structure_role
    if candidate_type == "complex":
        return record.structure_input_id.replace("-", " ").replace("_", " ")
    return record.candidate_id


def _explicit_resource_candidate_ids(resource: dict, valid_ids: set[str]) -> list[str]:
    values = [resource.get("candidate_id"), resource.get("related_candidate_id")]
    if isinstance(resource.get("related_candidate_ids"), list):
        values.extend(resource["related_candidate_ids"])
    return list(dict.fromkeys(str(v) for v in values if v and str(v) in valid_ids))


def _resource_role(resource: dict, resource_kind: str) -> str | None:
    for key in ("chain_role", "role", "source_role"):
        if isinstance(resource.get(key), str) and resource[key].strip():
            return resource[key].strip().lower()
    filename = (resource.get("original_filename") or "").lower()
    if any(marker in filename for marker in ("heavy", "_vh", "-vh", "_hc", "-hc")):
        return "antibody_heavy"
    if any(marker in filename for marker in ("light", "_vl", "-vl", "_lc", "-lc")):
        return "antibody_light"
    if any(marker in filename for marker in ("antigen", "target", "her2", "erbb2")):
        return "antigen"
    if "antibody" in filename or "trastuzumab" in filename:
        return "antibody"
    if "complex" in filename and resource_kind == "structure":
        return "complex"
    return None


def _unresolved_resource_summary(resource: dict, kind: str, status: str) -> dict:
    return {
        "resource_type": kind,
        "source_ref": resource.get("file_id") or resource.get("value") or resource.get("original_filename") or "unknown",
        "resource_binding_status": status,
        "reason": "no unique compatible candidate binding",
    }


def _reference_has_range(ref: dict) -> bool:
    return any(key in ref for key in ("residue_range", "start", "end", "residue_start", "residue_end", "range")) or bool(
        _RESIDUE_RANGE_RE.search(str(ref.get("value") or ""))
    )


def _observed_structure_metadata(
    *, storage: Storage, structure_files: list[dict],
    candidate_structure_materials: list[dict],
) -> tuple[list[ChainMapping], list[dict]]:
    mappings: list[ChainMapping] = []
    ranges: list[dict] = []
    resources = list(structure_files)
    resources.extend({
        "storage_path": material.get("value"),
        "chain_id": material.get("chain_id"),
        "chain_role": material.get("chain_role") or material.get("role"),
        "chain_roles": material.get("chain_roles") or {},
    } for material in candidate_structure_materials)
    seen: set[tuple[str, str]] = set()
    for resource in resources:
        source_ref = resource.get("storage_path")
        if not source_ref:
            continue
        explicit_roles = resource.get("chain_roles") or {}
        for chain_id, start, end in _parse_structure_chain_summary(storage, source_ref):
            key = (str(source_ref), chain_id)
            if key in seen:
                continue
            seen.add(key)
            explicit_role = explicit_roles.get(chain_id)
            if not explicit_role and resource.get("chain_id") == chain_id:
                explicit_role = resource.get("chain_role")
            role = explicit_role if explicit_role in {"antigen", "antibody_heavy", "antibody_light"} else "other"
            mappings.append(ChainMapping(
                chain_id=chain_id,
                chain_role=role,  # type: ignore[arg-type]
                mapping_confidence=1.0 if explicit_role else 0.0,
                source="explicit" if explicit_role else "unknown",
                source_ref=str(source_ref),
                chain_id_kind="observed",
            ))
            if start is not None and end is not None:
                ranges.append({
                    "chain_id": chain_id, "start": start, "end": end,
                    "source": "observed_structure", "source_ref": str(source_ref),
                })
    return mappings, ranges


def _parse_structure_chain_summary(
    storage: Storage, source_ref: str,
) -> list[tuple[str, int | None, int | None]]:
    try:
        path = Path(source_ref)
        if path.is_file():
            data = path.read_bytes()
        elif storage.exists(source_ref):
            data = storage.read_bytes(source_ref)
        else:
            return []
        text = data.decode("utf-8", errors="replace")
        suffix = PurePosixPath(source_ref).suffix.lower()
        fallback = _parse_structure_chain_ranges_from_text(text)
        if fallback:
            return fallback

        from Bio.PDB import MMCIFParser, PDBParser
        parser = MMCIFParser(QUIET=True) if suffix in {".cif", ".mmcif"} else PDBParser(QUIET=True)
        structure = parser.get_structure("step7_input", StringIO(text))
        model = next(structure.get_models(), None)
        if model is None:
            return []
        out: list[tuple[str, int | None, int | None]] = []
        for chain in model:
            residue_numbers = [
                residue.id[1] for residue in chain
                if isinstance(residue.id[1], int) and residue.id[0] == " "
            ]
            out.append((
                str(chain.id),
                min(residue_numbers) if residue_numbers else None,
                max(residue_numbers) if residue_numbers else None,
            ))
        return out
    except Exception:  # noqa: BLE001
        return []


def _parse_structure_chain_ranges_from_text(text: str) -> list[tuple[str, int | None, int | None]]:
    """Extract chain IDs + residue ranges from raw ATOM/HETATM records."""
    chain_ranges: dict[str, list[int]] = {}
    for line in text.splitlines():
        if not line.startswith("ATOM"):
            continue
        if len(line) < 27:
            continue
        chain_id = line[21].strip()
        if not chain_id or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9]{0,4}", chain_id):
            continue
        try:
            residue_no = int(line[22:26].strip())
        except ValueError:
            continue
        chain_ranges.setdefault(chain_id, []).append(residue_no)
    out: list[tuple[str, int | None, int | None]] = []
    for chain_id, residues in chain_ranges.items():
        if not residues:
            continue
        out.append((chain_id, min(residues), max(residues)))
    return out


def _chain_role_from_material(material_type: str) -> Optional[str]:
    return {
        "antibody_heavy_chain_sequence": "antibody_heavy",
        "antibody_light_chain_sequence": "antibody_light",
        "target_sequence": "antigen",
    }.get(material_type)


def _chain_role_from_fasta_file(file_record: dict, candidate_type: str | None) -> str:
    name = (file_record.get("original_filename") or "").lower()
    if any(marker in name for marker in ("light", "_l", "-l", "vl", "lc")):
        return "antibody_light"
    if any(marker in name for marker in ("heavy", "_h", "-h", "vh", "hc")):
        return "antibody_heavy"
    if candidate_type == "target_antigen":
        return "antigen"
    if candidate_type == "antibody":
        return "antibody_heavy"
    return "antigen"


def _chain_mapping_from_sequence_refs(sequence_refs: list[SequenceRef]) -> list[ChainMapping]:
    out: list[ChainMapping] = []
    placeholders = {
        "antigen": "predicted_antigen",
        "antibody_heavy": "predicted_antibody_heavy",
        "antibody_light": "predicted_antibody_light",
    }
    seen: set[str] = set()
    for ref in sequence_refs:
        role = ref.chain_role
        if role not in placeholders or role in seen:
            continue
        seen.add(role)
        out.append(
            ChainMapping(
                chain_id=placeholders[role],
                chain_role=role,  # type: ignore[arg-type]
                mapping_confidence=0.5,
                source="inferred",
                source_ref=ref.sequence_id,
                chain_id_kind="prediction_placeholder",
            )
        )
    return out


def _attach_antigen_antibody_mapping(
    prepared: list[StructureInputRecord],
    candidates: list[dict],
) -> None:
    records = {r.candidate_id: r for r in prepared}
    targets = [c for c in candidates if c.get("candidate_type") == "target_antigen" and c.get("candidate_id") in records]
    antibodies = [c for c in candidates if c.get("candidate_type") == "antibody" and c.get("candidate_id") in records]
    if not targets or not antibodies:
        return

    explicit_pairs: set[tuple[str, str]] = set()
    for candidate in candidates:
        candidate_id = candidate.get("candidate_id")
        related = candidate.get("related_candidate_ids") or []
        if candidate.get("related_candidate_id"):
            related = [*related, candidate.get("related_candidate_id")]
        for related_id in related:
            if candidate_id in records and related_id in records:
                types = {next((c.get("candidate_type") for c in candidates if c.get("candidate_id") == x), None) for x in (candidate_id, related_id)}
                if types == {"target_antigen", "antibody"}:
                    target_id = candidate_id if next(c for c in candidates if c.get("candidate_id") == candidate_id).get("candidate_type") == "target_antigen" else related_id
                    antibody_id = related_id if target_id == candidate_id else candidate_id
                    explicit_pairs.add((target_id, antibody_id))

    pairs: list[tuple[str, str, bool]] = [(t, a, True) for t, a in sorted(explicit_pairs)]
    if not pairs and len(targets) == 1 and len(antibodies) == 1:
        pairs = [(targets[0]["candidate_id"], antibodies[0]["candidate_id"], False)]
    elif not pairs:
        # Multiple candidates with no typed relationship: expose only pair
        # candidates involving each record, never a shared first-pair mapping.
        for target in targets:
            for antibody in antibodies:
                pair = {
                    "target_candidate_id": target["candidate_id"],
                    "antibody_candidate_id": antibody["candidate_id"],
                    "mapping_status": "ambiguous",
                }
                records[target["candidate_id"]].chain_pair_candidates.append(pair)
                records[antibody["candidate_id"]].chain_pair_candidates.append(pair)
        return

    for target_id, antibody_id, explicit in pairs:
        target_record = records[target_id]
        antibody_record = records[antibody_id]
        antigen_refs = [s.sequence_id for s in target_record.sequence_refs_for_prediction if s.chain_role == "antigen"]
        heavy_refs = [s.sequence_id for s in antibody_record.sequence_refs_for_prediction if s.chain_role == "antibody_heavy"]
        light_refs = [s.sequence_id for s in antibody_record.sequence_refs_for_prediction if s.chain_role == "antibody_light"]
        if target_record.prediction_required or antibody_record.prediction_required:
            status = "sequence_only_prediction_needed"
        elif not explicit:
            status = "ambiguous"
        elif antigen_refs and heavy_refs and light_refs:
            status = "complete"
        elif not (target_record.chain_mapping and antibody_record.chain_mapping):
            status = "missing_chain_ids"
        else:
            status = "partial"
        mapping = {
            "target_candidate_id": target_id,
            "antibody_candidate_id": antibody_id,
            "antigen_sequence_ids": antigen_refs,
            "antibody_heavy_sequence_ids": heavy_refs,
            "antibody_light_sequence_ids": light_refs,
            "mapping_status": status,
            "relationship_source": "explicit" if explicit else "ambiguous",
        }
        pair = {"target_candidate_id": target_id, "antibody_candidate_id": antibody_id, "mapping_status": status}
        for record in (target_record, antibody_record):
            record.antigen_antibody_mapping = mapping
            record.chain_pair_candidates.append(pair)


def _extract_residue_ranges(candidate: dict, refs_by_type: dict[str, list[dict]]) -> list[dict]:
    ranges: list[dict] = []

    def add_range(start: Any, end: Any, source: str, source_ref: str | None = None) -> None:
        try:
            s = int(start)
            e = int(end)
        except (TypeError, ValueError):
            return
        if s <= 0 or e < s:
            return
        item = {
            "start": s,
            "end": e,
            "source": source,
        }
        if source_ref:
            item["source_ref"] = source_ref
        if item not in ranges:
            ranges.append(item)

    def scan_obj(obj: dict, source: str) -> None:
        source_ref = obj.get("material_id") or obj.get("source") or obj.get("id_type")
        if isinstance(obj.get("residue_range"), dict):
            rr = obj["residue_range"]
            add_range(rr.get("start") or rr.get("residue_start"), rr.get("end") or rr.get("residue_end"), source, source_ref)
        if obj.get("start") is not None or obj.get("end") is not None:
            add_range(obj.get("start"), obj.get("end"), source, source_ref)
        if obj.get("residue_start") is not None or obj.get("residue_end") is not None:
            add_range(obj.get("residue_start"), obj.get("residue_end"), source, source_ref)
        text_values = []
        for key in ("range", "residue_range", "value", "notes", "context"):
            value = obj.get(key)
            if isinstance(value, str):
                text_values.append(value)
        for text in text_values:
            match = _RESIDUE_RANGE_RE.search(text)
            if match:
                add_range(match.group("start"), match.group("end"), source, source_ref)

    for material in candidate.get("materials") or []:
        if isinstance(material, dict):
            scan_obj(material, "candidate_material")
    for ref_list in refs_by_type.values():
        for ref in ref_list:
            if isinstance(ref, dict):
                scan_obj(ref, "referenced_input")
    return ranges


def _partial_input_hint_present(candidate: dict, refs_by_type: dict[str, list[dict]]) -> bool:
    text_parts: list[str] = []
    for material in candidate.get("materials") or []:
        if isinstance(material, dict):
            text_parts.extend(
                str(material.get(k) or "")
                for k in ("value", "notes", "context")
            )
    for ref_list in refs_by_type.values():
        for ref in ref_list:
            if isinstance(ref, dict):
                text_parts.extend(str(ref.get(k) or "") for k in ("value", "source", "notes", "context"))
    text = " ".join(text_parts).lower()
    return any(marker in text for marker in ("partial", "fragment", "domain", "residues"))


def _run_case_from_input_case(input_case: str | None) -> str:
    if input_case == "uploaded_structure_file":
        return "existing_complex_interface_evaluation"
    if input_case == "known_pdb_id":
        return "existing_complex_interface_evaluation"
    return "monomer_or_partial_structure_preparation"


def _extract_confidence_value(tool_name: str, tc: ToolCallRecord) -> float | None:
    # MVP: tool outputs are stored by reference; we don't peek into raw bodies
    # to extract numbers here. Confidence values stay None unless a future
    # tool returns a normalized scalar (Step 8 schema allows None).
    return None


def _confidence_type_for_step8_tool(tool_name: str) -> str:
    if tool_name == "PDBePISA_get_interfaces":
        return "interface_quality"
    if tool_name == "get_refinement_resolution_by_pdb_id":
        return "refinement_resolution"
    if tool_name == "CrystalStructure_validate":
        return "structure_quality"
    if tool_name in _STEP8_NIM_COMPLEX_TOOLS:
        return "prediction_confidence"
    return "other"


def _step8_tool_call_affects_partial(tc: ToolCallRecord) -> bool:
    if tc.run_status == "success":
        return False
    summary = tc.tool_input_summary or {}
    routing_decision = summary.get("routing_decision")
    if tc.run_status in {"failed", "dependency_unavailable", "partial"}:
        return True
    if routing_decision == "scope_unavailable":
        return True
    if routing_decision == "selected":
        return True
    return False


def _plan_step8_nim_complex_prediction(tool_name: str, sin: dict) -> ComplexPredictionPlan:
    input_case = sin.get("input_case")
    if input_case == "known_pdb_id":
        return ComplexPredictionPlan(
            tool_name=tool_name,
            input_status="not_applicable",
            runtime_status="not_applicable",
            can_invoke=False,
            structure_inputs=_compact_prediction_structure_inputs(sin),
            contract_notes=["existing PDB/interface route is preferred; complex prediction not needed"],
        )

    if input_case == "uploaded_structure_file" and _step8_has_explicit_complex_evidence(sin):
        return ComplexPredictionPlan(
            tool_name=tool_name,
            input_status="not_applicable",
            runtime_status="not_applicable",
            can_invoke=False,
            structure_inputs=_compact_prediction_structure_inputs(sin),
            contract_notes=["uploaded/local structure already has explicit complex chain evidence"],
        )

    sequence_inputs = _compact_prediction_sequence_inputs(sin)
    mapping = sin.get("antigen_antibody_mapping") or {}
    antigen_ids = list(mapping.get("antigen_sequence_ids") or [])
    heavy_ids = list(mapping.get("antibody_heavy_sequence_ids") or [])
    light_ids = list(mapping.get("antibody_light_sequence_ids") or [])

    roles = {entry.get("chain_role") for entry in sequence_inputs}
    if not antigen_ids:
        antigen_ids = [entry["sequence_id"] for entry in sequence_inputs if entry.get("chain_role") == "antigen"]
    if not heavy_ids:
        heavy_ids = [entry["sequence_id"] for entry in sequence_inputs if entry.get("chain_role") == "antibody_heavy"]
    if not light_ids:
        light_ids = [entry["sequence_id"] for entry in sequence_inputs if entry.get("chain_role") == "antibody_light"]

    missing: list[str] = []
    if not antigen_ids:
        missing.append("antigen_sequence")
    if not heavy_ids and "antibody" not in roles:
        missing.append("antibody_heavy_sequence")
    if not light_ids and "antibody" not in roles:
        missing.append("antibody_light_sequence")

    if missing:
        return ComplexPredictionPlan(
            tool_name=tool_name,
            input_status="input_missing",
            runtime_status="not_checked",
            can_invoke=False,
            missing_prediction_inputs=missing,
            sequence_inputs=sequence_inputs,
            structure_inputs=_compact_prediction_structure_inputs(sin),
            contract_notes=["missing antigen-antibody pair sequence input for complex prediction"],
        )

    return ComplexPredictionPlan(
        tool_name=tool_name,
        input_status="selected_but_deferred",
        runtime_status="runtime_unavailable",
        can_invoke=False,
        sequence_inputs=sequence_inputs,
        structure_inputs=_compact_prediction_structure_inputs(sin),
        contract_notes=[
            "antigen and antibody sequence refs are available",
            "NvidiaNIM wrapper/runtime/API-key contract is deferred",
        ],
    )


def _complex_prediction_plan_from_tool_call(tc: ToolCallRecord) -> ComplexPredictionPlan | None:
    summary = tc.tool_input_summary or {}
    raw = summary.get("complex_prediction_plan")
    if not isinstance(raw, dict):
        return None
    return ComplexPredictionPlan.model_validate(raw)


def _compact_prediction_sequence_inputs(sin: dict) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for seq in sin.get("sequence_refs_for_prediction") or []:
        if not isinstance(seq, dict):
            continue
        item = {
            "sequence_id": seq.get("sequence_id"),
            "chain_role": seq.get("chain_role"),
            "prediction_input_kind": seq.get("prediction_input_kind"),
            "source_kind": seq.get("source_kind"),
            "source_ref": seq.get("source_ref"),
            "sequence_value_status": seq.get("sequence_value_status"),
            "resource_binding_status": seq.get("resource_binding_status"),
        }
        sequence = seq.get("sequence")
        if isinstance(sequence, str) and sequence:
            item["sequence_length"] = len(sequence)
            item["sha256_prefix"] = hashlib.sha256(sequence.encode("utf-8")).hexdigest()[:12]
        out.append({k: v for k, v in item.items() if v not in (None, "", [])})
    return out


def _compact_prediction_structure_inputs(sin: dict) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for sref in sin.get("structure_refs") or []:
        if not isinstance(sref, dict):
            continue
        out.append(
            {
                k: v for k, v in {
                    "source_kind": sref.get("source_kind"),
                    "source_ref": sref.get("source_ref"),
                    "pdb_id": sref.get("pdb_id"),
                    "structure_format": sref.get("structure_format"),
                    "resource_binding_status": sref.get("resource_binding_status"),
                }.items() if v not in (None, "", [])
            }
        )
    return out


def _artifact_type_for_tool(tool_name: str) -> str:
    if tool_name == "CrystalStructure_validate":
        return "refinement_or_validation_report"
    if tool_name == "get_refinement_resolution_by_pdb_id":
        return "refinement_or_validation_report"
    if tool_name == "PDBePISA_get_interfaces":
        return "interface_analysis_raw_output"
    if tool_name in _STEP8_NIM_COMPLEX_TOOLS:
        return "predicted_complex_structure"
    return "other"


def _step8_local_structure_ref(storage: Storage, structure_refs: list[Any]) -> dict[str, Any] | None:
    for sref in structure_refs:
        if not isinstance(sref, dict):
            continue
        for key in ("storage_ref", "source_ref"):
            value = sref.get(key)
            if _is_concrete_pdb_path(storage, value):
                return {"value": value, "source": key}
    return None


def _step8_pdb_id(structure_refs: list[Any]) -> str | None:
    for sref in structure_refs:
        if isinstance(sref, dict) and sref.get("pdb_id"):
            value = str(sref["pdb_id"]).strip()
            if _looks_like_pdb_id(value):
                return value
    return None


def _extract_interface_features_for_step8(
    storage: Storage, tool_call: ToolCallRecord
) -> list[InterfaceFeature]:
    if tool_call.tool_name != "PDBePISA_get_interfaces":
        return []
    payload = _read_tool_output_payload(storage, tool_call)
    if not isinstance(payload, dict):
        return []
    raw_interfaces = payload.get("interfaces")
    if not isinstance(raw_interfaces, list):
        return []

    features: list[InterfaceFeature] = []
    for item in raw_interfaces[:25]:
        if not isinstance(item, dict):
            continue
        chain_a = _extract_scalar(item, ("chain_id_1", "chain_1", "chain_a", "chainId1"))
        chain_b = _extract_scalar(item, ("chain_id_2", "chain_2", "chain_b", "chainId2"))
        if not chain_a or not chain_b:
            chains = _extract_chain_ids(item)
            if len(chains) >= 2:
                chain_a, chain_b = chains[0], chains[1]
        if not chain_a or not chain_b:
            continue
        residues = item.get("interface_residues") or item.get("residues") or []
        if not isinstance(residues, list):
            residues = []
        features.append(
            InterfaceFeature(
                chain_id_1=str(chain_a),
                chain_id_2=str(chain_b),
                interface_residues=[_short(str(r)) for r in residues[:100]],
                metrics=InterfaceMetrics(
                    interface_area=_float_or_none(
                        item.get("interface_area") or item.get("area")
                    ),
                    solvation_energy=_float_or_none(
                        item.get("solvation_energy") or item.get("solvationEnergy")
                    ),
                    h_bond_count=_int_or_none(
                        item.get("h_bond_count") or item.get("hbonds")
                    ),
                    salt_bridge_count=_int_or_none(
                        item.get("salt_bridge_count") or item.get("salt_bridges")
                    ),
                ),
                quality_flags=[],
            )
        )
    return features


def _extract_interface_analysis_records_for_step8(
    storage: Storage, tool_call: ToolCallRecord
) -> list[InterfaceAnalysisRecord]:
    if tool_call.tool_name != "PDBePISA_get_interfaces":
        return []
    payload = _read_tool_output_payload(storage, tool_call)
    if not isinstance(payload, dict):
        return []
    raw_interfaces = payload.get("interfaces")
    if not isinstance(raw_interfaces, list):
        return []

    records: list[InterfaceAnalysisRecord] = []
    source_ref = _extract_scalar(payload, ("pdb_id", "source_ref", "query"))
    for item in raw_interfaces[:25]:
        if not isinstance(item, dict):
            continue
        chain_a = _extract_scalar(item, ("chain_id_1", "chain_1", "chain_a", "chainId1"))
        chain_b = _extract_scalar(item, ("chain_id_2", "chain_2", "chain_b", "chainId2"))
        if not chain_a or not chain_b:
            chains = _extract_chain_ids(item)
            if len(chains) >= 2:
                chain_a, chain_b = chains[0], chains[1]
        residues = item.get("interface_residues") or item.get("residues") or []
        residue_count = len(residues) if isinstance(residues, list) else None
        records.append(
            InterfaceAnalysisRecord(
                source_tool=tool_call.tool_name,
                source_tool_call_id=tool_call.tool_call_id,
                chain_pair={
                    k: v for k, v in {
                        "chain_id_1": str(chain_a) if chain_a else None,
                        "chain_id_2": str(chain_b) if chain_b else None,
                    }.items() if v
                },
                interface_residue_count=residue_count,
                interface_area=_float_or_none(item.get("interface_area") or item.get("area")),
                h_bond_count=_int_or_none(item.get("h_bond_count") or item.get("hbonds")),
                salt_bridge_count=_int_or_none(
                    item.get("salt_bridge_count") or item.get("salt_bridges")
                ),
                quality_flags=[],
                source_ref=str(source_ref) if source_ref else None,
            )
        )
    return records


def _extract_complex_structure_refs_for_step8(
    storage: Storage, sin: dict, tool_call: ToolCallRecord
) -> list[ComplexStructureRef]:
    summary = tool_call.tool_input_summary or {}
    refs: list[ComplexStructureRef] = []
    if tool_call.tool_name == "PDBePISA_get_interfaces":
        pdb_id = _step8_pdb_id(sin.get("structure_refs") or [])
        if pdb_id:
            refs.append(
                ComplexStructureRef(
                    source_kind="existing_pdb_complex",
                    source_ref=pdb_id,
                    pdb_id=pdb_id,
                    structure_format="pdb",
                    source_tool_call_id=tool_call.tool_call_id,
                    confidence_summary={"interface_evaluation": tool_call.run_status},
                )
            )
    elif tool_call.tool_name == "CrystalStructure_validate":
        local_ref = _step8_local_structure_ref(storage, sin.get("structure_refs") or [])
        if local_ref and _step8_has_explicit_complex_evidence(sin):
            refs.append(
                ComplexStructureRef(
                    source_kind="uploaded_local_complex",
                    source_ref=local_ref.get("source"),
                    storage_ref=str(local_ref["value"]),
                    structure_format=_format_for_file(str(local_ref["value"])),
                    source_tool_call_id=tool_call.tool_call_id,
                    confidence_summary={"validation": tool_call.run_status},
                )
            )
    elif tool_call.tool_name in _STEP8_NIM_COMPLEX_TOOLS:
        payload = _read_tool_output_payload(storage, tool_call)
        model_ref = _prediction_model_ref(payload)
        if model_ref:
            refs.append(
                ComplexStructureRef(
                    source_kind="predicted_complex",
                    source_ref=tool_call.tool_name,
                    storage_ref=model_ref,
                    structure_format=_format_for_file(model_ref),
                    source_tool_call_id=tool_call.tool_call_id,
                    confidence_summary=_prediction_confidence_summary(payload),
                )
            )
    return refs


def _dedupe_complex_structure_refs(refs: list[ComplexStructureRef]) -> list[ComplexStructureRef]:
    out: list[ComplexStructureRef] = []
    seen: set[tuple[Any, ...]] = set()
    for ref in refs:
        key = (ref.source_kind, ref.pdb_id, ref.storage_ref, ref.source_tool_call_id)
        if key in seen:
            continue
        seen.add(key)
        out.append(ref)
    return out


def _build_step8_downstream_handoff(
    *,
    complex_structure_refs: list[ComplexStructureRef],
    interface_features: list[InterfaceFeature],
    interface_analysis_records: list[InterfaceAnalysisRecord],
    confidence_records: list[StructureConfidenceRecord],
    tool_calls: list[ToolCallRecord],
    complex_prediction_plans: list[ComplexPredictionPlan],
) -> Step8DownstreamHandoff:
    missing: list[str] = []
    notes: list[str] = []
    true_complex_refs = [
        ref for ref in complex_structure_refs
        if ref.source_kind in {"existing_pdb_complex", "predicted_complex", "uploaded_local_complex"}
    ]
    has_complex = bool(true_complex_refs)
    has_interfaces = bool(interface_features or interface_analysis_records)
    confidence_types = {c.confidence_type for c in confidence_records}
    validation_available = "structure_quality" in confidence_types
    if not has_complex:
        missing.append("complex_structure_missing")
    if not has_interfaces:
        missing.append("interface_features_missing")
    if any(
        tc.tool_name in _STEP8_NIM_COMPLEX_TOOLS
        and tc.run_status == "dependency_unavailable"
        for tc in tool_calls
    ):
        missing.append("complex_prediction_unavailable")
        notes.append("NvidiaNIM complex prediction route is deferred/unavailable")
    for plan in complex_prediction_plans:
        for item in plan.missing_prediction_inputs:
            missing.append(item)
        if plan.input_status == "contract_unresolved":
            missing.append("complex_prediction_contract_unresolved")
        notes.extend(plan.contract_notes)

    structure_ref = None
    if true_complex_refs:
        first = true_complex_refs[0]
        structure_ref = first.storage_ref or first.pdb_id or first.source_ref
    validated_structure_ref = _validated_structure_ref_from_tool_calls(tool_calls)

    return Step8DownstreamHandoff(
        has_complex_structure=has_complex,
        has_validated_structure=validation_available,
        has_interface_features=has_interfaces,
        structure_for_variant_generation_ref=structure_ref,
        validated_structure_ref=validated_structure_ref,
        interface_quality_available="interface_quality" in confidence_types or has_interfaces,
        prediction_confidence_available="prediction_confidence" in confidence_types,
        refinement_resolution_available="refinement_resolution" in confidence_types,
        validation_available=validation_available,
        missing_for_step9=list(dict.fromkeys(missing)),
        handoff_notes=notes,
    )


def _summarize_prediction_input_status(plans: list[ComplexPredictionPlan]) -> str | None:
    if not plans:
        return None
    statuses = {plan.input_status for plan in plans}
    if "selected_but_deferred" in statuses:
        return "selected_but_deferred"
    if "input_missing" in statuses:
        return "input_missing"
    if "contract_unresolved" in statuses:
        return "contract_unresolved"
    if statuses == {"not_applicable"}:
        return "not_applicable"
    return sorted(statuses)[0]


def _summarize_missing_prediction_inputs(plans: list[ComplexPredictionPlan]) -> list[str]:
    out: list[str] = []
    for plan in plans:
        out.extend(plan.missing_prediction_inputs)
    return list(dict.fromkeys(out))


def _summarize_prediction_runtime_status(plans: list[ComplexPredictionPlan]) -> str | None:
    if not plans:
        return None
    statuses = {plan.runtime_status for plan in plans}
    if "runtime_unavailable" in statuses:
        return "runtime_unavailable"
    if "dependency_unavailable" in statuses:
        return "dependency_unavailable"
    if statuses == {"not_applicable"}:
        return "not_applicable"
    return sorted(statuses)[0]


def _summarize_prediction_contract_notes(plans: list[ComplexPredictionPlan]) -> list[str]:
    out: list[str] = []
    for plan in plans:
        out.extend(plan.contract_notes)
    return list(dict.fromkeys(out))


def _prediction_model_ref(payload: dict[str, Any] | None) -> str | None:
    if not isinstance(payload, dict):
        return None
    value = _extract_scalar(
        payload,
        (
            "model_ref",
            "model_path",
            "model_url",
            "structure_path",
            "structure_url",
            "output_path",
            "artifact_ref",
            "artifact_url",
            "storage_ref",
            "file_ref",
        ),
    )
    if not value:
        return None
    candidate = str(value)
    if _looks_like_raw_structure_body(candidate):
        return None
    return candidate


def _step8_has_explicit_complex_evidence(sin: dict) -> bool:
    status_values: list[str] = []
    for sref in sin.get("structure_refs") or []:
        if isinstance(sref, dict):
            status_values.append(str(sref.get("resource_binding_status") or ""))
    if any(status.lower() == "ambiguous" for status in status_values):
        return False

    roles = {
        str(item.get("chain_role") or "").lower()
        for item in sin.get("chain_mapping") or []
        if isinstance(item, dict)
    }
    has_antigen = "antigen" in roles or "target" in roles
    has_antibody = any(role in roles for role in ("antibody", "antibody_heavy", "antibody_light", "fab", "fc"))
    return has_antigen and has_antibody


def _validated_structure_ref_from_tool_calls(tool_calls: list[ToolCallRecord]) -> str | None:
    for tc in tool_calls:
        if tc.tool_name != "CrystalStructure_validate" or tc.run_status != "success":
            continue
        summary = tc.tool_input_summary or {}
        value = summary.get("pdb_id_or_path")
        if not value and isinstance(summary.get("arguments"), dict):
            value = summary["arguments"].get("pdb_id_or_path")
        if value:
            return str(value)
    return None


def _looks_like_raw_structure_body(value: str) -> bool:
    if len(value) > 500:
        return True
    upper = value[:500].upper()
    if upper.startswith(("ATOM", "HETATM", "HEADER")):
        return True
    if any(marker in upper for marker in ("\nATOM", "\nHETATM", "HEADER ", "\nHEADER")):
        return True
    lower = value[:500].lower()
    return "data_" in lower or "loop_" in lower


def _prediction_confidence_summary(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    out: dict[str, Any] = {}
    for key in ("ptm", "iptm", "plddt", "confidence", "score"):
        if key in payload and payload[key] is not None:
            out[key] = _short(payload[key])
    return out


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _compact_structure_refs_for_audit(structure_refs: list[Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for sref in structure_refs:
        if not isinstance(sref, dict):
            continue
        out.append(
            {
                "file_id": sref.get("file_id"),
                "pdb_id": sref.get("pdb_id"),
                "storage_ref": sref.get("storage_ref"),
            }
        )
    return out


def _compound_hit_from_call(candidate: dict, tc: ToolCallRecord, kwargs: dict, tool_name: str) -> CompoundHit:
    """Build a normalized CompoundHit. Raw payload stays at `tool_output_ref`."""
    smiles = kwargs.get("smiles") or _materials_by_type(candidate, {"payload_smiles", "linker_smiles", "compound_smiles"})
    if isinstance(smiles, list):
        smiles = (smiles[0] if smiles else {}).get("value", "")
    return CompoundHit(
        compound_id=new_artifact_id("compound_hit"),
        # Wrapper currently hits ZINC15 (per architecture audit). We record
        # the family (`ZINC`) and an honest version (`unknown`) so no record
        # ever claims `ZINC22` confirmation.
        source_library="ZINC",
        smiles=str(smiles or ""),
        similarity_score=None,
        source_database_version="unknown",
        source_tool_name=tool_name,
        source_runtime_status="success",
        notes=f"raw payload at tool_output_ref={tc.tool_output_ref}",
    )


def _short(v: Any) -> Any:
    if isinstance(v, str) and len(v) > 200:
        return v[:200] + "…"
    return v


def _compound_selection_context(candidate: dict) -> SelectionContext:
    smiles = _first_material_value(candidate.get("materials") or [], {"payload_smiles", "linker_smiles", "compound_smiles"})
    name = _first_material_value(candidate.get("materials") or [], {"payload_name", "linker_name", "compound_name"})
    zinc_id = _first_identifier_value(candidate.get("identifiers") or [], {"zinc_id"})
    chembl_id = _first_identifier_value(candidate.get("identifiers") or [], {"chembl_id"})
    pubchem_cid = _first_identifier_value(candidate.get("identifiers") or [], {"pubchem_cid"})
    return SelectionContext(
        signals={
            "smiles": bool(smiles),
            "compound_name": bool(name),
            "zinc_id": bool(zinc_id),
            "chembl_id": bool(chembl_id),
            "pubchem_cid": bool(pubchem_cid),
        },
        arg_hints={
            k: v for k, v in {
                "smiles": smiles,
                "query": name or smiles,
                "zinc_id": zinc_id,
                "chembl_id": chembl_id,
                "pubchem_cid": pubchem_cid,
                "compound_name": name,
            }.items() if v
        },
        note=f"step_09 compound candidate_id={candidate.get('candidate_id', '')}",
    )


def _compound_fallback_plans(candidate: dict) -> list[ToolInvocationPlan]:
    smiles = _first_material_value(candidate.get("materials") or [], {"payload_smiles", "linker_smiles", "compound_smiles"})
    name = _first_material_value(candidate.get("materials") or [], {"payload_name", "linker_name", "compound_name"})
    zinc_id = _first_identifier_value(candidate.get("identifiers") or [], {"zinc_id"})
    raw: list[tuple[str, dict[str, Any]]] = []
    if smiles:
        raw.append(("ZINC_search_by_smiles", {"smiles": smiles}))
    if zinc_id:
        raw.append(("ZINC_get_compound", {"zinc_id": zinc_id}))
    if not raw and name:
        raw.append(("ZINC_search_compounds", {"query": name}))
    return [
        ToolInvocationPlan(
            tool_name=tool,
            selection_reason="deterministic Step 9 compound fallback",
            arguments=args,
            argument_construction_reason="deterministic compound argument mapping",
            selected_by="deterministic_fallback",
        )
        for tool, args in raw
    ]


def _compound_argument_mapping(tool_name: str, arg_hints: dict) -> dict[str, Any]:
    if tool_name == "ZINC_search_by_smiles":
        return {"smiles": arg_hints.get("smiles") or ""}
    if tool_name == "ZINC_get_compound":
        return {"zinc_id": arg_hints.get("zinc_id") or ""}
    if tool_name == "ZINC_search_compounds":
        return {"query": arg_hints.get("query") or arg_hints.get("compound_name") or ""}
    return {"query": arg_hints.get("query") or arg_hints.get("compound_name") or arg_hints.get("smiles") or ""}


def _selection_summary(plan: ToolInvocationPlan) -> dict[str, Any]:
    return {
        "selected_by": plan.selected_by,
        "selection_reason": plan.selection_reason,
        "selection_policy_version": plan.selection_policy_version,
        "argument_construction_reason": plan.argument_construction_reason,
        "validation_status": plan.validation_status,
        "validation_warnings": plan.validation_warnings,
    }


def _apply_step7_tool_output_metadata(
    storage: Storage, record: StructureInputRecord, tool_call: ToolCallRecord, tool_name: str
) -> None:
    """Compact Step 7 tool output into normalized Step 7 artifact fields.

    This intentionally does not write raw tool payloads into normalized records.
    """
    compact: dict[str, Any] = {
        "tool_name": tool_name,
        "tool_call_id": tool_call.tool_call_id,
        "run_status": tool_call.run_status,
    }
    if tool_call.error_message:
        compact["error_message"] = str(tool_call.error_message)[:180]

    payload = _read_tool_output_payload(storage, tool_call)
    compact_output = _compact_step7_tool_output(tool_name, payload)
    if compact_output:
        compact["compact_output"] = compact_output

    record.step7_tool_output_metadata.append(compact)

    if tool_call.run_status != "success" or not compact_output:
        return

    if tool_name in {"RCSBData_get_entry", "RCSBData_get_assembly", "SAbDab_get_structure"}:
        entry_ref = _step7_normalized_struct_ref(
            source_kind="pdb_id",
            pdb_id=compact_output.get("pdb_id"),
            source_ref=compact_output.get("source_ref"),
            validation_status=compact_output.get("validation_status", "unknown"),
        )
        if entry_ref:
            if not any(
                existing.pdb_id == entry_ref.pdb_id and existing.storage_ref == entry_ref.storage_ref
                for existing in record.structure_refs
            ):
                record.structure_refs.append(entry_ref)

        for chain in compact_output.get("chain_mapping", []) or []:
            if not isinstance(chain, dict):
                continue
            chain_id = chain.get("chain_id")
            chain_role = chain.get("chain_role")
            chain_source = chain.get("mapping_confidence", 0.0)
            if not chain_id or not chain_role:
                continue
            if any(existing.chain_id == chain_id for existing in record.chain_mapping):
                continue
            try:
                record.chain_mapping.append(
                    ChainMapping(
                        chain_id=str(chain_id),
                        chain_role=chain_role,  # type: ignore[arg-type]
                        mapping_confidence=float(chain_source),
                        source="inferred",
                        source_ref=record.structure_input_id,
                        chain_id_kind="prediction_placeholder",
                    )
                )
            except Exception:  # noqa: BLE001
                continue

    if tool_name == "alphafold_get_prediction":
        ref_source = compact_output.get("uniprot") or compact_output.get("uniprot_id")
        model_ref = compact_output.get("model_ref")
        if model_ref:
            af_ref = _step7_normalized_struct_ref(
                source_kind="predicted_needed",
                pdb_id=None,
                source_ref=ref_source,
                storage_ref=model_ref,
                validation_status="unknown",
            )
            if af_ref and not any(
                (
                    existing.source_kind == "predicted_needed"
                    and existing.source_ref == af_ref.source_ref
                    and existing.storage_ref == af_ref.storage_ref
                )
                for existing in record.structure_refs
            ):
                record.structure_refs.append(af_ref)

    if tool_name in {"RCSBAdvSearch_search_structures", "PDBeSearch_search_structures"}:
        for hit in compact_output.get("hits", []) if isinstance(compact_output, dict) else []:
            if not isinstance(hit, dict):
                continue
            candidate = {
                **{k: _short(v) for k, v in hit.items() if k in {"pdb_id", "query", "source", "method"}},
                "resource_binding_status": "ambiguous",
                "run_status": tool_call.run_status,
                "tool_name": tool_name,
                "source_ref": hit.get("source_ref"),
                "chain_hints": hit.get("chain_hints", []),
            }
            record.database_search_candidates.append(candidate)


def _compact_step7_tool_output(tool_name: str, payload: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    if tool_name == "RCSBData_get_entry":
        return {
            "compact_type": "rcsb_entry",
            "pdb_id": _extract_scalar(payload, ("pdb_id", "id", "structure_id")),
            "source_ref": _extract_scalar(payload, ("source_ref", "pdb_id")),
            "status": payload.get("status"),
            "entry_metadata": _compact_struct_metadata(payload.get("entry") or payload),
            "chain_mapping": _extract_chain_mapping(payload),
            "validation_status": "unknown",
        }
    if tool_name == "RCSBData_get_assembly":
        return {
            "compact_type": "rcsb_assembly",
            "pdb_id": _extract_scalar(payload, ("pdb_id", "id", "structure_id")),
            "assembly_id": payload.get("assembly_id"),
            "source_ref": _extract_scalar(payload, ("pdb_id", "structure_id")),
            "status": payload.get("status"),
            "assembly_metadata": _compact_struct_metadata(payload.get("assembly") or payload.get("result") or {}),
            "chain_mapping": _extract_chain_mapping(payload.get("assembly") or payload),
            "validation_status": "unknown",
        }
    if tool_name == "SAbDab_get_structure":
        return {
            "compact_type": "sabdab_structure",
            "pdb_id": _extract_scalar(payload, ("pdb_id", "pdb_code", "id", "structure_id")),
            "source_ref": _extract_scalar(payload, ("pdb_id", "id", "structure_id")),
            "status": payload.get("status"),
            "source": payload.get("source"),
            "structure_metadata": _compact_struct_metadata(payload.get("structure") or payload.get("result") or {}),
            "chain_mapping": _extract_chain_mapping(payload.get("structure") or payload.get("result") or {}),
            "validation_status": "unknown",
        }
    if tool_name == "RCSBAdvSearch_search_structures":
        return {
            "compact_type": "rcsb_search",
            "query": _extract_scalar(payload, ("query",)),
            "status": payload.get("status"),
            "hits": _compact_search_hits(payload),
        }
    if tool_name == "PDBeSearch_search_structures":
        return {
            "compact_type": "pdbe_search",
            "query": _extract_scalar(payload, ("query",)),
            "status": payload.get("status"),
            "hits": _compact_search_hits(payload),
        }
    if tool_name == "alphafold_get_prediction":
        return {
            "compact_type": "alphafold_prediction",
            "uniprot": _extract_scalar(payload, ("uniprot", "qualifier", "query", "pdb_id")),
            "status": payload.get("status"),
            "model_ref": _prediction_model_ref(payload),
            "source": payload.get("source"),
        }
    return {
        "compact_type": "unknown",
        "status": payload.get("status"),
        "tool_output_keys": [k for k in payload.keys()],
    }


def _compact_search_hits(payload: Any) -> list[dict[str, Any]]:
    hits_raw = []
    if isinstance(payload, dict):
        for key in ("hits", "structures", "results", "items", "documents", "data"):
            if isinstance(payload.get(key), list):
                hits_raw = payload[key]
                break
    elif isinstance(payload, list):
        hits_raw = payload

    out: list[dict[str, Any]] = []
    for item in hits_raw[:25]:
        if isinstance(item, str):
            out.append({"pdb_id": item, "source_ref": item})
            continue
        if not isinstance(item, dict):
            continue
        pdb_id = _extract_scalar(item, ("pdb_id", "pdb_code", "id", "identifier"))
        chain_hints = _extract_chain_ids(item)
        out.append(
            {
                "pdb_id": pdb_id,
                "method": _extract_scalar(item, ("method", "method_type")),
                "resolution": _extract_scalar(item, ("resolution",)),
                "title": _extract_scalar(item, ("title",)),
                "chain_hints": chain_hints,
                "source_ref": pdb_id,
            }
        )
    return out


def _compact_struct_metadata(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    fields = (
        "title",
        "method",
        "resolution",
        "deposition_date",
        "release_date",
        "organism",
        "chains",
        "chain_count",
        "entity_count",
        "status",
    )
    out = {}
    for field in fields:
        value = payload.get(field)
        if value is not None:
            out[field] = _short(value)
    if "chains" in payload and not isinstance(payload.get("chains"), list):
        out.pop("chains", None)
    return out


def _extract_chain_mapping(payload: Any) -> list[dict[str, Any]]:
    if not payload:
        return []
    chain_ids = _extract_chain_ids(payload)
    chain_role_hints = _extract_chain_roles(payload)
    out: list[dict[str, Any]] = []
    for chain_id in chain_ids:
        role = chain_role_hints.get(chain_id, "other")
        out.append({
            "chain_id": chain_id,
            "chain_role": role,
            "mapping_confidence": 0.75 if role != "other" else 0.5,
        })
    return out


def _extract_chain_roles(payload: Any) -> dict[str, str]:
    out: dict[str, str] = {}
    if not isinstance(payload, dict):
        return out

    for key, value in payload.items():
        if not isinstance(key, str):
            continue
        k = key.lower()
        if "chain" not in k:
            continue
        role = None
        if "heavy" in k and "chain" in k:
            role = "antibody_heavy"
        elif "light" in k and "chain" in k:
            role = "antibody_light"
        elif "antigen" in k and "chain" in k:
            role = "antigen"
        if role and isinstance(value, str):
            out[str(value)] = role

    for chain_id in _extract_chain_ids(payload.get("chain_mapping") if isinstance(payload, dict) else None):
        if chain_id not in out:
            out.setdefault(chain_id, "other")
    return out


def _extract_chain_ids(payload: Any) -> list[str]:
    ids: list[str] = []

    def add_chain_id(value: Any) -> None:
        if isinstance(value, str):
            text = value.strip()
            if re.fullmatch(r"[A-Za-z][A-Za-z0-9]{0,4}", text):
                if text not in ids:
                    ids.append(text)

    def walk(obj: Any) -> None:
        if isinstance(obj, dict):
            for key, value in obj.items():
                if key in {"chain_id", "chain"} and isinstance(value, str):
                    add_chain_id(value)
                elif key == "chains":
                    walk(value)
                elif isinstance(value, (dict, list)):
                    walk(value)
        elif isinstance(obj, list):
            for entry in obj:
                walk(entry)

    walk(payload)
    return ids


def _extract_scalar(payload: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in payload and payload[key] not in ("", None):
            return payload[key]
    return None


def _read_tool_output_payload(storage: Storage, tool_call: ToolCallRecord) -> dict[str, Any] | None:
    if tool_call.run_status != "success" or not tool_call.tool_output_ref:
        return None
    try:
        raw = storage.read_json(tool_call.tool_output_ref) or {}
        output = raw.get("output")
        if isinstance(output, dict):
            return output
    except Exception:  # noqa: BLE001
        return None
    return None


def _step7_normalized_struct_ref(
    *,
    source_kind: str,
    pdb_id: Any,
    source_ref: Any,
    validation_status: str,
    storage_ref: Any = None,
) -> StructureRef | None:
    if not pdb_id and not storage_ref:
        return None
    result = StructureRef(
        pdb_id=str(pdb_id),
        source_kind=source_kind,  # type: ignore[arg-type]
        source_ref=(str(source_ref) if source_ref is not None else None),
        storage_ref=str(storage_ref) if storage_ref else None,
        structure_format="pdb",
        validation_status=validation_status,  # type: ignore[arg-type]
        related_candidate_ids=[],
        resource_binding_status="inferred",
        binding_confidence=0.6,
    )
    return result


def _skipped_tool_record(*, tool_name: str, agent_name: str, step_id: str, summary: dict[str, Any]) -> ToolCallRecord:
    return _nonexecuted_tool_record(
        tool_name=tool_name,
        agent_name=agent_name,
        step_id=step_id,
        run_status="skipped",
        summary=summary,
    )


def _nonexecuted_tool_record(
    *,
    tool_name: str,
    agent_name: str,
    step_id: str,
    run_status: str,
    summary: dict[str, Any],
) -> ToolCallRecord:
    now = now_iso()
    return ToolCallRecord(
        tool_call_id=new_tool_call_id(),
        tool_name=tool_name,
        agent_name=agent_name,
        step_id=step_id,
        run_status=run_status,  # type: ignore[arg-type]
        started_at=now,
        finished_at=now,
        tool_input_summary=summary,
        error_message=f"tool invocation not executed: {run_status}",
    )



def _first_material_value(materials: list[dict], types: set[str]) -> Optional[str]:
    for m in materials:
        if m.get("material_type") in types and m.get("value"):
            return str(m.get("value"))
    return None

def _first_identifier_value(identifiers: list[dict], types: set[str]) -> Optional[str]:
    for i in identifiers:
        if i.get("id_type") in types and i.get("id_value"):
            return str(i.get("id_value"))
    return None
