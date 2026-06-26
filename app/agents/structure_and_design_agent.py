"""StructureAndDesignAgent — Step 7, Step 8, Step 9.

This agent owns its own functional scope; it does not behave like a generic
tool-calling agent. The three entry points are:

- `run_step_7(run_id)` — assemble `prepared_structure_input_package` from
  Step 1/2/5 artifacts (uploaded PDB/CIF + FASTA, structured_query refs,
  candidate materials). One optional `RCSBData_get_entry` enrichment call per
  PDB id, by reference only.
- `run_step_8(run_id)` — per `StructureInputRecord`, route to a small subset
  of v0.2 inventory tools depending on `input_case`. Emit confidence /
  validation records and `output_artifacts[]` pointing at raw payloads
  stored under `tool_outputs/step_08/{tool_call_id}.json`.
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
        all_pdb_ids_seen: set[str] = set()
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
            # Enrich PDB references once via RCSBData_get_entry (mockable wrapper).
            for sref in record.structure_refs:
                if sref.pdb_id and sref.pdb_id not in all_pdb_ids_seen:
                    tc = self._call_tool(
                        run_id=run_id,
                        step_id=_STEP_07,
                        tool_name="RCSBData_get_entry",
                        kwargs={"pdb_id": sref.pdb_id},
                        output_dir="step_07",
                        label=f"enrich:{sref.pdb_id}",
                    )
                    tool_call_records.append(tc)
                    all_pdb_ids_seen.add(sref.pdb_id)
            prepared.append(record)
        _attach_antigen_antibody_mapping(prepared, candidates)

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
        if not (has_structure_file or has_pdb_id or has_sequence):
            return None

        # Decide input_case in deterministic source-priority order.
        if structure_files:
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
        for m in cand_structure_mats:
            source_ref = m.get("material_id") or m.get("value")
            structure_refs.append(
                StructureRef(
                    pdb_id=None,
                    file_id=None,
                    structure_format=_format_for_file(m.get("value_format") or ""),  # type: ignore[arg-type]
                    validation_status="unknown",
                    source_kind="candidate_material",
                    source_ref=source_ref,
                    related_candidate_ids=[candidate_id],
                    resource_binding_status="explicit",
                    binding_confidence=1.0,
                )
            )
        for f in structure_files:
            structure_refs.append(
                StructureRef(
                    pdb_id=None,
                    file_id=f.get("file_id"),
                    structure_format=_format_for_file(f.get("original_filename", "")),  # type: ignore[arg-type]
                    validation_status="unknown",
                    source_kind="uploaded_file",
                    source_ref=f.get("storage_path") or f.get("file_id"),
                    related_candidate_ids=[candidate_id],
                    resource_binding_status=f.get("resource_binding_status", "inferred"),
                    binding_confidence=float(f.get("binding_confidence", 0.6)),
                )
            )
        for ident in cand_pdb_ids:
            structure_refs.append(
                StructureRef(
                    pdb_id=ident.get("id_value"),
                    structure_format="pdb",
                    validation_status="unknown",
                    source_kind="pdb_id",
                    source_ref=ident.get("id_value"),
                    related_candidate_ids=[candidate_id],
                    resource_binding_status="explicit",
                    binding_confidence=1.0,
                )
            )
        for material in cand_pdb_material_refs:
            value = material.get("value")
            if value and not any(s.pdb_id == value for s in structure_refs):
                structure_refs.append(StructureRef(
                    pdb_id=value,
                    structure_format="pdb",
                    validation_status="unknown",
                    source_kind="candidate_material",
                    source_ref=material.get("material_id") or value,
                    related_candidate_ids=[candidate_id],
                    resource_binding_status="explicit",
                    binding_confidence=1.0,
                ))
        for ref in refs_by_type.get("pdb_id", []):
            value = ref.get("value")
            if value and not any(s.pdb_id == value for s in structure_refs):
                structure_refs.append(
                    StructureRef(
                        pdb_id=value,
                        structure_format="pdb",
                        validation_status="unknown",
                        source_kind="pdb_id",
                        source_ref=value,
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
            run_status = "ok"
            partial = False

            tools_to_call = self._tools_for_step8_case(sin)
            if not tools_to_call:
                run_status = "partial"
                partial = True

            for tool_name, kwargs, conf_type in tools_to_call:
                tc = self._call_tool(
                    run_id=run_id,
                    step_id=_STEP_08,
                    tool_name=tool_name,
                    kwargs=kwargs,
                    output_dir="step_08",
                    label=f"step08:{structure_input_id}:{tool_name}",
                )
                tool_calls.append(tc)
                if tc.run_status == "success":
                    confidence_records.append(
                        StructureConfidenceRecord(
                            confidence_type=conf_type,  # type: ignore[arg-type]
                            value=_extract_confidence_value(tool_name, tc),
                            source=tool_name,
                            source_tool_call_id=tc.tool_call_id,
                        )
                    )
                    if tc.tool_output_artifact_id and tc.tool_output_ref:
                        output_artifacts.append(
                            StructureOutputArtifact(
                                artifact_id=tc.tool_output_artifact_id,
                                related_candidate_id=candidate_id,
                                related_structure_input_id=structure_input_id,
                                artifact_type=_artifact_type_for_tool(tool_name),  # type: ignore[arg-type]
                                storage_ref=tc.tool_output_ref,
                                storage_type="local_run_storage",
                                content_type="json",
                                created_at=tc.finished_at,
                            )
                        )
                elif tc.run_status in {"dependency_unavailable", "failed", "skipped"}:
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
                    interface_features=[],
                    structure_confidence_records=confidence_records,
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

    def _tools_for_step8_case(
        self, sin: dict
    ) -> list[tuple[str, dict[str, Any], str]]:
        """Decide which Step 8 tools to call for one structure input record.

        Returns list of (tool_name, kwargs, confidence_type).
        """
        input_case = sin.get("input_case")
        out: list[tuple[str, dict, str]] = []

        if input_case == "uploaded_structure_file":
            file_ref = next(
                (s for s in sin.get("structure_refs") or [] if s.get("file_id") or s.get("pdb_id")),
                None,
            )
            ref_value = (file_ref or {}).get("file_id") or (file_ref or {}).get("pdb_id") or "uploaded"
            out.append(("CrystalStructure_validate", {"pdb_id_or_path": ref_value}, "structure_quality"))
            out.append(
                ("ProteinsPlus_profile_structure_quality",
                 {"pdb_id_or_path": ref_value}, "structure_quality")
            )

        elif input_case == "known_pdb_id":
            pdb_ref = next(
                (s for s in sin.get("structure_refs") or [] if s.get("pdb_id")),
                None,
            )
            pdb_id = (pdb_ref or {}).get("pdb_id")
            if pdb_id:
                out.append(("RCSBData_get_entry", {"pdb_id": pdb_id}, "structure_quality"))
                out.append(
                    ("get_refinement_resolution_by_pdb_id",
                     {"pdb_id": pdb_id}, "refinement_resolution")
                )
                out.append(
                    ("ProteinsPlus_profile_structure_quality",
                     {"pdb_id_or_path": pdb_id}, "structure_quality")
                )

        elif input_case == "sequence_only_input":
            seq_ref = next(
                (s for s in sin.get("sequence_refs_for_prediction") or []),
                None,
            )
            if seq_ref:
                # AlphaFold wrapper accepts UniProt identifiers. Step 7 now
                # represents identifiers explicitly instead of encoding them
                # as pseudo-sequences.
                uniprot = (
                    seq_ref.get("source_ref")
                    if seq_ref.get("prediction_input_kind") == "uniprot_id"
                    else None
                )
                if uniprot:
                    out.append(
                        ("alphafold_get_prediction", {"uniprot": uniprot}, "prediction_confidence")
                    )
                else:
                    # Can't run AF without a UniProt; lane partial.
                    pass

        return out

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

    def compatible_ids(resource: dict, resource_kind: str) -> tuple[list[str], str, float]:
        explicit_ids = _explicit_resource_candidate_ids(resource, set(bound))
        if explicit_ids:
            return explicit_ids, "explicit", 1.0
        role = _resource_role(resource, resource_kind)
        if role in {"target", "antigen"}:
            ids = by_type.get("target_antigen", [])
            return (ids, "inferred", 0.8) if len(ids) == 1 else ([], "ambiguous", 0.0)
        if role in {"antibody", "antibody_heavy", "antibody_light"}:
            ids = by_type.get("antibody", [])
            return (ids, "inferred", 0.8) if len(ids) == 1 else ([], "ambiguous", 0.0)
        if role == "complex":
            ids = by_type.get("adc_construct", [])
            if len(ids) == 1:
                return ids, "inferred", 0.7
            target_ids = by_type.get("target_antigen", [])
            return (target_ids, "inferred", 0.5) if len(target_ids) == 1 else ([], "ambiguous", 0.0)
        if resource_kind in {"structure", "pdb_id", "uniprot_id"}:
            ids = by_type.get("target_antigen", [])
            return (ids, "inferred", 0.5) if len(ids) == 1 else ([], "ambiguous", 0.0)
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
        from Bio.PDB import MMCIFParser, PDBParser

        text = data.decode("utf-8", errors="replace")
        suffix = PurePosixPath(source_ref).suffix.lower()
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


def _artifact_type_for_tool(tool_name: str) -> str:
    if tool_name == "CrystalStructure_validate":
        return "refinement_or_validation_report"
    if tool_name == "get_refinement_resolution_by_pdb_id":
        return "refinement_or_validation_report"
    if tool_name == "ProteinsPlus_profile_structure_quality":
        return "structure_quality_report"
    if tool_name == "RCSBData_get_entry":
        return "structure_quality_report"
    if tool_name == "alphafold_get_prediction":
        return "predicted_monomer_structure"
    return "other"


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


def _skipped_tool_record(*, tool_name: str, agent_name: str, step_id: str, summary: dict[str, Any]) -> ToolCallRecord:
    now = now_iso()
    return ToolCallRecord(
        tool_call_id=new_tool_call_id(),
        tool_name=tool_name,
        agent_name=agent_name,
        step_id=step_id,
        run_status="skipped",
        started_at=now,
        finished_at=now,
        tool_input_summary=summary,
        error_message="tool invocation plan validation_status=skipped",
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
