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


# ── shared utilities ────────────────────────────────────────────────────────

def _format_for_file(filename: str) -> str:
    ext = PurePosixPath(filename or "").suffix.lower()
    if ext in {".cif", ".mmcif"}:
        return "cif"
    return "pdb"


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

        for candidate in cct.get("candidate_records") or []:
            ctype = candidate.get("candidate_type")
            if ctype not in {"target_antigen", "antibody", "adc_construct"}:
                # compound_component / unknown candidates are handled in Step 9.
                continue
            record = self._build_structure_input_record(
                candidate=candidate,
                structure_files=structure_files,
                sequence_files=sequence_files,
                refs_by_type=refs_by_type,
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

        prep_status: str
        if not prepared:
            prep_status = "failed"
        elif any(r.missing_metadata_flags for r in prepared):
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
        cand_structure_mats = _materials_by_type(candidate, {"structure_file", "structure_ref"})
        cand_pdb_ids = _identifiers_by_type(candidate, {"pdb_id"})
        cand_uniprot_ids = _identifiers_by_type(candidate, {"uniprot_id"})
        cand_sequence_mats = _materials_by_type(
            candidate,
            {"antibody_heavy_chain_sequence", "antibody_light_chain_sequence", "target_sequence"},
        )

        # Top-level uploads. For target/antibody we accept any uploaded structure
        # file as a candidate-attached structure ref; explicit pairing is the
        # Step 8 evaluator's concern.
        has_structure_file = bool(cand_structure_mats) or bool(structure_files)
        has_pdb_id = bool(cand_pdb_ids) or bool(refs_by_type.get("pdb_id"))
        has_sequence = bool(cand_sequence_mats) or bool(sequence_files) or bool(
            cand_uniprot_ids or refs_by_type.get("uniprot_id")
        )
        if not (has_structure_file or has_pdb_id or has_sequence):
            return None

        # Decide input_case in order of strongest signal.
        if cand_structure_mats or structure_files:
            input_case = "uploaded_structure_file"
            structure_source = "user_uploaded"
        elif cand_pdb_ids or refs_by_type.get("pdb_id"):
            input_case = "known_pdb_id"
            structure_source = "structured_query.referenced_inputs"
        else:
            input_case = "sequence_only_input"
            structure_source = "fasta_or_uniprot"

        structure_refs: list[StructureRef] = []
        for m in cand_structure_mats:
            structure_refs.append(
                StructureRef(
                    pdb_id=None,
                    file_id=None,
                    structure_format=_format_for_file(m.get("value_format") or ""),  # type: ignore[arg-type]
                    validation_status="unknown",
                )
            )
        for f in structure_files:
            structure_refs.append(
                StructureRef(
                    pdb_id=None,
                    file_id=f.get("file_id"),
                    structure_format=_format_for_file(f.get("original_filename", "")),  # type: ignore[arg-type]
                    validation_status="unknown",
                )
            )
        for ident in cand_pdb_ids:
            structure_refs.append(
                StructureRef(
                    pdb_id=ident.get("id_value"),
                    structure_format="pdb",
                    validation_status="unknown",
                )
            )
        for ref in refs_by_type.get("pdb_id", []):
            value = ref.get("value")
            if value and not any(s.pdb_id == value for s in structure_refs):
                structure_refs.append(
                    StructureRef(pdb_id=value, structure_format="pdb", validation_status="unknown")
                )

        sequence_refs: list[SequenceRef] = []
        for m in cand_sequence_mats:
            sequence_refs.append(
                SequenceRef(
                    sequence_id=m.get("material_id", new_artifact_id("seq")),
                    chain_role=_chain_role_from_material(m.get("material_type", "")),
                    sequence=str(m.get("value") or ""),
                )
            )
        for f in sequence_files:
            sequence_refs.append(
                SequenceRef(
                    sequence_id=f.get("file_id", new_artifact_id("seq")),
                    chain_role="antibody_heavy" if ctype == "antibody" else "antigen",
                    sequence=f.get("storage_path") or f.get("original_filename") or "",
                )
            )
        for ident in cand_uniprot_ids:
            sequence_refs.append(
                SequenceRef(
                    sequence_id=ident.get("id_value", "uniprot"),
                    chain_role="antigen" if ctype == "target_antigen" else None,
                    sequence=f"uniprot:{ident.get('id_value')}",
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
        if input_case == "uploaded_structure_file" and not structure_refs:
            missing_flags.append("uploaded_structure_file_present_but_no_ref")
        if input_case == "known_pdb_id" and not any(s.pdb_id for s in structure_refs):
            missing_flags.append("pdb_id_referenced_but_no_structure_ref")
        if input_case == "sequence_only_input" and not sequence_refs:
            missing_flags.append("sequence_only_input_but_no_sequence_ref")

        return StructureInputRecord(
            structure_input_id=new_artifact_id("structure_input"),
            candidate_id=candidate_id,
            input_case=input_case,  # type: ignore[arg-type]
            structure_source=structure_source,
            assessment_intent=assessment_intent,
            structure_role=structure_role,  # type: ignore[arg-type]
            structure_refs=structure_refs,
            sequence_refs_for_prediction=sequence_refs,
            chain_mapping=[],
            chain_pair_candidates=[],
            antigen_antibody_mapping=None,
            residue_ranges=[],
            missing_metadata_flags=missing_flags,
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
                # alphafold_get_prediction expects a UniProt; we pass either an
                # explicit uniprot identifier or a placeholder derived from the
                # sequence_id so the wrapper records a deterministic call.
                value = seq_ref.get("sequence") or ""
                uniprot = value.replace("uniprot:", "") if value.startswith("uniprot:") else None
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

def _chain_role_from_material(material_type: str) -> Optional[str]:
    return {
        "antibody_heavy_chain_sequence": "antibody_heavy",
        "antibody_light_chain_sequence": "antibody_light",
        "target_sequence": "antigen",
    }.get(material_type)


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
