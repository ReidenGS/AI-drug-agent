"""CandidateContextAgent — Step 5.

Builds normalized candidate / material / identifier records from every source
the run has available so far:

- `structured_query.mentioned_entities` (target / antibody / payload / linker)
- `structured_query.referenced_inputs[]` (PDB / UniProt / ZINC / ChEMBL /
  DrugBank / PubChem / SMILES detected by the Step 2 parser)
- `raw_request_record.user_provided_context` (free-text fallback)
- `raw_request_record.uploaded_files[]` (PDB/CIF → structure material,
  FASTA → sequence material)

Per candidate, the agent calls at most one scoped MCP tool to enrich context.
Raw payloads land in `tool_outputs/step_05/{tool_call_id}.json` and are
referenced via `tool_call_records[].tool_output_ref`. Raw upstream payloads
NEVER appear inside `candidate_records[]`.

ZINC guard: ZINC ids land as `zinc_id` identifiers; no material is marked as
`ZINC22` unless the source record explicitly says so. Step 5 doesn't emit
`CompoundHit` records (those belong to Step 9).
"""

from __future__ import annotations

from pathlib import PurePosixPath
from typing import Any, Iterable, Optional

from ..mcp.client import MCPClient
from ..schemas.common import ToolCallRecord
from ..schemas.step_05_candidate_context_table import (
    ADCLinks,
    CandidateContextTable,
    CandidateRecord,
    Identifier,
    Material,
)
from ..services.artifact_registry_service import ArtifactRegistryService
from ..services.storage_service import Storage
from ..services.workflow_state_service import WorkflowStateService
from ..utils.ids import new_artifact_id, new_tool_call_id
from ..utils.time import now_iso


_AGENT_NAME = "candidate_context_agent"
_STEP_ID = "step_05"
_ARTIFACT_KEY = "candidate_context_table.json"


_PDB_EXTS = {".pdb", ".cif", ".mmcif", ".ent"}
_FASTA_EXTS = {".fasta", ".fa", ".faa", ".seq"}


class CandidateContextAgent:
    name = _AGENT_NAME

    def __init__(
        self,
        *,
        storage: Storage,
        registry: ArtifactRegistryService,
        workflow_state: WorkflowStateService,
        mcp_client: MCPClient,
    ) -> None:
        self.storage = storage
        self.registry = registry
        self.workflow_state = workflow_state
        self.mcp_client = mcp_client

    # ── public API ──────────────────────────────────────────────────────────
    def run(self, run_id: str) -> CandidateContextTable:
        reg = self.registry.get(run_id)
        if not reg.active_artifacts.structured_query_id:
            raise ValueError("Step 5 requires Step 2 structured_query in registry")
        if not reg.active_artifacts.run_step_plan_id:
            raise ValueError("Step 5 requires Step 4 run_step_plan in registry")

        sq = self.storage.read_json(self.storage.run_key(run_id, "inputs/structured_query.json"))
        raw = self.storage.read_json(
            self.storage.run_key(run_id, "inputs/raw_request_record.json")
        )

        entities = sq.get("mentioned_entities") or {}
        ctx = raw.get("user_provided_context") or {}
        refs = [r for r in (sq.get("referenced_inputs") or []) if isinstance(r, dict)]
        uploaded = raw.get("uploaded_files") or []
        sq_artifact_id = reg.active_artifacts.structured_query_id or ""

        # Index referenced_inputs by id_type once.
        refs_by_type: dict[str, list[dict]] = {}
        for r in refs:
            refs_by_type.setdefault(r.get("id_type", ""), []).append(r)

        # File-derived materials.
        structure_files = [
            f for f in uploaded
            if PurePosixPath(f.get("original_filename", "")).suffix.lower() in _PDB_EXTS
        ]
        sequence_files = [
            f for f in uploaded
            if PurePosixPath(f.get("original_filename", "")).suffix.lower() in _FASTA_EXTS
        ]

        tool_call_records: list[ToolCallRecord] = []
        candidate_records: list[CandidateRecord] = []
        missing_flags: list[str] = []

        # ── target candidate ────────────────────────────────────────────────
        target_text = (
            entities.get("target_or_antigen_text")
            or ctx.get("target_or_antigen_text")
        )
        target_cand = self._build_target_candidate(
            target_text=target_text,
            refs_by_type=refs_by_type,
            sq_artifact_id=sq_artifact_id,
            structure_files=structure_files,
            sequence_files=sequence_files,
        )
        if target_cand is None:
            missing_flags.append("mentioned_entities.target_or_antigen_text")
        else:
            if target_text:
                tc = self._enrich_with_tool(
                    run_id=run_id,
                    tool_name="SAbDab_search_structures",
                    arg_value=target_text,
                    label="target",
                )
                tool_call_records.append(tc)
                target_cand.candidate_notes = _notes_for(tc, "target context enrichment")
            candidate_records.append(target_cand)

        # ── antibody candidate ──────────────────────────────────────────────
        antibody_text = (
            entities.get("antibody_candidate_text") or ctx.get("candidate_text")
        )
        ab_cand = self._build_antibody_candidate(
            antibody_text=antibody_text,
            sequence_files=sequence_files,
            sq_artifact_id=sq_artifact_id,
        )
        if ab_cand is None:
            missing_flags.append("mentioned_entities.antibody_candidate_text")
        else:
            if antibody_text:
                tc = self._enrich_with_tool(
                    run_id=run_id,
                    tool_name="SAbDab_search_structures",
                    arg_value=antibody_text,
                    label="antibody",
                )
                tool_call_records.append(tc)
                ab_cand.candidate_notes = _notes_for(tc, "antibody context enrichment")
            candidate_records.append(ab_cand)

        # ── payload / linker / generic compound candidates ──────────────────
        payload_text = entities.get("payload_text") or ctx.get("payload_linker_text")
        linker_text = entities.get("linker_text")
        smiles_refs = refs_by_type.get("smiles", [])
        chembl_refs = refs_by_type.get("chembl_id", [])
        pubchem_refs = refs_by_type.get("pubchem_cid", [])
        zinc_refs = refs_by_type.get("zinc_id", [])
        drugbank_refs = refs_by_type.get("drugbank_id", [])

        compound_cands = self._build_compound_candidates(
            payload_text=payload_text,
            linker_text=linker_text,
            smiles_refs=smiles_refs,
            chembl_refs=chembl_refs,
            pubchem_refs=pubchem_refs,
            zinc_refs=zinc_refs,
            drugbank_refs=drugbank_refs,
            sq_artifact_id=sq_artifact_id,
        )
        if not compound_cands:
            missing_flags.append("mentioned_entities.payload_text")
        for cc in compound_cands:
            label = next(
                (m.value for m in cc.materials if m.material_type in {"payload_name", "linker_name", "compound_name"}),
                cc.candidate_label,
            )
            tool_name = (
                "ChEMBL_search_substructure"
                if any(m.material_type in {"linker_name", "linker_smiles"} for m in cc.materials)
                else "ChEMBL_search_molecules"
            )
            tc = self._enrich_with_tool(
                run_id=run_id,
                tool_name=tool_name,
                arg_value=str(label or ""),
                label="compound",
            )
            tool_call_records.append(tc)
            cc.candidate_notes = _notes_for(tc, "compound context enrichment")
            candidate_records.append(cc)

        # ── status + persist ────────────────────────────────────────────────
        success_count = sum(1 for tc in tool_call_records if tc.run_status == "success")
        if missing_flags or success_count < len(tool_call_records):
            build_status = "partial"
        elif tool_call_records:
            build_status = "ok"
        else:
            build_status = "failed"

        table = CandidateContextTable(
            run_id=run_id,
            created_at=now_iso(),
            context_build_status=build_status,  # type: ignore[arg-type]
            candidate_records=candidate_records,
            missing_context_flags=missing_flags,
            tool_call_records=tool_call_records,
        )

        artifact_id = new_artifact_id("candidate_context_table")
        self.storage.write_json(
            self.storage.run_key(run_id, _ARTIFACT_KEY),
            {"artifact_id": artifact_id, **table.model_dump()},
        )
        self.registry.update_active(run_id, candidate_context_table_id=artifact_id)
        self.workflow_state.mark(run_id, "step_05", "completed")
        return table

    # ── candidate builders ─────────────────────────────────────────────────
    def _build_target_candidate(
        self,
        *,
        target_text: Optional[str],
        refs_by_type: dict[str, list[dict]],
        sq_artifact_id: str,
        structure_files: list[dict],
        sequence_files: list[dict],
    ) -> Optional[CandidateRecord]:
        if not (target_text or refs_by_type.get("uniprot_id") or refs_by_type.get("pdb_id")
                or structure_files):
            return None

        materials: list[Material] = []
        if target_text:
            materials.append(self._material("target_antigen_name", target_text, "text"))

        for f in structure_files:
            materials.append(
                self._material(
                    "structure_file",
                    f.get("storage_path") or f.get("original_filename") or "",
                    _format_for_file(f),
                )
            )
        for ref in refs_by_type.get("pdb_id", []):
            materials.append(self._material("structure_ref", ref["value"], "text"))
        for f in sequence_files:
            materials.append(
                self._material(
                    "target_sequence",
                    f.get("storage_path") or f.get("original_filename") or "",
                    "fasta",
                )
            )

        identifiers = _identifiers_for(
            refs_by_type,
            ("uniprot_id", "pdb_id"),
            source_ids=[sq_artifact_id],
        )
        return CandidateRecord(
            candidate_id=new_artifact_id("candidate"),
            candidate_label=target_text or (identifiers[0].id_value if identifiers else "target"),
            candidate_type="target_antigen",
            source_records=[sq_artifact_id] if sq_artifact_id else [],
            identifiers=identifiers,
            materials=materials,
            adc_links=ADCLinks(),
            candidate_status="partially_ready_for_step6",
            candidate_notes=None,
        )

    def _build_antibody_candidate(
        self,
        *,
        antibody_text: Optional[str],
        sequence_files: list[dict],
        sq_artifact_id: str,
    ) -> Optional[CandidateRecord]:
        if not (antibody_text or sequence_files):
            return None
        materials: list[Material] = []
        if antibody_text:
            materials.append(self._material("antibody_name", antibody_text, "text"))
        for f in sequence_files:
            materials.append(
                self._material(
                    "antibody_heavy_chain_sequence",
                    f.get("storage_path") or f.get("original_filename") or "",
                    "fasta",
                )
            )
        return CandidateRecord(
            candidate_id=new_artifact_id("candidate"),
            candidate_label=antibody_text or "antibody_from_sequence_upload",
            candidate_type="antibody",
            source_records=[sq_artifact_id] if sq_artifact_id else [],
            identifiers=[],
            materials=materials,
            adc_links=ADCLinks(),
            candidate_status="partially_ready_for_step6",
            candidate_notes=None,
        )

    def _build_compound_candidates(
        self,
        *,
        payload_text: Optional[str],
        linker_text: Optional[str],
        smiles_refs: list[dict],
        chembl_refs: list[dict],
        pubchem_refs: list[dict],
        zinc_refs: list[dict],
        drugbank_refs: list[dict],
        sq_artifact_id: str,
    ) -> list[CandidateRecord]:
        results: list[CandidateRecord] = []

        if payload_text:
            mats: list[Material] = [self._material("payload_name", payload_text, "text")]
            mats.extend(self._smiles_materials(smiles_refs, "payload_smiles"))
            results.append(
                self._compound_candidate(
                    label=payload_text,
                    materials=mats,
                    identifiers=_identifiers_for(
                        {"chembl_id": chembl_refs, "pubchem_cid": pubchem_refs,
                         "zinc_id": zinc_refs, "drugbank_id": drugbank_refs},
                        ("chembl_id", "pubchem_cid", "zinc_id", "drugbank_id"),
                        source_ids=[sq_artifact_id],
                    ),
                    sq_artifact_id=sq_artifact_id,
                )
            )
        if linker_text:
            mats = [self._material("linker_name", linker_text, "text")]
            mats.extend(self._smiles_materials(smiles_refs, "linker_smiles"))
            results.append(
                self._compound_candidate(
                    label=linker_text,
                    materials=mats,
                    identifiers=[],
                    sq_artifact_id=sq_artifact_id,
                )
            )

        # Free-floating compound identifiers without payload/linker text.
        if not payload_text and not linker_text and (smiles_refs or chembl_refs or pubchem_refs
                                                     or zinc_refs or drugbank_refs):
            mats = list(self._smiles_materials(smiles_refs, "compound_smiles"))
            label = (
                (chembl_refs and chembl_refs[0]["value"])
                or (pubchem_refs and pubchem_refs[0]["value"])
                or (zinc_refs and zinc_refs[0]["value"])
                or "compound_from_identifiers"
            )
            results.append(
                self._compound_candidate(
                    label=label,
                    materials=mats,
                    identifiers=_identifiers_for(
                        {"chembl_id": chembl_refs, "pubchem_cid": pubchem_refs,
                         "zinc_id": zinc_refs, "drugbank_id": drugbank_refs},
                        ("chembl_id", "pubchem_cid", "zinc_id", "drugbank_id"),
                        source_ids=[sq_artifact_id],
                    ),
                    sq_artifact_id=sq_artifact_id,
                )
            )
        return results

    def _compound_candidate(
        self,
        *,
        label: str,
        materials: list[Material],
        identifiers: list[Identifier],
        sq_artifact_id: str,
    ) -> CandidateRecord:
        return CandidateRecord(
            candidate_id=new_artifact_id("candidate"),
            candidate_label=label,
            candidate_type="compound_component",
            source_records=[sq_artifact_id] if sq_artifact_id else [],
            identifiers=identifiers,
            materials=materials,
            adc_links=ADCLinks(),
            candidate_status="partially_ready_for_step6",
            candidate_notes=None,
        )

    # ── helpers ─────────────────────────────────────────────────────────────
    def _material(self, material_type: str, value: str, value_format: str) -> Material:
        return Material(
            material_id=new_artifact_id("material"),
            material_type=material_type,
            value=value,
            value_format=value_format,
            extraction_status="extracted",
            validation_status="unknown",
        )

    def _smiles_materials(self, smiles_refs: list[dict], material_type: str) -> Iterable[Material]:
        for ref in smiles_refs:
            yield Material(
                material_id=new_artifact_id("material"),
                material_type=material_type,
                value=ref["value"],
                value_format="smiles",
                extraction_status="extracted",
                validation_status="unknown",
            )

    def _enrich_with_tool(
        self,
        *,
        run_id: str,
        tool_name: str,
        arg_value: str,
        label: str,
    ) -> ToolCallRecord:
        tc_id = new_tool_call_id()
        started = now_iso()
        result = self.mcp_client.call_tool(
            agent_name=_AGENT_NAME,
            step_id=_STEP_ID,
            tool_name=tool_name,
            query=arg_value,
        )
        finished = now_iso()

        output_ref = None
        output_artifact_id = None
        if "payload" in result:
            output_artifact_id = new_artifact_id("tool_output")
            output_key = self.storage.run_key(
                run_id, "tool_outputs", "step_05", f"{tc_id}.json"
            )
            self.storage.write_json(
                output_key,
                {
                    "tool_call_id": tc_id,
                    "tool_name": tool_name,
                    "input": {"query": arg_value, "label": label},
                    "output": result["payload"],
                },
            )
            output_ref = output_key

        return ToolCallRecord(
            tool_call_id=tc_id,
            tool_name=tool_name,
            agent_name=_AGENT_NAME,
            step_id=_STEP_ID,
            run_status=result.get("run_status", "pending"),
            started_at=started,
            finished_at=finished,
            tool_input_summary={"query": arg_value, "label": label},
            tool_output_artifact_id=output_artifact_id,
            tool_output_ref=output_ref,
            error_message=result.get("error_message"),
        )

    def run_step(self, *, run_id: str, step_id: str, payload: dict[str, Any]) -> dict:  # noqa: ARG002
        return self.run(run_id).model_dump()


# ── module helpers ─────────────────────────────────────────────────────────

def _notes_for(tc: ToolCallRecord, label: str) -> Optional[str]:
    if tc.run_status == "success":
        return None
    return f"{label}: {tc.run_status}"


def _format_for_file(f: dict) -> str:
    ext = PurePosixPath(f.get("original_filename", "")).suffix.lower()
    if ext in {".cif", ".mmcif"}:
        return "cif"
    return "pdb"


def _identifiers_for(
    refs_by_type: dict[str, list[dict]],
    id_types: tuple[str, ...],
    *,
    source_ids: list[str],
) -> list[Identifier]:
    out: list[Identifier] = []
    for id_type in id_types:
        for ref in refs_by_type.get(id_type, []):
            out.append(
                Identifier(
                    id_type=id_type,
                    id_value=ref.get("value", ""),
                    source_ids=[s for s in source_ids if s],
                    confidence=0.9 if ref.get("source") == "raw_request_text" else 0.5,
                )
            )
    return out
