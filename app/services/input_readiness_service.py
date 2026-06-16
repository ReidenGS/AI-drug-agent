"""Step 3 — InputReadinessService.

Reads BOTH the Step 1 `raw_request_record` and the Step 2 `structured_query`,
then judges readiness from the combined signal. Structured-query entities can
satisfy a presence check even when the raw `user_provided_context` was sparse;
uploaded-file metadata is inspected for filename / content_type / sha256 to
infer a role (pdb/fasta/csv/json/unknown).
"""

from __future__ import annotations

from pathlib import PurePosixPath

from ..schemas.step_03_input_readiness import (
    BasicADCInputPresence,
    InputReadinessStatus,
    MissingInputItem,
    SourceRefs,
    UploadedFileCheck,
)
from ..utils.ids import new_artifact_id
from ..utils.time import now_iso
from .artifact_registry_service import ArtifactRegistryService
from .storage_service import Storage
from .workflow_state_service import WorkflowStateService


_ARTIFACT_KEY = "inputs/input_readiness_status.json"


_PDB_EXTS = {".pdb", ".cif", ".mmcif", ".ent"}
_FASTA_EXTS = {".fasta", ".fa", ".faa", ".seq"}
_CSV_EXTS = {".csv", ".tsv", ".xlsx"}
_JSON_EXTS = {".json"}
_IMG_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".svg"}


def _infer_file_role(filename: str, content_type: str | None) -> str:
    ext = PurePosixPath(filename or "").suffix.lower()
    if ext in _PDB_EXTS:
        return "pdb_or_cif_structure"
    if ext in _FASTA_EXTS:
        return "fasta_sequence"
    if ext in _CSV_EXTS:
        return "csv_or_table"
    if ext in _JSON_EXTS:
        return "json_metadata"
    if ext in _IMG_EXTS:
        return "image"
    ct = (content_type or "").lower()
    if "fasta" in ct:
        return "fasta_sequence"
    if "pdb" in ct or "chemical/x-pdb" in ct:
        return "pdb_or_cif_structure"
    if "csv" in ct or "tab-separated" in ct or "spreadsheet" in ct:
        return "csv_or_table"
    if "json" in ct:
        return "json_metadata"
    if ct.startswith("image/"):
        return "image"
    return "unknown"


def _referenced_id_types(structured_query: dict) -> set[str]:
    return {
        ref.get("id_type", "")
        for ref in (structured_query.get("referenced_inputs") or [])
        if isinstance(ref, dict)
    }


class InputReadinessService:
    def __init__(
        self,
        storage: Storage,
        registry: ArtifactRegistryService,
        workflow_state: WorkflowStateService,
    ) -> None:
        self.storage = storage
        self.registry = registry
        self.workflow_state = workflow_state

    def check(self, run_id: str) -> InputReadinessStatus:
        reg = self.registry.get(run_id)
        raw_id = reg.active_artifacts.raw_request_record_id
        sq_id = reg.active_artifacts.structured_query_id
        if not raw_id or not sq_id:
            raise ValueError("Step 3 requires Step 1 + Step 2 artifacts in registry")

        raw = self.storage.read_json(self.storage.run_key(run_id, "inputs/raw_request_record.json"))
        sq = self.storage.read_json(self.storage.run_key(run_id, "inputs/structured_query.json"))

        ctx = raw.get("user_provided_context") or {}
        entities = sq.get("mentioned_entities") or {}
        ref_id_types = _referenced_id_types(sq)
        uploaded_files = raw.get("uploaded_files") or []

        # ── file checks ────────────────────────────────────────────────────
        file_checks: list[UploadedFileCheck] = []
        any_structure_file = False
        any_sequence_file = False
        for f in uploaded_files:
            role = _infer_file_role(f.get("original_filename", ""), f.get("content_type"))
            file_checks.append(
                UploadedFileCheck(
                    file_id=f["file_id"],
                    exists=bool(f.get("storage_path")),
                    checksum_ok=bool(f.get("sha256")),
                    format_ok=role != "unknown",
                    inferred_role=role,  # type: ignore[arg-type]
                    storage_path_present=bool(f.get("storage_path")),
                    content_type_present=bool(f.get("content_type")),
                    size_bytes_present=f.get("size_bytes") is not None,
                    notes=None,
                )
            )
            if role == "pdb_or_cif_structure":
                any_structure_file = True
            if role == "fasta_sequence":
                any_sequence_file = True

        # ── presence + evidence ─────────────────────────────────────────────
        target_text = ctx.get("target_or_antigen_text") or entities.get("target_or_antigen_text")
        target_ev = (
            "raw_request_record.user_provided_context.target_or_antigen_text"
            if ctx.get("target_or_antigen_text")
            else (
                "structured_query.mentioned_entities.target_or_antigen_text"
                if entities.get("target_or_antigen_text")
                else None
            )
        )

        candidate_text = ctx.get("candidate_text") or entities.get("antibody_candidate_text")
        candidate_ev = (
            "raw_request_record.user_provided_context.candidate_text"
            if ctx.get("candidate_text")
            else (
                "structured_query.mentioned_entities.antibody_candidate_text"
                if entities.get("antibody_candidate_text")
                else None
            )
        )

        payload_text = entities.get("payload_text") or ctx.get("payload_linker_text")
        payload_ev = (
            "structured_query.mentioned_entities.payload_text"
            if entities.get("payload_text")
            else (
                "raw_request_record.user_provided_context.payload_linker_text"
                if ctx.get("payload_linker_text")
                else None
            )
        )

        linker_text = entities.get("linker_text") or ctx.get("payload_linker_text")
        linker_ev = (
            "structured_query.mentioned_entities.linker_text"
            if entities.get("linker_text")
            else (
                "raw_request_record.user_provided_context.payload_linker_text"
                if ctx.get("payload_linker_text")
                else None
            )
        )

        structure_present = (
            any_structure_file
            or "pdb_id" in ref_id_types
            or "structure_ref" in ref_id_types
        )
        sequence_present = (
            any_sequence_file
            or "uniprot_id" in ref_id_types
            or "antibody_heavy_chain_sequence" in (entities or {})
        )
        structure_or_sequence_present = structure_present or sequence_present
        structure_ev = None
        if any_structure_file:
            structure_ev = "raw_request_record.uploaded_files[*].inferred_role=pdb_or_cif_structure"
        elif "pdb_id" in ref_id_types:
            structure_ev = "structured_query.referenced_inputs[id_type=pdb_id]"
        elif any_sequence_file:
            structure_ev = "raw_request_record.uploaded_files[*].inferred_role=fasta_sequence"
        elif "uniprot_id" in ref_id_types:
            structure_ev = "structured_query.referenced_inputs[id_type=uniprot_id]"

        constraints_present = bool(ctx.get("constraints_text") or sq.get("user_constraints"))
        constraints_ev = (
            "raw_request_record.user_provided_context.constraints_text"
            if ctx.get("constraints_text")
            else ("structured_query.user_constraints" if sq.get("user_constraints") else None)
        )

        presence = BasicADCInputPresence(
            target_or_antigen_present=bool(target_text),
            antibody_candidate_present=bool(candidate_text),
            payload_present=bool(payload_text),
            linker_present=bool(linker_text),
            structure_or_sequence_present=structure_or_sequence_present,
            constraints_present=constraints_present,
            target_evidence=target_ev,
            antibody_evidence=candidate_ev,
            payload_evidence=payload_ev,
            linker_evidence=linker_ev,
            structure_or_sequence_evidence=structure_ev,
            constraints_evidence=constraints_ev,
        )

        # ── gap classification ─────────────────────────────────────────────
        missing: list[MissingInputItem] = []
        if not presence.target_or_antigen_present:
            missing.append(
                MissingInputItem(
                    field="user_provided_context.target_or_antigen_text",
                    severity="blocking",
                    message="Target / antigen not provided (neither raw context nor structured_query)",
                    category="target",
                    evidence_field=None,
                )
            )
        if not presence.antibody_candidate_present:
            missing.append(
                MissingInputItem(
                    field="user_provided_context.candidate_text",
                    severity="warning",
                    message="No explicit antibody candidate; Step 5 will rely on discovery",
                    category="antibody",
                )
            )
        if not presence.payload_present:
            missing.append(
                MissingInputItem(
                    field="user_provided_context.payload_linker_text",
                    severity="warning",
                    message="Payload not detected; Step 6 compound lanes will be partial/skipped",
                    category="payload_or_linker",
                )
            )
        if not presence.linker_present:
            missing.append(
                MissingInputItem(
                    field="user_provided_context.payload_linker_text",
                    severity="optional",
                    message="Linker not specified; defaults may be assumed downstream",
                    category="payload_or_linker",
                )
            )
        if not presence.structure_or_sequence_present:
            missing.append(
                MissingInputItem(
                    field="uploaded_files or structured_query.referenced_inputs",
                    severity="optional",
                    message="No structure/sequence reference found; Step 7-9 structure lanes will be partial",
                    category="structure_or_sequence",
                )
            )

        blocking = [m.message for m in missing if m.severity == "blocking"]
        if blocking:
            status_val: str = "blocked"
        elif any(m.severity == "warning" for m in missing):
            status_val = "needs_user_input"
        else:
            status_val = "ready"

        status = InputReadinessStatus(
            run_id=run_id,
            checked_at=now_iso(),
            source_refs=SourceRefs(raw_request_record_id=raw_id, structured_query_id=sq_id),
            input_readiness_status=status_val,  # type: ignore[arg-type]
            readiness_summary=f"{len(missing)} gap(s) identified",
            basic_adc_input_presence=presence,
            uploaded_file_checks=file_checks,
            missing_input_checklist=missing,
            blocking_reasons=blocking,
        )

        artifact_id = new_artifact_id("input_readiness_status")
        self.storage.write_json(
            self.storage.run_key(run_id, _ARTIFACT_KEY),
            {"artifact_id": artifact_id, **status.model_dump()},
        )
        self.registry.update_active(run_id, input_readiness_status_id=artifact_id)
        self.workflow_state.mark(run_id, "step_03", "completed")
        return status
