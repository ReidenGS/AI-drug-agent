"""Step 3 — InputReadinessService (deterministic input completeness check).

Reads BOTH the Step 1 `raw_request_record` and the Step 2 `structured_query`,
then judges readiness from the combined signal. The service is deterministic
by design — it does NOT use an LLM, does NOT call MCP tools, and does NOT
parse PDB/FASTA/CSV file contents. It looks only at metadata: filename
extension, content_type, sha256 / storage_path presence, plus the entities
and referenced IDs already produced by Step 2.

Signals (per `\u9879\u76ee\u6587\u4ef6/Step1_4_Orchestration_Component_Plan_v0.1.md §Step 3`):

- `adc_task_intent_present` — Step 2 says the user is asking for ADC work.
- `target_or_antigen_present` — context or Step 2 entity.
- `antibody_candidate_present` — context or Step 2 entity or FASTA upload.
- `payload_present` / `linker_present` — Step 2 entities or
  `payload_linker_text`.
- `structure_input_present` — PDB/CIF upload OR `pdb_id` reference.
- `sequence_input_present` — FASTA upload OR `uniprot_id` reference OR
  explicit chain sequence on a Step 2 entity.
- `structure_or_sequence_present` — either of the above (kept for back-
  compat with Step 5+ agents).
- `candidate_file_present` — CSV/XLSX/JSON upload (table of candidates).
- `constraints_present` — Step 2 user_constraints or raw constraints_text.

Severity policy:

| Gap | Severity | Run status floor |
|---|---|---|
| No `raw_user_query` at all | `blocking` | `blocked` |
| No ADC task intent (modality unknown / not ADC) | `blocking` | `blocked` |
| Missing target / antigen | `blocking` | `blocked` |
| Missing antibody candidate | `warning` | `needs_user_input` |
| Missing payload | `warning` | `needs_user_input` |
| Missing linker | `optional` | (does not block ready) |
| Missing structure / sequence | `optional` | (does not block ready) |
| Uploaded file missing `storage_path` | `warning` (or `blocking` if it is the *only* candidate file) | `needs_user_input` / `blocked` |
| Uploaded file with `unknown` role | `warning` | `needs_user_input` |
"""

from __future__ import annotations

import hashlib
from pathlib import PurePosixPath

from ..schemas.step_03_input_readiness import (
    BasicADCInputPresence,
    ClarificationRequest,
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


# Deterministic, non-LLM fallback questions keyed by gap category / slot.
# Used only when a Step 2 `missing_slot` carried no `suggested_question`.
_CLARIFICATION_FALLBACK_QUESTIONS = {
    "target": "What target or antigen should the ADC be designed against?",
    "antibody": "Which antibody candidate should we use, or should we run discovery?",
    "payload_or_linker": "Which payload and linker should the ADC use?",
    "payload": "Which payload should the ADC carry?",
    "linker": "Which linker chemistry should we use?",
    "structure_or_sequence": "Please provide a PDB/CIF file, PDB ID, UniProt ID, or protein sequence.",
    "structure": "Please provide a PDB/CIF file, PDB ID, UniProt ID, or protein sequence.",
    "sequence": "Please provide a protein or antibody sequence, or a UniProt ID.",
    "identifier": "Could you provide the relevant identifier (PDB ID, UniProt ID, or SMILES)?",
    "task_intent": "What workflow should we run (design, evaluate, screen, or review)?",
    "constraint": "Are there any constraints we should respect (e.g. DAR or affinity)?",
    "constraints": "Are there any constraints we should respect (e.g. DAR or affinity)?",
    "raw_user_query": "Could you describe what you'd like the pipeline to do?",
    "uploaded_file": "Could you re-check or re-upload the referenced file?",
    "other": "Could you provide more detail for this requirement?",
}


def _fallback_question(category: str, slot_name: str, reason: str) -> str:
    q = _CLARIFICATION_FALLBACK_QUESTIONS.get(category) or _CLARIFICATION_FALLBACK_QUESTIONS.get(
        slot_name
    )
    if q:
        return q
    reason = (reason or "").strip()
    if reason:
        return f"Could you clarify: {reason}"
    return _CLARIFICATION_FALLBACK_QUESTIONS["other"]


def _clarification_request_id(slot_name: str, slot_category: str, severity: str, question: str) -> str:
    """Stable id from slot identity + a short hash of the question content.

    Deterministic (no UUID / timestamp) so the same gap yields the same id
    across runs, which lets an answer store dedupe without churn.
    """
    raw = "|".join([slot_name, slot_category, severity, question])
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:10]
    return f"clr_{slot_name}_{digest}"


# Step 2 `missing_slots` slot_category -> Step 3 `GapCategory`. Keeps the
# structured Step 2 gap channel aligned with Step 3's checklist taxonomy.
_MISSING_SLOT_CATEGORY_TO_GAP = {
    "target": "target",
    "antibody": "antibody",
    "payload": "payload_or_linker",
    "linker": "payload_or_linker",
    "structure": "structure_or_sequence",
    "sequence": "structure_or_sequence",
    "identifier": "structure_or_sequence",
    "task_intent": "task_intent",
    "constraint": "constraints",
    "other": "other",
}


# `referenced_inputs` id_types that count as a protein/antibody SEQUENCE
# input (alongside FASTA uploads and an inline chain on a Step 2 entity).
_ANTIBODY_SEQUENCE_REF_ID_TYPES = {
    "antibody_heavy_chain_sequence",
    "antibody_light_chain_sequence",
    "antibody_sequence_reference",
}
_SEQUENCE_REF_ID_TYPES = {"uniprot_id", *_ANTIBODY_SEQUENCE_REF_ID_TYPES}

# Intents where Step 2's `missing_slots` (per its required_slot_schema) owns
# the gap assessment, so Step 3 must NOT re-impose the legacy new-ADC-design
# checklist (target / antibody / payload / linker). For these the readiness
# floor comes from `missing_slots` + presence signals only.
_NON_DESIGN_INTENTS = {
    "developability_assessment",
    "structure_analysis",
    "compound_screening",
    "literature_review",
    "patent_ip_review",
}


_PDB_EXTS = {".pdb", ".cif", ".mmcif", ".ent"}
_FASTA_EXTS = {".fasta", ".fa", ".faa", ".seq"}
_CSV_EXTS = {".csv", ".tsv", ".xlsx", ".xls"}
_JSON_EXTS = {".json"}
_IMG_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".svg"}
_DOC_EXTS = {".txt", ".md", ".pdf", ".docx"}


# `inferred_role` Literal in the schema is fixed; we only expose its values
# here. `.txt` / `.pdf` / `.docx` fall under `unknown` until the schema is
# widened — a `notes` line on the file check records the original extension.

def _infer_file_role(filename: str, content_type: str | None) -> tuple[str, str | None]:
    """Return (inferred_role, note).

    `note` carries a short human-readable hint when the extension/content type
    suggests a document we recognize but do not have a dedicated enum value
    for (`.txt`, `.pdf`, `.docx`). We keep `role="unknown"` in that case so
    we never claim a candidate-file or structure role we cannot back up.
    """
    ext = PurePosixPath(filename or "").suffix.lower()
    ct = (content_type or "").lower()

    if ext in _PDB_EXTS or "pdb" in ct or "chemical/x-pdb" in ct:
        return "pdb_or_cif_structure", None
    if ext in _FASTA_EXTS or "fasta" in ct or "x-fasta" in ct:
        return "fasta_sequence", None
    if (
        ext in _CSV_EXTS
        or "csv" in ct
        or "tab-separated" in ct
        or "spreadsheet" in ct
        or "ms-excel" in ct
    ):
        return "csv_or_table", None
    if ext in _JSON_EXTS or "json" in ct:
        return "json_metadata", None
    if ext in _IMG_EXTS or ct.startswith("image/"):
        return "image", None
    if ext in _DOC_EXTS or ct in {"text/plain", "application/pdf"}:
        return "unknown", f"document-like extension {ext or ct} — not classified as input"
    return "unknown", None


def _referenced_id_types(structured_query: dict) -> set[str]:
    return {
        ref.get("id_type", "")
        for ref in (structured_query.get("referenced_inputs") or [])
        if isinstance(ref, dict)
    }


def _has_explicit_chain_sequence(entities: dict) -> bool:
    """Detect a chain-sequence-like value on a Step 2 entity dict.

    The Step 2 schema's `MentionedEntities` is intentionally flat strings;
    this helper just looks for plausibly-long uppercase amino-acid runs in
    the antibody candidate text, which is how operators sometimes paste a
    raw sequence into the candidate textbox.
    """
    candidate = (entities or {}).get("antibody_candidate_text") or ""
    if not isinstance(candidate, str):
        return False
    upper = "".join(ch for ch in candidate if ch.isalpha())
    if len(upper) < 40:
        return False
    aa_set = set("ACDEFGHIKLMNPQRSTVWY")
    upper_only = upper.upper()
    return all(ch in aa_set for ch in upper_only)


def _is_adc_task_intent(structured_query: dict) -> tuple[bool, str | None]:
    """Decide whether Step 2's task_intent endorses an ADC design task.

    Conservative: requires modality == "ADC" OR an adc-flavored
    `task_type` OR an adc-flavored `primary_intent` (Step 2 batch 5).
    Anything else (e.g. `modality="unknown"`, `task_type="unknown"`)
    returns False so Step 3 can surface a blocking checklist item rather
    than silently letting non-ADC requests through.
    """
    intent = (structured_query or {}).get("task_intent") or {}
    modality = str(intent.get("modality") or "").strip().lower()
    task_type = str(intent.get("task_type") or "").strip().lower()
    primary_intent = str(intent.get("primary_intent") or "").strip().lower()
    if modality == "adc":
        return True, "structured_query.task_intent.modality=ADC"
    adc_task_types = {
        "adc_design",
        "candidate_screening",
        "candidate_evaluation",
        "structure_preparation",
        "optimization",
    }
    if task_type in adc_task_types and modality in {"", "unknown"}:
        # task_type alone is weaker evidence; still allow but record it.
        return True, f"structured_query.task_intent.task_type={task_type}"
    # Step 2 batch-5 primary_intent values that clearly imply ADC work.
    adc_primary_intents = {
        "new_adc_design",
        "existing_adc_evaluation",
        "developability_assessment",
        "structure_analysis",
        "compound_screening",
        "optimization",
    }
    if primary_intent in adc_primary_intents and modality in {"", "unknown"}:
        return True, f"structured_query.task_intent.primary_intent={primary_intent}"
    return False, None


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

        raw_user_query = (raw.get("raw_user_query") or "").strip()
        ctx = raw.get("user_provided_context") or {}
        entities = sq.get("mentioned_entities") or {}
        ref_id_types = _referenced_id_types(sq)
        uploaded_files = raw.get("uploaded_files") or []

        # ── file checks ────────────────────────────────────────────────────
        file_checks: list[UploadedFileCheck] = []
        any_structure_file = False
        any_sequence_file = False
        any_candidate_file = False
        candidate_file_evidence: str | None = None
        structure_file_evidence: str | None = None
        sequence_file_evidence: str | None = None
        failed_upload_count = 0
        unknown_role_count = 0
        for f in uploaded_files:
            role, note = _infer_file_role(
                f.get("original_filename", ""), f.get("content_type")
            )
            storage_path_present = bool(f.get("storage_path"))
            if not storage_path_present:
                failed_upload_count += 1
            if role == "unknown":
                unknown_role_count += 1
            file_checks.append(
                UploadedFileCheck(
                    file_id=f["file_id"],
                    exists=storage_path_present,
                    checksum_ok=bool(f.get("sha256")),
                    format_ok=role != "unknown",
                    inferred_role=role,  # type: ignore[arg-type]
                    storage_path_present=storage_path_present,
                    content_type_present=bool(f.get("content_type")),
                    size_bytes_present=f.get("size_bytes") is not None,
                    notes=note,
                )
            )
            if role == "pdb_or_cif_structure":
                any_structure_file = True
                if structure_file_evidence is None:
                    structure_file_evidence = (
                        "raw_request_record.uploaded_files[*].inferred_role=pdb_or_cif_structure"
                    )
            if role == "fasta_sequence":
                any_sequence_file = True
                if sequence_file_evidence is None:
                    sequence_file_evidence = (
                        "raw_request_record.uploaded_files[*].inferred_role=fasta_sequence"
                    )
            if role == "csv_or_table":
                any_candidate_file = True
                if candidate_file_evidence is None:
                    candidate_file_evidence = (
                        "raw_request_record.uploaded_files[*].inferred_role=csv_or_table"
                    )

        # ── ADC task intent ────────────────────────────────────────────────
        adc_task_intent, adc_intent_ev = _is_adc_task_intent(sq)

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

        # Split structure / sequence signals (the combined field stays for
        # downstream agents that already read it). Heavy/light/generic
        # antibody chain references in `referenced_inputs` are sequence input.
        antibody_seq_ref_types = sorted(ref_id_types & _ANTIBODY_SEQUENCE_REF_ID_TYPES)
        has_antibody_seq_ref = bool(antibody_seq_ref_types)
        structure_present = any_structure_file or "pdb_id" in ref_id_types
        sequence_present = (
            any_sequence_file
            or bool(ref_id_types & _SEQUENCE_REF_ID_TYPES)
            or _has_explicit_chain_sequence(entities)
        )
        structure_or_sequence_present = structure_present or sequence_present

        # Evidence picks the strongest source for each.
        structure_input_ev = None
        if any_structure_file:
            structure_input_ev = structure_file_evidence
        elif "pdb_id" in ref_id_types:
            structure_input_ev = "structured_query.referenced_inputs[id_type=pdb_id]"
        sequence_input_ev = None
        if any_sequence_file:
            sequence_input_ev = sequence_file_evidence
        elif has_antibody_seq_ref:
            sequence_input_ev = (
                "structured_query.referenced_inputs[id_type="
                f"{antibody_seq_ref_types[0]}]"
            )
        elif "uniprot_id" in ref_id_types:
            sequence_input_ev = "structured_query.referenced_inputs[id_type=uniprot_id]"
        elif _has_explicit_chain_sequence(entities):
            sequence_input_ev = (
                "structured_query.mentioned_entities.antibody_candidate_text~chain_sequence"
            )

        # Back-compat combined evidence — preserve previous behavior so
        # existing deeper tests still pass.
        structure_ev = None
        if any_structure_file:
            structure_ev = "raw_request_record.uploaded_files[*].inferred_role=pdb_or_cif_structure"
        elif "pdb_id" in ref_id_types:
            structure_ev = "structured_query.referenced_inputs[id_type=pdb_id]"
        elif any_sequence_file:
            structure_ev = "raw_request_record.uploaded_files[*].inferred_role=fasta_sequence"
        elif has_antibody_seq_ref:
            structure_ev = (
                "structured_query.referenced_inputs[id_type="
                f"{antibody_seq_ref_types[0]}]"
            )
        elif "uniprot_id" in ref_id_types:
            structure_ev = "structured_query.referenced_inputs[id_type=uniprot_id]"

        constraints_present = bool(ctx.get("constraints_text") or sq.get("user_constraints"))
        constraints_ev = (
            "raw_request_record.user_provided_context.constraints_text"
            if ctx.get("constraints_text")
            else ("structured_query.user_constraints" if sq.get("user_constraints") else None)
        )

        antibody_present = (
            bool(candidate_text) or any_sequence_file or has_antibody_seq_ref
        )
        antibody_ref_evidence = (
            "structured_query.referenced_inputs[id_type="
            f"{antibody_seq_ref_types[0]}]"
            if has_antibody_seq_ref
            else None
        )
        presence = BasicADCInputPresence(
            target_or_antigen_present=bool(target_text),
            antibody_candidate_present=antibody_present,
            payload_present=bool(payload_text),
            linker_present=bool(linker_text),
            structure_or_sequence_present=structure_or_sequence_present,
            constraints_present=constraints_present,
            adc_task_intent_present=adc_task_intent,
            structure_input_present=structure_present,
            sequence_input_present=sequence_present,
            candidate_file_present=any_candidate_file,
            target_evidence=target_ev,
            antibody_evidence=candidate_ev
            or (sequence_file_evidence if any_sequence_file else None)
            or antibody_ref_evidence,
            payload_evidence=payload_ev,
            linker_evidence=linker_ev,
            structure_or_sequence_evidence=structure_ev,
            constraints_evidence=constraints_ev,
            adc_task_intent_evidence=adc_intent_ev,
            structure_input_evidence=structure_input_ev,
            sequence_input_evidence=sequence_input_ev,
            candidate_file_evidence=candidate_file_evidence,
        )

        # ── gap classification ─────────────────────────────────────────────
        missing: list[MissingInputItem] = []
        if not raw_user_query:
            missing.append(
                MissingInputItem(
                    field="raw_request_record.raw_user_query",
                    severity="blocking",
                    message="No raw_user_query provided — Step 3 cannot judge intent",
                    category="raw_user_query",
                    evidence_field=None,
                )
            )
        if raw_user_query and not presence.adc_task_intent_present:
            intent = (sq or {}).get("task_intent") or {}
            modality = str(intent.get("modality") or "").strip().lower()
            try:
                modality_conf = float(intent.get("modality_confidence") or 0.0)
            except (TypeError, ValueError):
                modality_conf = 0.0
            # Block only when Step 2 confidently said "not ADC" (modality is
            # an explicit non-ADC value with non-trivial confidence). When
            # the modality is `unknown` / unset, downgrade to a warning so
            # the user can be asked to confirm intent before we declare the
            # request blocked.
            non_adc_confident = (
                modality not in {"", "unknown", "adc"} and modality_conf >= 0.5
            )
            severity = "blocking" if non_adc_confident else "warning"
            missing.append(
                MissingInputItem(
                    field="structured_query.task_intent",
                    severity=severity,  # type: ignore[arg-type]
                    message=(
                        "Step 2 did not classify the request as an ADC design "
                        "task (modality / task_type not ADC); confirm intent "
                        "before planning the pipeline"
                    ),
                    category="task_intent",
                    evidence_field=None,
                )
            )
        # The legacy new-ADC-design checklist (target blocking + antibody /
        # payload / linker gaps) only applies when the task is a design-style
        # request. For non-design intents (e.g. developability_assessment on
        # antibody heavy/light sequences) Step 2's `missing_slots` owns the
        # gap assessment, so Step 3 must NOT fabricate a target/payload/linker
        # requirement. `canonical_query`'s "unspecified" wording is never
        # parsed as a real entity here.
        primary_intent = str(
            ((sq or {}).get("task_intent") or {}).get("primary_intent") or ""
        ).strip().lower()
        apply_legacy_adc_checklist = primary_intent not in _NON_DESIGN_INTENTS

        if apply_legacy_adc_checklist:
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
                    message=(
                        "No structure/sequence reference found; Step 7-9 "
                        "structure lanes will be partial"
                    ),
                    category="structure_or_sequence",
                )
            )

        # Upload failures — `storage_path` missing means intake never saved
        # the file. Don't block on it unless the missing file was the only
        # signal we had for an otherwise-mandatory input.
        for fc in file_checks:
            if not fc.storage_path_present:
                sole_signal = (
                    fc.inferred_role == "pdb_or_cif_structure"
                    and not presence.target_or_antigen_present
                )
                missing.append(
                    MissingInputItem(
                        field=f"raw_request_record.uploaded_files[file_id={fc.file_id}].storage_path",
                        severity="blocking" if sole_signal else "warning",
                        message=(
                            f"Uploaded file {fc.file_id} has no storage_path; "
                            "intake did not persist it"
                        ),
                        category="uploaded_file",
                        evidence_field=None,
                    )
                )
            elif fc.inferred_role == "unknown":
                missing.append(
                    MissingInputItem(
                        field=(
                            f"raw_request_record.uploaded_files"
                            f"[file_id={fc.file_id}].inferred_role"
                        ),
                        severity="warning",
                        message=(
                            f"Uploaded file {fc.file_id} could not be "
                            "classified by extension or content_type"
                        ),
                        category="uploaded_file",
                        evidence_field=None,
                    )
                )

        # ── Step 2 structured missing_slots consumption (minimal) ───────────
        # Step 3 reflects the LLM-judged required-slot gaps reported by
        # Step 2: a `blocking` slot floors readiness to `blocked` and shows
        # up in the checklist; `warning` / `optional` slots are informational
        # gaps that never block on their own. We dedupe a slot against an
        # existing deterministic item only when that item already covers the
        # same category at the same-or-higher severity, so a genuinely new
        # blocking slot is always surfaced.
        self._consume_missing_slots(sq, missing)

        blocking = [m.message for m in missing if m.severity == "blocking"]
        if blocking:
            status_val: str = "blocked"
        elif any(m.severity == "warning" for m in missing):
            status_val = "needs_user_input"
        else:
            status_val = "ready"

        # Clarification requests are built from the SAME gaps but are NOT
        # subject to the checklist dedupe, so a Step 2 `suggested_question`
        # survives even when its category was deduped from the checklist.
        clarification_requests = self._build_clarification_requests(sq, missing)

        # User-facing response is a pure passthrough of Step 2's LLM-written
        # `structured_query.response` when readiness is not `ready`. Step 3
        # NEVER calls an LLM; if Step 2 left it empty we deterministically
        # join the clarification questions as a fallback.
        response = self._resolve_response(sq, status_val, clarification_requests)

        summary = self._readiness_summary(status_val, missing, presence)
        status = InputReadinessStatus(
            run_id=run_id,
            checked_at=now_iso(),
            source_refs=SourceRefs(raw_request_record_id=raw_id, structured_query_id=sq_id),
            input_readiness_status=status_val,  # type: ignore[arg-type]
            readiness_summary=summary,
            basic_adc_input_presence=presence,
            uploaded_file_checks=file_checks,
            missing_input_checklist=missing,
            blocking_reasons=blocking,
            clarification_requests=clarification_requests,
            response=response,
        )

        artifact_id = new_artifact_id("input_readiness_status")
        self.storage.write_json(
            self.storage.run_key(run_id, _ARTIFACT_KEY),
            {"artifact_id": artifact_id, **status.model_dump()},
        )
        self.registry.update_active(run_id, input_readiness_status_id=artifact_id)
        self.workflow_state.mark(run_id, "step_03", "completed")
        return status

    @staticmethod
    def _consume_missing_slots(
        structured_query: dict, missing: list[MissingInputItem]
    ) -> None:
        """Append Step 2 `missing_slots` to the Step 3 checklist (in place).

        Deterministic, additive, and backward compatible: when Step 2 did
        not populate `missing_slots` (old artifacts) this is a no-op. A slot
        is skipped only when an existing deterministic item already covers
        the same category at the same-or-higher severity, avoiding duplicate
        lines while always surfacing a new/stronger gap.
        """
        slots = structured_query.get("missing_slots") or []
        if not isinstance(slots, list):
            return
        severity_rank = {"optional": 0, "warning": 1, "blocking": 2}
        existing: dict[str, int] = {}
        for item in missing:
            rank = severity_rank.get(item.severity, 1)
            if rank > existing.get(item.category, -1):
                existing[item.category] = rank

        for slot in slots:
            if not isinstance(slot, dict):
                continue
            severity = slot.get("severity")
            if severity not in severity_rank:
                severity = "warning"
            slot_name = slot.get("slot_name") or "other"
            category = _MISSING_SLOT_CATEGORY_TO_GAP.get(
                slot.get("slot_category"), "other"
            )
            slot_rank = severity_rank[severity]
            if existing.get(category, -1) >= slot_rank:
                # Already represented at >= severity by a deterministic check.
                continue
            reason = slot.get("reason") or (
                f"Step 2 reported a missing required slot: {slot_name}"
            )
            question = slot.get("suggested_question")
            message = reason
            if isinstance(question, str) and question.strip():
                message = f"{reason} Suggested question: {question.strip()}"
            evidence = slot.get("evidence")
            missing.append(
                MissingInputItem(
                    field=f"structured_query.missing_slots[slot_name={slot_name}]",
                    severity=severity,  # type: ignore[arg-type]
                    message=message,
                    category=category,  # type: ignore[arg-type]
                    evidence_field=evidence if isinstance(evidence, str) else None,
                )
            )
            existing[category] = slot_rank

    @staticmethod
    def _build_clarification_requests(
        structured_query: dict, missing: list[MissingInputItem]
    ) -> list[ClarificationRequest]:
        """Turn required-slot gaps into stable, user-facing questions.

        Deterministic and additive. Step 2 `missing_slots` are the primary
        source and are read DIRECTLY (not via the deduped checklist) so a
        Step 2 `suggested_question` is never lost to checklist dedupe. A
        deterministic gap only becomes a request when its category was not
        already covered by a Step 2 slot. `blocking` / `warning` gaps yield
        requests; `optional` gaps stay checklist-only (less noise — see the
        dedicated test). Nothing here calls an LLM.
        """
        severity_with_question = {"blocking", "warning"}
        requests: list[ClarificationRequest] = []
        seen_ids: set[str] = set()
        covered_categories: set[str] = set()

        def _add(slot_name, slot_category, gap_category, severity, question, reason, source, evidence):
            request_id = _clarification_request_id(
                slot_name, slot_category, severity, question
            )
            if request_id in seen_ids:
                return
            seen_ids.add(request_id)
            covered_categories.add(gap_category)
            requests.append(
                ClarificationRequest(
                    request_id=request_id,
                    slot_name=slot_name,
                    slot_category=slot_category,
                    severity=severity,  # type: ignore[arg-type]
                    question=question,
                    reason=reason,
                    source=source,
                    evidence_field=evidence if isinstance(evidence, str) else None,
                )
            )

        # 1. Step 2 missing_slots — preserve the LLM's suggested_question.
        for slot in structured_query.get("missing_slots") or []:
            if not isinstance(slot, dict):
                continue
            severity = slot.get("severity")
            if severity not in {"blocking", "warning", "optional"}:
                severity = "warning"
            if severity not in severity_with_question:
                continue  # optional → checklist-only, no question
            slot_name = slot.get("slot_name") or "other"
            slot_category = slot.get("slot_category") or "other"
            gap_category = _MISSING_SLOT_CATEGORY_TO_GAP.get(slot_category, "other")
            reason = (slot.get("reason") or "").strip()
            suggested = slot.get("suggested_question")
            question = (
                suggested.strip()
                if isinstance(suggested, str) and suggested.strip()
                else _fallback_question(gap_category, slot_name, reason)
            )
            _add(
                slot_name, slot_category, gap_category, severity, question, reason,
                "step2_missing_slots", slot.get("evidence"),
            )

        # 2. Deterministic gaps not already covered by a Step 2 slot category.
        for item in missing:
            if item.field.startswith("structured_query.missing_slots"):
                continue  # already represented via the Step 2 loop above
            if item.severity not in severity_with_question:
                continue
            if item.category in covered_categories:
                continue
            question = _fallback_question(item.category, item.category, item.message)
            _add(
                item.category, item.category, item.category, item.severity,
                question, item.message, "deterministic_readiness", item.evidence_field,
            )

        return requests

    @staticmethod
    def _resolve_response(
        structured_query: dict,
        status_val: str,
        clarification_requests: list[ClarificationRequest],
    ) -> str | None:
        """Pass through Step 2's user-facing `response`, or fall back.

        Pure passthrough + deterministic fallback — NO LLM call. When
        readiness is `ready` there is nothing to ask, so `None`. Otherwise
        prefer Step 2's `structured_query.response`; if Step 2 left it empty,
        join the clarification questions (blocking first) as a fallback.
        """
        if status_val == "ready":
            return None
        step2_response = structured_query.get("response")
        if isinstance(step2_response, str) and step2_response.strip():
            return step2_response.strip()
        if not clarification_requests:
            return None
        ordered = sorted(
            clarification_requests,
            key=lambda c: 0 if c.severity == "blocking" else 1,
        )
        return " ".join(c.question for c in ordered if c.question)

    @staticmethod
    def _readiness_summary(
        status_val: str,
        missing: list[MissingInputItem],
        presence: BasicADCInputPresence,
    ) -> str:
        if status_val == "ready":
            return "Inputs sufficient for the fixed ADC pipeline."
        blockers = [m.category for m in missing if m.severity == "blocking"]
        warnings = [m.category for m in missing if m.severity == "warning"]
        parts: list[str] = []
        if blockers:
            parts.append(f"blocking: {', '.join(sorted(set(blockers)))}")
        if warnings:
            parts.append(f"warnings: {', '.join(sorted(set(warnings)))}")
        if not parts:
            parts.append("only optional gaps remain")
        return f"{status_val} — " + "; ".join(parts)
