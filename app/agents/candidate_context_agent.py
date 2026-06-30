"""CandidateContextAgent — Step 5.

Builds normalized candidate / material / identifier records from every source
the run has available so far:

- `structured_query.mentioned_entities` (target / antibody / payload / linker)
- `structured_query.referenced_inputs[]` (PDB / UniProt / ZINC / ChEMBL /
  DrugBank / PubChem / SMILES detected by the Step 2 parser)
- `raw_request_record.user_provided_context` (free-text fallback)
- `raw_request_record.uploaded_files[]` (PDB/CIF → structure material,
  FASTA → sequence material)

Per candidate, the agent may call zero or more scoped MCP tools to enrich
context, chosen via the metadata-driven `step_05_enrichment_registry` plus
the LLM Stage-1 relevance selector in `step_05_selection_policy`. The set
of executed tools per candidate is bounded by registry eligibility and
the LLM's relevance picks — never by a hand-coded per-candidate cap. Raw
payloads land in `tool_outputs/step_05/{tool_call_id}.json` and are
referenced via `tool_call_records[].tool_output_ref`. Raw upstream
payloads NEVER appear inside `candidate_records[]`.

Antibody full sequence → CDR3 → IEDB BCR lookup is wired via a synthetic
plan path in `_build_iedb_cdr3_plans` + `_enrich_with_iedb_cdr3`: the raw
CDR3 enters MCP arguments only and is never persisted on the candidate
or forwarded to the LLM payload. See `antibody_cdr3_extraction` for the
abnumber / anarci adapter contract.

ZINC guard: ZINC ids land as `zinc_id` identifiers; no material is marked as
`ZINC22` unless the source record explicitly says so. Step 5 doesn't emit
`CompoundHit` records (those belong to Step 9).
"""

from __future__ import annotations

import re
from pathlib import PurePosixPath
from typing import Any, Iterable, Optional

from ..llm.provider import LLMProvider, MockLLMProvider
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
from .antibody_cdr3_extraction import (
    CHAIN_TYPE_HEAVY,
    CHAIN_TYPE_LIGHT,
    Cdr3Result,
    STATUS_DEPENDENCY_UNAVAILABLE as _CDR3_STATUS_DEPENDENCY_UNAVAILABLE,
    STATUS_EXTRACTION_FAILED as _CDR3_STATUS_EXTRACTION_FAILED,
    STATUS_NO_VARIABLE_DOMAIN as _CDR3_STATUS_NO_VARIABLE_DOMAIN,
    STATUS_SUCCESS as _CDR3_STATUS_SUCCESS,
    extract_cdr3,
)
from .step_05_enrichment_registry import (
    EnrichmentPlan,
    plan_enrichment_for_record,
    skipped_low_information_chembl_name_queries,
)
from .step_05_selection_policy import (
    SELECTION_POLICY_VERSION as STEP5_SELECTION_POLICY_VERSION,
    Step5ToolDecision,
    select_step5_enrichment_plans,
    selection_provenance_for_tool_input_summary,
)


_AGENT_NAME = "candidate_context_agent"
_STEP_ID = "step_05"
_ARTIFACT_KEY = "candidate_context_table.json"


_PDB_EXTS = {".pdb", ".cif", ".mmcif", ".ent"}
_FASTA_EXTS = {".fasta", ".fa", ".faa", ".seq"}

# Step 5 CDR3 → IEDB extension.
_IEDB_BCR_TOOL_NAME = "iedb_search_bcr_sequences"
# Sentinel string written into the synthetic IEDB plan's ``query`` field
# so the agent's executor can recognise it without ever seeing the raw
# CDR3 string. The actual CDR3 lives in a transient stack-frame dict.
_IEDB_TRANSIENT_TOKEN = "<cdr3-transient>"
# Compact ``select`` columns the IEDB BCR endpoint returns. We do NOT
# request anything else — this is the audited downstream-needed subset.
_IEDB_SELECT_COLUMNS: tuple[str, ...] = (
    "receptor_group_id",
    "receptor_type",
    "receptor_name",
    "chain1_cdr3_seq",
    "chain2_cdr3_seq",
    "receptor_chain1_types",
    "receptor_chain2_types",
    "pdb_ids",
    "curated_source_antigens",
)
# Material types the agent treats as candidate-attached antibody
# sequences. ``Step 1 → Step 2`` populates these as paths; the sequence
# string itself only enters memory transiently inside
# ``_build_iedb_cdr3_plans``.
_ANTIBODY_HEAVY_SEQ_MATERIAL_TYPES = (
    "antibody_heavy_chain_sequence",
)
_ANTIBODY_LIGHT_SEQ_MATERIAL_TYPES = (
    "antibody_light_chain_sequence",
)
_ANTIBODY_GENERIC_SEQ_MATERIAL_TYPES = (
    "antibody_sequence_reference",
)
_ANTIBODY_SEQ_REFERENCE_ROLES = (
    "antibody_sequence_reference",
)
# referenced_inputs id_types that carry an inline antibody sequence (Step 2).
_ANTIBODY_SEQUENCE_REF_ID_TYPES = (
    "antibody_heavy_chain_sequence",
    "antibody_light_chain_sequence",
    "antibody_sequence_reference",
)


def _looks_like_antibody_name(text: str | None) -> bool:
    """Heuristic: is ``text`` a plausible antibody NAME (vs a sentence)?

    A real name is short (e.g. "trastuzumab", "Trastuzumab analog", "HER2
    antibody"): few words, no sentence punctuation, not a long phrase. A
    sentence-like label (a developability request paraphrase) must NOT be
    sent to SAbDab / TheraSAbDab as a name query.
    """
    t = (text or "").strip()
    if not t or len(t) > 60:
        return False
    if any(p in t for p in (".", "?", "!", ";", ":", ",")):
        return False
    return len(t.split()) <= 4

_HEAVY_CHAIN_HINTS = {
    "heavy", "heavychain", "heavy_chain", "vh", "hc", "igh", "ighv",
}
_LIGHT_CHAIN_HINTS = {
    "light", "lightchain", "light_chain", "vl", "lc", "kappa", "lambda",
    "igk", "igl", "igkv", "iglv",
}


# Structure-reference role inference. We never open the PDB; we look at
# the filename and the surrounding user text for keywords. Each pattern
# maps a keyword set to a more specific reference role; falls back to
# generic `structure_reference` when nothing matches. Order matters —
# the FIRST matching family wins so the more specific tags (Fab / Fc /
# N297) win over the generic antibody_arm category.
_STRUCTURE_ROLE_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("n297_site_reference",          ("n297", "n-297", "n 297")),
    ("glycan_or_glycosylation_reference",
        ("glycan", "glycosylation", "glycoform", "glyco-")),
    ("linker_attachment_site",       ("attachment_site", "attachment-site",
                                      "linker attachment", "conjugation site")),
    ("fab_structure_reference",      ("fab", "fab_")),
    ("fc_region_reference",          ("fc_", "_fc", "fc region", "fc-region")),
    ("antibody_arm_reference",       ("scfv", "vhh", "nanobody", "antibody_arm",
                                      "antibody-arm", "arm_")),
    ("antigen_structure_reference",  ("antigen", "ectodomain", "extracellular")),
    ("receptor_structure_reference", ("receptor", "her2", "egfr", "trop2",
                                      "tacstd2", "claudin", "cldn", "psma",
                                      "bcma", "ror1", "muc1", "folr1")),
    ("experimental_fragment_reference",
        ("fragment", "_frag", "frag_", "fragment_reference")),
)


def _infer_structure_role(
    filename: str, context_text: str
) -> str:
    """Pick the most specific structure-reference role we can defend.

    Inspects ONLY the filename and the surrounding context_text — never
    opens or reads the PDB file. Returns `structure_reference` as the
    safe fallback when no pattern matches.
    """
    haystack = " ".join([
        (filename or "").lower(),
        (context_text or "").lower(),
    ])
    for role, needles in _STRUCTURE_ROLE_PATTERNS:
        for needle in needles:
            if needle in haystack:
                return role
    return "structure_reference"


# Heuristic — words that signal the user explicitly tagged compounds as
# payload / linker candidates. Without these the IDs stay generic
# material_only entries (no payload_candidate / linker_candidate role).
_COMPOUND_PAYLOAD_KEYWORDS = (
    "payload candidate", "payload candidates",
    "payload reference", "payload references",
    "payload library",
    "as payload", "as payloads",
)
_COMPOUND_LINKER_KEYWORDS = (
    "linker candidate", "linker candidates",
    "linker reference", "linker references",
    "linker library",
)
_COMPOUND_LIGAND_KEYWORDS = (
    "ligand candidate", "ligand candidates", "ligand library",
    "targeting moiety", "targeting moieties",
)

_COMPOUND_HIT_LIST_KEYS = (
    "hits", "results", "molecules", "items", "records", "documents",
)
# Common envelope wrapper keys ToolUniverse and similar adapters use to nest
# the actual hit list one or two levels deep (e.g. live ChEMBL substructure
# search returns `{executor, status, payload: {data: {molecules: [...] }}}`).
# We unwrap these in a *bounded* way — no unbounded deep search through
# arbitrary dicts.
_COMPOUND_ENVELOPE_WRAPPER_KEYS = ("payload", "data", "output")
_MAX_COMPOUND_HITS_UNWRAP_DEPTH = 2
_MAX_COMPOUND_ENRICHMENT_HITS = 3
_LABELED_SMILES_RE = re.compile(
    r"\b(?P<role>payload|linker|compound)\s+smiles\s*[:=]?\s*(?P<value>[A-Za-z0-9@+\-\[\]\(\)=#$%/\\\.]+)",
    re.IGNORECASE,
)
_UNIPROT_ACCESSION_RE = re.compile(
    r"^(?:[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9][A-Z][A-Z0-9]{2}[0-9])(?:-\d+)?$"
)

_MIXED_CONTEXT_MARKERS = ("smiles", ";")


class CandidateContextAgent:
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
        # LLM-assisted Stage 1 tool relevance selection (Step 5). Defaults
        # to the deterministic MockLLMProvider so existing tests and
        # offline runs keep producing the same eligible-set as before
        # (the mock selects every tool whose coarse_input_requirements
        # match the candidate's signals). Production wiring (graph / API)
        # passes the configured provider from ``get_llm_provider()``.
        self.llm = llm or MockLLMProvider()

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
        enrichment_selection_audit: dict[str, dict] = {}
        scoped_step5_tools = self.mcp_client.list_tools(
            agent_name=_AGENT_NAME, step_id=_STEP_ID
        )

        # Batch-6 inputs from Step 2 enrichment.
        normalized_entities = sq.get("normalized_entities") or []
        entity_decompositions = sq.get("entity_decompositions") or []
        raw_user_query = raw.get("raw_user_query") or ""
        raw_user_query_lower = raw_user_query.lower()

        # Did the user explicitly nominate these IDs as payload / linker / ligand candidates?
        explicit_payload_role = any(
            kw in raw_user_query_lower for kw in _COMPOUND_PAYLOAD_KEYWORDS
        )
        explicit_linker_role = any(
            kw in raw_user_query_lower for kw in _COMPOUND_LINKER_KEYWORDS
        )
        explicit_ligand_role = any(
            kw in raw_user_query_lower for kw in _COMPOUND_LIGAND_KEYWORDS
        )

        # ── complete-ADC reference benchmark candidates (T-DM1, T-DXd, …) ──
        reference_adc_records = self._build_reference_adc_candidates(
            entity_decompositions=entity_decompositions,
            normalized_entities=normalized_entities,
            sq_artifact_id=sq_artifact_id,
        )
        candidate_records.extend(reference_adc_records)

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
            raw_user_query=raw_user_query,
            normalized_entities=normalized_entities,
        )
        if target_cand is None:
            missing_flags.append("mentioned_entities.target_or_antigen_text")
        else:
            tcs = self._execute_enrichment_plans(
                run_id=run_id,
                record=target_cand,
                scoped_tools=scoped_step5_tools,
                label="target",
                raw_user_query=raw_user_query,
                selection_audit=enrichment_selection_audit,
            )
            tool_call_records.extend(tcs)
            if tcs:
                target_cand.candidate_notes = _notes_for(tcs[-1], "target context enrichment")
            candidate_records.append(target_cand)

        # ── antibody candidate ──────────────────────────────────────────────
        antibody_text = (
            entities.get("antibody_candidate_text") or ctx.get("candidate_text")
        )
        inline_antibody_sequence_refs = [
            r
            for id_type in _ANTIBODY_SEQUENCE_REF_ID_TYPES
            for r in refs_by_type.get(id_type, [])
        ]
        ab_cand = self._build_antibody_candidate(
            antibody_text=antibody_text,
            sequence_files=sequence_files,
            inline_sequence_refs=inline_antibody_sequence_refs,
            sq_artifact_id=sq_artifact_id,
        )
        if ab_cand is None:
            missing_flags.append("mentioned_entities.antibody_candidate_text")
        else:
            tcs = self._execute_enrichment_plans(
                run_id=run_id,
                record=ab_cand,
                scoped_tools=scoped_step5_tools,
                label="antibody",
                raw_user_query=raw_user_query,
                selection_audit=enrichment_selection_audit,
            )
            tool_call_records.extend(tcs)
            if tcs:
                ab_cand.candidate_notes = _notes_for(tcs[-1], "antibody context enrichment")
            candidate_records.append(ab_cand)

        # ── payload / linker / generic compound candidates ──────────────────
        payload_text = entities.get("payload_text") or ctx.get("payload_linker_text")
        linker_text = entities.get("linker_text")
        smiles_refs = refs_by_type.get("smiles", [])
        labeled_smiles_refs = _extract_labeled_smiles_refs(raw_user_query, ctx)
        chembl_refs = refs_by_type.get("chembl_id", [])
        pubchem_refs = refs_by_type.get("pubchem_cid", [])
        zinc_refs = refs_by_type.get("zinc_id", [])
        drugbank_refs = refs_by_type.get("drugbank_id", [])

        compound_cands = self._build_compound_candidates(
            payload_text=payload_text,
            linker_text=linker_text,
            smiles_refs=smiles_refs,
            payload_smiles_refs=labeled_smiles_refs.get("payload", []),
            linker_smiles_refs=labeled_smiles_refs.get("linker", []),
            compound_smiles_refs=labeled_smiles_refs.get("compound", []),
            chembl_refs=chembl_refs,
            pubchem_refs=pubchem_refs,
            zinc_refs=zinc_refs,
            drugbank_refs=drugbank_refs,
            sq_artifact_id=sq_artifact_id,
            entity_decompositions=entity_decompositions,
            explicit_payload_role=explicit_payload_role,
            explicit_linker_role=explicit_linker_role,
            explicit_ligand_role=explicit_ligand_role,
        )
        if not compound_cands:
            missing_flags.append("mentioned_entities.payload_text")
        for cc in compound_cands:
            tcs = self._execute_enrichment_plans(
                run_id=run_id,
                record=cc,
                scoped_tools=scoped_step5_tools,
                label="compound",
                raw_user_query=raw_user_query,
                selection_audit=enrichment_selection_audit,
            )
            tool_call_records.extend(tcs)
            if tcs:
                cc.candidate_notes = _notes_for(tcs[-1], "compound context enrichment")
            candidate_records.append(cc)

        # ── status + persist ────────────────────────────────────────────────
        success_count = sum(1 for tc in tool_call_records if tc.run_status == "success")
        if missing_flags or success_count < len(tool_call_records):
            build_status = "partial"
        elif tool_call_records:
            build_status = "ok"
        else:
            build_status = "failed"

        # ── data-gap annotation + downstream hints (batch 6) ───────────────
        _annotate_data_gaps(
            candidate_records,
            target_text=target_text,
            antibody_text=antibody_text,
            payload_text=payload_text,
            linker_text=linker_text,
        )
        # Generalised per-candidate dedup. Different paths through
        # the build / enrichment / normalization layers can converge on
        # the same typed material or the same typed identifier for a
        # candidate (e.g. HER2 / ERBB2 both resolve to UniProt P04626,
        # uploaded heavy/light FASTA contributes a duplicate
        # ``target_sequence`` entry, etc.). The dedup key is strict —
        # different roles, different sources, or different stored
        # values are never merged — so payload vs linker, heavy vs
        # light, and structure files at different storage paths stay
        # distinct. Identifier provenance is preserved by merging
        # ``source_ids`` and keeping the higher ``confidence`` value.
        for rec in candidate_records:
            rec.materials = _dedupe_candidate_materials(rec.materials)
            rec.identifiers = _dedupe_candidate_identifiers(rec.identifiers)

        downstream_hints = _build_downstream_query_hints(
            candidate_records=candidate_records,
            normalized_entities=normalized_entities,
            target_text=target_text,
            antibody_text=antibody_text,
            payload_text=payload_text,
            linker_text=linker_text,
            entity_decompositions=entity_decompositions,
        )

        table = CandidateContextTable(
            run_id=run_id,
            created_at=now_iso(),
            context_build_status=build_status,  # type: ignore[arg-type]
            candidate_records=candidate_records,
            missing_context_flags=missing_flags,
            tool_call_records=tool_call_records,
            downstream_query_hints=downstream_hints,
            enrichment_selection_audit=enrichment_selection_audit,
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
        raw_user_query: str = "",
        normalized_entities: list[dict] | None = None,
    ) -> Optional[CandidateRecord]:
        if not (target_text or refs_by_type.get("uniprot_id") or refs_by_type.get("pdb_id")
                or structure_files):
            return None

        materials: list[Material] = []
        if target_text:
            materials.append(
                self._material_with_role(
                    "target_antigen_name", target_text, "text",
                    role="target", role_status="explicit",
                )
            )

        ctx_for_role = " ".join(
            [target_text or "", raw_user_query or ""]
        )
        for f in structure_files:
            filename = f.get("original_filename") or ""
            role = _infer_structure_role(filename, ctx_for_role)
            materials.append(
                self._material_with_role(
                    "structure_file",
                    f.get("storage_path") or filename or "",
                    _format_for_file(f),
                    role=role,
                    role_status="explicit" if role != "structure_reference" else "explicit",
                )
            )
        for ref in refs_by_type.get("pdb_id", []):
            role = _infer_structure_role("", ctx_for_role)
            materials.append(
                self._material_with_role(
                    "structure_ref", ref["value"], "text",
                    role=role, role_status="explicit",
                )
            )
        for f in sequence_files:
            materials.append(
                self._material_with_role(
                    "target_sequence",
                    f.get("storage_path") or f.get("original_filename") or "",
                    "fasta",
                    role="target_sequence_reference", role_status="explicit",
                )
            )

        identifiers = _identifiers_for(
            refs_by_type,
            ("uniprot_id", "pdb_id"),
            source_ids=[sq_artifact_id],
        )
        context_notes: list[str] = []
        data_gaps: list[str] = []
        _extend_target_identifiers_from_normalized_entities(
            identifiers,
            normalized_entities or [],
            sq_artifact_id=sq_artifact_id,
            context_notes=context_notes,
            data_gaps=data_gaps,
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
            candidate_role="partial_context",
            is_generated_candidate=False,
            context_status="partial",
            data_gaps=data_gaps,
            context_notes=context_notes,
        )

    def _build_antibody_candidate(
        self,
        *,
        antibody_text: Optional[str],
        sequence_files: list[dict],
        inline_sequence_refs: list[dict] | None = None,
        sq_artifact_id: str,
    ) -> Optional[CandidateRecord]:
        inline_sequence_refs = inline_sequence_refs or []
        if not (antibody_text or sequence_files or inline_sequence_refs):
            return None
        materials: list[Material] = []
        context_notes: list[str] = []
        has_sequence_input = bool(sequence_files or inline_sequence_refs)
        # Only emit an `antibody_name` material (which drives SAbDab /
        # TheraSAbDab name lookup) when the text is a plausible antibody NAME.
        # A sentence-like label on a sequence-only developability request must
        # NOT be sent as a name query — skip it and record a compact reason.
        if antibody_text:
            if _looks_like_antibody_name(antibody_text):
                materials.append(
                    self._material_with_role(
                        "antibody_name", antibody_text, "text",
                        role="antibody", role_status="explicit",
                    )
                )
            elif has_sequence_input:
                context_notes.append(
                    "antibody_name_lookup_skipped:sequence_only_input "
                    "(antibody label is not a usable name query)"
                )
        for f in sequence_files:
            material_type = _infer_antibody_sequence_material_type(f)
            materials.append(
                self._material_with_role(
                    material_type,
                    f.get("storage_path") or f.get("original_filename") or "",
                    "fasta",
                    role="antibody_sequence_reference", role_status="explicit",
                )
            )
        # Inline heavy/light/generic antibody sequences from Step 2
        # referenced_inputs become distinct candidate materials carrying the
        # inline amino-acid sequence (value_format reflects inline AA, not a
        # fasta ref). Step 6 available_fields exposes only a digest of these;
        # the raw sequence is never surfaced to the LLM / audit / summaries.
        for ref in inline_sequence_refs:
            if not isinstance(ref, dict):
                continue
            id_type = str(ref.get("id_type") or "")
            value = str(ref.get("value") or "").strip()
            if id_type not in _ANTIBODY_SEQUENCE_REF_ID_TYPES or not value:
                continue
            materials.append(
                self._material_with_role(
                    id_type, value, "amino_acid_sequence",
                    role="antibody_sequence_reference", role_status="explicit",
                )
            )
        if not materials:
            return None
        record = CandidateRecord(
            candidate_id=new_artifact_id("candidate"),
            candidate_label=antibody_text or "antibody_from_sequence_input",
            candidate_type="antibody",
            source_records=[sq_artifact_id] if sq_artifact_id else [],
            identifiers=[],
            materials=materials,
            adc_links=ADCLinks(),
            candidate_status="partially_ready_for_step6",
            candidate_notes=None,
            candidate_role="user_provided_candidate",
            is_generated_candidate=False,
            context_status="partial",
        )
        for note in context_notes:
            if note not in record.context_notes:
                record.context_notes.append(note)
        return record

    def _build_reference_adc_candidates(
        self,
        *,
        entity_decompositions: list[dict],
        normalized_entities: list[dict],
        sq_artifact_id: str,
    ) -> list[CandidateRecord]:
        """Create reference_benchmark candidates for whole ADC aliases.

        T-DM1 / T-DXd / Enhertu and similar — when the user wrote the
        whole-drug alias, Step 5 surfaces them so Step 13/14 can search
        by their name. They are NEVER `is_generated_candidate=True`, and
        their components are stored as `materials[*]` with `role_status`
        tracking whether the user wrote each component explicitly.
        """
        out: list[CandidateRecord] = []
        norm_by_original = {
            (ne.get("original_text") or "").lower(): ne
            for ne in normalized_entities if isinstance(ne, dict)
        }
        for decomp in entity_decompositions:
            if not isinstance(decomp, dict):
                continue
            original = decomp.get("original_text") or ""
            canonical = decomp.get("canonical_name") or original
            # Only treat as a reference benchmark when the alias resolves
            # to a `drug` entity_type in normalized_entities (whole ADC
            # products such as T-DM1 / T-DXd / Enhertu). Skip composite
            # linker_payload aliases like vc-MMAE — those are compound
            # materials, handled by `_build_compound_candidates`.
            norm_match = norm_by_original.get(original.lower())
            if not norm_match or norm_match.get("entity_type") != "drug":
                continue

            materials: list[Material] = [
                self._material_with_role(
                    "complete_adc_name", original, "text",
                    role="complete_adc", role_status="explicit",
                ),
                self._material_with_role(
                    "canonical_adc_name", canonical, "text",
                    role="complete_adc", role_status="inferred"
                    if canonical.lower() != original.lower() else "explicit",
                ),
            ]
            for comp in decomp.get("components") or []:
                if not isinstance(comp, dict):
                    continue
                role = comp.get("role") or "other"
                mat_type = {
                    "antibody": "antibody_name",
                    "payload": "payload_name",
                    "linker": "linker_name",
                    "linker_payload": "linker_payload_name",
                }.get(role, "component_name")
                materials.append(
                    self._material_with_role(
                        mat_type, comp.get("canonical_name") or "", "text",
                        role=role,
                        role_status=(
                            "explicit" if comp.get("inferred") is False else "inferred"
                        ),
                    )
                )

            identifiers: list[Identifier] = []
            if norm_match.get("canonical_id"):
                identifiers.append(
                    Identifier(
                        id_type=(
                            (norm_match.get("canonical_id_source") or "drug").lower() + "_id"
                        ),
                        id_value=str(norm_match["canonical_id"]),
                        source_ids=[sq_artifact_id] if sq_artifact_id else [],
                        confidence=0.9,
                    )
                )

            record = CandidateRecord(
                candidate_id=new_artifact_id("candidate"),
                candidate_label=original,
                candidate_type="adc_construct",
                source_records=[sq_artifact_id] if sq_artifact_id else [],
                identifiers=identifiers,
                materials=materials,
                adc_links=ADCLinks(),
                candidate_status="partially_ready_for_step6",
                candidate_notes=None,
                candidate_role="reference_benchmark",
                is_generated_candidate=False,
                context_status="complete_reference",
                context_notes=[
                    f"Reference ADC '{original}' carried for downstream "
                    "evidence / patent / case-study reasoning. "
                    "NOT a generated candidate."
                ],
            )
            out.append(record)
        return out

    def _build_compound_candidates(
        self,
        *,
        payload_text: Optional[str],
        linker_text: Optional[str],
        smiles_refs: list[dict],
        payload_smiles_refs: list[dict],
        linker_smiles_refs: list[dict],
        compound_smiles_refs: list[dict],
        chembl_refs: list[dict],
        pubchem_refs: list[dict],
        zinc_refs: list[dict],
        drugbank_refs: list[dict],
        sq_artifact_id: str,
        entity_decompositions: list[dict] | None = None,
        explicit_payload_role: bool = False,
        explicit_linker_role: bool = False,
        explicit_ligand_role: bool = False,
    ) -> list[CandidateRecord]:
        results: list[CandidateRecord] = []
        entity_decompositions = entity_decompositions or []

        if payload_text:
            mats: list[Material] = [
                self._material_with_role(
                    "payload_name", payload_text, "text",
                    role="payload", role_status="explicit",
                )
            ]
            mats.extend(
                self._smiles_materials(
                    payload_smiles_refs or smiles_refs,
                    "payload_smiles",
                    role="payload",
                    role_status="explicit" if payload_smiles_refs else "unknown",
                )
            )
            if linker_smiles_refs and not linker_text:
                mats.extend(
                    self._smiles_materials(
                        linker_smiles_refs,
                        "linker_smiles",
                        role="linker",
                        role_status="explicit",
                    )
                )
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
                    candidate_role="user_provided_candidate",
                    context_status="partial",
                )
            )

        # linker_payload aliases (vc-MMAE, deruxtecan) — preserve the whole
        # alias + emit decomposition materials when an entity_decomposition
        # marked the composite. Each component carries explicit/inferred via
        # `role_status`.
        for decomp in entity_decompositions:
            if not isinstance(decomp, dict):
                continue
            original = decomp.get("original_text") or ""
            # Skip whole-drug ADC entries (already handled).
            comps = decomp.get("components") or []
            roles_present = {c.get("role") for c in comps if isinstance(c, dict)}
            if not roles_present or "antibody" in roles_present:
                continue
            # linker_payload-shaped composite
            if not (
                "linker" in roles_present
                or "linker_payload" in roles_present
                or "payload" in roles_present
            ):
                continue
            mats = [
                self._material_with_role(
                    "linker_payload_name", original, "text",
                    role="linker_payload", role_status="explicit",
                )
            ]
            for c in comps:
                if not isinstance(c, dict):
                    continue
                comp_role = c.get("role") or "other"
                if comp_role not in {"linker", "payload", "linker_payload"}:
                    continue
                mat_type = {
                    "linker": "linker_name",
                    "payload": "payload_name",
                    "linker_payload": "linker_payload_name",
                }[comp_role]
                mats.append(
                    self._material_with_role(
                        mat_type, c.get("canonical_name") or "", "text",
                        role=comp_role,
                        role_status="explicit" if c.get("inferred") is False else "inferred",
                    )
                )
            results.append(
                self._compound_candidate(
                    label=original,
                    materials=mats,
                    identifiers=[],
                    sq_artifact_id=sq_artifact_id,
                    candidate_role="user_provided_candidate",
                    context_status="partial",
                )
            )

        if linker_text:
            mats = [
                self._material_with_role(
                    "linker_name", linker_text, "text",
                    role="linker", role_status="explicit",
                )
            ]
            mats.extend(
                self._smiles_materials(
                    linker_smiles_refs or ([] if payload_smiles_refs else smiles_refs),
                    "linker_smiles",
                    role="linker",
                    role_status="explicit" if linker_smiles_refs else "unknown",
                )
            )
            results.append(
                self._compound_candidate(
                    label=linker_text,
                    materials=mats,
                    identifiers=[],
                    sq_artifact_id=sq_artifact_id,
                    candidate_role="user_provided_candidate",
                    context_status="partial",
                )
            )

        # Free-floating compound identifiers without payload/linker text.
        if not payload_text and not linker_text and (
            smiles_refs or chembl_refs or pubchem_refs or zinc_refs or drugbank_refs
        ):
            # Material-only by default. The user must explicitly say
            # "payload candidates" / "linker candidates" / "ligand
            # candidates" for the role to be promoted.
            if explicit_payload_role:
                role = "payload_candidate"
                candidate_role = "user_provided_candidate"
            elif explicit_linker_role:
                role = "linker_candidate"
                candidate_role = "user_provided_candidate"
            elif explicit_ligand_role:
                role = "ligand_candidate"
                candidate_role = "user_provided_candidate"
            else:
                role = "compound"
                candidate_role = "material_only"
            role_status = "explicit" if role != "compound" else "unknown"
            mats = list(
                self._smiles_materials(
                    compound_smiles_refs or smiles_refs,
                    "compound_smiles",
                    role=role,
                    role_status=role_status,
                )
            )
            # Emit one material per source identifier so the role is
            # visible even when no SMILES was supplied. Material type
            # encodes the source DB (ChEMBL / PubChem / ZINC / DrugBank).
            for id_type, refs in (
                ("chembl_id", chembl_refs),
                ("pubchem_cid", pubchem_refs),
                ("zinc_id", zinc_refs),
                ("drugbank_id", drugbank_refs),
            ):
                for ref in refs:
                    mats.append(
                        self._material_with_role(
                            f"compound_identifier_{id_type}",
                            ref.get("value") or "", "text",
                            role=role, role_status=role_status,
                        )
                    )
            label = (
                (chembl_refs and chembl_refs[0]["value"])
                or (pubchem_refs and pubchem_refs[0]["value"])
                or (zinc_refs and zinc_refs[0]["value"])
                or "compound_from_identifiers"
            )
            # Attach a single role tag to each generic compound material
            # so downstream agents can tell "user said payload" apart
            # from "just an ID".
            for m in mats:
                if not m.role:
                    m.role = role
                if m.role_status == "unknown":
                    m.role_status = role_status  # type: ignore[assignment]
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
                    candidate_role=candidate_role,
                    context_status="material_pool" if candidate_role == "material_only" else "partial",
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
        candidate_role: str = "unknown",
        context_status: str = "unknown",
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
            candidate_role=candidate_role,  # type: ignore[arg-type]
            is_generated_candidate=False,
            context_status=context_status,  # type: ignore[arg-type]
        )

    def _material_with_role(
        self,
        material_type: str,
        value: str,
        value_format: str,
        *,
        role: str | None = None,
        role_status: str = "unknown",
    ) -> Material:
        m = self._material(material_type, value, value_format)
        m.role = role
        m.role_status = role_status  # type: ignore[assignment]
        return m

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

    def _smiles_materials(
        self,
        smiles_refs: list[dict],
        material_type: str,
        *,
        role: str | None = None,
        role_status: str = "unknown",
    ) -> Iterable[Material]:
        for ref in smiles_refs:
            yield self._material_with_role(
                material_type,
                ref["value"],
                "smiles",
                role=role,
                role_status=role_status,
            )

    def _execute_enrichment_plans(
        self,
        *,
        run_id: str,
        record: CandidateRecord,
        scoped_tools: Iterable[str],
        label: str,
        raw_user_query: str = "",
        selection_audit: dict[str, dict] | None = None,
    ) -> list[ToolCallRecord]:
        eligible_plans = plan_enrichment_for_record(
            record,
            scoped_tools=scoped_tools,
            candidate_category=record.candidate_type,
            name_query_sanitizer=_clean_chembl_name_query,
            smiles_query_sanitizer=_looks_like_smiles_query,
        )
        _annotate_skipped_low_information_chembl_names(
            record=record,
            scoped_tools=scoped_tools,
        )

        # CDR3 → IEDB extension. Transient: ``raw_cdr3_by_tool`` lives in
        # this stack frame ONLY and is keyed by the synthetic plan we
        # append below. It carries the raw CDR3 string so the IEDB MCP
        # call can build a real filter, but the raw CDR3 NEVER reaches
        # the audit, the LLM payload, or the persisted artifact.
        raw_cdr3_by_tool: dict[str, dict[str, Any]] = {}
        cdr3_audit_by_tool: dict[str, dict[str, Any]] = {}
        cdr3_plans = self._build_iedb_cdr3_plans(
            record=record,
            scoped_tools=scoped_tools,
            raw_cdr3_by_tool=raw_cdr3_by_tool,
            cdr3_audit_by_tool=cdr3_audit_by_tool,
        )
        eligible_plans = eligible_plans + cdr3_plans

        decisions, audit = select_step5_enrichment_plans(
            record=record,
            eligible_plans=eligible_plans,
            llm=self.llm,
            raw_user_query=raw_user_query,
        )
        if selection_audit is not None and audit.candidate_id:
            selection_audit[audit.candidate_id] = audit.to_compact()

        eligible_count = len(eligible_plans)
        # Real selected vs synthetic dependency-gap are reported as
        # distinct counters so a reviewer can never confuse a synthetic
        # ZINC dependency_unavailable record with an LLM-selected
        # successful execution.
        real_selected_count = sum(
            1 for d in decisions
            if d.selected and not d.plan.known_live_unavailable
        )
        known_unavailable_count = sum(
            1 for d in decisions
            if d.selected and d.plan.known_live_unavailable
        )
        skipped_count = sum(1 for d in decisions if not d.selected)
        fallback_reason = audit.fallback_reason

        out: list[ToolCallRecord] = []
        for decision in decisions:
            if not decision.selected:
                continue
            plan = decision.plan
            provenance = selection_provenance_for_tool_input_summary(
                decision,
                eligible_count=eligible_count,
                real_selected_count=real_selected_count,
                skipped_count=skipped_count,
                known_unavailable_count=known_unavailable_count,
                fallback_reason=fallback_reason,
            )
            if plan.known_live_unavailable:
                tc = _known_unavailable_tool_call(
                    plan=plan, label=label, provenance=provenance
                )
            elif plan.tool_name == _IEDB_BCR_TOOL_NAME and (
                _IEDB_TRANSIENT_TOKEN in plan.query
            ):
                tc = self._enrich_with_iedb_cdr3(
                    run_id=run_id,
                    record=record,
                    plan=plan,
                    label=label,
                    provenance=provenance,
                    raw_cdr3_by_tool=raw_cdr3_by_tool,
                    cdr3_audit_by_tool=cdr3_audit_by_tool,
                )
            else:
                summary = {
                    "query_kind": plan.query_kind,
                    "query_role": plan.query_role,
                    "material_type": plan.material_type,
                    "capability_type": plan.capability_type,
                    "output_extractor_type": plan.output_extractor_type,
                }
                summary.update(plan.extra_summary)
                summary.update(provenance)
                tc = self._enrich_with_tool(
                    run_id=run_id,
                    tool_name=plan.tool_name,
                    arg_value=plan.query,
                    label=label,
                    tool_arg_name=plan.schema_arg_name,
                    extra_summary=summary,
                )
            out.append(tc)
            if plan.output_extractor_type == "compound":
                _record_chembl_plan_outcome(
                    storage=self.storage,
                    record=record,
                    plan=plan,
                    tc=tc,
                )
            elif plan.output_extractor_type == "sabdab_structure" and record.candidate_type == "antibody":
                _annotate_antibody_sabdab_outcome(
                    storage=self.storage, record=record, tc=tc
                )
            elif plan.known_live_unavailable:
                gap = (
                    f"{plan.tool_name}({plan.query_kind}={plan.query[:80]}): "
                    f"dependency_unavailable"
                )
                if gap not in record.data_gaps:
                    record.data_gaps.append(gap)
        return out

    def _enrich_with_tool(
        self,
        *,
        run_id: str,
        tool_name: str,
        arg_value: str,
        label: str,
        tool_arg_name: str = "query",
        extra_summary: dict[str, Any] | None = None,
    ) -> ToolCallRecord:
        tc_id = new_tool_call_id()
        started = now_iso()
        result = self.mcp_client.call_tool(
            agent_name=_AGENT_NAME,
            step_id=_STEP_ID,
            tool_name=tool_name,
            **{tool_arg_name: arg_value},
        )
        finished = now_iso()
        summary = {"query": arg_value, "label": label}
        if tool_arg_name != "query":
            summary[tool_arg_name] = arg_value
        if extra_summary:
            summary.update(extra_summary)

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
                    "input": summary,
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
            tool_input_summary=summary,
            tool_output_artifact_id=output_artifact_id,
            tool_output_ref=output_ref,
            error_message=result.get("error_message"),
        )

    # ── CDR3 → IEDB extension ───────────────────────────────────────────
    def _read_sequence_material_text(self, material_value: str) -> str:
        """Read the antibody sequence string from a material `value`.

        Sequence materials usually carry a storage path. We deliberately
        treat the path as opaque: only the raw FASTA text is read from
        storage, only the sequence characters (no headers) are returned,
        and the full string is held in memory only long enough to build
        an MCP filter. Never logged. Never persisted on the candidate.
        """
        text: str = ""
        if not isinstance(material_value, str) or not material_value.strip():
            return ""
        # Path-shaped values get resolved through storage; literal sequences
        # (already amino-acid characters) are used as-is.
        if "/" in material_value or material_value.lower().endswith(
            tuple(_FASTA_EXTS)
        ):
            try:
                raw = self.storage.read_bytes(material_value) or b""
                text = raw.decode("utf-8", errors="ignore")
            except Exception:  # noqa: BLE001
                text = ""
        else:
            text = material_value
        # Strip FASTA headers / blank lines, keep AA chars only. We never
        # log this text and we never return it from public methods.
        lines = [
            line.strip() for line in text.splitlines()
            if line.strip() and not line.startswith(">")
        ]
        return "".join(lines).upper()

    def _candidate_sequence_materials(
        self, record: CandidateRecord
    ) -> list[tuple[str, str, str]]:
        """Return ``(material_id, chain_role, material_value)`` tuples
        for every antibody sequence material attached to ``record``."""
        out: list[tuple[str, str, str]] = []
        for material in record.materials or []:
            mt = material.material_type
            role = (material.role or "").lower()
            if mt in _ANTIBODY_HEAVY_SEQ_MATERIAL_TYPES:
                chain_role = "antibody_heavy"
            elif mt in _ANTIBODY_LIGHT_SEQ_MATERIAL_TYPES:
                chain_role = "antibody_light"
            elif mt in _ANTIBODY_GENERIC_SEQ_MATERIAL_TYPES:
                chain_role = "unknown"
            elif role in _ANTIBODY_SEQ_REFERENCE_ROLES:
                # Generic reference; let the numbering backend decide.
                chain_role = "unknown"
            else:
                continue
            value = material.value or ""
            if value:
                out.append((material.material_id, chain_role, value))
        return out

    def _build_iedb_cdr3_plans(
        self,
        *,
        record: CandidateRecord,
        scoped_tools: Iterable[str],
        raw_cdr3_by_tool: dict[str, dict[str, Any]],
        cdr3_audit_by_tool: dict[str, dict[str, Any]],
    ) -> list[EnrichmentPlan]:
        if _IEDB_BCR_TOOL_NAME not in set(scoped_tools):
            return []
        if record.candidate_type != "antibody":
            return []
        seq_materials = self._candidate_sequence_materials(record)
        if not seq_materials:
            return []

        plans: list[EnrichmentPlan] = []
        for material_id, chain_role, material_value in seq_materials:
            sequence = self._read_sequence_material_text(material_value)
            if not sequence:
                gap = (
                    f"iedb_cdr3_extraction_failed:source_material={material_id}"
                    ":empty_sequence"
                )
                if gap not in record.data_gaps:
                    record.data_gaps.append(gap)
                continue

            cdr3: Cdr3Result = extract_cdr3(
                sequence, expected_chain_role=chain_role  # type: ignore[arg-type]
            )
            if cdr3.status != _CDR3_STATUS_SUCCESS:
                self._record_cdr3_failure_gap(
                    record=record,
                    material_id=material_id,
                    cdr3=cdr3,
                )
                continue

            # Decide which IEDB filter key applies to this chain. Heavy
            # chains map to ``chain1_cdr3_seq``; light chains map to
            # ``chain2_cdr3_seq``. Unknown extracted chain types are not
            # queried — we do not invent a filter when the numbering
            # backend is unsure.
            if cdr3.chain_type == CHAIN_TYPE_HEAVY:
                filter_key = "chain1_cdr3_seq"
                derived_mat_type = "antibody_heavy_cdr3_sequence"
            elif cdr3.chain_type == CHAIN_TYPE_LIGHT:
                filter_key = "chain2_cdr3_seq"
                derived_mat_type = "antibody_light_cdr3_sequence"
            else:
                gap = (
                    f"iedb_cdr3_extraction_chain_type_unknown:"
                    f"source_material={material_id}"
                )
                if gap not in record.data_gaps:
                    record.data_gaps.append(gap)
                continue

            # Add a derived material whose `value` is a redacted marker —
            # NOT the raw CDR3 string. The raw string stays in
            # ``raw_cdr3_by_tool`` for the duration of this call only.
            redacted_value = (
                f"[redacted:cdr3 length={cdr3.cdr3_length} "
                f"sha256={cdr3.cdr3_sha256_prefix} "
                f"scheme={cdr3.numbering_scheme}]"
            )
            mat = Material(
                material_id=new_artifact_id("material"),
                material_type=derived_mat_type,
                value=redacted_value,
                value_format="redacted_marker",
                extraction_status="extracted",
                validation_status="valid",
                role="antibody",
                role_status="inferred",
            )
            _append_material_once(record, mat)
            note = (
                f"CDR3 extracted by antibody numbering backend "
                f"'{cdr3.backend}' (scheme={cdr3.numbering_scheme}); "
                f"length={cdr3.cdr3_length}; sha256_prefix="
                f"{cdr3.cdr3_sha256_prefix}; source_material={material_id}; "
                f"raw CDR3 used only for MCP arguments — not persisted."
            )
            if note not in record.context_notes:
                record.context_notes.append(note)

            # Build a synthetic plan. The ``query`` is a sentinel token,
            # NOT the raw CDR3, so eligibility audit / catalog / LLM
            # payload do not see the sequence.
            unique_token = f"{_IEDB_TRANSIENT_TOKEN}:{material_id}:{filter_key}"
            extra_summary = {
                "fallback_group": "iedb_bcr_cdr3",
                "provenance_policy": "cdr3_filtered_iedb_lookup",
                "confidence_policy": "context_only",
                "cdr3_chain_type": cdr3.chain_type,
                "cdr3_length": cdr3.cdr3_length,
                "cdr3_sha256_prefix": cdr3.cdr3_sha256_prefix,
                "cdr3_numbering_scheme": cdr3.numbering_scheme,
                "cdr3_backend": cdr3.backend,
                "cdr3_source_material_id": material_id,
                "iedb_filter_key": filter_key,
            }
            plan = EnrichmentPlan(
                tool_name=_IEDB_BCR_TOOL_NAME,
                query=unique_token,
                query_kind="cdr3_filter",
                query_role="antibody",
                material_type=derived_mat_type,
                schema_arg_name="filters",
                capability_type="bcell_receptor_cdr3_lookup",
                output_extractor_type="iedb_bcr_cdr3",
                provenance_policy="cdr3_filtered_iedb_lookup",
                confidence_policy="context_only",
                known_live_unavailable=False,
                known_unavailable_reason="",
                extra_summary=extra_summary,
            )
            plans.append(plan)
            raw_cdr3_by_tool[unique_token] = {
                "cdr3_sequence": cdr3.cdr3_sequence,
                "filter_key": filter_key,
            }
            cdr3_audit_by_tool[unique_token] = cdr3.to_compact_audit()
        return plans

    def _record_cdr3_failure_gap(
        self,
        *,
        record: CandidateRecord,
        material_id: str,
        cdr3: Cdr3Result,
    ) -> None:
        if cdr3.status == _CDR3_STATUS_DEPENDENCY_UNAVAILABLE:
            gap = (
                "iedb_cdr3_extraction_dependency_unavailable:"
                f"source_material={material_id}"
            )
            note = (
                "Step 5 did not query IEDB BCR by CDR3 because antibody "
                "numbering dependency (abnumber / anarci) is not "
                "installed. Full VH/VL sequence preserved for Step 6 "
                "PROSITE / Step 7 structure preparation."
            )
        elif cdr3.status == _CDR3_STATUS_NO_VARIABLE_DOMAIN:
            gap = (
                "iedb_cdr3_extraction_no_variable_domain:"
                f"source_material={material_id}"
            )
            note = (
                "Numbering backend reported no antibody variable domain; "
                "Step 5 did not query IEDB with the full sequence."
            )
        else:
            gap = (
                "iedb_cdr3_extraction_failed:"
                f"source_material={material_id}"
            )
            note = (
                "Numbering backend failed to extract CDR3; Step 5 did "
                "not query IEDB."
            )
        if gap not in record.data_gaps:
            record.data_gaps.append(gap)
        if note not in record.context_notes:
            record.context_notes.append(note)

    def _enrich_with_iedb_cdr3(
        self,
        *,
        run_id: str,
        record: CandidateRecord,
        plan: EnrichmentPlan,
        label: str,
        provenance: dict[str, Any],
        raw_cdr3_by_tool: dict[str, dict[str, Any]],
        cdr3_audit_by_tool: dict[str, dict[str, Any]],
    ) -> ToolCallRecord:
        """Execute the synthetic IEDB plan with raw CDR3 in MCP args
        ONLY. Audit and persisted summary keep only redacted metadata."""
        transient = raw_cdr3_by_tool.get(plan.query) or {}
        cdr3_sequence: str = transient.get("cdr3_sequence") or ""
        filter_key: str = transient.get("filter_key") or "chain1_cdr3_seq"
        cdr3_audit = cdr3_audit_by_tool.get(plan.query) or {}

        # Compact audit summary — never the raw CDR3 string.
        redacted_summary: dict[str, Any] = {
            "query_kind": plan.query_kind,
            "query_role": plan.query_role,
            "material_type": plan.material_type,
            "capability_type": plan.capability_type,
            "output_extractor_type": plan.output_extractor_type,
            "iedb_filter_key": filter_key,
            "cdr3_chain_type": plan.extra_summary.get("cdr3_chain_type"),
            "cdr3_length": plan.extra_summary.get("cdr3_length"),
            "cdr3_sha256_prefix": plan.extra_summary.get("cdr3_sha256_prefix"),
            "cdr3_numbering_scheme": plan.extra_summary.get("cdr3_numbering_scheme"),
            "cdr3_backend": plan.extra_summary.get("cdr3_backend"),
            "cdr3_source_material_id": plan.extra_summary.get(
                "cdr3_source_material_id"
            ),
            "select_columns_count": len(_IEDB_SELECT_COLUMNS),
            "cdr3_audit": cdr3_audit,
        }
        redacted_summary.update(plan.extra_summary)
        redacted_summary.update(provenance)
        # Defensive sweep: ensure no raw CDR3 leaks into the summary.
        if cdr3_sequence:
            for key, value in list(redacted_summary.items()):
                if isinstance(value, str) and cdr3_sequence in value:
                    redacted_summary[key] = "[redacted]"

        tc_id = new_tool_call_id()
        started = now_iso()
        # ``filters`` is a real MCP argument; ``select`` is a compact
        # column projection. Both contain operational values, NOT the
        # full antibody sequence and NOT placed on the candidate.
        mcp_args = {
            "filters": {filter_key: f"eq.{cdr3_sequence}"},
            "select": list(_IEDB_SELECT_COLUMNS),
        }
        result = self.mcp_client.call_tool(
            agent_name=_AGENT_NAME,
            step_id=_STEP_ID,
            tool_name=_IEDB_BCR_TOOL_NAME,
            **mcp_args,
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
                    "tool_name": _IEDB_BCR_TOOL_NAME,
                    # Persisted ``input`` is the REDACTED summary — never
                    # the raw CDR3 string and never the operational MCP
                    # arg dict. We accept the loss of a fully reproducible
                    # input echo in exchange for the privacy guarantee.
                    "input": redacted_summary,
                    "output": result["payload"],
                },
            )
            output_ref = output_key

        # Surface a compact context_note + non-leaking summary for the
        # candidate. Raw CDR3 still never reaches the record.
        if result.get("run_status") in (None, "success") and output_ref:
            record.candidate_notes = _notes_for(
                ToolCallRecord(
                    tool_call_id=tc_id,
                    tool_name=_IEDB_BCR_TOOL_NAME,
                    agent_name=_AGENT_NAME,
                    step_id=_STEP_ID,
                    run_status=result.get("run_status", "pending"),
                    started_at=started,
                    finished_at=finished,
                    tool_input_summary=redacted_summary,
                    tool_output_ref=output_ref,
                ),
                "iedb cdr3 context enrichment",
            ) or record.candidate_notes

        return ToolCallRecord(
            tool_call_id=tc_id,
            tool_name=_IEDB_BCR_TOOL_NAME,
            agent_name=_AGENT_NAME,
            step_id=_STEP_ID,
            run_status=result.get("run_status", "pending"),
            started_at=started,
            finished_at=finished,
            tool_input_summary=redacted_summary,
            tool_output_artifact_id=output_artifact_id,
            tool_output_ref=output_ref,
            error_message=result.get("error_message"),
        )

    def run_step(self, *, run_id: str, step_id: str, payload: dict[str, Any]) -> dict:  # noqa: ARG002
        return self.run(run_id).model_dump()


# ── module helpers ─────────────────────────────────────────────────────────

def _annotate_data_gaps(
    candidate_records: list[CandidateRecord],
    *,
    target_text: Optional[str],
    antibody_text: Optional[str],
    payload_text: Optional[str],
    linker_text: Optional[str],
) -> None:
    """Populate `data_gaps` / `missing_material_roles` / `context_notes`.

    The professor's TROP2-ADC-with-MMAE rule: when target + payload are
    present but antibody / linker are missing, Step 5 must mark the gap
    explicitly — not invent an antibody or linker.
    """
    has_target = bool(target_text)
    has_antibody = bool(antibody_text) or _has_role(
        candidate_records, "antibody"
    )
    has_payload = bool(payload_text) or _has_role(
        candidate_records, "payload"
    )
    has_linker = bool(linker_text) or _has_role(
        candidate_records, "linker"
    ) or _has_role(candidate_records, "linker_payload")

    # Each candidate gets a tailored gap list. Reference benchmarks
    # (T-DM1 / T-DXd) are complete by construction — skip.
    for rec in candidate_records:
        if rec.candidate_role == "reference_benchmark":
            continue
        gaps: list[str] = []
        missing_roles: list[str] = []
        if rec.candidate_type == "target_antigen":
            if not has_antibody:
                missing_roles.append("antibody")
                gaps.append("antibody not provided")
            if not has_payload:
                missing_roles.append("payload")
                gaps.append("payload not provided")
            if not has_linker:
                missing_roles.append("linker")
                gaps.append("linker not provided")
            if missing_roles:
                gaps.append("no complete ADC design specified by user")
        elif rec.candidate_type == "antibody":
            if not has_payload:
                missing_roles.append("payload")
                gaps.append("payload not provided for this antibody")
            if not has_linker:
                missing_roles.append("linker")
                gaps.append("linker not provided for this antibody")
        elif rec.candidate_type == "compound_component":
            # Compound records with no antibody / target context are
            # legitimate material_pool entries, but downstream still
            # needs to know there is no full ADC behind them.
            if not has_antibody:
                missing_roles.append("antibody")
                gaps.append("no antibody backbone defined for this compound")
            if not has_target:
                missing_roles.append("target")
                gaps.append("no target / antigen defined for this compound")

        if missing_roles:
            rec.data_gaps.extend(gaps)
            rec.missing_material_roles.extend(missing_roles)
            rec.context_notes.append(
                "Partial context: Step 5 preserves provided materials; "
                "downstream agents must NOT treat this as a generated "
                "ADC candidate."
            )
            # Status reflects partial-context honesty.
            if rec.context_status == "unknown":
                rec.context_status = "partial"


def _has_role(records: list[CandidateRecord], role: str) -> bool:
    for rec in records:
        for m in rec.materials:
            if m.role == role:
                return True
    return False


def _clean_chembl_name_query(value: str, role: str | None = None) -> str | None:
    query = " ".join((value or "").strip().split())
    if not query:
        return None
    role_suffixes = {
        "linker": (" linker",),
        "payload": (" payload",),
        "linker_payload": (" linker-payload", " linker payload"),
    }.get(role or "", ())
    lowered_query = query.lower()
    for suffix in role_suffixes:
        if lowered_query.endswith(suffix):
            query = query[: -len(suffix)].strip()
            break
    lowered = f" {query.lower()} "
    if any(marker in lowered for marker in _MIXED_CONTEXT_MARKERS):
        return None
    if "(" in query or ")" in query:
        return None
    if len(query) > 80:
        return None
    return query


def _annotate_skipped_low_information_chembl_names(
    *,
    record: CandidateRecord,
    scoped_tools: Iterable[str],
) -> None:
    for _material_type, query, _role in skipped_low_information_chembl_name_queries(
        record,
        scoped_tools=scoped_tools,
        candidate_category=record.candidate_type,
        name_query_sanitizer=_clean_chembl_name_query,
    ):
        gap = f"ChEMBL_search_molecules(name={query}): skipped_low_information_alias"
        if gap not in record.data_gaps:
            record.data_gaps.append(gap)
        note = (
            f"Skipped ChEMBL_search_molecules name query for low-information "
            f"alias '{query}'."
        )
        if note not in record.context_notes:
            record.context_notes.append(note)


def _looks_like_smiles_query(value: str) -> bool:
    query = (value or "").strip()
    if not query or any(ch.isspace() for ch in query):
        return False
    lowered = query.lower()
    if "smiles" in lowered or ";" in query:
        return False
    return bool(re.search(r"[A-Za-z]", query))


def _known_unavailable_tool_call(
    *,
    plan: EnrichmentPlan,
    label: str,
    provenance: dict[str, Any] | None = None,
) -> ToolCallRecord:
    now = now_iso()
    summary: dict[str, Any] = {
        "query": plan.query,
        "label": label,
        "query_kind": plan.query_kind,
        "query_role": plan.query_role,
        "material_type": plan.material_type,
        "capability_type": plan.capability_type,
        "output_extractor_type": plan.output_extractor_type,
        "provenance_policy": plan.provenance_policy,
        "confidence_policy": plan.confidence_policy,
        **plan.extra_summary,
    }
    if provenance:
        summary.update(provenance)
    return ToolCallRecord(
        tool_call_id=new_tool_call_id(),
        tool_name=plan.tool_name,
        agent_name=_AGENT_NAME,
        step_id=_STEP_ID,
        run_status="dependency_unavailable",
        started_at=now,
        finished_at=now,
        tool_input_summary=summary,
        error_message=plan.known_unavailable_reason or "known_unavailable",
    )


def _extend_target_identifiers_from_normalized_entities(
    identifiers: list[Identifier],
    normalized_entities: list[dict],
    *,
    sq_artifact_id: str,
    context_notes: list[str],
    data_gaps: list[str],
) -> None:
    for ne in normalized_entities or []:
        if not isinstance(ne, dict):
            continue
        if ne.get("entity_type") != "target_or_antigen":
            continue
        source = str(ne.get("canonical_id_source") or "").strip().lower()
        if source != "uniprot":
            continue
        accession = str(ne.get("canonical_id") or "").strip()
        original = str(ne.get("original_text") or ne.get("canonical_name") or "target")
        if not accession:
            continue
        if not _is_uniprot_accession(accession):
            gap = (
                "target_uniprot_id_not_promoted:invalid_normalized_entity_accession"
            )
            if gap not in data_gaps:
                data_gaps.append(gap)
            note = (
                "Step 2 normalized target entity carried canonical_id_source=UniProt "
                f"but canonical_id was not accession-like; original_text={original[:80]}"
            )
            if note not in context_notes:
                context_notes.append(note)
            continue
        ident = Identifier(
            id_type="uniprot_id",
            id_value=accession,
            source_ids=[sq_artifact_id] if sq_artifact_id else [],
            confidence=0.8,
        )
        _append_identifier_once_value(identifiers, ident)


def _is_uniprot_accession(value: str) -> bool:
    return bool(_UNIPROT_ACCESSION_RE.match((value or "").strip().upper()))


def _append_identifier_once_value(identifiers: list[Identifier], ident: Identifier) -> None:
    if any(
        existing.id_type == ident.id_type
        and existing.id_value.lower() == ident.id_value.lower()
        for existing in identifiers
    ):
        return
    identifiers.append(ident)


def _build_downstream_query_hints(
    *,
    candidate_records: list[CandidateRecord],
    normalized_entities: list[dict],
    target_text: Optional[str],
    antibody_text: Optional[str],
    payload_text: Optional[str],
    linker_text: Optional[str],
    entity_decompositions: list[dict],
) -> list[dict]:
    """Order: complete ADC → linker_payload → payload → linker → ligand →
    compound → target → conjugation chemistry → use / indication →
    antibody (only when explicitly provided).
    """
    hints: list[dict] = []
    seen: set[tuple[str, str]] = set()

    def _add(entity: str, role: str, explicit_or_inferred: str, source: str) -> None:
        entity = (entity or "").strip()
        if not entity:
            return
        key = (entity.lower(), role)
        if key in seen:
            return
        seen.add(key)
        hints.append(
            {
                "entity": entity,
                "role": role,
                "explicit_or_inferred": explicit_or_inferred,
                "source": source,
            }
        )

    # 1) complete ADCs
    for rec in candidate_records:
        if rec.candidate_role == "reference_benchmark":
            _add(rec.candidate_label, "complete_adc", "explicit",
                 "candidate_record.reference_benchmark")
    # 2) linker_payload
    for decomp in entity_decompositions or []:
        if not isinstance(decomp, dict):
            continue
        original = decomp.get("original_text") or ""
        comp_roles = {
            c.get("role") for c in (decomp.get("components") or [])
            if isinstance(c, dict)
        }
        if comp_roles and "antibody" not in comp_roles and (
            "linker_payload" in comp_roles or
            ("linker" in comp_roles and "payload" in comp_roles)
        ):
            _add(original, "linker_payload", "explicit",
                 "entity_decomposition.linker_payload")
    # 3) payload (explicit user mention OR normalized payload entity)
    if payload_text:
        _add(payload_text, "payload", "explicit",
             "mentioned_entities.payload_text")
    for ne in normalized_entities or []:
        if isinstance(ne, dict) and ne.get("entity_type") == "payload":
            _add(
                ne.get("original_text") or "",
                "payload",
                ne.get("explicit_or_inferred") or "inferred",
                "normalized_entities.payload",
            )
    # 4) linker
    if linker_text:
        _add(linker_text, "linker", "explicit",
             "mentioned_entities.linker_text")
    for ne in normalized_entities or []:
        if isinstance(ne, dict) and ne.get("entity_type") == "linker":
            _add(
                ne.get("original_text") or "",
                "linker",
                ne.get("explicit_or_inferred") or "inferred",
                "normalized_entities.linker",
            )
    # 5) compound (free-floating IDs that landed as material_only / payload_candidate)
    for rec in candidate_records:
        if rec.candidate_type != "compound_component":
            continue
        if rec.candidate_role == "user_provided_candidate":
            # Already covered by payload/linker_text above when applicable.
            continue
        for ident in rec.identifiers:
            _add(ident.id_value, "compound", "explicit",
                 f"identifier.{ident.id_type}")
    # 6) target
    if target_text:
        _add(target_text, "target", "explicit",
             "mentioned_entities.target_or_antigen_text")
    for ne in normalized_entities or []:
        if isinstance(ne, dict) and ne.get("entity_type") == "target_or_antigen":
            _add(
                ne.get("original_text") or "",
                "target",
                ne.get("explicit_or_inferred") or "inferred",
                "normalized_entities.target_or_antigen",
            )
    # 7) antibody ONLY when the user explicitly supplied one. This matches
    #    the professor rule that evidence / patent search should not be
    #    antibody-centered by default.
    if antibody_text:
        _add(antibody_text, "antibody", "explicit",
             "mentioned_entities.antibody_candidate_text")
    return hints


def _notes_for(tc: ToolCallRecord, label: str) -> Optional[str]:
    if tc.run_status == "success":
        return None
    return f"{label}: {tc.run_status}"


def _format_for_file(f: dict) -> str:
    ext = PurePosixPath(f.get("original_filename", "")).suffix.lower()
    if ext in {".cif", ".mmcif"}:
        return "cif"
    return "pdb"


def _infer_antibody_sequence_material_type(f: dict) -> str:
    """Infer heavy/light antibody sequence role from explicit file metadata.

    This is deliberately conservative: unknown FASTA files stay generic
    ``antibody_sequence_reference`` so Step 5 does not silently treat every
    antibody sequence upload as a heavy chain.
    """
    parts: list[str] = []
    for key in (
        "original_filename", "file_id", "role", "chain_role",
        "material_role", "description",
    ):
        value = f.get(key)
        if isinstance(value, str) and value.strip():
            parts.append(value.lower())
    haystack = " ".join(parts)
    tokens = set(re.split(r"[^a-z0-9]+", haystack))
    collapsed = re.sub(r"[^a-z0-9]+", "", haystack)
    if tokens.intersection(_HEAVY_CHAIN_HINTS) or any(
        hint in collapsed for hint in ("heavychain", "ighv")
    ):
        return "antibody_heavy_chain_sequence"
    if tokens.intersection(_LIGHT_CHAIN_HINTS) or any(
        hint in collapsed for hint in ("lightchain", "igkv", "iglv")
    ):
        return "antibody_light_chain_sequence"
    return "antibody_sequence_reference"


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


def _extract_labeled_smiles_refs(raw_user_query: str, ctx: dict) -> dict[str, list[dict]]:
    """Extract only explicitly labeled SMILES from user text/context.

    This is intentionally label-gated (`payload SMILES`, `linker SMILES`,
    `compound SMILES`). It does not attempt general SMILES recognition, so
    names such as vc-MMAE cannot be promoted to typed Step 6 inputs.
    """
    chunks = [raw_user_query or ""]
    for value in (ctx or {}).values():
        if isinstance(value, str):
            chunks.append(value)
    text = "\n".join(chunks)
    out: dict[str, list[dict]] = {"payload": [], "linker": [], "compound": []}
    seen: set[tuple[str, str]] = set()
    for match in _LABELED_SMILES_RE.finditer(text):
        role = match.group("role").lower()
        value = (match.group("value") or "").strip().strip(";,.)]")
        if not value:
            continue
        key = (role, value)
        if key in seen:
            continue
        seen.add(key)
        out[role].append(
            {
                "id_type": "smiles",
                "value": value,
                "source": "raw_request_labeled_smiles",
            }
        )
    return out


def _apply_compound_tool_enrichment(
    record: CandidateRecord,
    payload: Any,
    *,
    source_artifact_id: str | None,
    tool_name: str | None = None,
    query_kind: str | None = None,
    query_value: str | None = None,
) -> int:
    """Promote compact ChEMBL-style fields into Step 5 typed context.

    Raw Step 5 tool payloads stay in `tool_outputs/step_05/*.json`; this
    function copies only the fields Step 6 can legitimately consume:
    stable compound identifiers and canonical SMILES.

    ``tool_name`` / ``query_kind`` let the caller tag identifiers from a
    substructure search differently from name-confirmed hits: substructure
    matches are an UPPER BOUND on identity, not the user's exact compound,
    so we lower confidence and record a compact ``context_notes`` entry
    on the candidate. Returns the number of chembl_id identifiers that
    were actually promoted by this call (useful for zero-hit gap notes).
    """
    is_substructure = tool_name == "ChEMBL_search_substructure"
    chembl_confidence = 0.5 if is_substructure else 0.8
    source_ids = [source_artifact_id] if source_artifact_id else []
    hits = list(_iter_compound_hits(payload))[:_MAX_COMPOUND_ENRICHMENT_HITS]
    promoted_chembl_count = 0
    for hit in hits:
        if not isinstance(hit, dict):
            continue

        chembl_id = _first_text(
            hit,
            ("molecule_chembl_id",),
            ("chembl_id",),
            ("id",),
            ("molecule", "molecule_chembl_id"),
        )
        if chembl_id and chembl_id.upper().startswith("CHEMBL"):
            if _append_identifier_once(
                record,
                Identifier(
                    id_type="chembl_id",
                    id_value=chembl_id,
                    source_ids=source_ids,
                    confidence=chembl_confidence,
                ),
            ):
                promoted_chembl_count += 1

        smiles = _first_text(
            hit,
            ("molecule_structures", "canonical_smiles"),
            ("structure", "canonical_smiles"),
            ("canonical_smiles",),
            ("smiles",),
        )
        if smiles:
            material_type, role = _compound_enrichment_smiles_role(record)
            _append_material_once(
                record,
                Material(
                    material_id=new_artifact_id("material"),
                    material_type=material_type,
                    value=smiles,
                    value_format="smiles",
                    extraction_status="extracted",
                    validation_status="unknown",
                    role=role,
                    role_status="inferred",
                ),
            )

        name = _first_text(hit, ("pref_name",), ("molecule_name",), ("name",), ("label",))
        if name and not any(
            m.value.lower() == name.lower()
            for m in record.materials
            if m.material_type in {"compound_name", "payload_name", "linker_name", "linker_payload_name"}
        ):
            material_type, role = _compound_enrichment_name_role(record)
            _append_material_once(
                record,
                Material(
                    material_id=new_artifact_id("material"),
                    material_type=material_type,
                    value=name,
                    value_format="text",
                    extraction_status="extracted",
                    validation_status="unknown",
                    role=role,
                    role_status="inferred",
                ),
            )

    if is_substructure and promoted_chembl_count > 0:
        # Substructure matches are an UPPER BOUND on identity, never
        # confirmed exact identity. Record a compact context note and
        # data_gap so Step 6 / downstream readers do not mis-read the
        # promoted chembl_id as "user's exact compound = CHEMBLxxx".
        note = (
            f"ChEMBL substructure-derived chembl_id count={promoted_chembl_count}; "
            f"not confirmed exact identity"
            + (f"; query={query_value}" if query_value else "")
        )
        if note not in record.context_notes:
            record.context_notes.append(note)
        gap = "chembl_id_origin:substructure_derived_not_exact_identity"
        if gap not in record.data_gaps:
            record.data_gaps.append(gap)
    return promoted_chembl_count


def _iter_compound_hits(payload: Any, _depth: int = 0) -> Iterable[dict]:
    """Yield compact compound hit dicts from a tool envelope.

    Supports both the older mocked/test shapes (top-level ``hits`` /
    ``results`` / ``molecules`` / …) and the ToolUniverse live envelope
    where the hit list is nested under ``payload`` / ``data`` (e.g.
    ``output.payload.data.molecules`` for live ChEMBL substructure search).

    The unwrap is bounded: at most ``_MAX_COMPOUND_HITS_UNWRAP_DEPTH``
    layers of ``payload`` / ``data`` / ``output`` wrapper before we give
    up. We never recurse into arbitrary keys, never traverse lists of
    arbitrary structure beyond the hit list itself.
    """
    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict):
                yield item
        return
    if not isinstance(payload, dict):
        return
    for key in _COMPOUND_HIT_LIST_KEYS:
        value = payload.get(key)
        if isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    yield item
            return
    # No hit list at this level — controlled unwrap of well-known
    # envelope wrappers (live ToolUniverse / Step 5 tool_output record).
    if _depth < _MAX_COMPOUND_HITS_UNWRAP_DEPTH:
        for wrap_key in _COMPOUND_ENVELOPE_WRAPPER_KEYS:
            sub = payload.get(wrap_key)
            if isinstance(sub, (dict, list)):
                yielded_any = False
                for hit in _iter_compound_hits(sub, _depth + 1):
                    yielded_any = True
                    yield hit
                if yielded_any:
                    return
    # Last resort: treat the payload itself as a single hit-shaped dict.
    yield payload


def _first_text(obj: dict, *paths: tuple[str, ...]) -> str | None:
    for path in paths:
        cur: Any = obj
        for part in path:
            if not isinstance(cur, dict):
                cur = None
                break
            cur = cur.get(part)
        if isinstance(cur, str) and cur.strip():
            return cur.strip()
    return None


def _record_chembl_plan_outcome(
    *,
    storage: Storage,
    record: CandidateRecord,
    plan: Any,
    tc: ToolCallRecord,
) -> None:
    """Promote chembl_id / SMILES from a ChEMBL tool output AND record
    exactly one compact outcome data_gap so reviewers can distinguish:

    - exact name-confirmed identity vs substructure-derived upper bound,
    - zero-match upstream response vs ChEMBL upstream error,
    - tool live unavailable vs missing input.

    Outcomes are MUTUALLY EXCLUSIVE: at most one gap per (tool, query) is
    appended per call. ``upstream_error`` / ``dependency_unavailable`` /
    ``failed`` ALWAYS dominate ``zero_matches_returned``; the latter only
    fires when the wrapper truly returned ok with an empty result list.
    """
    short_query = (plan.query or "")[:80]
    prefix = f"{plan.tool_name}({plan.query_kind}={short_query})"

    def _push(gap: str) -> None:
        if gap not in record.data_gaps:
            record.data_gaps.append(gap)

    # Wrapper-level failure first — short-circuit so we never also report
    # zero_matches_returned for the same call.
    if tc.run_status == "dependency_unavailable":
        _push(f"{prefix}: dependency_unavailable")
        return
    if tc.run_status in {"failed", "partial"}:
        _push(f"{prefix}: failed: {(tc.error_message or 'unknown_error')[:120]}")
        return
    if tc.run_status != "success" or not tc.tool_output_ref:
        # Nothing to inspect — record a minimal "no output" gap and stop.
        if tc.run_status not in {"skipped", "not_run"}:
            _push(f"{prefix}: no_output_recorded:{tc.run_status}")
        return

    # We have a success run_status and an output ref. Inspect the envelope.
    try:
        raw = storage.read_json(tc.tool_output_ref) or {}
    except Exception:  # noqa: BLE001
        _push(f"{prefix}: failed: cannot_read_tool_output_ref")
        return
    output = raw.get("output") or {}

    # Live ToolUniverse upstream_error is reported under `output.status`
    # even though the MCP client run_status was "success". Treat that as
    # upstream_error and DO NOT also report zero_matches_returned.
    if isinstance(output, dict) and output.get("status") == "upstream_error":
        err = output.get("error_message") or "upstream_error"
        _push(f"{prefix}: upstream_error: {str(err)[:120]}")
        return

    # Genuine success path — promote any chembl_id / SMILES present, and
    # only flag zero_matches when the response truly carries an empty
    # result list.
    promoted = _apply_compound_tool_enrichment(
        record,
        output,
        source_artifact_id=tc.tool_output_artifact_id,
        tool_name=plan.tool_name,
        query_kind=plan.query_kind,
        query_value=short_query,
    )
    if promoted > 0:
        return
    if _chembl_payload_has_empty_results(output):
        # NOTE: phrasing avoids the literal substring "hits" so existing
        # raw-payload canary tests stay valid.
        _push(f"{prefix}: zero_matches_returned")
    else:
        # Successful call but no chembl_id / smiles discoverable. Record a
        # softer gap so the audit chain stays honest.
        _push(f"{prefix}: no_chembl_id_extracted")


def _chembl_payload_has_empty_results(payload: Any) -> bool:
    """True iff the wrapper output explicitly carries an empty list at one
    of the canonical result-list locations (top-level, payload, data)."""
    if not isinstance(payload, dict):
        return False
    candidates: list[Any] = [payload]
    inner_payload = payload.get("payload")
    if isinstance(inner_payload, dict):
        candidates.append(inner_payload)
        data = inner_payload.get("data")
        if isinstance(data, dict):
            candidates.append(data)
    for parent in candidates:
        if not isinstance(parent, dict):
            continue
        for key in ("molecules", "results", "items", "records", "documents"):
            if key in parent and isinstance(parent[key], list) and not parent[key]:
                return True
    return False


def _antibody_payload_has_sequence(payload: Any) -> bool:
    """True iff the payload contains a heavy/light antibody sequence field.

    Used after a SAbDab enrichment call to decide whether the live response
    actually carried a usable sequence string; if not, we record a data
    gap on the antibody candidate so Step 6's sequence lane stays honest.
    """
    if not isinstance(payload, dict):
        return False
    SEQUENCE_KEYS = (
        "heavy_chain_sequence", "light_chain_sequence", "vh_sequence",
        "vl_sequence", "heavy_chain_aa", "light_chain_aa", "seq", "sequence",
        "fv_sequence", "scfv_sequence",
    )

    def walk(obj: Any, depth: int = 0) -> bool:
        if depth > 4:
            return False
        if isinstance(obj, dict):
            for k, v in obj.items():
                if str(k).lower() in SEQUENCE_KEYS and isinstance(v, str) and len(v.strip()) >= 30:
                    return True
                if walk(v, depth + 1):
                    return True
        elif isinstance(obj, list):
            for item in obj:
                if walk(item, depth + 1):
                    return True
        return False
    return walk(payload)


def _annotate_antibody_sabdab_outcome(
    *, storage: Storage, record: CandidateRecord, tc: ToolCallRecord
) -> None:
    """Record a compact data_gap / context_note when SAbDab was queried
    but did NOT produce any usable heavy/light sequence field on the
    antibody candidate."""
    has_seq_material = any(
        m.material_type in {"antibody_heavy_chain_sequence", "antibody_light_chain_sequence"}
        for m in record.materials
    )
    if has_seq_material:
        return
    note: str | None = None
    gap: str | None = None
    if tc.run_status == "success" and tc.tool_output_ref:
        try:
            output = (storage.read_json(tc.tool_output_ref) or {}).get("output") or {}
        except Exception:  # noqa: BLE001
            output = {}
        if not _antibody_payload_has_sequence(output):
            note = (
                "SAbDab_search_structures returned ok but no antibody "
                "heavy/light sequence was extracted; Step 6 sequence lane "
                "remains missing input"
            )
            gap = "antibody_sequence_missing:sabdab_no_sequence_field"
    elif tc.run_status == "dependency_unavailable":
        note = (
            "SAbDab_search_structures dependency_unavailable; antibody "
            "heavy/light sequence not retrieved; Step 6 sequence lane "
            "remains missing input"
        )
        gap = "antibody_sequence_missing:sabdab_dependency_unavailable"
    elif tc.run_status in {"failed", "partial"} or tc.error_message:
        note = (
            f"SAbDab_search_structures failed: "
            f"{(tc.error_message or 'unknown_error')[:120]}; antibody "
            "heavy/light sequence not retrieved; Step 6 sequence lane "
            "remains missing input"
        )
        gap = "antibody_sequence_missing:sabdab_failed"
    if note and note not in record.context_notes:
        record.context_notes.append(note)
    if gap and gap not in record.data_gaps:
        record.data_gaps.append(gap)


def _append_identifier_once(record: CandidateRecord, ident: Identifier) -> bool:
    """Append an identifier if not already present. Returns True when newly added."""
    if any(
        existing.id_type == ident.id_type
        and existing.id_value.lower() == ident.id_value.lower()
        for existing in record.identifiers
    ):
        return False
    record.identifiers.append(ident)
    for source_id in ident.source_ids:
        if source_id and source_id not in record.source_records:
            record.source_records.append(source_id)
    return True


def _append_material_once(record: CandidateRecord, material: Material) -> None:
    if any(
        existing.material_type == material.material_type
        and existing.value.lower() == material.value.lower()
        for existing in record.materials
    ):
        return
    record.materials.append(material)


def _material_dedup_key(m: Material) -> tuple:
    """Strict per-material dedup key for one candidate.

    Two materials are considered duplicates only when ALL of the
    distinguishing axes match: ``material_type`` (so payload vs linker,
    heavy vs light stay distinct), normalized ``value`` (storage path
    / sequence string / identifier value, lowercased + stripped),
    ``value_format`` (e.g. ``text`` vs ``smiles`` vs ``fasta``), ``role``,
    and ``role_status``. Different roles or different sources never
    collapse together; only true byte-for-byte duplicates do.
    """
    return (
        (m.material_type or "").lower(),
        (m.value or "").strip().lower(),
        (m.value_format or "").lower(),
        (m.role or "").lower(),
        (m.role_status or "").lower(),
    )


def _dedupe_candidate_materials(materials: list[Material]) -> list[Material]:
    """Drop duplicate materials within ONE candidate.

    Order is preserved; the first occurrence wins so later identical
    additions (e.g. a UniProt-derived path that converges on the same
    canonical sequence reference) silently no-op. Materials with any
    distinguishing field different — material_type, normalized value,
    value_format, role, role_status — are NEVER merged.
    """
    seen: set[tuple] = set()
    out: list[Material] = []
    for m in materials:
        key = _material_dedup_key(m)
        if key in seen:
            continue
        seen.add(key)
        out.append(m)
    return out


def _dedupe_candidate_identifiers(
    identifiers: list[Identifier],
) -> list[Identifier]:
    """Drop duplicate identifiers within ONE candidate AND merge
    provenance.

    Dedup key is ``(id_type.lower(), id_value.strip().upper())``. When
    a duplicate is encountered, we merge its ``source_ids`` into the
    first occurrence (set-union preserving order) and lift the
    surviving ``confidence`` to the higher of the two values. Different
    ``id_type`` or different ``id_value`` are never merged — e.g. two
    distinct UniProt accessions stay distinct.
    """
    first_at: dict[tuple[str, str], int] = {}
    out: list[Identifier] = []
    for ident in identifiers:
        key = (
            (ident.id_type or "").lower(),
            (ident.id_value or "").strip().upper(),
        )
        if key in first_at:
            existing = out[first_at[key]]
            existing_sources = list(existing.source_ids or [])
            seen_set = set(existing_sources)
            for s in ident.source_ids or []:
                if s and s not in seen_set:
                    existing_sources.append(s)
                    seen_set.add(s)
            existing.source_ids = existing_sources
            try:
                new_conf = float(ident.confidence or 0)
                cur_conf = float(existing.confidence or 0)
            except (TypeError, ValueError):
                new_conf, cur_conf = 0.0, 0.0
            if new_conf > cur_conf:
                existing.confidence = ident.confidence
            continue
        first_at[key] = len(out)
        out.append(ident)
    return out


def _compound_enrichment_smiles_role(record: CandidateRecord) -> tuple[str, str]:
    mat_types = {m.material_type for m in record.materials}
    if "linker_payload_name" in mat_types:
        return "compound_smiles", "linker_payload"
    if "linker_name" in mat_types and "payload_name" not in mat_types:
        return "linker_smiles", "linker"
    if "payload_name" in mat_types and "linker_name" not in mat_types:
        return "payload_smiles", "payload"
    return "compound_smiles", "compound"


def _compound_enrichment_name_role(record: CandidateRecord) -> tuple[str, str]:
    mat_types = {m.material_type for m in record.materials}
    if "linker_payload_name" in mat_types:
        return "compound_name", "linker_payload"
    if "linker_name" in mat_types and "payload_name" not in mat_types:
        return "linker_name", "linker"
    if "payload_name" in mat_types and "linker_name" not in mat_types:
        return "payload_name", "payload"
    return "compound_name", "compound"
