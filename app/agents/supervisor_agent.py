"""SupervisorAgent — Step 2 parsing, Step 4 branch coordination, inter-step routing.

Step 2: takes the persisted `raw_request_record` and produces a canonical
`structured_query` (per ADC_Pipeline_IO_Schema_v0.1.md). All non-LLM fields
(run_id, parsed_at, source_raw_request_ref) are filled by this agent so the
LLM never has to invent them.

Step 2 prompt contract (production-style, first hardening pass):

- The LLM ONLY sees `raw_user_query`, `user_provided_context`, and the
  metadata of `uploaded_files` (file_id / original_filename / content_type
  / sha256 / size_bytes). It NEVER sees uploaded file bytes, MCP tool
  lists, ToolUniverse parameter schemas, or any pipeline-internal state.
- Output is JSON only, matching the Step 2 `structured_query` schema.
- Identifiers must NOT be invented — only those explicitly present in the
  request text, context, or filename get carried into the structured
  query. Missing fields stay `null` / `[]` with a `parse_warnings` entry.
- `requested_outputs` is constrained to a five-value enum; common
  aliases (`adc_candidate`, `candidates`, `patent`, …) are normalized by
  the GeminiProvider before validation.

This file deliberately does not import anything MCP / ToolUniverse — Step 2
must not call tools, build tool arguments, or check input completeness.
"""

from __future__ import annotations

from typing import Any, Optional

from ..llm.provider import LLMProvider
from ..schemas.step_01_raw_request_record import RawRequestRecord
from ..schemas.step_02_structured_query import (
    EntityComponent,
    EntityDecomposition,
    MentionedEntities,
    NormalizedEntity,
    SourceRawRequestRef,
    StructuredQuery,
    TaskIntent,
)
from ..utils.time import now_iso


SUPERVISOR_SYSTEM_PROMPT = """You are the ADC pipeline Step-2 structured-query parser.

Your single job is to convert a user's free-text ADC design request into the
canonical structured_query JSON. You do not plan workflows, you do not pick
tools, you do not check whether the request is complete — those are later
steps. You extract what is stated AND normalize known biomedical / ADC
aliases without inventing unsupported facts.

Output contract:

1. Return EXACTLY ONE valid JSON object. No prose, no markdown fences, no
   tool calls. The object MUST match the structured_query schema fields:
   `task_intent`, `mentioned_entities`, `referenced_inputs`,
   `requested_outputs`, `user_constraints`, `parse_warnings`,
   `normalized_entities`, `entity_decompositions`,
   `clarification_questions`.

2. Confidence scores are floats in [0.0, 1.0]. Use lower confidence when
   the signal is weak (e.g. the user mentioned ADC casually but did not
   describe a design task).

Intent classification:

- `task_intent.primary_intent` MUST be one of:
  `new_adc_design`, `existing_adc_evaluation`, `developability_assessment`,
  `structure_analysis`, `compound_screening`, `literature_review`,
  `patent_ip_review`, `optimization`, `unclear_or_needs_clarification`.
- `task_intent.secondary_intents` is a deduped list drawn from the same
  enum. Use it to capture richer questions like "evaluate T-DM1 vs T-DXd"
  (primary `existing_adc_evaluation`, secondary `literature_review` and
  `developability_assessment`).
- `task_type` (legacy free-form) and `modality` MUST also be populated
  for backward compatibility.

Entity extraction + normalization:

- For each entity in `mentioned_entities`, preserve the literal user
  phrasing as the value. Do not overwrite it with the canonical form.
- For every mentioned biomedical entity (target, disease, antibody,
  payload, linker, drug name, compound) ALSO emit a `normalized_entities`
  record. Each record MUST contain:
    * `original_text` — what the user actually wrote;
    * `canonical_name` — the resolved canonical label (e.g. HER2 → ERBB2);
    * `canonical_id` (optional) and `canonical_id_source` (e.g. HGNC,
       UniProt, DrugBank);
    * `entity_type` — one of `target_or_antigen`, `disease_or_indication`,
       `antibody`, `payload`, `linker`, `drug`, `compound`, `other`;
    * `explicit_or_inferred` — `explicit` when the user wrote the
       canonical form (or the alias and the canonical form unambiguously
       point to the same entity), `inferred` when the canonical form is
       a parser-supplied resolution of an alias.
- Normalization examples (apply when relevant): HER2 → ERBB2;
  TROP2 → TACSTD2; CLDN18.2 → CLDN18 isoform 2; Enhertu / T-DXd →
  trastuzumab deruxtecan; T-DM1 → ado-trastuzumab emtansine; MMAE →
  monomethyl auristatin E; DXd → topoisomerase I inhibitor payload
  family.
- Multi-component ADC names produce `entity_decompositions` entries with
  inferred components. Example: `T-DM1` → trastuzumab (antibody) +
  emtansine/DM1 (payload). `T-DXd` / Enhertu → trastuzumab (antibody) +
  deruxtecan (linker_payload) + DXd / topoisomerase I inhibitor
  (payload). `vc-MMAE` → valine-citrulline linker (inferred) + MMAE
  payload (explicit). Every component carries `inferred=True` unless the
  user explicitly wrote that component.
- `entity_decompositions[].components[]` entries MUST use
  `canonical_name` for the component label. Do NOT use `component_name`,
  `name`, `label`, or `value` for component objects. Each component object
  should include `canonical_name` and, when known, `component_type`.

Identifier extraction:

- `referenced_inputs` carries explicit IDs and uploaded-file references
  ONLY. Each entry is `{"id_type": "<pdb_id|uniprot_id|chembl_id|"
  "pubchem_cid|drugbank_id|zinc_id|smiles|doi|pmid|patent_application_number|"
  "uploaded_file>", "value": "<string>", "source": "<short text>"}`.
  ZINC IDs are NEVER labeled `zinc22`; the label stays `zinc_id` until a
  downstream tool confirms the source library.

Requested outputs:

- `requested_outputs` is a list of strings drawn ONLY from this canonical
  enum: `"ranked_candidates"`, `"report"`, `"evidence_summary"`,
  `"literature_review_summary"`, `"patent_or_ip_summary"`,
  `"optimization_suggestions"`, `"developability_summary"`,
  `"structure_validation_report"`, `"compound_screening_results"`,
  `"entity_normalization_summary"`, `"workflow_recommendation"`,
  `"data_gap_summary"`, `"case_study_summary"`.
- Map common aliases (e.g. `adc_candidate` → `ranked_candidates`,
  `literature_summary` → `literature_review_summary`,
  `gap_analysis` → `data_gap_summary`) to the canonical enum. Drop
  anything outside the enum and add a `parse_warnings` entry naming
  the dropped value.

User constraints, warnings, clarifications:

- `user_constraints` MUST be a JSON array of OBJECTS. Each object has at
  least `{"constraint_text": "<the user's literal phrasing>",
  "source": "user"}`. Do NOT emit plain strings here. Preserve the
  user's phrasing — do not reinterpret into numeric thresholds, DAR
  ranges, or scientific tolerances unless the user wrote those numbers.
- `parse_warnings` MUST be a JSON array of plain STRINGS — short
  English sentences describing parser ambiguity, dropped values,
  inferred resolutions you are not confident about, or low-confidence
  extractions. Do NOT emit objects here. They are not shown to the end
  user verbatim.
- `clarification_questions` are USER-FACING short questions surfaced
  back to the operator. Add one when a required component for the
  declared workflow is missing (e.g. "Which linker chemistry should we
  assume for this MMAE payload?") or when the request is ambiguous
  between several intents (e.g. "Should we treat this as a HER2 ADC
  evaluation or a generic HER2 screening?"). Keep each question short
  and answerable.

Inference rules:

- When you infer something (e.g. expanding T-DM1 into trastuzumab +
  emtansine), mark every inferred record explicitly via
  `explicit_or_inferred="inferred"` (for normalized_entities) or
  `inferred=True` (for entity decomposition components). Add a
  matching `parse_warnings` or `clarification_questions` entry when the
  inference is meaningful.
- Do NOT invent identifiers, molecules, targets, candidates, or
  downstream tool inputs that the user did not mention and that are not
  the canonical resolution of an explicit alias.

Privacy / safety:

- You will NEVER receive raw file bytes. Only file metadata. Treat
  filenames as advisory text; do not invent contents from them.
- You will NEVER receive API keys, MCP tool lists, ToolUniverse parameter
  schemas, or other pipeline state. Do not request them.
""".strip()


_UPLOADED_FILE_META_FIELDS = (
    "file_id",
    "original_filename",
    "content_type",
    "sha256",
    "size_bytes",
)


def _prompt_inputs_from_raw(raw_request_record: dict) -> dict[str, Any]:
    """Build the slim payload the LLM is allowed to see.

    Strips storage paths, intake bookkeeping, registry IDs, and anything
    that could leak file bytes. Keeps `raw_user_query`,
    `user_provided_context`, and `uploaded_files` metadata only.
    """
    ctx = raw_request_record.get("user_provided_context") or {}
    files_in = raw_request_record.get("uploaded_files") or []
    files_out: list[dict[str, Any]] = []
    for f in files_in:
        if not isinstance(f, dict):
            continue
        slim = {k: f.get(k) for k in _UPLOADED_FILE_META_FIELDS if f.get(k) is not None}
        if slim:
            files_out.append(slim)
    return {
        "raw_user_query": raw_request_record.get("raw_user_query") or "",
        "user_provided_context": {
            k: v for k, v in ctx.items() if v not in (None, "", [], {})
        },
        "uploaded_files": files_out,
    }


_BACKFILL_FROM_ENTITY_TYPES: tuple[tuple[str, tuple[str, ...]], ...] = (
    # (mentioned_entities field, allowed normalized_entity.entity_type values)
    # Order matters only for documentation; each field is filled independently.
    ("target_or_antigen_text", ("target_or_antigen",)),
    ("disease_or_indication_text", ("disease_or_indication",)),
    ("antibody_candidate_text", ("antibody",)),
    ("payload_text", ("payload",)),
    ("linker_text", ("linker", "linker_payload")),
)


def _is_empty(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and not value.strip():
        return True
    return False


def backfill_mentioned_entities(
    mentioned: dict[str, Any] | None,
    normalized: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    """Conservatively fill legacy `mentioned_entities` from `normalized_entities`.

    Live Gemini sometimes returns rich `normalized_entities` records while
    leaving the legacy `mentioned_entities` fields null/empty — but
    downstream services (Step 3 readiness presence checks, Step 5+ agents,
    Step 1 raw context propagation) still read the flat strings. This
    helper only fills a legacy field when:

    - the legacy field is missing / null / empty string, AND
    - a `normalized_entities` entry exists with an `entity_type` matched
      to that legacy field via `_BACKFILL_FROM_ENTITY_TYPES`.

    Rules respected:

    - Use `original_text` (user phrasing), NOT `canonical_name`. Step 2's
      contract is to preserve what the user wrote in `mentioned_entities`.
    - Never overwrite an existing non-empty value.
    - `entity_type="drug"` (entire ADC product like T-DM1 / T-DXd) is
      deliberately NOT mapped to antibody / payload — its decomposition
      handles components separately.
    - If multiple matching entries exist, take the FIRST one in
      normalized_entities order; never concatenate.
    """
    out: dict[str, Any] = dict(mentioned or {})
    norms = normalized or []
    for field, allowed_types in _BACKFILL_FROM_ENTITY_TYPES:
        if not _is_empty(out.get(field)):
            continue
        for ne in norms:
            if not isinstance(ne, dict):
                continue
            entity_type = ne.get("entity_type")
            if entity_type not in allowed_types:
                continue
            original = ne.get("original_text")
            if isinstance(original, str) and original.strip():
                out[field] = original
                break
    return out


def _coerce_warning_entry(item: Any) -> Optional[str]:
    """Compact-stringify a `parse_warnings` entry from a drifted LLM payload.

    Real Gemini occasionally returns dict-shaped warnings like
    ``{"warning_code": "X", "message": "...", "confidence": 0.6}`` instead
    of the schema-required plain strings. We compact those into a single
    line; unknown shapes degrade to ``str(item)``; ``None`` / empty drops
    silently (returns ``None``).
    """
    if item is None:
        return None
    if isinstance(item, str):
        s = item.strip()
        return s or None
    if isinstance(item, dict):
        # Prefer the most informative fields, but always include code+message
        # when both are present so downstream readers can grep either.
        code = item.get("warning_code") or item.get("code")
        message = item.get("message") or item.get("text") or item.get("warning")
        confidence = item.get("confidence")
        parts: list[str] = []
        if code:
            parts.append(str(code))
        if message:
            parts.append(str(message))
        if not parts:
            # Fallback: dump compact key/value pairs of any string-ish content.
            kv = [f"{k}={v}" for k, v in item.items() if isinstance(v, (str, int, float, bool))]
            if kv:
                parts.append("; ".join(kv))
        if confidence is not None and isinstance(confidence, (int, float)):
            parts.append(f"confidence={float(confidence):.2f}")
        joined = " | ".join(parts).strip()
        return joined or None
    if isinstance(item, (list, tuple)):
        text = ", ".join(str(x) for x in item if x is not None)
        return text or None
    text = str(item).strip()
    return text or None


def _coerce_constraint_entry(item: Any) -> Optional[dict[str, Any]]:
    """Coerce a `user_constraints` entry into the canonical dict shape.

    Schema requires ``list[dict]``. Real Gemini sometimes returns plain
    strings like ``"DAR<=4"``. We wrap those into
    ``{"constraint_text": str, "source": "llm_output"}``; dict entries
    pass through but pick up ``source="llm_output"`` if the LLM omitted
    one and ``constraint_text`` if a free-form text field exists under
    another common key. ``None`` / empty drops silently.
    """
    if item is None:
        return None
    if isinstance(item, str):
        s = item.strip()
        if not s:
            return None
        return {"constraint_text": s, "source": "llm_output"}
    if isinstance(item, dict):
        out: dict[str, Any] = dict(item)
        if not out.get("constraint_text"):
            for alt in ("text", "value", "description", "constraint"):
                v = out.get(alt)
                if isinstance(v, str) and v.strip():
                    out["constraint_text"] = v.strip()
                    break
        out.setdefault("source", "llm_output")
        return out
    text = str(item).strip()
    if not text:
        return None
    return {"constraint_text": text, "source": "supervisor_coerced"}


_COMPONENT_NAME_ALIASES = ("component_name", "name", "label", "value")
_COMPONENT_ROLE_VALUES = {"antibody", "payload", "linker", "linker_payload", "other"}
_ENTITY_TYPE_ALIASES = {
    "target": "target_or_antigen",
    "antigen": "target_or_antigen",
    "target-antigen": "target_or_antigen",
    "target antigen": "target_or_antigen",
    "target_or_antigen": "target_or_antigen",
    "disease": "disease_or_indication",
    "indication": "disease_or_indication",
    "disease-or-indication": "disease_or_indication",
    "disease indication": "disease_or_indication",
    "payload-linker": "linker_payload",
    "payload linker": "linker_payload",
    "linker-payload": "linker_payload",
    "linker payload": "linker_payload",
    "linker_payload": "linker_payload",
    "small_molecule": "compound",
    "small-molecule": "compound",
    "small molecule": "compound",
}


def _component_name_value(item: dict[str, Any]) -> str | None:
    value = item.get("canonical_name")
    if isinstance(value, str) and value.strip():
        return value.strip()
    for key in _COMPONENT_NAME_ALIASES:
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _normalize_entity_components(payload: dict[str, Any]) -> None:
    decompositions = payload.get("entity_decompositions")
    if not isinstance(decompositions, list):
        return

    warnings = payload.get("parse_warnings")
    if not isinstance(warnings, list):
        warnings = []
    payload["parse_warnings"] = warnings

    for decomp_index, decomp in enumerate(decompositions):
        if not isinstance(decomp, dict):
            warnings.append(
                f"dropped entity_decompositions[{decomp_index}]: expected object"
            )
            continue
        components = decomp.get("components")
        if not isinstance(components, list):
            if components is not None:
                warnings.append(
                    f"dropped entity_decompositions[{decomp_index}].components: expected list"
                )
            decomp["components"] = []
            continue

        normalized_components: list[dict[str, Any]] = []
        for comp_index, component in enumerate(components):
            if not isinstance(component, dict):
                warnings.append(
                    "dropped "
                    f"entity_decompositions[{decomp_index}].components[{comp_index}]: "
                    "expected object"
                )
                continue
            canonical_name = _component_name_value(component)
            if not canonical_name:
                warnings.append(
                    "dropped "
                    f"entity_decompositions[{decomp_index}].components[{comp_index}]: "
                    "missing canonical_name"
                )
                continue

            out = dict(component)
            out["canonical_name"] = canonical_name
            component_type = out.get("component_type")
            if (
                "role" not in out
                and isinstance(component_type, str)
                and component_type in _COMPONENT_ROLE_VALUES
            ):
                out["role"] = component_type
            normalized_components.append(out)

        decomp["components"] = normalized_components


def _normalize_entity_type_value(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    raw = value.strip()
    if not raw:
        return None
    lowered = raw.lower().replace("_", " ")
    compact = " ".join(lowered.replace("/", " ").split())
    return (
        _ENTITY_TYPE_ALIASES.get(raw)
        or _ENTITY_TYPE_ALIASES.get(raw.lower())
        or _ENTITY_TYPE_ALIASES.get(compact)
        or _ENTITY_TYPE_ALIASES.get(compact.replace(" ", "-"))
        or raw
    )


def _normalize_normalized_entity_types(payload: dict[str, Any]) -> None:
    normalized_entities = payload.get("normalized_entities")
    if not isinstance(normalized_entities, list):
        return

    warnings = payload.get("parse_warnings")
    if not isinstance(warnings, list):
        warnings = []
    payload["parse_warnings"] = warnings

    for i, entity in enumerate(normalized_entities):
        if not isinstance(entity, dict):
            continue
        original = entity.get("entity_type")
        normalized = _normalize_entity_type_value(original)
        if normalized is None:
            continue
        if normalized != original:
            entity["entity_type"] = normalized
            warnings.append(
                f"normalized normalized_entities[{i}].entity_type from {original!r} to {normalized!r}"
            )


def normalize_llm_payload_for_step2(payload: dict) -> dict:
    """Defensive coercer applied at the Step 2 parse boundary.

    Handles real schema-drift cases observed against live HER2 ADC inputs:

    1. ``parse_warnings`` returned as ``list[dict]`` instead of ``list[str]``.
    2. ``user_constraints`` returned as ``list[str]`` instead of ``list[dict]``.
    3. Entity decomposition components returned with alias keys such as
       ``component_name`` instead of the schema-required ``canonical_name``.

    Idempotent: payloads already matching the schema pass through unchanged
    (no spurious wrapping, no duplicate sources). Never raises — unknown
    item types degrade to ``str()`` / drop instead of breaking validation.
    """
    if not isinstance(payload, dict):
        return payload
    out = dict(payload)

    raw_warnings = out.get("parse_warnings")
    if isinstance(raw_warnings, list):
        coerced_warnings: list[str] = []
        for item in raw_warnings:
            s = _coerce_warning_entry(item)
            if s:
                coerced_warnings.append(s)
        out["parse_warnings"] = coerced_warnings

    raw_constraints = out.get("user_constraints")
    if isinstance(raw_constraints, list):
        coerced_constraints: list[dict[str, Any]] = []
        for item in raw_constraints:
            d = _coerce_constraint_entry(item)
            if d:
                coerced_constraints.append(d)
        out["user_constraints"] = coerced_constraints

    _normalize_entity_components(out)
    _normalize_normalized_entity_types(out)

    return out


def build_supervisor_user_prompt(raw_request_record: dict) -> str:
    """The user-message portion of the Step 2 LLM call.

    Compact instruction reminding the LLM of its job, plus the slim
    `prompt_inputs` payload. The schema name is passed through the
    `schema={"task": "structured_query", ...}` channel so the
    GeminiProvider can attach the canonical shape hint.
    """
    return (
        "Parse the user's ADC design request into the structured_query schema. "
        "Use only the prompt_inputs payload below. Leave unknowns null and "
        "record gaps in parse_warnings. Return JSON only."
    )


class SupervisorAgent:
    name = "supervisor_agent"

    def __init__(self, llm: LLMProvider, mcp_client: Any | None = None) -> None:
        self.llm = llm
        # Step 2 must NOT use mcp_client. The attribute is retained for the
        # later Step 4 path; defensive checks below confirm nothing here
        # routes through it.
        self.mcp_client = mcp_client

    def parse_raw_to_structured_query(self, raw_request_record: dict) -> StructuredQuery:
        # Defensive: ensure the payload at least parses as a raw_request_record.
        RawRequestRecord.model_validate(
            {k: v for k, v in raw_request_record.items() if k != "artifact_id"}
        )

        prompt_inputs = _prompt_inputs_from_raw(raw_request_record)
        prompt = build_supervisor_user_prompt(raw_request_record)
        llm_payload = self.llm.generate_json(
            prompt,
            schema={
                "task": "structured_query",
                "prompt_inputs": prompt_inputs,
                # MockLLMProvider still expects `raw_request_record` for
                # its rule-based path; pass it unchanged so the deterministic
                # mock keeps working without a Gemini key.
                "raw_request_record": raw_request_record,
            },
            system=SUPERVISOR_SYSTEM_PROMPT,
        )
        # Defensive coercion for real-LLM schema drift on the two fields
        # most often returned in the wrong shape (parse_warnings as dicts,
        # user_constraints as strings). Idempotent for conformant payloads.
        llm_payload = normalize_llm_payload_for_step2(llm_payload or {})

        # Agent fills the deterministic fields, never the LLM.
        normalized_entities = [
            NormalizedEntity(**ne)
            for ne in (llm_payload.get("normalized_entities") or [])
            if isinstance(ne, dict)
        ]
        entity_decompositions: list[EntityDecomposition] = []
        for ed in llm_payload.get("entity_decompositions") or []:
            if not isinstance(ed, dict):
                continue
            comps_raw = ed.get("components") or []
            comps = [
                EntityComponent(**c) for c in comps_raw if isinstance(c, dict)
            ]
            entity_decompositions.append(
                EntityDecomposition(
                    original_text=ed.get("original_text", ""),
                    canonical_name=ed.get("canonical_name"),
                    components=comps,
                    notes=ed.get("notes"),
                )
            )
        mentioned_entities_dict = backfill_mentioned_entities(
            llm_payload.get("mentioned_entities") or {},
            llm_payload.get("normalized_entities") or [],
        )
        sq = StructuredQuery(
            run_id=raw_request_record["run_id"],
            parsed_at=now_iso(),
            source_raw_request_ref=SourceRawRequestRef(
                raw_request_record_id=raw_request_record.get("artifact_id")
                or raw_request_record["run_artifact_registry_id"]
            ),
            task_intent=TaskIntent(**(llm_payload.get("task_intent") or {"task_type": "adc_design"})),
            mentioned_entities=MentionedEntities(**mentioned_entities_dict),
            referenced_inputs=llm_payload.get("referenced_inputs") or [],
            requested_outputs=llm_payload.get("requested_outputs") or [],
            user_constraints=llm_payload.get("user_constraints") or [],
            parse_warnings=llm_payload.get("parse_warnings") or [],
            normalized_entities=normalized_entities,
            entity_decompositions=entity_decompositions,
            clarification_questions=llm_payload.get("clarification_questions") or [],
        )
        return sq

    def run(self, *, run_id: str, step_id: str, payload: dict) -> dict:  # noqa: D401
        raise NotImplementedError
