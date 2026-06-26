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

import re
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


SUPERVISOR_SYSTEM_PROMPT = """You are the ADC pipeline structured-query parser.

You must convert the user's free-text request into one valid `structured_query`
JSON object. You are a lossless parser and conservative normalizer, not a
tool planner or biomedical reasoning step. Preserve explicit user inputs,
normalize common ADC / protein aliases, and never invent IDs, SMILES,
sequences, candidates, rankings, tool calls, or downstream results.

Use these fields:
- `task_intent`: choose the closest intent enum and add secondary intents
  only when the user clearly asks for them.
- `mentioned_entities`: literal user labels only, such as target,
  antibody, payload, linker, drug, disease, or compound names. Do not put
  bare SMILES, accessions, PDB IDs, file paths, or file contents here.
- `normalized_entities`: canonical forms for the literal mentions. Use
  `original_text`, `canonical_name`, optional explicit `canonical_id`,
  `canonical_id_source`, `entity_type`, and `explicit_or_inferred`.
- `entity_decompositions`: component breakdowns for composite ADC terms.
  Use `components[].canonical_name` and `component_type`; do not use
  `component_name`, `name`, `label`, or `value` inside component objects.
- `referenced_inputs`: explicit typed inputs only: UniProt/PDB/ChEMBL/
  PubChem/DrugBank/ZINC IDs, SMILES, uploaded files, inline protein
  sequences, antibody chain sequences, or explicitly supplied CDR3.

Input-to-output mapping:
- HER2 -> normalized ERBB2; TROP2 -> TACSTD2; MMAE -> monomethyl
  auristatin E; T-DM1 -> ado-trastuzumab emtansine; T-DXd / Enhertu ->
  trastuzumab deruxtecan.
- Explicit UniProt accession `P04626` -> `{"id_type": "uniprot_id",
  "value": "P04626", "source": "user"}`.
- Explicit PDB ID `7XYZ` -> `{"id_type": "pdb_id", "value": "7XYZ",
  "source": "user"}`.
- Uploaded file metadata with `file_id` -> `{"id_type": "uploaded_file",
  "value": "<file_id>", "source": "uploaded_file"}`. Do not use
  filenames or storage paths as `value`.
- Payload SMILES -> `{"id_type": "smiles", "value": "<smiles>",
  "source": "payload_smiles"}`. Linker SMILES uses
  `source="linker_smiles"`. Unlabeled compound SMILES uses
  `source="compound_smiles"`.
- Heavy chain / VH / HC / IGH / IGHV / H chain sequence or FASTA ->
  `antibody_heavy_chain_sequence`. Light chain / VL / LC / kappa /
  lambda / IGK / IGL / IGKV / IGLV / L chain -> `antibody_light_chain_sequence`.
  If the user only says antibody/protein sequence or FASTA with no chain
  hint, use `antibody_sequence_reference`; do not default to heavy.
- Do not infer heavy vs light from sequence content. Do not extract CDR3
  from a full sequence. Preserve only an explicitly supplied CDR3 as
  `antibody_cdr3_sequence`; downstream runtime extracts CDR3 when needed.
- Composite ADC terms stay literal in `mentioned_entities` and decompose
  separately. `vc-MMAE` -> valine-citrulline linker + monomethyl
  auristatin E payload. `T-DM1` -> trastuzumab antibody + DM1 payload.
  `T-DXd` / Enhertu -> trastuzumab antibody + deruxtecan linker-payload
  + DXd payload. Do not emit isolated `vc` as a standalone linker unless
  the user wrote it as a standalone linker.

Few-shot 1:
User: "Design an ADC against HER2 / ERBB2 (UniProt P04626) using
vc-MMAE. Payload SMILES CC(C)C[C@H](N(C)C(=O)C(C)C). Linker SMILES
NCCOC(=O)O. I attached antigen structure file_id f_pdb_001."
Output essentials:
{
  "mentioned_entities": {
    "target_or_antigen_text": "HER2",
    "payload_text": "vc-MMAE",
    "linker_text": "vc-MMAE"
  },
  "referenced_inputs": [
    {"id_type": "uniprot_id", "value": "P04626", "source": "user"},
    {"id_type": "smiles", "value": "CC(C)C[C@H](N(C)C(=O)C(C)C)",
     "source": "payload_smiles"},
    {"id_type": "smiles", "value": "NCCOC(=O)O",
     "source": "linker_smiles"},
    {"id_type": "uploaded_file", "value": "f_pdb_001",
     "source": "uploaded_file"}
  ],
  "normalized_entities": [
    {"original_text": "HER2", "canonical_name": "ERBB2",
     "canonical_id": "P04626", "canonical_id_source": "UniProt",
     "entity_type": "target_or_antigen",
     "explicit_or_inferred": "explicit"}
  ],
  "entity_decompositions": [
    {"original_text": "vc-MMAE",
     "canonical_name": "valine-citrulline-MMAE",
     "components": [
       {"canonical_name": "valine-citrulline",
        "component_type": "linker", "inferred": true},
       {"canonical_name": "monomethyl auristatin E",
        "component_type": "payload", "inferred": true}
     ]}
  ]
}

Few-shot 2:
User: "Use HER2 target with trastuzumab heavy_chain.fasta and
light_chain.fasta, plus vc-MMAE."
Output essentials:
{
  "mentioned_entities": {
    "target_or_antigen_text": "HER2",
    "antibody_candidate_text": "trastuzumab",
    "payload_text": "vc-MMAE",
    "linker_text": "vc-MMAE"
  },
  "referenced_inputs": [
    {"id_type": "uploaded_file", "value": "f_heavy_001",
     "source": "antibody_heavy_chain_sequence"},
    {"id_type": "uploaded_file", "value": "f_light_002",
     "source": "antibody_light_chain_sequence"}
  ]
}

Return exactly one JSON object matching the schema. Keep `parse_warnings`
as a string array and `user_constraints` as an object array with
`constraint_text` and `source`. Keep `requested_outputs` within the
schema enum: `ranked_candidates`, `report`, `evidence_summary`,
`literature_review_summary`, `patent_or_ip_summary`,
`optimization_suggestions`, `developability_summary`,
`structure_validation_report`, `compound_screening_results`,
`entity_normalization_summary`, `workflow_recommendation`,
`data_gap_summary`, `case_study_summary`. Do not emit tool plans, MCP
selections, ToolUniverse arguments, generated ADC candidates, candidate
rankings, liability flags, literature or patent queries, pose ensembles,
DAR designs, conjugation site recommendations, raw file bytes, raw FASTA,
storage paths, prompts, API keys, or raw tool payloads.
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
_LABELED_SMILES_RE = re.compile(
    r"\b(?P<role>payload|linker|compound)\s+SMILES\s*[:=]?\s*"
    r"(?P<value>[A-Za-z0-9@+\-\[\]\(\)=#$%/\\\.]+)",
    re.IGNORECASE,
)
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
    "antibody_candidate": "antibody",
    "antibody-candidate": "antibody",
    "antibody candidate": "antibody",
    "antibody_candidate_text": "antibody",
    "antibody candidate text": "antibody",
    "payload-linker": "linker_payload",
    "payload linker": "linker_payload",
    "linker-payload": "linker_payload",
    "linker payload": "linker_payload",
    "linker_payload": "linker_payload",
    "small_molecule": "compound",
    "small-molecule": "compound",
    "small molecule": "compound",
}


def _raw_text_chunks_for_step2(raw_request_record: dict | None) -> list[str]:
    if not isinstance(raw_request_record, dict):
        return []
    chunks: list[str] = []
    query = raw_request_record.get("raw_user_query")
    if isinstance(query, str) and query.strip():
        chunks.append(query)
    ctx = raw_request_record.get("user_provided_context") or {}
    if isinstance(ctx, dict):
        for value in ctx.values():
            if isinstance(value, str) and value.strip():
                chunks.append(value)
    return chunks


def _clean_labeled_smiles_value(value: str) -> str:
    cleaned = value.strip().rstrip(".,;")
    while cleaned.endswith(")") and cleaned.count(")") > cleaned.count("("):
        cleaned = cleaned[:-1].rstrip()
    while cleaned.endswith("]") and cleaned.count("]") > cleaned.count("["):
        cleaned = cleaned[:-1].rstrip()
    return cleaned


def _extract_labeled_smiles_refs(
    raw_request_record: dict | None,
) -> list[dict[str, str]]:
    refs: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for chunk in _raw_text_chunks_for_step2(raw_request_record):
        for match in _LABELED_SMILES_RE.finditer(chunk):
            role = match.group("role").lower()
            value = _clean_labeled_smiles_value(match.group("value"))
            if not value:
                continue
            key = (role, value)
            if key in seen:
                continue
            seen.add(key)
            refs.append(
                {"id_type": "smiles", "value": value, "source": f"{role}_smiles"}
            )
    return refs


def _is_smiles_like_text(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    s = value.strip()
    if len(s) < 2 or any(ch.isspace() for ch in s):
        return False
    if any(ch in s for ch in "=#[]()/\\@+"):
        return bool(re.search(r"[A-Za-z]", s))
    if any(ch.isdigit() for ch in s):
        return bool(re.search(r"[A-Za-z]", s)) and "-" not in s
    # Conservative simple organic strings such as CCO or NCCO. This
    # intentionally excludes names like MMAE, vc-MMAE, valine-citrulline.
    return bool(re.fullmatch(r"(?:Br|Cl|B|C|N|O|P|S|F|I|c|n|o|s|p){2,}", s))


def _ensure_referenced_input(refs: list[Any], new_ref: dict[str, str]) -> None:
    for existing in refs:
        if not isinstance(existing, dict):
            continue
        if (
            existing.get("id_type") == new_ref["id_type"]
            and existing.get("value") == new_ref["value"]
            and existing.get("source") == new_ref["source"]
        ):
            return
    refs.append(dict(new_ref))


def _normalize_typed_smiles_fields(
    payload: dict[str, Any],
    raw_request_record: dict | None,
) -> None:
    labeled_refs = _extract_labeled_smiles_refs(raw_request_record)
    if not labeled_refs:
        return

    refs = payload.get("referenced_inputs")
    if not isinstance(refs, list):
        refs = []
        payload["referenced_inputs"] = refs
    for ref in labeled_refs:
        _ensure_referenced_input(refs, ref)

    mentioned = payload.get("mentioned_entities")
    if not isinstance(mentioned, dict):
        return

    warnings = payload.get("parse_warnings")
    if not isinstance(warnings, list):
        warnings = []
        payload["parse_warnings"] = warnings

    expected_by_field = {
        "payload_text": {"payload_smiles", "compound_smiles"},
        "linker_text": {"linker_smiles", "compound_smiles"},
    }
    labeled_by_value = {
        ref["value"]: ref["source"]
        for ref in labeled_refs
        if ref.get("value") and ref.get("source")
    }
    for field, allowed_sources in expected_by_field.items():
        value = mentioned.get(field)
        if not _is_smiles_like_text(value):
            continue
        source = labeled_by_value.get(str(value).strip())
        if source not in allowed_sources:
            continue
        mentioned[field] = None
        note = (
            f"removed SMILES-like mentioned_entities.{field}; "
            f"retained as referenced_inputs[source={source}]"
        )
        if note not in warnings:
            warnings.append(note)


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


_ANTIBODY_HEAVY_CHAIN_SOURCE = "antibody_heavy_chain_sequence"
_ANTIBODY_LIGHT_CHAIN_SOURCE = "antibody_light_chain_sequence"
_ANTIBODY_GENERIC_CHAIN_SOURCE = "antibody_sequence_reference"

# Drifted id_type / source aliases the live LLM occasionally returns
# instead of the canonical antibody-chain source string. We promote
# only when the alias is unambiguous; ``antibody_sequence``,
# ``protein_sequence``, ``fasta`` etc. stay generic — we never invent a
# heavy/light role from a chain-silent alias.
_ANTIBODY_HEAVY_ID_TYPE_ALIASES = {
    "heavy_chain_sequence",
    "vh_sequence",
    "hc_sequence",
    "antibody_heavy_chain_sequence",
    "heavy_chain",
    "antibody_heavy_chain",
    "antibody_vh_sequence",
    "antibody_hc_sequence",
}
_ANTIBODY_LIGHT_ID_TYPE_ALIASES = {
    "light_chain_sequence",
    "vl_sequence",
    "lc_sequence",
    "kappa_sequence",
    "lambda_sequence",
    "antibody_light_chain_sequence",
    "light_chain",
    "antibody_light_chain",
    "antibody_vl_sequence",
    "antibody_lc_sequence",
    "antibody_kappa_sequence",
    "antibody_lambda_sequence",
}
_ANTIBODY_GENERIC_ID_TYPE_ALIASES = {
    "antibody_sequence",
    "antibody_sequence_reference",
    "protein_sequence",
    "fasta_sequence",
    "amino_acid_sequence",
}


def _normalize_antibody_chain_references(payload: dict[str, Any]) -> None:
    """Promote heavy / light / generic antibody-sequence drift in
    ``referenced_inputs`` to canonical id_types / source strings.

    Drift comes in two shapes:

    1. The LLM emits an entry with ``id_type`` in
       ``_ANTIBODY_*_ID_TYPE_ALIASES`` and a literal inline sequence in
       ``value`` — we rewrite ``id_type`` to the canonical chain string
       (``antibody_heavy_chain_sequence`` / ``…_light_…`` / generic)
       and fill ``source="user"`` if missing.
    2. The LLM emits an ``id_type="uploaded_file"`` entry with a
       drifted ``source`` (e.g. ``vh_sequence``) — we rewrite the
       ``source`` to the canonical chain string. The ``value`` (file_id)
       stays untouched.

    Generic / chain-silent aliases (``antibody_sequence``,
    ``protein_sequence``, ``fasta``) stay generic — they are coerced to
    ``antibody_sequence_reference`` so the downstream Step 5 agent sees
    a stable source string, but they are NEVER promoted to heavy.
    """
    refs = payload.get("referenced_inputs")
    if not isinstance(refs, list):
        return

    warnings = payload.get("parse_warnings")
    if not isinstance(warnings, list):
        warnings = []
    payload["parse_warnings"] = warnings

    for entry in refs:
        if not isinstance(entry, dict):
            continue
        id_type = entry.get("id_type")
        source = entry.get("source")
        id_type_lower = id_type.lower() if isinstance(id_type, str) else ""
        source_lower = source.lower() if isinstance(source, str) else ""

        # Case 1: inline drifted id_type.
        if id_type_lower in _ANTIBODY_HEAVY_ID_TYPE_ALIASES:
            new_type = _ANTIBODY_HEAVY_CHAIN_SOURCE
        elif id_type_lower in _ANTIBODY_LIGHT_ID_TYPE_ALIASES:
            new_type = _ANTIBODY_LIGHT_CHAIN_SOURCE
        elif id_type_lower in _ANTIBODY_GENERIC_ID_TYPE_ALIASES:
            new_type = _ANTIBODY_GENERIC_CHAIN_SOURCE
        else:
            new_type = None
        if new_type and id_type != new_type:
            old_id_type = id_type
            entry["id_type"] = new_type
            entry.setdefault("source", "user")
            warnings.append(
                f"normalized referenced_inputs.id_type from "
                f"{old_id_type!r} to {new_type!r}"
            )
            id_type = new_type
            id_type_lower = id_type.lower()

        # Case 2: uploaded_file entry whose source carries a drifted
        # chain alias. The source is the only place where the chain
        # hint can live on uploaded_file entries.
        if id_type == "uploaded_file" and source_lower:
            if source_lower in _ANTIBODY_HEAVY_ID_TYPE_ALIASES:
                if source != _ANTIBODY_HEAVY_CHAIN_SOURCE:
                    entry["source"] = _ANTIBODY_HEAVY_CHAIN_SOURCE
                    warnings.append(
                        f"normalized referenced_inputs.source from "
                        f"{source!r} to {_ANTIBODY_HEAVY_CHAIN_SOURCE!r}"
                    )
            elif source_lower in _ANTIBODY_LIGHT_ID_TYPE_ALIASES:
                if source != _ANTIBODY_LIGHT_CHAIN_SOURCE:
                    entry["source"] = _ANTIBODY_LIGHT_CHAIN_SOURCE
                    warnings.append(
                        f"normalized referenced_inputs.source from "
                        f"{source!r} to {_ANTIBODY_LIGHT_CHAIN_SOURCE!r}"
                    )
            elif source_lower in _ANTIBODY_GENERIC_ID_TYPE_ALIASES:
                if source != _ANTIBODY_GENERIC_CHAIN_SOURCE:
                    entry["source"] = _ANTIBODY_GENERIC_CHAIN_SOURCE
                    warnings.append(
                        f"normalized referenced_inputs.source from "
                        f"{source!r} to {_ANTIBODY_GENERIC_CHAIN_SOURCE!r}"
                    )


def normalize_llm_payload_for_step2(
    payload: dict,
    raw_request_record: dict | None = None,
) -> dict:
    """Defensive coercer applied at the Step 2 parse boundary.

    Handles real schema-drift cases observed against live HER2 ADC inputs:

    1. ``parse_warnings`` returned as ``list[dict]`` instead of ``list[str]``.
    2. ``user_constraints`` returned as ``list[str]`` instead of ``list[dict]``.
    3. Entity decomposition components returned with alias keys such as
       ``component_name`` instead of the schema-required ``canonical_name``.
    4. Antibody chain references returned with drifted id_type /
       source strings (``heavy_chain_sequence`` / ``vh_sequence`` / …).
       The normalizer promotes unambiguous heavy / light aliases and
       collapses chain-silent generic aliases — it NEVER fabricates a
       heavy/light role from a generic alias.

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
    _normalize_typed_smiles_fields(out, raw_request_record)
    _normalize_antibody_chain_references(out)

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
        llm_payload = normalize_llm_payload_for_step2(
            llm_payload or {},
            raw_request_record,
        )

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
        cleanup_payload = {
            "mentioned_entities": mentioned_entities_dict,
            "referenced_inputs": llm_payload.get("referenced_inputs") or [],
            "parse_warnings": llm_payload.get("parse_warnings") or [],
        }
        _normalize_typed_smiles_fields(cleanup_payload, raw_request_record)
        mentioned_entities_dict = cleanup_payload["mentioned_entities"]
        llm_payload["referenced_inputs"] = cleanup_payload["referenced_inputs"]
        llm_payload["parse_warnings"] = cleanup_payload["parse_warnings"]
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
