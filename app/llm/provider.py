"""LLM provider abstraction.

Two concrete implementations:
- `MockLLMProvider` (this file): deterministic rule-based provider used by
  tests and when no Gemini key is configured. Produces a `structured_query`
  payload from a `raw_request_record` payload without any network call.
- `GeminiProvider` (`gemini_provider.py`): wraps `google-genai`. Real network
  call lives there — never reach into google-genai from API or agent files.
"""

from __future__ import annotations

import re
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class LLMProvider(Protocol):
    name: str
    model: str

    def generate(self, prompt: str, *, system: str | None = None, **kwargs: Any) -> str: ...
    def generate_json(self, prompt: str, *, schema: dict, system: str | None = None) -> dict: ...


# ── identifier / entity detection ───────────────────────────────────────────

# ── selector-stage mocks ────────────────────────────────────────────────────

def _mock_stage1_selection(schema: dict) -> dict:
    """Deterministic Stage 1 selector mock.

    For every catalog entry whose `coarse_input_requirements` intersects the
    context `signals` dict (keys flagged True), emit a selection with a short
    reason. Tests can monkey-patch this to test malformed / hallucinated /
    empty-result paths.
    """
    catalog = schema.get("compact_catalog") or []
    signals = (schema.get("context") or {}).get("signals") or {}
    available = {k for k, v in signals.items() if v}
    selections: list[dict] = []
    for entry in catalog:
        reqs = entry.get("coarse_input_requirements") or []
        if not reqs:
            continue
        if any(req in available for req in reqs):
            selections.append({
                "tool_name": entry["tool_name"],
                "selection_reason": (
                    f"coarse_input {sorted(set(reqs) & available)} satisfied by context"
                ),
                "priority": 1,
                "required_context": sorted(set(reqs) & available),
            })
    return {"selections": selections, "selection_metadata": {"strategy": "mock_signals_match"}}


def _mock_stage2_arguments(schema: dict) -> dict:
    """Deterministic Stage 2 arg construction mock.

    For each schema property, look up the same key in `context.arg_hints`.
    If missing, leave it out — the validator will mark `required` gaps and
    the policy will try its deterministic fallback.
    """
    full_schema = schema.get("full_schema") or {}
    arg_hints = (schema.get("context") or {}).get("arg_hints") or {}
    properties = full_schema.get("properties") or {}
    args: dict = {}
    for name in properties:
        if name in arg_hints:
            args[name] = arg_hints[name]
    return {
        "arguments": args,
        "argument_construction_reason": (
            f"filled {sorted(args.keys())} from context.arg_hints"
        ),
        "missing_fields": [n for n in (full_schema.get("required") or []) if n not in args],
    }


_TARGET_HINTS = (
    "HER2", "EGFR", "TROP2", "BCMA", "CD19", "CD20", "CD22", "CD33", "CD30", "CD79",
    "Nectin-4", "B7-H3", "FOLR1", "MET", "MUC1", "ROR1", "PSMA", "Claudin18.2",
)
_PAYLOAD_HINTS = (
    "MMAE", "MMAF", "DM1", "DM4", "DXd", "SN-38", "PBD", "calicheamicin",
    "duocarmycin", "tubulysin", "amanitin",
)
_LINKER_HINTS = (
    "vc-PAB", "vc", "mc-vc", "GGFG", "valine-citrulline", "valine_citrulline",
    "cleavable", "non-cleavable", "thioether", "hydrazone", "disulfide",
)
# PDB IDs: 4-char, starts with digit. Avoid "1A2B" inside larger words.
_RE_PDB = re.compile(r"(?<![A-Z0-9])([1-9][A-Z0-9]{3})(?![A-Z0-9])", re.IGNORECASE)
# UniProt accessions: simplified canonical pattern (excludes [BJOUXZ] start, etc.)
_RE_UNIPROT = re.compile(
    r"\b([OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9](?:[A-Z][A-Z0-9]{2}[0-9]){1,2})\b"
)
_RE_ZINC = re.compile(r"\b(ZINC[0-9]{4,})\b", re.IGNORECASE)
_RE_CHEMBL = re.compile(r"\b(CHEMBL[0-9]+)\b", re.IGNORECASE)
_RE_DRUGBANK = re.compile(r"\b(DB[0-9]{5})\b")
_RE_PUBCHEM = re.compile(r"\b(?:CID[: _-]?)([0-9]{3,})\b", re.IGNORECASE)
# SMILES heuristic: a token of SMILES-legal characters with at least one
# bracket/bond char and length >= 5. Conservative — avoids matching plain words.
_SMILES_CHARS = r"A-Za-z0-9@+\-\[\]\(\)=#$%/\\.:"
_RE_SMILES_TOKEN = re.compile(rf"(?:^|\s)([{_SMILES_CHARS}]{{5,200}})(?=$|\s)")


def _find_first(text: str, hints: tuple[str, ...]) -> str | None:
    for h in hints:
        if re.search(rf"\b{re.escape(h)}\b", text, flags=re.IGNORECASE):
            return h
    return None


def _looks_like_smiles(token: str) -> bool:
    if len(token) < 5:
        return False
    # Must contain at least one SMILES-only signal char.
    if not re.search(r"[=\(\)\[\]#@]", token):
        return False
    # Avoid pure-word matches like "vc-MMAE" or "PEG-OH": SMILES needs at
    # least one upper-case atom + ring/bond syntax.
    if not re.search(r"[CNOPS]", token):
        return False
    # Reject tokens that are clearly just dash-joined names.
    if re.fullmatch(r"[A-Za-z]+(?:-[A-Za-z]+)+", token):
        return False
    return True


def _detect_referenced_inputs(text: str) -> list[dict]:
    seen: set[tuple[str, str]] = set()
    refs: list[dict] = []

    def _add(id_type: str, value: str) -> None:
        key = (id_type, value.upper())
        if key in seen:
            return
        seen.add(key)
        refs.append({"id_type": id_type, "value": value, "source": "raw_request_text"})

    for m in _RE_PDB.finditer(text):
        _add("pdb_id", m.group(1).upper())
    for m in _RE_UNIPROT.finditer(text):
        _add("uniprot_id", m.group(1).upper())
    for m in _RE_ZINC.finditer(text):
        _add("zinc_id", m.group(1).upper())
    for m in _RE_CHEMBL.finditer(text):
        _add("chembl_id", m.group(1).upper())
    for m in _RE_DRUGBANK.finditer(text):
        _add("drugbank_id", m.group(1))
    for m in _RE_PUBCHEM.finditer(text):
        _add("pubchem_cid", m.group(1))
    for m in _RE_SMILES_TOKEN.finditer(" " + text + " "):
        tok = m.group(1).strip()
        if _looks_like_smiles(tok):
            _add("smiles", tok)
    return refs


class MockLLMProvider:
    """Rule-based provider used for tests / no-API-key dev.

    Contract: given a `prompt` and a `schema` containing a
    `raw_request_record` snapshot, `generate_json` returns the inner
    structured_query payload (without run_id / parsed_at /
    source_raw_request_ref — those are supplied by the agent).
    """

    name = "mock"

    def __init__(self, model: str = "mock-supervisor-v1") -> None:
        self.model = model

    def generate(self, prompt: str, *, system: str | None = None, **kwargs: Any) -> str:
        raise NotImplementedError("MockLLMProvider only implements generate_json for now")

    def generate_json(self, prompt: str, *, schema: dict, system: str | None = None) -> dict:
        # Dispatch on `task` first so the selector can reuse the same
        # provider without colliding with the Supervisor parsing path.
        task = (schema or {}).get("task")
        if task == "tool_selection_stage_1":
            return _mock_stage1_selection(schema)
        if task == "tool_selection_stage_2":
            return _mock_stage2_arguments(schema)

        raw = (schema or {}).get("raw_request_record") or {}
        ctx = raw.get("user_provided_context") or {}
        user_query = raw.get("raw_user_query") or ""
        uploaded_files = raw.get("uploaded_files") or []
        haystack = " ".join(
            [
                user_query,
                ctx.get("target_or_antigen_text") or "",
                ctx.get("candidate_text") or "",
                ctx.get("payload_linker_text") or "",
                ctx.get("constraints_text") or "",
                ctx.get("notes") or "",
            ]
        )

        target = ctx.get("target_or_antigen_text") or _find_first(haystack, _TARGET_HINTS)
        candidate = ctx.get("candidate_text")
        payload = _find_first(haystack, _PAYLOAD_HINTS)
        linker = _find_first(haystack, _LINKER_HINTS)

        # If the user gave a free-form payload_linker_text but it didn't match
        # the hint list, surface it as the payload string anyway. We do NOT
        # invent identifiers.
        if not payload and ctx.get("payload_linker_text"):
            payload = ctx["payload_linker_text"]

        referenced = _detect_referenced_inputs(haystack)
        referenced.extend(_uploaded_file_refs(uploaded_files))

        # Constraint preservation: keep the user's explicit constraints
        # verbatim. Don't try to interpret numeric tolerances here.
        user_constraints: list[dict] = []
        if ctx.get("constraints_text"):
            user_constraints.append(
                {
                    "constraint_text": ctx["constraints_text"],
                    "source": "user_provided_context.constraints_text",
                }
            )

        # Aliases + decompositions (batch 5). Mock detects them
        # deterministically off the haystack text; never invents components
        # that aren't part of a known ADC's canonical recipe.
        normalized_entities, decompositions, mentioned_drugs = _detect_aliases(
            haystack, target=target, candidate=candidate, payload=payload,
            linker=linker,
        )

        # Mentioned candidate/payload from decomposition can fill in gaps
        # the surface text didn't reveal. Mark them inferred via the
        # normalized_entities records; mentioned_entities still mirrors
        # only what the user actually wrote.
        for decomp in decompositions:
            for comp in decomp.get("components") or []:
                if comp.get("role") == "antibody" and not candidate:
                    candidate = None  # leave mentioned_entities literal
                if comp.get("role") == "payload" and not payload:
                    # Only adopt the canonical payload name from the
                    # decomposition for the purpose of intent ranking;
                    # the literal payload_text field stays None.
                    pass

        # Crude ADC-vs-not heuristic for the mock provider. We deliberately do
        # NOT default to `adc_design` when the request has no ADC signal —
        # that would let the mock claim higher confidence than it can support.
        modality, task_type, modality_conf, task_conf, non_adc = _classify_intent(
            user_query, target, payload, ctx, mentioned_drugs=mentioned_drugs,
        )

        primary_intent, secondary_intents, intent_conf, requested_outputs = (
            _classify_primary_intent(
                user_query=user_query,
                ctx=ctx,
                target=target,
                candidate=candidate,
                payload=payload,
                referenced=referenced,
                mentioned_drugs=mentioned_drugs,
                non_adc=non_adc,
            )
        )

        warnings: list[str] = []
        if not target:
            warnings.append("target_or_antigen_text not detected from raw_request_record")
        if not payload:
            warnings.append("payload not detected; downstream readiness will mark gap")
        if not candidate:
            warnings.append("antibody candidate not detected; Step 5 may rely on discovery")
        if non_adc:
            warnings.append("request does not look like an ADC design task")

        clarifications = _clarification_questions(
            primary_intent=primary_intent,
            target=target,
            candidate=candidate,
            payload=payload,
            linker=linker,
            mentioned_drugs=mentioned_drugs,
            referenced=referenced,
            haystack=haystack,
        )

        return {
            "task_intent": {
                "task_type": task_type,
                "task_type_confidence": task_conf,
                "modality": modality,
                "modality_confidence": modality_conf,
                "user_goal_summary": user_query.strip() or "ADC design from user input",
                "primary_intent": primary_intent,
                "primary_intent_confidence": intent_conf,
                "secondary_intents": secondary_intents,
            },
            "mentioned_entities": {
                "target_or_antigen_text": target,
                "disease_or_indication_text": None,
                "antibody_candidate_text": candidate,
                "payload_text": payload,
                "linker_text": linker,
            },
            "referenced_inputs": referenced,
            "requested_outputs": requested_outputs,
            "user_constraints": user_constraints,
            "parse_warnings": warnings,
            "normalized_entities": normalized_entities,
            "entity_decompositions": decompositions,
            "clarification_questions": clarifications,
        }


# ── helpers reused by MockLLMProvider.generate_json above ──────────────────


_ADC_KEYWORDS = (
    "adc", "antibody-drug conjugate", "antibody drug conjugate",
    "payload", "linker", "conjugate",
)


def _classify_intent(
    user_query: str,
    target: str | None,
    payload: str | None,
    ctx: dict,
    mentioned_drugs: list[str] | None = None,
) -> tuple[str, str, float, float, bool]:
    """Return (modality, task_type, modality_conf, task_conf, non_adc).

    Recognized ADC signals include both the literal `adc` keywords and
    the presence of any known ADC drug name (`T-DM1`, `T-DXd`, Enhertu, …)
    which always imply ADC modality even when the user didn't write the
    string "ADC".
    """
    text = (user_query or "").lower() + " " + " ".join(
        str(v or "").lower() for v in ctx.values()
    )
    adc_signal = any(k in text for k in _ADC_KEYWORDS) or bool(
        mentioned_drugs or []
    )
    if adc_signal:
        modality = "ADC"
        task_type = "adc_design"
        modality_conf = 0.9
        task_conf = 0.8 if target and payload else 0.4
        return modality, task_type, modality_conf, task_conf, False
    # No ADC signal at all → keep modality "unknown" so Step 3 can flag it.
    return "unknown", "unknown", 0.0, 0.0, True


# ── batch 5: normalization + decomposition + intent classification ────────


# Canonical resolution table for the aliases the professor listed. Each
# entry: alias → (canonical_name, canonical_id, canonical_id_source,
# entity_type). `canonical_id_source` may be None when no authoritative
# identifier is available.
_NORMALIZATION_TABLE: dict[str, dict] = {
    "her2": {
        "canonical_name": "ERBB2",
        "canonical_id": "HGNC:3430",
        "canonical_id_source": "HGNC",
        "entity_type": "target_or_antigen",
    },
    "erbb2": {
        "canonical_name": "ERBB2",
        "canonical_id": "HGNC:3430",
        "canonical_id_source": "HGNC",
        "entity_type": "target_or_antigen",
    },
    "trop2": {
        "canonical_name": "TACSTD2",
        "canonical_id": "HGNC:11530",
        "canonical_id_source": "HGNC",
        "entity_type": "target_or_antigen",
    },
    "tacstd2": {
        "canonical_name": "TACSTD2",
        "canonical_id": "HGNC:11530",
        "canonical_id_source": "HGNC",
        "entity_type": "target_or_antigen",
    },
    "cldn18.2": {
        "canonical_name": "CLDN18 isoform 2",
        "canonical_id": "HGNC:2039",
        "canonical_id_source": "HGNC",
        "entity_type": "target_or_antigen",
    },
    "claudin18.2": {
        "canonical_name": "CLDN18 isoform 2",
        "canonical_id": "HGNC:2039",
        "canonical_id_source": "HGNC",
        "entity_type": "target_or_antigen",
    },
    "mmae": {
        "canonical_name": "monomethyl auristatin E",
        "canonical_id": None,
        "canonical_id_source": None,
        "entity_type": "payload",
    },
    "dxd": {
        "canonical_name": "topoisomerase I inhibitor (DXd payload family)",
        "canonical_id": None,
        "canonical_id_source": None,
        "entity_type": "payload",
    },
    "dm1": {
        "canonical_name": "emtansine",
        "canonical_id": None,
        "canonical_id_source": None,
        "entity_type": "payload",
    },
    "emtansine": {
        "canonical_name": "emtansine",
        "canonical_id": None,
        "canonical_id_source": None,
        "entity_type": "payload",
    },
    "deruxtecan": {
        "canonical_name": "deruxtecan",
        "canonical_id": None,
        "canonical_id_source": None,
        "entity_type": "linker",
    },
    "trastuzumab": {
        "canonical_name": "trastuzumab",
        "canonical_id": "DB00072",
        "canonical_id_source": "DrugBank",
        "entity_type": "antibody",
    },
    "t-dm1": {
        "canonical_name": "ado-trastuzumab emtansine",
        "canonical_id": "DB05773",
        "canonical_id_source": "DrugBank",
        "entity_type": "drug",
    },
    "ado-trastuzumab emtansine": {
        "canonical_name": "ado-trastuzumab emtansine",
        "canonical_id": "DB05773",
        "canonical_id_source": "DrugBank",
        "entity_type": "drug",
    },
    "t-dxd": {
        "canonical_name": "trastuzumab deruxtecan",
        "canonical_id": "DB14962",
        "canonical_id_source": "DrugBank",
        "entity_type": "drug",
    },
    "enhertu": {
        "canonical_name": "trastuzumab deruxtecan",
        "canonical_id": "DB14962",
        "canonical_id_source": "DrugBank",
        "entity_type": "drug",
    },
    "trastuzumab deruxtecan": {
        "canonical_name": "trastuzumab deruxtecan",
        "canonical_id": "DB14962",
        "canonical_id_source": "DrugBank",
        "entity_type": "drug",
    },
    "vc-mmae": {
        # decomposed below; this entry exists so the alias is also
        # surfaced in normalized_entities for traceability.
        "canonical_name": "vc-MMAE (valine-citrulline linker + MMAE)",
        "canonical_id": None,
        "canonical_id_source": None,
        "entity_type": "linker_payload",
    },
}


# Multi-component ADC decompositions. Components are emitted in the
# canonical-recipe order: antibody, linker, payload. Each component is
# `inferred=True` by default — callers that detect the user explicitly
# wrote the component can override.
_DECOMPOSITION_TABLE: dict[str, dict] = {
    "t-dm1": {
        "canonical_name": "ado-trastuzumab emtansine",
        "components": [
            {"role": "antibody", "canonical_name": "trastuzumab",
             "canonical_id": "DB00072", "canonical_id_source": "DrugBank"},
            {"role": "payload", "canonical_name": "emtansine (DM1)",
             "canonical_id": None, "canonical_id_source": None},
        ],
    },
    "t-dxd": {
        "canonical_name": "trastuzumab deruxtecan",
        "components": [
            {"role": "antibody", "canonical_name": "trastuzumab",
             "canonical_id": "DB00072", "canonical_id_source": "DrugBank"},
            {"role": "linker_payload", "canonical_name": "deruxtecan",
             "canonical_id": None, "canonical_id_source": None},
            {"role": "payload", "canonical_name":
                "DXd / topoisomerase I inhibitor",
             "canonical_id": None, "canonical_id_source": None},
        ],
    },
    "enhertu": {  # same recipe as T-DXd
        "canonical_name": "trastuzumab deruxtecan",
        "components": [
            {"role": "antibody", "canonical_name": "trastuzumab",
             "canonical_id": "DB00072", "canonical_id_source": "DrugBank"},
            {"role": "linker_payload", "canonical_name": "deruxtecan",
             "canonical_id": None, "canonical_id_source": None},
            {"role": "payload", "canonical_name":
                "DXd / topoisomerase I inhibitor",
             "canonical_id": None, "canonical_id_source": None},
        ],
    },
    "vc-mmae": {
        "canonical_name": "vc-MMAE (valine-citrulline linker + MMAE)",
        "components": [
            {"role": "linker", "canonical_name":
                "valine-citrulline (vc-PABC) linker",
             "canonical_id": None, "canonical_id_source": None},
            # MMAE payload is the explicit component (user wrote it).
            {"role": "payload", "canonical_name":
                "monomethyl auristatin E",
             "canonical_id": None, "canonical_id_source": None,
             "explicit": True},
        ],
    },
}


def _detect_aliases(
    haystack: str,
    *,
    target: str | None,
    candidate: str | None,
    payload: str | None,
    linker: str | None,
) -> tuple[list[dict], list[dict], list[str]]:
    """Return (normalized_entities, entity_decompositions, mentioned_drug_keys).

    Deterministic and conservative — only fires for alias keys that appear
    as standalone tokens (case-insensitive) in the haystack. Never invents
    canonical components that aren't in the recipe table.

    Component-explicitness rule (batch 5 follow-up): a decomposed
    component is `inferred=False` ONLY when the user wrote the component
    OUTSIDE a whole-ADC alias span. The whole ADC alias itself does not
    count — e.g. for `T-DXd`, the `DXd` substring inside the alias must
    NOT mark the DXd payload component as explicit. We scrub every
    decomposition-trigger alias from the haystack before measuring
    component explicitness. The recipe may still set `explicit: True`
    per component for non-whole-ADC aliases where the alias literally
    contains the component name as a meaningful token (e.g.
    vc-MMAE → MMAE payload).
    """
    lowered = " " + haystack.lower() + " "

    # Scrub multi-component alias spans BEFORE per-component explicitness
    # checks; matches inside the alias span don't count as user mentions.
    scrubbed_for_components = lowered
    for decomp_alias in _DECOMPOSITION_TABLE:
        scrubbed_for_components = re.sub(
            rf"(?<![A-Za-z0-9]){re.escape(decomp_alias)}(?![A-Za-z0-9])",
            lambda m: " " * len(m.group(0)),
            scrubbed_for_components,
        )

    seen_aliases: set[str] = set()
    norm_entries: list[dict] = []
    decomp_entries: list[dict] = []
    mentioned_drug_keys: list[str] = []

    explicit_text_lower = " " + " ".join(
        str(x or "") for x in (target, candidate, payload, linker)
    ).lower() + " "

    for alias, meta in _NORMALIZATION_TABLE.items():
        # token-style match: surrounded by non-alphanumeric chars.
        pattern = rf"(?<![A-Za-z0-9]){re.escape(alias)}(?![A-Za-z0-9])"
        m = re.search(pattern, lowered)
        if not m:
            continue
        if alias in seen_aliases:
            continue
        seen_aliases.add(alias)
        canonical = meta["canonical_name"]
        # If the user already wrote the canonical name explicitly, mark
        # explicit. Otherwise, the parser inferred the resolution.
        explicit = bool(
            re.search(
                rf"(?<![A-Za-z0-9]){re.escape(canonical.lower())}(?![A-Za-z0-9])",
                explicit_text_lower,
            )
        ) or (canonical.lower() == alias.lower())
        norm_entries.append(
            {
                "original_text": _original_span(haystack, alias),
                "canonical_name": canonical,
                "canonical_id": meta.get("canonical_id"),
                "canonical_id_source": meta.get("canonical_id_source"),
                "entity_type": meta.get("entity_type") or "other",
                "explicit_or_inferred": "explicit" if explicit else "inferred",
                "confidence": 0.9 if explicit else 0.7,
            }
        )

        # Emit decompositions only for known ADC drugs / multi-component
        # aliases.
        if alias in _DECOMPOSITION_TABLE:
            recipe = _DECOMPOSITION_TABLE[alias]
            comp_entries = []
            for c in recipe["components"]:
                recipe_explicit = alias == "vc-mmae" and bool(c.get("explicit"))
                explicit_comp = recipe_explicit or _component_in_text(
                    c["canonical_name"], scrubbed_for_components
                )
                comp_entries.append(
                    {
                        "role": c["role"],
                        "canonical_name": c["canonical_name"],
                        "canonical_id": c.get("canonical_id"),
                        "canonical_id_source": c.get("canonical_id_source"),
                        "inferred": not explicit_comp,
                    }
                )
            decomp_entries.append(
                {
                    "original_text": _original_span(haystack, alias),
                    "canonical_name": recipe["canonical_name"],
                    "components": comp_entries,
                }
            )
            mentioned_drug_keys.append(alias)
    return norm_entries, decomp_entries, mentioned_drug_keys


def _component_in_text(canonical: str, lowered_haystack: str) -> bool:
    # Treat a component "explicit" only if the canonical name (or its
    # first token before " /" / "(" / ",") appears as a standalone token
    # in the haystack.
    main = re.split(r"[ /(,]", canonical.lower(), maxsplit=1)[0].strip()
    if not main:
        return False
    return bool(
        re.search(rf"(?<![A-Za-z0-9]){re.escape(main)}(?![A-Za-z0-9])", lowered_haystack)
    )


def _original_span(haystack: str, alias: str) -> str:
    """Return the original-case span of `alias` from `haystack`."""
    m = re.search(rf"(?i)(?<![A-Za-z0-9]){re.escape(alias)}(?![A-Za-z0-9])", haystack)
    return m.group(0) if m else alias


# ── primary intent classification ─────────────────────────────────────────


def _classify_primary_intent(
    *,
    user_query: str,
    ctx: dict,
    target: str | None,
    candidate: str | None,
    payload: str | None,
    referenced: list[dict],
    mentioned_drugs: list[str],
    non_adc: bool,
) -> tuple[str, list[str], float, list[str]]:
    """Deterministic mock classifier for primary/secondary intents.

    Returns (primary_intent, secondary_intents, confidence, requested_outputs).
    Keeps the heuristic explicit so tests are easy to reason about — every
    keyword check below mirrors a professor benchmark example.
    """
    text = (user_query or "").lower() + " " + " ".join(
        str(v or "").lower() for v in (ctx or {}).values()
    )
    ref_id_types = {r.get("id_type") for r in referenced}
    has_pdb = "pdb_id" in ref_id_types
    has_zinc = "zinc_id" in ref_id_types
    has_chembl = "chembl_id" in ref_id_types

    primary = "unclear_or_needs_clarification"
    secondary: list[str] = []
    confidence = 0.0
    outputs: list[str] = []

    # Strong cues: known multi-component ADC drug → existing_adc_evaluation
    if mentioned_drugs:
        primary = "existing_adc_evaluation"
        secondary.append("literature_review")
        secondary.append("developability_assessment")
        outputs.extend([
            "evidence_summary", "developability_summary",
            "data_gap_summary", "case_study_summary", "report",
        ])
        # comparison cues → also literature_review_summary
        if any(
            kw in text for kw in (" vs ", "versus", "compare", "comparison")
        ):
            outputs.append("literature_review_summary")
        confidence = 0.85

    # Patent / IP cues
    elif any(
        kw in text for kw in (
            "patent", "ip ", "prior art", "freedom to operate", "fto",
        )
    ):
        primary = "patent_ip_review"
        secondary.append("literature_review")
        outputs.extend([
            "patent_or_ip_summary", "data_gap_summary", "report",
        ])
        confidence = 0.8

    # Structure analysis cues (PDB id, "structure", "validate", "interface")
    elif has_pdb or any(
        kw in text for kw in (
            "structure analysis", "validate the structure", "structure of",
            "structural analysis", "interface analysis", "binding mode",
        )
    ):
        primary = "structure_analysis"
        if has_zinc or has_chembl or "screen" in text or "library" in text:
            secondary.append("compound_screening")
            outputs.append("compound_screening_results")
        outputs.extend([
            "structure_validation_report", "data_gap_summary", "report",
        ])
        confidence = 0.75

    # Compound screening cues (ZINC / ChEMBL / "screen")
    elif has_zinc or has_chembl or any(
        kw in text for kw in (
            "screen ", "screening", "compound library", "shortlist compounds",
            "rank compounds",
        )
    ):
        primary = "compound_screening"
        secondary.append("developability_assessment")
        secondary.append("literature_review")
        outputs.extend([
            "compound_screening_results", "developability_summary",
            "data_gap_summary", "report",
        ])
        confidence = 0.7

    # Literature-only cues
    elif any(
        kw in text for kw in (
            "literature", "papers", "review the literature", "review papers",
            "summarize literature",
        )
    ):
        primary = "literature_review"
        outputs.extend([
            "literature_review_summary", "evidence_summary", "report",
        ])
        confidence = 0.75

    # Developability cues
    elif any(
        kw in text for kw in (
            "developability", "manufacturability", "aggregation", "stability"
        )
    ):
        primary = "developability_assessment"
        outputs.extend(["developability_summary", "report"])
        confidence = 0.7

    # Optimization cues
    elif any(
        kw in text for kw in (
            "optimize", "improve", "optimization", "tune ", "iterate",
        )
    ):
        primary = "optimization"
        outputs.extend(["optimization_suggestions", "report"])
        confidence = 0.65

    # New ADC design fallback (target + payload + linker / antibody all hint)
    elif target and (payload or "design" in text or "build" in text) and not non_adc:
        primary = "new_adc_design"
        secondary.append("structure_analysis")
        secondary.append("developability_assessment")
        outputs.extend([
            "ranked_candidates", "developability_summary",
            "data_gap_summary", "report",
        ])
        confidence = 0.7

    elif non_adc:
        primary = "unclear_or_needs_clarification"
        confidence = 0.2

    # Deduplicate while preserving order.
    secondary_unique: list[str] = []
    seen_s: set[str] = set()
    for s in secondary:
        if s and s != primary and s not in seen_s:
            secondary_unique.append(s)
            seen_s.add(s)
    outputs_unique: list[str] = []
    seen_o: set[str] = set()
    for o in outputs:
        if o and o not in seen_o:
            outputs_unique.append(o)
            seen_o.add(o)
    return primary, secondary_unique, confidence, outputs_unique


def _clarification_questions(
    *,
    primary_intent: str,
    target: str | None,
    candidate: str | None,
    payload: str | None,
    linker: str | None,
    mentioned_drugs: list[str],
    referenced: list[dict],
    haystack: str,
) -> list[str]:
    """User-facing clarification questions, distinct from parse_warnings.

    Each question maps to a benchmark scenario from the professor's
    feedback. Keep them short and answerable; the operator should be
    able to reply with one line.
    """
    questions: list[str] = []
    ref_id_types = {r.get("id_type") for r in referenced}

    if primary_intent == "new_adc_design":
        if not candidate:
            questions.append(
                "Which antibody backbone should we use for this new ADC design?"
            )
        if not linker:
            questions.append(
                "Which linker chemistry should we assume for this payload?"
            )

    if primary_intent == "structure_analysis":
        if "pdb_id" in ref_id_types and (
            "zinc_id" in ref_id_types or "chembl_id" in ref_id_types
        ):
            questions.append(
                "Is this a general HER2 / target compound screening, or a "
                "payload / linker workflow inside an ADC?"
            )

    if primary_intent == "literature_review":
        # Literature-only path with a payload mention (MMAE / trastuzumab)
        # but no explicit ADC target — ask whether the user means HER2 ADC
        # literature specifically.
        if "trastuzumab" in haystack.lower() and "mmae" in haystack.lower() and (
            not target or target.upper() not in {"HER2", "ERBB2"}
        ):
            questions.append(
                "Did you mean the HER2 ADC literature (trastuzumab + MMAE), "
                "or general antibody / payload literature?"
            )

    if primary_intent == "patent_ip_review" and not mentioned_drugs:
        # CLDN18.2 + deruxtecan-like payload path: surface a gap question
        # about which patent scope to search.
        if "cldn18" in haystack.lower() or "claudin18" in haystack.lower():
            questions.append(
                "Should we search patents for the deruxtecan payload family, "
                "the CLDN18.2 antibody backbone, or both?"
            )

    if primary_intent == "compound_screening" and not (
        candidate or "trastuzumab" in haystack.lower()
    ):
        questions.append(
            "These compounds have no antibody / linker context yet — should we "
            "treat this as standalone screening or as ADC payload candidates?"
        )

    if primary_intent == "unclear_or_needs_clarification":
        questions.append(
            "Could you clarify the workflow? Options include: design a new "
            "ADC, evaluate an existing ADC, run literature / patent review, "
            "or screen compounds."
        )

    # Dedup.
    out: list[str] = []
    seen: set[str] = set()
    for q in questions:
        if q not in seen:
            seen.add(q)
            out.append(q)
    return out


def _uploaded_file_refs(files: list) -> list[dict]:
    out: list[dict] = []
    for f in files or []:
        if not isinstance(f, dict):
            continue
        fid = f.get("file_id")
        if not fid:
            continue
        entry: dict = {
            "id_type": "uploaded_file",
            "value": fid,
            "source": "uploaded_files",
        }
        if f.get("original_filename"):
            entry["filename"] = f["original_filename"]
        out.append(entry)
    return out
