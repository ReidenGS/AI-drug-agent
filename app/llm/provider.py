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

        warnings: list[str] = []
        if not target:
            warnings.append("target_or_antigen_text not detected from raw_request_record")
        if not payload:
            warnings.append("payload not detected; downstream readiness will mark gap")
        if not candidate:
            warnings.append("antibody candidate not detected; Step 5 may rely on discovery")

        return {
            "task_intent": {
                "task_type": "adc_design",
                "task_type_confidence": 0.8 if target and payload else 0.4,
                "modality": "ADC",
                "modality_confidence": 0.9,
                "user_goal_summary": user_query.strip() or "ADC design from user input",
            },
            "mentioned_entities": {
                "target_or_antigen_text": target,
                "disease_or_indication_text": None,
                "antibody_candidate_text": candidate,
                "payload_text": payload,
                "linker_text": linker,
            },
            "referenced_inputs": referenced,
            "requested_outputs": [],
            "user_constraints": user_constraints,
            "parse_warnings": warnings,
        }
