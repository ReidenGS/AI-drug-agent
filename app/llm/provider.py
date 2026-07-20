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

from .json_task_validation import (
    extract_protein_variant_tokens,
    looks_like_masked_prompt_sequence,
    requests_protein_generation,
)


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


def _mock_stage1_multi_lane(schema: dict) -> dict:
    """Deterministic per-candidate multi-lane Stage 1 selector mock.

    For each lane in `schema["lanes"]`, picks every complementary tool from
    the intersection of the catalog and the lane's `allowed_tools` whose
    `coarse_input_requirements` intersect the lane's `signals`. Returns a
    flat `selections` list tagged with `lane_type`.
    """
    catalog = schema.get("compact_catalog") or []
    name_to_entry = {
        e.get("tool_name"): e for e in catalog
        if isinstance(e, dict) and e.get("tool_name")
    }
    lanes = schema.get("lanes") or []
    selections: list[dict] = []
    for lane in lanes:
        lane_type = lane.get("lane_type")
        allowed = [t for t in (lane.get("allowed_tools") or []) if isinstance(t, str)]
        signals = lane.get("signals") or {}
        available = {k for k, v in signals.items() if v}
        picks: list[dict] = []
        for tool_name in allowed:
            entry = name_to_entry.get(tool_name)
            if not entry:
                continue
            reqs = entry.get("coarse_input_requirements") or []
            if reqs and not any(r in available for r in reqs):
                continue
            picks.append({
                "lane_type": lane_type,
                "tool_name": tool_name,
                "selection_reason": (
                    f"coarse_input {sorted(set(reqs) & available)} satisfied by lane signals"
                    if reqs else "lane fallback (no coarse input requirements)"
                ),
                "priority": 1,
                "required_context": sorted(set(reqs) & available),
            })
        selections.extend(picks)
    return {
        "selections": selections,
        "selection_metadata": {"strategy": "mock_multi_lane_signals_match"},
    }


def _mock_stage2_multi_tool(schema: dict) -> dict:
    """Deterministic per-candidate multi-tool Stage 2 arg-construction mock.

    For each tool in `schema["tools"]`, copies any schema-required property
    that exists in that tool's per-lane `arg_hints`. Missing required
    fields surface as `missing_fields` per tool; the caller then tries
    deterministic mapping or marks the plan skipped.
    """
    tools_in = schema.get("tools") or []
    tools_out: list[dict] = []
    for t in tools_in:
        if not isinstance(t, dict):
            continue
        full_schema = t.get("full_schema") or {}
        arg_hints = t.get("arg_hints") or {}
        properties = full_schema.get("properties") or {}
        args: dict = {}
        for name in properties:
            if name in arg_hints:
                args[name] = arg_hints[name]
        tools_out.append({
            "lane_type": t.get("lane_type"),
            "tool_name": t.get("tool_name"),
            "arguments": args,
            "argument_construction_reason": (
                f"filled {sorted(args.keys())} from per-lane arg_hints"
            ),
            "missing_fields": [
                n for n in (full_schema.get("required") or []) if n not in args
            ],
        })
    return {"tools": tools_out}


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


def _mock_step6_schema_mapping_stage1(schema: dict) -> dict:
    catalog = schema.get("compact_catalog") or []
    return {
        "selections": [
            {
                "tool_name": entry.get("tool_name"),
                "selection_reason": "mock selected disclosed Step 6 tool",
                "priority": 1,
            }
            for entry in catalog
            if isinstance(entry, dict) and entry.get("tool_name")
        ],
        "selection_metadata": {"strategy": "mock_select_disclosed_catalog"},
    }


def _mock_step6_schema_mapping_stage2(schema: dict) -> dict:
    fields = schema.get("candidate_available_fields") or []
    tools = schema.get("tools") or []

    def match_field(arg_name: str) -> dict | None:
        lowered = arg_name.lower()
        for field in fields:
            if not isinstance(field, dict):
                continue
            value_kind = field.get("value_kind")
            id_type = field.get("id_type")
            field_type = field.get("field_type")
            if lowered in {"smiles", "canonical_smiles"} and value_kind == "smiles":
                return field
            if lowered in {"pdb_id", "pdb"} and id_type == "pdb_id":
                return field
            if lowered in {"pdb_id_or_path", "structure_file", "structure_ref"} and (
                id_type == "pdb_id" or value_kind == "structure_ref"
            ):
                return field
            if lowered in {"sequence", "protein_sequence"} and (
                field_type == "protein_sequence" and value_kind == "protein_sequence"
            ):
                return field
            if lowered in {"accession", "uniprot_id", "uniprot_accession"} and id_type == "uniprot_id":
                return field
            if lowered in {"molecule_chembl_id", "chembl_id"} and id_type == "chembl_id":
                return field
        return None

    out: list[dict] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        schema_obj = tool.get("full_schema") or {}
        required = list(schema_obj.get("required") or [])
        properties = schema_obj.get("properties") or {}
        mapping: dict[str, str] = {}
        missing: list[str] = []
        for arg_name in required:
            field = match_field(str(arg_name))
            if field is None:
                missing.append(str(arg_name))
            else:
                mapping[str(arg_name)] = field["field_ref"]
        if not required:
            for arg_name in properties:
                field = match_field(str(arg_name))
                if field is not None:
                    mapping[str(arg_name)] = field["field_ref"]
        out.append({
            "tool_name": tool.get("tool_name"),
            "can_invoke": not missing and bool(mapping),
            "argument_mapping": mapping,
            "missing_required_fields": missing,
            "argument_mapping_reason": "mock mapped schema args to available field refs",
        })
    return {"tools": out}


def _mock_step14_patent_tool_selection(schema: dict) -> dict:
    """Deterministic Step 14 single-stage patent planner mock.

    For each input ref + tool whose ``acceptable_supports`` intersects the
    ref's ``supports_tool_args``, emits ONE plan mapping the tool's official
    schema arg (via the catalog's ``supports_to_schema_arg``) to that
    ``input_ref_id``. It works purely from the catalog + input-ref metadata —
    never a resolved value (the request carries none). ``can_invoke`` is true
    because the emitted mapping fills the tool's identity arg; antibody refs are
    still emitted so the planner validator applies the scope gate.
    """
    catalog = schema.get("tool_catalog") or []
    input_refs = schema.get("input_refs") or []
    plans: list[dict] = []
    for ref in input_refs:
        if not isinstance(ref, dict):
            continue
        supports = [str(s) for s in (ref.get("supports_tool_args") or [])]
        supports_lower = {s.lower() for s in supports}
        if not supports_lower:
            continue
        for tool in catalog:
            if not isinstance(tool, dict):
                continue
            acceptable_order = [str(s) for s in (tool.get("acceptable_supports") or [])]
            s2a = {
                str(k).lower(): v
                for k, v in (tool.get("supports_to_schema_arg") or {}).items()
            }
            # Deterministic: first acceptable token (in catalog order) the ref
            # carries → its official schema arg.
            token = next(
                (t for t in acceptable_order if t.lower() in supports_lower), None
            )
            if token is None:
                continue
            schema_arg = s2a.get(token.lower())
            if not schema_arg:
                continue
            plans.append({
                "tool_name": tool.get("tool_name"),
                "can_invoke": True,
                "argument_mappings": [
                    {"schema_arg": schema_arg, "input_ref_id": ref.get("ref_id")}
                ],
                "argument_literals": [],
                "missing_required_args": [],
                "selection_reason": f"ref {ref.get('ref_id')} fills {schema_arg}",
            })
    return {"tool_plans": plans}


def _mock_patent_evidence_tool_selection(schema: dict) -> dict:
    """Catalog/scope/ref-driven unified planner; no hard-coded tool names."""
    catalog = schema.get("tool_catalog") or []
    input_refs = schema.get("input_refs") or []
    scope = schema.get("search_scope") or {}
    requested_lane_order = list(scope.get("requested_lanes") or [])
    requested_lanes = set(requested_lane_order)
    allowed_roles = set(scope.get("allowed_roles") or [])
    antibody_allowed = bool(scope.get("antibody_search_allowed"))
    plans: list[dict] = []
    for ref in input_refs:
        if not isinstance(ref, dict):
            continue
        role = str(ref.get("role") or "")
        if role == "antibody" and not antibody_allowed:
            continue
        if role not in allowed_roles:
            continue
        supports = {str(v).lower() for v in ref.get("supports_tool_args") or []}
        for tool in catalog:
            if not isinstance(tool, dict) or tool.get("search_lane") not in requested_lanes:
                continue
            if not (tool.get("runtime_availability") or {}).get("can_execute", False):
                continue
            mapping = {
                str(k).lower(): str(v)
                for k, v in (tool.get("supports_to_schema_arg") or {}).items()
            }
            token = next(
                (str(v) for v in tool.get("acceptable_supports") or [] if str(v).lower() in supports),
                None,
            )
            if token is None or token.lower() not in mapping:
                continue
            schema_arg = mapping[token.lower()]
            allowed_arg_roles = set(
                (tool.get("schema_arg_allowed_ref_roles") or {}).get(schema_arg)
                or []
            )
            if role not in allowed_arg_roles:
                continue
            plans.append(
                {
                    "tool_name": tool.get("tool_name"),
                    "can_invoke": True,
                    "argument_mappings": [
                        {
                            "schema_arg": schema_arg,
                            "input_ref_id": ref.get("ref_id"),
                        }
                    ],
                    "argument_literals": [],
                    "missing_required_args": [],
                    "selection_reason": "mock mapped supplied catalog to supplied ref",
                }
            )
    planned_lanes = {
        tool.get("search_lane")
        for plan in plans
        for tool in catalog
        if isinstance(tool, dict) and tool.get("tool_name") == plan.get("tool_name")
    }
    return {
        "lane_assessments": [
            {
                "search_lane": lane,
                "status": "planned" if lane in planned_lanes else "missing_inputs",
                "reason": (
                    "supplied refs support at least one runtime-available catalog tool"
                    if lane in planned_lanes
                    else "no supplied ref supports a runtime-available catalog tool"
                ),
            }
            for lane in requested_lane_order
        ],
        "tool_plans": plans,
    }


def _mock_step9_tool_selection_stage1(schema: dict) -> dict:
    catalog = schema.get("compact_catalog") or []
    return {
        "selections": [
            {
                "tool_name": entry.get("tool_name"),
                "lane_type": entry.get("lane_type"),
                "selection_reason": (
                    "mock selected Step 9 tool from the supplied active catalog"
                ),
            }
            for entry in catalog
            if isinstance(entry, dict) and entry.get("tool_name") and entry.get("lane_type")
        ]
    }


def _mock_step9_tool_schema_mapping_stage2(schema: dict) -> dict:
    fields = schema.get("step9_input_fields") or []
    tools = schema.get("tools") or []

    def match(arg_name: str) -> dict | None:
        lowered = arg_name.lower()
        for field in fields:
            if not isinstance(field, dict):
                continue
            supports = {str(a).lower() for a in (field.get("supports_tool_args") or [])}
            if lowered in supports:
                return field
        return None

    def literal(arg_name: str, full_schema: dict) -> Any | None:
        prop = (full_schema.get("properties") or {}).get(arg_name)
        if not isinstance(prop, dict):
            return None
        if "const" in prop:
            return prop.get("const")
        enum_values = prop.get("enum")
        if isinstance(enum_values, list) and len(enum_values) == 1:
            return enum_values[0]
        if "default" in prop:
            return prop.get("default")
        return None

    out: list[dict] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        full_schema = tool.get("full_schema") or {}
        required = [str(arg) for arg in (tool.get("required_fields") or full_schema.get("required") or [])]
        mappings: list[dict] = []
        literals: list[dict] = []
        missing: list[str] = []
        for arg in required:
            field = match(arg)
            if field is not None:
                mappings.append({"schema_arg": arg, "field_ref": field["field_ref"]})
                continue
            lit = literal(arg, full_schema)
            if lit is not None:
                literals.append({"schema_arg": arg, "literal_value": lit})
                continue
            missing.append(arg)
        out.append({
            "tool_name": tool.get("tool_name"),
            "lane_type": tool.get("lane_type"),
            "can_invoke": not missing,
            "argument_mappings": mappings,
            "argument_literals": literals,
            "missing_required_fields": missing,
            "skip_reason": "" if not missing else "missing_required_fields",
            "argument_mapping_reason": "mock mapped Step 9 schema args to available field refs",
        })
    return {"tools": out}


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
    pubchem_matches = list(_RE_PUBCHEM.finditer(text))
    pubchem_spans = [match.span() for match in pubchem_matches]

    def _add(id_type: str, value: str, *, source: str = "raw_request_text") -> None:
        key = (id_type, value.upper())
        if key in seen:
            return
        seen.add(key)
        refs.append({"id_type": id_type, "value": value, "source": source})

    for m in _RE_PDB.finditer(text):
        pdb_start, pdb_end = m.span(1)
        if any(
            pubchem_start < pdb_end and pdb_start < pubchem_end
            for pubchem_start, pubchem_end in pubchem_spans
        ):
            continue
        _add("pdb_id", m.group(1).upper())
    for m in _RE_UNIPROT.finditer(text):
        _add("uniprot_id", m.group(1).upper())
    # Protein point-mutation / variant (V777L, p.V777L). Structured as the
    # stable id_type="variant" so Step 5 / Step 9 (AlphaMissense / DynaMut2 /
    # ESM variant tools) can consume it without re-parsing the raw query.
    for token in extract_protein_variant_tokens(text):
        _add("variant", token, source="user")
    for m in _RE_ZINC.finditer(text):
        _add("zinc_id", m.group(1).upper())
    for m in _RE_CHEMBL.finditer(text):
        _add("chembl_id", m.group(1).upper())
    for m in _RE_DRUGBANK.finditer(text):
        _add("drugbank_id", m.group(1))
    for m in pubchem_matches:
        _add("pubchem_cid", m.group(1))
    for m in _RE_SMILES_TOKEN.finditer(" " + text + " "):
        tok = m.group(1).strip()
        if _looks_like_smiles(tok):
            _add("smiles", tok)
    return refs


def _mock_orchestrator_worker_routing(schema: dict) -> dict:
    """Test/offline-only deterministic routing over the supplied catalog.

    This helper is not a live LLM result and is not a production-provider
    failure fallback. It only emits agent/capability pairs present in the
    caller-provided compact AgentCard catalog.
    """
    intent = str(schema.get("compact_user_intent") or "").lower()
    structured_intent = schema.get("structured_intent")
    if not isinstance(structured_intent, dict):
        structured_intent = {}
    semantic_text = " ".join(
        [
            intent,
            str(structured_intent.get("primary_intent") or "").lower(),
            " ".join(
                str(value).lower()
                for value in structured_intent.get("secondary_intents", [])
                if isinstance(value, str)
            ),
            " ".join(
                str(value).lower()
                for value in structured_intent.get("requested_outputs", [])
                if isinstance(value, str)
            ),
        ]
    )
    available: dict[str, tuple[str, dict]] = {}
    for agent in schema.get("compact_card_catalog", []):
        if not isinstance(agent, dict) or not isinstance(agent.get("agent_id"), str):
            continue
        for capability in agent.get("capabilities", []):
            if isinstance(capability, dict) and isinstance(
                capability.get("capability_id"), str
            ):
                available[capability["capability_id"]] = (
                    agent["agent_id"], capability
                )

    requested: list[tuple[str, str, str, str]] = []
    if "developability" in semantic_text:
        requested.extend(
            [
                (
                    "step_05_candidate_context",
                    "Build normalized candidate context for downstream assessment.",
                    "Candidate context supports the requested developability assessment.",
                    "high",
                ),
                (
                    "step_06_developability",
                    "Assess developability and liability risks for normalized candidates.",
                    "The user requests developability assessment.",
                    "normal",
                ),
            ]
        )
    if any(token in semantic_text for token in ("structure", "protein design")):
        requested.append(
            (
                "structure_design_workflow",
                "Run the sequential structure and design workflow.",
                "The user requests structure or protein-design analysis.",
                "high",
            )
        )
    if any(
        token in semantic_text
        for token in (
            "literature",
            "scientific evidence",
            "evidence",
            "patent",
            "prior-art",
            "prior art",
            "intellectual property",
            "regulatory reference",
            "scientific_evidence_table",
            "patent_prior_art_table",
        )
    ):
        requested.append(
            (
                "patent_evidence_workflow",
                "Review scientific evidence, patent prior art, and regulatory references.",
                "The user intent or requested outputs require patent-evidence review.",
                "normal",
            )
        )
    available_artifacts = {
        str(item.get("artifact_name"))
        for item in schema.get("available_artifact_summary", [])
        if isinstance(item, dict) and item.get("available") is True
    }
    requested_ids = {item[0] for item in requested}
    prerequisites: list[tuple[str, str, str, str]] = []
    for capability_id, *_rest in requested:
        match = available.get(capability_id)
        if match is None:
            continue
        capability = match[1]
        for artifact_name in capability.get("required_input_artifact_names", []):
            if artifact_name in available_artifacts:
                continue
            producer = next(
                (
                    producer_id
                    for producer_id, (_agent_id, producer_capability) in available.items()
                    if artifact_name
                    in producer_capability.get("output_artifact_names", [])
                ),
                None,
            )
            if producer is not None and producer not in requested_ids:
                prerequisites.append(
                    (
                        producer,
                        f"Produce required artifact {artifact_name}.",
                        "The selected capability requires an artifact produced by this catalog capability.",
                        "high",
                    )
                )
                requested_ids.add(producer)
    requested = [*prerequisites, *requested]
    decisions = []
    for capability_id, objective, reason, priority in requested:
        match = available.get(capability_id)
        if match is None:
            continue
        decisions.append(
            {
                "agent_id": match[0],
                "capability_id": capability_id,
                "objective": objective,
                "selection_reason": reason,
                "priority": priority,
            }
        )
    return {
        "loop_decision": (
            "dispatch_next_workers" if decisions else "route_to_final_response"
        ),
        "decisions": decisions,
        "decision_summary": (
            "Selected matching capabilities from the supplied catalog."
            if decisions
            else "No supplied capability matches the compact user intent."
        ),
    }


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
        if task == "orchestrator_worker_routing":
            return _mock_orchestrator_worker_routing(schema)
        if task == "tool_selection_stage_1":
            return _mock_stage1_selection(schema)
        if task == "tool_selection_stage_2":
            return _mock_stage2_arguments(schema)
        if task == "tool_selection_stage_1_multi_lane":
            return _mock_stage1_multi_lane(schema)
        if task == "tool_selection_stage_2_multi_tool":
            return _mock_stage2_multi_tool(schema)
        if task == "step6_schema_mapping_stage_1":
            return _mock_step6_schema_mapping_stage1(schema)
        if task == "step6_schema_mapping_stage_2":
            return _mock_step6_schema_mapping_stage2(schema)
        if task == "step9_tool_selection_stage_1":
            return _mock_step9_tool_selection_stage1(schema)
        if task == "step9_tool_schema_mapping_stage_2":
            return _mock_step9_tool_schema_mapping_stage2(schema)
        if task == "step14_patent_tool_selection":
            return _mock_step14_patent_tool_selection(schema)
        if task == "patent_evidence_tool_selection":
            return _mock_patent_evidence_tool_selection(schema)

        raw = (schema or {}).get("raw_request_record") or {}
        ctx = raw.get("user_provided_context") or {}
        user_query = raw.get("raw_user_query") or ""
        uploaded_files = raw.get("uploaded_files") or []
        # Clarification follow-up answers (if this is a revision turn) are
        # the mock's stand-in for the LLM "remembering" the previous turn:
        # we fold the answer texts into the detection haystack and index them
        # by the slot they answered. The original query is preserved in
        # `user_query`, so intent classification is unchanged.
        answer_by_slot: dict[str, str] = {}
        answer_texts: list[str] = []
        for a in ctx.get("clarification_answers") or []:
            if not isinstance(a, dict):
                continue
            txt = str(a.get("answer_text") or "").strip()
            if not txt:
                continue
            answer_texts.append(txt)
            for key in (a.get("slot_name"), a.get("slot_category")):
                if key and key not in answer_by_slot:
                    answer_by_slot[key] = txt

        haystack = " ".join(
            [
                user_query,
                ctx.get("target_or_antigen_text") or "",
                ctx.get("candidate_text") or "",
                ctx.get("payload_linker_text") or "",
                ctx.get("constraints_text") or "",
                ctx.get("notes") or "",
                " ".join(answer_texts),
            ]
        )

        target = (
            ctx.get("target_or_antigen_text")
            or _find_first(haystack, _TARGET_HINTS)
            or answer_by_slot.get("target_or_antigen")
            or answer_by_slot.get("target")
        )
        candidate = (
            ctx.get("candidate_text")
            or answer_by_slot.get("antibody")
            or answer_by_slot.get("antibody_candidate")
        )
        payload = (
            _find_first(haystack, _PAYLOAD_HINTS)
            or answer_by_slot.get("payload")
        )
        linker = (
            _find_first(haystack, _LINKER_HINTS)
            or answer_by_slot.get("linker")
        )

        # If the user gave a free-form payload_linker_text but it didn't match
        # the hint list, surface it as the payload string anyway. We do NOT
        # invent identifiers.
        if not payload and ctx.get("payload_linker_text"):
            payload = ctx["payload_linker_text"]

        referenced = _detect_referenced_inputs(haystack)
        referenced.extend(_uploaded_file_refs(uploaded_files))
        prompt_sequence_ref = _detect_prompt_sequence_referenced_input(haystack, uploaded_files)
        if prompt_sequence_ref:
            referenced.append(prompt_sequence_ref)
        wants_protein_generation = requests_protein_generation(haystack)

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

        # Label detected protein variants as first-class normalized_entities
        # (entity_type="protein_variant") for traceability. The actionable
        # variant string is ALSO carried in referenced_inputs[id_type="variant"]
        # above; downstream (Step 5/9) consumes the referenced_input.
        for ref in referenced:
            if ref.get("id_type") == "variant":
                normalized_entities.append(
                    {
                        "original_text": ref["value"],
                        "canonical_name": ref["value"],
                        "entity_type": "protein_variant",
                        "explicit_or_inferred": "explicit",
                        "confidence": 0.9,
                    }
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

        missing_slots = _compute_missing_slots(
            primary_intent=primary_intent,
            target=target,
            candidate=candidate,
            payload=payload,
            linker=linker,
            referenced=referenced,
            normalized_entities=normalized_entities,
            mentioned_drugs=mentioned_drugs,
            wants_protein_generation=wants_protein_generation,
        )
        response = _compose_missing_slots_response(missing_slots)
        canonical_query = _compose_canonical_query(
            primary_intent=primary_intent,
            target=target,
            candidate=candidate,
            payload=payload,
            linker=linker,
            missing_slots=missing_slots,
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
            "missing_slots": missing_slots,
            "response": response,
            "canonical_query": canonical_query,
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
        v.lower() for v in ctx.values() if isinstance(v, str)
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
        v.lower() for v in (ctx or {}).values() if isinstance(v, str)
    )
    ref_id_types = {r.get("id_type") for r in referenced}
    has_pdb = "pdb_id" in ref_id_types
    has_zinc = "zinc_id" in ref_id_types
    has_chembl = "chembl_id" in ref_id_types
    # A concrete protein point-mutation / variant reference (Step 2
    # id_type="variant"/"mutation") signals a variant-evaluation / structure
    # analysis task (AlphaMissense / DynaMut2 / ESM variant scoring).
    has_variant = bool(ref_id_types & {"variant", "mutation"})

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
    elif has_pdb or has_variant or any(
        kw in text for kw in (
            "structure analysis", "validate the structure", "structure of",
            "structural analysis", "interface analysis", "binding mode",
            "variant scoring", "variant effect", "score the variant",
            "point mutation", "missense",
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

    # New ADC design fallback. Fires when the user shows a target + design /
    # payload signal, OR when there is an explicit design verb together with
    # an ADC signal even though the target is still missing — that missing
    # target is exactly the blocking gap Step 3 will surface.
    elif (not non_adc) and (
        (target and (payload or "design" in text or "build" in text))
        or (
            any(verb in text for verb in ("design", "build", "create", "develop"))
            and any(sig in text for sig in ("adc", "antibody-drug", "antibody drug"))
        )
    ):
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

    # Preserve explicit multi-intent review requests even when another strong
    # cue (for example a known ADC or developability request) owns the primary
    # intent. This remains test/offline Mock behavior; production providers
    # receive the same catalog and typed schema without this heuristic.
    if any(
        keyword in text
        for keyword in ("patent", "ip ", "prior art", "freedom to operate", "fto")
    ):
        secondary.append("patent_ip_review")
        outputs.append("patent_or_ip_summary")
    if any(
        keyword in text
        for keyword in (
            "literature",
            "papers",
            "scientific evidence",
            "review the literature",
            "review papers",
        )
    ):
        secondary.append("literature_review")
        outputs.append("literature_review_summary")

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


_ANTIBODY_SEQUENCE_SOURCES = {
    "antibody_heavy_chain_sequence",
    "antibody_light_chain_sequence",
    "antibody_sequence_reference",
}
_TARGET_SEQUENCE_SOURCES = {"target_sequence"}
_STRUCTURE_FILE_EXTS = (".pdb", ".cif", ".mmcif", ".ent")
_SEQUENCE_FILE_EXTS = (".fasta", ".fa", ".faa", ".seq")


def _missing_slot_signals(
    *,
    target: str | None,
    candidate: str | None,
    payload: str | None,
    linker: str | None,
    referenced: list[dict],
    normalized_entities: list[dict],
    mentioned_drugs: list[str],
) -> dict[str, bool]:
    """Deterministic satisfied/unsatisfied flags mirroring required_slot_schema.

    A slot is satisfied when ANY equivalent typed input is present — a flat
    mention string, a normalized entity of the right type, a typed
    referenced_input, an antibody-chain / sequence reference, or an uploaded
    file whose filename/source indicates structure or sequence. Only
    metadata is inspected — never file bytes or sequence content.
    """
    ref_id_types = {
        r.get("id_type") for r in referenced if isinstance(r, dict)
    }
    norm_types = {
        ne.get("entity_type") for ne in normalized_entities if isinstance(ne, dict)
    }

    smiles_sources = {
        (r.get("source") or "")
        for r in referenced
        if isinstance(r, dict) and r.get("id_type") == "smiles"
    }
    has_payload_smiles = bool(smiles_sources & {"payload_smiles", "compound_smiles"})
    has_linker_smiles = bool(smiles_sources & {"linker_smiles", "compound_smiles"})

    has_sequence_ref = False
    has_structure_upload = False
    has_sequence_upload = False
    prompt_file_ids = {
        str(r.get("value") or "")
        for r in referenced
        if isinstance(r, dict)
        and r.get("id_type") == "uploaded_file"
        and r.get("source") == "prompt_sequence"
    }
    for r in referenced:
        if not isinstance(r, dict):
            continue
        idt = (r.get("id_type") or "")
        src = (r.get("source") or "").lower()
        filename = (r.get("filename") or "").lower()
        if (
            idt in _ANTIBODY_SEQUENCE_SOURCES
            or src in _ANTIBODY_SEQUENCE_SOURCES
            or idt in _TARGET_SEQUENCE_SOURCES
            or src in _TARGET_SEQUENCE_SOURCES
        ):
            has_sequence_ref = True
        file_is_prompt_only = (
            idt == "uploaded_file"
            and str(r.get("value") or "") in prompt_file_ids
        )
        if filename.endswith(_STRUCTURE_FILE_EXTS) and not file_is_prompt_only:
            has_structure_upload = True
        if filename.endswith(_SEQUENCE_FILE_EXTS) and not file_is_prompt_only:
            has_sequence_upload = True

    target_satisfied = bool(target) or "uniprot_id" in ref_id_types or "target_or_antigen" in norm_types
    antibody_satisfied = (
        bool(candidate) or "antibody" in norm_types or has_sequence_ref
    )
    payload_satisfied = bool(payload) or "payload" in norm_types or has_payload_smiles
    linker_satisfied = (
        bool(linker)
        or bool(norm_types & {"linker", "linker_payload"})
        or has_linker_smiles
    )
    structure_or_sequence_satisfied = (
        "pdb_id" in ref_id_types
        or "uniprot_id" in ref_id_types
        or has_structure_upload
        or has_sequence_upload
        or has_sequence_ref
    )
    return {
        "target": target_satisfied,
        "antibody": antibody_satisfied,
        "payload": payload_satisfied,
        "linker": linker_satisfied,
        "structure_or_sequence": structure_or_sequence_satisfied,
        "any_analyzable": (
            payload_satisfied
            or linker_satisfied
            or antibody_satisfied
            or structure_or_sequence_satisfied
        ),
        "searchable_entity": (
            target_satisfied
            or antibody_satisfied
            or payload_satisfied
            or bool(mentioned_drugs)
        ),
    }


def _compute_missing_slots(
    *,
    primary_intent: str,
    target: str | None,
    candidate: str | None,
    payload: str | None,
    linker: str | None,
    referenced: list[dict],
    normalized_entities: list[dict],
    mentioned_drugs: list[str],
    wants_protein_generation: bool = False,
) -> list[dict]:
    """Mock missing_slots that follow the prompt's required_slot_schema.

    Deterministic and intentionally minimal: only the rules the Step 2 /
    Step 3 tests exercise. Each entry already matches the cleaned schema
    shape (so the supervisor normalizer is a no-op on this output).
    """
    sig = _missing_slot_signals(
        target=target,
        candidate=candidate,
        payload=payload,
        linker=linker,
        referenced=referenced,
        normalized_entities=normalized_entities,
        mentioned_drugs=mentioned_drugs,
    )
    slots: list[dict] = []

    def _add(slot_name, slot_category, severity, required_for, reason, question=None):
        slots.append(
            {
                "slot_name": slot_name,
                "slot_category": slot_category,
                "severity": severity,
                "required_for": list(required_for),
                "reason": reason,
                "suggested_question": question,
                "evidence": None,
            }
        )

    if primary_intent == "new_adc_design":
        if not sig["target"]:
            _add(
                "target_or_antigen", "target", "blocking", ["new_adc_design"],
                "No target/antigen provided for the new ADC design.",
                "What target or antigen should the ADC be designed against?",
            )
        if not sig["antibody"]:
            _add(
                "antibody", "antibody", "warning", ["new_adc_design"],
                "No antibody candidate provided; Step 5 will rely on discovery.",
                "Which antibody candidate should we use, or should we run discovery?",
            )
        if not sig["payload"]:
            _add(
                "payload", "payload", "warning", ["new_adc_design"],
                "No payload provided for the ADC.",
                "Which payload should the ADC carry?",
            )
        if not sig["linker"]:
            _add(
                "linker", "linker", "warning", ["new_adc_design"],
                "No linker chemistry specified.",
                "Which linker chemistry should we assume?",
            )
    elif primary_intent == "structure_analysis":
        if not sig["structure_or_sequence"]:
            _add(
                "structure_or_sequence", "structure", "blocking",
                ["structure_analysis"],
                "No structure or sequence input provided for structure analysis.",
                "Please provide a PDB/CIF file, PDB ID, UniProt ID, or protein sequence.",
            )
    elif primary_intent == "developability_assessment":
        if not sig["any_analyzable"]:
            _add(
                "structure_or_sequence", "structure", "blocking",
                ["developability_assessment"],
                "No analyzable molecule or protein input found.",
                "Please provide a compound/SMILES, protein sequence, UniProt ID, or structure.",
            )
    elif primary_intent in {"literature_review", "patent_ip_review"}:
        if not sig["searchable_entity"]:
            _add(
                "other", "other", "blocking", [primary_intent],
                "No searchable entity (target, drug, or compound) was identified.",
                "Which target, drug, or compound should we search for?",
            )

    # Cross-cutting: protein generation requires an explicit prompt_sequence
    # regardless of primary_intent. An ordinary heavy/light/target sequence
    # (or any other referenced input) never satisfies it.
    if wants_protein_generation:
        has_valid_prompt_sequence = any(
            isinstance(r, dict)
            and (
                (
                    r.get("id_type") == "prompt_sequence"
                    and looks_like_masked_prompt_sequence(r.get("value"))
                )
                or (r.get("id_type") == "uploaded_file" and r.get("source") == "prompt_sequence")
            )
            for r in referenced
        )
        if not has_valid_prompt_sequence:
            _add(
                "prompt_sequence", "sequence", "blocking", [primary_intent],
                "Protein generation requires an explicit masked prompt_sequence.",
                'Please provide the masked protein generation prompt '
                '(containing "_" or "<mask>") you want ESM to complete.',
            )

    return slots


_SLOT_PHRASES = {
    "target_or_antigen": "target or antigen",
    "antibody": "antibody candidate",
    "payload": "payload",
    "linker": "linker chemistry",
    "structure_or_sequence": "structure or sequence (PDB/CIF, PDB ID, UniProt ID, or protein sequence)",
    "pdb_id": "PDB ID",
    "uniprot_id": "UniProt ID",
    "smiles": "SMILES",
    "prompt_sequence": 'masked protein generation prompt (containing "_" or "<mask>")',
    "task_intent": "workflow you want to run",
    "constraint": "constraints",
    "other": "additional details",
}


def _join_phrases(phrases: list[str]) -> str:
    if not phrases:
        return ""
    if len(phrases) == 1:
        return phrases[0]
    if len(phrases) == 2:
        return f"{phrases[0]} and {phrases[1]}"
    return ", ".join(phrases[:-1]) + f", and {phrases[-1]}"


def _compose_missing_slots_response(missing_slots: list[dict]) -> str | None:
    """Compose a concise user-facing follow-up from the mock's missing_slots.

    Mirrors the prompt's contract: prioritize blocking slots, combine
    warnings compactly, phrase naturally. Optional slots are omitted.
    Returns None when there is nothing to ask for.
    """
    blocking = [
        _SLOT_PHRASES.get(s.get("slot_name"), "the required information")
        for s in missing_slots
        if s.get("severity") == "blocking"
    ]
    warning = [
        _SLOT_PHRASES.get(s.get("slot_name"), "the requested information")
        for s in missing_slots
        if s.get("severity") == "warning"
    ]
    if not blocking and not warning:
        return None
    parts: list[str] = []
    if blocking:
        parts.append(f"Please provide the {_join_phrases(blocking)} for the ADC.")
        if warning:
            parts.append(f"If available, also provide the {_join_phrases(warning)} you want to use.")
    else:
        parts.append(f"Please provide the {_join_phrases(warning)} you want to use.")
    return " ".join(parts)


_INTENT_TASK_PHRASES = {
    "new_adc_design": "Design a new antibody-drug conjugate",
    "existing_adc_evaluation": "Evaluate an existing antibody-drug conjugate",
    "developability_assessment": "Assess developability",
    "structure_analysis": "Analyze the structure",
    "compound_screening": "Screen compounds",
    "literature_review": "Review the literature",
    "patent_ip_review": "Review patents / IP",
    "optimization": "Optimize the design",
    "unclear_or_needs_clarification": "Clarify the requested workflow",
}


def _compose_canonical_query(
    *,
    primary_intent: str,
    target: str | None,
    candidate: str | None,
    payload: str | None,
    linker: str | None,
    missing_slots: list[dict],
) -> str:
    """Compose the mock's canonical (normalized) task description.

    Deterministic and compact — describes the task with whatever components
    are known and marks the rest 'unspecified'. It does NOT dump structured
    context and never invents unanswered fields. On a clarification turn the
    `target`/`candidate`/... already reflect the folded-in answers, so the
    canonical_query naturally updates (e.g. picks up "HER2").
    """
    head = _INTENT_TASK_PHRASES.get(primary_intent, "Process the ADC request")
    missing_names = {s.get("slot_name") for s in missing_slots}

    def _describe(slot: str, value: str | None, label: str) -> str:
        if value:
            return f"{label} {value}"
        if slot in missing_names:
            return f"{label} unspecified"
        return ""

    parts = [
        _describe("target_or_antigen", target, "target"),
        _describe("antibody", candidate, "antibody"),
        _describe("payload", payload, "payload"),
        _describe("linker", linker, "linker"),
    ]
    detail = "; ".join(p for p in parts if p)
    text = head if not detail else f"{head} ({detail})."
    return text[:800]


_PROMPT_SEQUENCE_LABEL_RE = re.compile(
    r"prompt[_ ]sequence|masked\s+(?:protein\s+)?prompt|generation\s+prompt|"
    r"sequence[_ ]completion\s+prompt",
    re.IGNORECASE,
)
_PROTEIN_LIKE_TOKEN_RE = re.compile(r"\b[ACDEFGHIKLMNPQRSTVWY_]{6,}\b", re.IGNORECASE)


def _detect_prompt_sequence_referenced_input(haystack: str, uploaded_files: list) -> dict | None:
    """Mock stand-in for LLM semantic role recognition: does the user's own
    text explicitly label an inline sequence or an uploaded file as the ESM
    generation prompt_sequence / masked protein prompt?

    Never inspects uploaded-file bytes — only query/context text (already in
    `haystack`) and filename metadata. Mask-marker content validation for the
    inline candidate is a separate, deterministic step
    (`looks_like_masked_prompt_sequence`) applied uniformly to every
    provider's output in
    `supervisor_agent._normalize_prompt_sequence_references` — this function
    only recognizes that a role was declared, exactly like every other
    referenced-input detector in this mock.
    """
    if _PROMPT_SEQUENCE_LABEL_RE.search(haystack):
        for token_match in _PROTEIN_LIKE_TOKEN_RE.finditer(haystack):
            token = token_match.group(0)
            has_mask = "_" in token or "<mask>" in token.lower()
            letters_only = token.replace("_", "")
            if not letters_only or not re.fullmatch(
                r"[ACDEFGHIKLMNPQRSTVWY]+", letters_only, re.IGNORECASE
            ):
                continue
            # A masked token (contains "_"/"<mask>") is unambiguous even
            # short; an unmasked token must be long enough to not collide
            # with ordinary short English words that happen to use only
            # amino-acid letters (e.g. "Please", "Capitals").
            min_len = 6 if has_mask else 12
            if len(token) < min_len:
                continue
            return {"id_type": "prompt_sequence", "value": token, "source": "user"}
        for f in uploaded_files or []:
            if isinstance(f, dict) and f.get("file_id"):
                return {
                    "id_type": "uploaded_file",
                    "value": f["file_id"],
                    "source": "prompt_sequence",
                }
        return None
    for f in uploaded_files or []:
        if not isinstance(f, dict):
            continue
        filename = str(f.get("original_filename") or "").lower()
        if "prompt" in filename and f.get("file_id"):
            return {
                "id_type": "uploaded_file",
                "value": f["file_id"],
                "source": "prompt_sequence",
            }
    return None


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
