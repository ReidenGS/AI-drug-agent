"""LLM-assisted tool selection with progressive disclosure.

Two stages, both bounded by the existing `MCPClient` + `scope_filter` +
`ToolUniversity v0.2 inventory` whitelist. The selector does NOT manage its
own permission rules — it ASKS the MCP client which tools are allowed for
the (agent_name, step_id) pair and only ever proposes / validates tools that
list reports.

Stage 1 — Tool Selection
    Build a compact catalog from `mcp_client.list_tools(agent_name, step_id)`.
    Each catalog entry carries ONLY:
        tool_name / short_description / capability_tags
        / coarse_input_requirements / step_id / agent_name
    `short_description` PREFERS the official ToolUniverse spec
    description (via `app.mcp.tooluniverse_adapter.get_tool_specifications`)
    for any tool TU recognizes; CAPABILITY_REGISTRY is only the fallback
    when TU has no entry or its metadata API is unavailable. The
    hand-written registry continues to own `capability_tags` and
    `coarse_input_requirements` (project-specific hints TU does not
    provide).
    The full JSON parameter schema is NOT included in Stage 1. Ask the
    LLM to pick zero or more tool names with a short reason. Drop
    anything that hallucinates a tool name not in the catalog, dedupe
    survivors.

Stage 2 — Argument Construction
    For every survivor only, look up the official ToolUniverse parameter
    schema via `get_tool_specification(...)`; fall back to
    `inspect.signature(...)` only when TU has no schema. `_live` is
    NEVER exposed to the LLM. Validate against the schema: required
    fields, integer/float/str/bool coercion. If the LLM can't produce
    valid args, fall back to a per-step deterministic argument mapping;
    if that also fails, the plan entry is `validation_status="skipped"`
    and the caller skips the MCP call instead of crashing.

`ToolInvocationPlan.selected_by` records `"llm"` or `"deterministic_fallback"`
so audit can tell which path was taken per tool.
"""

from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Literal, Optional

from pydantic import BaseModel, Field, ValidationError

from ..llm.provider import LLMProvider
from ..mcp.client import MCPClient
from ..mcp.tools._registry import _all_bindings

logger = logging.getLogger(__name__)

SELECTION_POLICY_VERSION = "v1"


# ── shared Stage 1 / Stage 2 prompt contract ────────────────────────────────
#
# These constants are the SINGLE source of truth for tool-selection prompts
# used by every agent that exercises `select_and_build_invocations`
# (DevelopabilityAgent / StructureAndDesignAgent / EvidenceAgent /
# PatentIPAgent today; any future step agent that selects MCP tools must
# import them too — agents MUST NOT define their own selector wording).
#
# Step 2 SupervisorAgent has a separate, parser-only prompt and does NOT
# import any of these constants. Searching the codebase for these names
# must surface zero hits under `app/agents/supervisor_agent.py`.


SELECTION_STAGE1_SYSTEM_PROMPT = """You are choosing MCP tools for the CURRENT agent / step ONLY.

Rules:

1. Use ONLY the `compact_catalog` provided. The catalog has already been
   scope-filtered by `agent_name` and `step_id` — every entry is a tool
   the current agent is allowed to call at the current step. You do not
   see, and must not assume, any tools outside this catalog. Do not draw
   on full-ToolUniverse knowledge that is not in the catalog.
2. Select zero or more `tool_name` values from the catalog. Each
   selection must be an exact `tool_name` string that appears in the
   catalog. Do not invent, paraphrase, or hallucinate tool names.
3. Each selection may carry a short `selection_reason`, a `priority`
   integer, and a `required_context` list — nothing else.
4. Do NOT construct arguments. Argument construction happens in Stage 2
   for the survivors only.
5. Do NOT infer hidden tools, do not request more tools, do not invent
   capability flags. The catalog is the entire surface available to you.
6. If the available context is insufficient to justify ANY selection,
   return an empty `selections` list.
7. Return EXACTLY ONE valid JSON object matching the shape declared in
   the schema. No prose, no markdown fences, no tool calls.
""".strip()


SELECTION_STAGE1_USER_PROMPT = (
    "Pick the MCP tools from the compact catalog that match the current "
    "context. Use tool_name only — Stage 2 will handle arguments."
)


SELECTION_STAGE2_SYSTEM_PROMPT = """You are constructing ARGUMENTS for ONE selected tool.

Rules:

1. Use ONLY the `full_schema` provided for the selected `tool_name`.
   It is the official parameter schema for that tool (ToolUniverse
   official when available, wrapper-signature fallback otherwise).
2. Use ONLY values present in `context.arg_hints` and `context.note` to
   fill arguments. Do NOT invent missing required IDs, SMILES, PDB IDs,
   UniProt accessions, PubChem CIDs, brand names, ChEMBL IDs, or any
   other identifiers the schema requires.
3. If a required input is missing from `context.arg_hints`, leave it
   missing and list it under `missing_fields`. Do NOT guess.
4. Output MUST NOT include the `_live` knob. That flag is set by the
   MCP client, never by the model.
5. Do NOT call the tool, do NOT request additional schema or catalog
   entries. You are an argument-construction step, not an execution step.
6. Return EXACTLY ONE valid JSON object matching the shape declared in
   the schema. No prose, no markdown fences, no tool calls.
""".strip()


SELECTION_STAGE2_USER_PROMPT = (
    "Construct arguments for the selected tool. Use only the provided "
    "official schema and the arg_hints / note from context."
)


# ── compact catalog ─────────────────────────────────────────────────────────

class CompactToolEntry(BaseModel):
    """The minimum the LLM needs to pick a tool — no full schema, no kwargs."""

    tool_name: str
    short_description: str
    capability_tags: list[str] = Field(default_factory=list)
    coarse_input_requirements: list[str] = Field(default_factory=list)
    step_id: str
    agent_name: str


class ToolInvocationPlan(BaseModel):
    """One per (tool, argument-set) pair the agent should attempt."""

    tool_name: str
    selection_reason: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    argument_construction_reason: str = ""
    priority: int = 0
    required_context: list[str] = Field(default_factory=list)
    selected_by: Literal["llm", "deterministic_fallback"]
    selection_policy_version: str = SELECTION_POLICY_VERSION
    validation_status: Literal["ok", "warning", "skipped"] = "ok"
    validation_warnings: list[str] = Field(default_factory=list)


@dataclass(slots=True)
class SelectionContext:
    """What the agent knows when asking the selector to plan tool calls.

    `signals` is a small dict of bool flags (`has_smiles`, `has_pdb_id`,
    `has_uniprot_id`, ...) so the Stage-1 mock LLM can pick tools whose
    `coarse_input_requirements` intersect with the available signals.

    `arg_hints` is what Stage 2 uses to fill in arguments — a flat mapping
    `param_name → value` (e.g. `{"smiles": "CCO", "pdb_id": "1n8z"}`).
    """

    signals: dict[str, bool]
    arg_hints: dict[str, Any]
    note: str = ""


# Capability registry for tools we route in this iteration. After the
# Stage-1 ToolUniverse-official-description migration, the `short_description`
# entries here serve ONLY as a fallback for:
#   - custom services / placeholder rows TU does not own
#   - tools whose TU metadata API is temporarily unavailable
#   - project-specific phrasing where TU's wording is misleading
# `capability_tags` and `coarse_input_requirements` remain the canonical
# source — ToolUniverse does NOT carry these hints.
CAPABILITY_REGISTRY: dict[str, dict] = {
    # ── Step 6 developability ────────────────────────────────────────────
    "DrugProps_pains_filter": {
        "short_description": "Pan-assay interference (PAINS) substructure filter",
        "capability_tags": ["small_molecule", "liability_filter"],
        "coarse_input_requirements": ["smiles", "compound_name"],
    },
    "DrugProps_lipinski_filter": {
        "short_description": "Lipinski rule-of-five drug-likeness check",
        "capability_tags": ["small_molecule", "drug_likeness"],
        "coarse_input_requirements": ["smiles"],
    },
    "DrugProps_calculate_qed": {
        "short_description": "Quantitative estimation of drug-likeness (QED)",
        "capability_tags": ["small_molecule", "drug_likeness"],
        "coarse_input_requirements": ["smiles"],
    },
    "SwissADME_calculate_adme": {
        "short_description": "SwissADME bulk ADME calculation",
        "capability_tags": ["small_molecule", "adme"],
        "coarse_input_requirements": ["smiles"],
    },
    "SwissADME_check_druglikeness": {
        "short_description": "SwissADME drug-likeness rule check",
        "capability_tags": ["small_molecule", "drug_likeness"],
        "coarse_input_requirements": ["smiles"],
    },
    "ADMETAI_predict_toxicity": {
        "short_description": "ADMET-AI compound toxicity prediction",
        "capability_tags": ["small_molecule", "toxicity"],
        "coarse_input_requirements": ["smiles"],
    },
    "ADMETAI_predict_physicochemical_properties": {
        "short_description": "ADMET-AI physicochemical property prediction",
        "capability_tags": ["small_molecule", "physchem"],
        "coarse_input_requirements": ["smiles"],
    },
    "PROSITE_scan_sequence": {
        "short_description": "PROSITE motif scan over a protein sequence",
        "capability_tags": ["protein_sequence", "motif"],
        "coarse_input_requirements": ["protein_sequence", "antibody_sequence"],
    },
    "IEDB_predict_mhci_binding": {
        "short_description": "IEDB MHC-I binding affinity prediction",
        "capability_tags": ["antibody_liability", "immunogenicity"],
        "coarse_input_requirements": ["protein_sequence"],
    },
    "EBIProteins_get_features": {
        "short_description": "EBI Proteins API feature lookup",
        "capability_tags": ["antigen_context", "protein_features"],
        "coarse_input_requirements": ["uniprot_id", "target_name"],
    },
    "EBIProteins_get_epitopes": {
        "short_description": "EBI Proteins API epitope lookup",
        "capability_tags": ["antigen_context", "epitope"],
        "coarse_input_requirements": ["uniprot_id"],
    },
    "ProteinsPlus_profile_structure_quality": {
        "short_description": "ProteinsPlus structure quality profiling",
        "capability_tags": ["structure_quality"],
        "coarse_input_requirements": ["pdb_id", "structure_file"],
    },
    "ChEMBL_search_activities": {
        "short_description": "ChEMBL bioactivity search",
        "capability_tags": ["small_molecule", "bioactivity_prior"],
        "coarse_input_requirements": ["chembl_id", "compound_name", "smiles"],
    },
    "ChEMBL_search_molecules": {
        "short_description": "ChEMBL molecule search by name / id",
        "capability_tags": ["small_molecule", "bioactivity_prior"],
        "coarse_input_requirements": ["chembl_id", "compound_name"],
    },
    # ── Step 9 compound screening ──────────────────────────────────────
    "ZINC_search_by_smiles": {
        "short_description": "ZINC similarity search by SMILES (ZINC15 endpoint)",
        "capability_tags": ["compound_screening", "zinc15"],
        "coarse_input_requirements": ["smiles"],
    },
    "ZINC_get_compound": {
        "short_description": "ZINC compound lookup by id (ZINC15 endpoint)",
        "capability_tags": ["compound_screening", "zinc15"],
        "coarse_input_requirements": ["zinc_id"],
    },
    "ZINC_search_compounds": {
        "short_description": "ZINC text search (ZINC15 endpoint)",
        "capability_tags": ["compound_screening", "zinc15"],
        "coarse_input_requirements": ["compound_name", "smiles"],
    },
    # ── Step 13 scientific evidence ───────────────────────────────────
    "EuropePMC_search_articles": {
        "short_description": "Europe PMC article search",
        "capability_tags": ["literature", "target_evidence"],
        "coarse_input_requirements": ["target_literature_query"],
    },
    "SemanticScholar_search_papers": {
        "short_description": "Semantic Scholar paper search",
        "capability_tags": ["literature", "target_evidence"],
        "coarse_input_requirements": ["target_literature_query"],
    },
    "LiteratureSearchTool": {
        "short_description": "General literature search",
        "capability_tags": ["literature", "payload_evidence"],
        "coarse_input_requirements": ["payload_literature_query"],
    },
    "PubTator3_LiteratureSearch": {
        "short_description": "PubTator literature search for biomedical entities",
        "capability_tags": ["literature", "candidate_evidence"],
        "coarse_input_requirements": ["candidate_literature_query"],
    },
    "MultiAgentLiteratureSearch": {
        "short_description": "Multi-agent literature search over a candidate shortlist",
        "capability_tags": ["literature", "shortlist_evidence"],
        "coarse_input_requirements": ["shortlist_literature_query"],
    },
    # ── Step 14 patent / prior art ────────────────────────────────────
    "PubChem_get_associated_patents_by_CID": {
        "short_description": "PubChem associated patents by compound CID",
        "capability_tags": ["patent", "pubchem"],
        "coarse_input_requirements": ["pubchem_cid"],
    },
    "drugbank_get_drug_references_by_drug_name_or_id": {
        "short_description": "DrugBank references by drug name or identifier",
        "capability_tags": ["patent", "drugbank", "drug_references"],
        "coarse_input_requirements": ["drug_name_or_id", "compound_name"],
    },
    "FDA_OrangeBook_get_patent_info": {
        "short_description": "FDA Orange Book patent and exclusivity lookup",
        "capability_tags": ["patent", "orange_book"],
        "coarse_input_requirements": ["brand_name", "application_number", "compound_name"],
    },
}


def _official_descriptions(tool_names: list[str]) -> dict[str, str]:
    """Best-effort lookup of official TU descriptions for the allowed set.

    Safe-by-design: any failure (TU not installed, metadata call raises,
    spec is missing the `description` field) returns an empty mapping
    for those names so the selector falls back to CAPABILITY_REGISTRY.
    Inventory scope is enforced inside the adapter — we only pass the
    agent's already-allowed names here, never a wider list. Never logs
    description bodies.
    """
    if not tool_names:
        return {}
    try:
        from ..mcp import tooluniverse_adapter
    except Exception:  # noqa: BLE001
        return {}
    try:
        specs = tooluniverse_adapter.get_tool_specifications(tool_names)
    except Exception:  # noqa: BLE001
        return {}
    out: dict[str, str] = {}
    for name, spec in (specs or {}).items():
        desc = (spec or {}).get("description")
        if isinstance(desc, str):
            desc = desc.strip()
            if desc:
                out[name] = desc
    return out


def _resolve_short_description(
    tool_name: str,
    official: dict[str, str],
    registry_meta: dict | None,
) -> str:
    """Pick the Stage-1 description for a tool.

    Priority: official TU description → CAPABILITY_REGISTRY override →
    de-snake-cased tool name. The official-first ordering means
    ToolUniverse-existing tools cannot drift from TU's wording, and
    custom / deferred / placeholder tools still get a project-curated
    line when TU has none.
    """
    desc = official.get(tool_name)
    if desc:
        return desc
    if registry_meta and registry_meta.get("short_description"):
        return str(registry_meta["short_description"])
    return tool_name.replace("_", " ")


def build_compact_catalog(
    *, mcp_client: MCPClient, agent_name: str, step_id: str,
) -> list[CompactToolEntry]:
    """Ask the MCPClient which tools are allowed, then strip to compact form.

    `short_description` comes from the official ToolUniverse spec when TU
    recognizes the name; CAPABILITY_REGISTRY is only the fallback. Tags
    + coarse input requirements still come from the hand-written
    registry — TU does not carry that project-specific metadata.
    """
    allowed = mcp_client.list_tools(agent_name=agent_name, step_id=step_id)
    if not allowed:
        return []
    sorted_allowed = sorted(allowed)
    official = _official_descriptions(sorted_allowed)
    out: list[CompactToolEntry] = []
    for name in sorted_allowed:
        meta = CAPABILITY_REGISTRY.get(name) or {}
        short_description = _resolve_short_description(name, official, meta)
        if meta:
            capability_tags = list(meta.get("capability_tags") or [])
            coarse_input_requirements = list(
                meta.get("coarse_input_requirements") or []
            )
        else:
            capability_tags = [step_id, agent_name]
            coarse_input_requirements = ["context"]
        out.append(
            CompactToolEntry(
                tool_name=name,
                short_description=short_description,
                capability_tags=capability_tags,
                coarse_input_requirements=coarse_input_requirements,
                step_id=step_id,
                agent_name=agent_name,
            )
        )
    return out


# ── Stage 2: signature → schema, validation ────────────────────────────────

_TYPE_HINT_MAP = {int: "integer", float: "number", str: "string", bool: "boolean", dict: "object", list: "array"}


def _normalize_official_schema(
    tool_name: str, spec: dict | None
) -> Optional[dict]:
    """Reshape a TU spec's `parameter` block to the selector's JSON shape.

    TU specs come in two relevant flavors:

    1. `{"parameter": {"type": "object", "properties": {...}, "required": [...]}}`
       — already the shape we want.
    2. `{"parameter": {"properties": {...}}}` with required-ness encoded
       on each property as `{"required": true}` (TU's older convention)
       and/or a top-level `required: [...]`. We honor both.

    Returns `None` when the spec lacks any usable parameter info. Never
    exposes a `_live` knob: TU specs do not declare it, and we filter it
    defensively just in case a future TU revision does.
    """
    if not spec:
        return None
    param = spec.get("parameter") or spec.get("parameters") or {}
    if not isinstance(param, dict):
        return None
    properties_in = param.get("properties") or {}
    if not isinstance(properties_in, dict):
        return None
    properties: dict[str, dict] = {}
    required: list[str] = []
    declared_required = param.get("required") or []
    if not isinstance(declared_required, list):
        declared_required = []
    for prop_name, prop_spec in properties_in.items():
        if prop_name == "_live" or prop_name.startswith("_"):
            continue
        if not isinstance(prop_spec, dict):
            continue
        # Normalize the per-property type. TU sometimes uses
        # `["string", "null"]` to mean optional/nullable; collapse to the
        # first non-null type for the selector's coercion table.
        ptype = prop_spec.get("type")
        if isinstance(ptype, list):
            non_null = [t for t in ptype if isinstance(t, str) and t != "null"]
            ptype = non_null[0] if non_null else "string"
        elif not isinstance(ptype, str):
            ptype = "string"
        normalized: dict[str, Any] = {"type": ptype}
        if "enum" in prop_spec and isinstance(prop_spec["enum"], list):
            normalized["enum"] = list(prop_spec["enum"])
        properties[prop_name] = normalized
        if (
            prop_spec.get("required") is True
            or prop_name in declared_required
        ):
            required.append(prop_name)
    if not properties:
        return None
    # Preserve order, deduplicate.
    seen: set[str] = set()
    required_unique: list[str] = []
    for r in required:
        if r not in seen and r in properties:
            seen.add(r)
            required_unique.append(r)
    return {
        "type": "object",
        "properties": properties,
        "required": required_unique,
    }


def _signature_schema_from_binding(tool_name: str) -> Optional[dict]:
    """Fallback: introspect the registered binding via `inspect.signature`.

    Returns a JSON-Schema-ish dict with required + properties; or None if
    we have no binding for that name (which is itself a validation
    failure upstream because the tool wasn't in scope).
    """
    fn = _binding_for(tool_name)
    if fn is None:
        return None
    sig = inspect.signature(fn)
    properties: dict[str, dict] = {}
    required: list[str] = []
    for name, param in sig.parameters.items():
        if name in {"_live"} or param.kind in {param.VAR_KEYWORD, param.VAR_POSITIONAL}:
            continue
        if name.startswith("_"):
            continue
        ann = param.annotation
        py_type = ann if ann is not inspect.Parameter.empty else str
        type_str = _TYPE_HINT_MAP.get(py_type, "string")
        properties[name] = {"type": type_str}
        if param.default is inspect.Parameter.empty:
            required.append(name)
    return {"type": "object", "properties": properties, "required": required}


def signature_schema_for(tool_name: str) -> Optional[dict]:
    """Stage-2 schema for a single tool — official TU first, then signature.

    The selector calls this only for a survivor of Stage 1 (so only for
    tools that the agent's `mcp_client.list_tools(...)` allowed). We do
    not bulk-prefetch schemas in Stage 1 — that would defeat progressive
    disclosure and pay metadata cost for tools the LLM never picked.
    """
    try:
        from ..mcp import tooluniverse_adapter
        spec = tooluniverse_adapter.get_tool_specification(tool_name)
    except Exception:  # noqa: BLE001
        spec = None
    official = _normalize_official_schema(tool_name, spec)
    if official is not None:
        return official
    return _signature_schema_from_binding(tool_name)


def _binding_for(tool_name: str) -> Optional[Callable[..., Any]]:
    for name, fn in _all_bindings():
        if name == tool_name:
            return fn
    return None


def validate_arguments(args: dict[str, Any], schema: dict) -> tuple[dict[str, Any], list[str]]:
    """Coerce/validate args against a `signature_schema_for(...)` shape.

    Returns (cleaned_args, warnings). Missing required field → warning;
    unknown extra arg → warning + dropped. Numeric/bool/string coercion best
    effort. NEVER raises — the caller decides whether to skip.
    """
    warnings: list[str] = []
    properties: dict[str, dict] = schema.get("properties") or {}
    required: list[str] = list(schema.get("required") or [])
    cleaned: dict[str, Any] = {}
    for name, spec in properties.items():
        if name not in args:
            if name in required:
                warnings.append(f"required argument `{name}` missing")
            continue
        value = args[name]
        target_type = spec.get("type")
        coerced, ok = _coerce(value, target_type)
        if not ok:
            warnings.append(f"argument `{name}` could not be coerced to {target_type}")
            if name in required:
                continue
        else:
            cleaned[name] = coerced
    for extra in (set(args) - set(properties)):
        warnings.append(f"argument `{extra}` not in schema; dropping")
    return cleaned, warnings


def _coerce(value: Any, target_type: Optional[str]) -> tuple[Any, bool]:
    if target_type is None:
        return value, True
    try:
        if target_type == "string":
            return ("" if value is None else str(value)), True
        if target_type == "integer":
            return int(value), True
        if target_type == "number":
            return float(value), True
        if target_type == "boolean":
            return bool(value), True
        if target_type == "object":
            return (value if isinstance(value, dict) else {}), isinstance(value, dict)
        if target_type == "array":
            return (value if isinstance(value, list) else []), isinstance(value, list)
    except (TypeError, ValueError):
        return value, False
    return value, True


# ── orchestration ──────────────────────────────────────────────────────────

def _validate_selections(
    raw: list[dict],
    allowed_names: set[str],
) -> tuple[list[dict], list[str]]:
    """Drop hallucinated / out-of-scope / duplicate selections."""
    warnings: list[str] = []
    seen: set[str] = set()
    cleaned: list[dict] = []
    for entry in raw:
        if not isinstance(entry, dict):
            warnings.append("stage-1 entry was not a dict; dropping")
            continue
        name = entry.get("tool_name")
        if not name:
            warnings.append("stage-1 entry missing tool_name; dropping")
            continue
        if name not in allowed_names:
            warnings.append(f"stage-1 entry `{name}` not in allowed list; dropping")
            continue
        if name in seen:
            warnings.append(f"stage-1 entry `{name}` duplicated; dropping")
            continue
        seen.add(name)
        cleaned.append(entry)
    return cleaned, warnings


def select_and_build_invocations(
    *,
    agent_name: str,
    step_id: str,
    mcp_client: MCPClient,
    llm: LLMProvider,
    context: SelectionContext,
    deterministic_fallback: Callable[[], list[ToolInvocationPlan]],
    deterministic_argument_mapping: Optional[Callable[[str, dict], dict[str, Any]]] = None,
) -> list[ToolInvocationPlan]:
    """The full Stage 1 + Stage 2 dance.

    `deterministic_fallback` is called when Stage 1 fails (malformed LLM
    output, empty selections, scope-failed entries). The returned plans must
    already be valid — they bypass Stage 2.

    `deterministic_argument_mapping(tool_name, arg_hints) -> dict` is called
    per tool when Stage 2 args fail validation. If it returns args that
    re-validate, the plan goes through with `selected_by="deterministic_fallback"`;
    otherwise the plan is `validation_status="skipped"`.
    """
    catalog = build_compact_catalog(
        mcp_client=mcp_client, agent_name=agent_name, step_id=step_id
    )
    allowed_names = {e.tool_name for e in catalog}
    if not catalog:
        logger.debug("Empty catalog for %s/%s; using deterministic fallback", agent_name, step_id)
        return deterministic_fallback()

    # Stage 1 — selection.
    stage1_payload: dict[str, Any] = {
        "task": "tool_selection_stage_1",
        "agent_name": agent_name,
        "step_id": step_id,
        "compact_catalog": [e.model_dump() for e in catalog],
        "context": {"signals": context.signals, "note": context.note},
    }
    try:
        stage1_response = llm.generate_json(
            SELECTION_STAGE1_USER_PROMPT,
            schema=stage1_payload,
            system=SELECTION_STAGE1_SYSTEM_PROMPT,
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("Stage 1 LLM call failed (%s); using deterministic fallback", e)
        return deterministic_fallback()

    raw_selections = (stage1_response or {}).get("selections") or []
    cleaned, scope_warnings = _validate_selections(raw_selections, allowed_names)
    if not cleaned:
        logger.debug(
            "Stage 1 produced no usable selections (warnings=%s); falling back",
            scope_warnings,
        )
        return deterministic_fallback()

    # Stage 2 — argument construction per tool.
    plans: list[ToolInvocationPlan] = []
    for entry in cleaned:
        tool_name = entry["tool_name"]
        schema = signature_schema_for(tool_name)
        if schema is None:
            plans.append(_skipped_plan(
                tool_name=tool_name, reason=entry.get("selection_reason") or "",
                priority=entry.get("priority") or 0,
                required_context=entry.get("required_context") or [],
                selected_by="llm",
                validation_warnings=[*scope_warnings, "no callable signature found"],
            ))
            continue

        stage2_payload = {
            "task": "tool_selection_stage_2",
            "agent_name": agent_name,
            "step_id": step_id,
            "tool_name": tool_name,
            "full_schema": schema,
            "context": {"arg_hints": context.arg_hints, "note": context.note},
        }
        try:
            stage2_response = llm.generate_json(
                SELECTION_STAGE2_USER_PROMPT,
                schema=stage2_payload,
                system=SELECTION_STAGE2_SYSTEM_PROMPT,
            )
        except Exception as e:  # noqa: BLE001
            stage2_response = {"arguments": {}, "argument_construction_reason": f"llm_error:{e}"}

        proposed = (stage2_response or {}).get("arguments") or {}
        construct_reason = (stage2_response or {}).get("argument_construction_reason") or ""
        cleaned_args, arg_warnings = validate_arguments(proposed, schema)

        # If LLM args fall short on required fields, try deterministic mapping.
        selected_by: Literal["llm", "deterministic_fallback"] = "llm"
        if _missing_required(cleaned_args, schema) and deterministic_argument_mapping is not None:
            fallback_args = deterministic_argument_mapping(tool_name, context.arg_hints) or {}
            fallback_clean, fallback_warnings = validate_arguments(fallback_args, schema)
            if not _missing_required(fallback_clean, schema):
                cleaned_args = fallback_clean
                arg_warnings = [*arg_warnings, *fallback_warnings,
                                "stage-2 args fell back to deterministic mapping"]
                selected_by = "deterministic_fallback"

        validation_status: Literal["ok", "warning", "skipped"]
        if _missing_required(cleaned_args, schema):
            validation_status = "skipped"
        elif arg_warnings:
            validation_status = "warning"
        else:
            validation_status = "ok"

        try:
            plan = ToolInvocationPlan(
                tool_name=tool_name,
                selection_reason=entry.get("selection_reason") or "",
                arguments=cleaned_args,
                argument_construction_reason=construct_reason,
                priority=int(entry.get("priority") or 0),
                required_context=list(entry.get("required_context") or []),
                selected_by=selected_by,
                validation_status=validation_status,
                validation_warnings=arg_warnings,
            )
        except ValidationError:
            plan = _skipped_plan(
                tool_name=tool_name, reason="model_validation_failed",
                priority=0, required_context=[],
                selected_by="llm", validation_warnings=arg_warnings,
            )
        plans.append(plan)

    return plans


def _missing_required(args: dict[str, Any], schema: dict) -> bool:
    for name in (schema.get("required") or []):
        if name not in args or args[name] in (None, ""):
            return True
    return False


def _skipped_plan(
    *, tool_name: str, reason: str, priority: int, required_context: list[str],
    selected_by: Literal["llm", "deterministic_fallback"],
    validation_warnings: Iterable[str],
) -> ToolInvocationPlan:
    return ToolInvocationPlan(
        tool_name=tool_name,
        selection_reason=reason,
        arguments={},
        argument_construction_reason="",
        priority=priority,
        required_context=required_context,
        selected_by=selected_by,
        validation_status="skipped",
        validation_warnings=list(validation_warnings),
    )
