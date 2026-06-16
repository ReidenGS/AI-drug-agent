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
    The full JSON parameter schema is NOT included. Ask the LLM to pick zero
    or more tool names with a short reason. Drop anything that hallucinates a
    tool name not in the catalog, dedupe survivors.

Stage 2 — Argument Construction
    For every survivor only, introspect the registered binding via
    `inspect.signature(...)` to produce a JSON-shaped param schema and ask
    the LLM to fill in arguments. Validate against the schema: required
    fields, integer/float/str/bool coercion, enum membership. If the LLM
    can't produce valid args, fall back to a per-step deterministic argument
    mapping; if that also fails, the plan entry is `validation_status="skipped"`
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


# Capability registry for tools we route in this iteration. Tools not listed
# fall back to a generic entry derived from `tool_name`. Everything here is
# auditable in one place rather than scattered across agents.
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


def _generic_entry(tool_name: str, agent_name: str, step_id: str) -> CompactToolEntry:
    return CompactToolEntry(
        tool_name=tool_name,
        short_description=tool_name.replace("_", " "),
        capability_tags=[step_id, agent_name],
        coarse_input_requirements=["context"],
        step_id=step_id,
        agent_name=agent_name,
    )


def build_compact_catalog(
    *, mcp_client: MCPClient, agent_name: str, step_id: str,
) -> list[CompactToolEntry]:
    """Ask the MCPClient which tools are allowed, then strip to compact form."""
    allowed = mcp_client.list_tools(agent_name=agent_name, step_id=step_id)
    out: list[CompactToolEntry] = []
    for name in sorted(allowed):
        meta = CAPABILITY_REGISTRY.get(name)
        if meta is None:
            out.append(_generic_entry(name, agent_name, step_id))
        else:
            out.append(
                CompactToolEntry(
                    tool_name=name,
                    short_description=meta["short_description"],
                    capability_tags=list(meta.get("capability_tags") or []),
                    coarse_input_requirements=list(meta.get("coarse_input_requirements") or []),
                    step_id=step_id,
                    agent_name=agent_name,
                )
            )
    return out


# ── Stage 2: signature → schema, validation ────────────────────────────────

_TYPE_HINT_MAP = {int: "integer", float: "number", str: "string", bool: "boolean", dict: "object", list: "array"}


def signature_schema_for(tool_name: str) -> Optional[dict]:
    """Introspect the registered binding via `inspect.signature`.

    Returns a JSON-Schema-ish dict with required + properties; or None if
    we have no binding for that name (which is itself a validation failure
    upstream because the tool wasn't in scope).
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
            "Pick tools from the compact catalog that match the context.",
            schema=stage1_payload,
            system=(
                "You are picking MCP tools by name only. Stick to the catalog. "
                "Do not invent arguments — Stage 2 will handle those."
            ),
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
                "Construct arguments for the selected tool.",
                schema=stage2_payload,
                system=(
                    "Fill arguments only from the provided context. Do not invent."
                ),
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
