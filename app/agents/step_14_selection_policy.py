"""Step 14 LLM patent-tool selection policy.

The Step 14 request is reference-only: the LLM chooses WHICH patent tools to
run purely from ``user_query`` + the input refs' ``role`` /
``supports_tool_args`` / ``source_path`` + ``patent_scope``. It never sees a
resolved runtime value and never constructs raw tool arguments — the runtime
resolver builds real arguments later from the selected input refs.

Only three Step 14 tools are ever allowed here:
- ``PubChem_get_associated_patents_by_CID`` (needs a ``cid`` ref)
- ``FDA_OrangeBook_get_patent_info`` (needs a ``brand_name`` or
  ``application_number`` ref)
- ``drugbank_get_drug_references_by_drug_name_or_id`` (needs a
  ``drug_name_or_id`` or ``query`` ref)

Structure mirrors the Step 9 selector (`step_09_selection_policy`): a stable
English-rules + JSON-shape + compact tool-catalog prompt prefix, a
run-specific dynamic suffix, and a strict validator that drops hallucinated
tools / input refs, unsatisfiable plans, and antibody plans when antibody
search is disabled — recording every rejection with a compact reason.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field

from ..llm.provider import LLMProvider
from ..schemas.step_14_patent_request import Step14PatentRequest
from .tool_selection_policy import (
    _normalize_official_schema,
    _official_descriptions,
    signature_schema_for,
)


STEP14_SELECTION_PROMPT_CACHE_LAYOUT_VERSION = "step14_selection_v3"


# ── Step 14 tool specs (the only 3 allowed tools) ───────────────────────────
#
# Only the request-level ref-role mapping is Step-14-owned here. The tool
# `description`, `required` args, and full parameter schema are pulled at
# catalog-build time from the SAME official/registry schema source Step 6/9
# use (`tool_selection_policy.signature_schema_for` → ToolUniverse official
# schema first, wrapper signature only as a fallback). This never invents
# output fields, never handwrites an external client, and never calls live
# ToolUniverse (schemas come from the offline TU spec / local binding).
#
# `acceptable_supports` = the `supports_tool_args` tokens (any one is enough)
# that let an input ref satisfy the tool's identity argument.
# `supports_to_schema_arg` = each token → the OFFICIAL schema argument it
# fills, so `acceptable_supports` stays aligned with the sourced schema even
# when a token name differs from the schema arg name (recorded in
# `arg_mapping_notes`).
@dataclass(frozen=True)
class Step14ToolSpec:
    tool_name: str
    acceptable_supports: tuple[str, ...]
    supports_to_schema_arg: dict[str, str]
    arg_mapping_notes: dict[str, str] = field(default_factory=dict)
    fallback_description: str = ""


STEP14_TOOL_SPECS: dict[str, Step14ToolSpec] = {
    "PubChem_get_associated_patents_by_CID": Step14ToolSpec(
        tool_name="PubChem_get_associated_patents_by_CID",
        acceptable_supports=("cid", "pubchem_cid"),
        supports_to_schema_arg={"cid": "cid", "pubchem_cid": "cid"},
        arg_mapping_notes={
            "pubchem_cid": (
                "Step 14 ref-role token 'pubchem_cid' fills the official/runtime "
                "arg 'cid'."
            ),
        },
        fallback_description=(
            "Retrieve patents associated with a PubChem compound by its CID."
        ),
    ),
    "FDA_OrangeBook_get_patent_info": Step14ToolSpec(
        tool_name="FDA_OrangeBook_get_patent_info",
        acceptable_supports=("brand_name", "application_number"),
        supports_to_schema_arg={
            "brand_name": "brand_name",
            "application_number": "application_number",
        },
        arg_mapping_notes={},
        fallback_description=(
            "Retrieve FDA Orange Book patent / exclusivity info for a drug "
            "product by brand name OR NDA/ANDA application number."
        ),
    ),
    "drugbank_get_drug_references_by_drug_name_or_id": Step14ToolSpec(
        tool_name="drugbank_get_drug_references_by_drug_name_or_id",
        acceptable_supports=("drug_name_or_id", "query"),
        # Official ToolUniverse schema's primary text arg is `query`. The
        # runtime wrapper ALSO accepts `drug_name_or_id` (it folds it into
        # `query`), but Step 14 fills the official arg name `query` so the
        # constructed args stay aligned with the sourced schema.
        supports_to_schema_arg={"drug_name_or_id": "query", "query": "query"},
        arg_mapping_notes={
            "drug_name_or_id": (
                "Official ToolUniverse schema arg is 'query'; the runtime "
                "wrapper also accepts 'drug_name_or_id' as an alias that folds "
                "into 'query'. Step 14 fills the official arg 'query'."
            ),
        },
        fallback_description=(
            "Retrieve DrugBank drug references (including patent references) "
            "by drug name or DrugBank ID / free-text query."
        ),
    ),
    # EuropePMC is a literature / prior-art SCIENTIFIC EVIDENCE search (the
    # ToolUniverse tool the Enola workflow used), NOT a patent-number-specific
    # database. Its official identity arg is `query`.
    "EuropePMC_search_articles": Step14ToolSpec(
        tool_name="EuropePMC_search_articles",
        acceptable_supports=("query",),
        supports_to_schema_arg={"query": "query"},
        arg_mapping_notes={},
        fallback_description=(
            "Search Europe PMC scientific literature / prior-art evidence by a "
            "free-text query (literature evidence, not a patent-number lookup)."
        ),
    ),
}


def _step14_schema_and_source(tool_name: str) -> tuple[Optional[dict], str]:
    """Return (official-first schema, schema_source label) for a Step 14 tool.

    Uses the shared `tool_selection_policy` helpers so the schema source is the
    SAME as Step 6 / Step 9: official ToolUniverse parameter schema first, the
    wrapper signature only as a fallback. No live ToolUniverse call — the spec
    comes from the offline TU spec registry / local binding.
    """
    try:
        from ..mcp import tooluniverse_adapter

        spec = tooluniverse_adapter.get_tool_specification(tool_name)
    except Exception:  # noqa: BLE001
        spec = None
    if _normalize_official_schema(tool_name, spec) is not None:
        return signature_schema_for(tool_name), "official_schema"
    schema = signature_schema_for(tool_name)
    if schema is not None:
        return schema, "signature_schema"
    return None, "fallback_binding_signature"


def acceptable_supports_for(tool_name: str) -> set[str]:
    return {s.lower() for s in STEP14_TOOL_SPECS[tool_name].acceptable_supports}


def schema_arg_for_support(tool_name: str, token: str) -> Optional[str]:
    """The official/runtime schema arg a supports token fills (case-insensitive)."""
    spec = STEP14_TOOL_SPECS[tool_name]
    lowered = {k.lower(): v for k, v in spec.supports_to_schema_arg.items()}
    return lowered.get(token.lower())


# Per-tool identity requirement: a tuple of OR-groups. A tool is invocable when
# EVERY group has at least one satisfied schema arg. FDA Orange Book's single
# group encodes its brand_name-OR-application_number semantics even though the
# official schema marks neither as `required`.
_STEP14_IDENTITY_GROUPS: dict[str, tuple[tuple[str, ...], ...]] = {
    "PubChem_get_associated_patents_by_CID": (("cid",),),
    "FDA_OrangeBook_get_patent_info": (("brand_name", "application_number"),),
    "drugbank_get_drug_references_by_drug_name_or_id": (("query",),),
    "EuropePMC_search_articles": (("query",),),
}


def _identity_args(tool_name: str) -> set[str]:
    return {a for group in _STEP14_IDENTITY_GROUPS.get(tool_name, ()) for a in group}


def _identity_missing(tool_name: str, satisfied_args: set[str]) -> list[str]:
    missing: list[str] = []
    for group in _STEP14_IDENTITY_GROUPS.get(tool_name, ()):
        if not (set(group) & satisfied_args):
            missing.append("|".join(group))
    return missing


def _tool_schema_props(tool_name: str) -> set[str]:
    schema, _ = _step14_schema_and_source(tool_name)
    return {str(k) for k in (schema or {}).get("properties") or {}}


def _ref_can_satisfy_schema_arg(
    tool_name: str, supports_tool_args: list[str], schema_arg: str
) -> Optional[str]:
    """Return the support token that lets an input ref fill ``schema_arg`` under
    the Step 14 mapping, or ``None`` if none of its tokens can."""
    for token in supports_tool_args:
        if schema_arg_for_support(tool_name, token) == schema_arg:
            return token
    return None


def _literal_allowed(tool_name: str, schema_arg: str, value: Any) -> bool:
    """Allow a static config literal only when the sourced schema supports it
    (enum/const/default match, or a boolean/integer/number type). Identity args
    are never fillable by a literal — those must be argument_mappings — so the
    LLM can never smuggle a runtime CID / brand name / query text as a literal.
    """
    if schema_arg in _identity_args(tool_name):
        return False
    schema, _ = _step14_schema_and_source(tool_name)
    prop = ((schema or {}).get("properties") or {}).get(schema_arg)
    if not isinstance(prop, dict):
        return False
    if "const" in prop:
        return value == prop["const"]
    if isinstance(prop.get("enum"), list):
        return value in prop["enum"]
    if "default" in prop:
        return value == prop["default"]
    ptype = prop.get("type")
    if ptype == "boolean":
        return isinstance(value, bool)
    if ptype == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if ptype == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    return False


STEP14_SELECTION_SYSTEM_PROMPT = """You plan Step 14 patent / prior-art tool calls.

You choose WHICH catalog tools to run AND map each tool's official
schema arguments to the request's input refs. You never see or invent runtime
values (CIDs, brand names, application numbers, drug / payload / linker text);
you only reference input_ref_ids and the tools' official schema arg names. A
separate runtime resolver reads the real values from storage after you plan.

For every selected tool, output one plan:
- tool_name: one of the catalog tool names.
- can_invoke: true only when the tool's required/identity argument is satisfied
  by argument_mappings.
- argument_mappings: list of {"schema_arg","input_ref_id"} pairs. schema_arg
  MUST be one of the tool's official schema_arg_names. input_ref_id MUST be a
  ref in input_refs whose supports_tool_args can fill that schema_arg.
- argument_literals: list of {"schema_arg","literal_value_json"} pairs, ONLY
  for static schema-supported config args (e.g. booleans / limits).
  literal_value_json is the value encoded as a JSON string, e.g. "25", "true",
  or "\\"some_enum\\"". The runtime decodes it to a real literal_value before
  validation. NEVER put a runtime identity value (cid, brand_name,
  application_number, query text) in a literal — those must be
  argument_mappings.
- missing_required_args: identity/required args you could not satisfy.
- selection_reason: short reason.

Rules:
1. Use only the catalog tools; never invent a tool.
2. Reference only input_ref_ids present in input_refs; never invent one.
3. Use only official schema args (each tool lists schema_arg_names).
4. Never invent runtime values; identity values come only from argument_mappings.
5. No duplicate schema_arg within a plan.
6. PubChem_get_associated_patents_by_CID identity arg is `cid`; refs whose
   supports_tool_args include cid or pubchem_cid fill it.
7. FDA_OrangeBook_get_patent_info is satisfied by `brand_name` OR
   `application_number` (either invokes it, even though official required is
   empty).
8. drugbank_get_drug_references_by_drug_name_or_id identity arg is `query`;
   refs whose supports_tool_args include query or drug_name_or_id fill it.
9. EuropePMC_search_articles is a literature / prior-art SCIENTIFIC EVIDENCE
   search (not a patent-number database); its identity arg is `query`, filled
   by refs whose supports_tool_args include query.
10. An input ref with role=antibody may be used ONLY when
    patent_scope.antibody_search_allowed is true.
11. If a tool cannot be satisfied, either omit it or set can_invoke=false and
    list missing_required_args. Never fabricate a call.
12. Return exactly one JSON object with this shape:
{
  "tool_plans": [
    {
      "tool_name": "one of the catalog tool names",
      "can_invoke": true,
      "argument_mappings": [{"schema_arg": "cid", "input_ref_id": "r_cid"}],
      "argument_literals": [],
      "missing_required_args": [],
      "selection_reason": "short reason"
    }
  ]
}

Example:
input_refs = [{"ref_id":"r_cid","role":"pubchem_cid",
"supports_tool_args":["cid","pubchem_cid"]},
{"ref_id":"r_payload","role":"payload","supports_tool_args":["query"]}].
Output:
{
  "tool_plans": [
    {"tool_name":"PubChem_get_associated_patents_by_CID","can_invoke":true,
     "argument_mappings":[{"schema_arg":"cid","input_ref_id":"r_cid"}],
     "argument_literals":[],"missing_required_args":[],
     "selection_reason":"CID ref satisfies PubChem cid"},
    {"tool_name":"drugbank_get_drug_references_by_drug_name_or_id",
     "can_invoke":true,
     "argument_mappings":[{"schema_arg":"query","input_ref_id":"r_payload"}],
     "argument_literals":[],"missing_required_args":[],
     "selection_reason":"payload ref fills DrugBank query"}
  ]
}

No-usable-tool example:
Output:
{"tool_plans": []}
""".strip()


STEP14_SELECTION_USER_PROMPT = (
    "Plan Step 14 patent tool calls: for each usable tool, map its official "
    "schema args to input_ref_ids. Do not invent tools, refs, schema args, or "
    "runtime values."
)


class Step14ArgumentMapping(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_arg: str
    input_ref_id: str


class Step14ArgumentLiteral(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_arg: str
    literal_value: Any = None


class Step14ToolPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool_name: str
    can_invoke: bool = False
    argument_mappings: list[Step14ArgumentMapping] = Field(default_factory=list)
    argument_literals: list[Step14ArgumentLiteral] = Field(default_factory=list)
    missing_required_args: list[str] = Field(default_factory=list)
    selection_reason: str = ""


class Step14RejectedToolPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool_name: str
    reason: str
    argument_mappings: list[dict] = Field(default_factory=list)


class Step14PlanningResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    catalog_tool_names: list[str] = Field(default_factory=list)
    tool_plans: list[Step14ToolPlan] = Field(default_factory=list)
    rejected_tool_plans: list[Step14RejectedToolPlan] = Field(default_factory=list)
    argument_mapping_audit: list[dict] = Field(default_factory=list)
    selection_source: str = "llm_step14"
    prompt_cache_layout_version: str = STEP14_SELECTION_PROMPT_CACHE_LAYOUT_VERSION


def build_step14_selection_catalog() -> list[dict[str, Any]]:
    """Return the stable compact Step 14 tool catalog (sorted by tool name).

    Each entry's `description`, `official_required_args`, `schema_arg_names`,
    and `schema_source` are sourced from the shared official/registry schema
    helpers (ToolUniverse official schema first, wrapper signature fallback).
    `acceptable_supports` + `supports_to_schema_arg` stay Step-14-owned but are
    aligned to the sourced schema arg names.
    """
    names = sorted(STEP14_TOOL_SPECS)
    official_descriptions = _official_descriptions(names)
    catalog: list[dict[str, Any]] = []
    for tool_name in names:
        spec = STEP14_TOOL_SPECS[tool_name]
        schema, schema_source = _step14_schema_and_source(tool_name)
        properties = (schema or {}).get("properties") or {}
        official_required = list((schema or {}).get("required") or [])
        description = official_descriptions.get(tool_name) or spec.fallback_description
        catalog.append(
            {
                "tool_name": tool_name,
                "description": description,
                "schema_source": schema_source,
                "official_required_args": sorted(official_required),
                "schema_arg_names": sorted(str(k) for k in properties),
                "acceptable_supports": list(spec.acceptable_supports),
                "supports_to_schema_arg": dict(spec.supports_to_schema_arg),
            }
        )
    return catalog


def _compact_input_refs(request: Step14PatentRequest) -> list[dict[str, Any]]:
    """Compact, value-free projection of the request's input refs.

    Only refs/roles/supports_tool_args/source_path/candidate_id are exposed to
    the LLM — never a resolved runtime value.
    """
    out: list[dict[str, Any]] = []
    for ref in request.input_refs:
        out.append(
            {
                "ref_id": ref.ref_id,
                "role": ref.role,
                "source_artifact": ref.source_artifact,
                "source_path": ref.source_path,
                "candidate_id": ref.candidate_id,
                "supports_tool_args": list(ref.supports_tool_args),
            }
        )
    return out


def build_step14_selection_payload(
    *, request: Step14PatentRequest, catalog: list[dict[str, Any]]
) -> dict[str, Any]:
    """Assemble the schema payload for the Step 14 selection LLM call.

    Stable keys: ``task`` / ``tool_catalog``. Dynamic keys: ``user_query`` /
    ``input_refs`` / ``patent_scope`` (split into the prompt's dynamic suffix
    by ``json_task_validation``).
    """
    return {
        "task": "step14_patent_tool_selection",
        "tool_catalog": catalog,
        "user_query": request.user_query or "",
        "input_refs": _compact_input_refs(request),
        "patent_scope": request.patent_scope.model_dump(),
    }


def plan_step14_tool_calls(
    *, llm: LLMProvider, request: Step14PatentRequest
) -> Step14PlanningResult:
    """Single-stage Step 14 planner: one LLM call returns tool plans with
    schema_arg → input_ref_id mappings; this function validates each plan
    against the sourced tool schema and the request's input refs.

    A plan is REJECTED (structural violation, audited compactly) when it
    references an unknown tool / schema_arg / input_ref_id, duplicates a
    schema_arg, uses an antibody ref while antibody search is off, maps an
    input ref that cannot fill the schema_arg, or supplies a disallowed literal.
    A structurally-valid plan whose identity arg is not covered is KEPT with
    ``can_invoke=false`` + ``missing_required_args`` (the runtime will not call
    it). The runtime builds kwargs strictly from the accepted mappings/literals
    — it never re-derives a mapping.
    """
    catalog = build_step14_selection_catalog()
    allowed_tools = set(STEP14_TOOL_SPECS)
    ref_by_id = {ref.ref_id: ref for ref in request.input_refs}
    antibody_allowed = bool(request.patent_scope.antibody_search_allowed)

    payload = build_step14_selection_payload(request=request, catalog=catalog)
    response = llm.generate_json(
        STEP14_SELECTION_USER_PROMPT,
        schema=payload,
        system=STEP14_SELECTION_SYSTEM_PROMPT,
    )

    tool_plans: list[Step14ToolPlan] = []
    rejected: list[Step14RejectedToolPlan] = []
    audit: list[dict] = []

    for i, entry in enumerate((response or {}).get("tool_plans") or []):
        if not isinstance(entry, dict):
            rejected.append(
                Step14RejectedToolPlan(tool_name=f"index:{i}", reason="plan_not_object")
            )
            continue
        tool_name = entry.get("tool_name")
        if not isinstance(tool_name, str) or not tool_name:
            rejected.append(
                Step14RejectedToolPlan(tool_name=f"index:{i}", reason="missing_tool_name")
            )
            continue
        raw_maps = entry.get("argument_mappings") or []
        raw_lits = entry.get("argument_literals") or []
        echo = [
            {"schema_arg": m.get("schema_arg"), "input_ref_id": m.get("input_ref_id")}
            for m in raw_maps
            if isinstance(m, dict)
        ]
        if tool_name not in allowed_tools:
            rejected.append(
                Step14RejectedToolPlan(
                    tool_name=tool_name, reason="unknown_tool", argument_mappings=echo
                )
            )
            continue

        props = _tool_schema_props(tool_name)
        errors: list[str] = []
        seen_args: set[str] = set()
        valid_maps: list[tuple[str, str, str]] = []  # (schema_arg, input_ref_id, token)

        for m in raw_maps:
            if not isinstance(m, dict):
                errors.append("mapping_not_object")
                continue
            arg = m.get("schema_arg")
            rid = m.get("input_ref_id")
            if not isinstance(arg, str) or not arg:
                errors.append("mapping_missing_schema_arg")
                continue
            if not isinstance(rid, str) or not rid:
                errors.append("mapping_missing_input_ref_id")
                continue
            if arg not in props:
                errors.append(f"unknown_schema_arg:{arg}")
                continue
            if arg in seen_args:
                errors.append(f"duplicate_schema_arg:{arg}")
                continue
            if rid not in ref_by_id:
                errors.append(f"unknown_input_ref_id:{rid}")
                continue
            ref = ref_by_id[rid]
            if ref.role == "antibody" and not antibody_allowed:
                errors.append("antibody_search_not_allowed")
                continue
            token = _ref_can_satisfy_schema_arg(tool_name, list(ref.supports_tool_args), arg)
            if token is None:
                errors.append(f"input_ref_cannot_satisfy_schema_arg:{arg}")
                continue
            seen_args.add(arg)
            valid_maps.append((arg, rid, token))

        valid_lits: list[tuple[str, Any]] = []
        for lit in raw_lits:
            if not isinstance(lit, dict):
                errors.append("literal_not_object")
                continue
            arg = lit.get("schema_arg")
            if not isinstance(arg, str) or not arg:
                errors.append("literal_missing_schema_arg")
                continue
            if arg not in props:
                errors.append(f"unknown_schema_arg:{arg}")
                continue
            if arg in seen_args:
                errors.append(f"duplicate_schema_arg:{arg}")
                continue
            # Parser-facing shape encodes the value as a JSON string in
            # `literal_value_json` (OpenAI strict / Gemini / Qwen). The OpenAI
            # provider already decodes it to `literal_value`; the json_object
            # path leaves it as `literal_value_json`, so decode it here. Accept
            # both so no provider needs a different shape.
            if "literal_value_json" in lit:
                raw_json = lit.get("literal_value_json")
                if not isinstance(raw_json, str):
                    errors.append(f"literal_value_json_not_string:{arg}")
                    continue
                try:
                    value = json.loads(raw_json)
                except (TypeError, ValueError):
                    errors.append(f"literal_value_json_invalid:{arg}")
                    continue
            else:
                value = lit.get("literal_value")
            if not _literal_allowed(tool_name, arg, value):
                errors.append(f"literal_not_allowed:{arg}")
                continue
            seen_args.add(arg)
            valid_lits.append((arg, value))

        if errors:
            rejected.append(
                Step14RejectedToolPlan(
                    tool_name=tool_name,
                    reason=";".join(errors),
                    argument_mappings=echo,
                )
            )
            continue

        satisfied_args = {a for a, _, _ in valid_maps} | {a for a, _ in valid_lits}
        missing = _identity_missing(tool_name, satisfied_args)
        # Authoritative can_invoke = identity satisfied, unless the LLM
        # explicitly refused (can_invoke=false) — never fabricated to true.
        can_invoke = (not missing) and (entry.get("can_invoke") is not False)

        plan = Step14ToolPlan(
            tool_name=tool_name,
            can_invoke=can_invoke,
            argument_mappings=[
                Step14ArgumentMapping(schema_arg=a, input_ref_id=r)
                for a, r, _ in valid_maps
            ],
            argument_literals=[
                Step14ArgumentLiteral(schema_arg=a, literal_value=v)
                for a, v in valid_lits
            ],
            missing_required_args=missing,
            selection_reason=str(entry.get("selection_reason") or ""),
        )
        tool_plans.append(plan)
        for a, r, tok in valid_maps:
            audit.append(
                {
                    "tool_name": tool_name,
                    "schema_arg": a,
                    "input_ref_id": r,
                    "satisfied_by_support": tok,
                    "can_invoke": can_invoke,
                }
            )

    return Step14PlanningResult(
        catalog_tool_names=[entry["tool_name"] for entry in catalog],
        tool_plans=tool_plans,
        rejected_tool_plans=rejected,
        argument_mapping_audit=audit,
    )
