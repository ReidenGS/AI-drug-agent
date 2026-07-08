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


STEP14_SELECTION_PROMPT_CACHE_LAYOUT_VERSION = "step14_selection_v2"


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


STEP14_SELECTION_SYSTEM_PROMPT = """You select Step 14 patent-search tools.

You only choose WHICH patent tools to run. You never resolve real values and
never construct raw tool arguments; a separate runtime resolver builds real
arguments from the selected input refs.

Rules:
1. Select only from the given tool catalog (exactly the 3 allowed tools).
2. Refer only to input_ref_ids present in the request's input_refs. Never
   invent an input_ref_id.
3. Never invent runtime values (CIDs, brand names, drug names, payload text).
4. PubChem_get_associated_patents_by_CID requires an input ref whose
   supports_tool_args includes cid or pubchem_cid.
5. FDA_OrangeBook_get_patent_info requires an input ref whose
   supports_tool_args includes brand_name or application_number.
6. drugbank_get_drug_references_by_drug_name_or_id requires an input ref whose
   supports_tool_args includes drug_name_or_id or query.
7. Text roles (payload, linker, linker_payload, compound, target,
   complete_adc) may drive DrugBank or FDA text lookup ONLY when their
   supports_tool_args include query / drug_name_or_id / brand_name.
8. An input ref with role=antibody may only be used when
   patent_scope.antibody_search_allowed is true. Otherwise do not select it.
9. If no valid tool can be selected, return an empty selected_tool_plans list.
   Never fabricate a fallback call.
10. Return exactly one valid JSON object with this shape:
{
  "selected_tool_plans": [
    {
      "tool_name": "one of the catalog tool names",
      "input_ref_ids": ["ref id from input_refs"],
      "selection_reason": "short reason",
      "missing_required_args": []
    }
  ]
}

Example:
Input situation: input_refs contains {"ref_id": "r1", "role": "pubchem_cid",
"supports_tool_args": ["cid", "pubchem_cid"]} and {"ref_id": "r2",
"role": "payload", "supports_tool_args": ["query"]}.
Output:
{
  "selected_tool_plans": [
    {
      "tool_name": "PubChem_get_associated_patents_by_CID",
      "input_ref_ids": ["r1"],
      "selection_reason": "r1 supports cid",
      "missing_required_args": []
    },
    {
      "tool_name": "drugbank_get_drug_references_by_drug_name_or_id",
      "input_ref_ids": ["r2"],
      "selection_reason": "r2 supports query text lookup",
      "missing_required_args": []
    }
  ]
}

No-valid-tool example:
Output:
{"selected_tool_plans": []}
""".strip()


STEP14_SELECTION_USER_PROMPT = (
    "Select Step 14 patent tools from the catalog using only the input refs' "
    "roles and supports_tool_args. Do not construct arguments or invent values."
)


class Step14SelectedToolPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool_name: str
    input_ref_ids: list[str] = Field(default_factory=list)
    selection_reason: str = ""
    missing_required_args: list[str] = Field(default_factory=list)


class Step14RejectedToolPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool_name: str
    input_ref_ids: list[str] = Field(default_factory=list)
    reason: str


class Step14SelectionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    catalog_tool_names: list[str] = Field(default_factory=list)
    selected_tool_plans: list[Step14SelectedToolPlan] = Field(default_factory=list)
    rejected_tool_plans: list[Step14RejectedToolPlan] = Field(default_factory=list)
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


def select_step14_tool_plans(
    *, llm: LLMProvider, request: Step14PatentRequest
) -> Step14SelectionResult:
    """Run Step 14 tool selection and validate the plans against the request.

    Validation drops/records: unknown tools, unknown input_ref_ids,
    unsatisfiable plans (no selected ref satisfies the tool's required args),
    and antibody plans when ``antibody_search_allowed`` is false.
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

    selected: list[Step14SelectedToolPlan] = []
    rejected: list[Step14RejectedToolPlan] = []

    for i, entry in enumerate((response or {}).get("selected_tool_plans") or []):
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
        raw_ref_ids = entry.get("input_ref_ids") or []
        ref_ids = [r for r in raw_ref_ids if isinstance(r, str) and r]
        if tool_name not in allowed_tools:
            rejected.append(
                Step14RejectedToolPlan(
                    tool_name=tool_name, input_ref_ids=ref_ids, reason="unknown_tool"
                )
            )
            continue

        # Unknown input_ref_ids: reject the whole plan (never invent refs).
        unknown = [r for r in ref_ids if r not in ref_by_id]
        if unknown:
            rejected.append(
                Step14RejectedToolPlan(
                    tool_name=tool_name,
                    input_ref_ids=ref_ids,
                    reason=f"unknown_input_ref_ids:{','.join(unknown)}",
                )
            )
            continue

        refs = [ref_by_id[r] for r in ref_ids]

        # Antibody gate: any antibody ref is rejected unless explicitly allowed.
        antibody_refs = [ref for ref in refs if ref.role == "antibody"]
        if antibody_refs and not antibody_allowed:
            rejected.append(
                Step14RejectedToolPlan(
                    tool_name=tool_name,
                    input_ref_ids=ref_ids,
                    reason="antibody_search_not_allowed",
                )
            )
            continue

        # Required-arg satisfaction: at least one selected ref must support the
        # tool's required/primary argument.
        acceptable = acceptable_supports_for(tool_name)
        satisfying = [
            ref
            for ref in refs
            if {s.lower() for s in ref.supports_tool_args} & acceptable
        ]
        if not satisfying:
            rejected.append(
                Step14RejectedToolPlan(
                    tool_name=tool_name,
                    input_ref_ids=ref_ids,
                    reason="no_input_ref_satisfies_required_args",
                )
            )
            continue

        selected.append(
            Step14SelectedToolPlan(
                tool_name=tool_name,
                input_ref_ids=ref_ids,
                selection_reason=str(entry.get("selection_reason") or ""),
                missing_required_args=[
                    str(a)
                    for a in (entry.get("missing_required_args") or [])
                    if isinstance(a, str)
                ],
            )
        )

    return Step14SelectionResult(
        catalog_tool_names=[entry["tool_name"] for entry in catalog],
        selected_tool_plans=selected,
        rejected_tool_plans=rejected,
    )
