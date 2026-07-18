"""Unified Step 13 + Step 14 reference-only tool selector.

The catalog is fail-closed and authoritative: all descriptions and complete
parameter schemas are fetched in one bulk call from ToolUniverse 1.2.2. A
wrapper signature is used only for compatibility auditing and can never
replace or modify official metadata.
"""

from __future__ import annotations

import copy
import inspect
import json
from dataclasses import dataclass
from typing import Any, Literal

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError
from pydantic import BaseModel, ConfigDict, Field

from ..llm.provider import LLMProvider
from ..mcp import tooluniverse_adapter
from ..mcp.outcome import PATENT_EVIDENCE_COMPOSITE_RUNTIME_AVAILABILITY
from ..mcp.tools._registry import _all_bindings
from ..schemas.patent_evidence_request import PatentEvidenceRequest
from ..schemas.patent_evidence_contract import (
    PATENT_EVIDENCE_SCHEMA_ARG_ALLOWED_ROLES,
    PATENT_EVIDENCE_SUPPORT_TO_SCHEMA_ARG,
)


PATENT_EVIDENCE_SELECTION_PROMPT_CACHE_LAYOUT_VERSION = (
    "patent_evidence_selection_v2_lane_assessments"
)


class OfficialMetadataUnavailableError(RuntimeError):
    """Fixed typed error raised when the official catalog cannot be built."""

    def __init__(self, tool_names: list[str]) -> None:
        self.tool_names = sorted(set(tool_names))
        super().__init__("official_metadata_unavailable:" + ",".join(self.tool_names))


class PatentEvidenceSelectionValidationError(RuntimeError):
    """Fail-closed semantic response validation error with compact code only."""


@dataclass(frozen=True)
class _OwnedTool:
    search_lane: str
    execution_step_id: str
    acceptable_supports: tuple[str, ...]
    supports_to_schema_arg: dict[str, str]
    identity_groups: tuple[tuple[str, ...], ...]
    required_ref_roles: tuple[str, ...] = ()


_OWNED_TOOLS: dict[str, _OwnedTool] = {
    "LiteratureSearchTool": _OwnedTool(
        "evidence",
        "step_13",
        ("research_topic", "query"),
        PATENT_EVIDENCE_SUPPORT_TO_SCHEMA_ARG["LiteratureSearchTool"],
        (("research_topic",),),
    ),
    "EuropePMC_search_articles": _OwnedTool(
        "evidence",
        "step_13",
        ("query",),
        PATENT_EVIDENCE_SUPPORT_TO_SCHEMA_ARG["EuropePMC_search_articles"],
        (("query",),),
    ),
    "openalex_search_works": _OwnedTool(
        "evidence",
        "step_13",
        ("query", "search"),
        PATENT_EVIDENCE_SUPPORT_TO_SCHEMA_ARG["openalex_search_works"],
        (("query", "search"),),
    ),
    "PubTator3_LiteratureSearch": _OwnedTool(
        "evidence",
        "step_13",
        ("query",),
        PATENT_EVIDENCE_SUPPORT_TO_SCHEMA_ARG["PubTator3_LiteratureSearch"],
        (("query",),),
    ),
    "PubTator3_get_annotations": _OwnedTool(
        "evidence",
        "step_13",
        ("pmids", "pmid"),
        PATENT_EVIDENCE_SUPPORT_TO_SCHEMA_ARG["PubTator3_get_annotations"],
        (("pmids",),),
    ),
    "SemanticScholar_search_papers": _OwnedTool(
        "evidence",
        "step_13",
        ("query",),
        PATENT_EVIDENCE_SUPPORT_TO_SCHEMA_ARG["SemanticScholar_search_papers"],
        (("query",),),
    ),
    "ChEMBL_search_documents": _OwnedTool(
        "evidence",
        "step_13",
        ("document_id", "title__contains", "title_contains", "title"),
        PATENT_EVIDENCE_SUPPORT_TO_SCHEMA_ARG["ChEMBL_search_documents"],
        (("document_id", "title__contains"),),
    ),
    "MultiAgentLiteratureSearch": _OwnedTool(
        "evidence",
        "step_13",
        ("query",),
        PATENT_EVIDENCE_SUPPORT_TO_SCHEMA_ARG["MultiAgentLiteratureSearch"],
        (("query",),),
    ),
    "PubChem_get_associated_patents_by_CID": _OwnedTool(
        "patent",
        "step_14",
        ("cid", "pubchem_cid"),
        PATENT_EVIDENCE_SUPPORT_TO_SCHEMA_ARG[
            "PubChem_get_associated_patents_by_CID"
        ],
        (("cid",),),
    ),
    "FDA_OrangeBook_get_patent_info": _OwnedTool(
        "patent",
        "step_14",
        ("brand_name", "application_number"),
        PATENT_EVIDENCE_SUPPORT_TO_SCHEMA_ARG["FDA_OrangeBook_get_patent_info"],
        (("brand_name", "application_number"),),
    ),
    "drugbank_get_drug_references_by_drug_name_or_id": _OwnedTool(
        "patent",
        "step_14",
        ("query", "drug_name_or_id"),
        PATENT_EVIDENCE_SUPPORT_TO_SCHEMA_ARG[
            "drugbank_get_drug_references_by_drug_name_or_id"
        ],
        (("query",),),
    ),
}

_MUTUALLY_EXCLUSIVE_SCHEMA_ARG_GROUPS: dict[str, tuple[tuple[str, ...], ...]] = {
    "EuropePMC_search_articles": (("limit", "page_size"),),
    "openalex_search_works": (("query", "search"), ("per_page", "limit")),
}

PATENT_EVIDENCE_TOOL_NAMES = tuple(sorted(_OWNED_TOOLS))

_RUNTIME_AVAILABILITY: dict[str, dict[str, Any]] = {
    tool_name: {
        "status": "available",
        "can_execute": True,
        "reason_code": None,
    }
    for tool_name in PATENT_EVIDENCE_TOOL_NAMES
}
_RUNTIME_AVAILABILITY["drugbank_get_drug_references_by_drug_name_or_id"] = {
    "status": "license_gated",
    "can_execute": False,
    "reason_code": "drugbank_license_required",
}
for _composite_name, _availability in (
    PATENT_EVIDENCE_COMPOSITE_RUNTIME_AVAILABILITY.items()
):
    _RUNTIME_AVAILABILITY[_composite_name] = dict(_availability)

_RUNTIME_CONSTRAINTS: dict[str, list[dict[str, Any]]] = {
    "MultiAgentLiteratureSearch": [
        {
            "schema_arg": "max_iterations",
            "constraint_schema": {"const": 1},
            "reason_code": "max_iterations_runtime_cap_1",
        }
    ]
}

_RUNTIME_EFFECTIVE_DEFAULTS: dict[str, dict[str, Any]] = {
    "MultiAgentLiteratureSearch": {
        "max_iterations": 1,
        "quality_threshold": 0.7,
    }
}


PATENT_EVIDENCE_SELECTION_SYSTEM_PROMPT = """
You plan scientific-evidence and patent/prior-art searches in one selection pass.

Use only the supplied 11-tool catalog. Never invent a tool, input ref, schema
argument, search lane, execution step, or runtime value. For every mapping,
map an official schema_arg to an existing input_ref_id whose
supports_tool_args explicitly supports that mapping. Never generate a CID,
PMID, brand name, application number, drug name, query text, or other runtime
identity value. Those values remain behind references and are resolved later.

Respect each catalog entry's runtime_availability, runtime_constraints,
runtime_effective_defaults, mutually_exclusive_schema_arg_groups, and
schema_arg_allowed_ref_roles. Never plan a tool whose can_execute is false.
MultiAgentLiteratureSearch currently permits max_iterations only as 1 and its
effective quality_threshold default is 0.7; do not emit a conflicting literal.

Scientific evidence results must never be represented as patent records.
EuropePMC_search_articles belongs to scientific evidence and is not a patent-
number database. When requested_lanes contains both evidence and patent, assess
both lanes. If inputs are insufficient, return a missing_inputs or
not_applicable assessment and no tool plan; never fabricate missing input.
PubTator3_get_annotations may be selected only for an existing ref explicitly
typed as role pmid or pmids.
Do not assume another search will later return a PMID and do not fabricate one.

For each requested lane, return exactly one lane_assessment with search_lane,
status (planned, missing_inputs, or not_applicable), and a compact reason.
Never omit, duplicate, or add a lane. planned requires at least one invocable
plan for that lane. missing_inputs and not_applicable must emit no tool plan for
that lane.

The same tool may have multiple plans for different input refs. Do not emit an
exact duplicate tool/ref/schema mapping. Antibody refs are usable only when
antibody_search_allowed is true. You cannot call MCP tools; validated runtime
code may execute accepted plans later.

Return exactly one JSON object:
{"lane_assessments":[{"search_lane":"evidence","status":"planned","reason":"short reason"}],
"tool_plans":[{"tool_name":"catalog tool","can_invoke":true,
"argument_mappings":[{"schema_arg":"official arg","input_ref_id":"existing ref"}],
"argument_literals":[{"schema_arg":"static config arg","literal_value_json":"valid JSON text"}],
"missing_required_args":[],"selection_reason":"short reason"}]}
""".strip()

PATENT_EVIDENCE_SELECTION_USER_PROMPT = (
    "Evaluate every requested search lane against the complete supplied catalog "
    "and return exact lane_assessments plus reference-only tool_plans."
)


def _official_schema(spec: dict[str, Any]) -> dict[str, Any] | None:
    raw = spec.get("parameter") or spec.get("parameters")
    if not isinstance(raw, dict):
        return None
    props = raw.get("properties")
    if not isinstance(props, dict) or not props:
        return None
    schema = copy.deepcopy(raw)
    schema.setdefault("type", "object")
    schema["properties"] = {
        str(k): copy.deepcopy(v)
        for k, v in props.items()
        if isinstance(v, dict) and not str(k).startswith("_")
    }
    declared = schema.get("required")
    required = list(declared) if isinstance(declared, list) else []
    for name, prop in schema["properties"].items():
        if prop.get("required") is True and name not in required:
            required.append(name)
        # ToolUniverse uses property-level required:boolean, which is not
        # standard JSON Schema. Promote it above, then remove it before the
        # official property schema reaches jsonschema validation.
        prop.pop("required", None)
    schema["required"] = [r for r in required if r in schema["properties"]]
    return schema if schema["properties"] else None


def _wrapper_identity_parity(tool_name: str, owned: _OwnedTool) -> bool:
    binding = dict(_all_bindings()).get(tool_name)
    if binding is None:
        return False
    signature = inspect.signature(binding)
    explicit = set(signature.parameters)
    # Every official identity/search arg must be explicitly accepted, or have
    # a declared domain alias that is explicit. **_extra alone never counts.
    for group in owned.identity_groups:
        for official_arg in group:
            aliases = {
                support
                for support, target in owned.supports_to_schema_arg.items()
                if target == official_arg
            }
            if official_arg not in explicit and not (aliases & explicit):
                return False
    return True


def _wrapper_full_schema_acceptance(tool_name: str, schema: dict[str, Any]) -> bool:
    binding = dict(_all_bindings()).get(tool_name)
    if binding is None:
        return False
    explicit = set(inspect.signature(binding).parameters)
    return set(schema.get("properties") or {}) <= explicit


def build_patent_evidence_catalog() -> list[dict[str, Any]]:
    """Build the stable 11-tool catalog exclusively from official specs."""
    specs = tooluniverse_adapter.get_tool_specifications(PATENT_EVIDENCE_TOOL_NAMES)
    unavailable: list[str] = []
    catalog: list[dict[str, Any]] = []
    for tool_name in PATENT_EVIDENCE_TOOL_NAMES:
        spec = specs.get(tool_name)
        description = str((spec or {}).get("description") or "").strip()
        schema = _official_schema(spec or {})
        if (
            not spec
            or spec.get("name") != tool_name
            or not description
            or schema is None
        ):
            unavailable.append(tool_name)
            continue
        owned = _OWNED_TOOLS[tool_name]
        props = schema["properties"]
        if any(arg not in props for arg in owned.supports_to_schema_arg.values()):
            unavailable.append(tool_name)
            continue
        identity_parity = _wrapper_identity_parity(tool_name, owned)
        full_signature_acceptance = _wrapper_full_schema_acceptance(tool_name, schema)
        runtime_availability = copy.deepcopy(_RUNTIME_AVAILABILITY[tool_name])
        full_executable_parity = bool(
            full_signature_acceptance and runtime_availability["can_execute"]
        )
        if not runtime_availability["can_execute"]:
            full_parity_status = (
                "not_executable_" + str(runtime_availability["status"])
            )
        elif full_executable_parity:
            full_parity_status = "verified"
        else:
            full_parity_status = "wrapper_schema_incompatible"
        if not full_signature_acceptance and runtime_availability["can_execute"]:
            runtime_availability = {
                "status": "wrapper_schema_incompatible",
                "can_execute": False,
                "reason_code": "wrapper_full_schema_parity_failed",
            }
        catalog.append(
            {
                "tool_name": tool_name,
                "description": description,
                "full_schema": schema,
                "schema_arg_names": sorted(props),
                "official_required_args": list(schema.get("required") or []),
                "search_lane": owned.search_lane,
                "execution_step_id": owned.execution_step_id,
                "acceptable_supports": list(owned.acceptable_supports),
                "supports_to_schema_arg": dict(owned.supports_to_schema_arg),
                "schema_arg_allowed_ref_roles": {
                    arg: sorted(roles)
                    for arg, roles in PATENT_EVIDENCE_SCHEMA_ARG_ALLOWED_ROLES[
                        tool_name
                    ].items()
                },
                "mutually_exclusive_schema_arg_groups": [
                    list(group)
                    for group in _MUTUALLY_EXCLUSIVE_SCHEMA_ARG_GROUPS.get(
                        tool_name, ()
                    )
                ],
                "metadata_authority": "tooluniverse_official_spec",
                "wrapper_identity_parity": identity_parity,
                "wrapper_full_schema_acceptance": full_signature_acceptance,
                "wrapper_full_executable_schema_parity": full_executable_parity,
                "wrapper_full_executable_schema_parity_status": full_parity_status,
                "runtime_availability": runtime_availability,
                "runtime_constraints": copy.deepcopy(
                    _RUNTIME_CONSTRAINTS.get(tool_name, [])
                ),
                "runtime_effective_defaults": copy.deepcopy(
                    _RUNTIME_EFFECTIVE_DEFAULTS.get(tool_name, {})
                ),
                "required_input_roles": [],
            }
        )
    if unavailable:
        raise OfficialMetadataUnavailableError(unavailable)
    return catalog


class PatentEvidenceArgumentMapping(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_arg: str
    input_ref_id: str


class PatentEvidenceArgumentLiteral(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_arg: str
    literal_value: Any = None


class PatentEvidenceToolPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tool_name: str
    search_lane: str
    execution_step_id: str
    can_invoke: bool = False
    argument_mappings: list[PatentEvidenceArgumentMapping] = Field(default_factory=list)
    argument_literals: list[PatentEvidenceArgumentLiteral] = Field(default_factory=list)
    missing_required_args: list[str] = Field(default_factory=list)
    selection_reason: str = ""


class PatentEvidenceLaneAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid")
    search_lane: Literal["evidence", "patent"]
    status: Literal["planned", "missing_inputs", "not_applicable"]
    reason: str


class PatentEvidenceRejectedPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tool_name: str
    reason: str


class PatentEvidencePlanningResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    catalog_tool_names: list[str]
    tool_plans: list[PatentEvidenceToolPlan] = Field(default_factory=list)
    lane_assessments: list[PatentEvidenceLaneAssessment] = Field(default_factory=list)
    rejected_tool_plans: list[PatentEvidenceRejectedPlan] = Field(default_factory=list)
    selection_audit: list[dict[str, Any]] = Field(default_factory=list)
    llm_call_count: int = 1
    selection_source: str = "llm_patent_evidence"
    prompt_cache_layout_version: str = (
        PATENT_EVIDENCE_SELECTION_PROMPT_CACHE_LAYOUT_VERSION
    )


def _compact_refs(request: PatentEvidenceRequest) -> list[dict[str, Any]]:
    return [
        {
            "ref_id": ref.ref_id,
            "source_artifact": ref.source_artifact,
            "source_path": ref.source_path,
            "role": ref.role,
            "candidate_id": ref.candidate_id,
            "supports_tool_args": list(ref.supports_tool_args),
        }
        for ref in request.input_refs
    ]


def build_patent_evidence_selection_payload(
    *, request: PatentEvidenceRequest, catalog: list[dict[str, Any]]
) -> dict[str, Any]:
    return {
        "task": "patent_evidence_tool_selection",
        "prompt_cache_layout_version": PATENT_EVIDENCE_SELECTION_PROMPT_CACHE_LAYOUT_VERSION,
        "tool_catalog": catalog,
        "user_query": request.user_query or "",
        "input_refs": _compact_refs(request),
        "search_scope": request.search_scope.model_dump(),
    }


def _json_schema_literal_error(schema: dict[str, Any], value: Any) -> str | None:
    """Validate one literal without ever returning or logging its value."""
    try:
        Draft202012Validator.check_schema(schema)
        error = next(Draft202012Validator(schema).iter_errors(value), None)
    except SchemaError:
        return "invalid_official_property_schema"
    if error is None:
        return None
    return f"json_schema_{error.validator or 'validation'}"


def plan_patent_evidence_tool_calls(
    *, llm: LLMProvider, request: PatentEvidenceRequest
) -> PatentEvidencePlanningResult:
    """Make exactly one LLM call, then deterministically validate every plan."""
    catalog = build_patent_evidence_catalog()
    by_tool = {entry["tool_name"]: entry for entry in catalog}
    ref_by_id = {ref.ref_id: ref for ref in request.input_refs}
    requested_lanes = set(request.search_scope.requested_lanes)
    allowed_roles = set(request.search_scope.allowed_roles)
    payload = build_patent_evidence_selection_payload(request=request, catalog=catalog)
    response = llm.generate_json(
        PATENT_EVIDENCE_SELECTION_USER_PROMPT,
        schema=payload,
        system=PATENT_EVIDENCE_SELECTION_SYSTEM_PROMPT,
    )

    accepted: list[PatentEvidenceToolPlan] = []
    rejected: list[PatentEvidenceRejectedPlan] = []
    audit: list[dict[str, Any]] = []
    exact_seen: set[str] = set()
    raw_known_plan_lanes: set[str] = set()
    for index, raw in enumerate((response or {}).get("tool_plans") or []):
        tool_name = raw.get("tool_name") if isinstance(raw, dict) else f"index:{index}"
        if not isinstance(raw, dict):
            rejected.append(
                PatentEvidenceRejectedPlan(
                    tool_name=tool_name, reason="plan_not_object"
                )
            )
            continue
        if tool_name not in by_tool:
            rejected.append(
                PatentEvidenceRejectedPlan(
                    tool_name=str(tool_name), reason="unknown_tool"
                )
            )
            continue
        entry = by_tool[tool_name]
        lane = entry["search_lane"]
        raw_known_plan_lanes.add(lane)
        if lane not in requested_lanes:
            rejected.append(
                PatentEvidenceRejectedPlan(
                    tool_name=tool_name, reason="tool_not_in_requested_lane"
                )
            )
            continue
        runtime_availability = entry["runtime_availability"]
        if not runtime_availability.get("can_execute"):
            rejected.append(
                PatentEvidenceRejectedPlan(
                    tool_name=tool_name,
                    reason=f"runtime_unavailable:{runtime_availability.get('status')}",
                )
            )
            continue
        props = entry["full_schema"]["properties"]
        owned = _OWNED_TOOLS[tool_name]
        errors: list[str] = []
        seen_args: set[str] = set()
        mappings: list[PatentEvidenceArgumentMapping] = []
        literals: list[PatentEvidenceArgumentLiteral] = []
        for mapping in raw.get("argument_mappings") or []:
            if not isinstance(mapping, dict):
                errors.append("mapping_not_object")
                continue
            arg, ref_id = mapping.get("schema_arg"), mapping.get("input_ref_id")
            if arg not in props:
                errors.append(f"unknown_schema_arg:{arg}")
                continue
            if arg in seen_args:
                errors.append(f"duplicate_schema_arg:{arg}")
                continue
            ref = ref_by_id.get(ref_id)
            if ref is None:
                errors.append(f"unknown_input_ref_id:{ref_id}")
                continue
            if (
                ref.role == "antibody"
                and not request.search_scope.antibody_search_allowed
            ):
                errors.append("antibody_search_not_allowed")
                continue
            if ref.role not in allowed_roles:
                errors.append(f"role_not_allowed:{ref.role}")
                continue
            supported = {
                owned.supports_to_schema_arg.get(token.lower())
                for token in ref.supports_tool_args
            }
            if arg not in supported:
                errors.append(f"input_ref_cannot_satisfy_schema_arg:{arg}")
                continue
            allowed_arg_roles = set(
                entry["schema_arg_allowed_ref_roles"].get(arg) or []
            )
            if ref.role not in allowed_arg_roles:
                errors.append(f"ref_role_not_allowed_for_schema_arg:{arg}")
                continue
            seen_args.add(arg)
            mappings.append(
                PatentEvidenceArgumentMapping(schema_arg=arg, input_ref_id=ref_id)
            )
        for literal in raw.get("argument_literals") or []:
            if not isinstance(literal, dict):
                errors.append("literal_not_object")
                continue
            arg = literal.get("schema_arg")
            if arg not in props:
                errors.append(f"unknown_schema_arg:{arg}")
                continue
            if arg in seen_args:
                errors.append(f"duplicate_schema_arg:{arg}")
                continue
            if arg in {a for group in owned.identity_groups for a in group}:
                errors.append(f"identity_literal_not_allowed:{arg}")
                continue
            try:
                value = (
                    json.loads(literal["literal_value_json"])
                    if "literal_value_json" in literal
                    else literal.get("literal_value")
                )
            except (TypeError, ValueError):
                errors.append(f"invalid_literal_json:{arg}")
                continue
            schema_error = _json_schema_literal_error(props[arg], value)
            if schema_error is not None:
                errors.append(f"literal_schema_invalid:{arg}:{schema_error}")
                continue
            runtime_constraint = next(
                (
                    constraint
                    for constraint in entry.get("runtime_constraints") or []
                    if constraint.get("schema_arg") == arg
                ),
                None,
            )
            if (
                runtime_constraint is not None
                and _json_schema_literal_error(
                    runtime_constraint["constraint_schema"], value
                )
                is not None
            ):
                errors.append(f"runtime_constraint_violation:{arg}")
                continue
            seen_args.add(arg)
            literals.append(
                PatentEvidenceArgumentLiteral(schema_arg=arg, literal_value=value)
            )
        for group in entry["mutually_exclusive_schema_arg_groups"]:
            if len(set(group) & seen_args) > 1:
                errors.append(
                    "mutually_exclusive_schema_args:" + "|".join(group)
                )
        if raw.get("can_invoke") is not True:
            errors.append("can_invoke_must_be_true_for_tool_plan")
        if errors:
            rejected.append(
                PatentEvidenceRejectedPlan(tool_name=tool_name, reason=";".join(errors))
            )
            continue
        satisfied = set(seen_args)
        missing: list[str] = []
        for group in owned.identity_groups:
            if not (set(group) & satisfied):
                missing.append("|".join(group))
        for arg in entry["official_required_args"]:
            if (
                arg not in satisfied
                and "default" not in props[arg]
                and arg not in {a for group in owned.identity_groups for a in group}
            ):
                missing.append(arg)
        signature = json.dumps(
            {
                "tool": tool_name,
                "mappings": sorted((m.schema_arg, m.input_ref_id) for m in mappings),
                "literals": sorted(
                    (literal.schema_arg, literal.literal_value) for literal in literals
                ),
            },
            sort_keys=True,
            default=str,
        )
        if signature in exact_seen:
            rejected.append(
                PatentEvidenceRejectedPlan(tool_name=tool_name, reason="duplicate_plan")
            )
            continue
        exact_seen.add(signature)
        can_invoke = not missing
        plan = PatentEvidenceToolPlan(
            tool_name=tool_name,
            search_lane=lane,
            execution_step_id=entry["execution_step_id"],
            can_invoke=can_invoke,
            argument_mappings=mappings,
            argument_literals=literals,
            missing_required_args=missing,
            selection_reason=str(raw.get("selection_reason") or ""),
        )
        accepted.append(plan)
        audit.append(
            {
                "tool_name": tool_name,
                "search_lane": lane,
                "execution_step_id": entry["execution_step_id"],
                "can_invoke": can_invoke,
            }
        )

    raw_assessments = (response or {}).get("lane_assessments")
    if not isinstance(raw_assessments, list):
        raise PatentEvidenceSelectionValidationError("lane_assessments_missing")
    assessment_by_lane: dict[str, PatentEvidenceLaneAssessment] = {}
    for raw in raw_assessments:
        if not isinstance(raw, dict):
            raise PatentEvidenceSelectionValidationError("lane_assessment_not_object")
        lane = raw.get("search_lane")
        status = raw.get("status")
        reason = raw.get("reason")
        if lane not in {"evidence", "patent"}:
            raise PatentEvidenceSelectionValidationError("lane_assessment_unknown_lane")
        if lane in assessment_by_lane:
            raise PatentEvidenceSelectionValidationError(
                "lane_assessment_duplicate_lane"
            )
        if lane not in requested_lanes:
            raise PatentEvidenceSelectionValidationError(
                "lane_assessment_unrequested_lane"
            )
        if status not in {"planned", "missing_inputs", "not_applicable"}:
            raise PatentEvidenceSelectionValidationError(
                "lane_assessment_invalid_status"
            )
        if (
            not isinstance(reason, str)
            or not reason.strip()
            or len(reason) > 300
            or any(ord(ch) < 32 and ch not in "\t\n\r" for ch in reason)
        ):
            raise PatentEvidenceSelectionValidationError(
                "lane_assessment_invalid_reason"
            )
        assessment_by_lane[lane] = PatentEvidenceLaneAssessment(
            search_lane=lane,
            status=status,
            reason=reason.strip(),
        )
    if set(assessment_by_lane) != requested_lanes:
        raise PatentEvidenceSelectionValidationError(
            "lane_assessment_requested_set_mismatch"
        )
    invocable_lanes = {plan.search_lane for plan in accepted if plan.can_invoke}
    for lane in request.search_scope.requested_lanes:
        assessment = assessment_by_lane[lane]
        if assessment.status == "planned" and lane not in invocable_lanes:
            raise PatentEvidenceSelectionValidationError(
                "lane_assessment_planned_without_accepted_plan"
            )
        if assessment.status != "planned" and (
            lane in invocable_lanes or lane in raw_known_plan_lanes
        ):
            raise PatentEvidenceSelectionValidationError(
                "lane_assessment_nonplanned_with_tool_plan"
            )
    return PatentEvidencePlanningResult(
        catalog_tool_names=list(PATENT_EVIDENCE_TOOL_NAMES),
        tool_plans=accepted,
        lane_assessments=[
            assessment_by_lane[lane] for lane in request.search_scope.requested_lanes
        ],
        rejected_tool_plans=rejected,
        selection_audit=audit,
    )
