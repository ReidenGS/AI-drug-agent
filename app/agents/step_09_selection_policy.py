"""Step 9 LLM relevance selection and schema mapping.

These selectors are audit-only for the current Step 9 iteration. Stage 1 asks
which hard-gate-allowed Step 9 tools are relevant. Stage 2 maps selected tool
schemas to available field refs / official schema literals. Neither stage
executes Step 9 protein/variant tools.
"""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ..llm.provider import LLMProvider
from ..schemas.step_09_structure_variant_and_compound_screening import (
    Step9AvailableField,
    Step9HardGateAllowedTool,
    Step9LaneStatus,
    Step9ToolSchemaRequirement,
)
from .tool_selection_policy import CAPABILITY_REGISTRY
from .tool_selection_policy import signature_schema_for


STEP9_STAGE1_PROMPT_CACHE_LAYOUT_VERSION = "step9_stage1_v1"
STEP9_STAGE2_PROMPT_CACHE_LAYOUT_VERSION = "step9_stage2_v1"

STEP9_STAGE1_SYSTEM_PROMPT = """You select relevant Step 9 tools.

Rules:
1. Select only from the allowed catalog.
2. Do not select blocked tools.
3. Do not construct tool arguments.
4. Do not invent variants, mutations, contigs, sequences, PDB IDs, structures, compounds, thresholds, or tiers.
5. Avoid redundant tools; if tools answer the same question, choose the best one.
6. Return exactly one valid JSON object matching the requested shape.
""".strip()


STEP9_STAGE1_USER_PROMPT = (
    "Select relevant Step 9 tools from the allowed catalog. Return only "
    "tool_name, lane_type, and selection_reason; do not construct arguments."
)


STEP9_STAGE2_SYSTEM_PROMPT = """You map selected Step 9 tool schemas to candidate field refs.

Rules:
1. Use only selected tools.
2. Use only official full_schema / required_fields and candidate available fields.
3. Do not output raw values.
4. Do not invent variants, mutations, contigs, thresholds, tiers, PDB IDs, structures, sequences, or compounds.
5. Output schema_arg -> field_ref using list-of-pairs.
6. Use argument_literals only for official enum/singleton/default-like schema literals.
7. If required args cannot be satisfied, can_invoke=false and list missing_required_fields.
8. Return exactly one valid JSON object matching the requested shape.
""".strip()


STEP9_STAGE2_USER_PROMPT = (
    "Map selected Step 9 tool schemas to candidate field refs and official "
    "schema literals. Return list-of-pairs only."
)


class Step9Stage1SelectionAudit(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool_name: str
    lane_type: str
    selection_reason: str = ""


class Step9Stage1SelectionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    catalog_tool_names: list[str] = Field(default_factory=list)
    selected_tools: list[Step9Stage1SelectionAudit] = Field(default_factory=list)
    rejected_tools_with_reason: list[dict[str, str]] = Field(default_factory=list)
    selection_source: str = "llm_stage1"
    prompt_cache_layout_version: str = STEP9_STAGE1_PROMPT_CACHE_LAYOUT_VERSION


class Step9Stage2ArgumentMapping(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_arg: str
    field_ref: str


class Step9Stage2ArgumentLiteral(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_arg: str
    literal_value: str | int | float | bool | None = None


class Step9Stage2MappedTool(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool_name: str
    lane_type: str
    can_invoke: bool
    argument_mappings: list[Step9Stage2ArgumentMapping] = Field(default_factory=list)
    argument_literals: list[Step9Stage2ArgumentLiteral] = Field(default_factory=list)
    missing_required_fields: list[str] = Field(default_factory=list)
    skip_reason: str = ""
    argument_mapping_reason: str = ""


class Step9Stage2MappingResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_survivors: list[str] = Field(default_factory=list)
    mapped_tools: list[Step9Stage2MappedTool] = Field(default_factory=list)
    uninvokable_tools: list[str] = Field(default_factory=list)
    uninvokable_tool_details: list[dict[str, Any]] = Field(default_factory=list)
    argument_mapping_audit: list[dict[str, Any]] = Field(default_factory=list)
    prompt_cache_layout_version: str = STEP9_STAGE2_PROMPT_CACHE_LAYOUT_VERSION


def build_step9_stage1_catalog(
    allowed_tools: list[Step9HardGateAllowedTool],
    schema_requirements: list[Step9ToolSchemaRequirement],
) -> list[dict[str, Any]]:
    """Build a stable allowed-only catalog, sorted by lane then tool."""

    req_by_tool_lane: dict[tuple[str, str], Step9ToolSchemaRequirement] = {}
    for req in schema_requirements:
        if req.hard_gate_decision != "allowed":
            continue
        key = (req.tool_name, req.lane_type)
        req_by_tool_lane.setdefault(key, req)

    seen: set[tuple[str, str]] = set()
    catalog: list[dict[str, Any]] = []
    for tool in sorted(allowed_tools, key=lambda t: (t.lane_type, t.tool_name)):
        key = (tool.tool_name, tool.lane_type)
        if key in seen:
            continue
        seen.add(key)
        meta = CAPABILITY_REGISTRY.get(tool.tool_name) or {}
        req = req_by_tool_lane.get(key)
        catalog.append(
            {
                "tool_name": tool.tool_name,
                "lane_type": tool.lane_type,
                "short_description": str(
                    meta.get("short_description") or tool.tool_name.replace("_", " ")
                ),
                "required_fields": list(req.required_fields if req else []),
                "schema_source": str(req.schema_source if req else "unavailable"),
                "capability_tags": list(meta.get("capability_tags") or []),
                "purpose": str(tool.rationale or ""),
            }
        )
    return catalog


def build_step9_stage1_payload(
    *,
    candidate_id: str,
    catalog: list[dict[str, Any]],
    readiness_projection: dict[str, Any],
    canonical_query: str = "",
    raw_user_query: str = "",
    step8_downstream_handoff_status: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Assemble the schema payload for the Step 9 Stage 1 LLM call."""

    return {
        "task": "step9_tool_selection_stage_1",
        "compact_catalog": catalog,
        "candidate_id": candidate_id,
        "lane_readiness_status": _compact_lane_statuses(
            readiness_projection.get("step9_lane_statuses") or []
        ),
        "step9_available_fields": _model_dump_list(
            readiness_projection.get("step9_available_fields") or []
        ),
        "step9_tool_schema_requirements": _compact_schema_requirements(
            readiness_projection.get("step9_tool_schema_requirements") or []
        ),
        "step9_hard_gate_allowed_tool_names": [entry["tool_name"] for entry in catalog],
        "blocked_summary": _compact_blocked_summary(
            readiness_projection.get("step9_lane_statuses") or []
        ),
        "user_intent_summary": {
            "canonical_query": _compact_text(canonical_query),
            "raw_user_query": _compact_text(raw_user_query),
        },
        "step8_downstream_handoff_status": step8_downstream_handoff_status or [],
        "candidate_context_refs": _candidate_context_refs(
            readiness_projection.get("step9_available_fields") or []
        ),
    }


def select_step9_stage1_tools(
    *,
    llm: LLMProvider,
    readiness_projection: dict[str, Any],
    candidate_id: str,
    canonical_query: str = "",
    raw_user_query: str = "",
    step8_downstream_handoff_status: list[dict[str, Any]] | None = None,
) -> Step9Stage1SelectionResult:
    """Run Stage 1 and validate selections against the hard-gate catalog."""

    catalog = build_step9_stage1_catalog(
        readiness_projection.get("step9_hard_gate_allowed_tools") or [],
        readiness_projection.get("step9_tool_schema_requirements") or [],
    )
    allowed = {(entry["tool_name"], entry["lane_type"]) for entry in catalog}
    allowed_names = {entry["tool_name"] for entry in catalog}
    payload = build_step9_stage1_payload(
        candidate_id=candidate_id,
        catalog=catalog,
        readiness_projection=readiness_projection,
        canonical_query=canonical_query,
        raw_user_query=raw_user_query,
        step8_downstream_handoff_status=step8_downstream_handoff_status,
    )
    response = llm.generate_json(
        STEP9_STAGE1_USER_PROMPT,
        schema=payload,
        system=STEP9_STAGE1_SYSTEM_PROMPT,
    )

    selected: list[Step9Stage1SelectionAudit] = []
    rejected: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for i, entry in enumerate((response or {}).get("selections") or []):
        if not isinstance(entry, dict):
            rejected.append({"tool_name": f"index:{i}", "reason": "selection_not_object"})
            continue
        tool_name = entry.get("tool_name")
        lane_type = entry.get("lane_type")
        if not isinstance(tool_name, str) or not tool_name:
            rejected.append({"tool_name": f"index:{i}", "reason": "missing_tool_name"})
            continue
        if not isinstance(lane_type, str) or not lane_type:
            rejected.append({"tool_name": tool_name, "reason": "missing_lane_type"})
            continue
        key = (tool_name, lane_type)
        if key not in allowed:
            reason = (
                "tool_not_in_allowed_catalog"
                if tool_name not in allowed_names
                else "tool_lane_not_in_allowed_catalog"
            )
            rejected.append({"tool_name": tool_name, "reason": reason})
            continue
        if key in seen:
            rejected.append({"tool_name": tool_name, "reason": "duplicate_selection"})
            continue
        seen.add(key)
        selected.append(
            Step9Stage1SelectionAudit(
                tool_name=tool_name,
                lane_type=lane_type,
                selection_reason=str(entry.get("selection_reason") or ""),
            )
        )

    return Step9Stage1SelectionResult(
        catalog_tool_names=[entry["tool_name"] for entry in catalog],
        selected_tools=selected,
        rejected_tools_with_reason=rejected,
        selection_source="llm_stage1",
    )


def build_step9_stage2_payload(
    *,
    candidate_id: str,
    selected_tools: list[Step9Stage1SelectionAudit] | list[dict[str, Any]],
    readiness_projection: dict[str, Any],
    canonical_query: str = "",
    raw_user_query: str = "",
    step8_downstream_handoff_status: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Assemble the Stage 2 schema payload.

    The stable selected-tool schema block is ``tools``. Candidate-specific
    field refs and Stage-1 reasons stay as dynamic payload keys; the shared
    prompt renderer splits them accordingly.
    """

    selected = [_selection_dict(item) for item in selected_tools]
    selected_pairs = {
        (item["tool_name"], item["lane_type"])
        for item in selected
        if item.get("tool_name") and item.get("lane_type")
    }
    reqs = _compact_schema_requirements(
        readiness_projection.get("step9_tool_schema_requirements") or []
    )
    req_by_pair = {
        (str(req.get("tool_name") or ""), str(req.get("lane_type") or "")): req
        for req in reqs
    }
    tools: list[dict[str, Any]] = []
    for tool_name, lane_type in sorted(selected_pairs, key=lambda p: (p[1], p[0])):
        req = req_by_pair.get((tool_name, lane_type), {})
        schema = signature_schema_for(tool_name) or _schema_from_requirement(req)
        tools.append(
            {
                "tool_name": tool_name,
                "lane_type": lane_type,
                "full_schema": _compact_schema(schema),
                "required_fields": list(req.get("required_fields") or schema.get("required") or []),
                "schema_source": str(req.get("schema_source") or "unavailable"),
            }
        )

    return {
        "task": "step9_tool_schema_mapping_stage_2",
        "candidate_id": candidate_id,
        "tools": tools,
        "selected_tools": selected,
        "step9_available_fields": _model_dump_list(
            readiness_projection.get("step9_available_fields") or []
        ),
        "step9_tool_schema_requirements": [
            req for req in reqs if (req.get("tool_name"), req.get("lane_type")) in selected_pairs
        ],
        "lane_readiness_status": _compact_lane_statuses(
            readiness_projection.get("step9_lane_statuses") or []
        ),
        "step8_downstream_handoff_status": step8_downstream_handoff_status or [],
        "user_intent_summary": {
            "canonical_query": _compact_text(canonical_query),
            "raw_user_query": _compact_text(raw_user_query),
        },
        "candidate_context_refs": _candidate_context_refs(
            readiness_projection.get("step9_available_fields") or []
        ),
    }


def select_step9_stage2_mappings(
    *,
    llm: LLMProvider,
    readiness_projection: dict[str, Any],
    selected_tools: list[Step9Stage1SelectionAudit],
    candidate_id: str,
    canonical_query: str = "",
    raw_user_query: str = "",
    step8_downstream_handoff_status: list[dict[str, Any]] | None = None,
) -> Step9Stage2MappingResult:
    payload = build_step9_stage2_payload(
        candidate_id=candidate_id,
        selected_tools=selected_tools,
        readiness_projection=readiness_projection,
        canonical_query=canonical_query,
        raw_user_query=raw_user_query,
        step8_downstream_handoff_status=step8_downstream_handoff_status,
    )
    stage2_tools = [tool for tool in payload["tools"] if isinstance(tool, dict)]
    schema_survivors = [str(tool.get("tool_name") or "") for tool in stage2_tools]
    if not stage2_tools:
        return Step9Stage2MappingResult(schema_survivors=[])

    try:
        response = llm.generate_json(
            STEP9_STAGE2_USER_PROMPT,
            schema=payload,
            system=STEP9_STAGE2_SYSTEM_PROMPT,
        )
        response_tools = [
            item for item in (response or {}).get("tools") or [] if isinstance(item, dict)
        ]
    except Exception as exc:  # noqa: BLE001
        response_tools = [
            {
                "tool_name": tool.get("tool_name"),
                "lane_type": tool.get("lane_type"),
                "can_invoke": False,
                "argument_mappings": [],
                "argument_literals": [],
                "missing_required_fields": list(tool.get("required_fields") or []),
                "skip_reason": f"stage2_provider_or_parse_error:{type(exc).__name__}",
                "argument_mapping_reason": "Stage 2 provider/parse error; no mapping trusted",
            }
            for tool in stage2_tools
        ]

    response_by_pair: dict[tuple[str, str], dict[str, Any]] = {}
    for item in response_tools:
        key = (str(item.get("tool_name") or ""), str(item.get("lane_type") or ""))
        response_by_pair.setdefault(key, item)

    fields = _model_dump_list(readiness_projection.get("step9_available_fields") or [])
    mapped: list[Step9Stage2MappedTool] = []
    uninvokable: list[str] = []
    details: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    for tool in stage2_tools:
        key = (str(tool.get("tool_name") or ""), str(tool.get("lane_type") or ""))
        response_item = response_by_pair.get(key) or {
            "tool_name": key[0],
            "lane_type": key[1],
            "can_invoke": False,
            "argument_mappings": [],
            "argument_literals": [],
            "missing_required_fields": list(tool.get("required_fields") or []),
            "skip_reason": "stage2_omitted_selected_tool",
            "argument_mapping_reason": "Stage 2 omitted selected tool",
        }
        validated = validate_step9_stage2_mapping(
            response_item=response_item,
            selected_tool=tool,
            available_fields=fields,
        )
        mapped.append(validated)
        audit.extend(_mapping_audit_entries(validated))
        if not validated.can_invoke:
            uninvokable.append(validated.tool_name)
            details.append(
                {
                    "tool_name": validated.tool_name,
                    "lane_type": validated.lane_type,
                    "missing_required_fields": list(validated.missing_required_fields),
                    "skip_reason": validated.skip_reason,
                }
            )

    return Step9Stage2MappingResult(
        schema_survivors=schema_survivors,
        mapped_tools=mapped,
        uninvokable_tools=uninvokable,
        uninvokable_tool_details=details,
        argument_mapping_audit=audit,
    )


def validate_step9_stage2_mapping(
    *,
    response_item: dict[str, Any],
    selected_tool: dict[str, Any],
    available_fields: list[dict[str, Any]],
) -> Step9Stage2MappedTool:
    tool_name = str(selected_tool.get("tool_name") or response_item.get("tool_name") or "")
    lane_type = str(selected_tool.get("lane_type") or response_item.get("lane_type") or "")
    schema = selected_tool.get("full_schema") if isinstance(selected_tool.get("full_schema"), dict) else {}
    properties = schema.get("properties") if isinstance(schema.get("properties"), dict) else {}
    required = [str(arg) for arg in (selected_tool.get("required_fields") or schema.get("required") or [])]

    valid_mappings: list[Step9Stage2ArgumentMapping] = []
    valid_literals: list[Step9Stage2ArgumentLiteral] = []
    missing: set[str] = {
        str(arg) for arg in (response_item.get("missing_required_fields") or []) if isinstance(arg, str)
    }
    field_refs = {str(field.get("field_ref") or ""): field for field in available_fields}
    seen_args: set[str] = set()
    warnings: list[str] = []

    for pair in response_item.get("argument_mappings") or []:
        if not isinstance(pair, dict):
            warnings.append("argument_mapping_not_object")
            continue
        arg = str(pair.get("schema_arg") or "")
        ref = str(pair.get("field_ref") or "")
        if not arg or not ref:
            warnings.append("argument_mapping_missing_schema_arg_or_field_ref")
            continue
        if arg in seen_args:
            warnings.append(f"duplicate_schema_arg:{arg}")
            continue
        if arg not in properties:
            warnings.append(f"schema_arg_not_in_full_schema:{arg}")
            continue
        field = field_refs.get(ref)
        if field is None:
            warnings.append(f"field_ref_not_available:{arg}")
            continue
        if not _step9_field_can_satisfy_arg(arg, field):
            warnings.append(f"field_ref_incompatible:{arg}")
            continue
        seen_args.add(arg)
        valid_mappings.append(Step9Stage2ArgumentMapping(schema_arg=arg, field_ref=ref))

    for pair in response_item.get("argument_literals") or []:
        if not isinstance(pair, dict):
            warnings.append("argument_literal_not_object")
            continue
        arg = str(pair.get("schema_arg") or "")
        if not arg:
            warnings.append("argument_literal_missing_schema_arg")
            continue
        if arg in seen_args:
            warnings.append(f"duplicate_schema_arg:{arg}")
            continue
        prop = properties.get(arg)
        if not isinstance(prop, dict):
            warnings.append(f"literal_schema_arg_not_in_full_schema:{arg}")
            continue
        ok, literal = _literal_allowed_by_schema(pair.get("literal_value"), prop)
        if not ok:
            warnings.append(f"literal_not_allowed:{arg}")
            continue
        seen_args.add(arg)
        valid_literals.append(Step9Stage2ArgumentLiteral(schema_arg=arg, literal_value=literal))

    for arg in required:
        if arg in seen_args:
            missing.discard(arg)
            continue
        literal = _deterministic_step9_literal_if_allowed(arg, schema)
        if literal is not None:
            valid_literals.append(Step9Stage2ArgumentLiteral(schema_arg=arg, literal_value=literal))
            seen_args.add(arg)
            missing.discard(arg)
            continue
        missing.add(arg)

    can_invoke = bool(response_item.get("can_invoke")) and not missing
    skip_reason = str(response_item.get("skip_reason") or "")
    if not can_invoke and not skip_reason:
        skip_reason = "missing_required_fields" if missing else "mapping_rejected"
    reason = str(response_item.get("argument_mapping_reason") or "")
    if warnings:
        reason = (reason + "; " if reason else "") + "warnings=" + ",".join(warnings)
    return Step9Stage2MappedTool(
        tool_name=tool_name,
        lane_type=lane_type,
        can_invoke=can_invoke,
        argument_mappings=valid_mappings,
        argument_literals=valid_literals,
        missing_required_fields=sorted(missing),
        skip_reason=skip_reason,
        argument_mapping_reason=reason,
    )


def _model_dump_list(items: list[Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for item in items:
        if hasattr(item, "model_dump"):
            out.append(item.model_dump())
        elif isinstance(item, dict):
            out.append(dict(item))
    return out


def _selection_dict(item: Step9Stage1SelectionAudit | dict[str, Any]) -> dict[str, Any]:
    if hasattr(item, "model_dump"):
        return item.model_dump()
    if isinstance(item, dict):
        return dict(item)
    return {}


def _compact_lane_statuses(items: list[Any]) -> list[dict[str, Any]]:
    statuses = _model_dump_list(items)
    return [
        {
            "candidate_id": item.get("candidate_id"),
            "lane_type": item.get("lane_type"),
            "candidate_type": item.get("candidate_type"),
            "status": item.get("status"),
            "allowed_tools": list(item.get("allowed_tools") or []),
            "missing_requirements": list(item.get("missing_requirements") or []),
            "available_field_refs": list(item.get("available_field_refs") or []),
        }
        for item in statuses
    ]


def _compact_schema_requirements(items: list[Any]) -> list[dict[str, Any]]:
    reqs = _model_dump_list(items)
    return [
        {
            "candidate_id": req.get("candidate_id"),
            "tool_name": req.get("tool_name"),
            "lane_type": req.get("lane_type"),
            "required_fields": list(req.get("required_fields") or []),
            "schema_source": req.get("schema_source"),
            "satisfiable_required_fields": list(
                req.get("satisfiable_required_fields") or []
            ),
            "missing_required_fields": list(req.get("missing_required_fields") or []),
            "hard_gate_decision": req.get("hard_gate_decision"),
            "reason": req.get("reason"),
        }
        for req in reqs
    ]


def _schema_from_requirement(req: dict[str, Any]) -> dict[str, Any]:
    required = [str(arg) for arg in (req.get("required_fields") or []) if isinstance(arg, str)]
    return {
        "type": "object",
        "properties": {arg: {"type": "string"} for arg in required},
        "required": required,
    }


def _compact_schema(schema: dict[str, Any]) -> dict[str, Any]:
    properties = schema.get("properties") if isinstance(schema.get("properties"), dict) else {}
    required = [str(arg) for arg in (schema.get("required") or []) if isinstance(arg, str)]
    compact_props: dict[str, Any] = {}
    for name, prop in sorted(properties.items(), key=lambda item: str(item[0])):
        if not isinstance(name, str) or name.startswith("_") or not isinstance(prop, dict):
            continue
        compact: dict[str, Any] = {}
        for key in ("type", "enum", "const", "default", "description"):
            if key in prop:
                compact[key] = prop[key]
        compact_props[name] = compact
    return {
        "type": "object",
        "properties": compact_props,
        "required": sorted(required),
    }


def _compact_blocked_summary(items: list[Step9LaneStatus]) -> list[dict[str, Any]]:
    statuses = _model_dump_list(items)
    summary: dict[tuple[str, str], dict[str, Any]] = {}
    for lane in statuses:
        key = (str(lane.get("candidate_id") or ""), str(lane.get("lane_type") or ""))
        bucket = summary.setdefault(
            key,
            {
                "candidate_id": key[0],
                "lane_type": key[1],
                "blocked_tool_count": 0,
                "missing_requirements": [],
            },
        )
        bucket["blocked_tool_count"] += len(lane.get("blocked_tools") or [])
        bucket["missing_requirements"] = sorted(
            set(bucket["missing_requirements"]) | set(lane.get("missing_requirements") or [])
        )
    return sorted(summary.values(), key=lambda item: (item["candidate_id"], item["lane_type"]))


def _candidate_context_refs(items: list[Step9AvailableField]) -> list[dict[str, str]]:
    fields = _model_dump_list(items)
    refs: list[dict[str, str]] = []
    for field in fields:
        refs.append(
            {
                "candidate_id": str(field.get("candidate_id") or ""),
                "field_ref": str(field.get("field_ref") or ""),
                "field_type": str(field.get("field_type") or ""),
                "value_kind": str(field.get("value_kind") or ""),
                "status": str(field.get("status") or ""),
            }
        )
    return refs


def _mapping_audit_entries(tool: Step9Stage2MappedTool) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for pair in tool.argument_mappings:
        entries.append(
            {
                "tool_name": tool.tool_name,
                "lane_type": tool.lane_type,
                "schema_arg": pair.schema_arg,
                "field_ref": pair.field_ref,
                "argument_value_source": "mapped_from_field_ref",
                "can_invoke": tool.can_invoke,
                "missing_required_fields": list(tool.missing_required_fields),
                "skip_reason": tool.skip_reason,
            }
        )
    for pair in tool.argument_literals:
        entries.append(
            {
                "tool_name": tool.tool_name,
                "lane_type": tool.lane_type,
                "schema_arg": pair.schema_arg,
                "literal_value": pair.literal_value,
                "argument_value_source": "mapped_from_official_schema_literal",
                "can_invoke": tool.can_invoke,
                "missing_required_fields": list(tool.missing_required_fields),
                "skip_reason": tool.skip_reason,
            }
        )
    if not entries:
        entries.append(
            {
                "tool_name": tool.tool_name,
                "lane_type": tool.lane_type,
                "can_invoke": tool.can_invoke,
                "missing_required_fields": list(tool.missing_required_fields),
                "skip_reason": tool.skip_reason,
            }
        )
    return entries


def _step9_field_can_satisfy_arg(arg: str, field: dict[str, Any]) -> bool:
    lowered = arg.lower().strip()
    value_kind = str(field.get("value_kind") or "").lower()
    field_type = str(field.get("field_type") or "").lower()
    field_ref = str(field.get("field_ref") or "").lower()
    provider = str(field.get("provider") or "").lower()
    source_ref = str(field.get("source_ref") or "").lower()

    if lowered in {"smiles", "canonical_smiles"}:
        return value_kind == "smiles" or field_type == "compound"
    if lowered == "query":
        return value_kind in {"name", "compound_name"} or (
            field_type in {"candidate_metadata", "compound"} and value_kind == "name"
        )
    if lowered == "zinc_id":
        return value_kind == "zinc_id" or "identifier:zinc_id:" in field_ref
    if lowered in {"chembl_id", "molecule_chembl_id"}:
        return value_kind == "chembl_id" or "identifier:chembl_id:" in field_ref
    if lowered in {"pubchem_cid", "cid"}:
        return value_kind == "pubchem_cid" or "identifier:pubchem_cid:" in field_ref
    if lowered in {"uniprot_id", "accession", "uniprot_accession"}:
        return value_kind == "uniprot_id" or "identifier:uniprot" in field_ref
    if lowered in {"variant", "variants", "mutation", "mutations"}:
        return value_kind in {"variant", "variants", "mutation", "protein_variant"}
    if lowered == "chain":
        return value_kind in {"chain", "chain_id", "chain_role"} or "chain" in field_ref
    if lowered == "pdb_id":
        return value_kind == "pdb_id" or "identifier:pdb_id:" in field_ref
    if lowered in {"input_pdb", "pdb_file", "structure_ref", "structure", "backbone", "path"}:
        return (
            provider == "step_08"
            and field_type in {"structure", "structure_ref", "complex_structure"}
            and value_kind in {"complex_structure_ref", "structure_ref", "pdb_id"}
        ) or (
            provider == "step_08"
            and ("complex" in source_ref or "pdb" in source_ref)
        )
    if lowered == "contigs":
        return value_kind in {"contigs", "design_contigs"} or "contig" in field_ref
    if lowered in {"prompt_sequence", "sequence", "sequence_value", "sequence_1", "sequence_2", "sequence_3", "sequence_a", "sequence_b"}:
        return field_type == "protein_sequence" or value_kind in {
            "sequence_material",
            "protein_sequence",
            "fasta_ref",
            "uploaded_fasta_ref",
        }
    return False


def _literal_allowed_by_schema(value: Any, prop: dict[str, Any]) -> tuple[bool, Any]:
    if "const" in prop:
        const = prop.get("const")
        return value == const or str(value) == str(const), const
    enum_values = prop.get("enum")
    if isinstance(enum_values, list) and enum_values:
        for allowed in enum_values:
            if value == allowed or str(value) == str(allowed):
                return True, allowed
        return False, value
    if "default" in prop:
        default = prop.get("default")
        return value == default or str(value) == str(default), default
    return False, value


def _deterministic_step9_literal_if_allowed(arg: str, schema: dict[str, Any]) -> Any | None:
    properties = schema.get("properties") if isinstance(schema.get("properties"), dict) else {}
    prop = properties.get(arg)
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


def _compact_text(value: str) -> str:
    text = " ".join(str(value or "").split())
    text = re.sub(
        r"\b[ACDEFGHIKLMNPQRSTVWY]{12,}\b",
        "[redacted_biological_sequence]",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\b(HEADER|ATOM|HETATM|MODEL|ENDMDL)\b.*", "[redacted_structure_payload]", text)
    text = re.sub(r"sk-[A-Za-z0-9_\-]{8,}", "[redacted_api_key]", text)
    return text[:800]
