"""Step 9 Stage 1 relevance selection.

This selector is audit-only for the current Step 9 iteration. It asks the LLM
which hard-gate-allowed Step 9 tools are relevant, validates the result against
that allowed catalog, and returns compact audit data. It does not construct
arguments and does not execute protein/variant tools.
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


STEP9_STAGE1_PROMPT_CACHE_LAYOUT_VERSION = "step9_stage1_v1"

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


def _model_dump_list(items: list[Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for item in items:
        if hasattr(item, "model_dump"):
            out.append(item.model_dump())
        elif isinstance(item, dict):
            out.append(dict(item))
    return out


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
