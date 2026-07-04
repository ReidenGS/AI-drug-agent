"""Step 9 dry-run runtime resolver.

This module turns Stage 2 schema mappings into an execution-plan audit without
calling MCP tools. Raw candidate values are intentionally not resolved into
persisted output; field refs are represented by compact metadata only.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


STEP9_DRY_RUN_EXECUTION_MODE = "dry_run_only"


class Step9DryRunResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    execution_plan: list[dict[str, Any]] = Field(default_factory=list)
    resolved_tools: list[str] = Field(default_factory=list)
    unresolved_tools: list[str] = Field(default_factory=list)
    resolver_audit: list[dict[str, Any]] = Field(default_factory=list)
    execution_mode: str = STEP9_DRY_RUN_EXECUTION_MODE


def build_step9_dry_run_execution_plan(
    *,
    mapped_tools: list[Any],
    available_fields: list[Any],
) -> Step9DryRunResult:
    """Build a dry-run execution plan from Stage 2 mapped tools.

    `mapped_tools` may be Pydantic models or already-dumped dicts. Field refs
    must exist in `available_fields` and have status `available` to resolve.
    """

    field_by_ref = {
        str(field.get("field_ref") or ""): field
        for field in _dump_list(available_fields)
        if isinstance(field.get("field_ref"), str)
    }
    execution_plan: list[dict[str, Any]] = []
    resolved_tools: list[str] = []
    unresolved_tools: list[str] = []
    resolver_audit: list[dict[str, Any]] = []

    for raw_tool in _dump_list(mapped_tools):
        tool_name = str(raw_tool.get("tool_name") or "")
        lane_type = str(raw_tool.get("lane_type") or "")
        missing = [
            str(item)
            for item in (raw_tool.get("missing_required_fields") or [])
            if isinstance(item, str)
        ]
        unresolved_refs: list[dict[str, str]] = []
        argument_plan: list[dict[str, Any]] = []
        argument_keys: list[str] = []

        for pair in raw_tool.get("argument_mappings") or []:
            if not isinstance(pair, dict):
                continue
            schema_arg = str(pair.get("schema_arg") or "")
            field_ref = str(pair.get("field_ref") or "")
            if not schema_arg or not field_ref:
                continue
            field = field_by_ref.get(field_ref)
            if field is None:
                unresolved_refs.append({"schema_arg": schema_arg, "field_ref": field_ref, "reason": "field_ref_not_available"})
                resolver_audit.append(_audit_entry(tool_name, lane_type, schema_arg, field_ref, None, "unresolved"))
                continue
            if str(field.get("status") or "available") != "available":
                unresolved_refs.append({"schema_arg": schema_arg, "field_ref": field_ref, "reason": "field_ref_not_available_status"})
                resolver_audit.append(_audit_entry(tool_name, lane_type, schema_arg, field_ref, field, "unresolved"))
                continue
            argument_keys.append(schema_arg)
            metadata = _field_metadata(field)
            argument_plan.append(
                {
                    "schema_arg": schema_arg,
                    "source": "field_ref",
                    "field_ref": field_ref,
                    "source_metadata": metadata,
                    "candidate_value_persisted": False,
                }
            )
            resolver_audit.append(_audit_entry(tool_name, lane_type, schema_arg, field_ref, field, "resolved"))

        for pair in raw_tool.get("argument_literals") or []:
            if not isinstance(pair, dict):
                continue
            schema_arg = str(pair.get("schema_arg") or "")
            if not schema_arg:
                continue
            argument_keys.append(schema_arg)
            argument_plan.append(
                {
                    "schema_arg": schema_arg,
                    "source": "official_schema_literal",
                    "literal_value": pair.get("literal_value"),
                    "candidate_value_persisted": False,
                }
            )
            resolver_audit.append(
                {
                    "tool_name": tool_name,
                    "lane_type": lane_type,
                    "schema_arg": schema_arg,
                    "source": "official_schema_literal",
                    "resolve_status": "resolved",
                }
            )

        can_resolve = bool(raw_tool.get("can_invoke")) and not missing and not unresolved_refs
        skip_reason = str(raw_tool.get("skip_reason") or "")
        if not can_resolve and not skip_reason:
            if missing:
                skip_reason = "missing_required_fields"
            elif unresolved_refs:
                skip_reason = "unresolved_field_refs"
            else:
                skip_reason = "stage2_uninvokable"

        record = {
            "tool_name": tool_name,
            "lane_type": lane_type,
            "can_resolve": can_resolve,
            "would_execute": can_resolve,
            "execution_mode": STEP9_DRY_RUN_EXECUTION_MODE,
            "argument_keys": sorted(set(argument_keys)),
            "argument_plan": argument_plan,
            "missing_required_fields": missing,
            "unresolved_field_refs": unresolved_refs,
            "skip_reason": "" if can_resolve else skip_reason,
        }
        execution_plan.append(record)
        if can_resolve:
            resolved_tools.append(tool_name)
        else:
            unresolved_tools.append(tool_name)

    return Step9DryRunResult(
        execution_plan=execution_plan,
        resolved_tools=resolved_tools,
        unresolved_tools=unresolved_tools,
        resolver_audit=resolver_audit,
    )


def _dump_list(items: list[Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for item in items:
        if hasattr(item, "model_dump"):
            out.append(item.model_dump())
        elif isinstance(item, dict):
            out.append(dict(item))
    return out


def _field_metadata(field: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "field_ref",
        "candidate_id",
        "provider",
        "field_type",
        "value_kind",
        "source_ref",
        "status",
    )
    return {key: field.get(key) for key in keys if field.get(key) is not None}


def _audit_entry(
    tool_name: str,
    lane_type: str,
    schema_arg: str,
    field_ref: str,
    field: dict[str, Any] | None,
    status: str,
) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "tool_name": tool_name,
        "lane_type": lane_type,
        "schema_arg": schema_arg,
        "field_ref": field_ref,
        "source": "field_ref",
        "resolve_status": status,
        "candidate_value_persisted": False,
    }
    if field is not None:
        entry.update(_field_metadata(field))
    return entry
