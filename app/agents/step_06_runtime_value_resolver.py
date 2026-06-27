"""Runtime resolver for Step 6 available field refs.

The resolver is intentionally separate from the LLM-safe projection:
available fields expose only digests and typed refs, while this module
retrieves the raw value immediately before a runtime MCP argument is built.
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

from .step_06_available_fields import AvailableField, project_candidate_available_fields


ResolveStatus = Literal["resolved", "missing", "unresolved"]


class ResolvedRuntimeValue(BaseModel):
    status: ResolveStatus
    field_ref: str
    raw_value: Optional[str] = Field(default=None, exclude=True)
    audit_metadata: dict[str, Any] = Field(default_factory=dict)
    error_message: Optional[str] = None


def resolve_runtime_value(
    *,
    candidate: dict[str, Any],
    field_ref: str,
) -> ResolvedRuntimeValue:
    """Resolve a candidate-local field_ref to the raw candidate value.

    `raw_value` is excluded from model dumps so callers cannot accidentally
    persist it in normalized artifacts. The compact audit metadata carries
    only field/type/digest information.
    """
    candidate_id = str(candidate.get("candidate_id") or "unknown")
    parsed = _parse_field_ref(field_ref)
    if parsed is None:
        return _unresolved(field_ref, "field_ref is not parseable")
    if parsed["candidate_id"] != candidate_id:
        return _unresolved(field_ref, "field_ref candidate_id does not match candidate")

    projection = project_candidate_available_fields(candidate)
    field_by_ref = {f.field_ref: f for f in projection.available_fields}
    field = field_by_ref.get(field_ref)
    if field is None:
        return _unresolved(field_ref, "field_ref is not present in candidate available fields")

    raw_value = _raw_value_for_ref(candidate, parsed)
    if raw_value in (None, ""):
        return ResolvedRuntimeValue(
            status="missing",
            field_ref=field_ref,
            audit_metadata=_audit_metadata(field),
            error_message="source value missing",
        )

    return ResolvedRuntimeValue(
        status="resolved",
        field_ref=field_ref,
        raw_value=str(raw_value),
        audit_metadata=_audit_metadata(field),
    )


def _raw_value_for_ref(candidate: dict[str, Any], parsed: dict[str, str]) -> str | None:
    if parsed["source_kind"] == "material":
        for material in candidate.get("materials") or []:
            if not isinstance(material, dict):
                continue
            if str(material.get("material_id") or "") == parsed["source_id"]:
                value = material.get("value")
                return str(value) if value not in (None, "") else None
        return None
    if parsed["source_kind"] == "identifier":
        for identifier in candidate.get("identifiers") or []:
            if not isinstance(identifier, dict):
                continue
            value = identifier.get("id_value")
            id_type = str(identifier.get("id_type") or "")
            if value in (None, ""):
                continue
            # The source_id is digest-derived because Step 5 identifiers do
            # not currently carry stable identifier_id fields.
            from .step_06_available_fields import _identifier_source_id  # local internal contract

            if _identifier_source_id(id_type, str(value)) == parsed["source_id"]:
                return str(value)
        return None
    if parsed["source_kind"] == "candidate_metadata":
        value = candidate.get(parsed["source_id"])
        return str(value) if value not in (None, "") else None
    return None


def _audit_metadata(field: AvailableField) -> dict[str, Any]:
    keys = (
        "field_ref",
        "candidate_id",
        "source_kind",
        "source_id",
        "field_name",
        "field_type",
        "value_kind",
        "semantic_role",
        "material_type",
        "id_type",
        "chain_role",
        "length",
        "sha256_prefix",
        "ref_length",
        "ref_sha256_prefix",
        "allowed_transforms",
        "privacy_class",
    )
    data = field.model_dump()
    return {k: data.get(k) for k in keys if data.get(k) is not None}


def _parse_field_ref(field_ref: str) -> dict[str, str] | None:
    parts = field_ref.split(":")
    if len(parts) != 5:
        return None
    prefix, candidate_id, source_kind, source_id, field_name = parts
    if prefix != "candidate":
        return None
    if source_kind not in {"material", "identifier", "candidate_metadata"}:
        return None
    if field_name not in {"value", "id_value"}:
        return None
    return {
        "candidate_id": candidate_id,
        "source_kind": source_kind,
        "source_id": source_id,
        "field_name": field_name,
    }


def _unresolved(field_ref: str, reason: str) -> ResolvedRuntimeValue:
    return ResolvedRuntimeValue(
        status="unresolved",
        field_ref=field_ref,
        audit_metadata={"field_ref": field_ref},
        error_message=reason,
    )
