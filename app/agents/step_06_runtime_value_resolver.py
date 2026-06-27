"""Runtime resolver for Step 6 available field refs.

The resolver is intentionally separate from the LLM-safe projection:
available fields expose only digests and typed refs, while this module
retrieves the raw value immediately before a runtime MCP argument is built.
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

from ..services.storage_service import Storage
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
    storage: Storage | None = None,
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

    try:
        raw_value = _raw_value_for_ref(candidate, parsed, storage=storage)
    except Exception as exc:  # noqa: BLE001
        return ResolvedRuntimeValue(
            status="unresolved",
            field_ref=field_ref,
            audit_metadata=_audit_metadata(field),
            error_message=f"unable to resolve runtime value: {exc}",
        )
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


def _raw_value_for_ref(
    candidate: dict[str, Any],
    parsed: dict[str, str],
    *,
    storage: Storage | None = None,
) -> str | None:
    if parsed["source_kind"] == "material":
        for material in candidate.get("materials") or []:
            if not isinstance(material, dict):
                continue
            if str(material.get("material_id") or "") == parsed["source_id"]:
                value = material.get("value")
                value_kind = _material_value_kind(material)
                if value_kind == "uploaded_fasta_ref":
                    if value in (None, ""):
                        return None
                    if storage is None:
                        return None
                    return _read_uploaded_fasta_sequence(storage, str(value))
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


def _material_value_kind(material: dict[str, Any]) -> str:
    mt = str(material.get("material_type") or "")
    value_kind = str(material.get("value_kind") or "").lower()
    if value_kind:
        return value_kind

    if mt in {"antibody_heavy_chain_sequence", "antibody_light_chain_sequence"}:
        return "uploaded_fasta_ref" if _material_value_looks_like_ref(material) else "protein_sequence"
    return ""


def _material_value_looks_like_ref(material: dict[str, Any]) -> bool:
    from .step_06_available_fields import _is_ref_shaped_material

    return _is_ref_shaped_material(material)


def _read_uploaded_fasta_sequence(storage: Storage, path: str) -> str:
    try:
        data = storage.read_bytes(path)
    except Exception as exc:
        raise ValueError("uploaded FASTA content could not be read from storage") from exc
    try:
        content = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("uploaded FASTA content is not utf-8 text") from exc
    sequences = _extract_fasta_sequences(content)
    if not sequences:
        raise ValueError("uploaded FASTA content has no sequences")

    # Do not concatenate multi-record FASTA bodies; emit first chain only and
    # keep resolver logic deterministic. Selection strategy is visible via the
    # compact resolver output path in tests and auditing.
    return sequences[0]


def _extract_fasta_sequences(content: str) -> list[str]:
    out: list[str] = []
    current: list[str] = []
    for line in content.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith(">"):
            if current:
                out.append("".join(current))
                current = []
            continue
        letters = "".join(ch for ch in stripped if ch.isalpha())
        if letters:
            current.append(letters)
    if current:
        out.append("".join(current))
    return out


def _unresolved(field_ref: str, reason: str) -> ResolvedRuntimeValue:
    return ResolvedRuntimeValue(
        status="unresolved",
        field_ref=field_ref,
        audit_metadata={"field_ref": field_ref},
        error_message=reason,
    )
