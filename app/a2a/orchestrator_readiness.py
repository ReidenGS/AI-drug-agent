"""Fail-closed Step 3 authority check before Step 4 worker routing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError

from app.schemas.step_01_raw_request_record import RawRequestRecord
from app.schemas.step_02_structured_query import StructuredQuery
from app.schemas.step_03_input_readiness import InputReadinessStatus
from app.services.raw_request_authority import (
    RawRequestAuthorityError,
    load_active_raw_request,
)

_READINESS_STORAGE_KEY = "inputs/input_readiness_status.json"
_STRUCTURED_QUERY_STORAGE_KEY = "inputs/structured_query.json"


class OrchestratorReadinessError(RuntimeError):
    """Fixed compact readiness gate error."""


@dataclass(frozen=True)
class ValidatedInputReadinessAuthority:
    """One stable typed Step 1→2→3 authority chain."""

    raw_request: RawRequestRecord
    structured_query: StructuredQuery
    readiness: InputReadinessStatus
    raw_request_artifact_id: str
    structured_query_artifact_id: str
    readiness_artifact_id: str


def _active_ids(run_id: str, registry: Any) -> tuple[Any, str, str, str]:
    try:
        run_registry = registry.get(run_id)
    except Exception:
        raise OrchestratorReadinessError(
            "input_readiness_status_unavailable"
        ) from None
    active = run_registry.active_artifacts
    if not active.raw_request_record_id or not active.structured_query_id:
        raise OrchestratorReadinessError(
            "input_readiness_status_source_mismatch"
        )
    if not active.input_readiness_status_id:
        raise OrchestratorReadinessError("input_readiness_status_missing")
    return (
        run_registry,
        active.raw_request_record_id,
        active.structured_query_id,
        active.input_readiness_status_id,
    )


def load_input_readiness_authority(
    *, run_id: str, registry: Any, storage: Any
) -> ValidatedInputReadinessAuthority:
    """Validate one stable active raw→structured→readiness source chain."""

    before, raw_id, structured_id, readiness_id = _active_ids(run_id, registry)
    try:
        raw = load_active_raw_request(
            run_id=run_id, registry=registry, storage=storage
        )
    except RawRequestAuthorityError as exc:
        raise OrchestratorReadinessError(str(exc)) from None

    structured_key = storage.run_key(run_id, _STRUCTURED_QUERY_STORAGE_KEY)
    try:
        if not storage.exists(structured_key):
            raise OrchestratorReadinessError("structured_query_storage_missing")
        structured_body = storage.read_json(structured_key)
    except OrchestratorReadinessError:
        raise
    except Exception:
        raise OrchestratorReadinessError("structured_query_unreadable") from None
    if (
        not isinstance(structured_body, dict)
        or structured_body.get("artifact_id") != structured_id
        or structured_body.get("run_id") != run_id
    ):
        raise OrchestratorReadinessError("structured_query_identity_mismatch")
    try:
        structured = StructuredQuery.model_validate(
            {
                name: value
                for name, value in structured_body.items()
                if name != "artifact_id"
            },
            strict=True,
        )
    except ValidationError:
        raise OrchestratorReadinessError("structured_query_schema_invalid") from None

    readiness_key = storage.run_key(run_id, _READINESS_STORAGE_KEY)
    try:
        if not storage.exists(readiness_key):
            raise OrchestratorReadinessError(
                "input_readiness_status_storage_missing"
            )
        readiness_body = storage.read_json(readiness_key)
    except OrchestratorReadinessError:
        raise
    except Exception:
        raise OrchestratorReadinessError(
            "input_readiness_status_unreadable"
        ) from None
    if (
        not isinstance(readiness_body, dict)
        or readiness_body.get("artifact_id") != readiness_id
        or readiness_body.get("run_id") != run_id
    ):
        raise OrchestratorReadinessError(
            "input_readiness_status_identity_mismatch"
        )
    try:
        readiness = InputReadinessStatus.model_validate(
            {
                name: value
                for name, value in readiness_body.items()
                if name != "artifact_id"
            },
            strict=True,
        )
    except ValidationError:
        raise OrchestratorReadinessError(
            "input_readiness_status_schema_invalid"
        ) from None

    after, after_raw, after_structured, after_readiness = _active_ids(
        run_id, registry
    )
    if (
        before.run_artifact_registry_id != after.run_artifact_registry_id
        or (raw_id, structured_id, readiness_id)
        != (after_raw, after_structured, after_readiness)
        or structured.source_raw_request_ref.raw_request_record_id != raw_id
        or readiness.source_refs.raw_request_record_id != raw_id
        or readiness.source_refs.structured_query_id != structured_id
    ):
        raise OrchestratorReadinessError(
            "input_readiness_status_source_mismatch"
        )
    return ValidatedInputReadinessAuthority(
        raw_request=raw,
        structured_query=structured,
        readiness=readiness,
        raw_request_artifact_id=raw_id,
        structured_query_artifact_id=structured_id,
        readiness_artifact_id=readiness_id,
    )


def load_input_readiness_status(
    *, run_id: str, registry: Any, storage: Any
) -> InputReadinessStatus:
    """Read and validate the active Step 3 artifact without side effects."""

    return load_input_readiness_authority(
        run_id=run_id, registry=registry, storage=storage
    ).readiness


def require_ready_input_authority(
    *, run_id: str, registry: Any, storage: Any
) -> ValidatedInputReadinessAuthority:
    """Require a stable source chain whose Step 3 status is semantically ready."""

    authority = load_input_readiness_authority(
        run_id=run_id, registry=registry, storage=storage
    )
    status = authority.readiness
    if status.input_readiness_status != "ready":
        raise OrchestratorReadinessError("input_readiness_not_ready")
    if (
        any(
            item.severity in {"blocking", "warning"}
            for item in status.missing_input_checklist
        )
        or bool(status.blocking_reasons)
        or any(
            not request.resolved
            and request.severity in {"blocking", "warning"}
            for request in status.clarification_requests
        )
    ):
        raise OrchestratorReadinessError(
            "input_readiness_status_semantic_invalid"
        )
    return authority


def require_ready_input_readiness(
    *, run_id: str, registry: Any, storage: Any
) -> InputReadinessStatus:
    """Require the validated Step 3 authority to be exactly ``ready``."""

    return require_ready_input_authority(
        run_id=run_id,
        registry=registry,
        storage=storage,
    ).readiness


__all__ = [
    "OrchestratorReadinessError",
    "ValidatedInputReadinessAuthority",
    "load_input_readiness_authority",
    "load_input_readiness_status",
    "require_ready_input_authority",
    "require_ready_input_readiness",
]
