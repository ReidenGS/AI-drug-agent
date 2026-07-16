"""Fail-closed authority reader for the active raw request artifact."""

from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from app.schemas.step_01_raw_request_record import RawRequestRecord

_RAW_REQUEST_STORAGE_KEY = "inputs/raw_request_record.json"


class RawRequestAuthorityError(RuntimeError):
    """Fixed compact raw-request authority failure."""


def load_active_raw_request(
    *, run_id: str, registry: Any, storage: Any
) -> RawRequestRecord:
    """Return the strictly validated active raw request without leaking data."""

    try:
        run_registry = registry.get(run_id)
    except Exception:
        raise RawRequestAuthorityError("raw_request_record_unavailable") from None
    artifact_id = run_registry.active_artifacts.raw_request_record_id
    if not artifact_id:
        raise RawRequestAuthorityError("raw_request_record_missing")
    key = storage.run_key(run_id, _RAW_REQUEST_STORAGE_KEY)
    try:
        if not storage.exists(key):
            raise RawRequestAuthorityError("raw_request_record_storage_missing")
        body = storage.read_json(key)
    except RawRequestAuthorityError:
        raise
    except Exception:
        raise RawRequestAuthorityError("raw_request_record_unreadable") from None
    if (
        not isinstance(body, dict)
        or body.get("artifact_id") != artifact_id
        or body.get("run_id") != run_id
    ):
        raise RawRequestAuthorityError("raw_request_record_identity_mismatch")
    try:
        record = RawRequestRecord.model_validate(
            {name: value for name, value in body.items() if name != "artifact_id"},
            strict=True,
        )
    except ValidationError:
        raise RawRequestAuthorityError("raw_request_record_schema_invalid") from None
    if (
        record.session_id is None
        or record.run_artifact_registry_id
        != run_registry.run_artifact_registry_id
    ):
        raise RawRequestAuthorityError("raw_request_record_identity_mismatch")
    return record


__all__ = ["RawRequestAuthorityError", "load_active_raw_request"]
