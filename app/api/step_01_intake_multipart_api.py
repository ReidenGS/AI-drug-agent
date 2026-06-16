"""Step 1 multipart intake — simulated frontend submission entrypoint.

Adds `POST /runs/multipart` so a frontend (or curl) can submit:
- text fields: raw_user_query, entry_source, submitted_by, user_provided_context
- one or more files via `files[]` parts

The existing JSON `POST /runs` is preserved (see app/api/run_api.py) and stays
the canonical internal / notebook / script ingestion path. Both endpoints
delegate to the same `IntakeService.submit(...)` and produce the same
`raw_request_record` shape — the only difference is *how* the files arrive.

Hardening (this file is the SINGLE place that enforces multipart limits — the
frontend cannot be trusted):

1. **Pre-validation before any IO.** `raw_user_query`, `entry_source`,
   `user_provided_context`, and file count are validated BEFORE we read or
   write any byte. A 422/413 on these never touches the storage layer.
2. **Per-file size cap.** Files are read with `read(cap + 1)` so the request
   short-circuits as soon as a file exceeds the limit; no oversized payload
   is ever fully buffered.
3. **Orphan cleanup on failure.** All file keys written for this request are
   tracked. If `IntakeService.submit(...)` raises (or anything else does
   after the first byte hits disk), every written key is deleted before the
   exception propagates. Behaviour: either the whole multipart request
   succeeds with all files persisted and a raw_request_record, or NOTHING is
   left on disk.

File persistence on success:
`adc_pilot/runs/{run_id}/inputs/files/{file_id}{ext}` — bytes only. File
metadata lands on `raw_request_record.uploaded_files[]`; the artifact JSON
never embeds raw bytes.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import PurePosixPath
from typing import Annotated, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from ..deps import get_registry_service, get_storage, get_workflow_state_service
from ..services.intake_service import IntakeService
from ..services.storage_service import Storage
from ..settings import get_settings
from ..utils.ids import new_file_id

router = APIRouter(prefix="/runs", tags=["runs"])
logger = logging.getLogger(__name__)


_ALLOWED_ENTRY_SOURCES = {"ui", "api", "notebook", "script"}


def _parse_user_context(raw: Optional[str]) -> dict:
    """`user_provided_context` arrives as a JSON string field (multipart has
    no native JSON), so we parse it here. Empty string or missing → {}."""
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=422,
            detail=f"user_provided_context must be valid JSON: {e.msg}",
        )
    if not isinstance(value, dict):
        raise HTTPException(
            status_code=422,
            detail="user_provided_context must be a JSON object",
        )
    return value


def _cleanup(storage: Storage, keys: list[str]) -> None:
    for key in keys:
        try:
            storage.delete(key)
        except Exception:  # noqa: BLE001
            logger.warning("Failed to clean up orphan upload at %s", key, exc_info=True)


@router.post("/multipart", status_code=201)
async def create_run_multipart(
    raw_user_query: Annotated[str, Form()],
    files: Annotated[list[UploadFile], File(description="Upload files (pdb/cif/fasta/csv/...)")] = [],  # noqa: B006
    entry_source: Annotated[str, Form()] = "ui",
    submitted_by: Annotated[Optional[str], Form()] = None,
    user_provided_context: Annotated[Optional[str], Form()] = None,
) -> dict:
    settings = get_settings()

    # 1. Text-field pre-validation — runs entirely before any file IO.
    if not raw_user_query or not raw_user_query.strip():
        raise HTTPException(status_code=422, detail="raw_user_query must be a non-empty string")
    if entry_source not in _ALLOWED_ENTRY_SOURCES:
        raise HTTPException(
            status_code=422,
            detail=f"entry_source must be one of {sorted(_ALLOWED_ENTRY_SOURCES)}",
        )
    ctx = _parse_user_context(user_provided_context)

    # 2. File-count pre-validation (filenameless / empty parts are ignored).
    real_files = [f for f in (files or []) if f and f.filename]
    if len(real_files) > settings.max_upload_files_per_run:
        raise HTTPException(
            status_code=413,
            detail=(
                f"Too many files: {len(real_files)} provided, "
                f"max is {settings.max_upload_files_per_run}"
            ),
        )

    # 3. Allocate run_id, then file IO + submit inside a single try/except so
    #    any failure path triggers the same orphan cleanup.
    storage = get_storage()
    intake = IntakeService(
        storage=storage,
        registry=get_registry_service(),
        workflow_state=get_workflow_state_service(),
    )
    run_id = intake.allocate_run_id()

    written_keys: list[str] = []
    try:
        uploaded_files_meta: list[dict] = []
        cap = settings.max_upload_bytes_per_file
        for f in real_files:
            data = await f.read(cap + 1)
            if len(data) > cap:
                raise HTTPException(
                    status_code=413,
                    detail=(
                        f"File {f.filename!r} exceeds per-file size limit "
                        f"({cap} bytes)"
                    ),
                )
            file_id = new_file_id()
            ext = PurePosixPath(f.filename).suffix
            key = storage.run_key(run_id, "inputs", "files", f"{file_id}{ext}")
            storage.write_bytes(key, data)
            written_keys.append(key)
            uploaded_files_meta.append(
                {
                    "file_id": file_id,
                    "original_filename": f.filename,
                    "storage_path": key,
                    "content_type": f.content_type or "application/octet-stream",
                    "sha256": "sha256:" + hashlib.sha256(data).hexdigest(),
                    "size_bytes": len(data),
                }
            )

        record = intake.submit(
            run_id=run_id,
            raw_user_query=raw_user_query,
            entry_source=entry_source,
            submitted_by=submitted_by,
            user_provided_context=ctx,
            uploaded_files=uploaded_files_meta,
        )
    except HTTPException:
        _cleanup(storage, written_keys)
        raise
    except Exception:
        _cleanup(storage, written_keys)
        raise

    return {
        "run_id": record.run_id,
        "uploaded_file_count": len(uploaded_files_meta),
        "raw_request_record": record.model_dump(),
    }
