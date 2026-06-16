"""Step 1 API — raw_request_record.

Per README_FOR_CLAUDE.md, each step API lives in its own file.

The canonical entry for Step 1 is `POST /runs` (see run_api.py), which mints
the `run_id` via IntakeService. This endpoint exists so the orchestrator can
re-read an already-created Step 1 artifact by run_id. It does NOT create a new
run — silently allocating a fresh run_id here would diverge from the URL's
run_id and break artifact addressability.
"""

from __future__ import annotations

from fastapi import APIRouter

from ..deps import get_storage
from ..utils.errors import NotFoundError, WorkflowStateError

router = APIRouter(prefix="/runs/{run_id}/steps/1", tags=["step-01-intake"])


@router.post("/execute")
def execute_step_01(run_id: str) -> dict:
    """Idempotent re-read of the Step 1 artifact for an existing run.

    - If the run has no `raw_request_record.json`, return 404 (callers should
      use `POST /runs` to mint a run).
    - If the artifact exists, return it as-is. Re-running Step 1 with a
      different payload is intentionally not supported here; the IntakeService
      owns run_id creation and we will not branch run_ids inside a step
      endpoint.
    """
    storage = get_storage()
    key = storage.run_key(run_id, "inputs/raw_request_record.json")
    if not storage.exists(key):
        raise NotFoundError(
            f"run {run_id} has no Step 1 artifact",
            detail={"hint": "Create a run via POST /runs before re-executing steps"},
        )
    payload = storage.read_json(key)
    if payload.get("run_id") != run_id:
        # Defensive check: refuse to serve a mis-addressed artifact.
        raise WorkflowStateError(
            "Stored raw_request_record.run_id does not match URL run_id",
            detail={"url_run_id": run_id, "stored_run_id": payload.get("run_id")},
        )
    return payload
