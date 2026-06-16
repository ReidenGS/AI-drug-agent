"""Step 1 — IntakeService.

Deterministic. Creates run_id, persists raw_request_record, initializes the
artifact registry, and (in real deployment) enqueues Step 2 onto SQS.
"""

from __future__ import annotations

from typing import Optional

from ..schemas.step_01_raw_request_record import (
    RawRequestRecord,
    UploadedFile,
    UserProvidedContext,
)
from ..utils.ids import new_run_id, new_artifact_id
from ..utils.time import now_iso
from .artifact_registry_service import ArtifactRegistryService
from .storage_service import Storage
from .workflow_state_service import WorkflowStateService


_ARTIFACT_KEY = "inputs/raw_request_record.json"


class IntakeService:
    def __init__(
        self,
        storage: Storage,
        registry: ArtifactRegistryService,
        workflow_state: WorkflowStateService,
    ) -> None:
        self.storage = storage
        self.registry = registry
        self.workflow_state = workflow_state

    def allocate_run_id(self) -> str:
        """Mint a run_id without persisting any artifact.

        Used by the multipart entrypoint so file bytes can be written under
        `adc_pilot/runs/{run_id}/inputs/files/...` before the
        raw_request_record is built. The same run_id is then passed back into
        `submit(run_id=...)`.
        """
        return new_run_id()

    def submit(
        self,
        *,
        raw_user_query: str,
        entry_source: str = "api",
        submitted_by: Optional[str] = None,
        user_provided_context: Optional[dict] = None,
        uploaded_files: Optional[list[dict]] = None,
        run_id: Optional[str] = None,
    ) -> RawRequestRecord:
        """Persist a raw_request_record for a run.

        `run_id` is normally allocated here. The multipart intake endpoint
        needs the run_id *before* writing file bytes (so storage paths land
        under the right run dir), so it allocates via `allocate_run_id()` and
        passes the result back in. JSON callers never need this.
        """
        run_id = run_id or new_run_id()
        self.workflow_state.init_run(run_id)
        registry = self.registry.init_registry(run_id)

        record = RawRequestRecord(
            run_id=run_id,
            run_artifact_registry_id=registry.run_artifact_registry_id,
            created_at=now_iso(),
            entry_source=entry_source,  # type: ignore[arg-type]
            submitted_by=submitted_by,
            raw_user_query=raw_user_query,
            user_provided_context=UserProvidedContext(**(user_provided_context or {})),
            uploaded_files=[UploadedFile(**f) for f in (uploaded_files or [])],
            intake_status="received",
        )
        artifact_id = new_artifact_id("raw_request_record")
        self.storage.write_json(
            self.storage.run_key(run_id, _ARTIFACT_KEY),
            {"artifact_id": artifact_id, **record.model_dump()},
        )
        self.registry.update_active(run_id, raw_request_record_id=artifact_id)
        self.workflow_state.mark(run_id, "step_01", "completed")
        return record
