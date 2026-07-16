"""Persist the SupervisorAgent's structured_query under the registry.

This helper is intentionally shared by the Step 2 API and the LangGraph
Step 2 node so they cannot drift.
"""

from __future__ import annotations

from ..agents.supervisor_agent import SupervisorAgent
from ..schemas.step_02_structured_query import StructuredQuery
from ..utils.errors import WorkflowStateError
from ..utils.ids import new_artifact_id
from .artifact_registry_service import ArtifactRegistryService
from .storage_service import Storage
from .workflow_state_service import WorkflowStateService


_ARTIFACT_KEY = "inputs/structured_query.json"
_HISTORY_KEY = "inputs/history/structured_query/{artifact_id}.json"


class StructuredQueryService:
    def __init__(
        self,
        storage: Storage,
        registry: ArtifactRegistryService,
        workflow_state: WorkflowStateService,
        supervisor: SupervisorAgent,
    ) -> None:
        self.storage = storage
        self.registry = registry
        self.workflow_state = workflow_state
        self.supervisor = supervisor

    def parse(self, run_id: str) -> StructuredQuery:
        reg = self.registry.get(run_id)
        raw_id = reg.active_artifacts.raw_request_record_id
        if not raw_id:
            raise WorkflowStateError("Step 2 requires Step 1 raw_request_record")

        raw_payload = self.storage.read_json(
            self.storage.run_key(run_id, "inputs/raw_request_record.json")
        )
        return self.parse_effective(run_id, raw_payload)

    def parse_effective(
        self, run_id: str, effective_raw_payload: dict
    ) -> StructuredQuery:
        """Parse an in-memory same-run projection without rewriting Step 1."""

        reg = self.registry.get(run_id)
        raw_id = reg.active_artifacts.raw_request_record_id
        if (
            not raw_id
            or effective_raw_payload.get("run_id") != run_id
            or effective_raw_payload.get("artifact_id") != raw_id
        ):
            raise WorkflowStateError("structured_query_source_invalid")
        sq = self.supervisor.parse_raw_to_structured_query(
            effective_raw_payload
        )

        artifact_id = new_artifact_id("structured_query")
        body = {"artifact_id": artifact_id, **sq.model_dump()}
        self.storage.write_json(
            self.storage.run_key(
                run_id,
                _HISTORY_KEY.format(artifact_id=artifact_id),
            ),
            body,
        )
        self.storage.write_json(
            self.storage.run_key(run_id, _ARTIFACT_KEY),
            body,
        )
        self.registry.update_active(run_id, structured_query_id=artifact_id)
        self.workflow_state.mark(run_id, "step_02", "completed")
        return sq


def structured_query_history_key(artifact_id: str) -> str:
    """Return the same-run immutable audit key for one Step 2 artifact."""

    return _HISTORY_KEY.format(artifact_id=artifact_id)
