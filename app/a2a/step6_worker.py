"""Step 6 Developability worker core for the shared HTTP A2A adapter.

Transport/envelope validation stays in :mod:`app.a2a.worker_server`. This
module validates the Step 6 artifact contract, loads the run-scoped normalized
candidate context from worker-owned storage, executes the real request-based
DevelopabilityAgent core, verifies persistence, and returns compact refs and
summaries only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

from python_a2a import AgentCard

from ..agents.developability_agent import DevelopabilityAgent
from ..schemas.common import ToolCallRecord
from .agent_cards import (
    AGENT_ID_STEP6,
    CAP_STEP6_DEVELOPABILITY,
    build_step6_agent_card,
    parse_adc_agent_contract,
    validate_adc_agent_contract,
)
from .contracts import (
    ToolCallSummary,
    WorkerArtifactRef,
    WorkerExecutionRequest,
    WorkerExecutionResult,
)
from .worker_server import (
    WorkerRequestRejected,
    assert_advertised_url_matches_port,
    create_worker_flask_app,
    serve_worker_http,
)

_CANDIDATE_CONTEXT_ARTIFACT_TYPE = "candidate_context_table"
_STRUCTURED_LIABILITY_ARTIFACT_TYPE = "structured_liability_summary"


@dataclass(frozen=True)
class _ArtifactSpec:
    name: str
    artifact_type: str
    storage_path: str
    required_field_keys: tuple[str, ...]
    registry_field: str
    expected_entity_type: Optional[str]
    expected_selection_mode: Optional[str]


class Step6A2AWorker:
    """Request-based Step 6 domain core implementing ``WorkerCore``."""

    AGENT_ID = AGENT_ID_STEP6
    CAPABILITY_ID = CAP_STEP6_DEVELOPABILITY

    def __init__(
        self,
        *,
        url: str,
        storage: Any,
        registry: Any,
        workflow_state: Any,
        mcp_client: Any,
        llm: Any = None,
        developability_agent_factory: Optional[
            Callable[[], DevelopabilityAgent]
        ] = None,
    ) -> None:
        self.url = url
        self.agent_id = self.AGENT_ID
        self.capability_ids = frozenset({self.CAPABILITY_ID})
        self._agent_card = build_step6_agent_card(url)
        self._storage = storage
        self._registry = registry
        self._workflow_state = workflow_state
        self._mcp_client = mcp_client
        self._llm = llm
        self._developability_agent_factory = (
            developability_agent_factory or self._default_agent_factory
        )
        self._input_spec, self._output_spec = self._derive_artifact_specs()

    @property
    def agent_card(self) -> AgentCard:
        return self._agent_card

    def health(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "agent_id": self.AGENT_ID,
            "capabilities": sorted(self.capability_ids),
        }

    def _default_agent_factory(self) -> DevelopabilityAgent:
        return DevelopabilityAgent(
            storage=self._storage,
            registry=self._registry,
            workflow_state=self._workflow_state,
            mcp_client=self._mcp_client,
            llm=self._llm,
        )

    def _derive_artifact_specs(self) -> tuple[_ArtifactSpec, _ArtifactSpec]:
        contract = parse_adc_agent_contract(self._agent_card)
        capability = next(
            cap
            for cap in contract.capabilities
            if cap.capability_id == self.CAPABILITY_ID
        )
        input_ref = next(
            ref
            for ref in capability.required_input_artifacts
            if ref.artifact_name == _CANDIDATE_CONTEXT_ARTIFACT_TYPE
        )
        field_requirement = capability.required_artifact_fields.get(
            input_ref.artifact_name
        )
        output_ref = next(
            ref
            for ref in capability.output_artifacts
            if ref.artifact_name == _STRUCTURED_LIABILITY_ARTIFACT_TYPE
        )
        return (
            _ArtifactSpec(
                name=input_ref.artifact_name,
                artifact_type=input_ref.artifact_name,
                storage_path=input_ref.storage_path,
                required_field_keys=tuple(
                    field_requirement.required_field_keys
                    if field_requirement
                    else ()
                ),
                registry_field="candidate_context_table_id",
                expected_entity_type=(
                    field_requirement.entity_type if field_requirement else None
                ),
                expected_selection_mode=(
                    field_requirement.default_selection_mode
                    if field_requirement
                    else None
                ),
            ),
            _ArtifactSpec(
                name=output_ref.artifact_name,
                artifact_type=output_ref.artifact_name,
                storage_path=output_ref.storage_path,
                required_field_keys=(),
                registry_field="structured_liability_summary_id",
                expected_entity_type=None,
                expected_selection_mode=None,
            ),
        )

    def execute_request(
        self, request: WorkerExecutionRequest
    ) -> WorkerExecutionResult:
        refs = request.input_projection.input_artifact_refs
        if self._input_spec.name not in refs:
            raise WorkerRequestRejected(
                result_status="validation_failed",
                error_code="missing_required_input_artifact_refs",
                message=(
                    "missing required input artifact refs: "
                    f"['{self._input_spec.name}']"
                ),
            )

        registry = self._get_registry(request.run_id)
        body = self._validate_and_read_ref(
            run_id=request.run_id,
            ref=refs[self._input_spec.name],
            registry=registry,
        )

        agent = self._developability_agent_factory()
        summary = agent.run_from_artifacts(
            request.run_id,
            candidate_context_table=body,
        )

        artifact_id = self._read_output_artifact_id(request.run_id)
        artifact_written = self._storage.exists(
            self._storage.run_key(
                request.run_id,
                self._output_spec.storage_path,
            )
        )
        if not artifact_id or not artifact_written:
            raise WorkerRequestRejected(
                result_status="tool_failed",
                error_code="structured_liability_summary_not_persisted",
                message=(
                    "Step 6 core returned but structured_liability_summary "
                    "was not persisted"
                ),
            )

        self._validate_output_artifact_identity(
            run_id=request.run_id,
            registry_artifact_id=artifact_id,
        )

        return self._build_result(request, summary, artifact_id)

    def _validate_and_read_ref(
        self,
        *,
        run_id: str,
        ref: Any,
        registry: Any,
    ) -> dict:
        spec = self._input_spec
        if ref.run_id != run_id:
            raise WorkerRequestRejected(
                result_status="validation_failed",
                error_code="artifact_ref_run_id_mismatch",
                message="candidate_context_table ref run_id does not match request run_id",
            )
        if ref.artifact_type != spec.artifact_type:
            raise WorkerRequestRejected(
                result_status="validation_failed",
                error_code="artifact_ref_type_mismatch",
                message=(
                    "candidate_context_table ref artifact_type "
                    f"'{ref.artifact_type}' != expected '{spec.artifact_type}'"
                ),
            )
        if ref.entity_type != spec.expected_entity_type:
            raise WorkerRequestRejected(
                result_status="validation_failed",
                error_code="artifact_ref_entity_type_mismatch",
                message=(
                    "candidate_context_table ref entity_type does not match "
                    "the AgentCard artifact contract"
                ),
            )
        if ref.selection_mode != spec.expected_selection_mode:
            raise WorkerRequestRejected(
                result_status="validation_failed",
                error_code="artifact_ref_selection_mode_unsupported",
                message=(
                    "candidate_context_table ref selection_mode is not "
                    "supported by the AgentCard artifact contract"
                ),
            )
        if not ref.can_read_from_db:
            raise WorkerRequestRejected(
                result_status="validation_failed",
                error_code="artifact_ref_not_db_readable",
                message="candidate_context_table ref is not marked can_read_from_db",
            )
        registry_id = getattr(registry.active_artifacts, spec.registry_field, None)
        if not registry_id or ref.artifact_id != registry_id:
            raise WorkerRequestRejected(
                result_status="validation_failed",
                error_code="artifact_ref_id_mismatch",
                message=(
                    "candidate_context_table ref artifact_id does not match the "
                    "current registry active artifact id"
                ),
            )
        missing_ref_fields = [
            field
            for field in spec.required_field_keys
            if field not in (ref.field_keys or [])
        ]
        if missing_ref_fields:
            raise WorkerRequestRejected(
                result_status="validation_failed",
                error_code="artifact_ref_field_keys_missing",
                message=(
                    "candidate_context_table ref field_keys missing required "
                    f"keys: {sorted(missing_ref_fields)}"
                ),
            )

        # The storage path comes only from the validated AgentCard contract;
        # request data never supplies or overrides it.
        storage_key = self._storage.run_key(run_id, spec.storage_path)
        if not self._storage.exists(storage_key):
            raise WorkerRequestRejected(
                result_status="validation_failed",
                error_code="artifact_not_found",
                message="candidate_context_table not found in worker storage",
            )
        body = self._storage.read_json(storage_key)
        missing_body_fields = [
            field
            for field in spec.required_field_keys
            if not isinstance(body, dict) or field not in body
        ]
        if missing_body_fields:
            raise WorkerRequestRejected(
                result_status="validation_failed",
                error_code="artifact_required_fields_missing",
                message=(
                    "candidate_context_table body missing required fields: "
                    f"{sorted(missing_body_fields)}"
                ),
            )
        return body

    def _validate_output_artifact_identity(
        self,
        *,
        run_id: str,
        registry_artifact_id: str,
    ) -> None:
        storage_key = self._storage.run_key(
            run_id,
            self._output_spec.storage_path,
        )
        try:
            persisted = self._storage.read_json(storage_key)
        except Exception as exc:  # noqa: BLE001 - compact, sanitized failure
            raise WorkerRequestRejected(
                result_status="tool_failed",
                error_code="structured_liability_artifact_identity_mismatch",
                message="persisted artifact identity body could not be read",
            ) from exc
        if not isinstance(persisted, dict):
            raise WorkerRequestRejected(
                result_status="tool_failed",
                error_code="structured_liability_artifact_identity_mismatch",
                message="persisted artifact identity body is not an object",
            )
        if persisted.get("artifact_id") != registry_artifact_id:
            raise WorkerRequestRejected(
                result_status="tool_failed",
                error_code="structured_liability_artifact_identity_mismatch",
                message=(
                    "persisted artifact_id identity does not match the "
                    "registry active artifact ID"
                ),
            )
        if persisted.get("run_id") != run_id:
            raise WorkerRequestRejected(
                result_status="tool_failed",
                error_code="structured_liability_artifact_identity_mismatch",
                message=(
                    "persisted run_id identity does not match the request run_id"
                ),
            )

    def _get_registry(self, run_id: str) -> Any:
        try:
            return self._registry.get(run_id)
        except Exception as exc:  # noqa: BLE001 - compact validation failure
            raise WorkerRequestRejected(
                result_status="validation_failed",
                error_code="run_registry_not_found",
                message=(
                    "run registry not found for run_id "
                    f"(reason: {type(exc).__name__})"
                ),
            ) from exc

    def _read_output_artifact_id(self, run_id: str) -> Optional[str]:
        try:
            registry = self._registry.get(run_id)
        except Exception:  # noqa: BLE001 - best effort for compact output ref
            return None
        return getattr(
            registry.active_artifacts,
            self._output_spec.registry_field,
            None,
        )

    def _build_result(
        self,
        request: WorkerExecutionRequest,
        summary: Any,
        artifact_id: str,
    ) -> WorkerExecutionResult:
        prefilter_status = str(getattr(summary, "prefilter_status", "failed"))
        if prefilter_status == "completed":
            result_status, execution_status, error_code = (
                "success",
                "completed",
                None,
            )
        elif prefilter_status in {"completed_with_missing_lanes", "partial"}:
            result_status, execution_status, error_code = (
                "partial",
                "completed",
                None,
            )
        else:
            result_status, execution_status, error_code = (
                "blocked",
                "failed",
                "developability_prefilter_blocked",
            )

        candidates = list(
            getattr(summary, "candidate_liability_results", []) or []
        )
        lanes = [
            lane
            for candidate in candidates
            for lane in (getattr(candidate, "lane_results", []) or [])
        ]
        records = [
            record
            for lane in lanes
            for record in (getattr(lane, "tool_call_records", []) or [])
        ]
        missing_flags = list(getattr(summary, "missing_input_flags", []) or [])
        tool_summary = _tool_call_summary(records)
        skipped_or_failed = sorted(
            {
                record.tool_name
                for record in records
                if record.run_status != "success"
            }
        )

        output_refs = {
            self._output_spec.name: WorkerArtifactRef(
                artifact_id=artifact_id,
                artifact_type=self._output_spec.artifact_type,
                storage_key=self._output_spec.storage_path,
                run_id=request.run_id,
            )
        }
        compact_summary = {
            "prefilter_status": prefilter_status,
            "candidate_count": len(candidates),
            "lane_count": len(lanes),
            "assessed_lane_count": sum(
                int(getattr(candidate, "assessed_lane_count", 0) or 0)
                for candidate in candidates
            ),
            "not_assessed_lane_count": sum(
                int(getattr(candidate, "not_assessed_lane_count", 0) or 0)
                for candidate in candidates
            ),
            "missing_input_flags_count": len(missing_flags),
            "output_artifact_present": True,
        }
        return WorkerExecutionResult(
            payload_type="worker_execution_result",
            payload_version="v1",
            run_id=request.run_id,
            task_id=request.task_id,
            routing_plan_id=request.routing_plan_id,
            routing_decision_id=request.routing_decision_id,
            agent_id=request.agent_id,
            capability_id=request.capability_id,
            execution_status=execution_status,  # type: ignore[arg-type]
            result_status=result_status,  # type: ignore[arg-type]
            error_code=error_code,
            output_artifact_refs=output_refs,
            compact_summary=compact_summary,
            tool_call_summary=tool_summary,
            skipped_or_failed_tools=skipped_or_failed,
        )


def _tool_call_summary(records: list[ToolCallRecord]) -> ToolCallSummary:
    attempted = success = failed = dependency_unavailable = skipped = 0
    for record in records:
        status = record.run_status
        if status in {"skipped", "not_run"}:
            skipped += 1
            continue
        attempted += 1
        if status == "success":
            success += 1
        elif status == "dependency_unavailable":
            dependency_unavailable += 1
        else:
            failed += 1
    return ToolCallSummary(
        attempted=attempted,
        success=success,
        failed=failed,
        dependency_unavailable=dependency_unavailable,
        skipped=skipped,
    )


def create_step6_flask_app(worker: Step6A2AWorker):
    """Create the Step 6 Flask app through the shared generic adapter."""
    return create_worker_flask_app(worker)


def run_step6_worker(
    *,
    url: str,
    host: str = "0.0.0.0",
    port: int = 8006,
    storage: Any = None,
    registry: Any = None,
    workflow_state: Any = None,
    mcp_client: Any = None,
    llm: Any = None,
) -> None:  # pragma: no cover - blocking production entrypoint
    """Build and serve the Step 6 worker; never scan or fall back ports."""
    from .. import deps

    assert_advertised_url_matches_port(url, port, agent_id=AGENT_ID_STEP6)
    worker = Step6A2AWorker(
        url=url,
        storage=storage or deps.get_storage(),
        registry=registry or deps.get_registry_service(),
        workflow_state=workflow_state or deps.get_workflow_state_service(),
        mcp_client=mcp_client or deps.get_mcp_client(),
        llm=llm if llm is not None else deps.get_llm_provider(),
    )
    validate_adc_agent_contract(worker.agent_card)
    serve_worker_http(worker, host=host, port=port)


__all__ = [
    "Step6A2AWorker",
    "create_step6_flask_app",
    "run_step6_worker",
]
