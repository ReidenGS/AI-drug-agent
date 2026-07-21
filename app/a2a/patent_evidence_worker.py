"""HTTP A2A worker for the unified Patent-Evidence domain core."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from python_a2a import AgentCard

from ..agents.patent_evidence_agent import (
    PatentEvidenceAgent,
    requested_lanes_from_structured_query,
)
from ..schemas.common import ToolCallRecord
from ..schemas.step_02_structured_query import StructuredQuery
from ..schemas.step_05_candidate_context_table import CandidateContextTable
from ..schemas.step_13_scientific_evidence_table import ScientificEvidenceTable
from ..schemas.step_14_patent_prior_art_table import PatentPriorArtTable
from .agent_cards import (
    AGENT_ID_PATENT_EVIDENCE,
    CAP_PATENT_EVIDENCE_WORKFLOW,
    build_patent_evidence_agent_card,
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


_REGISTRY_FIELDS = {
    "structured_query": "structured_query_id",
    "candidate_context_table": "candidate_context_table_id",
    "scientific_evidence_table": "scientific_evidence_table_id",
    "patent_prior_art_table": "patent_prior_art_table_id",
}


@dataclass(frozen=True)
class _ArtifactSpec:
    name: str
    storage_path: str
    registry_field: str
    required_fields: tuple[str, ...]
    readiness_field: str | None
    ready_values: tuple[str, ...]
    entity_type: str | None = None
    selection_mode: str | None = None


class PatentEvidenceA2AWorker:
    AGENT_ID = AGENT_ID_PATENT_EVIDENCE
    CAPABILITY_ID = CAP_PATENT_EVIDENCE_WORKFLOW

    def __init__(
        self,
        *,
        url: str,
        storage: Any,
        registry: Any,
        workflow_state: Any,
        mcp_client: Any,
        llm: Any,
        patent_evidence_agent_factory: Callable[[], PatentEvidenceAgent] | None = None,
    ) -> None:
        self.url = url
        self.agent_id = self.AGENT_ID
        self.capability_ids = frozenset({self.CAPABILITY_ID})
        self._agent_card = build_patent_evidence_agent_card(url)
        self._storage = storage
        self._registry = registry
        self._workflow_state = workflow_state
        self._mcp_client = mcp_client
        self._llm = llm
        self._agent_factory = (
            patent_evidence_agent_factory or self._default_agent_factory
        )
        self._required, self._outputs = self._derive_specs()

    @property
    def agent_card(self) -> AgentCard:
        return self._agent_card

    def health(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "agent_id": self.AGENT_ID,
            "capabilities": [self.CAPABILITY_ID],
        }

    def _default_agent_factory(self) -> PatentEvidenceAgent:
        return PatentEvidenceAgent(
            storage=self._storage,
            registry=self._registry,
            workflow_state=self._workflow_state,
            mcp_client=self._mcp_client,
            llm=self._llm,
        )

    def _derive_specs(
        self,
    ) -> tuple[dict[str, _ArtifactSpec], dict[str, _ArtifactSpec]]:
        contract = parse_adc_agent_contract(self._agent_card)
        capability = next(
            cap
            for cap in contract.capabilities
            if cap.capability_id == self.CAPABILITY_ID
        )

        def make(ref: Any) -> _ArtifactSpec:
            fields = capability.required_artifact_fields.get(ref.artifact_name)
            return _ArtifactSpec(
                name=ref.artifact_name,
                storage_path=ref.storage_path,
                registry_field=_REGISTRY_FIELDS[ref.artifact_name],
                required_fields=tuple(fields.required_field_keys if fields else ()),
                readiness_field=ref.readiness_status_field,
                ready_values=tuple(ref.ready_status_values),
                entity_type=fields.entity_type if fields else None,
                selection_mode=fields.default_selection_mode if fields else None,
            )

        return (
            {ref.artifact_name: make(ref) for ref in capability.required_input_artifacts},
            {ref.artifact_name: make(ref) for ref in capability.output_artifacts},
        )

    def execute_request(self, request: WorkerExecutionRequest) -> WorkerExecutionResult:
        if request.agent_id != self.AGENT_ID or request.capability_id != self.CAPABILITY_ID:
            raise WorkerRequestRejected(
                result_status="validation_failed",
                error_code="worker_identity_mismatch",
                message="worker request identity mismatch",
            )
        refs = request.input_projection.input_artifact_refs
        missing = sorted(set(self._required) - set(refs))
        if missing:
            raise WorkerRequestRejected(
                result_status="validation_failed",
                error_code="missing_required_input_artifact_refs",
                message=f"missing required input artifact refs: {missing}",
            )
        registry = self._get_registry(request.run_id)
        required_bodies = {
            name: self._read_and_validate(
                run_id=request.run_id,
                ref=refs[name],
                spec=spec,
                registry=registry,
                required=True,
            )
            for name, spec in self._required.items()
        }
        structured_query = StructuredQuery.model_validate(
            required_bodies["structured_query"], strict=True
        )
        lanes = requested_lanes_from_structured_query(structured_query)
        if not lanes:
            raise WorkerRequestRejected(
                result_status="validation_failed",
                error_code="patent_evidence_no_requested_lane",
                message="no Patent-Evidence lane requested",
            )

        run_result = self._agent_factory().run_from_artifacts(
            request.run_id,
            structured_query=required_bodies["structured_query"],
            candidate_context_table=required_bodies["candidate_context_table"],
        )
        persisted = {
            name: self._validate_output(request.run_id, spec)
            for name, spec in self._outputs.items()
        }
        records = [
            *run_result.evidence.tool_call_records,
            *run_result.patent.tool_call_records,
        ]
        requested_statuses = [
            run_result.evidence.review_status
            if lane == "evidence"
            else run_result.patent.patent_review_status
            for lane in lanes
        ]
        successes = sum(record.run_status == "success" for record in records)
        planning_audit = run_result.planning_audit
        inputs_unavailable = (
            planning_audit.accepted_count == 0
            and planning_audit.executed_count == 0
            and planning_audit.eligible_count == 0
            and bool(planning_audit.lane_assessments)
            and all(
                assessment.status in {"missing_inputs", "not_applicable"}
                for assessment in planning_audit.lane_assessments
            )
        )
        all_failed = successes == 0 and all(
            status not in {"not_requested"} for status in requested_statuses
        )
        if inputs_unavailable:
            result_status, execution_status, error_code = (
                "blocked",
                "failed",
                "patent_evidence_inputs_unavailable",
            )
            output_refs = {}
        elif all_failed:
            result_status, execution_status, error_code = (
                "tool_failed",
                "failed",
                "patent_evidence_workflow_failed",
            )
            output_refs: dict[str, WorkerArtifactRef] = {}
        else:
            is_partial = any(
                status
                in {
                    "partial",
                    "completed_with_warnings",
                    "failed",
                }
                for status in requested_statuses
            ) or any(
                record.run_status != "success" for record in records
            ) or planning_audit.rejected_count > 0
            result_status = "partial" if is_partial else "success"
            execution_status = "completed"
            error_code = None
            output_refs = {
                name: WorkerArtifactRef(
                    artifact_id=body["artifact_id"],
                    artifact_type=name,
                    run_id=request.run_id,
                )
                for name, body in persisted.items()
            }
        non_success = [
            {"tool_name": record.tool_name, "run_status": record.run_status}
            for record in records
            if record.run_status != "success"
        ]
        return WorkerExecutionResult(
            payload_type="worker_execution_result",
            payload_version="v1",
            run_id=request.run_id,
            task_id=request.task_id,
            routing_plan_id=request.routing_plan_id,
            routing_decision_id=request.routing_decision_id,
            agent_id=request.agent_id,
            capability_id=request.capability_id,
            retry_of_task_id=(
                request.retry_context.retry_of_task_id
                if request.retry_context is not None
                else None
            ),
            execution_status=execution_status,
            result_status=result_status,
            error_code=error_code,
            output_artifact_refs=output_refs,
            compact_summary={
                "lane_statuses": {
                    "evidence": run_result.evidence.review_status,
                    "patent": run_result.patent.patent_review_status,
                },
                "requested_lanes": lanes,
                "not_requested_lanes": sorted({"evidence", "patent"} - set(lanes)),
                "evidence_record_count": len(run_result.evidence.evidence_records),
                "patent_record_count": len(run_result.patent.patent_records),
                "output_presence": {
                    name: name in persisted for name in self._outputs
                },
                "non_success_tools": non_success,
            },
            tool_call_summary=_tool_summary(records),
            skipped_or_failed_tools=sorted(
                {item["tool_name"] for item in non_success}
            ),
            warnings=[],
        )

    def _get_registry(self, run_id: str) -> Any:
        try:
            return self._registry.get(run_id)
        except Exception as exc:  # noqa: BLE001
            raise WorkerRequestRejected(
                result_status="validation_failed",
                error_code="run_registry_not_found",
                message="run registry not found",
            ) from exc

    def _read_and_validate(
        self,
        *,
        run_id: str,
        ref: Any,
        spec: _ArtifactSpec,
        registry: Any,
        required: bool,
    ) -> dict[str, Any] | None:
        if ref.run_id != run_id or ref.artifact_type != spec.name:
            raise _input_error("artifact_ref_identity_mismatch")
        if ref.entity_type != spec.entity_type or ref.selection_mode != spec.selection_mode:
            raise _input_error("artifact_ref_projection_mismatch")
        if not ref.can_read_from_db:
            raise _input_error("artifact_ref_not_db_readable")
        active_id = getattr(registry.active_artifacts, spec.registry_field, None)
        if not active_id or ref.artifact_id != active_id:
            raise _input_error("artifact_ref_id_mismatch")
        if any(field not in (ref.field_keys or []) for field in spec.required_fields):
            raise _input_error("artifact_ref_field_keys_missing")
        key = self._storage.run_key(run_id, spec.storage_path)
        if not self._storage.exists(key):
            raise _input_error("artifact_not_found")
        try:
            body = self._storage.read_json(key)
        except Exception as exc:  # noqa: BLE001
            raise _input_error("input_artifact_malformed") from exc
        if not isinstance(body, dict):
            raise _input_error("input_artifact_malformed")
        if body.get("artifact_id") != active_id or body.get("artifact_id") != ref.artifact_id:
            raise _input_error("input_artifact_identity_mismatch")
        if body.get("run_id") != run_id:
            raise _input_error("input_artifact_identity_mismatch")
        if any(field not in body for field in spec.required_fields):
            raise _input_error("artifact_required_fields_missing")
        try:
            model = {
                "structured_query": StructuredQuery,
                "candidate_context_table": CandidateContextTable,
            }[spec.name]
            model.model_validate(body, strict=True)
        except Exception as exc:  # noqa: BLE001
            raise _input_error("input_artifact_schema_invalid") from exc
        if spec.readiness_field is not None:
            status = body.get(spec.readiness_field)
            if status not in spec.ready_values:
                if required:
                    raise _input_error("input_artifact_not_ready")
                return None
        return body

    def _validate_output(self, run_id: str, spec: _ArtifactSpec) -> dict[str, Any]:
        active = self._get_registry(run_id).active_artifacts
        active_id = getattr(active, spec.registry_field, None)
        key = self._storage.run_key(run_id, spec.storage_path)
        if not active_id or not self._storage.exists(key):
            raise _output_error("patent_evidence_outputs_not_persisted")
        try:
            body = self._storage.read_json(key)
        except Exception as exc:  # noqa: BLE001
            raise _output_error("patent_evidence_output_identity_mismatch") from exc
        if not isinstance(body, dict):
            raise _output_error("patent_evidence_output_identity_mismatch")
        if body.get("artifact_id") != active_id or body.get("run_id") != run_id:
            raise _output_error("patent_evidence_output_identity_mismatch")
        try:
            model = {
                "scientific_evidence_table": ScientificEvidenceTable,
                "patent_prior_art_table": PatentPriorArtTable,
            }[spec.name]
            model.model_validate(body, strict=True)
        except Exception as exc:  # noqa: BLE001
            raise _output_error("patent_evidence_output_schema_invalid") from exc
        return body


def _tool_summary(records: list[ToolCallRecord]) -> ToolCallSummary:
    attempted = success = failed = dependency = skipped = 0
    for record in records:
        if record.run_status in {"skipped", "not_run"}:
            skipped += 1
        else:
            attempted += 1
            if record.run_status == "success":
                success += 1
            elif record.run_status == "dependency_unavailable":
                dependency += 1
            else:
                failed += 1
    return ToolCallSummary(
        attempted=attempted,
        success=success,
        failed=failed,
        dependency_unavailable=dependency,
        skipped=skipped,
    )


def _input_error(code: str) -> WorkerRequestRejected:
    return WorkerRequestRejected(
        result_status="validation_failed", error_code=code, message=code
    )


def _output_error(code: str) -> WorkerRequestRejected:
    return WorkerRequestRejected(
        result_status="tool_failed", error_code=code, message=code
    )


def create_patent_evidence_flask_app(worker: PatentEvidenceA2AWorker):
    return create_worker_flask_app(worker)


def run_patent_evidence_worker(
    *,
    url: str,
    host: str = "0.0.0.0",
    port: int = 8014,
    storage: Any = None,
    registry: Any = None,
    workflow_state: Any = None,
    mcp_client: Any = None,
    llm: Any = None,
) -> None:  # pragma: no cover
    from .. import deps

    assert_advertised_url_matches_port(url, port, agent_id=AGENT_ID_PATENT_EVIDENCE)
    worker = PatentEvidenceA2AWorker(
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
    "PatentEvidenceA2AWorker",
    "create_patent_evidence_flask_app",
    "run_patent_evidence_worker",
]
