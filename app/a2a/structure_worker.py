"""Structure and Design workflow core for the shared HTTP A2A adapter.

One Orchestrator-facing ``structure_design_workflow`` request executes the
existing Step 7 -> Step 8 -> Step 9 production workflow inside this worker.
The generic adapter owns python-a2a transport validation; this module owns
artifact-contract validation, worker-local execution, persistence verification,
and the compact result only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

from python_a2a import AgentCard

from ..agents.structure_and_design_agent import StructureAndDesignAgent
from .agent_cards import (
    AGENT_ID_STRUCTURE,
    CAP_STRUCTURE_DESIGN_WORKFLOW,
    build_structure_agent_card,
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


@dataclass(frozen=True)
class _ArtifactSpec:
    name: str
    artifact_type: str
    storage_path: str
    registry_field: str
    required_field_keys: tuple[str, ...] = ()
    expected_entity_type: Optional[str] = None
    expected_selection_mode: Optional[str] = None
    readiness_status_field: Optional[str] = None
    ready_status_values: tuple[str, ...] = ()


@dataclass(frozen=True)
class _PersistedOutput:
    spec: _ArtifactSpec
    artifact_id: str
    body: dict[str, Any]


class StructureA2AWorker:
    """Request-based Structure worker implementing the generic WorkerCore."""

    AGENT_ID = AGENT_ID_STRUCTURE
    CAPABILITY_ID = CAP_STRUCTURE_DESIGN_WORKFLOW

    def __init__(
        self,
        *,
        url: str,
        storage: Any,
        registry: Any,
        workflow_state: Any,
        mcp_client: Any,
        llm: Any = None,
        structure_agent_factory: Optional[
            Callable[[], StructureAndDesignAgent]
        ] = None,
    ) -> None:
        self.url = url
        self.agent_id = self.AGENT_ID
        self.capability_ids = frozenset({self.CAPABILITY_ID})
        self._agent_card = build_structure_agent_card(url)
        self._storage = storage
        self._registry = registry
        self._workflow_state = workflow_state
        self._mcp_client = mcp_client
        self._llm = llm
        self._structure_agent_factory = (
            structure_agent_factory or self._default_agent_factory
        )
        (
            self._required_specs,
            self._optional_specs,
            self._output_specs,
            self._internal_execution_order,
        ) = self._derive_contract()

    @property
    def agent_card(self) -> AgentCard:
        return self._agent_card

    def health(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "agent_id": self.AGENT_ID,
            "capabilities": sorted(self.capability_ids),
        }

    def _default_agent_factory(self) -> StructureAndDesignAgent:
        return StructureAndDesignAgent(
            storage=self._storage,
            registry=self._registry,
            workflow_state=self._workflow_state,
            mcp_client=self._mcp_client,
            llm=self._llm,
        )

    def _derive_contract(
        self,
    ) -> tuple[
        tuple[_ArtifactSpec, ...],
        tuple[_ArtifactSpec, ...],
        tuple[_ArtifactSpec, ...],
        tuple[str, ...],
    ]:
        contract = parse_adc_agent_contract(self._agent_card)
        capability = next(
            cap
            for cap in contract.capabilities
            if cap.capability_id == self.CAPABILITY_ID
        )

        registry_fields = {
            "raw_request_record": "raw_request_record_id",
            "structured_query": "structured_query_id",
            "candidate_context_table": "candidate_context_table_id",
            "structured_liability_summary": "structured_liability_summary_id",
            "prepared_structure_input_package": (
                "prepared_structure_input_package_id"
            ),
            "structure_prediction_and_interface_results": (
                "structure_prediction_and_interface_results_id"
            ),
            "structure_variant_and_compound_screening": (
                "structure_variant_and_compound_screening_id"
            ),
        }

        def _spec(ref: Any) -> _ArtifactSpec:
            fields = capability.required_artifact_fields.get(ref.artifact_name)
            return _ArtifactSpec(
                name=ref.artifact_name,
                artifact_type=ref.artifact_name,
                storage_path=ref.storage_path,
                registry_field=registry_fields[ref.artifact_name],
                required_field_keys=tuple(
                    fields.required_field_keys if fields else ()
                ),
                expected_entity_type=(fields.entity_type if fields else None),
                expected_selection_mode=(
                    fields.default_selection_mode if fields else None
                ),
                readiness_status_field=ref.readiness_status_field,
                ready_status_values=tuple(ref.ready_status_values),
            )

        return (
            tuple(_spec(ref) for ref in capability.required_input_artifacts),
            tuple(_spec(ref) for ref in capability.optional_input_artifacts),
            tuple(_spec(ref) for ref in capability.output_artifacts),
            tuple(capability.internal_execution_order),
        )

    def execute_request(
        self,
        request: WorkerExecutionRequest,
    ) -> WorkerExecutionResult:
        refs = request.input_projection.input_artifact_refs
        missing = [spec.name for spec in self._required_specs if spec.name not in refs]
        if missing:
            raise WorkerRequestRejected(
                result_status="validation_failed",
                error_code="missing_required_input_artifact_refs",
                message=f"missing required input artifact refs: {sorted(missing)}",
            )

        registry = self._get_registry(request.run_id)
        bodies = {
            spec.name: self._validate_and_read_ref(
                run_id=request.run_id,
                spec=spec,
                ref=refs[spec.name],
                registry=registry,
            )
            for spec in self._required_specs
        }
        for spec in self._optional_specs:
            if spec.name in refs:
                self._validate_and_read_ref(
                    run_id=request.run_id,
                    spec=spec,
                    ref=refs[spec.name],
                    registry=registry,
                    optional=True,
                )

        agent = self._structure_agent_factory()
        agent.run_workflow_from_artifacts(
            request.run_id,
            raw_request_record=bodies["raw_request_record"],
            structured_query=bodies["structured_query"],
            candidate_context_table=bodies["candidate_context_table"],
        )

        outputs = self._validate_output_artifacts(request.run_id)
        return self._build_result(request, outputs)

    def _validate_and_read_ref(
        self,
        *,
        run_id: str,
        spec: _ArtifactSpec,
        ref: Any,
        registry: Any,
        optional: bool = False,
    ) -> dict[str, Any] | None:
        if ref.run_id != run_id:
            raise WorkerRequestRejected(
                result_status="validation_failed",
                error_code="artifact_ref_run_id_mismatch",
                message=f"input artifact ref '{spec.name}' run_id mismatch",
            )
        if ref.artifact_type != spec.artifact_type:
            raise WorkerRequestRejected(
                result_status="validation_failed",
                error_code="artifact_ref_type_mismatch",
                message=f"input artifact ref '{spec.name}' artifact_type mismatch",
            )
        if spec.expected_entity_type is not None and (
            ref.entity_type != spec.expected_entity_type
        ):
            raise WorkerRequestRejected(
                result_status="validation_failed",
                error_code="artifact_ref_entity_type_mismatch",
                message=f"input artifact ref '{spec.name}' entity_type mismatch",
            )
        if spec.expected_selection_mode is not None and (
            ref.selection_mode != spec.expected_selection_mode
        ):
            raise WorkerRequestRejected(
                result_status="validation_failed",
                error_code="artifact_ref_selection_mode_unsupported",
                message=f"input artifact ref '{spec.name}' selection_mode unsupported",
            )
        if not ref.can_read_from_db:
            raise WorkerRequestRejected(
                result_status="validation_failed",
                error_code="artifact_ref_not_db_readable",
                message=f"input artifact ref '{spec.name}' is not DB-readable",
            )
        registry_id = getattr(
            registry.active_artifacts,
            spec.registry_field,
            None,
        )
        if not registry_id or ref.artifact_id != registry_id:
            raise WorkerRequestRejected(
                result_status="validation_failed",
                error_code="artifact_ref_id_mismatch",
                message=(
                    f"input artifact ref '{spec.name}' does not match the "
                    "registry active artifact ID"
                ),
            )
        missing_ref_fields = [
            key
            for key in spec.required_field_keys
            if key not in (ref.field_keys or [])
        ]
        if missing_ref_fields:
            raise WorkerRequestRejected(
                result_status="validation_failed",
                error_code="artifact_ref_field_keys_missing",
                message=(
                    f"input artifact ref '{spec.name}' field_keys missing "
                    f"required keys: {sorted(missing_ref_fields)}"
                ),
            )

        # Only the published AgentCard controls storage lookup. No request field
        # is accepted as a storage path.
        storage_key = self._storage.run_key(run_id, spec.storage_path)
        if not self._storage.exists(storage_key):
            raise WorkerRequestRejected(
                result_status="validation_failed",
                error_code="artifact_not_found",
                message=f"input artifact '{spec.name}' not found in worker storage",
            )
        try:
            body = self._storage.read_json(storage_key)
        except Exception as exc:  # noqa: BLE001 - sanitized input failure
            raise WorkerRequestRejected(
                result_status="validation_failed",
                error_code="input_artifact_identity_mismatch",
                message=f"input artifact '{spec.name}' identity body could not be read",
            ) from exc
        self._validate_input_artifact_identity(
            spec=spec,
            body=body,
            ref=ref,
            registry_artifact_id=registry_id,
            run_id=run_id,
        )
        missing_body_fields = [
            key
            for key in spec.required_field_keys
            if key not in body
        ]
        if missing_body_fields:
            raise WorkerRequestRejected(
                result_status="validation_failed",
                error_code="artifact_required_fields_missing",
                message=(
                    f"input artifact '{spec.name}' body missing required fields: "
                    f"{sorted(missing_body_fields)}"
                ),
            )
        if not self._input_artifact_is_ready(spec=spec, body=body):
            if optional:
                return None
            raise WorkerRequestRejected(
                result_status="validation_failed",
                error_code="input_artifact_not_ready",
                message=f"input artifact '{spec.name}': input_artifact_not_ready",
            )
        return body

    @staticmethod
    def _input_artifact_is_ready(
        *, spec: _ArtifactSpec, body: dict[str, Any]
    ) -> bool:
        if spec.readiness_status_field is None:
            return True
        status = body.get(spec.readiness_status_field)
        return isinstance(status, str) and status in spec.ready_status_values

    @staticmethod
    def _validate_input_artifact_identity(
        *,
        spec: _ArtifactSpec,
        body: Any,
        ref: Any,
        registry_artifact_id: str,
        run_id: str,
    ) -> None:
        if not isinstance(body, dict):
            raise WorkerRequestRejected(
                result_status="validation_failed",
                error_code="input_artifact_identity_mismatch",
                message=f"input artifact '{spec.name}' identity body is not an object",
            )
        if (
            body.get("artifact_id") != registry_artifact_id
            or body.get("artifact_id") != ref.artifact_id
        ):
            raise WorkerRequestRejected(
                result_status="validation_failed",
                error_code="input_artifact_identity_mismatch",
                message=f"input artifact '{spec.name}' artifact_id identity mismatch",
            )
        if body.get("run_id") != run_id:
            raise WorkerRequestRejected(
                result_status="validation_failed",
                error_code="input_artifact_identity_mismatch",
                message=f"input artifact '{spec.name}' run_id identity mismatch",
            )

    def _validate_output_artifacts(
        self,
        run_id: str,
    ) -> tuple[_PersistedOutput, ...]:
        registry = self._get_registry(run_id)
        outputs: list[_PersistedOutput] = []
        for spec in self._output_specs:
            artifact_id = getattr(
                registry.active_artifacts,
                spec.registry_field,
                None,
            )
            storage_key = self._storage.run_key(run_id, spec.storage_path)
            if not artifact_id or not self._storage.exists(storage_key):
                raise WorkerRequestRejected(
                    result_status="tool_failed",
                    error_code="structure_workflow_artifact_not_persisted",
                    message=f"required internal output '{spec.name}' was not persisted",
                )
            try:
                body = self._storage.read_json(storage_key)
            except Exception as exc:  # noqa: BLE001 - sanitized compact failure
                raise WorkerRequestRejected(
                    result_status="tool_failed",
                    error_code="structure_workflow_artifact_identity_mismatch",
                    message=f"persisted output '{spec.name}' identity could not be read",
                ) from exc
            if not isinstance(body, dict):
                raise WorkerRequestRejected(
                    result_status="tool_failed",
                    error_code="structure_workflow_artifact_identity_mismatch",
                    message=f"persisted output '{spec.name}' identity is not an object",
                )
            if body.get("artifact_id") != artifact_id:
                raise WorkerRequestRejected(
                    result_status="tool_failed",
                    error_code="structure_workflow_artifact_identity_mismatch",
                    message=f"persisted output '{spec.name}' artifact_id mismatch",
                )
            if body.get("run_id") != run_id:
                raise WorkerRequestRejected(
                    result_status="tool_failed",
                    error_code="structure_workflow_artifact_identity_mismatch",
                    message=f"persisted output '{spec.name}' run_id mismatch",
                )
            outputs.append(
                _PersistedOutput(
                    spec=spec,
                    artifact_id=artifact_id,
                    body=body,
                )
            )
        return tuple(outputs)

    def _get_registry(self, run_id: str) -> Any:
        try:
            return self._registry.get(run_id)
        except Exception as exc:  # noqa: BLE001 - sanitized compact failure
            raise WorkerRequestRejected(
                result_status="validation_failed",
                error_code="run_registry_not_found",
                message=f"run registry not found (reason: {type(exc).__name__})",
            ) from exc

    def _build_result(
        self,
        request: WorkerExecutionRequest,
        outputs: tuple[_PersistedOutput, ...],
    ) -> WorkerExecutionResult:
        by_name = {output.spec.name: output for output in outputs}
        step7 = by_name["prepared_structure_input_package"].body
        step8 = by_name["structure_prediction_and_interface_results"].body
        step9 = by_name["structure_variant_and_compound_screening"].body

        step7_status = str(step7.get("structure_preparation_status", "failed"))
        step8_status = str(step8.get("structure_modeling_status", "failed"))
        step9_status = str(step9.get("screening_status", "failed"))
        if step7_status == "failed":
            result_status, execution_status, error_code = (
                "blocked",
                "failed",
                "structure_workflow_blocked",
            )
        elif (
            step7_status == "ok"
            and step8_status == "ok"
            and step9_status in {"ok", "skipped"}
        ):
            result_status, execution_status, error_code = (
                "success",
                "completed",
                None,
            )
        else:
            result_status, execution_status, error_code = (
                "partial",
                "completed",
                None,
            )

        records = [
            *list(step7.get("structure_tool_call_records") or []),
            *list(step8.get("tool_call_records") or []),
            *list(step9.get("tool_call_records") or []),
        ]
        compact_summary = {
            "internal_execution_order": list(self._internal_execution_order),
            "completed_internal_steps": list(self._internal_execution_order),
            "step7_status": step7_status,
            "step7_prepared_input_count": len(
                step7.get("prepared_structure_inputs") or []
            ),
            "step7_unresolved_resource_count": len(
                step7.get("unresolved_resource_refs") or []
            ),
            "step7_preparation_warning_count": len(
                step7.get("preparation_warnings") or []
            ),
            "step8_status": step8_status,
            "step8_candidate_result_count": len(
                step8.get("candidate_structure_results") or []
            ),
            "step8_output_artifact_count": len(
                step8.get("output_artifacts") or []
            ),
            "step9_status": step9_status,
            "step9_stage1_selected_tool_count": len(
                step9.get("step9_stage1_selected_tools") or []
            ),
            "step9_stage2_mapped_tool_count": len(
                step9.get("step9_stage2_mapped_tools") or []
            ),
            "step9_executed_tool_count": len(
                step9.get("step9_runtime_executed_tools") or []
            ),
            "output_artifact_count": len(outputs),
        }
        output_refs = {
            output.spec.name: WorkerArtifactRef(
                artifact_id=output.artifact_id,
                artifact_type=output.spec.artifact_type,
                storage_key=output.spec.storage_path,
                run_id=request.run_id,
            )
            for output in outputs
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
            retry_of_task_id=(
                request.retry_context.retry_of_task_id
                if request.retry_context is not None
                else None
            ),
            execution_status=execution_status,  # type: ignore[arg-type]
            result_status=result_status,  # type: ignore[arg-type]
            error_code=error_code,
            output_artifact_refs=output_refs,
            compact_summary=compact_summary,
            tool_call_summary=_tool_call_summary(records),
            skipped_or_failed_tools=sorted(
                {
                    str(record.get("tool_name"))
                    for record in records
                    if record.get("run_status") != "success"
                    and record.get("tool_name")
                }
            ),
        )


def _tool_call_summary(records: list[dict[str, Any]]) -> ToolCallSummary:
    attempted = success = failed = dependency_unavailable = skipped = 0
    for record in records:
        status = record.get("run_status")
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


def create_structure_flask_app(worker: StructureA2AWorker):
    """Create the Structure HTTP app through the shared generic adapter."""
    return create_worker_flask_app(worker)


def run_structure_worker(
    *,
    url: str,
    host: str = "0.0.0.0",
    port: int = 8009,
    storage: Any = None,
    registry: Any = None,
    workflow_state: Any = None,
    mcp_client: Any = None,
    llm: Any = None,
) -> None:  # pragma: no cover - blocking production entrypoint
    """Build and serve the Structure worker; never scan or fall back ports."""
    from .. import deps

    assert_advertised_url_matches_port(url, port, agent_id=AGENT_ID_STRUCTURE)
    worker = StructureA2AWorker(
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
    "StructureA2AWorker",
    "create_structure_flask_app",
    "run_structure_worker",
]
