"""Step 5 Candidate Context worker core (Turn B).

Domain half of the Step 5 A2A worker. It plugs into the generic
:class:`app.a2a.worker_server.A2AWorkerAdapter` (which owns all python-a2a
transport, envelope parsing, metadata/identity/privacy validation, and result
encoding). This module owns only Step 5 concerns:

- the Step 5 AgentCard,
- deterministic validation + resolution of the required input artifact refs,
- worker-local execution via ``CandidateContextAgent.run_from_artifacts`` (the
  request-based entry with NO run_step_plan gate),
- the compact Step 5 ``WorkerExecutionResult`` (build status, candidate count,
  tool-call summary, output artifact ref).

The Orchestrator must never call ``CandidateContextAgent`` directly; only this
worker process does, and only after the adapter + this core have validated the
task. No raw artifact body / sequence / API key / prompt / LLM response is ever
placed in the A2A result — only compact refs and summaries.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

from python_a2a import AgentCard

from ..agents.candidate_context_agent import CandidateContextAgent
from ..schemas.common import ToolCallRecord
from .agent_cards import (
    AGENT_ID_STEP5,
    CAP_STEP5_CANDIDATE_CONTEXT,
    build_step5_agent_card,
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
_CANDIDATE_CONTEXT_STORAGE_KEY = "candidate_context_table.json"


@dataclass(frozen=True)
class _RequiredArtifactSpec:
    """Per-required-artifact validation contract, derived from the AgentCard."""

    name: str
    artifact_type: str
    storage_path: str
    required_field_keys: tuple[str, ...]
    registry_field: str


class Step5A2AWorker:
    """Step 5 worker core (implements the generic ``WorkerCore`` protocol)."""

    AGENT_ID = AGENT_ID_STEP5
    CAPABILITY_ID = CAP_STEP5_CANDIDATE_CONTEXT

    def __init__(
        self,
        *,
        url: str,
        storage: Any,
        registry: Any,
        workflow_state: Any,
        mcp_client: Any,
        llm: Any = None,
        candidate_agent_factory: Optional[Callable[[], CandidateContextAgent]] = None,
    ) -> None:
        self.url = url
        self.agent_id = self.AGENT_ID
        self.capability_ids = frozenset({self.CAPABILITY_ID})
        self._agent_card = build_step5_agent_card(url)
        self._storage = storage
        self._registry = registry
        self._workflow_state = workflow_state
        self._mcp_client = mcp_client
        self._llm = llm
        # Injectable so tests can observe that CandidateContextAgent is NOT run on
        # the validation-failure paths. Default builds the real production agent.
        self._candidate_agent_factory = candidate_agent_factory or self._default_agent_factory
        # Derive the required-artifact validation contract straight from the card
        # so the worker and its published AgentCard never drift apart.
        self._required_specs = self._derive_required_specs()

    # ── WorkerCore surface ──────────────────────────────────────────────────
    @property
    def agent_card(self) -> AgentCard:
        return self._agent_card

    def health(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "agent_id": self.AGENT_ID,
            "capabilities": sorted(self.capability_ids),
        }

    def _default_agent_factory(self) -> CandidateContextAgent:
        return CandidateContextAgent(
            storage=self._storage,
            registry=self._registry,
            workflow_state=self._workflow_state,
            mcp_client=self._mcp_client,
            llm=self._llm,
        )

    def _derive_required_specs(self) -> tuple[_RequiredArtifactSpec, ...]:
        contract = parse_adc_agent_contract(self._agent_card)
        cap = next(c for c in contract.capabilities if c.capability_id == self.CAPABILITY_ID)
        specs: list[_RequiredArtifactSpec] = []
        for ref in cap.required_input_artifacts:
            field_req = cap.required_artifact_fields.get(ref.artifact_name)
            field_keys = tuple(field_req.required_field_keys) if field_req else ()
            specs.append(
                _RequiredArtifactSpec(
                    name=ref.artifact_name,
                    # For Step 5's entry artifacts the artifact_type == artifact_name.
                    artifact_type=ref.artifact_name,
                    storage_path=ref.storage_path,
                    required_field_keys=field_keys,
                    registry_field=f"{ref.artifact_name}_id",
                )
            )
        return tuple(specs)

    # ── worker-local execution ──────────────────────────────────────────────
    def execute_request(self, request: WorkerExecutionRequest) -> WorkerExecutionResult:
        """Validate + resolve the required input artifact refs, then run the
        request-based Step 5 core. Raises :class:`WorkerRequestRejected` for
        controlled validation/blocked/tool failures (the adapter turns those into
        a compact non-success result)."""
        run_id = request.run_id
        refs = request.input_projection.input_artifact_refs

        # (a) every required ref must be present.
        missing = [spec.name for spec in self._required_specs if spec.name not in refs]
        if missing:
            raise WorkerRequestRejected(
                result_status="validation_failed",
                error_code="missing_required_input_artifact_refs",
                message=f"missing required input artifact refs: {sorted(missing)}",
            )

        reg = self._get_registry(run_id)
        resolved_bodies: dict[str, dict] = {}
        for spec in self._required_specs:
            ref = refs[spec.name]
            resolved_bodies[spec.name] = self._validate_and_read_ref(
                run_id=run_id, spec=spec, ref=ref, registry=reg
            )

        # (b) run the request-based core (no run_step_plan gate). The worker reads
        # the bodies itself; only compact refs/summaries leave this process.
        agent = self._candidate_agent_factory()
        table = agent.run_from_artifacts(
            run_id,
            raw_request_record=resolved_bodies["raw_request_record"],
            structured_query=resolved_bodies["structured_query"],
            structured_query_artifact_id=refs["structured_query"].artifact_id,
        )

        # (c) confirm the artifact + registry pointer were really written.
        artifact_id = self._read_candidate_context_artifact_id(run_id)
        artifact_written = self._storage.exists(
            self._storage.run_key(run_id, _CANDIDATE_CONTEXT_STORAGE_KEY)
        )
        if not artifact_id or not artifact_written:
            raise WorkerRequestRejected(
                result_status="tool_failed",
                error_code="candidate_context_not_persisted",
                message="Step 5 core returned but candidate_context_table was not persisted",
            )

        self._validate_output_artifact_identity(
            run_id=run_id,
            registry_artifact_id=artifact_id,
        )

        return self._build_result(request, table, artifact_id)

    def _validate_and_read_ref(
        self, *, run_id: str, spec: _RequiredArtifactSpec, ref: Any, registry: Any
    ) -> dict:
        if ref.run_id != run_id:
            raise WorkerRequestRejected(
                result_status="validation_failed",
                error_code="artifact_ref_run_id_mismatch",
                message=f"input artifact ref '{spec.name}' run_id does not match request run_id",
            )
        if ref.artifact_type != spec.artifact_type:
            raise WorkerRequestRejected(
                result_status="validation_failed",
                error_code="artifact_ref_type_mismatch",
                message=(
                    f"input artifact ref '{spec.name}' artifact_type "
                    f"'{ref.artifact_type}' != expected '{spec.artifact_type}'"
                ),
            )
        if not ref.can_read_from_db:
            raise WorkerRequestRejected(
                result_status="validation_failed",
                error_code="artifact_ref_not_db_readable",
                message=f"input artifact ref '{spec.name}' is not marked can_read_from_db",
            )
        registry_id = getattr(registry.active_artifacts, spec.registry_field, None)
        if not registry_id or ref.artifact_id != registry_id:
            raise WorkerRequestRejected(
                result_status="validation_failed",
                error_code="artifact_ref_id_mismatch",
                message=(
                    f"input artifact ref '{spec.name}' artifact_id does not match the "
                    "current registry active artifact id"
                ),
            )
        missing_keys = [k for k in spec.required_field_keys if k not in (ref.field_keys or [])]
        if missing_keys:
            raise WorkerRequestRejected(
                result_status="validation_failed",
                error_code="artifact_ref_field_keys_missing",
                message=(
                    f"input artifact ref '{spec.name}' field_keys missing required "
                    f"keys: {sorted(missing_keys)}"
                ),
            )
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
        body_missing = [
            k for k in spec.required_field_keys if k not in body
        ]
        if body_missing:
            raise WorkerRequestRejected(
                result_status="validation_failed",
                error_code="artifact_required_fields_missing",
                message=(
                    f"input artifact '{spec.name}' body missing required fields: "
                    f"{sorted(body_missing)}"
                ),
            )
        return body

    @staticmethod
    def _validate_input_artifact_identity(
        *,
        spec: _RequiredArtifactSpec,
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

    def _get_registry(self, run_id: str) -> Any:
        try:
            return self._registry.get(run_id)
        except Exception as exc:  # noqa: BLE001 — surface as a compact validation failure
            raise WorkerRequestRejected(
                result_status="validation_failed",
                error_code="run_registry_not_found",
                message=f"run registry not found for run_id (reason: {type(exc).__name__})",
            ) from exc

    # ── result builder ──────────────────────────────────────────────────────
    def _build_result(
        self, request: WorkerExecutionRequest, table: Any, artifact_id: str
    ) -> WorkerExecutionResult:
        records: list[ToolCallRecord] = list(getattr(table, "tool_call_records", []) or [])
        summary = _tool_call_summary(records)
        skipped_or_failed = sorted({r.tool_name for r in records if r.run_status != "success"})
        build_status = str(getattr(table, "context_build_status", "unknown"))
        # Honest status mapping — a failed build is NEVER wrapped as success/partial.
        #   ok      -> success  / completed
        #   partial -> partial  / completed
        #   failed  -> tool_failed / failed  (A2A FAILED), error_code set
        #   other   -> partial  / completed  (conservative, still not success)
        if build_status == "ok":
            result_status, execution_status, error_code = "success", "completed", None
        elif build_status == "failed":
            result_status, execution_status, error_code = (
                "tool_failed",
                "failed",
                "candidate_context_build_failed",
            )
        else:  # "partial" or any other non-ok status
            result_status, execution_status, error_code = "partial", "completed", None

        # The artifact WAS persisted (checked by the caller), so we keep the
        # compact output ref / summary / tool summary for audit even on failure.
        output_refs = {
            _CANDIDATE_CONTEXT_ARTIFACT_TYPE: WorkerArtifactRef(
                artifact_id=artifact_id,
                artifact_type=_CANDIDATE_CONTEXT_ARTIFACT_TYPE,
                storage_key=_CANDIDATE_CONTEXT_STORAGE_KEY,
                run_id=request.run_id,
            )
        }
        missing_flags = [str(f) for f in (getattr(table, "missing_context_flags", []) or [])]
        compact_summary: dict[str, Any] = {
            "context_build_status": build_status,
            "candidate_count": len(getattr(table, "candidate_records", []) or []),
            "tool_call_count": len(records),
            "missing_context_flags_count": len(missing_flags),
            "missing_context_flags": missing_flags,
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
            tool_call_summary=summary,
            skipped_or_failed_tools=skipped_or_failed,
        )

    def _read_candidate_context_artifact_id(self, run_id: str) -> Optional[str]:
        try:
            reg = self._registry.get(run_id)
        except Exception:  # noqa: BLE001 — best-effort for the compact output ref
            return None
        return getattr(reg.active_artifacts, "candidate_context_table_id", None)

    def _validate_output_artifact_identity(
        self,
        *,
        run_id: str,
        registry_artifact_id: str,
    ) -> None:
        storage_key = self._storage.run_key(
            run_id,
            _CANDIDATE_CONTEXT_STORAGE_KEY,
        )
        try:
            persisted = self._storage.read_json(storage_key)
        except Exception as exc:  # noqa: BLE001 - compact, sanitized failure
            raise WorkerRequestRejected(
                result_status="tool_failed",
                error_code="candidate_context_artifact_identity_mismatch",
                message="persisted artifact identity body could not be read",
            ) from exc
        if not isinstance(persisted, dict):
            raise WorkerRequestRejected(
                result_status="tool_failed",
                error_code="candidate_context_artifact_identity_mismatch",
                message="persisted artifact identity body is not an object",
            )
        if persisted.get("artifact_id") != registry_artifact_id:
            raise WorkerRequestRejected(
                result_status="tool_failed",
                error_code="candidate_context_artifact_identity_mismatch",
                message=(
                    "persisted artifact_id identity does not match the "
                    "registry active artifact ID"
                ),
            )
        if persisted.get("run_id") != run_id:
            raise WorkerRequestRejected(
                result_status="tool_failed",
                error_code="candidate_context_artifact_identity_mismatch",
                message=(
                    "persisted run_id identity does not match the request run_id"
                ),
            )


# ── helpers ──────────────────────────────────────────────────────────────────
def _tool_call_summary(records: list[ToolCallRecord]) -> ToolCallSummary:
    attempted = success = failed = dependency_unavailable = skipped = 0
    for r in records:
        status = r.run_status
        if status in {"skipped", "not_run"}:
            skipped += 1
            continue
        attempted += 1
        if status == "success":
            success += 1
        elif status == "dependency_unavailable":
            dependency_unavailable += 1
        else:  # failed / error / pending / anything else
            failed += 1
    return ToolCallSummary(
        attempted=attempted,
        success=success,
        failed=failed,
        dependency_unavailable=dependency_unavailable,
        skipped=skipped,
    )


# ── server assembly (thin wrappers over the generic transport) ───────────────
def create_step5_flask_app(worker: Step5A2AWorker):
    """Create the Flask app serving the Step 5 worker over real HTTP A2A."""
    return create_worker_flask_app(worker)


def run_step5_worker(
    *,
    url: str,
    host: str = "0.0.0.0",
    port: int = 8005,
    storage: Any = None,
    registry: Any = None,
    workflow_state: Any = None,
    mcp_client: Any = None,
    llm: Any = None,
) -> None:  # pragma: no cover - production entrypoint, not exercised in unit tests
    """Production entrypoint: build the Step 5 worker from app.deps and serve it.

    Fails fast if ``host:port`` is already bound (no silent fallback to another
    port and no fallback to an in-process call). ``url`` is what the AgentCard
    advertises; its effective port (explicit, or scheme default 80/443) must
    match the bind ``port`` so discovery never drifts.
    """
    from .. import deps

    assert_advertised_url_matches_port(url, port, agent_id=AGENT_ID_STEP5)

    worker = Step5A2AWorker(
        url=url,
        storage=storage or deps.get_storage(),
        registry=registry or deps.get_registry_service(),
        workflow_state=workflow_state or deps.get_workflow_state_service(),
        mcp_client=mcp_client or deps.get_mcp_client(),
        llm=llm if llm is not None else deps.get_llm_provider(),
    )
    # Surface a clear error early if the card contract is malformed.
    validate_adc_agent_contract(worker.agent_card)
    serve_worker_http(worker, host=host, port=port)


__all__ = [
    "Step5A2AWorker",
    "create_step5_flask_app",
    "run_step5_worker",
]
