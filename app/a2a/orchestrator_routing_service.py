"""Turn F1-C2 Orchestrator routing-plan service without task dispatch.

The service composes frozen HTTP discovery, privacy-safe context projection,
one structured LLM proposal, deterministic routing validation, in-memory Task
construction, and compact plan persistence. It never sends a Task or executes a
worker. Runtime completion state is accepted only as a revalidation argument.
"""

from __future__ import annotations

import re
import threading
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from pydantic import ValidationError

from app.llm.json_task_validation import validate_task_shape
from app.llm.provider import LLMProvider
from app.schemas.step_03_input_readiness import InputReadinessStatus
from app.schemas.worker_routing_plan import (
    LoopDecision,
    OrchestratorRouteDecision,
    OrchestratorRoutingProposal,
    RejectedRoutingDecision,
    ValidatedRoutingDecision,
    WorkerRoutingPlan,
)
from app.services.artifact_registry_service import ArtifactRegistryService
from app.services.storage_service import Storage
from app.utils.ids import new_artifact_id, new_routing_plan_id
from app.utils.time import now_iso

from .contracts import WorkerExecutionResult
from .orchestrator_completion_validation import (
    CanonicalArtifactSpec,
    CompletionArtifactValidationError,
    artifact_id_fingerprint,
    artifact_requirement,
    canonical_artifact_specs,
    validate_worker_output_artifacts,
)
from .orchestrator_context_projection import (
    contains_unsafe_routing_text,
    project_orchestrator_context,
)
from .orchestrator_discovery import WorkerDiscoveryService
from .orchestrator_routing_prompt import (
    ORCHESTRATOR_ROUTING_SYSTEM_PROMPT,
    ORCHESTRATOR_ROUTING_USER_TASK,
)
from .orchestrator_routing_validation import (
    RoutingValidationResult,
    inspect_declared_artifact,
    validate_orchestrator_routing,
)
from .orchestrator_task_builder import (
    PreparedA2ATask,
    build_orchestrator_worker_task,
)

_PLAN_STORAGE_KEY = "inputs/worker_routing_plan.json"
_STRUCTURED_QUERY_STORAGE_KEY = "inputs/structured_query.json"
_READINESS_STORAGE_KEY = "inputs/input_readiness_status.json"
_COMPACT_CODE = re.compile(r"^[A-Za-z0-9_.:-]{1,128}$")
_SAFE_FIELD_NAME = re.compile(r"^[A-Za-z0-9_.:-]+$")


class OrchestratorRoutingServiceError(RuntimeError):
    """Compact fail-closed service error; message is always a fixed code."""


@dataclass(frozen=True)
class OrchestratorRoutingServiceResult:
    plan: WorkerRoutingPlan
    plan_artifact_id: str
    prepared_tasks: tuple[PreparedA2ATask, ...]
    reused_existing_plan: bool
    llm_called: bool
    discovery_performed: bool


class OrchestratorRoutingService:
    """Build and revalidate one compact routing plan per run."""

    def __init__(
        self,
        *,
        discovery: WorkerDiscoveryService,
        storage: Storage,
        registry: ArtifactRegistryService,
        llm: LLMProvider,
    ) -> None:
        self._discovery = discovery
        self._storage = storage
        self._registry = registry
        self._llm = llm
        self._meta_lock = threading.Lock()
        self._run_locks: dict[str, threading.Lock] = {}
        self._known_plan_id_by_run: dict[str, str] = {}
        self._known_artifact_id_by_run: dict[str, str] = {}
        self._discovered_runs: set[str] = set()

    def plan_for_run(self, run_id: str) -> OrchestratorRoutingServiceResult:
        """Create one plan for ``run_id`` or reuse its valid active plan."""
        with self._get_run_lock(run_id):
            active_plan_id = self._active_plan_artifact_id(run_id)
            if active_plan_id:
                plan = self._load_plan(run_id, active_plan_id)
                discovery_performed = self._ensure_discovery(run_id)
                return self._revalidate_loaded_plan(
                    run_id=run_id,
                    artifact_id=active_plan_id,
                    previous=plan,
                    completed_results=(),
                    discovery_performed=discovery_performed,
                )

            snapshot = self._discovery.discover_for_run(run_id)
            self._discovered_runs.add(run_id)
            compact_catalog = self._discovery.get_compact_card_catalog(run_id)
            structured_query = self._load_input_artifact(
                run_id=run_id,
                artifact_name="structured_query",
                storage_path=_STRUCTURED_QUERY_STORAGE_KEY,
            )
            readiness_body = self._load_input_artifact(
                run_id=run_id,
                artifact_name="input_readiness_status",
                storage_path=_READINESS_STORAGE_KEY,
            )
            try:
                readiness_model = InputReadinessStatus.model_validate(readiness_body)
            except ValidationError as exc:
                raise OrchestratorRoutingServiceError(
                    "input_readiness_status_schema_invalid"
                ) from exc
            artifact_summary = self._available_artifact_summary(run_id)

            if readiness_model.input_readiness_status != "ready":
                needs_input = (
                    readiness_model.input_readiness_status == "needs_user_input"
                )
                plan = self._empty_plan(
                    run_id=run_id,
                    routing_status="waiting" if needs_input else "blocked",
                    loop_decision="request_user_input" if needs_input else None,
                    llm_selection_source=(
                        "not_run_step3_needs_user_input"
                        if needs_input
                        else "not_run_step3_blocked"
                    ),
                    warning=(
                        "input_readiness_needs_user_input"
                        if needs_input
                        else "input_readiness_blocked"
                    ),
                    available_agent_ids=snapshot.available_agent_ids,
                    unavailable_agent_ids=snapshot.unavailable_agent_ids,
                )
                artifact_id = self._persist_new_plan(run_id, plan)
                return OrchestratorRoutingServiceResult(
                    plan=plan,
                    plan_artifact_id=artifact_id,
                    prepared_tasks=(),
                    reused_existing_plan=False,
                    llm_called=False,
                    discovery_performed=True,
                )

            projected = project_orchestrator_context(
                structured_query=structured_query,
                readiness=readiness_body,
                available_artifacts=artifact_summary,
                current_routing_context={
                    "available_agent_ids": snapshot.available_agent_ids,
                    "unavailable_agent_ids": snapshot.unavailable_agent_ids,
                    "completed_routes": [],
                    "compact_statuses": [],
                    "warning_codes": [],
                },
            )
            llm_schema = {
                "task": "orchestrator_worker_routing",
                "compact_card_catalog": compact_catalog,
                **projected,
            }
            try:
                raw_proposal = self._llm.generate_json(
                    ORCHESTRATOR_ROUTING_USER_TASK,
                    schema=llm_schema,
                    system=ORCHESTRATOR_ROUTING_SYSTEM_PROMPT,
                )
                validated_shape = validate_task_shape(
                    raw_proposal,
                    "orchestrator_worker_routing",
                    error_factory=lambda _message: OrchestratorRoutingServiceError(
                        "llm_response_schema_invalid"
                    ),
                )
                proposal = OrchestratorRoutingProposal.model_validate(validated_shape)
            except (OrchestratorRoutingServiceError, ValidationError):
                return self._persist_llm_failure(
                    run_id,
                    snapshot.available_agent_ids,
                    snapshot.unavailable_agent_ids,
                    "llm_response_schema_invalid",
                )
            except Exception as exc:  # noqa: BLE001 - never persist raw detail
                return self._persist_llm_failure(
                    run_id,
                    snapshot.available_agent_ids,
                    snapshot.unavailable_agent_ids,
                    self._llm_failure_code(exc),
                )

            validation = validate_orchestrator_routing(
                run_id=run_id,
                proposal=proposal,
                discovery=self._discovery,
                storage=self._storage,
                registry=self._registry,
            )
            routing_plan_id = new_routing_plan_id()
            prepared = self._prepare_tasks(
                run_id=run_id,
                routing_plan_id=routing_plan_id,
                validation=validation,
                previous_decisions=(),
                completed_routing_decision_ids=frozenset(),
            )
            plan = self._plan_from_validation(
                run_id=run_id,
                routing_plan_id=routing_plan_id,
                planned_at=now_iso(),
                proposal=proposal,
                validation=validation,
                prepared_task_count=len(prepared),
                available_agent_ids=snapshot.available_agent_ids,
                unavailable_agent_ids=snapshot.unavailable_agent_ids,
            )
            artifact_id = self._persist_new_plan(run_id, plan)
            return OrchestratorRoutingServiceResult(
                plan=plan,
                plan_artifact_id=artifact_id,
                prepared_tasks=tuple(prepared),
                reused_existing_plan=False,
                llm_called=True,
                discovery_performed=True,
            )

    def revalidate_for_run(
        self,
        run_id: str,
        *,
        completed_results: Sequence[WorkerExecutionResult],
        expected_routing_plan_id: str | None = None,
    ) -> OrchestratorRoutingServiceResult:
        """Revalidate using cumulative strict terminal worker proofs.

        Productive proofs release dependencies; validated failure attestations
        only explain worker-owned registry advancement.
        """
        with self._get_run_lock(run_id):
            artifact_id = self._active_plan_artifact_id(run_id)
            if not artifact_id:
                raise OrchestratorRoutingServiceError("worker_routing_plan_missing")
            previous = self._load_plan(run_id, artifact_id)
            if (
                expected_routing_plan_id is not None
                and previous.routing_plan_id != expected_routing_plan_id
            ):
                raise OrchestratorRoutingServiceError(
                    "worker_routing_plan_identity_mismatch"
                )
            discovery_performed = self._ensure_discovery(run_id)
            return self._revalidate_loaded_plan(
                run_id=run_id,
                artifact_id=artifact_id,
                previous=previous,
                completed_results=completed_results,
                discovery_performed=discovery_performed,
            )

    def _revalidate_loaded_plan(
        self,
        *,
        run_id: str,
        artifact_id: str,
        previous: WorkerRoutingPlan,
        completed_results: Sequence[WorkerExecutionResult],
        discovery_performed: bool,
    ) -> OrchestratorRoutingServiceResult:
        if previous.loop_decision != "dispatch_next_workers":
            if completed_results:
                raise OrchestratorRoutingServiceError("completion_unknown_decision")
            return OrchestratorRoutingServiceResult(
                plan=previous,
                plan_artifact_id=artifact_id,
                prepared_tasks=(),
                reused_existing_plan=True,
                llm_called=False,
                discovery_performed=discovery_performed,
            )

        completed_ids, terminal_ids = self._validate_completed_results(
            run_id=run_id,
            plan=previous,
            completed_results=completed_results,
        )
        proposal = OrchestratorRoutingProposal(
            loop_decision=previous.loop_decision,
            decisions=previous.proposed_decisions,
            decision_summary="Revalidate the persisted compact routing proposal.",
        )
        stable_ids = self._aligned_decision_ids(previous, proposal.decisions)
        validation = validate_orchestrator_routing(
            run_id=run_id,
            proposal=proposal,
            discovery=self._discovery,
            storage=self._storage,
            registry=self._registry,
            routing_decision_ids=stable_ids,
            completed_routing_decision_ids=completed_ids,
        )
        prepared = self._prepare_tasks(
            run_id=run_id,
            routing_plan_id=previous.routing_plan_id,
            validation=validation,
            previous_decisions=previous.validated_decisions,
            completed_routing_decision_ids=completed_ids,
            terminal_routing_decision_ids=terminal_ids,
        )
        retained_rejected = [
            item
            for item in previous.rejected_decisions
            if item.reason == "unsafe_llm_output"
        ]
        plan = self._plan_from_validation(
            run_id=run_id,
            routing_plan_id=previous.routing_plan_id,
            planned_at=previous.planned_at,
            proposal=proposal,
            validation=validation,
            prepared_task_count=len(prepared),
            available_agent_ids=previous.available_agent_ids,
            unavailable_agent_ids=previous.unavailable_agent_ids,
            retained_rejected=retained_rejected,
            completed_routing_decision_ids=completed_ids,
            terminal_routing_decision_ids=terminal_ids,
        )
        self._persist_existing_plan(run_id, artifact_id, plan)
        return OrchestratorRoutingServiceResult(
            plan=plan,
            plan_artifact_id=artifact_id,
            prepared_tasks=tuple(prepared),
            reused_existing_plan=True,
            llm_called=False,
            discovery_performed=discovery_performed,
        )

    def _ensure_discovery(self, run_id: str) -> bool:
        if run_id in self._discovered_runs:
            return False
        self._discovery.discover_for_run(run_id)
        self._discovered_runs.add(run_id)
        return True

    def _get_run_lock(self, run_id: str) -> threading.Lock:
        with self._meta_lock:
            return self._run_locks.setdefault(run_id, threading.Lock())

    def _active_plan_artifact_id(self, run_id: str) -> str | None:
        try:
            return self._registry.get(run_id).active_artifacts.worker_routing_plan_id
        except Exception as exc:  # noqa: BLE001
            raise OrchestratorRoutingServiceError("run_registry_unavailable") from exc

    def _load_input_artifact(
        self, *, run_id: str, artifact_name: str, storage_path: str
    ) -> dict[str, Any]:
        try:
            active = self._registry.get(run_id).active_artifacts
        except Exception as exc:  # noqa: BLE001
            raise OrchestratorRoutingServiceError("run_registry_unavailable") from exc
        registry_field = f"{artifact_name}_id"
        artifact_id = getattr(active, registry_field, None)
        if not artifact_id:
            raise OrchestratorRoutingServiceError(f"{artifact_name}_missing")
        key = self._storage.run_key(run_id, storage_path)
        if not self._storage.exists(key):
            raise OrchestratorRoutingServiceError(f"{artifact_name}_storage_missing")
        try:
            body = self._storage.read_json(key)
        except Exception as exc:  # noqa: BLE001
            raise OrchestratorRoutingServiceError(
                f"{artifact_name}_identity_mismatch"
            ) from exc
        if not isinstance(body, dict):
            raise OrchestratorRoutingServiceError(f"{artifact_name}_identity_mismatch")
        if body.get("artifact_id") != artifact_id or body.get("run_id") != run_id:
            raise OrchestratorRoutingServiceError(f"{artifact_name}_identity_mismatch")
        return body

    def _available_artifact_summary(self, run_id: str) -> list[dict[str, Any]]:
        active = self._registry.get(run_id).active_artifacts
        specs = self._canonical_artifact_specs(run_id)
        summary = []
        for artifact_name in sorted(specs):
            spec = specs[artifact_name]
            inspection = inspect_declared_artifact(
                run_id=run_id,
                artifact=spec.ref,
                requirement=self._artifact_requirement(spec),
                active=active,
                storage=self._storage,
            )
            summary.append(
                {
                    "artifact_name": artifact_name,
                    "available": inspection.state == "valid",
                    "present_field_names": [
                        field_name
                        for field_name in inspection.present_field_names
                        if _SAFE_FIELD_NAME.fullmatch(field_name)
                    ],
                }
            )
        return summary

    def _canonical_artifact_specs(
        self, run_id: str
    ) -> dict[str, CanonicalArtifactSpec]:
        try:
            return canonical_artifact_specs(
                run_id=run_id, discovery=self._discovery
            )
        except CompletionArtifactValidationError as exc:
            raise OrchestratorRoutingServiceError(str(exc)) from None

    @staticmethod
    def _artifact_requirement(
        spec: CanonicalArtifactSpec,
    ) -> Any:
        return artifact_requirement(spec)

    def _validate_completed_results(
        self,
        *,
        run_id: str,
        plan: WorkerRoutingPlan,
        completed_results: Sequence[WorkerExecutionResult],
    ) -> tuple[frozenset[str], frozenset[str]]:
        decisions = {
            decision.routing_decision_id: decision
            for decision in plan.validated_decisions
        }
        active = self._registry.get(run_id).active_artifacts
        completed_ids: set[str] = set()
        seen_decision_ids: set[str] = set()
        attested_outputs_by_decision: dict[str, set[str]] = {}
        for supplied in completed_results:
            if not isinstance(supplied, WorkerExecutionResult):
                raise OrchestratorRoutingServiceError("completion_result_type_invalid")
            try:
                result = WorkerExecutionResult.model_validate(
                    supplied.model_dump(mode="python", warnings=False),
                    strict=True,
                )
            except (AttributeError, ValidationError):
                raise OrchestratorRoutingServiceError(
                    "completion_proof_schema_invalid"
                ) from None
            decision_id = result.routing_decision_id
            if not decision_id or decision_id not in decisions:
                raise OrchestratorRoutingServiceError("completion_unknown_decision")
            if decision_id in seen_decision_ids:
                raise OrchestratorRoutingServiceError("completion_duplicate_decision")
            seen_decision_ids.add(decision_id)
            decision = decisions[decision_id]
            if not decision.task_id:
                raise OrchestratorRoutingServiceError("completion_decision_has_no_task")
            if (
                result.run_id != run_id
                or result.routing_plan_id != plan.routing_plan_id
                or result.task_id != decision.task_id
                or result.agent_id != decision.agent_id
                or result.capability_id != decision.capability_id
            ):
                raise OrchestratorRoutingServiceError("completion_identity_mismatch")
            productive = result.result_status in {"success", "partial"}
            expected = set(decision.expected_output_artifact_names)
            actual = set(result.output_artifact_refs)
            if productive and expected - actual:
                raise OrchestratorRoutingServiceError(
                    "completion_output_artifacts_missing"
                )
            if actual - expected:
                raise OrchestratorRoutingServiceError(
                    "completion_output_artifacts_unexpected"
                )
            try:
                validate_worker_output_artifacts(
                    run_id=run_id,
                    agent_id=decision.agent_id,
                    capability_id=decision.capability_id,
                    expected_output_artifact_names=expected,
                    output_artifact_refs=result.output_artifact_refs,
                    productive=productive,
                    discovery=self._discovery,
                    registry=self._registry,
                    storage=self._storage,
                    active=active,
                )
            except CompletionArtifactValidationError as exc:
                raise OrchestratorRoutingServiceError(str(exc)) from None
            attested_outputs_by_decision[decision_id] = actual
            if productive:
                completed_ids.add(decision_id)
        self._require_completion_proofs_for_advanced_outputs(
            plan=plan,
            active=active,
            attested_outputs_by_decision=attested_outputs_by_decision,
        )
        return frozenset(completed_ids), frozenset(seen_decision_ids)

    def _require_completion_proofs_for_advanced_outputs(
        self,
        *,
        plan: WorkerRoutingPlan,
        active: Any,
        attested_outputs_by_decision: dict[str, set[str]],
    ) -> None:
        """Require a validated terminal attestation for every advanced output."""
        baselines = active.worker_routing_plan_output_baselines
        for decision in plan.validated_decisions:
            for artifact_name in decision.expected_output_artifact_names:
                baseline = baselines.get(artifact_name)
                if baseline is None:
                    raise OrchestratorRoutingServiceError(
                        "completion_output_baseline_missing"
                    )
                current_artifact_id = getattr(
                    active, f"{artifact_name}_id", None
                )
                if artifact_id_fingerprint(current_artifact_id) != baseline:
                    attested = attested_outputs_by_decision.get(
                        decision.routing_decision_id, set()
                    )
                    if artifact_name not in attested:
                        raise OrchestratorRoutingServiceError(
                            "completion_proof_required"
                        )

    def _prepare_tasks(
        self,
        *,
        run_id: str,
        routing_plan_id: str,
        validation: RoutingValidationResult,
        previous_decisions: Sequence[ValidatedRoutingDecision],
        completed_routing_decision_ids: frozenset[str],
        terminal_routing_decision_ids: frozenset[str] = frozenset(),
    ) -> list[PreparedA2ATask]:
        previous_by_id = {
            item.routing_decision_id: item for item in previous_decisions
        }
        prepared: list[PreparedA2ATask] = []
        for item in validation.decisions:
            previous = previous_by_id.get(item.decision.routing_decision_id)
            if previous is not None and previous.task_id:
                item.decision = item.decision.model_copy(
                    update={"task_id": previous.task_id}
                )
            if (
                item.decision.validation_status != "ready"
                or not item.task_build_allowed
                or item.decision.routing_decision_id
                in terminal_routing_decision_ids
            ):
                continue
            built = build_orchestrator_worker_task(
                run_id=run_id,
                routing_plan_id=routing_plan_id,
                validated=item,
                task_id=item.decision.task_id,
            )
            item.decision = built.decision
            prepared.append(built)
        return prepared

    def _plan_from_validation(
        self,
        *,
        run_id: str,
        routing_plan_id: str,
        planned_at: str,
        proposal: OrchestratorRoutingProposal,
        validation: RoutingValidationResult,
        prepared_task_count: int,
        available_agent_ids: Sequence[str],
        unavailable_agent_ids: Sequence[str],
        retained_rejected: Sequence[RejectedRoutingDecision] = (),
        completed_routing_decision_ids: frozenset[str] = frozenset(),
        terminal_routing_decision_ids: frozenset[str] = frozenset(),
    ) -> WorkerRoutingPlan:
        rejected = [*validation.rejected_decisions]
        known_rejected_ids = {item.routing_decision_id for item in rejected}
        rejected.extend(
            item
            for item in retained_rejected
            if item.routing_decision_id not in known_rejected_ids
        )
        safe_proposed = [
            item for item in proposal.decisions if not self._decision_is_unsafe(item)
        ]
        status = self._routing_status(
            proposal=proposal,
            validation=validation,
            prepared_task_count=prepared_task_count,
            completed_routing_decision_ids=completed_routing_decision_ids,
            terminal_routing_decision_ids=terminal_routing_decision_ids,
        )
        warnings = self._compact_warnings(
            [*validation.warnings, *validation.plan_error_codes]
        )
        return WorkerRoutingPlan(
            run_id=run_id,
            routing_plan_id=routing_plan_id,
            planned_at=planned_at,
            loop_decision=proposal.loop_decision,
            routing_status=status,
            llm_selection_source="llm_primary_validated",
            proposed_decisions=safe_proposed,
            validated_decisions=[item.decision for item in validation.decisions],
            rejected_decisions=rejected,
            dependency_edges=validation.dependency_edges,
            ready_task_count=prepared_task_count,
            waiting_decision_count=sum(
                item.decision.validation_status == "waiting_for_dependencies"
                for item in validation.decisions
            ),
            rejected_decision_count=len(rejected),
            available_agent_ids=list(available_agent_ids),
            unavailable_agent_ids=list(unavailable_agent_ids),
            warnings=warnings,
        )

    @staticmethod
    def _routing_status(
        *,
        proposal: OrchestratorRoutingProposal,
        validation: RoutingValidationResult,
        prepared_task_count: int,
        completed_routing_decision_ids: frozenset[str],
        terminal_routing_decision_ids: frozenset[str] = frozenset(),
    ) -> str:
        active = [
            item
            for item in validation.decisions
            if item.decision.routing_decision_id
            not in terminal_routing_decision_ids
        ]
        if validation.plan_error_codes:
            return "rejected"
        if proposal.loop_decision == "route_to_final_response" and not active:
            return "completed"
        if (
            validation.decisions
            and not active
            and not validation.rejected_decisions
            and all(
                item.decision.routing_decision_id
                in completed_routing_decision_ids
                for item in validation.decisions
            )
        ):
            return "completed"
        if prepared_task_count or any(
            item.decision.validation_status == "ready" for item in active
        ):
            return "ready"
        if any(
            item.decision.validation_status == "waiting_for_dependencies"
            for item in active
        ) or proposal.loop_decision == "wait_for_dependencies":
            return "waiting"
        if not active and validation.rejected_decisions:
            return "rejected"
        if active and all(
            item.decision.validation_status == "blocked_missing_dependency"
            for item in active
        ):
            return "blocked"
        return "blocked"

    def _empty_plan(
        self,
        *,
        run_id: str,
        routing_status: str,
        loop_decision: LoopDecision | None = None,
        llm_selection_source: str,
        warning: str,
        available_agent_ids: Sequence[str],
        unavailable_agent_ids: Sequence[str],
    ) -> WorkerRoutingPlan:
        return WorkerRoutingPlan(
            run_id=run_id,
            routing_plan_id=new_routing_plan_id(),
            planned_at=now_iso(),
            loop_decision=loop_decision,
            routing_status=routing_status,
            llm_selection_source=llm_selection_source,
            warnings=self._compact_warnings([warning]),
            available_agent_ids=list(available_agent_ids),
            unavailable_agent_ids=list(unavailable_agent_ids),
        )

    def _persist_llm_failure(
        self,
        run_id: str,
        available_agent_ids: Sequence[str],
        unavailable_agent_ids: Sequence[str],
        code: str,
    ) -> OrchestratorRoutingServiceResult:
        plan = self._empty_plan(
            run_id=run_id,
            routing_status="llm_failed",
            llm_selection_source="llm_failed",
            warning=code,
            available_agent_ids=available_agent_ids,
            unavailable_agent_ids=unavailable_agent_ids,
        )
        artifact_id = self._persist_new_plan(run_id, plan)
        return OrchestratorRoutingServiceResult(
            plan=plan,
            plan_artifact_id=artifact_id,
            prepared_tasks=(),
            reused_existing_plan=False,
            llm_called=True,
            discovery_performed=True,
        )

    def _persist_new_plan(self, run_id: str, plan: WorkerRoutingPlan) -> str:
        artifact_id = new_artifact_id("worker_routing_plan")
        active = self._registry.get(run_id).active_artifacts
        output_names = {
            artifact_name
            for decision in plan.validated_decisions
            for artifact_name in decision.expected_output_artifact_names
        }
        output_baselines = {
            artifact_name: artifact_id_fingerprint(
                getattr(active, f"{artifact_name}_id", None)
            )
            for artifact_name in sorted(output_names)
        }
        self._write_plan(run_id, artifact_id, plan)
        self._registry.update_active(
            run_id,
            worker_routing_plan_id=artifact_id,
            worker_routing_plan_control_id=plan.routing_plan_id,
            worker_routing_plan_output_baselines=output_baselines,
        )
        self._known_plan_id_by_run[run_id] = plan.routing_plan_id
        self._known_artifact_id_by_run[run_id] = artifact_id
        self._load_plan(run_id, artifact_id)
        return artifact_id

    def _persist_existing_plan(
        self, run_id: str, artifact_id: str, plan: WorkerRoutingPlan
    ) -> None:
        self._write_plan(run_id, artifact_id, plan)
        self._load_plan(run_id, artifact_id)

    def _write_plan(
        self, run_id: str, artifact_id: str, plan: WorkerRoutingPlan
    ) -> None:
        self._storage.write_json(
            self._storage.run_key(run_id, _PLAN_STORAGE_KEY),
            {"artifact_id": artifact_id, **plan.model_dump()},
        )

    def _load_plan(self, run_id: str, artifact_id: str) -> WorkerRoutingPlan:
        try:
            active = self._registry.get(run_id).active_artifacts
        except Exception as exc:  # noqa: BLE001
            raise OrchestratorRoutingServiceError("run_registry_unavailable") from exc
        if active.worker_routing_plan_id != artifact_id:
            raise OrchestratorRoutingServiceError("worker_routing_plan_identity_mismatch")
        control_id = active.worker_routing_plan_control_id
        if not control_id:
            raise OrchestratorRoutingServiceError("worker_routing_plan_identity_mismatch")
        known_artifact_id = self._known_artifact_id_by_run.get(run_id)
        if known_artifact_id is not None and known_artifact_id != artifact_id:
            raise OrchestratorRoutingServiceError("worker_routing_plan_identity_mismatch")
        key = self._storage.run_key(run_id, _PLAN_STORAGE_KEY)
        if not self._storage.exists(key):
            raise OrchestratorRoutingServiceError("worker_routing_plan_storage_missing")
        try:
            body = self._storage.read_json(key)
        except Exception as exc:  # noqa: BLE001
            raise OrchestratorRoutingServiceError(
                "worker_routing_plan_identity_mismatch"
            ) from exc
        if not isinstance(body, dict):
            raise OrchestratorRoutingServiceError("worker_routing_plan_identity_mismatch")
        if body.get("artifact_id") != artifact_id or body.get("run_id") != run_id:
            raise OrchestratorRoutingServiceError("worker_routing_plan_identity_mismatch")
        try:
            plan = WorkerRoutingPlan.model_validate(
                {key: value for key, value in body.items() if key != "artifact_id"}
            )
        except Exception as exc:  # noqa: BLE001
            raise OrchestratorRoutingServiceError(
                "worker_routing_plan_schema_invalid"
            ) from exc
        if plan.routing_plan_id != control_id:
            raise OrchestratorRoutingServiceError("worker_routing_plan_identity_mismatch")
        known_plan_id = self._known_plan_id_by_run.get(run_id)
        if known_plan_id is not None and plan.routing_plan_id != known_plan_id:
            raise OrchestratorRoutingServiceError("worker_routing_plan_identity_mismatch")
        self._known_artifact_id_by_run[run_id] = artifact_id
        self._known_plan_id_by_run[run_id] = plan.routing_plan_id
        return plan

    @staticmethod
    def _decision_is_unsafe(item: OrchestratorRouteDecision) -> bool:
        return any(
            contains_unsafe_routing_text(value)
            for value in (
                item.agent_id,
                item.capability_id,
                item.objective,
                item.selection_reason,
            )
        )

    @staticmethod
    def _compact_warnings(values: Iterable[str]) -> list[str]:
        output: list[str] = []
        for value in values:
            compact = value if _COMPACT_CODE.fullmatch(str(value)) else "routing_warning_redacted"
            if compact not in output:
                output.append(compact)
        return output

    def _llm_failure_code(self, exc: Exception) -> str:
        provider = re.sub(
            r"[^a-z0-9_.-]", "_", str(getattr(self._llm, "name", "provider")).lower()
        )
        error_type = re.sub(r"[^a-z0-9_.-]", "_", type(exc).__name__.lower())
        return f"llm_error:{provider}:{error_type}"

    @staticmethod
    def _aligned_decision_ids(
        plan: WorkerRoutingPlan,
        decisions: Sequence[OrchestratorRouteDecision],
    ) -> list[str]:
        pools: dict[tuple[str, str], deque[str]] = defaultdict(deque)
        persisted_decisions = [
            *plan.validated_decisions,
            *(
                item
                for item in plan.rejected_decisions
                if item.reason != "unsafe_llm_output"
            ),
        ]
        for item in persisted_decisions:
            if item.agent_id is not None and item.capability_id is not None:
                pools[(item.agent_id, item.capability_id)].append(
                    item.routing_decision_id
                )
        aligned: list[str] = []
        for item in decisions:
            pool = pools[(item.agent_id, item.capability_id)]
            if not pool:
                raise OrchestratorRoutingServiceError(
                    "worker_routing_plan_decision_identity_mismatch"
                )
            aligned.append(pool.popleft())
        if any(pool for pool in pools.values()):
            raise OrchestratorRoutingServiceError(
                "worker_routing_plan_decision_identity_mismatch"
            )
        return aligned


__all__ = [
    "OrchestratorRoutingService",
    "OrchestratorRoutingServiceError",
    "OrchestratorRoutingServiceResult",
]
