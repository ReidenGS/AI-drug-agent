"""Deterministic Turn F1-C1 routing validation and artifact dependency DAG.

This module validates an LLM proposal against the frozen HTTP-discovered
AgentCards. It does not persist a routing plan, construct or send an A2A task,
or call a worker, LLM, or MCP tool.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence

from app.schemas.registry import ActiveArtifacts
from app.schemas.worker_routing_plan import (
    DependencyEdge,
    OrchestratorRouteDecision,
    OrchestratorRoutingProposal,
    RejectedRoutingDecision,
    ValidatedRoutingDecision,
)
from app.services.artifact_registry_service import ArtifactRegistryService
from app.services.storage_service import Storage
from app.utils.ids import new_routing_decision_id

from .agent_cards import AgentCapabilityContract, ContractArtifactRef
from .contracts import InputArtifactRef
from .orchestrator_context_projection import contains_unsafe_routing_text
from .orchestrator_discovery import (
    DispatchTarget,
    DispatchTargetValidationError,
    WorkerDiscoveryRunCache,
    WorkerUnavailableError,
)


class DiscoveryAuthority(Protocol):
    def get_full_card_cache(self, run_id: str) -> WorkerDiscoveryRunCache: ...

    def resolve_dispatch_target(
        self,
        run_id: str,
        *,
        agent_id: str,
        capability_id: str,
        dispatch_mode: str = "python_a2a",
    ) -> DispatchTarget: ...


@dataclass
class RuntimeValidatedDecision:
    """Audit decision plus in-memory-only execution authority and refs."""

    run_id: str
    proposed: OrchestratorRouteDecision
    decision: ValidatedRoutingDecision
    capability: AgentCapabilityContract
    dispatch_target: DispatchTarget
    input_artifact_refs: dict[str, InputArtifactRef] = field(default_factory=dict)
    task_build_allowed: bool = True


@dataclass
class RoutingValidationResult:
    run_id: str
    loop_decision: str
    decisions: list[RuntimeValidatedDecision] = field(default_factory=list)
    rejected_decisions: list[RejectedRoutingDecision] = field(default_factory=list)
    dependency_edges: list[DependencyEdge] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    plan_error_codes: list[str] = field(default_factory=list)
    invalidated_output_artifact_names: set[str] = field(default_factory=set)

    @property
    def ready_decisions(self) -> list[RuntimeValidatedDecision]:
        return [
            item
            for item in self.decisions
            if item.decision.validation_status == "ready" and item.task_build_allowed
        ]


@dataclass(frozen=True)
class ArtifactInspection:
    """Single AgentCard-declared artifact readiness/identity inspection."""

    state: str
    ref: InputArtifactRef | None = None
    code: str | None = None
    present_field_names: tuple[str, ...] = ()


def validate_orchestrator_routing(
    *,
    run_id: str,
    proposal: OrchestratorRoutingProposal,
    discovery: DiscoveryAuthority,
    storage: Storage,
    registry: ArtifactRegistryService,
    routing_decision_ids: Sequence[str] | None = None,
    completed_routing_decision_ids: set[str] | frozenset[str] = frozenset(),
) -> RoutingValidationResult:
    """Validate targets, privacy, artifacts and dependencies without dispatch."""
    if routing_decision_ids is not None and len(routing_decision_ids) != len(
        proposal.decisions
    ):
        raise ValueError("routing_decision_id_count_mismatch")
    if routing_decision_ids is not None and (
        any(not isinstance(item, str) or not item for item in routing_decision_ids)
        or len(set(routing_decision_ids)) != len(routing_decision_ids)
    ):
        raise ValueError("routing_decision_ids_invalid")
    stable_decision_ids = list(routing_decision_ids or ()) or [
        new_routing_decision_id() for _ in proposal.decisions
    ]
    result = RoutingValidationResult(run_id=run_id, loop_decision=proposal.loop_decision)
    loop_valid = _validate_loop_consistency(proposal, result)
    if not loop_valid:
        if proposal.loop_decision != "dispatch_next_workers":
            result.rejected_decisions = [
                _rejected(decision_id, proposed, "invalid_loop_decision")
                for decision_id, proposed in zip(
                    stable_decision_ids, proposal.decisions, strict=True
                )
            ]
        return result
    if proposal.loop_decision != "dispatch_next_workers":
        return result

    cache = discovery.get_full_card_cache(run_id)

    pair_counts: dict[tuple[str, str], int] = {}
    for decision in proposal.decisions:
        pair = (decision.agent_id, decision.capability_id)
        pair_counts[pair] = pair_counts.get(pair, 0) + 1

    candidates: list[RuntimeValidatedDecision] = []
    for routing_decision_id, proposed in zip(
        stable_decision_ids, proposal.decisions, strict=True
    ):
        if _decision_has_unsafe_text(proposed):
            unsafe_identity = contains_unsafe_routing_text(proposed.agent_id) or (
                contains_unsafe_routing_text(proposed.capability_id)
            )
            result.rejected_decisions.append(
                RejectedRoutingDecision(
                    routing_decision_id=routing_decision_id,
                    agent_id=None if unsafe_identity else proposed.agent_id,
                    capability_id=None if unsafe_identity else proposed.capability_id,
                    reason="unsafe_llm_output",
                )
            )
            continue
        if pair_counts[(proposed.agent_id, proposed.capability_id)] > 1:
            result.rejected_decisions.append(
                RejectedRoutingDecision(
                    routing_decision_id=routing_decision_id,
                    agent_id=proposed.agent_id,
                    capability_id=proposed.capability_id,
                    reason="duplicate_route",
                )
            )
            continue

        worker = cache.workers.get(proposed.agent_id)
        if worker is None:
            result.rejected_decisions.append(
                _rejected(routing_decision_id, proposed, "unknown_worker")
            )
            continue
        if not worker.is_available or worker.contract is None:
            result.rejected_decisions.append(
                _rejected(routing_decision_id, proposed, "rejected_unavailable")
            )
            continue
        capability = next(
            (
                item
                for item in worker.contract.capabilities
                if item.capability_id == proposed.capability_id
            ),
            None,
        )
        if capability is None:
            result.rejected_decisions.append(
                _rejected(routing_decision_id, proposed, "unknown_capability")
            )
            continue
        try:
            target = discovery.resolve_dispatch_target(
                run_id,
                agent_id=proposed.agent_id,
                capability_id=proposed.capability_id,
                dispatch_mode="python_a2a",
            )
        except WorkerUnavailableError:
            result.rejected_decisions.append(
                _rejected(routing_decision_id, proposed, "rejected_unavailable")
            )
            continue
        except DispatchTargetValidationError:
            result.rejected_decisions.append(
                _rejected(routing_decision_id, proposed, "dispatch_target_invalid")
            )
            continue

        candidates.append(
            RuntimeValidatedDecision(
                run_id=run_id,
                proposed=proposed,
                decision=ValidatedRoutingDecision(
                    routing_decision_id=routing_decision_id,
                    agent_id=proposed.agent_id,
                    capability_id=proposed.capability_id,
                    objective=proposed.objective,
                    selection_reason=proposed.selection_reason,
                    priority=proposed.priority,
                    validation_status="ready",
                    expected_output_artifact_names=[
                        item.artifact_name for item in capability.output_artifacts
                    ],
                ),
                capability=capability,
                dispatch_target=target,
                task_build_allowed=True,
            )
        )

    candidates = _reject_output_artifact_conflicts(candidates, result)
    result.decisions = candidates

    _validate_artifacts_and_build_dag(
        run_id=run_id,
        decisions=result.decisions,
        storage=storage,
        registry=registry,
        result=result,
        completed_routing_decision_ids=completed_routing_decision_ids,
    )
    return result


def _validate_loop_consistency(
    proposal: OrchestratorRoutingProposal, result: RoutingValidationResult
) -> bool:
    invalid = (
        proposal.loop_decision == "dispatch_next_workers" and not proposal.decisions
    ) or (
        proposal.loop_decision != "dispatch_next_workers" and bool(proposal.decisions)
    )
    if invalid:
        result.plan_error_codes.append("invalid_loop_decision")
        result.warnings.append("invalid_loop_decision")
        return False
    return True


def _reject_output_artifact_conflicts(
    candidates: list[RuntimeValidatedDecision],
    result: RoutingValidationResult,
) -> list[RuntimeValidatedDecision]:
    ownership: dict[str, list[RuntimeValidatedDecision]] = {}
    path_ownership: dict[str, list[RuntimeValidatedDecision]] = {}
    for item in candidates:
        for artifact in item.capability.output_artifacts:
            ownership.setdefault(artifact.artifact_name, []).append(item)
            path_ownership.setdefault(artifact.storage_path, []).append(item)
    conflicts = {
        artifact_name: producers
        for artifact_name, producers in ownership.items()
        if len(producers) > 1
    }
    path_conflicts = {}
    for storage_path, producers in path_ownership.items():
        artifact_names = {
            artifact.artifact_name
            for producer in producers
            for artifact in producer.capability.output_artifacts
            if artifact.storage_path == storage_path
        }
        if len(artifact_names) > 1:
            path_conflicts[storage_path] = producers
    if not conflicts and not path_conflicts:
        return candidates
    rejected_ids: set[str] = set()
    for artifact_name in sorted(conflicts):
        result.warnings.append(f"{artifact_name}:ambiguous_output_producer")
        rejected_ids.update(
            producer.decision.routing_decision_id
            for producer in conflicts[artifact_name]
        )
    if path_conflicts:
        result.warnings.append("output_storage_path:ambiguous_output_producer")
        for storage_path in sorted(path_conflicts):
            rejected_ids.update(
                producer.decision.routing_decision_id
                for producer in path_conflicts[storage_path]
            )
    for item in candidates:
        if item.decision.routing_decision_id in rejected_ids:
            item.task_build_allowed = False
            result.invalidated_output_artifact_names.update(
                artifact.artifact_name for artifact in item.capability.output_artifacts
            )
            result.rejected_decisions.append(
                RejectedRoutingDecision(
                    routing_decision_id=item.decision.routing_decision_id,
                    agent_id=item.decision.agent_id,
                    capability_id=item.decision.capability_id,
                    reason="output_artifact_conflict",
                )
            )
    return [
        item
        for item in candidates
        if item.decision.routing_decision_id not in rejected_ids
    ]


def _decision_has_unsafe_text(decision: OrchestratorRouteDecision) -> bool:
    return any(
        contains_unsafe_routing_text(value)
        for value in (
            decision.agent_id,
            decision.capability_id,
            decision.objective,
            decision.selection_reason,
        )
    )


def _rejected(
    routing_decision_id: str,
    proposed: OrchestratorRouteDecision,
    reason: str,
) -> RejectedRoutingDecision:
    unsafe_identity = contains_unsafe_routing_text(proposed.agent_id) or (
        contains_unsafe_routing_text(proposed.capability_id)
    )
    return RejectedRoutingDecision(
        routing_decision_id=routing_decision_id,
        agent_id=None if unsafe_identity else proposed.agent_id,
        capability_id=None if unsafe_identity else proposed.capability_id,
        reason=reason,
    )


def _validate_artifacts_and_build_dag(
    *,
    run_id: str,
    decisions: list[RuntimeValidatedDecision],
    storage: Storage,
    registry: ArtifactRegistryService,
    result: RoutingValidationResult,
    completed_routing_decision_ids: set[str] | frozenset[str],
) -> None:
    if not decisions:
        return
    producer_index: dict[str, list[RuntimeValidatedDecision]] = {}
    for item in decisions:
        if item.decision.routing_decision_id in completed_routing_decision_ids:
            continue
        for output in item.capability.output_artifacts:
            producer_index.setdefault(output.artifact_name, []).append(item)

    try:
        active = registry.get(run_id).active_artifacts
    except Exception:  # noqa: BLE001 - fixed compact code, never raw exception
        result.warnings.append("run_registry_unavailable")
        for item in decisions:
            item.decision.validation_status = "blocked_missing_dependency"
            item.decision.reason = "run_registry_unavailable"
            item.task_build_allowed = False
        return
    dependency_sources: dict[str, list[str]] = {}
    for item in decisions:
        if item.decision.routing_decision_id in completed_routing_decision_ids:
            item.input_artifact_refs = {}
            item.task_build_allowed = False
            continue
        refs: dict[str, InputArtifactRef] = {}
        missing_required: list[str] = []
        corrupt_required: list[str] = []
        not_ready_required: list[str] = []
        conflicted_required: list[str] = []
        for artifact in item.capability.required_input_artifacts:
            if artifact.artifact_name in result.invalidated_output_artifact_names:
                conflicted_required.append(artifact.artifact_name)
                continue
            if producer_index.get(artifact.artifact_name):
                missing_required.append(artifact.artifact_name)
                continue
            requirement = item.capability.required_artifact_fields.get(
                artifact.artifact_name
            )
            check = inspect_declared_artifact(
                run_id=run_id,
                artifact=artifact,
                requirement=requirement,
                active=active,
                storage=storage,
            )
            if check.state == "valid" and check.ref is not None:
                refs[artifact.artifact_name] = check.ref
            elif check.state == "missing":
                missing_required.append(artifact.artifact_name)
            elif check.state == "not_ready":
                not_ready_required.append(artifact.artifact_name)
                result.warnings.append(
                    f"{artifact.artifact_name}:artifact_not_ready"
                )
            else:
                corrupt_required.append(artifact.artifact_name)
                result.warnings.append(
                    f"{artifact.artifact_name}:{check.code or 'artifact_invalid'}"
                )

        for artifact in item.capability.optional_input_artifacts:
            if artifact.artifact_name in result.invalidated_output_artifact_names:
                continue
            if producer_index.get(artifact.artifact_name):
                continue
            check = inspect_declared_artifact(
                run_id=run_id,
                artifact=artifact,
                requirement=item.capability.required_artifact_fields.get(
                    artifact.artifact_name
                ),
                active=active,
                storage=storage,
            )
            if check.state == "valid" and check.ref is not None:
                refs[artifact.artifact_name] = check.ref
            elif check.state == "corrupt":
                result.warnings.append(
                    f"{artifact.artifact_name}:optional_artifact_invalid"
                )
            elif check.state == "not_ready":
                result.warnings.append(
                    f"{artifact.artifact_name}:optional_artifact_not_ready"
                )

        item.input_artifact_refs = refs
        item.decision.dependency_artifact_names = sorted(
            [
                *missing_required,
                *corrupt_required,
                *not_ready_required,
                *conflicted_required,
            ]
        )
        if conflicted_required:
            item.decision.validation_status = "blocked_missing_dependency"
            item.decision.reason = "dependency_producer_conflict"
            item.task_build_allowed = False
            continue
        if not_ready_required:
            item.decision.validation_status = "blocked_missing_dependency"
            item.decision.reason = "required_artifact_not_ready"
            item.task_build_allowed = False
            continue
        if corrupt_required:
            item.decision.validation_status = "blocked_missing_dependency"
            item.decision.reason = "required_artifact_invalid"
            item.task_build_allowed = False
            continue

        blocked = False
        waiting = False
        producers_for_item: list[str] = []
        for artifact_name in missing_required:
            producers = producer_index.get(artifact_name, [])
            if not producers:
                blocked = True
                continue
            if len(producers) > 1:
                blocked = True
                result.warnings.append(
                    f"{artifact_name}:ambiguous_dependency_producer"
                )
                item.decision.reason = "ambiguous_dependency_producer"
                continue
            producer = producers[0]
            waiting = True
            producers_for_item.append(producer.decision.routing_decision_id)
            result.dependency_edges.append(
                DependencyEdge(
                    artifact_name=artifact_name,
                    producer_agent_id=producer.decision.agent_id,
                    producer_capability_id=producer.decision.capability_id,
                    consumer_agent_id=item.decision.agent_id,
                    consumer_capability_id=item.decision.capability_id,
                )
            )
            dependency_sources.setdefault(
                item.decision.routing_decision_id, []
            ).append(producer.decision.routing_decision_id)
        item.decision.dependency_producers = producers_for_item
        if blocked:
            item.decision.validation_status = "blocked_missing_dependency"
            item.decision.reason = item.decision.reason or "missing_required_artifact"
            item.task_build_allowed = False
        elif waiting:
            item.decision.validation_status = "waiting_for_dependencies"
            item.decision.reason = "waiting_for_dependencies"
            item.task_build_allowed = False

    cycle_ids = _cycle_decision_ids(dependency_sources)
    if cycle_ids:
        result.warnings.append("dependency_cycle")
        for item in decisions:
            if item.decision.routing_decision_id in cycle_ids:
                item.decision.validation_status = "blocked_missing_dependency"
                item.decision.reason = "dependency_cycle"
                item.task_build_allowed = False

    by_id = {item.decision.routing_decision_id: item for item in decisions}
    changed = True
    while changed:
        changed = False
        for consumer_id, producer_ids in dependency_sources.items():
            consumer = by_id[consumer_id]
            if consumer.decision.validation_status != "waiting_for_dependencies":
                continue
            if any(
                by_id[producer_id].decision.validation_status
                == "blocked_missing_dependency"
                for producer_id in producer_ids
            ):
                consumer.decision.validation_status = "blocked_missing_dependency"
                consumer.decision.reason = "dependency_producer_blocked"
                consumer.task_build_allowed = False
                changed = True


def inspect_declared_artifact(
    *,
    run_id: str,
    artifact: ContractArtifactRef,
    requirement: Any,
    active: ActiveArtifacts,
    storage: Storage,
) -> ArtifactInspection:
    """Inspect one persisted artifact using its validated AgentCard contract."""
    registry_field = f"{artifact.artifact_name}_id"
    if registry_field not in ActiveArtifacts.model_fields:
        return ArtifactInspection("corrupt", code="unknown_artifact_registry_field")
    artifact_id = getattr(active, registry_field)
    if not artifact_id:
        return ArtifactInspection("missing")
    storage_key = storage.run_key(run_id, artifact.storage_path)
    if not storage.exists(storage_key):
        return ArtifactInspection("corrupt", code="artifact_storage_missing")
    try:
        body = storage.read_json(storage_key)
    except Exception:  # noqa: BLE001 - compact code only, never raw exception
        return ArtifactInspection("corrupt", code="artifact_json_unreadable")
    if not isinstance(body, dict):
        return ArtifactInspection("corrupt", code="artifact_body_not_object")
    if body.get("artifact_id") != artifact_id:
        return ArtifactInspection("corrupt", code="artifact_id_mismatch")
    if body.get("run_id") != run_id:
        return ArtifactInspection("corrupt", code="artifact_run_id_mismatch")
    field_keys = list(requirement.required_field_keys) if requirement else []
    if any(field_name not in body for field_name in field_keys):
        return ArtifactInspection("corrupt", code="artifact_required_fields_missing")
    if artifact.readiness_status_field is not None:
        readiness = body.get(artifact.readiness_status_field)
        if not isinstance(readiness, str) or readiness not in artifact.ready_status_values:
            return ArtifactInspection("not_ready", code="artifact_not_ready")
    schema_version = body.get("schema_version")
    return ArtifactInspection(
        "valid",
        ref=InputArtifactRef(
            artifact_id=artifact_id,
            run_id=run_id,
            artifact_type=artifact.artifact_name,
            artifact_role=artifact.artifact_name,
            schema_version=schema_version if isinstance(schema_version, str) else None,
            entity_type=requirement.entity_type if requirement else None,
            selection_mode=(
                requirement.default_selection_mode if requirement else None
            ),
            field_keys=field_keys,
            can_read_from_db=True,
        ),
        present_field_names=tuple(
            field_name for field_name in field_keys if field_name in body
        ),
    )


def _cycle_decision_ids(graph: dict[str, list[str]]) -> set[str]:
    visited: set[str] = set()
    active: list[str] = []
    cycles: set[str] = set()

    def visit(node: str) -> None:
        if node in active:
            cycles.update(active[active.index(node) :])
            return
        if node in visited:
            return
        active.append(node)
        for dependency in graph.get(node, []):
            visit(dependency)
        active.pop()
        visited.add(node)

    for node in graph:
        visit(node)
    return cycles
