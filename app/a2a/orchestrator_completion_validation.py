"""Shared validation of worker-owned output artifacts and completion proofs."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from hashlib import sha256
from typing import Any, Mapping, Protocol

from app.schemas.registry import ActiveArtifacts
from app.services.artifact_registry_service import ArtifactRegistryService
from app.services.storage_service import Storage

from .agent_cards import (
    AgentCapabilityContract,
    ArtifactFieldRequirement,
    ContractArtifactRef,
)
from .contracts import WorkerArtifactRef


class CompletionArtifactValidationError(RuntimeError):
    """Fixed compact validation code; never includes artifact data or paths."""


class CompletionDiscoveryAuthority(Protocol):
    def get_full_card_cache(self, run_id: str) -> Any: ...


@dataclass(frozen=True)
class CanonicalArtifactSpec:
    ref: ContractArtifactRef
    required_field_keys: tuple[str, ...]


@dataclass(frozen=True)
class ValidatedOutputArtifact:
    artifact_name: str
    artifact_id: str
    ready: bool


def artifact_id_fingerprint(artifact_id: str | None) -> str:
    """Return the routing-plan baseline fingerprint without exposing the ID."""
    if artifact_id is None:
        return "absent"
    return sha256(artifact_id.encode("utf-8")).hexdigest()


def canonical_artifact_specs(
    *, run_id: str, discovery: CompletionDiscoveryAuthority
) -> dict[str, CanonicalArtifactSpec]:
    """Build one fail-closed artifact contract catalog from frozen AgentCards."""
    cache = discovery.get_full_card_cache(run_id)
    refs: dict[str, ContractArtifactRef] = {}
    required_fields: dict[str, set[str]] = defaultdict(set)
    for worker in cache.workers.values():
        if worker.contract is None:
            continue
        for capability in worker.contract.capabilities:
            for ref in (
                *capability.required_input_artifacts,
                *capability.optional_input_artifacts,
                *capability.output_artifacts,
            ):
                previous = refs.get(ref.artifact_name)
                if previous is not None and not same_artifact_contract(previous, ref):
                    raise CompletionArtifactValidationError(
                        "artifact_contract_conflict"
                    )
                refs[ref.artifact_name] = ref
            for artifact_name, requirement in capability.required_artifact_fields.items():
                required_fields[artifact_name].update(
                    requirement.required_field_keys
                )
    return {
        artifact_name: CanonicalArtifactSpec(
            ref=ref,
            required_field_keys=tuple(sorted(required_fields[artifact_name])),
        )
        for artifact_name, ref in refs.items()
    }


def resolve_capability_contract(
    *,
    run_id: str,
    agent_id: str,
    capability_id: str,
    discovery: CompletionDiscoveryAuthority,
) -> AgentCapabilityContract:
    cache = discovery.get_full_card_cache(run_id)
    worker = cache.workers.get(agent_id)
    capability = (
        next(
            (
                item
                for item in worker.contract.capabilities
                if item.capability_id == capability_id
            ),
            None,
        )
        if worker is not None and worker.contract is not None
        else None
    )
    if capability is None:
        raise CompletionArtifactValidationError(
            "completion_capability_contract_mismatch"
        )
    return capability


def validate_worker_output_artifacts(
    *,
    run_id: str,
    agent_id: str,
    capability_id: str,
    expected_output_artifact_names: set[str],
    output_artifact_refs: Mapping[str, WorkerArtifactRef],
    productive: bool,
    discovery: CompletionDiscoveryAuthority,
    registry: ArtifactRegistryService,
    storage: Storage,
    active: ActiveArtifacts | None = None,
) -> dict[str, ValidatedOutputArtifact]:
    """Validate result refs against card, registry, baseline, and persisted body.

    Productive results require the exact output set and ready persisted bodies.
    Terminal failures may return a declared subset; a non-ready audit artifact is
    retained as invalid by the caller and can never satisfy a dependency.
    """
    actual = set(output_artifact_refs)
    if productive and expected_output_artifact_names - actual:
        raise CompletionArtifactValidationError(
            "completion_output_artifacts_missing"
        )
    if actual - expected_output_artifact_names:
        raise CompletionArtifactValidationError(
            "completion_output_artifacts_unexpected"
        )

    capability = resolve_capability_contract(
        run_id=run_id,
        agent_id=agent_id,
        capability_id=capability_id,
        discovery=discovery,
    )
    outputs = {item.artifact_name: item for item in capability.output_artifacts}
    if set(outputs) != expected_output_artifact_names:
        raise CompletionArtifactValidationError(
            "completion_capability_contract_mismatch"
        )
    specs = canonical_artifact_specs(run_id=run_id, discovery=discovery)
    active = active or registry.get(run_id).active_artifacts
    validated: dict[str, ValidatedOutputArtifact] = {}
    for artifact_name, output_ref in output_artifact_refs.items():
        contract = outputs[artifact_name]
        spec = specs.get(artifact_name)
        if spec is None or not same_artifact_contract(spec.ref, contract):
            raise CompletionArtifactValidationError(
                "completion_capability_contract_mismatch"
            )
        validated[artifact_name] = _validate_one_output(
            run_id=run_id,
            contract=contract,
            spec=spec,
            output_ref=output_ref,
            productive=productive,
            active=active,
            storage=storage,
        )
    return validated


def _validate_one_output(
    *,
    run_id: str,
    contract: ContractArtifactRef,
    spec: CanonicalArtifactSpec,
    output_ref: WorkerArtifactRef,
    productive: bool,
    active: ActiveArtifacts,
    storage: Storage,
) -> ValidatedOutputArtifact:
    registry_field = f"{contract.artifact_name}_id"
    if registry_field not in ActiveArtifacts.model_fields:
        raise CompletionArtifactValidationError(
            "completion_output_artifact_identity_mismatch"
        )
    active_artifact_id = getattr(active, registry_field)
    if (
        output_ref.artifact_type != contract.artifact_name
        or output_ref.run_id != run_id
        or output_ref.artifact_id != active_artifact_id
    ):
        raise CompletionArtifactValidationError(
            "completion_output_artifact_identity_mismatch"
        )
    if (
        output_ref.storage_key is not None
        and output_ref.storage_key != contract.storage_path
    ):
        raise CompletionArtifactValidationError(
            "completion_output_artifact_storage_key_mismatch"
        )
    baseline = active.worker_routing_plan_output_baselines.get(
        contract.artifact_name
    )
    if baseline is None:
        raise CompletionArtifactValidationError(
            "completion_output_baseline_missing"
        )
    if artifact_id_fingerprint(output_ref.artifact_id) == baseline:
        raise CompletionArtifactValidationError(
            "completion_output_artifact_not_new"
        )

    key = storage.run_key(run_id, contract.storage_path)
    if not storage.exists(key):
        raise CompletionArtifactValidationError("completion_output_artifact_invalid")
    try:
        body = storage.read_json(key)
    except Exception:  # noqa: BLE001 - fixed compact code only
        raise CompletionArtifactValidationError(
            "completion_output_artifact_invalid"
        ) from None
    if not isinstance(body, dict):
        raise CompletionArtifactValidationError("completion_output_artifact_invalid")
    if (
        body.get("artifact_id") != output_ref.artifact_id
        or body.get("run_id") != run_id
    ):
        raise CompletionArtifactValidationError("completion_output_artifact_invalid")
    if any(field not in body for field in spec.required_field_keys):
        raise CompletionArtifactValidationError("completion_output_artifact_invalid")

    ready = True
    if contract.readiness_status_field is not None:
        status = body.get(contract.readiness_status_field)
        ready = isinstance(status, str) and status in contract.ready_status_values
    if productive and not ready:
        raise CompletionArtifactValidationError("completion_output_artifact_invalid")
    return ValidatedOutputArtifact(
        artifact_name=contract.artifact_name,
        artifact_id=output_ref.artifact_id,
        ready=ready,
    )


def same_artifact_contract(
    left: ContractArtifactRef, right: ContractArtifactRef
) -> bool:
    return (
        left.artifact_name == right.artifact_name
        and left.storage_path == right.storage_path
        and left.readiness_status_field == right.readiness_status_field
        and set(left.ready_status_values) == set(right.ready_status_values)
    )


def artifact_requirement(
    spec: CanonicalArtifactSpec,
) -> ArtifactFieldRequirement | None:
    if not spec.required_field_keys:
        return None
    return ArtifactFieldRequirement(
        required_field_keys=list(spec.required_field_keys)
    )


__all__ = [
    "CanonicalArtifactSpec",
    "CompletionArtifactValidationError",
    "ValidatedOutputArtifact",
    "artifact_id_fingerprint",
    "artifact_requirement",
    "canonical_artifact_specs",
    "resolve_capability_contract",
    "same_artifact_contract",
    "validate_worker_output_artifacts",
]
