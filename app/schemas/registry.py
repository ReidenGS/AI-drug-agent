"""Run artifact registry — tracks active artifacts and snapshots."""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field


class ActiveArtifacts(BaseModel):
    raw_request_record_id: Optional[str] = None
    structured_query_id: Optional[str] = None
    input_readiness_status_id: Optional[str] = None
    # Latest same-run clarification revision artifact (additive; None until
    # a clarification answer is submitted).
    clarification_state_id: Optional[str] = None
    run_step_plan_id: Optional[str] = None
    # Turn D — additive: id of the persisted worker discovery snapshot for this
    # run (Orchestrator worker discovery + deterministic validation). Additive
    # only; does not affect run_step_plan or any existing artifact.
    worker_discovery_snapshot_id: Optional[str] = None
    # Turn F1 — compact deterministic routing plan; does not replace Step 4's
    # existing run_step_plan or the Turn D discovery snapshot.
    worker_routing_plan_id: Optional[str] = None
    # Independent persisted control identity for the active routing plan.
    # ``worker_routing_plan_id`` remains the artifact id; this field lets a
    # fresh process fail closed if the body's routing_plan_id is tampered.
    worker_routing_plan_control_id: Optional[str] = None
    # Planning-time fingerprints of selected producer output IDs. Values are
    # one-way hashes (or the fixed ``absent`` marker), never raw artifact IDs.
    # Completion must point at a different, newly active artifact.
    worker_routing_plan_output_baselines: dict[str, str] = Field(default_factory=dict)
    candidate_context_table_id: Optional[str] = None
    structured_liability_summary_id: Optional[str] = None
    prepared_structure_input_package_id: Optional[str] = None
    structure_prediction_and_interface_results_id: Optional[str] = None
    structure_variant_and_compound_screening_id: Optional[str] = None
    scoring_handoff_id: Optional[str] = None
    scoring_validation_id: Optional[str] = None
    ranking_table_id: Optional[str] = None
    scientific_evidence_table_id: Optional[str] = None
    patent_prior_art_table_id: Optional[str] = None


class RunArtifactRegistry(BaseModel):
    run_id: str
    run_artifact_registry_id: str
    version: int = 1
    created_at: str
    updated_at: str
    active_artifacts: ActiveArtifacts = Field(default_factory=ActiveArtifacts)
