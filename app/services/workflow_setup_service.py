"""Step 4 — WorkflowSetupService.

Reads BOTH `structured_query` and `input_readiness_status`, then emits a
per-step deterministic plan. The graph (and any future SQS scheduler) consults
`planned_steps[]` to decide whether to run / skip / mark partial for each
step.

Decision rules (MVP):
- readiness `blocked`          → WorkflowStateError, no plan written.
- readiness `needs_user_input` → plan_status `wait_for_input`, planned_steps
  still populated so callers see *why* per step.
- otherwise                    → plan_status `ready_to_execute`.

Per-step gating (Step 5..14):
- Target missing                 → Step 5/6 blocked; Step 7-9 blocked;
                                    Step 13/14 partial (still searchable).
- Antibody candidate missing     → Step 5 runs (discovery); Step 6 partial;
                                    Step 7-9 partial.
- Payload/linker missing         → compound lanes (Step 6 / Step 9 compound /
                                    Step 14 compound patents) partial.
- Structure/sequence missing     → Step 7-9 partial (no structure_lane).
- Plan never blocks evidence / patent agents purely on entity gaps; they may
  still find context from target/antibody names.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from ..schemas.step_04_run_step_plan import (
    DefaultExecutionPolicy,
    PlannedStep,
    RunStepPlan,
    SkippedStep,
)
from ..utils.errors import WorkflowStateError
from ..utils.ids import new_artifact_id
from ..utils.time import now_iso
from .artifact_registry_service import ArtifactRegistryService
from .storage_service import Storage
from .workflow_state_service import WorkflowStateService


_ARTIFACT_KEY = "inputs/run_step_plan.json"


# Ordered list of step ids we plan over. Lane categories per step are
# orthogonal to the run/skip decision.
_PIPELINE_STEPS: tuple[str, ...] = (
    "step_05_candidate_context",
    "step_06_developability",
    "step_07_structure_input",
    "step_08_structure_evaluation",
    "step_09_structure_design",
    "step_10_scoring_handoff",
    "step_11_scoring_validation",
    "step_12_ranking",
    "step_13_evidence",
    "step_14_patent_ip",
)


@dataclass(slots=True)
class _Signals:
    has_target: bool
    has_antibody: bool
    has_payload: bool
    has_linker: bool
    has_structure_or_sequence: bool
    readiness_status: str  # "ready" | "needs_user_input"


class WorkflowSetupService:
    def __init__(
        self,
        storage: Storage,
        registry: ArtifactRegistryService,
        workflow_state: WorkflowStateService,
    ) -> None:
        self.storage = storage
        self.registry = registry
        self.workflow_state = workflow_state

    # ── public API ──────────────────────────────────────────────────────────
    def plan(self, run_id: str) -> RunStepPlan:
        reg = self.registry.get(run_id)
        readiness_id = reg.active_artifacts.input_readiness_status_id
        sq_id = reg.active_artifacts.structured_query_id
        if not readiness_id:
            raise WorkflowStateError("Step 4 requires input_readiness_status from Step 3")
        if not sq_id:
            raise WorkflowStateError("Step 4 requires structured_query from Step 2")

        readiness = self.storage.read_json(
            self.storage.run_key(run_id, "inputs/input_readiness_status.json")
        )
        sq = self.storage.read_json(self.storage.run_key(run_id, "inputs/structured_query.json"))

        if readiness["input_readiness_status"] == "blocked":
            raise WorkflowStateError(
                "Cannot plan run while readiness=blocked",
                detail={"blocking_reasons": readiness.get("blocking_reasons", [])},
            )

        signals = _gather_signals(readiness=readiness, structured_query=sq)
        planned_steps, skipped_steps = _decide_per_step(signals)

        plan_status: str
        if signals.readiness_status == "needs_user_input":
            plan_status = "wait_for_input"
        else:
            plan_status = "ready_to_execute"

        plan = RunStepPlan(
            run_id=run_id,
            planned_at=now_iso(),
            plan_status=plan_status,  # type: ignore[arg-type]
            default_execution_policy=DefaultExecutionPolicy(),
            planned_steps=planned_steps,
            skipped_steps=skipped_steps,
            skipped_step_ids=[s.step_id for s in skipped_steps],
            planning_warnings=_build_warnings(signals),
            planning_notes=_build_notes(signals),
        )

        artifact_id = new_artifact_id("run_step_plan")
        self.storage.write_json(
            self.storage.run_key(run_id, _ARTIFACT_KEY),
            {"artifact_id": artifact_id, **plan.model_dump()},
        )
        self.registry.update_active(run_id, run_step_plan_id=artifact_id)
        self.workflow_state.mark(run_id, "step_04", "completed")
        return plan


# ── helpers ─────────────────────────────────────────────────────────────────

def _gather_signals(*, readiness: dict, structured_query: dict) -> _Signals:
    presence = readiness.get("basic_adc_input_presence") or {}
    entities = structured_query.get("mentioned_entities") or {}
    ref_id_types = {
        ref.get("id_type")
        for ref in (structured_query.get("referenced_inputs") or [])
        if isinstance(ref, dict)
    }
    return _Signals(
        has_target=bool(presence.get("target_or_antigen_present"))
        or bool(entities.get("target_or_antigen_text")),
        has_antibody=bool(presence.get("antibody_candidate_present"))
        or bool(entities.get("antibody_candidate_text")),
        has_payload=bool(presence.get("payload_present"))
        or bool(entities.get("payload_text")),
        has_linker=bool(presence.get("linker_present"))
        or bool(entities.get("linker_text")),
        has_structure_or_sequence=(
            bool(presence.get("structure_or_sequence_present"))
            or "pdb_id" in ref_id_types
            or "uniprot_id" in ref_id_types
        ),
        readiness_status=readiness.get("input_readiness_status", "needs_user_input"),
    )


def _planned(
    step_id: str,
    status: str,
    reason: str,
    *,
    required: Iterable[str] = (),
    lane_flags: dict[str, bool] | None = None,
) -> PlannedStep:
    return PlannedStep(
        step_id=step_id,
        planned_status=status,  # type: ignore[arg-type]
        reason=reason,
        required_artifact_refs=list(required),
        lane_flags=lane_flags or {},
    )


def _decide_per_step(s: _Signals) -> tuple[list[PlannedStep], list[SkippedStep]]:
    planned: list[PlannedStep] = []
    skipped: list[SkippedStep] = []

    # Step 5 — candidate context.
    if not s.has_target:
        planned.append(_planned(
            "step_05_candidate_context", "blocked",
            "target/antigen missing — cannot build candidate context",
            required=["candidate_context_table"],
        ))
        skipped.append(SkippedStep(
            step_id="step_05_candidate_context",
            reason_type="missing_input",
            reason="no target/antigen",
        ))
    else:
        planned.append(_planned(
            "step_05_candidate_context", "run",
            "target present; will assemble candidate/material records",
            required=["candidate_context_table"],
            lane_flags={
                "target_discovery_lane": True,
                "antibody_discovery_lane": not s.has_antibody,
                "compound_lane": bool(s.has_payload or s.has_linker),
                "structure_lane": s.has_structure_or_sequence,
            },
        ))

    # Step 6 — developability.
    if not s.has_target:
        planned.append(_planned(
            "step_06_developability", "blocked",
            "target/antigen missing — no candidate context to pre-filter",
        ))
        skipped.append(SkippedStep(
            step_id="step_06_developability",
            reason_type="dependency_missing",
            reason="Step 5 cannot run without target",
        ))
    else:
        compound_lane = bool(s.has_payload)
        antibody_lane = bool(s.has_antibody)
        # Without antibody OR payload, Step 6 has limited material to score.
        if not (antibody_lane or compound_lane):
            planned.append(_planned(
                "step_06_developability", "partial",
                "no antibody nor payload identified — lanes will run with target-only inputs",
                lane_flags={"compound_lane": False, "antibody_lane": False},
            ))
        elif not compound_lane:
            planned.append(_planned(
                "step_06_developability", "partial",
                "payload/linker missing — compound lanes will be skipped",
                lane_flags={"compound_lane": False, "antibody_lane": antibody_lane},
            ))
        elif not antibody_lane:
            planned.append(_planned(
                "step_06_developability", "partial",
                "antibody candidate missing — antibody-sequence lanes will be skipped",
                lane_flags={"compound_lane": True, "antibody_lane": False},
            ))
        else:
            planned.append(_planned(
                "step_06_developability", "run",
                "antibody + payload present; all lanes eligible",
                lane_flags={"compound_lane": True, "antibody_lane": True},
            ))

    # Steps 7-9 — structure lanes.
    structure_status = "partial" if not s.has_structure_or_sequence else "run"
    if not s.has_target:
        structure_status = "blocked"
    structure_reason = (
        "target missing"
        if not s.has_target
        else (
            "no structure or sequence reference — structure lanes will run with discovery only"
            if not s.has_structure_or_sequence
            else "structure or sequence reference available"
        )
    )
    for step_id in ("step_07_structure_input", "step_08_structure_evaluation", "step_09_structure_design"):
        planned.append(_planned(
            step_id, structure_status, structure_reason,
            lane_flags={
                "structure_lane": s.has_structure_or_sequence,
                "compound_lane": s.has_payload,
            },
        ))
        if structure_status == "blocked":
            skipped.append(SkippedStep(
                step_id=step_id, reason_type="dependency_missing", reason=structure_reason,
            ))

    # Step 10-12 — scoring/ranking. Always plan run; data sufficiency is
    # checked at execution time, not here.
    for step_id in ("step_10_scoring_handoff", "step_11_scoring_validation", "step_12_ranking"):
        planned.append(_planned(
            step_id, "run", "scoring/handoff/ranking always attempted post Step 9",
        ))

    # Step 13/14 — evidence + patent. Never globally blocked on entity gaps;
    # they may still find context from names. Compound-specific patent calls
    # will be skipped downstream if payload absent.
    planned.append(_planned(
        "step_13_evidence", "run",
        "evidence search always attempted (uses target/candidate/payload names if present)",
        lane_flags={"compound_lane": s.has_payload},
    ))
    planned.append(_planned(
        "step_14_patent_ip", "run",
        "patent / prior-art always attempted",
        lane_flags={
            "compound_lane": s.has_payload,
            "regulatory_lane": True,
        },
    ))

    return planned, skipped


def _build_warnings(s: _Signals) -> list[str]:
    warnings: list[str] = []
    if not s.has_antibody:
        warnings.append("antibody_candidate absent — Step 5 will rely on discovery")
    if not s.has_payload:
        warnings.append("payload absent — compound lanes will be partial/skipped")
    if not s.has_structure_or_sequence:
        warnings.append("structure/sequence absent — Step 7-9 will run discovery-only")
    return warnings


def _build_notes(s: _Signals) -> str:
    return (
        "Plan computed deterministically from input_readiness_status + structured_query. "
        f"Lane signals: target={s.has_target}, antibody={s.has_antibody}, "
        f"payload={s.has_payload}, linker={s.has_linker}, "
        f"structure_or_sequence={s.has_structure_or_sequence}."
    )


def planned_step_for(plan: dict | RunStepPlan, step_id: str) -> PlannedStep | None:
    """Lookup helper used by graph nodes and downstream agents."""
    if isinstance(plan, RunStepPlan):
        items = plan.planned_steps
        for p in items:
            if p.step_id == step_id:
                return p
        return None
    for p in plan.get("planned_steps") or []:
        if p.get("step_id") == step_id:
            return PlannedStep.model_validate(p)
    return None


@dataclass(slots=True)
class ExecutionDecision:
    """Gate decision shared by graph nodes and Step 5/6 API handlers.

    `allow=True` means the agent may run; the planned_step may still ask for
    `partial` execution, which is the agent's concern, not the gate's.
    `allow=False` means the step must NOT execute — graph marks the step
    `"skipped"` and the API returns 409.
    """

    allow: bool
    reason: str
    plan_status: str
    planned_status: str | None


def execution_decision(plan: dict | RunStepPlan | None, step_id: str) -> ExecutionDecision:
    """Decide whether `step_id` may run given the persisted Step 4 plan.

    Order of checks:
    1. If no plan exists, fail closed (no, the gate should ALWAYS see a plan).
    2. Top-level `plan_status` is the global gate: only `ready_to_execute`
       allows downstream execution. `wait_for_input` and `blocked` both block.
    3. Per-step `planned_status` is the local gate: `skip` and `blocked` stop
       this step even when the global plan is otherwise ready.
    """
    if plan is None:
        return ExecutionDecision(
            allow=False,
            reason="no run_step_plan found; Step 4 must complete first",
            plan_status="missing",
            planned_status=None,
        )

    plan_status = (
        plan.plan_status if isinstance(plan, RunStepPlan) else plan.get("plan_status", "")
    )
    if plan_status == "wait_for_input":
        return ExecutionDecision(
            allow=False,
            reason="plan_status=wait_for_input — user input required before Step 5+ can run",
            plan_status=plan_status,
            planned_status=None,
        )
    if plan_status == "blocked":
        return ExecutionDecision(
            allow=False,
            reason="plan_status=blocked — Step 5+ execution halted by Step 4 policy",
            plan_status=plan_status,
            planned_status=None,
        )
    if plan_status != "ready_to_execute":
        return ExecutionDecision(
            allow=False,
            reason=f"plan_status={plan_status!r} is not ready_to_execute",
            plan_status=plan_status,
            planned_status=None,
        )

    planned = planned_step_for(plan, step_id)
    planned_status = planned.planned_status if planned else None
    if planned and planned.planned_status in {"skip", "blocked"}:
        return ExecutionDecision(
            allow=False,
            reason=f"planned_status={planned.planned_status}: {planned.reason}",
            plan_status=plan_status,
            planned_status=planned_status,
        )
    return ExecutionDecision(
        allow=True,
        reason=(planned.reason if planned else "no per-step decision; allowing"),
        plan_status=plan_status,
        planned_status=planned_status,
    )
