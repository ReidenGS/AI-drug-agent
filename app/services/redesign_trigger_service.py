"""Step 18 scaffold service — convert review decisions into task records."""

from __future__ import annotations

from ..schemas.step_18_redesign_optimization_task_record import (
    MissingRedesignInput,
    NonRedesignOutcome,
    OptimizationGoal,
    RedesignOptimizationTaskRecord,
    RedesignTask,
    RedesignTriggerSource,
)
from ..utils.ids import new_artifact_id
from ..utils.time import now_iso
from .artifact_registry_service import ArtifactRegistryService
from .downstream_scaffold_utils import read_json_if_exists, safe_workflow_mark
from .storage_service import Storage
from .workflow_state_service import WorkflowStateService


_ARTIFACT_KEY = "redesign_optimization_task_record.json"


class RedesignTriggerService:
    def __init__(
        self,
        *,
        storage: Storage,
        registry: ArtifactRegistryService,
        workflow_state: WorkflowStateService,
    ) -> None:
        self.storage = storage
        self.registry = registry
        self.workflow_state = workflow_state

    def build_tasks(self, run_id: str) -> RedesignOptimizationTaskRecord:
        active = self.registry.get(run_id).active_artifacts
        review = read_json_if_exists(self.storage, run_id, "human_review_decision_record.json") or {}
        source_fields: set[str] = set()
        missing: list[MissingRedesignInput] = []
        tasks: list[RedesignTask] = []

        for decision in review.get("candidate_decisions") or []:
            wants_redesign = (
                decision.get("decision") == "redesign"
                or decision.get("recommended_next_action") == "redesign"
            )
            if not wants_redesign:
                continue
            source_fields.add("candidate_decisions")
            tasks.append(
                RedesignTask(
                    redesign_task_id=new_artifact_id("redesign_task"),
                    candidate_id=decision.get("candidate_id"),
                    trigger_reason="human_requested_redesign",
                    optimization_goals=[
                        OptimizationGoal(
                            goal_type="other",
                            goal_description=decision.get("decision_rationale"),
                            priority="medium",
                            source_ref="human_review_decision_record.candidate_decisions",
                        )
                    ],
                    recommended_redesign_scope="unknown",
                    requires_new_candidate_generation=False,
                    requires_pipeline_rerun=True,
                    suggested_rerun_start_step="step_05",
                    task_status="ready" if decision.get("candidate_id") else "needs_clarification",
                    task_notes="Scaffold task only; no redesigned candidate generated.",
                )
            )
            if not decision.get("candidate_id"):
                missing.append(
                    MissingRedesignInput(
                        missing_item="candidate_id",
                        severity="blocking",
                        message="Redesign requested but candidate_id is missing.",
                    )
                )

        for action in review.get("follow_up_actions") or []:
            if action.get("action_type") != "redesign":
                continue
            source_fields.add("follow_up_actions")
            candidate_ids = action.get("related_candidate_ids") or []
            if not candidate_ids:
                missing.append(
                    MissingRedesignInput(
                        missing_item="candidate_id",
                        severity="blocking",
                        message="Follow-up redesign action lacks related_candidate_ids.",
                    )
                )
                tasks.append(
                    RedesignTask(
                        redesign_task_id=new_artifact_id("redesign_task"),
                        candidate_id=None,
                        trigger_reason="human_requested_redesign",
                        optimization_goals=[
                            OptimizationGoal(
                                goal_type="other",
                                goal_description=action.get("action_notes"),
                                priority=action.get("priority", "medium"),
                                source_ref="human_review_decision_record.follow_up_actions",
                            )
                        ],
                        task_status="needs_clarification",
                        task_notes="Candidate target must be clarified before redesign.",
                    )
                )
            for cid in candidate_ids:
                tasks.append(
                    RedesignTask(
                        redesign_task_id=new_artifact_id("redesign_task"),
                        candidate_id=cid,
                        trigger_reason="human_requested_redesign",
                        optimization_goals=[
                            OptimizationGoal(
                                goal_type="other",
                                goal_description=action.get("action_notes"),
                                priority=action.get("priority", "medium"),
                                source_ref="human_review_decision_record.follow_up_actions",
                            )
                        ],
                        requires_pipeline_rerun=True,
                        suggested_rerun_start_step="step_05",
                        task_notes="Scaffold task only; no redesigned candidate generated.",
                    )
                )

        next_instruction = review.get("next_step_instruction")
        if next_instruction == "trigger_redesign":
            source_fields.add("next_step_instruction")
            if not tasks:
                missing.append(
                    MissingRedesignInput(
                        missing_item="candidate_id",
                        severity="blocking",
                        message="next_step_instruction requests redesign but no candidate-level redesign target exists.",
                    )
                )

        if tasks and any(t.task_status == "needs_clarification" for t in tasks):
            status = "needs_clarification"
        elif tasks:
            status = "triggered"
        else:
            status = "not_triggered"

        artifact = RedesignOptimizationTaskRecord(
            run_id=run_id,
            created_at=now_iso(),
            redesign_trigger_status=status,  # type: ignore[arg-type]
            trigger_source=RedesignTriggerSource(
                human_review_decision_record_id=(
                    active.human_review_decision_record_id or "missing"
                ),
                source_fields=sorted(source_fields) or ["other"],  # type: ignore[arg-type]
            ),
            redesign_tasks=tasks,
            non_redesign_outcome=NonRedesignOutcome(
                reason=None if tasks else _non_redesign_reason(next_instruction),
                next_step_instruction=_normalize_non_redesign_instruction(next_instruction),
            ),
            missing_redesign_inputs=missing,
            redesign_trigger_notes=(
                "Step 18 scaffold creates task records only. It does not generate redesigned "
                "candidates or rerun pipeline steps."
            ),
        )
        artifact_id = new_artifact_id("redesign_optimization_task_record")
        self.storage.write_json(
            self.storage.run_key(run_id, _ARTIFACT_KEY),
            {"artifact_id": artifact_id, **artifact.model_dump()},
        )
        self.registry.update_active(run_id, redesign_optimization_task_record_id=artifact_id)
        safe_workflow_mark(self.workflow_state, run_id, "step_18", "completed")
        return artifact


def _non_redesign_reason(instruction: str | None) -> str:
    if instruction == "proceed_to_output_package":
        return "proceed_to_output_package"
    if instruction == "stop_run":
        return "stop_run"
    if instruction == "revise_report":
        return "report_revision_only"
    if instruction == "request_more_data":
        return "request_more_data_only"
    return "no_redesign_requested"


def _normalize_non_redesign_instruction(instruction: str | None) -> str:
    allowed = {"proceed_to_output_package", "revise_report", "request_more_data", "stop_run", "other"}
    return instruction if instruction in allowed else "other"
