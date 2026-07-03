"""Step 17 scaffold service — persist human review decisions."""

from __future__ import annotations

from typing import Any

from ..schemas.step_17_human_review_decision_record import (
    CandidateDecision,
    FollowUpAction,
    HumanReviewDecisionRecord,
    ReportDecision,
    ReviewFeedback,
    ReviewedReportRef,
    ReviewerInfo,
)
from ..utils.ids import new_artifact_id
from ..utils.time import now_iso
from .artifact_registry_service import ArtifactRegistryService
from .downstream_scaffold_utils import read_json_if_exists, safe_workflow_mark
from .storage_service import Storage
from .workflow_state_service import WorkflowStateService


_ARTIFACT_KEY = "human_review_decision_record.json"


class HumanReviewService:
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

    def record(self, run_id: str, review_payload: dict[str, Any] | None = None) -> HumanReviewDecisionRecord:
        payload = review_payload or {}
        active = self.registry.get(run_id).active_artifacts
        report = read_json_if_exists(self.storage, run_id, "llm_design_review_report.json") or {}
        reviewed_report = ReviewedReportRef(
            llm_design_review_report_id=active.llm_design_review_report_id or "missing",
            report_storage_ref=(report.get("report_file") or {}).get("storage_ref"),
        )
        candidate_decisions = [
            CandidateDecision.model_validate(item)
            for item in payload.get("candidate_decisions", [])
        ]
        follow_up_actions = [
            FollowUpAction.model_validate(item)
            for item in payload.get("follow_up_actions", [])
        ]
        artifact = HumanReviewDecisionRecord(
            run_id=run_id,
            created_at=now_iso(),
            review_status=payload.get(
                "review_status",
                "completed" if payload else "needs_more_information",
            ),
            reviewed_report=reviewed_report,
            reviewer_info=ReviewerInfo.model_validate(payload.get("reviewer_info", {})),
            report_decision=ReportDecision.model_validate(payload.get("report_decision", {})),
            candidate_decisions=candidate_decisions,
            review_feedback=ReviewFeedback.model_validate(payload.get("review_feedback", {})),
            follow_up_actions=follow_up_actions,
            next_step_instruction=payload.get("next_step_instruction", "request_more_data"),
            review_notes=payload.get(
                "review_notes",
                "Scaffold record created without UI workflow." if not payload else None,
            ),
        )
        artifact_id = new_artifact_id("human_review_decision_record")
        self.storage.write_json(
            self.storage.run_key(run_id, _ARTIFACT_KEY),
            {"artifact_id": artifact_id, **artifact.model_dump()},
        )
        self.registry.update_active(run_id, human_review_decision_record_id=artifact_id)
        safe_workflow_mark(self.workflow_state, run_id, "step_17", "completed")
        return artifact
