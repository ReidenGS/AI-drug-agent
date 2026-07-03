"""Offline synthetic tests for Step 15-21 scaffold services."""

from __future__ import annotations

from app.services.design_review_service import DesignReviewService
from app.services.final_output_package_service import FinalOutputPackageService
from app.services.human_review_service import HumanReviewService
from app.services.ip_risk_integration_service import IPRiskIntegrationService
from app.services.pipeline_rerun_service import PipelineRerunService
from app.services.redesign_trigger_service import RedesignTriggerService
from app.services.run_tracking_service import RunTrackingService


def test_step15_21_scaffold_services_preserve_refs_and_do_not_live_call(
    local_storage,
    registry_service,
    workflow_state_service,
):
    run_id = "run_test_step15_21"
    registry_service.init_registry(run_id)
    workflow_state_service.init_run(run_id)
    _seed_upstream_artifacts(local_storage, registry_service, run_id)

    step15 = IPRiskIntegrationService(
        storage=local_storage,
        registry=registry_service,
        workflow_state=workflow_state_service,
    ).integrate(run_id)
    assert step15.candidate_ip_risk_records[0].source_patent_record_ids == ["pat_1"]
    assert step15.legal_disclaimer.startswith("For demonstration purposes only")
    assert step15.ip_filtering_policy.hard_filter_enabled is False

    step16 = DesignReviewService(
        storage=local_storage,
        registry=registry_service,
        workflow_state=workflow_state_service,
    ).create_report(run_id)
    assert step16.llm_call_records[0].run_status == "skipped"
    assert "scaffold_only" in (step16.llm_call_records[0].failure_reason or "")
    assert local_storage.exists(step16.report_file.storage_ref)

    step17 = HumanReviewService(
        storage=local_storage,
        registry=registry_service,
        workflow_state=workflow_state_service,
    ).record(
        run_id,
        review_payload={
            "review_status": "completed_with_conditions",
            "report_decision": {
                "decision": "approve_with_conditions",
                "reviewer_confidence": "medium",
            },
            "candidate_decisions": [
                {
                    "candidate_id": "cand_1",
                    "source_rank": 1,
                    "decision": "redesign",
                    "recommended_next_action": "redesign",
                    "decision_rationale": "Reduce IP and developability uncertainty.",
                }
            ],
            "next_step_instruction": "trigger_redesign",
        },
    )
    assert step17.candidate_decisions[0].candidate_id == "cand_1"

    step18 = RedesignTriggerService(
        storage=local_storage,
        registry=registry_service,
        workflow_state=workflow_state_service,
    ).build_tasks(run_id)
    assert step18.redesign_trigger_status == "triggered"
    assert step18.redesign_tasks[0].candidate_id == "cand_1"
    assert step18.redesign_tasks[0].requires_pipeline_rerun is True

    step19 = PipelineRerunService(
        storage=local_storage,
        registry=registry_service,
        workflow_state=workflow_state_service,
    ).record_rerun_status(run_id)
    assert step19.rerun_status == "partial"
    assert step19.rerun_task_results[0].rerun_status == "skipped"
    assert "No pipeline step was re-executed" in (step19.rerun_notes or "")

    step20 = FinalOutputPackageService(
        storage=local_storage,
        registry=registry_service,
        workflow_state=workflow_state_service,
    ).build_package(run_id)
    assert step20.output_package.package_format == "database_manifest"
    assert any(item.file_type == "design_review_report" for item in step20.included_files)
    assert step20.llm_call_records[0].run_status == "skipped"

    step21 = RunTrackingService(
        storage=local_storage,
        registry=registry_service,
        workflow_state=workflow_state_service,
    ).record_tracking(run_id)
    assert step21.memory_update_records[0].write_status == "skipped"
    assert "intentionally deferred" in (step21.memory_update_records[0].failure_reason or "")
    assert local_storage.exists(step21.tracked_artifacts.artifact_registry_snapshot_ref)

    combined = str(
        {
            "step15": step15.model_dump(),
            "step16": step16.model_dump(),
            "step17": step17.model_dump(),
            "step18": step18.model_dump(),
            "step19": step19.model_dump(),
            "step20": step20.model_dump(),
            "step21": step21.model_dump(),
        }
    )
    assert "raw_llm_response" not in combined
    assert "raw_tooluniverse_payload" not in combined
    assert "full_prompt" not in combined
    assert "CDR3" not in combined


def test_step15_missing_upstream_refs_warns_without_crashing(
    local_storage,
    registry_service,
    workflow_state_service,
):
    run_id = "run_missing_step15"
    registry_service.init_registry(run_id)
    workflow_state_service.init_run(run_id)
    out = IPRiskIntegrationService(
        storage=local_storage,
        registry=registry_service,
        workflow_state=workflow_state_service,
    ).integrate(run_id)
    assert out.ip_integration_status == "partial"
    assert out.candidate_ip_risk_records == []
    assert out.missing_ip_assessment_flags


def _seed_upstream_artifacts(local_storage, registry_service, run_id: str) -> None:
    local_storage.write_json(
        local_storage.run_key(run_id, "inputs/run_step_plan.json"),
        {
            "artifact_id": "plan_1",
            "run_id": run_id,
            "plan_status": "ready_to_execute",
            "planned_steps": [],
        },
    )
    local_storage.write_json(
        local_storage.run_key(run_id, "inputs/structured_query.json"),
        {"artifact_id": "sq_1", "run_id": run_id, "mentioned_entities": {}},
    )
    local_storage.write_json(
        local_storage.run_key(run_id, "candidate_context_table.json"),
        {
            "artifact_id": "cct_1",
            "run_id": run_id,
            "candidate_records": [
                {"candidate_id": "cand_1", "candidate_label": "payload A"}
            ],
        },
    )
    local_storage.write_json(
        local_storage.run_key(run_id, "ranking_table.json"),
        {
            "artifact_id": "rank_1",
            "run_id": run_id,
            "ranking_status": "completed",
            "ranked_candidates": [{"candidate_id": "cand_1", "rank": 1}],
        },
    )
    local_storage.write_json(
        local_storage.run_key(run_id, "scientific_evidence_table.json"),
        {"artifact_id": "evidence_1", "run_id": run_id, "evidence_records": []},
    )
    local_storage.write_json(
        local_storage.run_key(run_id, "patent_prior_art_table.json"),
        {
            "artifact_id": "patent_table_1",
            "run_id": run_id,
            "patent_records": [
                {
                    "patent_record_id": "pat_1",
                    "candidate_id": "cand_1",
                    "matched_entity_type": "payload",
                    "novelty_risk": "medium",
                    "confidence_level": "medium",
                    "source_refs": ["tool_outputs/step_14/tc_pat_1.json"],
                }
            ],
            "tool_call_records": [
                {"tool_call_id": "tc_pat_1", "run_status": "success"}
            ],
        },
    )
    registry_service.update_active(
        run_id,
        run_step_plan_id="plan_1",
        structured_query_id="sq_1",
        candidate_context_table_id="cct_1",
        ranking_table_id="rank_1",
        scientific_evidence_table_id="evidence_1",
        patent_prior_art_table_id="patent_table_1",
    )
