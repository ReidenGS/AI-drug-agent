"""Minimal schema instantiation coverage for Step 15-21 scaffold artifacts."""

from __future__ import annotations

from app.schemas.step_15_ip_risk_integration import IPRiskIntegratedShortlist
from app.schemas.step_16_llm_design_review_report import (
    DesignReviewReportFile,
    LLMDesignReviewReport,
)
from app.schemas.step_17_human_review_decision_record import (
    HumanReviewDecisionRecord,
    ReviewedReportRef,
)
from app.schemas.step_18_redesign_optimization_task_record import (
    RedesignOptimizationTaskRecord,
    RedesignTriggerSource,
)
from app.schemas.step_19_pipeline_rerun_result_record import PipelineRerunResultRecord
from app.schemas.step_20_final_output_package_record import (
    FinalOutputPackageRecord,
    OutputPackage,
)
from app.schemas.step_21_run_tracking_record import (
    RunTrackingMemoryUpdateRecord,
    TrackedArtifacts,
)


def test_step15_21_schema_minimal_instantiation():
    assert IPRiskIntegratedShortlist(run_id="run_test", created_at="now").step_id == "step_15"
    assert LLMDesignReviewReport(
        run_id="run_test",
        created_at="now",
        report_file=DesignReviewReportFile(
            report_artifact_id="file_1",
            storage_ref="runs/run_test/reports/report.md",
            generated_at="now",
        ),
    ).step_id == "step_16"
    assert HumanReviewDecisionRecord(
        run_id="run_test",
        created_at="now",
        reviewed_report=ReviewedReportRef(llm_design_review_report_id="rpt_1"),
    ).step_id == "step_17"
    assert RedesignOptimizationTaskRecord(
        run_id="run_test",
        created_at="now",
        trigger_source=RedesignTriggerSource(human_review_decision_record_id="hr_1"),
    ).step_id == "step_18"
    assert PipelineRerunResultRecord(
        run_id="run_test",
        created_at="now",
        source_redesign_task_record_id="rt_1",
    ).step_id == "step_19"
    assert FinalOutputPackageRecord(
        run_id="run_test",
        created_at="now",
        output_package=OutputPackage(
            package_artifact_id="pkg_1",
            storage_ref="runs/run_test/outputs/manifest.json",
            generated_at="now",
        ),
    ).step_id == "step_20"
    assert RunTrackingMemoryUpdateRecord(
        run_id="run_test",
        created_at="now",
        tracked_artifacts=TrackedArtifacts(
            run_artifact_registry_id="reg_1",
            artifact_registry_snapshot_ref="runs/run_test/registry/snapshot.json",
        ),
    ).step_id == "step_21"
