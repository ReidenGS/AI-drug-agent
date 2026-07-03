"""Step 20 scaffold service — create a final package manifest record."""

from __future__ import annotations

from ..schemas.step_16_llm_design_review_report import LLMCallRecord
from ..schemas.step_20_final_output_package_record import (
    FinalOutputPackageRecord,
    IncludedFile,
    OutputPackage,
    PackageWarning,
    UserFacingSummary,
)
from ..utils.ids import new_artifact_id, new_tool_call_id
from ..utils.time import now_iso
from .artifact_registry_service import ArtifactRegistryService
from .downstream_scaffold_utils import active_artifact_refs, missing_active_refs, read_json_if_exists, safe_workflow_mark
from .storage_service import Storage
from .workflow_state_service import WorkflowStateService


_ARTIFACT_KEY = "final_output_package_record.json"
_MANIFEST_KEY = "outputs/final_output_package_manifest.json"


class FinalOutputPackageService:
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

    def build_package(self, run_id: str) -> FinalOutputPackageRecord:
        refs = active_artifact_refs(self.registry, run_id)
        now = now_iso()
        package_artifact_id = new_artifact_id("final_output_package")
        manifest_key = self.storage.run_key(run_id, _MANIFEST_KEY)
        included = _included_files(self.storage, run_id, refs)
        warnings = [
            PackageWarning(
                warning_type="missing_required_artifact",
                message=f"Missing upstream artifact ref: {label}",
            )
            for _, label in missing_active_refs(
                self.registry,
                run_id,
                {
                    "ranking_table_id": "ranked_candidate_shortlist",
                    "scientific_evidence_table_id": "scientific_evidence_table",
                    "patent_prior_art_table_id": "patent_prior_art_table",
                    "ip_risk_integrated_shortlist_id": "ip_risk_integrated_shortlist",
                    "llm_design_review_report_id": "llm_design_review_report",
                    "human_review_decision_record_id": "human_review_decision_record",
                },
            )
        ]
        manifest = {
            "run_id": run_id,
            "package_artifact_id": package_artifact_id,
            "generated_at": now,
            "included_files": [item.model_dump() for item in included],
            "active_artifact_refs": refs,
            "scaffold_only": True,
        }
        self.storage.write_json(manifest_key, manifest)
        llm_call = LLMCallRecord(
            llm_call_id=new_tool_call_id(),
            llm_task_type="final_user_summary_generation",
            input_artifact_refs=list(refs.values()),
            run_status="skipped",
            failure_reason="scaffold_only: final natural-language LLM summary intentionally deferred",
        )
        artifact = FinalOutputPackageRecord(
            run_id=run_id,
            created_at=now,
            package_status="partial" if warnings else "completed_with_warnings",
            package_result_basis=(
                "original_and_rerun_results"
                if refs.get("pipeline_rerun_result_record_id")
                else "original_results"
            ),
            user_facing_summary=UserFacingSummary(
                summary_text="Scaffold package manifest created from available artifact refs.",
                summary_type="package_overview",
            ),
            output_package=OutputPackage(
                package_artifact_id=package_artifact_id,
                storage_ref=manifest_key,
                generated_at=now,
            ),
            included_files=included,
            llm_call_records=[llm_call],
            package_warnings=warnings,
            package_notes=(
                "Step 20 scaffold writes a database-manifest style package only; it does not "
                "generate PDF/DOCX exports or recompute scientific results."
            ),
        )
        artifact_id = new_artifact_id("final_output_package_record")
        self.storage.write_json(
            self.storage.run_key(run_id, _ARTIFACT_KEY),
            {"artifact_id": artifact_id, **artifact.model_dump()},
        )
        self.registry.update_active(run_id, final_output_package_record_id=artifact_id)
        safe_workflow_mark(self.workflow_state, run_id, "step_20", "completed")
        return artifact


def _included_files(storage: Storage, run_id: str, refs: dict[str, str]) -> list[IncludedFile]:
    items: list[IncludedFile] = []
    mappings = {
        "ranking_table_id": ("ranked_candidate_table", "ranking_table.json", "json"),
        "scientific_evidence_table_id": ("evidence_summary", "scientific_evidence_table.json", "json"),
        "patent_prior_art_table_id": ("patent_ip_summary", "patent_prior_art_table.json", "json"),
        "ip_risk_integrated_shortlist_id": ("patent_ip_summary", "ip_risk_integrated_shortlist.json", "json"),
        "human_review_decision_record_id": ("human_review_record", "human_review_decision_record.json", "json"),
        "pipeline_rerun_result_record_id": ("rerun_summary", "pipeline_rerun_result_record.json", "json"),
    }
    for ref_name, (file_type, key, file_format) in mappings.items():
        artifact_id = refs.get(ref_name)
        if not artifact_id:
            continue
        items.append(
            IncludedFile(
                file_artifact_id=artifact_id,
                file_type=file_type,  # type: ignore[arg-type]
                storage_ref=storage.run_key(run_id, key),
                file_format=file_format,  # type: ignore[arg-type]
            )
        )
    if refs.get("llm_design_review_report_id"):
        report = read_json_if_exists(storage, run_id, "llm_design_review_report.json") or {}
        report_file = report.get("report_file") or {}
        items.append(
            IncludedFile(
                file_artifact_id=report_file.get("report_artifact_id", refs["llm_design_review_report_id"]),
                file_type="design_review_report",
                storage_ref=report_file.get("storage_ref") or storage.run_key(run_id, "llm_design_review_report.json"),
                file_format=report_file.get("file_format") or "json",
            )
        )
    return items
