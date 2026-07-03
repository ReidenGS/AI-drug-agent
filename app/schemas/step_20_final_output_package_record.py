"""Step 20 — final output package metadata scaffold."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

from .step_16_llm_design_review_report import LLMCallRecord


class UserFacingSummary(BaseModel):
    summary_text: Optional[str] = None
    summary_type: Literal["final_recommendation_summary", "package_overview", "other"] = (
        "package_overview"
    )
    llm_call_id: Optional[str] = None


class OutputPackage(BaseModel):
    package_artifact_id: str
    storage_ref: str
    storage_type: Literal["database_record", "s3_path", "local_run_storage", "other"] = (
        "local_run_storage"
    )
    package_format: Literal["zip", "folder", "database_manifest", "other"] = (
        "database_manifest"
    )
    generated_at: str


class IncludedFile(BaseModel):
    file_artifact_id: str
    file_type: Literal[
        "user_facing_final_report",
        "final_rationale_summary",
        "design_review_report",
        "ranked_candidate_table",
        "scoring_summary",
        "evidence_summary",
        "patent_ip_summary",
        "human_review_record",
        "rerun_summary",
        "source_artifact_manifest",
        "other",
    ]
    storage_ref: str
    file_format: Literal["pdf", "docx", "markdown", "csv", "xlsx", "json", "html", "other"]


class PackageWarning(BaseModel):
    warning_type: Literal[
        "missing_required_artifact",
        "missing_optional_artifact",
        "partial_rerun_results",
        "summary_generation_failed",
        "report_file_missing",
        "export_failed",
        "source_artifact_unavailable",
        "other",
    ]
    message: str
    related_artifact_ref: Optional[str] = None


class FinalOutputPackageRecord(BaseModel):
    run_id: str
    step_id: str = "step_20"
    created_at: str
    package_status: Literal[
        "completed",
        "completed_with_warnings",
        "partial",
        "failed",
    ] = "partial"
    package_result_basis: Literal[
        "original_results",
        "rerun_results",
        "original_and_rerun_results",
    ] = "original_results"
    user_facing_summary: UserFacingSummary = Field(default_factory=UserFacingSummary)
    output_package: OutputPackage
    included_files: list[IncludedFile] = Field(default_factory=list)
    llm_call_records: list[LLMCallRecord] = Field(default_factory=list)
    package_warnings: list[PackageWarning] = Field(default_factory=list)
    package_notes: Optional[str] = None
