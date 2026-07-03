"""Step 16 — scaffold-only LLM design review report metadata."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class DesignReviewReportFile(BaseModel):
    report_artifact_id: str
    report_type: Literal["llm_design_review_report"] = "llm_design_review_report"
    storage_ref: str
    storage_type: Literal["database_record", "s3_path", "local_run_storage", "other"] = (
        "local_run_storage"
    )
    file_format: Literal["pdf", "docx", "markdown", "html", "json", "other"] = "markdown"
    generated_at: str


class DesignReviewReportMetadata(BaseModel):
    candidate_count: Optional[int] = None
    included_section_count: Optional[int] = None


class LLMCallRecord(BaseModel):
    llm_call_id: str
    llm_task_type: Literal[
        "design_review_report_generation",
        "report_revision",
        "fallback_summary",
        "final_user_summary_generation",
        "package_summary_revision",
        "other",
    ] = "design_review_report_generation"
    input_artifact_refs: list[str] = Field(default_factory=list)
    prompt_template_version: Optional[str] = None
    model_name: Optional[str] = None
    run_status: Literal["success", "failed", "skipped"] = "skipped"
    output_ref: Optional[str] = None
    failure_reason: Optional[str] = None


class DesignReviewWarning(BaseModel):
    warning_type: Literal[
        "missing_scoring_data",
        "missing_evidence_data",
        "missing_patent_data",
        "llm_output_incomplete",
        "report_generation_failed",
        "source_conflict",
        "other",
    ]
    candidate_id: Optional[str] = None
    message: str


class LLMDesignReviewReport(BaseModel):
    run_id: str
    step_id: str = "step_16"
    created_at: str
    design_review_status: Literal[
        "completed",
        "completed_with_warnings",
        "partial",
        "failed",
    ] = "partial"
    report_file: DesignReviewReportFile
    report_metadata: DesignReviewReportMetadata = Field(default_factory=DesignReviewReportMetadata)
    llm_call_records: list[LLMCallRecord] = Field(default_factory=list)
    design_review_warnings: list[DesignReviewWarning] = Field(default_factory=list)
    design_review_notes: Optional[str] = None
