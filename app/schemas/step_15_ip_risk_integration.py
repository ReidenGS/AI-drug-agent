"""Step 15 — IP filtering / risk integration scaffold output."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class IPRiskFilteringPolicy(BaseModel):
    default_behavior: Literal["label_only", "soft_filter", "hard_filter"] = "label_only"
    hard_filter_enabled: bool = False
    policy_version: Optional[str] = "v0.1"


class HumanReviewFlag(BaseModel):
    required: bool = False
    reason: Optional[
        Literal[
            "antibody_sequence_concern",
            "epitope_concern",
            "antigen_or_target_concern",
            "full_adc_construct_concern",
            "unclear_claim_scope",
            "insufficient_patent_text",
            "high_ip_risk",
            "other",
        ]
    ] = None


class CandidateIPRiskRecord(BaseModel):
    candidate_id: str
    source_rank: Optional[float] = None
    source_patent_record_ids: list[str] = Field(default_factory=list)
    source_tool_call_ids: list[str] = Field(default_factory=list)
    matched_entity_types: list[
        Literal[
            "payload",
            "linker",
            "linker_payload",
            "ligand",
            "compound",
            "drug_reference",
            "scaffold",
            "antibody_sequence",
            "epitope",
            "target",
            "full_adc_construct",
            "full_aoc_construct",
            "oligonucleotide_chemistry",
            "other",
            "unknown",
        ]
    ] = Field(default_factory=list)
    core_automated_ip_screen_scope: Literal[
        "payload_compound_drug_reference",
        "non_core_human_review_only",
        "mixed",
        "unknown",
    ] = "unknown"
    match_strength: Literal["exact", "high_similarity", "partial", "weak", "unknown"] = "unknown"
    claim_relevance: Literal[
        "direct_payload_or_compound_claim",
        "related_compound_or_scaffold",
        "drug_reference",
        "construct_level_claim",
        "background",
        "unclear",
        "unknown",
    ] = "unknown"
    novelty_risk: Literal["high", "medium", "low", "unknown"] = "unknown"
    confidence_level: Literal["high", "medium", "low", "unknown"] = "unknown"
    final_ip_risk_label: Literal["high", "medium", "low", "unknown"] = "unknown"
    human_review_flag: HumanReviewFlag = Field(default_factory=HumanReviewFlag)
    recommended_action: Literal[
        "proceed",
        "proceed_with_warning",
        "human_review_required",
        "deprioritize",
        "exclude_if_hard_filter_enabled",
    ] = "human_review_required"
    ip_risk_rationale: Optional[str] = None
    notes_limitations: Optional[str] = None


class IPFilteredShortlistItem(BaseModel):
    candidate_id: str
    original_rank: Optional[float] = None
    post_ip_status: Literal[
        "kept",
        "kept_with_warning",
        "human_review_required",
        "deprioritized",
        "excluded_by_hard_filter",
    ] = "human_review_required"
    final_ip_risk_label: Literal["high", "medium", "low", "unknown"] = "unknown"
    human_review_required: bool = True


class MissingIPAssessmentFlag(BaseModel):
    candidate_id: Optional[str] = None
    missing_item: Literal[
        "patent_records",
        "payload_or_compound_mapping",
        "claim_text",
        "source_database",
        "matched_entity_id",
        "confidence_level",
        "other",
    ]
    severity: Literal["warning", "blocking"] = "warning"
    message: str


class IPRiskIntegratedShortlist(BaseModel):
    run_id: str
    step_id: str = "step_15"
    created_at: str
    ip_integration_status: Literal[
        "completed",
        "completed_with_warnings",
        "partial",
        "failed",
    ] = "partial"
    legal_disclaimer: str = (
        "For demonstration purposes only. Not a formal legal opinion. "
        "Final patent risk assessment requires attorney review."
    )
    ip_filtering_policy: IPRiskFilteringPolicy = Field(default_factory=IPRiskFilteringPolicy)
    candidate_ip_risk_records: list[CandidateIPRiskRecord] = Field(default_factory=list)
    ip_filtered_shortlist: list[IPFilteredShortlistItem] = Field(default_factory=list)
    missing_ip_assessment_flags: list[MissingIPAssessmentFlag] = Field(default_factory=list)
    ip_integration_notes: Optional[str] = None
