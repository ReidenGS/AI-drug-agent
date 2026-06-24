"""Step 14 — patent_prior_art_table (PatentIPAgent output).

Canonical schema lives in ADC_Pipeline_IO_Schema_v0.1.md §Step 14. Raw
Orange Book / PubChem / DrugBank product-level fields stay in raw tool
outputs (referenced via ToolCallRecord.tool_output_ref); only the normalized
fields below land in `patent_records[]`.
"""

from __future__ import annotations

from typing import Literal, Optional
from pydantic import BaseModel, Field

from .common import ToolCallRecord


PatentSourceDatabase = Literal["PubChem", "DrugBank", "FDA_OrangeBook", "USPTO", "other"]

PatentMatchedEntityType = Literal[
    "payload",
    "linker",
    "linker_payload",
    "ligand",
    "compound",
    "drug_reference",
    "drug_application_or_regulatory_reference",
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

ConfidenceLabel = Literal["high", "medium", "low", "unknown"]


class PatentRecord(BaseModel):
    patent_record_id: str
    candidate_id: str
    matched_entity_type: PatentMatchedEntityType = "unknown"
    matched_material_id: Optional[str] = None
    source_database: PatentSourceDatabase = "other"
    patent_title: Optional[str] = None
    patent_number: Optional[str] = None
    patent_application_number: Optional[str] = None
    publication_date: Optional[str] = None
    filing_date: Optional[str] = None
    assignee: Optional[str] = None
    inventors: list[str] = Field(default_factory=list)
    claim_relevance: ConfidenceLabel = "unknown"
    novelty_risk: ConfidenceLabel = "unknown"
    confidence_level: ConfidenceLabel = "unknown"
    key_claim_or_prior_art_summary: Optional[str] = None
    source_url: Optional[str] = None
    source_ref: Optional[str] = None
    notes_limitations: Optional[str] = None
    # ── Step 14 systematic prior-art normalization additive fields ─────
    # Compact provenance / scoring; raw payload stays out by design.
    query_role: Optional[str] = None
    query_term: Optional[str] = None
    query_term_source: Optional[str] = None
    publication_year: Optional[int] = None
    jurisdiction: Optional[str] = None
    claim_focus: Optional[str] = None
    sources: list[str] = Field(default_factory=list)
    source_refs: list[str] = Field(default_factory=list)
    ip_relevance_score: Optional[float] = None
    relevance_rationale: Optional[str] = None


class PatentPriorArtTable(BaseModel):
    run_id: str
    step_id: str = "step_14_patent_ip"
    created_at: str
    patent_review_status: Literal[
        "completed", "completed_with_warnings", "partial", "failed"
    ] = "partial"
    legal_disclaimer: str = (
        "For demonstration purposes only. Not a formal legal opinion. "
        "Final patent risk assessment requires attorney review."
    )
    patent_records: list[PatentRecord] = Field(default_factory=list)
    tool_call_records: list[ToolCallRecord] = Field(default_factory=list)
    patent_review_notes: Optional[str] = None
