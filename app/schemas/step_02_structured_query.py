"""Step 2 — structured_query (SupervisorAgent output).

Schema additions in batch 5 (professor feedback):

- `task_intent.primary_intent` + `task_intent.secondary_intents`: explicit
  intent classification on top of the legacy `task_type` / `modality`
  pair. Backward compatible — old callers still read `task_type` and
  `modality`; new callers can branch on `primary_intent`.
- `normalized_entities[]`: alias-resolved entity records that preserve
  the user's original phrasing AND a canonical name / canonical id. Each
  entry marks `explicit_or_inferred`.
- `entity_decompositions[]`: when a single user term refers to a
  multi-component ADC (T-DM1 = trastuzumab + emtansine), this lists the
  inferred components with `inferred=True`.
- `clarification_questions[]`: user-facing questions distinct from the
  internal `parse_warnings` channel.

All fields are additive with safe defaults so existing artifacts and
existing downstream code (Step 3, Step 4) keep working without changes.
"""

from __future__ import annotations

from typing import Literal, Optional
from pydantic import BaseModel, Field


# ── Intent enum (additive — `task_type` stays as the legacy free-form field). ──
#
# Order chosen so JSON dumps read naturally. `unclear_or_needs_clarification`
# is the catch-all fallback when the parser cannot confidently classify.
PrimaryIntent = Literal[
    "new_adc_design",
    "existing_adc_evaluation",
    "developability_assessment",
    "structure_analysis",
    "compound_screening",
    "literature_review",
    "patent_ip_review",
    "optimization",
    "unclear_or_needs_clarification",
]


SecondaryIntent = PrimaryIntent  # same enum; reused for `secondary_intents`.


# ── Normalized-entity record. ────────────────────────────────────────────────
#
# `original_text` preserves the exact phrasing the user wrote (or the
# user_provided_context value); `canonical_name` is the resolved standard
# label (HER2 → ERBB2); `canonical_id` is an authoritative database
# identifier when one exists (e.g. HGNC symbol, UniProt accession,
# DrugBank id). `explicit_or_inferred` is mandatory — the parser must
# tell downstream callers whether the user said the canonical form, or
# whether the parser inferred it from an alias / family name.

EntityType = Literal[
    "target_or_antigen",
    "disease_or_indication",
    "antibody",
    "payload",
    "linker",
    "linker_payload",  # composite linker+payload reagent (vc-MMAE, deruxtecan)
    "drug",            # an entire ADC product (T-DM1, T-DXd, Enhertu, …)
    "compound",        # generic small molecule
    "other",
]


class NormalizedEntity(BaseModel):
    original_text: str
    canonical_name: Optional[str] = None
    canonical_id: Optional[str] = None
    canonical_id_source: Optional[str] = None  # e.g. "HGNC", "UniProt", "DrugBank"
    entity_type: EntityType = "other"
    explicit_or_inferred: Literal["explicit", "inferred"] = "inferred"
    confidence: float = 0.0
    notes: Optional[str] = None


# ── Entity decomposition (T-DM1 = trastuzumab + emtansine, etc.). ────────────


class EntityComponent(BaseModel):
    role: Literal["antibody", "payload", "linker", "linker_payload", "other"] = "other"
    canonical_name: str
    component_type: Optional[str] = None
    canonical_id: Optional[str] = None
    canonical_id_source: Optional[str] = None
    inferred: bool = True
    source: Optional[str] = None
    notes: Optional[str] = None


class EntityDecomposition(BaseModel):
    original_text: str
    canonical_name: Optional[str] = None
    components: list[EntityComponent] = Field(default_factory=list)
    notes: Optional[str] = None


# ── Legacy + Step 2 batch 5 task intent. ─────────────────────────────────────


class TaskIntent(BaseModel):
    task_type: str
    task_type_confidence: float = 0.0
    modality: str = "ADC"
    modality_confidence: float = 0.0
    user_goal_summary: str = ""
    # Batch 5 additions:
    primary_intent: PrimaryIntent = "unclear_or_needs_clarification"
    primary_intent_confidence: float = 0.0
    secondary_intents: list[SecondaryIntent] = Field(default_factory=list)


class SourceRawRequestRef(BaseModel):
    raw_request_record_id: str


class MentionedEntities(BaseModel):
    target_or_antigen_text: Optional[str] = None
    disease_or_indication_text: Optional[str] = None
    antibody_candidate_text: Optional[str] = None
    payload_text: Optional[str] = None
    linker_text: Optional[str] = None


class StructuredQuery(BaseModel):
    run_id: str
    step_id: str = "step_02_structured_query"
    parsed_at: str
    source_raw_request_ref: SourceRawRequestRef
    task_intent: TaskIntent
    mentioned_entities: MentionedEntities = Field(default_factory=MentionedEntities)
    referenced_inputs: list[dict] = Field(default_factory=list)
    requested_outputs: list[str] = Field(default_factory=list)
    user_constraints: list[dict] = Field(default_factory=list)
    parse_warnings: list[str] = Field(default_factory=list)
    # Batch 5 additions — all default to empty so existing artifacts /
    # callers that don't populate them continue to validate.
    normalized_entities: list[NormalizedEntity] = Field(default_factory=list)
    entity_decompositions: list[EntityDecomposition] = Field(default_factory=list)
    clarification_questions: list[str] = Field(default_factory=list)
