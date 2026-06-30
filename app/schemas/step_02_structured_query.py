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
- `missing_slots[]`: structured required-slot gaps the LLM judged against
  the inferred task intent + the user's query / file metadata. This is the
  machine-readable channel Step 3 consumes to decide blocked vs partial;
  it is distinct from `parse_warnings` (internal parse problems) and
  `clarification_questions` (free-text user prompts).

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


# ── Required-slot gap (Step 2 missing_slots channel). ────────────────────────
#
# The LLM reports which required slots for the inferred task intent are NOT
# satisfied by the user's query / context / uploaded-file metadata. Step 3
# consumes this: a `blocking` slot floors readiness to `blocked`; `warning`
# / `optional` slots stay informational (partial / gap) and never block.


MissingSlotName = Literal[
    "target_or_antigen",
    "antibody",
    "payload",
    "linker",
    "structure_or_sequence",
    "pdb_id",
    "uniprot_id",
    "smiles",
    "task_intent",
    "constraint",
    "other",
]


MissingSlotCategory = Literal[
    "target",
    "antibody",
    "payload",
    "linker",
    "structure",
    "sequence",
    "identifier",
    "task_intent",
    "constraint",
    "other",
]


MissingSlotSeverity = Literal["blocking", "warning", "optional"]


class MissingSlot(BaseModel):
    slot_name: MissingSlotName
    slot_category: MissingSlotCategory = "other"
    severity: MissingSlotSeverity = "warning"
    required_for: list[str] = Field(default_factory=list)
    reason: str = ""
    suggested_question: Optional[str] = None
    evidence: Optional[str] = None


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
    # Structured required-slot gaps (additive; defaults to [] so old
    # artifacts without this field still validate).
    missing_slots: list[MissingSlot] = Field(default_factory=list)
    # User-facing follow-up message the Step 2 LLM writes when missing_slots
    # is non-empty (additive; None/"" when nothing is missing). The program
    # only passes this through — it never re-phrases it. Must never carry raw
    # prompts, keys, file content, or full sequences.
    response: Optional[str] = None
    # LLM-generated canonical (normalized) natural-language description of the
    # CURRENT task. This is the stable working query downstream steps read;
    # `raw_user_query` (Step 1) stays the original, auditable user input and is
    # never overwritten. On a clarification turn the LLM updates this from
    # `previous_canonical_query` + `clarification_answers`. Stable field name —
    # NO query-like aliases (working_query / normalized_query / final_query /
    # rewritten_query / user_query_summary / query_for_downstream /
    # canonical_task / task_summary / query_summary). Never carries prompts,
    # keys, raw payloads, or full sequences; capped at ~800 chars.
    canonical_query: Optional[str] = None
