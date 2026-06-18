"""Step 2 batch-5 follow-up: conservative `mentioned_entities` backfill.

Live Gemini occasionally returns rich `normalized_entities` records while
leaving the legacy `mentioned_entities` fields null/empty. Downstream
services (Step 3 readiness presence checks, Step 5+ agents) still read
the flat strings, so SupervisorAgent now backfills them conservatively
from `normalized_entities` after parsing the LLM payload.

Rules under test:

- Backfill ONLY when the legacy field is missing/null/empty.
- Backfill uses `original_text` (user phrasing), not `canonical_name`.
- Never overwrite an existing non-empty value.
- `entity_type="drug"` (whole ADC products) does NOT map to antibody /
  payload — only the explicit per-component entity types do.
- `linker` and `linker_payload` may fill `linker_text`.
- Multiple matching entries → first wins, no concatenation.
- Step 2 still strips uploaded `storage_path` and never reads file
  bytes; the backfill change doesn't reopen those channels.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from app.agents.supervisor_agent import (
    SupervisorAgent,
    _prompt_inputs_from_raw,
    backfill_mentioned_entities,
)
from app.schemas.step_01_raw_request_record import (
    RawRequestRecord,
    UploadedFile,
    UserProvidedContext,
)


_FIXTURE_RUN_ID = "run_supervisor_step2_backfill_fixture"


def _raw(
    *,
    query: str = "Design a new TROP2 ADC with MMAE payload",
    ctx: dict | None = None,
    files: list[dict] | None = None,
) -> dict:
    rec = RawRequestRecord(
        run_id=_FIXTURE_RUN_ID,
        run_artifact_registry_id="reg_step2_backfill",
        created_at="2026-06-18T00:00:00Z",
        raw_user_query=query,
        user_provided_context=UserProvidedContext(**(ctx or {})),
        uploaded_files=[UploadedFile(**f) for f in (files or [])],
    )
    out = rec.model_dump()
    out["artifact_id"] = "raw_request_record_backfill"
    return out


class _PinnedLLM:
    """Returns a pre-baked structured_query payload — simulates Gemini."""

    name = "pinned"
    model = "pinned-v1"

    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload

    def generate(self, prompt: str, *, system: str | None = None, **kw: Any) -> str:
        raise NotImplementedError

    def generate_json(
        self, prompt: str, *, schema: dict, system: str | None = None
    ) -> dict:
        return self.payload


# ── unit-level helper coverage ────────────────────────────────────────────


def test_backfill_fills_target_and_payload_from_normalized_when_legacy_empty():
    """Live-Gemini observed case: target_or_antigen_text and payload_text
    null even though normalized_entities has both."""
    result = backfill_mentioned_entities(
        mentioned={
            "target_or_antigen_text": None,
            "antibody_candidate_text": None,
            "payload_text": None,
            "linker_text": None,
        },
        normalized=[
            {
                "original_text": "TROP2",
                "canonical_name": "TACSTD2",
                "entity_type": "target_or_antigen",
                "explicit_or_inferred": "explicit",
            },
            {
                "original_text": "MMAE",
                "canonical_name": "monomethyl auristatin E",
                "entity_type": "payload",
                "explicit_or_inferred": "explicit",
            },
        ],
    )
    assert result["target_or_antigen_text"] == "TROP2"
    assert result["payload_text"] == "MMAE"
    # Untouched, still null.
    assert result["antibody_candidate_text"] is None
    assert result["linker_text"] is None


def test_backfill_does_not_overwrite_existing_user_phrase():
    """If LLM already populated target_or_antigen_text, the backfill
    must leave it alone — even when normalized_entities has a canonical
    resolution like ERBB2."""
    result = backfill_mentioned_entities(
        mentioned={"target_or_antigen_text": "HER2 user phrase"},
        normalized=[
            {
                "original_text": "HER2",
                "canonical_name": "ERBB2",
                "entity_type": "target_or_antigen",
                "explicit_or_inferred": "explicit",
            }
        ],
    )
    assert result["target_or_antigen_text"] == "HER2 user phrase"


def test_backfill_uses_original_text_not_canonical_name():
    """User phrasing wins. `canonical_name` must not leak into the
    legacy `mentioned_entities` string fields."""
    result = backfill_mentioned_entities(
        mentioned={},
        normalized=[
            {
                "original_text": "TROP2",
                "canonical_name": "TACSTD2",  # canonical must NOT be used
                "entity_type": "target_or_antigen",
                "explicit_or_inferred": "explicit",
            }
        ],
    )
    assert result["target_or_antigen_text"] == "TROP2"
    assert result["target_or_antigen_text"] != "TACSTD2"


def test_backfill_ignores_drug_entity_for_antibody_and_payload():
    """T-DXd is `entity_type="drug"`. Its decomposition handles
    components separately; backfill must NOT map it into antibody or
    payload fields."""
    result = backfill_mentioned_entities(
        mentioned={
            "target_or_antigen_text": None,
            "antibody_candidate_text": None,
            "payload_text": None,
        },
        normalized=[
            {
                "original_text": "T-DXd",
                "canonical_name": "trastuzumab deruxtecan",
                "entity_type": "drug",
                "explicit_or_inferred": "explicit",
            }
        ],
    )
    assert result["antibody_candidate_text"] is None
    assert result["payload_text"] is None
    assert result["target_or_antigen_text"] is None


def test_backfill_linker_payload_entity_fills_linker_text_when_empty():
    """vc-MMAE-style linker_payload entries may fill linker_text when
    that field is still empty."""
    result = backfill_mentioned_entities(
        mentioned={"linker_text": None},
        normalized=[
            {
                "original_text": "vc-MMAE",
                "canonical_name": "vc-MMAE (valine-citrulline linker + MMAE)",
                "entity_type": "linker_payload",
                "explicit_or_inferred": "explicit",
            }
        ],
    )
    assert result["linker_text"] == "vc-MMAE"


def test_backfill_first_wins_no_concatenation():
    """If two target entities exist, take the first — never join them."""
    result = backfill_mentioned_entities(
        mentioned={},
        normalized=[
            {
                "original_text": "TROP2",
                "canonical_name": "TACSTD2",
                "entity_type": "target_or_antigen",
                "explicit_or_inferred": "explicit",
            },
            {
                "original_text": "HER2",
                "canonical_name": "ERBB2",
                "entity_type": "target_or_antigen",
                "explicit_or_inferred": "explicit",
            },
        ],
    )
    assert result["target_or_antigen_text"] == "TROP2"
    assert "HER2" not in (result.get("target_or_antigen_text") or "")


def test_backfill_disease_or_indication_only_from_disease_entity():
    result = backfill_mentioned_entities(
        mentioned={},
        normalized=[
            {
                "original_text": "HER2-positive breast cancer",
                "canonical_name": "HER2-positive breast cancer",
                "entity_type": "disease_or_indication",
                "explicit_or_inferred": "explicit",
            }
        ],
    )
    assert (
        result["disease_or_indication_text"] == "HER2-positive breast cancer"
    )


def test_backfill_treats_empty_string_as_missing():
    result = backfill_mentioned_entities(
        mentioned={"target_or_antigen_text": "   "},
        normalized=[
            {
                "original_text": "TROP2",
                "canonical_name": "TACSTD2",
                "entity_type": "target_or_antigen",
                "explicit_or_inferred": "explicit",
            }
        ],
    )
    assert result["target_or_antigen_text"] == "TROP2"


def test_backfill_safe_when_normalized_missing_or_malformed():
    """The helper must not crash on None / non-dict items."""
    result = backfill_mentioned_entities({"target_or_antigen_text": None}, None)
    assert result["target_or_antigen_text"] is None
    result = backfill_mentioned_entities(
        {"target_or_antigen_text": None},
        normalized=[None, "not a dict", {"entity_type": "target_or_antigen"}],
    )
    # The malformed entry has no original_text → still None.
    assert result["target_or_antigen_text"] is None


# ── end-to-end via SupervisorAgent + pinned LLM payload ──────────────────


def _gemini_style_payload(
    *,
    mentioned: dict[str, Any] | None = None,
    normalized: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "task_intent": {
            "task_type": "adc_design",
            "modality": "ADC",
            "modality_confidence": 0.9,
            "user_goal_summary": "Design TROP2 ADC with MMAE",
            "primary_intent": "new_adc_design",
            "primary_intent_confidence": 0.7,
            "secondary_intents": ["developability_assessment"],
        },
        "mentioned_entities": mentioned
        or {
            "target_or_antigen_text": None,
            "disease_or_indication_text": None,
            "antibody_candidate_text": None,
            "payload_text": None,
            "linker_text": None,
        },
        "referenced_inputs": [],
        "requested_outputs": ["ranked_candidates", "developability_summary"],
        "user_constraints": [],
        "parse_warnings": [],
        "normalized_entities": normalized or [],
        "entity_decompositions": [],
        "clarification_questions": [],
    }


def test_supervisor_backfills_target_and_payload_from_gemini_style_payload():
    llm = _PinnedLLM(
        _gemini_style_payload(
            normalized=[
                {
                    "original_text": "TROP2",
                    "canonical_name": "TACSTD2",
                    "entity_type": "target_or_antigen",
                    "explicit_or_inferred": "explicit",
                },
                {
                    "original_text": "MMAE",
                    "canonical_name": "monomethyl auristatin E",
                    "entity_type": "payload",
                    "explicit_or_inferred": "explicit",
                },
            ],
        )
    )
    agent = SupervisorAgent(llm=llm)
    sq = agent.parse_raw_to_structured_query(_raw())
    assert sq.mentioned_entities.target_or_antigen_text == "TROP2"
    assert sq.mentioned_entities.payload_text == "MMAE"
    # Normalized entities still carried through.
    norms = {ne.canonical_name for ne in sq.normalized_entities}
    assert "TACSTD2" in norms and "monomethyl auristatin E" in norms


def test_supervisor_does_not_overwrite_explicit_mentioned_entities():
    llm = _PinnedLLM(
        _gemini_style_payload(
            mentioned={
                "target_or_antigen_text": "HER2 user phrase",
                "disease_or_indication_text": None,
                "antibody_candidate_text": None,
                "payload_text": None,
                "linker_text": None,
            },
            normalized=[
                {
                    "original_text": "HER2",
                    "canonical_name": "ERBB2",
                    "entity_type": "target_or_antigen",
                    "explicit_or_inferred": "explicit",
                }
            ],
        )
    )
    agent = SupervisorAgent(llm=llm)
    sq = agent.parse_raw_to_structured_query(_raw())
    assert sq.mentioned_entities.target_or_antigen_text == "HER2 user phrase"


def test_supervisor_does_not_backfill_antibody_or_payload_from_drug_entity():
    llm = _PinnedLLM(
        _gemini_style_payload(
            normalized=[
                {
                    "original_text": "T-DXd",
                    "canonical_name": "trastuzumab deruxtecan",
                    "entity_type": "drug",
                    "explicit_or_inferred": "explicit",
                }
            ],
        )
    )
    agent = SupervisorAgent(llm=llm)
    sq = agent.parse_raw_to_structured_query(_raw())
    assert sq.mentioned_entities.antibody_candidate_text is None
    assert sq.mentioned_entities.payload_text is None
    assert sq.mentioned_entities.target_or_antigen_text is None


def test_supervisor_backfills_linker_text_from_linker_payload_entity():
    llm = _PinnedLLM(
        _gemini_style_payload(
            normalized=[
                {
                    "original_text": "vc-MMAE",
                    "canonical_name":
                        "vc-MMAE (valine-citrulline linker + MMAE)",
                    "entity_type": "linker_payload",
                    "explicit_or_inferred": "explicit",
                }
            ],
        )
    )
    agent = SupervisorAgent(llm=llm)
    sq = agent.parse_raw_to_structured_query(_raw())
    assert sq.mentioned_entities.linker_text == "vc-MMAE"


# ── Step 2 still never reads file bytes / strips storage_path ────────────


def test_step2_backfill_does_not_open_files(tmp_path):
    """Independent guarantee that the backfill path still respects the
    Step 2 file-byte privacy rule. We point a Gemini-style payload at an
    intake record with an attached file on disk and confirm
    `_prompt_inputs_from_raw` strips storage_path and the agent never
    reads the file content."""
    real = tmp_path / "trastuzumab.fasta"
    real.write_text("SENTINEL-SEQUENCE-DO-NOT-LEAK")
    raw = _raw(
        files=[
            {
                "file_id": "f1",
                "original_filename": "trastuzumab.fasta",
                "storage_path": str(real),
                "content_type": "text/x-fasta",
                "size_bytes": real.stat().st_size,
            }
        ]
    )
    # Prompt inputs strip storage_path.
    inputs = _prompt_inputs_from_raw(raw)
    blob = json.dumps(inputs)
    assert "SENTINEL-SEQUENCE-DO-NOT-LEAK" not in blob
    assert "storage_path" not in blob

    # End-to-end agent path with a pinned LLM that backfills payload.
    llm = _PinnedLLM(
        _gemini_style_payload(
            normalized=[
                {
                    "original_text": "TROP2",
                    "canonical_name": "TACSTD2",
                    "entity_type": "target_or_antigen",
                    "explicit_or_inferred": "explicit",
                }
            ],
        )
    )
    agent = SupervisorAgent(llm=llm)
    sq = agent.parse_raw_to_structured_query(raw)
    assert "SENTINEL-SEQUENCE-DO-NOT-LEAK" not in sq.model_dump_json()
    # Backfill still happened.
    assert sq.mentioned_entities.target_or_antigen_text == "TROP2"
