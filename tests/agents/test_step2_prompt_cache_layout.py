"""Step 2 prompt cache-friendly layout tests.

Step 2's real-provider prompt (`build_json_prompt`) is composed of a
**stable prefix** (system block + fixed JSON-only instruction + per-task
shape hint + the fixed task/developer instruction text) followed by a
**dynamic suffix** (the run-specific `prompt_inputs` payload, rendered as
JSON). The split is exposed via `build_json_prompt_sections` so it can be
verified directly instead of just asserted informally in prose.

Goal: the stable prefix should be byte-identical across different user
queries, uploaded files, and clarification turns (for the same code
version / provider task), so provider-side prefix-based prompt caching can
actually hit. All run/user-specific data — `raw_user_query`,
`user_provided_context`, `clarification_answers`, uploaded-file compact
metadata — must live only in the dynamic suffix, strictly after the stable
prefix ends. `storage_path`, `run_id`, `created_at`/`parsed_at`, and raw
file bytes must never appear anywhere in the rendered prompt text (Step 2
already never reads file bytes; this only guards the two prompt sections).

This file does not change Step 2 business logic — it only characterizes
the prompt-layout refactor in `app/llm/json_task_validation.py`.
"""

from __future__ import annotations

from app.agents.supervisor_agent import (
    SUPERVISOR_SYSTEM_PROMPT,
    SupervisorAgent,
    _prompt_inputs_from_raw,
    build_supervisor_user_prompt,
)
from app.llm.json_task_validation import build_json_prompt, build_json_prompt_sections
from app.llm.provider import MockLLMProvider
from app.schemas.step_01_raw_request_record import (
    RawRequestRecord,
    UploadedFile,
    UserProvidedContext,
)


def _raw(
    *,
    run_id: str = "run_cache_layout_fixture",
    query: str = "Design an ADC against HER2 with vc-MMAE.",
    ctx: dict | None = None,
    files: list[dict] | None = None,
) -> dict:
    rec = RawRequestRecord(
        run_id=run_id,
        run_artifact_registry_id="reg_cache_layout",
        created_at="2026-06-30T00:00:00Z",
        raw_user_query=query,
        user_provided_context=UserProvidedContext(**(ctx or {})),
        uploaded_files=[UploadedFile(**f) for f in (files or [])],
    )
    out = rec.model_dump()
    out["artifact_id"] = "art_cache_layout"
    return out


def _step2_sections(raw: dict) -> tuple[str, str]:
    """Build the Step 2 real-provider prompt sections exactly the way
    OpenAI/Qwen/Gemini providers do via `SupervisorAgent.parse_raw_to_structured_query`."""
    prompt_inputs = _prompt_inputs_from_raw(raw)
    prompt = build_supervisor_user_prompt(raw)
    schema = {
        "task": "structured_query",
        "prompt_inputs": prompt_inputs,
        "raw_request_record": raw,
    }
    return build_json_prompt_sections(prompt=prompt, schema=schema, system=SUPERVISOR_SYSTEM_PROMPT)


def _step2_full_prompt(raw: dict) -> str:
    prompt_inputs = _prompt_inputs_from_raw(raw)
    prompt = build_supervisor_user_prompt(raw)
    schema = {
        "task": "structured_query",
        "prompt_inputs": prompt_inputs,
        "raw_request_record": raw,
    }
    return build_json_prompt(prompt=prompt, schema=schema, system=SUPERVISOR_SYSTEM_PROMPT)


# ── 1. stable prefix is byte-identical across different raw_user_query ─────


def test_stable_prefix_byte_identical_across_different_queries():
    stable1, _ = _step2_sections(_raw(query="Design an ADC against HER2 with vc-MMAE."))
    stable2, _ = _step2_sections(_raw(query="Evaluate patents for a TROP2 ADC."))
    assert stable1 == stable2
    assert stable1  # non-empty


# ── 2. stable prefix is byte-identical across different uploaded files ─────


def test_stable_prefix_byte_identical_across_different_uploaded_files():
    raw1 = _raw(
        files=[{
            "file_id": "f1",
            "original_filename": "a.pdb",
            "storage_path": "/store/x/a.pdb",
            "content_type": "chemical/x-pdb",
            "size_bytes": 2048,
        }]
    )
    raw2 = _raw(
        files=[
            {
                "file_id": "f2",
                "original_filename": "b.fasta",
                "storage_path": "/store/y/b.fasta",
                "content_type": "text/x-fasta",
                "sha256": "deadbeef" * 4,
                "size_bytes": 999,
            },
            {
                "file_id": "f3",
                "original_filename": "c.pdb",
                "storage_path": "/store/z/c.pdb",
            },
        ]
    )
    stable1, _ = _step2_sections(raw1)
    stable2, _ = _step2_sections(raw2)
    assert stable1 == stable2


# ── 3. stable prefix is byte-identical: normal turn vs clarification turn ──


def test_stable_prefix_byte_identical_normal_vs_clarification_turn():
    raw_normal = _raw()
    raw_clarify = _raw(
        ctx={
            "previous_task_intent": {
                "primary_intent": "new_adc_design",
                "secondary_intents": [],
            },
            "previous_canonical_query": "Design a new ADC (target unspecified).",
            "previous_missing_slots": [
                {"slot_name": "target_or_antigen", "severity": "blocking"}
            ],
            "previous_clarification_requests": [
                {"request_id": "r1", "slot_name": "target_or_antigen",
                 "question": "What target?"}
            ],
            "clarification_answers": [
                {"request_id": "r1", "slot_name": "target_or_antigen",
                 "slot_category": "target", "answer_text": "HER2",
                 "answered_at": "2026-06-30T00:01:00Z"}
            ],
        }
    )
    stable_normal, _ = _step2_sections(raw_normal)
    stable_clarify, _ = _step2_sections(raw_clarify)
    assert stable_normal == stable_clarify


# ── 4. raw_user_query appears only in the dynamic suffix ───────────────────


def test_raw_user_query_only_in_dynamic_suffix_after_stable_prefix():
    raw = _raw(query="UNIQUE_QUERY_SENTINEL_XYZ")
    stable, dynamic = _step2_sections(raw)
    full = _step2_full_prompt(raw)

    assert "UNIQUE_QUERY_SENTINEL_XYZ" not in stable
    assert "UNIQUE_QUERY_SENTINEL_XYZ" in dynamic
    assert full == stable + dynamic
    assert full.index("UNIQUE_QUERY_SENTINEL_XYZ") >= len(stable)


# ── 5. uploaded file_id/original_filename/content_type/sha256/size_bytes ───
#      appear only in the dynamic suffix, after the stable prefix.


def test_uploaded_file_metadata_only_in_dynamic_suffix():
    raw = _raw(
        files=[{
            "file_id": "f_sentinel_001",
            "original_filename": "sentinel_name.pdb",
            "storage_path": "/secret/storage/sentinel_name.pdb",
            "content_type": "application/x-sentinel-pdb-marker",
            "sha256": "sentinelsha256marker",
            "size_bytes": 918273,
        }]
    )
    stable, dynamic = _step2_sections(raw)
    full = _step2_full_prompt(raw)

    for needle in (
        "f_sentinel_001",
        "sentinel_name.pdb",
        "application/x-sentinel-pdb-marker",
        "sentinelsha256marker",
        "918273",
    ):
        assert needle not in stable, needle
        assert needle in dynamic, needle
        assert full.index(needle) >= len(stable)

    # storage_path itself must never appear anywhere in the rendered prompt.
    assert "/secret/storage/" not in stable
    assert "/secret/storage/" not in dynamic
    assert "storage_path" not in dynamic


# ── 6. user_provided_context / clarification_answers only in dynamic suffix ─


def test_user_provided_context_and_clarification_answers_only_in_dynamic_suffix():
    raw = _raw(
        ctx={
            "constraints_text": "SENTINEL_CONSTRAINT_TEXT",
            "clarification_answers": [
                {"request_id": "r1", "slot_name": "payload",
                 "slot_category": "payload", "answer_text": "SENTINEL_ANSWER_TEXT",
                 "answered_at": "t"}
            ],
        }
    )
    stable, dynamic = _step2_sections(raw)
    assert "SENTINEL_CONSTRAINT_TEXT" not in stable
    assert "SENTINEL_CONSTRAINT_TEXT" in dynamic
    assert "SENTINEL_ANSWER_TEXT" not in stable
    assert "SENTINEL_ANSWER_TEXT" in dynamic


# ── 7. stable prefix excludes run_id / raw_user_query / file identifiers / ──
#      storage_path / timestamps / raw sequence / API key.


def test_stable_prefix_excludes_run_bookkeeping_identifiers_and_secrets():
    heavy_sequence = "EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGK"
    raw = _raw(
        run_id="run_SENTINEL_RUNID_marker",
        query=f"Design an ADC using antibody sequence {heavy_sequence}",
        files=[{
            "file_id": "f_x",
            "original_filename": "SENTINEL_FILENAME_marker.pdb",
            "storage_path": "/SENTINEL/STORAGE/PATH/marker",
        }],
    )
    stable, _ = _step2_sections(raw)
    for forbidden in (
        "run_SENTINEL_RUNID_marker",
        "SENTINEL_FILENAME_marker.pdb",
        "/SENTINEL/STORAGE/PATH/marker",
        "storage_path",
        "run_id",
        "created_at",
        "parsed_at",
        "art_cache_layout",  # artifact_id
        "reg_cache_layout",  # run_artifact_registry_id
        heavy_sequence,
        "api_key",
        "sk-",
    ):
        assert forbidden not in stable, forbidden


# ── 8. Existing Step 2 behavior is unchanged by the prompt-layout refactor ──


def test_step2_mock_provider_behavior_unchanged():
    """The MockLLMProvider path never renders `build_json_prompt` (it reads
    the schema dict directly), so it must be completely unaffected by the
    prompt-layout refactor."""
    agent = SupervisorAgent(llm=MockLLMProvider())
    sq = agent.parse_raw_to_structured_query(
        _raw(
            ctx={
                "target_or_antigen_text": "HER2",
                "candidate_text": "Trastuzumab analog",
                "payload_linker_text": "vc-MMAE",
            }
        )
    )
    assert sq.task_intent.task_type == "adc_design"
    assert sq.mentioned_entities.target_or_antigen_text == "HER2"


def test_build_json_prompt_still_equals_stable_plus_dynamic_concatenation():
    raw = _raw()
    stable, dynamic = _step2_sections(raw)
    assert _step2_full_prompt(raw) == stable + dynamic


def test_build_json_prompt_sections_unaffected_for_non_structured_query_tasks():
    """Other tasks (Step 5/6 selectors, etc.) must render exactly like
    before: the full schema dict dumped into the dynamic suffix, unaffected
    by the Step 2-only trimming."""
    schema = {
        "task": "tool_selection_stage_1",
        "compact_catalog": [{"tool_name": "DrugProps_pains_filter"}],
        "context": {"signals": {"has_smiles": True}},
    }
    stable, dynamic = build_json_prompt_sections(prompt="pick a tool", schema=schema, system=None)
    assert "DrugProps_pains_filter" in dynamic
    assert "has_smiles" in dynamic
    full = build_json_prompt(prompt="pick a tool", schema=schema, system=None)
    assert full == stable + dynamic
