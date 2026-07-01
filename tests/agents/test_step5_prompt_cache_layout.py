"""Step 5 prompt cache-friendly layout tests.

Step 5's real-provider Stage-1 selection prompt (`build_json_prompt`) is
composed of a **stable prefix** (the shared selection system prompt + the
Step 5 addendum, the fixed JSON-only instruction, the per-task shape hint,
the fixed user task text, and the stable tool catalog + rules metadata)
followed by a **dynamic suffix** (the candidate/run-specific context —
`context.candidate` + `context.signals`). The split is exposed via
`json_task_validation.build_json_prompt_sections`, driven by the exact
payload the production policy builds
(`step_05_selection_policy.build_step5_stage1_payload`).

Goal: the stable prefix must be byte-identical for the same Step 5
capability group across two different candidates (same eligible tools →
same catalog → cacheable prefix). All candidate/run-specific data
(candidate_id, material/identifier values, SMILES, IDs, data gaps, the raw
query excerpt) must live only in the dynamic suffix, strictly after the
stable prefix. run_id / material_id / storage_path / raw sequence / raw
CDR3 must never appear in the stable prefix.

These tests do not change Step 5 business logic — they characterize the
prompt-layout refactor only. The MockLLMProvider / selection policy read
the `schema` dict directly, so this text-only relocation cannot change
which tools are eligible / selected / skipped (asserted separately by the
existing Step 5 selection tests, which still pass).
"""

from __future__ import annotations

import json

from app.agents.step_05_enrichment_registry import (
    STEP_05_CAPABILITY_REGISTRY,
    plan_enrichment_for_record,
)
from app.agents.step_05_selection_policy import (
    STEP_05_SELECTION_SYSTEM_PROMPT,
    _build_compact_catalog,
    _build_signals,
    _compact_candidate_context,
    build_step5_stage1_payload,
)
from app.agents.tool_selection_policy import SELECTION_STAGE1_USER_PROMPT
from app.llm.json_task_validation import (
    build_json_prompt,
    build_json_prompt_sections,
)
from app.schemas.step_05_candidate_context_table import (
    CandidateRecord,
    Identifier,
    Material,
)


# ── helpers ─────────────────────────────────────────────────────────────────


def _mat(material_type: str, value: str, role: str | None = None,
         material_id: str | None = None) -> Material:
    return Material(
        material_id=material_id or f"mat_{material_type}",
        material_type=material_type,
        value=value,
        role=role,
    )


def _ident(id_type: str, value: str) -> Identifier:
    return Identifier(id_type=id_type, id_value=value)


def _record(
    *,
    candidate_id: str = "cand_alpha",
    candidate_label: str = "step5 cache layout candidate",
    candidate_type: str = "compound_component",
    materials: list[Material] | None = None,
    identifiers: list[Identifier] | None = None,
) -> CandidateRecord:
    return CandidateRecord(
        candidate_id=candidate_id,
        candidate_label=candidate_label,
        candidate_type=candidate_type,  # type: ignore[arg-type]
        materials=materials or [],
        identifiers=identifiers or [],
        candidate_role="user_provided_candidate",
        is_generated_candidate=False,
        context_status="partial",
    )


# Compound-lane scope so two candidates land in the same Step 5 capability
# group (same eligible tools → same catalog).
_COMPOUND_SCOPE = [
    "ChEMBL_search_molecules",
    "ChEMBL_search_substructure",
    "ChEMBL_get_molecule",
]


def _sections_for(record: CandidateRecord, *, scope=None,
                  raw_user_query: str = "") -> tuple[str, str]:
    """Build the Step 5 Stage-1 prompt sections exactly as the production
    policy would (catalog + signals + compact candidate context), then split
    them via `build_json_prompt_sections`."""
    plans = plan_enrichment_for_record(
        record,
        scoped_tools=scope or _COMPOUND_SCOPE,
        candidate_category=record.candidate_type,
        registry=STEP_05_CAPABILITY_REGISTRY,
    )
    catalog = _build_compact_catalog(plans)
    signals = _build_signals(plans)
    record_context = _compact_candidate_context(record, raw_user_query)
    payload = build_step5_stage1_payload(
        catalog=catalog, signals=signals, record_context=record_context,
    )
    return build_json_prompt_sections(
        prompt=SELECTION_STAGE1_USER_PROMPT,
        schema=payload,
        system=STEP_05_SELECTION_SYSTEM_PROMPT,
    )


def _full_prompt_for(record: CandidateRecord, *, scope=None,
                     raw_user_query: str = "") -> str:
    plans = plan_enrichment_for_record(
        record,
        scoped_tools=scope or _COMPOUND_SCOPE,
        candidate_category=record.candidate_type,
        registry=STEP_05_CAPABILITY_REGISTRY,
    )
    payload = build_step5_stage1_payload(
        catalog=_build_compact_catalog(plans),
        signals=_build_signals(plans),
        record_context=_compact_candidate_context(record, raw_user_query),
    )
    return build_json_prompt(
        prompt=SELECTION_STAGE1_USER_PROMPT,
        schema=payload,
        system=STEP_05_SELECTION_SYSTEM_PROMPT,
    )


# ── 1. same capability group, two different candidates -> stable identical ─


def test_stable_prefix_byte_identical_same_capability_group_two_candidates():
    rec_a = _record(
        candidate_id="cand_alpha",
        candidate_label="alpha payload",
        materials=[_mat("payload_name", "monomethyl auristatin E", "payload",
                        material_id="mat_a1")],
        identifiers=[_ident("chembl_id", "CHEMBL111")],
    )
    rec_b = _record(
        candidate_id="cand_beta",
        candidate_label="beta payload",
        materials=[_mat("payload_name", "deruxtecan", "payload",
                        material_id="mat_b1")],
        identifiers=[_ident("chembl_id", "CHEMBL222")],
    )
    stable_a, _ = _sections_for(rec_a)
    stable_b, _ = _sections_for(rec_b)
    assert stable_a == stable_b
    assert stable_a  # non-empty


# ── 2. dynamic candidate materials/IDs appear only in the dynamic suffix ───


def test_dynamic_candidate_values_only_in_dynamic_suffix():
    # Step 5's compact candidate context exposes candidate_id, identifier
    # id_values, material TYPES, and a truncated raw query excerpt — but NOT
    # raw material values. The dynamic sentinels below are exactly the
    # candidate/run-specific fields the production prompt already carries.
    rec = _record(
        candidate_id="cand_SENTINEL_ID",
        materials=[_mat("payload_smiles", "SENTINELSMILESXYZ", "payload",
                        material_id="mat_SENTINEL")],
        identifiers=[_ident("chembl_id", "CHEMBLSENTINEL42")],
    )
    stable, dynamic = _sections_for(rec, raw_user_query="SENTINEL_QUERY_TEXT")
    full = _full_prompt_for(rec, raw_user_query="SENTINEL_QUERY_TEXT")

    for needle in (
        "cand_SENTINEL_ID",       # candidate_id
        "CHEMBLSENTINEL42",       # identifier id_value
        "SENTINEL_QUERY_TEXT",    # raw_user_query_excerpt
    ):
        assert needle not in stable, needle
        assert needle in dynamic, needle
        assert full.index(needle) >= len(stable)
    # Raw material VALUES are never sent to the LLM (existing Step 5 privacy
    # rule) — the SMILES value appears in neither section.
    assert "SENTINELSMILESXYZ" not in stable
    assert "SENTINELSMILESXYZ" not in dynamic
    # full prompt is exactly the two sections concatenated.
    assert full == stable + dynamic


# ── 3. stable prefix excludes run/candidate/material identifiers & secrets ─


def test_stable_prefix_excludes_candidate_and_run_specific_data():
    raw_cdr3 = "CARDYGSSYW"
    raw_heavy = "EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGK"
    rec = _record(
        candidate_id="cand_RUNSPECIFIC",
        candidate_label="do-not-leak-label",
        materials=[
            _mat("payload_smiles", raw_heavy, "payload",
                 material_id="mat_SECRET_ID"),
        ],
        identifiers=[_ident("chembl_id", "CHEMBL_RUN_SECRET")],
    )
    stable, _ = _sections_for(
        rec, raw_user_query=f"design against {raw_cdr3} using {raw_heavy}"
    )
    for forbidden in (
        "cand_RUNSPECIFIC",
        "mat_SECRET_ID",
        "CHEMBL_RUN_SECRET",
        "do-not-leak-label",
        raw_cdr3,
        raw_heavy,
        "candidate_id",
        "material_id",
        "raw_user_query_excerpt",
        "storage_path",
        "run_id",
        "api_key",
        "sk-",
    ):
        assert forbidden not in stable, forbidden


# ── 4. stable prefix contains the English rules + stable tool metadata ─────


def test_stable_prefix_contains_english_rules_and_stable_tool_metadata():
    rec = _record(
        materials=[_mat("payload_name", "monomethyl auristatin E", "payload")],
        identifiers=[_ident("chembl_id", "CHEMBL111")],
    )
    stable, _ = _sections_for(rec)
    # Existing English selection rules (reused verbatim, not rewritten).
    assert "You are choosing MCP tools for the CURRENT agent / step ONLY." in stable
    assert (
        "You are selecting Step 5 context-enrichment tools for candidate "
        "material organization." in stable
    )
    assert "Use only exact `tool_name` values from `compact_catalog`." in stable
    # Stable, registry/ToolUniverse-derived tool metadata (catalog).
    assert "compact_catalog" in stable
    assert "ChEMBL_get_molecule" in stable
    assert "ChEMBL_search_molecules" in stable
    assert "capability_tags" in stable
    assert "coarse_input_requirements" in stable
    # Prompt text stays English — no CJK characters (guards against any
    # accidental translation of the production prompt into Chinese).
    assert not any("一" <= ch <= "鿿" for ch in stable)


# ── 5. tool metadata in the stable prefix is deterministically key-sorted ──


def test_stable_prefix_tool_metadata_is_key_sorted_and_deterministic():
    rec = _record(
        materials=[
            _mat("payload_name", "monomethyl auristatin E", "payload"),
            _mat("payload_smiles", "CCO", "payload"),
        ],
        identifiers=[_ident("chembl_id", "CHEMBL999")],
    )
    stable, _ = _sections_for(rec)
    # The stable schema block is rendered with sort_keys=True; parse it back
    # and confirm the top-level keys are emitted in sorted order and the
    # catalog is sorted by tool_name.
    marker = "Input schema/context JSON:\n"
    assert marker in stable
    block = stable.split(marker, 1)[1]
    parsed = json.loads(block)
    assert list(parsed.keys()) == sorted(parsed.keys())
    names = [e["tool_name"] for e in parsed["compact_catalog"]]
    assert names == sorted(names)
    # Each catalog entry's own keys are also sorted (sort_keys=True).
    for entry in parsed["compact_catalog"]:
        assert list(entry.keys()) == sorted(entry.keys())
    # Candidate/run-specific keys are NOT keys of the stable context block
    # (the word "candidate" still legitimately appears inside the fixed note
    # text, so check structural key membership, not substring).
    stable_context = parsed.get("context", {})
    assert "candidate" not in stable_context
    assert "signals" not in stable_context


# ── 6. build_json_prompt still equals stable + dynamic concatenation ───────


def test_full_prompt_equals_stable_plus_dynamic():
    rec = _record(
        materials=[_mat("payload_name", "monomethyl auristatin E", "payload")],
        identifiers=[_ident("chembl_id", "CHEMBL111")],
    )
    stable, dynamic = _sections_for(rec, raw_user_query="q")
    assert _full_prompt_for(rec, raw_user_query="q") == stable + dynamic


# ── 7. non-Step-5 tool_selection_stage_1 payloads are unaffected ───────────


def test_non_step5_stage1_payload_is_not_split():
    """Step 9/13/14 single-lane `tool_selection_stage_1` payloads carry no
    `context.candidate`, so the whole schema stays in the single dynamic
    block exactly as before — no trailing candidate block is added."""
    schema = {
        "task": "tool_selection_stage_1",
        "agent_name": "evidence_agent",
        "step_id": "step_13",
        "compact_catalog": [{"tool_name": "EuropePMC_search_articles"}],
        "context": {"signals": {"target_literature_query": True}, "note": "n"},
    }
    stable, dynamic = build_json_prompt_sections(
        prompt="pick", schema=schema, system="sys",
    )
    # No split: the dynamic suffix is the single "Input schema/context JSON"
    # block and there is no trailing candidate block.
    assert dynamic.startswith("Input schema/context JSON:\n")
    assert "Candidate/run-specific context JSON:" not in stable
    assert "Candidate/run-specific context JSON:" not in dynamic
    assert "EuropePMC_search_articles" in dynamic
    full = build_json_prompt(prompt="pick", schema=schema, system="sys")
    assert full == stable + dynamic
