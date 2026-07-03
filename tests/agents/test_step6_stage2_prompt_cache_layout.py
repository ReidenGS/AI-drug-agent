"""Step 6 Stage 2 prompt cache-friendly layout tests.

Step 6 Stage 2's real-provider mapping prompt (`build_json_prompt`) is
composed of a **stable prefix** (the existing English Stage 2 rules system
prompt, the fixed JSON-only instruction, the per-task shape hint, the fixed
user task text, and the SELECTED tools' official `full_schema` block) followed
by a **dynamic suffix** (the candidate/run-specific context — `candidate_id`,
`candidate_available_fields`, and the per-tool Stage-1 `selection_reason`).
The split is exposed via `json_task_validation.build_json_prompt_sections`,
driven by the exact payload the production selector builds
(`step_06_schema_mapping_selector.build_step6_stage2_payload`).

Goal: the stable prefix must be byte-identical for the same selected tool
schema set across candidates (Stage 2 only receives Stage-1-selected tools;
same set → same official schemas → cacheable prefix). All candidate/run data
(candidate_id, field_refs, available_fields, selection reasons) lives only in
the dynamic suffix. Stage 2 renders ONLY the selected tools, never the full
catalog. Raw sequences / PDB / CDR3 / storage_path never appear anywhere
(Stage 2 maps to LLM-safe digests only).

These tests do not change Stage 2 mapping. The selector/mock reads the
`schema` dict directly, so the text-only relocation cannot change mapping,
can_invoke, missing_required_fields, or audit (asserted by the existing
Step 6 tests, which still pass).
"""

from __future__ import annotations

import json

from app.agents.step_06_available_fields import project_candidate_available_fields
from app.agents.step_06_schema_mapping_selector import (
    STEP6_STAGE2_SCHEMA_MAPPING_SYSTEM_PROMPT,
    STEP6_STAGE2_SCHEMA_MAPPING_USER_PROMPT,
    build_step6_stage2_payload,
)
from app.llm.json_task_validation import (
    build_json_prompt,
    build_json_prompt_sections,
)


RAW_HEAVY = "EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGK"
SMILES = "CCO"
PDB_PATH = "adc_pilot/runs/run_x/inputs/complex.pdb"

# Deterministic official schemas (no candidate data) for two tools.
_SMILES_SCHEMA = {
    "type": "object",
    "properties": {"smiles": {"type": "string"}},
    "required": ["smiles"],
}
_SWISSADME_SCHEMA = {
    "type": "object",
    "properties": {
        "operation": {"type": "string", "enum": ["calculate_adme"]},
        "smiles": {"type": "string"},
    },
    "required": ["operation", "smiles"],
}


def _material(material_id: str, material_type: str, value: str, **extra):
    return {
        "material_id": material_id,
        "material_type": material_type,
        "value": value,
        "value_format": extra.pop("value_format", None),
        "role": extra.pop("role", None),
        "role_status": extra.pop("role_status", "unknown"),
    }


def _candidate(*, candidate_id="cand_a", materials=None, identifiers=None):
    return {
        "candidate_id": candidate_id,
        "candidate_label": "fixture",
        "candidate_type": "adc_construct",
        "materials": materials or [],
        "identifiers": identifiers or [],
    }


def _stage2_item(tool_name: str, schema: dict, selection_reason: str = "sel"):
    """One (stage1_entry, official_schema) pair, as the selector builds."""
    return ({"tool_name": tool_name, "selection_reason": selection_reason}, schema)


def _sections_for(candidate: dict, stage2_items):
    proj = project_candidate_available_fields(candidate)
    payload = build_step6_stage2_payload(
        agent_name="developability_agent",
        step_id="step_06",
        candidate_id=candidate["candidate_id"],
        stage2_items=stage2_items,
        available_fields=proj.available_fields,
    )
    stable, dynamic = build_json_prompt_sections(
        prompt=STEP6_STAGE2_SCHEMA_MAPPING_USER_PROMPT,
        schema=payload,
        system=STEP6_STAGE2_SCHEMA_MAPPING_SYSTEM_PROMPT,
    )
    full = build_json_prompt(
        prompt=STEP6_STAGE2_SCHEMA_MAPPING_USER_PROMPT,
        schema=payload,
        system=STEP6_STAGE2_SCHEMA_MAPPING_SYSTEM_PROMPT,
    )
    return stable, dynamic, full


def _stable_tool_names(stable: str) -> list[str]:
    block = stable.split("Input schema/context JSON:\n", 1)[1]
    parsed = json.loads(block)
    return [t["tool_name"] for t in parsed["tools"]]


# ── 1. same selected tool schema set, different candidate -> stable identical ─


def test_stable_prefix_byte_identical_same_tool_set_different_candidates():
    items = [_stage2_item("DrugProps_pains_filter", _SMILES_SCHEMA)]
    cand_a = _candidate(
        candidate_id="cand_alpha",
        materials=[_material("m_a", "payload_smiles", "CCO")],
    )
    cand_b = _candidate(
        candidate_id="cand_beta",
        materials=[_material("m_b", "payload_smiles", "CCCCO")],
    )
    # Different Stage-1 selection_reason text too — must not affect stable.
    stable_a, _, _ = _sections_for(
        cand_a, [_stage2_item("DrugProps_pains_filter", _SMILES_SCHEMA, "reason A")]
    )
    stable_b, _, _ = _sections_for(
        cand_b, [_stage2_item("DrugProps_pains_filter", _SMILES_SCHEMA, "reason B")]
    )
    assert stable_a == stable_b
    assert stable_a
    del items  # silence unused


# ── 2. different selected tool set -> stable differs, only selected tools ──


def test_different_selected_tool_set_changes_stable_and_only_selected_tools():
    cand = _candidate(materials=[_material("m", "payload_smiles", SMILES)])
    stable_one, _, _ = _sections_for(
        cand, [_stage2_item("DrugProps_pains_filter", _SMILES_SCHEMA)]
    )
    stable_two, _, _ = _sections_for(
        cand,
        [
            _stage2_item("DrugProps_pains_filter", _SMILES_SCHEMA),
            _stage2_item("SwissADME_calculate_adme", _SWISSADME_SCHEMA),
        ],
    )
    assert stable_one != stable_two
    # Only the selected tools are present — no full catalog.
    assert _stable_tool_names(stable_one) == ["DrugProps_pains_filter"]
    assert _stable_tool_names(stable_two) == [
        "DrugProps_pains_filter",
        "SwissADME_calculate_adme",
    ]


# ── 3. candidate field_refs / candidate_id / reasons only in dynamic suffix ─


def test_candidate_specific_data_only_in_dynamic_suffix():
    cand = _candidate(
        candidate_id="cand_SENTINEL_ID",
        materials=[_material("mat_SENTINEL", "payload_smiles", SMILES)],
    )
    stable, dynamic, full = _sections_for(
        cand,
        [_stage2_item("DrugProps_pains_filter", _SMILES_SCHEMA, "SENTINEL_REASON")],
    )
    field_ref = "candidate:cand_SENTINEL_ID:material:mat_SENTINEL:value"
    for needle in (
        "cand_SENTINEL_ID",   # candidate_id
        field_ref,            # candidate field_ref
        "SENTINEL_REASON",    # per-tool Stage-1 selection_reason
    ):
        assert needle not in stable, needle
        assert needle in dynamic, needle
        assert full.index(needle) >= len(stable)
    assert full == stable + dynamic


# ── 4. raw sequence/PDB/CDR3/storage_path absent in both sections ──────────


def test_no_raw_sequence_pdb_or_storage_path_anywhere():
    cand = _candidate(
        candidate_id="cand_raw",
        materials=[
            _material("m_seq", "antibody_heavy_chain_sequence", RAW_HEAVY),
            _material("m_struct", "structure_ref", PDB_PATH, value_format="pdb"),
        ],
    )
    stable, dynamic, full = _sections_for(
        cand, [_stage2_item("PROSITE_scan_sequence", _SMILES_SCHEMA)]
    )
    for forbidden in (RAW_HEAVY, PDB_PATH, "complex.pdb"):
        assert forbidden not in stable, forbidden
        assert forbidden not in dynamic, forbidden
        assert forbidden not in full, forbidden


# ── 5. selected tool full_schema appears in the stable prefix ─────────────


def test_selected_tool_full_schema_in_stable_prefix():
    cand = _candidate(materials=[_material("m", "payload_smiles", SMILES)])
    stable, _, _ = _sections_for(
        cand, [_stage2_item("SwissADME_calculate_adme", _SWISSADME_SCHEMA)]
    )
    block = stable.split("Input schema/context JSON:\n", 1)[1]
    parsed = json.loads(block)
    tool = parsed["tools"][0]
    assert tool["tool_name"] == "SwissADME_calculate_adme"
    # Official schema (properties / required / enum literal) is present.
    assert tool["full_schema"]["required"] == ["operation", "smiles"]
    assert tool["full_schema"]["properties"]["operation"]["enum"] == ["calculate_adme"]
    # selection_reason (candidate-specific) is NOT in the stable tools block.
    assert "selection_reason" not in tool


# ── 6. candidate_available_fields only in dynamic suffix ──────────────────


def test_candidate_available_fields_only_in_dynamic_suffix():
    cand = _candidate(materials=[_material("m", "payload_smiles", SMILES)])
    stable, dynamic, _ = _sections_for(
        cand, [_stage2_item("DrugProps_pains_filter", _SMILES_SCHEMA)]
    )
    # The stable SCHEMA block (not the rules text, which legitimately names
    # the field) must not carry candidate_available_fields as a key/data...
    stable_block = json.loads(stable.split("Input schema/context JSON:\n", 1)[1])
    assert "candidate_available_fields" not in stable_block
    # ...and the dynamic block carries the actual available-fields data.
    dynamic_block = json.loads(
        dynamic.split("Candidate/run-specific context JSON:\n", 1)[1]
    )
    assert "candidate_available_fields" in dynamic_block
    assert isinstance(dynamic_block["candidate_available_fields"], list)


# ── 7. stable tool block deterministic ordering + key-sorted ──────────────


def test_stable_tool_block_is_deterministic_and_key_sorted():
    cand = _candidate(materials=[_material("m", "payload_smiles", SMILES)])
    # Selection order reversed vs sorted; stable block must still be sorted.
    stable, _, _ = _sections_for(
        cand,
        [
            _stage2_item("SwissADME_calculate_adme", _SWISSADME_SCHEMA),
            _stage2_item("DrugProps_pains_filter", _SMILES_SCHEMA),
        ],
    )
    parsed = json.loads(stable.split("Input schema/context JSON:\n", 1)[1])
    # Tools sorted by tool_name regardless of selection order.
    assert [t["tool_name"] for t in parsed["tools"]] == [
        "DrugProps_pains_filter",
        "SwissADME_calculate_adme",
    ]
    # Top-level keys and each tool's keys are sort_keys=True ordered; no
    # candidate/run fields in the stable block.
    assert list(parsed.keys()) == sorted(parsed.keys())
    assert set(parsed.keys()) == {"agent_name", "step_id", "task", "tools"}
    for t in parsed["tools"]:
        assert list(t.keys()) == sorted(t.keys())
    assert "candidate_id" not in parsed
    # Prompt text stays English (no accidental Chinese translation).
    assert not any("一" <= ch <= "鿿" for ch in stable)


# ── 8. non-Step-6-Stage-2 payloads (provider shape) are not split ─────────


def test_provider_shape_stage2_payload_is_not_split():
    schema = {"task": "step6_schema_mapping_stage_2"}
    stable, dynamic = build_json_prompt_sections(
        prompt="map", schema=schema, system="sys",
    )
    assert dynamic.startswith("Input schema/context JSON:\n")
    assert "Candidate/run-specific context JSON:" not in stable
    assert "Candidate/run-specific context JSON:" not in dynamic
    full = build_json_prompt(prompt="map", schema=schema, system="sys")
    assert full == stable + dynamic
