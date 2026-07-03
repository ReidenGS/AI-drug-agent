"""Step 6 Stage 1 prompt cache-friendly layout tests.

Step 6 Stage 1's real-provider selection prompt (`build_json_prompt`) is
composed of a **stable prefix** (the existing English Stage 1 rules system
prompt, the fixed JSON-only instruction, the per-task shape hint, the fixed
user task text, and the program-disclosed compact catalog block) followed
by a **dynamic suffix** (the candidate/run-specific context —
`candidate_id`, `candidate_modality_summary`, `candidate_available_fields`,
`user_query_summary`, `disclosure_tags`). The split is exposed via
`json_task_validation.build_json_prompt_sections`, driven by the exact
payload the production selector builds
(`step_06_schema_mapping_selector.build_step6_stage1_payload`), whose
catalog comes from the unchanged progressive-disclosure result.

Goal: the stable prefix must be byte-identical for the same disclosed
category catalog across candidates (progressive disclosure picks the
catalog BEFORE Stage 1 → same disclosed set → cacheable prefix). All
candidate/run-specific data (candidate_id, field_refs, modality summary
values, user query summary, disclosure tags) must live only in the dynamic
suffix. Raw sequences / PDB / CDR3 / SMILES values / storage_path never
appear anywhere (Stage 1 only ever sees LLM-safe digests).

These tests do not change Step 6 business logic. The selector reads the
`schema` dict directly, so the text-only relocation cannot change
disclosure, selection, or audit (asserted by the existing Step 6 tests,
which still pass).
"""

from __future__ import annotations

import json

from app.agents.step_06_available_fields import project_candidate_available_fields
from app.agents.step_06_capability_registry import STEP_06_CAPABILITY_REGISTRY
from app.agents.step_06_schema_mapping_selector import (
    STEP6_STAGE1_SCHEMA_MAPPING_SYSTEM_PROMPT,
    STEP6_STAGE1_SCHEMA_MAPPING_USER_PROMPT,
    build_step6_stage1_catalog,
    build_step6_stage1_payload,
    disclose_step6_tools,
)
from app.llm.json_task_validation import (
    build_json_prompt,
    build_json_prompt_sections,
)


SCOPED = sorted({cap.tool_name for cap in STEP_06_CAPABILITY_REGISTRY})
RAW_HEAVY = "EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGK"
RAW_LIGHT = "DIQMTQSPSSLSASVGDRVTITCRASQDVNTAVAWYQQKPGKAPK"
SMILES = "CCO"
PDB_PATH = "adc_pilot/runs/run_x/inputs/complex.pdb"


class _MCP:
    def __init__(self, tools: list[str]):
        self.tools = tools

    def list_tools(self, *, agent_name: str, step_id: str) -> list[str]:
        return list(self.tools)


def _material(material_id: str, material_type: str, value: str, **extra):
    return {
        "material_id": material_id,
        "material_type": material_type,
        "value": value,
        "value_format": extra.pop("value_format", None),
        "role": extra.pop("role", None),
        "role_status": extra.pop("role_status", "unknown"),
    }


def _identifier(id_type: str, value: str):
    return {"id_type": id_type, "id_value": value, "source_ids": [], "confidence": 0.9}


def _candidate(*, candidate_id="cand_a", materials=None, identifiers=None):
    return {
        "candidate_id": candidate_id,
        "candidate_label": "fixture",
        "candidate_type": "adc_construct",
        "materials": materials or [],
        "identifiers": identifiers or [],
    }


def _sections_for(candidate: dict, *, scope=None, user_query_summary=""):
    """Build the Step 6 Stage-1 prompt sections exactly as the production
    selector would, then split them via `build_json_prompt_sections`."""
    proj = project_candidate_available_fields(candidate)
    mcp = _MCP(list(scope if scope is not None else SCOPED))
    scoped = set(mcp.list_tools(agent_name="developability_agent", step_id="step_06"))
    disclosure = disclose_step6_tools(
        scoped_tool_names=scoped,
        modality_summary=proj.modality_summary,
        available_fields=proj.available_fields,
        user_query_summary=user_query_summary,
    )
    catalog = build_step6_stage1_catalog(
        mcp_client=mcp,
        agent_name="developability_agent",
        step_id="step_06",
        disclosure=disclosure,
    )
    payload = build_step6_stage1_payload(
        agent_name="developability_agent",
        step_id="step_06",
        candidate_id=candidate["candidate_id"],
        catalog=catalog,
        modality_summary=proj.modality_summary,
        available_fields=proj.available_fields,
        user_query_summary=user_query_summary,
        disclosure_tags=disclosure.disclosure_tags,
    )
    stable, dynamic = build_json_prompt_sections(
        prompt=STEP6_STAGE1_SCHEMA_MAPPING_USER_PROMPT,
        schema=payload,
        system=STEP6_STAGE1_SCHEMA_MAPPING_SYSTEM_PROMPT,
    )
    full = build_json_prompt(
        prompt=STEP6_STAGE1_SCHEMA_MAPPING_USER_PROMPT,
        schema=payload,
        system=STEP6_STAGE1_SCHEMA_MAPPING_SYSTEM_PROMPT,
    )
    return stable, dynamic, full, disclosure


def _stable_catalog_names(stable: str) -> list[str]:
    marker = "Input schema/context JSON:\n"
    block = stable.split(marker, 1)[1]
    parsed = json.loads(block)
    return [e["tool_name"] for e in parsed["compact_catalog"]]


# ── 1. same Stage 1 category, different candidate fields -> stable identical ─


def test_stable_prefix_byte_identical_same_category_different_fields():
    # Two small-molecule candidates with different SMILES material ids/values
    # land in the same disclosed category → identical disclosed catalog.
    cand_a = _candidate(
        candidate_id="cand_alpha",
        materials=[_material("m_a", "payload_smiles", "CCO")],
    )
    cand_b = _candidate(
        candidate_id="cand_beta",
        materials=[_material("m_b", "payload_smiles", "CCCCO")],
        identifiers=[_identifier("chembl_id", "CHEMBL777")],
    )
    stable_a, _, _, disc_a = _sections_for(cand_a)
    stable_b, _, _, disc_b = _sections_for(cand_b)
    # Same disclosed category catalog...
    assert disc_a.disclosed_tool_names == disc_b.disclosed_tool_names
    # ...means byte-identical stable prefix.
    assert stable_a == stable_b
    assert stable_a


# ── 2. candidate field_refs / candidate_id / user_query only in dynamic ────


def test_candidate_specific_data_only_in_dynamic_suffix():
    cand = _candidate(
        candidate_id="cand_SENTINEL_ID",
        materials=[_material("mat_SENTINEL", "payload_smiles", SMILES)],
        identifiers=[_identifier("chembl_id", "CHEMBLSENTINEL42")],
    )
    stable, dynamic, full, _ = _sections_for(
        cand, user_query_summary="SENTINEL_QUERY_SUMMARY"
    )
    field_ref = "candidate:cand_SENTINEL_ID:material:mat_SENTINEL:value"
    for needle in (
        "cand_SENTINEL_ID",          # candidate_id
        field_ref,                    # candidate field_ref
        "SENTINEL_QUERY_SUMMARY",     # user_query_summary
    ):
        assert needle not in stable, needle
        assert needle in dynamic, needle
        assert full.index(needle) >= len(stable)
    # Raw identifier VALUES are never sent to the LLM (only sha256 digests),
    # so the raw ChEMBL id appears in neither section.
    assert "CHEMBLSENTINEL42" not in stable
    assert "CHEMBLSENTINEL42" not in dynamic
    assert full == stable + dynamic


# ── 3. heavy/light sequence candidate: sequence tools in, small-mol hidden ─


def test_sequence_candidate_stable_catalog_hides_small_molecule_tools():
    cand = _candidate(
        candidate_id="cand_seq",
        materials=[
            _material("m_h", "antibody_heavy_chain_sequence", RAW_HEAVY),
            _material("m_l", "antibody_light_chain_sequence", RAW_LIGHT),
        ],
    )
    stable, _, _, disclosure = _sections_for(cand)
    names = set(_stable_catalog_names(stable))
    # Sequence-relevant tools are disclosed.
    assert "PROSITE_scan_sequence" in names
    # Small-molecule / ADMET tools are hidden by progressive disclosure.
    for hidden in ("ADMETAI_predict_toxicity", "SwissADME_calculate_adme",
                   "DrugProps_pains_filter"):
        assert hidden not in names, hidden
    # The stable catalog exactly mirrors the disclosure result (no drift).
    assert names == set(disclosure.disclosed_tool_names)
    # Raw sequences never appear anywhere in the prompt.
    _, dynamic, full, _ = _sections_for(cand)
    assert RAW_HEAVY not in full and RAW_LIGHT not in full


# ── 4. small molecule candidate: ADMET tools in, sequence-only tools hidden ─


def test_small_molecule_candidate_stable_catalog_hides_sequence_tools():
    cand = _candidate(
        candidate_id="cand_sm",
        materials=[_material("m_s", "payload_smiles", SMILES)],
    )
    stable, _, _, disclosure = _sections_for(cand)
    names = set(_stable_catalog_names(stable))
    assert "ADMETAI_predict_toxicity" in names or "DrugProps_pains_filter" in names
    assert "PROSITE_scan_sequence" not in names
    assert names == set(disclosure.disclosed_tool_names)


# ── 5. structure candidate: structure tools disclosed per current logic ────


def test_structure_candidate_stable_catalog_contains_structure_tools():
    cand = _candidate(
        candidate_id="cand_struct",
        identifiers=[_identifier("pdb_id", "1N8Z")],
    )
    stable, _, _, disclosure = _sections_for(cand)
    names = set(_stable_catalog_names(stable))
    assert "PDBePISA_get_interfaces" in names
    assert names == set(disclosure.disclosed_tool_names)


# ── 6. mixed candidate: deterministic union in stable category order ───────


def test_mixed_candidate_stable_catalog_is_deterministic_union():
    cand = _candidate(
        candidate_id="cand_mixed",
        materials=[
            _material("m_s", "payload_smiles", SMILES),
            _material("m_h", "antibody_heavy_chain_sequence", RAW_HEAVY),
        ],
        identifiers=[_identifier("pdb_id", "1N8Z")],
    )
    stable, _, _, disclosure = _sections_for(cand)
    names = _stable_catalog_names(stable)
    # Union covers all three modalities.
    assert "PROSITE_scan_sequence" in names
    assert "PDBePISA_get_interfaces" in names
    assert any(n in names for n in ("ADMETAI_predict_toxicity", "DrugProps_pains_filter"))
    # Deterministic: same candidate reproduces the same catalog order.
    stable2, _, _, _ = _sections_for(cand)
    assert _stable_catalog_names(stable2) == names
    assert set(names) == set(disclosure.disclosed_tool_names)


# ── 7. fail_open candidate uses full scoped catalog + audit reason kept ────


def test_fail_open_candidate_uses_full_scoped_catalog_with_audit_reason():
    # Candidate with no recognizable modality → ambiguous fail-open.
    cand = _candidate(candidate_id="cand_ambig", materials=[], identifiers=[])
    stable, dynamic, _, disclosure = _sections_for(cand)
    assert disclosure.disclosure_summary.get("fail_open") is True
    assert "ambiguous_modality_fail_open" in disclosure.disclosure_tags
    names = set(_stable_catalog_names(stable))
    # Full scoped catalog is disclosed (existing production fail-open logic).
    assert names == set(disclosure.disclosed_tool_names)
    assert len(names) > 10
    # The audit reason travels in the dynamic suffix (disclosure_tags), not
    # the stable prefix.
    assert "ambiguous_modality_fail_open" in dynamic
    assert "ambiguous_modality_fail_open" not in stable


# ── 8. stable prefix excludes run/candidate/raw-data markers ───────────────


def test_stable_prefix_excludes_candidate_and_raw_data():
    cand = _candidate(
        candidate_id="cand_RUNSPECIFIC",
        materials=[
            _material("mat_SECRET", "antibody_heavy_chain_sequence", RAW_HEAVY),
            _material("mat_STRUCT", "structure_ref", PDB_PATH, value_format="pdb"),
        ],
        identifiers=[_identifier("uniprot_id", "P04626")],
    )
    stable, _, _, _ = _sections_for(
        cand, user_query_summary=f"analyze {RAW_HEAVY} for {PDB_PATH}"
    )
    for forbidden in (
        "cand_RUNSPECIFIC",
        "mat_SECRET",
        "mat_STRUCT",
        "P04626",
        RAW_HEAVY,
        PDB_PATH,
        "candidate_id",
        "candidate_available_fields",
        "candidate_modality_summary",
        "user_query_summary",
        "field_ref",
        "sha256_prefix",
        "storage_path",
        "run_id",
        "api_key",
        "sk-",
    ):
        assert forbidden not in stable, forbidden


# ── 9. stable catalog is deterministically ordered + key-sorted ────────────


def test_stable_catalog_is_ordered_and_key_sorted():
    cand = _candidate(
        candidate_id="cand_order",
        materials=[_material("m_s", "payload_smiles", SMILES)],
    )
    stable, _, _, _ = _sections_for(cand)
    marker = "Input schema/context JSON:\n"
    parsed = json.loads(stable.split(marker, 1)[1])
    # Top-level stable keys are sort_keys=True ordered and carry no
    # candidate/run fields.
    assert list(parsed.keys()) == sorted(parsed.keys())
    assert set(parsed.keys()) == {"agent_name", "compact_catalog", "step_id", "task"}
    # Each catalog entry's own keys are sorted.
    for entry in parsed["compact_catalog"]:
        assert list(entry.keys()) == sorted(entry.keys())
    # Prompt text stays English (no accidental Chinese translation).
    assert not any("一" <= ch <= "鿿" for ch in stable)


# ── 10. non-Step-6-Stage-1 payloads are unaffected (no split, no candidate) ─


def test_non_step6_stage1_payload_is_not_split():
    """A `step6_schema_mapping_stage_1` payload with no candidate/run keys
    (as the provider unit tests send) is NOT split — the whole schema stays
    in the single dynamic block, no trailing candidate block is added."""
    schema = {"task": "step6_schema_mapping_stage_1"}
    stable, dynamic = build_json_prompt_sections(
        prompt="pick", schema=schema, system="sys",
    )
    assert dynamic.startswith("Input schema/context JSON:\n")
    assert "Candidate/run-specific context JSON:" not in stable
    assert "Candidate/run-specific context JSON:" not in dynamic
    full = build_json_prompt(prompt="pick", schema=schema, system="sys")
    assert full == stable + dynamic
