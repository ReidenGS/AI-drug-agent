"""Step 9 Stage 1 prompt cache-friendly layout tests.

Stage 1's catalog is now the FULL active Step 9 tool set (6 tools), built
directly from `ACTIVE_STEP9_TOOLS` — independent of any per-candidate
hard-gate/readiness computation. These tests prove that invariant plus the
existing prompt-cache-friendly stable/dynamic split behavior.
"""

from __future__ import annotations

import json

from app.agents.step_09_selection_policy import (
    ACTIVE_STEP9_TOOLS,
    STEP9_STAGE1_SYSTEM_PROMPT,
    STEP9_STAGE1_USER_PROMPT,
    build_step9_stage1_catalog,
    build_step9_stage1_payload,
    select_step9_stage1_tools,
)
from app.llm.json_task_validation import build_json_prompt, build_json_prompt_sections


RAW_SEQ = "EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGK"
RAW_PDB = "HEADER TEST PDB\nATOM      1  N   GLY A   1"

_ACTIVE_TOOL_NAMES = {
    "NvidiaNIM_rfdiffusion",
    "NvidiaNIM_proteinmpnn",
    "ESM_generate_protein_sequence",
    "DynaMut2_predict_stability",
    "AlphaMissense_get_variant_score",
    "ESM_score_variant_sae_batch",
}


class _LLM:
    name = "fake"
    model = "fake"

    def __init__(self, response: dict):
        self.response = response

    def generate(self, prompt: str, *, system: str | None = None, **kwargs):
        raise NotImplementedError

    def generate_json(self, prompt: str, *, schema: dict, system: str | None = None) -> dict:
        self.prompt = prompt
        self.schema = schema
        self.system = system
        return self.response


def _projection(*, candidate_id="cand_a", missing=None, canonical_query="", raw_user_query=""):
    return {
        "input_fields": [
            {
                "field_ref": f"material:{candidate_id}_mat",
                "candidate_id": candidate_id,
                "source_step": "step_05",
                "source_artifact": "candidate_context_table",
                "source_path": "candidate_records[].materials[]",
                "field_name": "target_sequence",
                "field_type": "protein_sequence",
                "value_kind": "sequence_ref",
                "supports_tool_args": ["sequence", "prompt_sequence"],
                "can_resolve_at_runtime": True,
                "status": "available",
            }
        ],
        "candidate_summaries": [
            {
                "candidate_id": candidate_id,
                "candidate_type": "target_antigen",
                "field_count": 1,
                "field_types_present": ["protein_sequence"],
            }
        ],
        "handoff_summary": {"candidates": [{"candidate_id": candidate_id, "has_complex_structure": False}]},
        "missing_inputs": missing if missing is not None else [],
        "query_summary": {"canonical_query": canonical_query, "raw_user_query": raw_user_query},
    }


def _sections_for(projection: dict, *, candidate_id="cand_a"):
    catalog = build_step9_stage1_catalog()
    payload = build_step9_stage1_payload(
        candidate_id=candidate_id,
        catalog=catalog,
        projection=projection,
    )
    stable, dynamic = build_json_prompt_sections(
        prompt=STEP9_STAGE1_USER_PROMPT,
        schema=payload,
        system=STEP9_STAGE1_SYSTEM_PROMPT,
    )
    full = build_json_prompt(
        prompt=STEP9_STAGE1_USER_PROMPT,
        schema=payload,
        system=STEP9_STAGE1_SYSTEM_PROMPT,
    )
    return stable, dynamic, full


def _stable_catalog(stable: str) -> list[dict]:
    block = stable.split("Input schema/context JSON:\n", 1)[1]
    return json.loads(block)["compact_catalog"]


def test_active_catalog_is_always_all_six_tools_registry():
    assert set(ACTIVE_STEP9_TOOLS) == _ACTIVE_TOOL_NAMES


def test_stage1_catalog_always_contains_all_six_active_tools_regardless_of_readiness():
    catalog = build_step9_stage1_catalog()
    names = {entry["tool_name"] for entry in catalog}
    assert names == _ACTIVE_TOOL_NAMES


def test_stage1_catalog_has_no_zinc_or_chembl_tools():
    catalog = build_step9_stage1_catalog()
    names = {entry["tool_name"] for entry in catalog}
    assert not any(name.startswith(("ZINC_", "ChEMBL_")) for name in names)


def test_stage1_catalog_prefers_tooluniverse_official_description(monkeypatch):
    official = "OFFICIAL TU ProteinMPNN description for Step 9 selection"

    def _fake_specs(names):
        assert "NvidiaNIM_proteinmpnn" in names
        return {
            "NvidiaNIM_proteinmpnn": {
                "name": "NvidiaNIM_proteinmpnn",
                "description": official,
            }
        }

    from app.mcp import tooluniverse_adapter

    monkeypatch.setattr(tooluniverse_adapter, "get_tool_specifications", _fake_specs)
    catalog = build_step9_stage1_catalog()
    by_name = {entry["tool_name"]: entry for entry in catalog}

    assert by_name["NvidiaNIM_proteinmpnn"]["short_description"] == official
    assert set(by_name) == _ACTIVE_TOOL_NAMES


def test_stage1_catalog_falls_back_to_local_description_when_official_missing(monkeypatch):
    from app.mcp import tooluniverse_adapter

    monkeypatch.setattr(tooluniverse_adapter, "get_tool_specifications", lambda names: {})
    catalog = build_step9_stage1_catalog()
    by_name = {entry["tool_name"]: entry for entry in catalog}

    assert by_name["NvidiaNIM_proteinmpnn"]["short_description"] == (
        ACTIVE_STEP9_TOOLS["NvidiaNIM_proteinmpnn"]["short_description"]
    )
    assert set(by_name) == _ACTIVE_TOOL_NAMES


def test_same_active_catalog_different_candidate_context_stable_prefix_identical():
    projection_a = _projection(candidate_id="cand_alpha", canonical_query="screen alpha", raw_user_query="alpha query")
    projection_b = _projection(candidate_id="cand_beta", canonical_query="screen beta", raw_user_query="beta query")
    stable_a, _, _ = _sections_for(projection_a, candidate_id="cand_alpha")
    stable_b, _, _ = _sections_for(projection_b, candidate_id="cand_beta")
    assert stable_a == stable_b
    assert stable_a


def test_valid_empty_selection_respected_no_forced_selection():
    result = select_step9_stage1_tools(
        llm=_LLM({"selections": []}),
        projection=_projection(),
        candidate_id="cand_a",
    )
    assert set(result.catalog_tool_names) == _ACTIVE_TOOL_NAMES
    assert result.selected_tools == []
    assert result.rejected_tools_with_reason == []


def test_candidate_specific_values_only_after_stable_prefix():
    stable, dynamic, full = _sections_for(
        _projection(candidate_id="cand_SENTINEL", canonical_query="CANONICAL_SENTINEL", raw_user_query="RAW_QUERY_SENTINEL"),
        candidate_id="cand_SENTINEL",
    )
    for needle in ("cand_SENTINEL", "CANONICAL_SENTINEL", "RAW_QUERY_SENTINEL"):
        assert needle not in stable
        assert needle in dynamic
        assert full.index(needle) >= len(stable)
    assert full == stable + dynamic


def test_stable_prefix_declares_stage1_output_fields_and_few_shots():
    stable, dynamic, _ = _sections_for(
        _projection(candidate_id="cand_SENTINEL", canonical_query="CANONICAL_SENTINEL", raw_user_query="RAW_QUERY_SENTINEL"),
        candidate_id="cand_SENTINEL",
    )
    for needle in (
        '"selections"',
        '"tool_name"',
        '"lane_type"',
        '"selection_reason"',
        "Relevant-tool example",
        "AlphaMissense_get_variant_score",
        "Variant scoring is relevant and required inputs are available.",
        "No-relevant-tool example",
        '{"selections": []}',
    ):
        assert needle in stable
    for dynamic_only in ("cand_SENTINEL", "CANONICAL_SENTINEL", "RAW_QUERY_SENTINEL"):
        assert dynamic_only not in stable
        assert dynamic_only in dynamic


def test_stable_prefix_excludes_raw_sequence_pdb_fasta_a3m_and_api_key():
    stable, dynamic, full = _sections_for(
        _projection(canonical_query=f"design with {RAW_SEQ} and sk-secretvalue123", raw_user_query=f"{RAW_PDB}\n>seq\n{RAW_SEQ}\nA3M"),
    )
    for forbidden in (RAW_SEQ, "HEADER TEST PDB", "ATOM      1", "sk-secretvalue123"):
        assert forbidden not in stable
        assert forbidden not in dynamic
        assert forbidden not in full


def test_stable_catalog_sorted_by_lane_then_tool():
    stable, _, _ = _sections_for(_projection())
    pairs = [(entry["lane_type"], entry["tool_name"]) for entry in _stable_catalog(stable)]
    assert pairs == sorted(pairs)


def test_hallucinated_or_non_active_tool_is_rejected():
    llm = _LLM({
        "selections": [
            {
                "tool_name": "ESM_generate_protein_sequence",
                "lane_type": "protein_design",
                "selection_reason": "valid",
            },
            {
                "tool_name": "ZINC_search_by_smiles",
                "lane_type": "compound_screening",
                "selection_reason": "hallucinated non-active tool",
            },
            {
                "tool_name": "Imaginary_tool",
                "lane_type": "protein_design",
                "selection_reason": "hallucinated",
            },
        ]
    })
    result = select_step9_stage1_tools(
        llm=llm,
        projection=_projection(),
        candidate_id="cand_a",
    )
    assert [item.tool_name for item in result.selected_tools] == ["ESM_generate_protein_sequence"]
    rejected = {item["tool_name"]: item["reason"] for item in result.rejected_tools_with_reason}
    assert rejected["ZINC_search_by_smiles"] == "tool_not_in_active_catalog"
    assert rejected["Imaginary_tool"] == "tool_not_in_active_catalog"


def test_wrong_lane_for_known_tool_is_rejected():
    llm = _LLM({
        "selections": [
            {
                "tool_name": "ESM_generate_protein_sequence",
                "lane_type": "variant_evaluation",
                "selection_reason": "wrong lane",
            },
        ]
    })
    result = select_step9_stage1_tools(llm=llm, projection=_projection(), candidate_id="cand_a")
    assert result.selected_tools == []
    rejected = {item["tool_name"]: item["reason"] for item in result.rejected_tools_with_reason}
    assert rejected["ESM_generate_protein_sequence"] == "tool_lane_not_in_active_catalog"
