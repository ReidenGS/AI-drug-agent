"""Step 9 Stage 1 prompt cache-friendly layout tests."""

from __future__ import annotations

import json

from app.agents.step_09_selection_policy import (
    STEP9_STAGE1_SYSTEM_PROMPT,
    STEP9_STAGE1_USER_PROMPT,
    build_step9_stage1_catalog,
    build_step9_stage1_payload,
    select_step9_stage1_tools,
)
from app.llm.json_task_validation import build_json_prompt, build_json_prompt_sections
from app.schemas.step_09_structure_variant_and_compound_screening import (
    Step9AvailableField,
    Step9HardGateAllowedTool,
    Step9HardGateBlockedTool,
    Step9LaneStatus,
    Step9ToolSchemaRequirement,
)


RAW_SEQ = "EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGK"
RAW_PDB = "HEADER TEST PDB\nATOM      1  N   GLY A   1"


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


def _allowed(tool_name: str, lane_type: str, candidate_id: str = "cand_a"):
    return Step9HardGateAllowedTool(
        tool_name=tool_name,
        lane_type=lane_type,  # type: ignore[arg-type]
        candidate_id=candidate_id,
        rationale="fixture allowed",
    )


def _blocked(tool_name: str, lane_type: str, candidate_id: str = "cand_a"):
    return Step9HardGateBlockedTool(
        tool_name=tool_name,
        lane_type=lane_type,  # type: ignore[arg-type]
        candidate_id=candidate_id,
        reason="schema_required:pdb_id",
    )


def _req(tool_name: str, lane_type: str, *, decision: str = "allowed"):
    return Step9ToolSchemaRequirement(
        candidate_id="cand_a",
        tool_name=tool_name,
        lane_type=lane_type,  # type: ignore[arg-type]
        required_fields=["smiles"] if lane_type == "compound_screening" else ["sequence"],
        schema_source="signature",
        satisfiable_required_fields=["smiles"] if decision == "allowed" else [],
        missing_required_fields=[] if decision == "allowed" else ["pdb_id"],
        hard_gate_decision=decision,  # type: ignore[arg-type]
        reason="schema_requirements_satisfied" if decision == "allowed" else "schema_required:pdb_id",
    )


def _field(candidate_id: str = "cand_a", field_ref: str = "material:mat_a"):
    return Step9AvailableField(
        candidate_id=candidate_id,
        field_ref=field_ref,
        provider="step_05",
        field_type="compound",
        value_kind="smiles",
    )


def _lane(
    *,
    candidate_id: str = "cand_a",
    lane_type: str = "compound_screening",
    allowed_tools=None,
    blocked_tools=None,
):
    return Step9LaneStatus(
        lane_type=lane_type,  # type: ignore[arg-type]
        candidate_id=candidate_id,
        candidate_type="compound_component",
        status="ready" if allowed_tools else "blocked",
        allowed_tools=allowed_tools or [],
        blocked_tools=blocked_tools or [],
        missing_requirements=["schema_required:pdb_id"] if blocked_tools else [],
        available_field_refs=[f"material:{candidate_id}_mat"],
    )


def _projection(*, allowed=None, blocked=None, fields=None, lanes=None, reqs=None):
    allowed = allowed if allowed is not None else [_allowed("ZINC_search_by_smiles", "compound_screening")]
    blocked = blocked if blocked is not None else [_blocked("DynaMut2_predict_stability", "variant_evaluation")]
    reqs = reqs if reqs is not None else [
        _req(t.tool_name, t.lane_type, decision="allowed") for t in allowed
    ]
    return {
        "step9_hard_gate_allowed_tools": allowed,
        "step9_hard_gate_blocked_tools_with_reason": blocked,
        "step9_tool_schema_requirements": reqs,
        "step9_available_fields": fields if fields is not None else [_field()],
        "step9_lane_statuses": lanes if lanes is not None else [
            _lane(allowed_tools=[t.tool_name for t in allowed], blocked_tools=[b.tool_name for b in blocked])
        ],
    }


def _sections_for(projection: dict, *, candidate_id="cand_a", canonical_query="", raw_user_query=""):
    catalog = build_step9_stage1_catalog(
        projection["step9_hard_gate_allowed_tools"],
        projection["step9_tool_schema_requirements"],
    )
    payload = build_step9_stage1_payload(
        candidate_id=candidate_id,
        catalog=catalog,
        readiness_projection=projection,
        canonical_query=canonical_query,
        raw_user_query=raw_user_query,
        step8_downstream_handoff_status=[{"candidate_id": candidate_id, "has_complex_structure": False}],
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


def test_same_allowed_catalog_different_candidate_context_stable_prefix_identical():
    projection_a = _projection(fields=[_field("cand_alpha", "material:mat_alpha")])
    projection_b = _projection(fields=[_field("cand_beta", "material:mat_beta")])
    stable_a, _, _ = _sections_for(
        projection_a,
        candidate_id="cand_alpha",
        canonical_query="screen alpha",
        raw_user_query="alpha query",
    )
    stable_b, _, _ = _sections_for(
        projection_b,
        candidate_id="cand_beta",
        canonical_query="screen beta",
        raw_user_query="beta query",
    )
    assert stable_a == stable_b
    assert stable_a


def test_candidate_specific_values_only_after_stable_prefix():
    stable, dynamic, full = _sections_for(
        _projection(fields=[_field("cand_SENTINEL", "material:field_SENTINEL")]),
        candidate_id="cand_SENTINEL",
        canonical_query="CANONICAL_SENTINEL",
        raw_user_query="RAW_QUERY_SENTINEL",
    )
    for needle in ("cand_SENTINEL", "material:field_SENTINEL", "CANONICAL_SENTINEL", "RAW_QUERY_SENTINEL"):
        assert needle not in stable
        assert needle in dynamic
        assert full.index(needle) >= len(stable)
    assert full == stable + dynamic


def test_stable_prefix_contains_allowed_catalog_only_not_blocked_tools():
    projection = _projection(
        allowed=[_allowed("ZINC_search_by_smiles", "compound_screening")],
        blocked=[_blocked("DynaMut2_predict_stability", "variant_evaluation")],
    )
    stable, dynamic, full = _sections_for(projection)
    names = [entry["tool_name"] for entry in _stable_catalog(stable)]
    assert names == ["ZINC_search_by_smiles"]
    assert "DynaMut2_predict_stability" not in stable
    assert "DynaMut2_predict_stability" not in dynamic
    assert "DynaMut2_predict_stability" not in full


def test_stable_prefix_excludes_raw_sequence_pdb_fasta_a3m_and_api_key():
    stable, dynamic, full = _sections_for(
        _projection(),
        canonical_query=f"design with {RAW_SEQ} and sk-secretvalue123",
        raw_user_query=f"{RAW_PDB}\n>seq\n{RAW_SEQ}\nA3M",
    )
    for forbidden in (RAW_SEQ, "HEADER TEST PDB", "ATOM      1", "sk-secretvalue123"):
        assert forbidden not in stable
        assert forbidden not in dynamic
        assert forbidden not in full


def test_stable_catalog_sorted_by_lane_then_tool():
    projection = _projection(
        allowed=[
            _allowed("ZINC_search_by_smiles", "compound_screening"),
            _allowed("ESM_generate_protein_sequence", "protein_design"),
            _allowed("AlphaMissense_get_variant_score", "variant_evaluation"),
        ],
        reqs=[
            _req("ZINC_search_by_smiles", "compound_screening"),
            _req("ESM_generate_protein_sequence", "protein_design"),
            _req("AlphaMissense_get_variant_score", "variant_evaluation"),
        ],
    )
    stable, _, _ = _sections_for(projection)
    pairs = [(entry["lane_type"], entry["tool_name"]) for entry in _stable_catalog(stable)]
    assert pairs == sorted(pairs)


def test_protein_only_allowed_set_does_not_include_compound_tools():
    projection = _projection(
        allowed=[_allowed("ESM_generate_protein_sequence", "protein_design")],
        reqs=[_req("ESM_generate_protein_sequence", "protein_design")],
    )
    stable, _, _ = _sections_for(projection)
    names = {entry["tool_name"] for entry in _stable_catalog(stable)}
    assert names == {"ESM_generate_protein_sequence"}
    assert "ZINC_search_by_smiles" not in names


def test_variant_only_allowed_set_does_not_include_protein_design_tools():
    projection = _projection(
        allowed=[_allowed("AlphaMissense_get_variant_score", "variant_evaluation")],
        reqs=[_req("AlphaMissense_get_variant_score", "variant_evaluation")],
    )
    stable, _, _ = _sections_for(projection)
    names = {entry["tool_name"] for entry in _stable_catalog(stable)}
    assert names == {"AlphaMissense_get_variant_score"}
    assert "ESM_generate_protein_sequence" not in names


def test_hallucinated_or_blocked_tool_is_dropped_and_audited():
    projection = _projection(
        allowed=[_allowed("ZINC_search_by_smiles", "compound_screening")],
        blocked=[_blocked("DynaMut2_predict_stability", "variant_evaluation")],
    )
    llm = _LLM({
        "selections": [
            {
                "tool_name": "ZINC_search_by_smiles",
                "lane_type": "compound_screening",
                "selection_reason": "valid",
            },
            {
                "tool_name": "DynaMut2_predict_stability",
                "lane_type": "variant_evaluation",
                "selection_reason": "blocked",
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
        readiness_projection=projection,
        candidate_id="cand_a",
    )
    assert [item.tool_name for item in result.selected_tools] == ["ZINC_search_by_smiles"]
    rejected = {item["tool_name"]: item["reason"] for item in result.rejected_tools_with_reason}
    assert rejected["DynaMut2_predict_stability"] == "tool_not_in_allowed_catalog"
    assert rejected["Imaginary_tool"] == "tool_not_in_allowed_catalog"


def test_valid_empty_selection_respected_no_forced_selection():
    result = select_step9_stage1_tools(
        llm=_LLM({"selections": []}),
        readiness_projection=_projection(
            allowed=[_allowed("ZINC_search_by_smiles", "compound_screening")]
        ),
        candidate_id="cand_a",
    )
    assert result.catalog_tool_names == ["ZINC_search_by_smiles"]
    assert result.selected_tools == []
    assert result.rejected_tools_with_reason == []
