"""Step 9 Stage 2 prompt cache-friendly layout tests.

Stage 2 now consumes ONLY `Step9InputProjection.input_fields` (rendered as
the `step9_input_fields` payload key) — never Step 5/7/8 raw artifacts, and
never the legacy `step9_available_fields` hard-gate shape.
"""

from __future__ import annotations

import json
import re

import pytest

from app.agents import step_09_selection_policy as step9sel
from app.agents.step_09_input_projection import DuplicateStep9InputFieldError
from app.agents.step_09_selection_policy import (
    STEP9_STAGE2_SYSTEM_PROMPT,
    STEP9_STAGE2_USER_PROMPT,
    Step9Stage1SelectionAudit,
    build_step9_stage2_payload,
    select_step9_stage2_mappings,
    validate_step9_stage2_mapping,
)
from app.llm.json_task_validation import build_json_prompt, build_json_prompt_sections


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


class _ExplodingLLM:
    name = "must-not-call"
    model = "must-not-call"

    def generate(self, prompt: str, *, system: str | None = None, **kwargs):
        raise AssertionError("LLM should not be called")

    def generate_json(self, prompt: str, *, schema: dict, system: str | None = None) -> dict:
        raise AssertionError("LLM should not be called")


ESM_SEQUENCE_SCHEMA = {
    "type": "object",
    "properties": {
        "prompt_sequence": {"type": "string"},
        "task": {"type": "string", "enum": ["generate"]},
    },
    "required": ["task", "prompt_sequence"],
}


RF_DIFFUSION_SCHEMA = {
    "type": "object",
    "properties": {
        "input_pdb": {"type": "string"},
        "contigs": {"type": "string"},
    },
    "required": ["input_pdb", "contigs"],
}


ESM_SCORE_SCHEMA = {
    "type": "object",
    "properties": {
        "sequence": {"type": "string"},
        "variants": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "position": {"type": "integer"},
                    "ref_aa": {"type": "string"},
                    "alt_aa": {"type": "string"},
                },
                "required": ["position", "ref_aa", "alt_aa"],
            },
        },
        "model": {"type": "string", "enum": ["esmc-6b-2024-12"]},
    },
    "required": ["sequence", "variants"],
}


def _field(
    candidate_id="cand_a",
    field_ref="material:mat_a",
    value_kind="sequence_ref",
    field_type="protein_sequence",
    supports_tool_args=None,
    status="available",
    can_resolve_at_runtime=True,
):
    return {
        "field_ref": field_ref,
        "candidate_id": candidate_id,
        "source_step": "step_05",
        "source_artifact": "candidate_context_table",
        "source_path": "candidate_records[].materials[]",
        "field_name": "field",
        "field_type": field_type,
        "value_kind": value_kind,
        "supports_tool_args": supports_tool_args if supports_tool_args is not None else ["sequence", "prompt_sequence"],
        "can_resolve_at_runtime": can_resolve_at_runtime,
        "status": status,
    }


def _step8_complex_field(candidate_id="cand_a", field_ref="step8_complex_ref:cand_a:0"):
    return _field(
        candidate_id=candidate_id,
        field_ref=field_ref,
        value_kind="complex_structure_ref",
        field_type="complex_structure",
        supports_tool_args=["input_pdb", "pdb_file", "structure", "complex_structure", "backbone"],
    )


def _projection(fields=None, canonical_query="", raw_user_query=""):
    return {
        "input_fields": fields if fields is not None else [_field()],
        "candidate_summaries": [],
        "handoff_summary": {"candidates": []},
        "missing_inputs": [],
        "query_summary": {"canonical_query": canonical_query, "raw_user_query": raw_user_query},
    }


def _selected(tool_name="ESM_generate_protein_sequence", lane_type="protein_design", reason="selected"):
    return [Step9Stage1SelectionAudit(tool_name=tool_name, lane_type=lane_type, selection_reason=reason)]


def _sections_for(projection, *, selected=None, candidate_id="cand_a"):
    payload = build_step9_stage2_payload(
        candidate_id=candidate_id,
        selected_tools=selected or _selected(),
        projection=projection,
    )
    stable, dynamic = build_json_prompt_sections(
        prompt=STEP9_STAGE2_USER_PROMPT,
        schema=payload,
        system=STEP9_STAGE2_SYSTEM_PROMPT,
    )
    full = build_json_prompt(
        prompt=STEP9_STAGE2_USER_PROMPT,
        schema=payload,
        system=STEP9_STAGE2_SYSTEM_PROMPT,
    )
    return stable, dynamic, full


def _stable_tools(stable: str) -> list[dict]:
    block = stable.split("Input schema/context JSON:\n", 1)[1]
    return json.loads(block)["tools"]


def test_same_selected_schema_set_different_candidate_fields_stable_prefix_identical(monkeypatch):
    monkeypatch.setattr(step9sel, "signature_schema_for", lambda name: ESM_SEQUENCE_SCHEMA)
    stable_a, _, _ = _sections_for(
        _projection(fields=[_field("cand_alpha", "material:alpha")]),
        candidate_id="cand_alpha",
    )
    stable_b, _, _ = _sections_for(
        _projection(fields=[_field("cand_beta", "material:beta")]),
        candidate_id="cand_beta",
    )
    assert stable_a == stable_b


def test_no_selected_tools_skips_stage2_llm_call():
    result = select_step9_stage2_mappings(
        llm=_ExplodingLLM(),
        projection=_projection(),
        selected_tools=[],
        candidate_id="cand_no_selected",
    )
    assert result.schema_survivors == []
    assert result.mapped_tools == []
    assert result.uninvokable_tools == []
    assert result.argument_mapping_audit == []


def test_candidate_specific_data_only_after_stable_prefix(monkeypatch):
    monkeypatch.setattr(step9sel, "signature_schema_for", lambda name: ESM_SEQUENCE_SCHEMA)
    stable, dynamic, full = _sections_for(
        _projection(fields=[_field("cand_SENTINEL", "material:field_SENTINEL")], canonical_query="CANONICAL_SENTINEL", raw_user_query="RAW_SENTINEL"),
        candidate_id="cand_SENTINEL",
    )
    for needle in ("cand_SENTINEL", "material:field_SENTINEL", "CANONICAL_SENTINEL", "RAW_SENTINEL"):
        assert needle not in stable
        assert needle in dynamic
        assert full.index(needle) >= len(stable)


def test_stable_prefix_includes_only_selected_tool_schemas(monkeypatch):
    monkeypatch.setattr(step9sel, "signature_schema_for", lambda name: ESM_SEQUENCE_SCHEMA)
    stable, _, _ = _sections_for(
        _projection(),
        selected=_selected("ESM_generate_protein_sequence", "protein_design"),
    )
    names = [tool["tool_name"] for tool in _stable_tools(stable)]
    assert names == ["ESM_generate_protein_sequence"]
    assert "AlphaMissense_get_variant_score" not in names


def test_esm_score_stage2_payload_declares_variants_array_items(monkeypatch):
    fallback_schema_without_items = {
        "type": "object",
        "properties": {
            "sequence": {"type": "string"},
            "variants": {"type": "array"},
        },
        "required": ["sequence", "variants"],
    }
    monkeypatch.setattr(step9sel, "signature_schema_for", lambda name: fallback_schema_without_items)

    payload = build_step9_stage2_payload(
        candidate_id="cand_a",
        selected_tools=_selected("ESM_score_variant_sae_batch", "variant_evaluation"),
        projection=_projection(),
    )

    variants_schema = payload["tools"][0]["full_schema"]["properties"]["variants"]
    assert variants_schema["type"] == "array"
    assert variants_schema["items"]["type"] == "object"
    assert sorted(variants_schema["items"]["required"]) == ["alt_aa", "position", "ref_aa"]
    assert variants_schema["items"]["properties"]["position"]["type"] == "integer"
    assert variants_schema["items"]["properties"]["ref_aa"]["type"] == "string"
    assert variants_schema["items"]["properties"]["alt_aa"]["type"] == "string"


def test_stable_prefix_declares_stage2_output_fields_and_few_shots(monkeypatch):
    monkeypatch.setattr(step9sel, "signature_schema_for", lambda name: ESM_SEQUENCE_SCHEMA)
    stable, dynamic, _ = _sections_for(
        _projection(fields=[_field("cand_SENTINEL", "material:field_SENTINEL")], canonical_query="CANONICAL_SENTINEL", raw_user_query="RAW_SENTINEL"),
        candidate_id="cand_SENTINEL",
    )
    for needle in (
        '"tools"',
        '"can_invoke"',
        '"argument_mappings"',
        '"argument_literals"',
        '"literal_value_json"',
        '"missing_required_fields"',
        '"skip_reason"',
        '"argument_mapping_reason"',
        "DynaMut2_predict_stability",
        "ESM_score_variant_sae_batch",
        '"schema_arg": "variants"',
        '"position\\":777',
        '"field_ref": "identifier:mutation:V777L"',
        "supports_tool_args includes",
        "missing_required_fields",
    ):
        assert needle in stable
    assert "argument_json_literals" not in stable
    assert re.search(r"[\u4e00-\u9fff]", stable) is None
    for dynamic_only in ("cand_SENTINEL", "material:field_SENTINEL", "CANONICAL_SENTINEL", "RAW_SENTINEL"):
        assert dynamic_only not in stable
        assert dynamic_only in dynamic


def test_stable_prefix_excludes_raw_sequence_pdb_fasta_a3m_and_api_key(monkeypatch):
    monkeypatch.setattr(step9sel, "signature_schema_for", lambda name: ESM_SEQUENCE_SCHEMA)
    stable, dynamic, full = _sections_for(
        _projection(canonical_query=f"{RAW_SEQ} sk-secretvalue123", raw_user_query=f"{RAW_PDB}\n>seq\n{RAW_SEQ}\nA3M"),
    )
    for forbidden in (RAW_SEQ, "HEADER TEST PDB", "ATOM      1", "sk-secretvalue123"):
        assert forbidden not in stable
        assert forbidden not in dynamic
        assert forbidden not in full


def test_selected_tools_sorted_deterministically(monkeypatch):
    monkeypatch.setattr(step9sel, "signature_schema_for", lambda name: ESM_SEQUENCE_SCHEMA)
    stable, _, _ = _sections_for(
        _projection(),
        selected=[
            Step9Stage1SelectionAudit(tool_name="AlphaMissense_get_variant_score", lane_type="variant_evaluation"),
            Step9Stage1SelectionAudit(tool_name="ESM_generate_protein_sequence", lane_type="protein_design"),
        ],
    )
    pairs = [(tool["lane_type"], tool["tool_name"]) for tool in _stable_tools(stable)]
    assert pairs == sorted(pairs)


def test_field_refs_only_in_dynamic_suffix(monkeypatch):
    monkeypatch.setattr(step9sel, "signature_schema_for", lambda name: ESM_SEQUENCE_SCHEMA)
    stable, dynamic, _ = _sections_for(_projection(fields=[_field(field_ref="material:only_dynamic")]))
    assert "material:only_dynamic" not in stable
    assert "material:only_dynamic" in dynamic


def test_missing_required_fields_produce_uninvokable_not_fake_mapping(monkeypatch):
    monkeypatch.setattr(step9sel, "signature_schema_for", lambda name: RF_DIFFUSION_SCHEMA)
    result = select_step9_stage2_mappings(
        llm=_LLM({"tools": [{
            "tool_name": "NvidiaNIM_rfdiffusion",
            "lane_type": "protein_design",
            "can_invoke": True,
            "argument_mappings": [{"schema_arg": "input_pdb", "field_ref": "step8_complex_ref:cand_a:0"}],
            "argument_literals": [],
            "missing_required_fields": [],
            "skip_reason": "",
            "argument_mapping_reason": "mapped input only",
        }]}),
        projection=_projection(fields=[_step8_complex_field()]),
        selected_tools=_selected("NvidiaNIM_rfdiffusion", "protein_design"),
        candidate_id="cand_a",
    )
    tool = result.mapped_tools[0]
    assert tool.can_invoke is False
    assert tool.missing_required_fields == ["contigs"]


def test_official_literal_accepted_only_when_schema_permits():
    tool = {
        "tool_name": "ESM_generate_protein_sequence",
        "lane_type": "protein_design",
        "full_schema": ESM_SEQUENCE_SCHEMA,
        "required_fields": ["task", "prompt_sequence"],
    }
    valid = validate_step9_stage2_mapping(
        response_item={
            "tool_name": "ESM_generate_protein_sequence",
            "lane_type": "protein_design",
            "can_invoke": True,
            "argument_mappings": [{"schema_arg": "prompt_sequence", "field_ref": "material:sequence"}],
            "argument_literals": [{"schema_arg": "task", "literal_value": "generate"}],
            "missing_required_fields": [],
        },
        selected_tool=tool,
        available_fields=[_field(field_ref="material:sequence")],
    )
    invalid = validate_step9_stage2_mapping(
        response_item={
            "tool_name": "ESM_generate_protein_sequence",
            "lane_type": "protein_design",
            "can_invoke": True,
            "argument_mappings": [{"schema_arg": "prompt_sequence", "field_ref": "material:sequence"}],
            "argument_literals": [{"schema_arg": "task", "literal_value": "invented"}],
            "missing_required_fields": [],
        },
        selected_tool=tool,
        available_fields=[_field(field_ref="material:sequence")],
    )
    assert valid.can_invoke is True
    assert valid.argument_literals[0].literal_value == "generate"
    assert invalid.can_invoke is True  # deterministic singleton literal repairs it
    assert invalid.argument_literals[0].literal_value == "generate"
    assert "literal_not_allowed:task" in invalid.argument_mapping_reason


def test_json_literal_value_json_accepted_and_parsed_for_array_schema_arg():
    tool = {
        "tool_name": "ESM_score_variant_sae_batch",
        "lane_type": "variant_evaluation",
        "full_schema": ESM_SCORE_SCHEMA,
        "required_fields": ["sequence", "variants"],
    }
    mapped = validate_step9_stage2_mapping(
        response_item={
            "tool_name": "ESM_score_variant_sae_batch",
            "lane_type": "variant_evaluation",
            "can_invoke": True,
            "argument_mappings": [{"schema_arg": "sequence", "field_ref": "material:heavy_chain"}],
            "argument_literals": [
                {
                    "schema_arg": "variants",
                    "literal_value_json": '[{"position":777,"ref_aa":"V","alt_aa":"L"}]',
                },
                {"schema_arg": "model", "literal_value_json": '"esmc-6b-2024-12"'},
            ],
            "missing_required_fields": [],
        },
        selected_tool=tool,
        available_fields=[
            _field(field_ref="material:heavy_chain", supports_tool_args=["sequence"])
        ],
    )
    assert mapped.can_invoke is True
    literals = {item.schema_arg: item.literal_value for item in mapped.argument_literals}
    assert literals["variants"] == [{"position": 777, "ref_aa": "V", "alt_aa": "L"}]
    assert not isinstance(literals["variants"], str)
    assert literals["model"] == "esmc-6b-2024-12"


@pytest.mark.parametrize(
    "literal_json",
    [
        '["V777L"]',
        '[{"variant":"V777L"}]',
        '[{"position":777,"ref_aa":"V"}]',
    ],
)
def test_invalid_variants_literal_shape_hard_rejects_tool(literal_json):
    tool = {
        "tool_name": "ESM_score_variant_sae_batch",
        "lane_type": "variant_evaluation",
        "full_schema": ESM_SCORE_SCHEMA,
        "required_fields": ["sequence", "variants"],
    }
    mapped = validate_step9_stage2_mapping(
        response_item={
            "tool_name": "ESM_score_variant_sae_batch",
            "lane_type": "variant_evaluation",
            "can_invoke": True,
            "argument_mappings": [{"schema_arg": "sequence", "field_ref": "material:heavy_chain"}],
            "argument_literals": [{"schema_arg": "variants", "literal_value_json": literal_json}],
            "missing_required_fields": [],
        },
        selected_tool=tool,
        available_fields=[
            _field(field_ref="material:heavy_chain", supports_tool_args=["sequence"])
        ],
    )
    assert mapped.can_invoke is False
    assert mapped.skip_reason == "invalid_argument_literal_schema"
    assert "variants" in mapped.missing_required_fields
    assert mapped.argument_literals == []
    assert "invalid_variants_shape:variants" in mapped.argument_mapping_reason


def test_parsed_dict_argument_literals_from_openai_parser_are_accepted():
    tool = {
        "tool_name": "ESM_score_variant_sae_batch",
        "lane_type": "variant_evaluation",
        "full_schema": ESM_SCORE_SCHEMA,
        "required_fields": ["sequence", "variants"],
    }
    mapped = validate_step9_stage2_mapping(
        response_item={
            "tool_name": "ESM_score_variant_sae_batch",
            "lane_type": "variant_evaluation",
            "can_invoke": True,
            "argument_mappings": [{"schema_arg": "sequence", "field_ref": "material:heavy_chain"}],
            "argument_literals": {
                "variants": [{"position": 777, "ref_aa": "V", "alt_aa": "L"}],
                "model": "esmc-6b-2024-12",
            },
            "missing_required_fields": [],
        },
        selected_tool=tool,
        available_fields=[
            _field(field_ref="material:heavy_chain", supports_tool_args=["sequence"])
        ],
    )
    assert mapped.can_invoke is True
    assert {item.schema_arg for item in mapped.argument_literals} == {"variants", "model"}


def test_invalid_parsed_dict_variants_literal_shape_hard_rejects_tool():
    tool = {
        "tool_name": "ESM_score_variant_sae_batch",
        "lane_type": "variant_evaluation",
        "full_schema": ESM_SCORE_SCHEMA,
        "required_fields": ["sequence", "variants"],
    }
    mapped = validate_step9_stage2_mapping(
        response_item={
            "tool_name": "ESM_score_variant_sae_batch",
            "lane_type": "variant_evaluation",
            "can_invoke": True,
            "argument_mappings": [{"schema_arg": "sequence", "field_ref": "material:heavy_chain"}],
            "argument_literals": {"variants": [{"variant": "V777L"}]},
            "missing_required_fields": [],
        },
        selected_tool=tool,
        available_fields=[
            _field(field_ref="material:heavy_chain", supports_tool_args=["sequence"])
        ],
    )
    assert mapped.can_invoke is False
    assert mapped.skip_reason == "invalid_argument_literal_schema"
    assert "invalid_variants_shape:variants" in mapped.argument_mapping_reason


def test_invalid_json_literal_value_json_hard_rejects_tool():
    tool = {
        "tool_name": "ESM_score_variant_sae_batch",
        "lane_type": "variant_evaluation",
        "full_schema": ESM_SCORE_SCHEMA,
        "required_fields": ["sequence", "variants"],
    }
    mapped = validate_step9_stage2_mapping(
        response_item={
            "tool_name": "ESM_score_variant_sae_batch",
            "lane_type": "variant_evaluation",
            "can_invoke": True,
            "argument_mappings": [{"schema_arg": "sequence", "field_ref": "material:heavy_chain"}],
            "argument_literals": [{"schema_arg": "variants", "literal_value_json": "[not-json"}],
            "missing_required_fields": [],
        },
        selected_tool=tool,
        available_fields=[
            _field(field_ref="material:heavy_chain", supports_tool_args=["sequence"])
        ],
    )
    assert mapped.can_invoke is False
    assert mapped.skip_reason == "invalid_argument_literal_json"
    assert "literal_value_json_invalid:variants" in mapped.argument_mapping_reason


def test_duplicate_schema_arg_between_mapping_and_literal_is_audited_no_overwrite():
    tool = {
        "tool_name": "ESM_score_variant_sae_batch",
        "lane_type": "variant_evaluation",
        "full_schema": ESM_SCORE_SCHEMA,
        "required_fields": ["sequence", "variants"],
    }
    mapped = validate_step9_stage2_mapping(
        response_item={
            "tool_name": "ESM_score_variant_sae_batch",
            "lane_type": "variant_evaluation",
            "can_invoke": True,
            "argument_mappings": [
                {"schema_arg": "sequence", "field_ref": "material:heavy_chain"},
                {"schema_arg": "variants", "field_ref": "identifier:variant:V777L"},
            ],
            "argument_literals": [
                {
                    "schema_arg": "variants",
                    "literal_value_json": '[{"position":777,"ref_aa":"V","alt_aa":"L"}]',
                }
            ],
            "missing_required_fields": [],
        },
        selected_tool=tool,
        available_fields=[
            _field(field_ref="material:heavy_chain", supports_tool_args=["sequence"]),
            _field(
                field_ref="identifier:variant:V777L",
                value_kind="variant",
                field_type="variant",
                supports_tool_args=["variants"],
            ),
        ],
    )
    assert mapped.can_invoke is False
    assert mapped.skip_reason == "duplicate_schema_arg"
    assert [(item.schema_arg, item.field_ref) for item in mapped.argument_mappings] == [
        ("sequence", "material:heavy_chain"),
        ("variants", "identifier:variant:V777L"),
    ]
    assert mapped.argument_literals == []
    assert "duplicate_schema_arg:variants" in mapped.argument_mapping_reason


def test_duplicate_schema_arg_inside_literals_is_audited_no_silent_overwrite():
    tool = {
        "tool_name": "ESM_score_variant_sae_batch",
        "lane_type": "variant_evaluation",
        "full_schema": ESM_SCORE_SCHEMA,
        "required_fields": ["sequence", "variants"],
    }
    mapped = validate_step9_stage2_mapping(
        response_item={
            "tool_name": "ESM_score_variant_sae_batch",
            "lane_type": "variant_evaluation",
            "can_invoke": True,
            "argument_mappings": [{"schema_arg": "sequence", "field_ref": "material:heavy_chain"}],
            "argument_literals": [
                {
                    "schema_arg": "variants",
                    "literal_value_json": '[{"position":777,"ref_aa":"V","alt_aa":"L"}]',
                },
                {
                    "schema_arg": "variants",
                    "literal_value_json": '[{"position":888,"ref_aa":"A","alt_aa":"G"}]',
                },
            ],
            "missing_required_fields": [],
        },
        selected_tool=tool,
        available_fields=[
            _field(field_ref="material:heavy_chain", supports_tool_args=["sequence"])
        ],
    )
    assert mapped.can_invoke is False
    assert mapped.skip_reason == "duplicate_schema_arg"
    assert len(mapped.argument_literals) == 1
    assert mapped.argument_literals[0].literal_value == [
        {"position": 777, "ref_aa": "V", "alt_aa": "L"}
    ]
    assert "duplicate_schema_arg:variants" in mapped.argument_mapping_reason


def test_hallucinated_schema_arg_is_dropped_and_duplicate_does_not_overwrite():
    tool = {
        "tool_name": "ESM_generate_protein_sequence",
        "lane_type": "protein_design",
        "full_schema": ESM_SEQUENCE_SCHEMA,
        "required_fields": ["prompt_sequence"],
    }
    mapped = validate_step9_stage2_mapping(
        response_item={
            "tool_name": "ESM_generate_protein_sequence",
            "lane_type": "protein_design",
            "can_invoke": True,
            "argument_mappings": [
                {"schema_arg": "prompt_sequence", "field_ref": "material:sequence_a"},
                {"schema_arg": "prompt_sequence", "field_ref": "material:sequence_b"},
                {"schema_arg": "not_in_schema", "field_ref": "material:sequence_a"},
            ],
            "argument_literals": [],
            "missing_required_fields": [],
        },
        selected_tool=tool,
        available_fields=[_field(field_ref="material:sequence_a"), _field(field_ref="material:sequence_b")],
    )
    assert [(m.schema_arg, m.field_ref) for m in mapped.argument_mappings] == [
        ("prompt_sequence", "material:sequence_a")
    ]
    assert mapped.can_invoke is False
    assert mapped.skip_reason == "duplicate_schema_arg"
    assert "duplicate_schema_arg:prompt_sequence" in mapped.argument_mapping_reason
    assert "schema_arg_not_in_full_schema:not_in_schema" in mapped.argument_mapping_reason


def test_field_ref_not_in_projection_is_rejected():
    tool = {
        "tool_name": "ESM_generate_protein_sequence",
        "lane_type": "protein_design",
        "full_schema": ESM_SEQUENCE_SCHEMA,
        "required_fields": ["prompt_sequence"],
    }
    mapped = validate_step9_stage2_mapping(
        response_item={
            "tool_name": "ESM_generate_protein_sequence",
            "lane_type": "protein_design",
            "can_invoke": True,
            "argument_mappings": [{"schema_arg": "prompt_sequence", "field_ref": "material:not_in_projection"}],
            "argument_literals": [],
            "missing_required_fields": [],
        },
        selected_tool=tool,
        available_fields=[_field(field_ref="material:sequence_a")],
    )
    assert mapped.can_invoke is False
    assert "prompt_sequence" in mapped.missing_required_fields
    assert "field_ref_not_available:prompt_sequence" in mapped.argument_mapping_reason


def test_schema_arg_not_in_official_schema_is_rejected():
    tool = {
        "tool_name": "ESM_generate_protein_sequence",
        "lane_type": "protein_design",
        "full_schema": ESM_SEQUENCE_SCHEMA,
        "required_fields": ["prompt_sequence"],
    }
    mapped = validate_step9_stage2_mapping(
        response_item={
            "tool_name": "ESM_generate_protein_sequence",
            "lane_type": "protein_design",
            "can_invoke": True,
            "argument_mappings": [{"schema_arg": "not_a_real_arg", "field_ref": "material:sequence_a"}],
            "argument_literals": [],
            "missing_required_fields": [],
        },
        selected_tool=tool,
        available_fields=[_field(field_ref="material:sequence_a")],
    )
    assert "schema_arg_not_in_full_schema:not_a_real_arg" in mapped.argument_mapping_reason
    assert mapped.argument_mappings == []


def _run_mapping(monkeypatch, *, tool_name, lane_type, schema, fields, response=None):
    monkeypatch.setattr(step9sel, "signature_schema_for", lambda name: schema)
    return select_step9_stage2_mappings(
        llm=_LLM(response or {"tools": [{
            "tool_name": tool_name,
            "lane_type": lane_type,
            "can_invoke": True,
            "argument_mappings": [
                {"schema_arg": arg, "field_ref": field["field_ref"]}
                for arg, field in fields.items()
            ],
            "argument_literals": [],
            "missing_required_fields": [],
            "skip_reason": "",
            "argument_mapping_reason": "test mapping",
        }]}),
        projection=_projection(fields=list(fields.values())),
        selected_tools=_selected(tool_name, lane_type),
        candidate_id="cand_a",
    ).mapped_tools[0]


def test_alphamissense_selected_with_uniprot_and_variant_maps_both_args(monkeypatch):
    schema = {
        "type": "object",
        "properties": {"uniprot_id": {"type": "string"}, "variant": {"type": "string"}},
        "required": ["uniprot_id", "variant"],
    }
    tool = _run_mapping(
        monkeypatch,
        tool_name="AlphaMissense_get_variant_score",
        lane_type="variant_evaluation",
        schema=schema,
        fields={
            "uniprot_id": _field(field_ref="identifier:uniprot_id:P04626", value_kind="uniprot_id", field_type="identifier", supports_tool_args=["uniprot_id", "accession", "uniprot_accession"]),
            "variant": _field(field_ref="identifier:variant:V777L", value_kind="variant", field_type="variant", supports_tool_args=["variant", "variants", "mutation", "mutations"]),
        },
    )
    assert tool.can_invoke is True
    assert {m.schema_arg for m in tool.argument_mappings} == {"uniprot_id", "variant"}


def test_alphamissense_cannot_use_identifier_only_uniprot_as_raw_sequence(monkeypatch):
    schema = {
        "type": "object",
        "properties": {"sequence": {"type": "string"}},
        "required": ["sequence"],
    }
    tool = _run_mapping(
        monkeypatch,
        tool_name="ESM_generate_protein_sequence",
        lane_type="protein_design",
        schema=schema,
        fields={
            "sequence": _field(
                field_ref="identifier:uniprot_id:P04626",
                value_kind="uniprot_id",
                field_type="identifier",
                supports_tool_args=["uniprot_id", "accession", "uniprot_accession"],
            ),
        },
        response={"tools": [{
            "tool_name": "ESM_generate_protein_sequence",
            "lane_type": "protein_design",
            "can_invoke": True,
            "argument_mappings": [{"schema_arg": "sequence", "field_ref": "identifier:uniprot_id:P04626"}],
            "argument_literals": [],
            "missing_required_fields": [],
            "skip_reason": "",
            "argument_mapping_reason": "attempted identifier-only mapping",
        }]},
    )
    assert tool.can_invoke is False
    assert "sequence" in tool.missing_required_fields
    assert "field_ref_incompatible:sequence" in tool.argument_mapping_reason


def test_dynamut_selected_maps_pdb_chain_mutation_and_operation_literal(monkeypatch):
    schema = {
        "type": "object",
        "properties": {
            "operation": {"type": "string", "enum": ["predict_stability"]},
            "pdb_id": {"type": "string"},
            "chain": {"type": "string"},
            "mutation": {"type": "string"},
        },
        "required": ["operation", "pdb_id", "chain", "mutation"],
    }
    tool = _run_mapping(
        monkeypatch,
        tool_name="DynaMut2_predict_stability",
        lane_type="variant_evaluation",
        schema=schema,
        fields={
            "pdb_id": _field(field_ref="identifier:pdb_id:1N8Z", value_kind="pdb_id", field_type="structure_identifier", supports_tool_args=["pdb_id"]),
            "chain": _field(field_ref="identifier:chain:A", value_kind="chain_id", field_type="chain", supports_tool_args=["chain", "chain_id"]),
            "mutation": _field(field_ref="identifier:mutation:V777L", value_kind="mutation", field_type="variant", supports_tool_args=["variant", "variants", "mutation", "mutations"]),
        },
    )
    assert tool.can_invoke is True
    assert {m.schema_arg for m in tool.argument_mappings} == {"pdb_id", "chain", "mutation"}
    assert [(lit.schema_arg, lit.literal_value) for lit in tool.argument_literals] == [
        ("operation", "predict_stability")
    ]


def test_dynamut_without_true_pdb_id_uninvokable_missing_pdb_id(monkeypatch):
    schema = {
        "type": "object",
        "properties": {"pdb_id": {"type": "string"}, "chain": {"type": "string"}, "mutation": {"type": "string"}},
        "required": ["pdb_id", "chain", "mutation"],
    }
    tool = _run_mapping(
        monkeypatch,
        tool_name="DynaMut2_predict_stability",
        lane_type="variant_evaluation",
        schema=schema,
        fields={
            "chain": _field(field_ref="identifier:chain:A", value_kind="chain_id", field_type="chain", supports_tool_args=["chain", "chain_id"]),
            "mutation": _field(field_ref="identifier:mutation:V777L", value_kind="mutation", field_type="variant", supports_tool_args=["variant", "variants", "mutation", "mutations"]),
        },
    )
    assert tool.can_invoke is False
    assert "pdb_id" in tool.missing_required_fields


def test_rfdiffusion_without_contigs_uninvokable_missing_contigs(monkeypatch):
    schema = {
        "type": "object",
        "properties": {"input_pdb": {"type": "string"}, "contigs": {"type": "string"}},
        "required": ["input_pdb", "contigs"],
    }
    tool = _run_mapping(
        monkeypatch,
        tool_name="NvidiaNIM_rfdiffusion",
        lane_type="protein_design",
        schema=schema,
        fields={"input_pdb": _step8_complex_field()},
    )
    assert tool.can_invoke is False
    assert "contigs" in tool.missing_required_fields


def test_proteinmpnn_with_true_complex_maps_input_pdb(monkeypatch):
    schema = {
        "type": "object",
        "properties": {"input_pdb": {"type": "string"}},
        "required": ["input_pdb"],
    }
    tool = _run_mapping(
        monkeypatch,
        tool_name="NvidiaNIM_proteinmpnn",
        lane_type="protein_design",
        schema=schema,
        fields={"input_pdb": _step8_complex_field()},
    )
    assert tool.can_invoke is True
    assert tool.argument_mappings[0].schema_arg == "input_pdb"


def test_dynamut_missing_chain_uninvokable(monkeypatch):
    schema = {
        "type": "object",
        "properties": {"pdb_id": {"type": "string"}, "chain": {"type": "string"}, "mutation": {"type": "string"}},
        "required": ["pdb_id", "chain", "mutation"],
    }
    tool = _run_mapping(
        monkeypatch,
        tool_name="DynaMut2_predict_stability",
        lane_type="variant_evaluation",
        schema=schema,
        fields={
            "pdb_id": _field(field_ref="identifier:pdb_id:1N8Z", value_kind="pdb_id", field_type="structure_identifier", supports_tool_args=["pdb_id"]),
            "mutation": _field(field_ref="identifier:mutation:V777L", value_kind="mutation", field_type="variant", supports_tool_args=["variant", "variants", "mutation", "mutations"]),
        },
    )
    assert tool.can_invoke is False
    assert tool.missing_required_fields == ["chain"]


def test_esm_generate_sequence_maps_task_literal_and_prompt_sequence(monkeypatch):
    """A field explicitly marked as a masked generation prompt (value_kind
    `masked_prompt_sequence`, the ONLY value_kind `prompt_sequence` accepts)
    maps successfully. See the sibling ordinary-sequence tests below for the
    (much more common in production today) rejection case."""
    schema = {
        "type": "object",
        "properties": {
            "task": {"type": "string", "enum": ["generate"]},
            "prompt_sequence": {"type": "string"},
        },
        "required": ["task", "prompt_sequence"],
    }
    tool = _run_mapping(
        monkeypatch,
        tool_name="ESM_generate_protein_sequence",
        lane_type="protein_design",
        schema=schema,
        fields={
            "prompt_sequence": _field(
                field_ref="material:masked_prompt",
                value_kind="masked_prompt_sequence",
                supports_tool_args=["prompt_sequence"],
            )
        },
    )
    assert tool.can_invoke is True
    assert tool.argument_mappings[0].schema_arg == "prompt_sequence"
    assert tool.argument_literals[0].literal_value == "generate"


def test_esm_generate_ordinary_heavy_chain_sequence_cannot_map_to_prompt_sequence(monkeypatch):
    """Regression for the reported bug: an ordinary complete antibody
    heavy-chain sequence (`value_kind="sequence_ref"`, exactly what
    `Step9InputProjection` emits for a real heavy/light/target chain) must
    NOT satisfy `prompt_sequence` — even if the LLM (or a Mock) tries to map
    it there, `_step9_field_can_satisfy_arg`'s `supports_tool_args` gate
    rejects it because the projection layer never lists "prompt_sequence"
    for an ordinary sequence field."""
    schema = {
        "type": "object",
        "properties": {
            "task": {"type": "string", "enum": ["generate"]},
            "prompt_sequence": {"type": "string"},
        },
        "required": ["task", "prompt_sequence"],
    }
    heavy_chain_field = _field(
        field_ref="material:heavy_chain",
        value_kind="sequence_ref",
        field_type="protein_sequence",
        supports_tool_args=["sequence"],  # production Step9InputProjection contract
    )
    tool = validate_step9_stage2_mapping(
        response_item={
            "tool_name": "ESM_generate_protein_sequence",
            "lane_type": "protein_design",
            # An LLM (or a misbehaving Mock) that hallucinates this mapping
            # anyway must still be overridden by deterministic validation.
            "can_invoke": True,
            "argument_mappings": [{"schema_arg": "prompt_sequence", "field_ref": "material:heavy_chain"}],
            "argument_literals": [{"schema_arg": "task", "literal_value": "generate"}],
            "missing_required_fields": [],
        },
        selected_tool={
            "tool_name": "ESM_generate_protein_sequence",
            "lane_type": "protein_design",
            "full_schema": schema,
            "required_fields": ["task", "prompt_sequence"],
        },
        available_fields=[heavy_chain_field],
    )
    assert tool.can_invoke is False
    assert tool.argument_mappings == []
    assert "prompt_sequence" in tool.missing_required_fields
    assert "field_ref_incompatible:prompt_sequence" in tool.argument_mapping_reason


def test_no_raw_sequence_pdb_or_tooluniverse_payload_in_mapping_audit(monkeypatch):
    schema = {
        "type": "object",
        "properties": {"prompt_sequence": {"type": "string"}},
        "required": ["prompt_sequence"],
    }
    tool = _run_mapping(
        monkeypatch,
        tool_name="ESM_generate_protein_sequence",
        lane_type="protein_design",
        schema=schema,
        fields={
            "prompt_sequence": _field(
                field_ref="material:sequence_ref",
                value_kind="sequence_ref",
                field_type="protein_sequence",
            )
        },
    )
    blob = json.dumps(tool.model_dump())
    assert RAW_SEQ not in blob
    assert RAW_PDB not in blob
    assert "ToolUniverse" not in blob


def test_duplicate_field_ref_in_available_fields_raises_not_silent_overwrite():
    """A non-canonical `available_fields` list with two entries sharing the
    same field_ref must be rejected loudly, not resolved by whichever entry
    the dict-comprehension lookup happens to keep last."""
    tool = {
        "tool_name": "ESM_generate_protein_sequence",
        "lane_type": "protein_design",
        "full_schema": ESM_SEQUENCE_SCHEMA,
        "required_fields": ["prompt_sequence"],
    }
    duplicate_fields = [
        _field(field_ref="material:sequence", candidate_id="cand_a"),
        _field(field_ref="material:sequence", candidate_id="cand_b"),
    ]
    with pytest.raises(DuplicateStep9InputFieldError, match="material:sequence"):
        validate_step9_stage2_mapping(
            response_item={
                "tool_name": "ESM_generate_protein_sequence",
                "lane_type": "protein_design",
                "can_invoke": True,
                "argument_mappings": [{"schema_arg": "prompt_sequence", "field_ref": "material:sequence"}],
                "argument_literals": [],
                "missing_required_fields": [],
            },
            selected_tool=tool,
            available_fields=duplicate_fields,
        )


def test_select_stage2_mappings_end_to_end_only_ordinary_sequence_marks_esm_generate_uninvokable(
    monkeypatch,
):
    """End-to-end `select_step9_stage2_mappings`: ESM_generate_protein_sequence
    is selected (Stage 1 correctly judged it relevant to the query), only an
    ordinary heavy-chain protein_sequence field exists, and the (mock) LLM
    hallucinates mapping it to prompt_sequence anyway. The final mapped
    result must still be can_invoke=false with prompt_sequence in
    missing_required_fields, and the tool must show up in
    uninvokable_tools/uninvokable_tool_details."""
    schema = {
        "type": "object",
        "properties": {
            "task": {"type": "string", "enum": ["generate"]},
            "prompt_sequence": {"type": "string"},
        },
        "required": ["task", "prompt_sequence"],
    }
    monkeypatch.setattr(step9sel, "signature_schema_for", lambda name: schema)

    heavy_chain_field = _field(
        field_ref="material:heavy_chain",
        value_kind="sequence_ref",
        field_type="protein_sequence",
        supports_tool_args=["sequence"],
    )
    llm = _LLM(
        {
            "tools": [
                {
                    "tool_name": "ESM_generate_protein_sequence",
                    "lane_type": "protein_design",
                    "can_invoke": True,
                    "argument_mappings": [
                        {"schema_arg": "prompt_sequence", "field_ref": "material:heavy_chain"}
                    ],
                    "argument_literals": [{"schema_arg": "task", "literal_value": "generate"}],
                    "missing_required_fields": [],
                    "skip_reason": "",
                    "argument_mapping_reason": "hallucinated: mapped ordinary sequence to prompt_sequence",
                }
            ]
        }
    )
    result = select_step9_stage2_mappings(
        llm=llm,
        projection=_projection(fields=[heavy_chain_field]),
        selected_tools=_selected("ESM_generate_protein_sequence", "protein_design"),
        candidate_id="cand_a",
    )

    assert len(result.mapped_tools) == 1
    mapped = result.mapped_tools[0]
    assert mapped.tool_name == "ESM_generate_protein_sequence"
    assert mapped.can_invoke is False
    assert mapped.argument_mappings == []
    assert "prompt_sequence" in mapped.missing_required_fields

    assert "ESM_generate_protein_sequence" in result.uninvokable_tools
    details = {d["tool_name"]: d for d in result.uninvokable_tool_details}
    assert "prompt_sequence" in details["ESM_generate_protein_sequence"]["missing_required_fields"]
