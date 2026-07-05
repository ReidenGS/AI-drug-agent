"""Step 9 Stage 2 prompt cache-friendly layout tests."""

from __future__ import annotations

import json

from app.agents import step_09_selection_policy as step9sel
from app.agents.step_09_selection_policy import (
    STEP9_STAGE2_SYSTEM_PROMPT,
    STEP9_STAGE2_USER_PROMPT,
    Step9Stage1SelectionAudit,
    build_step9_stage2_payload,
    select_step9_stage2_mappings,
    validate_step9_stage2_mapping,
)
from app.llm.json_task_validation import build_json_prompt, build_json_prompt_sections
from app.schemas.step_09_structure_variant_and_compound_screening import (
    Step9AvailableField,
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


def _field(candidate_id="cand_a", field_ref="material:mat_a", value_kind="sequence_material", field_type="protein_sequence"):
    return Step9AvailableField(
        candidate_id=candidate_id,
        field_ref=field_ref,
        provider="step_05",
        field_type=field_type,
        value_kind=value_kind,
    )


def _step8_complex_field(candidate_id="cand_a", field_ref="step8_complex_ref:cand_a:0"):
    return Step9AvailableField(
        candidate_id=candidate_id,
        field_ref=field_ref,
        provider="step_08",
        field_type="structure",
        value_kind="complex_structure_ref",
        source_ref="1N8Z",
    )


def _req(tool_name="ESM_generate_protein_sequence", lane_type="protein_design", required=None):
    required = required or ["prompt_sequence"]
    return Step9ToolSchemaRequirement(
        candidate_id="cand_a",
        tool_name=tool_name,
        lane_type=lane_type,  # type: ignore[arg-type]
        required_fields=required,
        schema_source="signature",
        satisfiable_required_fields=required,
        missing_required_fields=[],
        hard_gate_decision="allowed",
        reason="schema_requirements_satisfied",
    )


def _lane(candidate_id="cand_a", tool_name="ESM_generate_protein_sequence"):
    return Step9LaneStatus(
        lane_type="protein_design",
        candidate_id=candidate_id,
        candidate_type="target_antigen",
        status="ready",
        allowed_tools=[tool_name],
        available_field_refs=[f"material:{candidate_id}_mat"],
    )


def _projection(fields=None, reqs=None, lanes=None):
    return {
        "step9_available_fields": fields if fields is not None else [_field()],
        "step9_tool_schema_requirements": reqs if reqs is not None else [_req(required=["task", "prompt_sequence"])],
        "step9_lane_statuses": lanes if lanes is not None else [_lane()],
    }


def _selected(tool_name="ESM_generate_protein_sequence", lane_type="protein_design", reason="selected"):
    return [Step9Stage1SelectionAudit(tool_name=tool_name, lane_type=lane_type, selection_reason=reason)]


def _sections_for(projection, *, selected=None, candidate_id="cand_a", canonical_query="", raw_user_query=""):
    payload = build_step9_stage2_payload(
        candidate_id=candidate_id,
        selected_tools=selected or _selected(),
        readiness_projection=projection,
        canonical_query=canonical_query,
        raw_user_query=raw_user_query,
        step8_downstream_handoff_status=[{"candidate_id": candidate_id, "has_complex_structure": False}],
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
        canonical_query="alpha query",
    )
    stable_b, _, _ = _sections_for(
        _projection(fields=[_field("cand_beta", "material:beta")]),
        candidate_id="cand_beta",
        canonical_query="beta query",
    )
    assert stable_a == stable_b


def test_candidate_specific_data_only_after_stable_prefix(monkeypatch):
    monkeypatch.setattr(step9sel, "signature_schema_for", lambda name: ESM_SEQUENCE_SCHEMA)
    stable, dynamic, full = _sections_for(
        _projection(fields=[_field("cand_SENTINEL", "material:field_SENTINEL")]),
        candidate_id="cand_SENTINEL",
        canonical_query="CANONICAL_SENTINEL",
        raw_user_query="RAW_SENTINEL",
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
    assert "AlphaMissense_get_variant_score" not in stable


def test_stable_prefix_excludes_raw_sequence_pdb_fasta_a3m_and_api_key(monkeypatch):
    monkeypatch.setattr(step9sel, "signature_schema_for", lambda name: ESM_SEQUENCE_SCHEMA)
    stable, dynamic, full = _sections_for(
        _projection(),
        canonical_query=f"{RAW_SEQ} sk-secretvalue123",
        raw_user_query=f"{RAW_PDB}\n>seq\n{RAW_SEQ}\nA3M",
    )
    for forbidden in (RAW_SEQ, "HEADER TEST PDB", "ATOM      1", "sk-secretvalue123"):
        assert forbidden not in stable
        assert forbidden not in dynamic
        assert forbidden not in full


def test_selected_tools_sorted_deterministically(monkeypatch):
    monkeypatch.setattr(step9sel, "signature_schema_for", lambda name: ESM_SEQUENCE_SCHEMA)
    stable, _, _ = _sections_for(
        _projection(reqs=[
            _req("ESM_generate_protein_sequence", "protein_design"),
            _req("AlphaMissense_get_variant_score", "variant_evaluation"),
        ]),
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
        readiness_projection=_projection(
            fields=[_step8_complex_field()],
            reqs=[_req("NvidiaNIM_rfdiffusion", "protein_design", ["input_pdb", "contigs"])],
        ),
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
        available_fields=[_field(field_ref="material:sequence").model_dump()],
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
        available_fields=[_field(field_ref="material:sequence").model_dump()],
    )
    assert valid.can_invoke is True
    assert valid.argument_literals[0].literal_value == "generate"
    assert invalid.can_invoke is True  # deterministic singleton literal repairs it
    assert invalid.argument_literals[0].literal_value == "generate"
    assert "literal_not_allowed:task" in invalid.argument_mapping_reason


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
        available_fields=[_field(field_ref="material:sequence_a").model_dump(), _field(field_ref="material:sequence_b").model_dump()],
    )
    assert mapped.can_invoke is True
    assert [(m.schema_arg, m.field_ref) for m in mapped.argument_mappings] == [
        ("prompt_sequence", "material:sequence_a")
    ]
    assert "duplicate_schema_arg:prompt_sequence" in mapped.argument_mapping_reason
    assert "schema_arg_not_in_full_schema:not_in_schema" in mapped.argument_mapping_reason


def _run_mapping(monkeypatch, *, tool_name, lane_type, schema, fields, response=None):
    monkeypatch.setattr(step9sel, "signature_schema_for", lambda name: schema)
    required = list(schema.get("required") or [])
    return select_step9_stage2_mappings(
        llm=_LLM(response or {"tools": [{
            "tool_name": tool_name,
            "lane_type": lane_type,
            "can_invoke": True,
            "argument_mappings": [
                {"schema_arg": arg, "field_ref": field.field_ref}
                for arg, field in fields.items()
            ],
            "argument_literals": [],
            "missing_required_fields": [],
            "skip_reason": "",
            "argument_mapping_reason": "test mapping",
        }]}),
        readiness_projection=_projection(
            fields=list(fields.values()),
            reqs=[_req(tool_name, lane_type, required)],
        ),
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
            "uniprot_id": _field(field_ref="identifier:uniprot_id:P04626", value_kind="uniprot_id", field_type="identifier"),
            "variant": _field(field_ref="identifier:variant:V777L", value_kind="variant", field_type="variant"),
        },
    )
    assert tool.can_invoke is True
    assert {m.schema_arg for m in tool.argument_mappings} == {"uniprot_id", "variant"}


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
            "pdb_id": _field(field_ref="identifier:pdb_id:1N8Z", value_kind="pdb_id", field_type="structure_identifier"),
            "chain": _field(field_ref="identifier:chain:A", value_kind="chain_id", field_type="chain"),
            "mutation": _field(field_ref="identifier:mutation:V777L", value_kind="mutation", field_type="variant"),
        },
    )
    assert tool.can_invoke is True
    assert {m.schema_arg for m in tool.argument_mappings} == {"pdb_id", "chain", "mutation"}
    assert [(lit.schema_arg, lit.literal_value) for lit in tool.argument_literals] == [
        ("operation", "predict_stability")
    ]


def test_dynamut_predicted_complex_without_true_pdb_id_uninvokable_missing_pdb_id(monkeypatch):
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
            "chain": _field(field_ref="identifier:chain:A", value_kind="chain_id", field_type="chain"),
            "mutation": _field(field_ref="identifier:mutation:V777L", value_kind="mutation", field_type="variant"),
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
    step8_field = Step9AvailableField(
        candidate_id="cand_a",
        field_ref="step8_complex_ref:cand_a:0",
        provider="step_08",
        field_type="structure",
        value_kind="complex_structure_ref",
        source_ref="1N8Z",
    )
    tool = _run_mapping(
        monkeypatch,
        tool_name="NvidiaNIM_rfdiffusion",
        lane_type="protein_design",
        schema=schema,
        fields={"input_pdb": step8_field},
    )
    assert tool.can_invoke is False
    assert "contigs" in tool.missing_required_fields


def test_proteinmpnn_with_true_complex_maps_input_pdb(monkeypatch):
    schema = {
        "type": "object",
        "properties": {"input_pdb": {"type": "string"}},
        "required": ["input_pdb"],
    }
    step8_field = Step9AvailableField(
        candidate_id="cand_a",
        field_ref="step8_complex_ref:cand_a:0",
        provider="step_08",
        field_type="structure",
        value_kind="complex_structure_ref",
        source_ref="1N8Z",
    )
    tool = _run_mapping(
        monkeypatch,
        tool_name="NvidiaNIM_proteinmpnn",
        lane_type="protein_design",
        schema=schema,
        fields={"input_pdb": step8_field},
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
            "pdb_id": _field(field_ref="identifier:pdb_id:1N8Z", value_kind="pdb_id", field_type="structure_identifier"),
            "mutation": _field(field_ref="identifier:mutation:V777L", value_kind="mutation", field_type="variant"),
        },
    )
    assert tool.can_invoke is False
    assert tool.missing_required_fields == ["chain"]


def test_esm_generate_sequence_maps_task_literal_and_prompt_sequence(monkeypatch):
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
        fields={"prompt_sequence": _field(field_ref="material:sequence", value_kind="sequence_material")},
    )
    assert tool.can_invoke is True
    assert tool.argument_mappings[0].schema_arg == "prompt_sequence"
    assert tool.argument_literals[0].literal_value == "generate"


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
                value_kind="sequence_material",
                field_type="protein_sequence",
            )
        },
    )
    blob = json.dumps(tool.model_dump())
    assert RAW_SEQ not in blob
    assert RAW_PDB not in blob
    assert "ToolUniverse" not in blob
