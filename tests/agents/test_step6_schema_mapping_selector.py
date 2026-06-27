from __future__ import annotations

from typing import Any

from app.agents.step_06_available_fields import project_candidate_available_fields
from app.agents.step_06_capability_registry import STEP_06_CAPABILITY_REGISTRY
from app.agents.step_06_schema_mapping_selector import (
    disclose_step6_tools,
    select_step6_schema_mapped_invocations,
)


SCOPED = {cap.tool_name for cap in STEP_06_CAPABILITY_REGISTRY}
RAW_SEQ = "EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGK"
SMILES = "CCO"
PDB_PATH = "adc_pilot/runs/run_x/inputs/complex.pdb"


def _candidate(*, materials=None, identifiers=None):
    return {
        "candidate_id": "cand_schema_map",
        "candidate_label": "fixture",
        "candidate_type": "adc_construct",
        "materials": materials or [],
        "identifiers": identifiers or [],
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


def _identifier(id_type: str, value: str):
    return {"id_type": id_type, "id_value": value, "source_ids": [], "confidence": 0.9}


def _projection(candidate: dict):
    return project_candidate_available_fields(candidate)


def test_disclosure_smiles_only_shows_compound_and_hides_sequence_structure():
    proj = _projection(_candidate(materials=[_material("m1", "payload_smiles", SMILES)]))
    result = disclose_step6_tools(
        scoped_tool_names=SCOPED,
        modality_summary=proj.modality_summary,
        available_fields=proj.available_fields,
    )
    disclosed = set(result.disclosed_tool_names)
    hidden = {item["tool_name"] for item in result.hidden_tools_with_reason}
    assert "DrugProps_pains_filter" in disclosed
    assert "SwissADME_calculate_adme" in disclosed
    assert "ADMETAI_predict_toxicity" in disclosed
    assert "BindingDB_get_targets_by_compound" in disclosed
    assert "PROSITE_scan_sequence" in hidden
    assert "PDBePISA_get_interfaces" in hidden


def test_disclosure_sequence_only_shows_sequence_and_hides_compound_structure():
    proj = _projection(_candidate(materials=[_material("m1", "antibody_heavy_chain_sequence", RAW_SEQ)]))
    result = disclose_step6_tools(
        scoped_tool_names=SCOPED,
        modality_summary=proj.modality_summary,
        available_fields=proj.available_fields,
    )
    disclosed = set(result.disclosed_tool_names)
    hidden = {item["tool_name"] for item in result.hidden_tools_with_reason}
    assert "PROSITE_scan_sequence" in disclosed
    assert "IEDB_predict_mhci_binding" in disclosed
    assert "DrugProps_pains_filter" in hidden
    assert "PDBePISA_get_interfaces" in hidden


def test_disclosure_uniprot_only_shows_antigen_feature_tools():
    proj = _projection(_candidate(identifiers=[_identifier("uniprot_id", "P04626")]))
    result = disclose_step6_tools(
        scoped_tool_names=SCOPED,
        modality_summary=proj.modality_summary,
        available_fields=proj.available_fields,
    )
    disclosed = set(result.disclosed_tool_names)
    assert "EBIProteins_get_features" in disclosed
    assert "EBIProteins_get_epitopes" in disclosed
    assert "GlyGen_get_glycoprotein" in disclosed
    assert "iPTMnet_get_ptm_sites" in disclosed


def test_disclosure_pdb_id_and_structure_ref_show_structure_category():
    pdb_proj = _projection(_candidate(identifiers=[_identifier("pdb_id", "1N8Z")]))
    pdb_result = disclose_step6_tools(
        scoped_tool_names=SCOPED,
        modality_summary=pdb_proj.modality_summary,
        available_fields=pdb_proj.available_fields,
    )
    assert "PDBePISA_get_interfaces" in set(pdb_result.disclosed_tool_names)

    ref_proj = _projection(
        _candidate(materials=[_material("m_struct", "structure_ref", PDB_PATH, value_format="pdb")])
    )
    ref_result = disclose_step6_tools(
        scoped_tool_names=SCOPED,
        modality_summary=ref_proj.modality_summary,
        available_fields=ref_proj.available_fields,
    )
    assert "PDBePISA_get_interfaces" in set(ref_result.disclosed_tool_names)


def test_disclosure_uploaded_pdb_path_ref_shows_structure_tools():
    uploaded = _projection(
        _candidate(materials=[_material("m_struct", "structure_ref", "/upload/complex.pdb", value_format="pdb")])
    )
    disclosed = set(disclose_step6_tools(
        scoped_tool_names=SCOPED,
        modality_summary=uploaded.modality_summary,
        available_fields=uploaded.available_fields,
    ).disclosed_tool_names)
    assert "PDBePISA_get_interfaces" in disclosed
    assert "ProteinsPlus_profile_structure_quality" in disclosed


def test_disclosure_mixed_inputs_is_union():
    proj = _projection(
        _candidate(
            materials=[
                _material("m_smiles", "payload_smiles", SMILES),
                _material("m_seq", "antibody_heavy_chain_sequence", RAW_SEQ),
            ],
            identifiers=[_identifier("pdb_id", "1N8Z"), _identifier("uniprot_id", "P04626")],
        )
    )
    result = disclose_step6_tools(
        scoped_tool_names=SCOPED,
        modality_summary=proj.modality_summary,
        available_fields=proj.available_fields,
    )
    disclosed = set(result.disclosed_tool_names)
    assert "DrugProps_pains_filter" in disclosed
    assert "PROSITE_scan_sequence" in disclosed
    assert "EBIProteins_get_features" in disclosed
    assert "PDBePISA_get_interfaces" in disclosed


def test_ambiguous_or_empty_modality_fails_open_with_reason():
    proj = _projection(_candidate(materials=[]))
    result = disclose_step6_tools(
        scoped_tool_names=SCOPED,
        modality_summary=proj.modality_summary,
        available_fields=proj.available_fields,
    )
    assert "ambiguous_modality_fail_open" in result.disclosure_tags
    assert len(result.disclosed_tool_names) > 10


class _MCP:
    def __init__(self, tools: list[str]):
        self.tools = tools

    def list_tools(self, *, agent_name: str, step_id: str) -> list[str]:
        return list(self.tools)

    def call_tool(self, **_kw: Any) -> dict:
        return {"run_status": "success", "payload": {}}


class _RecordingLLM:
    name = "rec"
    model = "rec"

    def __init__(self, *, stage1: dict, stage2: dict | None = None):
        self.stage1 = stage1
        self.stage2 = stage2 or {"tools": []}
        self.calls: list[dict] = []

    def generate(self, *_a, **_kw):
        raise NotImplementedError

    def generate_json(self, prompt, *, schema, system=None):
        self.calls.append({"prompt": prompt, "schema": schema, "system": system})
        if schema.get("task") == "step6_schema_mapping_stage_1":
            return self.stage1
        if schema.get("task") == "step6_schema_mapping_stage_2":
            return self.stage2
        return {}


def test_stage1_valid_empty_selection_produces_no_plans():
    proj = _projection(_candidate(materials=[_material("m1", "payload_smiles", SMILES)]))
    plans, _disclosure, audit = select_step6_schema_mapped_invocations(
        agent_name="developability_agent",
        step_id="step_06",
        mcp_client=_MCP(["DrugProps_pains_filter"]),
        llm=_RecordingLLM(stage1={"selections": []}),
        candidate_id="cand_schema_map",
        available_fields=proj.available_fields,
        modality_summary=proj.modality_summary,
    )
    assert plans == {}
    assert audit["stage1_call_status"] == "ok"


def test_stage2_cannot_map_structure_ref_to_pdb_id_required_schema(monkeypatch):
    proj = _projection(
        _candidate(materials=[_material("m_struct", "structure_ref", PDB_PATH, value_format="pdb")])
    )
    import app.agents.step_06_schema_mapping_selector as selector

    monkeypatch.setattr(
        selector,
        "signature_schema_for",
        lambda _name: {
            "type": "object",
            "properties": {"pdb_id": {"type": "string"}},
            "required": ["pdb_id"],
        },
    )
    struct_ref = next(f.field_ref for f in proj.available_fields if f.value_kind == "structure_ref")
    llm = _RecordingLLM(
        stage1={"selections": [{"tool_name": "PDBePISA_get_interfaces", "selection_reason": "structure"}]},
        stage2={"tools": [{
            "tool_name": "PDBePISA_get_interfaces",
            "can_invoke": True,
            "argument_mapping": {"pdb_id": struct_ref},
            "missing_required_fields": [],
            "argument_mapping_reason": "bad pdb mapping",
        }]},
    )
    plans, _disclosure, audit = select_step6_schema_mapped_invocations(
        agent_name="developability_agent",
        step_id="step_06",
        mcp_client=_MCP(["PDBePISA_get_interfaces"]),
        llm=llm,
        candidate_id="cand_schema_map",
        available_fields=proj.available_fields,
        modality_summary=proj.modality_summary,
    )
    plan = plans["structure_interface_quality"][0]
    assert plan.validation_status == "skipped"
    assert "pdb_id" in plan.missing_required_fields
    assert audit["stage2_uninvokable_tools"] == ["PDBePISA_get_interfaces"]


def test_stage2_cannot_use_uploaded_structure_ref_for_pdb_id_schema(monkeypatch):
    proj = _projection(
        _candidate(materials=[_material("m_struct", "structure_ref", "adc_pilot/runs/run_x/inputs/complex.pdb", value_format="pdb")])
    )
    import app.agents.step_06_schema_mapping_selector as selector

    monkeypatch.setattr(
        selector,
        "signature_schema_for",
        lambda _name: {
            "type": "object",
            "properties": {"pdb_id": {"type": "string"}},
            "required": ["pdb_id"],
        },
    )
    ref = next(f.field_ref for f in proj.available_fields if f.value_kind == "structure_ref")
    llm = _RecordingLLM(
        stage1={"selections": [{"tool_name": "PDBePISA_get_interfaces", "selection_reason": "structure"}]},
        stage2={"tools": [{
            "tool_name": "PDBePISA_get_interfaces",
            "can_invoke": True,
            "argument_mapping": {"pdb_id": ref},
            "missing_required_fields": [],
            "argument_mapping_reason": "bad map",
        }]},
    )
    plans, _disclosure, audit = select_step6_schema_mapped_invocations(
        agent_name="developability_agent",
        step_id="step_06",
        mcp_client=_MCP(["PDBePISA_get_interfaces"]),
        llm=llm,
        candidate_id="cand_schema_map",
        available_fields=proj.available_fields,
        modality_summary=proj.modality_summary,
    )
    plan = plans["structure_interface_quality"][0]
    assert plan.validation_status == "skipped"
    assert "pdb_id" in plan.missing_required_fields
    assert audit["stage2_uninvokable_tools"] == ["PDBePISA_get_interfaces"]


def test_stage2_maps_pdb_id_or_path_compatible_structure_schema(monkeypatch):
    proj = _projection(
        _candidate(materials=[_material("m_struct", "structure_ref", "adc_pilot/runs/run_x/inputs/complex.pdb", value_format="pdb")])
    )
    import app.agents.step_06_schema_mapping_selector as selector

    monkeypatch.setattr(
        selector,
        "signature_schema_for",
        lambda _name: {
            "type": "object",
            "properties": {
                "structure_file": {"type": "string"},
            },
            "required": ["structure_file"],
        },
    )
    ref = next(f.field_ref for f in proj.available_fields if f.value_kind == "structure_ref")
    llm = _RecordingLLM(
        stage1={"selections": [{"tool_name": "ProteinsPlus_profile_structure_quality", "selection_reason": "structure"}]},
        stage2={"tools": [{
            "tool_name": "ProteinsPlus_profile_structure_quality",
            "can_invoke": True,
            "argument_mapping": {"structure_file": ref},
            "missing_required_fields": [],
            "argument_mapping_reason": "ok",
        }]},
    )
    plans, _disclosure, _audit = select_step6_schema_mapped_invocations(
        agent_name="developability_agent",
        step_id="step_06",
        mcp_client=_MCP(["ProteinsPlus_profile_structure_quality"]),
        llm=llm,
        candidate_id="cand_schema_map",
        available_fields=proj.available_fields,
        modality_summary=proj.modality_summary,
    )
    plan = plans["structure_interface_quality"][0]
    assert plan.validation_status in {"ok", "warning"}
    assert plan.argument_field_refs["structure_file"] == ref


def test_stage2_maps_pdb_id_identifier_to_pdb_id_schema(monkeypatch):
    proj = _projection(_candidate(identifiers=[_identifier("pdb_id", "1N8Z")]))
    import app.agents.step_06_schema_mapping_selector as selector

    monkeypatch.setattr(
        selector,
        "signature_schema_for",
        lambda _name: {
            "type": "object",
            "properties": {"pdb_id": {"type": "string"}},
            "required": ["pdb_id"],
        },
    )
    pdb_ref = next(f.field_ref for f in proj.available_fields if f.id_type == "pdb_id")
    llm = _RecordingLLM(
        stage1={"selections": [{"tool_name": "PDBePISA_get_interfaces", "selection_reason": "structure"}]},
        stage2={"tools": [{
            "tool_name": "PDBePISA_get_interfaces",
            "can_invoke": True,
            "argument_mapping": {"pdb_id": pdb_ref},
            "missing_required_fields": [],
            "argument_mapping_reason": "ok",
        }]},
    )
    plans, _disclosure, audit = select_step6_schema_mapped_invocations(
        agent_name="developability_agent",
        step_id="step_06",
        mcp_client=_MCP(["PDBePISA_get_interfaces"]),
        llm=llm,
        candidate_id="cand_schema_map",
        available_fields=proj.available_fields,
        modality_summary=proj.modality_summary,
    )
    plan = plans["structure_interface_quality"][0]
    assert plan.validation_status == "ok"
    assert plan.argument_field_refs == {"pdb_id": pdb_ref}
    assert audit["stage2_mapped_tools"] == ["PDBePISA_get_interfaces"]


def test_stage_prompts_do_not_contain_raw_values():
    proj = _projection(
        _candidate(
            materials=[
                _material("m_seq", "antibody_heavy_chain_sequence", RAW_SEQ),
                _material("m_struct", "structure_ref", PDB_PATH, value_format="pdb"),
            ]
        )
    )
    llm = _RecordingLLM(stage1={"selections": []})
    select_step6_schema_mapped_invocations(
        agent_name="developability_agent",
        step_id="step_06",
        mcp_client=_MCP(["PROSITE_scan_sequence", "PDBePISA_get_interfaces"]),
        llm=llm,
        candidate_id="cand_schema_map",
        available_fields=proj.available_fields,
        modality_summary=proj.modality_summary,
    )
    blob = str(llm.calls)
    assert RAW_SEQ not in blob
    assert PDB_PATH not in blob
    assert "complex.pdb" not in blob
