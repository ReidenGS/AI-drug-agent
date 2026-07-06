"""Step 9 real MCP execution tests (Turn C).

These exercise `StructureAndDesignAgent.run_step_9` end to end: Stage 1
selection -> Stage 2 mapping -> runtime planning -> real value resolution
(`step_09_runtime_execution`) -> actual `mcp_client.call_tool` invocation.
Reuses the same `_seed` / `_mcp` pipeline helpers as
`test_structure_and_design_agent.py` so these run against the real Step
1-6 pipeline, not a synthetic shortcut.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from app.agents import step_09_selection_policy as step9sel
from app.agents.structure_and_design_agent import StructureAndDesignAgent
from app.mcp.client import LocalMCPClient
from app.services.tool_inventory_service import ToolInventoryService
from tests.agents.test_structure_and_design_agent import DEFAULT_XLSX, _mcp, _seed


RAW_PDB_BODY = "HEADER TEST PDB\nATOM      1  N   GLY A   1"


# `tests/agents/conftest.py` forces ToolUniverse offline for every test in
# this package by default (see its docstring). Offline, `signature_schema_for`
# falls back to introspecting each Step 9 tool's LOCAL Python binding — but
# NvidiaNIM_rfdiffusion/proteinmpnn/DynaMut2 are deliberately deferred
# (`_ni(*_a, **_kw)`, no named params), so that fallback yields empty
# properties/required and Stage 2 would reject every real argument mapping
# for reasons that have nothing to do with the runtime resolver under test.
# These are each tool's REAL official ToolUniverse schema (confirmed via a
# live-adapter run outside this offline test default) so Stage 2 behaves
# exactly as it would in production.
_REAL_STEP9_OFFICIAL_SCHEMAS = {
    "NvidiaNIM_rfdiffusion": {
        "type": "object",
        "properties": {
            "contigs": {"type": "string"},
            "input_pdb": {"type": "string"},
            "hotspot_res": {"type": "array"},
            "diffusion_steps": {"type": "integer"},
            "random_seed": {"type": "integer"},
        },
        "required": ["contigs", "input_pdb"],
    },
    "NvidiaNIM_proteinmpnn": {
        "type": "object",
        "properties": {
            "input_pdb": {"type": "string"},
            "ca_only": {"type": "boolean"},
            "use_soluble_model": {"type": "boolean"},
            "sampling_temp": {"type": "array"},
            "num_seq_per_target": {"type": "integer"},
        },
        "required": ["input_pdb"],
    },
    "DynaMut2_predict_stability": {
        "type": "object",
        "properties": {
            "operation": {"type": "string", "enum": ["predict_stability"]},
            "pdb_id": {"type": "string"},
            "chain": {"type": "string"},
            "mutation": {"type": "string"},
        },
        "required": ["operation", "pdb_id", "chain", "mutation"],
    },
    "AlphaMissense_get_variant_score": {
        "type": "object",
        "properties": {"uniprot_id": {"type": "string"}, "variant": {"type": "string"}},
        "required": ["uniprot_id", "variant"],
    },
}


@pytest.fixture(autouse=True)
def _real_step9_official_schemas(monkeypatch):
    original = step9sel.signature_schema_for

    def _patched(name: str):
        return _REAL_STEP9_OFFICIAL_SCHEMAS.get(name) or original(name)

    monkeypatch.setattr(step9sel, "signature_schema_for", _patched)


def _target_candidate_id(local_storage, run_id: str) -> str:
    cct = local_storage.read_json(local_storage.run_key(run_id, "candidate_context_table.json"))
    target = next(c for c in cct["candidate_records"] if c.get("candidate_type") == "target_antigen")
    return target["candidate_id"]


def _write_validated_backbone(local_storage, run_id: str, candidate_id: str) -> str:
    """Write a real backbone file + Step 7/8 artifacts so
    `step8_validated_structure_ref:<candidate_id>` resolves to it."""
    backbone_key = local_storage.run_key(run_id, "uploads", "validated_backbone.pdb")
    local_storage.write_bytes(backbone_key, b"fake validated backbone bytes")
    local_storage.write_json(
        local_storage.run_key(run_id, "prepared_structure_input_package.json"),
        {
            "prepared_structure_inputs": [
                {
                    "candidate_id": candidate_id,
                    "structure_input_id": f"si_{candidate_id}",
                    "structure_refs": [
                        {"source_ref": "mat_backbone", "storage_ref": backbone_key, "structure_format": "pdb"}
                    ],
                    "sequence_refs_for_prediction": [],
                }
            ]
        },
    )
    local_storage.write_json(
        local_storage.run_key(run_id, "structure_prediction_and_interface_results.json"),
        {
            "candidate_structure_results": [
                {
                    "candidate_id": candidate_id,
                    "downstream_handoff": {
                        "validated_structure_ref": "mat_backbone",
                        "has_complex_structure": True,
                    },
                }
            ]
        },
    )
    return backbone_key


def _add_uniprot_variant(local_storage, run_id: str, candidate_id: str) -> None:
    cct_path = local_storage.run_key(run_id, "candidate_context_table.json")
    cct = local_storage.read_json(cct_path)
    for candidate in cct["candidate_records"]:
        if candidate["candidate_id"] == candidate_id:
            candidate.setdefault("identifiers", []).extend(
                [
                    {"id_type": "uniprot_id", "id_value": "P04626"},
                    {"id_type": "variant", "id_value": "V777L"},
                ]
            )
    local_storage.write_json(cct_path, cct)


def _mcp_with_bindings(bindings: dict) -> LocalMCPClient:
    xlsx = os.environ.get("TOOL_INVENTORY_XLSX", str(DEFAULT_XLSX))
    return LocalMCPClient(inventory=ToolInventoryService(xlsx), bindings=bindings)


class _Step9SingleToolLLM:
    """Stage1 selects exactly one tool; Stage2 maps `input_pdb` to the
    validated-backbone field."""

    name = "step9-single-tool"
    model = "step9-single-tool"

    def __init__(self, *, tool_name: str, lane_type: str, field_ref: str, schema_arg: str = "input_pdb"):
        self.tool_name = tool_name
        self.lane_type = lane_type
        self.field_ref = field_ref
        self.schema_arg = schema_arg

    def generate(self, prompt, *, system=None, **kwargs):
        raise NotImplementedError

    def generate_json(self, prompt, *, schema, system=None):
        task = schema.get("task")
        if task == "step9_tool_selection_stage_1":
            return {
                "selections": [
                    {"tool_name": self.tool_name, "lane_type": self.lane_type, "selection_reason": "test"}
                ]
            }
        if task == "step9_tool_schema_mapping_stage_2":
            return {
                "tools": [
                    {
                        "tool_name": self.tool_name,
                        "lane_type": self.lane_type,
                        "can_invoke": True,
                        "argument_mappings": [{"schema_arg": self.schema_arg, "field_ref": self.field_ref}],
                        "argument_literals": [],
                        "missing_required_fields": [],
                        "skip_reason": "",
                        "argument_mapping_reason": "test mapping",
                    }
                ]
            }
        return {}


class _Step9TwoToolLLM:
    """Stage1 selects two tools; Stage2 maps both."""

    name = "step9-two-tool"
    model = "step9-two-tool"

    def __init__(self, *, candidate_id: str):
        self.candidate_id = candidate_id

    def generate(self, prompt, *, system=None, **kwargs):
        raise NotImplementedError

    def generate_json(self, prompt, *, schema, system=None):
        task = schema.get("task")
        if task == "step9_tool_selection_stage_1":
            return {
                "selections": [
                    {
                        "tool_name": "NvidiaNIM_proteinmpnn",
                        "lane_type": "protein_design",
                        "selection_reason": "backbone available",
                    },
                    {
                        "tool_name": "AlphaMissense_get_variant_score",
                        "lane_type": "variant_evaluation",
                        "selection_reason": "variant available",
                    },
                ]
            }
        if task == "step9_tool_schema_mapping_stage_2":
            return {
                "tools": [
                    {
                        "tool_name": "NvidiaNIM_proteinmpnn",
                        "lane_type": "protein_design",
                        "can_invoke": True,
                        "argument_mappings": [
                            {
                                "schema_arg": "input_pdb",
                                "field_ref": f"step8_validated_structure_ref:{self.candidate_id}",
                            }
                        ],
                        "argument_literals": [],
                        "missing_required_fields": [],
                        "skip_reason": "",
                        "argument_mapping_reason": "test mapping",
                    },
                    {
                        "tool_name": "AlphaMissense_get_variant_score",
                        "lane_type": "variant_evaluation",
                        "can_invoke": True,
                        "argument_mappings": [
                            {"schema_arg": "uniprot_id", "field_ref": "identifier:uniprot_id:P04626"},
                            {"schema_arg": "variant", "field_ref": "identifier:variant:V777L"},
                        ],
                        "argument_literals": [],
                        "missing_required_fields": [],
                        "skip_reason": "",
                        "argument_mapping_reason": "test mapping",
                    },
                ]
            }
        return {}


def test_selected_mapped_resolved_proteinmpnn_executes_once_with_real_kwargs(
    local_storage, registry_service, workflow_state_service
):
    run_id = _seed(local_storage, registry_service, workflow_state_service)
    candidate_id = _target_candidate_id(local_storage, run_id)
    backbone_key = _write_validated_backbone(local_storage, run_id, candidate_id)

    captured_calls: list[dict] = []

    def _capture_proteinmpnn(**kwargs):
        captured_calls.append(kwargs)
        return {"status": "success", "output_pdb_ref": "fake_designed_backbone"}

    from app.mcp.tools._registry import _all_bindings

    bindings = dict(_all_bindings())
    bindings["NvidiaNIM_proteinmpnn"] = _capture_proteinmpnn

    agent = StructureAndDesignAgent(
        storage=local_storage,
        registry=registry_service,
        workflow_state=workflow_state_service,
        mcp_client=_mcp_with_bindings(bindings),
        llm=_Step9SingleToolLLM(
            tool_name="NvidiaNIM_proteinmpnn",
            lane_type="protein_design",
            field_ref=f"step8_validated_structure_ref:{candidate_id}",
        ),
    )
    artifact = agent.run_step_9(run_id)

    assert len(captured_calls) == 1
    assert captured_calls[0]["input_pdb"] == backbone_key

    assert artifact.step9_runtime_executed_tools == ["NvidiaNIM_proteinmpnn"]
    assert len(artifact.tool_call_records) == 1
    tc = artifact.tool_call_records[0]
    assert tc.tool_name == "NvidiaNIM_proteinmpnn"
    assert tc.run_status == "success"
    assert tc.step_id == "step_09"
    assert artifact.screening_status == "ok"

    # The real backbone storage path never leaks into the persisted summary.
    assert backbone_key not in json.dumps(tc.tool_input_summary)
    assert backbone_key not in json.dumps(artifact.model_dump())


def test_multiple_selected_mapped_resolved_tools_all_execute(
    local_storage, registry_service, workflow_state_service
):
    run_id = _seed(local_storage, registry_service, workflow_state_service)
    candidate_id = _target_candidate_id(local_storage, run_id)
    _write_validated_backbone(local_storage, run_id, candidate_id)
    _add_uniprot_variant(local_storage, run_id, candidate_id)

    from app.mcp.tools._registry import _all_bindings

    bindings = dict(_all_bindings())
    bindings["NvidiaNIM_proteinmpnn"] = lambda **kw: {"status": "success", "designed_sequences": ["MK"]}

    agent = StructureAndDesignAgent(
        storage=local_storage,
        registry=registry_service,
        workflow_state=workflow_state_service,
        mcp_client=_mcp_with_bindings(bindings),
        llm=_Step9TwoToolLLM(candidate_id=candidate_id),
    )
    artifact = agent.run_step_9(run_id)

    assert set(artifact.step9_runtime_executed_tools) == {
        "NvidiaNIM_proteinmpnn",
        "AlphaMissense_get_variant_score",
    }
    assert len(artifact.tool_call_records) == 2
    assert {tc.run_status for tc in artifact.tool_call_records} == {"success"}
    assert artifact.screening_status == "ok"


def test_one_success_one_failure_reports_partial_status_and_both_records(
    local_storage, registry_service, workflow_state_service
):
    run_id = _seed(local_storage, registry_service, workflow_state_service)
    candidate_id = _target_candidate_id(local_storage, run_id)
    _write_validated_backbone(local_storage, run_id, candidate_id)
    _add_uniprot_variant(local_storage, run_id, candidate_id)

    from app.mcp.tools._registry import _all_bindings

    def _fail_proteinmpnn(**kwargs):
        raise RuntimeError("simulated ToolUniverse upstream failure")

    bindings = dict(_all_bindings())
    bindings["NvidiaNIM_proteinmpnn"] = _fail_proteinmpnn
    # AlphaMissense_get_variant_score keeps its real (non-live mocked-success) binding.

    agent = StructureAndDesignAgent(
        storage=local_storage,
        registry=registry_service,
        workflow_state=workflow_state_service,
        mcp_client=_mcp_with_bindings(bindings),
        llm=_Step9TwoToolLLM(candidate_id=candidate_id),
    )
    artifact = agent.run_step_9(run_id)

    assert len(artifact.tool_call_records) == 2
    statuses = {tc.tool_name: tc.run_status for tc in artifact.tool_call_records}
    assert statuses["NvidiaNIM_proteinmpnn"] == "failed"
    assert statuses["AlphaMissense_get_variant_score"] == "success"
    assert artifact.screening_status == "partial"


def test_unresolved_tool_does_not_execute_and_gets_skipped_audit(
    local_storage, registry_service, workflow_state_service
):
    """Stage2 maps a tool's required arg to a field_ref that cannot be
    resolved to a real value (points at a nonexistent structure ref) — the
    tool must never be invoked, and the skip must be explicit."""
    run_id = _seed(local_storage, registry_service, workflow_state_service)
    candidate_id = _target_candidate_id(local_storage, run_id)
    # No Step7/Step8 artifacts written at all: step8_validated_structure_ref
    # will not even exist as a projected field, so Stage2's mapping targets a
    # field_ref that is not in the projection.

    captured_calls: list[dict] = []

    def _capture_proteinmpnn(**kwargs):
        captured_calls.append(kwargs)
        return {"status": "success"}

    from app.mcp.tools._registry import _all_bindings

    bindings = dict(_all_bindings())
    bindings["NvidiaNIM_proteinmpnn"] = _capture_proteinmpnn

    agent = StructureAndDesignAgent(
        storage=local_storage,
        registry=registry_service,
        workflow_state=workflow_state_service,
        mcp_client=_mcp_with_bindings(bindings),
        llm=_Step9SingleToolLLM(
            tool_name="NvidiaNIM_proteinmpnn",
            lane_type="protein_design",
            field_ref=f"step8_validated_structure_ref:{candidate_id}",
        ),
    )
    artifact = agent.run_step_9(run_id)

    assert captured_calls == []
    assert artifact.tool_call_records == []
    assert artifact.step9_runtime_executed_tools == []
    assert artifact.screening_status == "skipped"
    records = artifact.step9_runtime_execution_records
    assert len(records) == 1
    assert records[0]["tool_name"] == "NvidiaNIM_proteinmpnn"
    assert records[0]["run_status"] == "skipped"
    assert records[0]["unresolved_reasons"]


def test_non_selected_tool_never_executes(local_storage, registry_service, workflow_state_service):
    """Stage1 selects only ProteinMPNN; DynaMut2/AlphaMissense/ESM/rfdiffusion
    must never be invoked even though real inputs exist for them."""
    run_id = _seed(local_storage, registry_service, workflow_state_service)
    candidate_id = _target_candidate_id(local_storage, run_id)
    _write_validated_backbone(local_storage, run_id, candidate_id)
    _add_uniprot_variant(local_storage, run_id, candidate_id)

    from app.mcp.tools._registry import _all_bindings

    called_tools: list[str] = []

    def _tracking_wrapper(tool_name, real_fn):
        def _wrapped(**kwargs):
            called_tools.append(tool_name)
            return real_fn(**kwargs)
        return _wrapped

    bindings = dict(_all_bindings())
    for tool_name in (
        "NvidiaNIM_proteinmpnn",
        "AlphaMissense_get_variant_score",
        "DynaMut2_predict_stability",
        "ESM_generate_protein_sequence",
        "NvidiaNIM_rfdiffusion",
        "ESM_score_variant_sae_batch",
    ):
        bindings[tool_name] = _tracking_wrapper(tool_name, bindings[tool_name])

    agent = StructureAndDesignAgent(
        storage=local_storage,
        registry=registry_service,
        workflow_state=workflow_state_service,
        mcp_client=_mcp_with_bindings(bindings),
        llm=_Step9SingleToolLLM(
            tool_name="NvidiaNIM_proteinmpnn",
            lane_type="protein_design",
            field_ref=f"step8_validated_structure_ref:{candidate_id}",
        ),
    )
    agent.run_step_9(run_id)

    assert called_tools == ["NvidiaNIM_proteinmpnn"]


def test_zinc_and_chembl_never_execute_even_if_llm_tries_to_select_them(
    local_storage, registry_service, workflow_state_service
):
    class _HostileLLM:
        name = "hostile"
        model = "hostile"

        def generate(self, prompt, *, system=None, **kwargs):
            raise NotImplementedError

        def generate_json(self, prompt, *, schema, system=None):
            task = schema.get("task")
            if task == "step9_tool_selection_stage_1":
                return {
                    "selections": [
                        {
                            "tool_name": "ZINC_search_by_smiles",
                            "lane_type": "compound_screening",
                            "selection_reason": "hallucinated",
                        }
                    ]
                }
            return {}

    run_id = _seed(local_storage, registry_service, workflow_state_service)
    agent = StructureAndDesignAgent(
        storage=local_storage,
        registry=registry_service,
        workflow_state=workflow_state_service,
        mcp_client=_mcp(),
        llm=_HostileLLM(),
    )
    artifact = agent.run_step_9(run_id)

    assert artifact.tool_call_records == []
    assert artifact.step9_runtime_executed_tools == []
    rejected = {item["tool_name"]: item["reason"] for item in artifact.step9_stage1_rejected_tools_with_reason}
    assert rejected["ZINC_search_by_smiles"] == "tool_not_in_active_catalog"
    blob = json.dumps(artifact.model_dump())
    assert "ZINC_get_compound" not in blob


def test_privacy_no_raw_pdb_body_or_path_in_normalized_artifact(
    local_storage, registry_service, workflow_state_service
):
    run_id = _seed(local_storage, registry_service, workflow_state_service)
    candidate_id = _target_candidate_id(local_storage, run_id)
    backbone_key = _write_validated_backbone(local_storage, run_id, candidate_id)

    from app.mcp.tools._registry import _all_bindings

    bindings = dict(_all_bindings())
    bindings["NvidiaNIM_proteinmpnn"] = lambda **kw: {
        "status": "success",
        "raw_pdb_body": RAW_PDB_BODY,
    }

    agent = StructureAndDesignAgent(
        storage=local_storage,
        registry=registry_service,
        workflow_state=workflow_state_service,
        mcp_client=_mcp_with_bindings(bindings),
        llm=_Step9SingleToolLLM(
            tool_name="NvidiaNIM_proteinmpnn",
            lane_type="protein_design",
            field_ref=f"step8_validated_structure_ref:{candidate_id}",
        ),
    )
    artifact = agent.run_step_9(run_id)
    tc = artifact.tool_call_records[0]

    normalized_blob = local_storage.read_json(
        local_storage.run_key(run_id, "compound_screening_artifact.json")
    )
    normalized_text = json.dumps(normalized_blob)
    assert RAW_PDB_BODY not in normalized_text
    assert backbone_key not in normalized_text
    assert "ATOM      1" not in normalized_text

    # The raw ToolUniverse payload (including the fake raw PDB body) is only
    # reachable via tool_outputs/step_09/<tool_call_id>.json.
    assert tc.tool_output_ref is not None
    assert "tool_outputs/step_09/" in tc.tool_output_ref.replace("\\", "/")
    raw_output = local_storage.read_json(tc.tool_output_ref)
    assert raw_output["output"]["raw_pdb_body"] == RAW_PDB_BODY
