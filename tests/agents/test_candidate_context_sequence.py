"""Step 5 inline antibody heavy/light sequence propagation (developability).

Step 2 emits referenced_inputs with id_type antibody_heavy_chain_sequence /
antibody_light_chain_sequence. Step 5 must materialize them into distinct
candidate materials (inline amino-acid sequence value_format), must NOT
record antibody_sequence_missing, and must NOT call SAbDab/TheraSAbDab with a
sentence-like label. Step 6 must then see both chains as available_fields and
expand into separate heavy/light calls — all without raw sequence leakage
into Step 5/6 normalized artifacts, tool_input_summary, selection_audit, or
the persisted Step 6 summary.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from app.agents.candidate_context_agent import CandidateContextAgent
from app.agents.developability_agent import DevelopabilityAgent
from app.agents.structure_and_design_agent import StructureAndDesignAgent
from app.agents.supervisor_agent import SupervisorAgent
from app.agents.step_06_available_fields import project_candidate_available_fields
from app.agents.step_09_input_projection import project_step9_inputs
from app.agents.step_09_runtime_execution import resolve_step9_field_value
from app.llm.provider import MockLLMProvider
from app.mcp.client import LocalMCPClient
from app.schemas.step_02_structured_query import (
    SourceRawRequestRef,
    StructuredQuery,
    TaskIntent,
)
from app.services.input_readiness_service import InputReadinessService
from app.services.intake_service import IntakeService
from app.services.structured_query_service import StructuredQueryService
from app.services.tool_inventory_service import ToolInventoryService
from app.services.workflow_setup_service import WorkflowSetupService
from app.utils.ids import new_artifact_id
from app.utils.time import now_iso


PROJECT_ROOT = Path(__file__).resolve().parents[2].parent
DEFAULT_XLSX = PROJECT_ROOT / "项目文件" / "ToolUniversity_inventory_v0.2.xlsx"

HEAVY = "EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGK"
LIGHT = "DIQMTQSPSSLSASVGDRVTITCRASQGISSYLNWYQQKPGK"


def _inventory_or_skip() -> ToolInventoryService:
    xlsx = os.environ.get("TOOL_INVENTORY_XLSX", str(DEFAULT_XLSX))
    if not Path(xlsx).exists():
        pytest.skip(f"Inventory xlsx not at {xlsx}")
    return ToolInventoryService(xlsx)


def _seed_sequence_run(
    local_storage, registry_service, workflow_state_service,
    *, antibody_text: str | None = "these antibody heavy and light chain sequences",
    referenced_inputs: list[dict] | None = None,
) -> str:
    """Step 1 + a directly-written Step 2 carrying inline heavy/light refs,
    then Step 3 + Step 4 so Step 5 can run."""
    if referenced_inputs is None:
        referenced_inputs = [
            {"id_type": "antibody_heavy_chain_sequence", "value": HEAVY, "source": "user"},
            {"id_type": "antibody_light_chain_sequence", "value": LIGHT, "source": "user"},
        ]
    rec = IntakeService(local_storage, registry_service, workflow_state_service).submit(
        raw_user_query="Run a developability pre-filter on antibody heavy/light sequences.",
        user_provided_context={},
    )
    run_id = rec.run_id
    sq = StructuredQuery(
        run_id=run_id,
        parsed_at=now_iso(),
        source_raw_request_ref=SourceRawRequestRef(
            raw_request_record_id=registry_service.get(run_id).active_artifacts.raw_request_record_id
        ),
        task_intent=TaskIntent(
            task_type="developability_assessment",
            primary_intent="developability_assessment",
        ),
        mentioned_entities={"antibody_candidate_text": antibody_text},
        referenced_inputs=referenced_inputs,
        missing_slots=[],
        canonical_query="developability/liability pre-filter for antibody heavy/light sequences",
    )
    sq_id = new_artifact_id("structured_query")
    local_storage.write_json(
        local_storage.run_key(run_id, "inputs/structured_query.json"),
        {"artifact_id": sq_id, **sq.model_dump()},
    )
    registry_service.update_active(run_id, structured_query_id=sq_id)
    workflow_state_service.mark(run_id, "step_02", "completed")
    InputReadinessService(local_storage, registry_service, workflow_state_service).check(run_id)
    WorkflowSetupService(local_storage, registry_service, workflow_state_service).plan(run_id)
    return run_id, sq_id


def _run_step5(local_storage, registry_service, workflow_state_service, run_id) -> dict:
    inventory = _inventory_or_skip()
    CandidateContextAgent(
        storage=local_storage, registry=registry_service,
        workflow_state=workflow_state_service,
        mcp_client=LocalMCPClient(inventory=inventory),
    ).run(run_id)
    return local_storage.read_json(
        local_storage.run_key(run_id, "candidate_context_table.json")
    )


def _run_step5_without_external_tools(
    local_storage, registry_service, workflow_state_service, run_id
) -> dict:
    CandidateContextAgent(
        storage=local_storage,
        registry=registry_service,
        workflow_state=workflow_state_service,
        mcp_client=LocalMCPClient(),
    ).run(run_id)
    return local_storage.read_json(
        local_storage.run_key(run_id, "candidate_context_table.json")
    )


def _antibody_record(cct: dict) -> dict:
    abs_ = [c for c in cct["candidate_records"] if c["candidate_type"] == "antibody"]
    assert abs_, "expected an antibody candidate record"
    return abs_[0]


def test_heavy_light_refs_become_two_distinct_materials(
    local_storage, registry_service, workflow_state_service
):
    run_id, sq_id = _seed_sequence_run(local_storage, registry_service, workflow_state_service)
    cct = _run_step5(local_storage, registry_service, workflow_state_service, run_id)
    ab = _antibody_record(cct)
    seq_mats = [
        m for m in ab["materials"]
        if m["material_type"] in {"antibody_heavy_chain_sequence", "antibody_light_chain_sequence"}
    ]
    assert {m["material_type"] for m in seq_mats} == {
        "antibody_heavy_chain_sequence", "antibody_light_chain_sequence"
    }
    for m in seq_mats:
        assert m["value_format"] == "amino_acid_sequence"
        assert m["role"] == "antibody_sequence_reference"
        assert m["role_status"] == "explicit"
    # source_records include the Step 2 structured_query artifact id.
    assert sq_id in ab["source_records"]


def test_heavy_ref_only_becomes_heavy_material(local_storage, registry_service, workflow_state_service):
    run_id, _ = _seed_sequence_run(
        local_storage, registry_service, workflow_state_service,
        referenced_inputs=[{"id_type": "antibody_heavy_chain_sequence", "value": HEAVY, "source": "user"}],
        antibody_text="trastuzumab",
    )
    cct = _run_step5(local_storage, registry_service, workflow_state_service, run_id)
    ab = _antibody_record(cct)
    mat_types = {m["material_type"] for m in ab["materials"] if m["value_format"] == "amino_acid_sequence"}
    assert mat_types == {"antibody_heavy_chain_sequence"}
    assert not any(m["material_type"] == "antibody_light_chain_sequence" for m in ab["materials"])


def test_light_ref_only_becomes_light_material(local_storage, registry_service, workflow_state_service):
    run_id, _ = _seed_sequence_run(
        local_storage, registry_service, workflow_state_service,
        referenced_inputs=[{"id_type": "antibody_light_chain_sequence", "value": LIGHT, "source": "user"}],
        antibody_text="trastuzumab",
    )
    cct = _run_step5(local_storage, registry_service, workflow_state_service, run_id)
    ab = _antibody_record(cct)
    mat_types = {m["material_type"] for m in ab["materials"] if m["value_format"] == "amino_acid_sequence"}
    assert mat_types == {"antibody_light_chain_sequence"}
    assert not any(m["material_type"] == "antibody_heavy_chain_sequence" for m in ab["materials"])


@pytest.mark.parametrize("sequence", ["ACDE", "ACDEFGHIKLMNPQRSTVWY" * 40])
def test_explicit_target_sequence_fixture_reaches_step7_and_step9_runtime(
    local_storage, registry_service, workflow_state_service, sequence
):
    class _ExplicitTargetSequenceProvider:
        """Test-only semantic fixture; production normalization stays real."""

        name = "test-only-explicit-target-sequence"
        model = "test-only"

        def __init__(self):
            self.inner = MockLLMProvider()
            self.calls = []

        def generate_json(self, prompt, *, schema, system=None):
            self.calls.append(
                json.loads(json.dumps({"prompt": prompt, "schema": schema}))
            )
            result = self.inner.generate_json(prompt, schema=schema, system=system)
            result["task_intent"] = {
                "task_type": "structure_preparation",
                "task_type_confidence": 1.0,
                "modality": "ADC",
                "modality_confidence": 1.0,
                "user_goal_summary": "Analyze an explicitly typed target sequence.",
                "primary_intent": "structure_analysis",
                "primary_intent_confidence": 1.0,
                "secondary_intents": [],
            }
            result["referenced_inputs"] = [
                {"id_type": "target_sequence", "value": sequence, "source": "user"}
            ]
            result["missing_slots"] = [
                slot
                for slot in result.get("missing_slots") or []
                if slot.get("slot_name") != "structure_or_sequence"
            ]
            result["response"] = None
            return result

    provider = _ExplicitTargetSequenceProvider()
    raw = IntakeService(
        local_storage, registry_service, workflow_state_service
    ).submit(
        raw_user_query=f"Analyze target protein sequence: {sequence}",
        user_provided_context={},
    )
    run_id = raw.run_id
    structured = StructuredQueryService(
        local_storage,
        registry_service,
        workflow_state_service,
        SupervisorAgent(llm=provider),
    ).parse(run_id)
    assert sequence in json.dumps(provider.calls)
    assert structured.referenced_inputs == [
        {"id_type": "target_sequence", "value": sequence, "source": "user"}
    ]
    readiness = InputReadinessService(
        local_storage, registry_service, workflow_state_service
    ).check(run_id)
    assert readiness.input_readiness_status == "ready", readiness.model_dump()
    WorkflowSetupService(
        local_storage, registry_service, workflow_state_service
    ).plan(run_id)
    cct = _run_step5_without_external_tools(
        local_storage, registry_service, workflow_state_service, run_id
    )
    target = next(
        item for item in cct["candidate_records"]
        if item["candidate_type"] == "target_antigen"
    )
    material = next(
        item for item in target["materials"]
        if item["material_type"] == "target_sequence"
    )
    assert material["value"] == sequence
    assert material["value_format"] == "amino_acid_sequence"
    assert cct.get("tool_call_records") == []

    step7 = StructureAndDesignAgent(
        storage=local_storage,
        registry=registry_service,
        workflow_state=workflow_state_service,
        mcp_client=LocalMCPClient(),
    ).run_step_7(run_id)
    prepared = next(
        item for item in step7.prepared_structure_inputs
        if item.candidate_id == target["candidate_id"]
    )
    sequence_ref = next(
        item for item in prepared.sequence_refs_for_prediction
        if item.source_ref == material["material_id"]
    )
    assert sequence_ref.sequence_value_status == "inline"
    assert sequence_ref.prediction_input_kind == "amino_acid_sequence"
    assert prepared.input_case == "sequence_only_input"
    assert prepared.missing_metadata_flags == []

    projection = project_step9_inputs(
        candidate_context_table=cct,
        prepared_structure_input_package=step7.model_dump(mode="json"),
        structure_prediction_and_interface_results=None,
    )
    field = next(
        item for item in projection["input_fields"]
        if item.field_ref == f"material:{material['material_id']}"
    )
    assert field.status == "available"
    assert field.missing_reason is None
    assert field.supports_tool_args == ["sequence"]
    value, error = resolve_step9_field_value(
        field.model_dump(mode="json"),
        candidate_context_table=cct,
        prepared_inputs=step7.model_dump(mode="json")["prepared_structure_inputs"],
        step8_result={},
        storage=local_storage,
    )
    assert error is None
    assert value == sequence
    tool_statuses = [
        (record.tool_name, record.run_status, record.error_message)
        for record in step7.structure_tool_call_records
    ]
    assert sum(status == "dependency_unavailable" for _, status, _ in tool_statuses) == 1
    assert sum(status == "skipped" for _, status, _ in tool_statuses) == 6
    assert not any(status in {"success", "failed"} for _, status, _ in tool_statuses)
    assert sequence_ref.msa_status == "dependency_unavailable"
    assert step7.structure_preparation_status == "partial"
    assert step7.preparation_warnings == [
        {
            "tool_name": "NvidiaNIM_msa_search",
            "run_status": "dependency_unavailable",
            "reason": "wrapper_not_wired",
            "chain_role": "antigen",
        }
    ]
    unavailable = next(
        record
        for record in step7.structure_tool_call_records
        if record.run_status == "dependency_unavailable"
    )
    assert unavailable.tool_name == "NvidiaNIM_msa_search"
    assert unavailable.tool_input_summary["routing_decision"] == "selected"
    assert unavailable.error_message == "wrapper_not_wired"


def test_step5_does_not_record_antibody_sequence_missing(
    local_storage, registry_service, workflow_state_service
):
    run_id, _ = _seed_sequence_run(local_storage, registry_service, workflow_state_service)
    cct = _run_step5(local_storage, registry_service, workflow_state_service, run_id)
    ab = _antibody_record(cct)
    assert not [g for g in ab["data_gaps"] if "antibody_sequence_missing" in g]


def test_antibody_sequence_reference_referenced_input_is_not_materialized(
    local_storage, registry_service, workflow_state_service
):
    run_id, _ = _seed_sequence_run(
        local_storage,
        registry_service,
        workflow_state_service,
        referenced_inputs=[
            {
                "id_type": "antibody_sequence_reference",
                "value": HEAVY + LIGHT,
                "source": "user",
            },
        ],
        antibody_text="trastuzumab",
    )
    cct = _run_step5(local_storage, registry_service, workflow_state_service, run_id)
    ab = _antibody_record(cct)

    assert all(
        m["material_type"]
        not in {"antibody_sequence_reference", "antibody_heavy_chain_sequence", "antibody_light_chain_sequence"}
        for m in ab["materials"]
    )
    assert not any(HEAVY in gap or LIGHT in gap for gap in ab["data_gaps"])
    assert any(
        "antibody_sequence_role_unresolved" in note for note in ab["context_notes"]
    )
    assert any(
        "antibody_sequence_chain_role_unresolved" in gap or "heavy_or_light" in gap
        for gap in ab["data_gaps"]
    )


def test_step5_skips_name_lookup_for_sentence_label(
    local_storage, registry_service, workflow_state_service
):
    run_id, _ = _seed_sequence_run(local_storage, registry_service, workflow_state_service)
    cct = _run_step5(local_storage, registry_service, workflow_state_service, run_id)
    ab = _antibody_record(cct)
    # No antibody_name material → no name-driven SAbDab/TheraSAbDab query.
    assert not any(m["material_type"] == "antibody_name" for m in ab["materials"])
    called = {tc["tool_name"] for tc in cct.get("tool_call_records", [])}
    assert not any("SAbDab" in t for t in called)
    assert any("antibody_name_lookup_skipped" in n for n in ab["context_notes"])


@pytest.mark.parametrize(
    "label",
    [
        "antibody protein sequences",
        "antibody heavy and light chain sequence developability pre-filter",
    ],
)
def test_step5_skips_generic_sequence_task_labels_as_antibody_names(
    local_storage, registry_service, workflow_state_service, label
):
    run_id, _ = _seed_sequence_run(
        local_storage,
        registry_service,
        workflow_state_service,
        antibody_text=label,
    )
    cct = _run_step5(local_storage, registry_service, workflow_state_service, run_id)
    ab = _antibody_record(cct)
    assert any(m["material_type"] == "antibody_heavy_chain_sequence" for m in ab["materials"])
    assert any(m["material_type"] == "antibody_light_chain_sequence" for m in ab["materials"])
    assert not any(m["material_type"] == "antibody_name" for m in ab["materials"])
    called = {tc["tool_name"] for tc in cct.get("tool_call_records", [])}
    assert "SAbDab_search_structures" not in called
    assert "TheraSAbDab_search_therapeutics" not in called


def test_step6_available_fields_expose_both_chains_as_digests(
    local_storage, registry_service, workflow_state_service
):
    run_id, _ = _seed_sequence_run(local_storage, registry_service, workflow_state_service)
    cct = _run_step5(local_storage, registry_service, workflow_state_service, run_id)
    ab = _antibody_record(cct)
    proj = project_candidate_available_fields(ab)
    chain_roles = {
        f.chain_role for f in proj.available_fields if f.field_type == "protein_sequence"
    }
    assert {"heavy", "light"} <= chain_roles
    assert proj.modality_summary.has_antibody_heavy_sequence
    assert proj.modality_summary.has_antibody_light_sequence
    # Raw sequence is never in the LLM-safe projection (digest only).
    blob = proj.model_dump_json()
    assert HEAVY not in blob and LIGHT not in blob


def test_step6_chain_expansion_separate_calls_no_sequence_leak(
    local_storage, registry_service, workflow_state_service
):
    run_id, _ = _seed_sequence_run(local_storage, registry_service, workflow_state_service)
    inventory = _inventory_or_skip()
    _run_step5(local_storage, registry_service, workflow_state_service, run_id)

    seen_sequences: list[str] = []

    def _prosite(**kwargs):
        if "sequence" in kwargs:
            seen_sequences.append(kwargs["sequence"])
        return {"status": "mocked", "motifs": []}

    DevelopabilityAgent(
        storage=local_storage, registry=registry_service,
        workflow_state=workflow_state_service,
        mcp_client=LocalMCPClient(
            inventory=inventory, bindings={"PROSITE_scan_sequence": _prosite}
        ),
    ).run(run_id)

    # Separate heavy + light resolution happened at runtime (two distinct
    # raw sequences were injected into the MCP call).
    assert HEAVY in seen_sequences
    assert LIGHT in seen_sequences

    # No raw sequence in the persisted Step 6 normalized summary / audit.
    summary = local_storage.read_json(
        local_storage.run_key(run_id, "structured_liability_summary.json")
    )
    blob = json.dumps(summary)
    assert HEAVY not in blob
    assert LIGHT not in blob
