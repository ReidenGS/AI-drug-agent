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
from app.agents.step_06_available_fields import project_candidate_available_fields
from app.mcp.client import LocalMCPClient
from app.schemas.step_02_structured_query import (
    SourceRawRequestRef,
    StructuredQuery,
    TaskIntent,
)
from app.services.input_readiness_service import InputReadinessService
from app.services.intake_service import IntakeService
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
) -> str:
    """Step 1 + a directly-written Step 2 carrying inline heavy/light refs,
    then Step 3 + Step 4 so Step 5 can run."""
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
        referenced_inputs=[
            {"id_type": "antibody_heavy_chain_sequence", "value": HEAVY, "source": "user"},
            {"id_type": "antibody_light_chain_sequence", "value": LIGHT, "source": "user"},
        ],
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


def test_step5_does_not_record_antibody_sequence_missing(
    local_storage, registry_service, workflow_state_service
):
    run_id, _ = _seed_sequence_run(local_storage, registry_service, workflow_state_service)
    cct = _run_step5(local_storage, registry_service, workflow_state_service, run_id)
    ab = _antibody_record(cct)
    assert not [g for g in ab["data_gaps"] if "antibody_sequence_missing" in g]


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
