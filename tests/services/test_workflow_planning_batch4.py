"""Step 4 batch-4 deterministic planning + gating regression.

Adds coverage requested by
`\u9879\u76ee\u6587\u4ef6/Step1_4_Orchestration_Component_Plan_v0.1.md §Step 4`:

- Lane-flag canonical names — `antibody_lane`, `compound_lane`,
  `structure_lane`, `evidence_lane`, `patent_lane` — appear on the
  planned_steps where the doc says they should.
- Step 13/14 block when there's literally no query context.
- Step 13/14 keep running when ANY context exists (target keyword,
  candidate name, payload, referenced ID, or user_goal_summary).
- Plan never invents artifacts; gated steps leave the registry clean.
- Static + runtime guarantees that Step 4 imports zero LLM / MCP / A2A
  code and never builds the ToolUniverse singleton.
"""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest

from app.services.input_readiness_service import InputReadinessService
from app.services.intake_service import IntakeService
from app.services.workflow_setup_service import (
    WorkflowSetupService,
    execution_decision,
    planned_step_for,
)
from app.schemas.step_02_structured_query import (
    MentionedEntities,
    SourceRawRequestRef,
    StructuredQuery,
    TaskIntent,
)
from app.utils.ids import new_artifact_id
from app.utils.time import now_iso


def _bootstrap(
    local_storage, registry_service, workflow_state_service,
    *,
    target=None, candidate=None, payload=None, linker=None,
    raw_context=None, referenced_inputs=None, user_goal_summary="",
    raw_user_query="HER2 ADC",
):
    intake = IntakeService(local_storage, registry_service, workflow_state_service)
    rec = intake.submit(
        raw_user_query=raw_user_query,
        user_provided_context=raw_context or {},
    )
    reg = registry_service.get(rec.run_id)
    sq = StructuredQuery(
        run_id=rec.run_id,
        parsed_at=now_iso(),
        source_raw_request_ref=SourceRawRequestRef(
            raw_request_record_id=reg.active_artifacts.raw_request_record_id
        ),
        task_intent=TaskIntent(
            task_type="adc_design",
            modality="ADC",
            modality_confidence=0.9,
            user_goal_summary=user_goal_summary,
        ),
        mentioned_entities=MentionedEntities(
            target_or_antigen_text=target,
            antibody_candidate_text=candidate,
            payload_text=payload,
            linker_text=linker,
        ),
        referenced_inputs=referenced_inputs or [],
    )
    sq_id = new_artifact_id("structured_query")
    local_storage.write_json(
        local_storage.run_key(rec.run_id, "inputs/structured_query.json"),
        {"artifact_id": sq_id, **sq.model_dump()},
    )
    registry_service.update_active(rec.run_id, structured_query_id=sq_id)
    workflow_state_service.mark(rec.run_id, "step_02", "completed")
    InputReadinessService(local_storage, registry_service, workflow_state_service).check(rec.run_id)
    return rec.run_id


def _plan(local_storage, registry_service, workflow_state_service, run_id):
    return WorkflowSetupService(
        local_storage, registry_service, workflow_state_service
    ).plan(run_id)


# ── canonical lane_flags on planned_steps ──────────────────────────────────


def test_step6_partial_emits_antibody_and_compound_lane_flags(
    local_storage, registry_service, workflow_state_service
):
    """Antibody missing → Step 6 partial with antibody_lane=False, compound_lane=True."""
    run_id = _bootstrap(
        local_storage, registry_service, workflow_state_service,
        target="HER2", payload="MMAE", linker="vc",
        raw_context={"target_or_antigen_text": "HER2", "payload_linker_text": "vc-MMAE"},
    )
    plan = _plan(local_storage, registry_service, workflow_state_service, run_id)
    s6 = planned_step_for(plan, "step_06_developability")
    assert s6.planned_status == "partial"
    assert s6.lane_flags.get("antibody_lane") is False
    assert s6.lane_flags.get("compound_lane") is True


def test_step6_partial_emits_compound_lane_false_when_payload_missing(
    local_storage, registry_service, workflow_state_service
):
    run_id = _bootstrap(
        local_storage, registry_service, workflow_state_service,
        target="HER2", candidate="Trastuzumab",
        raw_context={"target_or_antigen_text": "HER2", "candidate_text": "Trastuzumab"},
    )
    plan = _plan(local_storage, registry_service, workflow_state_service, run_id)
    s6 = planned_step_for(plan, "step_06_developability")
    assert s6.planned_status == "partial"
    assert s6.lane_flags.get("compound_lane") is False
    assert s6.lane_flags.get("antibody_lane") is True


def test_step7_to_9_partial_carries_structure_lane_false(
    local_storage, registry_service, workflow_state_service
):
    run_id = _bootstrap(
        local_storage, registry_service, workflow_state_service,
        target="HER2", candidate="Trastuzumab", payload="MMAE", linker="vc",
        raw_context={
            "target_or_antigen_text": "HER2",
            "candidate_text": "Trastuzumab",
            "payload_linker_text": "vc-MMAE",
        },
    )
    plan = _plan(local_storage, registry_service, workflow_state_service, run_id)
    for sid in (
        "step_07_structure_input",
        "step_08_structure_evaluation",
        "step_09_structure_design",
    ):
        p = planned_step_for(plan, sid)
        assert p.planned_status == "partial"
        assert p.lane_flags.get("structure_lane") is False


def test_step5_lane_flags_carry_antibody_lane_when_candidate_present(
    local_storage, registry_service, workflow_state_service
):
    run_id = _bootstrap(
        local_storage, registry_service, workflow_state_service,
        target="HER2", candidate="Trastuzumab", payload="MMAE", linker="vc",
        raw_context={
            "target_or_antigen_text": "HER2",
            "candidate_text": "Trastuzumab",
            "payload_linker_text": "vc-MMAE",
        },
    )
    plan = _plan(local_storage, registry_service, workflow_state_service, run_id)
    s5 = planned_step_for(plan, "step_05_candidate_context")
    assert s5.planned_status == "run"
    assert s5.lane_flags.get("antibody_lane") is True
    assert s5.lane_flags.get("antibody_discovery_lane") is False
    assert s5.lane_flags.get("compound_lane") is True


# ── Step 13 / 14 carry evidence_lane / patent_lane ─────────────────────────


def test_step13_planned_step_carries_evidence_lane_true_when_context_present(
    local_storage, registry_service, workflow_state_service
):
    run_id = _bootstrap(
        local_storage, registry_service, workflow_state_service,
        target="HER2", candidate="Trastuzumab", payload="MMAE", linker="vc",
        raw_context={
            "target_or_antigen_text": "HER2",
            "candidate_text": "Trastuzumab",
            "payload_linker_text": "vc-MMAE",
        },
    )
    plan = _plan(local_storage, registry_service, workflow_state_service, run_id)
    s13 = planned_step_for(plan, "step_13_evidence")
    s14 = planned_step_for(plan, "step_14_patent_ip")
    assert s13.planned_status == "run"
    assert s14.planned_status == "run"
    assert s13.lane_flags.get("evidence_lane") is True
    assert s14.lane_flags.get("patent_lane") is True
    assert s13.lane_flags.get("compound_lane") is True
    assert s14.lane_flags.get("compound_lane") is True
    assert s14.lane_flags.get("regulatory_lane") is True


# ── Step 13 / 14 block when no query context (referenced-input only path) ──


def _make_minimal_run_with_referenced_id_only(
    local_storage, registry_service, workflow_state_service, *, referenced_inputs
):
    """A run whose readiness is `needs_user_input` because target is
    missing — but we manually flip the readiness back to `ready` after
    Step 3 so we can drive Step 4 directly against the "no entities but
    some referenced IDs" case."""
    intake = IntakeService(local_storage, registry_service, workflow_state_service)
    rec = intake.submit(
        raw_user_query="HER2 ADC reference",
        user_provided_context={
            "target_or_antigen_text": "HER2",
            "candidate_text": "Trastuzumab",
            "payload_linker_text": "vc-MMAE",
        },
    )
    reg = registry_service.get(rec.run_id)
    sq = StructuredQuery(
        run_id=rec.run_id,
        parsed_at=now_iso(),
        source_raw_request_ref=SourceRawRequestRef(
            raw_request_record_id=reg.active_artifacts.raw_request_record_id
        ),
        task_intent=TaskIntent(
            task_type="adc_design", modality="ADC", modality_confidence=0.9,
        ),
        mentioned_entities=MentionedEntities(target_or_antigen_text=None),
        referenced_inputs=referenced_inputs,
    )
    sq_id = new_artifact_id("structured_query")
    local_storage.write_json(
        local_storage.run_key(rec.run_id, "inputs/structured_query.json"),
        {"artifact_id": sq_id, **sq.model_dump()},
    )
    registry_service.update_active(rec.run_id, structured_query_id=sq_id)
    workflow_state_service.mark(rec.run_id, "step_02", "completed")
    InputReadinessService(
        local_storage, registry_service, workflow_state_service
    ).check(rec.run_id)
    # Hand-edit readiness to `needs_user_input` (target_present remained
    # True from raw context) so Step 4 doesn't raise on entry.
    readiness_key = local_storage.run_key(rec.run_id, "inputs/input_readiness_status.json")
    readiness = local_storage.read_json(readiness_key)
    readiness["input_readiness_status"] = "needs_user_input"
    local_storage.write_json(readiness_key, readiness)
    return rec.run_id


def test_step13_14_blocked_when_no_query_context_at_all(
    local_storage, registry_service, workflow_state_service
):
    """No entities, no referenced IDs, no goal summary → evidence/patent blocked."""
    intake = IntakeService(local_storage, registry_service, workflow_state_service)
    # target_or_antigen_text is supplied raw so readiness isn't blocked,
    # but everything else is empty so query_context is False at Step 4.
    rec = intake.submit(
        raw_user_query="",
        user_provided_context={"target_or_antigen_text": "HER2"},
    )
    reg = registry_service.get(rec.run_id)
    sq = StructuredQuery(
        run_id=rec.run_id,
        parsed_at=now_iso(),
        source_raw_request_ref=SourceRawRequestRef(
            raw_request_record_id=reg.active_artifacts.raw_request_record_id
        ),
        task_intent=TaskIntent(
            task_type="adc_design", modality="ADC", modality_confidence=0.9,
            user_goal_summary="",  # empty
        ),
        mentioned_entities=MentionedEntities(
            target_or_antigen_text=None, antibody_candidate_text=None,
            payload_text=None, linker_text=None,
        ),
        referenced_inputs=[],
    )
    sq_id = new_artifact_id("structured_query")
    local_storage.write_json(
        local_storage.run_key(rec.run_id, "inputs/structured_query.json"),
        {"artifact_id": sq_id, **sq.model_dump()},
    )
    registry_service.update_active(rec.run_id, structured_query_id=sq_id)
    workflow_state_service.mark(rec.run_id, "step_02", "completed")
    InputReadinessService(
        local_storage, registry_service, workflow_state_service
    ).check(rec.run_id)
    # Force readiness to needs_user_input AND clear target presence so
    # Step 4 sees the no-query-context branch (target was in raw context
    # but we want to drive the worst case of "no signal anywhere").
    readiness_key = local_storage.run_key(
        rec.run_id, "inputs/input_readiness_status.json"
    )
    readiness = local_storage.read_json(readiness_key)
    readiness["input_readiness_status"] = "needs_user_input"
    presence = readiness.get("basic_adc_input_presence") or {}
    for k in (
        "target_or_antigen_present",
        "antibody_candidate_present",
        "payload_present",
        "linker_present",
        "structure_or_sequence_present",
    ):
        presence[k] = False
    readiness["basic_adc_input_presence"] = presence
    local_storage.write_json(readiness_key, readiness)

    plan = WorkflowSetupService(
        local_storage, registry_service, workflow_state_service
    ).plan(rec.run_id)

    s13 = planned_step_for(plan, "step_13_evidence")
    s14 = planned_step_for(plan, "step_14_patent_ip")
    assert s13.planned_status == "blocked"
    assert s14.planned_status == "blocked"
    assert s13.lane_flags.get("evidence_lane") is False
    assert s14.lane_flags.get("patent_lane") is False
    # Both steps recorded in skipped_steps too.
    assert "step_13_evidence" in plan.skipped_step_ids
    assert "step_14_patent_ip" in plan.skipped_step_ids


def test_step13_14_run_when_only_referenced_id_present(
    local_storage, registry_service, workflow_state_service
):
    """Just a referenced PubChem CID is enough query context — evidence
    and patent can still search by it."""
    run_id = _make_minimal_run_with_referenced_id_only(
        local_storage, registry_service, workflow_state_service,
        referenced_inputs=[{
            "id_type": "pubchem_cid", "value": "2244", "source": "raw_request_text",
        }],
    )
    plan = WorkflowSetupService(
        local_storage, registry_service, workflow_state_service
    ).plan(run_id)
    s13 = planned_step_for(plan, "step_13_evidence")
    s14 = planned_step_for(plan, "step_14_patent_ip")
    assert s13.planned_status == "run"
    assert s14.planned_status == "run"
    assert s13.lane_flags.get("evidence_lane") is True
    assert s14.lane_flags.get("patent_lane") is True


# ── execution_decision contracts for skip / blocked / partial / run ────────


@pytest.mark.parametrize(
    "planned_status,allow",
    [("run", True), ("partial", True), ("skip", False), ("blocked", False)],
)
def test_execution_decision_per_step_planned_status_matrix(planned_status, allow):
    plan = {
        "plan_status": "ready_to_execute",
        "planned_steps": [
            {"step_id": "step_05_candidate_context",
             "planned_status": planned_status,
             "reason": "test", "required_artifact_refs": [], "lane_flags": {}},
        ],
    }
    d = execution_decision(plan, "step_05_candidate_context")
    assert d.allow is allow
    assert d.planned_status == planned_status


# ── Step 4 is deterministic — no LLM / MCP / A2A imports anywhere ──────────


def test_workflow_setup_service_module_has_no_llm_or_mcp_imports():
    module_path = Path(
        importlib.import_module("app.services.workflow_setup_service").__file__
    )
    tree = ast.parse(module_path.read_text())
    bad: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if mod.startswith(("app.llm", "app.mcp", "app.a2a")):
                bad.append(f"from {mod} import …")
            if mod in {"openai", "anthropic", "google.genai", "google_genai"}:
                bad.append(f"from {mod} import …")
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith(("app.llm", "app.mcp", "app.a2a")):
                    bad.append(f"import {alias.name}")
                if alias.name in {"openai", "anthropic", "google.genai"}:
                    bad.append(f"import {alias.name}")
    assert bad == [], f"Step 4 must not import LLM/MCP/A2A: {bad}"


def test_workflow_plan_does_not_build_tooluniverse_singleton(
    local_storage, registry_service, workflow_state_service, monkeypatch
):
    from app.mcp import tooluniverse_adapter

    tooluniverse_adapter._reset_for_tests()
    sentinel = {"built": False}

    def _explode():
        sentinel["built"] = True
        raise AssertionError("Step 4 must not build the ToolUniverse singleton")

    monkeypatch.setattr(tooluniverse_adapter, "_get_universe", _explode)

    run_id = _bootstrap(
        local_storage, registry_service, workflow_state_service,
        target="HER2", candidate="Trastuzumab", payload="MMAE", linker="vc",
        raw_context={
            "target_or_antigen_text": "HER2",
            "candidate_text": "Trastuzumab",
            "payload_linker_text": "vc-MMAE",
        },
    )
    WorkflowSetupService(
        local_storage, registry_service, workflow_state_service
    ).plan(run_id)
    assert sentinel["built"] is False


# ── gated steps do not produce artifacts ───────────────────────────────────


def test_gated_step5_leaves_registry_unchanged(
    local_storage, registry_service, workflow_state_service
):
    """When the plan blocks Step 5, the registry must NOT gain a
    candidate_context_table_id and the workflow_state must read `skipped`."""
    run_id = _bootstrap(
        local_storage, registry_service, workflow_state_service,
        target="HER2", candidate="Trastuzumab", payload="MMAE", linker="vc",
        raw_context={
            "target_or_antigen_text": "HER2",
            "candidate_text": "Trastuzumab",
            "payload_linker_text": "vc-MMAE",
        },
    )
    plan = WorkflowSetupService(
        local_storage, registry_service, workflow_state_service
    ).plan(run_id)
    plan_dict = plan.model_dump()
    for p in plan_dict["planned_steps"]:
        if p["step_id"] == "step_05_candidate_context":
            p["planned_status"] = "blocked"
            p["reason"] = "test override"
    local_storage.write_json(
        local_storage.run_key(run_id, "inputs/run_step_plan.json"),
        {"artifact_id": "rp_x", **plan_dict},
    )

    from app.graph.nodes import make_node_step_05
    from app.mcp.client import LocalMCPClient

    node = make_node_step_05(
        local_storage, registry_service, workflow_state_service, LocalMCPClient()
    )
    state = node({"run_id": run_id})

    assert state["results"]["step_05"]["executed"] is False
    assert state["results"]["step_05"]["planned_status"] == "blocked"
    reg = registry_service.get(run_id)
    assert reg.active_artifacts.candidate_context_table_id is None
    ws = workflow_state_service.get(run_id)
    assert ws["steps"]["step_05"] == "skipped"
