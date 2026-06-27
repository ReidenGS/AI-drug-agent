"""Step 6 DevelopabilityAgent MVP tests."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from app.agents.candidate_context_agent import CandidateContextAgent
from app.agents.developability_agent import DevelopabilityAgent
from app.agents.supervisor_agent import SupervisorAgent
from app.llm.provider import MockLLMProvider
from app.mcp.client import LocalMCPClient
from app.services.input_readiness_service import InputReadinessService
from app.services.intake_service import IntakeService
from app.services.structured_query_service import StructuredQueryService
from app.services.tool_inventory_service import ToolInventoryService
from app.services.workflow_setup_service import WorkflowSetupService
from app.utils.errors import WorkflowStateError


PROJECT_ROOT = Path(__file__).resolve().parents[2].parent
DEFAULT_XLSX = PROJECT_ROOT / "\u9879\u76ee\u6587\u4ef6" / "ToolUniversity_inventory_v0.2.xlsx"


def _bindings(canned: dict[str, dict]) -> dict:
    def make(payload):
        def _fn(**_kwargs):
            return payload
        return _fn
    return {name: make(p) for name, p in canned.items()}


def _seed_through_step_5(
    local_storage,
    registry_service,
    workflow_state_service,
    *,
    step5_bindings: dict[str, dict] | None = None,
) -> str:
    intake = IntakeService(local_storage, registry_service, workflow_state_service)
    rec = intake.submit(
        raw_user_query="HER2 ADC with vc-MMAE",
        user_provided_context={
            "target_or_antigen_text": "HER2",
            "candidate_text": "Trastuzumab analog",
            "payload_linker_text": "vc-MMAE",
        },
    )
    StructuredQueryService(
        local_storage, registry_service, workflow_state_service, SupervisorAgent(llm=MockLLMProvider())
    ).parse(rec.run_id)
    InputReadinessService(local_storage, registry_service, workflow_state_service).check(rec.run_id)
    WorkflowSetupService(local_storage, registry_service, workflow_state_service).plan(rec.run_id)
    CandidateContextAgent(
        storage=local_storage,
        registry=registry_service,
        workflow_state=workflow_state_service,
        mcp_client=LocalMCPClient(bindings=step5_bindings or _bindings({
            "SAbDab_search_structures": {"hits": [{"pdb_id": "1n8z"}]},
            "ChEMBL_search_molecules": {"hits": [{"chembl_id": "CHEMBL1201585"}]},
            "ChEMBL_search_substructure": {"hits": [{"chembl_id": "CHEMBL_linker"}]},
        })),
    ).run(rec.run_id)
    _add_step6_typed_fixture_inputs(local_storage, rec.run_id)
    return rec.run_id


def _add_step6_typed_fixture_inputs(local_storage, run_id: str) -> None:
    """Make legacy Step 6 tests explicit about typed executable inputs.

    Production Step 6 must not treat names as SMILES, sequences, or accessions.
    These tests exercise interpretation behavior, so their fixture injects
    typed fields directly into the Step 5 artifact.
    """
    key = local_storage.run_key(run_id, "candidate_context_table.json")
    cct = local_storage.read_json(key)
    for cand in cct.get("candidate_records") or []:
        materials = cand.setdefault("materials", [])
        mat_types = {m.get("material_type") for m in materials}
        if any(t in mat_types for t in {"payload_name", "linker_name", "compound_name"}):
            if not any(t in mat_types for t in {"payload_smiles", "linker_smiles", "compound_smiles"}):
                materials.append({
                    "material_id": "mat_fixture_payload_smiles",
                    "material_type": "payload_smiles",
                    "value": "CCO",
                    "value_format": None,
                    "extraction_status": "extracted",
                    "validation_status": "unknown",
                    "role": "payload",
                    "role_status": "explicit",
                })
        if any(t in mat_types for t in {"antibody_name"}):
            if "antibody_heavy_chain_sequence" not in mat_types:
                materials.append({
                    "material_id": "mat_fixture_antibody_heavy_chain_sequence",
                    "material_type": "antibody_heavy_chain_sequence",
                    "value": "EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGK",
                    "value_format": None,
                    "extraction_status": "extracted",
                    "validation_status": "unknown",
                    "role": "antibody",
                    "role_status": "explicit",
                })
        if "target_antigen_name" in mat_types:
            identifiers = cand.setdefault("identifiers", [])
            if not any(i.get("id_type") == "uniprot_id" for i in identifiers):
                identifiers.append({
                    "id_type": "uniprot_id",
                    "id_value": "P04626",
                    "source_ids": [],
                    "confidence": 0.9,
                })
    local_storage.write_json(key, cct)


# ── 1. missing Step 5 artifact ───────────────────────────────────────────────

def test_step6_requires_step5_artifact(
    local_storage, registry_service, workflow_state_service
):
    intake = IntakeService(local_storage, registry_service, workflow_state_service)
    rec = intake.submit(
        raw_user_query="hello", user_provided_context={"target_or_antigen_text": "HER2"}
    )
    agent = DevelopabilityAgent(
        storage=local_storage,
        registry=registry_service,
        workflow_state=workflow_state_service,
        mcp_client=LocalMCPClient(),
    )
    with pytest.raises(WorkflowStateError, match="Step 5"):
        agent.run(rec.run_id)


# ── 2. happy path: produces structured_liability_summary ─────────────────────

def test_step6_produces_summary_from_step5(
    local_storage, registry_service, workflow_state_service
):
    run_id = _seed_through_step_5(local_storage, registry_service, workflow_state_service)
    agent = DevelopabilityAgent(
        storage=local_storage,
        registry=registry_service,
        workflow_state=workflow_state_service,
        mcp_client=LocalMCPClient(),
    )
    summary = agent.run(run_id)
    assert summary.run_id == run_id
    assert summary.step_id == "step_06_developability"
    assert summary.candidate_liability_results, "Step 6 should produce one row per Step 5 candidate"
    # registry updated
    reg = registry_service.get(run_id)
    assert reg.active_artifacts.structured_liability_summary_id is not None
    # workflow_state updated
    state = workflow_state_service.get(run_id)
    assert state["steps"]["step_06"] == "completed"


def test_step6_selection_audit_uses_schema_mapping_fields(
    local_storage, registry_service, workflow_state_service
):
    run_id = _seed_through_step_5(local_storage, registry_service, workflow_state_service)
    DevelopabilityAgent(
        storage=local_storage,
        registry=registry_service,
        workflow_state=workflow_state_service,
        mcp_client=LocalMCPClient(),
    ).run(run_id)
    persisted = local_storage.read_json(
        local_storage.run_key(run_id, "structured_liability_summary.json")
    )
    audit = persisted["selection_audit"]

    for old_key in (
        "step_06_runtime_eligible_tools_by_candidate_lane",
        "step_06_stage1_allowed_tools_by_lane",
        "step_06_suppressed_tools_with_reason",
        "argument_construction_source_distribution",
    ):
        assert old_key not in audit

    for new_key in (
        "step_06_stage1_scope_tool_names",
        "step_06_stage1_catalog_tool_names",
        "step_06_stage1_disclosed_tool_names",
        "step_06_stage1_hidden_tools_with_reason",
        "step_06_stage1_disclosure_summary",
        "step_06_stage1_selected_tools",
        "step_06_stage2_schema_survivors",
        "step_06_stage2_mapped_tools",
        "step_06_runtime_resolved_tools",
        "step_06_executed_tools",
        "step_06_recorded_tool_call_tools",
        "step_06_stage2_uninvokable_tool_details",
        "step_06_runtime_chain_expanded_tools",
        "step_06_runtime_chain_expansion_details",
        "step_06_selection_progress",
        "argument_mapping_source_distribution",
    ):
        assert new_key in audit

    scope = set(audit["step_06_stage1_scope_tool_names"])
    catalog = set(audit["step_06_stage1_catalog_tool_names"])
    disclosed = set(audit["step_06_stage1_disclosed_tool_names"])
    selected = set(audit["step_06_stage1_selected_tools"])
    mapped = set(audit["step_06_stage2_mapped_tools"])
    resolved = set(audit["step_06_runtime_resolved_tools"])
    executed = set(audit["step_06_executed_tools"])
    recorded = set(audit["step_06_recorded_tool_call_tools"])

    assert scope, "scope should contain full Step 6 MCP-scoped tool surface"
    assert catalog, "catalog should contain LLM-visible disclosed tools"
    assert catalog == disclosed
    assert catalog <= scope
    assert selected <= catalog
    assert mapped <= selected
    assert resolved <= mapped
    assert executed <= resolved
    assert executed <= recorded
    assert isinstance(audit["step_06_runtime_chain_expansion_details"], list)

    assert isinstance(audit["step_06_selection_progress"], list)
    assert isinstance(audit["step_06_stage2_uninvokable_tool_details"], list)


# ── 3. inventory scope: Step 6 agent only calls Step 6 tools ─────────────────

def test_step6_only_calls_step6_inventory_tools(
    local_storage, registry_service, workflow_state_service
):
    xlsx = os.environ.get("TOOL_INVENTORY_XLSX", str(DEFAULT_XLSX))
    if not Path(xlsx).exists():
        pytest.skip(f"Inventory xlsx not available at {xlsx}")

    # Use inventory-scoped client: any non-Step-6 tool the agent tries would
    # come back as "skipped". The agent's lane router only ever picks Step 6
    # tool names, so we expect no skipped-by-scope outcomes here.
    run_id = _seed_through_step_5(local_storage, registry_service, workflow_state_service)
    inventory = ToolInventoryService(xlsx)
    mcp = LocalMCPClient(inventory=inventory)

    agent = DevelopabilityAgent(
        storage=local_storage,
        registry=registry_service,
        workflow_state=workflow_state_service,
        mcp_client=mcp,
    )
    agent.run(run_id)
    persisted = local_storage.read_json(
        local_storage.run_key(run_id, "structured_liability_summary.json")
    )

    # Every recorded tool call must be a Step 6 tool from v0.2 inventory.
    step6_tool_names = {
        e.tool_name
        for e in inventory.load()
        if (e.step_id or "").strip() == "6"
    }
    assert step6_tool_names, "inventory has no Step 6 tools — fixture problem"

    for cand in persisted["candidate_liability_results"]:
        for lane in cand["lane_results"]:
            for tc in lane["tool_call_records"]:
                assert tc["tool_name"] in step6_tool_names, (
                    f"Step 6 agent called non-Step-6 tool: {tc['tool_name']}"
                )
                # Turn B may record skipped tools when Stage 2 cannot map
                # required args to field refs. It must still never be a
                # Step-scope rejection.
                assert tc.get("error_message") != "tool_not_in_agent_scope"


# ── 4. unwired wrappers → dependency_unavailable, status partial ─────────────

def test_step6_handles_unwired_wrappers_gracefully(
    local_storage, registry_service, workflow_state_service
):
    """The dependency-unavailable path on Step 6 must keep producing a
    partial summary instead of crashing.

    After the Step 6 batch-1 migration, several Step 6 wrappers
    (`DrugProps_pains_filter`, `BindingDB_get_targets_by_compound`,
    `PROSITE_scan_sequence`, `EBIProteins_get_epitopes`,
    `EBIProteins_get_antigen`) are `tooluniverse_adapter`-backed and
    return success envelopes in mock mode. To keep the
    `dependency_unavailable` contract under test, force every binding to
    `_ni` so `LocalMCPClient` surfaces `dependency_unavailable`
    end-to-end.
    """
    from app.mcp.tools._registry import _all_bindings

    def _force_ni(*_a, **_kw):
        raise NotImplementedError

    forced = {name: _force_ni for name in dict(_all_bindings())}

    run_id = _seed_through_step_5(local_storage, registry_service, workflow_state_service)
    agent = DevelopabilityAgent(
        storage=local_storage,
        registry=registry_service,
        workflow_state=workflow_state_service,
        mcp_client=LocalMCPClient(bindings=forced),
    )
    summary = agent.run(run_id)
    persisted = local_storage.read_json(
        local_storage.run_key(run_id, "structured_liability_summary.json")
    )
    assert summary.prefilter_status in {"partial", "completed_with_missing_lanes"}
    # at least one tool call recorded as dependency_unavailable
    dep_unavail = [
        tc
        for cand in persisted["candidate_liability_results"]
        for lane in cand["lane_results"]
        for tc in lane["tool_call_records"]
        if tc["run_status"] == "dependency_unavailable"
    ]
    assert dep_unavail, "expected dependency_unavailable for unwired wrappers"


# ── 5. raw payload isolation ─────────────────────────────────────────────────

def test_step6_raw_payload_does_not_leak_into_normalized_records(
    local_storage, registry_service, workflow_state_service
):
    run_id = _seed_through_step_5(local_storage, registry_service, workflow_state_service)
    bindings = _bindings({
        "DrugProps_pains_filter": {"hits": [{"alert": "michael_acceptor"}]},
        "PROSITE_scan_sequence": {"hits": [{"motif": "GLYCOSYLATION"}]},
        "EBIProteins_get_features": {"hits": [{"feature": "epitope"}]},
        "ProteinsPlus_profile_structure_quality": {"hits": [{"quality": "low"}]},
        "ChEMBL_search_activities": {"hits": [{"assay_id": "A1"}]},
    })
    agent = DevelopabilityAgent(
        storage=local_storage,
        registry=registry_service,
        workflow_state=workflow_state_service,
        mcp_client=LocalMCPClient(bindings=bindings),
    )
    agent.run(run_id)
    persisted = local_storage.read_json(
        local_storage.run_key(run_id, "structured_liability_summary.json")
    )

    # No raw `hits` key should appear in normalized liability fields.
    for cand in persisted["candidate_liability_results"]:
        for lane in cand["lane_results"]:
            # liability_flags and lane_summary must not embed raw payload
            assert "hits" not in json.dumps(lane["liability_flags"])
            assert "hits" not in (lane.get("lane_summary") or "")
            for tc in lane["tool_call_records"]:
                if tc["run_status"] == "success":
                    assert tc["tool_output_ref"]
                    assert local_storage.exists(tc["tool_output_ref"])
                    raw = local_storage.read_json(tc["tool_output_ref"])
                    assert "output" in raw  # raw payload lives only here


# ── 6. small-molecule lane interpretation: PAINS alert → compact flag ────────

def _find_lane(persisted: dict, lane_type: str) -> dict | None:
    for cand in persisted["candidate_liability_results"]:
        for lane in cand["lane_results"]:
            if lane["lane_type"] == lane_type and lane["run_status"] not in {"skipped"}:
                return lane
    return None


def test_step6_small_molecule_lane_emits_compact_pains_flag(
    local_storage, registry_service, workflow_state_service
):
    run_id = _seed_through_step_5(local_storage, registry_service, workflow_state_service)
    bindings = _bindings({
        "DrugProps_pains_filter": {
            "status": "mocked",
            "source": "DrugProps_pains_filter",
            "alerts": [
                {"alert_name": "michael_acceptor_A", "smarts": "C=CC(=O)"},
                {"alert_name": "quinone_B"},
            ],
        },
    })
    agent = DevelopabilityAgent(
        storage=local_storage,
        registry=registry_service,
        workflow_state=workflow_state_service,
        mcp_client=LocalMCPClient(bindings=bindings),
    )
    agent.run(run_id)
    persisted = local_storage.read_json(
        local_storage.run_key(run_id, "structured_liability_summary.json")
    )

    lane = _find_lane(persisted, "payload_linker_compound_liability")
    assert lane, "small-molecule lane should run when payload material is present"
    assert lane["liability_flags"], "PAINS alerts must produce at least one compact flag"
    for flag in lane["liability_flags"]:
        assert flag["flag_type"], "flag_type required"
        assert flag["severity"] in {"low", "medium", "high"}
        assert flag["evidence_summary"], "evidence_summary required"
        assert flag["source_tool"] == "DrugProps_pains_filter"
        # source_ref must point at the tool_output_ref so the raw payload stays
        # outside the normalized record.
        assert flag["source_ref"]
        assert local_storage.exists(flag["source_ref"])
        # no full alert array embedded
        assert "smarts" not in flag["evidence_summary"]
        assert "smarts" not in json.dumps(flag)
    assert lane["lane_risk_category"] in {"low", "medium", "high"}
    assert lane["lane_risk_category"] != "unknown"
    # raw 'alerts' array must not have leaked
    assert "smarts" not in json.dumps(lane["liability_flags"])
    assert "smarts" not in (lane.get("lane_summary") or "")


# ── 7. sequence/protein lane interpretation: motif hit → compact flag ────────

def test_step6_sequence_lane_emits_compact_motif_flag(
    local_storage, registry_service, workflow_state_service
):
    run_id = _seed_through_step_5(local_storage, registry_service, workflow_state_service)
    bindings = _bindings({
        "PROSITE_scan_sequence": {
            "status": "mocked",
            "source": "PROSITE_scan_sequence",
            "motifs": [
                {"name": "ASN_GLYCOSYLATION", "start": 12, "end": 15, "raw_match": "NXT"},
                {"name": "DEAMIDATION_SITE", "start": 40, "end": 41},
            ],
        },
    })
    agent = DevelopabilityAgent(
        storage=local_storage,
        registry=registry_service,
        workflow_state=workflow_state_service,
        mcp_client=LocalMCPClient(bindings=bindings),
    )
    agent.run(run_id)
    persisted = local_storage.read_json(
        local_storage.run_key(run_id, "structured_liability_summary.json")
    )

    lane = _find_lane(persisted, "antibody_protein_sequence_liability")
    assert lane, "sequence lane should run when antibody material is present"
    assert lane["liability_flags"]
    flag_types = {f["flag_type"] for f in lane["liability_flags"]}
    assert "motif_match" in flag_types or any("motif" in ft for ft in flag_types)
    for flag in lane["liability_flags"]:
        assert flag["source_tool"] == "PROSITE_scan_sequence"
        assert flag["source_ref"] and local_storage.exists(flag["source_ref"])
        # raw match string must not be embedded
        assert "raw_match" not in json.dumps(flag)
        assert "NXT" not in json.dumps(flag)
    assert lane["lane_risk_category"] in {"medium", "high"}
    # raw motif list not embedded in summary
    assert "raw_match" not in (lane.get("lane_summary") or "")


# ── 8. no-signal payload: lane succeeds with no flags, risk low/unknown ──────

def test_step6_no_signal_payload_produces_no_flags_and_no_unknown_alone(
    local_storage, registry_service, workflow_state_service
):
    run_id = _seed_through_step_5(local_storage, registry_service, workflow_state_service)
    bindings = _bindings({
        "DrugProps_pains_filter": {
            "status": "mocked",
            "source": "DrugProps_pains_filter",
            "alerts": [],
            "passes": True,
        },
    })
    agent = DevelopabilityAgent(
        storage=local_storage,
        registry=registry_service,
        workflow_state=workflow_state_service,
        mcp_client=LocalMCPClient(bindings=bindings),
    )
    agent.run(run_id)
    persisted = local_storage.read_json(
        local_storage.run_key(run_id, "structured_liability_summary.json")
    )

    lane = _find_lane(persisted, "payload_linker_compound_liability")
    assert lane, "small-molecule lane should run"
    assert lane["liability_flags"] == []
    assert lane["lane_risk_category"] in {"low", "unknown"}
    assert lane["lane_summary"]
    assert "no interpreted liability signal" in lane["lane_summary"].lower()


# ── 9. dep_unavailable preserved, risk stays unknown ─────────────────────────

def test_step6_dependency_unavailable_keeps_unknown_risk(
    local_storage, registry_service, workflow_state_service
):
    from app.mcp.tools._registry import _all_bindings

    def _force_ni(*_a, **_kw):
        raise NotImplementedError

    forced = {name: _force_ni for name in dict(_all_bindings())}
    run_id = _seed_through_step_5(local_storage, registry_service, workflow_state_service)
    agent = DevelopabilityAgent(
        storage=local_storage,
        registry=registry_service,
        workflow_state=workflow_state_service,
        mcp_client=LocalMCPClient(bindings=forced),
    )
    agent.run(run_id)
    persisted = local_storage.read_json(
        local_storage.run_key(run_id, "structured_liability_summary.json")
    )
    # at least one lane has only dep_unavailable tool calls and risk stays unknown
    dep_unavail_lanes = [
        lane
        for cand in persisted["candidate_liability_results"]
        for lane in cand["lane_results"]
        if any(tc["run_status"] == "dependency_unavailable" for tc in lane["tool_call_records"])
        and not any(tc["run_status"] == "success" for tc in lane["tool_call_records"])
    ]
    assert dep_unavail_lanes
    for lane in dep_unavail_lanes:
        assert lane["liability_flags"] == []
        assert lane["lane_risk_category"] == "unknown"


# ── 10. lane-scoped interpretation: small-molecule interpreters never apply
#       to antibody/protein lanes, even if a small-molecule tool happens to be
#       called there ──────────────────────────────────────────────────────────

def test_step6_small_molecule_interpreters_do_not_apply_cross_lane():
    """Direct unit-level guard for `interpret_tool_payload`.

    A PAINS payload routed against the antibody lane must produce no flags —
    PAINS, Lipinski, QED, SwissADME, ADMETAI are small-molecule-only.
    """
    from app.agents.step_06_interpretation import interpret_tool_payload

    pains_payload = {"alerts": [{"alert_name": "michael_acceptor_A"}]}
    # In-lane: yields a flag.
    in_lane = interpret_tool_payload(
        "DrugProps_pains_filter",
        pains_payload,
        source_ref="ref://x",
        lane_type="payload_linker_compound_liability",
    )
    assert in_lane and in_lane[0]["flag_type"] == "pains_alert"
    # Out-of-lane: must produce no flags.
    for wrong_lane in (
        "antibody_protein_sequence_liability",
        "antigen_protein_feature_context",
        "structure_interface_quality",
        "compound_bioactivity_prior_context",
    ):
        assert (
            interpret_tool_payload(
                "DrugProps_pains_filter",
                pains_payload,
                source_ref="ref://x",
                lane_type=wrong_lane,
            )
            == []
        )

    # Reverse direction: PROSITE motifs must not be interpreted on the
    # small-molecule lane.
    motif_payload = {"motifs": [{"name": "ASN_GLYCOSYLATION"}]}
    assert interpret_tool_payload(
        "PROSITE_scan_sequence",
        motif_payload,
        source_ref="ref://x",
        lane_type="antibody_protein_sequence_liability",
    )
    assert (
        interpret_tool_payload(
            "PROSITE_scan_sequence",
            motif_payload,
            source_ref="ref://x",
            lane_type="payload_linker_compound_liability",
        )
        == []
    )


# ── 11. antibody lane summary records ADC-specific unassessed aspects ────────

def test_step6_antibody_lane_summary_notes_adc_unassessed_aspects(
    local_storage, registry_service, workflow_state_service
):
    """Antibody/protein lane must explicitly call out that ADC-specific
    developability (DAR, N297 glycosylation, Fc linker attachment,
    heavy/light chain pairing) is **not** assessed in Step 6.
    """
    run_id = _seed_through_step_5(local_storage, registry_service, workflow_state_service)
    bindings = _bindings({
        "PROSITE_scan_sequence": {
            "motifs": [{"name": "ASN_GLYCOSYLATION", "start": 12}],
        },
    })
    agent = DevelopabilityAgent(
        storage=local_storage,
        registry=registry_service,
        workflow_state=workflow_state_service,
        mcp_client=LocalMCPClient(bindings=bindings),
    )
    agent.run(run_id)
    persisted = local_storage.read_json(
        local_storage.run_key(run_id, "structured_liability_summary.json")
    )

    lane = _find_lane(persisted, "antibody_protein_sequence_liability")
    assert lane, "antibody lane should run"
    summary = (lane.get("lane_summary") or "").lower()
    # ADC-specific aspects must be called out as unassessed.
    for token in ("dar", "n297"):
        assert token in summary, (
            f"antibody lane summary must mention {token!r} as unassessed; got: {summary!r}"
        )
    assert "downstream" in summary or "not assessed" in summary or "unassessed" in summary
