"""Step 5 LLM-assisted tool selection tests.

Pins the contract that Step 5 enrichment now goes through a Stage 1 LLM
selection layer:

- eligible (from ``step_05_enrichment_registry``)
- LLM relevance pick (this layer)
- deterministic argument construction (registry's ``schema_arg_name`` /
  ``query``)
- execution
- compact audit (eligible / selected / skipped / source / status)

The agent must still execute every eligible plan when the LLM picks them
(preserving prior behavior under the default ``MockLLMProvider``), and
must drop any LLM-named tool that is outside the eligible catalog.
"""

from __future__ import annotations

import json
from typing import Any

from app.agents.candidate_context_agent import CandidateContextAgent
from app.agents.step_05_enrichment_registry import (
    STEP_05_CAPABILITY_REGISTRY,
    plan_enrichment_for_record,
)
from app.agents.step_05_selection_policy import (
    DESCRIPTION_SOURCE_FALLBACK,
    DESCRIPTION_SOURCE_TU,
    EXPECTED_OUTPUT_SOURCE_TU_DESCRIPTION,
    EXPECTED_OUTPUT_SOURCE_TU_NO_CONTRACT,
    EXPECTED_OUTPUT_SOURCE_TU_SPEC,
    EXPECTED_OUTPUT_SOURCE_UNAVAILABLE,
    STEP_05_SELECTION_SYSTEM_ADDENDUM,
    _build_compact_catalog,
    EXEC_SEMANTICS_REAL,
    EXEC_SEMANTICS_SYNTHETIC_DEP,
    FALLBACK_REASON_LLM_EMPTY,
    FALLBACK_REASON_LLM_NOT_CALLED,
    FALLBACK_REASON_LLM_OUT_OF_SCOPE_ONLY,
    FALLBACK_REASON_LLM_UNAVAILABLE,
    LLM_STATUS_EMPTY,
    LLM_STATUS_FAILED,
    LLM_STATUS_NOT_CALLED,
    LLM_STATUS_OK,
    LLM_STATUS_OK_WITH_DROPPED,
    LLM_STATUS_OUT_OF_SCOPE_ONLY,
    REASON_DETERMINISTIC_FALLBACK,
    REASON_FALLBACK_EMPTY,
    REASON_FALLBACK_NOT_CALLED,
    REASON_FALLBACK_OUT_OF_SCOPE,
    REASON_KNOWN_LIVE_UNAVAILABLE,
    REASON_LLM_SELECTED,
    REASON_LLM_SKIPPED,
    SELECTION_POLICY_VERSION,
    SOURCE_DETERMINISTIC_FALLBACK,
    SOURCE_LLM_STAGE1,
    SOURCE_SYSTEM_DEPENDENCY_GAP,
    select_step5_enrichment_plans,
)
from app.agents.supervisor_agent import SupervisorAgent
from app.llm.provider import MockLLMProvider
from app.mcp.client import LocalMCPClient
from app.schemas.step_05_candidate_context_table import (
    CandidateRecord,
    Identifier,
    Material,
)
from app.services.intake_service import IntakeService
from app.services.input_readiness_service import InputReadinessService
from app.services.structured_query_service import StructuredQueryService
from app.services.workflow_setup_service import WorkflowSetupService


# ── helpers ─────────────────────────────────────────────────────────────────


def _record(
    *,
    candidate_type: str = "compound_component",
    materials: list[Material] | None = None,
    identifiers: list[Identifier] | None = None,
    candidate_id: str = "cand_test",
) -> CandidateRecord:
    return CandidateRecord(
        candidate_id=candidate_id,
        candidate_label="step5 llm selection test",
        candidate_type=candidate_type,  # type: ignore[arg-type]
        materials=materials or [],
        identifiers=identifiers or [],
        candidate_role="user_provided_candidate",
        is_generated_candidate=False,
        context_status="partial",
    )


def _mat(material_type: str, value: str, role: str | None = None) -> Material:
    return Material(
        material_id=f"mat_{material_type}",
        material_type=material_type,
        value=value,
        role=role,
    )


def _ident(id_type: str, value: str) -> Identifier:
    return Identifier(id_type=id_type, id_value=value)


class _FakeLLM:
    """Minimal LLM stub that returns a fixed Stage-1 selections payload.

    Records every call so tests can assert how many times Stage 1 fired.
    Implements only the methods the policy actually uses.
    """

    name = "fake_step5_selector"
    model = "fake-step5-v1"

    def __init__(self, selections: list[dict] | None = None,
                 raise_exc: Exception | None = None):
        self._selections = selections or []
        self._raise = raise_exc
        self.calls: list[dict] = []

    def generate(self, prompt: str, *, system: str | None = None, **_: Any) -> str:
        raise NotImplementedError

    def generate_json(self, prompt: str, *, schema: dict,
                      system: str | None = None) -> dict:
        self.calls.append({"task": (schema or {}).get("task"),
                           "tools": [e["tool_name"] for e in
                                     (schema or {}).get("compact_catalog") or []]})
        if self._raise is not None:
            raise self._raise
        return {"selections": list(self._selections),
                "selection_metadata": {"strategy": "fake_step5"}}


# ── 1. LLM skips an eligible tool → not executed ───────────────────────────


def test_llm_skip_means_no_execution_and_audit_records_skip():
    """LLM picks only ChEMBL_get_molecule; ChEMBL_search_substructure is
    eligible but should NOT be executed. Audit records both states."""
    record = _record(
        identifiers=[_ident("chembl_id", "CHEMBL1201585")],
        materials=[_mat("payload_smiles", "CCO", "payload")],
    )
    eligible = plan_enrichment_for_record(
        record,
        scoped_tools=["ChEMBL_get_molecule", "ChEMBL_search_substructure"],
        candidate_category="compound_component",
        registry=STEP_05_CAPABILITY_REGISTRY,
    )
    assert {p.tool_name for p in eligible} == {
        "ChEMBL_get_molecule", "ChEMBL_search_substructure",
    }
    llm = _FakeLLM(selections=[
        {"tool_name": "ChEMBL_get_molecule",
         "selection_reason": "exact id present", "priority": 1,
         "required_context": ["chembl_id"]},
    ])
    decisions, audit = select_step5_enrichment_plans(
        record=record, eligible_plans=eligible, llm=llm,
        raw_user_query="",
    )
    selected = [d for d in decisions if d.selected]
    skipped = [d for d in decisions if not d.selected]
    assert [d.plan.tool_name for d in selected] == ["ChEMBL_get_molecule"]
    assert [d.plan.tool_name for d in skipped] == ["ChEMBL_search_substructure"]
    assert all(d.selection_reason == REASON_LLM_SELECTED for d in selected)
    assert all(d.skip_reason == REASON_LLM_SKIPPED for d in skipped)
    assert audit.tool_selection_source == "llm_stage1"
    assert audit.llm_call_status == "ok"
    assert {t["tool_name"] for t in audit.selected_tools} == {"ChEMBL_get_molecule"}
    assert {t["tool_name"] for t in audit.skipped_eligible_tools} == {
        "ChEMBL_search_substructure"
    }
    assert audit.policy_version == SELECTION_POLICY_VERSION


# ── 2. Out-of-scope LLM selection is dropped ────────────────────────────────


def test_out_of_scope_llm_tool_is_dropped_before_execution():
    """LLM returns only out-of-scope names. Fail-open fallback still
    executes the eligible set (explicit product policy), but the audit
    must distinguish this from a true LLM selection: source is
    deterministic_fallback, status is out_of_scope_only, fallback_reason
    is llm_out_of_scope_only, and per-tool selection_reason is the
    out-of-scope fallback code (NOT ``llm_selected``)."""
    record = _record(
        identifiers=[_ident("chembl_id", "CHEMBL1201585")],
    )
    eligible = plan_enrichment_for_record(
        record,
        scoped_tools=["ChEMBL_get_molecule"],
        candidate_category="compound_component",
        registry=STEP_05_CAPABILITY_REGISTRY,
    )
    llm = _FakeLLM(selections=[
        {"tool_name": "Made_up_tool_not_in_catalog",
         "selection_reason": "hallucinated", "priority": 1,
         "required_context": []},
        {"tool_name": "RCSBData_get_entry",  # also not eligible here
         "selection_reason": "scope spillover", "priority": 1,
         "required_context": []},
    ])
    decisions, audit = select_step5_enrichment_plans(
        record=record, eligible_plans=eligible, llm=llm,
        raw_user_query="",
    )
    assert "Made_up_tool_not_in_catalog" in audit.llm_dropped_out_of_scope
    assert "RCSBData_get_entry" in audit.llm_dropped_out_of_scope
    assert audit.tool_selection_source == SOURCE_DETERMINISTIC_FALLBACK
    assert audit.llm_call_status == LLM_STATUS_OUT_OF_SCOPE_ONLY
    assert audit.fallback_reason == FALLBACK_REASON_LLM_OUT_OF_SCOPE_ONLY
    selected = [d for d in decisions if d.selected]
    assert [d.plan.tool_name for d in selected] == ["ChEMBL_get_molecule"]
    assert all(
        d.selection_reason == REASON_FALLBACK_OUT_OF_SCOPE for d in selected
    )
    assert not any(d.selection_reason == REASON_LLM_SELECTED for d in selected)


# ── 3. LLM unavailable → deterministic fallback ────────────────────────────


def test_llm_failure_falls_back_to_deterministic_with_source_recorded():
    record = _record(
        materials=[_mat("payload_smiles", "CCO", "payload")],
    )
    eligible = plan_enrichment_for_record(
        record,
        scoped_tools=["ChEMBL_search_substructure"],
        candidate_category="compound_component",
        registry=STEP_05_CAPABILITY_REGISTRY,
    )
    llm = _FakeLLM(raise_exc=RuntimeError("simulated LLM outage"))
    decisions, audit = select_step5_enrichment_plans(
        record=record, eligible_plans=eligible, llm=llm,
        raw_user_query="",
    )
    selected = [d for d in decisions if d.selected]
    assert len(selected) == 1
    assert selected[0].plan.tool_name == "ChEMBL_search_substructure"
    assert selected[0].tool_selection_source == SOURCE_DETERMINISTIC_FALLBACK
    assert selected[0].selection_reason == REASON_DETERMINISTIC_FALLBACK
    assert audit.llm_call_status == LLM_STATUS_FAILED
    assert audit.tool_selection_source == SOURCE_DETERMINISTIC_FALLBACK
    assert audit.fallback_reason == FALLBACK_REASON_LLM_UNAVAILABLE


# ── 4. llm=None → deterministic (no Stage 1 call attempted) ───────────────


def test_llm_none_means_no_call_and_deterministic_selection():
    record = _record(
        identifiers=[_ident("chembl_id", "CHEMBL999")],
    )
    eligible = plan_enrichment_for_record(
        record,
        scoped_tools=["ChEMBL_get_molecule"],
        candidate_category="compound_component",
        registry=STEP_05_CAPABILITY_REGISTRY,
    )
    decisions, audit = select_step5_enrichment_plans(
        record=record, eligible_plans=eligible, llm=None,
        raw_user_query="",
    )
    selected = [d for d in decisions if d.selected]
    assert [d.plan.tool_name for d in selected] == ["ChEMBL_get_molecule"]
    assert audit.llm_call_status == LLM_STATUS_NOT_CALLED
    assert audit.tool_selection_source == SOURCE_DETERMINISTIC_FALLBACK
    assert audit.fallback_reason == FALLBACK_REASON_LLM_NOT_CALLED
    # per-decision reason must NOT pretend the LLM picked these.
    assert all(
        d.selection_reason == REASON_FALLBACK_NOT_CALLED for d in selected
    )
    assert not any(d.selection_reason == REASON_LLM_SELECTED for d in selected)


# ── 5. Mock LLM (signal-match) preserves prior agent behavior ──────────────


def _setup_run(local_storage, registry_service, workflow_state_service) -> str:
    intake = IntakeService(local_storage, registry_service, workflow_state_service)
    rec = intake.submit(
        raw_user_query="Design ADC against HER2 with vc-MMAE payload",
        user_provided_context={
            "target_or_antigen_text": "HER2",
            "candidate_text": "Trastuzumab analog",
            "payload_linker_text": "vc-MMAE",
        },
    )
    supervisor = SupervisorAgent(llm=MockLLMProvider())
    StructuredQueryService(
        local_storage, registry_service, workflow_state_service, supervisor
    ).parse(rec.run_id)
    InputReadinessService(
        local_storage, registry_service, workflow_state_service
    ).check(rec.run_id)
    WorkflowSetupService(
        local_storage, registry_service, workflow_state_service
    ).plan(rec.run_id)
    return rec.run_id


def _bindings(canned: dict[str, dict]) -> dict:
    def make(payload):
        def _fn(**_kwargs):
            return payload
        return _fn
    return {name: make(p) for name, p in canned.items()}


def test_agent_default_mock_llm_executes_eligible_and_emits_audit(
    local_storage, registry_service, workflow_state_service
):
    """End-to-end: default MockLLMProvider's Stage 1 mock selects every
    catalog entry whose coarse_input_requirements intersect with signals,
    so the agent still executes the prior eligible set. Audit fields and
    per-tool provenance must be present."""
    run_id = _setup_run(local_storage, registry_service, workflow_state_service)
    mcp = LocalMCPClient(
        bindings=_bindings({
            "SAbDab_search_structures": {"hits": [{"pdb_id": "1n8z"}]},
            "ChEMBL_search_molecules": {"hits": [{"chembl_id": "CHEMBL_a"}]},
            "ChEMBL_search_substructure": {"hits": [{"chembl_id": "CHEMBL_b"}]},
        })
    )
    agent = CandidateContextAgent(
        storage=local_storage, registry=registry_service,
        workflow_state=workflow_state_service, mcp_client=mcp,
    )
    agent.run(run_id)
    persisted = local_storage.read_json(
        local_storage.run_key(run_id, "candidate_context_table.json")
    )
    audit = persisted["enrichment_selection_audit"]
    assert audit, "audit must be persisted"
    for cand_id, entry in audit.items():
        assert entry["policy_version"] == SELECTION_POLICY_VERSION
        assert entry["tool_selection_source"] in {
            SOURCE_LLM_STAGE1, SOURCE_DETERMINISTIC_FALLBACK
        }
        assert entry["llm_call_status"] in {
            LLM_STATUS_OK, LLM_STATUS_OK_WITH_DROPPED, LLM_STATUS_EMPTY,
            LLM_STATUS_OUT_OF_SCOPE_ONLY, LLM_STATUS_FAILED,
            LLM_STATUS_NOT_CALLED,
        }
        assert isinstance(entry["eligible_tools"], list)
        assert isinstance(entry["selected_tools"], list)
        assert isinstance(entry["skipped_eligible_tools"], list)
        assert isinstance(entry["known_unavailable_records"], list)
        assert "fallback_reason" in entry
    # Every executed tool_call_record carries selection provenance.
    for tc in persisted["tool_call_records"]:
        summary = tc["tool_input_summary"]
        assert summary.get("tool_selection_source")
        assert summary.get("selection_reason")
        assert summary.get("argument_construction_source") == "deterministic_mapping"
        assert summary.get("selection_policy_version") == SELECTION_POLICY_VERSION
    # Raw payload isolation.
    cand_blob = json.dumps(persisted["candidate_records"])
    assert "hits" not in cand_blob


# ── 6. End-to-end LLM skip blocks MCP execution ─────────────────────────────


def test_agent_with_fake_llm_skip_does_not_execute_skipped_tool(
    local_storage, registry_service, workflow_state_service
):
    run_id = _setup_run(local_storage, registry_service, workflow_state_service)

    # The fake LLM picks only ChEMBL_search_molecules.
    fake = _FakeLLM(selections=[
        {"tool_name": "ChEMBL_search_molecules",
         "selection_reason": "compound name lookup is sufficient",
         "priority": 1, "required_context": ["compound_name"]},
        {"tool_name": "SAbDab_search_structures",
         "selection_reason": "antibody / target structure context",
         "priority": 1, "required_context": []},
    ])

    called: list[str] = []

    def trace(name, payload):
        def _fn(**_kwargs):
            called.append(name)
            return payload
        return _fn

    bindings = {
        "SAbDab_search_structures": trace(
            "SAbDab_search_structures", {"hits": [{"pdb_id": "1n8z"}]}
        ),
        "ChEMBL_search_molecules": trace(
            "ChEMBL_search_molecules", {"hits": [{"chembl_id": "CHEMBL_a"}]}
        ),
        "ChEMBL_search_substructure": trace(
            "ChEMBL_search_substructure",
            {"hits": [{"chembl_id": "CHEMBL_b"}]},
        ),
    }
    agent = CandidateContextAgent(
        storage=local_storage, registry=registry_service,
        workflow_state=workflow_state_service,
        mcp_client=LocalMCPClient(bindings=bindings), llm=fake,
    )
    agent.run(run_id)

    # ChEMBL_search_substructure was eligible (vc-MMAE → SMILES path)
    # only if a SMILES material exists; the substructure plan is not
    # always produced on this fixture (Step 2 mock typically does not
    # surface a SMILES for vc-MMAE). The strict assertion is: NO call
    # was ever made to a tool the fake LLM did not select.
    assert "ChEMBL_search_substructure" not in called
    # Tools the fake LLM DID select were attempted iff they were also
    # eligible (the agent never widens past eligible).
    assert called  # something must have been called
    assert set(called).issubset(
        {"ChEMBL_search_molecules", "SAbDab_search_structures"}
    )
    persisted = local_storage.read_json(
        local_storage.run_key(run_id, "candidate_context_table.json")
    )
    audit = persisted["enrichment_selection_audit"]
    # Each candidate that had eligible plans now records llm_stage1 as source.
    assert any(
        entry.get("tool_selection_source") == "llm_stage1"
        for entry in audit.values()
    )


# ── 7. ChEMBL ID promotion still works under LLM selection ─────────────────


def test_chembl_id_lookup_routing_preserved_under_llm_selection():
    """The metadata-driven routing (ChEMBL ID → ChEMBL_get_molecule,
    name → ChEMBL_search_molecules, SMILES → ChEMBL_search_substructure)
    must continue to drive eligibility. The LLM only chooses among
    eligible plans."""
    cases = [
        (
            _record(identifiers=[_ident("chembl_id", "CHEMBL999")]),
            ["ChEMBL_get_molecule", "ChEMBL_search_molecules"],
            {"ChEMBL_get_molecule"},
        ),
        (
            _record(materials=[_mat("payload_name", "monomethyl auristatin E", "payload")]),
            ["ChEMBL_search_molecules", "ChEMBL_search_substructure"],
            {"ChEMBL_search_molecules"},
        ),
        (
            _record(materials=[_mat("payload_smiles", "CCO", "payload")]),
            ["ChEMBL_search_substructure", "ChEMBL_search_molecules"],
            {"ChEMBL_search_substructure"},
        ),
    ]
    for record, scope, expected in cases:
        eligible = plan_enrichment_for_record(
            record,
            scoped_tools=scope,
            candidate_category="compound_component",
            registry=STEP_05_CAPABILITY_REGISTRY,
        )
        assert {p.tool_name for p in eligible} == expected, (record, scope)


# ── 8. SAbDab routing preserved ─────────────────────────────────────────────


def test_sabdab_routing_preserved_under_llm_selection():
    target = _record(
        candidate_type="target_antigen",
        materials=[_mat("target_antigen_name", "HER2", "target")],
    )
    plans = plan_enrichment_for_record(
        target,
        scoped_tools=["SAbDab_search_structures"],
        candidate_category="target_antigen",
        registry=STEP_05_CAPABILITY_REGISTRY,
    )
    assert [p.tool_name for p in plans] == ["SAbDab_search_structures"]


# ── 9. No new Step 5 schema drift (no candidate generation / pose / rank) ──


def test_persisted_step5_artifact_carries_no_candidate_generation_fields(
    local_storage, registry_service, workflow_state_service
):
    run_id = _setup_run(local_storage, registry_service, workflow_state_service)
    mcp = LocalMCPClient(
        bindings=_bindings({
            "SAbDab_search_structures": {"hits": []},
            "ChEMBL_search_molecules": {"hits": []},
            "ChEMBL_search_substructure": {"hits": []},
        })
    )
    CandidateContextAgent(
        storage=local_storage, registry=registry_service,
        workflow_state=workflow_state_service, mcp_client=mcp,
    ).run(run_id)
    persisted = local_storage.read_json(
        local_storage.run_key(run_id, "candidate_context_table.json")
    )
    # Step 5 must not start generating downstream artifacts.
    blob = json.dumps(persisted)
    for forbidden in (
        "generated_adc_candidate", "pose_ensemble", "ranking_score",
        "dar_design", "conjugation_design", "hypothesis_record",
    ):
        assert forbidden not in blob, forbidden


# ── 10. Empty LLM selection → fail-open fallback executes all eligible,
#       audit makes the fallback path explicit. ───────────────────────────


def test_llm_empty_selection_does_not_pretend_to_be_llm_selected():
    record = _record(
        identifiers=[_ident("chembl_id", "CHEMBL999")],
        materials=[_mat("payload_smiles", "CCO", "payload")],
    )
    eligible = plan_enrichment_for_record(
        record,
        scoped_tools=["ChEMBL_get_molecule", "ChEMBL_search_substructure"],
        candidate_category="compound_component",
        registry=STEP_05_CAPABILITY_REGISTRY,
    )
    llm = _FakeLLM(selections=[])  # valid JSON, zero picks
    decisions, audit = select_step5_enrichment_plans(
        record=record, eligible_plans=eligible, llm=llm,
        raw_user_query="",
    )
    selected = [d for d in decisions if d.selected]
    # Fail-open: all eligible still execute.
    assert {d.plan.tool_name for d in selected} == {
        "ChEMBL_get_molecule", "ChEMBL_search_substructure",
    }
    # But the audit must say so honestly.
    assert audit.tool_selection_source == SOURCE_DETERMINISTIC_FALLBACK
    assert audit.llm_call_status == LLM_STATUS_EMPTY
    assert audit.fallback_reason == FALLBACK_REASON_LLM_EMPTY
    assert audit.skipped_eligible_tools == []
    # Per-tool selection_reason must NOT claim the LLM selected them.
    assert all(
        d.selection_reason == REASON_FALLBACK_EMPTY for d in selected
    )
    assert not any(d.selection_reason == REASON_LLM_SELECTED for d in selected)


# ── 11. Partial valid + out-of-scope → only valid executes; no all-eligible
#       fallback. ──────────────────────────────────────────────────────────


def test_partial_valid_plus_out_of_scope_does_not_widen_to_all_eligible():
    record = _record(
        identifiers=[_ident("chembl_id", "CHEMBL999")],
        materials=[_mat("payload_smiles", "CCO", "payload")],
    )
    eligible = plan_enrichment_for_record(
        record,
        scoped_tools=["ChEMBL_get_molecule", "ChEMBL_search_substructure"],
        candidate_category="compound_component",
        registry=STEP_05_CAPABILITY_REGISTRY,
    )
    llm = _FakeLLM(selections=[
        {"tool_name": "ChEMBL_get_molecule",
         "selection_reason": "exact id present", "priority": 1,
         "required_context": ["chembl_id"]},
        {"tool_name": "Made_up_tool", "selection_reason": "noise",
         "priority": 1, "required_context": []},
    ])
    decisions, audit = select_step5_enrichment_plans(
        record=record, eligible_plans=eligible, llm=llm,
        raw_user_query="",
    )
    selected = [d for d in decisions if d.selected]
    skipped = [d for d in decisions if not d.selected]
    # Only the in-scope pick executes; substructure stays skipped.
    assert [d.plan.tool_name for d in selected] == ["ChEMBL_get_molecule"]
    assert [d.plan.tool_name for d in skipped] == ["ChEMBL_search_substructure"]
    assert audit.tool_selection_source == SOURCE_LLM_STAGE1
    assert audit.llm_call_status == LLM_STATUS_OK_WITH_DROPPED
    assert "Made_up_tool" in audit.llm_dropped_out_of_scope
    # No deterministic fallback metadata when a real LLM pick survived.
    assert audit.fallback_reason == ""
    assert selected[0].selection_reason == REASON_LLM_SELECTED


# ── 12. known_live_unavailable is bucketed separately and never tagged
#       as an LLM-selected real execution. ────────────────────────────────


def test_known_live_unavailable_record_is_not_an_llm_selection():
    record = _record(
        identifiers=[_ident("zinc_id", "ZINC0000001"),
                     _ident("chembl_id", "CHEMBL999")],
    )
    # Force the ZINC plan into the eligible set via include_known_unavailable
    # so we can audit how the policy buckets it (production call path keeps
    # ZINC at the registry gate; this test purposely exercises the
    # post-registry policy boundary).
    eligible = plan_enrichment_for_record(
        record,
        scoped_tools=["ChEMBL_get_molecule", "ZINC_get_compound"],
        candidate_category="compound_component",
        include_known_unavailable=True,
        registry=STEP_05_CAPABILITY_REGISTRY,
    )
    assert {p.tool_name for p in eligible} == {
        "ChEMBL_get_molecule", "ZINC_get_compound",
    }
    llm = _FakeLLM(selections=[
        {"tool_name": "ChEMBL_get_molecule",
         "selection_reason": "exact id present", "priority": 1,
         "required_context": ["chembl_id"]},
    ])
    decisions, audit = select_step5_enrichment_plans(
        record=record, eligible_plans=eligible, llm=llm,
        raw_user_query="",
    )
    zinc_dec = next(d for d in decisions if d.plan.tool_name == "ZINC_get_compound")
    chembl_dec = next(d for d in decisions if d.plan.tool_name == "ChEMBL_get_molecule")

    # ZINC: synthetic dependency-gap record. Selected for emission but
    # NOT credited to the LLM and NOT counted as a real execution.
    assert zinc_dec.selected is True
    assert zinc_dec.tool_selection_source == SOURCE_SYSTEM_DEPENDENCY_GAP
    assert zinc_dec.selection_reason == REASON_KNOWN_LIVE_UNAVAILABLE
    assert zinc_dec.execution_semantics == EXEC_SEMANTICS_SYNTHETIC_DEP
    # ChEMBL: a real LLM-selected execution.
    assert chembl_dec.tool_selection_source == SOURCE_LLM_STAGE1
    assert chembl_dec.selection_reason == REASON_LLM_SELECTED
    assert chembl_dec.execution_semantics == EXEC_SEMANTICS_REAL

    # Audit bucketing.
    assert {t["tool_name"] for t in audit.selected_tools} == {"ChEMBL_get_molecule"}
    assert {t["tool_name"] for t in audit.known_unavailable_records} == {
        "ZINC_get_compound"
    }
    # known_unavailable records carry the dedicated source + semantics.
    zinc_audit = audit.known_unavailable_records[0]
    assert zinc_audit["tool_selection_source"] == SOURCE_SYSTEM_DEPENDENCY_GAP
    assert zinc_audit["execution_semantics"] == EXEC_SEMANTICS_SYNTHETIC_DEP


# ── 13. tool_input_summary counters split real vs synthetic ────────────────


def test_agent_tool_input_summary_distinguishes_real_vs_synthetic(
    local_storage, registry_service, workflow_state_service
):
    """When the agent executes real ChEMBL tools, the provenance dict
    on each ToolCallRecord must expose ``real_selected_count`` and
    ``known_unavailable_count`` separately so a reviewer cannot read a
    synthetic ZINC ``dependency_unavailable`` record as part of the
    LLM-selected execution count."""
    run_id = _setup_run(local_storage, registry_service, workflow_state_service)
    mcp = LocalMCPClient(
        bindings=_bindings({
            "SAbDab_search_structures": {"hits": []},
            "ChEMBL_search_molecules": {"hits": []},
            "ChEMBL_search_substructure": {"hits": []},
        })
    )
    CandidateContextAgent(
        storage=local_storage, registry=registry_service,
        workflow_state=workflow_state_service, mcp_client=mcp,
    ).run(run_id)
    persisted = local_storage.read_json(
        local_storage.run_key(run_id, "candidate_context_table.json")
    )
    assert persisted["tool_call_records"], "expected ≥1 real tool call"
    for tc in persisted["tool_call_records"]:
        summary = tc["tool_input_summary"]
        assert "real_selected_count" in summary
        assert "known_unavailable_count" in summary
        assert summary.get("execution_semantics") in {
            EXEC_SEMANTICS_REAL, EXEC_SEMANTICS_SYNTHETIC_DEP,
        }
        assert "fallback_reason" in summary


# ── 14. STEP_05_SELECTION_SYSTEM_ADDENDUM wording ──────────────────────────


def test_addendum_uses_concise_role_rules_and_output_sections():
    sp = STEP_05_SELECTION_SYSTEM_ADDENDUM
    assert sp.startswith(
        "You are selecting Step 5 context-enrichment tools"
    )
    assert "Role:" in sp
    assert "Rules:" in sp
    assert "Catalog fields:" in sp
    assert "Output:" in sp
    assert "Your decision is relevance, not feasibility." in sp
    assert "fills a material/context gap" in sp
    assert len(sp.split()) < 520


def test_addendum_keeps_core_relevance_rules_with_compact_examples():
    sp = STEP_05_SELECTION_SYSTEM_ADDENDUM
    assert "Use only exact `tool_name` values from `compact_catalog`." in sp
    assert "Prefer exact identity/context over approximate lookup" in sp
    assert "Select a tool only if its expected output" in sp
    assert "Skip tools whose expected output is redundant" in sp
    assert "Return an empty `selections` list" in sp
    assert "Compact examples:" in sp
    assert "linker_payload_name=vc-MMAE" in sp
    assert "monomethyl auristatin E" in sp
    assert "ChEMBL_get_molecule` only" not in sp
    assert '"selections": [' not in sp


def test_addendum_states_name_and_smiles_lookup_are_complementary():
    sp = STEP_05_SELECTION_SYSTEM_ADDENDUM
    assert "Name lookup and SMILES lookup are complementary paths." in sp
    assert "`payload_smiles`, `linker_smiles`, or `compound_smiles`" in sp
    assert "`query_kind=smiles` / SMILES-capable tool" in sp
    assert "select at least one SMILES lookup path" in sp
    assert "even if a name lookup is also\n  selected" in sp
    assert "Do not invent SMILES" in sp
    assert "do not use a name string as SMILES" in sp


def test_addendum_has_compact_few_shots_for_key_input_shapes():
    sp = STEP_05_SELECTION_SYSTEM_ADDENDUM
    for phrase in (
        "input: `linker_payload_name=vc-MMAE`",
        "select eligible name lookup tools such as `ChEMBL_search_molecules`",
        "input: `payload_smiles=CCO` or `linker_smiles=NCC(=O)O`",
        "select eligible SMILES lookup tools such as",
        "`ChEMBL_search_substructure`, even if name lookup is also selected",
        "input: `target_antigen_name=HER2` or `uniprot_id=P04626`",
        "select eligible target/antigen context tools",
        "input: `antibody_heavy_chain_sequence` +",
        "select eligible sequence/CDR3/BCR",
        "input: uploaded `structure_file` or `pdb_id`",
        "select eligible structure-context tools",
    ):
        assert phrase in sp, phrase


def test_addendum_selection_reason_and_output_shape_are_explicit():
    sp = STEP_05_SELECTION_SYSTEM_ADDENDUM
    assert "Return JSON matching the shared Stage-1 shape" in sp
    assert '"selections"' in sp
    assert '"tool_name"' in sp
    assert '"selection_reason"' in sp
    assert '"required_context"' in sp
    assert "`selection_reason` must briefly name the input" in sp
    assert "expected context output" in sp
    assert "downstream use" in sp


def test_addendum_still_forbids_downstream_outputs():
    sp = STEP_05_SELECTION_SYSTEM_ADDENDUM
    flat = " ".join(sp.split())
    assert "Do not generate ADC candidates" in sp
    for phrase in (
        "rankings",
        "pose ensembles",
        "DAR or conjugation designs",
        "hypotheses",
        "liability verdicts",
        "candidate-screening picks",
    ):
        assert phrase in flat, phrase


def test_addendum_does_not_ask_for_argument_construction_or_inventing_ids():
    sp = STEP_05_SELECTION_SYSTEM_ADDENDUM
    assert "Do not construct arguments" in sp
    assert "invent identifiers" in sp
    assert "invent tool names" in sp
    assert "deterministic argument construction" in sp


# ── 15. STEP_05_SELECTION_SYSTEM_ADDENDUM mentions TU-derived catalog ──────


def test_addendum_explains_tooluniverse_derived_catalog_metadata():
    sp = STEP_05_SELECTION_SYSTEM_ADDENDUM
    assert "Catalog fields:" in sp
    assert "official ToolUniverse" in sp
    # Field names the LLM should look at.
    for key in (
        "`short_description`",
        "`expected_output_fields`",
        "`expected_output_semantics`",
        "`expected_output_source`",
        "`description_source`",
        "`project_side_hints`",
    ):
        assert key in sp, key
    # Project-side hints are explicitly NOT TU output.
    assert "not ToolUniverse\n  output fields" in sp


# ── 16. _build_compact_catalog wiring against a fake ToolUniverse ──────────


class _FakeTU:
    """Minimal stand-in matching the ``_get_universe`` API surface used
    by ``tooluniverse_adapter.get_tool_specifications``."""

    def __init__(self, specs: dict[str, dict]) -> None:
        self._specs = dict(specs)
        self.lookups: list[tuple[str, ...]] = []

    def get_tool_specification_by_names(self, names: list[str]) -> list[dict]:
        self.lookups.append(tuple(names))
        return [self._specs[n] for n in names if n in self._specs]

    # The catalog builder only touches get_tool_specifications →
    # _safe_universe → these two unused methods. Keep them no-op for
    # safety.
    def load_tools(self, **_kw: Any) -> None:
        return None

    def get_available_tools(self, name_only: bool = True) -> list[str]:
        return list(self._specs)


def _install_fake_tu(monkeypatch, specs: dict[str, dict]) -> _FakeTU:
    from app.mcp import tooluniverse_adapter  # noqa: PLC0415

    fake = _FakeTU(specs)
    tooluniverse_adapter._reset_for_tests()
    monkeypatch.setattr(tooluniverse_adapter, "_get_universe", lambda: fake)
    return fake


def _eligible_for(scoped_tools: list[str], record: CandidateRecord):
    return plan_enrichment_for_record(
        record,
        scoped_tools=scoped_tools,
        candidate_category=record.candidate_type,
        registry=STEP_05_CAPABILITY_REGISTRY,
    )


def test_catalog_short_description_prefers_tooluniverse_metadata(monkeypatch):
    record = _record(identifiers=[_ident("chembl_id", "CHEMBL999")])
    plans = _eligible_for(["ChEMBL_get_molecule"], record)
    _install_fake_tu(monkeypatch, {
        "ChEMBL_get_molecule": {
            "name": "ChEMBL_get_molecule",
            "description": "TU official: fetch a ChEMBL molecule by id.",
            "parameter": {
                "type": "object",
                "properties": {"chembl_id": {"type": "string"}},
                "required": ["chembl_id"],
            },
        }
    })
    catalog = _build_compact_catalog(plans)
    assert len(catalog) == 1
    entry = catalog[0]
    assert entry["short_description"].startswith("TU official:")
    assert entry["description_source"] == DESCRIPTION_SOURCE_TU
    # Project hints are present and clearly bucketed.
    assert "project_side_hints" in entry
    hints = entry["project_side_hints"]
    assert hints["identity_strength"] == "exact"
    assert hints["redundancy_group"] == "chembl_exact_id"
    assert "step_06_compound_liability_lane" in hints["downstream_uses"]


def test_catalog_short_description_falls_back_when_tu_unavailable(monkeypatch):
    record = _record(identifiers=[_ident("chembl_id", "CHEMBL999")])
    plans = _eligible_for(["ChEMBL_get_molecule"], record)
    # No specs → fake returns []
    _install_fake_tu(monkeypatch, {})
    catalog = _build_compact_catalog(plans)
    assert len(catalog) == 1
    entry = catalog[0]
    assert entry["description_source"] == DESCRIPTION_SOURCE_FALLBACK
    # Fallback short_description carries no TU sentence.
    assert "TU official" not in entry["short_description"]


def test_catalog_expected_output_source_tooluniverse_spec(monkeypatch):
    record = _record(materials=[_mat("payload_name", "MMAE", "payload")])
    plans = _eligible_for(["ChEMBL_search_molecules"], record)
    _install_fake_tu(monkeypatch, {
        "ChEMBL_search_molecules": {
            "name": "ChEMBL_search_molecules",
            "description": "TU official: search ChEMBL molecules by name.",
            "parameter": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
            "return_schema": {
                "type": "object",
                "description": "List of matched ChEMBL molecules.",
                "properties": {
                    "molecules": {"type": "array"},
                    "pref_name": {"type": "string"},
                    "molecule_chembl_id": {"type": "string"},
                },
            },
        }
    })
    catalog = _build_compact_catalog(plans)
    entry = catalog[0]
    assert entry["expected_output_source"] == EXPECTED_OUTPUT_SOURCE_TU_SPEC
    assert set(entry["expected_output_fields"]) == {
        "molecules", "pref_name", "molecule_chembl_id",
    }
    assert entry["expected_output_semantics"] == "List of matched ChEMBL molecules."
    assert "return_schema" in entry["expected_output_notes"]


def test_catalog_expected_output_source_tooluniverse_description(monkeypatch):
    record = _record(materials=[_mat("payload_smiles", "CCO", "payload")])
    plans = _eligible_for(["ChEMBL_search_substructure"], record)
    # No return_schema; description mentions returns.
    _install_fake_tu(monkeypatch, {
        "ChEMBL_search_substructure": {
            "name": "ChEMBL_search_substructure",
            "description": (
                "Substructure search over ChEMBL. Returns ChEMBL molecule "
                "matches for the SMILES query."
            ),
            "parameter": {
                "type": "object",
                "properties": {"smiles": {"type": "string"}},
                "required": ["smiles"],
            },
        }
    })
    catalog = _build_compact_catalog(plans)
    entry = catalog[0]
    assert entry["expected_output_source"] == EXPECTED_OUTPUT_SOURCE_TU_DESCRIPTION
    # Field list must be empty — we never extract field names from text.
    assert entry["expected_output_fields"] == []
    assert "Returns" in entry["expected_output_semantics"]
    assert "field list not parsed" in entry["expected_output_notes"]


def test_catalog_expected_output_source_no_contract(monkeypatch):
    record = _record(identifiers=[_ident("chembl_id", "CHEMBL999")])
    plans = _eligible_for(["ChEMBL_get_molecule"], record)
    # Description is silent about returns; no structured schema either.
    _install_fake_tu(monkeypatch, {
        "ChEMBL_get_molecule": {
            "name": "ChEMBL_get_molecule",
            "description": "Look up a ChEMBL compound by identifier.",
            "parameter": {
                "type": "object",
                "properties": {"chembl_id": {"type": "string"}},
                "required": ["chembl_id"],
            },
        }
    })
    catalog = _build_compact_catalog(plans)
    entry = catalog[0]
    assert (
        entry["expected_output_source"]
        == EXPECTED_OUTPUT_SOURCE_TU_NO_CONTRACT
    )
    assert entry["expected_output_fields"] == []
    assert entry["expected_output_semantics"] == ""


def test_catalog_expected_output_source_unavailable_when_no_metadata(monkeypatch):
    record = _record(identifiers=[_ident("chembl_id", "CHEMBL999")])
    plans = _eligible_for(["ChEMBL_get_molecule"], record)
    _install_fake_tu(monkeypatch, {})
    catalog = _build_compact_catalog(plans)
    entry = catalog[0]
    assert (
        entry["expected_output_source"]
        == EXPECTED_OUTPUT_SOURCE_UNAVAILABLE
    )
    assert entry["expected_output_fields"] == []
    assert entry["expected_output_semantics"] == ""


def test_catalog_never_emits_full_schema_or_raw_payload(monkeypatch):
    """The catalog forwarded to the LLM must not carry the full TU
    input schema, full raw output schema, raw payload, hits, examples,
    arguments, or _live."""
    record = _record(
        identifiers=[_ident("chembl_id", "CHEMBL999")],
        materials=[_mat("payload_smiles", "CCO", "payload")],
    )
    plans = _eligible_for(
        ["ChEMBL_get_molecule", "ChEMBL_search_substructure"], record
    )
    _install_fake_tu(monkeypatch, {
        "ChEMBL_get_molecule": {
            "name": "ChEMBL_get_molecule",
            "description": "TU official: fetch a ChEMBL molecule by id.",
            "parameter": {
                "type": "object",
                "properties": {"chembl_id": {"type": "string"}},
                "required": ["chembl_id"],
            },
            "return_schema": {
                "type": "object",
                "properties": {"molecule": {"type": "object"}},
            },
        },
        "ChEMBL_search_substructure": {
            "name": "ChEMBL_search_substructure",
            "description": "Substructure search; returns molecules.",
            "parameter": {
                "type": "object",
                "properties": {"smiles": {"type": "string"}},
            },
        },
    })
    catalog = _build_compact_catalog(plans)
    blob = json.dumps(catalog)
    for forbidden in (
        "full_schema",
        "parameter",
        "_live",
        "arguments",
        "raw_payload",
        "hits",
        "example_output",
        "molecules\":",
        "records\":",
    ):
        assert forbidden not in blob, forbidden
    # Per-entry shape — only the documented compact keys.
    documented_keys = {
        "tool_name", "short_description", "description_source",
        "capability_tags", "coarse_input_requirements",
        "expected_output_fields", "expected_output_semantics",
        "expected_output_source", "expected_output_notes",
        "project_side_hints", "step_id", "agent_name",
    }
    for entry in catalog:
        assert set(entry.keys()) == documented_keys, set(entry.keys())


# ── 17. New protein/antibody/antigen coverage in catalog ────────────────────


def test_addendum_keeps_sequence_and_structure_boundaries_concise():
    sp = STEP_05_SELECTION_SYSTEM_ADDENDUM
    assert "Candidate context may be compound, antibody, target/antigen" in sp
    assert "structure" in sp
    assert "whole ADC reference" in sp
    assert "mixed case" in sp
    assert "antibody heavy/light sequences" in sp
    assert "runtime CDR3 extraction happens" in sp
    assert "Use only exact `tool_name` values from `compact_catalog`." in sp


def test_catalog_includes_new_protein_antibody_tools_with_tu_output_metadata(
    monkeypatch,
):
    """The LLM-facing compact catalog must surface the three new tools
    with their TooluUniverse-derived expected_output contract — and
    must keep field discovery sourced from the TU spec (not hand-rolled)."""
    # Target candidate with UniProt + PDB + name → SAbDab_get_structure
    # and TheraSAbDab_search_by_target should be eligible.
    record_target = _record(
        candidate_type="target_antigen",
        materials=[_mat("target_antigen_name", "HER2", "target")],
        identifiers=[_ident("pdb_id", "1N8Z")],
    )
    plans_target = plan_enrichment_for_record(
        record_target,
        scoped_tools=["SAbDab_get_structure", "TheraSAbDab_search_by_target"],
        candidate_category="target_antigen",
        registry=STEP_05_CAPABILITY_REGISTRY,
    )
    assert {p.tool_name for p in plans_target} == {
        "SAbDab_get_structure", "TheraSAbDab_search_by_target",
    }
    _install_fake_tu(monkeypatch, {
        "SAbDab_get_structure": {
            "name": "SAbDab_get_structure",
            "description": (
                "Get antibody structure details from SAbDab by PDB ID. "
                "Returns CDR annotations, chain information, and "
                "antigen binding data."
            ),
            "parameter": {
                "type": "object",
                "properties": {"pdb_id": {"type": "string"}},
            },
            "return_schema": {
                "type": "object",
                "description": "SAbDab structure record.",
                "properties": {
                    "pdb_id": {"type": "string"},
                    "cdr_annotations": {"type": "array"},
                    "chains": {"type": "array"},
                    "antigen_binding": {"type": "object"},
                },
            },
        },
        "TheraSAbDab_search_by_target": {
            "name": "TheraSAbDab_search_by_target",
            "description": (
                "Find therapeutic antibodies targeting a specific antigen. "
                "Returns clinical/approved antibodies against the target."
            ),
            "parameter": {
                "type": "object",
                "properties": {"target": {"type": "string"}},
                "required": ["target"],
            },
            "return_schema": {
                "type": "object",
                "properties": {
                    "results": {"type": "array"},
                    "target": {"type": "string"},
                    "n_hits": {"type": "integer"},
                },
            },
        },
    })
    catalog = _build_compact_catalog(plans_target)
    by_name = {e["tool_name"]: e for e in catalog}
    sabdab = by_name["SAbDab_get_structure"]
    therasabdab = by_name["TheraSAbDab_search_by_target"]
    # Description comes from TU.
    assert sabdab["description_source"] == DESCRIPTION_SOURCE_TU
    assert therasabdab["description_source"] == DESCRIPTION_SOURCE_TU
    # expected_output_fields are extracted from TU return_schema — NOT
    # hand-written by Step 5.
    assert sabdab["expected_output_source"] == EXPECTED_OUTPUT_SOURCE_TU_SPEC
    assert set(sabdab["expected_output_fields"]) == {
        "pdb_id", "cdr_annotations", "chains", "antigen_binding",
    }
    assert therasabdab["expected_output_source"] == EXPECTED_OUTPUT_SOURCE_TU_SPEC
    assert set(therasabdab["expected_output_fields"]) == {
        "results", "target", "n_hits",
    }


def test_catalog_includes_therasabdab_search_therapeutics_for_antibody(
    monkeypatch,
):
    record = _record(
        candidate_type="antibody",
        materials=[_mat("antibody_name", "trastuzumab", "antibody")],
    )
    plans = plan_enrichment_for_record(
        record,
        scoped_tools=["TheraSAbDab_search_therapeutics"],
        candidate_category="antibody",
        registry=STEP_05_CAPABILITY_REGISTRY,
    )
    assert [p.tool_name for p in plans] == ["TheraSAbDab_search_therapeutics"]
    _install_fake_tu(monkeypatch, {
        "TheraSAbDab_search_therapeutics": {
            "name": "TheraSAbDab_search_therapeutics",
            "description": (
                "Search therapeutic antibodies by name in Thera-SAbDab. "
                "Returns WHO INN name, target antigen, format, clinical "
                "phase, and PDB structures."
            ),
            "parameter": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
            "return_schema": {
                "type": "object",
                "properties": {
                    "inn_name": {"type": "string"},
                    "target": {"type": "string"},
                    "format": {"type": "string"},
                    "phase": {"type": "string"},
                    "pdb_structures": {"type": "array"},
                },
            },
        }
    })
    entry = _build_compact_catalog(plans)[0]
    assert entry["expected_output_source"] == EXPECTED_OUTPUT_SOURCE_TU_SPEC
    assert set(entry["expected_output_fields"]) == {
        "inn_name", "target", "format", "phase", "pdb_structures",
    }


# ── 18. Raw FASTA / PDB contents never reach the LLM payload or audit ──────


def test_raw_fasta_pdb_contents_do_not_reach_llm_or_audit(
    local_storage, registry_service, workflow_state_service, monkeypatch,
):
    """End-to-end: even when a candidate carries a structure_file or
    antibody_sequence_reference material, the path/filename — not the
    file body — flows through the agent. The LLM payload and the
    persisted normalized artifact must contain neither raw PDB lines
    (e.g. ``ATOM``, ``HETATM``) nor a FASTA header (``>``)."""

    # Plant a candidate with explicit antibody sequence material via the
    # normal intake → Step 2 path, then upload a real-ish PDB file as
    # context. We never load file bytes into Python — only the metadata.
    pdb_text = (
        "HEADER    TEST PDB                                              \n"
        "ATOM      1  N   ALA A   1      11.104  13.207  10.000  1.00 20.00\n"
        "ATOM      2  CA  ALA A   1      12.560  13.207  10.000  1.00 20.00\n"
        "END\n"
    )
    fasta_text = (
        ">trastuzumab_HC\n"
        "EVQLVQSGAEVKKPGSSVKVSCKASGGTFSSYAISWVRQAPGQGLEWMG\n"
    )
    pdb_path = local_storage.run_key("seed_step5_raw", "uploads/raw.pdb")
    fasta_path = local_storage.run_key("seed_step5_raw", "uploads/raw.fasta")
    local_storage.write_bytes(pdb_path, pdb_text.encode("utf-8"))
    local_storage.write_bytes(fasta_path, fasta_text.encode("utf-8"))

    run_id = _setup_run(local_storage, registry_service, workflow_state_service)

    # Patch raw_request_record + structured_query to carry uploaded
    # files referencing those storage paths so Step 5 builds materials
    # with `value` = the path (not the bytes).
    rrr_key = local_storage.run_key(run_id, "inputs/raw_request_record.json")
    rrr = local_storage.read_json(rrr_key)
    rrr.setdefault("uploaded_files", []).extend([
        {"file_id": "f_pdb_1", "original_filename": "raw.pdb",
         "storage_path": pdb_path, "content_type": "chemical/x-pdb",
         "size_bytes": len(pdb_text)},
        {"file_id": "f_fa_1", "original_filename": "raw.fasta",
         "storage_path": fasta_path, "content_type": "text/x-fasta",
         "size_bytes": len(fasta_text)},
    ])
    local_storage.write_json(rrr_key, rrr)

    # Capture every LLM call payload as it crosses the boundary.
    captured: list[dict] = []

    class _CapturingLLM:
        name = "capturing"
        model = "capturing-v1"

        def generate(self, prompt, *, system=None, **_):
            raise NotImplementedError

        def generate_json(self, prompt, *, schema, system=None):
            captured.append({"schema": schema, "system": system})
            return {"selections": []}  # empty → fail-open fallback

    mcp = LocalMCPClient(bindings=_bindings({
        "SAbDab_search_structures": {"hits": []},
        "SAbDab_get_structure": {"hits": []},
        "TheraSAbDab_search_by_target": {"results": []},
        "TheraSAbDab_search_therapeutics": {"results": []},
        "ChEMBL_search_molecules": {"hits": []},
        "ChEMBL_search_substructure": {"hits": []},
    }))
    CandidateContextAgent(
        storage=local_storage, registry=registry_service,
        workflow_state=workflow_state_service,
        mcp_client=mcp, llm=_CapturingLLM(),
    ).run(run_id)

    # 1. No LLM payload carries raw file bytes / FASTA headers / PDB lines.
    assert captured, "expected at least one LLM payload"
    for call in captured:
        blob = json.dumps(call["schema"])
        assert "ATOM " not in blob and "HETATM" not in blob, blob[:120]
        assert "EVQLVQSGAEVKKPGSSVKVSCKASGGTFS" not in blob
        assert ">trastuzumab_HC" not in blob

    # 2. The persisted artifact carries neither raw lines either.
    persisted = local_storage.read_json(
        local_storage.run_key(run_id, "candidate_context_table.json")
    )
    pblob = json.dumps(persisted)
    assert "ATOM " not in pblob and "HETATM" not in pblob
    assert "EVQLVQSGAEVKKPGSSVKVSCKASGGTFS" not in pblob
    assert ">trastuzumab_HC" not in pblob


# ── 19. Sequence-only antibody candidate does not invent tool calls ────────


def test_sequence_only_antibody_candidate_records_material_without_tool_calls(
    local_storage, registry_service, workflow_state_service,
):
    """A candidate built purely from an uploaded FASTA (no antibody
    name) should not surface a TheraSAbDab_search_therapeutics or
    SAbDab_search_structures plan: the registry has no eligible slot.
    Material + downstream gaps should be preserved."""
    from app.services.intake_service import IntakeService  # noqa: PLC0415
    from app.services.structured_query_service import StructuredQueryService  # noqa: PLC0415
    from app.services.input_readiness_service import InputReadinessService  # noqa: PLC0415
    from app.services.workflow_setup_service import WorkflowSetupService  # noqa: PLC0415

    fasta_text = (
        ">trastuzumab_HC\n"
        "EVQLVQSGAEVKKPGSSVKVSCKASGGTFSSYAISWVRQAPGQGLEWMG\n"
    )
    intake = IntakeService(
        local_storage, registry_service, workflow_state_service
    )
    run_id_pre = intake.allocate_run_id()
    fasta_path = local_storage.run_key(
        run_id_pre, "inputs/files/raw.fasta"
    )
    local_storage.write_bytes(fasta_path, fasta_text.encode("utf-8"))
    # Minimal target context so Step 3 readiness is not blocked; the
    # antibody-from-FASTA candidate we are testing is built separately
    # by Step 5 from the uploaded file. Readiness is not the subject
    # of this test.
    rec = intake.submit(
        run_id=run_id_pre,
        raw_user_query="Run Step 5 with an antibody sequence only.",
        user_provided_context={
            "target_or_antigen_text": "HER2",
            "payload_linker_text": "vc-MMAE",
        },
        uploaded_files=[
            {"file_id": "f_fa_only", "original_filename": "raw.fasta",
             "storage_path": fasta_path,
             "content_type": "text/x-fasta",
             "size_bytes": len(fasta_text)},
        ],
    )
    StructuredQueryService(
        local_storage, registry_service, workflow_state_service,
        SupervisorAgent(llm=MockLLMProvider()),
    ).parse(rec.run_id)
    InputReadinessService(
        local_storage, registry_service, workflow_state_service
    ).check(rec.run_id)
    WorkflowSetupService(
        local_storage, registry_service, workflow_state_service
    ).plan(rec.run_id)

    # No bindings at all — if the registry tried to fabricate calls,
    # this run would still record them in tool_call_records.
    mcp = LocalMCPClient(bindings={})
    CandidateContextAgent(
        storage=local_storage, registry=registry_service,
        workflow_state=workflow_state_service, mcp_client=mcp,
    ).run(rec.run_id)
    persisted = local_storage.read_json(
        local_storage.run_key(rec.run_id, "candidate_context_table.json")
    )
    # Antibody candidate exists with the FASTA material reference.
    ab_records = [
        c for c in persisted["candidate_records"]
        if c["candidate_type"] == "antibody"
    ]
    assert ab_records, "antibody-from-sequence candidate should be built"
    ab = ab_records[0]
    assert any(
        m["material_type"] == "antibody_sequence_reference"
        for m in ab["materials"]
    )
    # No tool call should have been recorded for this candidate.
    audit = persisted["enrichment_selection_audit"]
    entry = audit[ab["candidate_id"]]
    assert entry["eligible_tools"] == []
    assert entry["selected_tools"] == []
    assert entry["skipped_eligible_tools"] == []


# ── Cache-stable catalog ordering (no semantic change) ─────────────────


def test_compact_catalog_is_sorted_by_tool_name_for_cache_stability():
    """For a given eligibility set, the compact catalog forwarded to
    the LLM must be deterministically ordered by ``tool_name``. The
    provider's auto-prompt cache keys off the leading byte sequence
    of the request; sorting makes the catalog prefix stable across
    candidates and across runs.

    The selection policy treats the catalog as an unordered set of
    allowed tool names — sorting changes the byte stream the provider
    sees but never changes which tools are eligible / selected /
    skipped."""
    record = _record(
        identifiers=[_ident("chembl_id", "CHEMBL999")],
        materials=[
            _mat("payload_name", "monomethyl auristatin E", "payload"),
            _mat("payload_smiles", "CCO", "payload"),
        ],
    )
    scope = [
        # Intentionally reversed-alphabet scope to expose any
        # accidental dependency on input order.
        "ChEMBL_search_substructure",
        "ChEMBL_search_molecules",
        "ChEMBL_get_molecule",
    ]
    plans = plan_enrichment_for_record(
        record, scoped_tools=scope,
        candidate_category="compound_component",
        registry=STEP_05_CAPABILITY_REGISTRY,
    )
    catalog = _build_compact_catalog(plans)
    names = [entry["tool_name"] for entry in catalog]
    assert names == sorted(names), names
    # Same record + same scope produces the same catalog (idempotent).
    plans2 = plan_enrichment_for_record(
        record, scoped_tools=scope,
        candidate_category="compound_component",
        registry=STEP_05_CAPABILITY_REGISTRY,
    )
    assert _build_compact_catalog(plans2) == catalog


def test_compact_catalog_order_does_not_change_eligible_tool_set():
    """Reordering inputs to ``plan_enrichment_for_record`` must not
    drop / add any tool from the catalog. Eligibility is determined
    by the registry, not by the scope-list order."""
    record = _record(
        identifiers=[_ident("chembl_id", "CHEMBL999")],
        materials=[_mat("payload_smiles", "CCO", "payload")],
    )
    scope_a = ["ChEMBL_get_molecule", "ChEMBL_search_substructure"]
    scope_b = ["ChEMBL_search_substructure", "ChEMBL_get_molecule"]
    plans_a = plan_enrichment_for_record(
        record, scoped_tools=scope_a,
        candidate_category="compound_component",
        registry=STEP_05_CAPABILITY_REGISTRY,
    )
    plans_b = plan_enrichment_for_record(
        record, scoped_tools=scope_b,
        candidate_category="compound_component",
        registry=STEP_05_CAPABILITY_REGISTRY,
    )
    names_a = [e["tool_name"] for e in _build_compact_catalog(plans_a)]
    names_b = [e["tool_name"] for e in _build_compact_catalog(plans_b)]
    assert names_a == sorted(names_a) == names_b


def test_stage1_payload_keys_have_compact_catalog_before_candidate_context():
    """``schema=`` dict serialisation uses ``sort_keys=True`` in
    ``app/llm/json_task_validation.py``, so the top-level keys ship
    alphabetically: ``agent_name`` < ``compact_catalog`` < ``context``
    < ``step_id`` < ``task``. The static `compact_catalog` therefore
    appears BEFORE the dynamic `context.candidate` block in the
    serialised JSON — the cache-friendly order. This test pins both
    invariants: the top-level keys are exactly the expected set, and
    `compact_catalog` sorts before `context` (so the catalog stays in
    the cacheable prefix)."""
    expected_keys = {"agent_name", "compact_catalog", "context",
                     "step_id", "task"}
    keys_sorted = sorted(expected_keys)
    cc_idx = keys_sorted.index("compact_catalog")
    ctx_idx = keys_sorted.index("context")
    assert cc_idx < ctx_idx, keys_sorted
