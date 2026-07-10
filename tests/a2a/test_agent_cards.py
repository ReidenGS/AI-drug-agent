"""Turn A — AgentCard builders, adc_agent_contract validator, compact catalog.

These tests exercise the real production artifact contract. The storage paths
asserted here are the exact paths the live Step 5 / Step 6 / structure agents
read and write (verified against app/agents/*.py). The structure worker exposes
ONE Orchestrator-facing capability (``structure_design_workflow``); Step 7/8/9
are its internal serial steps, not independently dispatchable A2A capabilities.
No test-only caps, mocks, allowlists, or narrowed constraints are introduced.
"""

from __future__ import annotations

import json

import pytest
from python_a2a import AgentCard

from app.a2a.agent_cards import (
    AGENT_ID_STEP5,
    AGENT_ID_STEP6,
    AGENT_ID_STRUCTURE,
    CAP_STEP5_CANDIDATE_CONTEXT,
    CAP_STEP6_DEVELOPABILITY,
    CAP_STRUCTURE_DESIGN_WORKFLOW,
    STEP_07_STRUCTURE_INPUT,
    STEP_08_STRUCTURE_EVALUATION,
    STEP_09_STRUCTURE_DESIGN,
    AgentContractError,
    build_compact_card_catalog,
    build_compact_card_for_agent,
    build_step5_agent_card,
    build_step6_agent_card,
    build_structure_agent_card,
    parse_adc_agent_contract,
    validate_adc_agent_contract,
)

STEP5_URL = "http://step5-worker:8005"
STEP6_URL = "http://step6-worker:8006"
STRUCTURE_URL = "http://structure-worker:8009"

# The three old, now-removed structure capability ids. They must NOT reappear as
# capability ids / skill ids anywhere.
_OLD_STRUCTURE_CAPABILITY_IDS = {
    "step_07_structure_input",
    "step_08_structure_evaluation",
    "step_09_structure_design",
}


def _contract(card: AgentCard) -> dict:
    return card.capabilities["adc_agent_contract"]


def _only_cap(contract: dict, capability_id: str) -> dict:
    caps = [c for c in contract["capabilities"] if c["capability_id"] == capability_id]
    assert caps, f"capability {capability_id} not found"
    return caps[0]


def _required_paths(contract: dict, capability_id: str) -> dict[str, str]:
    cap = _only_cap(contract, capability_id)
    return {r["artifact_name"]: r["storage_path"] for r in cap["required_input_artifacts"]}


def _optional_paths(contract: dict, capability_id: str) -> dict[str, str]:
    cap = _only_cap(contract, capability_id)
    return {r["artifact_name"]: r["storage_path"] for r in cap["optional_input_artifacts"]}


def _output_pairs(contract: dict, capability_id: str) -> list[tuple[str, str]]:
    cap = _only_cap(contract, capability_id)
    return [(r["artifact_name"], r["storage_path"]) for r in cap["output_artifacts"]]


# ── 1. all three cards build and are python_a2a.AgentCard ────────────────────
def test_cards_build_as_python_a2a_agent_cards():
    for card in (
        build_step5_agent_card(STEP5_URL),
        build_step6_agent_card(STEP6_URL),
        build_structure_agent_card(STRUCTURE_URL),
    ):
        assert isinstance(card, AgentCard)
        assert card.skills, "each card must publish at least one AgentSkill"


# ── 2. each card carries capabilities['adc_agent_contract'] ──────────────────
@pytest.mark.parametrize(
    "builder, url, agent_id",
    [
        (build_step5_agent_card, STEP5_URL, AGENT_ID_STEP5),
        (build_step6_agent_card, STEP6_URL, AGENT_ID_STEP6),
        (build_structure_agent_card, STRUCTURE_URL, AGENT_ID_STRUCTURE),
    ],
)
def test_card_has_adc_agent_contract(builder, url, agent_id):
    card = builder(url)
    assert "adc_agent_contract" in card.capabilities
    contract = _contract(card)
    assert contract["agent_id"] == agent_id
    assert contract["agent_role"] == "worker"
    parsed = validate_adc_agent_contract(card)
    assert parsed.agent_id == agent_id


# ── 3. Step 5 input/output contract == production read/write paths ───────────
def test_step5_artifact_contract_matches_production():
    contract = _contract(build_step5_agent_card(STEP5_URL))
    assert _required_paths(contract, CAP_STEP5_CANDIDATE_CONTEXT) == {
        "raw_request_record": "inputs/raw_request_record.json",
        "structured_query": "inputs/structured_query.json",
        "run_step_plan": "inputs/run_step_plan.json",
    }
    assert _output_pairs(contract, CAP_STEP5_CANDIDATE_CONTEXT) == [
        ("candidate_context_table", "candidate_context_table.json"),
    ]
    # Step 5 is a single_step capability with no internal workflow.
    cap = _only_cap(contract, CAP_STEP5_CANDIDATE_CONTEXT)
    assert cap["execution_mode"] == "single_step"
    assert cap["internal_execution_order"] == []


# ── 4. Step 6 input/output contract == production read/write paths ───────────
def test_step6_artifact_contract_matches_production():
    contract = _contract(build_step6_agent_card(STEP6_URL))
    assert _required_paths(contract, CAP_STEP6_DEVELOPABILITY) == {
        "candidate_context_table": "candidate_context_table.json",
        "run_step_plan": "inputs/run_step_plan.json",
    }
    assert _output_pairs(contract, CAP_STEP6_DEVELOPABILITY) == [
        ("structured_liability_summary", "structured_liability_summary.json"),
    ]
    cap = _only_cap(contract, CAP_STEP6_DEVELOPABILITY)
    assert cap["execution_mode"] == "single_step"
    assert cap["internal_execution_order"] == []


# ── Structure worker: single capability ──────────────────────────────────────
def test_structure_card_has_exactly_one_skill_and_capability():
    card = build_structure_agent_card(STRUCTURE_URL)
    # Exactly one AgentSkill.
    assert len(card.skills) == 1
    assert card.skills[0].id == CAP_STRUCTURE_DESIGN_WORKFLOW

    contract = _contract(card)
    cap_ids = [c["capability_id"] for c in contract["capabilities"]]
    assert cap_ids == [CAP_STRUCTURE_DESIGN_WORKFLOW]


def test_structure_old_capability_ids_are_gone_everywhere():
    card = build_structure_agent_card(STRUCTURE_URL)
    contract = _contract(card)

    skill_ids = {s.id for s in card.skills}
    assert not (skill_ids & _OLD_STRUCTURE_CAPABILITY_IDS)

    contract_cap_ids = {c["capability_id"] for c in contract["capabilities"]}
    assert not (contract_cap_ids & _OLD_STRUCTURE_CAPABILITY_IDS)

    compact = build_compact_card_for_agent(card)
    compact_cap_ids = {c["capability_id"] for c in compact["capabilities"]}
    assert not (compact_cap_ids & _OLD_STRUCTURE_CAPABILITY_IDS)


def test_structure_execution_mode_and_internal_order():
    contract = _contract(build_structure_agent_card(STRUCTURE_URL))
    cap = _only_cap(contract, CAP_STRUCTURE_DESIGN_WORKFLOW)
    assert cap["execution_mode"] == "sequential_workflow"
    assert cap["internal_execution_order"] == [
        STEP_07_STRUCTURE_INPUT,
        STEP_08_STRUCTURE_EVALUATION,
        STEP_09_STRUCTURE_DESIGN,
    ]
    # And those steps are declared as supported_step_ids.
    for step in cap["internal_execution_order"]:
        assert step in cap["supported_step_ids"]


def test_structure_required_inputs_are_workflow_entry_only():
    contract = _contract(build_structure_agent_card(STRUCTURE_URL))
    required = _required_paths(contract, CAP_STRUCTURE_DESIGN_WORKFLOW)
    assert required == {
        "raw_request_record": "inputs/raw_request_record.json",
        "structured_query": "inputs/structured_query.json",
        "candidate_context_table": "candidate_context_table.json",
    }
    # Internal Step 7/8 outputs and Step 4 control plan are NOT required inputs.
    for forbidden in (
        "run_step_plan",
        "prepared_structure_input_package",
        "structure_prediction_and_interface_results",
    ):
        assert forbidden not in required


def test_structure_optional_inputs_include_liability_summary():
    contract = _contract(build_structure_agent_card(STRUCTURE_URL))
    assert _optional_paths(contract, CAP_STRUCTURE_DESIGN_WORKFLOW) == {
        "structured_liability_summary": "structured_liability_summary.json",
    }


def test_structure_outputs_are_step7_8_9_in_order():
    contract = _contract(build_structure_agent_card(STRUCTURE_URL))
    assert _output_pairs(contract, CAP_STRUCTURE_DESIGN_WORKFLOW) == [
        ("prepared_structure_input_package", "prepared_structure_input_package.json"),
        (
            "structure_prediction_and_interface_results",
            "structure_prediction_and_interface_results.json",
        ),
        ("structure_variant_and_compound_screening", "compound_screening_artifact.json"),
    ]


def test_structure_description_wording_is_accurate():
    """Orchestrator/LLM-visible strings must describe the ACTIVE workflow only:
    no compound screening / ZINC / ChEMBL wording. The legacy output artifact
    name ``structure_variant_and_compound_screening`` is retained for JSON/API
    compatibility and is intentionally not covered by this text check."""
    card = build_structure_agent_card(STRUCTURE_URL)
    contract = _contract(card)
    cap = _only_cap(contract, CAP_STRUCTURE_DESIGN_WORKFLOW)

    agent_desc = contract["description"].lower()
    cap_summary = cap["capability_summary"].lower()

    for text in (agent_desc, cap_summary):
        assert "compound screening" not in text
        assert "compound" not in text
        assert "zinc" not in text
        assert "chembl" not in text

    # Still expresses the active structure design + variant evaluation workflow.
    assert "design" in agent_desc and "variant evaluation" in agent_desc
    assert "design" in cap_summary and "variant evaluation" in cap_summary

    # The compact catalog (LLM-facing) carries the same corrected summary.
    compact = build_compact_card_for_agent(card)
    compact_summary = compact["capabilities"][0]["capability_summary"].lower()
    for needle in ("compound screening", "compound", "zinc", "chembl"):
        assert needle not in compact_summary
    assert "design" in compact_summary and "variant evaluation" in compact_summary


def test_structure_required_artifact_fields_use_real_schema_keys():
    contract = _contract(build_structure_agent_card(STRUCTURE_URL))
    cap = _only_cap(contract, CAP_STRUCTURE_DESIGN_WORKFLOW)
    raf = cap["required_artifact_fields"]
    assert set(raf) == {"raw_request_record", "structured_query", "candidate_context_table"}

    assert raf["raw_request_record"]["required_field_keys"] == [
        "raw_user_query",
        "user_provided_context",
        "uploaded_files",
    ]
    assert raf["structured_query"]["required_field_keys"] == [
        "task_intent",
        "referenced_inputs",
        "requested_outputs",
        "user_constraints",
        "normalized_entities",
        "canonical_query",
    ]
    assert raf["candidate_context_table"]["entity_type"] == "candidate"
    assert raf["candidate_context_table"]["default_selection_mode"] == "all_in_artifact"
    assert raf["candidate_context_table"]["required_field_keys"] == [
        "candidate_records",
        "downstream_query_hints",
    ]


# ── 6. AgentCard url can be a Docker internal service name ───────────────────
def test_card_url_accepts_docker_internal_service_name():
    card = build_step6_agent_card("http://step6-worker:8006")
    assert card.url == "http://step6-worker:8006"
    validate_adc_agent_contract(card)


# ── 7. invalid adc_agent_contract is rejected (no silent pass) ───────────────
def test_missing_contract_is_rejected():
    card = AgentCard(name="x", description="d", url="http://x:1", version="1.0.0")
    with pytest.raises(AgentContractError):
        parse_adc_agent_contract(card)


def test_malformed_contract_unknown_field_is_rejected():
    card = build_step5_agent_card(STEP5_URL)
    card.capabilities["adc_agent_contract"]["totally_unknown_field"] = True
    with pytest.raises(AgentContractError):
        parse_adc_agent_contract(card)


def test_contract_missing_agent_id_is_rejected():
    card = build_step5_agent_card(STEP5_URL)
    card.capabilities["adc_agent_contract"]["agent_id"] = ""
    with pytest.raises(AgentContractError):
        parse_adc_agent_contract(card)


def test_contract_bad_dispatch_mode_is_rejected():
    card = build_step5_agent_card(STEP5_URL)
    card.capabilities["adc_agent_contract"]["dispatch_modes"] = ["local_call"]
    with pytest.raises(AgentContractError):
        parse_adc_agent_contract(card)


def test_routable_worker_without_url_is_rejected():
    card = build_step5_agent_card(STEP5_URL)
    card.url = ""
    with pytest.raises(AgentContractError):
        parse_adc_agent_contract(card)


def test_output_artifact_missing_storage_path_is_rejected():
    card = build_step5_agent_card(STEP5_URL)
    outputs = card.capabilities["adc_agent_contract"]["capabilities"][0]["output_artifacts"]
    outputs[0]["storage_path"] = ""
    with pytest.raises(AgentContractError):
        parse_adc_agent_contract(card)


def test_skills_capability_misalignment_is_rejected():
    card = build_step6_agent_card(STEP6_URL)
    card.skills = []
    with pytest.raises(AgentContractError):
        parse_adc_agent_contract(card)


# ── execution-mode / internal-order validation failures ──────────────────────
def test_sequential_workflow_without_order_is_rejected():
    card = build_structure_agent_card(STRUCTURE_URL)
    _only_cap(_contract(card), CAP_STRUCTURE_DESIGN_WORKFLOW)  # sanity
    cap = card.capabilities["adc_agent_contract"]["capabilities"][0]
    cap["internal_execution_order"] = []
    with pytest.raises(AgentContractError):
        parse_adc_agent_contract(card)


def test_internal_order_with_unknown_step_is_rejected():
    card = build_structure_agent_card(STRUCTURE_URL)
    cap = card.capabilities["adc_agent_contract"]["capabilities"][0]
    cap["internal_execution_order"] = [
        STEP_07_STRUCTURE_INPUT,
        "step_99_unknown",
        STEP_09_STRUCTURE_DESIGN,
    ]
    with pytest.raises(AgentContractError):
        parse_adc_agent_contract(card)


def test_internal_order_with_duplicate_step_is_rejected():
    card = build_structure_agent_card(STRUCTURE_URL)
    cap = card.capabilities["adc_agent_contract"]["capabilities"][0]
    cap["internal_execution_order"] = [
        STEP_07_STRUCTURE_INPUT,
        STEP_07_STRUCTURE_INPUT,
        STEP_09_STRUCTURE_DESIGN,
    ]
    with pytest.raises(AgentContractError):
        parse_adc_agent_contract(card)


def test_single_step_with_multiple_internal_steps_is_rejected():
    card = build_step5_agent_card(STEP5_URL)
    cap = card.capabilities["adc_agent_contract"]["capabilities"][0]
    cap["supported_step_ids"] = ["step_05_candidate_context", "step_05b_extra"]
    cap["internal_execution_order"] = ["step_05_candidate_context", "step_05b_extra"]
    with pytest.raises(AgentContractError):
        parse_adc_agent_contract(card)


# ── required_artifact_fields validation failures ─────────────────────────────
def test_required_artifact_fields_referencing_undeclared_artifact_is_rejected():
    card = build_structure_agent_card(STRUCTURE_URL)
    cap = card.capabilities["adc_agent_contract"]["capabilities"][0]
    # structured_liability_summary is optional, not required — cannot carry a
    # required field contract.
    cap["required_artifact_fields"]["structured_liability_summary"] = {
        "required_field_keys": ["prefilter_status"],
        "entity_type": None,
        "default_selection_mode": None,
    }
    with pytest.raises(AgentContractError):
        parse_adc_agent_contract(card)


def test_required_artifact_fields_empty_key_is_rejected():
    card = build_structure_agent_card(STRUCTURE_URL)
    cap = card.capabilities["adc_agent_contract"]["capabilities"][0]
    cap["required_artifact_fields"]["structured_query"]["required_field_keys"] = [
        "task_intent",
        "  ",
    ]
    with pytest.raises(AgentContractError):
        parse_adc_agent_contract(card)


def test_required_artifact_fields_duplicate_key_is_rejected():
    card = build_structure_agent_card(STRUCTURE_URL)
    cap = card.capabilities["adc_agent_contract"]["capabilities"][0]
    cap["required_artifact_fields"]["structured_query"]["required_field_keys"] = [
        "task_intent",
        "task_intent",
    ]
    with pytest.raises(AgentContractError):
        parse_adc_agent_contract(card)


# ── compact catalog ──────────────────────────────────────────────────────────
def _all_cards():
    return [
        build_step5_agent_card(STEP5_URL),
        build_step6_agent_card(STEP6_URL),
        build_structure_agent_card(STRUCTURE_URL),
    ]


def test_compact_catalog_excludes_urls_and_raw_material():
    catalog = build_compact_card_catalog(_all_cards())
    blob = json.dumps(catalog).lower()

    for needle in ("http://", "step5-worker", "step6-worker", "structure-worker", "8005", "8009"):
        assert needle not in blob, f"compact catalog leaked endpoint token: {needle}"

    for needle in (
        "api_key",
        "apikey",
        "authorization",
        "bearer",
        "raw_sequence",
        "fasta",
        "pdb",
        "cif",
        "a3m",
        "tooluniverse",
        "full_prompt",
        "raw_llm_response",
    ):
        assert needle not in blob, f"compact catalog leaked forbidden token: {needle}"


def test_compact_catalog_structure_has_single_capability_with_routing_info():
    catalog = build_compact_card_catalog(_all_cards())
    by_agent = {entry["agent_id"]: entry for entry in catalog}
    assert set(by_agent) == {AGENT_ID_STEP5, AGENT_ID_STEP6, AGENT_ID_STRUCTURE}

    structure = by_agent[AGENT_ID_STRUCTURE]
    assert len(structure["capabilities"]) == 1
    cap = structure["capabilities"][0]
    assert cap["capability_id"] == CAP_STRUCTURE_DESIGN_WORKFLOW
    assert cap["execution_mode"] == "sequential_workflow"
    assert cap["internal_execution_order"] == [
        STEP_07_STRUCTURE_INPUT,
        STEP_08_STRUCTURE_EVALUATION,
        STEP_09_STRUCTURE_DESIGN,
    ]
    assert cap["required_input_artifact_names"] == [
        "raw_request_record",
        "structured_query",
        "candidate_context_table",
    ]
    assert cap["optional_input_artifact_names"] == ["structured_liability_summary"]
    assert cap["output_artifact_names"] == [
        "prepared_structure_input_package",
        "structure_prediction_and_interface_results",
        "structure_variant_and_compound_screening",
    ]
    # Field NAMES exposed, no values/bodies.
    assert set(cap["required_artifact_field_names"]) == {
        "raw_request_record",
        "structured_query",
        "candidate_context_table",
    }


def test_compact_catalog_step5_step6_single_step_no_regression():
    catalog = build_compact_card_catalog(_all_cards())
    by_agent = {entry["agent_id"]: entry for entry in catalog}

    step5 = by_agent[AGENT_ID_STEP5]
    assert len(step5["capabilities"]) == 1
    assert step5["capabilities"][0]["capability_id"] == CAP_STEP5_CANDIDATE_CONTEXT
    assert step5["capabilities"][0]["execution_mode"] == "single_step"
    assert step5["capabilities"][0]["internal_execution_order"] == []

    step6 = by_agent[AGENT_ID_STEP6]
    assert len(step6["capabilities"]) == 1
    assert step6["capabilities"][0]["capability_id"] == CAP_STEP6_DEVELOPABILITY
    assert step6["capabilities"][0]["execution_mode"] == "single_step"
    assert step6["capabilities"][0]["required_input_artifact_names"] == [
        "candidate_context_table",
        "run_step_plan",
    ]
    assert step6["capabilities"][0]["output_artifact_names"] == [
        "structured_liability_summary",
    ]


# ── skills/capabilities 1:1 alignment still holds for all cards ──────────────
@pytest.mark.parametrize(
    "builder, url",
    [
        (build_step5_agent_card, STEP5_URL),
        (build_step6_agent_card, STEP6_URL),
        (build_structure_agent_card, STRUCTURE_URL),
    ],
)
def test_skills_capabilities_one_to_one(builder, url):
    card = builder(url)
    parsed = parse_adc_agent_contract(card)
    skill_ids = {s.id for s in card.skills}
    cap_ids = {c.capability_id for c in parsed.capabilities}
    assert skill_ids == cap_ids
