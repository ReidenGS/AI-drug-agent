"""Turn A — AgentCard builders, adc_agent_contract validator, compact catalog.

These tests exercise the real production artifact contract. The storage paths
asserted here are the exact paths the live Step 5 / Step 6 / structure agents
read and write (verified against app/agents/*.py). No test-only caps, mocks,
allowlists, or narrowed constraints are introduced.
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
    CAP_STEP7_STRUCTURE_INPUT,
    CAP_STEP8_STRUCTURE_EVALUATION,
    CAP_STEP9_STRUCTURE_DESIGN,
    AgentContractError,
    build_compact_card_catalog,
    build_step5_agent_card,
    build_step6_agent_card,
    build_structure_agent_card,
    parse_adc_agent_contract,
    validate_adc_agent_contract,
)

STEP5_URL = "http://step5-worker:8005"
STEP6_URL = "http://step6-worker:8006"
STRUCTURE_URL = "http://structure-worker:8009"


def _contract(card: AgentCard) -> dict:
    return card.capabilities["adc_agent_contract"]


def _required_paths(contract: dict, capability_id: str) -> dict[str, str]:
    cap = next(c for c in contract["capabilities"] if c["capability_id"] == capability_id)
    return {r["artifact_name"]: r["storage_path"] for r in cap["required_input_artifacts"]}


def _optional_paths(contract: dict, capability_id: str) -> dict[str, str]:
    cap = next(c for c in contract["capabilities"] if c["capability_id"] == capability_id)
    return {r["artifact_name"]: r["storage_path"] for r in cap["optional_input_artifacts"]}


def _output_paths(contract: dict, capability_id: str) -> dict[str, str]:
    cap = next(c for c in contract["capabilities"] if c["capability_id"] == capability_id)
    return {r["artifact_name"]: r["storage_path"] for r in cap["output_artifacts"]}


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
    # parse_adc_agent_contract validates and returns without raising.
    parsed = validate_adc_agent_contract(card)
    assert parsed.agent_id == agent_id


# ── 3. Step 5 input/output contract == production read/write paths ───────────
def test_step5_artifact_contract_matches_production():
    contract = _contract(build_step5_agent_card(STEP5_URL))
    required = _required_paths(contract, CAP_STEP5_CANDIDATE_CONTEXT)
    assert required == {
        "raw_request_record": "inputs/raw_request_record.json",
        "structured_query": "inputs/structured_query.json",
        "run_step_plan": "inputs/run_step_plan.json",
    }
    assert _output_paths(contract, CAP_STEP5_CANDIDATE_CONTEXT) == {
        "candidate_context_table": "candidate_context_table.json",
    }


# ── 4. Step 6 input/output contract == production read/write paths ───────────
def test_step6_artifact_contract_matches_production():
    contract = _contract(build_step6_agent_card(STEP6_URL))
    required = _required_paths(contract, CAP_STEP6_DEVELOPABILITY)
    assert required == {
        "candidate_context_table": "candidate_context_table.json",
        "run_step_plan": "inputs/run_step_plan.json",
    }
    assert _output_paths(contract, CAP_STEP6_DEVELOPABILITY) == {
        "structured_liability_summary": "structured_liability_summary.json",
    }


# ── 5. structure worker Step 7/8/9 contracts == production read/write paths ──
def test_structure_step7_8_9_artifact_contracts_match_production():
    contract = _contract(build_structure_agent_card(STRUCTURE_URL))

    # Step 7
    assert _required_paths(contract, CAP_STEP7_STRUCTURE_INPUT) == {
        "raw_request_record": "inputs/raw_request_record.json",
        "structured_query": "inputs/structured_query.json",
        "candidate_context_table": "candidate_context_table.json",
        "run_step_plan": "inputs/run_step_plan.json",
    }
    assert _output_paths(contract, CAP_STEP7_STRUCTURE_INPUT) == {
        "prepared_structure_input_package": "prepared_structure_input_package.json",
    }

    # Step 8
    assert _required_paths(contract, CAP_STEP8_STRUCTURE_EVALUATION) == {
        "prepared_structure_input_package": "prepared_structure_input_package.json",
        "run_step_plan": "inputs/run_step_plan.json",
    }
    assert _output_paths(contract, CAP_STEP8_STRUCTURE_EVALUATION) == {
        "structure_prediction_and_interface_results": "structure_prediction_and_interface_results.json",
    }

    # Step 9 required
    assert _required_paths(contract, CAP_STEP9_STRUCTURE_DESIGN) == {
        "candidate_context_table": "candidate_context_table.json",
        "run_step_plan": "inputs/run_step_plan.json",
    }
    # Step 9 optional
    assert _optional_paths(contract, CAP_STEP9_STRUCTURE_DESIGN) == {
        "prepared_structure_input_package": "prepared_structure_input_package.json",
        "structure_prediction_and_interface_results": "structure_prediction_and_interface_results.json",
        "raw_request_record": "inputs/raw_request_record.json",
        "structured_query": "inputs/structured_query.json",
    }
    # Step 9 output — note the artifact NAME differs from its storage key.
    assert _output_paths(contract, CAP_STEP9_STRUCTURE_DESIGN) == {
        "structure_variant_and_compound_screening": "compound_screening_artifact.json",
    }


# ── 6. AgentCard url can be a Docker internal service name ───────────────────
def test_card_url_accepts_docker_internal_service_name():
    card = build_step6_agent_card("http://step6-worker:8006")
    assert card.url == "http://step6-worker:8006"
    # A routable worker with a docker-internal url passes validation.
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
    # Drop the published skill so it no longer aligns with the contract capability.
    card.skills = []
    with pytest.raises(AgentContractError):
        parse_adc_agent_contract(card)


# ── 10 & 11. compact catalog excludes secrets/raw but keeps routing info ─────
def _all_cards():
    return [
        build_step5_agent_card(STEP5_URL),
        build_step6_agent_card(STEP6_URL),
        build_structure_agent_card(STRUCTURE_URL),
    ]


def test_compact_catalog_excludes_urls_and_raw_material():
    catalog = build_compact_card_catalog(_all_cards())
    blob = json.dumps(catalog).lower()

    # No endpoint URL / host / port / scheme.
    for needle in ("http://", "step5-worker", "step6-worker", "structure-worker", "8005", "8009"):
        assert needle not in blob, f"compact catalog leaked endpoint token: {needle}"

    # No auth / secrets / raw payloads / raw biological data.
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


def test_compact_catalog_retains_routing_info():
    catalog = build_compact_card_catalog(_all_cards())
    by_agent = {entry["agent_id"]: entry for entry in catalog}
    assert set(by_agent) == {AGENT_ID_STEP5, AGENT_ID_STEP6, AGENT_ID_STRUCTURE}

    step6 = by_agent[AGENT_ID_STEP6]
    cap = next(c for c in step6["capabilities"] if c["capability_id"] == CAP_STEP6_DEVELOPABILITY)
    assert cap["required_input_artifact_names"] == ["candidate_context_table", "run_step_plan"]
    assert cap["output_artifact_names"] == ["structured_liability_summary"]
    assert step6["routable"] is True

    structure = by_agent[AGENT_ID_STRUCTURE]
    cap_ids = {c["capability_id"] for c in structure["capabilities"]}
    assert cap_ids == {
        CAP_STEP7_STRUCTURE_INPUT,
        CAP_STEP8_STRUCTURE_EVALUATION,
        CAP_STEP9_STRUCTURE_DESIGN,
    }
