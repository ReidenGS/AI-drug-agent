from app.a2a.agent_cards import (
    build_patent_evidence_agent_card,
    parse_adc_agent_contract,
    validate_adc_agent_contract,
)


def test_patent_evidence_agent_card_exact_contract_and_no_tool_details():
    card = build_patent_evidence_agent_card("http://worker:8014")
    contract = parse_adc_agent_contract(card)
    validate_adc_agent_contract(card)
    assert contract.agent_id == "patent_evidence_agent"
    assert contract.step_id == "step_13_14_patent_evidence"
    assert contract.display_name == "Patent and Evidence Agent"
    assert contract.dispatch_modes == ["python_a2a"]
    assert contract.routable is True and contract.status == "active"
    capability = contract.capabilities[0]
    assert capability.capability_id == "patent_evidence_workflow"
    assert capability.execution_mode == "single_step"
    assert capability.supported_step_ids == ["step_13_evidence", "step_14_patent_ip"]
    assert capability.supported_intents == [
        "literature_review",
        "patent_ip_review",
        "new_adc_design",
        "existing_adc_evaluation",
        "optimization",
    ]
    assert capability.supported_lane_flags == [
        "scientific_evidence_lane",
        "patent_prior_art_lane",
        "regulatory_reference_lane",
    ]
    required = {ref.artifact_name: ref for ref in capability.required_input_artifacts}
    assert set(required) == {"structured_query", "candidate_context_table"}
    assert capability.optional_input_artifacts == []
    assert required["structured_query"].storage_path == "inputs/structured_query.json"
    assert required["candidate_context_table"].ready_status_values == ["ok", "partial"]
    assert {ref.artifact_name for ref in capability.output_artifacts} == {
        "scientific_evidence_table",
        "patent_prior_art_table",
    }
    serialized = str(card.to_dict())
    assert contract.description
    assert capability.skill_name
    assert capability.capability_summary
    for forbidden in (
        "EuropePMC_search_articles",
        "PubChem_get_associated_patents_by_CID",
        "official schema",
        "runtime_availability",
        "endpoint",
    ):
        assert forbidden not in serialized
