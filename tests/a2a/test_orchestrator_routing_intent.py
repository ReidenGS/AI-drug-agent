"""Production Step 1-3 projection into real-card Orchestrator routing."""

from __future__ import annotations

import pytest

from app.a2a.agent_cards import (
    AGENT_ID_STEP5,
    AGENT_ID_STEP6,
    AGENT_ID_STRUCTURE,
    CAP_STEP5_CANDIDATE_CONTEXT,
    CAP_STEP6_DEVELOPABILITY,
    CAP_STRUCTURE_DESIGN_WORKFLOW,
    build_step5_agent_card,
    build_step6_agent_card,
    build_structure_agent_card,
)
from app.a2a.orchestrator_discovery import (
    ExpectedWorkerEndpoint,
    WorkerDiscoveryService,
)
from app.a2a.orchestrator_routing_service import OrchestratorRoutingService
from app.agents.supervisor_agent import SupervisorAgent
from app.llm.provider import MockLLMProvider
from app.services.input_readiness_service import InputReadinessService
from app.services.intake_service import IntakeService
from app.services.structured_query_service import StructuredQueryService
from tests.a2a.test_orchestrator_discovery import _agentserver_stub

QUERY = (
    "Assess HER2 ADC developability and prepare structure-guided protein "
    "design using a trastuzumab-like antibody and vc-MMAE payload."
)


@pytest.fixture(autouse=True)
def _localhost_proxy_isolation(monkeypatch):
    for name in ("NO_PROXY", "no_proxy"):
        monkeypatch.setenv(name, "127.0.0.1,localhost")
    for name in (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "http_proxy",
        "https_proxy",
        "ALL_PROXY",
        "all_proxy",
    ):
        monkeypatch.delenv(name, raising=False)


def test_real_step2_step3_projection_routes_all_requested_real_cards(
    local_storage,
    registry_service,
    workflow_state_service,
):
    """Uses real services and HTTP cards; MockLLM is offline-only routing."""
    record = IntakeService(
        local_storage, registry_service, workflow_state_service
    ).submit(
        raw_user_query=QUERY,
        user_provided_context={
            "target_or_antigen_text": "HER2",
            "candidate_text": "trastuzumab-like antibody",
            "payload_linker_text": "vc-MMAE",
        },
    )
    structured = StructuredQueryService(
        local_storage,
        registry_service,
        workflow_state_service,
        SupervisorAgent(llm=MockLLMProvider()),
    ).parse(record.run_id)
    readiness = InputReadinessService(
        local_storage, registry_service, workflow_state_service
    ).check(record.run_id)
    assert readiness.input_readiness_status == "ready"
    assert "developability" in structured.task_intent.user_goal_summary.lower()
    assert "structure-guided protein design" in (
        structured.task_intent.user_goal_summary.lower()
    )
    assert "developability" not in structured.canonical_query.lower()
    assert "structure" not in structured.canonical_query.lower()

    step5 = _agentserver_stub(base_builder=build_step5_agent_card)
    step6 = _agentserver_stub(base_builder=build_step6_agent_card)
    structure = _agentserver_stub(base_builder=build_structure_agent_card)
    try:
        discovery = WorkerDiscoveryService(
            expected_workers=[
                ExpectedWorkerEndpoint(
                    AGENT_ID_STEP5,
                    (CAP_STEP5_CANDIDATE_CONTEXT,),
                    step5.url,
                ),
                ExpectedWorkerEndpoint(
                    AGENT_ID_STEP6,
                    (CAP_STEP6_DEVELOPABILITY,),
                    step6.url,
                ),
                ExpectedWorkerEndpoint(
                    AGENT_ID_STRUCTURE,
                    (CAP_STRUCTURE_DESIGN_WORKFLOW,),
                    structure.url,
                ),
            ],
            storage=local_storage,
            registry=registry_service,
            discovery_timeout_seconds=2,
            health_timeout_seconds=2,
        )
        result = OrchestratorRoutingService(
            discovery=discovery,
            storage=local_storage,
            registry=registry_service,
            llm=MockLLMProvider(),
        ).plan_for_run(record.run_id)
    finally:
        step5.close()
        step6.close()
        structure.close()

    decisions = {
        item.capability_id: item for item in result.plan.validated_decisions
    }
    assert set(decisions) == {
        CAP_STEP5_CANDIDATE_CONTEXT,
        CAP_STEP6_DEVELOPABILITY,
        CAP_STRUCTURE_DESIGN_WORKFLOW,
    }, (
        result.plan.routing_status,
        result.plan.warnings,
        result.plan.rejected_decisions,
        result.plan.proposed_decisions,
    )
    assert decisions[CAP_STEP5_CANDIDATE_CONTEXT].validation_status == "ready"
    assert decisions[CAP_STEP6_DEVELOPABILITY].validation_status == (
        "waiting_for_dependencies"
    )
    assert decisions[CAP_STRUCTURE_DESIGN_WORKFLOW].validation_status == (
        "waiting_for_dependencies"
    )
    assert [item.decision.agent_id for item in result.prepared_tasks] == [
        AGENT_ID_STEP5
    ]
    edges = {
        (
            edge.producer_capability_id,
            edge.consumer_capability_id,
            edge.artifact_name,
        )
        for edge in result.plan.dependency_edges
    }
    assert edges == {
        (
            CAP_STEP5_CANDIDATE_CONTEXT,
            CAP_STEP6_DEVELOPABILITY,
            "candidate_context_table",
        ),
        (
            CAP_STEP5_CANDIDATE_CONTEXT,
            CAP_STRUCTURE_DESIGN_WORKFLOW,
            "candidate_context_table",
        ),
    }
