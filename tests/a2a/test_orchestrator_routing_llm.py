import inspect

from app.a2a.agent_cards import (
    build_compact_card_catalog,
    build_patent_evidence_agent_card,
    build_step5_agent_card,
    build_step6_agent_card,
    build_structure_agent_card,
)
from app.llm.provider import MockLLMProvider, _mock_orchestrator_worker_routing


def _schema(intent: str, *, remove_capabilities: set[str] | None = None):
    catalog = build_compact_card_catalog(
        [
            build_step5_agent_card("http://step5-worker:8005"),
            build_step6_agent_card("http://step6-worker:8006"),
            build_structure_agent_card("http://structure-worker:8009"),
            build_patent_evidence_agent_card("http://patent-worker:8014"),
        ]
    )
    remove_capabilities = remove_capabilities or set()
    for agent in catalog:
        agent["capabilities"] = [
            capability
            for capability in agent["capabilities"]
            if capability["capability_id"] not in remove_capabilities
        ]
    return {
        "task": "orchestrator_worker_routing",
        "compact_user_intent": intent,
        "structured_intent": {},
        "compact_card_catalog": catalog,
    }


def test_mock_developability_uses_only_catalog_routes():
    out = MockLLMProvider().generate_json(
        "route", schema=_schema("assess developability")
    )
    assert [x["capability_id"] for x in out["decisions"]] == [
        "step_05_candidate_context",
        "step_06_developability",
    ]


def test_mock_structure_and_unrelated():
    assert [
        item["capability_id"]
        for item in MockLLMProvider().generate_json(
            "route", schema=_schema("structure and protein design")
        )["decisions"]
    ] == ["step_05_candidate_context", "structure_design_workflow"]
    out = MockLLMProvider().generate_json("route", schema=_schema("already satisfied"))
    assert out["loop_decision"] == "route_to_final_response" and out["decisions"] == []


def test_mock_never_invents_removed_step6_capability():
    out = MockLLMProvider().generate_json(
        "route",
        schema=_schema(
            "assess developability",
            remove_capabilities={"step_06_developability"},
        ),
    )
    assert [item["capability_id"] for item in out["decisions"]] == [
        "step_05_candidate_context"
    ]


def test_mock_routes_patent_evidence_semantics_and_requested_outputs():
    for intent in (
        "review scientific literature evidence",
        "search patent prior-art and IP",
        "find regulatory reference",
    ):
        out = MockLLMProvider().generate_json("route", schema=_schema(intent))
        assert [item["capability_id"] for item in out["decisions"]] == [
            "step_05_candidate_context",
            "patent_evidence_workflow",
        ]

    schema = _schema("evaluate this program")
    schema["structured_intent"] = {
        "requested_outputs": ["patent_prior_art_table"]
    }
    out = MockLLMProvider().generate_json("route", schema=schema)
    assert [item["capability_id"] for item in out["decisions"]] == [
        "step_05_candidate_context",
        "patent_evidence_workflow"
    ]


def test_mock_never_invents_removed_patent_evidence_capability():
    out = MockLLMProvider().generate_json(
        "route",
        schema=_schema(
            "review literature evidence and patent IP",
            remove_capabilities={"patent_evidence_workflow"},
        ),
    )
    assert out["loop_decision"] == "route_to_final_response"
    assert out["decisions"] == []


def test_mock_with_no_matching_catalog_capability_emits_no_route_or_task():
    out = MockLLMProvider().generate_json(
        "route",
        schema=_schema(
            "assess developability",
            remove_capabilities={
                "step_05_candidate_context",
                "step_06_developability",
            },
        ),
    )
    assert out["loop_decision"] == "route_to_final_response"
    assert out["decisions"] == []
    assert "task_id" not in str(out).lower()


def test_mock_boundary_is_explicitly_test_offline_only():
    """Mock routing does not construct Tasks or call workers.

    It is deterministic test/offline behavior, not a live LLM result and not
    a production-provider failure fallback.
    """
    out = MockLLMProvider().generate_json(
        "route", schema=_schema("assess developability")
    )
    assert set(out) == {"loop_decision", "decisions", "decision_summary"}
    source = inspect.getsource(_mock_orchestrator_worker_routing)
    assert "python_a2a" not in source
    assert "send_task" not in source
    assert "execute_request" not in source
    assert "Task(" not in source
