import json

from app.a2a.agent_cards import (
    build_compact_card_catalog,
    build_step5_agent_card,
    build_step6_agent_card,
    build_structure_agent_card,
)
from app.a2a.orchestrator_context_projection import project_orchestrator_context
from app.a2a.orchestrator_routing_prompt import (
    ORCHESTRATOR_ROUTING_FEW_SHOTS,
    ORCHESTRATOR_ROUTING_SYSTEM_PROMPT,
    ORCHESTRATOR_ROUTING_USER_TASK,
)
from app.llm.json_task_validation import build_json_prompt_sections


def _catalog():
    catalog = build_compact_card_catalog(
        [
            build_structure_agent_card("http://structure-worker:8009"),
            build_step6_agent_card("http://step6-worker:8006"),
            build_step5_agent_card("http://step5-worker:8005"),
        ]
    )
    for index, agent in enumerate(catalog):
        agent["endpoint"] = f"http://private-{index}:9999"
        agent["availability"] = "available"
        agent["agent_failure_reason"] = "private failure"
        agent["discovery_error"] = "private discovery"
        agent["timestamp"] = "2026-private"
    extra_capability = dict(catalog[0]["capabilities"][0])
    extra_capability["capability_id"] = "aaa_test_sort_only"
    catalog[0]["capabilities"].append(extra_capability)
    catalog[0]["capabilities"] = list(reversed(catalog[0]["capabilities"]))
    return catalog


def _projected(
    query: str,
    *,
    available: bool,
    completed_status: str,
    run_id: str,
    artifact_id: str,
):
    return project_orchestrator_context(
        structured_query={
            "canonical_query": query,
            "task_intent": {
                "primary_intent": "developability_assessment",
                "secondary_intents": ["structure_analysis"],
            },
            "requested_outputs": ["developability_summary"],
            "missing_slots": [{"slot_name": "antibody"}],
            "run_id": run_id,
        },
        readiness={
            "input_readiness_status": "blocked" if not available else "ready",
            "missing_input_checklist": [
                {
                    "severity": "blocking",
                    "category": "antibody",
                    "message": "SENSITIVE_DYNAMIC_SENTINEL",
                }
            ],
        },
        available_artifacts=[
            {
                "artifact_name": "candidate_context_table",
                "available": available,
                "present_field_names": ["candidate_records"],
                "artifact_id": artifact_id,
                "path": "/private/artifact.json",
            }
        ],
        current_routing_context={
            "available_agent_ids": ["step_05_candidate_context_agent"],
            "unavailable_agent_ids": ["step_06_developability_agent"],
            "completed_routes": [
                {
                    "agent_id": "step_05_candidate_context_agent",
                    "capability_id": "step_05_candidate_context",
                    "status": completed_status,
                    "output_artifact_names": ["candidate_context_table"],
                }
            ],
        },
    )


def _sections(
    query: str,
    *,
    available: bool,
    completed_status: str,
    run_id: str = "run-private",
    artifact_id: str = "artifact-private",
    system: str | None = ORCHESTRATOR_ROUTING_SYSTEM_PROMPT,
):
    schema = {
        "task": "orchestrator_worker_routing",
        "compact_card_catalog": _catalog(),
        **_projected(
            query,
            available=available,
            completed_status=completed_status,
            run_id=run_id,
            artifact_id=artifact_id,
        ),
    }
    return build_json_prompt_sections(
        prompt=ORCHESTRATOR_ROUTING_USER_TASK,
        schema=schema,
        system=system,
    )


def test_real_prompt_sections_have_exact_cache_layout_and_stability():
    stable_a, dynamic_a = _sections(
        "QUERY_ALPHA",
        available=True,
        completed_status="completed",
        run_id="run-alpha",
        artifact_id="artifact-alpha",
    )
    stable_b, dynamic_b = _sections(
        "QUERY_BETA",
        available=False,
        completed_status="failed",
        run_id="run-beta",
        artifact_id="artifact-beta",
    )
    assert stable_a == stable_b
    assert dynamic_a != dynamic_b

    stable_markers = [
        "System instructions:",
        ORCHESTRATOR_ROUTING_SYSTEM_PROMPT,
        "Return exactly one valid JSON object.",
        "Expected top-level shape:",
        "User/developer task:",
        ORCHESTRATOR_ROUTING_USER_TASK,
        "Few-shot examples JSON:",
        "Compact AgentCard catalog JSON:",
    ]
    positions = [stable_a.index(marker) for marker in stable_markers]
    assert positions == sorted(positions)
    few_shot_text = stable_a.split("Few-shot examples JSON:\n", 1)[1].split(
        "\n\nCompact AgentCard catalog JSON:", 1
    )[0]
    assert json.loads(few_shot_text) == ORCHESTRATOR_ROUTING_FEW_SHOTS
    assert len(json.loads(few_shot_text)) == 2

    dynamic_keys = [
        "compact_user_intent",
        "structured_intent",
        "input_readiness_summary",
        "available_artifact_summary",
        "current_routing_context",
    ]
    assert [dynamic_a.index(f'"{key}"') for key in dynamic_keys] == sorted(
        dynamic_a.index(f'"{key}"') for key in dynamic_keys
    )
    assert "QUERY_ALPHA" in dynamic_a
    assert "candidate_context_table" in dynamic_a
    assert "blocking_gap_categories" in dynamic_a


def test_system_block_strictly_follows_argument_and_user_task_is_forwarded():
    stable_with_system, _ = _sections(
        "QUERY_ALPHA", available=True, completed_status="completed"
    )
    stable_without_system, _ = _sections(
        "QUERY_ALPHA",
        available=True,
        completed_status="completed",
        system=None,
    )
    assert stable_with_system.count(ORCHESTRATOR_ROUTING_SYSTEM_PROMPT) == 1
    assert ORCHESTRATOR_ROUTING_SYSTEM_PROMPT not in stable_without_system
    task_marker = f"User/developer task:\n{ORCHESTRATOR_ROUTING_USER_TASK}\n\n"
    assert task_marker in stable_with_system
    assert task_marker in stable_without_system
    for stable in (stable_with_system, stable_without_system):
        assert stable.index("Few-shot examples JSON:") < stable.index(
            "Compact AgentCard catalog JSON:"
        )


def test_catalog_is_canonical_and_stable_prefix_excludes_runtime_state():
    stable, dynamic = _sections(
        "QUERY_ALPHA", available=True, completed_status="completed"
    )
    catalog_text = stable.split("Compact AgentCard catalog JSON:\n", 1)[1]
    catalog = json.loads(catalog_text)
    assert [agent["agent_id"] for agent in catalog] == sorted(
        agent["agent_id"] for agent in catalog
    )
    for agent in catalog:
        ids = [cap["capability_id"] for cap in agent["capabilities"]]
        assert ids == sorted(ids)
    for forbidden in (
        "http://private-",
        '"availability"',
        "agent_failure_reason",
        "discovery_error",
        "run-private",
        "artifact-private",
        "2026-private",
        "QUERY_ALPHA",
    ):
        assert forbidden not in stable
    assert "QUERY_ALPHA" in dynamic
    assert "run-private" not in dynamic
    assert "artifact-private" not in dynamic


def test_full_prompt_contains_no_raw_sensitive_material():
    sensitive_query = (
        ">seq\n" + "A" * 60
        + "\nHEADER PRIVATE\nATOM  x\nAPI_KEY=secret-token "
        "raw ToolUniverse payload full prompt raw LLM response"
    )
    stable, dynamic = _sections(
        sensitive_query, available=True, completed_status="completed"
    )
    full = stable + dynamic
    for forbidden in (
        ">seq",
        "A" * 60,
        "HEADER PRIVATE",
        "ATOM  x",
        "secret-token",
        "raw ToolUniverse payload",
        "full prompt",
        "raw LLM response",
    ):
        assert forbidden not in full
