from app.a2a.orchestrator_context_projection import project_orchestrator_context
from app.schemas.step_02_structured_query import (
    MissingSlot,
    SourceRawRequestRef,
    StructuredQuery,
    TaskIntent,
)
from app.schemas.step_03_input_readiness import (
    ClarificationRequest,
    InputReadinessStatus,
    MissingInputItem,
    SourceRefs,
)


def test_real_step2_step3_models_project_only_safe_categories():
    sensitive = "SENSITIVE_FREE_TEXT_SENTINEL"
    query = StructuredQuery(
        run_id="run-private",
        parsed_at="2026-07-13T00:00:00Z",
        source_raw_request_ref=SourceRawRequestRef(raw_request_record_id="raw-private"),
        task_intent=TaskIntent(
            task_type="adc_design",
            user_goal_summary="Assess HER2 developability",
            primary_intent="developability_assessment",
        ),
        requested_outputs=["developability_summary"],
        missing_slots=[
            MissingSlot(
                slot_name="antibody",
                slot_category="antibody",
                severity="blocking",
                reason=sensitive,
            )
        ],
        canonical_query="Assess HER2 developability",
        response=f"response {sensitive}",
    )
    readiness = InputReadinessStatus(
        run_id="run-private",
        checked_at="2026-07-13T00:00:01Z",
        source_refs=SourceRefs(
            raw_request_record_id="raw-private",
            structured_query_id="sq-private",
        ),
        input_readiness_status="blocked",
        missing_input_checklist=[
            MissingInputItem(
                field="antibody",
                severity="blocking",
                category="antibody",
                message=f"blocking {sensitive}",
            ),
            MissingInputItem(
                field="constraints",
                severity="warning",
                category="constraints",
                message=f"warning {sensitive}",
            ),
        ],
        blocking_reasons=[f"reason {sensitive}"],
        clarification_requests=[
            ClarificationRequest(
                request_id="clarification-1",
                slot_name="antibody",
                slot_category="antibody",
                severity="blocking",
                question=f"question {sensitive}",
                reason=f"reason {sensitive}",
            )
        ],
        response=f"readiness response {sensitive}",
    )

    out = project_orchestrator_context(
        structured_query=query.model_dump(),
        readiness=readiness.model_dump(),
        available_artifacts=[],
    )

    assert sensitive not in str(out)
    summary = out["input_readiness_summary"]
    assert summary["missing_slot_names"] == ["antibody"]
    assert summary["blocking_gap_categories"] == ["antibody"]
    assert summary["warning_gap_categories"] == ["constraints"]


def test_projection_redacts_raw_material_and_nested_context():
    raw = "A" * 60
    out = project_orchestrator_context(
        structured_query={
            "canonical_query": (
                f"Assess HER2 developability using {raw} "
                "sk-secret12345 /data/private/a.pdb"
            ),
            "task_intent": {"primary_intent": "developability_assessment"},
            "missing_slots": [],
        },
        readiness={"input_readiness_status": "ready"},
        available_artifacts=[],
        current_routing_context={
            "completed_routes": [],
            "tool_call_records": [raw],
            "nested": {"raw": raw},
        },
    )
    blob = str(out)
    assert raw not in blob
    assert "sk-secret" not in blob
    assert "/data/private" not in blob
    assert "tool_call_records" not in blob
    assert "nested" not in blob


def test_projection_redacts_structure_alignment_and_private_markers():
    sentinels = [
        ">seq\nAAAAAA--AAAA",
        "HEADER SECRET\nATOM  x",
        "data_demo\nloop_\n_atom_site.id",
        "NVIDIA_API_KEY=token123",
        "raw ToolUniverse payload",
        "full prompt",
        "raw LLM response",
        "C:\\private\\x.cif",
    ]
    for sentinel in sentinels:
        out = project_orchestrator_context(
            structured_query={
                "canonical_query": "structure design " + sentinel,
                "task_intent": {},
                "missing_slots": [],
            },
            readiness={"input_readiness_status": "ready"},
            available_artifacts=[],
        )
        assert sentinel not in str(out)


def test_projection_preserves_distinct_user_goal_and_canonical_query():
    out = project_orchestrator_context(
        structured_query={
            "canonical_query": "Evaluate an existing HER2 ADC.",
            "task_intent": {
                "user_goal_summary": (
                    "Assess developability and prepare structure-guided "
                    "protein design."
                ),
                "primary_intent": "existing_adc_evaluation",
            },
            "missing_slots": [],
        },
        readiness={"input_readiness_status": "ready"},
        available_artifacts=[],
    )
    intent = out["compact_user_intent"]
    assert intent == (
        "User goal: Assess developability and prepare structure-guided protein "
        "design. Canonical query: Evaluate an existing HER2 ADC."
    )
