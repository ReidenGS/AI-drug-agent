"""Compact graph-resume-state adapter (LangGraph checkpointer seam).

The adapter builds a compact, privacy-safe resume state from the artifact
store. No checkpointer is wired this round; the artifact store is the source
of truth. These tests pin: it stores only ids + canonical_query + compact
summaries, never raw payloads/sequences/keys.
"""

from __future__ import annotations

from app.graph.compact_resume_state import build_compact_resume_state
from app.schemas.step_02_structured_query import (
    MissingSlot,
    SourceRawRequestRef,
    StructuredQuery,
    TaskIntent,
)
from app.services.intake_service import IntakeService
from app.services.input_readiness_service import InputReadinessService
from app.utils.ids import new_artifact_id
from app.utils.time import now_iso


def _seed(local_storage, registry_service, workflow_state_service) -> str:
    rec = IntakeService(local_storage, registry_service, workflow_state_service).submit(
        raw_user_query="design an ADC", user_provided_context={}
    )
    run_id = rec.run_id
    sq = StructuredQuery(
        run_id=run_id,
        parsed_at=now_iso(),
        source_raw_request_ref=SourceRawRequestRef(
            raw_request_record_id=registry_service.get(run_id).active_artifacts.raw_request_record_id
        ),
        task_intent=TaskIntent(task_type="adc_design", modality="ADC"),
        canonical_query="Design a new antibody-drug conjugate (target unspecified).",
        missing_slots=[
            MissingSlot(slot_name="target_or_antigen", slot_category="target", severity="blocking")
        ],
    )
    sq_id = new_artifact_id("structured_query")
    local_storage.write_json(
        local_storage.run_key(run_id, "inputs/structured_query.json"),
        {"artifact_id": sq_id, **sq.model_dump()},
    )
    registry_service.update_active(run_id, structured_query_id=sq_id)
    workflow_state_service.mark(run_id, "step_02", "completed")
    InputReadinessService(local_storage, registry_service, workflow_state_service).check(run_id)
    return run_id


def test_compact_state_has_only_compact_fields(
    local_storage, registry_service, workflow_state_service
):
    run_id = _seed(local_storage, registry_service, workflow_state_service)
    state = build_compact_resume_state(
        local_storage, registry_service, workflow_state_service, run_id
    )
    assert set(state) == {
        "run_id",
        "current_step",
        "active_artifacts",
        "canonical_query",
        "missing_slots_summary",
        "clarification_request_ids",
    }
    assert state["run_id"] == run_id
    assert state["canonical_query"].startswith("Design a new antibody-drug conjugate")
    assert state["missing_slots_summary"] == [
        {"slot_name": "target_or_antigen", "severity": "blocking"}
    ]
    assert state["clarification_request_ids"]  # target request id present
    # active_artifacts are ids only.
    assert state["active_artifacts"]["structured_query_id"]


def test_compact_state_no_raw_payload_or_secrets(
    local_storage, registry_service, workflow_state_service
):
    run_id = _seed(local_storage, registry_service, workflow_state_service)
    state = build_compact_resume_state(
        local_storage, registry_service, workflow_state_service, run_id
    )
    blob = str(state).lower()
    # Only a slot-name/severity summary — not full slot objects with reasons.
    assert "suggested_question" not in blob
    assert "reason" not in blob
    assert "api_key" not in blob
    assert "system instructions" not in blob
