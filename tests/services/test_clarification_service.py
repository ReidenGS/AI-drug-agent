"""Step 3 clarification multi-turn loop — minimal backend closed loop.

Covers: saving answers, request-id validation, deterministic duplicate
handling, next-turn context construction (original query + previous intent +
answers), the Step 2/3 re-parse loop, resolved/unresolved bookkeeping, and
privacy. No LangGraph memory/checkpointer is used; Step 3 calls no LLM.
"""

from __future__ import annotations

import pytest

from app.agents.supervisor_agent import SupervisorAgent
from app.llm.provider import MockLLMProvider
from app.services.clarification_service import ClarificationService
from app.services.input_readiness_service import InputReadinessService
from app.services.intake_service import IntakeService
from app.services.structured_query_service import StructuredQueryService


def _supervisor() -> SupervisorAgent:
    return SupervisorAgent(llm=MockLLMProvider())


def _turn_one(local_storage, registry_service, workflow_state_service, query: str):
    """Run Step 1→3 turn one and return (run_id, readiness)."""
    rec = IntakeService(local_storage, registry_service, workflow_state_service).submit(
        raw_user_query=query, user_provided_context={}
    )
    StructuredQueryService(
        local_storage, registry_service, workflow_state_service, _supervisor()
    ).parse(rec.run_id)
    readiness = InputReadinessService(
        local_storage, registry_service, workflow_state_service
    ).check(rec.run_id)
    return rec.run_id, readiness


def _target_request_id(readiness) -> str:
    reqs = [c for c in readiness.clarification_requests if c.slot_name == "target_or_antigen"]
    assert reqs, "turn one must produce a target_or_antigen clarification request"
    return reqs[0].request_id


# ── A. basic save ─────────────────────────────────────────────────────────────


def test_submit_clarification_answer_persists_state(
    local_storage, registry_service, workflow_state_service
):
    run_id, readiness = _turn_one(
        local_storage, registry_service, workflow_state_service,
        "I want to design a new antibody-drug conjugate.",
    )
    assert readiness.input_readiness_status == "blocked"
    rid = _target_request_id(readiness)

    svc = ClarificationService(local_storage, registry_service, workflow_state_service)
    state = svc.submit_clarification_answer(
        run_id, [{"request_id": rid, "answer_text": "HER2", "answered_at": "t"}]
    )
    assert state.clarification_answers[0].answer_text == "HER2"
    assert state.clarification_answers[0].target_slot_name == "target_or_antigen"
    assert rid in state.resolved_request_ids
    assert rid not in state.unresolved_request_ids
    # The other (warning) requests stay unresolved.
    assert state.unresolved_request_ids
    # State is persisted and pointed at by the registry (non-destructive).
    reg = registry_service.get(run_id)
    assert reg.active_artifacts.clarification_state_id is not None
    # Original Step 2 / Step 3 artifacts are untouched.
    assert reg.active_artifacts.structured_query_id == state.source_structured_query_id
    assert reg.active_artifacts.input_readiness_status_id == state.source_input_readiness_status_id


# ── B. request validation ─────────────────────────────────────────────────────


def test_unknown_request_id_raises(
    local_storage, registry_service, workflow_state_service
):
    run_id, _ = _turn_one(
        local_storage, registry_service, workflow_state_service,
        "I want to design a new antibody-drug conjugate.",
    )
    svc = ClarificationService(local_storage, registry_service, workflow_state_service)
    with pytest.raises(ValueError, match="unknown clarification request_id"):
        svc.submit_clarification_answer(
            run_id, [{"request_id": "clr_does_not_exist", "answer_text": "HER2", "answered_at": "t"}]
        )


def test_duplicate_request_id_rejected(
    local_storage, registry_service, workflow_state_service
):
    """Deterministic policy: duplicate answers for the same request in one
    submission are rejected (not silently overridden)."""
    run_id, readiness = _turn_one(
        local_storage, registry_service, workflow_state_service,
        "I want to design a new antibody-drug conjugate.",
    )
    rid = _target_request_id(readiness)
    svc = ClarificationService(local_storage, registry_service, workflow_state_service)
    with pytest.raises(ValueError, match="duplicate clarification answer"):
        svc.submit_clarification_answer(
            run_id,
            [
                {"request_id": rid, "answer_text": "HER2", "answered_at": "t"},
                {"request_id": rid, "answer_text": "EGFR", "answered_at": "t"},
            ],
        )


def test_run_without_clarification_requests_raises_clearly(
    local_storage, registry_service, workflow_state_service
):
    """Old/ready runs with no clarification_requests give a clear error, not a
    crash."""
    run_id, readiness = _turn_one(
        local_storage, registry_service, workflow_state_service,
        "Design a HER2 ADC with vc-MMAE and trastuzumab",
    )
    # Full-ish context → no clarification requests (ready or warning-only).
    svc = ClarificationService(local_storage, registry_service, workflow_state_service)
    if not readiness.clarification_requests:
        with pytest.raises(ValueError, match="no Step 3 clarification_requests"):
            svc.submit_clarification_answer(
                run_id, [{"request_id": "clr_x", "answer_text": "y", "answered_at": "t"}]
            )


def test_empty_answers_raises(
    local_storage, registry_service, workflow_state_service
):
    run_id, _ = _turn_one(
        local_storage, registry_service, workflow_state_service,
        "I want to design a new antibody-drug conjugate.",
    )
    svc = ClarificationService(local_storage, registry_service, workflow_state_service)
    with pytest.raises(ValueError, match="no clarification answers"):
        svc.submit_clarification_answer(run_id, [])


# ── C. next-turn context ───────────────────────────────────────────────────────


def test_next_run_preserves_original_query_and_carries_context(
    local_storage, registry_service, workflow_state_service
):
    run_id, readiness = _turn_one(
        local_storage, registry_service, workflow_state_service,
        "I want to design a new antibody-drug conjugate.",
    )
    rid = _target_request_id(readiness)
    svc = ClarificationService(local_storage, registry_service, workflow_state_service)
    state = svc.submit_clarification_answer(
        run_id, [{"request_id": rid, "answer_text": "HER2", "answered_at": "t"}]
    )
    next_raw = local_storage.read_json(
        local_storage.run_key(state.next_run_id, "inputs/raw_request_record.json")
    )
    # Original query preserved (not replaced by "HER2").
    assert next_raw["raw_user_query"].startswith(
        "I want to design a new antibody-drug conjugate."
    )
    assert next_raw["raw_user_query"] != "HER2"
    ctx = next_raw["user_provided_context"]
    assert ctx["previous_task_intent"]["primary_intent"] == "new_adc_design"
    assert ctx["previous_missing_slots"]
    assert ctx["previous_clarification_requests"]
    answers = ctx["clarification_answers"]
    assert answers[0]["answer_text"] == "HER2"
    assert answers[0]["slot_name"] == "target_or_antigen"


def test_next_run_does_not_clobber_unrelated_context_fields(
    local_storage, registry_service, workflow_state_service
):
    # Turn one WITH an unrelated context field set.
    rec = IntakeService(local_storage, registry_service, workflow_state_service).submit(
        raw_user_query="I want to design a new antibody-drug conjugate.",
        user_provided_context={"constraints_text": "DAR<=4"},
    )
    StructuredQueryService(
        local_storage, registry_service, workflow_state_service, _supervisor()
    ).parse(rec.run_id)
    readiness = InputReadinessService(
        local_storage, registry_service, workflow_state_service
    ).check(rec.run_id)
    rid = _target_request_id(readiness)
    svc = ClarificationService(local_storage, registry_service, workflow_state_service)
    state = svc.submit_clarification_answer(
        rec.run_id, [{"request_id": rid, "answer_text": "HER2", "answered_at": "t"}]
    )
    next_raw = local_storage.read_json(
        local_storage.run_key(state.next_run_id, "inputs/raw_request_record.json")
    )
    assert next_raw["user_provided_context"]["constraints_text"] == "DAR<=4"


# ── D. Step 2/3 loop behavior ──────────────────────────────────────────────────


def test_clarification_loop_resolves_target_blocking(
    local_storage, registry_service, workflow_state_service
):
    run_id, readiness = _turn_one(
        local_storage, registry_service, workflow_state_service,
        "I want to design a new antibody-drug conjugate.",
    )
    assert readiness.input_readiness_status == "blocked"
    rid = _target_request_id(readiness)
    svc = ClarificationService(local_storage, registry_service, workflow_state_service)
    result = svc.submit_and_reparse(
        run_id, [{"request_id": rid, "answer_text": "HER2", "answered_at": "t"}], _supervisor()
    )
    sq2 = result.structured_query
    r2 = result.input_readiness_status
    # Turn 2: HER2 recognized as target; intent preserved.
    assert sq2.mentioned_entities.target_or_antigen_text == "HER2"
    assert sq2.task_intent.primary_intent == "new_adc_design"
    # target_or_antigen is no longer a (blocking) missing slot.
    assert not any(
        m.slot_name == "target_or_antigen" and m.severity == "blocking"
        for m in sq2.missing_slots
    )
    # Not blocked on target anymore (warning-only → needs_user_input).
    assert r2.input_readiness_status in {"needs_user_input", "ready", "partial"}
    assert r2.input_readiness_status != "blocked" or not any(
        "target" in r.lower() for r in r2.blocking_reasons
    )


def test_step3_reparse_does_not_overwrite_original_run(
    local_storage, registry_service, workflow_state_service
):
    run_id, readiness = _turn_one(
        local_storage, registry_service, workflow_state_service,
        "I want to design a new antibody-drug conjugate.",
    )
    orig_sq = local_storage.read_json(
        local_storage.run_key(run_id, "inputs/structured_query.json")
    )
    rid = _target_request_id(readiness)
    svc = ClarificationService(local_storage, registry_service, workflow_state_service)
    result = svc.submit_and_reparse(
        run_id, [{"request_id": rid, "answer_text": "HER2", "answered_at": "t"}], _supervisor()
    )
    # The revision lives under a NEW run id.
    assert result.next_run_id != run_id
    # Original run's structured_query is unchanged (still target-less).
    after = local_storage.read_json(
        local_storage.run_key(run_id, "inputs/structured_query.json")
    )
    assert after == orig_sq


# ── E. prompt ──────────────────────────────────────────────────────────────────


def test_prompt_includes_clarification_rules():
    from app.agents.supervisor_agent import SUPERVISOR_SYSTEM_PROMPT as sp

    assert "clarification_answers" in sp
    assert "previous_task_intent" in sp
    # Tells the model not to treat a short answer as a new standalone task.
    assert "standalone task" in sp


# ── F. privacy ─────────────────────────────────────────────────────────────────


def test_clarification_state_does_not_leak_keys_or_prompt(
    local_storage, registry_service, workflow_state_service
):
    run_id, readiness = _turn_one(
        local_storage, registry_service, workflow_state_service,
        "I want to design a new antibody-drug conjugate.",
    )
    rid = _target_request_id(readiness)
    svc = ClarificationService(local_storage, registry_service, workflow_state_service)
    state = svc.submit_clarification_answer(
        run_id, [{"request_id": rid, "answer_text": "HER2", "answered_at": "t"}]
    )
    blob = str(state.model_dump()).lower()
    assert "api_key" not in blob
    assert "system instructions" not in blob
    assert "bearer " not in blob


def test_sequence_answer_is_user_input_not_copied_to_normalized_artifacts(
    local_storage, registry_service, workflow_state_service
):
    """A sequence-shaped answer is the user's own input carried in the
    clarification context; it must NOT be copied into the turn-2
    structured_query's normalized fields (entities/normalized_entities)."""
    run_id, readiness = _turn_one(
        local_storage, registry_service, workflow_state_service,
        "I want to design a new antibody-drug conjugate.",
    )
    seq = "EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGK"
    # Answer the antibody warning request with a raw sequence.
    ab_reqs = [c for c in readiness.clarification_requests if c.slot_name == "antibody"]
    assert ab_reqs
    svc = ClarificationService(local_storage, registry_service, workflow_state_service)
    result = svc.submit_and_reparse(
        run_id,
        [{"request_id": ab_reqs[0].request_id, "answer_text": seq, "answered_at": "t"}],
        _supervisor(),
    )
    sq2 = result.structured_query
    # The sequence is not promoted into normalized entity canonical names.
    norm_blob = str([ne.model_dump() for ne in sq2.normalized_entities])
    assert seq not in norm_blob


# ── canonical_query carry-over ───────────────────────────────────────────────


def test_next_context_includes_previous_canonical_query(
    local_storage, registry_service, workflow_state_service
):
    run_id, readiness = _turn_one(
        local_storage, registry_service, workflow_state_service,
        "I want to design a new antibody-drug conjugate.",
    )
    rid = _target_request_id(readiness)
    svc = ClarificationService(local_storage, registry_service, workflow_state_service)
    state = svc.submit_clarification_answer(
        run_id, [{"request_id": rid, "answer_text": "HER2", "answered_at": "t"}]
    )
    next_raw = local_storage.read_json(
        local_storage.run_key(state.next_run_id, "inputs/raw_request_record.json")
    )
    ctx = next_raw["user_provided_context"]
    assert ctx.get("previous_canonical_query")
    assert "antibody-drug conjugate" in ctx["previous_canonical_query"].lower()
    # Original raw query preserved (auditable), not replaced by a marker.
    assert "I want to design a new antibody-drug conjugate." in next_raw["raw_user_query"]


def test_second_turn_updates_canonical_query_and_clears_target_block(
    local_storage, registry_service, workflow_state_service
):
    run_id, readiness = _turn_one(
        local_storage, registry_service, workflow_state_service,
        "I want to design a new antibody-drug conjugate.",
    )
    rid = _target_request_id(readiness)
    svc = ClarificationService(local_storage, registry_service, workflow_state_service)
    res = svc.submit_and_reparse(
        run_id, [{"request_id": rid, "answer_text": "HER2", "answered_at": "t"}], _supervisor()
    )
    sq = res.structured_query
    assert sq.task_intent.primary_intent == "new_adc_design"
    assert "HER2" in (sq.canonical_query or "")
    assert not any(
        m.slot_name == "target_or_antigen" and m.severity == "blocking"
        for m in sq.missing_slots
    )
    assert res.input_readiness_status.input_readiness_status != "blocked"
