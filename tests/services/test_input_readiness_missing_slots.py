"""Step 3 minimal consumption of Step 2 ``missing_slots``.

Step 3 stays deterministic: it reflects the LLM-judged required-slot gaps
reported by Step 2 without implementing any user-interaction loop.

- A ``blocking`` missing_slot floors readiness to ``blocked`` and appears in
  ``missing_input_checklist``.
- ``warning`` / ``optional`` missing_slots are informational gaps that never
  block on their own.
- Old artifacts without ``missing_slots`` are unaffected.
"""

from __future__ import annotations

from app.schemas.step_02_structured_query import (
    MissingSlot,
    SourceRawRequestRef,
    StructuredQuery,
    TaskIntent,
)
from app.services.input_readiness_service import InputReadinessService
from app.services.intake_service import IntakeService
from app.utils.ids import new_artifact_id
from app.utils.time import now_iso


def _bootstrap_step_2(
    local_storage,
    registry_service,
    workflow_state_service,
    run_id: str,
    *,
    missing_slots: list[MissingSlot] | None = None,
) -> None:
    reg = registry_service.get(run_id)
    sq = StructuredQuery(
        run_id=run_id,
        parsed_at=now_iso(),
        source_raw_request_ref=SourceRawRequestRef(
            raw_request_record_id=reg.active_artifacts.raw_request_record_id
        ),
        task_intent=TaskIntent(task_type="adc_design", modality="ADC"),
        missing_slots=missing_slots or [],
    )
    sq_id = new_artifact_id("structured_query")
    local_storage.write_json(
        local_storage.run_key(run_id, "inputs/structured_query.json"),
        {"artifact_id": sq_id, **sq.model_dump()},
    )
    registry_service.update_active(run_id, structured_query_id=sq_id)
    workflow_state_service.mark(run_id, "step_02", "completed")


def _full_context_run(local_storage, registry_service, workflow_state_service) -> str:
    """Intake with full ADC context so deterministic checks produce NO
    blocking item — isolating the missing_slots contribution."""
    intake = IntakeService(local_storage, registry_service, workflow_state_service)
    rec = intake.submit(
        raw_user_query="design HER2 ADC with vc-MMAE",
        user_provided_context={
            "target_or_antigen_text": "HER2",
            "candidate_text": "Trastuzumab",
            "payload_linker_text": "vc-MMAE",
        },
    )
    return rec.run_id


def test_step3_consumes_blocking_missing_slot(
    local_storage, registry_service, workflow_state_service
):
    run_id = _full_context_run(local_storage, registry_service, workflow_state_service)
    _bootstrap_step_2(
        local_storage,
        registry_service,
        workflow_state_service,
        run_id,
        missing_slots=[
            MissingSlot(
                slot_name="structure_or_sequence",
                slot_category="structure",
                severity="blocking",
                required_for=["structure_analysis"],
                reason="No structure or sequence input provided.",
                suggested_question="Please provide a PDB/CIF file, PDB ID, UniProt ID, or sequence.",
            )
        ],
    )
    out = InputReadinessService(
        local_storage, registry_service, workflow_state_service
    ).check(run_id)

    assert out.input_readiness_status == "blocked"
    slot_items = [
        m
        for m in out.missing_input_checklist
        if m.field.startswith("structured_query.missing_slots")
    ]
    assert slot_items, "blocking missing_slot must appear in the checklist"
    assert slot_items[0].severity == "blocking"
    # suggested_question is surfaced in the checklist message.
    assert "Suggested question" in slot_items[0].message
    assert any("structure" in r.lower() for r in out.blocking_reasons)


def test_step3_does_not_block_on_warning_only_missing_slot(
    local_storage, registry_service, workflow_state_service
):
    run_id = _full_context_run(local_storage, registry_service, workflow_state_service)
    _bootstrap_step_2(
        local_storage,
        registry_service,
        workflow_state_service,
        run_id,
        missing_slots=[
            MissingSlot(
                slot_name="linker",
                slot_category="linker",
                severity="warning",
                required_for=["new_adc_design"],
                reason="No linker chemistry specified.",
            )
        ],
    )
    out = InputReadinessService(
        local_storage, registry_service, workflow_state_service
    ).check(run_id)

    assert out.input_readiness_status != "blocked"
    assert out.input_readiness_status == "needs_user_input"
    assert not out.blocking_reasons


def test_step3_optional_missing_slot_is_informational_only(
    local_storage, registry_service, workflow_state_service
):
    run_id = _full_context_run(local_storage, registry_service, workflow_state_service)
    _bootstrap_step_2(
        local_storage,
        registry_service,
        workflow_state_service,
        run_id,
        missing_slots=[
            MissingSlot(
                slot_name="constraint",
                slot_category="constraint",
                severity="optional",
                reason="No explicit constraints.",
            )
        ],
    )
    out = InputReadinessService(
        local_storage, registry_service, workflow_state_service
    ).check(run_id)
    assert out.input_readiness_status != "blocked"
    assert not out.blocking_reasons


def test_step3_old_artifact_without_missing_slots_unaffected(
    local_storage, registry_service, workflow_state_service
):
    run_id = _full_context_run(local_storage, registry_service, workflow_state_service)
    # Bootstrap, then strip missing_slots from the persisted artifact to
    # simulate an artifact produced before the field existed.
    _bootstrap_step_2(local_storage, registry_service, workflow_state_service, run_id)
    key = local_storage.run_key(run_id, "inputs/structured_query.json")
    sq = local_storage.read_json(key)
    sq.pop("missing_slots", None)
    local_storage.write_json(key, sq)

    out = InputReadinessService(
        local_storage, registry_service, workflow_state_service
    ).check(run_id)
    # No slot-derived checklist items; status driven purely by deterministic
    # checks (full context → ready).
    assert not [
        m
        for m in out.missing_input_checklist
        if m.field.startswith("structured_query.missing_slots")
    ]
    assert out.input_readiness_status == "ready"


def test_step3_blocking_slot_does_not_duplicate_existing_category(
    local_storage, registry_service, workflow_state_service
):
    """When Step 3's deterministic check already blocks the same category at
    the same severity, the slot is not added as a duplicate line."""
    intake = IntakeService(local_storage, registry_service, workflow_state_service)
    rec = intake.submit(raw_user_query="design an ADC", user_provided_context={})
    run_id = rec.run_id
    _bootstrap_step_2(
        local_storage,
        registry_service,
        workflow_state_service,
        run_id,
        missing_slots=[
            MissingSlot(
                slot_name="target_or_antigen",
                slot_category="target",
                severity="blocking",
                reason="No target.",
            )
        ],
    )
    out = InputReadinessService(
        local_storage, registry_service, workflow_state_service
    ).check(run_id)
    assert out.input_readiness_status == "blocked"
    target_items = [m for m in out.missing_input_checklist if m.category == "target"]
    # Deterministic target check already present; slot deduped against it.
    assert len(target_items) == 1
    assert not target_items[0].field.startswith("structured_query.missing_slots")


# ── Step 3 clarification_requests (minimal backend skeleton) ─────────────────


def test_step3_blocking_slot_generates_clarification_request(
    local_storage, registry_service, workflow_state_service
):
    run_id = _full_context_run(local_storage, registry_service, workflow_state_service)
    _bootstrap_step_2(
        local_storage,
        registry_service,
        workflow_state_service,
        run_id,
        missing_slots=[
            MissingSlot(
                slot_name="structure_or_sequence",
                slot_category="structure",
                severity="blocking",
                required_for=["structure_analysis"],
                reason="No structure or sequence input provided.",
                suggested_question="Please provide a PDB/CIF file, PDB ID, UniProt ID, or sequence.",
            )
        ],
    )
    out = InputReadinessService(
        local_storage, registry_service, workflow_state_service
    ).check(run_id)

    assert out.input_readiness_status == "blocked"
    crs = out.clarification_requests
    assert crs, "blocking missing_slot must yield a clarification request"
    cr = next(c for c in crs if c.slot_name == "structure_or_sequence")
    assert cr.severity == "blocking"
    assert cr.source == "step2_missing_slots"
    assert cr.question == "Please provide a PDB/CIF file, PDB ID, UniProt ID, or sequence."
    assert cr.resolved is False
    assert cr.request_id.startswith("clr_structure_or_sequence_")


def test_step3_warning_slot_generates_request_without_blocking(
    local_storage, registry_service, workflow_state_service
):
    run_id = _full_context_run(local_storage, registry_service, workflow_state_service)
    _bootstrap_step_2(
        local_storage,
        registry_service,
        workflow_state_service,
        run_id,
        missing_slots=[
            MissingSlot(
                slot_name="linker",
                slot_category="linker",
                severity="warning",
                reason="No linker chemistry specified.",
                suggested_question="Which linker chemistry should we use?",
            )
        ],
    )
    out = InputReadinessService(
        local_storage, registry_service, workflow_state_service
    ).check(run_id)

    assert out.input_readiness_status != "blocked"
    assert out.input_readiness_status == "needs_user_input"
    linker_reqs = [c for c in out.clarification_requests if c.slot_name == "linker"]
    assert linker_reqs and linker_reqs[0].severity == "warning"
    assert not out.blocking_reasons


def test_step3_optional_slot_stays_checklist_only_no_request(
    local_storage, registry_service, workflow_state_service
):
    """Design choice: optional slots are informational and do NOT generate a
    clarification request (less noise); they remain on the checklist."""
    run_id = _full_context_run(local_storage, registry_service, workflow_state_service)
    _bootstrap_step_2(
        local_storage,
        registry_service,
        workflow_state_service,
        run_id,
        missing_slots=[
            MissingSlot(
                slot_name="constraint",
                slot_category="constraint",
                severity="optional",
                reason="No explicit constraints.",
                suggested_question="Any constraints?",
            )
        ],
    )
    out = InputReadinessService(
        local_storage, registry_service, workflow_state_service
    ).check(run_id)
    assert not any(c.slot_name == "constraint" for c in out.clarification_requests)
    assert any(
        m.field.startswith("structured_query.missing_slots") and m.category == "constraints"
        for m in out.missing_input_checklist
    )


def test_step3_clarification_preserves_step2_question_when_checklist_dedupes(
    local_storage, registry_service, workflow_state_service
):
    """The core fix: when a deterministic check and a Step 2 slot share a
    category, the checklist may dedupe the Step 2 entry, but the Step 2
    `suggested_question` MUST survive in clarification_requests."""
    intake = IntakeService(local_storage, registry_service, workflow_state_service)
    rec = intake.submit(raw_user_query="design an ADC", user_provided_context={})
    run_id = rec.run_id
    step2_question = "What target or antigen should the ADC be designed against?"
    _bootstrap_step_2(
        local_storage,
        registry_service,
        workflow_state_service,
        run_id,
        missing_slots=[
            MissingSlot(
                slot_name="target_or_antigen",
                slot_category="target",
                severity="blocking",
                reason="No target.",
                suggested_question=step2_question,
            )
        ],
    )
    out = InputReadinessService(
        local_storage, registry_service, workflow_state_service
    ).check(run_id)

    # Checklist deduped the Step 2 target slot against the deterministic one.
    target_items = [m for m in out.missing_input_checklist if m.category == "target"]
    assert len(target_items) == 1
    assert not target_items[0].field.startswith("structured_query.missing_slots")

    # But the Step 2 suggested_question survives in clarification_requests,
    # sourced from Step 2 (not the deterministic fallback).
    target_reqs = [c for c in out.clarification_requests if c.slot_category == "target"]
    assert len(target_reqs) == 1
    assert target_reqs[0].question == step2_question
    assert target_reqs[0].source == "step2_missing_slots"


def test_step3_deterministic_gap_yields_request_when_no_step2_slot(
    local_storage, registry_service, workflow_state_service
):
    """Old Step 2 artifacts (no missing_slots) still surface a question for a
    deterministic blocking gap, sourced as deterministic_readiness."""
    intake = IntakeService(local_storage, registry_service, workflow_state_service)
    rec = intake.submit(raw_user_query="design an ADC", user_provided_context={})
    run_id = rec.run_id
    _bootstrap_step_2(local_storage, registry_service, workflow_state_service, run_id)
    # Simulate a pre-missing_slots artifact.
    key = local_storage.run_key(run_id, "inputs/structured_query.json")
    sq = local_storage.read_json(key)
    sq.pop("missing_slots", None)
    local_storage.write_json(key, sq)

    out = InputReadinessService(
        local_storage, registry_service, workflow_state_service
    ).check(run_id)
    assert out.input_readiness_status == "blocked"
    target_reqs = [c for c in out.clarification_requests if c.slot_category == "target"]
    assert target_reqs and target_reqs[0].source == "deterministic_readiness"


def test_step3_request_id_is_deterministic_across_runs(
    local_storage, registry_service, workflow_state_service
):
    def _run_once() -> list[str]:
        intake = IntakeService(local_storage, registry_service, workflow_state_service)
        rec = intake.submit(raw_user_query="design an ADC", user_provided_context={})
        _bootstrap_step_2(
            local_storage,
            registry_service,
            workflow_state_service,
            rec.run_id,
            missing_slots=[
                MissingSlot(
                    slot_name="target_or_antigen",
                    slot_category="target",
                    severity="blocking",
                    reason="No target.",
                    suggested_question="What target or antigen should the ADC be designed against?",
                )
            ],
        )
        out = InputReadinessService(
            local_storage, registry_service, workflow_state_service
        ).check(rec.run_id)
        return [c.request_id for c in out.clarification_requests]

    first = _run_once()
    second = _run_once()
    assert first == second
    assert all(rid.startswith("clr_") for rid in first)


def test_step3_old_artifact_without_missing_slots_has_empty_or_deterministic_requests(
    local_storage, registry_service, workflow_state_service
):
    """Backward compatible: a fully satisfied run with no missing_slots has
    no clarification requests."""
    run_id = _full_context_run(local_storage, registry_service, workflow_state_service)
    _bootstrap_step_2(local_storage, registry_service, workflow_state_service, run_id)
    key = local_storage.run_key(run_id, "inputs/structured_query.json")
    sq = local_storage.read_json(key)
    sq.pop("missing_slots", None)
    local_storage.write_json(key, sq)
    out = InputReadinessService(
        local_storage, registry_service, workflow_state_service
    ).check(run_id)
    assert out.input_readiness_status == "ready"
    assert out.clarification_requests == []


def test_step3_clarification_requests_do_not_leak_sequences_or_keys(
    local_storage, registry_service, workflow_state_service
):
    heavy = "EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGK"
    run_id = _full_context_run(local_storage, registry_service, workflow_state_service)
    _bootstrap_step_2(
        local_storage,
        registry_service,
        workflow_state_service,
        run_id,
        missing_slots=[
            MissingSlot(
                slot_name="structure_or_sequence",
                slot_category="structure",
                severity="blocking",
                reason="No structure or sequence input provided.",
                suggested_question="Please provide a PDB/CIF file, PDB ID, UniProt ID, or sequence.",
            )
        ],
    )
    out = InputReadinessService(
        local_storage, registry_service, workflow_state_service
    ).check(run_id)
    blob = str([c.model_dump() for c in out.clarification_requests])
    assert heavy not in blob
    assert "api_key" not in blob.lower()
    assert "system instructions" not in blob.lower()


# ── Step 3 user-facing response passthrough (no LLM) ─────────────────────────


def _set_step2_response(local_storage, run_id: str, response) -> None:
    key = local_storage.run_key(run_id, "inputs/structured_query.json")
    sq = local_storage.read_json(key)
    sq["response"] = response
    local_storage.write_json(key, sq)


def test_step3_passes_through_step2_response_when_blocked(
    local_storage, registry_service, workflow_state_service
):
    intake = IntakeService(local_storage, registry_service, workflow_state_service)
    rec = intake.submit(raw_user_query="design an ADC", user_provided_context={})
    run_id = rec.run_id
    _bootstrap_step_2(
        local_storage,
        registry_service,
        workflow_state_service,
        run_id,
        missing_slots=[
            MissingSlot(
                slot_name="target_or_antigen",
                slot_category="target",
                severity="blocking",
                reason="No target.",
                suggested_question="What target or antigen should the ADC be designed against?",
            )
        ],
    )
    step2_msg = "Please provide the target or antigen for the ADC."
    _set_step2_response(local_storage, run_id, step2_msg)

    out = InputReadinessService(
        local_storage, registry_service, workflow_state_service
    ).check(run_id)
    assert out.input_readiness_status == "blocked"
    assert out.response == step2_msg


def test_step3_passes_through_step2_response_when_needs_user_input(
    local_storage, registry_service, workflow_state_service
):
    run_id = _full_context_run(local_storage, registry_service, workflow_state_service)
    _bootstrap_step_2(
        local_storage,
        registry_service,
        workflow_state_service,
        run_id,
        missing_slots=[
            MissingSlot(
                slot_name="linker",
                slot_category="linker",
                severity="warning",
                reason="No linker chemistry specified.",
            )
        ],
    )
    step2_msg = "Please provide the linker chemistry you want to use."
    _set_step2_response(local_storage, run_id, step2_msg)
    out = InputReadinessService(
        local_storage, registry_service, workflow_state_service
    ).check(run_id)
    assert out.input_readiness_status == "needs_user_input"
    assert out.response == step2_msg


def test_step3_ready_status_has_no_response(
    local_storage, registry_service, workflow_state_service
):
    run_id = _full_context_run(local_storage, registry_service, workflow_state_service)
    _bootstrap_step_2(local_storage, registry_service, workflow_state_service, run_id)
    # Even if Step 2 left a stray response, a ready run does not surface it.
    _set_step2_response(local_storage, run_id, "stray message")
    out = InputReadinessService(
        local_storage, registry_service, workflow_state_service
    ).check(run_id)
    assert out.input_readiness_status == "ready"
    assert out.response is None


def test_step3_deterministic_fallback_response_when_step2_absent(
    local_storage, registry_service, workflow_state_service
):
    """Step 2 left no response, but Step 3 has clarification_requests → the
    fallback joins those questions deterministically (no LLM)."""
    intake = IntakeService(local_storage, registry_service, workflow_state_service)
    rec = intake.submit(raw_user_query="design an ADC", user_provided_context={})
    run_id = rec.run_id
    _bootstrap_step_2(
        local_storage,
        registry_service,
        workflow_state_service,
        run_id,
        missing_slots=[
            MissingSlot(
                slot_name="target_or_antigen",
                slot_category="target",
                severity="blocking",
                reason="No target.",
                suggested_question="What target or antigen should the ADC be designed against?",
            )
        ],
    )
    # No structured_query.response set at all.
    out = InputReadinessService(
        local_storage, registry_service, workflow_state_service
    ).check(run_id)
    assert out.input_readiness_status == "blocked"
    assert out.response
    assert out.clarification_requests
    # Fallback is built from the clarification question(s).
    assert any(c.question in out.response for c in out.clarification_requests)


def test_step3_response_does_not_leak_sequences_or_keys(
    local_storage, registry_service, workflow_state_service
):
    heavy = "EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGK"
    run_id = _full_context_run(local_storage, registry_service, workflow_state_service)
    _bootstrap_step_2(
        local_storage,
        registry_service,
        workflow_state_service,
        run_id,
        missing_slots=[
            MissingSlot(
                slot_name="linker",
                slot_category="linker",
                severity="warning",
                reason="No linker chemistry specified.",
            )
        ],
    )
    out = InputReadinessService(
        local_storage, registry_service, workflow_state_service
    ).check(run_id)
    blob = out.response or ""
    assert heavy not in blob
    assert "api_key" not in blob.lower()
    assert "system instructions" not in blob.lower()
