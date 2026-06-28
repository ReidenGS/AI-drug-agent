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
