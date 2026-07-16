"""Step 3 minimal consumption of Step 2 ``missing_slots``.

Step 3 stays deterministic: it reflects the LLM-judged required-slot gaps
reported by Step 2 without implementing any user-interaction loop.

- A recoverable ``blocking`` missing_slot floors readiness to
  ``needs_user_input`` and appears in ``missing_input_checklist``.
- ``warning`` / ``optional`` missing_slots are informational gaps that never
  block on their own.
- Old artifacts without ``missing_slots`` are unaffected.
"""

from __future__ import annotations

import pytest

from app.agents.supervisor_agent import SupervisorAgent
from app.llm.provider import MockLLMProvider
from app.schemas.step_02_structured_query import (
    MissingSlot,
    SourceRawRequestRef,
    StructuredQuery,
    TaskIntent,
)
from app.services.input_readiness_service import InputReadinessService
from app.services.intake_service import IntakeService
from app.services.structured_query_service import StructuredQueryService
from app.utils.ids import new_artifact_id
from app.utils.time import now_iso


def _make_slot(
    slot_name: str,
    slot_category: str,
    severity: str,
    reason: str,
    *,
    suggested_question: str | None = None,
) -> MissingSlot:
    return MissingSlot(
        slot_name=slot_name,
        slot_category=slot_category,
        severity=severity,
        reason=reason,
        suggested_question=suggested_question,
        required_for=["developability_assessment"],
    )


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


def _run_real_step2_and_step3(
    local_storage, registry_service, workflow_state_service, query: str
):
    """Exercise the production Step 2/3 path with the deterministic mock LLM.

    The mock isolates external LLM access; parsing, projection, persistence,
    missing-slot generation, and readiness evaluation are production code.
    """
    record = IntakeService(
        local_storage, registry_service, workflow_state_service
    ).submit(raw_user_query=query, user_provided_context={})
    structured = StructuredQueryService(
        local_storage,
        registry_service,
        workflow_state_service,
        SupervisorAgent(llm=MockLLMProvider()),
    ).parse(record.run_id)
    readiness = InputReadinessService(
        local_storage, registry_service, workflow_state_service
    ).check(record.run_id)
    return structured, readiness


@pytest.mark.parametrize(
    "id_type",
    [
        "target_sequence",
        "antibody_heavy_chain_sequence",
        "antibody_light_chain_sequence",
    ],
)
def test_invalid_typed_sequence_becomes_step3_clarification_without_worker(
    local_storage,
    registry_service,
    workflow_state_service,
    id_type,
):
    invalid = "ACDE?FG"

    class _InvalidTypedSequenceProvider:
        """Test-only LLM fixture; normalization and Step 3 are production."""

        name = "test-only-invalid-target-sequence"
        model = "test-only"

        def __init__(self):
            self.inner = MockLLMProvider()
            self.call_count = 0

        def generate_json(self, prompt, *, schema, system=None):
            self.call_count += 1
            result = self.inner.generate_json(
                prompt, schema=schema, system=system
            )
            result["referenced_inputs"] = [
                {
                    "id_type": id_type,
                    "value": invalid,
                    "source": "user",
                }
            ]
            result["missing_slots"] = []
            result["response"] = None
            return result

    provider = _InvalidTypedSequenceProvider()
    raw = IntakeService(
        local_storage, registry_service, workflow_state_service
    ).submit(
        raw_user_query=f"Analyze the HER2 structure using sequence {invalid}.",
        user_provided_context={},
    )
    structured = StructuredQueryService(
        local_storage,
        registry_service,
        workflow_state_service,
        SupervisorAgent(llm=provider),
    ).parse(raw.run_id)
    readiness = InputReadinessService(
        local_storage, registry_service, workflow_state_service
    ).check(raw.run_id)

    assert provider.call_count == 1
    assert structured.referenced_inputs == []
    assert invalid not in " ".join(structured.parse_warnings)
    expected_warning = (
        "dropped invalid target_sequence referenced_input"
        if id_type == "target_sequence"
        else "dropped invalid antibody chain sequence referenced_input"
    )
    assert expected_warning in structured.parse_warnings
    slots = [
        slot
        for slot in structured.missing_slots
        if slot.slot_name == "structure_or_sequence"
    ]
    assert len(slots) == 1
    assert slots[0].model_dump() == {
        "slot_name": "structure_or_sequence",
        "slot_category": "sequence",
        "severity": "blocking",
        "required_for": [],
        "reason": "The supplied protein sequence could not be used.",
        "suggested_question": "Please provide a valid protein sequence.",
        "evidence": None,
    }
    assert readiness.input_readiness_status == "needs_user_input"
    assert readiness.blocking_reasons
    requests = [
        request
        for request in readiness.clarification_requests
        if request.slot_name == "structure_or_sequence" and not request.resolved
    ]
    assert len(requests) == 1
    assert requests[0].severity == "blocking"
    assert requests[0].question == "Please provide a valid protein sequence."
    assert "Please provide a valid protein sequence." in (readiness.response or "")
    active = registry_service.get(raw.run_id).active_artifacts
    assert active.candidate_context_table_id is None
    assert active.prepared_structure_input_package_id is None
    assert not any(
        "tool_call_records" in key
        for key in local_storage.list_prefix(local_storage.run_key(raw.run_id))
    )


def test_invalid_redundant_sequence_does_not_block_valid_pdb(
    local_storage,
    registry_service,
    workflow_state_service,
):
    class _InvalidSequenceAndValidPdbProvider:
        """Test-only LLM fixture returning two explicit typed roles."""

        name = "test-only-invalid-sequence-valid-pdb"
        model = "test-only"

        def __init__(self):
            self.inner = MockLLMProvider()

        def generate_json(self, prompt, *, schema, system=None):
            result = self.inner.generate_json(
                prompt, schema=schema, system=system
            )
            result["referenced_inputs"] = [
                {
                    "id_type": "target_sequence",
                    "value": "ACDE?FG",
                    "source": "user",
                },
                {"id_type": "pdb_id", "value": "1N8Z", "source": "user"},
            ]
            result["missing_slots"] = []
            result["response"] = None
            return result

    raw = IntakeService(
        local_storage, registry_service, workflow_state_service
    ).submit(
        raw_user_query="Analyze the HER2 structure using PDB 1N8Z.",
        user_provided_context={},
    )
    structured = StructuredQueryService(
        local_storage,
        registry_service,
        workflow_state_service,
        SupervisorAgent(llm=_InvalidSequenceAndValidPdbProvider()),
    ).parse(raw.run_id)
    readiness = InputReadinessService(
        local_storage, registry_service, workflow_state_service
    ).check(raw.run_id)

    assert structured.referenced_inputs == [
        {"id_type": "pdb_id", "value": "1N8Z", "source": "user"}
    ]
    assert not any(
        slot.slot_name == "structure_or_sequence"
        for slot in structured.missing_slots
    )
    assert readiness.input_readiness_status == "ready"
    assert not any(
        request.slot_name == "structure_or_sequence"
        for request in readiness.clarification_requests
    )


def test_natural_structure_query_without_pdb_requests_recoverable_input(
    local_storage, registry_service, workflow_state_service
):
    structured, readiness = _run_real_step2_and_step3(
        local_storage,
        registry_service,
        workflow_state_service,
        "Analyze the structure of HER2 and report the binding interface.",
    )

    assert structured.task_intent.primary_intent == "structure_analysis"
    assert any(
        slot.slot_name == "structure_or_sequence" and slot.severity == "blocking"
        for slot in structured.missing_slots
    )
    assert readiness.input_readiness_status == "needs_user_input"
    assert any(
        request.slot_name == "structure_or_sequence"
        and request.severity == "blocking"
        for request in readiness.clarification_requests
    )
    assert not any(
        request.slot_name in {"payload", "linker"}
        for request in readiness.clarification_requests
    )


def test_natural_masked_generation_without_prompt_has_no_adc_component_noise(
    local_storage, registry_service, workflow_state_service
):
    structured, readiness = _run_real_step2_and_step3(
        local_storage,
        registry_service,
        workflow_state_service,
        "Optimize and generate a protein sequence for EGFR using masked protein generation.",
    )

    assert any(
        slot.slot_name == "prompt_sequence" and slot.severity == "blocking"
        for slot in structured.missing_slots
    )
    assert readiness.input_readiness_status == "needs_user_input"
    assert any(
        request.slot_name == "prompt_sequence"
        and request.severity == "blocking"
        for request in readiness.clarification_requests
    )
    assert not any(
        request.slot_name in {"payload", "linker"}
        for request in readiness.clarification_requests
    )


def test_uploaded_prompt_fasta_does_not_satisfy_structure_or_sequence(
    local_storage, registry_service, workflow_state_service
):
    record = IntakeService(
        local_storage, registry_service, workflow_state_service
    ).submit(
        raw_user_query="Analyze a protein structure using a separate generation prompt.",
        user_provided_context={},
        uploaded_files=[
            {
                "file_id": "file_prompt_fasta",
                "original_filename": "generation_prompt.fasta",
                "storage_path": "adc_pilot/runs/x/inputs/files/generation_prompt.fasta",
                "content_type": "text/x-fasta",
                "sha256": "a" * 64,
                "size_bytes": 128,
            }
        ],
    )
    run_id = record.run_id
    sq = StructuredQuery(
        run_id=run_id,
        parsed_at=now_iso(),
        source_raw_request_ref=SourceRawRequestRef(
            raw_request_record_id=registry_service.get(
                run_id
            ).active_artifacts.raw_request_record_id
        ),
        task_intent=TaskIntent(
            task_type="structure_preparation",
            primary_intent="structure_analysis",
        ),
        referenced_inputs=[
            {
                "id_type": "uploaded_file",
                "value": "file_prompt_fasta",
                "source": "prompt_sequence",
            }
        ],
        missing_slots=[
            MissingSlot(
                slot_name="structure_or_sequence",
                slot_category="sequence",
                severity="blocking",
                reason="A generation prompt is not a complete analysis input.",
                suggested_question="Please provide a structure or complete sequence.",
            )
        ],
    )
    sq_id = new_artifact_id("structured_query")
    local_storage.write_json(
        local_storage.run_key(run_id, "inputs/structured_query.json"),
        {"artifact_id": sq_id, **sq.model_dump(mode="json")},
    )
    registry_service.update_active(run_id, structured_query_id=sq_id)
    workflow_state_service.mark(run_id, "step_02", "completed")

    readiness = InputReadinessService(
        local_storage, registry_service, workflow_state_service
    ).check(run_id)
    assert readiness.input_readiness_status == "needs_user_input"
    assert readiness.basic_adc_input_presence.sequence_input_present is False
    assert readiness.basic_adc_input_presence.structure_or_sequence_present is False
    assert any(
        request.slot_name == "structure_or_sequence"
        for request in readiness.clarification_requests
    )


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
            _make_slot(
                "structure_or_sequence",
                "structure",
                "blocking",
                "No structure or sequence input provided.",
                suggested_question=(
                    "Please provide a PDB/CIF file, PDB ID, UniProt ID, or sequence."
                ),
            )
        ],
    )
    out = InputReadinessService(
        local_storage, registry_service, workflow_state_service
    ).check(run_id)

    assert out.input_readiness_status == "needs_user_input"
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


def test_step3_consumes_blocking_sequence_role_slot(
    local_storage, registry_service, workflow_state_service
):
    run_id = _full_context_run(local_storage, registry_service, workflow_state_service)
    _bootstrap_step_2(
        local_storage,
        registry_service,
        workflow_state_service,
        run_id,
        missing_slots=[
            _make_slot(
                "sequence_role",
                "sequence",
                "blocking",
                "Need FASTA role before proceeding.",
                suggested_question="Please confirm whether sequence is heavy chain, light chain, or target antigen.",
            )
        ],
    )
    out = InputReadinessService(
        local_storage, registry_service, workflow_state_service
    ).check(run_id)

    assert out.input_readiness_status == "needs_user_input"
    seq_items = [
        m
        for m in out.missing_input_checklist
        if m.field.startswith("structured_query.missing_slots")
        and m.category == "structure_or_sequence"
    ]
    assert seq_items, "sequence_role blocking slot must map to structure/sequence checklist"
    requests = [
        r for r in out.clarification_requests if r.slot_name == "sequence_role"
    ]
    assert requests and requests[0].severity == "blocking"
    assert requests[0].source == "step2_missing_slots"


def test_step3_does_not_block_on_warning_sequence_role_slot(
    local_storage, registry_service, workflow_state_service
):
    run_id = _full_context_run(local_storage, registry_service, workflow_state_service)
    _bootstrap_step_2(
        local_storage,
        registry_service,
        workflow_state_service,
        run_id,
        missing_slots=[
            _make_slot(
                "sequence_role",
                "sequence",
                "warning",
                "Sequence role is ambiguous but can be inferred later.",
                suggested_question="Please confirm sequence role.",
            )
        ],
    )
    out = InputReadinessService(
        local_storage, registry_service, workflow_state_service
    ).check(run_id)

    assert out.input_readiness_status == "needs_user_input"
    assert not out.blocking_reasons
    assert any(
        m.field.startswith("structured_query.missing_slots")
        for m in out.missing_input_checklist
    )


def test_step3_does_not_block_on_optional_sequence_role_slot(
    local_storage, registry_service, workflow_state_service
):
    run_id = _full_context_run(local_storage, registry_service, workflow_state_service)
    _bootstrap_step_2(
        local_storage,
        registry_service,
        workflow_state_service,
        run_id,
        missing_slots=[
            _make_slot(
                "sequence_role",
                "sequence",
                "optional",
                "Sequence role hint is optional.",
                suggested_question="Optional hint.",
            )
        ],
    )
    out = InputReadinessService(
        local_storage, registry_service, workflow_state_service
    ).check(run_id)

    assert out.input_readiness_status == "ready"
    assert not any(r.slot_name == "sequence_role" for r in out.clarification_requests)
    # Optional sequence-role hints should remain checklist-only and never
    # block; whether they appear as a separate checklist line is
    # implementation-defined when the same warning is already present
    # deterministically.


def test_step3_no_sequence_role_item_when_missing_slot_absent(
    local_storage, registry_service, workflow_state_service
):
    run_id = _full_context_run(local_storage, registry_service, workflow_state_service)
    _bootstrap_step_2(
        local_storage, registry_service, workflow_state_service, run_id
    )
    out = InputReadinessService(
        local_storage, registry_service, workflow_state_service
    ).check(run_id)
    assert out.input_readiness_status == "ready"
    assert not any(
        m.field.startswith("structured_query.missing_slots")
        and "slot_name=sequence_role" in m.field
        for m in out.missing_input_checklist
    )


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
            _make_slot("linker", "linker", "warning", "No linker chemistry specified.")
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
            _make_slot("constraint", "constraint", "optional", "No explicit constraints.")
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
            _make_slot("target_or_antigen", "target", "blocking", "No target.")
        ],
    )
    out = InputReadinessService(
        local_storage, registry_service, workflow_state_service
    ).check(run_id)
    assert out.input_readiness_status == "needs_user_input"
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
            _make_slot(
                "structure_or_sequence",
                "structure",
                "blocking",
                "No structure or sequence input provided.",
                suggested_question=(
                    "Please provide a PDB/CIF file, PDB ID, UniProt ID, or sequence."
                ),
            )
        ],
    )
    out = InputReadinessService(
        local_storage, registry_service, workflow_state_service
    ).check(run_id)

    assert out.input_readiness_status == "needs_user_input"
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
            _make_slot(
                "linker",
                "linker",
                "warning",
                "No linker chemistry specified.",
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
            _make_slot(
                "constraint",
                "constraint",
                "optional",
                "No explicit constraints.",
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
            _make_slot(
                "target_or_antigen",
                "target",
                "blocking",
                "No target.",
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
    assert out.input_readiness_status == "needs_user_input"
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


@pytest.mark.parametrize(
    ("accession", "expected_present"),
    [("P04626", True), ("not-a-uniprot-accession", False)],
)
def test_step3_only_counts_valid_typed_uniprot_as_sequence_input(
    local_storage,
    registry_service,
    workflow_state_service,
    accession,
    expected_present,
):
    run_id = _full_context_run(local_storage, registry_service, workflow_state_service)
    _bootstrap_step_2(local_storage, registry_service, workflow_state_service, run_id)
    key = local_storage.run_key(run_id, "inputs/structured_query.json")
    sq = local_storage.read_json(key)
    sq["referenced_inputs"] = [
        {"id_type": "uniprot_id", "value": accession, "source": "user"}
    ]
    local_storage.write_json(key, sq)

    out = InputReadinessService(
        local_storage, registry_service, workflow_state_service
    ).check(run_id)
    assert out.basic_adc_input_presence.sequence_input_present is expected_present
    assert (
        out.basic_adc_input_presence.structure_or_sequence_present
        is expected_present
    )
    expected_evidence = (
        "structured_query.referenced_inputs[id_type=uniprot_id]"
        if expected_present
        else None
    )
    assert out.basic_adc_input_presence.sequence_input_evidence == expected_evidence
    assert (
        out.basic_adc_input_presence.structure_or_sequence_evidence
        == expected_evidence
    )


def test_step3_target_sequence_presence_and_evidence_are_consistent(
    local_storage,
    registry_service,
    workflow_state_service,
):
    run_id = _full_context_run(local_storage, registry_service, workflow_state_service)
    _bootstrap_step_2(local_storage, registry_service, workflow_state_service, run_id)
    key = local_storage.run_key(run_id, "inputs/structured_query.json")
    sq = local_storage.read_json(key)
    sq["referenced_inputs"] = [
        {
            "id_type": "target_sequence",
            "value": "ACDEFGHIK",
            "source": "user",
        }
    ]
    local_storage.write_json(key, sq)

    out = InputReadinessService(
        local_storage, registry_service, workflow_state_service
    ).check(run_id)
    expected = "structured_query.referenced_inputs[id_type=target_sequence]"
    assert out.basic_adc_input_presence.sequence_input_present is True
    assert out.basic_adc_input_presence.structure_or_sequence_present is True
    assert out.basic_adc_input_presence.sequence_input_evidence == expected
    assert out.basic_adc_input_presence.structure_or_sequence_evidence == expected


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
    assert out.input_readiness_status == "needs_user_input"
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
    assert out.input_readiness_status == "needs_user_input"
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


# ── antibody heavy/light sequence developability assessment ──────────────────


def _seed_sequence_developability(
    local_storage, registry_service, workflow_state_service, *, missing_slots=None
) -> str:
    """A developability_assessment run carrying antibody heavy/light chain
    sequence referenced_inputs (no target/payload/linker)."""
    intake = IntakeService(local_storage, registry_service, workflow_state_service)
    rec = intake.submit(
        raw_user_query=(
            "Run a developability/liability pre-filter on these antibody "
            "heavy and light chain sequences."
        ),
        user_provided_context={},
    )
    run_id = rec.run_id
    sq = StructuredQuery(
        run_id=run_id,
        parsed_at=now_iso(),
        source_raw_request_ref=SourceRawRequestRef(
            raw_request_record_id=registry_service.get(run_id).active_artifacts.raw_request_record_id
        ),
        task_intent=TaskIntent(
            task_type="developability_assessment",
            primary_intent="developability_assessment",
        ),
        referenced_inputs=[
            {"id_type": "antibody_heavy_chain_sequence", "value": "EVQLHEAVY", "source": "user"},
            {"id_type": "antibody_light_chain_sequence", "value": "DIQMLIGHT", "source": "user"},
        ],
        missing_slots=missing_slots or [],
        canonical_query=(
            "developability/liability pre-filter for antibody heavy/light "
            "sequences, not new ADC design"
        ),
    )
    sq_id = new_artifact_id("structured_query")
    local_storage.write_json(
        local_storage.run_key(run_id, "inputs/structured_query.json"),
        {"artifact_id": sq_id, **sq.model_dump()},
    )
    registry_service.update_active(run_id, structured_query_id=sq_id)
    workflow_state_service.mark(run_id, "step_02", "completed")
    return run_id


def test_step3_sequence_developability_not_blocked(
    local_storage, registry_service, workflow_state_service
):
    run_id = _seed_sequence_developability(
        local_storage, registry_service, workflow_state_service
    )
    out = InputReadinessService(
        local_storage, registry_service, workflow_state_service
    ).check(run_id)
    assert out.input_readiness_status != "blocked"
    assert not out.blocking_reasons
    # No target gap fabricated from the legacy ADC checklist.
    assert not any(m.category == "target" for m in out.missing_input_checklist)


def test_step3_sequence_developability_does_not_ask_for_target(
    local_storage, registry_service, workflow_state_service
):
    run_id = _seed_sequence_developability(
        local_storage, registry_service, workflow_state_service
    )
    out = InputReadinessService(
        local_storage, registry_service, workflow_state_service
    ).check(run_id)
    assert not out.response or "target" not in out.response.lower()


def test_step3_sequence_input_present_and_evidence(
    local_storage, registry_service, workflow_state_service
):
    run_id = _seed_sequence_developability(
        local_storage, registry_service, workflow_state_service
    )
    out = InputReadinessService(
        local_storage, registry_service, workflow_state_service
    ).check(run_id)
    p = out.basic_adc_input_presence
    assert p.sequence_input_present is True
    assert p.structure_or_sequence_present is True
    assert p.antibody_candidate_present is True
    assert p.sequence_input_evidence == (
        "structured_query.referenced_inputs[id_type=antibody_heavy_chain_sequence]"
    )
    assert p.antibody_evidence == (
        "structured_query.referenced_inputs[id_type=antibody_heavy_chain_sequence]"
    )


def test_step3_sequence_developability_requests_clarification_for_blocking_slot(
    local_storage, registry_service, workflow_state_service
):
    """A blocking but answerable Step 2 slot remains outside worker routing."""
    run_id = _seed_sequence_developability(
        local_storage, registry_service, workflow_state_service,
        missing_slots=[
            MissingSlot(
                slot_name="structure_or_sequence",
                slot_category="sequence",
                severity="blocking",
                reason="Sequences could not be parsed.",
                suggested_question="Please re-upload the antibody sequences.",
            )
        ],
    )
    out = InputReadinessService(
        local_storage, registry_service, workflow_state_service
    ).check(run_id)
    assert out.input_readiness_status == "needs_user_input"
    assert out.blocking_reasons


def test_step3_sequence_developability_no_raw_sequence_leak(
    local_storage, registry_service, workflow_state_service
):
    run_id = _seed_sequence_developability(
        local_storage, registry_service, workflow_state_service
    )
    out = InputReadinessService(
        local_storage, registry_service, workflow_state_service
    ).check(run_id)
    blob = str(out.model_dump())
    assert "EVQLHEAVY" not in blob
    assert "DIQMLIGHT" not in blob
