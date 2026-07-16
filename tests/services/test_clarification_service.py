"""Step 3 clarification multi-turn loop — minimal backend closed loop.

Covers: saving answers, request-id validation, deterministic duplicate
handling, next-turn context construction (original query + previous intent +
answers), the Step 2/3 re-parse loop, resolved/unresolved bookkeeping, and
privacy. No LangGraph memory/checkpointer is used; Step 3 calls no LLM.
"""

from __future__ import annotations

import json

import pytest

from app.a2a.orchestrator_readiness import OrchestratorReadinessError
from app.agents.supervisor_agent import SupervisorAgent
from app.llm.provider import MockLLMProvider
from app.services.clarification_service import (
    ClarificationConflictError,
    ClarificationRequestError,
    ClarificationReparseError,
    ClarificationService,
)
from app.services.input_readiness_service import InputReadinessService
from app.services.intake_service import IntakeService
from app.services.structured_query_service import StructuredQueryService


def _supervisor() -> SupervisorAgent:
    return SupervisorAgent(llm=MockLLMProvider())


class _CapturingSupervisor:
    def __init__(self):
        self.inner = _supervisor()
        self.inputs: list[dict] = []

    def parse_raw_to_structured_query(self, payload: dict):
        self.inputs.append(json.loads(json.dumps(payload)))
        return self.inner.parse_raw_to_structured_query(payload)


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


def _revision_key(storage, run_id: str, revision_id: str) -> str:
    return storage.run_key(run_id, "clarification", f"{revision_id}.json")


@pytest.mark.parametrize(
    "mutate",
    [
        lambda body: body["clarification_answers"][0].update(
            target_slot_name="payload"
        ),
        lambda body: body["clarification_answers"][0].update(
            target_slot_category="payload"
        ),
        lambda body: body["clarification_answers"][0].update(
            answer_text="tampered-safe-answer"
        ),
        lambda body: body.update(
            submission_fingerprint="clarification_submission_" + "0" * 64
        ),
        lambda body: body.update(resolved_request_ids=[]),
        lambda body: body.update(unresolved_request_ids=[]),
        lambda body: body.update(
            revision_status="completed",
            output_structured_query_id=None,
            output_input_readiness_status_id=None,
            failure_code=None,
        ),
        lambda body: body.update(
            revision_status="submitted",
            output_structured_query_id="structured_query_000000000000",
            output_input_readiness_status_id=None,
            failure_code=None,
        ),
        lambda body: body.update(
            revision_status="reparse_failed",
            output_structured_query_id="structured_query_000000000000",
            output_input_readiness_status_id=None,
            failure_code="clarification_step2_failed",
        ),
        lambda body: body.update(
            revision_status="reparse_failed",
            output_structured_query_id=None,
            output_input_readiness_status_id=None,
            failure_code="clarification_step3_failed",
        ),
    ],
    ids=[
        "slot-name",
        "slot-category",
        "answer-with-stale-fingerprint",
        "fingerprint",
        "resolved-ids",
        "unresolved-ids",
        "phase-combination",
        "submitted-with-output",
        "step2-failure-with-output",
        "step3-failure-without-output",
    ],
)
def test_revision_authority_tampering_fails_before_llm_or_new_files(
    local_storage,
    registry_service,
    workflow_state_service,
    mutate,
):
    run_id, readiness = _turn_one(
        local_storage,
        registry_service,
        workflow_state_service,
        "I want to design a new antibody-drug conjugate.",
    )
    request_id = _target_request_id(readiness)
    service = ClarificationService(
        local_storage, registry_service, workflow_state_service
    )
    state = service.submit_clarification_answer(
        run_id,
        [{"request_id": request_id, "answer_text": "HER2", "answered_at": "t"}],
    )
    key = _revision_key(local_storage, run_id, state.revision_id)
    body = local_storage.read_json(key)
    mutate(body)
    local_storage.write_json(key, body)
    before = set(local_storage.list_prefix(local_storage.run_key(run_id)))
    supervisor = _CapturingSupervisor()

    with pytest.raises(
        ClarificationConflictError,
        match="^clarification_revision_authority_invalid$",
    ):
        ClarificationService(
            local_storage, registry_service, workflow_state_service
        ).submit_and_reparse(
            run_id,
            [
                {
                    "request_id": request_id,
                    "answer_text": "HER2",
                    "answered_at": "later",
                }
            ],
            supervisor,
        )

    assert supervisor.inputs == []
    assert set(local_storage.list_prefix(local_storage.run_key(run_id))) == before
    active = registry_service.get(run_id).active_artifacts
    assert active.worker_routing_plan_id is None
    assert active.candidate_context_table_id is None


# ── A. basic save ─────────────────────────────────────────────────────────────


def test_submit_clarification_answer_persists_state(
    local_storage, registry_service, workflow_state_service
):
    run_id, readiness = _turn_one(
        local_storage, registry_service, workflow_state_service,
        "I want to design a new antibody-drug conjugate.",
    )
    assert readiness.input_readiness_status == "needs_user_input"
    rid = _target_request_id(readiness)

    svc = ClarificationService(local_storage, registry_service, workflow_state_service)
    state = svc.submit_clarification_answer(
        run_id, [{"request_id": rid, "answer_text": "HER2", "answered_at": "t"}]
    )
    assert state.clarification_answers[0].answer_text == "HER2"
    assert state.clarification_answers[0].target_slot_name == "target_or_antigen"
    assert rid in state.resolved_request_ids
    assert rid not in state.unresolved_request_ids
    assert state.revision_status == "submitted"
    assert state.revision_id.startswith("clarification_state_")
    assert state.submission_fingerprint.startswith("clarification_submission_")
    assert "HER2" not in state.submission_fingerprint
    assert state.output_structured_query_id is None
    assert state.output_input_readiness_status_id is None
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
    with pytest.raises(
        ClarificationRequestError, match="^clarification_request_invalid$"
    ):
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
    with pytest.raises(
        ClarificationRequestError, match="^clarification_request_invalid$"
    ):
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
        with pytest.raises(
            ClarificationRequestError, match="^clarification_source_not_ready$"
        ):
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
    with pytest.raises(
        ClarificationRequestError, match="^clarification_request_invalid$"
    ):
        svc.submit_clarification_answer(run_id, [])


# ── C. next-turn context ───────────────────────────────────────────────────────


def test_same_run_effective_input_preserves_query_and_carries_context(
    local_storage, registry_service, workflow_state_service
):
    run_id, readiness = _turn_one(
        local_storage, registry_service, workflow_state_service,
        "I want to design a new antibody-drug conjugate.",
    )
    rid = _target_request_id(readiness)
    svc = ClarificationService(local_storage, registry_service, workflow_state_service)
    original = local_storage.read_json(
        local_storage.run_key(run_id, "inputs/raw_request_record.json")
    )
    supervisor = _CapturingSupervisor()
    result = svc.submit_and_reparse(
        run_id,
        [{"request_id": rid, "answer_text": "HER2", "answered_at": "t"}],
        supervisor,
    )
    effective = supervisor.inputs[0]
    assert result.run_id == run_id
    assert effective["raw_user_query"].startswith(
        "I want to design a new antibody-drug conjugate."
    )
    assert effective["raw_user_query"] != "HER2"
    ctx = effective["user_provided_context"]
    assert ctx["previous_task_intent"]["primary_intent"] == "new_adc_design"
    assert ctx["previous_missing_slots"]
    assert ctx["previous_clarification_requests"]
    answers = ctx["clarification_answers"]
    assert answers[0]["answer_text"] == "HER2"
    assert answers[0]["slot_name"] == "target_or_antigen"
    assert local_storage.read_json(
        local_storage.run_key(run_id, "inputs/raw_request_record.json")
    ) == original


def test_same_run_effective_input_does_not_clobber_context_fields(
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
    supervisor = _CapturingSupervisor()
    svc.submit_and_reparse(
        rec.run_id,
        [{"request_id": rid, "answer_text": "HER2", "answered_at": "t"}],
        supervisor,
    )
    assert supervisor.inputs[0]["user_provided_context"]["constraints_text"] == (
        "DAR<=4"
    )


# ── D. Step 2/3 loop behavior ──────────────────────────────────────────────────


def test_clarification_loop_resolves_target_blocking(
    local_storage, registry_service, workflow_state_service
):
    run_id, readiness = _turn_one(
        local_storage, registry_service, workflow_state_service,
        "I want to design a new antibody-drug conjugate.",
    )
    assert readiness.input_readiness_status == "needs_user_input"
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


def test_step2_step3_reparse_preserves_old_bodies_in_same_run_history(
    local_storage, registry_service, workflow_state_service
):
    run_id, readiness = _turn_one(
        local_storage, registry_service, workflow_state_service,
        "I want to design a new antibody-drug conjugate.",
    )
    orig_sq = local_storage.read_json(
        local_storage.run_key(run_id, "inputs/structured_query.json")
    )
    orig_readiness = local_storage.read_json(
        local_storage.run_key(run_id, "inputs/input_readiness_status.json")
    )
    rid = _target_request_id(readiness)
    svc = ClarificationService(local_storage, registry_service, workflow_state_service)
    result = svc.submit_and_reparse(
        run_id, [{"request_id": rid, "answer_text": "HER2", "answered_at": "t"}], _supervisor()
    )
    assert result.run_id == run_id
    after = local_storage.read_json(
        local_storage.run_key(run_id, "inputs/structured_query.json")
    )
    assert after["artifact_id"] != orig_sq["artifact_id"]
    assert local_storage.read_json(
        local_storage.run_key(
            run_id,
            "inputs/history/structured_query",
            f"{orig_sq['artifact_id']}.json",
        )
    ) == orig_sq
    assert local_storage.read_json(
        local_storage.run_key(
            run_id,
            "inputs/history/input_readiness_status",
            f"{orig_readiness['artifact_id']}.json",
        )
    ) == orig_readiness


def test_legacy_missing_history_is_materialized_before_revision_and_recovery(
    local_storage, registry_service, workflow_state_service, monkeypatch
):
    run_id, readiness = _turn_one(
        local_storage,
        registry_service,
        workflow_state_service,
        "Analyze the structure of HER2 and report the binding interface.",
    )
    request = next(
        item
        for item in readiness.clarification_requests
        if item.slot_name == "structure_or_sequence"
    )
    structured_body = local_storage.read_json(
        local_storage.run_key(run_id, "inputs/structured_query.json")
    )
    readiness_body = local_storage.read_json(
        local_storage.run_key(run_id, "inputs/input_readiness_status.json")
    )
    structured_history = local_storage.run_key(
        run_id,
        "inputs/history/structured_query",
        f"{structured_body['artifact_id']}.json",
    )
    readiness_history = local_storage.run_key(
        run_id,
        "inputs/history/input_readiness_status",
        f"{readiness_body['artifact_id']}.json",
    )
    assert local_storage.delete(structured_history)
    assert local_storage.delete(readiness_history)

    supervisor = _CapturingSupervisor()
    original_check = InputReadinessService.check
    checks = 0

    def fail_once(service, checked_run_id):
        nonlocal checks
        checks += 1
        if checks == 1:
            raise RuntimeError("test-only-step3-failure")
        return original_check(service, checked_run_id)

    monkeypatch.setattr(InputReadinessService, "check", fail_once)
    payload = [
        {
            "request_id": request.request_id,
            "answer_text": "Use PDB 1N8Z.",
            "answered_at": "t",
        }
    ]
    service = ClarificationService(
        local_storage, registry_service, workflow_state_service
    )
    with pytest.raises(
        ClarificationReparseError, match="clarification_reparse_failed"
    ):
        service.submit_and_reparse(run_id, payload, supervisor)
    revision_id = registry_service.get(
        run_id
    ).active_artifacts.clarification_state_id
    failed = local_storage.read_json(
        _revision_key(local_storage, run_id, revision_id)
    )
    assert failed["failure_code"] == "clarification_step3_failed"
    assert local_storage.read_json(structured_history) == structured_body
    assert local_storage.read_json(readiness_history) == readiness_body
    assert len(supervisor.inputs) == 1

    recovered = ClarificationService(
        local_storage, registry_service, workflow_state_service
    ).submit_and_reparse(run_id, payload, supervisor)

    assert recovered.state.revision_id == revision_id
    assert recovered.state.revision_status == "completed"
    assert len(supervisor.inputs) == 1
    assert checks == 2
    assert len(
        local_storage.list_prefix(
            local_storage.run_key(run_id, "inputs/history/structured_query")
        )
    ) == 2
    assert len(
        local_storage.list_prefix(
            local_storage.run_key(
                run_id, "inputs/history/input_readiness_status"
            )
        )
    ) == 2


def test_existing_source_history_is_never_overwritten_when_content_differs(
    local_storage, registry_service, workflow_state_service
):
    run_id, readiness = _turn_one(
        local_storage,
        registry_service,
        workflow_state_service,
        "I want to design a new antibody-drug conjugate.",
    )
    structured_id = registry_service.get(
        run_id
    ).active_artifacts.structured_query_id
    history_key = local_storage.run_key(
        run_id,
        "inputs/history/structured_query",
        f"{structured_id}.json",
    )
    tampered = local_storage.read_json(history_key)
    tampered["canonical_query"] = "different-history-content"
    local_storage.write_json(history_key, tampered)
    before = local_storage.read_bytes(history_key)
    request_id = _target_request_id(readiness)

    with pytest.raises(
        ClarificationConflictError, match="^structured_query_history_invalid$"
    ):
        ClarificationService(
            local_storage, registry_service, workflow_state_service
        ).submit_clarification_answer(
            run_id,
            [
                {
                    "request_id": request_id,
                    "answer_text": "HER2",
                    "answered_at": "t",
                }
            ],
        )

    assert local_storage.read_bytes(history_key) == before
    assert registry_service.get(
        run_id
    ).active_artifacts.clarification_state_id is None


class _CapturingProvider:
    name = "test-only-capturing-mock"
    model = "test-only"

    def __init__(self):
        self.inner = MockLLMProvider()
        self.calls: list[dict] = []

    def generate_json(self, prompt, *, schema, system=None):
        self.calls.append(
            json.loads(
                json.dumps(
                    {"prompt": prompt, "schema": schema, "system": system}
                )
            )
        )
        return self.inner.generate_json(prompt, schema=schema, system=system)


def test_long_sequence_clarification_enters_step2_and_becomes_target_sequence(
    local_storage,
    registry_service,
    workflow_state_service,
):
    raw_value = "ACDEFGHIKLMNPQRSTVWY" * 30
    run_id, readiness = _turn_one(
        local_storage,
        registry_service,
        workflow_state_service,
        "Analyze the structure of HER2 and report the binding interface.",
    )
    request = next(
        item
        for item in readiness.clarification_requests
        if item.slot_name == "structure_or_sequence"
    )
    class _ExplicitTargetSequenceProvider(_CapturingProvider):
        """Test-only role fixture; not evidence of live LLM recognition."""

        def generate_json(self, prompt, *, schema, system=None):
            result = super().generate_json(prompt, schema=schema, system=system)
            result["referenced_inputs"] = [
                {"id_type": "target_sequence", "value": raw_value, "source": "user"}
            ]
            result["missing_slots"] = [
                slot
                for slot in result.get("missing_slots") or []
                if slot.get("slot_name") != "structure_or_sequence"
            ]
            result["response"] = None
            return result

    provider = _ExplicitTargetSequenceProvider()
    supervisor = SupervisorAgent(llm=provider)
    service = ClarificationService(
        local_storage, registry_service, workflow_state_service
    )
    payload = [
        {
            "request_id": request.request_id,
            "answer_text": raw_value,
            "answered_at": "t",
        }
    ]
    result = service.submit_and_reparse(run_id, payload, supervisor)

    assert len(provider.calls) == 1
    provider_text = json.dumps(provider.calls, sort_keys=True)
    assert raw_value in provider_text
    answer = result.state.clarification_answers[0]
    assert answer.answer_text == raw_value
    assert result.input_readiness_status is not None
    assert result.input_readiness_status.input_readiness_status == "ready"
    assert len(
        local_storage.list_prefix(local_storage.run_key(run_id, "clarification"))
    ) == 1
    structured = local_storage.read_json(
        local_storage.run_key(run_id, "inputs/structured_query.json")
    )
    assert {
        "id_type": "target_sequence",
        "value": raw_value,
        "source": "user",
    } in structured["referenced_inputs"]
    assert not any(
        "clarification_input" in key
        for key in local_storage.list_prefix(local_storage.run_key(run_id))
    )

    files_before = set(local_storage.list_prefix(local_storage.run_key(run_id)))
    replay = ClarificationService(
        local_storage, registry_service, workflow_state_service
    ).submit_and_reparse(run_id, payload, supervisor)
    assert replay.state.revision_id == result.state.revision_id
    assert len(provider.calls) == 1
    assert set(local_storage.list_prefix(local_storage.run_key(run_id))) == files_before
    all_keys = local_storage.list_prefix(local_storage.run_key(run_id))
    assert not any("checkpoint" in key or "tool_call_records" in key for key in all_keys)
    active = registry_service.get(run_id).active_artifacts
    assert active.worker_routing_plan_id is None
    assert active.candidate_context_table_id is None


@pytest.mark.parametrize("short_answer", ["1N8Z", "HER2", "MMAE"])
def test_short_clarification_answers_remain_visible_to_step2(
    local_storage,
    registry_service,
    workflow_state_service,
    short_answer,
):
    run_id, readiness = _turn_one(
        local_storage,
        registry_service,
        workflow_state_service,
        "Analyze the structure of an ADC candidate and report the interface.",
    )
    request = next(
        item
        for item in readiness.clarification_requests
        if not item.resolved and item.severity in {"blocking", "warning"}
    )
    provider = _CapturingProvider()
    ClarificationService(
        local_storage, registry_service, workflow_state_service
    ).submit_and_reparse(
        run_id,
        [
            {
                "request_id": request.request_id,
                "answer_text": short_answer,
                "answered_at": "t",
            }
        ],
        SupervisorAgent(llm=provider),
    )
    assert short_answer in json.dumps(provider.calls)


def test_clarification_revision_preserves_same_run_session_and_raw_identity(
    local_storage, registry_service, workflow_state_service
):
    run_id, readiness = _turn_one(
        local_storage,
        registry_service,
        workflow_state_service,
        "I want to design a new antibody-drug conjugate.",
    )
    original = local_storage.read_json(
        local_storage.run_key(run_id, "inputs/raw_request_record.json")
    )
    result = ClarificationService(
        local_storage, registry_service, workflow_state_service
    ).submit_and_reparse(
        run_id,
        [
            {
                "request_id": _target_request_id(readiness),
                "answer_text": "HER2",
                "answered_at": "t",
            }
        ],
        _supervisor(),
    )
    current = local_storage.read_json(
        local_storage.run_key(run_id, "inputs/raw_request_record.json")
    )
    assert result.run_id == run_id
    assert current == original
    assert registry_service.get(run_id).active_artifacts.raw_request_record_id == (
        original["artifact_id"]
    )


def test_clarification_rejects_tampered_raw_identity_before_revision_write(
    local_storage, registry_service, workflow_state_service
):
    run_id, readiness = _turn_one(
        local_storage,
        registry_service,
        workflow_state_service,
        "I want to design a new antibody-drug conjugate.",
    )
    key = local_storage.run_key(run_id, "inputs/raw_request_record.json")
    body = local_storage.read_json(key)
    body["run_id"] = "sk-live-tampered-run"
    local_storage.write_json(key, body)

    with pytest.raises(
        OrchestratorReadinessError,
        match="^raw_request_record_identity_mismatch$",
    ) as caught:
        ClarificationService(
            local_storage, registry_service, workflow_state_service
        ).submit_clarification_answer(
            run_id,
            [
                {
                    "request_id": _target_request_id(readiness),
                    "answer_text": "HER2",
                    "answered_at": "t",
                }
            ],
        )

    assert "sk-live" not in repr(caught.value)
    assert len(local_storage.list_prefix(local_storage.run_key(run_id, "clarification"))) == 0


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


def test_effective_context_includes_previous_canonical_query(
    local_storage, registry_service, workflow_state_service
):
    run_id, readiness = _turn_one(
        local_storage, registry_service, workflow_state_service,
        "I want to design a new antibody-drug conjugate.",
    )
    rid = _target_request_id(readiness)
    svc = ClarificationService(local_storage, registry_service, workflow_state_service)
    supervisor = _CapturingSupervisor()
    svc.submit_and_reparse(
        run_id,
        [{"request_id": rid, "answer_text": "HER2", "answered_at": "t"}],
        supervisor,
    )
    effective = supervisor.inputs[0]
    ctx = effective["user_provided_context"]
    assert ctx.get("previous_canonical_query")
    assert "antibody-drug conjugate" in ctx["previous_canonical_query"].lower()
    # Original raw query preserved (auditable), not replaced by a marker.
    assert "I want to design a new antibody-drug conjugate." in effective[
        "raw_user_query"
    ]


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


def test_fresh_service_replays_completed_revision_from_persisted_storage(
    local_storage, registry_service, workflow_state_service
):
    run_id, readiness = _turn_one(
        local_storage,
        registry_service,
        workflow_state_service,
        "Analyze the structure of HER2 and report the binding interface.",
    )
    request = next(
        item
        for item in readiness.clarification_requests
        if item.slot_name == "structure_or_sequence"
    )
    first_supervisor = _CapturingSupervisor()
    payload = [
        {
            "request_id": request.request_id,
            "answer_text": "Use PDB 1N8Z.",
            "answered_at": "2026-07-15T00:00:00Z",
        }
    ]
    first = ClarificationService(
        local_storage, registry_service, workflow_state_service
    ).submit_and_reparse(run_id, payload, first_supervisor)
    files = set(local_storage.list_prefix(local_storage.prefix))

    replay_supervisor = _CapturingSupervisor()
    payload[0]["answered_at"] = "2099-01-01T00:00:00Z"
    replay = ClarificationService(
        local_storage, registry_service, workflow_state_service
    ).submit_and_reparse(run_id, payload, replay_supervisor)

    assert replay.state.revision_id == first.state.revision_id
    assert replay.input_readiness_status == first.input_readiness_status
    assert replay_supervisor.inputs == []
    assert set(local_storage.list_prefix(local_storage.prefix)) == files
