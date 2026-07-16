"""Same-run clarification HTTP loop over production Step 1/2/3 services.

MockLLMProvider and small failing wrappers isolate external network access.
They are not live LLM, MCP, ToolUniverse, worker, or biomedical-tool evidence.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

import app.api.step_03_input_readiness_api as clarification_api
import app.api.step_02_structured_query_api as step2_api
import app.deps as deps
from app.a2a.orchestrator_readiness import require_ready_input_readiness
from app.graph.orchestrator_execution_graph import execution_graph_config
from app.llm.provider import MockLLMProvider
from app.main import app
from app.services.input_readiness_service import InputReadinessService
from app.settings import get_settings
from app.utils.ids import new_file_id


@pytest.fixture
def client(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("STORAGE_MODE", "local")
    monkeypatch.setenv("LOCAL_STORAGE_ROOT", str(tmp_path / "store"))
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    monkeypatch.delenv("LANGGRAPH_CHECKPOINT_DATABASE_URL", raising=False)
    cached = (
        get_settings,
        deps.get_storage,
        deps.get_registry_service,
        deps.get_workflow_state_service,
        deps.get_llm_provider,
        deps.get_worker_discovery_service,
        deps.get_orchestrator_routing_service,
    )
    for function in cached:
        function.cache_clear()
    with TestClient(app) as test_client:
        yield test_client
    for function in cached:
        function.cache_clear()


def _first_round(client: TestClient, query: str):
    created = client.post("/runs", json={"raw_user_query": query})
    assert created.status_code == 200
    source = created.json()
    step2 = client.post(f"/runs/{source['run_id']}/steps/2/execute")
    step3 = client.post(f"/runs/{source['run_id']}/steps/3/execute")
    assert step2.status_code == step3.status_code == 200
    readiness = step3.json()
    assert readiness["input_readiness_status"] == "needs_user_input"
    return source, readiness


def _answer_payload(
    readiness: dict[str, Any],
    *,
    slot_name: str | None = None,
    answer_text: str = "test-only clarification answer",
) -> dict[str, Any]:
    request = next(
        item
        for item in readiness["clarification_requests"]
        if slot_name is None or item["slot_name"] == slot_name
    )
    return {
        "answers": [
            {
                "request_id": request["request_id"],
                "answer_text": answer_text,
                "answered_at": "2026-07-15T00:00:00Z",
            }
        ]
    }


def _stored_files() -> set[str]:
    storage = deps.get_storage()
    return set(storage.list_prefix(storage.prefix))


def _run_ids() -> set[str]:
    run_marker = f"{deps.get_storage().prefix}/runs/"
    return {
        name.split(run_marker, 1)[1].split("/", 1)[0]
        for name in _stored_files()
        if run_marker in name
    }


def _assert_no_step4_side_effects(run_id: str):
    active = deps.get_registry_service().get(run_id).active_artifacts
    assert active.worker_discovery_snapshot_id is None
    assert active.worker_routing_plan_id is None
    assert active.worker_routing_plan_control_id is None
    assert active.candidate_context_table_id is None
    assert active.structured_liability_summary_id is None
    assert active.prepared_structure_input_package_id is None
    stored_names = _stored_files()
    assert not any("worker_routing_plan" in name for name in stored_names)
    assert not any("tool_call_records" in name for name in stored_names)
    assert app.state.orchestrator_checkpoint_runtime is None


@pytest.mark.parametrize(
    ("query", "slot_name", "answer"),
    [
        (
            "Analyze the structure of HER2 and report the binding interface.",
            "structure_or_sequence",
            "Use PDB 1N8Z.",
        ),
        (
            "Optimize and generate a protein sequence for EGFR using masked protein generation.",
            "prompt_sequence",
            "Use prompt_sequence ACDEFGHIK_LMNPQRST.",
        ),
    ],
)
def test_clarification_http_round_reparses_step2_step3_in_same_run(
    client: TestClient, query: str, slot_name: str, answer: str
):
    source, readiness = _first_round(client, query)
    run_id = source["run_id"]
    registry = deps.get_registry_service()
    before = registry.get(run_id)
    raw_id = before.active_artifacts.raw_request_record_id
    structured_id = before.active_artifacts.structured_query_id
    readiness_id = before.active_artifacts.input_readiness_status_id
    raw_before = deps.get_storage().read_bytes(
        deps.get_storage().run_key(run_id, "inputs/raw_request_record.json")
    )

    response = client.post(
        f"/runs/{run_id}/steps/3/clarifications",
        json=_answer_payload(
            readiness, slot_name=slot_name, answer_text=answer
        ),
    )

    assert response.status_code == 200, response.text
    compact = response.json()
    after = registry.get(run_id)
    assert set(compact) == {
        "run_id",
        "session_id",
        "clarification_revision_id",
        "input_readiness_status",
        "response",
        "clarification_requests",
    }
    assert compact["run_id"] == run_id
    assert compact["session_id"] == source["session_id"]
    assert compact["clarification_revision_id"] == (
        after.active_artifacts.clarification_state_id
    )
    assert compact["input_readiness_status"] == "ready"
    assert compact["clarification_requests"] == []
    assert _run_ids() == {run_id}
    assert after.active_artifacts.raw_request_record_id == raw_id
    assert after.active_artifacts.structured_query_id != structured_id
    assert after.active_artifacts.input_readiness_status_id != readiness_id
    assert deps.get_storage().read_bytes(
        deps.get_storage().run_key(run_id, "inputs/raw_request_record.json")
    ) == raw_before
    assert require_ready_input_readiness(
        run_id=run_id,
        registry=registry,
        storage=deps.get_storage(),
    ).input_readiness_status == "ready"
    _assert_no_step4_side_effects(run_id)


def test_multiple_missing_slot_rounds_stay_in_one_run(client: TestClient):
    source, first = _first_round(
        client, "I want to design a new antibody-drug conjugate."
    )
    run_id = source["run_id"]
    first_response = client.post(
        f"/runs/{run_id}/steps/3/clarifications",
        json=_answer_payload(
            first, slot_name="target_or_antigen", answer_text="HER2"
        ),
    )
    assert first_response.status_code == 200
    second = first_response.json()
    assert second["run_id"] == run_id
    assert second["input_readiness_status"] == "needs_user_input"

    answers = []
    values = {"antibody": "trastuzumab", "payload": "MMAE"}
    for request in second["clarification_requests"]:
        answers.append(
            {
                "request_id": request["request_id"],
                "answer_text": values.get(request["slot_name"], "provided"),
                "answered_at": "2026-07-15T00:01:00Z",
            }
        )
    final_response = client.post(
        f"/runs/{run_id}/steps/3/clarifications",
        json={"answers": answers},
    )
    assert final_response.status_code == 200, final_response.text
    final = final_response.json()
    assert final["run_id"] == run_id
    assert final["session_id"] == source["session_id"]
    assert final["input_readiness_status"] == "ready"
    assert final["clarification_revision_id"] != second[
        "clarification_revision_id"
    ]
    revision_files = deps.get_storage().list_prefix(
        deps.get_storage().run_key(run_id, "clarification")
    )
    assert len(revision_files) == 2
    assert sorted(
        deps.get_storage().read_json(key)["revision_number"]
        for key in revision_files
    ) == [1, 2]
    assert _run_ids() == {run_id}
    _assert_no_step4_side_effects(run_id)


def test_completed_submission_replay_is_persisted_and_exactly_idempotent(
    client: TestClient, monkeypatch
):
    source, readiness = _first_round(
        client, "Analyze the structure of HER2 and report the binding interface."
    )

    class _CountingProvider:
        name = "test-only-counting-mock"
        model = "test-only"

        def __init__(self):
            self.inner = MockLLMProvider()
            self.call_count = 0

        def generate_json(self, *args, **kwargs):
            self.call_count += 1
            return self.inner.generate_json(*args, **kwargs)

    provider = _CountingProvider()
    monkeypatch.setattr(clarification_api, "get_llm_provider", lambda: provider)
    payload = _answer_payload(
        readiness,
        slot_name="structure_or_sequence",
        answer_text="Use PDB 1N8Z.",
    )
    url = f"/runs/{source['run_id']}/steps/3/clarifications"
    first = client.post(url, json=payload)
    assert first.status_code == 200
    files_after_first = _stored_files()
    calls_after_first = provider.call_count

    replay_payload = {
        "answers": [
            {
                **payload["answers"][0],
                "answered_at": "2099-01-01T00:00:00Z",
            }
        ]
    }
    replay = client.post(url, json=replay_payload)

    assert replay.status_code == 200
    assert replay.json() == first.json()
    assert provider.call_count == calls_after_first == 1
    assert _stored_files() == files_after_first
    assert _run_ids() == {source["run_id"]}

    changed_answer = {
        "answers": [
            {
                **payload["answers"][0],
                "answer_text": "Use PDB 2ABC.",
            }
        ]
    }
    resolved = client.post(url, json=changed_answer)
    assert resolved.status_code == 422
    assert resolved.json() == {"detail": "clarification_request_invalid"}
    assert provider.call_count == 1
    assert _stored_files() == files_after_first


def test_masked_prompt_body_is_sent_to_step2_provider_without_typed_ref_artifact(
    client: TestClient, monkeypatch
):
    source, readiness = _first_round(
        client,
        "Optimize and generate a protein sequence for EGFR using masked protein generation.",
    )
    raw_prompt = "A" * 400 + "_" + "C" * 401

    class _CapturingProvider:
        name = "test-only-capturing-mock"
        model = "test-only"

        def __init__(self):
            self.inner = MockLLMProvider()
            self.calls: list[str] = []

        def generate_json(self, prompt, *, schema, system=None):
            self.calls.append(f"{prompt!r}{schema!r}{system!r}")
            return self.inner.generate_json(prompt, schema=schema, system=system)

    provider = _CapturingProvider()
    monkeypatch.setattr(clarification_api, "get_llm_provider", lambda: provider)
    response = client.post(
        f"/runs/{source['run_id']}/steps/3/clarifications",
        json=_answer_payload(
            readiness,
            slot_name="prompt_sequence",
            answer_text=f"Use as prompt_sequence: {raw_prompt}",
        ),
    )

    assert response.status_code == 200, response.text
    assert response.json()["input_readiness_status"] == "ready"
    assert len(provider.calls) == 1
    assert raw_prompt in provider.calls[0]
    assert raw_prompt not in response.text
    revision_id = response.json()["clarification_revision_id"]
    revision = deps.get_storage().read_json(
        deps.get_storage().run_key(
            source["run_id"], "clarification", f"{revision_id}.json"
        )
    )
    answer = revision["clarification_answers"][0]
    assert answer["answer_text"] == f"Use as prompt_sequence: {raw_prompt}"
    assert not any(
        "clarification_input" in key for key in _stored_files()
    )
    _assert_no_step4_side_effects(source["run_id"])


def test_pdb_body_cannot_satisfy_prompt_sequence_slot(client: TestClient):
    source, readiness = _first_round(
        client,
        "Optimize and generate a protein sequence for EGFR using masked protein generation.",
    )
    pdb_body = (
        "HEADER TEST\n"
        "ATOM      1  CA  ALA A   1      11.000  12.000  13.000\nEND"
    )
    response = client.post(
        f"/runs/{source['run_id']}/steps/3/clarifications",
        json=_answer_payload(
            readiness,
            slot_name="prompt_sequence",
            answer_text=pdb_body,
        ),
    )
    assert response.status_code == 200, response.text
    compact = response.json()
    assert compact["input_readiness_status"] == "needs_user_input"
    assert any(
        request["slot_name"] == "prompt_sequence"
        for request in compact["clarification_requests"]
    )
    structured = deps.get_storage().read_json(
        deps.get_storage().run_key(
            source["run_id"], "inputs/structured_query.json"
        )
    )
    assert not any(
        item.get("id_type") == "prompt_sequence"
        for item in structured["referenced_inputs"]
    )
    _assert_no_step4_side_effects(source["run_id"])


def test_invalid_then_valid_target_sequence_clarifications_stay_in_same_run(
    client: TestClient,
    monkeypatch,
):
    invalid = "ACDE?FG"
    valid = "ACDEFGHIKLMNPQRSTVWY"
    source, initial_readiness = _first_round(
        client,
        "Analyze the structure of HER2 and report the binding interface.",
    )
    run_id = source["run_id"]
    session_id = source["session_id"]
    registry = deps.get_registry_service()
    storage = deps.get_storage()
    initial_active = registry.get(run_id).active_artifacts
    raw_id = initial_active.raw_request_record_id
    initial_structured_id = initial_active.structured_query_id
    initial_readiness_id = initial_active.input_readiness_status_id
    raw_key = storage.run_key(run_id, "inputs/raw_request_record.json")
    raw_bytes = storage.read_bytes(raw_key)

    class _SequenceClarificationProvider:
        """Test-only semantic fixture; production same-run services execute."""

        name = "test-only-sequence-clarification"
        model = "test-only"

        def __init__(self):
            self.inner = MockLLMProvider()
            self.call_count = 0

        def generate_json(self, prompt, *, schema, system=None):
            result = self.inner.generate_json(
                prompt, schema=schema, system=system
            )
            value = invalid if self.call_count == 0 else valid
            self.call_count += 1
            result["referenced_inputs"] = [
                {
                    "id_type": "target_sequence",
                    "value": value,
                    "source": "user",
                }
            ]
            result["missing_slots"] = [
                slot
                for slot in result.get("missing_slots") or []
                if slot.get("slot_name") != "structure_or_sequence"
            ]
            result["response"] = None
            return result

    provider = _SequenceClarificationProvider()
    monkeypatch.setattr(clarification_api, "get_llm_provider", lambda: provider)
    first_payload = _answer_payload(
        initial_readiness,
        slot_name="structure_or_sequence",
        answer_text=invalid,
    )
    url = f"/runs/{run_id}/steps/3/clarifications"
    invalid_response = client.post(url, json=first_payload)

    assert invalid_response.status_code == 200, invalid_response.text
    invalid_compact = invalid_response.json()
    assert invalid_compact["run_id"] == run_id
    assert invalid_compact["session_id"] == session_id
    assert invalid_compact["input_readiness_status"] == "needs_user_input"
    assert any(
        request["slot_name"] == "structure_or_sequence"
        for request in invalid_compact["clarification_requests"]
    )
    invalid_active = registry.get(run_id).active_artifacts
    assert invalid_active.raw_request_record_id == raw_id
    assert invalid_active.structured_query_id != initial_structured_id
    assert invalid_active.input_readiness_status_id != initial_readiness_id
    invalid_structured = storage.read_json(
        storage.run_key(run_id, "inputs/structured_query.json")
    )
    assert invalid not in [
        ref.get("value") for ref in invalid_structured["referenced_inputs"]
    ]
    assert invalid not in " ".join(invalid_structured["parse_warnings"])
    assert "dropped invalid target_sequence referenced_input" in (
        invalid_structured["parse_warnings"]
    )
    invalid_slots = [
        slot
        for slot in invalid_structured["missing_slots"]
        if slot["slot_name"] == "structure_or_sequence"
    ]
    assert len(invalid_slots) == 1
    invalid_revision = storage.read_json(
        storage.run_key(
            run_id,
            "clarification",
            f"{invalid_compact['clarification_revision_id']}.json",
        )
    )
    assert invalid_revision["revision_status"] == "completed"
    assert invalid_revision["failure_code"] is None
    files_after_invalid = _stored_files()
    replay = client.post(url, json=first_payload)
    assert replay.status_code == 200
    assert replay.json() == invalid_compact
    assert provider.call_count == 1
    assert _stored_files() == files_after_invalid

    valid_response = client.post(
        url,
        json=_answer_payload(
            invalid_compact,
            slot_name="structure_or_sequence",
            answer_text=valid,
        ),
    )

    assert valid_response.status_code == 200, valid_response.text
    valid_compact = valid_response.json()
    assert valid_compact["run_id"] == run_id
    assert valid_compact["session_id"] == session_id
    assert valid_compact["input_readiness_status"] == "ready"
    assert valid_compact["clarification_requests"] == []
    assert valid_compact["clarification_revision_id"] != invalid_compact[
        "clarification_revision_id"
    ]
    final_active = registry.get(run_id).active_artifacts
    assert final_active.raw_request_record_id == raw_id
    assert final_active.structured_query_id != invalid_active.structured_query_id
    assert (
        final_active.input_readiness_status_id
        != invalid_active.input_readiness_status_id
    )
    final_structured = storage.read_json(
        storage.run_key(run_id, "inputs/structured_query.json")
    )
    assert {
        "id_type": "target_sequence",
        "value": valid,
        "source": "user",
    } in final_structured["referenced_inputs"]
    assert not any(
        slot["slot_name"] == "structure_or_sequence"
        for slot in final_structured["missing_slots"]
    )
    revision_files = storage.list_prefix(
        storage.run_key(run_id, "clarification")
    )
    assert sorted(
        storage.read_json(key)["revision_number"] for key in revision_files
    ) == [1, 2]
    assert provider.call_count == 2
    assert _run_ids() == {run_id}
    assert storage.read_bytes(raw_key) == raw_bytes
    _assert_no_step4_side_effects(run_id)


def test_target_fasta_does_not_erase_antibody_sequence_role_clarification(
    client: TestClient,
    monkeypatch,
):
    """Test-only LLM transport; production Step 1/2/3 services execute."""

    target_file_id = new_file_id()
    generic_antibody_sequence = "EVQLVESGGGLVQPGGSLRLSCAAS"
    heavy_sequence = "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWV"

    class _RoleClarificationProvider:
        name = "test-only-role-clarification"
        model = "test-only"

        def __init__(self):
            self.inner = MockLLMProvider()
            self.call_count = 0

        def generate_json(self, prompt, *, schema, system=None):
            result = self.inner.generate_json(prompt, schema=schema, system=system)
            result["task_intent"] = {
                "task_type": "structure_analysis",
                "primary_intent": "structure_analysis",
                "user_goal_summary": "Analyze target and antibody sequences.",
            }
            result["mentioned_entities"] = {
                "target_or_antigen_text": "HER2",
                "antibody_candidate_text": "test antibody",
            }
            result["referenced_inputs"] = [
                {
                    "id_type": "uploaded_file",
                    "value": target_file_id,
                    "source": "target_sequence",
                },
                (
                    {
                        "id_type": "antibody_sequence_reference",
                        "value": generic_antibody_sequence,
                        "source": "user",
                    }
                    if self.call_count == 0
                    else {
                        "id_type": "antibody_heavy_chain_sequence",
                        "value": heavy_sequence,
                        "source": "user",
                    }
                ),
            ]
            result["missing_slots"] = (
                [
                    {
                        "slot_name": "sequence_role",
                        "slot_category": "sequence",
                        "severity": "blocking",
                        "required_for": ["structure_analysis"],
                        "reason": "The antibody sequence chain role is unresolved.",
                        "suggested_question": (
                            "Is the antibody sequence heavy or light chain?"
                        ),
                        "evidence": "structured_query.referenced_inputs[1]",
                    }
                ]
                if self.call_count == 0
                else []
            )
            result["response"] = (
                "Please identify the antibody chain role."
                if self.call_count == 0
                else None
            )
            self.call_count += 1
            return result

    provider = _RoleClarificationProvider()
    monkeypatch.setattr(step2_api, "get_llm_provider", lambda: provider)
    monkeypatch.setattr(clarification_api, "get_llm_provider", lambda: provider)
    created = client.post(
        "/runs",
        json={
            "raw_user_query": (
                "The uploaded FASTA is the HER2 target sequence. The supplied "
                "antibody sequence still needs a chain role."
            ),
            "uploaded_files": [
                {
                    "file_id": target_file_id,
                    "original_filename": "target.fasta",
                    "storage_path": f"inputs/files/{target_file_id}.fasta",
                    "content_type": "text/x-fasta",
                    "role": "target_sequence",
                }
            ],
        },
    )
    assert created.status_code == 200
    source = created.json()
    run_id = source["run_id"]
    session_id = source["session_id"]
    storage = deps.get_storage()
    registry = deps.get_registry_service()
    raw_key = storage.run_key(run_id, "inputs/raw_request_record.json")
    raw_bytes = storage.read_bytes(raw_key)
    raw_id = registry.get(run_id).active_artifacts.raw_request_record_id

    first_step2 = client.post(f"/runs/{run_id}/steps/2/execute")
    first_step3 = client.post(f"/runs/{run_id}/steps/3/execute")
    assert first_step2.status_code == first_step3.status_code == 200
    first_sq = first_step2.json()
    first_readiness = first_step3.json()
    assert first_sq["referenced_inputs"] == [
        {
            "id_type": "uploaded_file",
            "value": target_file_id,
            "source": "target_sequence",
        },
        {
            "id_type": "antibody_sequence_reference",
            "value": generic_antibody_sequence,
            "source": "user",
        },
    ]
    assert [slot["slot_name"] for slot in first_sq["missing_slots"]] == [
        "sequence_role"
    ]
    assert first_sq["missing_slots"][0]["severity"] == "blocking"
    assert first_readiness["input_readiness_status"] == "needs_user_input"
    assert any(
        request["slot_name"] == "sequence_role"
        for request in first_readiness["clarification_requests"]
    )
    _assert_no_step4_side_effects(run_id)

    clarified = client.post(
        f"/runs/{run_id}/steps/3/clarifications",
        json=_answer_payload(
            first_readiness,
            slot_name="sequence_role",
            answer_text="The supplied antibody sequence is the heavy chain.",
        ),
    )
    assert clarified.status_code == 200, clarified.text
    compact = clarified.json()
    second_sq = storage.read_json(
        storage.run_key(run_id, "inputs/structured_query.json")
    )
    assert compact["run_id"] == run_id
    assert compact["session_id"] == session_id
    assert compact["input_readiness_status"] == "ready"
    assert compact["clarification_requests"] == []
    assert second_sq["referenced_inputs"] == [
        {
            "id_type": "uploaded_file",
            "value": target_file_id,
            "source": "target_sequence",
        },
        {
            "id_type": "antibody_heavy_chain_sequence",
            "value": heavy_sequence,
            "source": "user",
        },
    ]
    assert second_sq["missing_slots"] == []
    assert provider.call_count == 2
    assert _run_ids() == {run_id}
    assert registry.get(run_id).active_artifacts.raw_request_record_id == raw_id
    assert storage.read_bytes(raw_key) == raw_bytes
    _assert_no_step4_side_effects(run_id)


def test_provider_failure_reuses_same_revision_and_recovers_same_run(
    client: TestClient, monkeypatch
):
    source, readiness = _first_round(
        client, "Analyze the structure of HER2 and report the binding interface."
    )
    sentinel = "sensitive-provider-sentinel"

    class _FailOnceProvider:
        name = "test-only-fail-once"
        model = "test-only"

        def __init__(self):
            self.inner = MockLLMProvider()
            self.call_count = 0

        def generate_json(self, *args, **kwargs):
            self.call_count += 1
            if self.call_count == 1:
                raise ValueError(sentinel)
            return self.inner.generate_json(*args, **kwargs)

    provider = _FailOnceProvider()
    monkeypatch.setattr(clarification_api, "get_llm_provider", lambda: provider)
    payload = _answer_payload(
        readiness,
        slot_name="structure_or_sequence",
        answer_text="Use PDB 1N8Z.",
    )
    url = f"/runs/{source['run_id']}/steps/3/clarifications"
    before_runs = _run_ids()
    failed = client.post(url, json=payload)
    assert failed.status_code == 503
    assert failed.json() == {"detail": "clarification_reparse_failed"}
    assert sentinel not in failed.text
    active = deps.get_registry_service().get(
        source["run_id"]
    ).active_artifacts
    revision_id = active.clarification_state_id
    revision_key = deps.get_storage().run_key(
        source["run_id"], "clarification", f"{revision_id}.json"
    )
    failed_revision = deps.get_storage().read_json(revision_key)
    assert failed_revision["revision_status"] == "reparse_failed"
    assert failed_revision["failure_code"] == "clarification_step2_failed"
    assert failed_revision["output_structured_query_id"] is None
    assert _run_ids() == before_runs == {source["run_id"]}

    recovered = client.post(url, json=payload)

    assert recovered.status_code == 200, recovered.text
    assert recovered.json()["run_id"] == source["run_id"]
    assert recovered.json()["clarification_revision_id"] == revision_id
    assert provider.call_count == 2
    completed = deps.get_storage().read_json(revision_key)
    assert completed["revision_status"] == "completed"
    assert completed["failure_code"] is None
    assert completed["output_structured_query_id"] is not None
    assert completed["output_input_readiness_status_id"] is not None
    assert _run_ids() == {source["run_id"]}
    _assert_no_step4_side_effects(source["run_id"])


def test_step3_failure_recovery_does_not_repeat_step2_llm(
    client: TestClient, monkeypatch
):
    source, readiness = _first_round(
        client, "Analyze the structure of HER2 and report the binding interface."
    )

    class _CountingProvider:
        name = "test-only-counting-mock"
        model = "test-only"

        def __init__(self):
            self.inner = MockLLMProvider()
            self.call_count = 0

        def generate_json(self, *args, **kwargs):
            self.call_count += 1
            return self.inner.generate_json(*args, **kwargs)

    provider = _CountingProvider()
    monkeypatch.setattr(clarification_api, "get_llm_provider", lambda: provider)
    original_check = InputReadinessService.check
    readiness_calls = 0

    def _fail_once(service, run_id):
        nonlocal readiness_calls
        readiness_calls += 1
        if readiness_calls == 1:
            raise ValueError("sensitive-step3-sentinel")
        return original_check(service, run_id)

    monkeypatch.setattr(InputReadinessService, "check", _fail_once)
    payload = _answer_payload(
        readiness,
        slot_name="structure_or_sequence",
        answer_text="Use PDB 1N8Z.",
    )
    url = f"/runs/{source['run_id']}/steps/3/clarifications"
    failed = client.post(url, json=payload)
    assert failed.status_code == 503
    revision_id = deps.get_registry_service().get(
        source["run_id"]
    ).active_artifacts.clarification_state_id
    revision_key = deps.get_storage().run_key(
        source["run_id"], "clarification", f"{revision_id}.json"
    )
    failed_revision = deps.get_storage().read_json(revision_key)
    assert failed_revision["failure_code"] == "clarification_step3_failed"
    assert failed_revision["output_structured_query_id"] is not None
    assert failed_revision["output_input_readiness_status_id"] is None

    recovered = client.post(url, json=payload)

    assert recovered.status_code == 200
    assert recovered.json()["clarification_revision_id"] == revision_id
    assert provider.call_count == 1
    assert readiness_calls == 2
    assert deps.get_storage().read_json(revision_key)["revision_status"] == (
        "completed"
    )


def test_different_submission_on_failed_source_conflicts_without_new_effect(
    client: TestClient, monkeypatch
):
    source, readiness = _first_round(
        client, "Analyze the structure of HER2 and report the binding interface."
    )

    class _AlwaysFailProvider:
        name = "test-only-failing-provider"
        model = "test-only"
        call_count = 0

        def generate_json(self, *_args, **_kwargs):
            self.call_count += 1
            raise ValueError("sensitive-provider-sentinel")

    provider = _AlwaysFailProvider()
    monkeypatch.setattr(clarification_api, "get_llm_provider", lambda: provider)
    url = f"/runs/{source['run_id']}/steps/3/clarifications"
    first = client.post(
        url,
        json=_answer_payload(
            readiness,
            slot_name="structure_or_sequence",
            answer_text="Use PDB 1N8Z.",
        ),
    )
    assert first.status_code == 503
    before = _stored_files()

    conflict = client.post(
        url,
        json=_answer_payload(
            readiness,
            slot_name="structure_or_sequence",
            answer_text="Use PDB 2ABC.",
        ),
    )

    assert conflict.status_code == 409
    assert conflict.json() == {"detail": "clarification_submission_conflict"}
    assert provider.call_count == 1
    assert _stored_files() == before


@pytest.mark.parametrize(
    ("artifact_name", "mutate", "expected_code"),
    [
        (
            "raw_request_record",
            lambda body: body.update(
                {"artifact_id": "raw_request_record_20990101_deadbeef"}
            ),
            "raw_request_record_identity_mismatch",
        ),
        (
            "input_readiness_status",
            lambda body: body.update({"run_id": "run_20990101_deadbeef"}),
            "input_readiness_status_identity_mismatch",
        ),
        (
            "input_readiness_status",
            lambda body: body["source_refs"].update(
                {"structured_query_id": "structured_query_20990101_deadbeef"}
            ),
            "input_readiness_status_source_mismatch",
        ),
        (
            "input_readiness_status",
            lambda body: body["clarification_requests"][0].update(
                {"severity": "invalid_severity"}
            ),
            "input_readiness_status_schema_invalid",
        ),
        (
            "structured_query",
            lambda body: body.update(
                {"artifact_id": "structured_query_20990101_deadbeef"}
            ),
            "structured_query_identity_mismatch",
        ),
        (
            "structured_query",
            lambda body: body["source_raw_request_ref"].update(
                {"raw_request_record_id": "raw_request_record_20990101_deadbeef"}
            ),
            "input_readiness_status_source_mismatch",
        ),
    ],
)
def test_clarification_rejects_tampered_source_before_any_new_file(
    client: TestClient, artifact_name: str, mutate, expected_code: str
):
    source, readiness = _first_round(
        client, "I want to design a new antibody-drug conjugate."
    )
    storage = deps.get_storage()
    key = storage.run_key(source["run_id"], "inputs", f"{artifact_name}.json")
    body = storage.read_json(key)
    mutate(body)
    storage.write_json(key, body)
    before = _stored_files()

    response = client.post(
        f"/runs/{source['run_id']}/steps/3/clarifications",
        json=_answer_payload(readiness),
    )

    assert response.status_code == 409
    assert response.json() == {"detail": expected_code}
    assert _stored_files() == before
    assert deps.get_registry_service().get(
        source["run_id"]
    ).active_artifacts.clarification_state_id is None


def test_stale_or_resolved_request_and_routed_run_fail_closed(
    client: TestClient
):
    source, readiness = _first_round(
        client, "I want to design a new antibody-drug conjugate."
    )
    run_id = source["run_id"]
    before = _stored_files()
    stale = client.post(
        f"/runs/{run_id}/steps/3/clarifications",
        json={
            "answers": [
                {
                    "request_id": "clr_stale_request",
                    "answer_text": "HER2",
                    "answered_at": "2026-07-15T00:00:00Z",
                }
            ]
        },
    )
    assert stale.status_code == 422
    assert stale.json() == {"detail": "clarification_request_invalid"}
    assert _stored_files() == before

    deps.get_registry_service().update_active(
        run_id, worker_routing_plan_id="worker_routing_plan_testonly"
    )
    before_routed = _stored_files()
    routed = client.post(
        f"/runs/{run_id}/steps/3/clarifications",
        json=_answer_payload(readiness),
    )
    assert routed.status_code == 409
    assert routed.json() == {"detail": "clarification_run_already_routed"}
    assert _stored_files() == before_routed


def test_clarification_body_cannot_override_run_or_session(client: TestClient):
    source, readiness = _first_round(
        client, "I want to design a new antibody-drug conjugate."
    )
    payload = _answer_payload(
        readiness, slot_name="target_or_antigen", answer_text="HER2"
    )
    payload["run_id"] = "run_20990101_deadbeef"
    payload["session_id"] = "sess_deadbeefdeadbeef"
    before = _stored_files()

    response = client.post(
        f"/runs/{source['run_id']}/steps/3/clarifications", json=payload
    )

    assert response.status_code == 422
    assert _stored_files() == before


def test_new_task_reuses_session_but_creates_independent_run(client: TestClient):
    first = client.post("/runs", json={"raw_user_query": "first task"}).json()
    second = client.post(
        "/runs",
        json={
            "raw_user_query": "second independent task",
            "session_id": first["session_id"],
        },
    ).json()
    assert second["run_id"] != first["run_id"]
    assert second["session_id"] == first["session_id"]
    first_registry = deps.get_registry_service().get(first["run_id"])
    second_registry = deps.get_registry_service().get(second["run_id"])
    assert first_registry.run_artifact_registry_id != (
        second_registry.run_artifact_registry_id
    )
    assert first_registry.active_artifacts.raw_request_record_id != (
        second_registry.active_artifacts.raw_request_record_id
    )
    assert execution_graph_config(first["run_id"]) != execution_graph_config(
        second["run_id"]
    )
    assert _run_ids() == {first["run_id"], second["run_id"]}
