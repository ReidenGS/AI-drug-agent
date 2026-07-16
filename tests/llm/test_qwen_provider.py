"""QwenProvider JSON generation tests.

These tests stub the provider's SDK-facing call and never hit DashScope.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from app.a2a.orchestrator_routing_prompt import (
    ORCHESTRATOR_ROUTING_SYSTEM_PROMPT,
    ORCHESTRATOR_ROUTING_USER_TASK,
)
from app.llm.qwen_provider import QwenProvider, QwenProviderError


def _provider_with_responses(responses: list[Any]) -> QwenProvider:
    provider = QwenProvider(api_key="qwen-fake-key", max_retries=2)
    calls = iter(responses)

    def _fake_generate_content(prompt: str, *, system: str | None = None) -> Any:
        return next(calls)

    provider._generate_content = _fake_generate_content  # type: ignore[method-assign]
    return provider


def _chat_response(content: str) -> Any:
    message = SimpleNamespace(content=content)
    choice = SimpleNamespace(message=message)
    return SimpleNamespace(choices=[choice])


def test_orchestrator_system_role_is_not_duplicated_in_user_message():
    calls: list[dict[str, Any]] = []

    def _create(**kwargs: Any) -> Any:
        calls.append(kwargs)
        return _chat_response(
            '{"loop_decision":"route_to_final_response","decisions":[],'
            '"decision_summary":"No worker is needed."}'
        )

    provider = QwenProvider(api_key="qwen-fake-key", max_retries=0)
    provider._client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=_create))
    )
    provider.generate_json(
        ORCHESTRATOR_ROUTING_USER_TASK,
        schema={
            "task": "orchestrator_worker_routing",
            "compact_card_catalog": [
                {
                    "agent_id": "step_06_developability_agent",
                    "capabilities": [
                        {"capability_id": "step_06_developability"}
                    ],
                }
            ],
            "compact_user_intent": "Already satisfied",
            "structured_intent": {},
            "input_readiness_summary": {"input_readiness_status": "ready"},
            "available_artifact_summary": [],
            "current_routing_context": {},
        },
        system=ORCHESTRATOR_ROUTING_SYSTEM_PROMPT,
    )

    messages = calls[0]["messages"]
    assert [message["role"] for message in messages] == ["system", "user"]
    assert messages[0]["content"] == ORCHESTRATOR_ROUTING_SYSTEM_PROMPT
    user = messages[1]["content"]
    assert ORCHESTRATOR_ROUTING_SYSTEM_PROMPT not in user
    assert ORCHESTRATOR_ROUTING_USER_TASK in user
    assert user.count('"input_situation"') == 2
    assert "Compact AgentCard catalog JSON:" in user
    assert "step_06_developability_agent" in user
    assert sum(
        message["content"].count(ORCHESTRATOR_ROUTING_SYSTEM_PROMPT)
        for message in messages
    ) == 1


def _chat_response_with_usage(
    content: str,
    *,
    prompt: int | None = None,
    completion: int | None = None,
    total: int | None = None,
    cached: int | None = None,
) -> Any:
    message = SimpleNamespace(content=content)
    choice = SimpleNamespace(message=message)
    details = (
        SimpleNamespace(cached_tokens=cached) if cached is not None else None
    )
    usage = SimpleNamespace(
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=total,
        prompt_tokens_details=details,
    )
    return SimpleNamespace(choices=[choice], usage=usage)


def test_structured_query_accepts_valid_json():
    provider = _provider_with_responses([
        _chat_response(
            '{"task_intent":{"task_type":"adc_design"},'
            '"mentioned_entities":{"target_or_antigen_text":"HER2"}}'
        )
    ])
    out = provider.generate_json(
        "parse", schema={"raw_request_record": {"raw_user_query": "HER2"}}
    )
    assert out["task_intent"]["task_type"] == "adc_design"
    assert out["mentioned_entities"]["target_or_antigen_text"] == "HER2"
    assert out["referenced_inputs"] == []
    assert out["parse_warnings"] == []


def test_step6_schema_mapping_stage1_accepts_valid_payload_without_task_intent():
    provider = _provider_with_responses([
        _chat_response(
            '{"selections":[{"tool_name":"DrugProps_pains_filter",'
            '"selection_reason":"smiles"}]}'
        )
    ])
    out = provider.generate_json(
        "pick", schema={"task": "step6_schema_mapping_stage_1"}
    )
    assert out["selections"][0]["tool_name"] == "DrugProps_pains_filter"


def test_step6_schema_mapping_stage2_accepts_valid_payload_without_task_intent():
    provider = _provider_with_responses([
        _chat_response(
            '{"tools":[{"tool_name":"DrugProps_pains_filter",'
            '"can_invoke":true,'
            '"argument_mapping":{"smiles":"candidate:c1:material:m1:value"},'
            '"missing_required_fields":[],'
            '"argument_mapping_reason":"mapped"}]}'
        )
    ])
    out = provider.generate_json(
        "map", schema={"task": "step6_schema_mapping_stage_2"}
    )
    assert out["tools"][0]["argument_mapping"]["smiles"].startswith("candidate:")


def test_malformed_json_retries_and_returns_corrected_payload():
    provider = _provider_with_responses([
        _chat_response('{"selections": ['),
        _chat_response('{"selections":[{"tool_name":"DrugProps_calculate_qed"}]}'),
    ])
    out = provider.generate_json("pick", schema={"task": "tool_selection_stage_1"})
    assert out["selections"][0]["tool_name"] == "DrugProps_calculate_qed"


def test_invalid_task_shape_raises_clear_error_without_leakage():
    provider = _provider_with_responses([
        _chat_response('{"selections":"not-a-list"}')
    ] * 3)
    with pytest.raises(QwenProviderError) as excinfo:
        provider.generate_json(
            "USER PROMPT WITH PRIVATE CONTEXT",
            schema={"task": "step6_schema_mapping_stage_1"},
        )
    msg = str(excinfo.value)
    assert "step6_schema_mapping_stage_1" in msg
    assert "selections" in msg
    assert "qwen-fake-key" not in msg
    assert "USER PROMPT WITH PRIVATE CONTEXT" not in msg
    assert "not-a-list" not in msg


def test_response_without_choices_raises_clear_error():
    provider = _provider_with_responses([SimpleNamespace(choices=[])] * 3)
    with pytest.raises(QwenProviderError, match="no choices"):
        provider.generate_json("pick", schema={"task": "tool_selection_stage_1"})


def test_usage_events_include_qwen_provider_and_model():
    provider = _provider_with_responses([
        _chat_response_with_usage(
            '{"task_intent":{"task_type":"adc_design"},'
            '"mentioned_entities":{}}',
            prompt=120,
            completion=45,
            total=165,
            cached=32,
        )
    ])
    provider.generate_json("parse", schema={"raw_request_record": {}})
    assert provider.usage_events == [
        {
            "provider": "qwen",
            "model": provider.model,
            "task": "structured_query",
            "attempt": 0,
            "prompt_tokens": 120,
            "completion_tokens": 45,
            "total_tokens": 165,
            "cached_prompt_tokens": 32,
        }
    ]


def test_error_message_does_not_contain_api_key_prompt_or_raw_response():
    provider = QwenProvider(api_key="qwen-secret-key", max_retries=0)

    def _fake_generate_content(prompt: str, *, system: str | None = None) -> Any:
        return _chat_response("RAW_RESPONSE_SECRET")

    provider._generate_content = _fake_generate_content  # type: ignore[method-assign]
    with pytest.raises(QwenProviderError) as excinfo:
        provider.generate_json(
            "PROMPT_SECRET",
            schema={"task": "tool_selection_stage_1"},
        )
    msg = str(excinfo.value)
    assert "qwen-secret-key" not in msg
    assert "PROMPT_SECRET" not in msg
    assert "RAW_RESPONSE_SECRET" not in msg


def test_no_api_key_raises_value_error():
    with pytest.raises(ValueError, match="non-empty api_key"):
        QwenProvider(api_key="")


def test_generate_json_retries_on_timeout_and_returns_response():
    class _Timeout(Exception):
        """Stub exception that looks like a transport timeout."""

    provider = QwenProvider(api_key="qwen-fake-key", max_retries=1, timeout=1.0)
    calls = iter([
        _Timeout("connection timed out"),
        _chat_response('{"selections":[{"tool_name":"DrugProps_calculate_qed"}]}'),
    ])

    def _fake_generate_content(prompt: str, *, system: str | None = None) -> Any:
        value = next(calls)
        if isinstance(value, Exception):
            raise value
        return value

    provider._generate_content = _fake_generate_content  # type: ignore[method-assign]
    out = provider.generate_json(
        "pick", schema={"task": "step6_schema_mapping_stage_1"}
    )

    assert out["selections"][0]["tool_name"] == "DrugProps_calculate_qed"
    # usage is recorded only after the successful retry attempt.
    assert len(provider.usage_events) == 1
    assert provider.usage_events[0]["attempt"] == 1


def test_generate_json_reports_timeout_without_task_intent_leakage():
    class _Timeout(Exception):
        """Timeout stub to exercise timeout normalization branch."""

    provider = QwenProvider(api_key="qwen-fake-key", max_retries=0, timeout=1.0)

    def _fake_generate_content(prompt: str, *, system: str | None = None) -> Any:
        raise _Timeout("timed out while waiting")

    provider._generate_content = _fake_generate_content  # type: ignore[method-assign]
    with pytest.raises(QwenProviderError) as excinfo:
        provider.generate_json(
            "pick", schema={"task": "step6_schema_mapping_stage_1"}
        )
    msg = str(excinfo.value)
    assert "timed out" in msg.lower()
    assert "non-empty" not in msg
