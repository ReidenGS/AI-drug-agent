"""OpenAIProvider JSON generation tests.

These tests stub the SDK call inside the provider (``_generate_content``)
so they never hit the OpenAI API. They verify shared JSON-task validation
flows through the OpenAI surface the same way it flows through Gemini.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from app.llm.openai_provider import OpenAIProvider, OpenAIProviderError


def _provider_with_responses(responses: list[Any]) -> OpenAIProvider:
    provider = OpenAIProvider(api_key="sk-fake-key", max_retries=2)
    calls = iter(responses)

    def _fake_generate_content(prompt: str, *, system: str | None = None) -> Any:
        return next(calls)

    provider._generate_content = _fake_generate_content  # type: ignore[method-assign]
    return provider


def _chat_response(content: str) -> Any:
    """Build a fake Chat Completions response with one choice."""
    message = SimpleNamespace(content=content)
    choice = SimpleNamespace(message=message)
    return SimpleNamespace(choices=[choice])


# ── basic JSON happy paths ─────────────────────────────────────────────────


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
    # Shared normalization filled empty lists for absent optional fields.
    assert out["referenced_inputs"] == []
    assert out["parse_warnings"] == []


def test_structured_query_promotes_component_name_alias():
    provider = _provider_with_responses([
        _chat_response(
            '{"task_intent":{"task_type":"adc_design"},'
            '"mentioned_entities":{"payload_text":"vc-MMAE"},'
            '"entity_decompositions":[{"original_text":"vc-MMAE",'
            '"components":[{"component_name":"valine-citrulline",'
            '"component_type":"linker","inferred":true}]}]}'
        )
    ])
    out = provider.generate_json(
        "parse", schema={"task": "structured_query", "raw_request_record": {}}
    )
    comp = out["entity_decompositions"][0]["components"][0]
    assert comp["canonical_name"] == "valine-citrulline"
    assert comp["component_type"] == "linker"
    assert comp["role"] == "linker"


def test_stage1_multi_lane_passes_shared_validator():
    provider = _provider_with_responses([
        _chat_response(
            '{"selections":['
            '{"lane_type":"payload_linker_compound_liability",'
            ' "tool_name":"DrugProps_pains_filter",'
            ' "selection_reason":"smiles present"}]}'
        )
    ])
    out = provider.generate_json(
        "pick", schema={"task": "tool_selection_stage_1_multi_lane"}
    )
    assert out["selections"][0]["tool_name"] == "DrugProps_pains_filter"


def test_stage2_multi_tool_passes_shared_validator():
    provider = _provider_with_responses([
        _chat_response(
            '{"tools":[{"lane_type":"payload_linker_compound_liability",'
            '"tool_name":"DrugProps_pains_filter",'
            '"arguments":{"smiles":"CCO"}}]}'
        )
    ])
    out = provider.generate_json(
        "args", schema={"task": "tool_selection_stage_2_multi_tool"}
    )
    assert out["tools"][0]["arguments"] == {"smiles": "CCO"}


def test_stage1_single_tool_path_passes_shared_validator():
    provider = _provider_with_responses([
        _chat_response('{"selections":[{"tool_name":"DrugProps_calculate_qed"}]}')
    ])
    out = provider.generate_json(
        "pick", schema={"task": "tool_selection_stage_1"}
    )
    assert out["selections"][0]["tool_name"] == "DrugProps_calculate_qed"


# ── error paths ───────────────────────────────────────────────────────────


def test_invalid_top_level_shape_raises_clear_error_without_prompt_leakage():
    bad = '{"arguments":{}}'  # missing `selections` for stage_1
    provider = _provider_with_responses([_chat_response(bad)] * 3)
    with pytest.raises(OpenAIProviderError) as excinfo:
        provider.generate_json("pick", schema={"task": "tool_selection_stage_1"})
    msg = str(excinfo.value)
    # Compact error referencing the contract, not the prompt or response body.
    assert "tool_selection_stage_1" in msg
    assert "selections" in msg
    # Make sure secrets and raw text don't leak through.
    assert "sk-" not in msg
    assert "System instructions" not in msg


def test_malformed_json_retries_and_returns_corrected_payload():
    provider = _provider_with_responses([
        _chat_response('{"selections": ['),  # malformed
        _chat_response('{"selections":[{"tool_name":"DrugProps_calculate_qed"}]}'),
    ])
    out = provider.generate_json("pick", schema={"task": "tool_selection_stage_1"})
    assert out["selections"][0]["tool_name"] == "DrugProps_calculate_qed"


def test_response_without_choices_raises_clear_error():
    provider = _provider_with_responses(
        [SimpleNamespace(choices=[])] * 3
    )
    with pytest.raises(OpenAIProviderError, match="no choices"):
        provider.generate_json("pick", schema={"task": "tool_selection_stage_1"})


def test_stage1_multi_lane_rejects_selection_missing_lane_type():
    provider = _provider_with_responses(
        [_chat_response('{"selections":[{"tool_name":"DrugProps_pains_filter"}]}')] * 3
    )
    with pytest.raises(OpenAIProviderError, match="multi_lane.*lane_type"):
        provider.generate_json(
            "pick", schema={"task": "tool_selection_stage_1_multi_lane"}
        )


def test_no_api_key_raises_value_error():
    with pytest.raises(ValueError, match="non-empty api_key"):
        OpenAIProvider(api_key="")


# ── usage event recording ──────────────────────────────────────────────────


def _chat_response_with_usage(
    content: str, *, prompt: int | None = None,
    completion: int | None = None, total: int | None = None,
) -> Any:
    message = SimpleNamespace(content=content)
    choice = SimpleNamespace(message=message)
    usage = SimpleNamespace(
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=total,
    )
    return SimpleNamespace(choices=[choice], usage=usage)


def test_usage_events_record_token_counts_per_attempt():
    provider = _provider_with_responses([
        _chat_response_with_usage(
            '{"task_intent":{"task_type":"adc_design"},'
            '"mentioned_entities":{"target_or_antigen_text":"HER2"}}',
            prompt=120, completion=45, total=165,
        )
    ])
    provider.generate_json("parse", schema={"raw_request_record": {}})
    assert len(provider.usage_events) == 1
    evt = provider.usage_events[0]
    assert evt == {
        "provider": "openai",
        "model": provider.model,
        "task": "structured_query",
        "attempt": 0,
        "prompt_tokens": 120,
        "completion_tokens": 45,
        "total_tokens": 165,
        # The stub response does not carry prompt_tokens_details, so
        # the cache field degrades to None. A dedicated test below
        # exercises the present-and-populated path.
        "cached_prompt_tokens": None,
    }


def test_usage_events_recorded_for_each_retry_attempt():
    """Each provider call costs real tokens. A retry triggered by a
    malformed first response must still log the first attempt."""
    provider = _provider_with_responses([
        _chat_response_with_usage(
            "not json at all", prompt=50, completion=10, total=60,
        ),
        _chat_response_with_usage(
            '{"task_intent":{"task_type":"adc_design"},'
            '"mentioned_entities":{}}',
            prompt=80, completion=12, total=92,
        ),
    ])
    provider.generate_json("parse", schema={"raw_request_record": {}})
    assert [e["attempt"] for e in provider.usage_events] == [0, 1]
    assert provider.usage_events[0]["total_tokens"] == 60
    assert provider.usage_events[1]["total_tokens"] == 92


def test_usage_event_degrades_to_null_when_response_has_no_usage_field():
    """Some SDK / proxy responses omit the usage block. The event must
    still be appended with token fields set to None — never crash."""
    message = SimpleNamespace(content='{"task_intent":{"task_type":"adc_design"},"mentioned_entities":{}}')
    choice = SimpleNamespace(message=message)
    bare_response = SimpleNamespace(choices=[choice])  # no `usage`
    provider = _provider_with_responses([bare_response])
    provider.generate_json("parse", schema={"raw_request_record": {}})
    assert len(provider.usage_events) == 1
    evt = provider.usage_events[0]
    assert evt["prompt_tokens"] is None
    assert evt["completion_tokens"] is None
    assert evt["total_tokens"] is None


def test_usage_events_carry_no_prompt_response_or_api_key():
    """Defensive content check: the compact event must never include
    prompt body, response body, schema payload, or the API key."""
    secret_prompt = "USER PROMPT BODY WITH A SECRET HER2 SCAFFOLD"
    secret_response = (
        '{"task_intent":{"task_type":"adc_design"},'
        '"mentioned_entities":{"target_or_antigen_text":"HER2_SECRET"}}'
    )
    provider = OpenAIProvider(api_key="sk-very-secret-key", max_retries=0)

    def _fake_generate_content(prompt: str, *, system: str | None = None) -> Any:
        # The provider must NOT carry `prompt` through to the event.
        return _chat_response_with_usage(secret_response, prompt=10, completion=5, total=15)

    provider._generate_content = _fake_generate_content  # type: ignore[method-assign]
    provider.generate_json(secret_prompt, schema={"raw_request_record": {}})
    assert len(provider.usage_events) == 1
    import json as _json
    blob = _json.dumps(provider.usage_events[0])
    assert "sk-very-secret-key" not in blob
    assert "USER PROMPT BODY" not in blob
    assert "SECRET HER2 SCAFFOLD" not in blob
    assert "HER2_SECRET" not in blob


# ── cached prompt tokens (OpenAI auto prompt caching) ─────────────────────


def _chat_response_with_cache(
    content: str, *, prompt: int, completion: int, total: int,
    cached: int | None,
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


def test_usage_event_reads_cached_prompt_tokens_from_prompt_tokens_details():
    provider = _provider_with_responses([
        _chat_response_with_cache(
            '{"task_intent":{"task_type":"adc_design"},"mentioned_entities":{}}',
            prompt=1000, completion=200, total=1200, cached=512,
        )
    ])
    provider.generate_json("parse", schema={"raw_request_record": {}})
    evt = provider.usage_events[0]
    assert evt["prompt_tokens"] == 1000
    assert evt["cached_prompt_tokens"] == 512
    assert evt["total_tokens"] == 1200


def test_usage_event_cached_prompt_tokens_none_when_no_details():
    provider = _provider_with_responses([
        _chat_response_with_cache(
            '{"task_intent":{"task_type":"adc_design"},"mentioned_entities":{}}',
            prompt=1000, completion=50, total=1050, cached=None,
        )
    ])
    provider.generate_json("parse", schema={"raw_request_record": {}})
    evt = provider.usage_events[0]
    assert evt["cached_prompt_tokens"] is None


def test_usage_event_dict_shaped_usage_with_details_dict():
    """Some proxies return usage as a dict instead of an SDK object;
    the cache reader must still degrade safely."""
    message = SimpleNamespace(
        content='{"task_intent":{"task_type":"adc_design"},"mentioned_entities":{}}'
    )
    choice = SimpleNamespace(message=message)
    response = SimpleNamespace(choices=[choice], usage={
        "prompt_tokens": 800,
        "completion_tokens": 40,
        "total_tokens": 840,
        "prompt_tokens_details": {"cached_tokens": 256},
    })
    provider = _provider_with_responses([response])
    provider.generate_json("parse", schema={"raw_request_record": {}})
    evt = provider.usage_events[0]
    assert evt["cached_prompt_tokens"] == 256


def test_cached_field_does_not_leak_prompt_or_response(monkeypatch):
    """Regression: the cache field is a single int; even when present
    the event must not carry prompt body, response body, or API key."""
    provider = OpenAIProvider(api_key="sk-cache-secret", max_retries=0)

    def _fake_generate_content(prompt: str, *, system: str | None = None) -> Any:
        return _chat_response_with_cache(
            '{"task_intent":{"task_type":"adc_design"},'
            '"mentioned_entities":{"target_or_antigen_text":"HER2_CACHE_SECRET"}}',
            prompt=10, completion=5, total=15, cached=4,
        )

    provider._generate_content = _fake_generate_content  # type: ignore[method-assign]
    provider.generate_json("PROMPT CACHE BODY", schema={"raw_request_record": {}})
    import json as _json
    blob = _json.dumps(provider.usage_events[0])
    assert "sk-cache-secret" not in blob
    assert "PROMPT CACHE BODY" not in blob
    assert "HER2_CACHE_SECRET" not in blob


# ── system prompt is NOT duplicated into the user message ─────────────────


def test_openai_system_prompt_is_not_embedded_into_user_message():
    """``OpenAIProvider`` must forward ``system`` as a dedicated
    role=system message and NOT also prepend it into the user message
    body. Embedding it twice burns prompt tokens on every Step 2 /
    Step 5 call."""
    sentinel_system = "SENTINEL_SYSTEM_BLOCK_OPENAI_only"
    user_prompt_body = "USER_PROMPT_SENTINEL_for_openai"

    captured: dict[str, Any] = {}

    provider = OpenAIProvider(api_key="sk-fake-dedup", max_retries=0)

    def _fake_generate_content(prompt: str, *, system: str | None = None) -> Any:
        captured["prompt"] = prompt
        captured["system"] = system
        return _chat_response(
            '{"task_intent":{"task_type":"adc_design"},'
            '"mentioned_entities":{}}'
        )

    provider._generate_content = _fake_generate_content  # type: ignore[method-assign]
    provider.generate_json(
        user_prompt_body,
        schema={"raw_request_record": {"raw_user_query": "HER2"}},
        system=sentinel_system,
    )

    # System sentinel travels via the dedicated `system` kwarg.
    assert captured["system"] == sentinel_system
    # And does NOT appear in the user-message body — so Chat Completions
    # never receives the same system block twice.
    assert sentinel_system not in captured["prompt"], captured["prompt"][:200]
    # The user-message body still carries the user prompt + the expected
    # task-shape preamble produced by ``build_json_prompt`` so the LLM
    # has everything it needs.
    assert user_prompt_body in captured["prompt"]
    assert "expected top-level shape" in captured["prompt"].lower() or (
        "task_intent" in captured["prompt"]
    ), captured["prompt"][:200]


def test_openai_handles_none_system_without_embedding_anything():
    """When the caller does not pass a system block, the user-message
    body still must not embed an empty `System instructions:` header."""
    user_prompt_body = "USER_PROMPT_NO_SYSTEM"
    captured: dict[str, Any] = {}
    provider = OpenAIProvider(api_key="sk-fake-no-sys", max_retries=0)

    def _fake_generate_content(prompt: str, *, system: str | None = None) -> Any:
        captured["prompt"] = prompt
        captured["system"] = system
        return _chat_response(
            '{"task_intent":{"task_type":"adc_design"},'
            '"mentioned_entities":{}}'
        )

    provider._generate_content = _fake_generate_content  # type: ignore[method-assign]
    provider.generate_json(
        user_prompt_body,
        schema={"raw_request_record": {}},
        system=None,
    )
    assert captured["system"] is None
    assert "System instructions" not in captured["prompt"]
    assert user_prompt_body in captured["prompt"]


def test_openai_usage_events_shape_unchanged_after_dedup():
    """Defensive sanity: the dedup fix does not strip any field from
    ``usage_events``. Every recorded entry still has the 8 documented
    keys."""
    provider = _provider_with_responses([
        _chat_response_with_usage(
            '{"task_intent":{"task_type":"adc_design"},'
            '"mentioned_entities":{}}',
            prompt=100, completion=20, total=120,
        )
    ])
    provider.generate_json(
        "parse", schema={"raw_request_record": {}},
        system="SOME SYSTEM PROMPT",
    )
    assert len(provider.usage_events) == 1
    assert set(provider.usage_events[0].keys()) == {
        "provider", "model", "task", "attempt",
        "prompt_tokens", "completion_tokens", "total_tokens",
        "cached_prompt_tokens",
    }
    import json as _json
    blob = _json.dumps(provider.usage_events[0])
    assert "sk-fake-key" not in blob
    assert "SOME SYSTEM PROMPT" not in blob
