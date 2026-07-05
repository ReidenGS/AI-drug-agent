"""OpenAIProvider JSON generation tests.

These tests stub the SDK call inside the provider (``_generate_content``)
so they never hit the OpenAI API. They verify shared JSON-task validation
flows through the OpenAI surface the same way it flows through Gemini.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from app.llm.openai_provider import (
    OpenAIProvider,
    OpenAIProviderError,
    _RESPONSE_MODEL_FOR_TASK,
    _ToolSelectionStage1Response,
    _Step6SchemaMappingStage1Response,
    _Step6SchemaMappingStage2ParserResponse,
    _Step9SchemaMappingStage2Response,
    _Step9ToolSelectionStage1Response,
)


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


def test_step6_schema_mapping_stage1_passes_without_task_intent():
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


def test_step6_schema_mapping_stage2_passes_without_task_intent():
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


def test_step9_tool_selection_stage1_passes_without_task_intent():
    provider = _provider_with_responses([
        _chat_response(
            '{"selections":[{"tool_name":"ZINC_search_by_smiles",'
            '"lane_type":"compound_screening",'
            '"selection_reason":"smiles available"}]}'
        )
    ])
    out = provider.generate_json(
        "pick", schema={"task": "step9_tool_selection_stage_1"}
    )
    assert out["selections"][0]["lane_type"] == "compound_screening"


def test_step9_schema_mapping_stage2_passes_without_task_intent():
    provider = _provider_with_responses([
        _chat_response(
            '{"tools":[{"tool_name":"ZINC_search_by_smiles",'
            '"lane_type":"compound_screening","can_invoke":true,'
            '"argument_mappings":[{"schema_arg":"smiles","field_ref":"material:m1"}],'
            '"argument_literals":[],"missing_required_fields":[],'
            '"skip_reason":"","argument_mapping_reason":"mapped"}]}'
        )
    ])
    out = provider.generate_json(
        "map", schema={"task": "step9_tool_schema_mapping_stage_2"}
    )
    assert out["tools"][0]["argument_mappings"][0]["schema_arg"] == "smiles"


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


def test_step6_schema_mapping_stage1_rejects_malformed_selections():
    provider = _provider_with_responses(
        [_chat_response('{"selections":"not-a-list"}')] * 3
    )
    with pytest.raises(OpenAIProviderError, match="step6_schema_mapping_stage_1.*selections"):
        provider.generate_json(
            "pick", schema={"task": "step6_schema_mapping_stage_1"}
        )


def test_step6_schema_mapping_stage2_rejects_malformed_tools():
    provider = _provider_with_responses(
        [_chat_response(
            '{"tools":[{"tool_name":"DrugProps_pains_filter",'
            '"can_invoke":"yes","argument_mapping":{},'
            '"missing_required_fields":[]}]}'
        )] * 3
    )
    with pytest.raises(OpenAIProviderError, match="step6_schema_mapping_stage_2.*can_invoke"):
        provider.generate_json(
            "map", schema={"task": "step6_schema_mapping_stage_2"}
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


# ── official structured-output parser path + json_object fallback ─────────


def _fake_openai_client(*, parse=None, create=None, has_parse: bool = True):
    """Build a fake OpenAI client exposing `beta.chat.completions.parse` and
    `chat.completions.create`, recording calls to each."""
    calls: dict[str, list[dict[str, Any]]] = {"parse": [], "create": []}
    completions = SimpleNamespace()

    if has_parse:
        def _parse(**kw: Any) -> Any:
            calls["parse"].append(kw)
            return parse(**kw) if callable(parse) else parse
        completions.parse = _parse

    def _create(**kw: Any) -> Any:
        calls["create"].append(kw)
        return create(**kw) if callable(create) else create
    completions.create = _create

    chat = SimpleNamespace(completions=completions)
    client = SimpleNamespace(beta=SimpleNamespace(chat=chat), chat=chat)
    client._calls = calls  # type: ignore[attr-defined]
    return client


def _parsed_response(parsed_obj: Any, *, usage: Any = None) -> Any:
    message = SimpleNamespace(parsed=parsed_obj, content=None, refusal=None)
    choice = SimpleNamespace(message=message)
    return SimpleNamespace(choices=[choice], usage=usage)


def _provider_with_client(client: Any, *, max_retries: int = 2) -> OpenAIProvider:
    provider = OpenAIProvider(api_key="sk-fake-parser", max_retries=max_retries)
    provider._client = client  # type: ignore[assignment]  # bypass real SDK construction
    return provider


# ── strict-schema compliance (real production requirement, not fake client) ─


def _strict_schema_violations(schema: Any, path: str = "root") -> list[str]:
    """Return object/array nodes that violate OpenAI strict structured output:
    an object with `additionalProperties` != False, or an array with an
    unconstrained (empty) `items`."""
    problems: list[str] = []
    if isinstance(schema, dict):
        if schema.get("type") == "object" and "properties" in schema:
            if schema.get("additionalProperties") is not False:
                problems.append(f"{path}: additionalProperties={schema.get('additionalProperties')!r}")
        if schema.get("type") == "array" and not schema.get("items"):
            problems.append(f"{path}: unconstrained items={schema.get('items')!r}")
        for key, value in schema.items():
            problems += _strict_schema_violations(value, f"{path}.{key}")
    elif isinstance(schema, list):
        for i, value in enumerate(schema):
            problems += _strict_schema_violations(value, f"{path}[{i}]")
    return problems


def test_parser_models_produce_real_strict_schema():
    """Every parser-supported response model must yield a genuinely strict
    schema via the SDK's own `to_strict_json_schema`: no `additionalProperties:
    true` and no unconstrained array items anywhere. This is the real
    production gate — not a fake-client assertion that `parse` was called."""
    from openai.lib._pydantic import to_strict_json_schema

    # Currently ACTIVE parser tasks. step6_schema_mapping_stage_2 is
    # temporarily disabled from the parser path (json_object instead) while
    # the Step 6 Stage 2 prompt still asks for the legacy dynamic-dict shape.
    assert set(_RESPONSE_MODEL_FOR_TASK) == {
        "tool_selection_stage_1",
        "step6_schema_mapping_stage_1",
        "step9_tool_selection_stage_1",
        "step9_tool_schema_mapping_stage_2",
    }
    assert "step6_schema_mapping_stage_2" not in _RESPONSE_MODEL_FOR_TASK
    for task, model in _RESPONSE_MODEL_FOR_TASK.items():
        schema = to_strict_json_schema(model)
        violations = _strict_schema_violations(schema)
        assert not violations, f"{task}/{model.__name__} strict-schema violations: {violations}"


def test_disabled_step6_stage2_parser_model_is_still_strict_for_future_use():
    """The Step 6 Stage 2 list-of-pairs parser model is TEMPORARILY DISABLED
    from the active parser path but kept for future re-enable. It must still
    produce a genuinely strict schema so re-enabling is a one-line change."""
    from openai.lib._pydantic import to_strict_json_schema

    schema = to_strict_json_schema(_Step6SchemaMappingStage2ParserResponse)
    violations = _strict_schema_violations(schema)
    assert not violations, f"disabled step6 stage2 model strict-schema violations: {violations}"


def test_step6_stage2_uses_json_object_path_not_parser():
    """Even when the parser IS available, step6_schema_mapping_stage_2 must go
    through the json_object path (returning the legacy dynamic-dict shape),
    because it is disabled from the parser registry."""
    client = _fake_openai_client(
        has_parse=True,
        create=_chat_response(
            '{"tools":[{"tool_name":"DrugProps_pains_filter","can_invoke":true,'
            '"argument_mapping":{"smiles":"candidate:c1:material:m1:value"},'
            '"missing_required_fields":[],"argument_mapping_reason":"mapped"}]}'
        ),
    )
    provider = _provider_with_client(client)
    out = provider.generate_json("map", schema={"task": "step6_schema_mapping_stage_2"})
    assert client._calls["parse"] == []  # parser NOT used for disabled task
    assert len(client._calls["create"]) == 1
    assert out["tools"][0]["argument_mapping"] == {"smiles": "candidate:c1:material:m1:value"}


def test_structured_query_not_routed_to_strict_parser():
    """structured_query (deeply-nested / variant fields) must NOT be routed to
    the strict parser — it stays on the json_object path."""
    assert "structured_query" not in _RESPONSE_MODEL_FOR_TASK


# ── behavioral parser tests (supported tasks: stage_1 / step6 stage_1) ─────


def test_openai_stage1_uses_structured_parser_and_returns_dict():
    parsed = _ToolSelectionStage1Response(
        selections=[{"tool_name": "DrugProps_calculate_qed"}]
    )
    client = _fake_openai_client(parse=_parsed_response(parsed))
    provider = _provider_with_client(client)

    out = provider.generate_json("pick", schema={"task": "tool_selection_stage_1"})

    assert len(client._calls["parse"]) == 1
    assert client._calls["create"] == []  # parser succeeded → no fallback
    assert client._calls["parse"][0]["response_format"] is _ToolSelectionStage1Response
    assert isinstance(out, dict)
    assert not isinstance(out, _ToolSelectionStage1Response)
    assert out["selections"][0]["tool_name"] == "DrugProps_calculate_qed"


def test_openai_step6_stage1_uses_structured_parser_and_validated():
    parsed = _Step6SchemaMappingStage1Response(
        selections=[{"tool_name": "DrugProps_pains_filter", "selection_reason": "smiles"}]
    )
    client = _fake_openai_client(parse=_parsed_response(parsed))
    provider = _provider_with_client(client)
    out = provider.generate_json("pick", schema={"task": "step6_schema_mapping_stage_1"})
    assert client._calls["parse"] and not client._calls["create"]
    assert client._calls["parse"][0]["response_format"] is _Step6SchemaMappingStage1Response
    assert out["selections"][0]["tool_name"] == "DrugProps_pains_filter"


def test_openai_step9_stage1_uses_structured_parser_and_returns_dict():
    parsed = _Step9ToolSelectionStage1Response(
        selections=[{
            "tool_name": "ZINC_search_by_smiles",
            "lane_type": "compound_screening",
            "selection_reason": "smiles available",
        }]
    )
    client = _fake_openai_client(parse=_parsed_response(parsed))
    provider = _provider_with_client(client)
    out = provider.generate_json(
        "pick", schema={"task": "step9_tool_selection_stage_1"}
    )
    assert client._calls["parse"] and not client._calls["create"]
    assert client._calls["parse"][0]["response_format"] is _Step9ToolSelectionStage1Response
    assert isinstance(out, dict)
    assert not isinstance(out, _Step9ToolSelectionStage1Response)
    assert out["selections"][0]["tool_name"] == "ZINC_search_by_smiles"


def test_openai_step9_stage2_uses_structured_parser_and_returns_dict():
    parsed = _Step9SchemaMappingStage2Response(
        tools=[{
            "tool_name": "ZINC_search_by_smiles",
            "lane_type": "compound_screening",
            "can_invoke": True,
            "argument_mappings": [
                {"schema_arg": "smiles", "field_ref": "material:m1"},
            ],
            "argument_literals": [],
            "missing_required_fields": [],
            "skip_reason": "",
            "argument_mapping_reason": "mapped",
        }]
    )
    client = _fake_openai_client(parse=_parsed_response(parsed))
    provider = _provider_with_client(client)
    out = provider.generate_json(
        "map", schema={"task": "step9_tool_schema_mapping_stage_2"}
    )
    assert client._calls["parse"] and not client._calls["create"]
    assert client._calls["parse"][0]["response_format"] is _Step9SchemaMappingStage2Response
    assert isinstance(out, dict)
    assert not isinstance(out, _Step9SchemaMappingStage2Response)
    assert out["tools"][0]["argument_mappings"] == [
        {"schema_arg": "smiles", "field_ref": "material:m1"}
    ]


def test_openai_step9_stage2_duplicate_schema_arg_raises_then_retry_succeeds():
    dup = _Step9SchemaMappingStage2Response(
        tools=[{
            "tool_name": "ZINC_search_by_smiles",
            "lane_type": "compound_screening",
            "can_invoke": True,
            "argument_mappings": [
                {"schema_arg": "smiles", "field_ref": "material:m1"},
                {"schema_arg": "smiles", "field_ref": "material:m2"},
            ],
            "argument_literals": [],
            "missing_required_fields": [],
        }]
    )
    good = _Step9SchemaMappingStage2Response(
        tools=[{
            "tool_name": "ZINC_search_by_smiles",
            "lane_type": "compound_screening",
            "can_invoke": True,
            "argument_mappings": [
                {"schema_arg": "smiles", "field_ref": "material:m1"},
            ],
            "argument_literals": [],
            "missing_required_fields": [],
        }]
    )
    seq = iter([_parsed_response(dup), _parsed_response(good)])
    client = _fake_openai_client(parse=lambda **kw: next(seq))
    provider = _provider_with_client(client, max_retries=1)
    out = provider.generate_json(
        "map", schema={"task": "step9_tool_schema_mapping_stage_2"}
    )
    assert len(client._calls["parse"]) == 2
    assert out["tools"][0]["argument_mappings"] == [
        {"schema_arg": "smiles", "field_ref": "material:m1"}
    ]


def test_openai_parser_output_flows_through_validate_task_shape():
    """Parser output still runs the shared per-task validator: a stage_1 entry
    missing tool_name is rejected (with retry) rather than silently accepted."""
    bad = _ToolSelectionStage1Response()  # empty selections is valid-empty
    # Build a parsed object whose model_dump omits tool_name by using a dict.
    bad_dict = {"selections": [{"selection_reason": "no tool_name here"}]}
    client = _fake_openai_client(parse=_parsed_response(bad_dict))
    provider = _provider_with_client(client, max_retries=0)
    with pytest.raises(OpenAIProviderError, match="tool_selection_stage_1.*tool_name"):
        provider.generate_json("pick", schema={"task": "tool_selection_stage_1"})
    del bad


def test_openai_parser_accepts_dict_parsed_shape():
    """Some SDK/proxy variants set `message.parsed` to a dict; still a dict out."""
    client = _fake_openai_client(
        parse=_parsed_response({"selections": [{"tool_name": "DrugProps_calculate_qed"}]})
    )
    provider = _provider_with_client(client)
    out = provider.generate_json("pick", schema={"task": "tool_selection_stage_1"})
    assert isinstance(out, dict)
    assert out["selections"][0]["tool_name"] == "DrugProps_calculate_qed"


def test_openai_falls_back_to_json_object_when_no_parser_api():
    """SDK without a `parse` method → fall back to json_object create."""
    client = _fake_openai_client(
        has_parse=False,
        create=_chat_response('{"selections":[{"tool_name":"DrugProps_calculate_qed"}]}'),
    )
    provider = _provider_with_client(client)
    out = provider.generate_json("pick", schema={"task": "tool_selection_stage_1"})
    assert client._calls["parse"] == []  # no parse method existed
    assert len(client._calls["create"]) == 1  # fell back to json_object
    assert out["selections"][0]["tool_name"] == "DrugProps_calculate_qed"


def test_openai_falls_back_on_parser_incompatibility_error():
    """Parser raising an incompatibility error → json_object fallback."""
    def _parse_boom(**kw: Any) -> Any:
        raise TypeError("invalid schema for response_format: additionalProperties")

    client = _fake_openai_client(
        parse=_parse_boom,
        create=_chat_response('{"selections":[{"tool_name":"DrugProps_calculate_qed"}]}'),
    )
    provider = _provider_with_client(client)
    out = provider.generate_json("pick", schema={"task": "tool_selection_stage_1"})
    assert len(client._calls["parse"]) == 1  # attempted
    assert len(client._calls["create"]) == 1  # then fell back
    assert out["selections"][0]["tool_name"] == "DrugProps_calculate_qed"


def test_openai_fallback_records_usage_events_and_no_leak():
    """Fallback path still records compact usage and leaks nothing."""
    client = _fake_openai_client(
        has_parse=False,
        create=_chat_response_with_usage(
            '{"selections":[{"tool_name":"DrugProps_calculate_qed"}]}',
            prompt=70, completion=15, total=85,
        ),
    )
    provider = _provider_with_client(client)
    provider.generate_json(
        "USER_PROMPT_SECRET", schema={"task": "tool_selection_stage_1"}, system="SYS_SECRET",
    )
    assert len(provider.usage_events) == 1
    evt = provider.usage_events[0]
    assert evt["task"] == "tool_selection_stage_1"
    assert evt["total_tokens"] == 85
    import json as _json
    blob = _json.dumps(evt)
    for secret in ("sk-fake-parser", "USER_PROMPT_SECRET", "SYS_SECRET"):
        assert secret not in blob


def test_openai_parser_records_usage_events_and_no_leak():
    parsed = _ToolSelectionStage1Response(selections=[{"tool_name": "DrugProps_calculate_qed"}])
    usage = SimpleNamespace(prompt_tokens=200, completion_tokens=30, total_tokens=230)
    client = _fake_openai_client(parse=_parsed_response(parsed, usage=usage))
    provider = _provider_with_client(client)
    provider.generate_json(
        "USER_PROMPT_SECRET", schema={"task": "tool_selection_stage_1"}, system="SYS_SECRET",
    )
    assert len(provider.usage_events) == 1
    evt = provider.usage_events[0]
    assert evt["prompt_tokens"] == 200 and evt["total_tokens"] == 230
    import json as _json
    blob = _json.dumps(evt)
    for secret in ("sk-fake-parser", "USER_PROMPT_SECRET", "SYS_SECRET"):
        assert secret not in blob


def test_openai_parser_genuine_error_propagates_no_silent_fallback():
    """A non-incompatibility error from the parser is NOT masked as a
    fallback — it propagates (no fake success), and create is never called."""
    def _parse_network(**kw: Any) -> Any:
        raise RuntimeError("connection reset by peer")

    client = _fake_openai_client(
        parse=_parse_network,
        create=_chat_response('{"selections":[{"tool_name":"DrugProps_calculate_qed"}]}'),
    )
    provider = _provider_with_client(client, max_retries=0)
    with pytest.raises(RuntimeError, match="connection reset"):
        provider.generate_json("pick", schema={"task": "tool_selection_stage_1"})
    assert client._calls["create"] == []  # never silently fell back


def test_openai_structured_query_stays_on_json_object_path():
    """structured_query is NOT strict-parser-routed: it uses json_object and
    still runs validate + normalize (adc_candidate → ranked_candidates)."""
    client = _fake_openai_client(
        parse=_parsed_response({"should": "not be used"}),
        create=_chat_response(
            '{"task_intent":{"task_type":"adc_design"},"mentioned_entities":{},'
            '"requested_outputs":["adc_candidate","report"]}'
        ),
    )
    provider = _provider_with_client(client)
    out = provider.generate_json("parse", schema={"task": "structured_query"})
    assert client._calls["parse"] == []  # parser NOT used for structured_query
    assert len(client._calls["create"]) == 1
    assert out["requested_outputs"] == ["ranked_candidates", "report"]


# ── Step 6 Stage 2 parser model: TEMPORARILY DISABLED, kept for future ──────
# These exercise the KEPT list-of-pairs parser model + `to_external_dict`
# directly (NOT the provider routing, which is json_object for this task now),
# so the conversion + duplicate-guard code stays covered for a future
# re-enable. See `test_step6_stage2_uses_json_object_path_not_parser` for the
# active routing behavior.


def test_future_step6_stage2_model_list_of_pairs_converts_to_dynamic_dict():
    parsed = _Step6SchemaMappingStage2ParserResponse(
        tools=[{
            "tool_name": "SwissADME_calculate_adme",
            "can_invoke": True,
            "argument_mappings": [
                {"schema_arg": "smiles", "field_ref": "candidate:c1:material:m1:value"},
            ],
            "argument_literals": [
                {"schema_arg": "operation", "literal_value": "calculate_adme"},
            ],
            "missing_required_fields": [],
            "argument_mapping_reason": "smiles mapped; operation is a schema literal",
        }]
    )
    external = parsed.to_external_dict()
    tool = external["tools"][0]
    assert tool["argument_mapping"] == {"smiles": "candidate:c1:material:m1:value"}
    assert tool["argument_literals"] == {"operation": "calculate_adme"}
    assert tool["can_invoke"] is True
    assert tool["missing_required_fields"] == []
    assert tool["argument_mapping_reason"].startswith("smiles mapped")


def test_future_step6_stage2_model_empty_literals_becomes_empty_dict():
    parsed = _Step6SchemaMappingStage2ParserResponse(
        tools=[{
            "tool_name": "DrugProps_pains_filter",
            "can_invoke": True,
            "argument_mappings": [
                {"schema_arg": "smiles", "field_ref": "candidate:c1:material:m1:value"},
            ],
            "argument_literals": [],
            "missing_required_fields": [],
        }]
    )
    external = parsed.to_external_dict()
    assert external["tools"][0]["argument_literals"] == {}
    assert external["tools"][0]["argument_mapping"] == {"smiles": "candidate:c1:material:m1:value"}


def test_future_step6_stage2_model_can_invoke_false_empty_mappings_missing_fields():
    parsed = _Step6SchemaMappingStage2ParserResponse(
        tools=[{
            "tool_name": "PDBePISA_get_interfaces",
            "can_invoke": False,
            "argument_mappings": [],
            "argument_literals": [],
            "missing_required_fields": ["pdb_id"],
            "argument_mapping_reason": "no pdb_id available",
        }]
    )
    external = parsed.to_external_dict()
    tool = external["tools"][0]
    assert tool["can_invoke"] is False
    assert tool["argument_mapping"] == {}
    assert tool["argument_literals"] == {}
    assert tool["missing_required_fields"] == ["pdb_id"]


def test_future_step6_stage2_model_duplicate_schema_arg_raises_no_silent_overwrite():
    """Duplicate argument_mapping schema_arg raises (never a silent overwrite)
    at conversion time, so a future re-enable keeps that guarantee."""
    dup = _Step6SchemaMappingStage2ParserResponse(
        tools=[{
            "tool_name": "SwissADME_calculate_adme",
            "can_invoke": True,
            "argument_mappings": [
                {"schema_arg": "smiles", "field_ref": "ref_a"},
                {"schema_arg": "smiles", "field_ref": "ref_b"},
            ],
            "argument_literals": [],
            "missing_required_fields": [],
        }]
    )
    with pytest.raises(OpenAIProviderError, match="duplicate argument_mapping"):
        dup.to_external_dict()


def test_openai_step6_stage2_fallback_json_object_accepts_old_dict_shape():
    """When the parser is unavailable, the json_object fallback still returns
    the existing dynamic-dict shape unchanged (no list-of-pairs conversion)."""
    client = _fake_openai_client(
        has_parse=False,
        create=_chat_response(
            '{"tools":[{"tool_name":"DrugProps_pains_filter","can_invoke":true,'
            '"argument_mapping":{"smiles":"candidate:c1:material:m1:value"},'
            '"missing_required_fields":[],"argument_mapping_reason":"mapped"}]}'
        ),
    )
    provider = _provider_with_client(client)
    out = provider.generate_json("map", schema={"task": "step6_schema_mapping_stage_2"})
    assert client._calls["parse"] == []
    assert len(client._calls["create"]) == 1
    assert out["tools"][0]["argument_mapping"] == {"smiles": "candidate:c1:material:m1:value"}


def test_openai_step6_stage2_output_passes_validate_task_shape():
    """The converted external dict passes the shared step6 stage2 validator
    (bad shape would raise). Here a valid tool round-trips."""
    from app.llm.json_task_validation import validate_task_shape
    parsed = _Step6SchemaMappingStage2ParserResponse(
        tools=[{
            "tool_name": "SwissADME_calculate_adme",
            "can_invoke": True,
            "argument_mappings": [{"schema_arg": "smiles", "field_ref": "candidate:x"}],
            "argument_literals": [{"schema_arg": "operation", "literal_value": "calculate_adme"}],
            "missing_required_fields": [],
        }]
    )
    external = parsed.to_external_dict()
    # Direct validator round-trip (same call the provider makes).
    validated = validate_task_shape(external, "step6_schema_mapping_stage_2", error_factory=OpenAIProviderError)
    assert validated["tools"][0]["can_invoke"] is True


def test_openai_unsupported_task_skips_parser_uses_json_object():
    """A task without a structured-output model goes straight to json_object;
    the parser is never attempted."""
    client = _fake_openai_client(
        parse=_parsed_response(_ToolSelectionStage1Response()),  # would be wrong if used
        create=_chat_response(
            '{"tools":[{"lane_type":"payload_linker_compound_liability",'
            '"tool_name":"DrugProps_pains_filter","arguments":{"smiles":"CCO"}}]}'
        ),
    )
    provider = _provider_with_client(client)
    out = provider.generate_json("args", schema={"task": "tool_selection_stage_2_multi_tool"})
    assert client._calls["parse"] == []  # unsupported task → no parser
    assert len(client._calls["create"]) == 1
    assert out["tools"][0]["arguments"] == {"smiles": "CCO"}


def test_openai_parser_refusal_is_surfaced_then_retried():
    """A structured-output refusal raises a compact error and is retried."""
    refusal_msg = SimpleNamespace(parsed=None, content=None, refusal="I can't help with that")
    refusal_resp = SimpleNamespace(choices=[SimpleNamespace(message=refusal_msg)], usage=None)
    good = _parsed_response(_ToolSelectionStage1Response(selections=[{"tool_name": "DrugProps_calculate_qed"}]))
    parse_seq = iter([refusal_resp, good])
    client = _fake_openai_client(parse=lambda **kw: next(parse_seq))
    provider = _provider_with_client(client, max_retries=1)
    out = provider.generate_json("pick", schema={"task": "tool_selection_stage_1"})
    assert len(client._calls["parse"]) == 2  # retried after refusal
    assert out["selections"][0]["tool_name"] == "DrugProps_calculate_qed"
    # both attempts recorded in usage_events
    assert [e["attempt"] for e in provider.usage_events] == [0, 1]


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
