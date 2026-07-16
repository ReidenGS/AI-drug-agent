"""GeminiProvider JSON generation tests.

These tests monkeypatch the provider's model call and never hit the network.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from app.a2a.orchestrator_routing_prompt import (
    ORCHESTRATOR_ROUTING_SYSTEM_PROMPT,
    ORCHESTRATOR_ROUTING_USER_TASK,
)
from app.llm.gemini_provider import GeminiProvider, GeminiProviderError


def _provider_with_responses(responses: list[object]) -> GeminiProvider:
    provider = GeminiProvider(api_key="fake-key", max_retries=2)
    calls = iter(responses)

    def _fake_generate_content(prompt: str) -> object:
        return next(calls)

    provider._generate_content = _fake_generate_content  # type: ignore[method-assign]
    return provider


def test_orchestrator_combined_prompt_contains_system_exactly_once():
    calls: list[dict[str, Any]] = []

    def _generate_content(**kwargs: Any) -> Any:
        calls.append(kwargs)
        return SimpleNamespace(
            parsed={
                "loop_decision": "route_to_final_response",
                "decisions": [],
                "decision_summary": "No worker is needed.",
            }
        )

    provider = GeminiProvider(api_key="fake-key", max_retries=0)
    provider._client = SimpleNamespace(
        models=SimpleNamespace(generate_content=_generate_content)
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

    combined = calls[0]["contents"]
    assert combined.count(ORCHESTRATOR_ROUTING_SYSTEM_PROMPT) == 1
    assert ORCHESTRATOR_ROUTING_USER_TASK in combined
    assert combined.count('"input_situation"') == 2
    assert "Compact AgentCard catalog JSON:" in combined
    assert "step_06_developability_agent" in combined


def test_stage1_accepts_json_mode_text_response():
    provider = _provider_with_responses(
        [
            SimpleNamespace(
                text='{"selections":[{"tool_name":"DrugProps_calculate_qed"}],'
                '"selection_metadata":{"strategy":"test"}}'
            )
        ]
    )

    out = provider.generate_json(
        "pick",
        schema={"task": "tool_selection_stage_1", "compact_catalog": []},
    )

    assert out["selections"][0]["tool_name"] == "DrugProps_calculate_qed"


def test_stage2_accepts_parsed_dict_response():
    provider = _provider_with_responses(
        [
            SimpleNamespace(
                parsed={
                    "arguments": {"smiles": "CCO"},
                    "argument_construction_reason": "context",
                    "missing_fields": [],
                }
            )
        ]
    )

    out = provider.generate_json(
        "args",
        schema={"task": "tool_selection_stage_2", "tool_name": "DrugProps_calculate_qed"},
    )

    assert out["arguments"] == {"smiles": "CCO"}


def test_structured_query_extracts_fenced_json_and_fills_optional_lists():
    provider = _provider_with_responses(
        [
            SimpleNamespace(
                text=(
                    "```json\n"
                    '{"task_intent":{"task_type":"adc_design"},'
                    '"mentioned_entities":{"target_or_antigen_text":"HER2"}}'
                    "\n```"
                )
            )
        ]
    )

    out = provider.generate_json("parse", schema={"raw_request_record": {"raw_user_query": "HER2"}})

    assert out["task_intent"]["task_type"] == "adc_design"
    assert out["mentioned_entities"]["target_or_antigen_text"] == "HER2"
    assert out["referenced_inputs"] == []
    assert out["parse_warnings"] == []


def test_structured_query_promotes_component_name_alias():
    provider = _provider_with_responses(
        [
            SimpleNamespace(
                text=(
                    '{"task_intent":{"task_type":"adc_design"},'
                    '"mentioned_entities":{"payload_text":"vc-MMAE"},'
                    '"entity_decompositions":[{"original_text":"vc-MMAE",'
                    '"components":[{"component_name":"valine-citrulline",'
                    '"component_type":"linker","inferred":true}]}]}'
                )
            )
        ]
    )

    out = provider.generate_json(
        "parse", schema={"task": "structured_query", "raw_request_record": {}}
    )

    comp = out["entity_decompositions"][0]["components"][0]
    assert comp["canonical_name"] == "valine-citrulline"
    assert comp["component_type"] == "linker"
    assert comp["role"] == "linker"


def test_malformed_json_retries_and_returns_corrected_response():
    provider = _provider_with_responses(
        [
            SimpleNamespace(text='{"selections": ['),
            SimpleNamespace(text='{"selections": [{"tool_name": "ZINC_search_by_smiles"}]}'),
        ]
    )

    out = provider.generate_json("pick", schema={"task": "tool_selection_stage_1"})

    assert out["selections"][0]["tool_name"] == "ZINC_search_by_smiles"


def test_invalid_shape_raises_clear_error_after_retries():
    provider = _provider_with_responses(
        [
            SimpleNamespace(text='{"arguments": {}}'),
            SimpleNamespace(text='{"arguments": {}}'),
            SimpleNamespace(text='{"arguments": {}}'),
        ]
    )

    with pytest.raises(GeminiProviderError, match="tool_selection_stage_1.*selections"):
        provider.generate_json("pick", schema={"task": "tool_selection_stage_1"})


# ── Step 6 per-candidate Stage 1 / Stage 2 task shapes ──────────────────────


def _provider_with_response(payload_text: str) -> GeminiProvider:
    return _provider_with_responses([SimpleNamespace(text=payload_text)])


def test_stage1_multi_lane_accepts_valid_payload():
    provider = _provider_with_response(
        '{"selections": ['
        '{"lane_type": "payload_linker_compound_liability",'
        ' "tool_name": "DrugProps_pains_filter",'
        ' "selection_reason": "smiles present",'
        ' "required_context": ["smiles"]},'
        '{"lane_type": "structure_interface_quality",'
        ' "tool_name": "ProteinsPlus_profile_structure_quality"}],'
        '"selection_metadata": {"strategy": "test"}}'
    )
    out = provider.generate_json(
        "pick", schema={"task": "tool_selection_stage_1_multi_lane"}
    )
    assert out["selections"][0]["lane_type"] == "payload_linker_compound_liability"
    assert out["selections"][0]["tool_name"] == "DrugProps_pains_filter"


def test_stage1_multi_lane_does_not_fall_into_structured_query_validator():
    """Regression: prior to the fix, the multi-lane task fell through to the
    structured_query validator and raised about a missing `task_intent`.
    """
    provider = _provider_with_responses(
        # Burn the 3 retries with the same valid Stage 1 multi-lane payload —
        # validation should pass on the first attempt; no `task_intent`
        # complaint should ever surface.
        [SimpleNamespace(text='{"selections": []}')]
    )
    out = provider.generate_json(
        "pick", schema={"task": "tool_selection_stage_1_multi_lane"}
    )
    assert out["selections"] == []


def test_stage1_multi_lane_rejects_non_list_selections():
    provider = _provider_with_responses(
        [SimpleNamespace(text='{"selections": "not-a-list"}')] * 3
    )
    with pytest.raises(GeminiProviderError, match="multi_lane.*selections"):
        provider.generate_json(
            "pick", schema={"task": "tool_selection_stage_1_multi_lane"}
        )


def test_stage1_multi_lane_rejects_selection_missing_lane_type():
    provider = _provider_with_responses(
        [SimpleNamespace(text='{"selections": [{"tool_name": "DrugProps_pains_filter"}]}')] * 3
    )
    with pytest.raises(GeminiProviderError, match="multi_lane.*lane_type"):
        provider.generate_json(
            "pick", schema={"task": "tool_selection_stage_1_multi_lane"}
        )


def test_stage1_multi_lane_rejects_selection_missing_tool_name():
    provider = _provider_with_responses(
        [SimpleNamespace(text='{"selections": [{"lane_type": "payload_linker_compound_liability"}]}')] * 3
    )
    with pytest.raises(GeminiProviderError, match="multi_lane.*tool_name"):
        provider.generate_json(
            "pick", schema={"task": "tool_selection_stage_1_multi_lane"}
        )


def test_stage1_multi_lane_rejects_non_list_required_context():
    provider = _provider_with_responses(
        [SimpleNamespace(text=(
            '{"selections":[{"lane_type":"payload_linker_compound_liability",'
            '"tool_name":"DrugProps_pains_filter",'
            '"required_context":"smiles"}]}'
        ))] * 3
    )
    with pytest.raises(GeminiProviderError, match="multi_lane.*required_context"):
        provider.generate_json(
            "pick", schema={"task": "tool_selection_stage_1_multi_lane"}
        )


def test_stage2_multi_tool_accepts_valid_payload():
    provider = _provider_with_response(
        '{"tools": ['
        '{"lane_type": "payload_linker_compound_liability",'
        ' "tool_name": "DrugProps_pains_filter",'
        ' "arguments": {"smiles": "CCO"},'
        ' "argument_construction_reason": "smiles from context",'
        ' "missing_fields": []}]}'
    )
    out = provider.generate_json(
        "args", schema={"task": "tool_selection_stage_2_multi_tool"}
    )
    assert out["tools"][0]["arguments"] == {"smiles": "CCO"}


def test_stage2_multi_tool_rejects_non_list_tools():
    provider = _provider_with_responses(
        [SimpleNamespace(text='{"tools": "not-a-list"}')] * 3
    )
    with pytest.raises(GeminiProviderError, match="multi_tool.*tools"):
        provider.generate_json(
            "args", schema={"task": "tool_selection_stage_2_multi_tool"}
        )


def test_stage2_multi_tool_rejects_entry_with_non_object_arguments():
    provider = _provider_with_responses(
        [SimpleNamespace(text=(
            '{"tools":[{"lane_type":"payload_linker_compound_liability",'
            '"tool_name":"DrugProps_pains_filter",'
            '"arguments":"not-an-object"}]}'
        ))] * 3
    )
    with pytest.raises(GeminiProviderError, match="multi_tool.*arguments"):
        provider.generate_json(
            "args", schema={"task": "tool_selection_stage_2_multi_tool"}
        )


# ── Step 6 schema mapping task shapes ──────────────────────────────────────


def test_step6_schema_mapping_stage1_accepts_valid_payload_without_task_intent():
    provider = _provider_with_response(
        '{"selections":[{"tool_name":"DrugProps_pains_filter",'
        '"selection_reason":"smiles"}]}'
    )
    out = provider.generate_json(
        "pick", schema={"task": "step6_schema_mapping_stage_1"}
    )
    assert out["selections"][0]["tool_name"] == "DrugProps_pains_filter"


def test_step6_schema_mapping_stage2_accepts_valid_payload_without_task_intent():
    provider = _provider_with_response(
        '{"tools":[{"tool_name":"DrugProps_pains_filter",'
        '"can_invoke":true,'
        '"argument_mapping":{"smiles":"candidate:c1:material:m1:value"},'
        '"missing_required_fields":[],'
        '"argument_mapping_reason":"mapped"}]}'
    )
    out = provider.generate_json(
        "map", schema={"task": "step6_schema_mapping_stage_2"}
    )
    assert out["tools"][0]["can_invoke"] is True


def test_step6_schema_mapping_stage1_rejects_non_list_selections():
    provider = _provider_with_responses(
        [SimpleNamespace(text='{"selections":"not-a-list"}')] * 3
    )
    with pytest.raises(GeminiProviderError, match="step6_schema_mapping_stage_1.*selections"):
        provider.generate_json(
            "pick", schema={"task": "step6_schema_mapping_stage_1"}
        )


def test_step6_schema_mapping_stage2_rejects_malformed_tools():
    provider = _provider_with_responses(
        [SimpleNamespace(text=(
            '{"tools":[{"tool_name":"DrugProps_pains_filter",'
            '"can_invoke":"yes","argument_mapping":{},'
            '"missing_required_fields":[]}]}'
        ))] * 3
    )
    with pytest.raises(GeminiProviderError, match="step6_schema_mapping_stage_2.*can_invoke"):
        provider.generate_json(
            "map", schema={"task": "step6_schema_mapping_stage_2"}
        )


def test_stage2_multi_tool_rejects_entry_missing_lane_type_or_tool_name():
    provider = _provider_with_responses(
        [SimpleNamespace(text='{"tools":[{"arguments":{}}]}')] * 3
    )
    with pytest.raises(GeminiProviderError, match="multi_tool.*lane_type"):
        provider.generate_json(
            "args", schema={"task": "tool_selection_stage_2_multi_tool"}
        )


# ── usage event recording ──────────────────────────────────────────────────


def _response_with_usage(
    text: str, *, prompt: int | None = None,
    completion: int | None = None, total: int | None = None,
) -> SimpleNamespace:
    usage = SimpleNamespace(
        prompt_token_count=prompt,
        candidates_token_count=completion,
        total_token_count=total,
    )
    return SimpleNamespace(text=text, usage_metadata=usage)


def test_gemini_usage_events_record_token_counts_from_usage_metadata():
    provider = _provider_with_responses([
        _response_with_usage(
            '{"task_intent":{"task_type":"adc_design"},'
            '"mentioned_entities":{}}',
            prompt=200, completion=80, total=280,
        )
    ])
    provider.generate_json("parse", schema={"raw_request_record": {}})
    assert len(provider.usage_events) == 1
    evt = provider.usage_events[0]
    assert evt["provider"] == "gemini"
    assert evt["model"] == provider.model
    assert evt["task"] == "structured_query"
    assert evt["attempt"] == 0
    # Gemini exposes prompt_token_count / candidates_token_count /
    # total_token_count; we project to the OpenAI-shaped names.
    assert evt["prompt_tokens"] == 200
    assert evt["completion_tokens"] == 80
    assert evt["total_tokens"] == 280


def test_gemini_usage_events_recorded_for_each_retry_attempt():
    provider = _provider_with_responses([
        _response_with_usage(
            "not json", prompt=11, completion=2, total=13,
        ),
        _response_with_usage(
            '{"task_intent":{"task_type":"adc_design"},"mentioned_entities":{}}',
            prompt=33, completion=4, total=37,
        ),
    ])
    provider.generate_json("parse", schema={"raw_request_record": {}})
    assert [e["attempt"] for e in provider.usage_events] == [0, 1]
    assert provider.usage_events[0]["total_tokens"] == 13
    assert provider.usage_events[1]["total_tokens"] == 37


def test_gemini_usage_event_degrades_to_null_when_no_usage_metadata():
    provider = _provider_with_responses([
        SimpleNamespace(text='{"task_intent":{"task_type":"adc_design"},"mentioned_entities":{}}')
        # no usage_metadata at all
    ])
    provider.generate_json("parse", schema={"raw_request_record": {}})
    assert len(provider.usage_events) == 1
    evt = provider.usage_events[0]
    assert evt["prompt_tokens"] is None
    assert evt["completion_tokens"] is None
    assert evt["total_tokens"] is None


def test_gemini_usage_event_reads_cached_content_token_count_when_present():
    usage = SimpleNamespace(
        prompt_token_count=900,
        candidates_token_count=80,
        total_token_count=980,
        cached_content_token_count=300,
    )
    response = SimpleNamespace(
        text='{"task_intent":{"task_type":"adc_design"},"mentioned_entities":{}}',
        usage_metadata=usage,
    )
    provider = _provider_with_responses([response])
    provider.generate_json("parse", schema={"raw_request_record": {}})
    evt = provider.usage_events[0]
    assert evt["cached_prompt_tokens"] == 300
    assert evt["prompt_tokens"] == 900


def test_gemini_usage_event_cached_prompt_tokens_none_when_missing():
    """Older google-genai SDK versions don't expose
    ``cached_content_token_count``. The field must degrade to None,
    not raise."""
    usage = SimpleNamespace(
        prompt_token_count=400,
        candidates_token_count=20,
        total_token_count=420,
    )
    response = SimpleNamespace(
        text='{"task_intent":{"task_type":"adc_design"},"mentioned_entities":{}}',
        usage_metadata=usage,
    )
    provider = _provider_with_responses([response])
    provider.generate_json("parse", schema={"raw_request_record": {}})
    evt = provider.usage_events[0]
    assert evt["cached_prompt_tokens"] is None
    # Shape parity with OpenAI: same 8 keys must be present.
    assert set(evt.keys()) == {
        "provider", "model", "task", "attempt",
        "prompt_tokens", "completion_tokens", "total_tokens",
        "cached_prompt_tokens",
    }


def test_gemini_usage_events_carry_no_prompt_response_or_api_key():
    secret_prompt = "Gemini PROMPT body with HER2 SECRET payload"
    secret_response = (
        '{"task_intent":{"task_type":"adc_design"},'
        '"mentioned_entities":{"target_or_antigen_text":"HER2_SECRET"}}'
    )
    provider = GeminiProvider(api_key="gemini-very-secret-key", max_retries=0)

    def _fake_generate_content(prompt: str) -> object:
        return _response_with_usage(secret_response, prompt=10, completion=5, total=15)

    provider._generate_content = _fake_generate_content  # type: ignore[method-assign]
    provider.generate_json(secret_prompt, schema={"raw_request_record": {}})
    assert len(provider.usage_events) == 1
    import json as _json
    blob = _json.dumps(provider.usage_events[0])
    assert "gemini-very-secret-key" not in blob
    assert "PROMPT body" not in blob
    assert "HER2_SECRET" not in blob


# ── Gemini path STILL embeds system block into the user prompt body ──────


def test_gemini_user_prompt_body_still_includes_system_block():
    """The Gemini provider's ``_generate_content`` signature takes only
    a prompt today — there is no separate role=system message. The
    user message body therefore MUST still include the system block,
    otherwise the model loses the system instructions entirely.

    If Gemini ever gains a real system-role channel, this test will
    intentionally need to be updated alongside that change."""
    sentinel_system = "SENTINEL_SYSTEM_BLOCK_GEMINI_only"
    user_prompt_body = "USER_PROMPT_SENTINEL_for_gemini"
    captured: dict = {}

    provider = GeminiProvider(api_key="gemini-fake-dedup", max_retries=0)

    def _fake_generate_content(prompt: str) -> object:
        captured["prompt"] = prompt
        return SimpleNamespace(
            text=('{"task_intent":{"task_type":"adc_design"},'
                  '"mentioned_entities":{}}'),
        )

    provider._generate_content = _fake_generate_content  # type: ignore[method-assign]
    provider.generate_json(
        user_prompt_body,
        schema={"raw_request_record": {"raw_user_query": "HER2"}},
        system=sentinel_system,
    )
    # System sentinel MUST appear in the user prompt body — Gemini has
    # no separate system-role channel here.
    assert sentinel_system in captured["prompt"]
    # And the user task body is still present too.
    assert user_prompt_body in captured["prompt"]
