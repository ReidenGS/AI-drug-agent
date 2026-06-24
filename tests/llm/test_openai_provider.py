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
