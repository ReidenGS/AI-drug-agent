"""GeminiProvider JSON generation tests.

These tests monkeypatch the provider's model call and never hit the network.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.llm.gemini_provider import GeminiProvider, GeminiProviderError


def _provider_with_responses(responses: list[object]) -> GeminiProvider:
    provider = GeminiProvider(api_key="fake-key", max_retries=2)
    calls = iter(responses)

    def _fake_generate_content(prompt: str) -> object:
        return next(calls)

    provider._generate_content = _fake_generate_content  # type: ignore[method-assign]
    return provider


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
