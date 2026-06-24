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


def test_stage2_multi_tool_rejects_entry_missing_lane_type_or_tool_name():
    provider = _provider_with_responses(
        [SimpleNamespace(text='{"tools":[{"arguments":{}}]}')] * 3
    )
    with pytest.raises(GeminiProviderError, match="multi_tool.*lane_type"):
        provider.generate_json(
            "args", schema={"task": "tool_selection_stage_2_multi_tool"}
        )
