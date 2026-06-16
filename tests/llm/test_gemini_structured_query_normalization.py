"""GeminiProvider.structured_query requested_outputs normalization.

These tests monkeypatch the provider's model call and never hit the network.
They verify that Gemini outputs which don't match the canonical Step 2 enum
are coerced into compliant `list[str]` values, with anything unmappable
recorded under `parse_warnings` instead of raising.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

from app.llm.gemini_provider import GeminiProvider


def _provider_with_response(payload: dict) -> GeminiProvider:
    provider = GeminiProvider(api_key="fake-key", max_retries=0)
    response = SimpleNamespace(text=json.dumps(payload))

    def _fake_generate_content(prompt: str) -> object:
        return response

    provider._generate_content = _fake_generate_content  # type: ignore[method-assign]
    return provider


def _base_payload(requested_outputs: list[object]) -> dict:
    return {
        "task_intent": {"task_type": "adc_design"},
        "mentioned_entities": {},
        "referenced_inputs": [],
        "requested_outputs": requested_outputs,
        "user_constraints": [],
        "parse_warnings": [],
    }


def test_structured_query_maps_adc_candidate_entity_dict():
    """Case A: dict shape `{entity_type: adc_candidate}` → mapped to canonical."""
    provider = _provider_with_response(
        _base_payload([{"entity_type": "adc_candidate"}])
    )

    out = provider.generate_json("parse", schema={"task": "structured_query"})

    assert out["requested_outputs"] == ["ranked_candidates"]
    assert out["parse_warnings"] == []


def test_structured_query_preserves_canonical_string_value():
    """Case B: legal canonical string passes through untouched, no warning."""
    provider = _provider_with_response(
        _base_payload(["evidence_summary"])
    )

    out = provider.generate_json("parse", schema={"task": "structured_query"})

    assert out["requested_outputs"] == ["evidence_summary"]
    assert out["parse_warnings"] == []


def test_structured_query_drops_unknown_entity_with_warning():
    """Case C: fully unknown entity_type is dropped + warning recorded."""
    provider = _provider_with_response(
        _base_payload([{"entity_type": "totally_unknown_xyz"}])
    )

    out = provider.generate_json("parse", schema={"task": "structured_query"})

    assert out["requested_outputs"] == []
    assert len(out["parse_warnings"]) == 1
    warning = out["parse_warnings"][0]
    assert "totally_unknown_xyz" in warning
    assert "dropped" in warning


def test_structured_query_dedupes_after_alias_mapping():
    """Both `adc_candidate` and `ranked_candidates` collapse to one entry."""
    provider = _provider_with_response(
        _base_payload(
            [
                {"entity_type": "adc_candidate"},
                "ranked_candidates",
                "evidence_summary",
            ]
        )
    )

    out = provider.generate_json("parse", schema={"task": "structured_query"})

    assert out["requested_outputs"] == ["ranked_candidates", "evidence_summary"]
    assert out["parse_warnings"] == []


def test_structured_query_handles_mixed_legal_and_garbage():
    """Legal entries kept, garbage dropped with one warning each."""
    provider = _provider_with_response(
        _base_payload(
            [
                "report",
                {"entity_type": "totally_unknown_xyz"},
                42,
            ]
        )
    )

    out = provider.generate_json("parse", schema={"task": "structured_query"})

    assert out["requested_outputs"] == ["report"]
    assert len(out["parse_warnings"]) == 2
