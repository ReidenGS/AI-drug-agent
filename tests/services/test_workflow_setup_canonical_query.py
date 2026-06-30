"""Step 4 downstream query consumption prefers canonical_query.

The natural-language working-query signal must read the stable
``canonical_query`` first (fallback to legacy ``task_intent.user_goal_summary``),
and must NOT read any wrong query-like alias field.
"""

from __future__ import annotations

from app.services.workflow_setup_service import _gather_signals


def _readiness() -> dict:
    return {"basic_adc_input_presence": {}}


def test_canonical_query_drives_query_context_when_no_entities():
    sq = {
        "mentioned_entities": {},
        "referenced_inputs": [],
        "task_intent": {"user_goal_summary": ""},
        "canonical_query": "Design a new antibody-drug conjugate targeting HER2.",
    }
    signals = _gather_signals(readiness=_readiness(), structured_query=sq)
    assert signals.has_query_context is True


def test_falls_back_to_user_goal_summary_when_canonical_empty():
    sq = {
        "mentioned_entities": {},
        "referenced_inputs": [],
        "task_intent": {"user_goal_summary": "Some goal text"},
        "canonical_query": None,
    }
    signals = _gather_signals(readiness=_readiness(), structured_query=sq)
    assert signals.has_query_context is True


def test_wrong_alias_field_is_not_read_as_query():
    # A stray alias must NOT count as query context — only canonical_query /
    # user_goal_summary are consulted.
    sq = {
        "mentioned_entities": {},
        "referenced_inputs": [],
        "task_intent": {"user_goal_summary": ""},
        "canonical_query": None,
        "working_query": "Design a HER2 ADC",
        "query_summary": "Design a HER2 ADC",
    }
    signals = _gather_signals(readiness=_readiness(), structured_query=sq)
    assert signals.has_query_context is False
