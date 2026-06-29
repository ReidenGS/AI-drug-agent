"""Shared JSON task validation / normalization for Step 2 ``missing_slots``.

These exercise the single source of truth all three providers (OpenAI /
Gemini / Qwen) and the SupervisorAgent boundary share, so drift handling is
identical regardless of which provider produced the payload.
"""

from __future__ import annotations

import pytest

from app.llm.json_task_validation import (
    normalize_missing_slots,
    shape_instruction,
    validate_task_shape,
)


class _Err(Exception):
    pass


def _err(msg: str) -> _Err:
    return _Err(msg)


def _structured_query(**overrides) -> dict:
    base = {
        "task_intent": {"task_type": "adc_design", "primary_intent": "new_adc_design"},
        "mentioned_entities": {},
    }
    base.update(overrides)
    return base


# ── shape hint ──────────────────────────────────────────────────────────────


def test_shape_instruction_documents_missing_slots():
    shape = shape_instruction("structured_query")
    assert "missing_slots" in shape
    assert "slot_name" in shape
    assert "severity" in shape


# ── validator accepts well-formed missing_slots ─────────────────────────────


def test_validator_accepts_valid_missing_slots():
    data = _structured_query(
        missing_slots=[
            {
                "slot_name": "target_or_antigen",
                "slot_category": "target",
                "severity": "blocking",
                "required_for": ["new_adc_design"],
                "reason": "no target",
                "suggested_question": "What target?",
            }
        ]
    )
    out = validate_task_shape(data, "structured_query", error_factory=_err)
    assert out["missing_slots"][0]["slot_name"] == "target_or_antigen"


def test_validator_defaults_absent_missing_slots():
    out = validate_task_shape(_structured_query(), "structured_query", error_factory=_err)
    assert out["missing_slots"] == []


def test_validator_does_not_crash_on_malformed_missing_slots():
    """A single malformed item must not fail the whole structured_query."""
    data = _structured_query(
        missing_slots=[
            {"slot_name": "payload", "severity": "warning"},
            None,
            "free text gap",
            {"slot_name": "weird", "severity": "nope", "required_for": "new_adc_design"},
        ]
    )
    out = validate_task_shape(data, "structured_query", error_factory=_err)
    slots = out["missing_slots"]
    assert all(isinstance(s, dict) for s in slots)
    assert all(s["severity"] in {"blocking", "warning", "optional"} for s in slots)
    # `required_for` string coerced to list (the unknown slot_name → other).
    coerced = [s for s in slots if s["required_for"] == ["new_adc_design"]]
    assert coerced and coerced[0]["slot_name"] == "other"
    # The free-text string entry also became an `other` slot.
    assert any(s["reason"] == "free text gap" for s in slots)


# ── normalize_missing_slots direct ──────────────────────────────────────────


def test_normalize_absent():
    assert normalize_missing_slots({})["missing_slots"] == []


def test_normalize_dict_to_list():
    out = normalize_missing_slots({"missing_slots": {"slot_name": "linker"}})
    assert out["missing_slots"][0]["slot_name"] == "linker"
    assert out["missing_slots"][0]["slot_category"] == "linker"


def test_normalize_string_to_other_slot():
    out = normalize_missing_slots({"missing_slots": "give me a target"})
    assert out["missing_slots"][0]["slot_name"] == "other"
    assert out["missing_slots"][0]["reason"] == "give me a target"


def test_normalize_unknown_enum_coerced_to_safe_defaults():
    out = normalize_missing_slots(
        {
            "missing_slots": [
                {
                    "slot_name": "antigen",  # alias → target_or_antigen
                    "slot_category": "bogus",  # → backfilled from slot_name
                    "severity": "fatal",  # → warning
                }
            ]
        }
    )
    slot = out["missing_slots"][0]
    assert slot["slot_name"] == "target_or_antigen"
    assert slot["slot_category"] == "target"
    assert slot["severity"] == "warning"


def test_normalize_non_list_non_dict_non_str_dropped_with_warning():
    out = normalize_missing_slots({"missing_slots": 42})
    assert out["missing_slots"] == []
    assert any("malformed missing_slots" in w for w in out["parse_warnings"])


def test_normalize_suggested_question_non_string_stringified():
    out = normalize_missing_slots(
        {"missing_slots": [{"slot_name": "payload", "suggested_question": 123}]}
    )
    assert out["missing_slots"][0]["suggested_question"] == "123"


def test_normalize_is_idempotent():
    once = normalize_missing_slots(
        {"missing_slots": [{"slot_name": "payload", "severity": "warning"}]}
    )
    twice = normalize_missing_slots({"missing_slots": list(once["missing_slots"])})
    assert twice["missing_slots"] == once["missing_slots"]


# ── response normalization ───────────────────────────────────────────────────


def test_validator_accepts_string_response():
    data = _structured_query(response="Please provide the target.")
    out = validate_task_shape(data, "structured_query", error_factory=_err)
    assert out["response"] == "Please provide the target."


def test_validator_defaults_absent_response_to_none():
    out = validate_task_shape(_structured_query(), "structured_query", error_factory=_err)
    assert out["response"] is None


def test_validator_does_not_crash_on_non_string_response():
    out = validate_task_shape(
        _structured_query(response={"message": "need target"}),
        "structured_query",
        error_factory=_err,
    )
    assert out["response"] == "need target"


def test_normalize_response_scalar_and_none():
    from app.llm.json_task_validation import normalize_response

    assert normalize_response({})["response"] is None
    assert normalize_response({"response": None})["response"] is None
    assert normalize_response({"response": 7})["response"] == "7"
    assert normalize_response({"response": "  hi  "})["response"] == "hi"


def test_normalize_response_truncates_overlong():
    from app.llm.json_task_validation import normalize_response

    out = normalize_response({"response": "y" * 800})
    assert len(out["response"]) == 500
    assert any("truncated response" in w for w in out["parse_warnings"])
