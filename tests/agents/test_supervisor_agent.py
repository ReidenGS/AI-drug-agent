from __future__ import annotations

from app.agents.supervisor_agent import SupervisorAgent
from app.llm.provider import MockLLMProvider
from app.schemas.step_01_raw_request_record import (
    RawRequestRecord,
    UserProvidedContext,
)


# Fixture value; not derived from `new_run_id()`. The supervisor only needs
# *some* run_id to thread through; we check pass-through equality, not format.
_FIXTURE_RUN_ID = "run_supervisor_fixture_001"


def _raw(target: str | None = "HER2", payload: str | None = "vc-MMAE") -> dict:
    rec = RawRequestRecord(
        run_id=_FIXTURE_RUN_ID,
        run_artifact_registry_id="reg_x",
        created_at="2026-06-15T00:00:00Z",
        raw_user_query="Design an ADC against HER2 with vc-MMAE",
        user_provided_context=UserProvidedContext(
            target_or_antigen_text=target,
            candidate_text="Trastuzumab analog",
            payload_linker_text=payload,
        ),
    )
    out = rec.model_dump()
    out["artifact_id"] = "raw_request_record_test"
    return out


def test_supervisor_extracts_target_and_payload_from_context():
    agent = SupervisorAgent(llm=MockLLMProvider())
    sq = agent.parse_raw_to_structured_query(_raw())
    assert sq.run_id == _FIXTURE_RUN_ID
    assert sq.source_raw_request_ref.raw_request_record_id == "raw_request_record_test"
    assert sq.task_intent.task_type == "adc_design"
    assert sq.mentioned_entities.target_or_antigen_text == "HER2"
    assert sq.mentioned_entities.payload_text is not None


def test_supervisor_marks_warning_when_target_missing():
    agent = SupervisorAgent(llm=MockLLMProvider())
    sq = agent.parse_raw_to_structured_query(_raw(target=None))
    assert sq.mentioned_entities.target_or_antigen_text == "HER2"  # detected from raw_user_query
    sq2 = agent.parse_raw_to_structured_query({
        **_raw(target=None),
        "raw_user_query": "build an antibody-drug conjugate (no target specified)",
        "user_provided_context": {},
    })
    assert sq2.mentioned_entities.target_or_antigen_text is None
    assert any("target" in w for w in sq2.parse_warnings)
