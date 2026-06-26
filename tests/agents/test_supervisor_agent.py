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


# ── Step 2 LLM schema-drift coercion (Gemini-like malformed payloads) ────────


class _DriftedLLM:
    """Fake provider that emits Step 2 payloads exhibiting real Gemini drift."""

    name = "drifted"
    model = "test"

    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def generate(self, prompt: str, *, system=None, **kw):  # pragma: no cover
        raise NotImplementedError

    def generate_json(self, prompt: str, *, schema: dict, system=None) -> dict:
        return self.payload


def _drifted_payload(**overrides):
    base = {
        "task_intent": {
            "task_type": "adc_design",
            "primary_intent": "developability_assessment",
        },
        "mentioned_entities": {"target_or_antigen_text": "HER2"},
        "referenced_inputs": [],
        "requested_outputs": ["developability_summary"],
        "user_constraints": [],
        "parse_warnings": [],
        "normalized_entities": [],
        "entity_decompositions": [],
        "clarification_questions": [],
    }
    base.update(overrides)
    return base


def test_supervisor_coerces_parse_warnings_dicts_to_strings():
    """Real Gemini sometimes returns parse_warnings as a list of dicts.
    The schema requires list[str]; SupervisorAgent must compact-stringify
    dict entries instead of crashing Pydantic validation."""
    drift = _drifted_payload(parse_warnings=[
        {"warning_code": "MISSING_TARGET", "message": "no target supplied", "confidence": 0.6},
        {"message": "ambiguous payload"},
        "kept as-is",
    ])
    agent = SupervisorAgent(llm=_DriftedLLM(drift))
    sq = agent.parse_raw_to_structured_query(_raw())
    assert isinstance(sq.parse_warnings, list)
    assert all(isinstance(w, str) for w in sq.parse_warnings)
    assert any("MISSING_TARGET" in w for w in sq.parse_warnings)
    assert any("no target supplied" in w for w in sq.parse_warnings)
    assert any("ambiguous payload" in w for w in sq.parse_warnings)
    assert any(w == "kept as-is" for w in sq.parse_warnings)


def test_supervisor_coerces_user_constraints_strings_to_dicts():
    """Real Gemini sometimes returns user_constraints as a list of strings.
    The schema requires list[dict]; SupervisorAgent must wrap each string
    in a compact constraint dict."""
    drift = _drifted_payload(user_constraints=[
        "no PBD payloads",
        "DAR<=4",
        {"constraint_text": "already a dict", "source": "llm"},
    ])
    agent = SupervisorAgent(llm=_DriftedLLM(drift))
    sq = agent.parse_raw_to_structured_query(_raw())
    assert isinstance(sq.user_constraints, list)
    assert all(isinstance(c, dict) for c in sq.user_constraints)
    texts = [c.get("constraint_text") for c in sq.user_constraints]
    assert "no PBD payloads" in texts
    assert "DAR<=4" in texts
    assert "already a dict" in texts
    # Source attribution preserved / injected
    for c in sq.user_constraints:
        if c.get("constraint_text") in ("no PBD payloads", "DAR<=4"):
            assert c.get("source") in {"llm_output", "llm", "supervisor_coerced"}


def test_supervisor_handles_mixed_unknown_types_without_crashing():
    drift = _drifted_payload(
        parse_warnings=[123, None, {"message": "ok"}, ["nested", "list"]],
        user_constraints=[42, None, "valid_text", {"constraint_text": "ok"}],
    )
    agent = SupervisorAgent(llm=_DriftedLLM(drift))
    sq = agent.parse_raw_to_structured_query(_raw())
    # No crash, all entries valid types.
    assert all(isinstance(w, str) for w in sq.parse_warnings)
    assert all(isinstance(c, dict) for c in sq.user_constraints)
    # The valid entries survived.
    assert any("ok" in w for w in sq.parse_warnings)
    assert any(c.get("constraint_text") == "valid_text" for c in sq.user_constraints)
    assert any(c.get("constraint_text") == "ok" for c in sq.user_constraints)


def test_supervisor_does_not_regress_mock_llm_path():
    """MockLLMProvider returns string parse_warnings and dict user_constraints
    today; the coercer must be a no-op for already-conformant payloads."""
    agent = SupervisorAgent(llm=MockLLMProvider())
    sq = agent.parse_raw_to_structured_query(_raw())
    assert all(isinstance(w, str) for w in sq.parse_warnings)
    assert all(isinstance(c, dict) for c in sq.user_constraints)


def test_supervisor_does_not_store_raw_llm_response_or_prompt():
    """Defensive: structured_query artifact must not embed full prompt /
    raw LLM payload. We assert on the model dump rather than disk because
    the agent constructs the artifact deterministically."""
    sentinel_prompt = "SECRET_PROMPT_BODY_DO_NOT_LEAK"
    sentinel_raw = "SECRET_RAW_LLM_BODY_DO_NOT_LEAK"

    class _SentinelLLM(_DriftedLLM):
        def generate_json(self, prompt: str, *, schema: dict, system=None) -> dict:
            # Drop the sentinel into the payload's free-text fields the
            # supervisor explicitly does NOT propagate.
            return _drifted_payload(
                parse_warnings=[f"warning containing {sentinel_raw}"],
                clarification_questions=[f"contains {sentinel_raw}"],
            )

    raw = _raw()
    raw["raw_user_query"] = sentinel_prompt
    agent = SupervisorAgent(llm=_SentinelLLM({}))
    sq = agent.parse_raw_to_structured_query(raw)
    blob = sq.model_dump_json()
    # SECRET_RAW comes from the LLM's parse_warnings entry; that's expected to
    # surface (warnings are user-visible). The PROMPT sentinel is the user's
    # own query — also expected. The point of this test: the supervisor must
    # not invent OTHER raw-LLM dumping channels beyond the documented fields.
    # i.e. there's no `raw_llm_response`, `prompt_inputs`, or `full_prompt`
    # field accidentally created on the artifact.
    forbidden_keys = {"raw_llm_response", "prompt_inputs", "full_prompt", "llm_payload"}
    for k in forbidden_keys:
        assert f'"{k}":' not in blob, f"structured_query must not expose `{k}`"


def test_supervisor_promotes_component_name_to_canonical_name():
    drift = _drifted_payload(
        entity_decompositions=[
            {
                "original_text": "vc-MMAE",
                "components": [
                    {
                        "component_name": "valine-citrulline",
                        "component_type": "linker",
                        "inferred": True,
                    }
                ],
            }
        ]
    )
    agent = SupervisorAgent(llm=_DriftedLLM(drift))
    sq = agent.parse_raw_to_structured_query(_raw())
    comp = sq.entity_decompositions[0].components[0]
    assert comp.canonical_name == "valine-citrulline"
    assert comp.component_type == "linker"
    assert comp.role == "linker"
    assert comp.inferred is True


def test_supervisor_keeps_existing_component_canonical_name():
    drift = _drifted_payload(
        entity_decompositions=[
            {
                "original_text": "T-DM1",
                "components": [
                    {
                        "canonical_name": "emtansine",
                        "component_name": "wrong alias",
                        "role": "payload",
                    }
                ],
            }
        ]
    )
    agent = SupervisorAgent(llm=_DriftedLLM(drift))
    sq = agent.parse_raw_to_structured_query(_raw())
    comp = sq.entity_decompositions[0].components[0]
    assert comp.canonical_name == "emtansine"
    assert comp.role == "payload"


def test_supervisor_promotes_component_name_label_value_aliases():
    drift = _drifted_payload(
        entity_decompositions=[
            {
                "original_text": "multi-component ADC",
                "components": [
                    {"name": "trastuzumab", "component_type": "antibody"},
                    {"label": "deruxtecan", "component_type": "linker_payload"},
                    {"value": "DXd", "component_type": "payload", "source": "llm"},
                ],
            }
        ]
    )
    agent = SupervisorAgent(llm=_DriftedLLM(drift))
    sq = agent.parse_raw_to_structured_query(_raw())
    comps = sq.entity_decompositions[0].components
    assert [c.canonical_name for c in comps] == ["trastuzumab", "deruxtecan", "DXd"]
    assert [c.role for c in comps] == ["antibody", "linker_payload", "payload"]
    assert comps[2].source == "llm"


def test_supervisor_drops_components_without_usable_name_and_warns():
    drift = _drifted_payload(
        entity_decompositions=[
            {
                "original_text": "bad decomposition",
                "components": [
                    {},
                    {"component_name": "   ", "component_type": "payload"},
                    ["not", "a", "dict"],
                    {"canonical_name": "MMAE", "component_type": "payload"},
                ],
            }
        ]
    )
    agent = SupervisorAgent(llm=_DriftedLLM(drift))
    sq = agent.parse_raw_to_structured_query(_raw())
    assert [c.canonical_name for c in sq.entity_decompositions[0].components] == ["MMAE"]
    warnings = " | ".join(sq.parse_warnings)
    assert "components[0]" in warnings and "missing canonical_name" in warnings
    assert "components[1]" in warnings and "missing canonical_name" in warnings
    assert "components[2]" in warnings and "expected object" in warnings
    blob = sq.model_dump_json()
    for k in ("raw_llm_response", "full_prompt", "prompt_inputs"):
        assert f'"{k}":' not in blob


def test_supervisor_normalizes_entity_type_aliases_before_pydantic():
    drift = _drifted_payload(
        normalized_entities=[
            {
                "original_text": "vc-MMAE",
                "canonical_name": "vc-MMAE",
                "entity_type": "payload-linker",
                "explicit_or_inferred": "explicit",
            },
            {
                "original_text": "small molecule X",
                "canonical_name": "small molecule X",
                "entity_type": "small molecule",
                "explicit_or_inferred": "explicit",
            },
            {
                "original_text": "trastuzumab analog",
                "canonical_name": "trastuzumab analog",
                "entity_type": "antibody_candidate",
                "explicit_or_inferred": "explicit",
            },
        ]
    )
    agent = SupervisorAgent(llm=_DriftedLLM(drift))
    sq = agent.parse_raw_to_structured_query(_raw())
    assert [e.entity_type for e in sq.normalized_entities] == [
        "linker_payload",
        "compound",
        "antibody",
    ]
    warnings = " | ".join(sq.parse_warnings)
    assert "payload-linker" in warnings
    assert "linker_payload" in warnings
    assert "antibody_candidate" in warnings


def test_supervisor_repairs_explicit_payload_smiles_in_payload_text():
    raw = _raw()
    raw["raw_user_query"] = (
        "Evaluate HER2 ADC with payload-linker vc-MMAE. payload SMILES CCO."
    )
    drift = _drifted_payload(
        mentioned_entities={"target_or_antigen_text": "HER2", "payload_text": "CCO"}
    )
    agent = SupervisorAgent(llm=_DriftedLLM(drift))
    sq = agent.parse_raw_to_structured_query(raw)
    assert sq.mentioned_entities.payload_text is None
    assert {
        "id_type": "smiles",
        "value": "CCO",
        "source": "payload_smiles",
    } in sq.referenced_inputs
    assert any("payload_text" in w and "payload_smiles" in w for w in sq.parse_warnings)


def test_supervisor_repairs_explicit_linker_smiles_in_linker_text():
    raw = _raw()
    raw["raw_user_query"] = (
        "Evaluate HER2 ADC with vc-MMAE. linker SMILES NCC(=O)O."
    )
    drift = _drifted_payload(
        mentioned_entities={
            "target_or_antigen_text": "HER2",
            "linker_text": "NCC(=O)O",
        }
    )
    agent = SupervisorAgent(llm=_DriftedLLM(drift))
    sq = agent.parse_raw_to_structured_query(raw)
    assert sq.mentioned_entities.linker_text is None
    assert {
        "id_type": "smiles",
        "value": "NCC(=O)O",
        "source": "linker_smiles",
    } in sq.referenced_inputs
    assert not any(r.get("value") == "NCC(=O)O)" for r in sq.referenced_inputs)
    assert any("linker_text" in w and "linker_smiles" in w for w in sq.parse_warnings)


def test_supervisor_extracts_parenthetical_labeled_smiles_without_closing_context_paren():
    raw = _raw()
    raw["raw_user_query"] = "Evaluate HER2 ADC with vc-MMAE."
    raw["user_provided_context"]["payload_linker_text"] = (
        "vc-MMAE (payload SMILES CCO; linker SMILES NCC(=O)O)"
    )
    drift = _drifted_payload(
        mentioned_entities={
            "target_or_antigen_text": "HER2",
            "payload_text": "vc-MMAE",
            "linker_text": "vc-MMAE",
        }
    )
    agent = SupervisorAgent(llm=_DriftedLLM(drift))
    sq = agent.parse_raw_to_structured_query(raw)
    refs = {
        (r.get("id_type"), r.get("value"), r.get("source"))
        for r in sq.referenced_inputs
    }
    assert ("smiles", "CCO", "payload_smiles") in refs
    assert ("smiles", "NCC(=O)O", "linker_smiles") in refs
    assert ("smiles", "NCC(=O)O)", "linker_smiles") not in refs


def test_supervisor_cleans_typed_smiles_after_normalized_entity_backfill():
    raw = _raw()
    raw["raw_user_query"] = "Assess HER2 ADC. payload SMILES CCO."
    drift = _drifted_payload(
        mentioned_entities={"target_or_antigen_text": "HER2"},
        normalized_entities=[
            {
                "original_text": "CCO",
                "canonical_name": "CCO",
                "entity_type": "payload",
                "explicit_or_inferred": "explicit",
            }
        ],
    )
    agent = SupervisorAgent(llm=_DriftedLLM(drift))
    sq = agent.parse_raw_to_structured_query(raw)
    assert sq.mentioned_entities.payload_text is None
    assert {
        "id_type": "smiles",
        "value": "CCO",
        "source": "payload_smiles",
    } in sq.referenced_inputs


def test_supervisor_preserves_real_payload_and_linker_names_with_typed_smiles():
    raw = _raw()
    raw["raw_user_query"] = (
        "Evaluate HER2 ADC with payload MMAE and linker vc-MMAE. "
        "payload SMILES CCO. linker SMILES NCC(=O)O."
    )
    drift = _drifted_payload(
        mentioned_entities={
            "target_or_antigen_text": "HER2",
            "payload_text": "MMAE",
            "linker_text": "vc-MMAE",
        }
    )
    agent = SupervisorAgent(llm=_DriftedLLM(drift))
    sq = agent.parse_raw_to_structured_query(raw)
    assert sq.mentioned_entities.payload_text == "MMAE"
    assert sq.mentioned_entities.linker_text == "vc-MMAE"
    refs = {
        (r.get("id_type"), r.get("value"), r.get("source"))
        for r in sq.referenced_inputs
    }
    assert ("smiles", "CCO", "payload_smiles") in refs
    assert ("smiles", "NCC(=O)O", "linker_smiles") in refs


def test_supervisor_preserves_sn38_carbonate_name_with_typed_smiles():
    raw = _raw()
    raw["raw_user_query"] = (
        "Evaluate a TROP2 ADC with linker-payload SN-38 carbonate. "
        "Payload SMILES C1=CC=C2C(=C1)C(=O)N(C)C3=CC=CC=C23."
    )
    drift = _drifted_payload(
        mentioned_entities={
            "target_or_antigen_text": "TROP2",
            "payload_text": "SN-38 carbonate",
        }
    )
    agent = SupervisorAgent(llm=_DriftedLLM(drift))
    sq = agent.parse_raw_to_structured_query(raw)
    assert sq.mentioned_entities.payload_text == "SN-38 carbonate"
    assert {
        "id_type": "smiles",
        "value": "C1=CC=C2C(=C1)C(=O)N(C)C3=CC=CC=C23",
        "source": "payload_smiles",
    } in sq.referenced_inputs
