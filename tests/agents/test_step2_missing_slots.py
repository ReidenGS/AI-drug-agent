"""Step 2 ``missing_slots`` — schema, prompt, normalizer, and mock output.

Covers the structured required-slot gap channel added on top of Step 2:

- ``StructuredQuery`` validates ``missing_slots`` and still accepts old
  artifacts that predate the field.
- The supervisor prompt advertises the ``required_slot_schema`` and tells
  the model ``missing_slots`` is a structured gap channel.
- ``normalize_llm_payload_for_step2`` coerces absent / dict / string /
  malformed-list drift into a clean ``list[dict]`` without crashing.
- ``MockLLMProvider`` emits the typical required-slot gaps per intent and
  does not over-report when an equivalent typed input is present.
- No raw file content / full sequence / API key / prompt leakage.
"""

from __future__ import annotations

from app.agents.supervisor_agent import (
    SUPERVISOR_SYSTEM_PROMPT,
    SupervisorAgent,
    normalize_llm_payload_for_step2,
)
from app.llm.provider import MockLLMProvider
from app.schemas.step_02_structured_query import (
    MissingSlot,
    SourceRawRequestRef,
    StructuredQuery,
    TaskIntent,
)
from app.utils.time import now_iso


# ── schema ──────────────────────────────────────────────────────────────────


def _sq(**overrides) -> StructuredQuery:
    base = dict(
        run_id="run_ms",
        parsed_at=now_iso(),
        source_raw_request_ref=SourceRawRequestRef(raw_request_record_id="reg_ms"),
        task_intent=TaskIntent(task_type="adc_design"),
    )
    base.update(overrides)
    return StructuredQuery(**base)


def test_schema_accepts_missing_slots():
    sq = _sq(
        missing_slots=[
            MissingSlot(
                slot_name="target_or_antigen",
                slot_category="target",
                severity="blocking",
                required_for=["new_adc_design"],
                reason="no target",
                suggested_question="What target?",
            )
        ]
    )
    assert sq.missing_slots[0].slot_name == "target_or_antigen"
    assert sq.missing_slots[0].severity == "blocking"


def test_schema_accepts_conditional_sequence_role_missing_slot():
    sq = _sq(
        missing_slots=[
            MissingSlot(
                slot_name="sequence_role",
                slot_category="sequence",
                severity="blocking",
                required_for=["uploaded_fasta_role_routing"],
                reason="uploaded FASTA role is unclear",
                suggested_question="What role should the uploaded FASTA have?",
            )
        ]
    )
    assert sq.missing_slots[0].slot_name == "sequence_role"
    assert sq.missing_slots[0].slot_category == "sequence"


def test_schema_defaults_missing_slots_empty():
    assert _sq().missing_slots == []


def test_schema_backward_compatible_old_artifact_without_missing_slots():
    """An artifact dumped before the field existed must still validate."""
    payload = _sq().model_dump()
    payload.pop("missing_slots", None)
    assert "missing_slots" not in payload
    restored = StructuredQuery.model_validate(payload)
    assert restored.missing_slots == []


# ── prompt ──────────────────────────────────────────────────────────────────


def test_prompt_advertises_required_slot_schema_and_missing_slots():
    sp = SUPERVISOR_SYSTEM_PROMPT
    assert "required_slot_schema" in sp
    assert "missing_slots" in sp
    # Intent-specific blocking slots are described.
    assert "target_or_antigen" in sp
    assert "structure_or_sequence" in sp
    # The prompt frames missing_slots as a structured gap channel, separate
    # from parse_warnings, and reminds the model not to over-block.
    assert "structured gap channel" in sp
    assert "equivalent typed input" in sp


def test_prompt_describes_conditional_uploaded_fasta_role_slot():
    sp = SUPERVISOR_SYSTEM_PROMPT
    assert "conditional uploaded FASTA role" in sp
    assert "blocking sequence_role ONLY when an uploaded FASTA/sequence file exists" in sp
    assert "Do not emit it when no FASTA exists" in sp


# ── normalizer drift handling ───────────────────────────────────────────────


def test_normalizer_missing_slots_absent_becomes_empty_list():
    out = normalize_llm_payload_for_step2({"task_intent": {"task_type": "x"}})
    assert out["missing_slots"] == []


def test_normalizer_missing_slots_dict_wrapped_to_list():
    out = normalize_llm_payload_for_step2(
        {
            "missing_slots": {
                "slot_name": "payload",
                "severity": "warning",
                "reason": "no payload",
            }
        }
    )
    assert isinstance(out["missing_slots"], list)
    assert out["missing_slots"][0]["slot_name"] == "payload"
    assert out["missing_slots"][0]["slot_category"] == "payload"  # backfilled
    assert any("container to a list" in w for w in out["parse_warnings"])


def test_normalizer_missing_slots_preserves_sequence_role_aliases():
    out = normalize_llm_payload_for_step2(
        {
            "missing_slots": [
                {
                    "slot_name": "fasta_role",
                    "severity": "blocking",
                    "reason": "ambiguous uploaded FASTA",
                }
            ]
        }
    )
    assert out["missing_slots"][0]["slot_name"] == "sequence_role"
    assert out["missing_slots"][0]["slot_category"] == "sequence"


def test_normalizer_promotes_other_sequence_fasta_role_gap():
    out = normalize_llm_payload_for_step2(
        {
            "missing_slots": [
                {
                    "slot_name": "other",
                    "slot_category": "sequence",
                    "severity": "blocking",
                    "reason": "Uploaded FASTA role is unclear.",
                }
            ]
        }
    )
    assert out["missing_slots"][0]["slot_name"] == "sequence_role"
    assert out["missing_slots"][0]["slot_category"] == "sequence"


def test_normalizer_missing_slots_string_becomes_other_slot():
    out = normalize_llm_payload_for_step2({"missing_slots": "need a target"})
    assert out["missing_slots"][0]["slot_name"] == "other"
    assert out["missing_slots"][0]["reason"] == "need a target"


def test_normalizer_missing_slots_malformed_list_entries_dropped_not_crash():
    out = normalize_llm_payload_for_step2(
        {
            "missing_slots": [
                {"slot_name": "target_or_antigen", "severity": "blocking"},
                None,
                123,
                {"slot_name": "totally_unknown", "severity": "explode"},
            ]
        }
    )
    slots = out["missing_slots"]
    # The two object entries survive (unknown enum coerced to safe defaults);
    # None / int are dropped with a compact warning.
    names = [s["slot_name"] for s in slots]
    assert "target_or_antigen" in names
    assert "other" in names  # the unknown slot_name coerced
    assert all(s["severity"] in {"blocking", "warning", "optional"} for s in slots)
    assert any("malformed missing_slots" in w for w in out["parse_warnings"])


def test_normalizer_missing_slots_is_idempotent():
    once = normalize_llm_payload_for_step2(
        {"missing_slots": [{"slot_name": "linker", "severity": "warning"}]}
    )
    twice = normalize_llm_payload_for_step2(dict(once))
    assert twice["missing_slots"] == once["missing_slots"]


# ── mock provider output ────────────────────────────────────────────────────


def _raw(query: str, ctx: dict | None = None, files: list | None = None) -> dict:
    return {
        "run_id": "run_x",
        "run_artifact_registry_id": "reg_x",
        "raw_user_query": query,
        "user_provided_context": ctx or {},
        "uploaded_files": files or [],
    }


def _parse(query: str, ctx: dict | None = None, files: list | None = None) -> dict:
    return MockLLMProvider().generate_json(
        "parse", schema={"raw_request_record": _raw(query, ctx, files)}
    )


def _slots_by_name(out: dict) -> dict[str, dict]:
    return {s["slot_name"]: s for s in out["missing_slots"]}


def test_mock_emits_blocking_target_for_bare_design_request():
    out = _parse("I want to design an ADC")
    slots = _slots_by_name(out)
    assert "target_or_antigen" in slots
    assert slots["target_or_antigen"]["severity"] == "blocking"
    assert slots["target_or_antigen"]["suggested_question"]


def test_mock_does_not_emit_target_missing_when_her2_present():
    out = _parse("Design an ADC against HER2 with MMAE")
    slots = _slots_by_name(out)
    assert "target_or_antigen" not in slots
    # payload satisfied by MMAE → not missing either.
    assert "payload" not in slots


def test_mock_does_not_emit_target_missing_with_uniprot_accession():
    out = _parse("Design an ADC", ctx={"target_or_antigen_text": ""})
    # Use an explicit UniProt accession in the query — satisfies target slot.
    out = _parse("Design an ADC targeting the antigen with UniProt P04626")
    slots = _slots_by_name(out)
    assert "target_or_antigen" not in slots


def test_mock_structure_analysis_without_structure_blocks():
    out = _parse("Run a structure analysis of the antibody-antigen complex")
    assert out["task_intent"]["primary_intent"] == "structure_analysis"
    slots = _slots_by_name(out)
    assert slots["structure_or_sequence"]["severity"] == "blocking"


def test_mock_structure_analysis_satisfied_by_pdb_id():
    out = _parse("Run a structure analysis using PDB 1N8Z")
    slots = _slots_by_name(out)
    assert "structure_or_sequence" not in slots


def test_mock_structure_analysis_satisfied_by_uniprot():
    out = _parse("Run a structure analysis for UniProt P04626")
    slots = _slots_by_name(out)
    assert "structure_or_sequence" not in slots


def test_mock_structure_analysis_satisfied_by_uploaded_pdb_file():
    out = _parse(
        "Run a structure analysis of the attached complex",
        files=[
            {
                "file_id": "f_pdb_1",
                "original_filename": "complex.pdb",
                "content_type": "chemical/x-pdb",
                "sha256": "a" * 64,
                "size_bytes": 1024,
            }
        ],
    )
    slots = _slots_by_name(out)
    assert "structure_or_sequence" not in slots


def test_mock_structure_analysis_satisfied_by_heavy_light_sequence_refs():
    out = _parse(
        "Run a structure analysis of trastuzumab",
        files=[
            {
                "file_id": "f_h",
                "original_filename": "heavy_chain.fasta",
                "content_type": "text/x-fasta",
                "sha256": "b" * 64,
                "size_bytes": 512,
            },
            {
                "file_id": "f_l",
                "original_filename": "light_chain.fasta",
                "content_type": "text/x-fasta",
                "sha256": "c" * 64,
                "size_bytes": 512,
            },
        ],
    )
    slots = _slots_by_name(out)
    assert "structure_or_sequence" not in slots


def test_mock_missing_slots_survive_full_supervisor_parse():
    """End-to-end through SupervisorAgent: blocking target slot is preserved
    as a typed MissingSlot on the StructuredQuery."""
    agent = SupervisorAgent(llm=MockLLMProvider())
    sq = agent.parse_raw_to_structured_query(
        {
            "run_id": "run_e2e",
            "run_artifact_registry_id": "reg_e2e",
            "artifact_id": "art_e2e",
            "created_at": "2026-06-28T00:00:00Z",
            "raw_user_query": "I want to design an ADC",
            "user_provided_context": {},
            "uploaded_files": [],
        }
    )
    blocking = [m for m in sq.missing_slots if m.severity == "blocking"]
    assert any(m.slot_name == "target_or_antigen" for m in blocking)


def test_mock_missing_slots_do_not_leak_sequences_or_keys():
    heavy = "EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGK"
    out = _parse(
        f"Design an ADC against HER2 using antibody sequence {heavy}",
    )
    blob = str(out["missing_slots"])
    assert heavy not in blob
    assert "api_key" not in blob.lower()
    assert "system instructions" not in blob.lower()


# ── Step 2 user-facing `response` field ──────────────────────────────────────


def test_schema_accepts_response_and_defaults_none():
    assert _sq().response is None
    assert _sq(response="Please provide the target.").response == "Please provide the target."


def test_schema_backward_compatible_old_artifact_without_response():
    payload = _sq().model_dump()
    payload.pop("response", None)
    assert "response" not in payload
    restored = StructuredQuery.model_validate(payload)
    assert restored.response is None


def test_prompt_includes_response_rules():
    sp = SUPERVISOR_SYSTEM_PROMPT
    assert "response" in sp
    assert "user-facing" in sp.lower()
    # Tells the model to prioritize blocking slots in the message.
    assert "blocking" in sp.lower()


def test_normalizer_response_absent_becomes_none():
    out = normalize_llm_payload_for_step2({"task_intent": {"task_type": "x"}})
    assert out["response"] is None


def test_normalizer_response_non_string_scalar_coerced():
    out = normalize_llm_payload_for_step2({"response": 123})
    assert out["response"] == "123"


def test_normalizer_response_list_compacted():
    out = normalize_llm_payload_for_step2({"response": ["need target", "need payload"]})
    assert out["response"] == "need target need payload"


def test_normalizer_response_dict_compacted():
    out = normalize_llm_payload_for_step2(
        {"response": {"message": "Please provide the target."}}
    )
    assert out["response"] == "Please provide the target."


def test_normalizer_response_overlong_trimmed():
    long = "x" * 900
    out = normalize_llm_payload_for_step2({"response": long})
    assert len(out["response"]) == 500
    assert any("truncated response" in w for w in out["parse_warnings"])


def test_mock_emits_response_when_missing_slots_present():
    out = _parse("I want to design an ADC")
    assert out["response"]
    assert "target" in out["response"].lower()


def test_mock_response_none_when_no_missing_slots():
    out = _parse("Design HER2 ADC with vc-MMAE and trastuzumab")
    assert out["missing_slots"] == []
    assert out["response"] is None


def test_mock_response_warning_only_combines_compactly():
    out = _parse("Design a HER2 ADC with MMAE")
    assert "target" not in out["response"].lower()
    assert "antibody" in out["response"].lower()
    assert "linker" in out["response"].lower()


def test_supervisor_preserves_response_into_structured_query():
    agent = SupervisorAgent(llm=MockLLMProvider())
    sq = agent.parse_raw_to_structured_query(
        {
            "run_id": "run_resp",
            "run_artifact_registry_id": "reg_resp",
            "artifact_id": "art_resp",
            "created_at": "2026-06-28T00:00:00Z",
            "raw_user_query": "I want to design an ADC",
            "user_provided_context": {},
            "uploaded_files": [],
        }
    )
    assert sq.response
    assert "target" in sq.response.lower()


def test_mock_response_does_not_leak_sequence_or_keys():
    heavy = "EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGK"
    out = _parse(f"Design an ADC using antibody sequence {heavy}")
    blob = (out.get("response") or "")
    assert heavy not in blob
    assert "api_key" not in blob.lower()
    assert "system instructions" not in blob.lower()


# ── Step 2 canonical_query (stable working-query field) ──────────────────────

_QUERY_ALIASES = (
    "working_query", "normalized_query", "final_query", "rewritten_query",
    "user_query_summary", "query_for_downstream", "canonical_task",
    "task_summary", "query_summary",
)


def test_schema_accepts_canonical_query_and_defaults_none():
    assert _sq().canonical_query is None
    assert _sq(canonical_query="Design ADC for HER2").canonical_query == "Design ADC for HER2"


def test_schema_backward_compatible_without_canonical_query():
    payload = _sq().model_dump()
    payload.pop("canonical_query", None)
    restored = StructuredQuery.model_validate(payload)
    assert restored.canonical_query is None


def test_schema_has_no_query_alias_fields():
    fields = set(StructuredQuery.model_fields)
    assert "canonical_query" in fields
    for alias in _QUERY_ALIASES:
        assert alias not in fields


def test_prompt_requires_canonical_query_and_forbids_aliases():
    sp = SUPERVISOR_SYSTEM_PROMPT
    assert "canonical_query" in sp
    for alias in _QUERY_ALIASES:
        assert alias in sp  # named explicitly as forbidden


def test_normalizer_canonical_query_absent_is_none():
    out = normalize_llm_payload_for_step2({"task_intent": {"task_type": "x"}})
    assert out["canonical_query"] is None


def test_normalizer_canonical_query_promotes_alias_and_removes_it():
    out = normalize_llm_payload_for_step2(
        {"working_query": "Design a HER2 ADC", "parse_warnings": []}
    )
    assert out["canonical_query"] == "Design a HER2 ADC"
    assert "working_query" not in out
    assert any("promoted query alias" in w for w in out["parse_warnings"])


def test_normalizer_canonical_query_overlong_truncated():
    out = normalize_llm_payload_for_step2({"canonical_query": "x" * 1200})
    assert len(out["canonical_query"]) == 800


def test_mock_first_turn_emits_canonical_query():
    out = _parse("I want to design a new antibody-drug conjugate.")
    cq = out["canonical_query"]
    assert cq and "antibody-drug conjugate" in cq.lower()
    assert "unspecified" in cq.lower()  # missing slots noted, not invented
    for alias in _QUERY_ALIASES:
        assert alias not in out


def test_mock_second_turn_canonical_query_includes_her2_and_keeps_intent():
    out = _parse(
        "I want to design a new antibody-drug conjugate.",
        ctx={
            "previous_task_intent": {"primary_intent": "new_adc_design", "secondary_intents": []},
            "previous_canonical_query": "Design a new antibody-drug conjugate (target unspecified).",
            "clarification_answers": [
                {"request_id": "r1", "slot_name": "target_or_antigen",
                 "slot_category": "target", "answer_text": "HER2", "answered_at": "t"}
            ],
        },
    )
    assert out["task_intent"]["primary_intent"] == "new_adc_design"
    assert "HER2" in (out["canonical_query"] or "")
    slots = _slots_by_name(out)
    assert "target_or_antigen" not in slots


def test_final_artifact_has_no_query_alias_keys():
    """Even if the LLM emits a wrong alias, the StructuredQuery artifact must
    not contain it."""
    agent = SupervisorAgent(llm=MockLLMProvider())
    sq = agent.parse_raw_to_structured_query(
        {
            "run_id": "run_cq", "run_artifact_registry_id": "reg_cq",
            "artifact_id": "art_cq", "created_at": "2026-06-29T00:00:00Z",
            "raw_user_query": "I want to design a new antibody-drug conjugate.",
            "user_provided_context": {}, "uploaded_files": [],
        }
    )
    dumped = sq.model_dump()
    assert "canonical_query" in dumped
    for alias in _QUERY_ALIASES:
        assert alias not in dumped


def test_mock_canonical_query_no_leakage():
    heavy = "EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGK"
    out = _parse(f"Design an ADC using antibody sequence {heavy}")
    cq = out.get("canonical_query") or ""
    assert heavy not in cq
    assert "api_key" not in cq.lower()


# ── Step 2 prompt_sequence (ESM masked generation prompt) ──────────────────

_HEAVY_CHAIN_SEQ = "EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGK"
_UNMASKED_TOKEN = "MKTAYIAKQNNVGA"
_MASKED_TOKEN = "MKT_YIAKQNNVGA"


def test_schema_accepts_prompt_sequence_missing_slot():
    sq = _sq(
        missing_slots=[
            MissingSlot(
                slot_name="prompt_sequence",
                slot_category="sequence",
                severity="blocking",
                required_for=["unclear_or_needs_clarification"],
                reason="Protein generation requires an explicit masked prompt_sequence.",
                suggested_question="Please provide the masked protein generation prompt.",
            )
        ]
    )
    assert sq.missing_slots[0].slot_name == "prompt_sequence"
    assert sq.missing_slots[0].severity == "blocking"


def test_normalizer_accepts_prompt_sequence_slot_name_without_downgrading_to_other():
    """`prompt_sequence` must survive `normalize_missing_slots` as itself —
    it must never be coerced to the generic `other` slot_name."""
    out = normalize_llm_payload_for_step2(
        {
            "missing_slots": [
                {
                    "slot_name": "prompt_sequence",
                    "severity": "blocking",
                    "reason": "no masked prompt provided",
                }
            ]
        }
    )
    assert out["missing_slots"][0]["slot_name"] == "prompt_sequence"
    assert out["missing_slots"][0]["slot_category"] == "sequence"


def test_normalizer_keeps_uploaded_file_prompt_sequence_reference_untouched():
    """Step 2 has no storage access — an uploaded_file entry explicitly
    labeled `source="prompt_sequence"` is passed through unchanged; its
    content (mask-marker presence) is validated later at Step 5."""
    out = normalize_llm_payload_for_step2(
        {
            "referenced_inputs": [
                {"id_type": "uploaded_file", "value": "f_prompt_1", "source": "prompt_sequence"},
            ],
        }
    )
    assert out["referenced_inputs"] == [
        {"id_type": "uploaded_file", "value": "f_prompt_1", "source": "prompt_sequence"},
    ]
    assert not any(s["slot_name"] == "prompt_sequence" for s in out["missing_slots"])


def test_normalizer_drops_inline_prompt_sequence_without_mask_marker_and_blocks():
    """An inline `id_type="prompt_sequence"` value without "_"/"<mask>" must
    never be treated as usable — it is dropped (never downgraded to a plain
    sequence id_type) and a blocking missing_slot is raised."""
    out = normalize_llm_payload_for_step2(
        {"referenced_inputs": [{"id_type": "prompt_sequence", "value": _UNMASKED_TOKEN, "source": "user"}]},
        {"raw_user_query": "Please generate a protein sequence.", "user_provided_context": {}},
    )
    assert out["referenced_inputs"] == []
    slots = {s["slot_name"]: s for s in out["missing_slots"]}
    assert slots["prompt_sequence"]["severity"] == "blocking"
    assert any("dropped" in w and "prompt_sequence" in w for w in out["parse_warnings"])
    assert _UNMASKED_TOKEN not in str(out["parse_warnings"])


def test_normalizer_keeps_inline_prompt_sequence_with_mask_marker():
    out = normalize_llm_payload_for_step2(
        {"referenced_inputs": [{"id_type": "prompt_sequence", "value": _MASKED_TOKEN, "source": "user"}]},
        {"raw_user_query": "Please generate a protein sequence.", "user_provided_context": {}},
    )
    assert out["referenced_inputs"] == [
        {"id_type": "prompt_sequence", "value": _MASKED_TOKEN, "source": "user"}
    ]
    assert not any(s["slot_name"] == "prompt_sequence" for s in out["missing_slots"])


def test_normalizer_no_blocking_slot_when_generation_not_requested():
    out = normalize_llm_payload_for_step2(
        {"referenced_inputs": []},
        {"raw_user_query": "Assess developability of this antibody.", "user_provided_context": {}},
    )
    assert not any(s["slot_name"] == "prompt_sequence" for s in out["missing_slots"])


def test_mock_ordinary_heavy_chain_sequence_with_generation_request_blocks_prompt_sequence():
    out = _parse(
        f"Please generate a protein sequence using this heavy chain {_HEAVY_CHAIN_SEQ} as reference"
    )
    slots = _slots_by_name(out)
    assert slots["prompt_sequence"]["severity"] == "blocking"
    assert not any(r.get("id_type") == "prompt_sequence" for r in out["referenced_inputs"])


def test_mock_explicit_masked_inline_prompt_sequence_satisfies_slot():
    out = _parse(f"Please generate a protein sequence. Use {_MASKED_TOKEN} as the prompt_sequence.")
    slots = _slots_by_name(out)
    assert "prompt_sequence" not in slots
    refs = {(r["id_type"], r["value"]) for r in out["referenced_inputs"]}
    assert ("prompt_sequence", _MASKED_TOKEN) in refs


def test_mock_ordinary_developability_assessment_does_not_require_prompt_sequence():
    out = _parse(f"Assess developability of this antibody heavy chain {_HEAVY_CHAIN_SEQ}")
    slots = _slots_by_name(out)
    assert "prompt_sequence" not in slots


def test_full_supervisor_parse_uploaded_file_declared_prompt_sequence_not_blocking():
    """Step 2 cannot read uploaded-file bytes; declaring a file as the
    masked prompt is enough to satisfy the slot without inspecting it."""
    agent = SupervisorAgent(llm=MockLLMProvider())
    sq = agent.parse_raw_to_structured_query(
        {
            "run_id": "run_prompt_seq_file",
            "run_artifact_registry_id": "reg_prompt_seq_file",
            "artifact_id": "art_prompt_seq_file",
            "created_at": "2026-07-06T00:00:00Z",
            "raw_user_query": "Please generate a protein sequence using the attached masked prompt file.",
            "user_provided_context": {},
            "uploaded_files": [
                {
                    "file_id": "f_prompt_1",
                    "original_filename": "generation_prompt.fasta",
                    "storage_path": "adc_pilot/runs/x/inputs/files/f_prompt_1.fasta",
                    "content_type": "text/x-fasta",
                    "sha256": "d" * 64,
                    "size_bytes": 128,
                }
            ],
        }
    )
    blocking = [m for m in sq.missing_slots if m.severity == "blocking"]
    assert not any(m.slot_name == "prompt_sequence" for m in blocking)
    assert any(
        r.get("id_type") == "uploaded_file" and r.get("source") == "prompt_sequence"
        for r in sq.referenced_inputs
    )


def test_full_supervisor_parse_inline_prompt_sequence_without_mask_stays_blocking():
    agent = SupervisorAgent(llm=MockLLMProvider())
    sq = agent.parse_raw_to_structured_query(
        {
            "run_id": "run_prompt_seq_nomask",
            "run_artifact_registry_id": "reg_prompt_seq_nomask",
            "artifact_id": "art_prompt_seq_nomask",
            "created_at": "2026-07-06T00:00:00Z",
            "raw_user_query": f"Please generate a protein sequence. Use {_UNMASKED_TOKEN} as the prompt_sequence.",
            "user_provided_context": {},
            "uploaded_files": [],
        }
    )
    blocking = [m for m in sq.missing_slots if m.severity == "blocking"]
    assert any(m.slot_name == "prompt_sequence" for m in blocking)
    assert not any(r.get("id_type") == "prompt_sequence" for r in sq.referenced_inputs)


def test_mock_prompt_sequence_missing_slots_do_not_leak_raw_tokens():
    out = _parse(
        f"Please generate a protein sequence using this heavy chain {_HEAVY_CHAIN_SEQ} as reference"
    )
    blob = str(out["missing_slots"]) + str(out.get("response") or "")
    assert _HEAVY_CHAIN_SEQ not in blob
