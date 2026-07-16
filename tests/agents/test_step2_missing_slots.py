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

import json

import pytest

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
    assert "source` is its only downstream" in sp
    assert "Never infer this role from" in sp


@pytest.mark.parametrize(
    "source",
    [
        "target_sequence",
        "antibody_heavy_chain_sequence",
        "antibody_light_chain_sequence",
        "prompt_sequence",
    ],
)
def test_normalizer_preserves_canonical_uploaded_fasta_source(source):
    out = normalize_llm_payload_for_step2(
        {
            "referenced_inputs": [
                {
                    "id_type": "uploaded_file",
                    "value": "file_sequence",
                    "source": source,
                }
            ],
            "missing_slots": [],
        },
        {
            "raw_user_query": "Analyze the uploaded file.",
            "user_provided_context": {},
            "uploaded_files": [
                {
                    "file_id": "file_sequence",
                    "original_filename": "generic.fasta",
                }
            ],
        },
    )
    assert out["referenced_inputs"] == [
        {
            "id_type": "uploaded_file",
            "value": "file_sequence",
            "source": source,
        }
    ]
    assert not any(
        slot["slot_name"] == "sequence_role" for slot in out["missing_slots"]
    )


def test_normalizer_keeps_unassigned_fasta_generic_and_blocks_sequence_role():
    out = normalize_llm_payload_for_step2(
        {
            "referenced_inputs": [
                {
                    "id_type": "uploaded_file",
                    "value": "file_sequence",
                    "source": "uploaded_file",
                }
            ],
            "missing_slots": [],
        },
        {
            "raw_user_query": "Use target heavy light keywords.",
            "user_provided_context": {
                "target_or_antigen_text": "target",
                "candidate_text": "heavy light antibody",
            },
            "uploaded_files": [
                {
                    "file_id": "file_sequence",
                    "original_filename": "target_heavy_light.fasta",
                }
            ],
        },
    )
    assert out["referenced_inputs"] == [
        {
            "id_type": "uploaded_file",
            "value": "file_sequence",
            "source": "uploaded_file",
        }
    ]
    slots = [
        slot for slot in out["missing_slots"] if slot["slot_name"] == "sequence_role"
    ]
    assert len(slots) == 1
    assert slots[0]["severity"] == "blocking"


def test_assigned_fasta_does_not_delete_llm_sequence_role_semantics():
    existing = {
        "slot_name": "sequence_role",
        "slot_category": "sequence",
        "severity": "blocking",
        "required_for": ["structure_analysis"],
        "reason": "The antibody sequence chain role is unresolved.",
        "suggested_question": "Is the antibody sequence heavy or light chain?",
        "evidence": "structured_query.referenced_inputs[1]",
    }
    out = normalize_llm_payload_for_step2(
        {
            "referenced_inputs": [
                {
                    "id_type": "uploaded_file",
                    "value": "file_target",
                    "source": "target_sequence",
                },
                {
                    "id_type": "antibody_sequence_reference",
                    "value": "EVQLVESGGGLVQPGGSLRLSCAAS",
                    "source": "user",
                },
            ],
            "missing_slots": [existing],
        },
        {
            "raw_user_query": "The uploaded FASTA is the target sequence.",
            "user_provided_context": {},
            "uploaded_files": [
                {
                    "file_id": "file_target",
                    "original_filename": "target.fasta",
                    "role": "target_sequence",
                }
            ],
        },
    )
    assert out["referenced_inputs"][0]["source"] == "target_sequence"
    assert out["referenced_inputs"][1]["id_type"] == (
        "antibody_sequence_reference"
    )
    assert out["missing_slots"] == [existing]


def test_unassigned_fasta_dedupes_sequence_role_and_preserves_first_evidence():
    first = {
        "slot_name": "sequence_role",
        "slot_category": "sequence",
        "severity": "warning",
        "required_for": ["structure_analysis"],
        "reason": "Existing semantic reason.",
        "suggested_question": "Existing semantic question?",
        "evidence": "existing_semantic_evidence",
    }
    duplicate = {**first, "reason": "duplicate must be removed"}
    out = normalize_llm_payload_for_step2(
        {
            "referenced_inputs": [
                {
                    "id_type": "uploaded_file",
                    "value": "file_unassigned",
                    "source": "uploaded_file",
                }
            ],
            "missing_slots": [first, duplicate],
        },
        {
            "raw_user_query": "Use the upload.",
            "user_provided_context": {},
            "uploaded_files": [
                {
                    "file_id": "file_unassigned",
                    "original_filename": "sequence.fasta",
                }
            ],
        },
    )
    slots = [
        slot
        for slot in out["missing_slots"]
        if slot["slot_name"] == "sequence_role"
    ]
    assert len(slots) == 1
    assert slots[0] == {**first, "severity": "blocking"}


@pytest.mark.parametrize(
    ("metadata", "source"),
    [
        ({"role": "target_sequence"}, "target_sequence"),
        ({"chain_role": "heavy"}, "antibody_heavy_chain_sequence"),
        ({"chain_role": "light"}, "antibody_light_chain_sequence"),
        ({"role": "prompt_sequence"}, "prompt_sequence"),
    ],
)
def test_explicit_uploaded_file_metadata_normalizes_into_step2_source(
    metadata,
    source,
):
    out = normalize_llm_payload_for_step2(
        {"referenced_inputs": [], "missing_slots": []},
        {
            "raw_user_query": "Analyze the uploaded file.",
            "user_provided_context": {},
            "uploaded_files": [
                {
                    "file_id": "file_sequence",
                    "original_filename": "generic.fasta",
                    **metadata,
                }
            ],
        },
    )
    assert out["referenced_inputs"] == [
        {
            "id_type": "uploaded_file",
            "value": "file_sequence",
            "source": source,
        }
    ]
    assert not any(
        slot["slot_name"] == "sequence_role" for slot in out["missing_slots"]
    )


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


@pytest.mark.parametrize(
    "sequence",
    ["ACDE", "ACDEFGHIKLMNPQRSTVWY" * 40],
)
def test_initial_inline_target_sequence_is_visible_and_canonical(sequence):
    class _CapturingProvider:
        """Test-only fixture: explicit Step2 role output, not live semantics."""

        name = "test-only-capturing-mock"
        model = "test-only"

        def __init__(self):
            self.inner = MockLLMProvider()
            self.calls = []

        def generate_json(self, prompt, *, schema, system=None):
            self.calls.append({"prompt": prompt, "schema": schema, "system": system})
            result = self.inner.generate_json(prompt, schema=schema, system=system)
            result["referenced_inputs"] = [
                {"id_type": "target_sequence", "value": sequence, "source": "user"}
            ]
            result["missing_slots"] = [
                slot
                for slot in result.get("missing_slots") or []
                if slot.get("slot_name") != "structure_or_sequence"
            ]
            result["response"] = None
            return result

    provider = _CapturingProvider()
    sq = SupervisorAgent(llm=provider).parse_raw_to_structured_query(
        {
            "run_id": "run_20260715_abcdef12",
            "run_artifact_registry_id": "raw_request_record_20260715_abcdef12",
            "artifact_id": "raw_request_record_20260715_abcdef12",
            "created_at": "2026-07-15T00:00:00Z",
            "raw_user_query": f"Analyze target sequence: {sequence}",
            "user_provided_context": {},
            "uploaded_files": [],
        }
    )
    assert sequence in json.dumps(provider.calls)
    assert any(
        ref.get("id_type") == "target_sequence" and ref.get("value") == sequence
        for ref in sq.referenced_inputs
    )
    assert not any(slot.slot_name == "structure_or_sequence" for slot in sq.missing_slots)


@pytest.mark.parametrize(
    "query",
    [
        "Analyze target sequence: ACDEFGHIKLMNPQ",
        "Protein sequence: PLEASE",
    ],
)
def test_mock_does_not_infer_target_sequence_from_raw_query(query):
    out = _parse(query)
    assert not any(
        ref.get("id_type") == "target_sequence"
        for ref in out["referenced_inputs"]
    )


@pytest.mark.parametrize(
    "generic_id_type",
    ["protein_sequence", "fasta_sequence", "amino_acid_sequence"],
)
def test_normalizer_does_not_promote_generic_sequence_aliases(generic_id_type):
    out = normalize_llm_payload_for_step2(
        {
            "referenced_inputs": [
                {
                    "id_type": generic_id_type,
                    "value": "MKTAYIAKQNNVG",
                    "source": "user",
                }
            ],
            "missing_slots": [
                {
                    "slot_name": "structure_or_sequence",
                    "slot_category": "sequence",
                    "severity": "blocking",
                    "reason": "Step2 did not assign a consumable role.",
                }
            ],
        }
    )
    assert out["referenced_inputs"][0]["id_type"] == generic_id_type
    assert out["missing_slots"][0]["slot_name"] == "structure_or_sequence"


@pytest.mark.parametrize(
    "id_type",
    [
        "target_sequence",
        "antibody_heavy_chain_sequence",
        "antibody_light_chain_sequence",
    ],
)
def test_invalid_canonical_typed_sequence_restores_exact_blocking_slot(id_type):
    invalid = "ACDE?FG"
    out = normalize_llm_payload_for_step2(
        {
            "referenced_inputs": [
                {"id_type": id_type, "value": invalid, "source": "user"}
            ],
            "missing_slots": [],
        }
    )

    assert out["referenced_inputs"] == []
    assert invalid not in json.dumps(out["parse_warnings"])
    assert out["parse_warnings"] == [
        (
            "dropped invalid target_sequence referenced_input"
            if id_type == "target_sequence"
            else "dropped invalid antibody chain sequence referenced_input"
        )
    ]
    assert out["missing_slots"] == [
        {
            "slot_name": "structure_or_sequence",
            "slot_category": "sequence",
            "severity": "blocking",
            "required_for": [],
            "reason": "The supplied protein sequence could not be used.",
            "suggested_question": "Please provide a valid protein sequence.",
            "evidence": None,
        }
    ]


def test_invalid_typed_sequence_does_not_duplicate_existing_structure_slot():
    out = normalize_llm_payload_for_step2(
        {
            "referenced_inputs": [
                {
                    "id_type": "target_sequence",
                    "value": "ACDE?FG",
                    "source": "user",
                }
            ],
            "missing_slots": [
                {
                    "slot_name": "structure_or_sequence",
                    "slot_category": "structure",
                    "severity": "warning",
                    "required_for": ["structure_analysis"],
                    "reason": "Existing reason.",
                    "suggested_question": "Existing question?",
                    "evidence": "existing-safe-evidence",
                }
            ],
        }
    )

    slots = [
        slot
        for slot in out["missing_slots"]
        if slot["slot_name"] == "structure_or_sequence"
    ]
    assert len(slots) == 1
    assert slots[0] == {
        "slot_name": "structure_or_sequence",
        "slot_category": "structure",
        "severity": "blocking",
        "required_for": ["structure_analysis"],
        "reason": "Existing reason.",
        "suggested_question": "Existing question?",
        "evidence": "existing-safe-evidence",
    }


def test_invalid_typed_sequence_is_redundant_when_valid_pdb_remains():
    out = normalize_llm_payload_for_step2(
        {
            "referenced_inputs": [
                {
                    "id_type": "target_sequence",
                    "value": "ACDE?FG",
                    "source": "user",
                },
                {"id_type": "pdb_id", "value": "1N8Z", "source": "user"},
            ],
            "missing_slots": [],
        }
    )

    assert out["referenced_inputs"] == [
        {"id_type": "pdb_id", "value": "1N8Z", "source": "user"}
    ]
    assert not any(
        slot["slot_name"] == "structure_or_sequence"
        for slot in out["missing_slots"]
    )


@pytest.mark.parametrize(
    "prompt_ref",
    [
        {
            "id_type": "prompt_sequence",
            "value": "ACDEFGHIK_LMNPQRST",
            "source": "user",
        },
        {
            "id_type": "uploaded_file",
            "value": "file_prompt",
            "source": "prompt_sequence",
        },
    ],
)
def test_prompt_sequence_does_not_clear_previous_structure_or_sequence_slot(
    prompt_ref,
):
    out = normalize_llm_payload_for_step2(
        {"referenced_inputs": [prompt_ref], "missing_slots": []},
        {
            "raw_user_query": "Analyze a protein structure.",
            "user_provided_context": {
                "previous_missing_slots": [
                    {
                        "slot_name": "structure_or_sequence",
                        "slot_category": "sequence",
                        "severity": "blocking",
                    }
                ]
            },
            "uploaded_files": [
                {
                    "file_id": "file_prompt",
                    "original_filename": "generation_prompt.fasta",
                }
            ],
        },
    )
    assert out["referenced_inputs"] == [prompt_ref]
    assert any(
        slot["slot_name"] == "structure_or_sequence"
        and slot["severity"] == "blocking"
        for slot in out["missing_slots"]
    )


@pytest.mark.parametrize(
    ("accession", "slot_remains"),
    [("P04626", False), ("not-a-uniprot-accession", True)],
)
def test_only_valid_typed_uniprot_satisfies_previous_structure_or_sequence_slot(
    accession,
    slot_remains,
):
    out = normalize_llm_payload_for_step2(
        {
            "referenced_inputs": [
                {"id_type": "uniprot_id", "value": accession, "source": "user"}
            ],
            "missing_slots": [],
        },
        {
            "raw_user_query": "Analyze a protein structure.",
            "user_provided_context": {
                "previous_missing_slots": [
                    {
                        "slot_name": "structure_or_sequence",
                        "slot_category": "sequence",
                        "severity": "blocking",
                    }
                ]
            },
            "uploaded_files": [],
        },
    )
    assert (
        any(slot["slot_name"] == "structure_or_sequence" for slot in out["missing_slots"])
        is slot_remains
    )


@pytest.mark.parametrize("accession", ["P04626", "P04626-2"])
def test_normalizer_preserves_valid_uniprot_and_isoform(accession):
    out = normalize_llm_payload_for_step2(
        {
            "referenced_inputs": [
                {"id_type": "uniprot_id", "value": accession, "source": "user"}
            ],
            "missing_slots": [],
        }
    )
    assert out["referenced_inputs"] == [
        {"id_type": "uniprot_id", "value": accession, "source": "user"}
    ]
    assert not any(
        slot["slot_name"] == "structure_or_sequence"
        for slot in out["missing_slots"]
    )


def test_invalid_uniprot_is_dropped_and_restores_blocking_structure_slot():
    invalid = "not-a-uniprot-accession"
    out = normalize_llm_payload_for_step2(
        {
            "referenced_inputs": [
                {"id_type": "uniprot_id", "value": invalid, "source": "user"}
            ],
            "missing_slots": [],
        }
    )
    assert out["referenced_inputs"] == []
    assert out["parse_warnings"] == [
        "dropped invalid uniprot_id referenced_input"
    ]
    assert out["dropped_referenced_input_counts"] == {"uniprot_id": 1}
    assert invalid not in json.dumps(out["parse_warnings"])
    slots = [
        slot
        for slot in out["missing_slots"]
        if slot["slot_name"] == "structure_or_sequence"
    ]
    assert len(slots) == 1
    assert slots[0]["severity"] == "blocking"


@pytest.mark.parametrize(
    "valid_ref",
    [
        {"id_type": "pdb_id", "value": "1N8Z", "source": "user"},
        {
            "id_type": "target_sequence",
            "value": "ACDEFGHIK",
            "source": "user",
        },
    ],
)
def test_invalid_uniprot_does_not_block_another_valid_input(valid_ref):
    out = normalize_llm_payload_for_step2(
        {
            "referenced_inputs": [
                {
                    "id_type": "uniprot_id",
                    "value": "not-a-uniprot-accession",
                    "source": "user",
                },
                valid_ref,
            ],
            "missing_slots": [],
        }
    )
    assert out["referenced_inputs"] == [valid_ref]
    assert not any(
        slot["slot_name"] == "structure_or_sequence"
        for slot in out["missing_slots"]
    )


def test_mock_prompt_file_does_not_satisfy_structure_analysis_input():
    out = _parse(
        "Analyze the structure of HER2; the attached file is the prompt_sequence.",
        files=[
            {
                "file_id": "file_prompt",
                "original_filename": "generation_prompt.fasta",
                "content_type": "text/x-fasta",
            }
        ],
    )
    assert any(
        ref.get("id_type") == "uploaded_file"
        and ref.get("value") == "file_prompt"
        and ref.get("source") == "prompt_sequence"
        for ref in out["referenced_inputs"]
    )
    assert _slots_by_name(out)["structure_or_sequence"]["severity"] == "blocking"


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


# ── Step 2 protein variant / point mutation structuring ──────────────────────

_VARIANT_QUERY = (
    "Evaluate the HER2 variant V777L using UniProt P04626. "
    "Use variant scoring only; do not generate protein sequences."
)


def _refs_by_type(out: dict) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for ref in out.get("referenced_inputs") or []:
        if isinstance(ref, dict):
            grouped.setdefault(str(ref.get("id_type")), []).append(ref)
    return grouped


def test_mock_structures_uniprot_and_variant_referenced_inputs():
    out = _parse(_VARIANT_QUERY)
    grouped = _refs_by_type(out)
    assert any(r["value"] == "P04626" for r in grouped.get("uniprot_id", []))
    variants = grouped.get("variant", [])
    assert len(variants) == 1
    assert variants[0]["value"] == "V777L"
    assert variants[0]["source"] == "user"


def test_mock_labels_variant_as_protein_variant_entity():
    out = _parse(_VARIANT_QUERY)
    pv = [
        e for e in out.get("normalized_entities") or []
        if e.get("entity_type") == "protein_variant"
    ]
    assert pv and pv[0]["original_text"] == "V777L"


def test_mock_variant_not_placed_in_mentioned_entities():
    out = _parse(_VARIANT_QUERY)
    mentioned = out.get("mentioned_entities") or {}
    blob = str(mentioned)
    assert "V777L" not in blob
    # HER2 remains the target; the variant is not the target/antibody/payload.
    assert mentioned.get("antibody_candidate_text") in (None, "")
    assert mentioned.get("payload_text") in (None, "")


def test_normalizer_does_not_crash_on_protein_variant_entity_and_derives_variant():
    """A real-LLM payload that labels the mention entity_type="protein_variant"
    but omits the typed referenced_input must NOT raise, and the normalizer
    derives the stable id_type="variant" referenced_input."""
    out = normalize_llm_payload_for_step2(
        {
            "task_intent": {"task_type": "structure_analysis"},
            "normalized_entities": [
                {
                    "original_text": "V777L",
                    "canonical_name": "V777L",
                    "entity_type": "protein_variant",
                    "explicit_or_inferred": "explicit",
                }
            ],
            "referenced_inputs": [],
        }
    )
    variants = _refs_by_type(out).get("variant", [])
    assert variants and variants[0]["value"] == "V777L"
    # entity_type survives (schema now allows protein_variant).
    assert out["normalized_entities"][0]["entity_type"] == "protein_variant"


def test_normalizer_canonicalizes_variant_id_type_aliases():
    out = normalize_llm_payload_for_step2(
        {
            "task_intent": {"task_type": "structure_analysis"},
            "referenced_inputs": [
                {"id_type": "protein_variant", "value": "V777L", "source": "user"},
            ],
        }
    )
    grouped = _refs_by_type(out)
    assert "protein_variant" not in grouped
    assert grouped["variant"][0]["value"] == "V777L"


def test_normalizer_variant_derivation_is_idempotent_no_duplicates():
    payload = {
        "task_intent": {"task_type": "structure_analysis"},
        "normalized_entities": [
            {
                "original_text": "V777L",
                "entity_type": "protein_variant",
                "explicit_or_inferred": "explicit",
            }
        ],
        "referenced_inputs": [
            {"id_type": "variant", "value": "V777L", "source": "user"},
        ],
    }
    out = normalize_llm_payload_for_step2(payload)
    assert len(_refs_by_type(out).get("variant", [])) == 1


def test_full_supervisor_parse_structures_variant_without_crash():
    agent = SupervisorAgent(llm=MockLLMProvider())
    sq = agent.parse_raw_to_structured_query(
        {
            "run_id": "run_variant",
            "run_artifact_registry_id": "reg_variant",
            "artifact_id": "art_variant",
            "created_at": "2026-07-06T00:00:00Z",
            "raw_user_query": _VARIANT_QUERY,
            "user_provided_context": {},
            "uploaded_files": [],
        }
    )
    by_type = {r["id_type"]: r for r in sq.referenced_inputs if isinstance(r, dict)}
    assert by_type["uniprot_id"]["value"] == "P04626"
    assert by_type["variant"]["value"] == "V777L"
    # protein_variant entity type is preserved through the strict schema.
    assert any(e.entity_type == "protein_variant" for e in sq.normalized_entities)


def test_step2_prompt_documents_variant_referenced_input():
    assert '"id_type": "variant"' in SUPERVISOR_SYSTEM_PROMPT
    assert "V777L" in SUPERVISOR_SYSTEM_PROMPT
