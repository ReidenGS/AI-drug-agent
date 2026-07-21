from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.schemas.patent_evidence_request import (
    PatentEvidenceInputRef,
    PatentEvidenceRequest,
    PatentEvidenceSearchScope,
)


def _ref(**overrides):
    values = {
        "ref_id": "r_payload",
        "source_artifact": "candidate_context_table",
        "source_path": "candidate_records[0].payload",
        "role": "payload",
        "supports_tool_args": ["query"],
    }
    values.update(overrides)
    return PatentEvidenceInputRef(**values)


@pytest.mark.parametrize(
    "scope",
    [
        {"requested_lanes": []},
        {"requested_lanes": ["evidence", "evidence"]},
        {"allowed_roles": []},
        {"allowed_roles": ["payload", "payload"]},
        {"allowed_roles": ["unknown_role"]},
    ],
)
def test_scope_rejects_empty_duplicate_or_unknown_values(scope):
    with pytest.raises(ValidationError):
        PatentEvidenceSearchScope(**scope)


@pytest.mark.parametrize(
    "overrides",
    [
        {"ref_id": ""},
        {"ref_id": "../unsafe"},
        {"source_artifact": "../artifact"},
        {"source_path": "../payload"},
        {"role": "arbitrary"},
        {"supports_tool_args": ["query", "query"]},
        {"supports_tool_args": ["unknown_arg"]},
        {"supports_tool_args": ["bad-token"]},
    ],
)
def test_input_ref_rejects_unsafe_or_unknown_fields(overrides):
    with pytest.raises(ValidationError):
        _ref(**overrides)


def test_request_requires_unique_ref_ids_and_safe_artifact_refs():
    with pytest.raises(ValidationError, match="ref_id values must be unique"):
        PatentEvidenceRequest(
            run_id="run_20260717_abcdef12", input_refs=[_ref(), _ref()]
        )
    with pytest.raises(ValidationError):
        PatentEvidenceRequest(
            run_id="run_20260717_abcdef12",
            source_artifact_refs={"candidate_context_table": "../secret"},
        )


def test_valid_reference_only_request_round_trips():
    request = PatentEvidenceRequest(
        run_id="run_20260717_abcdef12",
        source_artifact_refs={"candidate_context_table": "artifacts/cct_123"},
        input_refs=[_ref()],
    )
    assert request.input_refs[0].supports_tool_args == ["query"]
    assert request.search_scope.requested_lanes == ["evidence", "patent"]


@pytest.mark.parametrize(
    ("role", "support"),
    [
        ("pmid", "cid"),
        ("application_number", "brand_name"),
        ("brand_name", "application_number"),
        ("pubchem_cid", "pmids"),
        ("title", "document_id"),
    ],
)
def test_support_tokens_must_match_typed_ref_role(role, support):
    with pytest.raises(
        ValidationError, match="supports_tool_args is incompatible with ref role"
    ):
        _ref(role=role, supports_tool_args=[support])


@pytest.mark.parametrize(
    "typed_support",
    [
        "pmid",
        "pmids",
        "cid",
        "pubchem_cid",
        "brand_name",
        "application_number",
        "document_id",
        "title",
        "title_contains",
        "title__contains",
    ],
)
def test_query_role_does_not_gain_typed_identifier_supports(typed_support):
    with pytest.raises(
        ValidationError, match="supports_tool_args is incompatible with ref role"
    ):
        _ref(role="query", supports_tool_args=[typed_support])


def test_long_query_and_request_note_sequence_have_no_semantic_length_caps():
    long_text = "HER2 ADC evidence " * 1000
    notes = [f"note {index} " + ("x" * 1000) for index in range(30)]
    request = PatentEvidenceRequest(
        run_id="run_20260717_abcdef12",
        user_query=long_text,
        request_notes=notes,
    )
    assert request.user_query == long_text.strip()
    assert request.request_notes == notes


def test_request_reuses_strict_production_run_id_type():
    with pytest.raises(ValidationError):
        PatentEvidenceRequest(run_id="run_pe_a")
