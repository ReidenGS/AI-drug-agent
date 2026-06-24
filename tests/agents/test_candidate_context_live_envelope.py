"""Step 5 live-envelope ChEMBL enrichment + raw isolation tests.

Pinned by a real observed regression: ChEMBL live envelopes nest the hit
list under ``payload.data.molecules`` (and sometimes ``payload.molecules``
or ``data.molecules``). The Step 5 enrichment must promote the compact
fields (chembl_id, canonical_smiles, pref_name) from those shapes while
keeping the raw envelope OUT of the normalized candidate_record.
"""

from __future__ import annotations

import json

import pytest

from app.agents.candidate_context_agent import (
    _apply_compound_tool_enrichment,
    _iter_compound_hits,
)
from app.schemas.step_05_candidate_context_table import CandidateRecord


def _new_compound_candidate() -> CandidateRecord:
    return CandidateRecord(
        candidate_id="cand_test",
        candidate_label="vc-MMAE",
        candidate_type="compound_component",
        materials=[],
        identifiers=[],
        candidate_role="user_provided_candidate",
        is_generated_candidate=False,
        context_status="partial",
    )


# ── _iter_compound_hits: envelope unwrap ───────────────────────────────────


def test_iter_compound_hits_unwraps_payload_data_molecules():
    """Live ChEMBL substructure envelope shape."""
    envelope = {
        "executor": "tooluniverse",
        "status": "ok",
        "payload": {
            "data": {
                "molecules": [
                    {"molecule_chembl_id": "CHEMBL2107839",
                     "molecule_structures": {"canonical_smiles": "CCO"}},
                    {"molecule_chembl_id": "CHEMBL999"},
                ]
            }
        },
    }
    hits = list(_iter_compound_hits(envelope))
    assert [h.get("molecule_chembl_id") for h in hits] == ["CHEMBL2107839", "CHEMBL999"]


def test_iter_compound_hits_unwraps_payload_molecules():
    envelope = {"payload": {"molecules": [{"molecule_chembl_id": "CHEMBL1"}]}}
    hits = list(_iter_compound_hits(envelope))
    assert [h.get("molecule_chembl_id") for h in hits] == ["CHEMBL1"]


def test_iter_compound_hits_unwraps_data_molecules():
    envelope = {"data": {"molecules": [{"molecule_chembl_id": "CHEMBL2"}]}}
    hits = list(_iter_compound_hits(envelope))
    assert [h.get("molecule_chembl_id") for h in hits] == ["CHEMBL2"]


def test_iter_compound_hits_preserves_legacy_top_level_hits():
    """Old mocked/test shape must keep working unchanged."""
    payload = {"hits": [{"chembl_id": "CHEMBL_OLD"}, {"chembl_id": "CHEMBL_OLDER"}]}
    hits = list(_iter_compound_hits(payload))
    assert [h.get("chembl_id") for h in hits] == ["CHEMBL_OLD", "CHEMBL_OLDER"]


def test_iter_compound_hits_does_not_unbounded_descend():
    """Sanity: unrelated deeply-nested keys must NOT be discovered."""
    payload = {"foo": {"bar": {"baz": {"molecules": [{"x": 1}]}}}}
    hits = list(_iter_compound_hits(payload))
    # `foo` is not in the wrapper key allowlist, so the payload is treated
    # as a single hit-shaped dict (last-resort fallback).
    assert hits == [payload]


# ── End-to-end: enrichment promotes ChEMBL fields from live envelope ───────


def test_live_envelope_promotes_chembl_id_and_smiles_to_candidate():
    cand = _new_compound_candidate()
    envelope = {
        "executor": "tooluniverse",
        "status": "ok",
        "source": "ChEMBL_search_substructure",
        "payload": {
            "data": {
                "molecules": [
                    {
                        "molecule_chembl_id": "CHEMBL2107839",
                        "molecule_structures": {"canonical_smiles": "CCO"},
                        "pref_name": "monomethyl auristatin E",
                    }
                ]
            }
        },
    }
    _apply_compound_tool_enrichment(
        cand, envelope, source_artifact_id="tool_output_abc",
    )

    ids = [(i.id_type, i.id_value, i.confidence) for i in cand.identifiers]
    assert ("chembl_id", "CHEMBL2107839", 0.8) in ids
    # source_ids carries the artifact id, NOT the raw envelope.
    for i in cand.identifiers:
        if i.id_type == "chembl_id":
            assert i.source_ids == ["tool_output_abc"]

    # SMILES promoted as a typed material.
    smiles_materials = [m for m in cand.materials if m.value_format == "smiles"]
    assert smiles_materials, "canonical_smiles should be promoted to a material"
    assert any(m.value == "CCO" for m in smiles_materials)


# ── Raw-payload isolation: envelope wrapper keys MUST NOT leak ─────────────


def test_normalized_candidate_record_does_not_contain_raw_envelope_keys():
    cand = _new_compound_candidate()
    envelope = {
        "executor": "tooluniverse",
        "status": "ok",
        "source": "ChEMBL_search_substructure",
        "arguments": {"smiles": "CCO"},
        "payload": {
            "data": {
                "molecules": [{
                    "molecule_chembl_id": "CHEMBL2107839",
                    "molecule_structures": {"canonical_smiles": "CCO"},
                    "raw_full_structure_blob": "SECRET_RAW_DO_NOT_LEAK",
                }],
            },
        },
    }
    _apply_compound_tool_enrichment(
        cand, envelope, source_artifact_id="tool_output_abc",
    )
    dumped = cand.model_dump_json()
    for forbidden in (
        "executor", "payload", "data", "molecule_structures",
        "raw_full_structure_blob", "SECRET_RAW_DO_NOT_LEAK",
        "arguments", "tooluniverse",
    ):
        # Some keys (e.g. "data") may appear inside ChEMBL_id strings, so
        # we check for the exact JSON key shape `"<name>":`.
        json_key = f'"{forbidden}":'
        assert json_key not in dumped, (
            f"normalized candidate_record leaked envelope key {forbidden!r}: {dumped}"
        )


# ── Multiple hits dedupe + cap ────────────────────────────────────────────


def test_multiple_live_hits_capped_and_deduped():
    cand = _new_compound_candidate()
    envelope = {
        "payload": {"data": {"molecules": [
            {"molecule_chembl_id": "CHEMBL1"},
            {"molecule_chembl_id": "CHEMBL2"},
            {"molecule_chembl_id": "CHEMBL3"},
            {"molecule_chembl_id": "CHEMBL4"},  # beyond cap=3
            {"molecule_chembl_id": "CHEMBL1"},  # duplicate
        ]}}
    }
    _apply_compound_tool_enrichment(cand, envelope, source_artifact_id="tc1")
    ids = sorted(i.id_value for i in cand.identifiers if i.id_type == "chembl_id")
    assert ids == ["CHEMBL1", "CHEMBL2", "CHEMBL3"], ids
