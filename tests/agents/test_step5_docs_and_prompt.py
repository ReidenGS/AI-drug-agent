"""Step 5 prompt-addendum + dependency-doc presence audit.

The Step 5 addendum is intentionally compact: it should state the role,
core catalog-only rules, short input→selection examples, CDR3 boundary,
and no-candidate-generation boundary without embedding long JSON samples
or wrapper-specific implementation details. The separate CDR3 dependency
note records setup details.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.agents.step_05_selection_policy import (
    STEP_05_SELECTION_SYSTEM_ADDENDUM,
)


# Project tree layout: <…>/国外ai医药/程序/执行文件 + <…>/国外ai医药/任务执行.
# The dependency note lives next to 程序, not under it. Resolve via the
# common ancestor (国外ai医药).
_COMMON_ANCESTOR = Path(__file__).resolve().parents[4]
_WEEK5_DOC = (
    _COMMON_ANCESTOR
    / "任务执行"
    / "week5"
    / "step5_cdr3_numbering_dependency_note_2026-06-26.md"
)


# ── 1. Selection addendum: concise role / CDR3 / boundary coverage ────


def test_addendum_is_concise_and_has_compact_few_shot_examples():
    sp = STEP_05_SELECTION_SYSTEM_ADDENDUM
    assert len(sp.split()) < 520
    assert "Compact examples:" in sp
    assert "input: `linker_payload_name=vc-MMAE`" in sp
    assert "input: `payload_smiles=CCO`" in sp
    assert "input: `target_antigen_name=HER2`" in sp
    assert "input: `antibody_heavy_chain_sequence`" in sp
    assert "input: uploaded `structure_file` or `pdb_id`" in sp
    assert '"selections": [' not in sp
    assert "Role:" in sp
    assert "Rules:" in sp
    assert "Output:" in sp


def test_addendum_states_smiles_and_name_lookup_complementarity():
    sp = STEP_05_SELECTION_SYSTEM_ADDENDUM
    assert "Name lookup and SMILES lookup are complementary paths." in sp
    assert "select at least one SMILES lookup path" in sp
    assert "even if a name lookup is also" in sp
    assert "Do not invent SMILES" in sp
    assert "do not use a name string as SMILES" in sp


def test_addendum_states_full_sequence_must_go_through_cdr3_before_iedb():
    sp = STEP_05_SELECTION_SYSTEM_ADDENDUM
    assert "antibody heavy/light sequences" in sp
    assert "runtime CDR3 extraction happens" in sp
    assert "before any IEDB BCR lookup" in sp
    assert "do not assume\nfull VH/VL sequences are sent to IEDB" in sp
    assert "If CDR3 extraction is unavailable" in sp
    assert "runtime records a data gap" in sp


def test_addendum_restates_catalog_only_and_no_invented_tools():
    sp = STEP_05_SELECTION_SYSTEM_ADDENDUM
    assert "Use only exact `tool_name` values from `compact_catalog`." in sp
    assert "Do not construct arguments" in sp
    assert "invent identifiers" in sp
    assert "invent tool names" in sp


def test_addendum_forbids_new_adc_candidate_generation():
    sp = STEP_05_SELECTION_SYSTEM_ADDENDUM
    assert "Step 5 does not generate new ADC design\ncandidates." in sp
    assert "Do not generate ADC candidates" in sp


# ── 2. CDR3 dependency note ──────────────────────────────────────────


@pytest.fixture(scope="module")
def cdr3_doc_text() -> str:
    if not _WEEK5_DOC.exists():
        pytest.skip(
            "CDR3 dependency note not present in this checkout: "
            f"{_WEEK5_DOC}"
        )
    return _WEEK5_DOC.read_text(encoding="utf-8")


def test_cdr3_doc_records_python_dependencies(cdr3_doc_text: str):
    assert "abnumber" in cdr3_doc_text
    assert "anarci" in cdr3_doc_text


def test_cdr3_doc_records_system_binary_and_install_example(
    cdr3_doc_text: str,
):
    assert "HMMER" in cdr3_doc_text or "hmmer" in cdr3_doc_text.lower()
    assert "hmmscan" in cdr3_doc_text
    assert "brew install hmmer" in cdr3_doc_text


def test_cdr3_doc_records_chain_filter_mapping(cdr3_doc_text: str):
    assert "chain1_cdr3_seq" in cdr3_doc_text
    assert "chain2_cdr3_seq" in cdr3_doc_text


def test_cdr3_doc_records_privacy_boundary(cdr3_doc_text: str):
    assert (
        "Raw full sequences and raw CDR3 strings must not be written"
        in cdr3_doc_text
    )
    assert (
        "records a compact data gap and does not call IEDB"
        in cdr3_doc_text
    )
