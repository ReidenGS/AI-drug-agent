"""Unit tests for the opt-in Gemini Step 2 benchmark smoke validators.

These tests do not call Gemini and do not touch networked code. They only
exercise the compact validator functions used by
scripts/run_gemini_step2_benchmark_smoke.py.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "run_gemini_step2_benchmark_smoke.py"
)
SPEC = importlib.util.spec_from_file_location("gemini_step2_benchmark_smoke", SCRIPT_PATH)
assert SPEC and SPEC.loader
smoke = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = smoke
SPEC.loader.exec_module(smoke)

compact_summary = smoke.compact_summary
validate_case_1_her2_tdm1_tdxd = smoke.validate_case_1_her2_tdm1_tdxd
validate_case_2_trop2_mmae = smoke.validate_case_2_trop2_mmae
validate_case_3_her2_pdb_zinc = smoke.validate_case_3_her2_pdb_zinc
validate_case_4_cldn18_patent = smoke.validate_case_4_cldn18_patent
validate_case_5_literature_trastuzumab_mmae = smoke.validate_case_5_literature_trastuzumab_mmae
validate_case_6_chembl_zinc_payloads = smoke.validate_case_6_chembl_zinc_payloads
exception_reason = smoke._exception_reason


def _sq(
    *,
    primary: str,
    secondary: list[str] | None = None,
    requested: list[str] | None = None,
    normalized: list[dict] | None = None,
    decompositions: list[dict] | None = None,
    refs: list[dict] | None = None,
    questions: list[str] | None = None,
) -> dict:
    return {
        "task_intent": {
            "primary_intent": primary,
            "secondary_intents": secondary or [],
        },
        "requested_outputs": requested or [],
        "normalized_entities": normalized or [],
        "entity_decompositions": decompositions or [],
        "referenced_inputs": refs or [],
        "clarification_questions": questions or [],
    }


def test_case1_validator_accepts_her2_tdm1_tdxd_minimum():
    sq = _sq(
        primary="existing_adc_evaluation",
        secondary=["literature_review"],
        requested=["evidence_summary"],
        normalized=[{"canonical_name": "ERBB2"}],
        decompositions=[
            {"original_text": "T-DM1"},
            {"original_text": "T-DXd"},
        ],
    )
    assert validate_case_1_her2_tdm1_tdxd(sq) == []


def test_case2_validator_requires_trop2_mmae_and_clarification():
    sq = _sq(
        primary="new_adc_design",
        normalized=[
            {"canonical_name": "TACSTD2"},
            {"canonical_name": "monomethyl auristatin E"},
        ],
        questions=["Which antibody backbone and linker chemistry should we assume?"],
    )
    assert validate_case_2_trop2_mmae(sq) == []

    bad = _sq(primary="new_adc_design", normalized=[{"canonical_name": "TACSTD2"}])
    failures = validate_case_2_trop2_mmae(bad)
    assert "missing MMAE normalization" in failures
    assert "missing antibody/linker clarification question" in failures


def test_case3_validator_accepts_pdb_and_zinc_structure_screening():
    sq = _sq(
        primary="structure_analysis",
        secondary=["compound_screening"],
        requested=["structure_validation_report"],
        refs=[
            {"id_type": "pdb_id", "value": "1N8Z"},
            {"id_type": "zinc_id", "value": "ZINC12345678"},
        ],
    )
    assert validate_case_3_her2_pdb_zinc(sq) == []


def test_case4_validator_accepts_cldn18_patent_minimum():
    sq = _sq(
        primary="patent_ip_review",
        requested=["patent_or_ip_summary"],
        normalized=[{"canonical_name": "CLDN18 isoform 2"}],
    )
    assert validate_case_4_cldn18_patent(sq) == []


def test_case5_validator_requires_literature_output_and_question():
    sq = _sq(
        primary="literature_review",
        requested=["literature_review_summary"],
        questions=["Did you mean HER2 ADC literature or general payload literature?"],
    )
    assert validate_case_5_literature_trastuzumab_mmae(sq) == []

    bad = _sq(primary="literature_review", requested=["evidence_summary"])
    assert validate_case_5_literature_trastuzumab_mmae(bad) == [
        "missing clarification question for unclear HER2/ADC context"
    ]


def test_case6_validator_accepts_chembl_zinc_payload_candidates():
    sq = _sq(
        primary="compound_screening",
        secondary=["developability_assessment"],
        requested=["compound_screening_results"],
        refs=[
            {"id_type": "chembl_id", "value": "CHEMBL1201585"},
            {"id_type": "zinc_id", "value": "ZINC12345678"},
        ],
    )
    assert validate_case_6_chembl_zinc_payloads(sq) == []


def test_compact_summary_counts_without_raw_payloads():
    sq = _sq(
        primary="compound_screening",
        secondary=["literature_review"],
        requested=["compound_screening_results", "report"],
        normalized=[{"canonical_name": "ERBB2"}],
        decompositions=[{"original_text": "T-DM1"}],
        questions=["Clarify payload context?"],
    )
    summary = compact_summary("case", sq, [])
    assert summary == {
        "case_name": "case",
        "status": "PASS",
        "primary_intent": "compound_screening",
        "secondary_intents": ["literature_review"],
        "requested_outputs_count": 2,
        "normalized_entities_count": 1,
        "decompositions_count": 1,
        "clarification_questions_count": 1,
        "reason": "",
    }


def test_exception_reason_sanitizes_provider_payload():
    exc = RuntimeError(
        "503 UNAVAILABLE. {'error': {'message': 'This model is currently "
        "experiencing high demand', 'status': 'UNAVAILABLE'}}"
    )
    assert exception_reason(exc) == "RuntimeError: 503 UNAVAILABLE"
