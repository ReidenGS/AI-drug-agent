"""Step 2 batch-5 benchmark tests (professor examples).

Six benchmark requests; each exercises the deterministic MockLLMProvider
through the SupervisorAgent and asserts:

1. HER2 ADC benchmark comparing T-DM1 and T-DXd / Enhertu.
2. New TROP2 ADC with MMAE payload (linker / antibody missing).
3. HER2 PDB 1N8Z + ZINC compounds — structure_analysis primary,
   compound_screening secondary, ambiguity clarification surfaced.
4. CLDN18.2 ADC patent / IP review with deruxtecan-family payload.
5. Literature-only "trastuzumab + MMAE" request — HER2 inferred, not
   forced explicit.
6. ChEMBL / ZINC compounds as possible HER2 payload candidates.

Each benchmark verifies primary_intent + secondary_intents,
requested_outputs, normalized_entities (HER2→ERBB2, TROP2→TACSTD2, …),
entity_decompositions for multi-component drugs, and
clarification_questions where relevant.

Also asserts that:
- `parse_warnings` are internal (warnings exist for missing fields).
- `clarification_questions` are user-facing (distinct from parse_warnings).
- Step 2 path imports nothing from app.mcp / app.a2a, and never reads
  file bytes.
"""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest

from app.agents.supervisor_agent import SupervisorAgent
from app.llm.provider import MockLLMProvider
from app.schemas.step_01_raw_request_record import (
    RawRequestRecord,
    UploadedFile,
    UserProvidedContext,
)


_FIXTURE_RUN_ID = "run_step2_benchmark_fixture"


def _raw(
    *,
    query: str,
    ctx: dict | None = None,
    files: list[dict] | None = None,
) -> dict:
    rec = RawRequestRecord(
        run_id=_FIXTURE_RUN_ID,
        run_artifact_registry_id="reg_step2_bench",
        created_at="2026-06-18T00:00:00Z",
        raw_user_query=query,
        user_provided_context=UserProvidedContext(**(ctx or {})),
        uploaded_files=[UploadedFile(**f) for f in (files or [])],
    )
    out = rec.model_dump()
    out["artifact_id"] = "raw_request_record_bench"
    return out


def _parse(query: str, ctx: dict | None = None, files: list[dict] | None = None):
    agent = SupervisorAgent(llm=MockLLMProvider())
    return agent.parse_raw_to_structured_query(
        _raw(query=query, ctx=ctx, files=files)
    )


def _norm_index(sq, predicate) -> list:
    return [ne for ne in sq.normalized_entities if predicate(ne)]


def _decomp_index(sq, original_substr: str):
    return [
        d for d in sq.entity_decompositions
        if original_substr.lower() in d.original_text.lower()
    ]


def _only_decomp(sq, original_substr: str):
    matches = _decomp_index(sq, original_substr)
    assert matches, f"missing decomposition for {original_substr!r}"
    return matches[0]


# ── Benchmark 1: HER2 ADC comparing T-DM1 vs T-DXd ─────────────────────────


def test_benchmark1_her2_compare_tdm1_vs_tdxd():
    sq = _parse(
        "We want to evaluate the HER2 ADC benchmark comparing T-DM1 vs "
        "T-DXd (Enhertu) for breast cancer treatment.",
    )
    assert sq.task_intent.primary_intent == "existing_adc_evaluation"
    assert "literature_review" in sq.task_intent.secondary_intents
    assert "developability_assessment" in sq.task_intent.secondary_intents
    # Confidence is non-trivial.
    assert sq.task_intent.primary_intent_confidence >= 0.5

    # HER2 → ERBB2 normalization.
    her2_norms = _norm_index(
        sq, lambda ne: ne.canonical_name == "ERBB2"
    )
    assert her2_norms, "HER2 must normalize to ERBB2"
    # T-DM1 + T-DXd normalized + decomposed.
    drug_names = {
        ne.canonical_name for ne in sq.normalized_entities
        if ne.entity_type == "drug"
    }
    assert "ado-trastuzumab emtansine" in drug_names
    assert "trastuzumab deruxtecan" in drug_names
    tdm1_decomp = _decomp_index(sq, "T-DM1")
    assert tdm1_decomp, "T-DM1 must produce a decomposition"
    tdm1_components = {c.canonical_name for c in tdm1_decomp[0].components}
    assert any("trastuzumab" in c for c in tdm1_components)
    assert any("emtansine" in c.lower() for c in tdm1_components)
    tdxd_decomp = _decomp_index(sq, "T-DXd")
    assert tdxd_decomp
    tdxd_roles = {c.role for c in tdxd_decomp[0].components}
    assert "antibody" in tdxd_roles and (
        "linker_payload" in tdxd_roles or "payload" in tdxd_roles
    )

    # Requested outputs cover the professor's expected set.
    for out in (
        "evidence_summary",
        "developability_summary",
        "data_gap_summary",
        "case_study_summary",
        "report",
    ):
        assert out in sq.requested_outputs, f"missing {out!r}"


# ── whole-ADC alias decomposition inferred flags ───────────────────────────

def test_tdxd_only_decomposition_marks_all_components_inferred():
    sq = _parse("Evaluate T-DXd.")
    decomp = _only_decomp(sq, "T-DXd")
    assert {c.role for c in decomp.components} == {
        "antibody",
        "linker_payload",
        "payload",
    }
    assert all(c.inferred is True for c in decomp.components)


def test_enhertu_only_decomposition_marks_all_components_inferred():
    sq = _parse("Evaluate Enhertu.")
    decomp = _only_decomp(sq, "Enhertu")
    assert {c.role for c in decomp.components} == {
        "antibody",
        "linker_payload",
        "payload",
    }
    assert all(c.inferred is True for c in decomp.components)


def test_tdm1_only_decomposition_marks_all_components_inferred():
    sq = _parse("Evaluate T-DM1.")
    decomp = _only_decomp(sq, "T-DM1")
    assert {c.role for c in decomp.components} == {"antibody", "payload"}
    assert all(c.inferred is True for c in decomp.components)


def test_tdxd_with_explicit_dxd_payload_marks_payload_explicit():
    sq = _parse("Evaluate T-DXd with DXd payload.")
    decomp = _only_decomp(sq, "T-DXd")
    components_by_role = {c.role: c for c in decomp.components}
    assert components_by_role["antibody"].inferred is True
    assert components_by_role["linker_payload"].inferred is True
    assert components_by_role["payload"].inferred is False


# ── Benchmark 2: New TROP2 ADC with MMAE (linker/antibody missing) ─────────


def test_benchmark2_new_trop2_adc_with_mmae():
    sq = _parse(
        "Design a new TROP2 ADC with MMAE payload for solid tumors.",
    )
    assert sq.task_intent.primary_intent == "new_adc_design"
    assert "structure_analysis" in sq.task_intent.secondary_intents
    assert "developability_assessment" in sq.task_intent.secondary_intents

    # TROP2 → TACSTD2.
    trop2_norm = _norm_index(
        sq, lambda ne: ne.canonical_name == "TACSTD2"
    )
    assert trop2_norm

    # MMAE → monomethyl auristatin E.
    mmae_norm = _norm_index(
        sq, lambda ne: ne.canonical_name == "monomethyl auristatin E"
    )
    assert mmae_norm

    # Clarification questions about missing linker / antibody.
    cqs = " | ".join(sq.clarification_questions).lower()
    assert "linker" in cqs or "antibody" in cqs

    # parse_warnings remains internal and includes antibody/linker gaps.
    warning_blob = " | ".join(sq.parse_warnings).lower()
    assert "antibody" in warning_blob or "candidate" in warning_blob


# ── Benchmark 3: HER2 PDB 1N8Z + ZINC compounds ───────────────────────────


def test_benchmark3_her2_pdb_zinc_screening():
    sq = _parse(
        "Validate the structure of HER2 using PDB 1N8Z and screen ZINC12345 "
        "and ZINC67890 compounds against it.",
    )
    assert sq.task_intent.primary_intent == "structure_analysis"
    assert "compound_screening" in sq.task_intent.secondary_intents
    assert "structure_validation_report" in sq.requested_outputs
    assert "compound_screening_results" in sq.requested_outputs

    # ZINC IDs and PDB ID picked up via referenced_inputs.
    ref_types = {r["id_type"] for r in sq.referenced_inputs}
    assert "pdb_id" in ref_types
    assert "zinc_id" in ref_types
    # ZINC never labeled zinc22.
    for ref in sq.referenced_inputs:
        assert ref.get("id_type") != "zinc22"

    # Clarification: general HER2 screening vs ADC payload/linker workflow.
    cqs = " | ".join(sq.clarification_questions).lower()
    assert "her2" in cqs or "adc" in cqs


# ── Benchmark 4: CLDN18.2 ADC patent review with deruxtecan-like payload ──


def test_benchmark4_cldn18_2_patent_review():
    sq = _parse(
        "Run a patent / IP review on CLDN18.2 ADCs with deruxtecan-like "
        "payloads — we want to understand prior art.",
    )
    assert sq.task_intent.primary_intent == "patent_ip_review"
    # secondary may include literature_review.
    assert "literature_review" in sq.task_intent.secondary_intents

    # CLDN18.2 → CLDN18 isoform 2.
    cldn_norm = _norm_index(
        sq, lambda ne: ne.canonical_name == "CLDN18 isoform 2"
    )
    assert cldn_norm
    # deruxtecan normalized.
    deruxtecan_norm = _norm_index(
        sq, lambda ne: ne.canonical_name == "deruxtecan"
    )
    assert deruxtecan_norm

    assert "patent_or_ip_summary" in sq.requested_outputs
    assert "data_gap_summary" in sq.requested_outputs

    # Clarification surfaced for patent scope.
    cqs = " | ".join(sq.clarification_questions).lower()
    assert "patent" in cqs or "deruxtecan" in cqs or "cldn18" in cqs


# ── Benchmark 5: Literature-only trastuzumab + MMAE ───────────────────────


def test_benchmark5_literature_only_trastuzumab_mmae():
    sq = _parse(
        "Please review the literature on trastuzumab and MMAE for me.",
    )
    assert sq.task_intent.primary_intent == "literature_review"
    assert "literature_review_summary" in sq.requested_outputs
    assert "evidence_summary" in sq.requested_outputs

    # HER2 / ERBB2 must NOT be marked as an explicit user entity here —
    # the user said "trastuzumab + MMAE" without writing HER2.
    her2_norms = [
        ne for ne in sq.normalized_entities
        if ne.canonical_name == "ERBB2"
    ]
    # If the parser inferred ERBB2 from trastuzumab, it must mark inferred.
    for ne in her2_norms:
        assert ne.explicit_or_inferred == "inferred"

    # Trastuzumab normalized to canonical antibody record.
    trastuzumab_norms = _norm_index(
        sq, lambda ne: ne.canonical_name == "trastuzumab"
    )
    assert trastuzumab_norms

    # Clarification surfaced — is this HER2 ADC literature?
    cqs = " | ".join(sq.clarification_questions).lower()
    assert "her2" in cqs or "adc" in cqs


# ── Benchmark 6: ChEMBL / ZINC compounds as HER2 payload candidates ───────


def test_benchmark6_chembl_zinc_as_her2_payload_candidates():
    sq = _parse(
        "Screen CHEMBL1201585 and ZINC98765 compounds as possible payload "
        "candidates for a HER2 ADC.",
    )
    assert sq.task_intent.primary_intent == "compound_screening"
    assert "developability_assessment" in sq.task_intent.secondary_intents
    assert "literature_review" in sq.task_intent.secondary_intents

    # Source IDs preserved.
    types = {r["id_type"] for r in sq.referenced_inputs}
    assert "chembl_id" in types
    assert "zinc_id" in types

    # Warn that this is not a complete ADC workflow (antibody/linker
    # missing). The mock surfaces it via clarification_questions; the
    # important thing is the gap is visible to the user.
    cqs = " | ".join(sq.clarification_questions).lower()
    pw = " | ".join(sq.parse_warnings).lower()
    assert (
        "antibody" in cqs or "linker" in cqs or "antibody" in pw
        or "candidate" in pw
    )

    # Compound screening output requested.
    assert "compound_screening_results" in sq.requested_outputs


# ── parse_warnings vs clarification_questions separation ──────────────────


def test_parse_warnings_and_clarification_questions_are_distinct_channels():
    sq = _parse(
        "Design a new TROP2 ADC with MMAE payload.",
    )
    # Internal parser warnings include missing-field gaps.
    warning_text = " ".join(sq.parse_warnings).lower()
    assert "antibody" in warning_text or "candidate" in warning_text
    # User-facing clarification questions exist and are NOT the same
    # strings as the parser warnings.
    assert sq.clarification_questions
    overlap = set(sq.parse_warnings) & set(sq.clarification_questions)
    assert not overlap, (
        f"parse_warnings and clarification_questions should not overlap, "
        f"got: {overlap}"
    )


# ── Step 2 path imports nothing from app.mcp / app.a2a / file bytes ───────


@pytest.mark.parametrize(
    "module_name",
    [
        "app.agents.supervisor_agent",
        "app.llm.provider",
        "app.llm.gemini_provider",
        "app.schemas.step_02_structured_query",
    ],
)
def test_step2_modules_have_no_mcp_a2a_imports(module_name):
    module_path = Path(importlib.import_module(module_name).__file__)
    tree = ast.parse(module_path.read_text())
    bad: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if mod.startswith(("app.mcp", "app.a2a")):
                bad.append(f"from {mod} import …")
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith(("app.mcp", "app.a2a")):
                    bad.append(f"import {alias.name}")
    assert bad == [], f"{module_name} must not import MCP/A2A: {bad}"


def test_step2_never_reads_uploaded_file_bytes(tmp_path):
    """Sanity: even if a file exists on disk under storage_path, Step 2
    never opens it — the prompt-inputs builder strips storage_path and the
    mock provider only inspects metadata."""
    real_file = tmp_path / "trastuzumab.fasta"
    real_file.write_text("MAGNETIC-FAKE-SEQUENCE")
    sq = _parse(
        "Design a HER2 ADC with vc-MMAE.",
        ctx={
            "target_or_antigen_text": "HER2",
            "candidate_text": "Trastuzumab",
            "payload_linker_text": "vc-MMAE",
        },
        files=[
            {
                "file_id": "f_fasta_001",
                "original_filename": "trastuzumab.fasta",
                "storage_path": str(real_file),
                "content_type": "text/x-fasta",
                "size_bytes": real_file.stat().st_size,
            }
        ],
    )
    blob = sq.model_dump_json()
    # The file's fake byte content must not appear anywhere.
    assert "MAGNETIC-FAKE-SEQUENCE" not in blob


# ── vc-MMAE decomposition: explicit MMAE, inferred linker ─────────────────


def test_vc_mmae_decomposition_marks_explicit_payload_inferred_linker():
    sq = _parse(
        "Design a HER2 ADC with vc-MMAE payload.",
        ctx={
            "target_or_antigen_text": "HER2",
            "payload_linker_text": "vc-MMAE",
        },
    )
    vc_decomp = _decomp_index(sq, "vc-MMAE")
    assert vc_decomp
    components_by_role = {c.role: c for c in vc_decomp[0].components}
    assert "linker" in components_by_role
    assert "payload" in components_by_role
    # MMAE component must be explicit (user wrote it via vc-MMAE alias).
    # Linker component remains inferred unless the user wrote it.
    assert components_by_role["payload"].inferred is False
    assert components_by_role["linker"].inferred is True
