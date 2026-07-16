"""Step 3 batch-3 readiness signals + severity policy regression.

Covers the production plan in
`\u9879\u76ee\u6587\u4ef6/Step1_4_Orchestration_Component_Plan_v0.1.md §Step 3`:

- target_in_structured_query satisfies target readiness (already covered
  in deeper tests; sanity-check that the new evidence fields are populated).
- user_provided_context target satisfies target readiness.
- uploaded PDB / CIF inferred as structure.
- uploaded FASTA inferred as sequence.
- uploaded CSV / XLSX inferred as candidate table; `has_candidate_file`
  signal flips on.
- missing target blocks.
- missing antibody warning (not blocking, not silent pass).
- missing payload/linker warnings (payload warning, linker optional).
- non-ADC request not ready.
- failed upload produces checklist item.
- Step 3 path uses no LLM / no MCP.
"""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest

from app.services.intake_service import IntakeService
from app.services.input_readiness_service import (
    InputReadinessService,
    _infer_file_role,
    _is_adc_task_intent,
)
from app.schemas.step_02_structured_query import (
    MentionedEntities,
    SourceRawRequestRef,
    StructuredQuery,
    TaskIntent,
)
from app.utils.ids import new_artifact_id, new_file_id
from app.utils.time import now_iso


def _bootstrap_step_2(
    local_storage,
    registry_service,
    workflow_state_service,
    run_id,
    *,
    target=None,
    candidate=None,
    payload=None,
    linker=None,
    referenced_inputs=None,
    task_type="adc_design",
    modality="ADC",
    modality_confidence=0.9,
):
    reg = registry_service.get(run_id)
    sq = StructuredQuery(
        run_id=run_id,
        parsed_at=now_iso(),
        source_raw_request_ref=SourceRawRequestRef(
            raw_request_record_id=reg.active_artifacts.raw_request_record_id
        ),
        task_intent=TaskIntent(
            task_type=task_type,
            modality=modality,
            modality_confidence=modality_confidence,
        ),
        mentioned_entities=MentionedEntities(
            target_or_antigen_text=target,
            antibody_candidate_text=candidate,
            payload_text=payload,
            linker_text=linker,
        ),
        referenced_inputs=referenced_inputs or [],
    )
    sq_id = new_artifact_id("structured_query")
    local_storage.write_json(
        local_storage.run_key(run_id, "inputs/structured_query.json"),
        {"artifact_id": sq_id, **sq.model_dump()},
    )
    registry_service.update_active(run_id, structured_query_id=sq_id)
    workflow_state_service.mark(run_id, "step_02", "completed")


# ── _infer_file_role pure-function coverage ────────────────────────────────


@pytest.mark.parametrize(
    "filename,content_type,expected",
    [
        ("complex.pdb", "chemical/x-pdb", "pdb_or_cif_structure"),
        ("complex.cif", None, "pdb_or_cif_structure"),
        ("model.mmcif", None, "pdb_or_cif_structure"),
        ("model.ent", None, "pdb_or_cif_structure"),
        ("heavy.fasta", "text/x-fasta", "fasta_sequence"),
        ("heavy.fa", None, "fasta_sequence"),
        ("seqs.faa", None, "fasta_sequence"),
        ("candidates.csv", "text/csv", "csv_or_table"),
        ("candidates.tsv", None, "csv_or_table"),
        ("candidates.xlsx", "application/vnd.ms-excel", "csv_or_table"),
        ("meta.json", "application/json", "json_metadata"),
        ("scheme.svg", None, "image"),
        ("readme.txt", "text/plain", "unknown"),
        ("notes.docx", None, "unknown"),
        ("brief.pdf", "application/pdf", "unknown"),
        ("opaque.bin", None, "unknown"),
    ],
)
def test_infer_file_role_extension_and_content_type(filename, content_type, expected):
    role, note = _infer_file_role(filename, content_type)
    assert role == expected
    if expected == "unknown" and (filename.endswith((".txt", ".pdf", ".docx"))
                                  or content_type in {"text/plain", "application/pdf"}):
        assert note is not None
        assert "document" in note


# ── target presence via structured_query vs raw context ────────────────────


def test_target_in_user_context_satisfies_target_readiness(
    local_storage, registry_service, workflow_state_service
):
    intake = IntakeService(local_storage, registry_service, workflow_state_service)
    rec = intake.submit(
        raw_user_query="HER2 ADC", user_provided_context={"target_or_antigen_text": "HER2"}
    )
    _bootstrap_step_2(
        local_storage, registry_service, workflow_state_service, rec.run_id,
    )
    out = InputReadinessService(
        local_storage, registry_service, workflow_state_service
    ).check(rec.run_id)
    assert out.basic_adc_input_presence.target_or_antigen_present
    assert (
        out.basic_adc_input_presence.target_evidence
        == "raw_request_record.user_provided_context.target_or_antigen_text"
    )


# ── uploaded files: structure / sequence / candidate ──────────────────────


def _intake_with_file(
    intake, raw_user_query, ctx, filename, content_type, *, file_id=None
):
    return intake.submit(
        raw_user_query=raw_user_query,
        user_provided_context=ctx,
        uploaded_files=[
            {
                "file_id": file_id or new_file_id(),
                "original_filename": filename,
                "storage_path": f"/upload/{filename}",
                "content_type": content_type,
                "sha256": "sha256:fixture",
                "size_bytes": 1024,
            }
        ],
    )


def test_uploaded_cif_inferred_as_structure_present(
    local_storage, registry_service, workflow_state_service
):
    intake = IntakeService(local_storage, registry_service, workflow_state_service)
    rec = _intake_with_file(
        intake,
        "HER2 ADC",
        {
            "target_or_antigen_text": "HER2",
            "candidate_text": "Trastuzumab",
            "payload_linker_text": "vc-MMAE",
        },
        "1n8z.cif",
        None,
    )
    _bootstrap_step_2(
        local_storage, registry_service, workflow_state_service, rec.run_id,
        target="HER2", candidate="Trastuzumab", payload="MMAE", linker="vc",
    )
    out = InputReadinessService(
        local_storage, registry_service, workflow_state_service
    ).check(rec.run_id)
    assert out.uploaded_file_checks[0].inferred_role == "pdb_or_cif_structure"
    assert out.basic_adc_input_presence.structure_input_present
    assert out.basic_adc_input_presence.structure_or_sequence_present
    assert out.basic_adc_input_presence.structure_input_evidence is not None


def test_uploaded_fasta_inferred_as_sequence_present(
    local_storage, registry_service, workflow_state_service
):
    intake = IntakeService(local_storage, registry_service, workflow_state_service)
    rec = _intake_with_file(
        intake,
        "HER2 ADC",
        {
            "target_or_antigen_text": "HER2",
            "candidate_text": "Trastuzumab",
            "payload_linker_text": "vc-MMAE",
        },
        "trastuzumab.fasta",
        "text/x-fasta",
    )
    file_id = rec.uploaded_files[0].file_id
    _bootstrap_step_2(
        local_storage, registry_service, workflow_state_service, rec.run_id,
        target="HER2", candidate="Trastuzumab", payload="MMAE", linker="vc",
        referenced_inputs=[
            {
                "id_type": "uploaded_file",
                "value": file_id,
                "source": "antibody_heavy_chain_sequence",
            }
        ],
    )
    out = InputReadinessService(
        local_storage, registry_service, workflow_state_service
    ).check(rec.run_id)
    assert out.uploaded_file_checks[0].inferred_role == "fasta_sequence"
    assert out.basic_adc_input_presence.sequence_input_present
    assert out.basic_adc_input_presence.sequence_input_evidence == (
        "structured_query.referenced_inputs[id_type=uploaded_file,"
        "source=antibody_heavy_chain_sequence]"
    )
    assert not out.basic_adc_input_presence.structure_input_present
    assert out.basic_adc_input_presence.sequence_input_evidence is not None


@pytest.mark.parametrize("filename,content_type", [
    ("candidates.csv", "text/csv"),
    ("candidates.xlsx", "application/vnd.ms-excel"),
])
def test_uploaded_csv_or_xlsx_inferred_as_candidate_file(
    local_storage, registry_service, workflow_state_service, filename, content_type
):
    intake = IntakeService(local_storage, registry_service, workflow_state_service)
    rec = _intake_with_file(
        intake,
        "Rank these ADC candidates",
        {
            "target_or_antigen_text": "HER2",
            "candidate_text": "from spreadsheet",
            "payload_linker_text": "vc-MMAE",
        },
        filename,
        content_type,
    )
    _bootstrap_step_2(
        local_storage, registry_service, workflow_state_service, rec.run_id,
        target="HER2", candidate="from spreadsheet", payload="MMAE", linker="vc",
    )
    out = InputReadinessService(
        local_storage, registry_service, workflow_state_service
    ).check(rec.run_id)
    assert out.uploaded_file_checks[0].inferred_role == "csv_or_table"
    assert out.basic_adc_input_presence.candidate_file_present
    assert (
        out.basic_adc_input_presence.candidate_file_evidence
        == "raw_request_record.uploaded_files[*].inferred_role=csv_or_table"
    )


# ── severity policy ────────────────────────────────────────────────────────


def test_missing_target_blocks_overall_status(
    local_storage, registry_service, workflow_state_service
):
    intake = IntakeService(local_storage, registry_service, workflow_state_service)
    rec = intake.submit(raw_user_query="design", user_provided_context={})
    _bootstrap_step_2(local_storage, registry_service, workflow_state_service, rec.run_id)
    out = InputReadinessService(
        local_storage, registry_service, workflow_state_service
    ).check(rec.run_id)
    assert out.input_readiness_status == "needs_user_input"
    cats = [(m.category, m.severity) for m in out.missing_input_checklist]
    assert ("target", "blocking") in cats


def test_missing_antibody_is_warning_not_silent_pass(
    local_storage, registry_service, workflow_state_service
):
    intake = IntakeService(local_storage, registry_service, workflow_state_service)
    rec = intake.submit(
        raw_user_query="HER2 ADC MMAE",
        user_provided_context={
            "target_or_antigen_text": "HER2",
            "payload_linker_text": "vc-MMAE",
        },
    )
    _bootstrap_step_2(
        local_storage, registry_service, workflow_state_service, rec.run_id,
        target="HER2", payload="MMAE", linker="vc",
    )
    out = InputReadinessService(
        local_storage, registry_service, workflow_state_service
    ).check(rec.run_id)
    assert out.input_readiness_status == "needs_user_input"
    cats = [(m.category, m.severity) for m in out.missing_input_checklist]
    assert ("antibody", "warning") in cats


def test_missing_payload_warning_and_linker_optional(
    local_storage, registry_service, workflow_state_service
):
    intake = IntakeService(local_storage, registry_service, workflow_state_service)
    rec = intake.submit(
        raw_user_query="HER2 ADC with trastuzumab",
        user_provided_context={
            "target_or_antigen_text": "HER2",
            "candidate_text": "Trastuzumab",
        },
    )
    _bootstrap_step_2(
        local_storage, registry_service, workflow_state_service, rec.run_id,
        target="HER2", candidate="Trastuzumab",
    )
    out = InputReadinessService(
        local_storage, registry_service, workflow_state_service
    ).check(rec.run_id)
    cats = {(m.category, m.severity) for m in out.missing_input_checklist}
    # Payload missing → warning; linker missing → optional.
    payload_severities = {sev for cat, sev in cats if cat == "payload_or_linker"}
    assert "warning" in payload_severities
    assert "optional" in payload_severities
    assert out.input_readiness_status == "needs_user_input"


# ── ADC task intent / non-ADC blocking ────────────────────────────────────


def test_is_adc_task_intent_helper_accepts_modality_adc():
    ok, ev = _is_adc_task_intent({"task_intent": {"modality": "ADC"}})
    assert ok and ev == "structured_query.task_intent.modality=ADC"


def test_is_adc_task_intent_helper_accepts_task_type_alone():
    ok, ev = _is_adc_task_intent(
        {"task_intent": {"modality": "unknown", "task_type": "adc_design"}}
    )
    assert ok and ev.endswith("task_type=adc_design")


def test_is_adc_task_intent_helper_rejects_unknown():
    ok, ev = _is_adc_task_intent(
        {"task_intent": {"modality": "unknown", "task_type": "unknown"}}
    )
    assert not ok and ev is None


def test_non_adc_request_high_confidence_blocks_pipeline(
    local_storage, registry_service, workflow_state_service
):
    """`modality="other"` with non-trivial confidence → blocking task_intent gap."""
    intake = IntakeService(local_storage, registry_service, workflow_state_service)
    rec = intake.submit(
        raw_user_query="Summarize bispecific patents please",
        user_provided_context={"target_or_antigen_text": "HER2"},
    )
    _bootstrap_step_2(
        local_storage, registry_service, workflow_state_service, rec.run_id,
        target="HER2", task_type="patent_review", modality="other",
        modality_confidence=0.8,
    )
    out = InputReadinessService(
        local_storage, registry_service, workflow_state_service
    ).check(rec.run_id)
    assert out.input_readiness_status == "blocked"
    cats = [(m.category, m.severity) for m in out.missing_input_checklist]
    assert ("task_intent", "blocking") in cats
    assert not out.basic_adc_input_presence.adc_task_intent_present


def test_non_adc_request_low_confidence_only_warns(
    local_storage, registry_service, workflow_state_service
):
    """`modality="unknown"` or low confidence → warning, not block."""
    intake = IntakeService(local_storage, registry_service, workflow_state_service)
    rec = intake.submit(
        raw_user_query="Build something",
        user_provided_context={
            "target_or_antigen_text": "HER2",
            "candidate_text": "Trastuzumab",
        },
    )
    _bootstrap_step_2(
        local_storage, registry_service, workflow_state_service, rec.run_id,
        target="HER2", candidate="Trastuzumab",
        task_type="unknown", modality="unknown", modality_confidence=0.0,
    )
    out = InputReadinessService(
        local_storage, registry_service, workflow_state_service
    ).check(rec.run_id)
    assert out.input_readiness_status == "needs_user_input"
    cats = {(m.category, m.severity) for m in out.missing_input_checklist}
    assert ("task_intent", "warning") in cats
    # And it must NEVER be `ready` for a non-ADC request.
    assert out.input_readiness_status != "ready"


def test_empty_raw_user_query_is_blocking(
    local_storage, registry_service, workflow_state_service
):
    intake = IntakeService(local_storage, registry_service, workflow_state_service)
    rec = intake.submit(
        raw_user_query="   ",  # whitespace-only counts as empty
        user_provided_context={"target_or_antigen_text": "HER2"},
    )
    _bootstrap_step_2(
        local_storage, registry_service, workflow_state_service, rec.run_id,
        target="HER2",
    )
    out = InputReadinessService(
        local_storage, registry_service, workflow_state_service
    ).check(rec.run_id)
    assert out.input_readiness_status == "blocked"
    cats = [(m.category, m.severity) for m in out.missing_input_checklist]
    assert ("raw_user_query", "blocking") in cats


# ── failed upload checklist ────────────────────────────────────────────────


def test_failed_upload_surfaces_checklist_item(
    local_storage, registry_service, workflow_state_service
):
    """If intake recorded the file but no storage_path, Step 3 says so."""
    intake = IntakeService(local_storage, registry_service, workflow_state_service)
    # Submit through intake then hand-edit the persisted raw_request_record
    # to simulate the upload-persistence failure (intake itself blocks on
    # missing storage_path, which is the correct end-to-end behavior).
    rec = intake.submit(
        raw_user_query="HER2 ADC",
        user_provided_context={
            "target_or_antigen_text": "HER2",
            "candidate_text": "Trastuzumab",
            "payload_linker_text": "vc-MMAE",
        },
    )
    key = local_storage.run_key(rec.run_id, "inputs/raw_request_record.json")
    raw = local_storage.read_json(key)
    raw["uploaded_files"] = [
        {
            "file_id": "f_failed",
            "original_filename": "broken.pdb",
            "storage_path": "",  # ← persistence failure
            "content_type": "chemical/x-pdb",
            "sha256": None,
            "size_bytes": None,
        }
    ]
    local_storage.write_json(key, raw)
    _bootstrap_step_2(
        local_storage, registry_service, workflow_state_service, rec.run_id,
        target="HER2", candidate="Trastuzumab", payload="MMAE", linker="vc",
    )
    out = InputReadinessService(
        local_storage, registry_service, workflow_state_service
    ).check(rec.run_id)
    upload_items = [m for m in out.missing_input_checklist if m.category == "uploaded_file"]
    assert upload_items, "expected an uploaded_file checklist entry"
    assert any("f_failed" in m.field for m in upload_items)
    assert any(m.severity == "warning" for m in upload_items)


def test_unknown_role_upload_surfaces_warning(
    local_storage, registry_service, workflow_state_service
):
    intake = IntakeService(local_storage, registry_service, workflow_state_service)
    rec = _intake_with_file(
        intake,
        "HER2 ADC",
        {
            "target_or_antigen_text": "HER2",
            "candidate_text": "Trastuzumab",
            "payload_linker_text": "vc-MMAE",
        },
        "notes.txt",
        "text/plain",
        file_id="f_notes",
    )
    _bootstrap_step_2(
        local_storage, registry_service, workflow_state_service, rec.run_id,
        target="HER2", candidate="Trastuzumab", payload="MMAE", linker="vc",
    )
    out = InputReadinessService(
        local_storage, registry_service, workflow_state_service
    ).check(rec.run_id)
    upload_warnings = [
        m for m in out.missing_input_checklist
        if m.category == "uploaded_file" and "f_notes" in m.field
    ]
    assert upload_warnings
    assert upload_warnings[0].severity == "warning"


# ── readiness_summary readability ─────────────────────────────────────────


def test_readiness_summary_is_user_legible(
    local_storage, registry_service, workflow_state_service
):
    intake = IntakeService(local_storage, registry_service, workflow_state_service)
    rec = intake.submit(
        raw_user_query="HER2 ADC", user_provided_context={"target_or_antigen_text": "HER2"}
    )
    _bootstrap_step_2(
        local_storage, registry_service, workflow_state_service, rec.run_id, target="HER2",
    )
    out = InputReadinessService(
        local_storage, registry_service, workflow_state_service
    ).check(rec.run_id)
    assert "needs_user_input" in out.readiness_summary
    assert "antibody" in out.readiness_summary or "payload_or_linker" in out.readiness_summary


# ── Step 3 path uses no LLM / no MCP ──────────────────────────────────────


def test_input_readiness_module_source_has_no_llm_or_mcp_imports():
    """Static guarantee: the Step 3 module never imports LLM / MCP code."""
    module_path = Path(
        importlib.import_module("app.services.input_readiness_service").__file__
    )
    tree = ast.parse(module_path.read_text())
    bad: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if mod.startswith(("app.llm", "app.mcp", "app.a2a")):
                bad.append(f"from {mod} import …")
            if mod in {"openai", "anthropic", "google.genai"}:
                bad.append(f"from {mod} import …")
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith(("app.llm", "app.mcp", "app.a2a")):
                    bad.append(f"import {alias.name}")
                if alias.name in {"openai", "anthropic", "google.genai"}:
                    bad.append(f"import {alias.name}")
    assert bad == [], f"Step 3 must not import LLM/MCP: {bad}"


def test_input_readiness_check_does_not_touch_mcp(
    local_storage, registry_service, workflow_state_service, monkeypatch
):
    """Run-time guarantee: importing the MCP universe singleton stays unbuilt."""
    from app.mcp import tooluniverse_adapter

    tooluniverse_adapter._reset_for_tests()
    sentinel = {"built": False}

    def _explode():
        sentinel["built"] = True
        raise AssertionError("Step 3 must not build the ToolUniverse singleton")

    monkeypatch.setattr(tooluniverse_adapter, "_get_universe", _explode)
    intake = IntakeService(local_storage, registry_service, workflow_state_service)
    rec = intake.submit(
        raw_user_query="HER2 ADC with vc-MMAE",
        user_provided_context={
            "target_or_antigen_text": "HER2",
            "candidate_text": "Trastuzumab",
            "payload_linker_text": "vc-MMAE",
        },
    )
    _bootstrap_step_2(
        local_storage, registry_service, workflow_state_service, rec.run_id,
        target="HER2", candidate="Trastuzumab", payload="MMAE", linker="vc",
    )
    InputReadinessService(
        local_storage, registry_service, workflow_state_service
    ).check(rec.run_id)
    assert sentinel["built"] is False
