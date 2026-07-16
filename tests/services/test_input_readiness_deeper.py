"""Step 3 deeper readiness tests (structured_query + uploaded file metadata)."""

from __future__ import annotations

from app.services.intake_service import IntakeService
from app.services.input_readiness_service import InputReadinessService
from app.schemas.step_02_structured_query import (
    MentionedEntities,
    SourceRawRequestRef,
    StructuredQuery,
    TaskIntent,
)
from app.utils.ids import new_artifact_id, new_file_id
from app.utils.time import now_iso


def _bootstrap_step_2(
    local_storage, registry_service, workflow_state_service, run_id, *,
    target=None, candidate=None, payload=None, linker=None, referenced_inputs=None,
):
    reg = registry_service.get(run_id)
    sq = StructuredQuery(
        run_id=run_id,
        parsed_at=now_iso(),
        source_raw_request_ref=SourceRawRequestRef(
            raw_request_record_id=reg.active_artifacts.raw_request_record_id
        ),
        task_intent=TaskIntent(task_type="adc_design"),
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


def test_step3_uses_structured_query_target_even_when_raw_sparse(
    local_storage, registry_service, workflow_state_service
):
    """Raw context has no target but structured_query.mentioned_entities does."""
    intake = IntakeService(local_storage, registry_service, workflow_state_service)
    rec = intake.submit(raw_user_query="design ADC", user_provided_context={})
    _bootstrap_step_2(
        local_storage, registry_service, workflow_state_service, rec.run_id,
        target="HER2", payload="MMAE", linker="vc",
    )
    out = InputReadinessService(local_storage, registry_service, workflow_state_service).check(rec.run_id)
    assert out.basic_adc_input_presence.target_or_antigen_present
    assert out.basic_adc_input_presence.target_evidence == (
        "structured_query.mentioned_entities.target_or_antigen_text"
    )


def test_step3_missing_target_is_blocking(
    local_storage, registry_service, workflow_state_service
):
    intake = IntakeService(local_storage, registry_service, workflow_state_service)
    rec = intake.submit(raw_user_query="x", user_provided_context={})
    _bootstrap_step_2(local_storage, registry_service, workflow_state_service, rec.run_id)
    out = InputReadinessService(local_storage, registry_service, workflow_state_service).check(rec.run_id)
    assert out.input_readiness_status == "needs_user_input"
    blocking = [m for m in out.missing_input_checklist if m.severity == "blocking"]
    assert blocking and blocking[0].category == "target"


def test_step3_missing_antibody_is_warning_not_blocking(
    local_storage, registry_service, workflow_state_service
):
    intake = IntakeService(local_storage, registry_service, workflow_state_service)
    rec = intake.submit(raw_user_query="x", user_provided_context={"target_or_antigen_text": "HER2"})
    _bootstrap_step_2(
        local_storage, registry_service, workflow_state_service, rec.run_id, target="HER2"
    )
    out = InputReadinessService(local_storage, registry_service, workflow_state_service).check(rec.run_id)
    assert out.input_readiness_status == "needs_user_input"
    cats = [(m.category, m.severity) for m in out.missing_input_checklist]
    assert ("antibody", "warning") in cats


def test_step3_pdb_referenced_input_satisfies_structure_present(
    local_storage, registry_service, workflow_state_service
):
    intake = IntakeService(local_storage, registry_service, workflow_state_service)
    rec = intake.submit(
        raw_user_query="HER2 ADC MMAE",
        user_provided_context={
            "target_or_antigen_text": "HER2",
            "candidate_text": "Trastuzumab",
            "payload_linker_text": "vc-MMAE",
        },
    )
    _bootstrap_step_2(
        local_storage, registry_service, workflow_state_service, rec.run_id,
        target="HER2", candidate="Trastuzumab", payload="MMAE", linker="vc",
        referenced_inputs=[{"id_type": "pdb_id", "value": "1N8Z", "source": "raw_request_text"}],
    )
    out = InputReadinessService(local_storage, registry_service, workflow_state_service).check(rec.run_id)
    assert out.basic_adc_input_presence.structure_or_sequence_present
    assert out.basic_adc_input_presence.structure_or_sequence_evidence == (
        "structured_query.referenced_inputs[id_type=pdb_id]"
    )


def test_step3_uploaded_pdb_file_inferred_as_structure(
    local_storage, registry_service, workflow_state_service
):
    intake = IntakeService(local_storage, registry_service, workflow_state_service)
    rec = intake.submit(
        raw_user_query="HER2 ADC",
        user_provided_context={
            "target_or_antigen_text": "HER2",
            "candidate_text": "Trastuzumab",
            "payload_linker_text": "vc-MMAE",
        },
        uploaded_files=[
            {
                "file_id": new_file_id(),
                "original_filename": "complex.pdb",
                "storage_path": "/upload/complex.pdb",
                "content_type": "chemical/x-pdb",
                "sha256": "sha256:abc",
                "size_bytes": 1024,
            }
        ],
    )
    _bootstrap_step_2(
        local_storage, registry_service, workflow_state_service, rec.run_id,
        target="HER2", candidate="Trastuzumab", payload="MMAE", linker="vc",
    )
    out = InputReadinessService(local_storage, registry_service, workflow_state_service).check(rec.run_id)
    assert out.basic_adc_input_presence.structure_or_sequence_present
    role = out.uploaded_file_checks[0].inferred_role
    assert role == "pdb_or_cif_structure"


def test_step3_uploaded_fasta_file_inferred_as_sequence(
    local_storage, registry_service, workflow_state_service
):
    intake = IntakeService(local_storage, registry_service, workflow_state_service)
    file_id = new_file_id()
    rec = intake.submit(
        raw_user_query="HER2 ADC",
        user_provided_context={
            "target_or_antigen_text": "HER2",
            "candidate_text": "Trastuzumab",
            "payload_linker_text": "vc-MMAE",
        },
        uploaded_files=[
            {
                "file_id": file_id,
                "original_filename": "heavy_chain.fasta",
                "storage_path": "/upload/heavy_chain.fasta",
                "sha256": "sha256:def",
            }
        ],
    )
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
    out = InputReadinessService(local_storage, registry_service, workflow_state_service).check(rec.run_id)
    assert out.basic_adc_input_presence.structure_or_sequence_present
    assert out.uploaded_file_checks[0].inferred_role == "fasta_sequence"
