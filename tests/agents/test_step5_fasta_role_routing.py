from __future__ import annotations

from app.agents.candidate_context_agent import CandidateContextAgent
from app.agents.step_06_available_fields import project_candidate_available_fields
from app.mcp.client import LocalMCPClient
from app.schemas.step_02_structured_query import (
    MentionedEntities,
    SourceRawRequestRef,
    StructuredQuery,
    TaskIntent,
)
from app.services.input_readiness_service import InputReadinessService
from app.services.intake_service import IntakeService
from app.services.workflow_setup_service import WorkflowSetupService
from app.utils.ids import new_artifact_id, new_file_id
from app.utils.time import now_iso


def _bootstrap(
    local_storage,
    registry_service,
    workflow_state_service,
    *,
    target: str | None = "HER2",
    antibody: str | None = "Trastuzumab",
    raw_user_query: str | None = None,
    uploaded_files: list[dict] | None = None,
):
    rec = IntakeService(local_storage, registry_service, workflow_state_service).submit(
        raw_user_query=raw_user_query or "Run developability context extraction.",
        user_provided_context={
            "target_or_antigen_text": target or "",
            "candidate_text": antibody or "",
        },
    )
    run_id = rec.run_id
    sq = StructuredQuery(
        run_id=run_id,
        parsed_at=now_iso(),
        source_raw_request_ref=SourceRawRequestRef(
            raw_request_record_id=registry_service.get(run_id).active_artifacts.raw_request_record_id
        ),
        task_intent=TaskIntent(task_type="developability_assessment"),
        mentioned_entities=MentionedEntities(
            target_or_antigen_text=target,
            antibody_candidate_text=antibody,
        ),
        referenced_inputs=[],
        raw_context={},
    )
    sq_id = new_artifact_id("structured_query")
    local_storage.write_json(
        local_storage.run_key(run_id, "inputs/structured_query.json"),
        {"artifact_id": sq_id, **sq.model_dump()},
    )
    registry_service.update_active(run_id, structured_query_id=sq_id)
    workflow_state_service.mark(run_id, "step_02", "completed")
    InputReadinessService(local_storage, registry_service, workflow_state_service).check(run_id)
    WorkflowSetupService(local_storage, registry_service, workflow_state_service).plan(run_id)

    if uploaded_files:
        # Intake already persists uploaded files into raw_request_record.json.
        # Re-open and patch to include the files for this test fixture.
        raw = local_storage.read_json(local_storage.run_key(run_id, "inputs/raw_request_record.json"))
        raw["uploaded_files"] = uploaded_files
        local_storage.write_json(local_storage.run_key(run_id, "inputs/raw_request_record.json"), raw)
    return run_id


def _run_step5(local_storage, registry_service, workflow_state_service, run_id):
    table = CandidateContextAgent(
        storage=local_storage,
        registry=registry_service,
        workflow_state=workflow_state_service,
        mcp_client=LocalMCPClient(),
    ).run(run_id)
    return table


def _record_by_type(table, candidate_type: str):
    return next(c for c in table.candidate_records if c.candidate_type == candidate_type)


def test_step5_uploaded_heavy_and_light_fasta_files_route_to_antibody_candidate(
    local_storage, registry_service, workflow_state_service
):
    heavy_fasta = {
        "file_id": new_file_id(),
        "original_filename": "heavy_chain.fasta",
        "storage_path": "/runs/inputs/heavy_chain.fasta",
    }
    light_fasta = {
        "file_id": new_file_id(),
        "original_filename": "light_chain.fasta",
        "storage_path": "/runs/inputs/light_chain.fasta",
    }
    run_id = _bootstrap(
        local_storage=local_storage,
        registry_service=registry_service,
        workflow_state_service=workflow_state_service,
        uploaded_files=[heavy_fasta, light_fasta],
        raw_user_query="Assess antibody developability for Trastuzumab and HER2 target.",
    )
    table = _run_step5(local_storage, registry_service, workflow_state_service, run_id)
    target = _record_by_type(table, "target_antigen")
    antibody = _record_by_type(table, "antibody")

    assert {m.material_type for m in target.materials if m.role == "target_sequence_reference"} == set()
    assert any(
        m.material_type == "antibody_heavy_chain_sequence" and m.value == heavy_fasta["storage_path"]
        for m in antibody.materials
    )
    assert any(
        m.material_type == "antibody_light_chain_sequence" and m.value == light_fasta["storage_path"]
        for m in antibody.materials
    )

    projection = project_candidate_available_fields(antibody.model_dump())
    assert projection.modality_summary.has_antibody_heavy_sequence
    assert projection.modality_summary.has_antibody_light_sequence


def test_step5_antibody_named_fasta_routes_to_antibody_sequence_reference(
    local_storage, registry_service, workflow_state_service
):
    uploaded = [
        {
            "file_id": new_file_id(),
            "original_filename": "trastuzumab.fasta",
            "storage_path": "/runs/inputs/trastuzumab.fasta",
        }
    ]
    run_id = _bootstrap(
        local_storage=local_storage,
        registry_service=registry_service,
        workflow_state_service=workflow_state_service,
        uploaded_files=uploaded,
        raw_user_query="Use the uploaded Trastuzumab FASTA.",
    )
    table = _run_step5(local_storage, registry_service, workflow_state_service, run_id)
    antibody = _record_by_type(table, "antibody")

    antibody_mats = [m for m in antibody.materials if m.material_type == "antibody_sequence_reference"]
    assert len(antibody_mats) == 1
    assert antibody_mats[0].value == uploaded[0]["storage_path"]
    assert not any(
        m.material_type in {"antibody_heavy_chain_sequence", "antibody_light_chain_sequence"}
        for m in antibody.materials
    )
    assert uploaded[0]["storage_path"] not in [m.value for m in _record_by_type(table, "target_antigen").materials]


def test_step5_antigen_cues_route_fasta_to_target_sequence(
    local_storage, registry_service, workflow_state_service
):
    antigen_fasta = {
        "file_id": new_file_id(),
        "original_filename": "antigen.fasta",
        "storage_path": "/runs/inputs/antigen.fasta",
    }
    run_id = _bootstrap(
        local_storage=local_storage,
        registry_service=registry_service,
        workflow_state_service=workflow_state_service,
        uploaded_files=[antigen_fasta],
        raw_user_query="Analyze target FASTA for HER2 antigen.",
    )
    table = _run_step5(local_storage, registry_service, workflow_state_service, run_id)
    target = _record_by_type(table, "target_antigen")

    target_fasta = [
        m for m in target.materials if m.material_type == "target_sequence"
    ]
    assert len(target_fasta) == 1
    assert target_fasta[0].value == antigen_fasta["storage_path"]

    projection = project_candidate_available_fields(target.model_dump())
    assert projection.modality_summary.has_antigen_sequence is True


def test_step5_ambiguous_fasta_keeps_unassigned_notes_not_target_sequence(
    local_storage, registry_service, workflow_state_service
):
    ambiguous = {
        "file_id": new_file_id(),
        "original_filename": "sequence.fasta",
        "storage_path": "/runs/inputs/sequence.fasta",
    }
    run_id = _bootstrap(
        local_storage=local_storage,
        registry_service=registry_service,
        workflow_state_service=workflow_state_service,
        uploaded_files=[ambiguous],
        raw_user_query="Use the provided sequence file.",
    )
    table = _run_step5(local_storage, registry_service, workflow_state_service, run_id)

    target = _record_by_type(table, "target_antigen")
    assert not any(
        m.material_type == "target_sequence" and m.value == ambiguous["storage_path"]
        for m in target.materials
    )
    assert any(
        (ambiguous["storage_path"] in gap) or (ambiguous["file_id"] in gap)
        for gap in target.data_gaps
    )
    assert any(
        note.startswith("sequence_file_unassigned_for_target_skipped:")
        for note in target.context_notes
    )
