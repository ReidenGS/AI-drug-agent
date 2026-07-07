"""Step 5 protein variant / point-mutation identifierization.

Step 2 structures a protein variant as
``referenced_inputs[id_type="variant", value="V777L"]`` (alongside
``uniprot_id``). Step 5 must attach it as a typed identifier on the target
candidate so Step 9's input projection can surface
``identifier:uniprot_id:*`` + ``identifier:variant:*`` for the AlphaMissense /
DynaMut2 / ESM variant tools. A variant is never treated as the
target/antibody/payload itself.
"""

from __future__ import annotations

from app.agents.candidate_context_agent import CandidateContextAgent
from app.agents.step_09_input_projection import project_step9_inputs
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
from app.utils.ids import new_artifact_id
from app.utils.time import now_iso


VARIANT_QUERY = (
    "Evaluate the HER2 variant V777L using UniProt P04626. "
    "Use variant scoring only; do not generate protein sequences."
)


def _bootstrap(local_storage, registry_service, workflow_state_service, *, referenced_inputs):
    rec = IntakeService(local_storage, registry_service, workflow_state_service).submit(
        raw_user_query=VARIANT_QUERY,
        user_provided_context={"target_or_antigen_text": "HER2"},
    )
    run_id = rec.run_id
    sq = StructuredQuery(
        run_id=run_id,
        parsed_at=now_iso(),
        source_raw_request_ref=SourceRawRequestRef(
            raw_request_record_id=registry_service.get(run_id).active_artifacts.raw_request_record_id
        ),
        task_intent=TaskIntent(task_type="structure_analysis", primary_intent="structure_analysis"),
        mentioned_entities=MentionedEntities(target_or_antigen_text="HER2"),
        referenced_inputs=referenced_inputs,
        canonical_query=VARIANT_QUERY,
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
    return run_id, sq_id


def _run_step5(local_storage, registry_service, workflow_state_service, run_id):
    return CandidateContextAgent(
        storage=local_storage,
        registry=registry_service,
        workflow_state=workflow_state_service,
        mcp_client=LocalMCPClient(),
    ).run(run_id)


def _target(table):
    return next(c for c in table.candidate_records if c.candidate_type == "target_antigen")


def test_step5_variant_referenced_input_becomes_target_identifier(
    local_storage, registry_service, workflow_state_service
):
    run_id, sq_id = _bootstrap(
        local_storage, registry_service, workflow_state_service,
        referenced_inputs=[
            {"id_type": "uniprot_id", "value": "P04626", "source": "user"},
            {"id_type": "variant", "value": "V777L", "source": "user"},
        ],
    )
    table = _run_step5(local_storage, registry_service, workflow_state_service, run_id)
    target = _target(table)
    idents = {i.id_type: i for i in target.identifiers}
    assert idents["uniprot_id"].id_value == "P04626"
    assert idents["variant"].id_value == "V777L"
    assert sq_id in idents["variant"].source_ids

    # The variant is an identifier, never the candidate label/type itself.
    assert target.candidate_label != "V777L"
    # And it is not materialized as a target/antibody/payload material.
    assert all(m.value != "V777L" for m in target.materials)


def test_step5_variant_flows_to_step9_projection(
    local_storage, registry_service, workflow_state_service
):
    """End-to-end Step 5 -> Step 9 projection: the persisted
    candidate_context_table yields both uniprot_id and variant input fields."""
    run_id, _ = _bootstrap(
        local_storage, registry_service, workflow_state_service,
        referenced_inputs=[
            {"id_type": "uniprot_id", "value": "P04626", "source": "user"},
            {"id_type": "variant", "value": "V777L", "source": "user"},
        ],
    )
    _run_step5(local_storage, registry_service, workflow_state_service, run_id)
    cct = local_storage.read_json(
        local_storage.run_key(run_id, "candidate_context_table.json")
    )
    projection = project_step9_inputs(
        candidate_context_table=cct,
        prepared_structure_input_package=None,
        structure_prediction_and_interface_results=None,
    )
    refs = {f.field_ref for f in projection["input_fields"]}
    assert "identifier:uniprot_id:P04626" in refs
    assert "identifier:variant:V777L" in refs


def test_step5_mutation_id_type_also_identifierized(
    local_storage, registry_service, workflow_state_service
):
    run_id, _ = _bootstrap(
        local_storage, registry_service, workflow_state_service,
        referenced_inputs=[
            {"id_type": "uniprot_id", "value": "P04626", "source": "user"},
            {"id_type": "mutation", "value": "V777L", "source": "user"},
        ],
    )
    table = _run_step5(local_storage, registry_service, workflow_state_service, run_id)
    target = _target(table)
    assert any(i.id_type == "mutation" and i.id_value == "V777L" for i in target.identifiers)
