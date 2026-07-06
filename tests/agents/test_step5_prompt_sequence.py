"""Step 5 explicit ESM prompt_sequence (masked generation prompt) materialization.

Step 2 hands Step 5 exactly two shapes for a user-declared masked generation
prompt:

    inline:   {"id_type": "prompt_sequence", "value": "<masked>", "source": "user"}
    uploaded: {"id_type": "uploaded_file", "value": "<file_id>", "source": "prompt_sequence"}

Step 5 is the first step with storage access, so it reads the uploaded file's
content and runs the deterministic mask / protein-format validation Step 2
could not. A valid prompt becomes a dedicated `prompt_sequence` material whose
`value` is a STORAGE REF (never the raw masked prompt) — the raw generation
prompt must never land in the persisted candidate_context_table artifact.
Invalid / unreadable / mask-less inputs are NOT materialized as executable
prompts; they surface a compact data_gap instead. An ordinary complete
heavy/light/target sequence never becomes a prompt_sequence.
"""

from __future__ import annotations

import json

from app.agents.candidate_context_agent import CandidateContextAgent
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


# Masked generation prompts (contain "_" or "<mask>" positions). Kept short and
# never asserted-on directly inside a persisted artifact (privacy).
INLINE_MASKED_PROMPT = "EVQLVESGGG___LRLSCAAS_GFNIKDTYIHW"
UPLOADED_MASKED_FASTA = ">design_prompt\nEVQL<mask>VESGGG<mask>LRLSCAASGF\n"
ORDINARY_HEAVY = "EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGK"
ORDINARY_LIGHT = "DIQMTQSPSSLSASVGDRVTITCRASQGISSYLNWYQQKPGK"


def _bootstrap(
    local_storage,
    registry_service,
    workflow_state_service,
    *,
    referenced_inputs: list[dict],
    uploaded_files: list[dict] | None = None,
    raw_user_query: str = "Generate a candidate protein sequence from my masked prompt.",
):
    rec = IntakeService(local_storage, registry_service, workflow_state_service).submit(
        raw_user_query=raw_user_query,
        user_provided_context={},
    )
    run_id = rec.run_id
    sq = StructuredQuery(
        run_id=run_id,
        parsed_at=now_iso(),
        source_raw_request_ref=SourceRawRequestRef(
            raw_request_record_id=registry_service.get(run_id).active_artifacts.raw_request_record_id
        ),
        task_intent=TaskIntent(task_type="structure_analysis", primary_intent="structure_analysis"),
        mentioned_entities=MentionedEntities(),
        referenced_inputs=referenced_inputs,
        canonical_query=raw_user_query,
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
        raw = local_storage.read_json(local_storage.run_key(run_id, "inputs/raw_request_record.json"))
        raw["uploaded_files"] = uploaded_files
        local_storage.write_json(local_storage.run_key(run_id, "inputs/raw_request_record.json"), raw)
    return run_id, sq_id


def _run_step5(local_storage, registry_service, workflow_state_service, run_id):
    return CandidateContextAgent(
        storage=local_storage,
        registry=registry_service,
        workflow_state=workflow_state_service,
        mcp_client=LocalMCPClient(),
    ).run(run_id)


def _prompt_material(table):
    for cand in table.candidate_records:
        for m in cand.materials:
            if m.material_type == "prompt_sequence":
                return cand, m
    return None, None


def _all_prompt_data_gaps(table) -> list[str]:
    gaps: list[str] = []
    for cand in table.candidate_records:
        gaps.extend(g for g in cand.data_gaps if g.startswith("prompt_sequence_missing"))
    return gaps


# ── inline masked prompt ─────────────────────────────────────────────────────

def test_inline_masked_prompt_materialized_as_storage_ref(
    local_storage, registry_service, workflow_state_service
):
    run_id, sq_id = _bootstrap(
        local_storage, registry_service, workflow_state_service,
        referenced_inputs=[
            {"id_type": "prompt_sequence", "value": INLINE_MASKED_PROMPT, "source": "user"}
        ],
    )
    table = _run_step5(local_storage, registry_service, workflow_state_service, run_id)
    cand, mat = _prompt_material(table)
    assert mat is not None, "expected a prompt_sequence material"
    assert mat.material_type == "prompt_sequence"
    assert mat.value_format == "masked_amino_acid_sequence"
    assert mat.role == "protein_generation_prompt"
    assert mat.role_status == "explicit"
    # value is a STORAGE REF, never the raw masked prompt.
    assert mat.value != INLINE_MASKED_PROMPT
    assert "prompt_sequences" in mat.value
    assert mat.content_descriptor is not None
    assert mat.content_descriptor["has_mask"] is True
    assert mat.content_descriptor["value_is_storage_ref"] is True
    assert mat.content_descriptor["prompt_length"] == len(INLINE_MASKED_PROMPT)
    # traceable to the Step 2 structured_query artifact.
    assert sq_id in cand.source_records
    # The raw masked prompt itself is stored (storage-ref only) — the resolver
    # can recover it — but only the ref lives in the material value.
    assert local_storage.read_bytes(mat.value).decode("utf-8") == INLINE_MASKED_PROMPT


def test_inline_masked_prompt_no_raw_leak_in_persisted_artifact(
    local_storage, registry_service, workflow_state_service
):
    run_id, _ = _bootstrap(
        local_storage, registry_service, workflow_state_service,
        referenced_inputs=[
            {"id_type": "prompt_sequence", "value": INLINE_MASKED_PROMPT, "source": "user"}
        ],
    )
    _run_step5(local_storage, registry_service, workflow_state_service, run_id)
    persisted = local_storage.read_json(
        local_storage.run_key(run_id, "candidate_context_table.json")
    )
    blob = json.dumps(persisted)
    assert INLINE_MASKED_PROMPT not in blob
    # The residue-only body (mask markers removed) must not leak either.
    assert INLINE_MASKED_PROMPT.replace("_", "") not in blob


# ── uploaded masked prompt file ──────────────────────────────────────────────

def test_uploaded_prompt_sequence_file_materialized(
    local_storage, registry_service, workflow_state_service
):
    file_id = new_file_id()
    storage_path = local_storage.run_key("uploads_probe", "inputs", "design_prompt.fasta")
    local_storage.write_bytes(storage_path, UPLOADED_MASKED_FASTA.encode("utf-8"))
    run_id, _ = _bootstrap(
        local_storage, registry_service, workflow_state_service,
        referenced_inputs=[
            {"id_type": "uploaded_file", "value": file_id, "source": "prompt_sequence"}
        ],
        uploaded_files=[
            {
                "file_id": file_id,
                "original_filename": "design_prompt.fasta",
                "storage_path": storage_path,
            }
        ],
    )
    table = _run_step5(local_storage, registry_service, workflow_state_service, run_id)
    _, mat = _prompt_material(table)
    assert mat is not None
    assert mat.material_type == "prompt_sequence"
    assert mat.value_format == "masked_amino_acid_sequence"
    assert mat.content_descriptor["source_kind"] == "uploaded_file"
    assert mat.content_descriptor["has_mask"] is True
    # Stored masked prompt has the FASTA header stripped, mask markers kept.
    stored = local_storage.read_bytes(mat.value).decode("utf-8")
    assert "<mask>" in stored
    assert "design_prompt" not in stored  # FASTA header line dropped
    assert not stored.startswith(">")
    # No raw FASTA / prompt content in the persisted artifact.
    persisted = local_storage.read_json(
        local_storage.run_key(run_id, "candidate_context_table.json")
    )
    assert "<mask>" not in json.dumps(persisted)
    assert "EVQLVESGGG" not in json.dumps(persisted)


# ── invalid prompts: mask marker missing / bad format ────────────────────────

def test_inline_prompt_sequence_without_mask_is_not_materialized(
    local_storage, registry_service, workflow_state_service
):
    run_id, _ = _bootstrap(
        local_storage, registry_service, workflow_state_service,
        referenced_inputs=[
            {"id_type": "prompt_sequence", "value": ORDINARY_HEAVY, "source": "user"}
        ],
    )
    table = _run_step5(local_storage, registry_service, workflow_state_service, run_id)
    _, mat = _prompt_material(table)
    assert mat is None, "an unmasked sequence must never become a prompt_sequence"
    gaps = _all_prompt_data_gaps(table)
    assert any("prompt_sequence_mask_missing" in g for g in gaps)
    # It must NOT have been silently downgraded to a plain protein sequence.
    assert not any(
        m.material_type in {"antibody_heavy_chain_sequence", "target_sequence", "prompt_sequence"}
        for c in table.candidate_records
        for m in c.materials
    )


def test_uploaded_prompt_sequence_file_without_mask_is_not_materialized(
    local_storage, registry_service, workflow_state_service
):
    file_id = new_file_id()
    storage_path = local_storage.run_key("uploads_probe2", "inputs", "not_masked.fasta")
    local_storage.write_bytes(storage_path, f">seq\n{ORDINARY_LIGHT}\n".encode("utf-8"))
    run_id, _ = _bootstrap(
        local_storage, registry_service, workflow_state_service,
        referenced_inputs=[
            {"id_type": "uploaded_file", "value": file_id, "source": "prompt_sequence"}
        ],
        uploaded_files=[
            {
                "file_id": file_id,
                "original_filename": "not_masked.fasta",
                "storage_path": storage_path,
            }
        ],
    )
    table = _run_step5(local_storage, registry_service, workflow_state_service, run_id)
    _, mat = _prompt_material(table)
    assert mat is None
    gaps = _all_prompt_data_gaps(table)
    assert any("prompt_sequence_mask_missing" in g for g in gaps)
    # Compact gap references the file_id, never the raw sequence.
    assert any(file_id in g for g in gaps)
    assert ORDINARY_LIGHT not in json.dumps(
        local_storage.read_json(local_storage.run_key(run_id, "candidate_context_table.json"))
    )


def test_inline_prompt_with_mask_but_nonprotein_body_rejected(
    local_storage, registry_service, workflow_state_service
):
    # Contains a mask marker but the residue body is plain text, not amino
    # acids — must not be treated as a protein generation prompt.
    run_id, _ = _bootstrap(
        local_storage, registry_service, workflow_state_service,
        referenced_inputs=[
            {"id_type": "prompt_sequence", "value": "please_fill_here!!123", "source": "user"}
        ],
    )
    table = _run_step5(local_storage, registry_service, workflow_state_service, run_id)
    _, mat = _prompt_material(table)
    assert mat is None
    gaps = _all_prompt_data_gaps(table)
    assert any("prompt_sequence_invalid_format" in g for g in gaps)


# ── ordinary chains never become prompt_sequence ─────────────────────────────

def test_ordinary_heavy_light_sequences_do_not_become_prompt_sequence(
    local_storage, registry_service, workflow_state_service
):
    run_id, _ = _bootstrap(
        local_storage, registry_service, workflow_state_service,
        referenced_inputs=[
            {"id_type": "antibody_heavy_chain_sequence", "value": ORDINARY_HEAVY, "source": "user"},
            {"id_type": "antibody_light_chain_sequence", "value": ORDINARY_LIGHT, "source": "user"},
        ],
        raw_user_query="Assess developability of these antibody heavy/light sequences.",
    )
    table = _run_step5(local_storage, registry_service, workflow_state_service, run_id)
    _, mat = _prompt_material(table)
    assert mat is None
    # The ordinary chains still route to the antibody candidate (unchanged).
    antibody = next(c for c in table.candidate_records if c.candidate_type == "antibody")
    assert {m.material_type for m in antibody.materials} >= {
        "antibody_heavy_chain_sequence",
        "antibody_light_chain_sequence",
    }


def test_no_prompt_sequence_candidate_when_no_prompt_refs(
    local_storage, registry_service, workflow_state_service
):
    run_id, _ = _bootstrap(
        local_storage, registry_service, workflow_state_service,
        referenced_inputs=[
            {"id_type": "antibody_heavy_chain_sequence", "value": ORDINARY_HEAVY, "source": "user"},
        ],
        raw_user_query="Assess developability of this antibody heavy chain.",
    )
    table = _run_step5(local_storage, registry_service, workflow_state_service, run_id)
    assert all(
        m.material_type != "prompt_sequence"
        for c in table.candidate_records
        for m in c.materials
    )
    assert not _all_prompt_data_gaps(table)
