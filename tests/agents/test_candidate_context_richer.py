"""Step 5 richer candidate/material/identifier extraction tests."""

from __future__ import annotations

import json

import pytest

from app.agents.candidate_context_agent import CandidateContextAgent
from app.mcp.client import LocalMCPClient
from app.services.input_readiness_service import InputReadinessService
from app.services.intake_service import IntakeService
from app.services.workflow_setup_service import WorkflowSetupService
from app.schemas.step_02_structured_query import (
    MentionedEntities,
    SourceRawRequestRef,
    StructuredQuery,
    TaskIntent,
)
from app.utils.ids import new_artifact_id, new_file_id
from app.utils.time import now_iso


def _bootstrap(
    local_storage, registry_service, workflow_state_service, *,
    target="HER2", candidate="Trastuzumab", payload=None, linker=None,
    referenced_inputs=None, uploaded_files=None, raw_context=None,
    entity_decompositions=None, normalized_entities=None,
):
    intake = IntakeService(local_storage, registry_service, workflow_state_service)
    rec = intake.submit(
        raw_user_query="x",
        user_provided_context=raw_context
        or {"target_or_antigen_text": target, "candidate_text": candidate},
        uploaded_files=uploaded_files,
    )
    reg = registry_service.get(rec.run_id)
    sq = StructuredQuery(
        run_id=rec.run_id,
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
        entity_decompositions=entity_decompositions or [],
        normalized_entities=normalized_entities or [],
    )
    sq_id = new_artifact_id("structured_query")
    local_storage.write_json(
        local_storage.run_key(rec.run_id, "inputs/structured_query.json"),
        {"artifact_id": sq_id, **sq.model_dump()},
    )
    registry_service.update_active(rec.run_id, structured_query_id=sq_id)
    workflow_state_service.mark(rec.run_id, "step_02", "completed")
    InputReadinessService(local_storage, registry_service, workflow_state_service).check(rec.run_id)
    WorkflowSetupService(local_storage, registry_service, workflow_state_service).plan(rec.run_id)
    return rec.run_id


def _run_step5(local_storage, registry_service, workflow_state_service, run_id):
    agent = CandidateContextAgent(
        storage=local_storage,
        registry=registry_service,
        workflow_state=workflow_state_service,
        mcp_client=LocalMCPClient(),
    )
    return agent.run(run_id)


def _types(table):
    return {m.material_type for c in table.candidate_records for m in c.materials}


# ── 1. structured_query-only entities create candidates ─────────────────────

def test_step5_builds_target_antibody_payload_candidates_from_sq(
    local_storage, registry_service, workflow_state_service
):
    run_id = _bootstrap(
        local_storage, registry_service, workflow_state_service,
        payload="MMAE", linker="vc",
        raw_context={
            "target_or_antigen_text": "HER2",
            "candidate_text": "Trastuzumab",
            "payload_linker_text": "vc-MMAE",
        },
    )
    table = _run_step5(local_storage, registry_service, workflow_state_service, run_id)
    cand_types = {c.candidate_type for c in table.candidate_records}
    assert cand_types.issuperset({"target_antigen", "antibody", "compound_component"})
    mat_types = _types(table)
    assert "target_antigen_name" in mat_types
    assert "antibody_name" in mat_types
    assert "payload_name" in mat_types


def test_step5_promotes_target_uniprot_from_normalized_entities(
    local_storage, registry_service, workflow_state_service
):
    run_id = _bootstrap(
        local_storage, registry_service, workflow_state_service,
        target="HER2",
        referenced_inputs=[],
        normalized_entities=[{
            "original_text": "HER2",
            "canonical_name": "ERBB2",
            "canonical_id": "P04626",
            "canonical_id_source": "UniProt",
            "entity_type": "target_or_antigen",
            "explicit_or_inferred": "inferred",
        }],
    )
    table = _run_step5(local_storage, registry_service, workflow_state_service, run_id)
    target = next(c for c in table.candidate_records if c.candidate_type == "target_antigen")
    ids = [i for i in target.identifiers if i.id_type == "uniprot_id"]
    assert len(ids) == 1
    assert ids[0].id_value == "P04626"
    assert ids[0].confidence == 0.8


def test_step5_does_not_promote_non_target_uniprot_normalized_entity(
    local_storage, registry_service, workflow_state_service
):
    run_id = _bootstrap(
        local_storage, registry_service, workflow_state_service,
        target="HER2",
        referenced_inputs=[],
        normalized_entities=[{
            "original_text": "payload protein",
            "canonical_name": "payload protein",
            "canonical_id": "P04626",
            "canonical_id_source": "UniProt",
            "entity_type": "payload",
            "explicit_or_inferred": "inferred",
        }],
    )
    table = _run_step5(local_storage, registry_service, workflow_state_service, run_id)
    target = next(c for c in table.candidate_records if c.candidate_type == "target_antigen")
    assert not any(i.id_type == "uniprot_id" for i in target.identifiers)


def test_step5_invalid_target_uniprot_canonical_id_records_gap_not_identifier(
    local_storage, registry_service, workflow_state_service
):
    run_id = _bootstrap(
        local_storage, registry_service, workflow_state_service,
        target="HER2",
        referenced_inputs=[],
        normalized_entities=[{
            "original_text": "HER2",
            "canonical_name": "ERBB2",
            "canonical_id": "not-a-uniprot",
            "canonical_id_source": "UniProt",
            "entity_type": "target_or_antigen",
            "explicit_or_inferred": "inferred",
        }],
    )
    table = _run_step5(local_storage, registry_service, workflow_state_service, run_id)
    target = next(c for c in table.candidate_records if c.candidate_type == "target_antigen")
    assert not any(i.id_type == "uniprot_id" for i in target.identifiers)
    assert any("target_uniprot_id_not_promoted" in g for g in target.data_gaps)
    assert any("canonical_id was not accession-like" in n for n in target.context_notes)


def test_step5_normalized_uniprot_allows_step6_antigen_feature_lane(
    local_storage, registry_service, workflow_state_service
):
    from app.agents.developability_agent import DevelopabilityAgent

    run_id = _bootstrap(
        local_storage, registry_service, workflow_state_service,
        target="HER2",
        payload=None,
        linker=None,
        referenced_inputs=[],
        normalized_entities=[{
            "original_text": "HER2",
            "canonical_name": "ERBB2",
            "canonical_id": "P04626",
            "canonical_id_source": "UniProt",
            "entity_type": "target_or_antigen",
            "explicit_or_inferred": "inferred",
        }],
    )
    step5_table = _run_step5(local_storage, registry_service, workflow_state_service, run_id)
    target_id = next(
        c.candidate_id
        for c in step5_table.candidate_records
        if c.candidate_type == "target_antigen"
    )
    DevelopabilityAgent(
        storage=local_storage,
        registry=registry_service,
        workflow_state=workflow_state_service,
        mcp_client=LocalMCPClient(bindings={
            "EBIProteins_get_features": lambda **kw: {"features": []},
        }),
    ).run(run_id)
    persisted = local_storage.read_json(
        local_storage.run_key(run_id, "structured_liability_summary.json")
    )
    target_result = next(
        cand for cand in persisted["candidate_liability_results"]
        if cand["candidate_id"] == target_id
    )
    lanes = {lane["lane_type"]: lane for lane in target_result["lane_results"]}
    antigen = lanes["antigen_protein_feature_context"]
    assert antigen["run_status"] in {"ok", "partial"}
    assert antigen["input_status"] == "sufficient"
    assert any(
        tc["tool_name"] == "EBIProteins_get_features"
        for tc in antigen["tool_call_records"]
    )


# ── 2. raw context fallback ──────────────────────────────────────────────────

def test_step5_falls_back_to_raw_context_when_sq_entities_empty(
    local_storage, registry_service, workflow_state_service
):
    """structured_query has no mentioned_entities populated, but raw context
    does — Step 5 must still produce candidates."""
    run_id = _bootstrap(
        local_storage, registry_service, workflow_state_service,
        target=None, candidate=None,  # SQ entities empty
        raw_context={
            "target_or_antigen_text": "HER2",
            "candidate_text": "Trastuzumab",
            "payload_linker_text": "vc-MMAE",
        },
    )
    # but the bootstrap _will_ have failed readiness without target. Make
    # readiness pass by re-seeding SQ with the target in mentioned_entities.
    # We re-bootstrap with target in SQ to keep readiness sane.
    run_id = _bootstrap(
        local_storage, registry_service, workflow_state_service,
        target="HER2", candidate=None, payload=None, linker=None,
        raw_context={
            "target_or_antigen_text": "HER2",
            "candidate_text": "Trastuzumab",
            "payload_linker_text": "vc-MMAE",
        },
    )
    table = _run_step5(local_storage, registry_service, workflow_state_service, run_id)
    labels = {c.candidate_label for c in table.candidate_records}
    assert "Trastuzumab" in labels
    assert "vc-MMAE" in labels  # payload from raw context


# ── 3. uploaded PDB file → structure material/ref ───────────────────────────

def test_step5_uploaded_pdb_creates_structure_material(
    local_storage, registry_service, workflow_state_service
):
    run_id = _bootstrap(
        local_storage, registry_service, workflow_state_service,
        payload="MMAE", linker="vc",
        raw_context={
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
    table = _run_step5(local_storage, registry_service, workflow_state_service, run_id)
    assert "structure_file" in _types(table)


# ── 4. uploaded FASTA → antibody sequence material ──────────────────────────

def test_step5_uploaded_fasta_creates_sequence_material(
    local_storage, registry_service, workflow_state_service
):
    file_id = new_file_id()
    run_id = _bootstrap(
        local_storage, registry_service, workflow_state_service,
        payload="MMAE", linker="vc",
        raw_context={
            "target_or_antigen_text": "HER2",
            "candidate_text": "Trastuzumab",
            "payload_linker_text": "vc-MMAE",
        },
        referenced_inputs=[
            {
                "id_type": "uploaded_file",
                "value": file_id,
                "source": "antibody_heavy_chain_sequence",
            }
        ],
        uploaded_files=[
            {
                "file_id": file_id,
                "original_filename": "heavy_chain.fasta",
                "storage_path": "/upload/heavy_chain.fasta",
                "sha256": "sha256:def",
            }
        ],
    )
    table = _run_step5(local_storage, registry_service, workflow_state_service, run_id)
    assert "antibody_heavy_chain_sequence" in _types(table)


@pytest.mark.parametrize(
    ("source", "expected_material_type"),
    [
        ("target_sequence", "target_sequence"),
        ("antibody_heavy_chain_sequence", "antibody_heavy_chain_sequence"),
        ("antibody_light_chain_sequence", "antibody_light_chain_sequence"),
        ("prompt_sequence", "prompt_sequence"),
    ],
)
def test_step5_uploaded_fasta_role_comes_only_from_step2_source(
    local_storage,
    registry_service,
    workflow_state_service,
    source,
    expected_material_type,
):
    file_id = new_file_id()
    storage_key = f"test_inputs/{file_id}.fasta"
    local_storage.write_bytes(
        storage_key,
        (">test\nACD_EF" if source == "prompt_sequence" else ">test\nACDEFG").encode(),
    )
    run_id = _bootstrap(
        local_storage,
        registry_service,
        workflow_state_service,
        referenced_inputs=[
            {"id_type": "uploaded_file", "value": file_id, "source": source}
        ],
        uploaded_files=[
            {
                "file_id": file_id,
                "original_filename": "sequence.fasta",
                "storage_path": storage_key,
                "content_type": "text/x-fasta",
                "sha256": "sha256:authority",
            }
        ],
    )

    table = _run_step5(
        local_storage, registry_service, workflow_state_service, run_id
    )
    matching = [
        material
        for candidate in table.candidate_records
        for material in candidate.materials
        if material.material_type == expected_material_type
    ]
    assert len(matching) == 1
    assert not any(
        material.material_type
        in {
            "target_sequence",
            "antibody_heavy_chain_sequence",
            "antibody_light_chain_sequence",
            "prompt_sequence",
        }
        and material.material_type != expected_material_type
        for candidate in table.candidate_records
        for material in candidate.materials
    )


def test_step5_does_not_infer_unassigned_fasta_role_from_filename_or_query(
    local_storage,
    registry_service,
    workflow_state_service,
):
    file_id = new_file_id()
    run_id = _bootstrap(
        local_storage,
        registry_service,
        workflow_state_service,
        referenced_inputs=[
            {
                "id_type": "uploaded_file",
                "value": file_id,
                "source": "uploaded_file",
            }
        ],
        uploaded_files=[
            {
                "file_id": file_id,
                "original_filename": "target_heavy_light_sequence.fasta",
                "storage_path": "/upload/target_heavy_light_sequence.fasta",
                "content_type": "text/x-fasta",
            }
        ],
        raw_context={
            "target_or_antigen_text": "HER2 target",
            "candidate_text": "trastuzumab heavy light antibody",
        },
    )
    table = _run_step5(
        local_storage, registry_service, workflow_state_service, run_id
    )

    role_materials = [
        material.material_type
        for candidate in table.candidate_records
        for material in candidate.materials
        if material.material_type
        in {
            "target_sequence",
            "antibody_heavy_chain_sequence",
            "antibody_light_chain_sequence",
            "prompt_sequence",
        }
    ]
    assert role_materials == []
    assert any(
        "uploaded_fasta_sequence_file_unassigned" in gap
        for candidate in table.candidate_records
        for gap in candidate.data_gaps
    )


# ── 5. provided SMILES becomes a compound_smiles material ───────────────────

def test_step5_smiles_referenced_input_becomes_payload_smiles(
    local_storage, registry_service, workflow_state_service
):
    run_id = _bootstrap(
        local_storage, registry_service, workflow_state_service,
        payload="MMAE", linker="vc",
        raw_context={
            "target_or_antigen_text": "HER2",
            "candidate_text": "Trastuzumab",
            "payload_linker_text": "vc-MMAE",
        },
        referenced_inputs=[
            {"id_type": "smiles", "value": "CC(=O)NCCC1=CN(c2ccc(O)cc2)C(=O)C1", "source": "raw_request_text"},
        ],
    )
    table = _run_step5(local_storage, registry_service, workflow_state_service, run_id)
    # Either payload_smiles attached to payload candidate, or compound_smiles
    # on the fallback compound candidate.
    mt = _types(table)
    assert "payload_smiles" in mt or "compound_smiles" in mt


def test_step5_extracts_explicit_payload_and_linker_smiles_from_raw_context(
    local_storage, registry_service, workflow_state_service
):
    run_id = _bootstrap(
        local_storage, registry_service, workflow_state_service,
        payload="CCO", linker="vc-MMAE",
        raw_context={
            "target_or_antigen_text": "HER2",
            "candidate_text": "Trastuzumab",
            "payload_linker_text": "vc-MMAE (payload SMILES CCO; linker SMILES NCC(=O)O)",
        },
    )
    table = _run_step5(local_storage, registry_service, workflow_state_service, run_id)
    assert any(
        m.material_type == "payload_smiles" and m.value == "CCO"
        for c in table.candidate_records for m in c.materials
    )
    assert any(
        m.material_type == "linker_smiles" and m.value == "NCC(=O)O"
        for c in table.candidate_records for m in c.materials
    )


def test_step5_routes_name_materials_to_chembl_molecules_not_substructure(
    local_storage, registry_service, workflow_state_service
):
    run_id = _bootstrap(
        local_storage, registry_service, workflow_state_service,
        payload="MMAE",
        linker="valine-citrulline",
        raw_context={
            "target_or_antigen_text": "HER2",
            "candidate_text": "Trastuzumab",
            "payload_linker_text": "MMAE with valine-citrulline linker",
        },
    )
    calls: list[tuple[str, dict]] = []

    def _record(tool_name: str, payload: dict):
        def _fn(**kwargs):
            calls.append((tool_name, kwargs))
            return payload
        return _fn

    agent = CandidateContextAgent(
        storage=local_storage,
        registry=registry_service,
        workflow_state=workflow_state_service,
        mcp_client=LocalMCPClient(bindings={
            "SAbDab_search_structures": _record("SAbDab_search_structures", {"hits": []}),
            "ChEMBL_search_molecules": _record("ChEMBL_search_molecules", {"molecules": []}),
            "ChEMBL_search_substructure": _record("ChEMBL_search_substructure", {"molecules": []}),
        }),
    )
    table = agent.run(run_id)

    chembl_calls = [c for c in calls if c[0].startswith("ChEMBL")]
    assert ("ChEMBL_search_molecules", {"query": "MMAE"}) in chembl_calls
    assert ("ChEMBL_search_molecules", {"query": "valine-citrulline"}) in chembl_calls
    assert not any(name == "ChEMBL_search_substructure" for name, _ in chembl_calls)
    persisted = local_storage.read_json(local_storage.run_key(run_id, "candidate_context_table.json"))
    summaries = [
        tc["tool_input_summary"] for tc in persisted["tool_call_records"]
        if tc["tool_name"].startswith("ChEMBL")
    ]
    assert {s["query_kind"] for s in summaries} == {"name"}
    assert {s["query"] for s in summaries}.issuperset({"MMAE", "valine-citrulline"})
    assert table.context_build_status in {"ok", "partial"}


def test_step5_routes_smiles_materials_to_chembl_substructure(
    local_storage, registry_service, workflow_state_service
):
    run_id = _bootstrap(
        local_storage, registry_service, workflow_state_service,
        payload="MMAE",
        raw_context={
            "target_or_antigen_text": "HER2",
            "candidate_text": "Trastuzumab",
            "payload_linker_text": "MMAE payload SMILES CCO",
        },
    )
    calls: list[tuple[str, dict]] = []

    def _record(tool_name: str, payload: dict):
        def _fn(**kwargs):
            calls.append((tool_name, kwargs))
            return payload
        return _fn

    agent = CandidateContextAgent(
        storage=local_storage,
        registry=registry_service,
        workflow_state=workflow_state_service,
        mcp_client=LocalMCPClient(bindings={
            "SAbDab_search_structures": _record("SAbDab_search_structures", {"hits": []}),
            "ChEMBL_search_molecules": _record("ChEMBL_search_molecules", {"molecules": []}),
            "ChEMBL_search_substructure": _record("ChEMBL_search_substructure", {"molecules": []}),
        }),
    )
    agent.run(run_id)

    assert ("ChEMBL_search_molecules", {"query": "MMAE"}) in calls
    assert ("ChEMBL_search_substructure", {"smiles": "CCO"}) in calls
    persisted = local_storage.read_json(local_storage.run_key(run_id, "candidate_context_table.json"))
    sub = [
        tc for tc in persisted["tool_call_records"]
        if tc["tool_name"] == "ChEMBL_search_substructure"
    ]
    assert sub
    assert all(tc["tool_input_summary"]["query_kind"] == "smiles" for tc in sub)
    assert all(tc["tool_input_summary"]["query"] == "CCO" for tc in sub)


def test_step5_skips_mixed_payload_linker_text_as_chembl_query(
    local_storage, registry_service, workflow_state_service
):
    mixed = "vc-MMAE (payload SMILES CCO; linker SMILES NCC(=O)O)"
    run_id = _bootstrap(
        local_storage, registry_service, workflow_state_service,
        payload=None,
        linker=None,
        raw_context={
            "target_or_antigen_text": "HER2",
            "candidate_text": "Trastuzumab",
            "payload_linker_text": mixed,
        },
    )
    calls: list[tuple[str, dict]] = []

    def _record(tool_name: str, payload: dict):
        def _fn(**kwargs):
            calls.append((tool_name, kwargs))
            return payload
        return _fn

    agent = CandidateContextAgent(
        storage=local_storage,
        registry=registry_service,
        workflow_state=workflow_state_service,
        mcp_client=LocalMCPClient(bindings={
            "SAbDab_search_structures": _record("SAbDab_search_structures", {"hits": []}),
            "ChEMBL_search_molecules": _record("ChEMBL_search_molecules", {"molecules": []}),
            "ChEMBL_search_substructure": _record("ChEMBL_search_substructure", {"molecules": []}),
        }),
    )
    agent.run(run_id)

    chembl_queries = [
        kwargs.get("query") or kwargs.get("smiles")
        for name, kwargs in calls
        if name.startswith("ChEMBL")
    ]
    assert mixed not in chembl_queries
    assert "CCO" in chembl_queries
    assert "NCC(=O)O" in chembl_queries


def test_step5_searches_clean_decomposition_component_names_separately(
    local_storage, registry_service, workflow_state_service
):
    run_id = _bootstrap(
        local_storage, registry_service, workflow_state_service,
        payload="vc-MMAE",
        raw_context={
            "target_or_antigen_text": "HER2",
            "candidate_text": "Trastuzumab",
            "payload_linker_text": "vc-MMAE",
        },
        entity_decompositions=[{
            "original_text": "vc-MMAE",
            "canonical_name": "vc-MMAE",
            "components": [
                {"role": "linker", "canonical_name": "valine-citrulline", "inferred": True},
                {"role": "payload", "canonical_name": "monomethyl auristatin E", "inferred": False},
            ],
        }],
    )
    calls: list[tuple[str, dict]] = []

    def _record(tool_name: str, payload: dict):
        def _fn(**kwargs):
            calls.append((tool_name, kwargs))
            return payload
        return _fn

    CandidateContextAgent(
        storage=local_storage,
        registry=registry_service,
        workflow_state=workflow_state_service,
        mcp_client=LocalMCPClient(bindings={
            "SAbDab_search_structures": _record("SAbDab_search_structures", {"hits": []}),
            "ChEMBL_search_molecules": _record("ChEMBL_search_molecules", {"molecules": []}),
            "ChEMBL_search_substructure": _record("ChEMBL_search_substructure", {"molecules": []}),
        }),
    ).run(run_id)

    molecule_queries = [
        kwargs.get("query") for name, kwargs in calls
        if name == "ChEMBL_search_molecules"
    ]
    assert "vc-MMAE" in molecule_queries
    assert "valine-citrulline" in molecule_queries
    assert "monomethyl auristatin E" in molecule_queries
    assert not any(name == "ChEMBL_search_substructure" for name, _ in calls)


def test_step5_skips_low_information_vc_linker_name_query_and_records_gap(
    local_storage, registry_service, workflow_state_service
):
    run_id = _bootstrap(
        local_storage, registry_service, workflow_state_service,
        payload=None,
        linker="vc",
        raw_context={
            "target_or_antigen_text": "HER2",
            "candidate_text": "Trastuzumab",
            "payload_linker_text": "vc linker",
        },
    )
    calls: list[tuple[str, dict]] = []

    def _record(tool_name: str, payload: dict):
        def _fn(**kwargs):
            calls.append((tool_name, kwargs))
            return payload
        return _fn

    table = CandidateContextAgent(
        storage=local_storage,
        registry=registry_service,
        workflow_state=workflow_state_service,
        mcp_client=LocalMCPClient(bindings={
            "SAbDab_search_structures": _record("SAbDab_search_structures", {"hits": []}),
            "ChEMBL_search_molecules": _record("ChEMBL_search_molecules", {"molecules": [
                {"molecule_chembl_id": "CHEMBL_SHOULD_NOT_PROMOTE"}
            ]}),
        }),
    ).run(run_id)

    assert ("ChEMBL_search_molecules", {"query": "vc"}) not in calls
    linker_records = [
        c for c in table.candidate_records
        if c.candidate_type == "compound_component"
        and any(m.material_type == "linker_name" and m.value == "vc" for m in c.materials)
    ]
    assert linker_records
    rec = linker_records[0]
    assert not rec.identifiers
    assert any(
        gap == "ChEMBL_search_molecules(name=vc): skipped_low_information_alias"
        for gap in rec.data_gaps
    )
    assert any("low-information alias 'vc'" in note for note in rec.context_notes)


# ── 6. ZINC id becomes identifier and is NOT labeled ZINC22 ─────────────────

def test_step5_zinc_id_does_not_default_to_zinc22(
    local_storage, registry_service, workflow_state_service
):
    run_id = _bootstrap(
        local_storage, registry_service, workflow_state_service,
        payload="MMAE", linker="vc",
        raw_context={
            "target_or_antigen_text": "HER2",
            "candidate_text": "Trastuzumab",
            "payload_linker_text": "vc-MMAE",
        },
        referenced_inputs=[
            {"id_type": "zinc_id", "value": "ZINC12345678", "source": "raw_request_text"},
        ],
    )
    table = _run_step5(local_storage, registry_service, workflow_state_service, run_id)
    zinc_ids = [
        ident for c in table.candidate_records for ident in c.identifiers
        if ident.id_type == "zinc_id"
    ]
    assert zinc_ids and zinc_ids[0].id_value == "ZINC12345678"
    # No record should claim ZINC22.
    import json
    blob = json.dumps(table.model_dump(), default=str)
    assert "ZINC22" not in blob


# ── 7. raw enrichment payload never leaks into candidate_records ────────────

def test_step5_raw_enrichment_does_not_leak_into_normalized_records(
    local_storage, registry_service, workflow_state_service
):
    run_id = _bootstrap(
        local_storage, registry_service, workflow_state_service,
        payload="MMAE", linker="vc",
        raw_context={
            "target_or_antigen_text": "HER2",
            "candidate_text": "Trastuzumab",
            "payload_linker_text": "vc-MMAE",
        },
    )
    agent = CandidateContextAgent(
        storage=local_storage,
        registry=registry_service,
        workflow_state=workflow_state_service,
        mcp_client=LocalMCPClient(bindings={
            "SAbDab_search_structures": lambda **kw: {"hits": [{"pdb_id": "1n8z"}]},
            "ChEMBL_search_molecules": lambda **kw: {"hits": [{"chembl_id": "CHEMBL1"}]},
        }),
    )
    agent.run(run_id)
    import json
    persisted = local_storage.read_json(
        local_storage.run_key(run_id, "candidate_context_table.json")
    )
    assert "hits" not in json.dumps(persisted["candidate_records"])
    for tc in persisted["tool_call_records"]:
        if tc.get("run_status") == "success":
            assert tc["tool_output_ref"]


def test_step5_promotes_chembl_smiles_and_id_for_payload_candidate(
    local_storage, registry_service, workflow_state_service
):
    run_id = _bootstrap(
        local_storage, registry_service, workflow_state_service,
        payload="MMAE", linker=None,
        raw_context={
            "target_or_antigen_text": "HER2",
            "candidate_text": "Trastuzumab",
            "payload_linker_text": "MMAE",
        },
    )
    agent = CandidateContextAgent(
        storage=local_storage,
        registry=registry_service,
        workflow_state=workflow_state_service,
        mcp_client=LocalMCPClient(bindings={
            "SAbDab_search_structures": lambda **kw: {"hits": [{"pdb_id": "1n8z"}]},
            "ChEMBL_search_molecules": lambda **kw: {
                "hits": [{
                    "molecule_chembl_id": "CHEMBL1201585",
                    "pref_name": "MONOMETHYL AURISTATIN E",
                    "molecule_structures": {"canonical_smiles": "CCO"},
                    "raw_match": "SECRET_RAW_FIELD_DO_NOT_LEAK",
                }]
            },
        }),
    )
    table = agent.run(run_id)

    payload_records = [
        c for c in table.candidate_records
        if c.candidate_type == "compound_component"
        and any(m.material_type == "payload_name" for m in c.materials)
    ]
    assert payload_records
    rec = payload_records[0]
    assert any(i.id_type == "chembl_id" and i.id_value == "CHEMBL1201585" for i in rec.identifiers)
    assert any(m.material_type == "payload_smiles" and m.value == "CCO" for m in rec.materials)

    persisted = local_storage.read_json(
        local_storage.run_key(run_id, "candidate_context_table.json")
    )
    import json
    normalized_blob = json.dumps(persisted["candidate_records"])
    assert "hits" not in normalized_blob
    assert "raw_match" not in normalized_blob
    assert "SECRET_RAW_FIELD_DO_NOT_LEAK" not in normalized_blob


def test_step5_chembl_id_material_plans_get_molecule_and_promotes_smiles(
    local_storage, registry_service, workflow_state_service
):
    run_id = _bootstrap(
        local_storage, registry_service, workflow_state_service,
        payload=None,
        linker=None,
        referenced_inputs=[
            {"id_type": "chembl_id", "value": "CHEMBL1201585", "source": "raw_request_text"}
        ],
        raw_context={
            "target_or_antigen_text": "HER2",
            "candidate_text": "Trastuzumab",
        },
    )
    calls: list[tuple[str, dict]] = []

    def _record(tool_name: str, payload: dict):
        def _fn(**kwargs):
            calls.append((tool_name, kwargs))
            return payload
        return _fn

    table = CandidateContextAgent(
        storage=local_storage,
        registry=registry_service,
        workflow_state=workflow_state_service,
        mcp_client=LocalMCPClient(bindings={
            "SAbDab_search_structures": _record("SAbDab_search_structures", {"hits": []}),
            "ChEMBL_get_molecule": _record("ChEMBL_get_molecule", {
                "molecule_chembl_id": "CHEMBL1201585",
                "molecule_structures": {"canonical_smiles": "CCO"},
                "pref_name": "ASPIRIN",
                "raw_match": "SECRET_RAW_FIELD_DO_NOT_LEAK",
            }),
            "ChEMBL_search_molecules": _record("ChEMBL_search_molecules", {"molecules": []}),
        }),
    ).run(run_id)

    assert ("ChEMBL_get_molecule", {"chembl_id": "CHEMBL1201585"}) in calls
    compounds = [c for c in table.candidate_records if c.candidate_type == "compound_component"]
    assert compounds
    rec = compounds[0]
    assert any(i.id_type == "chembl_id" and i.id_value == "CHEMBL1201585" for i in rec.identifiers)
    assert any(m.material_type == "compound_smiles" and m.value == "CCO" for m in rec.materials)

    persisted = local_storage.read_json(local_storage.run_key(run_id, "candidate_context_table.json"))
    normalized_blob = json.dumps(persisted["candidate_records"])
    assert "raw_match" not in normalized_blob
    assert "SECRET_RAW_FIELD_DO_NOT_LEAK" not in normalized_blob


def test_step5_promotes_composite_linker_payload_smiles_as_compound_smiles(
    local_storage, registry_service, workflow_state_service
):
    run_id = _bootstrap(
        local_storage, registry_service, workflow_state_service,
        payload="vc-MMAE", linker=None,
        raw_context={
            "target_or_antigen_text": "HER2",
            "candidate_text": "Trastuzumab",
            "payload_linker_text": "vc-MMAE",
        },
        entity_decompositions=[{
            "original_text": "vc-MMAE",
            "canonical_name": "vc-MMAE",
            "components": [
                {"role": "linker", "canonical_name": "vc", "inferred": False},
                {"role": "payload", "canonical_name": "MMAE", "inferred": False},
            ],
        }],
    )
    agent = CandidateContextAgent(
        storage=local_storage,
        registry=registry_service,
        workflow_state=workflow_state_service,
        mcp_client=LocalMCPClient(bindings={
            "SAbDab_search_structures": lambda **kw: {"hits": [{"pdb_id": "1n8z"}]},
            "ChEMBL_search_molecules": lambda **kw: {
                "results": [{
                    "chembl_id": "CHEMBL_COMPOSITE",
                    "canonical_smiles": "NCCO",
                    "name": "vc-MMAE",
                }]
            },
        }),
    )
    table = agent.run(run_id)

    compound_records = [
        c for c in table.candidate_records
        if c.candidate_type == "compound_component"
    ]
    assert compound_records
    assert any(
        m.material_type == "compound_smiles" and m.value == "NCCO"
        for c in compound_records for m in c.materials
    )


# ── per-candidate material + identifier dedup ───────────────────────────


def test_step5_dedupes_target_uniprot_identifier_from_multipath_normalization(
    local_storage, registry_service, workflow_state_service,
):
    """HER2 and ERBB2 both resolve to UniProt P04626 via normalized
    entities, AND P04626 also appears as an explicit referenced_input.
    The final target candidate must carry exactly ONE P04626 identifier
    with merged provenance from every contributing path."""
    run_id = _bootstrap(
        local_storage, registry_service, workflow_state_service,
        target="HER2",
        referenced_inputs=[
            {"id_type": "uniprot_id", "value": "P04626",
             "source": "raw_request_text"},
        ],
        normalized_entities=[
            {"original_text": "HER2", "canonical_name": "ERBB2",
             "canonical_id": "P04626", "canonical_id_source": "UniProt",
             "entity_type": "target_or_antigen",
             "explicit_or_inferred": "explicit"},
            {"original_text": "ERBB2", "canonical_name": "ERBB2",
             "canonical_id": "P04626", "canonical_id_source": "UniProt",
             "entity_type": "target_or_antigen",
             "explicit_or_inferred": "explicit"},
        ],
    )
    table = _run_step5(
        local_storage, registry_service, workflow_state_service, run_id,
    )
    target = next(
        c for c in table.candidate_records if c.candidate_type == "target_antigen"
    )
    uniprot_ids = [i for i in target.identifiers if i.id_type == "uniprot_id"]
    assert [i.id_value for i in uniprot_ids] == ["P04626"]
    # Provenance is preserved: source_ids contains at least the
    # structured_query artifact id that fed each path.
    assert uniprot_ids[0].source_ids, uniprot_ids[0].source_ids


def test_step5_does_not_merge_distinct_uniprot_identifiers(
    local_storage, registry_service, workflow_state_service,
):
    run_id = _bootstrap(
        local_storage, registry_service, workflow_state_service,
        target="HER2",
        referenced_inputs=[
            {"id_type": "uniprot_id", "value": "P04626",
             "source": "raw_request_text"},
            {"id_type": "uniprot_id", "value": "Q9NQB7",
             "source": "raw_request_text"},
        ],
    )
    table = _run_step5(
        local_storage, registry_service, workflow_state_service, run_id,
    )
    target = next(
        c for c in table.candidate_records if c.candidate_type == "target_antigen"
    )
    ids = sorted(i.id_value for i in target.identifiers if i.id_type == "uniprot_id")
    assert ids == ["P04626", "Q9NQB7"]


def test_step5_dedupes_identical_target_sequence_material_per_candidate(
    local_storage, registry_service, workflow_state_service,
):
    """Two paths that converge on the same FASTA reference for one
    target produce a single ``target_sequence`` material."""
    fasta = {
        "file_id": new_file_id(),
        "original_filename": "her2_extracellular.fasta",
        "storage_path": "/runs/abc/inputs/files/her2_extracellular.fasta",
        "content_type": "text/x-fasta",
        "size_bytes": 12,
    }
    run_id = _bootstrap(
        local_storage, registry_service, workflow_state_service,
        target="HER2",
        # The same uploaded FASTA listed twice — simulates the
        # multipath case where intake / Step 2 contributed the same
        # reference under two different drift paths.
        uploaded_files=[fasta, dict(fasta)],
        referenced_inputs=[
            {
                "id_type": "uploaded_file",
                "value": fasta["file_id"],
                "source": "target_sequence",
            }
        ],
    )
    table = _run_step5(
        local_storage, registry_service, workflow_state_service, run_id,
    )
    target = next(
        c for c in table.candidate_records if c.candidate_type == "target_antigen"
    )
    seq_materials = [m for m in target.materials
                     if m.material_type == "target_sequence"]
    assert len(seq_materials) == 1, [m.value for m in seq_materials]


def test_step5_keeps_distinct_target_sequence_materials_at_different_paths(
    local_storage, registry_service, workflow_state_service,
):
    """If the user uploads two genuinely different FASTAs that both
    fall into the target-sequence channel, dedup does NOT merge them —
    different ``value`` means different references."""
    f1 = {
        "file_id": new_file_id(),
        "original_filename": "her2_a.fasta",
        "storage_path": "/runs/abc/inputs/files/her2_a.fasta",
        "content_type": "text/x-fasta", "size_bytes": 12,
    }
    f2 = {
        "file_id": new_file_id(),
        "original_filename": "her2_b.fasta",
        "storage_path": "/runs/abc/inputs/files/her2_b.fasta",
        "content_type": "text/x-fasta", "size_bytes": 12,
    }
    run_id = _bootstrap(
        local_storage, registry_service, workflow_state_service,
        target="HER2",
        uploaded_files=[f1, f2],
        referenced_inputs=[
            {
                "id_type": "uploaded_file",
                "value": item["file_id"],
                "source": "target_sequence",
            }
            for item in (f1, f2)
        ],
    )
    table = _run_step5(
        local_storage, registry_service, workflow_state_service, run_id,
    )
    target = next(
        c for c in table.candidate_records if c.candidate_type == "target_antigen"
    )
    seq_materials = [m for m in target.materials
                     if m.material_type == "target_sequence"]
    assert len(seq_materials) == 2
    assert {m.value for m in seq_materials} == {
        f1["storage_path"], f2["storage_path"],
    }


def test_step5_does_not_merge_structure_files_with_different_paths(
    local_storage, registry_service, workflow_state_service,
):
    f1 = {
        "file_id": new_file_id(),
        "original_filename": "antigen.pdb",
        "storage_path": "/runs/abc/inputs/files/antigen.pdb",
        "content_type": "chemical/x-pdb", "size_bytes": 12,
    }
    f2 = {
        "file_id": new_file_id(),
        "original_filename": "antigen.pdb",  # same name, different path
        "storage_path": "/runs/xyz/inputs/files/antigen.pdb",
        "content_type": "chemical/x-pdb", "size_bytes": 12,
    }
    run_id = _bootstrap(
        local_storage, registry_service, workflow_state_service,
        target="HER2", uploaded_files=[f1, f2],
    )
    table = _run_step5(
        local_storage, registry_service, workflow_state_service, run_id,
    )
    target = next(
        c for c in table.candidate_records if c.candidate_type == "target_antigen"
    )
    structure_mats = [m for m in target.materials
                      if m.material_type == "structure_file"]
    assert len(structure_mats) == 2, [m.value for m in structure_mats]
    assert {m.value for m in structure_mats} == {
        f1["storage_path"], f2["storage_path"],
    }


def test_step5_does_not_merge_payload_and_linker_name_with_same_value(
    local_storage, registry_service, workflow_state_service,
):
    """payload_name and linker_name with the same string value answer
    different downstream questions; dedup must keep them separate."""
    run_id = _bootstrap(
        local_storage, registry_service, workflow_state_service,
        target="HER2",
        payload="duplicate-name",
        linker="duplicate-name",
    )
    table = _run_step5(
        local_storage, registry_service, workflow_state_service, run_id,
    )
    name_types = sorted({
        m.material_type
        for c in table.candidate_records
        for m in c.materials
        if m.material_type in {"payload_name", "linker_name"}
    })
    assert name_types == ["linker_name", "payload_name"]


def test_step5_does_not_merge_heavy_and_light_chain_sequence_materials(
    local_storage, registry_service, workflow_state_service,
):
    """Heavy and light chain materials may carry an identical value in
    pathological drift cases (e.g. a normalizer collapsing both paths
    onto a single redacted marker). They must stay distinct because
    Step 5's CDR3 path branches on the chain ``material_type``."""
    from app.agents.candidate_context_agent import _dedupe_candidate_materials
    from app.schemas.step_05_candidate_context_table import Material
    heavy = Material(
        material_id="m_heavy",
        material_type="antibody_heavy_chain_sequence",
        value="/runs/abc/heavy.fasta",
        value_format="fasta",
        role="antibody_sequence_reference",
        role_status="explicit",
    )
    light = Material(
        material_id="m_light",
        material_type="antibody_light_chain_sequence",
        value="/runs/abc/heavy.fasta",  # same value, different role/type
        value_format="fasta",
        role="antibody_sequence_reference",
        role_status="explicit",
    )
    deduped = _dedupe_candidate_materials([heavy, light])
    assert [m.material_type for m in deduped] == [
        "antibody_heavy_chain_sequence",
        "antibody_light_chain_sequence",
    ]


def test_step5_dedupes_exact_duplicate_material_for_one_candidate():
    """The unit-level path: a fully equivalent material on the same
    candidate collapses to one. All distinguishing axes must match."""
    from app.agents.candidate_context_agent import _dedupe_candidate_materials
    from app.schemas.step_05_candidate_context_table import Material
    a = Material(
        material_id="m_a",
        material_type="target_sequence",
        value="/runs/x/inputs/files/her2.fasta",
        value_format="fasta",
        role="target_sequence_reference",
        role_status="explicit",
    )
    b = Material(
        material_id="m_b",  # different artifact id, otherwise identical
        material_type="target_sequence",
        value="/runs/x/inputs/files/her2.fasta",
        value_format="fasta",
        role="target_sequence_reference",
        role_status="explicit",
    )
    deduped = _dedupe_candidate_materials([a, b])
    assert len(deduped) == 1
    # First occurrence wins; provenance is the surviving artifact.
    assert deduped[0].material_id == "m_a"


def test_step5_identifier_dedup_merges_source_ids_and_keeps_higher_confidence():
    from app.agents.candidate_context_agent import _dedupe_candidate_identifiers
    from app.schemas.step_05_candidate_context_table import Identifier
    a = Identifier(
        id_type="uniprot_id", id_value="P04626",
        source_ids=["sq_artifact_1"], confidence=0.5,
    )
    b = Identifier(
        id_type="uniprot_id", id_value="P04626",
        source_ids=["sq_artifact_2"], confidence=0.9,
    )
    c = Identifier(
        id_type="uniprot_id", id_value="P04626",
        source_ids=["sq_artifact_1"], confidence=0.2,  # duplicate source
    )
    deduped = _dedupe_candidate_identifiers([a, b, c])
    assert len(deduped) == 1
    survivor = deduped[0]
    assert survivor.id_value == "P04626"
    assert survivor.source_ids == ["sq_artifact_1", "sq_artifact_2"]
    assert survivor.confidence == 0.9


def test_step5_raw_content_isolation_holds_after_dedup_pass(
    local_storage, registry_service, workflow_state_service,
):
    """The dedup pass must not loosen the raw-payload isolation
    guarantee on the persisted normalized artifact."""
    fasta = {
        "file_id": new_file_id(),
        "original_filename": "antigen.fasta",
        "storage_path": "/runs/abc/inputs/files/antigen.fasta",
        "content_type": "text/x-fasta", "size_bytes": 12,
    }
    run_id = _bootstrap(
        local_storage, registry_service, workflow_state_service,
        target="HER2", uploaded_files=[fasta, dict(fasta)],
    )
    table = _run_step5(
        local_storage, registry_service, workflow_state_service, run_id,
    )
    blob = json.dumps(table.model_dump(), default=str)
    # No FASTA header, no PDB ATOM/HETATM line, no raw CDR3.
    assert ">antigen" not in blob
    assert "ATOM " not in blob
    assert "HETATM" not in blob
    # And no plausible raw heavy-chain stretch.
    assert "EVQLVQSGAEVKKPGSSVKVSCKAS" not in blob
