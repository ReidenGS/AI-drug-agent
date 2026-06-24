"""Step 5 richer candidate/material/identifier extraction tests."""

from __future__ import annotations

import json

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
    entity_decompositions=None,
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
                "original_filename": "heavy_chain.fasta",
                "storage_path": "/upload/heavy_chain.fasta",
                "sha256": "sha256:def",
            }
        ],
    )
    table = _run_step5(local_storage, registry_service, workflow_state_service, run_id)
    assert "antibody_heavy_chain_sequence" in _types(table)


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
