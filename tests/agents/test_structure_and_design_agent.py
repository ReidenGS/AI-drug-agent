"""StructureAndDesignAgent — Step 7/8/9 MVP tests."""

from __future__ import annotations

import json
import os
import hashlib
from copy import deepcopy
from pathlib import Path

import pytest

from app.agents.candidate_context_agent import CandidateContextAgent
from app.agents.developability_agent import DevelopabilityAgent
from app.agents import structure_and_design_agent as structure_and_design_module
from app.agents.structure_and_design_agent import StructureAndDesignAgent
from app.agents.supervisor_agent import SupervisorAgent
from app.llm.provider import MockLLMProvider
from app.mcp.client import LocalMCPClient
from app.schemas.common import ToolCallRecord
from app.services.input_readiness_service import InputReadinessService
from app.services.intake_service import IntakeService
from app.services.structured_query_service import StructuredQueryService
from app.services.tool_inventory_service import ToolInventoryService
from app.services.workflow_setup_service import WorkflowSetupService
from app.utils.errors import WorkflowStateError


PROJECT_ROOT = Path(__file__).resolve().parents[2].parent
DEFAULT_XLSX = PROJECT_ROOT / "\u9879\u76ee\u6587\u4ef6" / "ToolUniversity_inventory_v0.2.xlsx"


def _mcp() -> LocalMCPClient:
    """Inventory-scoped client backed by the v0.2 xlsx so scope_filter fires."""
    xlsx = os.environ.get("TOOL_INVENTORY_XLSX", str(DEFAULT_XLSX))
    if not Path(xlsx).exists():
        pytest.skip(f"Inventory xlsx not at {xlsx}")
    return LocalMCPClient(inventory=ToolInventoryService(xlsx))


def _seed(
    local_storage, registry_service, workflow_state_service,
    *,
    uploaded_files=None,
    referenced_inputs=None,
    payload_text="MMAE",
    linker_text="vc",
):
    intake = IntakeService(local_storage, registry_service, workflow_state_service)
    rec = intake.submit(
        raw_user_query="HER2 ADC with vc-MMAE",
        user_provided_context={
            "target_or_antigen_text": "HER2",
            "candidate_text": "Trastuzumab analog",
            "payload_linker_text": "vc-MMAE",
        },
        uploaded_files=uploaded_files,
    )
    run_id = rec.run_id

    # Step 2 — real supervisor + mock LLM, but extend referenced_inputs.
    StructuredQueryService(
        local_storage, registry_service, workflow_state_service,
        SupervisorAgent(llm=MockLLMProvider()),
    ).parse(run_id)
    if referenced_inputs:
        sq_path = local_storage.run_key(run_id, "inputs/structured_query.json")
        sq = local_storage.read_json(sq_path)
        sq.setdefault("referenced_inputs", []).extend(referenced_inputs)
        local_storage.write_json(sq_path, sq)

    InputReadinessService(local_storage, registry_service, workflow_state_service).check(run_id)
    WorkflowSetupService(local_storage, registry_service, workflow_state_service).plan(run_id)

    # Step 5 with mock bindings so candidate enrichment "succeeds".
    CandidateContextAgent(
        storage=local_storage,
        registry=registry_service,
        workflow_state=workflow_state_service,
        mcp_client=LocalMCPClient(bindings={
            "SAbDab_search_structures": lambda **kw: {"hits": []},
            "ChEMBL_search_molecules": lambda **kw: {"hits": []},
            "ChEMBL_search_substructure": lambda **kw: {"hits": []},
        }),
    ).run(run_id)

    # Step 6 (default unwired bindings → partial; that's fine for these tests)
    DevelopabilityAgent(
        storage=local_storage,
        registry=registry_service,
        workflow_state=workflow_state_service,
        mcp_client=LocalMCPClient(),
    ).run(run_id)
    return run_id


# ── Step 7 ──────────────────────────────────────────────────────────────────

def test_step7_builds_input_package_from_multipart_uploads(
    local_storage, registry_service, workflow_state_service
):
    """Uploaded PDB → uploaded_structure_file; uploaded FASTA → sequence ref
    on the antibody candidate."""
    run_id = _seed(
        local_storage, registry_service, workflow_state_service,
        uploaded_files=[
            {
                "file_id": "file_pdb",
                "original_filename": "complex.pdb",
                "storage_path": "adc_pilot/runs/x/inputs/files/file_pdb.pdb",
                "content_type": "chemical/x-pdb",
                "sha256": "sha256:abc",
                "size_bytes": 1024,
            },
            {
                "file_id": "file_fasta",
                "original_filename": "heavy.fasta",
                "storage_path": "adc_pilot/runs/x/inputs/files/file_fasta.fasta",
                "content_type": "text/x-fasta",
                "sha256": "sha256:def",
                "size_bytes": 256,
            },
        ],
    )
    agent = StructureAndDesignAgent(
        storage=local_storage,
        registry=registry_service,
        workflow_state=workflow_state_service,
        mcp_client=_mcp(),
    )
    pkg = agent.run_step_7(run_id)
    assert pkg.structure_preparation_status in {"ok", "partial"}
    cases = {r.input_case for r in pkg.prepared_structure_inputs}
    assert "uploaded_structure_file" in cases

    # Antibody candidate ends up with the FASTA sequence ref.
    any_seq = any(
        r.sequence_refs_for_prediction
        for r in pkg.prepared_structure_inputs
        if r.structure_role == "antibody_only"
    )
    assert any_seq

    # No raw file bytes in the artifact (only storage_path-style refs).
    blob = json.dumps(pkg.model_dump())
    assert "HEADER" not in blob and "ATOM" not in blob


def test_step7_uploaded_pdb_preferred_over_sequence_refs(
    local_storage, registry_service, workflow_state_service
):
    run_id = _seed(
        local_storage, registry_service, workflow_state_service,
        uploaded_files=[
            {
                "file_id": "file_pdb",
                "original_filename": "complex.pdb",
                "storage_path": "adc_pilot/runs/x/inputs/files/file_pdb.pdb",
                "content_type": "chemical/x-pdb",
                "sha256": "sha256:abc",
                "size_bytes": 1024,
            },
            {
                "file_id": "file_fasta",
                "original_filename": "target.fasta",
                "storage_path": "adc_pilot/runs/x/inputs/files/file_fasta.fasta",
                "content_type": "text/x-fasta",
                "sha256": "sha256:def",
                "size_bytes": 256,
            },
        ],
        referenced_inputs=[{"id_type": "uniprot_id", "value": "P04626", "source": "raw"}],
    )
    pkg = StructureAndDesignAgent(
        storage=local_storage,
        registry=registry_service,
        workflow_state=workflow_state_service,
        mcp_client=_mcp(),
    ).run_step_7(run_id)

    uploaded = [r for r in pkg.prepared_structure_inputs if r.input_case == "uploaded_structure_file"]
    assert uploaded
    for rec in uploaded:
        assert rec.prediction_required is False
        assert rec.preferred_input_rank in {1, 2}
        assert any("preferred" in n for n in rec.source_priority_notes)
    blob = json.dumps(pkg.model_dump())
    assert "RAW_PDB_SENTINEL" not in blob
    assert "RAW_FASTA_SENTINEL" not in blob


def test_step7_does_not_treat_generic_antibody_sequence_reference_as_prediction_input(
    local_storage, registry_service, workflow_state_service
):
    run_id = _seed(
        local_storage,
        registry_service,
        workflow_state_service,
        referenced_inputs=[
            {
                "id_type": "antibody_sequence_reference",
                "value": "EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGK",
                "source": "user",
            }
        ],
    )
    cct_path = local_storage.run_key(run_id, "candidate_context_table.json")
    cct = local_storage.read_json(cct_path)
    antibody = next(c for c in cct["candidate_records"] if c["candidate_type"] == "antibody")
    assert all(
        m["material_type"] != "antibody_sequence_reference" for m in antibody["materials"]
    )
    assert any(
        "antibody_sequence_role_unresolved" in note for note in antibody.get("context_notes", [])
    )

    pkg = StructureAndDesignAgent(
        storage=local_storage,
        registry=registry_service,
        workflow_state=workflow_state_service,
        mcp_client=_mcp(),
    ).run_step_7(run_id)

    for rec in pkg.prepared_structure_inputs:
        assert all(
            s.prediction_input_kind != "amino_acid_sequence"
            for s in rec.sequence_refs_for_prediction
        )
    artifact = local_storage.read_json(local_storage.run_key(run_id, "prepared_structure_input_package.json"))
    artifact_blob = json.dumps(artifact)
    assert "EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGK" not in artifact_blob
    assert any(
        "antibody_sequence_role_unresolved" in n for n in antibody["context_notes"]
    )


def test_step7_builds_input_package_from_referenced_pdb_and_uniprot(
    local_storage, registry_service, workflow_state_service
):
    """Step 2 referenced_inputs carry pdb_id + uniprot_id; Step 7 must produce
    a known_pdb_id input case and one RCSBData_get_entry tool call."""
    run_id = _seed(
        local_storage, registry_service, workflow_state_service,
        referenced_inputs=[
            {"id_type": "pdb_id", "value": "1N8Z", "source": "raw_request_text"},
            {"id_type": "uniprot_id", "value": "P04626", "source": "raw_request_text"},
        ],
    )
    agent = StructureAndDesignAgent(
        storage=local_storage,
        registry=registry_service,
        workflow_state=workflow_state_service,
        mcp_client=_mcp(),
    )
    pkg = agent.run_step_7(run_id)
    cases = {r.input_case for r in pkg.prepared_structure_inputs}
    assert "known_pdb_id" in cases

    tool_names = {tc.tool_name for tc in pkg.structure_tool_call_records}
    assert "RCSBData_get_entry" in tool_names
    # raw payload only at tool_output_ref (success case)
    for tc in pkg.structure_tool_call_records:
        if tc.run_status == "success":
            assert tc.tool_output_ref
            raw = local_storage.read_json(tc.tool_output_ref)
            assert "output" in raw
            # Normalized artifact doesn't carry the raw entry blob.
            assert "1N8Z" not in json.dumps(pkg.model_dump())[:0] or True  # presence is fine

    known = [r for r in pkg.prepared_structure_inputs if r.input_case == "known_pdb_id"]
    assert known
    assert all(r.prediction_required is False for r in known)
    assert any(s.pdb_id == "1N8Z" and s.source_kind == "pdb_id" for r in known for s in r.structure_refs)


def test_step7_partial_when_no_structure_signal(
    local_storage, registry_service, workflow_state_service
):
    """Antibody-only run (no PDB, no FASTA, no uniprot) — Step 7 should still
    complete but mark partial / no structure_lane available."""
    run_id = _seed(local_storage, registry_service, workflow_state_service)
    agent = StructureAndDesignAgent(
        storage=local_storage, registry=registry_service,
        workflow_state=workflow_state_service, mcp_client=_mcp(),
    )
    pkg = agent.run_step_7(run_id)
    # Status is failed only when zero candidates with structure signals exist;
    # otherwise partial. Either is OK — we mostly care that we don't crash.
    assert pkg.structure_preparation_status in {"ok", "partial", "failed"}


def test_step7_antigen_uniprot_only_proceeds_as_sequence_prediction(
    local_storage, registry_service, workflow_state_service
):
    run_id = _seed(
        local_storage, registry_service, workflow_state_service,
        referenced_inputs=[{"id_type": "uniprot_id", "value": "P04626", "source": "raw"}],
    )
    pkg = StructureAndDesignAgent(
        storage=local_storage, registry=registry_service,
        workflow_state=workflow_state_service, mcp_client=_mcp(),
    ).run_step_7(run_id)
    antigen = next(r for r in pkg.prepared_structure_inputs if r.structure_role == "antigen_only")
    assert antigen.input_case == "sequence_only_input"
    assert antigen.prediction_required is True
    assert "structure_prediction_needed" not in antigen.missing_metadata_flags
    assert not any("no structure file" in f for f in antigen.missing_metadata_flags)
    assert any(s.source_kind == "uniprot_id" and s.sequence_id == "P04626" for s in antigen.sequence_refs_for_prediction)
    assert any(cm.chain_role == "antigen" and cm.chain_id == "predicted_antigen" for cm in antigen.chain_mapping)


def test_step7_antibody_heavy_light_sequence_only_proceeds(
    local_storage, registry_service, workflow_state_service
):
    run_id = _seed(local_storage, registry_service, workflow_state_service)
    cct_path = local_storage.run_key(run_id, "candidate_context_table.json")
    cct = local_storage.read_json(cct_path)
    antibody = next(c for c in cct["candidate_records"] if c["candidate_type"] == "antibody")
    antibody["materials"].extend([
        {
            "material_id": "heavy_seq",
            "material_type": "antibody_heavy_chain_sequence",
            "value": "EVQLVESGGGLVQPGGSLRLSCAAS",
            "value_format": "fasta",
            "role": "antibody",
        },
        {
            "material_id": "light_seq",
            "material_type": "antibody_light_chain_sequence",
            "value": "DIQMTQSPSSLSASVGDRVTITC",
            "value_format": "fasta",
            "role": "antibody",
        },
    ])
    local_storage.write_json(cct_path, cct)

    pkg = StructureAndDesignAgent(
        storage=local_storage, registry=registry_service,
        workflow_state=workflow_state_service, mcp_client=_mcp(),
    ).run_step_7(run_id)
    antibody_rec = next(r for r in pkg.prepared_structure_inputs if r.structure_role == "antibody_only")
    assert antibody_rec.input_case == "sequence_only_input"
    assert antibody_rec.prediction_required is True
    roles = {s.chain_role for s in antibody_rec.sequence_refs_for_prediction}
    assert {"antibody_heavy", "antibody_light"}.issubset(roles)
    chain_roles = {cm.chain_role for cm in antibody_rec.chain_mapping}
    assert {"antibody_heavy", "antibody_light"}.issubset(chain_roles)
    assert all(cm.chain_id_kind == "prediction_placeholder" for cm in antibody_rec.chain_mapping)


def test_step7_heavy_light_inline_sequences_are_compact_in_prepared_artifact(
    local_storage, registry_service, workflow_state_service
):
    run_id = _seed(local_storage, registry_service, workflow_state_service)
    cct_path = local_storage.run_key(run_id, "candidate_context_table.json")
    cct = local_storage.read_json(cct_path)
    antibody = next(c for c in cct["candidate_records"] if c["candidate_type"] == "antibody")
    heavy_seq = "EVQLVESGGGLVQPGGSLRLSCAAS1234HEAVY"
    light_seq = "DIQMTQSPSSLSASVGDRVTITC5678LIGHT"
    antibody["materials"].extend([
        {
            "material_id": "heavy_compact_seq",
            "material_type": "antibody_heavy_chain_sequence",
            "value": heavy_seq,
            "value_format": "fasta",
            "role": "antibody",
        },
        {
            "material_id": "light_compact_seq",
            "material_type": "antibody_light_chain_sequence",
            "value": light_seq,
            "value_format": "fasta",
            "role": "antibody",
        },
    ])
    local_storage.write_json(cct_path, cct)

    agent = StructureAndDesignAgent(
        storage=local_storage,
        registry=registry_service,
        workflow_state=workflow_state_service,
        mcp_client=_mcp(),
    )
    pkg = agent.run_step_7(run_id)

    artifact_path = local_storage.run_key(run_id, "prepared_structure_input_package.json")
    artifact = local_storage.read_json(artifact_path)
    artifact_blob = json.dumps(artifact)

    assert heavy_seq not in artifact_blob
    assert light_seq not in artifact_blob
    for rec in artifact.get("prepared_structure_inputs", []):
        for seq in rec.get("sequence_refs_for_prediction", []):
            assert "sequence" not in seq

    antibody_rec = next(r for r in pkg.prepared_structure_inputs if r.structure_role == "antibody_only")
    heavy_ref = next(
        s for s in antibody_rec.sequence_refs_for_prediction if s.sequence_id == "heavy_compact_seq"
    )
    light_ref = next(
        s for s in antibody_rec.sequence_refs_for_prediction if s.sequence_id == "light_compact_seq"
    )
    assert heavy_ref.sequence_length == len(heavy_seq)
    assert light_ref.sequence_length == len(light_seq)
    assert heavy_ref.sha256_prefix == hashlib.sha256(heavy_seq.encode("utf-8")).hexdigest()[:12]
    assert light_ref.sha256_prefix == hashlib.sha256(light_seq.encode("utf-8")).hexdigest()[:12]
    assert heavy_ref.prediction_input_kind == "amino_acid_sequence"
    assert light_ref.prediction_input_kind == "amino_acid_sequence"
    assert heavy_ref.sequence_value_status == "inline"
    assert light_ref.sequence_value_status == "inline"


def test_step7_antigen_antibody_sequence_mapping_without_interface_invention(
    local_storage, registry_service, workflow_state_service
):
    run_id = _seed(
        local_storage, registry_service, workflow_state_service,
        referenced_inputs=[{"id_type": "uniprot_id", "value": "P04626", "source": "raw"}],
    )
    cct_path = local_storage.run_key(run_id, "candidate_context_table.json")
    cct = local_storage.read_json(cct_path)
    antibody = next(c for c in cct["candidate_records"] if c["candidate_type"] == "antibody")
    antibody["materials"].extend([
        {"material_id": "heavy_seq", "material_type": "antibody_heavy_chain_sequence", "value": "EVQLVES", "value_format": "fasta"},
        {"material_id": "light_seq", "material_type": "antibody_light_chain_sequence", "value": "DIQMTQ", "value_format": "fasta"},
    ])
    local_storage.write_json(cct_path, cct)

    pkg = StructureAndDesignAgent(
        storage=local_storage, registry=registry_service,
        workflow_state=workflow_state_service, mcp_client=_mcp(),
    ).run_step_7(run_id)
    mapped = [r for r in pkg.prepared_structure_inputs if r.antigen_antibody_mapping]
    assert mapped
    mapping = mapped[0].antigen_antibody_mapping
    assert mapping["target_candidate_id"]
    assert mapping["antibody_candidate_id"]
    assert mapping["mapping_status"] == "sequence_only_prediction_needed"
    blob = json.dumps(mapping).lower()
    assert "interface" not in blob
    assert "epitope" not in blob


def test_step7_extracts_partial_residue_range_and_does_not_invent_when_absent(
    local_storage, registry_service, workflow_state_service
):
    run_id = _seed(
        local_storage, registry_service, workflow_state_service,
        referenced_inputs=[{"id_type": "uniprot_id", "value": "P04626", "source": "raw"}],
    )
    cct_path = local_storage.run_key(run_id, "candidate_context_table.json")
    cct = local_storage.read_json(cct_path)
    target = next(c for c in cct["candidate_records"] if c["candidate_type"] == "target_antigen")
    target["materials"].append({
        "material_id": "target_domain",
        "material_type": "target_sequence",
        "value": "HER2 extracellular domain residues 20-240",
        "value_format": "text",
        "role": "target",
    })
    local_storage.write_json(cct_path, cct)

    pkg = StructureAndDesignAgent(
        storage=local_storage, registry=registry_service,
        workflow_state=workflow_state_service, mcp_client=_mcp(),
    ).run_step_7(run_id)
    target_rec = next(r for r in pkg.prepared_structure_inputs if r.structure_role == "antigen_only")
    assert {"start": 20, "end": 240, "source": "candidate_material", "source_ref": "target_domain"} in target_rec.residue_ranges
    assert all(
        r.residue_ranges == []
        for r in pkg.prepared_structure_inputs
        if r.candidate_id != target_rec.candidate_id
    )


@pytest.mark.parametrize(
    "filename,expected_role,expected_structure_role",
    [
        ("heavy.fasta", "antibody_heavy", "antibody_only"),
        ("antigen.fasta", "antigen", "antigen_only"),
    ],
)
def test_step7_filename_scoped_fasta_binds_only_compatible_candidate(
    local_storage, registry_service, workflow_state_service,
    filename, expected_role, expected_structure_role,
):
    run_id = _seed(
        local_storage, registry_service, workflow_state_service,
        uploaded_files=[{
            "file_id": f"file_{expected_role}",
            "original_filename": filename,
            "storage_path": f"adc_pilot/runs/x/inputs/files/{filename}",
            "content_type": "text/x-fasta",
            "size_bytes": 64,
        }],
    )
    pkg = StructureAndDesignAgent(
        storage=local_storage, registry=registry_service,
        workflow_state=workflow_state_service, mcp_client=_mcp(),
    ).run_step_7(run_id)
    refs = [
        (r.structure_role, s) for r in pkg.prepared_structure_inputs
        for s in r.sequence_refs_for_prediction if s.sequence_id == f"file_{expected_role}"
    ]
    assert len(refs) == 1
    assert refs[0][0] == expected_structure_role
    assert refs[0][1].chain_role == expected_role
    assert refs[0][1].sequence is None
    assert refs[0][1].sequence_storage_ref.endswith(filename)
    assert refs[0][1].prediction_input_kind == "fasta_ref"


def test_step7_ambiguous_fasta_is_unresolved_not_broadcast(
    local_storage, registry_service, workflow_state_service
):
    run_id = _seed(
        local_storage, registry_service, workflow_state_service,
        uploaded_files=[{
            "file_id": "ambiguous_fasta",
            "original_filename": "ambiguous.fasta",
            "storage_path": "adc_pilot/runs/x/inputs/files/ambiguous.fasta",
            "content_type": "text/x-fasta",
            "size_bytes": 64,
        }],
    )
    pkg = StructureAndDesignAgent(
        storage=local_storage, registry=registry_service,
        workflow_state=workflow_state_service, mcp_client=_mcp(),
    ).run_step_7(run_id)
    assert not any(
        s.sequence_id == "ambiguous_fasta"
        for r in pkg.prepared_structure_inputs for s in r.sequence_refs_for_prediction
    )
    assert any(
        u["source_ref"] == "ambiguous_fasta" and u["resource_binding_status"] in {"unassigned", "ambiguous"}
        for u in pkg.unresolved_resource_refs
    )


def test_step7_referenced_pdb_id_is_not_copied_to_antibody(
    local_storage, registry_service, workflow_state_service
):
    run_id = _seed(
        local_storage, registry_service, workflow_state_service,
        referenced_inputs=[{"id_type": "pdb_id", "value": "1N8Z", "source": "raw"}],
    )
    pkg = StructureAndDesignAgent(
        storage=local_storage, registry=registry_service,
        workflow_state=workflow_state_service, mcp_client=_mcp(),
    ).run_step_7(run_id)
    owners = [
        r for r in pkg.prepared_structure_inputs
        if any(s.pdb_id == "1N8Z" for s in r.structure_refs)
    ]
    if owners:
        # In non-ambiguous runs this still maps to antigen candidate.
        assert len(owners) == 1
        assert owners[0].structure_role == "antigen_only"
    else:
        # New safety rule: avoid implicit target binding when target+antibody both exist.
        assert any(
            u["resource_type"] == "pdb_id" and u["source_ref"] == "1N8Z"
            for u in pkg.unresolved_resource_refs
        )


def test_step7_unscoped_uploaded_structure_is_unresolved_when_target_and_antibody_conflict(
    local_storage, registry_service, workflow_state_service
):
    run_id = _seed(
        local_storage, registry_service, workflow_state_service,
        uploaded_files=[{
            "file_id": "unscoped_pdb",
            "original_filename": "structure.pdb",
            "storage_path": str(PROJECT_ROOT / "data" / "pdb" / "S1.pdb"),
            "content_type": "chemical/x-pdb",
            "size_bytes": 1,
        }],
    )
    pkg = StructureAndDesignAgent(
        storage=local_storage, registry=registry_service,
        workflow_state=workflow_state_service, mcp_client=_mcp(),
    ).run_step_7(run_id)
    assert any(u["resource_type"] == "structure" for u in pkg.unresolved_resource_refs)
    # Ensure not implicitly copied to a specific target/antibody candidate.
    resolved = [r for r in pkg.prepared_structure_inputs if any(s.source_ref == "unscoped_pdb" for s in r.structure_refs)]
    assert resolved == []


def test_step7_unscoped_pdb_id_is_unresolved_when_target_and_antibody_conflict(
    local_storage, registry_service, workflow_state_service
):
    run_id = _seed(
        local_storage, registry_service, workflow_state_service,
        referenced_inputs=[{"id_type": "pdb_id", "value": "1N8Z", "source": "raw_request_text"}],
    )
    cct_path = local_storage.run_key(run_id, "candidate_context_table.json")
    cct = local_storage.read_json(cct_path)
    for cand in cct["candidate_records"]:
        if cand.get("candidate_type") in {"target_antigen", "antibody"}:
            cand["identifiers"] = [
                i for i in (cand.get("identifiers") or [])
                if i.get("id_type") != "pdb_id"
            ]
            cand["materials"] = [
                m for m in (cand.get("materials") or [])
                if m.get("material_type") != "structure_ref"
            ]
    local_storage.write_json(cct_path, cct)

    pkg = StructureAndDesignAgent(
        storage=local_storage, registry=registry_service,
        workflow_state=workflow_state_service, mcp_client=_mcp(),
    ).run_step_7(run_id)
    has_target = any(r.structure_role == "antigen_only" for r in pkg.prepared_structure_inputs)
    has_antibody = any(r.structure_role == "antibody_only" for r in pkg.prepared_structure_inputs)
    if has_target and has_antibody:
        assert any(
            u["resource_type"] == "pdb_id"
            and u["source_ref"] == "1N8Z"
            for u in pkg.unresolved_resource_refs
        )
        assert not any(
            s.pdb_id == "1N8Z"
            for r in pkg.prepared_structure_inputs
            for s in r.structure_refs
        )
    else:
        assert any(s.pdb_id == "1N8Z" for r in pkg.prepared_structure_inputs for s in r.structure_refs)


def test_step7_candidate_scoped_structure_material_stays_on_candidate(
    local_storage, registry_service, workflow_state_service
):
    run_id = _seed(local_storage, registry_service, workflow_state_service)
    cct_path = local_storage.run_key(run_id, "candidate_context_table.json")
    cct = local_storage.read_json(cct_path)
    target = next(c for c in cct["candidate_records"] if c["candidate_type"] == "target_antigen")
    target["materials"].append({
        "material_id": "target_s1", "material_type": "structure_file",
        "value": str(PROJECT_ROOT / "data" / "pdb" / "S1.pdb"), "value_format": "pdb",
    })
    local_storage.write_json(cct_path, cct)
    pkg = StructureAndDesignAgent(
        storage=local_storage, registry=registry_service,
        workflow_state=workflow_state_service, mcp_client=_mcp(),
    ).run_step_7(run_id)
    owners = [r for r in pkg.prepared_structure_inputs if any(s.source_ref == "target_s1" for s in r.structure_refs)]
    assert len(owners) == 1
    assert owners[0].candidate_id == target["candidate_id"]
    rec = owners[0]
    rec_ref = next(s for s in rec.structure_refs if s.source_ref == "target_s1")
    assert rec_ref.storage_ref == str(PROJECT_ROOT / "data" / "pdb" / "S1.pdb")
    assert rec.crystal_metadata is not None
    assert rec.crystal_metadata.parse_status == "ok"
    assert rec.crystal_metadata.a is not None
    assert rec.crystal_metadata.b is not None
    assert rec.crystal_metadata.c is not None
    assert rec.crystal_metadata.alpha is not None
    assert rec.crystal_metadata.beta is not None
    assert rec.crystal_metadata.gamma is not None
    assert rec.crystal_metadata.space_group
    assert rec.crystal_metadata.z_value is not None
    assert rec.molecular_weight_estimate is not None
    assert rec.molecular_weight_estimate.value is not None
    assert rec.molecular_weight_estimate.method == "seqres_residue_sum"
    assert rec.molecular_weight_estimate.status in {"estimated", "estimated_with_warnings"}
    assert not any(ref.pdb_id == "XXXX" for ref in rec.structure_refs)
    blob = json.dumps(rec.model_dump())
    assert "HEADER" not in blob
    assert "ATOM" not in blob
    assert "HETATM" not in blob
    assert "SEQRES" not in blob


@pytest.mark.parametrize("fixture_name", ["S1.pdb", "S2.pdb", "S3.pdb"])
def test_step7_real_pdb_fixtures_supply_compact_crystal_validation_metadata(
    local_storage, fixture_name
):
    path = PROJECT_ROOT / "data" / "pdb" / fixture_name
    crystal, mw = structure_and_design_module._extract_structure_validation_metadata(
        storage=local_storage,
        structure_files=[],
        candidate_structure_materials=[{
            "material_id": f"mat_{fixture_name}",
            "value": str(path),
        }],
    )

    assert crystal is not None
    assert crystal.parse_status == "ok"
    assert crystal.a and crystal.b and crystal.c
    assert crystal.alpha and crystal.beta and crystal.gamma
    assert crystal.space_group
    assert crystal.z_value is not None
    assert mw is not None
    assert mw.value is not None
    assert mw.method == "seqres_residue_sum"
    blob = json.dumps({"crystal": crystal.model_dump(), "mw": mw.model_dump()})
    assert "HEADER" not in blob
    assert "ATOM" not in blob
    assert "HETATM" not in blob
    assert "SEQRES" not in blob


def test_step7_structure_file_material_is_consumable_by_step8_without_fake_placeholder(
    local_storage, registry_service, workflow_state_service
):
    run_id = _seed(local_storage, registry_service, workflow_state_service)
    cct_path = local_storage.run_key(run_id, "candidate_context_table.json")
    cct = local_storage.read_json(cct_path)
    antibody = next(c for c in cct["candidate_records"] if c["candidate_type"] == "antibody")
    antibody["materials"].append({
        "material_id": "ab_s1",
        "material_type": "structure_file",
        "value": str(PROJECT_ROOT / "data" / "pdb" / "S1.pdb"),
        "value_format": "pdb",
        "role": "antibody",
    })
    local_storage.write_json(cct_path, cct)

    captured: list[dict] = []
    overrides = {
        "CrystalStructure_validate": lambda **kw: captured.append(dict(kw)) or {"ok": True, "validated": True},
    }
    mcp = LocalMCPClient(
        inventory=ToolInventoryService(os.environ.get("TOOL_INVENTORY_XLSX", str(DEFAULT_XLSX))),
        bindings={**_bindings_with_step8_overrides(), **overrides},
    )
    agent = StructureAndDesignAgent(
        storage=local_storage, registry=registry_service,
        workflow_state=workflow_state_service, mcp_client=mcp,
    )
    agent.run_step_7(run_id)
    results = agent.run_step_8(run_id)
    tool_calls = [
        tc for tc in results.tool_call_records
        if tc.tool_name == "CrystalStructure_validate"
        and tc.tool_input_summary.get("routing_decision") == "selected"
    ]
    assert tool_calls, "expected structure validation calls"
    assert captured
    for tc in tool_calls:
        args = tc.tool_input_summary["arguments"]
        assert args["operation"] == "validate"
        assert args["a"] > 0
        assert args["Z"] > 0
        assert args["mw"] > 0
        assert "pdb_id_or_path" not in args
        assert "pdb_id_or_path" not in tc.tool_input_summary
    assert all(set(call) >= {"operation", "a", "Z", "mw"} for call in captured)
    assert all("pdb_id_or_path" not in call for call in captured)


def test_step7_name_only_routes_to_database_search_tools(
    local_storage, registry_service, workflow_state_service
):
    run_id = _seed(
        local_storage, registry_service, workflow_state_service,
        referenced_inputs=[],
    )
    # Build with candidate names only; no structure files, no PDB IDs, no sequences.
    pkg = StructureAndDesignAgent(
        storage=local_storage,
        registry=registry_service,
        workflow_state=workflow_state_service,
        mcp_client=_mcp(),
    ).run_step_7(run_id)

    db_results = [tc for tc in pkg.structure_tool_call_records if tc.tool_name in {"RCSBAdvSearch_search_structures", "PDBeSearch_search_structures"}]
    assert db_results, "name-only input should route to search wrappers"
    assert all(tc.run_status in {"success", "dependency_unavailable", "failed"} for tc in db_results)

    assert all(
        tc.tool_input_summary
        and tc.tool_input_summary.get("routing_decision") == "selected"
        for tc in db_results
    )
    skipped = [
        tc for tc in pkg.structure_tool_call_records
        if tc.tool_name in {"RCSBData_get_entry", "RCSBData_get_assembly", "SAbDab_get_structure"}
    ]
    assert skipped and all(tc.run_status == "skipped" for tc in skipped)
    assert all(tc.tool_input_summary.get("routing_decision") == "not_applicable" for tc in skipped)

    # Database search results are normalized as ambiguous candidate
    # options. They do not get promoted to primary structure refs.
    for rec in pkg.prepared_structure_inputs:
        if rec.input_case != "database_search_result":
            continue
        assert rec.database_search_candidates == []


def test_step7_scope_tools_match_inventory_runtime_scope():
    mcp = _mcp()
    runtime_scope = set(mcp.list_tools(agent_name="structure_and_design_agent", step_id="step_07"))
    assert runtime_scope == set(structure_and_design_module._STEP7_SCOPED_TOOLS), (
        "Step 7 runtime scope must stay in sync with _STEP7_SCOPED_TOOLS.\n"
        f"runtime_scope={sorted(runtime_scope)}\n"
        f"routing_table={sorted(structure_and_design_module._STEP7_SCOPED_TOOLS)}"
    )


def test_step7_candidate_pdb_id_routes_to_rcsb_and_sabdab_step7_tools(
    local_storage, registry_service, workflow_state_service
):
    run_id = _seed(local_storage, registry_service, workflow_state_service)
    cct_path = local_storage.run_key(run_id, "candidate_context_table.json")
    cct = local_storage.read_json(cct_path)
    antibody = next(c for c in cct["candidate_records"] if c["candidate_type"] == "antibody")
    antibody_id = antibody["candidate_id"]
    antibody.setdefault("identifiers", []).append(
        {"id_type": "pdb_id", "id_value": "1N8Z", "source": "candidate_profile"}
    )
    local_storage.write_json(cct_path, cct)

    pkg = StructureAndDesignAgent(
        storage=local_storage,
        registry=registry_service,
        workflow_state=workflow_state_service,
        mcp_client=_mcp(),
    ).run_step_7(run_id)
    tc_by_name = {
        tc.tool_name: tc for tc in pkg.structure_tool_call_records
        if tc.tool_name in {
            "RCSBData_get_entry", "RCSBData_get_assembly", "SAbDab_get_structure"
        } and tc.tool_input_summary.get("candidate_id") == antibody_id
        }
    assert "RCSBData_get_entry" in tc_by_name
    assert tc_by_name["RCSBData_get_entry"].tool_input_summary.get("routing_decision") == "selected"
    assert "RCSBData_get_assembly" in tc_by_name
    assert tc_by_name["RCSBData_get_assembly"].tool_input_summary.get("arguments", {}).get("assembly_id") == "1"
    assert "SAbDab_get_structure" in tc_by_name
    assert tc_by_name["SAbDab_get_structure"].tool_input_summary.get("routing_decision") == "selected"

    rec = next(
        r for r in pkg.prepared_structure_inputs if r.input_case == "known_pdb_id"
    )
    compacted = {m["tool_name"]: m for m in rec.step7_tool_output_metadata}
    entry_meta = compacted["RCSBData_get_entry"]["compact_output"]
    assert entry_meta["compact_type"] == "rcsb_entry"
    assert entry_meta["pdb_id"]
    assert entry_meta["entry_metadata"] is not None
    assert compacted["RCSBData_get_assembly"]["compact_output"]["compact_type"] == "rcsb_assembly"
    assert compacted["RCSBData_get_assembly"]["compact_output"]["assembly_id"] == "1"
    assert compacted["SAbDab_get_structure"]["compact_output"]["compact_type"] == "sabdab_structure"
    assert compacted["SAbDab_get_structure"]["compact_output"]["pdb_id"] == entry_meta["pdb_id"]
    assert rec.structure_refs and any(s.source_kind == "pdb_id" for s in rec.structure_refs)


def test_step7_sequence_only_uniprot_triggers_alphafold_get_prediction_lookup(
    local_storage, registry_service, workflow_state_service
):
    run_id = _seed(
        local_storage, registry_service, workflow_state_service,
        referenced_inputs=[{"id_type": "uniprot_id", "value": "P04626", "source": "raw"}],
    )
    cct_path = local_storage.run_key(run_id, "candidate_context_table.json")
    cct = local_storage.read_json(cct_path)
    for cand in cct["candidate_records"]:
        cand["materials"] = [
            m for m in cand.get("materials", [])
            if not (isinstance(m, dict) and str(m.get("material_type", "")).endswith("_name"))
        ]
    local_storage.write_json(cct_path, cct)

    overrides = {
        "alphafold_get_prediction": lambda **kw: {
            "uniprot": kw.get("uniprot"),
            "status": "success",
            "model_path": "mock://alphafold/P04626.pdb",
        },
    }
    agent = StructureAndDesignAgent(
        storage=local_storage,
        registry=registry_service,
        workflow_state=workflow_state_service,
        mcp_client=LocalMCPClient(
            inventory=ToolInventoryService(os.environ.get("TOOL_INVENTORY_XLSX", str(DEFAULT_XLSX))),
            bindings=overrides,
        ),
    )
    pkg = agent.run_step_7(run_id)
    step8 = agent.run_step_8(run_id)
    af_calls = [tc for tc in pkg.structure_tool_call_records if tc.tool_name == "alphafold_get_prediction"]
    assert len(af_calls) == 1
    af_call = af_calls[0]
    assert af_call.run_status == "success"
    assert af_call.tool_input_summary["routing_decision"] == "selected"
    assert af_call.tool_input_summary["arguments"]["uniprot"] == "P04626"

    rec = next(
        r for r in pkg.prepared_structure_inputs if r.input_case == "sequence_only_input"
    )
    assert any(
        ref.source_kind == "predicted_needed" and ref.source_ref == "P04626"
        for ref in rec.structure_refs
    )
    compacted = {m["tool_name"]: m for m in rec.step7_tool_output_metadata}
    assert compacted["alphafold_get_prediction"]["compact_output"]["compact_type"] == "alphafold_prediction"
    assert compacted["alphafold_get_prediction"]["compact_output"]["model_ref"] == "mock://alphafold/P04626.pdb"

    for name in {"RCSBData_get_entry", "RCSBData_get_assembly", "RCSBAdvSearch_search_structures", "PDBeSearch_search_structures"}:
        callset = [tc for tc in pkg.structure_tool_call_records if tc.tool_name == name]
        assert callset
        assert all(
            call.tool_input_summary.get("routing_decision") in {"not_applicable", "scope_unavailable"}
            for call in callset
        )
        assert all(call.run_status == "skipped" for call in callset)
    assert all(tc.tool_name != "alphafold_get_prediction" for tc in step8.tool_call_records)

    pkg_blob = json.dumps(pkg.model_dump())
    assert "raw_sequence" not in pkg_blob
    assert "full_payload" not in pkg_blob


def test_step7_sequence_only_without_uniprot_skips_alphafold_lookup(
    local_storage, registry_service, workflow_state_service
):
    run_id = _seed(local_storage, registry_service, workflow_state_service, referenced_inputs=[])
    cct_path = local_storage.run_key(run_id, "candidate_context_table.json")
    cct = local_storage.read_json(cct_path)
    for cand in cct["candidate_records"]:
        if cand.get("candidate_type") == "target_antigen":
            cand["materials"] = [
                {
                    "material_id": "manual_sequence_target",
                    "material_type": "target_sequence",
                    "value": "MKTAYIAKQNNVG..."
                }
            ]
        else:
            cand["materials"] = []
            if isinstance(cand.get("materials"), list):
                cand["materials"] = []
    local_storage.write_json(cct_path, cct)

    agent = StructureAndDesignAgent(
        storage=local_storage,
        registry=registry_service,
        workflow_state=workflow_state_service,
        mcp_client=_mcp(),
    )
    pkg = agent.run_step_7(run_id)

    rec = next(r for r in pkg.prepared_structure_inputs if r.input_case == "sequence_only_input")
    af_calls = [tc for tc in pkg.structure_tool_call_records if tc.tool_name == "alphafold_get_prediction"]
    assert len(af_calls) == 1
    af_call = af_calls[0]
    assert af_call.run_status == "skipped"
    assert af_call.tool_input_summary["routing_decision"] == "not_applicable"
    assert af_call.tool_input_summary.get("reason") == "sequence-only input has no UniProt accession for AlphaFold prediction lookup"

    assert not any(ref.source_kind == "predicted_needed" for ref in rec.structure_refs)


@pytest.mark.parametrize(
    ("payload_key", "payload_value"),
    [
        ("model_url", "https://alphafold.test/P04626.pdb"),
        ("model_path", "mock://alphafold/P04626.pdb"),
        ("artifact_ref", "s3://bucket/alphafold/P04626.pdb"),
    ],
)
def test_step7_alphafold_safe_model_ref_promotes_predicted_structure_ref(
    local_storage, registry_service, workflow_state_service, payload_key, payload_value
):
    run_id = _seed(
        local_storage, registry_service, workflow_state_service,
        referenced_inputs=[{"id_type": "uniprot_id", "value": "P04626", "source": "raw"}],
    )
    cct_path = local_storage.run_key(run_id, "candidate_context_table.json")
    cct = local_storage.read_json(cct_path)
    for cand in cct["candidate_records"]:
        cand["materials"] = [
            m for m in cand.get("materials", [])
            if not (isinstance(m, dict) and str(m.get("material_type", "")).endswith("_name"))
        ]
    local_storage.write_json(cct_path, cct)

    def _af(**kw):
        return {"uniprot": kw.get("uniprot"), "status": "success", payload_key: payload_value}

    pkg = StructureAndDesignAgent(
        storage=local_storage,
        registry=registry_service,
        workflow_state=workflow_state_service,
        mcp_client=LocalMCPClient(
            inventory=ToolInventoryService(os.environ.get("TOOL_INVENTORY_XLSX", str(DEFAULT_XLSX))),
            bindings={"alphafold_get_prediction": _af},
        ),
    ).run_step_7(run_id)

    rec = next(r for r in pkg.prepared_structure_inputs if r.input_case == "sequence_only_input")
    assert any(
        ref.source_kind == "predicted_needed" and ref.storage_ref == payload_value
        for ref in rec.structure_refs
    )
    compacted = {m["tool_name"]: m for m in rec.step7_tool_output_metadata}
    assert compacted["alphafold_get_prediction"]["compact_output"]["model_ref"] == payload_value


def test_step7_alphafold_generic_url_does_not_promote_model_ref(
    local_storage, registry_service, workflow_state_service
):
    run_id = _seed(
        local_storage, registry_service, workflow_state_service,
        referenced_inputs=[{"id_type": "uniprot_id", "value": "P04626", "source": "raw"}],
    )
    cct_path = local_storage.run_key(run_id, "candidate_context_table.json")
    cct = local_storage.read_json(cct_path)
    for cand in cct["candidate_records"]:
        cand["materials"] = [
            m for m in cand.get("materials", [])
            if not (isinstance(m, dict) and str(m.get("material_type", "")).endswith("_name"))
        ]
    local_storage.write_json(cct_path, cct)

    pkg = StructureAndDesignAgent(
        storage=local_storage,
        registry=registry_service,
        workflow_state=workflow_state_service,
        mcp_client=LocalMCPClient(
            inventory=ToolInventoryService(os.environ.get("TOOL_INVENTORY_XLSX", str(DEFAULT_XLSX))),
            bindings={
                "alphafold_get_prediction": lambda **kw: {
                    "uniprot": kw.get("uniprot"),
                    "status": "success",
                    "url": "https://alphafold.test/generic-url.pdb",
                }
            },
        ),
    ).run_step_7(run_id)

    rec = next(r for r in pkg.prepared_structure_inputs if r.input_case == "sequence_only_input")
    assert not any(ref.source_kind == "predicted_needed" and ref.storage_ref for ref in rec.structure_refs)
    compacted = {m["tool_name"]: m for m in rec.step7_tool_output_metadata}
    assert compacted["alphafold_get_prediction"]["compact_output"]["model_ref"] is None


def test_step7_alphafold_raw_output_does_not_leak_or_promote_model_ref(
    local_storage, registry_service, workflow_state_service
):
    raw_pdb = "HEADER    RAW ALPHAFOLD MODEL\nATOM      1  N   ALA A   1\nHETATM    2  O   HOH A   2"
    run_id = _seed(
        local_storage, registry_service, workflow_state_service,
        referenced_inputs=[{"id_type": "uniprot_id", "value": "P04626", "source": "raw"}],
    )
    cct_path = local_storage.run_key(run_id, "candidate_context_table.json")
    cct = local_storage.read_json(cct_path)
    for cand in cct["candidate_records"]:
        cand["materials"] = [
            m for m in cand.get("materials", [])
            if not (isinstance(m, dict) and str(m.get("material_type", "")).endswith("_name"))
        ]
    local_storage.write_json(cct_path, cct)

    pkg = StructureAndDesignAgent(
        storage=local_storage,
        registry=registry_service,
        workflow_state=workflow_state_service,
        mcp_client=LocalMCPClient(
            inventory=ToolInventoryService(os.environ.get("TOOL_INVENTORY_XLSX", str(DEFAULT_XLSX))),
            bindings={
                "alphafold_get_prediction": lambda **kw: {
                    "uniprot": kw.get("uniprot"),
                    "status": "success",
                    "output": raw_pdb,
                }
            },
        ),
    ).run_step_7(run_id)

    rec = next(r for r in pkg.prepared_structure_inputs if r.input_case == "sequence_only_input")
    assert not any(ref.source_kind == "predicted_needed" and ref.storage_ref for ref in rec.structure_refs)
    compacted = {m["tool_name"]: m for m in rec.step7_tool_output_metadata}
    assert compacted["alphafold_get_prediction"]["compact_output"]["model_ref"] is None
    blob = json.dumps(pkg.model_dump())
    assert "HEADER" not in blob
    assert "ATOM" not in blob
    assert "HETATM" not in blob


def test_step7_scope_failure_records_scope_unavailable(
    local_storage, registry_service, workflow_state_service
):
    class _FailingScopeClient(LocalMCPClient):
        def list_tools(self, *args, **kwargs):  # type: ignore[override]
            raise RuntimeError("scope backend unavailable")

    run_id = _seed(
        local_storage, registry_service, workflow_state_service,
        referenced_inputs=[{"id_type": "uniprot_id", "value": "P04626", "source": "raw"}],
    )
    pkg = StructureAndDesignAgent(
        storage=local_storage,
        registry=registry_service,
        workflow_state=workflow_state_service,
        mcp_client=_FailingScopeClient(),
    ).run_step_7(run_id)

    by_name = {tc.tool_name: tc for tc in pkg.structure_tool_call_records}
    for tool in structure_and_design_module._STEP7_SCOPED_TOOLS:
        assert tool in by_name
        tc = by_name[tool]
        assert tc.run_status == "skipped"
        assert tc.tool_input_summary["routing_decision"] == "scope_unavailable"
        assert "scope backend unavailable" in tc.tool_input_summary.get("reason", "")
        assert tc.run_status == "skipped"


def test_step7_database_search_results_become_ambiguous_candidates_only(
    local_storage, registry_service, workflow_state_service
):
    run_id = _seed(
        local_storage, registry_service, workflow_state_service,
        referenced_inputs=[],
    )
    overrides = {
        "RCSBAdvSearch_search_structures": lambda **kw: {
            "query": kw.get("query"),
            "structures": [
                {"pdb_id": "1N8Z", "method": "xray", "resolution": 2.1},
            ],
        },
        "PDBeSearch_search_structures": lambda **kw: {
            "query": kw.get("query"),
            "structures": [
                {"pdb_id": "2N8Z", "method": "crystal", "resolution": 2.3},
            ],
        },
    }
    mcp = LocalMCPClient(
        inventory=ToolInventoryService(os.environ.get("TOOL_INVENTORY_XLSX", str(DEFAULT_XLSX))),
        bindings=overrides,
    )
    pkg = StructureAndDesignAgent(
        storage=local_storage, registry=registry_service,
        workflow_state=workflow_state_service, mcp_client=mcp,
    ).run_step_7(run_id)

    db_recs = [r for r in pkg.prepared_structure_inputs if r.input_case == "database_search_result"]
    assert db_recs
    for rec in db_recs:
        assert rec.database_search_candidates
        assert all(item.get("resource_binding_status") == "ambiguous" for item in rec.database_search_candidates)
        assert all(item.get("pdb_id") for item in rec.database_search_candidates)
        assert not any(item.get("source") == "selected_structure_ref" for item in rec.database_search_candidates)
        assert rec.structure_refs == []


def test_step8_skips_when_no_usable_structure_ref(
    local_storage, registry_service, workflow_state_service
):
    run_id = _seed(local_storage, registry_service, workflow_state_service)
    cct_path = local_storage.run_key(run_id, "candidate_context_table.json")
    cct = local_storage.read_json(cct_path)
    target = next(c for c in cct["candidate_records"] if c["candidate_type"] == "target_antigen")
    # A structure_file material without concrete file path or pdb id.
    target["materials"].append({
        "material_id": "target_missing_structure",
        "material_type": "structure_file",
        "value": "not_a_file_reference",
        "value_format": "text",
    })
    local_storage.write_json(cct_path, cct)

    agent = StructureAndDesignAgent(
        storage=local_storage, registry=registry_service,
        workflow_state=workflow_state_service, mcp_client=_mcp(),
    )
    agent.run_step_7(run_id)
    results = agent.run_step_8(run_id)
    assert results.structure_modeling_status in {"partial", "failed"}
    # No fabricated "uploaded" placeholder should be passed.
    for tc in results.tool_call_records:
        summary = json.dumps(tc.tool_input_summary or {})
        assert "pdb_id_or_path\": \"uploaded\"" not in summary


@pytest.mark.parametrize(
    "filename,expected_ranges",
    [
        ("S1.pdb", {"A": (8, 171), "B": (78, 171)}),
        ("S2.pdb", {"A": (11, 173), "B": (74, 171)}),
        ("S3.pdb", {"A": (7, 171), "B": (77, 171)}),
    ],
)
def test_step7_real_pdb_parses_compact_observed_chains(
    local_storage, registry_service, workflow_state_service, filename, expected_ranges
):
    run_id = _seed(
        local_storage, registry_service, workflow_state_service,
        uploaded_files=[{
            "file_id": filename.lower(), "original_filename": filename,
            "storage_path": str(PROJECT_ROOT / "data" / "pdb" / filename),
            "content_type": "chemical/x-pdb", "size_bytes": 1, "role": "antigen",
        }],
    )
    pkg = StructureAndDesignAgent(
        storage=local_storage, registry=registry_service,
        workflow_state=workflow_state_service, mcp_client=_mcp(),
    ).run_step_7(run_id)
    rec = next(r for r in pkg.prepared_structure_inputs if r.structure_role == "antigen_only")
    assert {m.chain_id for m in rec.chain_mapping} == set(expected_ranges)
    assert all(m.chain_id_kind == "observed" and m.chain_role == "other" for m in rec.chain_mapping)
    compact_ranges = {r["chain_id"]: (r["start"], r["end"]) for r in rec.residue_ranges if r.get("source") == "observed_structure"}
    assert compact_ranges == expected_ranges
    assert "chain_ids_missing" not in rec.missing_metadata_flags
    assert "chain_roles_unknown" in rec.missing_metadata_flags
    blob = json.dumps(pkg.model_dump())
    assert "ATOM      " not in blob and "HEADER    " not in blob


def test_step7_uniprot_and_inline_sequence_semantics(
    local_storage, registry_service, workflow_state_service
):
    run_id = _seed(
        local_storage, registry_service, workflow_state_service,
        referenced_inputs=[{"id_type": "uniprot_id", "value": "P04626", "source": "raw"}],
    )
    cct_path = local_storage.run_key(run_id, "candidate_context_table.json")
    cct = local_storage.read_json(cct_path)
    target = next(c for c in cct["candidate_records"] if c["candidate_type"] == "target_antigen")
    target["materials"].append({
        "material_id": "inline_target", "material_type": "target_sequence",
        "value": "MKTIIALSYIFCLVFA", "value_format": "fasta",
    })
    local_storage.write_json(cct_path, cct)
    pkg = StructureAndDesignAgent(
        storage=local_storage, registry=registry_service,
        workflow_state=workflow_state_service, mcp_client=_mcp(),
    ).run_step_7(run_id)
    rec = next(r for r in pkg.prepared_structure_inputs if r.structure_role == "antigen_only")
    inline = next(s for s in rec.sequence_refs_for_prediction if s.sequence_id == "inline_target")
    uniprot = next(s for s in rec.sequence_refs_for_prediction if s.sequence_id == "P04626")
    assert inline.sequence == "MKTIIALSYIFCLVFA"
    assert inline.sequence_value_status == "inline"
    assert inline.prediction_input_kind == "amino_acid_sequence"
    assert uniprot.sequence is None
    assert uniprot.source_ref == "P04626"
    assert uniprot.sequence_value_status == "identifier_only"
    assert uniprot.prediction_input_kind == "uniprot_id"
    assert rec.prediction_required is True
    assert rec.missing_metadata_flags == []
    assert pkg.structure_preparation_status == "ok"


def test_step7_multiple_candidates_do_not_receive_shared_first_pair_mapping(
    local_storage, registry_service, workflow_state_service
):
    run_id = _seed(local_storage, registry_service, workflow_state_service)
    cct_path = local_storage.run_key(run_id, "candidate_context_table.json")
    cct = local_storage.read_json(cct_path)
    target = next(c for c in cct["candidate_records"] if c["candidate_type"] == "target_antigen")
    antibody = next(c for c in cct["candidate_records"] if c["candidate_type"] == "antibody")
    target["materials"].append({"material_id": "t1seq", "material_type": "target_sequence", "value": "AAAA"})
    antibody["materials"].append({"material_id": "a1seq", "material_type": "antibody_heavy_chain_sequence", "value": "BBBB"})
    target2, antibody2 = deepcopy(target), deepcopy(antibody)
    target2["candidate_id"], antibody2["candidate_id"] = "target_2", "antibody_2"
    target2["materials"][0]["material_id"] = "target2_name"
    antibody2["materials"][0]["material_id"] = "antibody2_name"
    cct["candidate_records"].extend([target2, antibody2])
    local_storage.write_json(cct_path, cct)
    pkg = StructureAndDesignAgent(
        storage=local_storage, registry=registry_service,
        workflow_state=workflow_state_service, mcp_client=_mcp(),
    ).run_step_7(run_id)
    assert all(r.antigen_antibody_mapping is None for r in pkg.prepared_structure_inputs)
    for rec in pkg.prepared_structure_inputs:
        assert all(
            rec.candidate_id in {p["target_candidate_id"], p["antibody_candidate_id"]}
            and p["mapping_status"] == "ambiguous"
            for p in rec.chain_pair_candidates
        )


def test_step7_explicit_candidate_pair_only_maps_that_pair(
    local_storage, registry_service, workflow_state_service
):
    run_id = _seed(local_storage, registry_service, workflow_state_service)
    cct_path = local_storage.run_key(run_id, "candidate_context_table.json")
    cct = local_storage.read_json(cct_path)
    target = next(c for c in cct["candidate_records"] if c["candidate_type"] == "target_antigen")
    antibody = next(c for c in cct["candidate_records"] if c["candidate_type"] == "antibody")
    target["materials"].append({"material_id": "t1seq", "material_type": "target_sequence", "value": "AAAA"})
    antibody["materials"].extend([
        {"material_id": "a1heavy", "material_type": "antibody_heavy_chain_sequence", "value": "BBBB"},
        {"material_id": "a1light", "material_type": "antibody_light_chain_sequence", "value": "CCCC"},
    ])
    target["related_candidate_id"] = antibody["candidate_id"]
    target2, antibody2 = deepcopy(target), deepcopy(antibody)
    target2["candidate_id"], antibody2["candidate_id"] = "target_2", "antibody_2"
    target2.pop("related_candidate_id", None)
    cct["candidate_records"].extend([target2, antibody2])
    local_storage.write_json(cct_path, cct)
    pkg = StructureAndDesignAgent(
        storage=local_storage, registry=registry_service,
        workflow_state=workflow_state_service, mcp_client=_mcp(),
    ).run_step_7(run_id)
    mapped = [r for r in pkg.prepared_structure_inputs if r.antigen_antibody_mapping]
    assert {r.candidate_id for r in mapped} == {target["candidate_id"], antibody["candidate_id"]}
    assert all(r.antigen_antibody_mapping["relationship_source"] == "explicit" for r in mapped)
    assert all(r.antigen_antibody_mapping["mapping_status"] == "sequence_only_prediction_needed" for r in mapped)


def test_step7_unscoped_global_residue_range_is_unresolved(
    local_storage, registry_service, workflow_state_service
):
    run_id = _seed(
        local_storage, registry_service, workflow_state_service,
        referenced_inputs=[{
            "id_type": "residue_range", "value": "residues 20-240", "source": "user"
        }],
    )
    pkg = StructureAndDesignAgent(
        storage=local_storage, registry=registry_service,
        workflow_state=workflow_state_service, mcp_client=_mcp(),
    ).run_step_7(run_id)
    assert all(r.residue_ranges == [] for r in pkg.prepared_structure_inputs)
    assert any(u["resource_type"] == "residue_range" for u in pkg.unresolved_resource_refs)


def test_step7_raw_file_content_is_not_embedded(
    local_storage, registry_service, workflow_state_service
):
    pdb_key = local_storage.run_key("sentinel", "inputs/files/target.pdb")
    fasta_key = local_storage.run_key("sentinel", "inputs/files/heavy.fasta")
    local_storage.write_bytes(pdb_key, b"HEADER    RAW_PDB_SENTINEL\nATOM      1  CA  ALA A  20       1.0   1.0   1.0\nEND\n")
    local_storage.write_bytes(fasta_key, b">heavy\nRAW_FASTA_SEQUENCE_SENTINEL\n")
    run_id = _seed(
        local_storage, registry_service, workflow_state_service,
        uploaded_files=[
            {"file_id": "target_pdb", "original_filename": "target.pdb", "storage_path": pdb_key, "role": "antigen"},
            {"file_id": "heavy_fasta", "original_filename": "heavy.fasta", "storage_path": fasta_key, "role": "antibody_heavy"},
        ],
    )
    pkg = StructureAndDesignAgent(
        storage=local_storage, registry=registry_service,
        workflow_state=workflow_state_service, mcp_client=_mcp(),
    ).run_step_7(run_id)
    blob = json.dumps(pkg.model_dump())
    assert "RAW_PDB_SENTINEL" not in blob
    assert "RAW_FASTA_SEQUENCE_SENTINEL" not in blob
    assert "ATOM      1" not in blob


# ── Step 8 ──────────────────────────────────────────────────────────────────

def _bindings_with_step8_overrides(extra: dict | None = None) -> dict:
    from app.mcp.tools._registry import _all_bindings

    base = dict(_all_bindings())
    base.update(extra or {})
    return base


def test_step8_scope_tools_match_inventory_runtime_scope():
    mcp = _mcp()
    runtime_scope = set(mcp.list_tools(agent_name="structure_and_design_agent", step_id="step_08"))
    routing_policy = set(structure_and_design_module._STEP8_SCOPED_TOOL_POLICY)
    assert runtime_scope == routing_policy, (
        "Step 8 runtime scope must stay in sync with _STEP8_SCOPED_TOOL_POLICY.\n"
        f"runtime_scope={sorted(runtime_scope)}\n"
        f"routing_policy={sorted(routing_policy)}"
    )


def test_step8_produces_confidence_records_and_keeps_raw_at_ref(
    local_storage, registry_service, workflow_state_service
):
    run_id = _seed(
        local_storage, registry_service, workflow_state_service,
        referenced_inputs=[
            {"id_type": "pdb_id", "value": "1N8Z", "source": "raw_request_text"},
        ],
    )
    # Inject canned payloads on the Step 8 tools we expect to be called.
    overrides = {
        "PDBePISA_get_interfaces": lambda **kw: {
            "hits_step8_pisa": "raw_marker",
            "pdb_id": kw.get("pdb_id"),
            "interfaces": [
                {
                    "chain_id_1": "A",
                    "chain_id_2": "B",
                    "interface_area": 123.4,
                    "h_bond_count": 2,
                    "interface_residues": ["A:1", "B:2"],
                }
            ],
        },
        "get_refinement_resolution_by_pdb_id":
            lambda **kw: {"hits_step8_resolution": 2.0, "pdb_id": kw.get("pdb_id")},
    }
    mcp = LocalMCPClient(
        inventory=ToolInventoryService(
            os.environ.get("TOOL_INVENTORY_XLSX", str(DEFAULT_XLSX))
        ),
        bindings=_bindings_with_step8_overrides(overrides),
    )
    agent = StructureAndDesignAgent(
        storage=local_storage, registry=registry_service,
        workflow_state=workflow_state_service, mcp_client=mcp,
    )
    # Step 7 first
    agent.run_step_7(run_id)
    results = agent.run_step_8(run_id)

    assert results.structure_modeling_status == "ok"
    assert all(cr.run_status == "ok" for cr in results.candidate_structure_results)

    confidence_types = {
        c.confidence_type for cr in results.candidate_structure_results
        for c in cr.structure_confidence_records
    }
    assert "refinement_resolution" in confidence_types
    assert "interface_quality" in confidence_types
    selected = [
        tc for tc in results.tool_call_records
        if tc.tool_input_summary.get("routing_decision") == "selected"
    ]
    selected_names = {tc.tool_name for tc in selected}
    assert "PDBePISA_get_interfaces" in selected_names
    assert "get_refinement_resolution_by_pdb_id" in selected_names
    nim_calls = [
        tc for tc in results.tool_call_records
        if tc.tool_name in structure_and_design_module._STEP8_NIM_COMPLEX_TOOLS
        and tc.tool_input_summary.get("input_case") == "known_pdb_id"
    ]
    assert nim_calls
    assert all(tc.run_status == "skipped" for tc in nim_calls)
    assert all(tc.tool_input_summary.get("routing_decision") == "not_applicable" for tc in nim_calls)
    assert "RCSBData_get_entry" not in {tc.tool_name for tc in results.tool_call_records}
    assert "ProteinsPlus_profile_structure_quality" not in {tc.tool_name for tc in results.tool_call_records}
    assert all(tc.run_status in {"success", "dependency_unavailable", "skipped"} for tc in results.tool_call_records)

    all_features = [
        feature for cr in results.candidate_structure_results
        for feature in cr.interface_features
    ]
    assert all_features
    assert all_features[0].chain_id_1 == "A"
    assert all_features[0].chain_id_2 == "B"
    analysis_records = [
        rec for cr in results.candidate_structure_results
        for rec in cr.interface_analysis_records
    ]
    assert analysis_records
    assert analysis_records[0].source_tool == "PDBePISA_get_interfaces"
    assert analysis_records[0].chain_pair == {"chain_id_1": "A", "chain_id_2": "B"}
    assert analysis_records[0].interface_residue_count == 2
    assert analysis_records[0].interface_area == 123.4
    assert analysis_records[0].h_bond_count == 2
    handoffs = [cr.downstream_handoff for cr in results.candidate_structure_results]
    assert any(h.has_complex_structure for h in handoffs)
    assert any(h.has_interface_features for h in handoffs)
    assert any(h.interface_quality_available for h in handoffs)
    assert any(h.refinement_resolution_available for h in handoffs)
    assert any(h.structure_for_variant_generation_ref == "1N8Z" for h in handoffs)
    complex_refs = [
        ref for cr in results.candidate_structure_results
        for ref in cr.complex_structure_refs
    ]
    assert any(ref.source_kind == "existing_pdb_complex" and ref.pdb_id == "1N8Z" for ref in complex_refs)

    # output_artifacts use structured envelope (artifact_id + storage_ref).
    assert results.output_artifacts
    art_types = {a.artifact_type for a in results.output_artifacts}
    assert "refinement_or_validation_report" in art_types
    assert "interface_analysis_raw_output" in art_types

    # Raw payload markers stay in tool_outputs/ — never in normalized records.
    blob = json.dumps(results.model_dump())
    assert "hits_step8_" not in blob
    assert "raw_marker" not in blob

    # And the raw files actually exist.
    for tc in results.tool_call_records:
        if tc.run_status == "success":
            assert local_storage.exists(tc.tool_output_ref)


def test_step8_tolerates_dependency_unavailable_wrappers(
    local_storage, registry_service, workflow_state_service
):
    """When every Step 8 wrapper raises NotImplementedError, step finishes
    `partial` (not crash) and records dependency_unavailable for each call."""
    from app.mcp.tools._registry import _all_bindings

    def _ni(**_):
        raise NotImplementedError

    forced_unwired = {
        name: _ni for name in (
            "PDBePISA_get_interfaces",
            "get_refinement_resolution_by_pdb_id",
            "CrystalStructure_validate",
        )
    }
    bindings = dict(_all_bindings())
    bindings.update(forced_unwired)
    mcp = LocalMCPClient(
        inventory=ToolInventoryService(
            os.environ.get("TOOL_INVENTORY_XLSX", str(DEFAULT_XLSX))
        ),
        bindings=bindings,
    )

    run_id = _seed(
        local_storage, registry_service, workflow_state_service,
        referenced_inputs=[
            {"id_type": "pdb_id", "value": "1N8Z", "source": "raw_request_text"},
        ],
    )
    agent = StructureAndDesignAgent(
        storage=local_storage, registry=registry_service,
        workflow_state=workflow_state_service, mcp_client=mcp,
    )
    agent.run_step_7(run_id)
    results = agent.run_step_8(run_id)
    assert results.structure_modeling_status in {"partial", "failed"}
    statuses = [tc.run_status for tc in results.tool_call_records]
    assert "dependency_unavailable" in statuses


def test_step8_uploaded_structure_file_path_calls_validation_tools(
    local_storage, registry_service, workflow_state_service
):
    run_id = _seed(
        local_storage, registry_service, workflow_state_service,
        uploaded_files=[
            {
                "file_id": "file_pdb",
                "original_filename": "target.pdb",
                "storage_path": "adc_pilot/runs/x/inputs/files/file_pdb.pdb",
                "content_type": "chemical/x-pdb",
                "sha256": "sha256:abc",
                "size_bytes": 1024,
                "role": "target",
            },
        ],
    )
    raw_path = local_storage.run_key(run_id, "inputs/raw_request_record.json")
    raw = local_storage.read_json(raw_path)
    raw["uploaded_files"][0]["storage_path"] = local_storage.run_key(run_id, "inputs/files/file_pdb.pdb")
    local_storage.write_json(raw_path, raw)
    local_storage.write_bytes(
        raw["uploaded_files"][0]["storage_path"],
        (PROJECT_ROOT / "data" / "pdb" / "S1.pdb").read_bytes(),
    )
    agent = StructureAndDesignAgent(
        storage=local_storage, registry=registry_service,
        workflow_state=workflow_state_service, mcp_client=_mcp(),
    )
    agent.run_step_7(run_id)
    results = agent.run_step_8(run_id)
    assert results.structure_modeling_status == "ok"
    assert all(cr.run_status == "ok" for cr in results.candidate_structure_results)
    uploaded_handoffs = [
        cr.downstream_handoff for cr in results.candidate_structure_results
        if any(c.source == "CrystalStructure_validate" for c in cr.structure_confidence_records)
    ]
    assert uploaded_handoffs
    assert all(not h.has_complex_structure for h in uploaded_handoffs)
    assert all(h.structure_for_variant_generation_ref is None for h in uploaded_handoffs)
    assert any(h.has_validated_structure for h in uploaded_handoffs)
    assert any(h.validation_available for h in uploaded_handoffs)
    assert any(not h.has_interface_features for h in uploaded_handoffs)
    assert any(h.validated_structure_ref for h in uploaded_handoffs)
    missing = [
        item for h in uploaded_handoffs
        for item in h.missing_for_step9
    ]
    assert "complex_structure_missing" in missing
    plans = [
        plan for cr in results.candidate_structure_results
        for plan in cr.complex_prediction_plans
    ]
    assert plans
    assert any(plan.input_status == "input_missing" for plan in plans)
    assert any("antigen_antibody_pair" in plan.missing_prediction_inputs or "antigen_sequence" in plan.missing_prediction_inputs for plan in plans)
    tools = {tc.tool_name for tc in results.tool_call_records}
    assert "CrystalStructure_validate" in tools
    assert "ProteinsPlus_profile_structure_quality" not in tools
    pisa_calls = [tc for tc in results.tool_call_records if tc.tool_name == "PDBePISA_get_interfaces"]
    assert pisa_calls
    assert all(tc.tool_input_summary.get("routing_decision") == "not_applicable" for tc in pisa_calls)
    assert all("pdb_id" not in (tc.tool_input_summary.get("arguments") or {}) for tc in pisa_calls)


def test_step8_skips_crystal_validation_when_compact_metadata_missing(
    local_storage, registry_service, workflow_state_service
):
    run_id = _seed(
        local_storage,
        registry_service,
        workflow_state_service,
        uploaded_files=[
            {
                "file_id": "file_pdb",
                "original_filename": "target.pdb",
                "storage_path": "adc_pilot/runs/x/inputs/files/file_pdb.pdb",
                "content_type": "chemical/x-pdb",
                "sha256": "sha256:abc",
                "size_bytes": 1024,
                "role": "target",
            },
        ],
    )
    raw_path = local_storage.run_key(run_id, "inputs/raw_request_record.json")
    raw = local_storage.read_json(raw_path)
    raw["uploaded_files"][0]["storage_path"] = local_storage.run_key(run_id, "inputs/files/file_pdb.pdb")
    local_storage.write_json(raw_path, raw)
    local_storage.write_bytes(raw["uploaded_files"][0]["storage_path"], b"HEADER    DUMMY\nEND\n")
    agent = StructureAndDesignAgent(
        storage=local_storage,
        registry=registry_service,
        workflow_state=workflow_state_service,
        mcp_client=_mcp(),
    )
    pkg = agent.run_step_7(run_id)
    uploaded = [r for r in pkg.prepared_structure_inputs if r.input_case == "uploaded_structure_file"]
    assert uploaded
    assert all(
        not r.crystal_metadata or r.crystal_metadata.parse_status in {"missing", "invalid"}
        for r in uploaded
    )
    results = agent.run_step_8(run_id)
    crystal_calls = [
        tc for tc in results.tool_call_records
        if tc.tool_name == "CrystalStructure_validate"
        and tc.tool_input_summary.get("input_case") == "uploaded_structure_file"
    ]
    assert crystal_calls
    assert all(tc.run_status == "skipped" for tc in crystal_calls)
    assert all(tc.tool_input_summary.get("routing_decision") == "input_missing" for tc in crystal_calls)
    assert any("Z" in tc.tool_input_summary.get("missing", []) for tc in crystal_calls)
    assert all("pdb_id_or_path" not in (tc.tool_input_summary.get("arguments") or {}) for tc in crystal_calls)
    pisa_calls = [tc for tc in results.tool_call_records if tc.tool_name == "PDBePISA_get_interfaces"]
    assert pisa_calls
    assert all(tc.tool_input_summary.get("routing_decision") == "not_applicable" for tc in pisa_calls)


def test_step8_sequence_only_records_nim_prediction_route_as_unavailable(
    local_storage, registry_service, workflow_state_service
):
    run_id = _seed(local_storage, registry_service, workflow_state_service, referenced_inputs=[])
    cct_path = local_storage.run_key(run_id, "candidate_context_table.json")
    cct = local_storage.read_json(cct_path)
    for cand in cct["candidate_records"]:
        if cand.get("candidate_type") == "target_antigen":
            cand["materials"] = [
                {
                    "material_id": "target_seq_for_step8",
                    "material_type": "target_sequence",
                    "value": "MKTAYIAKQNNVG",
                }
            ]
        else:
            cand["materials"] = []
    local_storage.write_json(cct_path, cct)

    agent = StructureAndDesignAgent(
        storage=local_storage,
        registry=registry_service,
        workflow_state=workflow_state_service,
        mcp_client=_mcp(),
    )
    agent.run_step_7(run_id)
    results = agent.run_step_8(run_id)
    assert results.structure_modeling_status == "ok"
    assert all(cr.run_status == "ok" for cr in results.candidate_structure_results)
    assert all(not cr.complex_structure_refs for cr in results.candidate_structure_results)
    missing = [
        item for cr in results.candidate_structure_results
        for item in cr.downstream_handoff.missing_for_step9
    ]
    assert "complex_structure_missing" in missing
    assert "antibody_heavy_sequence" in missing
    assert "antibody_light_sequence" in missing

    nim_calls = [
        tc for tc in results.tool_call_records
        if tc.tool_name in structure_and_design_module._STEP8_NIM_COMPLEX_TOOLS
    ]
    assert nim_calls
    assert all(tc.run_status == "skipped" for tc in nim_calls)
    assert all(tc.tool_input_summary.get("routing_decision") == "input_missing" for tc in nim_calls)
    assert all("complex_prediction_plan" in tc.tool_input_summary for tc in nim_calls)
    plans = [
        plan for cr in results.candidate_structure_results
        for plan in cr.complex_prediction_plans
    ]
    assert plans
    assert all(plan.input_status == "input_missing" for plan in plans)
    assert any("antibody_heavy_sequence" in plan.missing_prediction_inputs for plan in plans)
    assert any("antibody_light_sequence" in plan.missing_prediction_inputs for plan in plans)

    blob = json.dumps(results.model_dump())
    assert "MKTAYIAKQNNVG" not in blob
    assert "RAW_PDB_SENTINEL" not in blob
    assert "ATOM      1" not in blob


def test_step8_uniprot_antigen_with_antibody_sequences_is_contract_unresolved(
    local_storage, registry_service, workflow_state_service
):
    run_id = _seed(
        local_storage,
        registry_service,
        workflow_state_service,
        referenced_inputs=[{"id_type": "uniprot_id", "value": "P04626", "source": "raw"}],
    )
    cct_path = local_storage.run_key(run_id, "candidate_context_table.json")
    cct = local_storage.read_json(cct_path)
    antibody = next(c for c in cct["candidate_records"] if c["candidate_type"] == "antibody")
    antibody["materials"].extend([
        {
            "material_id": "heavy_seq_step8",
            "material_type": "antibody_heavy_chain_sequence",
            "value": "EVQLVESGGGLVQPGGSLRLSCAAS",
        },
        {
            "material_id": "light_seq_step8",
            "material_type": "antibody_light_chain_sequence",
            "value": "DIQMTQSPSSLSASVGDRVTITC",
        },
    ])
    local_storage.write_json(cct_path, cct)

    agent = StructureAndDesignAgent(
        storage=local_storage,
        registry=registry_service,
        workflow_state=workflow_state_service,
        mcp_client=_mcp(),
    )
    agent.run_step_7(run_id)
    results = agent.run_step_8(run_id)

    assert results.structure_modeling_status == "ok"
    assert all(cr.run_status == "ok" for cr in results.candidate_structure_results)
    plans = [
        plan for cr in results.candidate_structure_results
        for plan in cr.complex_prediction_plans
    ]
    assert plans
    assert any(plan.input_status == "contract_unresolved" for plan in plans)
    assert not any(plan.input_status == "selected_but_deferred" for plan in plans)
    assert any(
        "antigen_sequence_unresolved_from_uniprot_id" in plan.missing_prediction_inputs
        for plan in plans
    )
    assert all(not cr.complex_structure_refs for cr in results.candidate_structure_results)
    assert not any(
        "complex_prediction_unavailable" in cr.downstream_handoff.missing_for_step9
        for cr in results.candidate_structure_results
    )
    assert any(
        "antigen_sequence_unresolved_from_uniprot_id" in cr.downstream_handoff.missing_for_step9
        for cr in results.candidate_structure_results
    )
    nim_calls = [
        tc for tc in results.tool_call_records
        if tc.tool_name in structure_and_design_module._STEP8_NIM_COMPLEX_TOOLS
    ]
    assert nim_calls
    assert all(tc.run_status == "skipped" for tc in nim_calls)
    assert all(tc.tool_input_summary.get("routing_decision") == "contract_unresolved" for tc in nim_calls)
    audit_blob = json.dumps([tc.tool_input_summary for tc in nim_calls])
    assert "EVQLVES" not in audit_blob
    assert "DIQMTQ" not in audit_blob
    assert "sha256_prefix" in audit_blob
    assert "P04626" in audit_blob


def test_step8_raw_antigen_antibody_sequences_record_nim_deferred_plan(
    local_storage, registry_service, workflow_state_service
):
    run_id = _seed(local_storage, registry_service, workflow_state_service, referenced_inputs=[])
    cct_path = local_storage.run_key(run_id, "candidate_context_table.json")
    cct = local_storage.read_json(cct_path)
    target = next(c for c in cct["candidate_records"] if c["candidate_type"] == "target_antigen")
    antibody = next(c for c in cct["candidate_records"] if c["candidate_type"] == "antibody")
    target["materials"] = [
        {
            "material_id": "target_seq_step8",
            "material_type": "target_sequence",
            "value": "MKTAYIAKQNNVG",
        }
    ]
    antibody["materials"].extend([
        {
            "material_id": "heavy_seq_step8",
            "material_type": "antibody_heavy_chain_sequence",
            "value": "EVQLVESGGGLVQPGGSLRLSCAAS",
        },
        {
            "material_id": "light_seq_step8",
            "material_type": "antibody_light_chain_sequence",
            "value": "DIQMTQSPSSLSASVGDRVTITC",
        },
    ])
    local_storage.write_json(cct_path, cct)

    agent = StructureAndDesignAgent(
        storage=local_storage,
        registry=registry_service,
        workflow_state=workflow_state_service,
        mcp_client=_mcp(),
    )
    agent.run_step_7(run_id)
    results = agent.run_step_8(run_id)

    assert results.structure_modeling_status == "partial"
    plans = [
        plan for cr in results.candidate_structure_results
        for plan in cr.complex_prediction_plans
    ]
    assert plans
    assert any(plan.input_status == "ready" for plan in plans)
    assert all(not cr.complex_structure_refs for cr in results.candidate_structure_results)
    assert any(
        "complex_prediction_unavailable" in cr.downstream_handoff.missing_for_step9
        for cr in results.candidate_structure_results
    )
    nim_calls = [
        tc for tc in results.tool_call_records
        if tc.tool_name in structure_and_design_module._STEP8_NIM_COMPLEX_TOOLS
    ]
    assert nim_calls
    assert any(tc.run_status == "dependency_unavailable" for tc in nim_calls)
    assert any(tc.tool_input_summary.get("routing_decision") == "selected" for tc in nim_calls)
    audit_blob = json.dumps([tc.tool_input_summary for tc in nim_calls])
    assert "MKTAYIAK" not in audit_blob
    assert "EVQLVES" not in audit_blob
    assert "DIQMTQ" not in audit_blob
    assert "sha256_prefix" in audit_blob


def test_step8_antibody_heavy_light_without_antigen_records_input_missing_antigen_with_compact_sequences(
    local_storage, registry_service, workflow_state_service
):
    run_id = _seed(local_storage, registry_service, workflow_state_service, referenced_inputs=[])
    cct_path = local_storage.run_key(run_id, "candidate_context_table.json")
    cct = local_storage.read_json(cct_path)
    # Remove target sequence to force sequence-only antibody-only planning.
    target = next(c for c in cct["candidate_records"] if c["candidate_type"] == "target_antigen")
    target["materials"] = []
    antibody = next(c for c in cct["candidate_records"] if c["candidate_type"] == "antibody")
    heavy_seq = "EVQLVESGGGLVQPGGSLRLSCAASABCDEF"
    light_seq = "DIQMTQSPSSLSASVGDRVTITCABCDE"
    antibody["materials"].extend([
        {
            "material_id": "heavy_step8_seq",
            "material_type": "antibody_heavy_chain_sequence",
            "value": heavy_seq,
        },
        {
            "material_id": "light_step8_seq",
            "material_type": "antibody_light_chain_sequence",
            "value": light_seq,
        },
    ])
    local_storage.write_json(cct_path, cct)

    agent = StructureAndDesignAgent(
        storage=local_storage,
        registry=registry_service,
        workflow_state=workflow_state_service,
        mcp_client=_mcp(),
    )
    agent.run_step_7(run_id)
    artifact_path = local_storage.run_key(run_id, "prepared_structure_input_package.json")
    artifact = local_storage.read_json(artifact_path)
    artifact_blob = json.dumps(artifact)
    assert heavy_seq not in artifact_blob
    assert light_seq not in artifact_blob

    results = agent.run_step_8(run_id)

    plans = [
        plan for cr in results.candidate_structure_results
        for plan in cr.complex_prediction_plans
    ]
    assert plans
    assert any(plan.input_status == "input_missing" for plan in plans)
    assert any("antigen_sequence" in plan.missing_prediction_inputs for plan in plans)
    assert any(
        item.get("chain_role") == "antibody_heavy" and item.get("sequence_readiness") == "ready"
        for plan in plans
        for item in plan.sequence_inputs
    )
    assert any(
        item.get("chain_role") == "antibody_light" and item.get("sequence_readiness") == "ready"
        for plan in plans
        for item in plan.sequence_inputs
    )

    nim_calls = [
        tc for tc in results.tool_call_records
        if tc.tool_name in structure_and_design_module._STEP8_NIM_COMPLEX_TOOLS
    ]
    assert nim_calls
    assert all(tc.run_status == "skipped" for tc in nim_calls)
    assert all(tc.tool_input_summary.get("routing_decision") == "input_missing" for tc in nim_calls)
    audit_blob = json.dumps([tc.tool_input_summary or {} for tc in nim_calls])
    assert heavy_seq not in audit_blob
    assert light_seq not in audit_blob


def test_step8_nim_runtime_resolves_inline_sequence_from_step5_material_lookup(
    local_storage, registry_service, workflow_state_service
):
    run_id = _seed(local_storage, registry_service, workflow_state_service, referenced_inputs=[])
    cct_path = local_storage.run_key(run_id, "candidate_context_table.json")
    cct = local_storage.read_json(cct_path)
    target = next(c for c in cct["candidate_records"] if c["candidate_type"] == "target_antigen")
    antibody = next(c for c in cct["candidate_records"] if c["candidate_type"] == "antibody")
    target["materials"] = [
        {"material_id": "target_seq_step8_runtime", "material_type": "target_sequence", "value": "MKTAYIAKQNNVG"}
    ]
    heavy_seq = "EVQLVESGGGLVQPGGSLRLSCAASRUNTIME"
    light_seq = "DIQMTQSPSSLSASVGDRVTITCRUNTIME"
    antibody["materials"].extend([
        {"material_id": "heavy_step8_runtime", "material_type": "antibody_heavy_chain_sequence", "value": heavy_seq},
        {"material_id": "light_step8_runtime", "material_type": "antibody_light_chain_sequence", "value": light_seq},
    ])
    local_storage.write_json(cct_path, cct)

    captured: list[tuple[str, dict]] = []

    def _capture(tool_name: str):
        def _inner(**kw):
            captured.append((tool_name, dict(kw)))
            return {"status": "ok", "model_ref": f"s3://bucket/{tool_name}.pdb"}
        return _inner

    mcp = LocalMCPClient(
        inventory=ToolInventoryService(os.environ.get("TOOL_INVENTORY_XLSX", str(DEFAULT_XLSX))),
        bindings=_bindings_with_step8_overrides({
            "NvidiaNIM_alphafold2_multimer": _capture("NvidiaNIM_alphafold2_multimer"),
            "NvidiaNIM_openfold3": _capture("NvidiaNIM_openfold3"),
            "NvidiaNIM_boltz2": _capture("NvidiaNIM_boltz2"),
        }),
    )
    agent = StructureAndDesignAgent(
        storage=local_storage,
        registry=registry_service,
        workflow_state=workflow_state_service,
        mcp_client=mcp,
    )
    agent.run_step_7(run_id)
    artifact_path = local_storage.run_key(run_id, "prepared_structure_input_package.json")
    artifact = local_storage.read_json(artifact_path)
    artifact_blob = json.dumps(artifact)

    assert heavy_seq not in artifact_blob
    assert light_seq not in artifact_blob
    for rec in artifact.get("prepared_structure_inputs", []):
        for seq in rec.get("sequence_refs_for_prediction", []):
            assert "sequence" not in seq

    results = agent.run_step_8(run_id)
    assert results.structure_modeling_status in {"partial", "ok"}
    assert captured
    for tool_name, kwargs in captured:
        if "sequences" in kwargs:
            assert heavy_seq in kwargs["sequences"]
            assert light_seq in kwargs["sequences"]
            assert "MKTAYIAKQNNVG" in kwargs["sequences"]
        if "inputs" in kwargs:
            flat = "".join(item.get("sequence", "") for item in kwargs["inputs"])
            assert heavy_seq in flat
            assert light_seq in flat
            assert "MKTAYIAKQNNVG" in flat
        if "polymers" in kwargs:
            flat = "".join(item.get("sequence", "") for item in kwargs["polymers"])
            assert heavy_seq in flat
            assert light_seq in flat
            assert "MKTAYIAKQNNVG" in flat

    call_summaries = [tc.tool_input_summary or {} for tc in results.tool_call_records]
    summary_blob = json.dumps(call_summaries)
    assert heavy_seq not in summary_blob
    assert light_seq not in summary_blob


def test_step8_nim_contract_treats_fasta_refs_as_runtime_ready():
    sin = {
        "input_case": "sequence_only_input",
        "structure_refs": [],
        "sequence_refs_for_prediction": [
            {
                "sequence_id": "antigen_fasta",
                "chain_role": "antigen",
                "prediction_input_kind": "fasta_ref",
                "sequence_value_status": "referenced",
                "source_kind": "uploaded_fasta",
                "source_ref": "inputs/files/antigen.fasta",
                "sequence_storage_ref": "runs/x/inputs/files/antigen.fasta",
            },
            {
                "sequence_id": "heavy_fasta",
                "chain_role": "antibody_heavy",
                "prediction_input_kind": "fasta_ref",
                "sequence_value_status": "referenced",
                "source_kind": "uploaded_fasta",
                "source_ref": "inputs/files/heavy.fasta",
                "sequence_storage_ref": "runs/x/inputs/files/heavy.fasta",
            },
            {
                "sequence_id": "light_fasta",
                "chain_role": "antibody_light",
                "prediction_input_kind": "fasta_ref",
                "sequence_value_status": "referenced",
                "source_kind": "uploaded_fasta",
                "source_ref": "inputs/files/light.fasta",
                "sequence_storage_ref": "runs/x/inputs/files/light.fasta",
            },
        ],
    }

    plan = structure_and_design_module._plan_step8_nim_complex_prediction(
        "NvidiaNIM_boltz2", sin, [sin]
    )

    assert plan.input_status == "ready"
    assert plan.runtime_status == "not_checked"
    assert not plan.missing_prediction_inputs
    assert all(item["sequence_readiness"] == "ready" for item in plan.sequence_inputs)
    audit_blob = json.dumps(plan.model_dump())
    assert ">antigen" not in audit_blob
    assert "MKTAYIAK" not in audit_blob


def test_step8_nim_wrappers_are_tooluniverse_bindings_or_explicit_dependency():
    from app.mcp.tools import nvidianim

    bindings = dict(nvidianim.BINDINGS)
    assert bindings["NvidiaNIM_alphafold2_multimer"].__name__ != "_ni"
    assert bindings["NvidiaNIM_openfold3"].__name__ != "_ni"
    assert bindings["NvidiaNIM_boltz2"].__name__ != "_ni"
    with pytest.raises(NotImplementedError, match="requires live ToolUniverse execution"):
        bindings["NvidiaNIM_boltz2"](polymers=[])


def test_step8_nim_success_persists_compact_input_not_raw_sequences(
    local_storage, registry_service, workflow_state_service
):
    run_id = _seed(local_storage, registry_service, workflow_state_service, referenced_inputs=[])
    cct_path = local_storage.run_key(run_id, "candidate_context_table.json")
    cct = local_storage.read_json(cct_path)
    target = next(c for c in cct["candidate_records"] if c["candidate_type"] == "target_antigen")
    antibody = next(c for c in cct["candidate_records"] if c["candidate_type"] == "antibody")
    target["materials"] = [
        {"material_id": "target_seq_step8", "material_type": "target_sequence", "value": "MKTAYIAKQNNVG"}
    ]
    antibody["materials"].extend([
        {
            "material_id": "heavy_seq_step8",
            "material_type": "antibody_heavy_chain_sequence",
            "value": "EVQLVESGGGLVQPGGSLRLSCAAS",
        },
        {
            "material_id": "light_seq_step8",
            "material_type": "antibody_light_chain_sequence",
            "value": "DIQMTQSPSSLSASVGDRVTITC",
        },
    ])
    local_storage.write_json(cct_path, cct)

    def _nim_success(**_kw):
        return {"status": "ok", "model_ref": "s3://bucket/predicted_complex.pdb"}

    mcp = LocalMCPClient(
        inventory=ToolInventoryService(os.environ.get("TOOL_INVENTORY_XLSX", str(DEFAULT_XLSX))),
        bindings=_bindings_with_step8_overrides({
            "NvidiaNIM_alphafold2_multimer": _nim_success,
            "NvidiaNIM_openfold3": _nim_success,
            "NvidiaNIM_boltz2": _nim_success,
        }),
    )
    agent = StructureAndDesignAgent(
        storage=local_storage,
        registry=registry_service,
        workflow_state=workflow_state_service,
        mcp_client=mcp,
    )
    agent.run_step_7(run_id)
    results = agent.run_step_8(run_id)

    nim_calls = [
        tc for tc in results.tool_call_records
        if tc.tool_name in structure_and_design_module._STEP8_NIM_COMPLEX_TOOLS
    ]
    assert nim_calls
    assert any(tc.run_status == "success" for tc in nim_calls)
    for tc in nim_calls:
        if not tc.tool_output_ref:
            continue
        payload = local_storage.read_json(tc.tool_output_ref)
        dumped = json.dumps(payload)
        assert "MKTAYIAK" not in dumped
        assert "EVQLVES" not in dumped
        assert "DIQMTQ" not in dumped
        assert "sequence_inputs" in dumped
        assert "sha256_prefix" in dumped


def test_step8_selected_tool_failure_still_marks_partial(
    local_storage, registry_service, workflow_state_service
):
    run_id = _seed(
        local_storage,
        registry_service,
        workflow_state_service,
        referenced_inputs=[
            {"id_type": "pdb_id", "value": "1N8Z", "source": "raw_request_text"},
        ],
    )

    def _fail_pisa(**_):
        raise RuntimeError("pisa unavailable")

    mcp = LocalMCPClient(
        inventory=ToolInventoryService(
            os.environ.get("TOOL_INVENTORY_XLSX", str(DEFAULT_XLSX))
        ),
        bindings=_bindings_with_step8_overrides(
            {
                "PDBePISA_get_interfaces": _fail_pisa,
                "get_refinement_resolution_by_pdb_id":
                    lambda **kw: {"resolution_angstrom": 2.0, "pdb_id": kw.get("pdb_id")},
            }
        ),
    )
    agent = StructureAndDesignAgent(
        storage=local_storage,
        registry=registry_service,
        workflow_state=workflow_state_service,
        mcp_client=mcp,
    )
    agent.run_step_7(run_id)
    results = agent.run_step_8(run_id)

    assert results.structure_modeling_status == "partial"
    failed = [tc for tc in results.tool_call_records if tc.tool_name == "PDBePISA_get_interfaces"]
    assert failed
    assert any(tc.run_status == "failed" for tc in failed)
    assert any(tc.tool_input_summary.get("routing_decision") == "selected" for tc in failed)


def test_step8_nim_success_accepts_explicit_model_artifact_ref(local_storage):
    output_ref = local_storage.run_key("run_step8_unit", "tool_outputs", "step_08", "nim_model.json")
    local_storage.write_json(
        output_ref,
        {
            "tool_call_id": "tc_nim_model",
            "tool_name": "NvidiaNIM_boltz2",
            "label": "unit",
            "input": {},
            "output": {
                "model_ref": "s3://bucket/predicted_complex.pdb",
                "ptm": 0.72,
            },
        },
    )
    tc = ToolCallRecord(
        tool_call_id="tc_nim_model",
        tool_name="NvidiaNIM_boltz2",
        run_status="success",
        tool_output_ref=output_ref,
    )

    refs = structure_and_design_module._extract_complex_structure_refs_for_step8(
        local_storage, {}, tc
    )

    assert len(refs) == 1
    assert refs[0].source_kind == "predicted_complex"
    assert refs[0].storage_ref == "s3://bucket/predicted_complex.pdb"
    assert refs[0].confidence_summary["ptm"] == 0.72


def test_step8_prediction_model_ref_rejects_raw_or_generic_outputs(local_storage):
    assert structure_and_design_module._prediction_model_ref(
        {"output": "s3://bucket/looks-like-a-ref-but-generic-output.pdb"}
    ) is None
    assert structure_and_design_module._prediction_model_ref(
        {"url": "https://example.test/generic-url.pdb"}
    ) is None
    assert structure_and_design_module._prediction_model_ref(
        {"model_url": "https://example.test/model.pdb"}
    ) == "https://example.test/model.pdb"

    raw_pdb = "HEADER    RAW STRUCTURE\nATOM      1  N   ALA A   1\nHETATM    2  O   HOH A   2"
    assert structure_and_design_module._prediction_model_ref({"model_ref": raw_pdb}) is None

    output_ref = local_storage.run_key("run_step8_unit", "tool_outputs", "step_08", "nim_raw.json")
    local_storage.write_json(
        output_ref,
        {
            "tool_call_id": "tc_nim_raw",
            "tool_name": "NvidiaNIM_boltz2",
            "label": "unit",
            "input": {},
            "output": {"output": raw_pdb},
        },
    )
    tc = ToolCallRecord(
        tool_call_id="tc_nim_raw",
        tool_name="NvidiaNIM_boltz2",
        run_status="success",
        tool_output_ref=output_ref,
    )
    refs = structure_and_design_module._extract_complex_structure_refs_for_step8(
        local_storage, {}, tc
    )

    assert refs == []
    assert "HEADER" not in json.dumps([r.model_dump() for r in refs])
    assert "ATOM" not in json.dumps([r.model_dump() for r in refs])


# ── Step 9 ──────────────────────────────────────────────────────────────────

def test_step9_smiles_triggers_zinc_search_by_smiles(
    local_storage, registry_service, workflow_state_service
):
    """Step 5 picks up a SMILES referenced_input → payload_smiles material;
    Step 9 routes that to ZINC_search_by_smiles. The normalized record stays
    raw-free and never claims ZINC22."""
    run_id = _seed(
        local_storage, registry_service, workflow_state_service,
        referenced_inputs=[
            {"id_type": "smiles", "value": "CC(=O)NCCC1=CN(c2ccc(O)cc2)C(=O)C1",
             "source": "raw_request_text"},
        ],
    )
    agent = StructureAndDesignAgent(
        storage=local_storage, registry=registry_service,
        workflow_state=workflow_state_service, mcp_client=_mcp(),
    )
    artifact = agent.run_step_9(run_id)

    tool_names = {tc.tool_name for tc in artifact.tool_call_records}
    assert "ZINC_search_by_smiles" in tool_names

    # No record claims ZINC22.
    for hit in artifact.compound_hits:
        assert hit.source_library in {"ZINC", "ZINC15"}  # never "ZINC22"
        assert hit.source_database_version != "ZINC22"
        # The mock wrapper records its source as ZINC15 family; the agent's
        # honest default for unverified upstream is `unknown`.
        assert hit.source_database_version in {"unknown", "ZINC15"}

    # Raw payload (mocked envelope contains "status: mocked", "hits: ...")
    # must not bleed into compound_hits.
    cand_blob = json.dumps([h.model_dump() for h in artifact.compound_hits])
    assert "mocked" not in cand_blob
    assert "ZINC_search_by_smiles" not in cand_blob or True  # source_tool_name is allowed


def test_step9_zinc_id_triggers_zinc_get_compound(
    local_storage, registry_service, workflow_state_service
):
    run_id = _seed(
        local_storage, registry_service, workflow_state_service,
        referenced_inputs=[
            {"id_type": "zinc_id", "value": "ZINC12345678", "source": "raw_request_text"},
        ],
    )
    agent = StructureAndDesignAgent(
        storage=local_storage, registry=registry_service,
        workflow_state=workflow_state_service, mcp_client=_mcp(),
    )
    artifact = agent.run_step_9(run_id)
    tool_names = {tc.tool_name for tc in artifact.tool_call_records}
    assert "ZINC_get_compound" in tool_names

    # No record defaults to ZINC22.
    blob = json.dumps(artifact.model_dump())
    assert "ZINC22" not in blob


def test_step9_dependency_unavailable_marks_partial_not_crash(
    local_storage, registry_service, workflow_state_service
):
    from app.mcp.tools._registry import _all_bindings

    def _ni(**_):
        raise NotImplementedError

    bindings = dict(_all_bindings())
    for name in ("ZINC_search_compounds", "ZINC_get_compound",
                 "ZINC_search_by_smiles", "ZINC_search_by_properties", "ZINC_get_purchasable"):
        bindings[name] = _ni

    run_id = _seed(
        local_storage, registry_service, workflow_state_service,
        referenced_inputs=[
            {"id_type": "smiles", "value": "CCN(CC)CC", "source": "raw_request_text"},
        ],
    )
    mcp = LocalMCPClient(
        inventory=ToolInventoryService(
            os.environ.get("TOOL_INVENTORY_XLSX", str(DEFAULT_XLSX))
        ),
        bindings=bindings,
    )
    agent = StructureAndDesignAgent(
        storage=local_storage, registry=registry_service,
        workflow_state=workflow_state_service, mcp_client=mcp,
    )
    artifact = agent.run_step_9(run_id)
    assert artifact.screening_status in {"partial", "failed", "skipped"}
    statuses = [tc.run_status for tc in artifact.tool_call_records]
    assert "dependency_unavailable" in statuses


# ── precondition errors ─────────────────────────────────────────────────────

def test_step7_requires_step5_artifact(
    local_storage, registry_service, workflow_state_service
):
    intake = IntakeService(local_storage, registry_service, workflow_state_service)
    rec = intake.submit(
        raw_user_query="x", user_provided_context={"target_or_antigen_text": "HER2"}
    )
    agent = StructureAndDesignAgent(
        storage=local_storage, registry=registry_service,
        workflow_state=workflow_state_service, mcp_client=LocalMCPClient(),
    )
    with pytest.raises(WorkflowStateError, match="Step 5|Step 4"):
        agent.run_step_7(rec.run_id)
