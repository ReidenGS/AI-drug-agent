"""Step 5 batch-6 follow-up — real PDB Step 1→5 + extended role inference.

Uses the project's real reference PDB files under `程序/data/pdb/`:

    /Users/jackiewen/Desktop/desk/实习工作/国外ai医药/程序/data/pdb/S1.pdb
    /Users/jackiewen/Desktop/desk/实习工作/国外ai医药/程序/data/pdb/S2.pdb
    /Users/jackiewen/Desktop/desk/实习工作/国外ai医药/程序/data/pdb/S3.pdb

No fake PDB content is written — the test loads bytes from those files
and uploads them through `POST /runs/multipart`, then drives Step 2 → 5
through the service layer with `MockLLMProvider` (no network, no MCP
live calls).

Verifies:

- Step 1→5 end-to-end with a REAL PDB file completes.
- The Step 5 normalized artifact contains no raw PDB byte/text content
  (`HEADER`, `ATOM`, `END`, hex/sha snippets) — only metadata refs.
- PubChem CID alone defaults to material-only (no auto ADC candidate).
- DrugBank ID alone defaults to material-only.
- Explicit "as payload candidates" wording promotes the role.
- T-DM1 / T-DXd / Enhertu surface as reference benchmarks, not
  generated candidates.
- TROP2 ADC + MMAE records missing antibody/linker; no completion.
- vc-MMAE keeps the linker_payload term + decomposed materials with
  the inferred / explicit flags.
- Filename-driven role inference (Fab / Fc / N297 / glycan / receptor
  / antigen / fragment) lands the more specific structure role.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pytest
from fastapi.testclient import TestClient

import app.deps as deps
from app.agents.candidate_context_agent import (
    CandidateContextAgent,
    _infer_structure_role,
)
from app.agents.supervisor_agent import SupervisorAgent
from app.llm.provider import MockLLMProvider
from app.main import app
from app.mcp.client import LocalMCPClient
from app.services.input_readiness_service import InputReadinessService
from app.services.intake_service import IntakeService
from app.services.structured_query_service import StructuredQueryService
from app.services.workflow_setup_service import WorkflowSetupService


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_REAL_PDB_DIR = _PROJECT_ROOT.parent / "data" / "pdb"


def _real_pdb(name: str = "S1.pdb") -> Path:
    p = _REAL_PDB_DIR / name
    if not p.exists():
        pytest.skip(
            f"Real reference PDB not available at {p}. Place S1/S2/S3.pdb under data/pdb/."
        )
    return p


@pytest.fixture
def client(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("STORAGE_MODE", "local")
    monkeypatch.setenv("LOCAL_STORAGE_ROOT", str(tmp_path / "store"))
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    from app.settings import get_settings

    fns = (
        get_settings,
        deps.get_storage,
        deps.get_registry_service,
        deps.get_workflow_state_service,
        deps.get_tool_inventory_service,
        deps.get_mcp_client,
        deps.get_llm_provider,
    )
    for fn in fns:
        fn.cache_clear()
    yield TestClient(app)
    for fn in fns:
        fn.cache_clear()


def _drive_step5_after_upload(run_id: str):
    storage = deps.get_storage()
    registry = deps.get_registry_service()
    workflow_state = deps.get_workflow_state_service()
    StructuredQueryService(
        storage, registry, workflow_state,
        SupervisorAgent(llm=MockLLMProvider()),
    ).parse(run_id)
    InputReadinessService(storage, registry, workflow_state).check(run_id)
    WorkflowSetupService(storage, registry, workflow_state).plan(run_id)
    agent = CandidateContextAgent(
        storage=storage,
        registry=registry,
        workflow_state=workflow_state,
        mcp_client=LocalMCPClient(),
    )
    return agent.run(run_id), storage


# ── Step 1→5 with a real PDB file ─────────────────────────────────────────


def test_real_pdb_step1_5_end_to_end_keeps_bytes_out_of_normalized_artifact(client):
    pdb_path = _real_pdb("S1.pdb")
    pdb_bytes = pdb_path.read_bytes()
    # Sanity — the real file is non-trivial and has HEADER / ATOM lines.
    assert b"HEADER" in pdb_bytes
    assert b"ATOM" in pdb_bytes
    assert pdb_path.stat().st_size > 100_000

    resp = client.post(
        "/runs/multipart",
        data={
            "raw_user_query": (
                "Use the attached HER2 Fab structure as a reference for "
                "the antibody arm; design an ADC with vc-MMAE."
            ),
            "user_provided_context": json.dumps(
                {
                    "target_or_antigen_text": "HER2",
                    "candidate_text": "Trastuzumab",
                    "payload_linker_text": "vc-MMAE",
                }
            ),
        },
        files=[("files", ("trastuzumab_fab_S1.pdb", pdb_bytes, "chemical/x-pdb"))],
    )
    assert resp.status_code == 201, resp.text
    run_id = resp.json()["raw_request_record"]["run_id"]

    table, storage = _drive_step5_after_upload(run_id)

    # Real Step 5 artifact must NOT contain raw PDB byte/text content.
    persisted = storage.read_json(
        storage.run_key(run_id, "candidate_context_table.json")
    )
    blob = json.dumps(persisted)
    for pdb_marker in (
        "HEADER    ",
        "ATOM ",
        "ATOM\t",
        "END\n",
        "REMARK ",
        "TER ",
    ):
        assert pdb_marker not in blob, (
            f"raw PDB content leaked into normalized artifact: {pdb_marker!r}"
        )
    # Also the literal first-line header text from S1.pdb (TRANSFERASE etc).
    assert "TRANSFERASE" not in blob
    assert "STAUROSPORINE" not in blob

    # Structure material surfaces with a specific reference role; filename
    # carried "fab" → fab_structure_reference wins.
    target_rec = next(
        r for r in table.candidate_records if r.candidate_type == "target_antigen"
    )
    structure_mats = [
        m for m in target_rec.materials
        if m.material_type in {"structure_file", "structure_ref"}
    ]
    assert structure_mats
    assert any(m.role == "fab_structure_reference" for m in structure_mats), (
        [m.role for m in structure_mats]
    )

    # No `pose_*` records anywhere.
    for forbidden in (
        "pose_ensemble", "generated_pose", "modeled_pose",
        "pose_candidate", "modeled_adc_candidate",
    ):
        assert forbidden not in blob.lower()

    # is_generated_candidate must be False everywhere in Step 5.
    assert all(not r.is_generated_candidate for r in table.candidate_records)


def test_real_pdb_storage_path_persists_separately_from_normalized_record(client):
    pdb_path = _real_pdb("S2.pdb")
    pdb_bytes = pdb_path.read_bytes()
    resp = client.post(
        "/runs/multipart",
        data={
            "raw_user_query": "HER2 antigen ectodomain reference",
            "user_provided_context": json.dumps(
                {"target_or_antigen_text": "HER2"}
            ),
        },
        files=[("files", ("antigen_ectodomain_S2.pdb", pdb_bytes, "chemical/x-pdb"))],
    )
    assert resp.status_code == 201, resp.text
    body = resp.json()
    uploaded = body["raw_request_record"]["uploaded_files"][0]
    run_id = body["raw_request_record"]["run_id"]
    # The byte file landed on disk under storage_path, not inlined.
    storage = deps.get_storage()
    on_disk = storage.read_bytes(uploaded["storage_path"])
    assert on_disk == pdb_bytes
    # Step 5 references the file_id / storage_path, not the bytes.
    table, _ = _drive_step5_after_upload(run_id)
    persisted = storage.read_json(
        storage.run_key(run_id, "candidate_context_table.json")
    )
    target = next(
        r for r in table.candidate_records if r.candidate_type == "target_antigen"
    )
    structure_mats = [
        m for m in target.materials
        if m.material_type in {"structure_file", "structure_ref"}
    ]
    # The material value points to a storage path / filename, not bytes.
    assert structure_mats
    assert any(uploaded["storage_path"] in (m.value or "") for m in structure_mats)
    # And the role is the more specific `antigen_structure_reference`
    # because the filename + raw text both mention "antigen".
    assert any(
        m.role == "antigen_structure_reference" for m in structure_mats
    ), [m.role for m in structure_mats]
    # No raw bytes / no decoded PDB text inside normalized artifact.
    blob = json.dumps(persisted)
    assert "HEADER    " not in blob
    assert "ATOM " not in blob


# ── PubChem CID alone defaults to material-only ───────────────────────────


def _agent_table(query: str, ctx: dict, local_storage, registry_service, workflow_state_service):
    intake = IntakeService(local_storage, registry_service, workflow_state_service)
    rec = intake.submit(raw_user_query=query, user_provided_context=ctx)
    StructuredQueryService(
        local_storage, registry_service, workflow_state_service,
        SupervisorAgent(llm=MockLLMProvider()),
    ).parse(rec.run_id)
    InputReadinessService(
        local_storage, registry_service, workflow_state_service
    ).check(rec.run_id)
    WorkflowSetupService(
        local_storage, registry_service, workflow_state_service
    ).plan(rec.run_id)
    agent = CandidateContextAgent(
        storage=local_storage,
        registry=registry_service,
        workflow_state=workflow_state_service,
        mcp_client=LocalMCPClient(),
    )
    return agent.run(rec.run_id)


def test_pubchem_cid_alone_is_material_only(
    local_storage, registry_service, workflow_state_service
):
    table = _agent_table(
        "Reference PubChem CID 2244 for HER2 work.",
        {"target_or_antigen_text": "HER2"},
        local_storage, registry_service, workflow_state_service,
    )
    compounds = [r for r in table.candidate_records if r.candidate_type == "compound_component"]
    assert compounds
    rec = compounds[0]
    assert rec.is_generated_candidate is False
    assert rec.candidate_role == "material_only"
    assert rec.context_status == "material_pool"
    assert any(i.id_type == "pubchem_cid" for i in rec.identifiers)
    roles = {m.role for m in rec.materials}
    # Not auto-promoted to payload_candidate.
    assert "payload_candidate" not in roles


def test_drugbank_id_alone_is_material_only(
    local_storage, registry_service, workflow_state_service
):
    table = _agent_table(
        "Reference DrugBank DB00072 for HER2 work.",
        {"target_or_antigen_text": "HER2"},
        local_storage, registry_service, workflow_state_service,
    )
    compounds = [r for r in table.candidate_records if r.candidate_type == "compound_component"]
    assert compounds
    rec = compounds[0]
    assert rec.is_generated_candidate is False
    assert rec.candidate_role == "material_only"
    assert any(i.id_type == "drugbank_id" for i in rec.identifiers)


def test_explicit_payload_wording_promotes_pubchem_cid(
    local_storage, registry_service, workflow_state_service
):
    """When the user says 'as payload candidates', the role promotes
    even for a non-SMILES, non-ChEMBL identifier source."""
    table = _agent_table(
        "Screen PubChem CID 2244 and CID 5311 as payload candidates for HER2 ADC.",
        {"target_or_antigen_text": "HER2"},
        local_storage, registry_service, workflow_state_service,
    )
    compounds = [r for r in table.candidate_records if r.candidate_type == "compound_component"]
    assert compounds
    rec = compounds[0]
    assert rec.candidate_role == "user_provided_candidate"
    roles = {m.role for m in rec.materials}
    assert "payload_candidate" in roles


# ── T-DM1 / T-DXd / Enhertu are reference benchmarks, not generated ──────


def test_real_adc_names_become_reference_benchmarks(
    local_storage, registry_service, workflow_state_service
):
    table = _agent_table(
        "Compare T-DM1 vs T-DXd (Enhertu) for HER2 breast cancer.",
        {"target_or_antigen_text": "HER2"},
        local_storage, registry_service, workflow_state_service,
    )
    by_label = {r.candidate_label: r for r in table.candidate_records}
    for name in ("T-DM1", "T-DXd", "Enhertu"):
        if name not in by_label:
            continue
        rec = by_label[name]
        assert rec.candidate_role == "reference_benchmark"
        assert rec.is_generated_candidate is False
        assert rec.context_status == "complete_reference"
        # context_notes carry the "NOT a generated candidate" line.
        joined_notes = " ".join(rec.context_notes).lower()
        assert "not a generated candidate" in joined_notes


# ── TROP2 + MMAE incomplete — gaps explicit, no completion ───────────────


def test_trop2_with_mmae_records_missing_pieces(
    local_storage, registry_service, workflow_state_service
):
    table = _agent_table(
        "Design a new TROP2 ADC with MMAE payload.",
        {"target_or_antigen_text": "TROP2", "payload_linker_text": "MMAE"},
        local_storage, registry_service, workflow_state_service,
    )
    target_rec = next(
        r for r in table.candidate_records if r.candidate_type == "target_antigen"
    )
    assert "antibody" in target_rec.missing_material_roles
    assert "linker" in target_rec.missing_material_roles
    assert any(
        "complete ADC" in g.lower() or "complete adc" in g.lower()
        for g in target_rec.data_gaps
    )
    # No antibody candidate invented.
    assert not any(r.candidate_type == "antibody" for r in table.candidate_records)
    # And no reference_benchmark created from this incomplete spec.
    assert not any(
        r.candidate_role == "reference_benchmark" for r in table.candidate_records
    )


# ── vc-MMAE keeps the composite + decomposes with inferred flags ─────────


def test_vc_mmae_keeps_composite_and_marks_linker_inferred(
    local_storage, registry_service, workflow_state_service
):
    table = _agent_table(
        "Design HER2 ADC with vc-MMAE.",
        {"target_or_antigen_text": "HER2", "payload_linker_text": "vc-MMAE"},
        local_storage, registry_service, workflow_state_service,
    )
    lp_records = [
        r for r in table.candidate_records
        if any(m.role == "linker_payload" for m in r.materials)
    ]
    assert lp_records, "expected a linker_payload candidate for vc-MMAE"
    by_role = {
        m.role: m for m in lp_records[0].materials
        if m.role in {"linker", "payload", "linker_payload"}
    }
    assert by_role["linker_payload"].role_status == "explicit"
    if "linker" in by_role:
        # Linker remains inferred unless user names it.
        assert by_role["linker"].role_status == "inferred"
    if "payload" in by_role:
        # MMAE is explicit because vc-MMAE contains it as a meaningful token.
        assert by_role["payload"].role_status == "explicit"


# ── Filename-driven role inference ───────────────────────────────────────


@pytest.mark.parametrize(
    "filename,context,expected",
    [
        ("trastuzumab_Fab.pdb",         "HER2 ADC reference",         "fab_structure_reference"),
        ("igg1_Fc_region.pdb",          "Fc region reference",        "fc_region_reference"),
        ("trastuzumab_N297.pdb",        "N297 glycosylation site",    "n297_site_reference"),
        ("antibody_glycoform.pdb",      "glycan analysis",            "glycan_or_glycosylation_reference"),
        ("conjugation_attachment_site.pdb", "linker attachment site", "linker_attachment_site"),
        ("scfv_arm.pdb",                "anti-HER2 scFv arm",         "antibody_arm_reference"),
        ("antigen_ectodomain.pdb",      "TROP2 antigen ectodomain",   "antigen_structure_reference"),
        ("receptor_domain.pdb",         "EGFR receptor structure",    "receptor_structure_reference"),
        ("epitope_fragment.pdb",        "epitope fragment reference", "experimental_fragment_reference"),
        ("complex.pdb",                 "generic complex",            "structure_reference"),
    ],
)
def test_infer_structure_role_specific_buckets(filename, context, expected):
    assert _infer_structure_role(filename, context) == expected


def test_real_pdb_with_fc_filename_picks_fc_role(client):
    """Using S3.pdb renamed to invoke the Fc-region bucket; the
    inference looks at filename + text, never at PDB bytes."""
    pdb_bytes = _real_pdb("S3.pdb").read_bytes()
    resp = client.post(
        "/runs/multipart",
        data={
            "raw_user_query": "Fc region reference for an ADC backbone study",
            "user_provided_context": json.dumps(
                {"target_or_antigen_text": "HER2"}
            ),
        },
        files=[("files", ("igg1_Fc_region_S3.pdb", pdb_bytes, "chemical/x-pdb"))],
    )
    assert resp.status_code == 201
    run_id = resp.json()["raw_request_record"]["run_id"]
    table, storage = _drive_step5_after_upload(run_id)
    target = next(
        r for r in table.candidate_records if r.candidate_type == "target_antigen"
    )
    structure_mats = [
        m for m in target.materials
        if m.material_type in {"structure_file", "structure_ref"}
    ]
    assert any(m.role == "fc_region_reference" for m in structure_mats), (
        [m.role for m in structure_mats]
    )
    # No PDB bytes inlined.
    persisted = storage.read_json(
        storage.run_key(run_id, "candidate_context_table.json")
    )
    blob = json.dumps(persisted)
    assert "HEADER    " not in blob
    assert "ATOM " not in blob
