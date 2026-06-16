"""Multipart Step 1 intake — simulated frontend submission.

Scenarios covered:
1. text-only multipart (no files)
2. multipart + single PDB file
3. multipart + multi-file (pdb + fasta)
4. JSON POST /runs still works (regression)
5. Step 2/3 read multipart-uploaded files and infer their role
6. Step 5 builds materials from the multipart-uploaded PDB + FASTA
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import app.deps as deps
from app.agents.candidate_context_agent import CandidateContextAgent
from app.agents.supervisor_agent import SupervisorAgent
from app.llm.provider import MockLLMProvider
from app.main import app
from app.mcp.client import LocalMCPClient
from app.services.input_readiness_service import InputReadinessService
from app.services.structured_query_service import StructuredQueryService
from app.services.workflow_setup_service import WorkflowSetupService


@pytest.fixture
def client(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("STORAGE_MODE", "local")
    monkeypatch.setenv("LOCAL_STORAGE_ROOT", str(tmp_path / "store"))
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    from app.settings import get_settings

    for fn in (
        get_settings,
        deps.get_storage,
        deps.get_registry_service,
        deps.get_workflow_state_service,
        deps.get_tool_inventory_service,
        deps.get_mcp_client,
        deps.get_llm_provider,
    ):
        fn.cache_clear()
    yield TestClient(app)
    for fn in (
        get_settings,
        deps.get_storage,
        deps.get_registry_service,
        deps.get_workflow_state_service,
        deps.get_tool_inventory_service,
        deps.get_mcp_client,
        deps.get_llm_provider,
    ):
        fn.cache_clear()


_PDB_BYTES = (
    b"HEADER    HYDROLASE                               01-JAN-26   XXXX\n"
    b"ATOM      1  N   ALA A   1      11.104  13.207  10.000  1.00 20.00           N\n"
    b"END\n"
)
_FASTA_BYTES = (
    b">heavy_chain\nEVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVR\n"
    b">light_chain\nDIQMTQSPSSLSASVGDRVTITC\n"
)


# ── 1. text-only multipart ────────────────────────────────────────────────────

def test_multipart_text_only_creates_run(client: TestClient):
    resp = client.post(
        "/runs/multipart",
        data={
            "raw_user_query": "Design ADC against HER2 with vc-MMAE",
            "entry_source": "ui",
            "submitted_by": "tester@nyu.edu",
            "user_provided_context": json.dumps(
                {
                    "target_or_antigen_text": "HER2",
                    "candidate_text": "Trastuzumab analog",
                    "payload_linker_text": "vc-MMAE",
                }
            ),
        },
    )
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["uploaded_file_count"] == 0
    rec = body["raw_request_record"]
    assert rec["uploaded_files"] == []
    assert rec["entry_source"] == "ui"
    assert rec["submitted_by"] == "tester@nyu.edu"
    assert rec["user_provided_context"]["target_or_antigen_text"] == "HER2"

    # Stored artifact matches the response.
    storage = deps.get_storage()
    persisted = storage.read_json(
        storage.run_key(rec["run_id"], "inputs/raw_request_record.json")
    )
    assert persisted["raw_user_query"] == rec["raw_user_query"]


# ── 2. multipart + single PDB file ───────────────────────────────────────────

def test_multipart_single_pdb_persists_and_indexes_metadata(client: TestClient):
    resp = client.post(
        "/runs/multipart",
        data={
            "raw_user_query": "HER2 ADC with attached complex",
            "user_provided_context": json.dumps({"target_or_antigen_text": "HER2"}),
        },
        files=[("files", ("complex.pdb", _PDB_BYTES, "chemical/x-pdb"))],
    )
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["uploaded_file_count"] == 1

    uploaded = body["raw_request_record"]["uploaded_files"][0]
    for k in ("file_id", "original_filename", "storage_path",
              "content_type", "sha256", "size_bytes"):
        assert uploaded.get(k), f"missing metadata field: {k}"

    assert uploaded["original_filename"] == "complex.pdb"
    assert uploaded["content_type"] == "chemical/x-pdb"
    assert uploaded["size_bytes"] == len(_PDB_BYTES)
    assert uploaded["sha256"] == "sha256:" + hashlib.sha256(_PDB_BYTES).hexdigest()

    # File actually landed on disk under the run directory.
    storage = deps.get_storage()
    on_disk = storage.read_bytes(uploaded["storage_path"])
    assert on_disk == _PDB_BYTES
    # And the path is under the run's inputs/files/ dir.
    assert uploaded["storage_path"].endswith(".pdb")
    assert "/runs/" in uploaded["storage_path"]
    assert "/inputs/files/" in uploaded["storage_path"]

    # Raw bytes never leak into the JSON artifact.
    persisted = storage.read_json(
        storage.run_key(body["run_id"], "inputs/raw_request_record.json")
    )
    assert "HYDROLASE" not in json.dumps(persisted)


# ── 3. multipart + multi-file ────────────────────────────────────────────────

def test_multipart_multi_file_persists_each_separately(client: TestClient):
    resp = client.post(
        "/runs/multipart",
        data={
            "raw_user_query": "HER2 ADC, structure + sequence attached",
            "user_provided_context": json.dumps({"target_or_antigen_text": "HER2"}),
        },
        files=[
            ("files", ("complex.pdb", _PDB_BYTES, "chemical/x-pdb")),
            ("files", ("ab_chains.fasta", _FASTA_BYTES, "text/x-fasta")),
        ],
    )
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["uploaded_file_count"] == 2

    files = body["raw_request_record"]["uploaded_files"]
    by_name = {f["original_filename"]: f for f in files}
    assert {"complex.pdb", "ab_chains.fasta"} == set(by_name)

    # Distinct file_ids and storage paths.
    file_ids = {f["file_id"] for f in files}
    paths = {f["storage_path"] for f in files}
    assert len(file_ids) == 2
    assert len(paths) == 2

    storage = deps.get_storage()
    assert storage.read_bytes(by_name["complex.pdb"]["storage_path"]) == _PDB_BYTES
    assert storage.read_bytes(by_name["ab_chains.fasta"]["storage_path"]) == _FASTA_BYTES
    assert by_name["ab_chains.fasta"]["sha256"] == (
        "sha256:" + hashlib.sha256(_FASTA_BYTES).hexdigest()
    )


# ── 4. JSON POST /runs regression ────────────────────────────────────────────

def test_json_post_runs_still_works(client: TestClient):
    """The original internal/notebook ingestion path must keep working."""
    resp = client.post(
        "/runs",
        json={
            "raw_user_query": "HER2 ADC vc-MMAE",
            "user_provided_context": {
                "target_or_antigen_text": "HER2",
                "payload_linker_text": "vc-MMAE",
            },
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["run_id"].startswith("run_")
    assert body["raw_request_record"]["uploaded_files"] == []


# ── 5. Step 3 reads multipart-uploaded files for role inference ─────────────

def test_multipart_run_lets_step3_infer_file_roles(client: TestClient):
    resp = client.post(
        "/runs/multipart",
        data={
            "raw_user_query": "HER2 ADC vc-MMAE with complex.pdb + sequences",
            "user_provided_context": json.dumps(
                {
                    "target_or_antigen_text": "HER2",
                    "candidate_text": "Trastuzumab",
                    "payload_linker_text": "vc-MMAE",
                }
            ),
        },
        files=[
            ("files", ("complex.pdb", _PDB_BYTES, "chemical/x-pdb")),
            ("files", ("heavy.fasta", _FASTA_BYTES, "text/x-fasta")),
        ],
    )
    assert resp.status_code == 201
    run_id = resp.json()["run_id"]

    storage = deps.get_storage()
    reg = deps.get_registry_service()
    ws = deps.get_workflow_state_service()
    StructuredQueryService(storage, reg, ws, SupervisorAgent(llm=MockLLMProvider())).parse(run_id)
    readiness = InputReadinessService(storage, reg, ws).check(run_id)

    roles = [c.inferred_role for c in readiness.uploaded_file_checks]
    assert "pdb_or_cif_structure" in roles
    assert "fasta_sequence" in roles
    assert readiness.basic_adc_input_presence.structure_or_sequence_present


# ── 6. Step 5 builds materials from multipart-uploaded files ─────────────────

def test_multipart_run_lets_step5_consume_uploaded_files(client: TestClient):
    resp = client.post(
        "/runs/multipart",
        data={
            "raw_user_query": "HER2 ADC with attached PDB and sequences",
            "user_provided_context": json.dumps(
                {
                    "target_or_antigen_text": "HER2",
                    "candidate_text": "Trastuzumab",
                    "payload_linker_text": "vc-MMAE",
                }
            ),
        },
        files=[
            ("files", ("complex.pdb", _PDB_BYTES, "chemical/x-pdb")),
            ("files", ("heavy.fasta", _FASTA_BYTES, "text/x-fasta")),
        ],
    )
    run_id = resp.json()["run_id"]

    storage = deps.get_storage()
    reg = deps.get_registry_service()
    ws = deps.get_workflow_state_service()
    StructuredQueryService(storage, reg, ws, SupervisorAgent(llm=MockLLMProvider())).parse(run_id)
    InputReadinessService(storage, reg, ws).check(run_id)
    WorkflowSetupService(storage, reg, ws).plan(run_id)
    table = CandidateContextAgent(
        storage=storage,
        registry=reg,
        workflow_state=ws,
        mcp_client=LocalMCPClient(),
    ).run(run_id)

    material_types = {m.material_type for c in table.candidate_records for m in c.materials}
    assert "structure_file" in material_types
    assert "antibody_heavy_chain_sequence" in material_types
