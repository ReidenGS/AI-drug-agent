"""Step 1 multipart hardening: pre-validation, limits, orphan cleanup.

Each test verifies BOTH that the request is rejected with the right status
AND that the storage / registry / workflow_state are left untouched.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import app.deps as deps
from app.main import app


_PDB_BYTES = b"HEADER    DUMMY\nATOM      1  N   ALA A   1      0.0  0.0  0.0\nEND\n"


@pytest.fixture
def client(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("STORAGE_MODE", "local")
    monkeypatch.setenv("LOCAL_STORAGE_ROOT", str(tmp_path / "store"))
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    # Lower limits make the cap tests fast — 3 files, 1 KiB.
    monkeypatch.setenv("MAX_UPLOAD_FILES_PER_RUN", "3")
    monkeypatch.setenv("MAX_UPLOAD_BYTES_PER_FILE", str(1024))
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


def _disk_is_empty(storage) -> bool:
    """No file artifact, no JSON artifact landed anywhere under the prefix."""
    return storage.list_prefix(storage.prefix) == []


# ── pre-validation rejects before any IO ────────────────────────────────────

def test_invalid_entry_source_returns_422_and_writes_nothing(client: TestClient):
    resp = client.post(
        "/runs/multipart",
        data={
            "raw_user_query": "HER2 ADC",
            "entry_source": "frontend",  # not in {ui, api, notebook, script}
            "user_provided_context": json.dumps({"target_or_antigen_text": "HER2"}),
        },
        files=[("files", ("complex.pdb", _PDB_BYTES, "chemical/x-pdb"))],
    )
    assert resp.status_code == 422
    assert "entry_source" in resp.json()["detail"]
    assert _disk_is_empty(deps.get_storage())


def test_non_object_user_context_returns_422_and_writes_nothing(client: TestClient):
    resp = client.post(
        "/runs/multipart",
        data={
            "raw_user_query": "HER2 ADC",
            "user_provided_context": json.dumps(["HER2", "MMAE"]),  # array, not object
        },
        files=[("files", ("complex.pdb", _PDB_BYTES, "chemical/x-pdb"))],
    )
    assert resp.status_code == 422
    assert "JSON object" in resp.json()["detail"]
    assert _disk_is_empty(deps.get_storage())


def test_malformed_json_user_context_returns_422_and_writes_nothing(client: TestClient):
    resp = client.post(
        "/runs/multipart",
        data={
            "raw_user_query": "HER2 ADC",
            "user_provided_context": "{not json",
        },
        files=[("files", ("complex.pdb", _PDB_BYTES, "chemical/x-pdb"))],
    )
    assert resp.status_code == 422
    assert _disk_is_empty(deps.get_storage())


def test_empty_raw_user_query_returns_422_and_writes_nothing(client: TestClient):
    resp = client.post(
        "/runs/multipart",
        data={
            "raw_user_query": "   ",  # whitespace only
            "user_provided_context": json.dumps({"target_or_antigen_text": "HER2"}),
        },
        files=[("files", ("complex.pdb", _PDB_BYTES, "chemical/x-pdb"))],
    )
    assert resp.status_code == 422
    assert "raw_user_query" in resp.json()["detail"]
    assert _disk_is_empty(deps.get_storage())


def test_invalid_session_returns_422_before_any_io(client: TestClient):
    response = client.post(
        "/runs/multipart",
        data={
            "raw_user_query": "HER2 structure",
            "session_id": "sess_NOT_OPAQUE",
        },
        files=[("files", ("complex.pdb", _PDB_BYTES, "chemical/x-pdb"))],
    )

    assert response.status_code == 422
    assert response.json()["detail"] == "session_id_invalid"
    assert _disk_is_empty(deps.get_storage())


def test_multipart_persists_explicit_session(client: TestClient):
    session_id = "sess_0123456789abcdef"
    response = client.post(
        "/runs/multipart",
        data={"raw_user_query": "HER2 structure", "session_id": session_id},
        files=[("files", ("complex.pdb", _PDB_BYTES, "chemical/x-pdb"))],
    )

    assert response.status_code == 201
    body = response.json()
    assert body["session_id"] == session_id
    assert body["raw_request_record"]["session_id"] == session_id
    persisted = deps.get_storage().read_json(
        deps.get_storage().run_key(
            body["run_id"], "inputs/raw_request_record.json"
        )
    )
    assert persisted["session_id"] == session_id


# ── upload limits ────────────────────────────────────────────────────────────

def test_too_many_files_returns_413_and_writes_nothing(client: TestClient):
    # Limit set to 3 by the fixture; send 4.
    files = [
        ("files", (f"f{i}.pdb", _PDB_BYTES, "chemical/x-pdb")) for i in range(4)
    ]
    resp = client.post(
        "/runs/multipart",
        data={
            "raw_user_query": "HER2 ADC",
            "user_provided_context": json.dumps({"target_or_antigen_text": "HER2"}),
        },
        files=files,
    )
    assert resp.status_code == 413
    assert "Too many files" in resp.json()["detail"]
    assert _disk_is_empty(deps.get_storage())


def test_oversized_file_returns_413_and_cleans_up_earlier_files(client: TestClient):
    """Send one small file (accepted), then one oversized (rejected). The
    earlier-written file must be cleaned up so we leave no orphan bytes."""
    big = b"X" * 4096  # > 1 KiB limit
    files = [
        ("files", ("ok.pdb", _PDB_BYTES, "chemical/x-pdb")),
        ("files", ("too_big.fasta", big, "text/x-fasta")),
    ]
    resp = client.post(
        "/runs/multipart",
        data={
            "raw_user_query": "HER2 ADC",
            "user_provided_context": json.dumps({"target_or_antigen_text": "HER2"}),
        },
        files=files,
    )
    assert resp.status_code == 413
    assert "exceeds per-file size limit" in resp.json()["detail"]
    # No file (small or big) and no raw_request_record should be on disk.
    assert _disk_is_empty(deps.get_storage())


# ── orphan cleanup on IntakeService failure ─────────────────────────────────

def test_intake_failure_cleans_up_written_files(tmp_path: Path, monkeypatch):
    """Simulate an IntakeService.submit crash mid-handler; ALL written upload
    bytes must be cleaned up and no raw_request_record should be persisted."""
    monkeypatch.setenv("STORAGE_MODE", "local")
    monkeypatch.setenv("LOCAL_STORAGE_ROOT", str(tmp_path / "store"))
    monkeypatch.setenv("MAX_UPLOAD_FILES_PER_RUN", "3")
    monkeypatch.setenv("MAX_UPLOAD_BYTES_PER_FILE", str(1024))
    from app.settings import get_settings

    for fn in (
        get_settings,
        deps.get_storage,
        deps.get_registry_service,
        deps.get_workflow_state_service,
    ):
        fn.cache_clear()

    from app.services import intake_service as intake_mod

    def _exploding_submit(self, **kwargs):  # noqa: ANN001
        # Files have already been written by the time we get here.
        raise RuntimeError("simulated downstream failure")

    monkeypatch.setattr(intake_mod.IntakeService, "submit", _exploding_submit)

    # `raise_server_exceptions=False` so we observe the 500 response rather
    # than the raised exception bubbling out of TestClient.
    isolated_client = TestClient(app, raise_server_exceptions=False)
    resp = isolated_client.post(
        "/runs/multipart",
        data={
            "raw_user_query": "HER2 ADC",
            "user_provided_context": json.dumps({"target_or_antigen_text": "HER2"}),
        },
        files=[
            ("files", ("a.pdb", _PDB_BYTES, "chemical/x-pdb")),
            ("files", ("b.fasta", b">x\nACDEFG\n", "text/x-fasta")),
        ],
    )
    assert resp.status_code == 500

    storage = deps.get_storage()
    leftover_files = [p for p in storage.list_prefix(storage.prefix) if "/inputs/files/" in p]
    assert leftover_files == [], f"orphan upload bytes left after failure: {leftover_files}"
    # And no raw_request_record artifact either, because submit never ran.
    raw_records = [
        p for p in storage.list_prefix(storage.prefix)
        if p.endswith("raw_request_record.json")
    ]
    assert raw_records == []


# ── happy path still works under the lowered fixture limits ────────────────

def test_within_limits_still_succeeds_under_hardening(client: TestClient):
    """Sanity: at-limit (3 files, each well below 1 KiB) still completes."""
    files = [
        ("files", (f"f{i}.pdb", _PDB_BYTES, "chemical/x-pdb")) for i in range(3)
    ]
    resp = client.post(
        "/runs/multipart",
        data={
            "raw_user_query": "HER2 ADC",
            "user_provided_context": json.dumps({"target_or_antigen_text": "HER2"}),
        },
        files=files,
    )
    assert resp.status_code == 201, resp.text
    assert resp.json()["uploaded_file_count"] == 3
