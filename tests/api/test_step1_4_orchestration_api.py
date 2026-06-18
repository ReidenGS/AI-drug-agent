"""Step 1 → 4 API end-to-end regression.

Uses FastAPI's TestClient against the actual route layer (`POST /runs`,
`/runs/{id}/steps/2/execute`, `/steps/3/execute`, `/steps/4/execute`).
Every test runs offline against `MockLLMProvider` — no Gemini, no MCP,
no ToolUniverse, no file bytes printed.

Scenarios:

- **A. ready** — full ADC context flows Step 1 → 2 → 3 → 4 and produces
  a `ready_to_execute` plan; all four registry artifacts are present.
- **B. needs_user_input** — target only; readiness reports
  `needs_user_input`; Step 4 still writes a plan with
  `plan_status="wait_for_input"`.
- **C. blocked** — no target; readiness reports `blocked`; Step 4
  refuses to plan and the API surfaces the structured 409 error rather
  than silently writing a fake `ready_to_execute` plan.
- **multipart D** — multipart upload metadata (PDB / FASTA) survives
  Step 1 intake and is picked up by Step 3 file-role inference WITHOUT
  reading the file bytes (the readiness service only inspects metadata).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import app.deps as deps
from app.main import app


_PDB_BYTES = (
    b"HEADER    HYDROLASE                               01-JAN-26   XXXX\n"
    b"ATOM      1  N   ALA A   1      11.104  13.207  10.000  1.00 20.00           N\n"
    b"END\n"
)
_FASTA_BYTES = (
    b">heavy_chain\nEVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVR\n"
    b">light_chain\nDIQMTQSPSSLSASVGDRVTITC\n"
)


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


def _create_run(client: TestClient, ctx: dict, *, query="HER2 ADC") -> str:
    resp = client.post(
        "/runs",
        json={"raw_user_query": query, "user_provided_context": ctx},
    )
    assert resp.status_code == 200, resp.text
    return resp.json()["run_id"]


def _run_steps_2_3_4(client: TestClient, run_id: str) -> dict:
    """Execute Step 2 → 3 → 4 in sequence, returning the final-step response."""
    r2 = client.post(f"/runs/{run_id}/steps/2/execute")
    assert r2.status_code == 200, r2.text
    r3 = client.post(f"/runs/{run_id}/steps/3/execute")
    assert r3.status_code == 200, r3.text
    r4 = client.post(f"/runs/{run_id}/steps/4/execute")
    return {"step_2": r2, "step_3": r3, "step_4": r4}


def _registry_artifacts(run_id: str) -> dict:
    return deps.get_registry_service().get(run_id).active_artifacts.model_dump()


# ── A. ready ───────────────────────────────────────────────────────────────


def test_scenario_A_ready_creates_all_four_artifacts(client: TestClient):
    run_id = _create_run(
        client,
        ctx={
            "target_or_antigen_text": "HER2",
            "candidate_text": "Trastuzumab analog",
            "payload_linker_text": "vc-MMAE",
        },
        query="Design an ADC against HER2 with vc-MMAE payload",
    )
    out = _run_steps_2_3_4(client, run_id)
    assert out["step_4"].status_code == 200, out["step_4"].text

    plan = out["step_4"].json()
    assert plan["plan_status"] == "ready_to_execute"

    arts = _registry_artifacts(run_id)
    assert arts["raw_request_record_id"]
    assert arts["structured_query_id"]
    assert arts["input_readiness_status_id"]
    assert arts["run_step_plan_id"]


# ── B. needs_user_input ────────────────────────────────────────────────────


def test_scenario_B_needs_user_input_writes_wait_for_input_plan(client: TestClient):
    run_id = _create_run(
        client,
        ctx={
            "target_or_antigen_text": "HER2",
            "candidate_text": "Trastuzumab",
        },
        query="Build an ADC against HER2",
    )
    out = _run_steps_2_3_4(client, run_id)

    assert out["step_3"].json()["input_readiness_status"] == "needs_user_input"
    assert out["step_4"].status_code == 200, out["step_4"].text
    plan = out["step_4"].json()
    assert plan["plan_status"] == "wait_for_input"
    # Step 4 must still write a plan — callers need per-step reasons.
    arts = _registry_artifacts(run_id)
    assert arts["run_step_plan_id"]


def test_scenario_B_step5_api_is_gated_with_409(client: TestClient):
    """The wait_for_input plan must hard-gate Step 5 at the API layer."""
    run_id = _create_run(
        client,
        ctx={
            "target_or_antigen_text": "HER2",
            "candidate_text": "Trastuzumab",
        },
        query="Build an ADC against HER2",
    )
    _run_steps_2_3_4(client, run_id)

    resp = client.post(f"/runs/{run_id}/steps/5/execute")
    assert resp.status_code == 409, resp.text
    body = resp.json()
    assert body["code"] == "workflow_state_error"
    assert body["detail"]["plan_status"] == "wait_for_input"
    # Step 5 artifact must NOT have been written.
    assert _registry_artifacts(run_id)["candidate_context_table_id"] is None


# ── C. blocked ─────────────────────────────────────────────────────────────


def test_scenario_C_blocked_does_not_create_a_ready_plan(client: TestClient):
    run_id = _create_run(
        client,
        ctx={},  # no target → readiness=blocked
        query="Help me get started — no specifics yet",
    )
    r2 = client.post(f"/runs/{run_id}/steps/2/execute")
    assert r2.status_code == 200
    r3 = client.post(f"/runs/{run_id}/steps/3/execute")
    assert r3.status_code == 200
    assert r3.json()["input_readiness_status"] == "blocked"

    # Step 4 refuses to plan — surfaces 409 (WorkflowStateError) with the
    # blocking reasons attached. The plan must NOT be written.
    r4 = client.post(f"/runs/{run_id}/steps/4/execute")
    assert r4.status_code == 409, r4.text
    body = r4.json()
    assert body["code"] == "workflow_state_error"
    detail = body.get("detail") or {}
    assert "blocking_reasons" in detail
    arts = _registry_artifacts(run_id)
    assert arts["run_step_plan_id"] is None

    # And Step 5 stays 409 because no plan exists at all.
    r5 = client.post(f"/runs/{run_id}/steps/5/execute")
    assert r5.status_code == 409
    assert r5.json()["detail"]["plan_status"] == "missing"


# ── D. multipart upload metadata flows to Step 3 file-role readiness ──────


def test_multipart_upload_flows_to_step3_role_inference_without_file_bytes(
    client: TestClient, monkeypatch
):
    """Step 1 multipart accepts a PDB + FASTA; Step 3 sees the inferred
    roles via metadata ONLY. We patch `read_bytes` to assert no Step 3
    code path tries to read the persisted file contents."""
    resp = client.post(
        "/runs/multipart",
        data={
            "raw_user_query": "HER2 ADC with attached complex and chain",
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
            ("files", ("trastuzumab.fasta", _FASTA_BYTES, "text/x-fasta")),
        ],
    )
    assert resp.status_code == 201, resp.text
    body = resp.json()
    run_id = body["raw_request_record"]["run_id"]
    uploaded = body["raw_request_record"]["uploaded_files"]
    # Sanity: storage paths + sha256 populated for both files.
    by_name = {f["original_filename"]: f for f in uploaded}
    for name in ("complex.pdb", "trastuzumab.fasta"):
        assert by_name[name]["storage_path"]
        assert by_name[name]["sha256"].startswith("sha256:")
    assert by_name["complex.pdb"]["sha256"] == "sha256:" + hashlib.sha256(_PDB_BYTES).hexdigest()

    # Patch read_bytes so any attempt to read an UPLOADED-FILE byte stream
    # during Step 2/3/4 fails loudly. (JSON artifacts like
    # `registry/current.json` go through read_bytes too — we let those
    # through and only fail on the uploaded files' storage_paths.)
    storage = deps.get_storage()
    sentinel = {"forbidden_calls": 0}
    orig_read_bytes = storage.read_bytes
    uploaded_paths = {f["storage_path"] for f in uploaded}

    def _guarded_read_bytes(path: str):
        if path in uploaded_paths:
            sentinel["forbidden_calls"] += 1
            raise AssertionError(
                f"Step 2/3/4 must not read uploaded-file bytes; read_bytes({path!r}) was called"
            )
        return orig_read_bytes(path)

    monkeypatch.setattr(storage, "read_bytes", _guarded_read_bytes)
    try:
        _run_steps_2_3_4(client, run_id)
    finally:
        storage.read_bytes = orig_read_bytes  # type: ignore[method-assign]

    # Inferred roles are visible on the Step 3 artifact.
    readiness_key = storage.run_key(run_id, "inputs/input_readiness_status.json")
    readiness = storage.read_json(readiness_key)
    roles_by_filename: dict[str, str] = {}
    file_id_to_filename = {f["file_id"]: f["original_filename"] for f in uploaded}
    for fc in readiness["uploaded_file_checks"]:
        roles_by_filename[file_id_to_filename[fc["file_id"]]] = fc["inferred_role"]
    assert roles_by_filename["complex.pdb"] == "pdb_or_cif_structure"
    assert roles_by_filename["trastuzumab.fasta"] == "fasta_sequence"
    # And no uploaded-file byte was read.
    assert sentinel["forbidden_calls"] == 0

    # Plan is ready_to_execute (full ADC context + a structure/sequence file).
    plan_key = storage.run_key(run_id, "inputs/run_step_plan.json")
    plan = storage.read_json(plan_key)
    assert plan["plan_status"] == "ready_to_execute"


# ── no live Gemini / no MCP touched anywhere in this suite ────────────────


def test_step1_4_path_does_not_touch_mcp_or_live_gemini(client: TestClient, monkeypatch):
    """Sanity guard: Step 2/3/4 path must not build the ToolUniverse
    singleton and must not invoke Gemini (LLM_PROVIDER=mock)."""
    from app.mcp import tooluniverse_adapter

    tooluniverse_adapter._reset_for_tests()
    sentinel = {"tu_built": False}

    def _explode():
        sentinel["tu_built"] = True
        raise AssertionError("Step 2/3/4 must not build ToolUniverse")

    monkeypatch.setattr(tooluniverse_adapter, "_get_universe", _explode)

    run_id = _create_run(
        client,
        ctx={
            "target_or_antigen_text": "HER2",
            "candidate_text": "Trastuzumab",
            "payload_linker_text": "vc-MMAE",
        },
        query="Design HER2 ADC with vc-MMAE",
    )
    _run_steps_2_3_4(client, run_id)
    assert sentinel["tu_built"] is False
    # LLM provider stayed on the mock — sanity-check via the dep helper.
    from app.llm.provider import MockLLMProvider

    assert isinstance(deps.get_llm_provider(), MockLLMProvider)
