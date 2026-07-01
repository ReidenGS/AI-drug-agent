"""Step 5 antibody full-sequence → CDR3 → IEDB BCR lookup tests.

Pins the safe link:

- Full VH/VL sequence on an antibody candidate triggers CDR3 extraction
  via a real antibody-numbering backend (abnumber → anarci).
- Only when extraction succeeds does the agent build an IEDB BCR
  call. Heavy CDR3 → ``filters={"chain1_cdr3_seq": "eq.<cdr3>"}``,
  light CDR3 → ``filters={"chain2_cdr3_seq": "eq.<cdr3>"}``.
- When extraction fails (dependency_unavailable / no_variable_domain /
  extraction_failed), Step 5 records a clear data_gap and does NOT
  query IEDB with the full sequence.
- Raw CDR3 may enter MCP arguments but NEVER reaches LLM payload,
  audit, candidate_records, or the persisted normalized artifact.
- tool_input_summary keeps only the redacted CDR3 metadata
  (length / sha256_prefix / chain_type / source material id).
"""

from __future__ import annotations

import json
import shutil
from typing import Any

import pytest

from app.agents import antibody_cdr3_extraction
from app.agents import candidate_context_agent as cca_module
from app.agents.antibody_cdr3_extraction import (
    CHAIN_TYPE_HEAVY,
    CHAIN_TYPE_LIGHT,
    CHAIN_TYPE_UNKNOWN,
    Cdr3Result,
    STATUS_DEPENDENCY_UNAVAILABLE,
    STATUS_EXTRACTION_FAILED,
    STATUS_NO_VARIABLE_DOMAIN,
    STATUS_SUCCESS,
    extract_cdr3,
)
from app.agents.candidate_context_agent import CandidateContextAgent
from app.agents.supervisor_agent import SupervisorAgent
from app.llm.provider import MockLLMProvider
from app.mcp.client import LocalMCPClient
from app.services.intake_service import IntakeService
from app.services.input_readiness_service import InputReadinessService
from app.services.structured_query_service import StructuredQueryService
from app.services.workflow_setup_service import WorkflowSetupService


_FULL_HEAVY = (
    "EVQLVQSGAEVKKPGSSVKVSCKASGGTFSSYAISWVRQAPGQGLEWMGGIIPIFGTANY"
    "AQKFQGRVTITADESTSTAYMELSSLRSEDTAVYYCARGGYDFWSGYYTFDYWGQGTLVTV"
)
_FULL_LIGHT = (
    "DIQMTQSPSSLSASVGDRVTITCRASQDVNTAVAWYQQKPGKAPKLLIYSASFLYSGVP"
    "SRFSGSRSGTDFTLTISSLQPEDFATYYCQQHYTTPPTFGQGTKVEIK"
)


# ── 0. extract_cdr3 contract ─────────────────────────────────────────────


def test_extract_cdr3_degrades_to_dependency_unavailable_when_backends_missing(
    monkeypatch,
):
    """In an environment with neither abnumber nor anarci, the adapter
    must return ``dependency_unavailable`` — never invent a CDR3."""
    # Force both backends to "not installed".
    monkeypatch.setattr(
        antibody_cdr3_extraction, "_try_extract_via_abnumber",
        lambda *_a, **_k: None,
    )
    monkeypatch.setattr(
        antibody_cdr3_extraction, "_try_extract_via_anarci",
        lambda *_a, **_k: None,
    )
    result = extract_cdr3(
        _FULL_HEAVY, expected_chain_role="antibody_heavy"
    )
    assert result.status == STATUS_DEPENDENCY_UNAVAILABLE
    assert result.cdr3_sequence == ""
    assert result.cdr3_length == 0
    assert result.cdr3_sha256_prefix == ""
    # The compact audit projection must be raw-CDR3-free by construction.
    audit = result.to_compact_audit()
    assert "cdr3_sequence" not in audit
    assert audit["status"] == STATUS_DEPENDENCY_UNAVAILABLE


def test_extract_cdr3_rejects_short_sequence():
    result = extract_cdr3("EVQLV")  # < 70 chars
    assert result.status == STATUS_NO_VARIABLE_DOMAIN
    assert result.cdr3_sequence == ""


def test_extract_cdr3_rejects_non_aa_characters():
    bad = "X" * 200  # X is not in the canonical AA alphabet
    result = extract_cdr3(bad, expected_chain_role="antibody_heavy")
    assert result.status == STATUS_EXTRACTION_FAILED
    assert result.cdr3_sequence == ""


@pytest.mark.skipif(
    shutil.which("hmmscan") is None,
    reason="HMMER hmmscan is required by ANARCI / abnumber",
)
def test_extract_cdr3_real_dependency_smoke():
    pytest.importorskip("abnumber")
    result = extract_cdr3(
        _FULL_HEAVY, expected_chain_role="antibody_heavy"
    )
    assert result.status == STATUS_SUCCESS
    assert result.chain_type == CHAIN_TYPE_HEAVY
    assert result.backend in {"abnumber", "anarci"}
    assert result.numbering_scheme == "IMGT"
    assert result.source_sequence_length == len(_FULL_HEAVY)
    assert result.cdr3_length > 0
    assert result.cdr3_sha256_prefix
    assert "cdr3_sequence" not in result.to_compact_audit()


# ── 1. Agent: mocked numbering success → IEDB filter built correctly ────


def _seed_run_with_antibody_sequence_material(
    storage, registry_service, workflow_state_service,
    *, sequence_text: str, chain_role: str,
    candidate_text: str | None = None,
    filename: str | None = None,
) -> str:
    """Submit a run, parse Step 2 / 3 / 4, then patch the antibody
    candidate in Step 5 by writing the raw_request_record and
    structured_query so a sequence_files material is attached."""
    intake = IntakeService(storage, registry_service, workflow_state_service)
    run_id = intake.allocate_run_id()
    fasta_name = filename or (
        "heavy_chain.fasta" if chain_role == "antibody_heavy"
        else "light_chain.fasta"
    )
    fasta_text = f">fixture_chain\n{sequence_text}\n"
    fasta_path = storage.run_key(run_id, "inputs/files", fasta_name)
    storage.write_bytes(fasta_path, fasta_text.encode("utf-8"))
    intake.submit(
        run_id=run_id,
        raw_user_query="ADC against HER2 with provided antibody sequence.",
        user_provided_context={
            "target_or_antigen_text": "HER2",
            "candidate_text": candidate_text or "Trastuzumab analog",
            "payload_linker_text": "vc-MMAE",
        },
        uploaded_files=[{
            "file_id": "f_seq_fixture",
            "original_filename": fasta_name,
            "storage_path": fasta_path,
            "content_type": "text/x-fasta",
            "size_bytes": len(fasta_text),
        }],
    )
    StructuredQueryService(
        storage, registry_service, workflow_state_service,
        SupervisorAgent(llm=MockLLMProvider()),
    ).parse(run_id)
    InputReadinessService(
        storage, registry_service, workflow_state_service
    ).check(run_id)
    WorkflowSetupService(
        storage, registry_service, workflow_state_service
    ).plan(run_id)
    return run_id


def _patch_chain_specific_material(
    storage, run_id: str, *, chain_role: str, sequence_text: str,
):
    """Force the Step 5 antibody candidate to carry a heavy- or
    light-chain sequence material instead of the generic sequence
    reference. Step 2 fixtures store the literal sequence under
    ``material.value`` so the agent's
    ``_read_sequence_material_text`` resolves it directly without
    storage round-trip."""
    cct_key = storage.run_key(run_id, "candidate_context_table.json")
    cct = storage.read_json(cct_key)
    target_material_type = (
        "antibody_heavy_chain_sequence" if chain_role == "antibody_heavy"
        else "antibody_light_chain_sequence"
    )
    for c in cct["candidate_records"]:
        if c["candidate_type"] != "antibody":
            continue
        for m in c["materials"]:
            if m["material_type"] in (
                "antibody_heavy_chain_sequence",
                "antibody_light_chain_sequence",
            ):
                m["material_type"] = target_material_type
                m["value"] = sequence_text
                m["value_format"] = "amino_acid"
    storage.write_json(cct_key, cct)


def _mock_cdr3_success(
    monkeypatch, *, chain_type: str, cdr3: str, backend: str = "fake_abnumber",
):
    def _fake(sequence: str, *, expected_chain_role="unknown"):
        from app.agents.antibody_cdr3_extraction import _build_result
        return _build_result(
            status=STATUS_SUCCESS,
            backend=backend,
            sequence_len=len(sequence),
            chain_type=chain_type,
            cdr3=cdr3,
            notes="mocked numbering success",
        )
    monkeypatch.setattr(cca_module, "extract_cdr3", _fake)


def _mock_cdr3_status(monkeypatch, *, status: str, backend: str = ""):
    def _fake(sequence: str, *, expected_chain_role="unknown"):
        from app.agents.antibody_cdr3_extraction import _build_result
        return _build_result(
            status=status,
            backend=backend,
            sequence_len=len(sequence),
            chain_type=CHAIN_TYPE_UNKNOWN,
            cdr3="",
            warnings=[f"mocked {status}"],
            notes=f"mocked {status}",
        )
    monkeypatch.setattr(cca_module, "extract_cdr3", _fake)


def test_uploaded_heavy_fasta_path_routes_iedb_without_artifact_patch(
    local_storage, registry_service, workflow_state_service, monkeypatch,
):
    run_id = _seed_run_with_antibody_sequence_material(
        local_storage, registry_service, workflow_state_service,
        sequence_text=_FULL_HEAVY, chain_role="antibody_heavy",
        filename="heavy_chain.fasta",
    )
    _mock_cdr3_success(
        monkeypatch, chain_type=CHAIN_TYPE_HEAVY,
        cdr3="ARGGYDFWSGYYTFDY", backend="fake_abnumber",
    )
    captured_calls: list[dict] = []

    def _iedb_capture(**kwargs):
        captured_calls.append({
            "filters": kwargs.get("filters"),
            "select": kwargs.get("select"),
        })
        return {"records": [{"receptor_group_id": 42}]}

    CandidateContextAgent(
        storage=local_storage,
        registry=registry_service,
        workflow_state=workflow_state_service,
        mcp_client=LocalMCPClient(bindings={
            "SAbDab_search_structures": lambda **_: {"hits": []},
            "ChEMBL_search_molecules": lambda **_: {"hits": []},
            "ChEMBL_search_substructure": lambda **_: {"hits": []},
            "iedb_search_bcr_sequences": _iedb_capture,
        }),
    ).run(run_id)

    assert captured_calls
    assert captured_calls[-1]["filters"] == {
        "chain1_cdr3_seq": "eq.ARGGYDFWSGYYTFDY"
    }
    cct = local_storage.read_json(
        local_storage.run_key(run_id, "candidate_context_table.json")
    )
    ab = next(
        c for c in cct["candidate_records"] if c["candidate_type"] == "antibody"
    )
    assert any(
        m["material_type"] == "antibody_heavy_chain_sequence"
        for m in ab["materials"]
    )
    blob = json.dumps(cct)
    assert _FULL_HEAVY not in blob
    assert "ARGGYDFWSGYYTFDY" not in blob


def test_iedb_bcr_default_path_reports_not_live_not_mocked(
    local_storage, registry_service, workflow_state_service, monkeypatch,
):
    run_id = _seed_run_with_antibody_sequence_material(
        local_storage, registry_service, workflow_state_service,
        sequence_text=_FULL_HEAVY, chain_role="antibody_heavy",
        filename="heavy_chain.fasta",
    )
    _mock_cdr3_success(
        monkeypatch, chain_type=CHAIN_TYPE_HEAVY,
        cdr3="ARGGYDFWSGYYTFDY", backend="fake_abnumber",
    )

    CandidateContextAgent(
        storage=local_storage,
        registry=registry_service,
        workflow_state=workflow_state_service,
        mcp_client=LocalMCPClient(),
    ).run(run_id)

    cct = local_storage.read_json(
        local_storage.run_key(run_id, "candidate_context_table.json")
    )
    calls = [
        tc for tc in cct.get("tool_call_records", [])
        if tc.get("tool_name") == "iedb_search_bcr_sequences"
    ]
    assert calls
    assert all(tc["run_status"] == "dependency_unavailable" for tc in calls)
    assert all(
        tc["tool_input_summary"].get("execution_semantics") == "not_live_mcp_execution"
        for tc in calls
    )
    for tc in calls:
        output = local_storage.read_json(tc["tool_output_ref"])["output"]
        assert output["status"] == "not_live"
        assert output["status"] != "mocked"
    blob = json.dumps(cct)
    assert _FULL_HEAVY not in blob
    assert "ARGGYDFWSGYYTFDY" not in blob


def test_uploaded_light_fasta_path_routes_iedb_without_artifact_patch(
    local_storage, registry_service, workflow_state_service, monkeypatch,
):
    run_id = _seed_run_with_antibody_sequence_material(
        local_storage, registry_service, workflow_state_service,
        sequence_text=_FULL_LIGHT, chain_role="antibody_light",
        filename="light_chain.fasta",
    )
    _mock_cdr3_success(
        monkeypatch, chain_type=CHAIN_TYPE_LIGHT, cdr3="QQHYTTPPT",
        backend="fake_abnumber",
    )
    captured_calls: list[dict] = []

    def _iedb_capture(**kwargs):
        captured_calls.append({
            "filters": kwargs.get("filters"),
            "select": kwargs.get("select"),
        })
        return {"records": [{"receptor_group_id": 43}]}

    CandidateContextAgent(
        storage=local_storage,
        registry=registry_service,
        workflow_state=workflow_state_service,
        mcp_client=LocalMCPClient(bindings={
            "SAbDab_search_structures": lambda **_: {"hits": []},
            "ChEMBL_search_molecules": lambda **_: {"hits": []},
            "ChEMBL_search_substructure": lambda **_: {"hits": []},
            "iedb_search_bcr_sequences": _iedb_capture,
        }),
    ).run(run_id)

    assert captured_calls
    assert captured_calls[-1]["filters"] == {
        "chain2_cdr3_seq": "eq.QQHYTTPPT"
    }
    cct = local_storage.read_json(
        local_storage.run_key(run_id, "candidate_context_table.json")
    )
    ab = next(
        c for c in cct["candidate_records"] if c["candidate_type"] == "antibody"
    )
    assert any(
        m["material_type"] == "antibody_light_chain_sequence"
        for m in ab["materials"]
    )
    blob = json.dumps(cct)
    assert _FULL_LIGHT not in blob
    assert "QQHYTTPPT" not in blob


@pytest.mark.skipif(
    shutil.which("hmmscan") is None,
    reason="HMMER hmmscan is required by ANARCI / abnumber",
)
def test_uploaded_heavy_fasta_path_with_real_numbering_reaches_iedb(
    local_storage, registry_service, workflow_state_service,
):
    pytest.importorskip("abnumber")
    run_id = _seed_run_with_antibody_sequence_material(
        local_storage, registry_service, workflow_state_service,
        sequence_text=_FULL_HEAVY, chain_role="antibody_heavy",
        filename="heavy_chain.fasta",
    )
    captured_calls: list[dict] = []

    def _iedb_capture(**kwargs):
        captured_calls.append({
            "filters": kwargs.get("filters"),
            "select": kwargs.get("select"),
        })
        return {"records": [{"receptor_group_id": 44}]}

    CandidateContextAgent(
        storage=local_storage,
        registry=registry_service,
        workflow_state=workflow_state_service,
        mcp_client=LocalMCPClient(bindings={
            "SAbDab_search_structures": lambda **_: {"hits": []},
            "ChEMBL_search_molecules": lambda **_: {"hits": []},
            "ChEMBL_search_substructure": lambda **_: {"hits": []},
            "iedb_search_bcr_sequences": _iedb_capture,
        }),
    ).run(run_id)

    assert captured_calls
    filters = captured_calls[-1]["filters"]
    assert set(filters) == {"chain1_cdr3_seq"}
    filter_value = filters["chain1_cdr3_seq"]
    assert filter_value.startswith("eq.")
    raw_cdr3 = filter_value.removeprefix("eq.")
    assert raw_cdr3
    cct = local_storage.read_json(
        local_storage.run_key(run_id, "candidate_context_table.json")
    )
    blob = json.dumps(cct)
    assert _FULL_HEAVY not in blob
    assert raw_cdr3 not in blob


def test_heavy_chain_success_routes_iedb_with_chain1_cdr3_seq_filter(
    local_storage, registry_service, workflow_state_service, monkeypatch,
):
    run_id = _seed_run_with_antibody_sequence_material(
        local_storage, registry_service, workflow_state_service,
        sequence_text=_FULL_HEAVY, chain_role="antibody_heavy",
    )
    # First agent run (no patch yet) — populates the antibody candidate
    # so we can rewrite its material_type before the real run we test.
    LocalMCPClient_bindings = {
        "SAbDab_search_structures": lambda **_: {"hits": []},
        "ChEMBL_search_molecules": lambda **_: {"hits": []},
        "ChEMBL_search_substructure": lambda **_: {"hits": []},
        "iedb_search_bcr_sequences": lambda **kwargs: {
            "filters_received": kwargs.get("filters"),
            "select_received": kwargs.get("select"),
            "records": [{
                "receptor_group_id": 42,
                "receptor_type": "IgG",
                "receptor_name": "fixture",
            }],
        },
    }
    mcp = LocalMCPClient(bindings=LocalMCPClient_bindings)
    CandidateContextAgent(
        storage=local_storage, registry=registry_service,
        workflow_state=workflow_state_service, mcp_client=mcp,
    ).run(run_id)
    _patch_chain_specific_material(
        local_storage, run_id,
        chain_role="antibody_heavy", sequence_text=_FULL_HEAVY,
    )

    _mock_cdr3_success(
        monkeypatch, chain_type=CHAIN_TYPE_HEAVY,
        cdr3="ARGGYDFWSGYYTFDY", backend="fake_abnumber",
    )

    captured_calls: list[dict] = []
    def _iedb_capture(**kwargs):
        captured_calls.append({"filters": kwargs.get("filters"),
                               "select": kwargs.get("select")})
        return {"records": [{"receptor_group_id": 42}]}
    mcp = LocalMCPClient(bindings={
        **LocalMCPClient_bindings,
        "iedb_search_bcr_sequences": _iedb_capture,
    })
    CandidateContextAgent(
        storage=local_storage, registry=registry_service,
        workflow_state=workflow_state_service, mcp_client=mcp,
    ).run(run_id)

    assert captured_calls, "IEDB should have been called"
    only = captured_calls[-1]
    assert only["filters"] == {"chain1_cdr3_seq": "eq.ARGGYDFWSGYYTFDY"}
    assert "chain1_cdr3_seq" in only["filters"]
    # select projection is the audited compact list, not "everything".
    assert isinstance(only["select"], list) and only["select"]
    assert "chain1_cdr3_seq" in only["select"]
    assert "chain2_cdr3_seq" in only["select"]


def test_light_chain_success_routes_iedb_with_chain2_cdr3_seq_filter(
    local_storage, registry_service, workflow_state_service, monkeypatch,
):
    run_id = _seed_run_with_antibody_sequence_material(
        local_storage, registry_service, workflow_state_service,
        sequence_text=_FULL_LIGHT, chain_role="antibody_light",
    )
    mcp_bindings = {
        "SAbDab_search_structures": lambda **_: {"hits": []},
        "ChEMBL_search_molecules": lambda **_: {"hits": []},
        "ChEMBL_search_substructure": lambda **_: {"hits": []},
        "iedb_search_bcr_sequences": lambda **_: {"records": []},
    }
    CandidateContextAgent(
        storage=local_storage, registry=registry_service,
        workflow_state=workflow_state_service,
        mcp_client=LocalMCPClient(bindings=mcp_bindings),
    ).run(run_id)
    _patch_chain_specific_material(
        local_storage, run_id,
        chain_role="antibody_light", sequence_text=_FULL_LIGHT,
    )
    _mock_cdr3_success(
        monkeypatch, chain_type=CHAIN_TYPE_LIGHT, cdr3="QQHYTTPPT",
    )

    captured: list[dict] = []
    def _iedb_capture(**kwargs):
        captured.append({"filters": kwargs.get("filters"),
                         "select": kwargs.get("select")})
        return {"records": []}
    CandidateContextAgent(
        storage=local_storage, registry=registry_service,
        workflow_state=workflow_state_service,
        mcp_client=LocalMCPClient(bindings={
            **mcp_bindings,
            "iedb_search_bcr_sequences": _iedb_capture,
        }),
    ).run(run_id)
    assert captured
    assert captured[-1]["filters"] == {"chain2_cdr3_seq": "eq.QQHYTTPPT"}


# ── 2. Numbering dependency unavailable / failure paths ─────────────────


def test_dependency_unavailable_does_not_call_iedb_and_records_gap(
    local_storage, registry_service, workflow_state_service, monkeypatch,
):
    run_id = _seed_run_with_antibody_sequence_material(
        local_storage, registry_service, workflow_state_service,
        sequence_text=_FULL_HEAVY, chain_role="antibody_heavy",
    )
    mcp_bindings = {
        "SAbDab_search_structures": lambda **_: {"hits": []},
        "ChEMBL_search_molecules": lambda **_: {"hits": []},
        "ChEMBL_search_substructure": lambda **_: {"hits": []},
        "iedb_search_bcr_sequences": lambda **_: {"records": []},
    }
    CandidateContextAgent(
        storage=local_storage, registry=registry_service,
        workflow_state=workflow_state_service,
        mcp_client=LocalMCPClient(bindings=mcp_bindings),
    ).run(run_id)
    _patch_chain_specific_material(
        local_storage, run_id,
        chain_role="antibody_heavy", sequence_text=_FULL_HEAVY,
    )
    _mock_cdr3_status(monkeypatch, status=STATUS_DEPENDENCY_UNAVAILABLE)

    iedb_calls: list[dict] = []
    def _iedb_capture(**kwargs):
        iedb_calls.append(kwargs)
        return {"records": []}
    CandidateContextAgent(
        storage=local_storage, registry=registry_service,
        workflow_state=workflow_state_service,
        mcp_client=LocalMCPClient(bindings={
            **mcp_bindings,
            "iedb_search_bcr_sequences": _iedb_capture,
        }),
    ).run(run_id)
    assert iedb_calls == []

    cct = local_storage.read_json(
        local_storage.run_key(run_id, "candidate_context_table.json")
    )
    ab = next(
        c for c in cct["candidate_records"] if c["candidate_type"] == "antibody"
    )
    assert any(
        "iedb_cdr3_extraction_dependency_unavailable" in g
        for g in ab["data_gaps"]
    ), ab["data_gaps"]


def test_extraction_failure_does_not_call_iedb_and_records_gap(
    local_storage, registry_service, workflow_state_service, monkeypatch,
):
    run_id = _seed_run_with_antibody_sequence_material(
        local_storage, registry_service, workflow_state_service,
        sequence_text=_FULL_HEAVY, chain_role="antibody_heavy",
    )
    mcp_bindings = {
        "SAbDab_search_structures": lambda **_: {"hits": []},
        "ChEMBL_search_molecules": lambda **_: {"hits": []},
        "ChEMBL_search_substructure": lambda **_: {"hits": []},
        "iedb_search_bcr_sequences": lambda **_: {"records": []},
    }
    CandidateContextAgent(
        storage=local_storage, registry=registry_service,
        workflow_state=workflow_state_service,
        mcp_client=LocalMCPClient(bindings=mcp_bindings),
    ).run(run_id)
    _patch_chain_specific_material(
        local_storage, run_id,
        chain_role="antibody_heavy", sequence_text=_FULL_HEAVY,
    )
    _mock_cdr3_status(monkeypatch, status=STATUS_EXTRACTION_FAILED)

    iedb_calls: list[dict] = []
    def _iedb_capture(**kwargs):
        iedb_calls.append(kwargs)
        return {"records": []}
    CandidateContextAgent(
        storage=local_storage, registry=registry_service,
        workflow_state=workflow_state_service,
        mcp_client=LocalMCPClient(bindings={
            **mcp_bindings,
            "iedb_search_bcr_sequences": _iedb_capture,
        }),
    ).run(run_id)
    assert iedb_calls == []
    cct = local_storage.read_json(
        local_storage.run_key(run_id, "candidate_context_table.json")
    )
    ab = next(
        c for c in cct["candidate_records"] if c["candidate_type"] == "antibody"
    )
    assert any(
        "iedb_cdr3_extraction_failed" in g for g in ab["data_gaps"]
    ), ab["data_gaps"]


# ── 3. Privacy: raw sequence + raw CDR3 never reach LLM / artifact ──────


def test_raw_sequence_and_raw_cdr3_never_reach_llm_or_artifact(
    local_storage, registry_service, workflow_state_service, monkeypatch,
):
    run_id = _seed_run_with_antibody_sequence_material(
        local_storage, registry_service, workflow_state_service,
        sequence_text=_FULL_HEAVY, chain_role="antibody_heavy",
    )
    mcp_bindings = {
        "SAbDab_search_structures": lambda **_: {"hits": []},
        "ChEMBL_search_molecules": lambda **_: {"hits": []},
        "ChEMBL_search_substructure": lambda **_: {"hits": []},
        "iedb_search_bcr_sequences": lambda **_: {"records": [
            {"receptor_group_id": 99, "receptor_type": "IgG"}
        ]},
    }
    CandidateContextAgent(
        storage=local_storage, registry=registry_service,
        workflow_state=workflow_state_service,
        mcp_client=LocalMCPClient(bindings=mcp_bindings),
    ).run(run_id)
    _patch_chain_specific_material(
        local_storage, run_id,
        chain_role="antibody_heavy", sequence_text=_FULL_HEAVY,
    )

    cdr3_raw = "ARGGYDFWSGYYTFDY"
    _mock_cdr3_success(monkeypatch, chain_type=CHAIN_TYPE_HEAVY, cdr3=cdr3_raw)

    captured_llm: list[dict] = []

    class _CapturingLLM:
        name = "capturing"
        model = "capturing-v1"

        def generate(self, prompt, *, system=None, **_):
            raise NotImplementedError

        def generate_json(self, prompt, *, schema, system=None):
            captured_llm.append({"schema": schema})
            return {"selections": []}

    CandidateContextAgent(
        storage=local_storage, registry=registry_service,
        workflow_state=workflow_state_service,
        mcp_client=LocalMCPClient(bindings=mcp_bindings),
        llm=_CapturingLLM(),
    ).run(run_id)

    # No LLM payload may carry the raw CDR3 or the full sequence.
    assert captured_llm
    for call in captured_llm:
        blob = json.dumps(call["schema"])
        assert cdr3_raw not in blob
        assert _FULL_HEAVY not in blob

    # No persisted normalized artifact may carry them either.
    cct = local_storage.read_json(
        local_storage.run_key(run_id, "candidate_context_table.json")
    )
    cct_blob = json.dumps(cct)
    assert cdr3_raw not in cct_blob
    assert _FULL_HEAVY not in cct_blob

    # tool_call_records exist for IEDB and carry redacted summary fields.
    iedb_records = [
        tc for tc in cct["tool_call_records"]
        if tc["tool_name"] == "iedb_search_bcr_sequences"
    ]
    assert iedb_records, "IEDB tool_call_record should be present"
    summary = iedb_records[0]["tool_input_summary"]
    assert summary["iedb_filter_key"] in {"chain1_cdr3_seq", "chain2_cdr3_seq"}
    assert summary["cdr3_chain_type"] == CHAIN_TYPE_HEAVY
    assert summary["cdr3_length"] == len(cdr3_raw)
    assert summary["cdr3_sha256_prefix"]
    assert summary["cdr3_source_material_id"]
    # tool_input_summary carries neither raw cdr3 nor full sequence.
    summary_blob = json.dumps(summary)
    assert cdr3_raw not in summary_blob
    assert _FULL_HEAVY not in summary_blob

    # Derived material on the candidate carries only redacted marker.
    ab = next(
        c for c in cct["candidate_records"] if c["candidate_type"] == "antibody"
    )
    derived = [
        m for m in ab["materials"]
        if m["material_type"] == "antibody_heavy_cdr3_sequence"
    ]
    assert derived, "derived heavy CDR3 material should be present"
    val = derived[0]["value"]
    assert val.startswith("[redacted:cdr3 ")
    assert cdr3_raw not in val
    assert derived[0]["value_format"] == "redacted_marker"
    assert derived[0]["role_status"] == "inferred"


# ── 4. Tool-outputs file: also no raw sequence ──────────────────────────


def test_tool_outputs_file_carries_redacted_input_only(
    local_storage, registry_service, workflow_state_service, monkeypatch,
):
    run_id = _seed_run_with_antibody_sequence_material(
        local_storage, registry_service, workflow_state_service,
        sequence_text=_FULL_HEAVY, chain_role="antibody_heavy",
    )
    mcp_bindings = {
        "SAbDab_search_structures": lambda **_: {"hits": []},
        "ChEMBL_search_molecules": lambda **_: {"hits": []},
        "ChEMBL_search_substructure": lambda **_: {"hits": []},
        "iedb_search_bcr_sequences": lambda **_: {"records": [{"x": 1}]},
    }
    CandidateContextAgent(
        storage=local_storage, registry=registry_service,
        workflow_state=workflow_state_service,
        mcp_client=LocalMCPClient(bindings=mcp_bindings),
    ).run(run_id)
    _patch_chain_specific_material(
        local_storage, run_id,
        chain_role="antibody_heavy", sequence_text=_FULL_HEAVY,
    )
    _mock_cdr3_success(monkeypatch, chain_type=CHAIN_TYPE_HEAVY,
                       cdr3="ARGGYDFWSGYYTFDY")
    CandidateContextAgent(
        storage=local_storage, registry=registry_service,
        workflow_state=workflow_state_service,
        mcp_client=LocalMCPClient(bindings=mcp_bindings),
    ).run(run_id)

    cct = local_storage.read_json(
        local_storage.run_key(run_id, "candidate_context_table.json")
    )
    iedb_records = [
        tc for tc in cct["tool_call_records"]
        if tc["tool_name"] == "iedb_search_bcr_sequences"
    ]
    assert iedb_records
    tc = iedb_records[0]
    assert tc["tool_output_ref"]
    output_doc = local_storage.read_json(tc["tool_output_ref"])
    blob = json.dumps(output_doc.get("input") or {})
    assert "ARGGYDFWSGYYTFDY" not in blob
    assert _FULL_HEAVY not in blob


# ── 5. Step 6 tools were not pulled into Step 5 scope ───────────────────


def test_step6_protein_tools_did_not_leak_into_step5_scope(
    local_storage, registry_service, workflow_state_service,
):
    """Confirm we did NOT widen the Step 5 MCP scope to bring in
    EBIProteins / PROSITE / GlyGen / iPTMnet / AlphaFold or any Step 6
    sequence-liability tool while implementing the CDR3 link."""
    from app.deps import get_mcp_client  # noqa: PLC0415
    forbidden = {
        "EBIProteins_get_features",
        "EBIProteins_get_epitopes",
        "PROSITE_scan_sequence",
        "GlyGen_lookup",
        "iPTMnet_lookup",
        "ProteinsPlus_profile_structure_quality",
        "AlphaFold_get_prediction",
    }
    # Reuse the same in-memory client the rest of the suite uses.
    intake = IntakeService(local_storage, registry_service, workflow_state_service)
    run_id = intake.allocate_run_id()
    intake.submit(
        run_id=run_id,
        raw_user_query="placeholder",
        user_provided_context={"target_or_antigen_text": "HER2",
                                "payload_linker_text": "vc-MMAE"},
    )
    StructuredQueryService(
        local_storage, registry_service, workflow_state_service,
        SupervisorAgent(llm=MockLLMProvider()),
    ).parse(run_id)
    InputReadinessService(
        local_storage, registry_service, workflow_state_service
    ).check(run_id)
    WorkflowSetupService(
        local_storage, registry_service, workflow_state_service
    ).plan(run_id)
    # `LocalMCPClient` doesn't enforce scope filtering at the
    # inventory level, but every fixture-driven Step 5 test in this
    # suite passes through the existing scope_filter in production
    # paths. Here we assert via the registry: registry-eligible tools
    # for an antibody candidate must not name any forbidden Step 6
    # tool (regardless of which scoped catalog is asserted).
    from app.agents.step_05_enrichment_registry import (
        STEP_05_CAPABILITY_REGISTRY,
    )
    registry_tool_names = {c.tool_name for c in STEP_05_CAPABILITY_REGISTRY}
    assert forbidden.isdisjoint(registry_tool_names)
