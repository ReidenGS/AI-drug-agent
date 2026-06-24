"""Step 2 SupervisorAgent prompt-hardening tests.

Coverage parity with the production prompt contract in
`\u9879\u76ee\u6587\u4ef6/Step1_4_Orchestration_Component_Plan_v0.1.md §Step 2`:

- Full ADC request (HER2 + trastuzumab + vc-MMAE).
- Target-only request.
- PDB / UniProt / ChEMBL / PubChem / DrugBank / ZINC ID extraction.
- Uploaded PDB / FASTA metadata is referenced by file_id.
- Ambiguous request flagged.
- Non-ADC request stays non-ADC (no force-coercion).
- `requested_outputs` alias normalization (`adc_candidate → ranked_candidates`).
- Unknown requested output dropped + parse_warnings entry.
- Step 2 never calls MCP / never includes ToolUniverse arguments.
- LLM prompt payload contains only `raw_user_query`,
  `user_provided_context`, and slim uploaded-file metadata — never raw
  file bytes, never `storage_path`, never registry IDs.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

from app.agents.supervisor_agent import (
    SUPERVISOR_SYSTEM_PROMPT,
    SupervisorAgent,
    _prompt_inputs_from_raw,
    build_supervisor_user_prompt,
)
from app.llm.gemini_provider import GeminiProvider
from app.llm.provider import MockLLMProvider
from app.schemas.step_01_raw_request_record import (
    RawRequestRecord,
    UploadedFile,
    UserProvidedContext,
)


_FIXTURE_RUN_ID = "run_supervisor_step2_fixture"


def _raw(
    *,
    query: str = "Design an ADC against HER2 with trastuzumab analog and vc-MMAE.",
    ctx: dict | None = None,
    files: list[dict] | None = None,
) -> dict:
    rec = RawRequestRecord(
        run_id=_FIXTURE_RUN_ID,
        run_artifact_registry_id="reg_step2",
        created_at="2026-06-18T00:00:00Z",
        raw_user_query=query,
        user_provided_context=UserProvidedContext(**(ctx or {})),
        uploaded_files=[UploadedFile(**f) for f in (files or [])],
    )
    out = rec.model_dump()
    out["artifact_id"] = "raw_request_record_test"
    return out


# ── deterministic mock path ────────────────────────────────────────────────


def test_step2_full_adc_request_extracts_target_candidate_payload():
    agent = SupervisorAgent(llm=MockLLMProvider())
    sq = agent.parse_raw_to_structured_query(
        _raw(
            ctx={
                "target_or_antigen_text": "HER2",
                "candidate_text": "Trastuzumab analog",
                "payload_linker_text": "vc-MMAE",
            }
        )
    )
    assert sq.task_intent.task_type == "adc_design"
    assert sq.task_intent.modality == "ADC"
    assert 0.0 <= sq.task_intent.task_type_confidence <= 1.0
    assert 0.0 <= sq.task_intent.modality_confidence <= 1.0
    assert sq.mentioned_entities.target_or_antigen_text == "HER2"
    assert sq.mentioned_entities.antibody_candidate_text == "Trastuzumab analog"
    assert sq.mentioned_entities.payload_text == "MMAE"
    assert sq.mentioned_entities.linker_text == "vc"


def test_step2_target_only_request_marks_payload_gap():
    agent = SupervisorAgent(llm=MockLLMProvider())
    sq = agent.parse_raw_to_structured_query(
        _raw(query="Build an ADC against HER2.", ctx={})
    )
    assert sq.mentioned_entities.target_or_antigen_text == "HER2"
    assert sq.mentioned_entities.payload_text is None
    assert any("payload" in w.lower() for w in sq.parse_warnings)


def test_step2_prompt_requires_component_canonical_name():
    assert "entity_decompositions[].components[]" in SUPERVISOR_SYSTEM_PROMPT
    assert "MUST use\n  `canonical_name`" in SUPERVISOR_SYSTEM_PROMPT
    assert "Do NOT use `component_name`" in SUPERVISOR_SYSTEM_PROMPT
    assert "`name`, `label`, or `value`" in SUPERVISOR_SYSTEM_PROMPT
    assert "when known, `component_type`" in SUPERVISOR_SYSTEM_PROMPT


def test_step2_prompt_requires_typed_smiles_referenced_inputs():
    sp = SUPERVISOR_SYSTEM_PROMPT
    assert "`mentioned_entities.payload_text`, `mentioned_entities.linker_text`, and" in sp
    assert "are NAME / LABEL fields" in sp
    assert "Do NOT put bare SMILES strings into `payload_text` or `linker_text`" in sp
    assert "`payload SMILES <value>`, `linker SMILES <value>`" in sp
    assert '"source": "payload_smiles"' in sp
    assert '"source": "linker_smiles"' in sp
    assert '"source": "compound_smiles"' in sp
    assert "If the name and SMILES appear" in sp
    assert "inconsistent, add a `parse_warnings` entry" in sp


def test_step2_prompt_typed_smiles_few_shot_keeps_smiles_out_of_name_fields():
    sp = SUPERVISOR_SYSTEM_PROMPT
    assert "Evaluate a TROP2 ADC with antibody sacituzumab analog" in sp
    assert "Payload SMILES C1=CC=C2C(=C1)C(=O)N(C)C3=CC=CC=C23" in sp
    assert "Linker SMILES NCCOC(=O)O" in sp
    assert '"payload_text": "SN-38 carbonate"' in sp
    assert '"linker_text": "SN-38 carbonate"' in sp
    assert (
        '{"id_type": "smiles", "value": '
        '"C1=CC=C2C(=C1)C(=O)N(C)C3=CC=CC=C23", "source": "payload_smiles"}'
    ) in sp
    assert (
        '{"id_type": "smiles", "value": "NCCOC(=O)O", '
        '"source": "linker_smiles"}'
    ) in sp
    assert '"payload_text": "C1=CC' not in sp
    assert '"linker_text": "NCCOC' not in sp


@pytest.mark.parametrize(
    "query,id_type,value",
    [
        ("Use PDB 1N8Z as the reference complex.", "pdb_id", "1N8Z"),
        ("Target HER2 (UniProt P04626).", "uniprot_id", "P04626"),
        ("Reference compound CHEMBL1201585.", "chembl_id", "CHEMBL1201585"),
        ("Use PubChem CID 2244 as a baseline.", "pubchem_cid", "2244"),
        ("Compare against DrugBank DB00072.", "drugbank_id", "DB00072"),
        ("Screen ZINC12345678 against my library.", "zinc_id", "ZINC12345678"),
    ],
)
def test_step2_extracts_explicit_ids(query, id_type, value):
    agent = SupervisorAgent(llm=MockLLMProvider())
    sq = agent.parse_raw_to_structured_query(_raw(query=query))
    ids = {(r.get("id_type"), r.get("value")) for r in sq.referenced_inputs}
    assert (id_type, value) in ids


def test_step2_zinc_id_never_labeled_zinc22():
    agent = SupervisorAgent(llm=MockLLMProvider())
    sq = agent.parse_raw_to_structured_query(
        _raw(query="Screen ZINC12345678 against my library.")
    )
    for ref in sq.referenced_inputs:
        assert "ZINC22" not in str(ref).upper().replace("ZINC2 ", "")
        # type is `zinc_id`, never `zinc22`.
        assert ref.get("id_type") != "zinc22"


def test_step2_extracts_uploaded_pdb_and_fasta_metadata():
    agent = SupervisorAgent(llm=MockLLMProvider())
    sq = agent.parse_raw_to_structured_query(
        _raw(
            query="HER2 ADC with attached structure and sequence files.",
            files=[
                {
                    "file_id": "f_pdb_001",
                    "original_filename": "1n8z.pdb",
                    "storage_path": "/store/runs/x/1n8z.pdb",
                    "content_type": "chemical/x-pdb",
                    "size_bytes": 12345,
                },
                {
                    "file_id": "f_fasta_002",
                    "original_filename": "trastuzumab.fasta",
                    "storage_path": "/store/runs/x/trastuzumab.fasta",
                    "content_type": "text/x-fasta",
                    "size_bytes": 678,
                },
            ],
        )
    )
    refs = {(r.get("id_type"), r.get("value")) for r in sq.referenced_inputs}
    assert ("uploaded_file", "f_pdb_001") in refs
    assert ("uploaded_file", "f_fasta_002") in refs


def test_step2_ambiguous_request_marks_warning_but_does_not_invent_target():
    agent = SupervisorAgent(llm=MockLLMProvider())
    sq = agent.parse_raw_to_structured_query(
        _raw(query="I'd like to look at ADCs in general — no specifics yet.")
    )
    assert sq.mentioned_entities.target_or_antigen_text is None
    assert sq.mentioned_entities.payload_text is None
    assert any("target" in w.lower() for w in sq.parse_warnings)
    assert any("payload" in w.lower() for w in sq.parse_warnings)


def test_step2_non_adc_request_is_not_forced_to_adc():
    agent = SupervisorAgent(llm=MockLLMProvider())
    sq = agent.parse_raw_to_structured_query(
        _raw(query="Please summarize patents on bispecific antibodies for me.")
    )
    # Mock provider should not declare this an ADC design task.
    assert sq.task_intent.modality != "ADC"
    assert sq.task_intent.task_type != "adc_design"
    assert any(
        "adc" in w.lower() for w in sq.parse_warnings
    ), "expected a parse_warnings entry flagging non-ADC request"


def test_step2_supervisor_does_not_call_mcp(monkeypatch):
    """Step 2 must not touch MCP — even if the agent has a client wired."""

    class _Sentinel:
        def __getattr__(self, name):
            raise AssertionError(
                f"Step 2 SupervisorAgent must NOT access mcp_client.{name}"
            )

    agent = SupervisorAgent(llm=MockLLMProvider(), mcp_client=_Sentinel())
    sq = agent.parse_raw_to_structured_query(_raw())
    assert sq.run_id == _FIXTURE_RUN_ID  # path completed without touching MCP.


def test_step2_output_has_no_tooluniverse_arguments():
    """Step 2 output is structured_query only — never tool args / schemas."""
    agent = SupervisorAgent(llm=MockLLMProvider())
    sq = agent.parse_raw_to_structured_query(_raw())
    blob = sq.model_dump_json()
    for forbidden in (
        "tool_name",
        "parameter_schema",
        "tooluniverse",
        "operation",  # would only appear if a TU arg leaked
        "compact_catalog",
    ):
        assert forbidden not in blob.lower(), (
            f"Step 2 structured_query unexpectedly contains {forbidden!r}: {blob}"
        )


# ── prompt-input shape: storage paths and bytes never reach the LLM ────────


def test_prompt_inputs_strip_storage_paths_and_pipeline_state():
    raw = _raw(
        ctx={"constraints_text": "Prefer DAR=4"},
        files=[
            {
                "file_id": "f1",
                "original_filename": "1n8z.pdb",
                "storage_path": "/secret/local/storage/1n8z.pdb",
                "content_type": "chemical/x-pdb",
                "sha256": "abcd1234",
                "size_bytes": 4096,
            }
        ],
    )
    payload = _prompt_inputs_from_raw(raw)
    assert set(payload) == {"raw_user_query", "user_provided_context", "uploaded_files"}
    assert payload["uploaded_files"][0]["file_id"] == "f1"
    # storage_path must NOT leak into the LLM prompt.
    blob = json.dumps(payload)
    assert "/secret/local/storage" not in blob
    assert "storage_path" not in blob
    # Pipeline bookkeeping must not leak either.
    for key in ("run_id", "run_artifact_registry_id", "step_id", "artifact_id", "intake_status"):
        assert key not in blob


def test_supervisor_user_prompt_does_not_mention_mcp_or_tooluniverse():
    prompt = build_supervisor_user_prompt(_raw())
    low = prompt.lower()
    assert "mcp" not in low
    assert "tooluniverse" not in low
    assert "tool_name" not in low


def test_supervisor_system_prompt_enumerates_canonical_outputs():
    sp = SUPERVISOR_SYSTEM_PROMPT
    for required_value in (
        "ranked_candidates",
        "report",
        "evidence_summary",
        "patent_or_ip_summary",
        "optimization_suggestions",
    ):
        assert required_value in sp, (
            f"system prompt missing canonical requested_output {required_value!r}"
        )


# ── alias normalization through GeminiProvider (monkeypatched, no network) ─


def _gemini_with_response(payload: dict) -> GeminiProvider:
    provider = GeminiProvider(api_key="fake-key", max_retries=0)
    response = SimpleNamespace(text=json.dumps(payload))
    provider._generate_content = lambda prompt: response  # type: ignore[method-assign]
    return provider


def test_step2_requested_outputs_adc_candidate_alias_normalized():
    """End-to-end through the supervisor: `adc_candidate` → `ranked_candidates`."""
    gemini = _gemini_with_response(
        {
            "task_intent": {
                "task_type": "adc_design",
                "task_type_confidence": 0.7,
                "modality": "ADC",
                "modality_confidence": 0.9,
                "user_goal_summary": "ADC design with shortlist",
            },
            "mentioned_entities": {
                "target_or_antigen_text": "HER2",
                "payload_text": "MMAE",
            },
            "referenced_inputs": [],
            "requested_outputs": ["adc_candidate", "ranked_candidates", "report"],
            "user_constraints": [],
            "parse_warnings": [],
        }
    )
    agent = SupervisorAgent(llm=gemini)
    sq = agent.parse_raw_to_structured_query(_raw())
    # Aliases collapsed and deduped.
    assert sq.requested_outputs == ["ranked_candidates", "report"]


def test_step2_unknown_requested_output_dropped_with_warning():
    gemini = _gemini_with_response(
        {
            "task_intent": {
                "task_type": "adc_design",
                "modality": "ADC",
            },
            "mentioned_entities": {},
            "referenced_inputs": [],
            "requested_outputs": ["report", "frobnication_summary"],
            "user_constraints": [],
            "parse_warnings": [],
        }
    )
    agent = SupervisorAgent(llm=gemini)
    sq = agent.parse_raw_to_structured_query(_raw())
    assert sq.requested_outputs == ["report"]
    assert any("frobnication_summary" in w for w in sq.parse_warnings)
    assert any("dropped" in w.lower() for w in sq.parse_warnings)


# ── schema passed to LLM has task=structured_query but no file bytes ───────


class _RecordingLLM:
    name = "recording"
    model = "recording-v1"

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def generate(self, prompt: str, *, system: str | None = None, **kw: Any) -> str:
        raise NotImplementedError

    def generate_json(self, prompt: str, *, schema: dict, system: str | None = None) -> dict:
        self.calls.append({"prompt": prompt, "schema": schema, "system": system})
        return {
            "task_intent": {"task_type": "adc_design", "modality": "ADC"},
            "mentioned_entities": {},
            "referenced_inputs": [],
            "requested_outputs": [],
            "user_constraints": [],
            "parse_warnings": [],
        }


def test_step2_llm_schema_carries_task_and_slim_prompt_inputs():
    llm = _RecordingLLM()
    agent = SupervisorAgent(llm=llm)
    agent.parse_raw_to_structured_query(
        _raw(
            files=[
                {
                    "file_id": "f1",
                    "original_filename": "1n8z.pdb",
                    "storage_path": "/secret/x/1n8z.pdb",
                    "content_type": "chemical/x-pdb",
                    "size_bytes": 4096,
                }
            ],
        )
    )
    assert llm.calls
    schema = llm.calls[0]["schema"]
    assert schema["task"] == "structured_query"
    prompt_inputs = schema["prompt_inputs"]
    assert set(prompt_inputs) == {
        "raw_user_query", "user_provided_context", "uploaded_files",
    }
    # storage_path stripped, bytes never present.
    blob = json.dumps(prompt_inputs)
    assert "storage_path" not in blob
    assert "/secret/x/" not in blob
