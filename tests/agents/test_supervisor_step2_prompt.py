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
    sp = SUPERVISOR_SYSTEM_PROMPT
    assert "`entity_decompositions`" in sp
    assert "`components[].canonical_name`" in sp
    assert "`component_type`" in sp
    assert "`component_name`, `name`, `label`, or `value`" in sp


def test_step2_prompt_requires_typed_smiles_referenced_inputs():
    sp = SUPERVISOR_SYSTEM_PROMPT
    assert "Do not put\n  bare SMILES" in sp
    assert "Payload SMILES ->" in sp
    assert "Linker SMILES uses" in sp
    assert "Unlabeled compound SMILES uses" in sp
    assert '"source": "payload_smiles"' in sp
    assert '"source": "linker_smiles"' in sp
    assert "compound_smiles" in sp


def test_step2_prompt_typed_smiles_few_shot_keeps_smiles_out_of_name_fields():
    sp = SUPERVISOR_SYSTEM_PROMPT
    assert "Few-shot 1:" in sp
    assert "Design an ADC against HER2 / ERBB2 (UniProt P04626)" in sp
    assert "Payload SMILES CC(C)C[C@H](N(C)C(=O)C(C)C)" in sp
    assert "Linker SMILES\nNCCOC(=O)O" in sp
    assert '"payload_text": "vc-MMAE"' in sp
    assert '"linker_text": "vc-MMAE"' in sp
    assert (
        '{"id_type": "smiles", "value": '
        '"CC(C)C[C@H](N(C)C(=O)C(C)C)"'
    ) in sp
    assert (
        '{"id_type": "smiles", "value": "NCCOC(=O)O",\n'
        '     "source": "linker_smiles"}'
    ) in sp
    assert '"payload_text": "CC(' not in sp
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


def test_step2_explicit_pubchem_span_suppresses_only_overlapping_pdb_match():
    agent = SupervisorAgent(llm=MockLLMProvider())
    pubchem_only = agent.parse_raw_to_structured_query(
        _raw(query="Use PubChem CID 2244 as a baseline.")
    )
    assert pubchem_only.referenced_inputs == [
        {
            "id_type": "pubchem_cid",
            "value": "2244",
            "source": "raw_request_text",
        }
    ]

    mixed = agent.parse_raw_to_structured_query(
        _raw(query="PDB 1N8Z and PubChem CID 2244")
    )
    assert mixed.referenced_inputs == [
        {"id_type": "pdb_id", "value": "1N8Z", "source": "raw_request_text"},
        {
            "id_type": "pubchem_cid",
            "value": "2244",
            "source": "raw_request_text",
        },
    ]


def test_step2_non_overlapping_numeric_pdb_id_remains_allowed():
    agent = SupervisorAgent(llm=MockLLMProvider())
    structured = agent.parse_raw_to_structured_query(
        _raw(query="Use PDB 2244 as the reference structure.")
    )
    assert structured.referenced_inputs == [
        {"id_type": "pdb_id", "value": "2244", "source": "raw_request_text"}
    ]


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


def test_step2_prompt_states_narrow_lossless_role():
    sp = SUPERVISOR_SYSTEM_PROMPT
    assert "lossless parser and conservative normalizer" in sp
    assert "not a\ntool planner or biomedical reasoning step" in sp.lower()
    assert "never invent IDs, SMILES" in sp


def test_step2_prompt_uses_second_person_voice():
    """Wording cleanup: drop third-person 'the LLM' / 'the model' from
    the prompt body and stop repeating 'Step 2'. The opening role
    sentence is allowed; the rest of the prompt addresses the model
    directly."""
    sp = SUPERVISOR_SYSTEM_PROMPT
    assert "the LLM" not in sp
    assert "the model" not in sp
    # Opening role sentence is in second person, no "Step-2" tag.
    assert sp.startswith("You are the ADC pipeline structured-query parser.")
    # Body no longer repeats "Step 2" as a subject.
    assert "Step 2" not in sp
    # Body uses imperatives / "You must" / "Do not" framings.
    assert "You must" in sp
    assert "Do not" in sp


def test_step2_prompt_lists_four_field_responsibilities():
    sp = SUPERVISOR_SYSTEM_PROMPT
    assert "Use these fields:" in sp
    assert "`mentioned_entities`: literal user labels only" in sp
    assert "`normalized_entities`: canonical forms" in sp
    assert "`entity_decompositions`: component breakdowns" in sp
    assert "`referenced_inputs`: explicit typed inputs only" in sp


def test_step2_prompt_closes_normalized_entity_type_enum():
    sp = SUPERVISOR_SYSTEM_PROMPT
    allowed = {
        "target_or_antigen",
        "disease_or_indication",
        "antibody",
        "payload",
        "linker",
        "linker_payload",
        "drug",
        "compound",
        "protein_variant",
        "other",
    }
    entity_type_rule = sp.split("`entity_type` must be exactly one of", 1)[1].split(
        ". Never invent", 1
    )[0]
    assert {value for value in allowed if f"`{value}`" in entity_type_rule} == allowed
    assert "Never invent, extend,\n  or return any other `entity_type` value." in sp


def test_step2_prompt_adc_combination_rules_for_vc_mmae_and_t_dm1():
    sp = SUPERVISOR_SYSTEM_PROMPT
    assert "Composite ADC terms" in sp
    assert "vc-MMAE" in sp and "T-DM1" in sp and "T-DXd" in sp and "Enhertu" in sp
    assert "stay literal in `mentioned_entities`" in sp
    assert "valine-citrulline linker + monomethyl" in sp
    assert "trastuzumab antibody + DM1 payload" in sp


def test_step2_prompt_forbids_isolated_low_information_vc_alias():
    sp = SUPERVISOR_SYSTEM_PROMPT
    assert "Do not emit isolated `vc`" in sp
    assert "unless\n  the user wrote it as a standalone linker" in sp


def test_step2_prompt_typed_identifier_rules_for_uniprot_pdb_uploaded_smiles():
    sp = SUPERVISOR_SYSTEM_PROMPT
    assert "Input-to-output mapping:" in sp
    assert "Explicit UniProt accession" in sp
    assert "`P04626`" in sp
    assert "Explicit PDB ID `7XYZ`" in sp
    assert "Uploaded file metadata with `file_id`" in sp
    assert '"source": "uploaded_file"' in sp
    assert "Payload SMILES" in sp
    assert "Linker SMILES" in sp
    assert "Unlabeled compound SMILES" in sp


def test_step2_prompt_carries_her2_vc_mmae_few_shot():
    sp = SUPERVISOR_SYSTEM_PROMPT
    assert "Few-shot 1:" in sp
    assert "Design an ADC against HER2 / ERBB2 (UniProt P04626) using" in sp
    assert "Payload SMILES CC(C)C[C@H](N(C)C(=O)C(C)C)" in sp
    assert "Linker SMILES\nNCCOC(=O)O" in sp
    assert '"payload_text": "vc-MMAE"' in sp
    assert '"linker_text": "vc-MMAE"' in sp
    assert (
        '{"id_type": "uniprot_id", "value": "P04626", "source": "user"}'
        in sp
    )
    assert (
        '{"id_type": "smiles", "value": "CC(C)C[C@H](N(C)C(=O)C(C)C)",\n'
        '     "source": "payload_smiles"}'
    ) in sp
    assert (
        '{"id_type": "uploaded_file", "value": "f_pdb_001",\n'
        '     "source": "uploaded_file"}'
    ) in sp
    # The few-shot must NOT teach the LLM to leak SMILES into name fields
    # or to emit a bare "vc" as a standalone linker_text.
    assert '"payload_text": "CC(' not in sp
    assert '"linker_text": "NCCOC' not in sp
    assert '"linker_text": "vc"' not in sp


def test_step2_prompt_few_shot_explains_normalizer_invariants():
    sp = SUPERVISOR_SYSTEM_PROMPT
    assert "Notes the LLM must follow" not in sp
    assert '{"id_type": "uniprot_id", "value": "P04626", "source": "user"}' in sp
    assert "`components[].canonical_name`" in sp
    assert "`component_name`, `name`, `label`, or `value`" in sp


def test_step2_prompt_boundary_block_forbids_downstream_outputs():
    sp = SUPERVISOR_SYSTEM_PROMPT
    for forbidden_phrase in (
        "generated ADC candidates",
        "candidate\nrankings",
        "liability flags",
        "tool plans, MCP\nselections",
        "literature or patent queries",
        "pose ensembles",
        "DAR designs",
    ):
        assert forbidden_phrase in sp, forbidden_phrase


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


# ── Antibody heavy / light chain prompt + normalizer tests ──────────────


def test_step2_prompt_lists_heavy_chain_signal_keywords():
    sp = SUPERVISOR_SYSTEM_PROMPT
    sp_lower = sp.lower()
    for hint in ("heavy chain", "VH", "HC", "IGH", "IGHV"):
        assert hint.lower() in sp_lower, hint
    assert "antibody_heavy_chain_sequence" in sp


def test_step2_prompt_lists_light_chain_signal_keywords():
    sp = SUPERVISOR_SYSTEM_PROMPT
    sp_lower = sp.lower()
    for hint in ("light chain", "VL", "LC", "kappa", "lambda",
                 "IGK", "IGL", "IGKV", "IGLV"):
        assert hint.lower() in sp_lower, hint
    assert "antibody_light_chain_sequence" in sp


def test_step2_prompt_forbids_default_to_heavy_when_chain_silent():
    sp = SUPERVISOR_SYSTEM_PROMPT
    assert "If an antibody FASTA/sequence lacks heavy/light role" in sp
    assert "do not use\n  `antibody_sequence_reference`" in sp
    assert "report blocking `sequence_role`" in sp


def test_step2_prompt_reports_sequence_role_only_for_ambiguous_uploaded_fasta():
    sp = SUPERVISOR_SYSTEM_PROMPT
    assert "blocking `sequence_role`" in sp
    assert "uploaded FASTA/sequence file exists" in sp
    assert "user or explicit file `role`/`chain_role`" in sp
    assert "Never infer this role from\n  filename tokens or sequence content" in sp
    assert "Do not emit it when no FASTA exists" in sp
    assert "or the role is clear" in sp


def test_step2_prompt_forbids_inferring_chain_from_sequence_content():
    sp = SUPERVISOR_SYSTEM_PROMPT
    assert "Do not infer heavy vs light from sequence content." in sp


def test_step2_prompt_forbids_extracting_cdr3_at_parse_time():
    sp = SUPERVISOR_SYSTEM_PROMPT
    assert "Do not extract CDR3\n  from a full sequence." in sp
    assert "antibody_cdr3_sequence" in sp


def test_step2_prompt_heavy_light_few_shot_uses_file_ids_only():
    sp = SUPERVISOR_SYSTEM_PROMPT
    assert "Few-shot 2:" in sp
    assert "heavy_chain.fasta and\nlight_chain.fasta" in sp
    assert (
        '{"id_type": "uploaded_file", "value": "f_heavy_001",\n'
        '     "source": "antibody_heavy_chain_sequence"}'
    ) in sp
    assert (
        '{"id_type": "uploaded_file", "value": "f_light_002",\n'
        '     "source": "antibody_light_chain_sequence"}'
    ) in sp
    # Raw FASTA / storage paths / filenames must not appear in the few-shot.
    assert "EVQLVQSGAEVKKPGSSVKVSCKAS" not in sp
    assert "/store/runs/" not in sp
    assert '"value": "heavy_chain.fasta"' not in sp
    assert '"value": "light_chain.fasta"' not in sp


def test_step2_referenced_inputs_id_type_enum_includes_chain_signals():
    sp = SUPERVISOR_SYSTEM_PROMPT
    assert "antibody_heavy_chain_sequence" in sp
    assert "antibody_light_chain_sequence" in sp
    assert "antibody_sequence_reference" in sp
    assert "antibody_cdr3_sequence" in sp


# ── Normalizer behaviour ─────────────────────────────────────────────────


def _normalize(payload: dict) -> dict:
    from app.agents.supervisor_agent import normalize_llm_payload_for_step2
    return normalize_llm_payload_for_step2(payload)


def test_normalizer_promotes_heavy_chain_alias_id_type():
    payload = {
        "referenced_inputs": [
            {"id_type": "heavy_chain_sequence",
             "value": "EVQLVQSGAEVKKPGSSVKVSCKAS"},
        ],
        "parse_warnings": [],
    }
    out = _normalize(payload)
    assert out["referenced_inputs"][0]["id_type"] == "antibody_heavy_chain_sequence"
    assert out["referenced_inputs"][0]["source"] == "user"
    assert any("heavy_chain_sequence" in w for w in out["parse_warnings"])


def test_normalizer_promotes_vh_and_hc_alias_id_types():
    for alias in ("vh_sequence", "hc_sequence"):
        out = _normalize({"referenced_inputs": [
            {"id_type": alias, "value": "EVQLVQSGAEVKKPG"},
        ], "parse_warnings": []})
        assert out["referenced_inputs"][0]["id_type"] == (
            "antibody_heavy_chain_sequence"
        ), alias


def test_normalizer_promotes_light_chain_alias_id_types():
    for alias in ("light_chain_sequence", "vl_sequence", "lc_sequence",
                  "kappa_sequence", "lambda_sequence"):
        out = _normalize({"referenced_inputs": [
            {"id_type": alias, "value": "DIQMTQSPSSLSASVGD"},
        ], "parse_warnings": []})
        assert out["referenced_inputs"][0]["id_type"] == (
            "antibody_light_chain_sequence"
        ), alias


def test_normalizer_keeps_chain_silent_sequence_aliases_non_executable():
    """Chain-silent aliases never invent heavy/light or target authority."""
    expected = {
        "antibody_sequence": "antibody_sequence_reference",
        "protein_sequence": "protein_sequence",
        "fasta_sequence": "fasta_sequence",
        "amino_acid_sequence": "amino_acid_sequence",
    }
    for alias, canonical in expected.items():
        out = _normalize({"referenced_inputs": [
            {"id_type": alias, "value": "EVQLVQSGAEVKKPGSSVKVSCKAS"},
        ], "parse_warnings": []})
        new_type = out["referenced_inputs"][0]["id_type"]
        assert new_type == canonical, (alias, new_type)
        assert "antibody_heavy_chain_sequence" not in new_type
        assert "antibody_light_chain_sequence" not in new_type


def test_normalizer_repairs_uploaded_file_drifted_source_for_heavy_and_light():
    payload = {
        "referenced_inputs": [
            {"id_type": "uploaded_file", "value": "f_heavy_001",
             "source": "vh_sequence"},
            {"id_type": "uploaded_file", "value": "f_light_002",
             "source": "kappa_sequence"},
            {"id_type": "uploaded_file", "value": "f_generic_003",
             "source": "protein_sequence"},
            {"id_type": "uploaded_file", "value": "f_silent_004",
             "source": "uploaded_file"},
        ],
        "parse_warnings": [],
    }
    out = _normalize(payload)
    refs = out["referenced_inputs"]
    assert refs[0]["source"] == "antibody_heavy_chain_sequence"
    assert refs[1]["source"] == "antibody_light_chain_sequence"
    assert refs[2]["source"] == "protein_sequence"
    # Chain-silent uploaded_file source stays as the existing value.
    assert refs[3]["source"] == "uploaded_file"
    # file_id values untouched.
    assert refs[0]["value"] == "f_heavy_001"
    assert refs[1]["value"] == "f_light_002"
    assert refs[2]["value"] == "f_generic_003"
    assert refs[3]["value"] == "f_silent_004"


def test_normalizer_is_idempotent_on_canonical_payload():
    payload = {
        "referenced_inputs": [
            {"id_type": "uploaded_file", "value": "f_heavy_001",
             "source": "antibody_heavy_chain_sequence"},
            {"id_type": "antibody_light_chain_sequence",
             "value": "DIQMTQSPSSLSASVGD", "source": "user"},
        ],
        "parse_warnings": [],
    }
    out = _normalize(payload)
    assert out["referenced_inputs"][0]["source"] == "antibody_heavy_chain_sequence"
    assert out["referenced_inputs"][1]["id_type"] == "antibody_light_chain_sequence"
    assert out["referenced_inputs"][1]["source"] == "user"
    assert out["parse_warnings"] == []


# ── prompt_sequence (ESM masked generation prompt) prompt text ────────────


def test_step2_prompt_states_prompt_sequence_requirement_concisely():
    sp = SUPERVISOR_SYSTEM_PROMPT
    assert "prompt_sequence" in sp
    assert '"id_type": "prompt_sequence"' in sp
    assert "masked generation prompt" in sp
    assert 'mask positions such as\n  "_" or "<mask>"' in sp
    assert "never an ordinary complete heavy/light/target sequence" in sp
    assert "blocking `missing_slots` slot \"prompt_sequence\"" in sp


def test_step2_prompt_describes_conditional_prompt_sequence_slot():
    sp = SUPERVISOR_SYSTEM_PROMPT
    assert "conditional prompt_sequence" in sp
    assert "blocking prompt_sequence ONLY when the user asks to generate" in sp
    assert "does NOT satisfy this slot" in sp


def test_step2_prompt_does_not_ask_llm_to_inspect_uploaded_file_content():
    """The LLM never sees uploaded-file bytes — the prompt must not imply
    it should inspect file content or check for mask markers inside a file
    to decide whether something is a prompt_sequence."""
    sp_lower = SUPERVISOR_SYSTEM_PROMPT.lower()
    for forbidden in (
        "inspect uploaded file content",
        "inspect the uploaded file",
        "if the file contains",
        "check header/content",
        "check the file content",
        "read the uploaded file",
        "read the file content",
    ):
        assert forbidden not in sp_lower, forbidden


def test_step2_referenced_inputs_enum_includes_prompt_sequence():
    sp = SUPERVISOR_SYSTEM_PROMPT
    assert "prompt_sequence" in sp
    assert '"source": "prompt_sequence"' in sp


# ── prompt_sequence normalizer behaviour ────────────────────────────────────


def test_normalizer_prompt_sequence_id_type_not_swept_into_generic_antibody_alias():
    """`prompt_sequence` must stay its own id_type — it must never be
    coerced into `antibody_sequence_reference` alongside the generic
    `protein_sequence` / `fasta_sequence` aliases."""
    out = _normalize(
        {
            "referenced_inputs": [
                {"id_type": "prompt_sequence", "value": "MKT_YIAKQNNVGA", "source": "user"},
            ],
            "parse_warnings": [],
        }
    )
    assert out["referenced_inputs"][0]["id_type"] == "prompt_sequence"
