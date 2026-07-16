"""MockLLMProvider deterministic parser tests."""

from __future__ import annotations

from app.llm.provider import MockLLMProvider


def _raw(query: str, ctx: dict | None = None) -> dict:
    return {
        "run_id": "run_x",
        "run_artifact_registry_id": "reg_x",
        "raw_user_query": query,
        "user_provided_context": ctx or {},
    }


def _parse(query: str, ctx: dict | None = None) -> dict:
    return MockLLMProvider().generate_json(
        "parse",
        schema={"raw_request_record": _raw(query, ctx)},
    )


# ── entity detection ────────────────────────────────────────────────────────

def test_mock_parser_detects_her2_and_mmae():
    out = _parse("Design an ADC against HER2 with vc-MMAE payload")
    assert out["mentioned_entities"]["target_or_antigen_text"] == "HER2"
    assert out["mentioned_entities"]["payload_text"] == "MMAE"
    assert out["mentioned_entities"]["linker_text"] == "vc"


def test_mock_parser_detects_pdb_id():
    out = _parse("Use PDB 1N8Z as the reference complex.")
    refs = {r["id_type"]: r["value"] for r in out["referenced_inputs"]}
    assert refs.get("pdb_id") == "1N8Z"


def test_mock_parser_detects_chembl_id_and_drugbank():
    out = _parse("Compound CHEMBL1201585 and reference DB00072 for context.")
    types = {r["id_type"] for r in out["referenced_inputs"]}
    assert "chembl_id" in types
    assert "drugbank_id" in types


def test_mock_parser_detects_zinc_without_defaulting_to_zinc22():
    out = _parse("Screen against ZINC12345678 library.")
    zinc = [r for r in out["referenced_inputs"] if r["id_type"] == "zinc_id"]
    assert zinc and zinc[0]["value"] == "ZINC12345678"
    # The parser must NOT label this as ZINC22 anywhere.
    assert all("ZINC22" not in r.get("value", "") for r in out["referenced_inputs"])


def test_mock_parser_detects_uniprot_accession():
    out = _parse("Target is HER2 (UniProt P04626).")
    types = {r["id_type"]: r["value"] for r in out["referenced_inputs"]}
    assert types.get("uniprot_id") == "P04626"


def test_mock_parser_detects_smiles_string():
    out = _parse("Use the warhead C(=O)CC1=CC=CC=C1 as payload reference")
    smiles = [r for r in out["referenced_inputs"] if r["id_type"] == "smiles"]
    assert smiles, "should detect at least one SMILES-like token"
    assert any("C1=CC=CC=C1" in r["value"] for r in smiles)


def test_mock_parser_preserves_user_constraints():
    out = _parse(
        "HER2 ADC with MMAE",
        ctx={"target_or_antigen_text": "HER2", "constraints_text": "Prefer DAR=4, avoid PEG"},
    )
    constraints = out["user_constraints"]
    assert constraints
    assert constraints[0]["constraint_text"] == "Prefer DAR=4, avoid PEG"
    assert constraints[0]["source"] == "user_provided_context.constraints_text"


def test_mock_parser_emits_warnings_for_missing_target():
    out = _parse("Just build an ADC.")
    assert out["mentioned_entities"]["target_or_antigen_text"] is None
    assert any("target" in w for w in out["parse_warnings"])


# ── tool-selection mocks ───────────────────────────────────────────────────

def test_mock_multilane_stage1_selects_all_matching_allowed_tools():
    allowed_tools = [
        "DrugProps_pains_filter",
        "DrugProps_lipinski_filter",
        "DrugProps_calculate_qed",
        "SwissADME_calculate_adme",
        "SwissADME_check_druglikeness",
    ]
    catalog = [
        {
            "tool_name": tool_name,
            "coarse_input_requirements": ["smiles"],
        }
        for tool_name in allowed_tools
    ]

    out = MockLLMProvider().generate_json(
        "pick",
        schema={
            "task": "tool_selection_stage_1_multi_lane",
            "compact_catalog": catalog,
            "lanes": [
                {
                    "lane_type": "payload_linker_compound_liability",
                    "allowed_tools": allowed_tools,
                    "signals": {"smiles": True},
                }
            ],
        },
    )

    selections = out["selections"]
    assert [entry["tool_name"] for entry in selections] == allowed_tools
    assert len(selections) == 5


def test_mock_step9_stage1_audits_selection_from_supplied_active_catalog():
    out = MockLLMProvider().generate_json(
        "pick",
        schema={
            "task": "step9_tool_selection_stage_1",
            "compact_catalog": [
                {
                    "tool_name": "ESM_generate_protein_sequence",
                    "lane_type": "protein_design",
                }
            ],
        },
    )

    selection = out["selections"][0]
    assert selection["selection_reason"] == (
        "mock selected Step 9 tool from the supplied active catalog"
    )
    normalized_reason = selection["selection_reason"].lower()
    assert "hard-gate" not in normalized_reason
    assert "hard gate" not in normalized_reason
    assert "readiness gate" not in normalized_reason
