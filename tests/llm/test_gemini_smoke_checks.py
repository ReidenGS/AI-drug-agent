from __future__ import annotations

import pytest

from app.llm.gemini_smoke_checks import (
    stage1_payload,
    stage2_payload,
    structured_query_payload,
    top_level_keys,
    validate_stage1,
    validate_stage2,
    validate_step1_6_artifacts,
    validate_structured_query,
)
from app.services.storage_local import LocalStorage


def test_structured_query_payload_and_validator():
    prompt, schema = structured_query_payload()

    assert "HER2" in prompt
    assert schema["raw_request_record"]["raw_user_query"]
    out = validate_structured_query(
        {
            "task_intent": {"task_type": "adc_design"},
            "mentioned_entities": {"target_or_antigen_text": "HER2"},
            "referenced_inputs": [],
            "parse_warnings": [],
        }
    )

    assert top_level_keys(out) == [
        "mentioned_entities",
        "parse_warnings",
        "referenced_inputs",
        "task_intent",
    ]


def test_stage1_payload_validator_rejects_out_of_catalog_selection():
    _, schema = stage1_payload()

    with pytest.raises(AssertionError, match="out-of-catalog"):
        validate_stage1({"selections": [{"tool_name": "NotInCatalog"}]}, schema)


def test_stage1_payload_validator_accepts_catalog_selection():
    _, schema = stage1_payload()

    out = validate_stage1({"selections": [{"tool_name": "DrugProps_pains_filter"}]}, schema)

    assert out["selections"][0]["tool_name"] == "DrugProps_pains_filter"


def test_stage2_payload_and_validator():
    _, schema = stage2_payload()

    assert schema["tool_name"] == "DrugProps_pains_filter"
    assert schema["context"]["arg_hints"] == {"smiles": "CCO"}
    out = validate_stage2({"arguments": {"smiles": "CCO"}})

    assert out["arguments"]["smiles"] == "CCO"


def test_step1_6_artifact_validator_reads_structured_query_from_inputs(tmp_path):
    storage = LocalStorage(root=str(tmp_path), prefix="adc_pilot")
    run_id = "run_smoke"
    artifacts = {
        "structured_query": "structured_query_ok",
        "structured_liability_summary": "structured_liability_summary_ok",
    }
    storage.write_json(
        storage.run_key(run_id, "inputs", "structured_query.json"),
        {"artifact_id": "structured_query_ok", "task_intent": {"task_type": "adc_design"}},
    )
    storage.write_json(
        storage.run_key(run_id, "structured_liability_summary.json"),
        {
            "artifact_id": "structured_liability_summary_ok",
            "candidate_liability_results": [
                {
                    "lane_results": [
                        {
                            "tool_call_records": [
                                {
                                    "tool_input_summary": {
                                        "selection_policy_version": "v1",
                                    }
                                }
                            ]
                        }
                    ]
                }
            ],
        },
    )

    out = validate_step1_6_artifacts(storage=storage, run_id=run_id, artifacts=artifacts)

    assert out["structured_query"]["artifact_id"] == "structured_query_ok"
    assert out["liability"]["artifact_id"] == "structured_liability_summary_ok"
