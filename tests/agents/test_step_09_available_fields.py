"""Step 9 readiness projection hard-gate behavior.

These tests validate that official-tool required argument metadata is treated as
the primary gate signal (with legacy fallback only for missing metadata).
"""

from __future__ import annotations

from app.agents import step_09_available_fields as step9


def _seed_candidate_context_table() -> dict:
    return {
        "candidate_records": [
            {
                "candidate_id": "cand_t1",
                "candidate_type": "target_antigen",
                "materials": [
                    {
                        "material_id": "tgt_seq",
                        "material_type": "target_sequence",
                        "value": "MKTAYIAKQNNVG",
                    }
                ],
                "identifiers": [
                    {
                        "id_type": "uniprot_id",
                        "id_value": "P00533",
                    },
                ],
            },
            {
                "candidate_id": "cand_c1",
                "candidate_type": "compound_component",
                "materials": [
                    {
                        "material_id": "cmp_smiles",
                        "material_type": "payload_smiles",
                        "value": "CCO",
                    }
                ],
            },
        ]
    }


def _seed_ready_step8_structure_reference() -> dict:
    return {
        "candidate_structure_results": [
            {
                "candidate_id": "cand_t1",
                "downstream_handoff": {
                    "validated_structure_ref": "s3://tests/structure.pdb",
                },
            }
        ]
    }


def test_step9_hard_gate_schema_required_args_override_legacy_for_dynamut(monkeypatch):
    candidate_context = _seed_candidate_context_table()
    monkeypatch.setattr(
        step9,
        "_TOOL_REQUIRED_ARGS_CACHE",
        {},
        raising=False,
    )

    def _schema_for(name: str):
        if name == "DynaMut2_predict_stability":
            return {
                "type": "object",
                "properties": {
                    "structure_ref": {"type": "string"},
                    "mutation": {"type": "string"},
                },
                "required": ["structure_ref"],
            }
        return None

    monkeypatch.setattr(step9, "signature_schema_for", _schema_for)

    readiness = step9.project_step9_readiness(
        candidate_context_table=candidate_context,
        prepared_structure_input_package={},
        structure_prediction_and_interface_results={},
        compound_context_text="",
    )
    blocked = {
        entry.tool_name: entry.reason
        for entry in readiness["step9_hard_gate_blocked_tools_with_reason"]
    }
    assert blocked["DynaMut2_predict_stability"].startswith("schema_required:")
    assert "structure_ref" in blocked["DynaMut2_predict_stability"]


def test_step9_hard_gate_schema_required_args_satisfied_allows_tool(monkeypatch):
    candidate_context = _seed_candidate_context_table()
    # add explicit variant to satisfy mutation requirement.
    candidate_context["candidate_records"][0]["identifiers"].append(
        {"id_type": "mutation", "id_value": "p.V600E"}
    )

    monkeypatch.setattr(
        step9,
        "_TOOL_REQUIRED_ARGS_CACHE",
        {},
        raising=False,
    )

    def _schema_for(name: str):
        if name == "DynaMut2_predict_stability":
            return {
                "type": "object",
                "properties": {
                    "structure_ref": {"type": "string"},
                    "mutation": {"type": "string"},
                },
                "required": ["structure_ref", "mutation"],
            }
        return None

    monkeypatch.setattr(step9, "signature_schema_for", _schema_for)

    readiness = step9.project_step9_readiness(
        candidate_context_table=candidate_context,
        prepared_structure_input_package={},
        structure_prediction_and_interface_results=_seed_ready_step8_structure_reference(),
        compound_context_text="",
    )
    allowed = {entry.tool_name for entry in readiness["step9_hard_gate_allowed_tools"]}
    blocked = {entry.tool_name: entry.reason for entry in readiness["step9_hard_gate_blocked_tools_with_reason"]}

    assert "DynaMut2_predict_stability" in allowed
    assert "NvidiaNIM_rfdiffusion" in blocked
