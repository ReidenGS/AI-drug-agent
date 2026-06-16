"""Step 11 — ScoringValidationService."""

from __future__ import annotations

from app.services.scoring_handoff_service import ScoringHandoffService
from app.services.scoring_validation_service import ScoringValidationService


def _bootstrap_through_step_10(local_storage, registry_service, workflow_state_service):
    # Minimal Step 1-9 seed via importing the helper from test_scoring_handoff
    # would create a cross-module dependency; build the artifacts directly here.
    from tests.services.test_scoring_handoff import _seed_through_step_9

    run_id = _seed_through_step_9(
        local_storage, registry_service, workflow_state_service, with_smiles=True
    )
    ScoringHandoffService(
        local_storage, registry_service, workflow_state_service
    ).prepare(run_id)
    return run_id


def test_step11_awaiting_external_input_when_no_result_file(
    local_storage, registry_service, workflow_state_service
):
    run_id = _bootstrap_through_step_10(local_storage, registry_service, workflow_state_service)
    result = ScoringValidationService(
        local_storage, registry_service, workflow_state_service
    ).validate(run_id)
    assert result.validation_status == "awaiting_external_input"
    assert result.row_count == 0
    assert result.validated_candidate_ids == []
    assert result.external_scoring_input_ref is None
    assert "No external scoring result" in (result.notes or "")


def test_step11_validates_well_formed_scoring_result(
    local_storage, registry_service, workflow_state_service
):
    run_id = _bootstrap_through_step_10(local_storage, registry_service, workflow_state_service)

    # Pull candidate ids straight from the handoff package.
    handoff = local_storage.read_json(
        local_storage.run_key(run_id, "scoring_handoff_package.json")
    )
    cand_ids = handoff["candidate_ids"][:2]

    external = {
        "scored_at": "2026-06-15T12:00:00Z",
        "candidates": [
            {
                "candidate_id": cand_ids[0],
                "total_score": 7.4,
                "dimensions": {"docking_score": -8.2, "developability_score": 6.5},
            },
            {
                "candidate_id": cand_ids[1] if len(cand_ids) > 1 else cand_ids[0] + "_dup",
                "total_score": 5.1,
                "dimensions": {"developability_score": 4.0},
            },
        ],
    }
    local_storage.write_json(
        local_storage.run_key(run_id, "inputs/external_scoring_result.json"), external
    )
    result = ScoringValidationService(
        local_storage, registry_service, workflow_state_service
    ).validate(run_id)
    assert result.validation_status in {"completed", "completed_with_warnings"}
    assert result.row_count == 2
    assert cand_ids[0] in result.validated_candidate_ids
    # External input ref is recorded; raw rows are NOT inlined into artifact.
    assert result.external_scoring_input_ref


def test_step11_flags_missing_required_field_as_error(
    local_storage, registry_service, workflow_state_service
):
    run_id = _bootstrap_through_step_10(local_storage, registry_service, workflow_state_service)
    handoff = local_storage.read_json(
        local_storage.run_key(run_id, "scoring_handoff_package.json")
    )
    external = {
        "candidates": [
            # missing total_score → error
            {"candidate_id": handoff["candidate_ids"][0]},
            # bogus candidate_id → warning
            {"candidate_id": "candidate_does_not_exist", "total_score": 5.0},
            # out-of-range total_score → warning
            {"candidate_id": handoff["candidate_ids"][0], "total_score": 99.0},
        ],
    }
    local_storage.write_json(
        local_storage.run_key(run_id, "inputs/external_scoring_result.json"), external
    )
    result = ScoringValidationService(
        local_storage, registry_service, workflow_state_service
    ).validate(run_id)
    severities = {i.severity for i in result.issues}
    assert "error" in severities
    assert "warning" in severities
    assert result.validation_status in {"completed_with_warnings", "failed"}


def test_step11_does_not_embed_raw_rows_in_artifact(
    local_storage, registry_service, workflow_state_service
):
    run_id = _bootstrap_through_step_10(local_storage, registry_service, workflow_state_service)
    handoff = local_storage.read_json(
        local_storage.run_key(run_id, "scoring_handoff_package.json")
    )
    external = {
        "scored_at": "2026-06-15T12:00:00Z",
        "candidates": [
            {"candidate_id": handoff["candidate_ids"][0], "total_score": 7.4,
             "yufei_private_field_xyz": "should_not_leak"},
        ],
    }
    local_storage.write_json(
        local_storage.run_key(run_id, "inputs/external_scoring_result.json"), external
    )
    result = ScoringValidationService(
        local_storage, registry_service, workflow_state_service
    ).validate(run_id)
    import json
    assert "yufei_private_field_xyz" not in json.dumps(result.model_dump())
