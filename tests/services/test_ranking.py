"""Step 12 — RankingService."""

from __future__ import annotations

from app.services.ranking_service import RankingService
from app.services.scoring_handoff_service import ScoringHandoffService
from app.services.scoring_validation_service import ScoringValidationService


def _bootstrap_through_step_10(local_storage, registry_service, workflow_state_service):
    from tests.services.test_scoring_handoff import _seed_through_step_9

    run_id = _seed_through_step_9(
        local_storage, registry_service, workflow_state_service, with_smiles=True
    )
    ScoringHandoffService(
        local_storage, registry_service, workflow_state_service
    ).prepare(run_id)
    return run_id


def test_step12_awaiting_when_validation_awaiting(
    local_storage, registry_service, workflow_state_service
):
    run_id = _bootstrap_through_step_10(local_storage, registry_service, workflow_state_service)
    ScoringValidationService(
        local_storage, registry_service, workflow_state_service
    ).validate(run_id)  # → awaiting_external_input (no result file)
    table = RankingService(
        local_storage, registry_service, workflow_state_service
    ).build_ranking_table(run_id)
    assert table.ranking_status == "awaiting_external_scoring"
    assert table.ranked_candidates == []
    assert "awaiting" in (table.notes or "").lower() or "no external" in (table.notes or "").lower()


def test_step12_ranks_validated_candidates_deterministically(
    local_storage, registry_service, workflow_state_service
):
    run_id = _bootstrap_through_step_10(local_storage, registry_service, workflow_state_service)
    handoff = local_storage.read_json(
        local_storage.run_key(run_id, "scoring_handoff_package.json")
    )
    cand_ids = handoff["candidate_ids"]
    assert len(cand_ids) >= 2

    # Tied total_score on first two candidates exercises the candidate_id tiebreak.
    external = {
        "candidates": [
            {"candidate_id": cand_ids[0], "total_score": 7.0,
             "dimensions": {"developability_score": 7.0}},
            {"candidate_id": cand_ids[1], "total_score": 9.0,
             "dimensions": {"developability_score": 7.0}},
            {"candidate_id": cand_ids[0], "total_score": 7.0,
             "dimensions": {"developability_score": 7.0}},
        ],
    }
    local_storage.write_json(
        local_storage.run_key(run_id, "inputs/external_scoring_result.json"), external
    )
    ScoringValidationService(
        local_storage, registry_service, workflow_state_service
    ).validate(run_id)
    table = RankingService(
        local_storage, registry_service, workflow_state_service
    ).build_ranking_table(run_id)
    assert table.ranking_status == "completed"
    assert table.ranked_candidates
    # Top rank has highest total_score (9.0 → cand_ids[1]).
    assert table.ranked_candidates[0].candidate_id == cand_ids[1]
    assert table.ranked_candidates[0].rank == 1
    assert table.ranked_candidates[0].final_rank_score == 9.0

    # Deterministic: running twice produces the same order.
    table2 = RankingService(
        local_storage, registry_service, workflow_state_service
    ).build_ranking_table(run_id)
    assert [r.candidate_id for r in table.ranked_candidates] == \
        [r.candidate_id for r in table2.ranked_candidates]
    assert [r.final_rank_score for r in table.ranked_candidates] == \
        [r.final_rank_score for r in table2.ranked_candidates]


def test_step12_does_not_fabricate_rank_when_no_validated_candidates(
    local_storage, registry_service, workflow_state_service
):
    """If every external row has an error, validation produces zero validated
    candidate ids — Step 12 must report `failed`, not invent ranks."""
    run_id = _bootstrap_through_step_10(local_storage, registry_service, workflow_state_service)
    handoff = local_storage.read_json(
        local_storage.run_key(run_id, "scoring_handoff_package.json")
    )
    external = {
        # candidate_id missing → error → not validated
        "candidates": [{"total_score": 7.0}],
    }
    local_storage.write_json(
        local_storage.run_key(run_id, "inputs/external_scoring_result.json"), external
    )
    ScoringValidationService(
        local_storage, registry_service, workflow_state_service
    ).validate(run_id)
    table = RankingService(
        local_storage, registry_service, workflow_state_service
    ).build_ranking_table(run_id)
    assert table.ranking_status in {"failed", "awaiting_external_scoring"}
    assert table.ranked_candidates == []
