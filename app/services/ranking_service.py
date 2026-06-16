"""Step 12 — deterministic ranking.

Reads Step 10 handoff and Step 11 validation. Behaviour:
- If Step 11 is `awaiting_external_input`, Step 12 emits a table with
  `ranking_status="awaiting_external_scoring"` and **no fabricated ranks**.
- If Step 11 is `completed` / `completed_with_warnings` and at least one
  candidate validated, Step 12 reads the external scoring file directly and
  sorts validated candidates by `total_score` descending. Ties broken by
  `candidate_id` ascending so results are repeatable.
- No LLM, no MCP.
"""

from __future__ import annotations

from ..schemas.step_12_ranking_table import RankedCandidate, RankingTable
from ..utils.errors import WorkflowStateError
from ..utils.ids import new_artifact_id
from ..utils.time import now_iso
from .artifact_registry_service import ArtifactRegistryService
from .storage_service import Storage
from .workflow_state_service import WorkflowStateService


_ARTIFACT_KEY = "ranking_table.json"


class RankingService:
    def __init__(
        self,
        storage: Storage,
        registry: ArtifactRegistryService,
        workflow_state: WorkflowStateService,
    ) -> None:
        self.storage = storage
        self.registry = registry
        self.workflow_state = workflow_state

    def build_ranking_table(self, run_id: str) -> RankingTable:
        reg = self.registry.get(run_id)
        if not reg.active_artifacts.scoring_validation_id:
            raise WorkflowStateError("Step 12 requires Step 11 scoring_validation")

        validation = self.storage.read_json(
            self.storage.run_key(run_id, "scoring_validation.json")
        )
        status = validation.get("validation_status")
        if status == "awaiting_external_input":
            return self._persist(
                run_id,
                RankingTable(
                    run_id=run_id,
                    created_at=now_iso(),
                    ranking_status="awaiting_external_scoring",
                    ranked_candidates=[],
                    shortlist_size=0,
                    storage_path="",
                    source_scoring_validation_id=reg.active_artifacts.scoring_validation_id,
                    notes=(
                        "No external scoring result yet — ranking deferred. "
                        "When the Yufei AEE module drops a result file, re-run "
                        "Step 11 then Step 12."
                    ),
                ),
            )

        validated_ids = set(validation.get("validated_candidate_ids") or [])
        if not validated_ids:
            return self._persist(
                run_id,
                RankingTable(
                    run_id=run_id,
                    created_at=now_iso(),
                    ranking_status="failed",
                    ranked_candidates=[],
                    shortlist_size=0,
                    storage_path="",
                    source_scoring_validation_id=reg.active_artifacts.scoring_validation_id,
                    notes="Step 11 produced no validated candidates; cannot rank.",
                ),
            )

        input_ref = validation.get("external_scoring_input_ref")
        if not input_ref or not self.storage.exists(input_ref):
            return self._persist(
                run_id,
                RankingTable(
                    run_id=run_id,
                    created_at=now_iso(),
                    ranking_status="failed",
                    ranked_candidates=[],
                    shortlist_size=0,
                    storage_path="",
                    source_scoring_validation_id=reg.active_artifacts.scoring_validation_id,
                    notes="External scoring input file referenced by Step 11 is missing.",
                ),
            )

        external = self.storage.read_json(input_ref)
        rows = [
            r for r in (external.get("candidates") or [])
            if r.get("candidate_id") in validated_ids
            and _is_number(r.get("total_score"))
        ]
        # Descending by total_score; stable tiebreak by candidate_id.
        rows.sort(key=lambda r: (-float(r["total_score"]), r.get("candidate_id") or ""))
        ranked = [
            RankedCandidate(
                rank=i + 1,
                candidate_id=row["candidate_id"],
                final_rank_score=float(row["total_score"]),
                notes=row.get("notes"),
            )
            for i, row in enumerate(rows)
        ]

        return self._persist(
            run_id,
            RankingTable(
                run_id=run_id,
                created_at=now_iso(),
                ranking_status="completed",
                ranked_candidates=ranked,
                shortlist_size=len(ranked),
                storage_path=self.storage.run_key(run_id, _ARTIFACT_KEY),
                source_scoring_validation_id=reg.active_artifacts.scoring_validation_id,
                notes=None,
            ),
        )

    def _persist(self, run_id: str, table: RankingTable) -> RankingTable:
        artifact_id = new_artifact_id("ranking_table")
        self.storage.write_json(
            self.storage.run_key(run_id, _ARTIFACT_KEY),
            {"artifact_id": artifact_id, **table.model_dump()},
        )
        self.registry.update_active(run_id, ranking_table_id=artifact_id)
        self.workflow_state.mark(run_id, "step_12", "completed")
        return table


def _is_number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)
