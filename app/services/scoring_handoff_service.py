"""Step 10 — ScoringHandoffService.

Deterministic. Aggregates Step 5/6/7/8/9 artifacts into a normalized handoff
package for the external Yufei AEE scoring module. No external service is
called; the package's `handoff_status` is `awaiting_external_scoring` until a
downstream scoring artifact is recorded (Step 11 / Step 12).

Raw MCP outputs are NOT carried in this package — only Step-level artifact
ids and normalized per-candidate summary fields.
"""

from __future__ import annotations

from ..schemas.step_10_scoring_handoff import (
    CandidateSummary,
    CompoundScreeningSummary,
    ScoringHandoff,
    SourceArtifactRefs,
    StructureConfidenceSummary,
)
from ..utils.errors import WorkflowStateError
from ..utils.ids import new_artifact_id
from ..utils.time import now_iso
from .artifact_registry_service import ArtifactRegistryService
from .storage_service import Storage
from .workflow_state_service import WorkflowStateService


_ARTIFACT_KEY = "scoring_handoff_package.json"


class ScoringHandoffService:
    def __init__(
        self,
        storage: Storage,
        registry: ArtifactRegistryService,
        workflow_state: WorkflowStateService,
    ) -> None:
        self.storage = storage
        self.registry = registry
        self.workflow_state = workflow_state

    def prepare(self, run_id: str) -> ScoringHandoff:
        reg = self.registry.get(run_id)
        active = reg.active_artifacts
        # We minimally require Step 5 (candidate context); Step 6/7/8/9 are
        # optional — if they're missing the package still goes out with empty
        # subsections so the external module sees the gap.
        if not active.candidate_context_table_id:
            raise WorkflowStateError("Step 10 requires Step 5 candidate_context_table")

        cct = self.storage.read_json(
            self.storage.run_key(run_id, "candidate_context_table.json")
        )
        sls = (
            self.storage.read_json(self.storage.run_key(run_id, "structured_liability_summary.json"))
            if active.structured_liability_summary_id
            else None
        )
        spi = (
            self.storage.read_json(
                self.storage.run_key(run_id, "structure_prediction_and_interface_results.json")
            )
            if active.structure_prediction_and_interface_results_id
            else None
        )
        cs = (
            self.storage.read_json(self.storage.run_key(run_id, "compound_screening_artifact.json"))
            if active.structure_variant_and_compound_screening_id
            else None
        )

        liability_by_cid = {
            r.get("candidate_id"): r
            for r in (sls or {}).get("candidate_liability_results", [])
        }
        structure_by_cid: dict[str, list[StructureConfidenceSummary]] = {}
        for cr in (spi or {}).get("candidate_structure_results", []):
            structure_by_cid.setdefault(cr.get("candidate_id"), []).append(
                StructureConfidenceSummary(
                    structure_input_id=cr.get("structure_input_id"),
                    run_status=cr.get("run_status"),
                    confidence_count=len(cr.get("structure_confidence_records") or []),
                    partial_run_flag=bool(cr.get("partial_run_flag")),
                )
            )
        compound_hits = (cs or {}).get("compound_hits", [])

        candidate_summaries: list[CandidateSummary] = []
        missing: list[str] = []
        if not spi:
            missing.append("structure_prediction_and_interface_results is missing")

        refs = SourceArtifactRefs(
            candidate_context_table_id=active.candidate_context_table_id,
            structured_liability_summary_id=active.structured_liability_summary_id,
            prepared_structure_input_package_id=active.prepared_structure_input_package_id,
            structure_prediction_and_interface_results_id=(
                active.structure_prediction_and_interface_results_id
            ),
            structure_variant_and_compound_screening_id=(
                active.structure_variant_and_compound_screening_id
            ),
        )
        for candidate in cct.get("candidate_records") or []:
            cid = candidate.get("candidate_id")
            liability = liability_by_cid.get(cid) or {}
            # Compound hits attach by best-effort label match on payload_name.
            cand_label = candidate.get("candidate_label") or ""
            attached_hits = []
            for hit in compound_hits:
                if cand_label and cand_label.lower() in (hit.get("smiles") or "").lower():
                    attached_hits.append(hit)
            candidate_summaries.append(
                CandidateSummary(
                    candidate_id=cid,
                    candidate_label=cand_label or None,
                    candidate_type=candidate.get("candidate_type"),
                    developability_label=liability.get("candidate_overall_liability_label"),
                    recommended_action=liability.get("recommended_action"),
                    structure_confidence=structure_by_cid.get(cid, []),
                    compound_screening=[
                        CompoundScreeningSummary(
                            compound_id=h.get("compound_id"),
                            source_library=h.get("source_library"),
                            source_database_version=h.get("source_database_version"),
                            source_tool_name=h.get("source_tool_name"),
                            source_runtime_status=h.get("source_runtime_status"),
                        )
                        for h in attached_hits or compound_hits  # all hits if no label match
                    ] if compound_hits else [],
                    source_artifact_refs=refs,
                )
            )

        handoff_status = (
            "partial" if missing or not candidate_summaries else "awaiting_external_scoring"
        )

        notes = (
            "Step 10 prepared a normalized handoff package for the external "
            "Yufei AEE scoring module. No scoring has been performed in-process. "
            f"Drop the result at `{run_id}/inputs/external_scoring_result.json` "
            "to unblock Step 11."
        )

        artifact_id = new_artifact_id("scoring_handoff_package")
        package_key = self.storage.run_key(run_id, _ARTIFACT_KEY)

        package = ScoringHandoff(
            run_id=run_id,
            created_at=now_iso(),
            handoff_status=handoff_status,  # type: ignore[arg-type]
            candidate_ids=[c.candidate_id for c in candidate_summaries],
            candidate_summaries=candidate_summaries,
            payload_storage_path=package_key,
            external_module="yufei_aee",
            expected_result_storage_path=(
                self.storage.run_key(run_id, "inputs/external_scoring_result.json")
            ),
            missing_inputs=missing,
            notes=notes,
        )

        self.storage.write_json(
            package_key, {"artifact_id": artifact_id, **package.model_dump()}
        )
        self.registry.update_active(run_id, scoring_handoff_id=artifact_id)
        self.workflow_state.mark(run_id, "step_10", "completed")
        return package
