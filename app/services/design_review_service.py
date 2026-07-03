"""Step 16 scaffold service — write report metadata without calling an LLM."""

from __future__ import annotations

from ..schemas.step_16_llm_design_review_report import (
    DesignReviewReportFile,
    DesignReviewReportMetadata,
    DesignReviewWarning,
    LLMCallRecord,
    LLMDesignReviewReport,
)
from ..utils.ids import new_artifact_id, new_tool_call_id
from ..utils.time import now_iso
from .artifact_registry_service import ArtifactRegistryService
from .downstream_scaffold_utils import active_artifact_refs, missing_active_refs, read_json_if_exists, safe_workflow_mark
from .storage_service import Storage
from .workflow_state_service import WorkflowStateService


_ARTIFACT_KEY = "llm_design_review_report.json"
_REPORT_KEY = "reports/step_16_design_review_scaffold.md"


class DesignReviewService:
    def __init__(
        self,
        *,
        storage: Storage,
        registry: ArtifactRegistryService,
        workflow_state: WorkflowStateService,
    ) -> None:
        self.storage = storage
        self.registry = registry
        self.workflow_state = workflow_state

    def create_report(self, run_id: str) -> LLMDesignReviewReport:
        refs = active_artifact_refs(self.registry, run_id)
        ip_shortlist = read_json_if_exists(self.storage, run_id, "ip_risk_integrated_shortlist.json")
        candidate_count = len((ip_shortlist or {}).get("candidate_ip_risk_records") or [])
        warnings = [
            _warning_for_missing(label)
            for _, label in missing_active_refs(
                self.registry,
                run_id,
                {
                    "ip_risk_integrated_shortlist_id": "ip_risk_integrated_shortlist",
                    "ranking_table_id": "ranked_candidate_shortlist",
                    "candidate_context_table_id": "candidate_context_table",
                    "scientific_evidence_table_id": "scientific_evidence_table",
                    "patent_prior_art_table_id": "patent_prior_art_table",
                    "structured_query_id": "structured_query",
                    "run_step_plan_id": "run_step_plan",
                },
            )
        ]
        report_artifact_id = new_artifact_id("llm_design_review_report_file")
        generated_at = now_iso()
        report_key = self.storage.run_key(run_id, _REPORT_KEY)
        report_text = _render_scaffold_report(run_id, generated_at, refs, candidate_count)
        self.storage.write_bytes(report_key, report_text.encode("utf-8"))

        llm_call = LLMCallRecord(
            llm_call_id=new_tool_call_id(),
            input_artifact_refs=list(refs.values()),
            run_status="skipped",
            failure_reason="scaffold_only: real LLM generation intentionally deferred",
        )
        artifact = LLMDesignReviewReport(
            run_id=run_id,
            created_at=generated_at,
            design_review_status=("partial" if warnings else "completed_with_warnings"),
            report_file=DesignReviewReportFile(
                report_artifact_id=report_artifact_id,
                storage_ref=report_key,
                generated_at=generated_at,
            ),
            report_metadata=DesignReviewReportMetadata(
                candidate_count=candidate_count,
                included_section_count=7,
            ),
            llm_call_records=[llm_call],
            design_review_warnings=warnings,
            design_review_notes=(
                "Scaffold-only report metadata. No production prompt was designed and no LLM call was made."
            ),
        )
        artifact_id = new_artifact_id("llm_design_review_report")
        self.storage.write_json(
            self.storage.run_key(run_id, _ARTIFACT_KEY),
            {"artifact_id": artifact_id, **artifact.model_dump()},
        )
        self.registry.update_active(run_id, llm_design_review_report_id=artifact_id)
        safe_workflow_mark(self.workflow_state, run_id, "step_16", "completed")
        return artifact


def _warning_for_missing(label: str) -> DesignReviewWarning:
    mapping = {
        "scientific_evidence_table": "missing_evidence_data",
        "patent_prior_art_table": "missing_patent_data",
        "ranked_candidate_shortlist": "missing_scoring_data",
    }
    return DesignReviewWarning(
        warning_type=mapping.get(label, "other"),  # type: ignore[arg-type]
        message=f"Missing upstream artifact ref: {label}",
    )


def _render_scaffold_report(
    run_id: str,
    generated_at: str,
    refs: dict[str, str],
    candidate_count: int,
) -> str:
    lines = [
        "# Step 16 Design Review Scaffold",
        "",
        f"- run_id: {run_id}",
        f"- generated_at: {generated_at}",
        "- status: scaffold_only",
        "- llm_generation: not_run",
        f"- candidate_count: {candidate_count}",
        "",
        "## Source Artifact Summary",
    ]
    for name, artifact_id in sorted(refs.items()):
        lines.append(f"- {name}: {artifact_id}")
    lines.extend(
        [
            "",
            "## Limitations",
            "- This scaffold does not contain production report prose.",
            "- No prompt, raw LLM response, or external ToolUniverse payload is stored here.",
        ]
    )
    return "\n".join(lines) + "\n"
