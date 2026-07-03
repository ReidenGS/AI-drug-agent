"""Step 15 scaffold service — integrate existing patent refs into IP labels."""

from __future__ import annotations

from typing import Any

from ..schemas.step_15_ip_risk_integration import (
    CandidateIPRiskRecord,
    HumanReviewFlag,
    IPFilteredShortlistItem,
    IPRiskIntegratedShortlist,
    MissingIPAssessmentFlag,
)
from ..utils.ids import new_artifact_id
from ..utils.time import now_iso
from .artifact_registry_service import ArtifactRegistryService
from .downstream_scaffold_utils import missing_active_refs, read_json_if_exists, safe_workflow_mark
from .storage_service import Storage
from .workflow_state_service import WorkflowStateService


_ARTIFACT_KEY = "ip_risk_integrated_shortlist.json"
_RISK_ORDER = {"unknown": 0, "low": 1, "medium": 2, "high": 3}
_CORE_ENTITY_TYPES = {"payload", "linker", "linker_payload", "ligand", "compound", "drug_reference", "scaffold"}


class IPRiskIntegrationService:
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

    def integrate(self, run_id: str) -> IPRiskIntegratedShortlist:
        reg = self.registry.get(run_id)
        active = reg.active_artifacts
        ranking = read_json_if_exists(self.storage, run_id, "ranking_table.json")
        patent = read_json_if_exists(self.storage, run_id, "patent_prior_art_table.json")
        cct = read_json_if_exists(self.storage, run_id, "candidate_context_table.json")

        candidates = _candidate_rows(ranking=ranking, cct=cct)
        patent_records = (patent or {}).get("patent_records") or []
        tool_calls_by_id = {
            tc.get("tool_call_id"): tc
            for tc in (patent or {}).get("tool_call_records") or []
            if tc.get("tool_call_id")
        }
        grouped = _group_patents_by_candidate(patent_records)

        flags = [
            MissingIPAssessmentFlag(
                candidate_id=None,
                missing_item="other",
                severity="warning",
                message=f"Missing upstream artifact: {artifact_label}",
            )
            for _, artifact_label in missing_active_refs(
                self.registry,
                run_id,
                {
                    "ranking_table_id": "ranked_candidate_shortlist",
                    "patent_prior_art_table_id": "patent_prior_art_table",
                    "candidate_context_table_id": "candidate_context_table",
                    "run_step_plan_id": "run_step_plan",
                },
            )
        ]
        if not candidates:
            flags.append(
                MissingIPAssessmentFlag(
                    candidate_id=None,
                    missing_item="payload_or_compound_mapping",
                    severity="warning",
                    message="No candidate rows were available from Step 12 or Step 5.",
                )
            )

        risk_records: list[CandidateIPRiskRecord] = []
        shortlist: list[IPFilteredShortlistItem] = []
        for row in candidates:
            cid = row["candidate_id"]
            records = grouped.get(cid, [])
            if not records:
                flags.append(
                    MissingIPAssessmentFlag(
                        candidate_id=cid,
                        missing_item="patent_records",
                        severity="warning",
                        message="No Step 14 patent/prior-art record mapped to this candidate.",
                    )
                )
            source_tool_call_ids = sorted(
                {
                    ref
                    for rec in records
                    for ref in _source_tool_call_ids(rec)
                    if ref in tool_calls_by_id or ref.startswith("tc_")
                }
            )
            entity_types = sorted(
                {
                    _normalize_entity_type(rec.get("matched_entity_type"))
                    for rec in records
                }
            ) or ["unknown"]
            risk_label = _max_label(rec.get("novelty_risk") for rec in records)
            confidence = _max_label(rec.get("confidence_level") for rec in records)
            human_required = risk_label in {"high", "unknown"} or not records
            recommended_action = (
                "human_review_required"
                if human_required
                else ("proceed_with_warning" if risk_label == "medium" else "proceed")
            )
            risk_records.append(
                CandidateIPRiskRecord(
                    candidate_id=cid,
                    source_rank=row.get("rank"),
                    source_patent_record_ids=[
                        rec.get("patent_record_id")
                        for rec in records
                        if rec.get("patent_record_id")
                    ],
                    source_tool_call_ids=source_tool_call_ids,
                    matched_entity_types=entity_types,  # type: ignore[arg-type]
                    core_automated_ip_screen_scope=_screen_scope(entity_types),
                    match_strength="unknown",
                    claim_relevance=_claim_relevance(records),
                    novelty_risk=risk_label,  # type: ignore[arg-type]
                    confidence_level=confidence,  # type: ignore[arg-type]
                    final_ip_risk_label=risk_label,  # type: ignore[arg-type]
                    human_review_flag=HumanReviewFlag(
                        required=human_required,
                        reason=(
                            "high_ip_risk"
                            if risk_label == "high"
                            else ("insufficient_patent_text" if human_required else None)
                        ),
                    ),
                    recommended_action=recommended_action,  # type: ignore[arg-type]
                    ip_risk_rationale=(
                        "Scaffold label derived from Step 14 normalized novelty/confidence labels; "
                        "no patent retrieval or legal conclusion was performed."
                    ),
                    notes_limitations=(
                        "Automated scope is limited to compact Step 14 prior-art metadata."
                    ),
                )
            )
            shortlist.append(
                IPFilteredShortlistItem(
                    candidate_id=cid,
                    original_rank=row.get("rank"),
                    post_ip_status=(
                        "human_review_required"
                        if human_required
                        else ("kept_with_warning" if risk_label == "medium" else "kept")
                    ),
                    final_ip_risk_label=risk_label,  # type: ignore[arg-type]
                    human_review_required=human_required,
                )
            )

        status = "completed" if risk_records and not flags else ("partial" if risk_records else "partial")
        artifact = IPRiskIntegratedShortlist(
            run_id=run_id,
            created_at=now_iso(),
            ip_integration_status=status,  # type: ignore[arg-type]
            candidate_ip_risk_records=risk_records,
            ip_filtered_shortlist=shortlist,
            missing_ip_assessment_flags=flags,
            ip_integration_notes=(
                "Step 15 scaffold consumed existing Step 14 records only. It did not call "
                "patent tools and does not provide legal advice."
            ),
        )
        artifact_id = new_artifact_id("ip_risk_integrated_shortlist")
        self.storage.write_json(
            self.storage.run_key(run_id, _ARTIFACT_KEY),
            {"artifact_id": artifact_id, **artifact.model_dump()},
        )
        self.registry.update_active(run_id, ip_risk_integrated_shortlist_id=artifact_id)
        safe_workflow_mark(self.workflow_state, run_id, "step_15", "completed")
        return artifact


def _candidate_rows(*, ranking: dict[str, Any] | None, cct: dict[str, Any] | None) -> list[dict[str, Any]]:
    ranked = (ranking or {}).get("ranked_candidates") or []
    if ranked:
        return [
            {"candidate_id": r.get("candidate_id"), "rank": r.get("rank")}
            for r in ranked
            if r.get("candidate_id")
        ]
    rows = (cct or {}).get("candidate_records") or []
    return [
        {"candidate_id": r.get("candidate_id"), "rank": None}
        for r in rows
        if r.get("candidate_id")
    ]


def _group_patents_by_candidate(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for rec in records:
        cid = rec.get("candidate_id")
        if cid:
            out.setdefault(cid, []).append(rec)
    return out


def _source_tool_call_ids(rec: dict[str, Any]) -> list[str]:
    refs = rec.get("source_refs") or []
    direct = rec.get("source_ref")
    if direct:
        refs.append(direct)
    return [str(ref).split("/")[-1].replace(".json", "") for ref in refs if ref]


def _normalize_entity_type(value: Any) -> str:
    value = str(value or "unknown")
    if value == "drug_application_or_regulatory_reference":
        return "drug_reference"
    allowed = {
        "payload",
        "linker",
        "linker_payload",
        "ligand",
        "compound",
        "drug_reference",
        "scaffold",
        "antibody_sequence",
        "epitope",
        "target",
        "full_adc_construct",
        "full_aoc_construct",
        "oligonucleotide_chemistry",
        "other",
        "unknown",
    }
    return value if value in allowed else "other"


def _max_label(values) -> str:
    best = "unknown"
    for value in values:
        value = str(value or "unknown")
        if value not in _RISK_ORDER:
            value = "unknown"
        if _RISK_ORDER[value] > _RISK_ORDER[best]:
            best = value
    return best


def _screen_scope(entity_types: list[str]) -> str:
    known = {e for e in entity_types if e != "unknown"}
    if not known:
        return "unknown"
    core = known & _CORE_ENTITY_TYPES
    non_core = known - _CORE_ENTITY_TYPES
    if core and non_core:
        return "mixed"
    if core:
        return "payload_compound_drug_reference"
    return "non_core_human_review_only"


def _claim_relevance(records: list[dict[str, Any]]) -> str:
    entity_types = {_normalize_entity_type(rec.get("matched_entity_type")) for rec in records}
    if entity_types & {"payload", "linker", "linker_payload", "compound"}:
        return "direct_payload_or_compound_claim"
    if "scaffold" in entity_types:
        return "related_compound_or_scaffold"
    if "drug_reference" in entity_types:
        return "drug_reference"
    if entity_types & {"full_adc_construct", "antibody_sequence", "target", "epitope"}:
        return "construct_level_claim"
    return "unknown"
