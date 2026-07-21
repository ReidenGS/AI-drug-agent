"""Unified request-based Patent-Evidence domain core (Turn H2).

This entrypoint owns one planning pass for both lanes. It never calls the
legacy EvidenceAgent/PatentIPAgent planners and never invents runtime argument
mappings after deterministic validation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from ..llm.provider import LLMProvider
from ..mcp.client import MCPClient
from ..schemas.common import ToolCallRecord
from ..schemas.patent_evidence_audit import (
    PatentEvidencePlanningAudit,
    PatentEvidenceResolverAuditEntry,
)
from ..schemas.patent_evidence_contract import (
    PATENT_EVIDENCE_ID_TYPE_REF_AUTHORITY,
)
from ..schemas.patent_evidence_request import (
    PatentEvidenceInputRef,
    PatentEvidenceRequest,
    PatentEvidenceSearchScope,
)
from ..schemas.step_13_scientific_evidence_table import ScientificEvidenceTable
from ..schemas.step_14_patent_prior_art_table import (
    PatentLookupSummary,
    PatentPriorArtTable,
)
from ..schemas.step_02_structured_query import StructuredQuery
from ..schemas.step_05_candidate_context_table import CandidateContextTable
from ..services.artifact_registry_service import ArtifactRegistryService
from ..services.storage_service import Storage
from ..services.workflow_state_service import WorkflowStateService
from ..utils.ids import new_artifact_id, new_tool_call_id
from ..utils.time import now_iso
from .evidence_agent import (
    _dedup_and_sort_by_relevance as dedup_evidence_hits,
    _extract_hits as extract_evidence_hits,
    _record_from_merged as evidence_record_from_merged,
)
from .patent_evidence_selection_policy import plan_patent_evidence_tool_calls
from .patent_ip_agent import _record_from_merged as patent_record_from_merged
from .step_14_prior_art import dedup_and_sort_by_relevance, extract_hits


_AGENT_NAME = "patent_evidence_agent"
_EVIDENCE_STEP = "step_13"
_PATENT_STEP = "step_14"
_SAFE_CANDIDATE_ID = re.compile(r"^[a-z][a-z0-9_]{0,127}$")
_INDEXED_SEGMENT = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)(?:\[(\d+)\])?$")
_PATENT_SOURCE_DB = {
    "drugbank_get_drug_references_by_drug_name_or_id": "DrugBank",
}


def requested_lanes_from_structured_query(
    structured_query: StructuredQuery,
) -> list[str]:
    """Derive lane authority only from typed StructuredQuery fields."""
    outputs = set(structured_query.requested_outputs)
    intents = {
        structured_query.task_intent.primary_intent,
        *structured_query.task_intent.secondary_intents,
    }
    lanes: list[str] = []
    if "literature_review_summary" in outputs or "literature_review" in intents:
        lanes.append("evidence")
    if "patent_or_ip_summary" in outputs or "patent_ip_review" in intents:
        lanes.append("patent")
    return lanes


@dataclass(frozen=True)
class PatentEvidenceRunResult:
    evidence: ScientificEvidenceTable
    patent: PatentPriorArtTable
    planning_audit: PatentEvidencePlanningAudit
    resolver_audit: list[PatentEvidenceResolverAuditEntry]


class PatentEvidenceAgent:
    """One planner, one resolver authority, two persisted output artifacts."""

    name = _AGENT_NAME

    def __init__(
        self,
        *,
        storage: Storage,
        registry: ArtifactRegistryService,
        workflow_state: WorkflowStateService,
        mcp_client: MCPClient,
        llm: LLMProvider,
    ) -> None:
        self.storage = storage
        self.registry = registry
        self.workflow_state = workflow_state
        self.mcp_client = mcp_client
        self.llm = llm

    def run_from_artifacts(
        self,
        run_id: str,
        *,
        structured_query: dict[str, Any],
        candidate_context_table: dict[str, Any],
    ) -> PatentEvidenceRunResult:
        structured_model = StructuredQuery.model_validate(structured_query, strict=True)
        candidate_model = CandidateContextTable.model_validate(
            candidate_context_table, strict=True
        )
        lanes = requested_lanes_from_structured_query(structured_model)
        if not lanes:
            raise ValueError("patent_evidence_no_requested_lane")
        artifacts = {
            "structured_query": structured_model.model_dump(),
            "candidate_context_table": candidate_model.model_dump(),
        }
        request = _build_request(run_id=run_id, artifacts=artifacts, lanes=lanes)
        planning = plan_patent_evidence_tool_calls(llm=self.llm, request=request)
        refs = {ref.ref_id: ref for ref in request.input_refs}
        resolver_audit: list[PatentEvidenceResolverAuditEntry] = []
        tool_records: dict[str, list[ToolCallRecord]] = {
            "evidence": [],
            "patent": [],
        }
        evidence_hits: list[Any] = []
        patent_hits: list[Any] = []
        lookup_summaries: list[PatentLookupSummary] = []
        executed = 0

        for plan in planning.tool_plans:
            lane = plan.search_lane
            if lane not in lanes:
                continue
            if not plan.can_invoke:
                tool_records[lane].append(
                    _skipped_plan_record(plan, "plan_not_invokable")
                )
                continue
            kwargs: dict[str, Any] = {}
            mapped_refs: list[str] = []
            unresolved_reason: str | None = None
            primary_ref: PatentEvidenceInputRef | None = None
            for mapping in plan.argument_mappings:
                ref = refs[mapping.input_ref_id]
                resolved, value_or_reason = _resolve_ref(artifacts, ref)
                resolver_audit.append(
                    PatentEvidenceResolverAuditEntry(
                        ref_id=ref.ref_id,
                        source_artifact=ref.source_artifact,
                        source_path=ref.source_path,
                        role=ref.role,
                        resolved=resolved,
                        unresolved_reason=None if resolved else value_or_reason,
                    )
                )
                if not resolved:
                    unresolved_reason = value_or_reason
                    break
                kwargs[mapping.schema_arg] = value_or_reason
                mapped_refs.append(ref.ref_id)
                primary_ref = primary_ref or ref
            if unresolved_reason is not None:
                tool_records[lane].append(
                    _skipped_plan_record(plan, unresolved_reason)
                )
                continue
            for literal in plan.argument_literals:
                kwargs[literal.schema_arg] = literal.literal_value

            record, result = self._execute_plan(
                run_id=run_id,
                plan=plan,
                kwargs=kwargs,
                mapped_ref_ids=mapped_refs,
            )
            executed += 1
            tool_records[lane].append(record)
            if record.run_status != "success":
                continue
            envelope = result.get("payload")
            inner = _inner_payload(envelope)
            if lane == "evidence":
                evidence_hits.extend(
                    extract_evidence_hits(
                        inner,
                        source_tool=plan.tool_name,
                        source_ref=record.tool_output_ref,
                        query_role=primary_ref.role if primary_ref else "query",
                        query_term="",
                        candidate_id=(primary_ref.candidate_id or "") if primary_ref else "",
                        limit=None,
                    )
                )
                continue
            normalized = result.get("normalized_output")
            if isinstance(normalized, dict) and normalized.get("source_type") in {
                "pubchem_associated_reference",
                "fda_orange_book_application_row",
            }:
                lookup_summaries.append(
                    PatentLookupSummary(
                        tool_name=plan.tool_name,
                        source_type=normalized["source_type"],
                        record_count=int(normalized.get("record_count") or 0),
                        tool_output_ref=record.tool_output_ref or "",
                        functional_limitation=str(
                            normalized.get("functional_limitation") or ""
                        ),
                    )
                )
                continue
            patent_hits.extend(
                extract_hits(
                    inner,
                    source_tool=plan.tool_name,
                    source_database=_PATENT_SOURCE_DB.get(plan.tool_name, "other"),
                    source_ref=record.tool_output_ref,
                    query_role=primary_ref.role if primary_ref else "query",
                    query_term="",
                    query_term_source=(
                        f"input_ref:{primary_ref.ref_id}" if primary_ref else "input_ref"
                    ),
                    candidate_id=(primary_ref.candidate_id or "") if primary_ref else "",
                )
            )

        planning_audit = PatentEvidencePlanningAudit(
            prompt_cache_layout_version=planning.prompt_cache_layout_version,
            llm_call_count=planning.llm_call_count,
            catalog_visible_count=len(planning.catalog_tool_names),
            eligible_count=len(planning.eligible_tool_names),
            selected_count=planning.llm_selected_plan_count,
            accepted_count=len(planning.tool_plans),
            rejected_count=len(planning.rejected_tool_plans),
            executed_count=executed,
            lane_assessments=[
                {
                    "search_lane": assessment.search_lane,
                    "status": assessment.status,
                    "reason": f"llm_assessment_{assessment.status}",
                }
                for assessment in planning.lane_assessments
            ],
            rejections=[
                rejection.model_dump() for rejection in planning.rejected_tool_plans
            ],
        )
        evidence_records = [
            evidence_record_from_merged(item, target_text=None)
            for item in dedup_evidence_hits(evidence_hits)
        ]
        patent_records = [
            patent_record_from_merged(item)
            for item in dedup_and_sort_by_relevance(patent_hits)
        ]
        evidence_status = _evidence_status(
            requested="evidence" in lanes,
            records=tool_records["evidence"],
        )
        patent_status = _patent_status(
            requested="patent" in lanes,
            records=tool_records["patent"],
        )
        evidence = ScientificEvidenceTable(
            run_id=run_id,
            created_at=now_iso(),
            review_status=evidence_status,
            evidence_records=evidence_records,
            tool_call_records=tool_records["evidence"],
            patent_evidence_planning_audit=planning_audit,
            patent_evidence_resolver_audit=resolver_audit,
        )
        patent = PatentPriorArtTable(
            run_id=run_id,
            created_at=now_iso(),
            patent_review_status=patent_status,
            patent_records=patent_records,
            tool_call_records=tool_records["patent"],
            patent_review_notes=(
                "Unified reference-only Patent-Evidence execution. PubChem and FDA "
                "lookup rows remain typed summaries and are not confirmed patents."
            ),
            step14_request_source="patent_evidence_unified",
            patent_evidence_planning_audit=planning_audit,
            patent_evidence_resolver_audit=resolver_audit,
            lookup_summaries=lookup_summaries,
        )
        self._persist(run_id=run_id, evidence=evidence, patent=patent)
        self._mark_workflow_state(run_id=run_id, evidence=evidence, patent=patent)
        return PatentEvidenceRunResult(
            evidence=evidence,
            patent=patent,
            planning_audit=planning_audit,
            resolver_audit=resolver_audit,
        )

    def _mark_workflow_state(
        self,
        *,
        run_id: str,
        evidence: ScientificEvidenceTable,
        patent: PatentPriorArtTable,
    ) -> None:
        evidence_state = {
            "ok": "completed",
            "partial": "completed",
            "failed": "failed",
            "not_requested": "skipped",
        }[evidence.review_status]
        patent_state = {
            "completed": "completed",
            "completed_with_warnings": "completed",
            "partial": "completed",
            "failed": "failed",
            "not_requested": "skipped",
        }[patent.patent_review_status]
        self.workflow_state.mark(run_id, "step_13", evidence_state)
        self.workflow_state.mark(run_id, "step_14", patent_state)

    def _execute_plan(
        self,
        *,
        run_id: str,
        plan: Any,
        kwargs: dict[str, Any],
        mapped_ref_ids: list[str],
    ) -> tuple[ToolCallRecord, dict[str, Any]]:
        tool_call_id = new_tool_call_id()
        started = now_iso()
        result = self.mcp_client.call_tool(
            agent_name=_AGENT_NAME,
            step_id=plan.execution_step_id,
            tool_name=plan.tool_name,
            **kwargs,
        )
        finished = now_iso()
        output_ref = None
        output_artifact_id = None
        if "payload" in result:
            output_artifact_id = new_artifact_id("tool_output")
            step_dir = "step_13" if plan.search_lane == "evidence" else "step_14"
            output_ref = self.storage.run_key(
                run_id, "tool_outputs", step_dir, f"{tool_call_id}.json"
            )
            self.storage.write_json(
                output_ref,
                {
                    "tool_call_id": tool_call_id,
                    "tool_name": plan.tool_name,
                    "raw_envelope": result["payload"],
                },
            )
        compact_error = result.get("reason") or result.get("envelope_status")
        return (
            ToolCallRecord(
                tool_call_id=tool_call_id,
                tool_name=plan.tool_name,
                agent_name=_AGENT_NAME,
                step_id=plan.execution_step_id,
                run_status=result.get("run_status", "failed"),
                started_at=started,
                finished_at=finished,
                tool_input_summary={
                    "search_lane": plan.search_lane,
                    "input_ref_ids": mapped_ref_ids,
                    "mapped_schema_args": [
                        mapping.schema_arg for mapping in plan.argument_mappings
                    ],
                    "literal_schema_args": [
                        literal.schema_arg for literal in plan.argument_literals
                    ],
                    "selected_by": "llm_patent_evidence",
                    "validation_status": "accepted",
                },
                tool_output_artifact_id=output_artifact_id,
                tool_output_ref=output_ref,
                error_message=(
                    str(compact_error) if result.get("run_status") != "success" else None
                ),
            ),
            result,
        )

    def _persist(
        self,
        *,
        run_id: str,
        evidence: ScientificEvidenceTable,
        patent: PatentPriorArtTable,
    ) -> None:
        evidence_id = new_artifact_id("scientific_evidence_table")
        patent_id = new_artifact_id("patent_prior_art_table")
        self.storage.write_json(
            self.storage.run_key(run_id, "scientific_evidence_table.json"),
            {"artifact_id": evidence_id, **evidence.model_dump()},
        )
        self.storage.write_json(
            self.storage.run_key(run_id, "patent_prior_art_table.json"),
            {"artifact_id": patent_id, **patent.model_dump()},
        )
        self.registry.update_active(
            run_id,
            scientific_evidence_table_id=evidence_id,
            patent_prior_art_table_id=patent_id,
        )


def _build_request(
    *, run_id: str, artifacts: dict[str, dict[str, Any]], lanes: list[str]
) -> PatentEvidenceRequest:
    refs: list[PatentEvidenceInputRef] = []

    def add(
        *,
        artifact: str,
        path: str,
        role: str,
        supports: list[str],
        candidate_id: str | None = None,
    ) -> None:
        refs.append(
            PatentEvidenceInputRef(
                ref_id=f"ref_{len(refs) + 1:05d}",
                source_artifact=artifact,
                source_path=path,
                role=role,
                candidate_id=(
                    candidate_id
                    if candidate_id and _SAFE_CANDIDATE_ID.fullmatch(candidate_id)
                    else None
                ),
                supports_tool_args=supports,
            )
        )

    sq = artifacts["structured_query"]
    if str(sq.get("canonical_query") or "").strip():
        add(
            artifact="structured_query",
            path="canonical_query",
            role="query",
            supports=["query", "research_topic"],
        )
    for index, item in enumerate(sq.get("referenced_inputs") or []):
        if not isinstance(item, dict) or not item.get("value"):
            continue
        mapping = PATENT_EVIDENCE_ID_TYPE_REF_AUTHORITY.get(item.get("id_type"))
        if mapping:
            add(
                artifact="structured_query",
                path=f"referenced_inputs[{index}].value",
                role=mapping[0],
                supports=list(mapping[1]),
            )
    cct = artifacts["candidate_context_table"]
    for index, hint in enumerate(cct.get("downstream_query_hints") or []):
        if not isinstance(hint, dict) or not hint.get("entity"):
            continue
        role = hint.get("role")
        if role in {
            "linker_payload",
            "payload",
            "linker",
            "compound",
            "target",
            "complete_adc",
            "antibody",
        }:
            add(
                artifact="candidate_context_table",
                path=f"downstream_query_hints[{index}].entity",
                role=role,
                supports=["query", "research_topic"],
            )
    for candidate_index, candidate in enumerate(cct.get("candidate_records") or []):
        if not isinstance(candidate, dict):
            continue
        candidate_id = candidate.get("candidate_id")
        for identifier_index, identifier in enumerate(candidate.get("identifiers") or []):
            if not isinstance(identifier, dict) or not identifier.get("id_value"):
                continue
            role_support = PATENT_EVIDENCE_ID_TYPE_REF_AUTHORITY.get(
                identifier.get("id_type")
            )
            if role_support:
                add(
                    artifact="candidate_context_table",
                    path=(
                        f"candidate_records[{candidate_index}].identifiers"
                        f"[{identifier_index}].id_value"
                    ),
                    role=role_support[0],
                    supports=list(role_support[1]),
                    candidate_id=candidate_id,
                )
    antibody_search_allowed = any(
        isinstance(hint, dict)
        and hint.get("role") == "antibody"
        and hint.get("explicit_or_inferred") == "explicit"
        for hint in cct.get("downstream_query_hints") or []
    )
    return PatentEvidenceRequest(
        run_id=run_id,
        user_query=sq.get("canonical_query"),
        source_artifact_refs={
            "structured_query": "inputs/structured_query.json",
            "candidate_context_table": "candidate_context_table.json",
        },
        input_refs=refs,
        search_scope=PatentEvidenceSearchScope(
            requested_lanes=lanes,
            antibody_search_allowed=antibody_search_allowed,
        ),
        request_notes=["constructed_from_validated_worker_artifacts"],
    )


def _resolve_ref(
    artifacts: dict[str, dict[str, Any]], ref: PatentEvidenceInputRef
) -> tuple[bool, Any]:
    body = artifacts.get(ref.source_artifact)
    if not isinstance(body, dict):
        return False, "source_artifact_unavailable"
    current: Any = body
    for segment in ref.source_path.split("."):
        match = _INDEXED_SEGMENT.fullmatch(segment)
        if match is None or not isinstance(current, dict):
            return False, "source_path_invalid"
        current = current.get(match.group(1))
        if match.group(2) is not None:
            if not isinstance(current, list):
                return False, "source_path_not_list"
            index = int(match.group(2))
            if index >= len(current):
                return False, "source_path_index_missing"
            current = current[index]
    if current is None or (isinstance(current, str) and not current.strip()):
        return False, "source_value_missing"
    return True, current


def _skipped_plan_record(plan: Any, reason: str) -> ToolCallRecord:
    now = now_iso()
    return ToolCallRecord(
        tool_call_id=new_tool_call_id(),
        tool_name=plan.tool_name,
        agent_name=_AGENT_NAME,
        step_id=plan.execution_step_id,
        run_status="skipped",
        started_at=now,
        finished_at=now,
        tool_input_summary={
            "search_lane": plan.search_lane,
            "input_ref_ids": [
                mapping.input_ref_id for mapping in plan.argument_mappings
            ],
            "validation_status": "not_executed",
            "skip_reason": reason,
        },
        error_message=reason,
    )


def _inner_payload(envelope: Any) -> Any:
    if isinstance(envelope, dict) and isinstance(envelope.get("payload"), dict):
        return envelope["payload"]
    return envelope


def _evidence_status(*, requested: bool, records: list[ToolCallRecord]) -> str:
    if not requested:
        return "not_requested"
    successes = sum(record.run_status == "success" for record in records)
    if successes and successes == len(records):
        return "ok"
    if successes:
        return "partial"
    return "failed"


def _patent_status(*, requested: bool, records: list[ToolCallRecord]) -> str:
    if not requested:
        return "not_requested"
    successes = sum(record.run_status == "success" for record in records)
    if successes and successes == len(records):
        return "completed"
    if successes:
        return "completed_with_warnings"
    return "failed"


__all__ = [
    "PatentEvidenceAgent",
    "PatentEvidenceRunResult",
    "requested_lanes_from_structured_query",
]
