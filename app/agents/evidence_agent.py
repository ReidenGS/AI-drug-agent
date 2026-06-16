"""EvidenceAgent — Step 13 (deterministic query construction MVP).

Reads Step 2 / 5 / 10 / 12 artifacts (all three terminal upstreams are
optional except Step 2 + Step 5) and routes literature search queries to the
Step 13 MCP tools. No LLM is involved at this stage; query construction is
rule-based so output is repeatable across runs.

Shortlist fallback order (most-specific first):

    Step 12 ranking_table (when ranking_status="completed" and ranked_candidates non-empty)
        → Step 10 scoring_handoff_package.candidate_ids (when handoff exists)
        → Step 5 candidate_context_table candidate_ids (always present)

The chosen source is recorded on every `MultiAgentLiteratureSearch` call via
`tool_input_summary.shortlist_source` so audit / tests can tell which Step
provided the scope.

Tool routing (each call recorded as one `ToolCallRecord`):
- target_or_antigen_text     → `EuropePMC_search_articles` + `SemanticScholar_search_papers`
- payload_text               → `LiteratureSearchTool`
- candidate_label (per-cand) → `PubTator3_LiteratureSearch`
- resolved shortlist         → `MultiAgentLiteratureSearch`

Per-candidate `EvidenceRecord` is built from each successful tool call. Raw
upstream payloads land at `tool_outputs/step_13/{tool_call_id}.json`;
`scientific_evidence_table` only carries the normalized record + the
`tool_output_ref` so raw bodies never leak into the artifact.

If wrappers raise `NotImplementedError` (or the MCP client returns
`dependency_unavailable`), the step still completes with `review_status="partial"`.
"""

from __future__ import annotations

from typing import Any, Optional

from ..agents.tool_selection_policy import (
    SelectionContext,
    ToolInvocationPlan,
    select_and_build_invocations,
)
from ..llm.provider import LLMProvider, MockLLMProvider
from ..mcp.client import MCPClient
from ..schemas.common import ToolCallRecord
from ..schemas.step_13_scientific_evidence_table import (
    EvidenceRecord,
    ScientificEvidenceTable,
)
from ..services.artifact_registry_service import ArtifactRegistryService
from ..services.storage_service import Storage
from ..services.workflow_state_service import WorkflowStateService
from ..utils.errors import WorkflowStateError
from ..utils.ids import new_artifact_id, new_tool_call_id
from ..utils.time import now_iso


_AGENT_NAME = "evidence_agent"
_STEP_ID = "step_13"
_ARTIFACT_KEY = "scientific_evidence_table.json"


class EvidenceAgent:
    name = _AGENT_NAME

    def __init__(
        self,
        *,
        storage: Storage,
        registry: ArtifactRegistryService,
        workflow_state: WorkflowStateService,
        mcp_client: MCPClient,
        llm: LLMProvider | None = None,
    ) -> None:
        self.storage = storage
        self.registry = registry
        self.workflow_state = workflow_state
        self.mcp_client = mcp_client
        self.llm = llm or MockLLMProvider()

    def run(self, run_id: str) -> ScientificEvidenceTable:
        reg = self.registry.get(run_id)
        if not reg.active_artifacts.candidate_context_table_id:
            raise WorkflowStateError("Step 13 requires Step 5 candidate_context_table")
        if not reg.active_artifacts.structured_query_id:
            raise WorkflowStateError("Step 13 requires Step 2 structured_query")

        sq = self.storage.read_json(
            self.storage.run_key(run_id, "inputs/structured_query.json")
        )
        cct = self.storage.read_json(
            self.storage.run_key(run_id, "candidate_context_table.json")
        )
        ranking = (
            self.storage.read_json(self.storage.run_key(run_id, "ranking_table.json"))
            if reg.active_artifacts.ranking_table_id
            else None
        )
        handoff = (
            self.storage.read_json(self.storage.run_key(run_id, "scoring_handoff_package.json"))
            if reg.active_artifacts.scoring_handoff_id
            else None
        )

        entities = sq.get("mentioned_entities") or {}
        target = entities.get("target_or_antigen_text")
        payload = entities.get("payload_text") or entities.get("linker_text")
        candidates = cct.get("candidate_records") or []
        shortlist, shortlist_source = _resolve_shortlist(ranking, handoff, candidates)

        tool_calls: list[ToolCallRecord] = []
        evidence_records: list[EvidenceRecord] = []

        # ── target evidence ──────────────────────────────────────────────
        if target:
            for plan in self._plans_for_query(
                tool_names=["EuropePMC_search_articles", "SemanticScholar_search_papers"],
                query=str(target),
                signal="target_literature_query",
            ):
                tc = self._call_tool(
                    run_id=run_id, tool_name=plan.tool_name,
                    kwargs=plan.arguments,
                    label=f"target:{target}",
                    extra_input_summary=_selection_summary(plan),
                )
                tool_calls.append(tc)
                if tc.run_status == "success":
                    evidence_records.append(self._record(
                        candidate_id="",
                        target=str(target),
                        evidence_type="target_literature",
                        key_finding=(
                            f"Search executed via {plan.tool_name}. Raw payload at "
                            f"tool_output_ref={tc.tool_output_ref}."
                        ),
                        source=plan.tool_name,
                        confidence=0.4,
                    ))

        # ── payload evidence ─────────────────────────────────────────────
        if payload:
            for plan in self._plans_for_query(
                tool_names=["LiteratureSearchTool"],
                query=str(payload),
                signal="payload_literature_query",
            ):
                tc = self._call_tool(
                    run_id=run_id, tool_name=plan.tool_name,
                    kwargs=plan.arguments,
                    label=f"payload:{payload}",
                    extra_input_summary=_selection_summary(plan),
                )
                tool_calls.append(tc)
                if tc.run_status == "success":
                    evidence_records.append(self._record(
                        candidate_id="",
                        target=str(target) if target else None,
                        mechanism=str(payload),
                        evidence_type="payload_literature",
                        key_finding=(
                            f"Search executed via {plan.tool_name}. Raw payload at "
                            f"tool_output_ref={tc.tool_output_ref}."
                        ),
                        source=plan.tool_name,
                        confidence=0.4,
                    ))

        # ── per-candidate evidence ───────────────────────────────────────
        for candidate in candidates:
            label = candidate.get("candidate_label") or ""
            if not label:
                continue
            for plan in self._plans_for_query(
                tool_names=["PubTator3_LiteratureSearch"],
                query=label,
                signal="candidate_literature_query",
            ):
                tc = self._call_tool(
                    run_id=run_id, tool_name=plan.tool_name,
                    kwargs=plan.arguments,
                    label=f"candidate:{candidate.get('candidate_id')}",
                    extra_input_summary=_selection_summary(plan),
                )
                tool_calls.append(tc)
                if tc.run_status == "success":
                    evidence_records.append(self._record(
                        candidate_id=candidate.get("candidate_id", ""),
                        target=str(target) if target else None,
                        evidence_type="candidate_literature",
                        key_finding=(
                            f"Search executed via {plan.tool_name} for {label}. "
                            f"Raw payload at tool_output_ref={tc.tool_output_ref}."
                        ),
                        source=plan.tool_name,
                        confidence=0.3,
                    ))

        # ── shortlist multi-agent search ────────────────────────────────
        if shortlist:
            q = ", ".join([str(cid) for cid in shortlist[:5] if cid])
            if q:
                for plan in self._plans_for_query(
                    tool_names=["MultiAgentLiteratureSearch"],
                    query=f"shortlist:{q}",
                    signal="shortlist_literature_query",
                ):
                    tc = self._call_tool(
                        run_id=run_id, tool_name=plan.tool_name,
                        kwargs=plan.arguments,
                        label=f"shortlist[{shortlist_source}]",
                        extra_input_summary={"shortlist_source": shortlist_source, **_selection_summary(plan)},
                    )
                    tool_calls.append(tc)
                # MultiAgentLiteratureSearch may report dependency_unavailable
                # per Week 3 audit (total_papers=0 in real mode); we record the
                # call either way and don't fabricate findings.

        review_status = self._status(tool_calls, evidence_records)
        table = ScientificEvidenceTable(
            run_id=run_id,
            created_at=now_iso(),
            review_status=review_status,  # type: ignore[arg-type]
            evidence_records=evidence_records,
            tool_call_records=tool_calls,
        )

        artifact_id = new_artifact_id("scientific_evidence_table")
        self.storage.write_json(
            self.storage.run_key(run_id, _ARTIFACT_KEY),
            {"artifact_id": artifact_id, **table.model_dump()},
        )
        self.registry.update_active(run_id, scientific_evidence_table_id=artifact_id)
        self.workflow_state.mark(run_id, "step_13", "completed")
        return table

    # ── helpers ─────────────────────────────────────────────────────────
    def _plans_for_query(self, *, tool_names: list[str], query: str, signal: str) -> list[ToolInvocationPlan]:
        def fallback() -> list[ToolInvocationPlan]:
            return [
                ToolInvocationPlan(
                    tool_name=name,
                    selection_reason="deterministic Step 13 query fallback",
                    arguments={"query": query},
                    argument_construction_reason="deterministic evidence query mapping",
                    selected_by="deterministic_fallback",
                )
                for name in tool_names
            ]

        plans = select_and_build_invocations(
            agent_name=_AGENT_NAME,
            step_id=_STEP_ID,
            mcp_client=self.mcp_client,
            llm=self.llm,
            context=SelectionContext(
                signals={signal: True},
                arg_hints={"query": query},
                note=f"step_13 {signal}",
            ),
            deterministic_fallback=fallback,
            deterministic_argument_mapping=lambda _tool, hints: {"query": hints.get("query") or query},
        )
        selected = [p for p in plans if p.tool_name in set(tool_names)]
        return selected or fallback()

    def _call_tool(
        self,
        *,
        run_id: str,
        tool_name: str,
        kwargs: dict[str, Any],
        label: str,
        extra_input_summary: Optional[dict[str, Any]] = None,
    ) -> ToolCallRecord:
        tc_id = new_tool_call_id()
        started = now_iso()
        result = self.mcp_client.call_tool(
            agent_name=_AGENT_NAME, step_id=_STEP_ID, tool_name=tool_name, **kwargs
        )
        finished = now_iso()
        output_ref = None
        output_artifact_id = None
        if "payload" in result:
            output_artifact_id = new_artifact_id("tool_output")
            output_key = self.storage.run_key(
                run_id, "tool_outputs", "step_13", f"{tc_id}.json"
            )
            self.storage.write_json(output_key, {
                "tool_call_id": tc_id, "tool_name": tool_name,
                "label": label, "input": kwargs, "output": result["payload"],
            })
            output_ref = output_key
        return ToolCallRecord(
            tool_call_id=tc_id, tool_name=tool_name,
            agent_name=_AGENT_NAME, step_id=_STEP_ID,
            run_status=result.get("run_status", "pending"),
            started_at=started, finished_at=finished,
            tool_input_summary={"label": label, **kwargs, **(extra_input_summary or {})},
            tool_output_artifact_id=output_artifact_id,
            tool_output_ref=output_ref,
            error_message=result.get("error_message"),
        )

    @staticmethod
    def _record(
        *, candidate_id: str, key_finding: str, source: str,
        evidence_type: Optional[str] = None,
        target: Optional[str] = None, mechanism: Optional[str] = None,
        confidence: float = 0.0,
    ) -> EvidenceRecord:
        return EvidenceRecord(
            evidence_id=new_artifact_id("evidence"),
            candidate_id=candidate_id,
            therapeutic_area=None,
            disease_context=None,
            target=target,
            mechanism=mechanism,
            evidence_type=evidence_type,
            key_finding=key_finding,
            source=source,
            confidence_score=confidence,
        )

    @staticmethod
    def _status(calls: list[ToolCallRecord], records: list[EvidenceRecord]) -> str:
        if not calls:
            return "failed"
        any_success = any(t.run_status == "success" for t in calls)
        any_partial = any(
            t.run_status in {"failed", "dependency_unavailable", "skipped"} for t in calls
        )
        if any_success and not any_partial:
            return "ok"
        if any_success or records:
            return "partial"
        return "failed"


def _resolve_shortlist(
    ranking: Optional[dict],
    handoff: Optional[dict],
    candidates: list[dict],
) -> tuple[list[str], str]:
    """Return (shortlist_candidate_ids, source).

    Precedence:
    1. `step_12_ranking` — only when ranking_status="completed" AND at least
       one ranked candidate exists.
    2. `step_10_handoff` — when scoring_handoff_package was prepared.
    3. `step_05_candidates` — final fallback over the full Step 5 list.
    """
    if ranking and ranking.get("ranking_status") == "completed":
        ranked = [rc.get("candidate_id") for rc in ranking.get("ranked_candidates") or []]
        ranked = [cid for cid in ranked if cid]
        if ranked:
            return ranked, "step_12_ranking"
    if handoff:
        handoff_ids = [cid for cid in (handoff.get("candidate_ids") or []) if cid]
        if handoff_ids:
            return handoff_ids, "step_10_handoff"
    return [c.get("candidate_id") for c in candidates if c.get("candidate_id")], "step_05_candidates"


def _selection_summary(plan: ToolInvocationPlan) -> dict[str, Any]:
    return {
        "selected_by": plan.selected_by,
        "selection_reason": plan.selection_reason,
        "selection_policy_version": plan.selection_policy_version,
        "argument_construction_reason": plan.argument_construction_reason,
        "validation_status": plan.validation_status,
        "validation_warnings": plan.validation_warnings,
    }
