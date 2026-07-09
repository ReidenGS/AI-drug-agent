"""PatentIPAgent — Step 14 hint-driven patent / prior-art normalization MVP.

Reads Step 2 / 5 / 9 / 10 / 12 artifacts (Step 5 required) and routes patent
queries to the Step 14 MCP tools. Queries are driven by Step 5
`candidate_context_table.downstream_query_hints` so the search scope follows
professor guidance and is NEVER antibody-centered by default:

    linker_payload → payload → linker → compound → target → complete_adc
    → antibody (only when Step 5 explicitly captured one)

Inferred query expansion adds two patent-specific roles:
- `conjugation_chemistry` — only when linker / payload / linker_payload hint
  exists; the synthesized term references the existing hint entity.
- `use_or_indication` — only when a target hint exists; the synthesized term
  references the target entity. Both carry
  `query_term_source="inferred_expansion"`.

Antibody queries are emitted only when Step 5 wrote an antibody hint (i.e.
the user explicitly supplied an antibody candidate).

Tool routing (all calls go through the inventory-scoped MCP client; no new
external API client):
- `PubChem_get_associated_patents_by_CID` — per compound candidate with
  `pubchem_cid` identifier.
- `drugbank_get_drug_references_by_drug_name_or_id` — per
  compound-candidate payload/linker/compound name AND per text hint
  (linker_payload/payload/linker/compound/target/complete_adc/antibody/
  conjugation_chemistry/use_or_indication).
- `FDA_OrangeBook_get_patent_info` — same as DrugBank.

Scope fallback order for **candidate-bound** queries only (PubChem CID +
compound-candidate name flows):

    Step 12 ranking_table (when `ranking_status="completed"` and
        `ranked_candidates` non-empty)
    → Step 10 scoring_handoff_package
    → Step 5 candidates

Entity-level hint queries carry their own `shortlist_source` derived from
the hint's origin (e.g. `mentioned_entities.payload_text` →
`step_02_structured_query`; otherwise `step_05_downstream_hint`).

Per successful tool call, raw payload lands at
`tool_outputs/step_14/{tool_call_id}.json`. Compact prior-art hits are
extracted (title, patent_number, assignee, year, link, claim_focus), then
deduped across all tool calls by patent_number (preferred) or
(normalized_title + normalized_assignee). Deterministic IP-relevance scoring
ranks the merged literature/patent hits. Step 14 NEVER writes
`ranking_table.json` — ADC candidate ranking remains owned by Step 12.

If wrappers raise `NotImplementedError` (or the MCP client returns
`dependency_unavailable`), the step still completes with
`patent_review_status="partial"`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional

from ..agents.tool_selection_policy import (
    SelectionContext,
    ToolInvocationPlan,
    select_and_build_invocations,
)
from ..agents.step_14_prior_art import (
    MergedHit,
    dedup_and_sort_by_relevance,
    extract_hits,
)
from ..agents.step_14_runtime_resolver import (
    Step14ResolvedRef,
    resolve_step14_input_ref,
)
from ..agents.step_14_selection_policy import (
    Step14ToolPlan,
    plan_step14_tool_calls,
)
from ..llm.provider import LLMProvider, MockLLMProvider
from ..mcp.client import MCPClient
from ..schemas.common import ToolCallRecord
from ..schemas.step_14_patent_prior_art_table import (
    PatentPriorArtTable,
    PatentRecord,
)
from ..schemas.step_14_patent_request import Step14PatentRequest
from ..services.artifact_registry_service import ArtifactRegistryService
from ..services.storage_service import Storage
from ..services.workflow_state_service import WorkflowStateService
from ..utils.errors import WorkflowStateError
from ..utils.ids import new_artifact_id, new_tool_call_id
from ..utils.time import now_iso


_AGENT_NAME = "patent_ip_agent"
_STEP_ID = "step_14"
_ARTIFACT_KEY = "patent_prior_art_table.json"

# Demo-friendly default; the public ceiling allows production-scale review.
DEFAULT_TOTAL_LIMIT = 50
MAX_TOTAL_LIMIT = 1000
DEFAULT_PER_QUERY_LIMIT = 25

# Hint role priority. Payload / linker first; antibody only when Step 5 hint
# present; complete_adc is comparator scope and sits late.
_ROLE_PRIORITY: tuple[str, ...] = (
    "linker_payload",
    "payload",
    "linker",
    "compound",
    "target",
    "complete_adc",
    "antibody",
)

# Hint roles → text tools that get called with the hint entity.
_TEXT_TOOLS = (
    "drugbank_get_drug_references_by_drug_name_or_id",
    "FDA_OrangeBook_get_patent_info",
)

# Tool name → source_database label used in extracted records.
_TOOL_SOURCE_DB: dict[str, str] = {
    "PubChem_get_associated_patents_by_CID": "PubChem",
    "drugbank_get_drug_references_by_drug_name_or_id": "DrugBank",
    "FDA_OrangeBook_get_patent_info": "FDA_OrangeBook",
    # EuropePMC is a literature / prior-art evidence source. It is NOT one of
    # the PatentSourceDatabase Literal values, so its provenance is preserved
    # in `PatentRecord.sources` / `source_refs` (compact) while the
    # `source_database` Literal field falls back to "other" — we never mislabel
    # a literature hit as PubChem/FDA/DrugBank/USPTO.
    "EuropePMC_search_articles": "EuropePMC",
}


@dataclass
class _QueryPlan:
    role: str               # query_role (linker_payload, payload, …, conjugation_chemistry, …)
    term: str               # query_term (entity text or CID)
    term_source: str        # query_term_source (e.g. mentioned_entities.payload_text)
    tools: tuple[str, ...]  # which tools to call for this plan
    shortlist_source: str   # provenance label recorded on every call
    candidate_id: Optional[str] = None
    matched_entity_type: str = "unknown"


class PatentIPAgent:
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

    def run(
        self,
        run_id: str,
        *,
        total_limit: int = DEFAULT_TOTAL_LIMIT,
        per_query_limit: int = DEFAULT_PER_QUERY_LIMIT,
    ) -> PatentPriorArtTable:
        total_limit = max(1, min(int(total_limit), MAX_TOTAL_LIMIT))
        per_query_limit = max(1, min(int(per_query_limit), MAX_TOTAL_LIMIT))

        reg = self.registry.get(run_id)
        if not reg.active_artifacts.candidate_context_table_id:
            raise WorkflowStateError("Step 14 requires Step 5 candidate_context_table")

        cct = self.storage.read_json(
            self.storage.run_key(run_id, "candidate_context_table.json")
        )
        sq = (
            self.storage.read_json(self.storage.run_key(run_id, "inputs/structured_query.json"))
            if reg.active_artifacts.structured_query_id
            else None
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

        downstream_hints = cct.get("downstream_query_hints") or []
        candidates = cct.get("candidate_records") or []
        compound_candidates = [
            c for c in candidates if c.get("candidate_type") == "compound_component"
        ]

        # Candidate scope (only constrains candidate-bound plans).
        scope_ids, scope_source = _resolve_scope(
            ranking=ranking, handoff=handoff,
            cct_candidate_ids=[c.get("candidate_id") for c in candidates],
        )
        if scope_ids:
            compound_candidates = [
                c for c in compound_candidates if c.get("candidate_id") in scope_ids
            ]

        plans = list(
            _build_query_plans(
                downstream_hints=downstream_hints,
                compound_candidates=compound_candidates,
                scope_source=scope_source,
                sq_entities=(sq or {}).get("mentioned_entities") if sq else None,
            )
        )

        tool_calls: list[ToolCallRecord] = []
        all_raw_hits = []
        # Track which tool_calls produced at least one extracted hit so we
        # only emit synthetic search-execution receipts for the "empty raw"
        # case (preserves backward-compat assertions like "at least one OB
        # row exists").
        productive_tc_ids: set[str] = set()
        for plan in plans[:max(1, total_limit)]:
            for tool_name in plan.tools:
                tc, hits = self._call_and_extract(
                    run_id=run_id,
                    plan=plan,
                    tool_name=tool_name,
                    per_query_limit=per_query_limit,
                )
                tool_calls.append(tc)
                if hits:
                    productive_tc_ids.add(tc.tool_call_id)
                    all_raw_hits.extend(hits)

        merged_hits = dedup_and_sort_by_relevance(all_raw_hits)[:total_limit]

        patent_records: list[PatentRecord] = []
        # Extracted prior-art rows.
        for m in merged_hits:
            patent_records.append(_record_from_merged(m))
        # Synthetic receipt rows: one per successful tool call whose payload
        # contained no extractable hits. Keeps "at least one OB row /
        # at least one DrugBank row" coverage for empty mock envelopes.
        for tc in tool_calls:
            if tc.run_status != "success":
                continue
            if tc.tool_call_id in productive_tc_ids:
                continue
            patent_records.append(_record_from_receipt(tc))

        review_status = self._status(tool_calls, patent_records)
        table = PatentPriorArtTable(
            run_id=run_id,
            created_at=now_iso(),
            patent_review_status=review_status,  # type: ignore[arg-type]
            patent_records=patent_records,
            tool_call_records=tool_calls,
            patent_review_notes=(
                "Step 14 normalized prior-art rows are compact. Raw "
                "upstream payloads are referenced via "
                "tool_call_records[].tool_output_ref / "
                "patent_records[].source_refs."
            ),
            step14_request_source="discovery_run",
        )

        artifact_id = new_artifact_id("patent_prior_art_table")
        self.storage.write_json(
            self.storage.run_key(run_id, _ARTIFACT_KEY),
            {"artifact_id": artifact_id, **table.model_dump()},
        )
        self.registry.update_active(run_id, patent_prior_art_table_id=artifact_id)
        self.workflow_state.mark(run_id, "step_14", "completed")
        return table

    # ── request-based entrypoint (Step 14 single-stage planner) ─────────────
    def run_from_request(
        self,
        request: Step14PatentRequest,
        *,
        total_limit: int = DEFAULT_TOTAL_LIMIT,
        per_query_limit: int = DEFAULT_PER_QUERY_LIMIT,
    ) -> PatentPriorArtTable:
        """Request-driven Step 14 routing (single-stage LLM planner).

        One LLM call returns tool plans with schema_arg → input_ref_id
        mappings. The runtime builds kwargs STRICTLY from the LLM's accepted
        ``argument_mappings`` / ``argument_literals`` — it never re-derives a
        mapping from ``supports_to_schema_arg``. For each mapped input ref the
        runtime resolver reads the real value from storage. A plan the planner
        marked ``can_invoke=false`` is not called (uninvokable record); a plan
        whose mapped ref cannot be resolved is not called
        (skipped / input_missing). No fake success. Raw payloads stay in
        ``tool_output_ref``; extraction / dedup / scoring reuse ``run``'s
        helpers. Step 14 never writes ``ranking_table``.
        """
        total_limit = max(1, min(int(total_limit), MAX_TOTAL_LIMIT))
        per_query_limit = max(1, min(int(per_query_limit), MAX_TOTAL_LIMIT))
        run_id = request.run_id

        planning = plan_step14_tool_calls(llm=self.llm, request=request)
        ref_by_id = {ref.ref_id: ref for ref in request.input_refs}

        tool_calls: list[ToolCallRecord] = []
        resolver_audit: list[dict] = []
        all_raw_hits: list = []
        productive_tc_ids: set[str] = set()
        resolved_cache: dict[str, Step14ResolvedRef] = {}

        def _resolve(ref_id: str) -> Step14ResolvedRef:
            if ref_id not in resolved_cache:
                resolved = resolve_step14_input_ref(self.storage, request, ref_by_id[ref_id])
                resolved_cache[ref_id] = resolved
                resolver_audit.append(resolved.audit_entry())
            return resolved_cache[ref_id]

        for plan in planning.tool_plans[: max(1, total_limit)]:
            if not plan.can_invoke:
                tool_calls.append(_uninvokable_record(plan))
                continue
            # Build kwargs STRICTLY from the LLM's accepted mappings/literals.
            kwargs: dict[str, Any] = {}
            primary: Optional[Step14ResolvedRef] = None
            unresolved_reason: Optional[str] = None
            for mapping in plan.argument_mappings:
                resolved = _resolve(mapping.input_ref_id)
                if not resolved.resolved or not resolved.value:
                    unresolved_reason = resolved.unresolved_reason or "input_missing"
                    break
                kwargs[mapping.schema_arg] = resolved.value
                primary = primary or resolved
            if unresolved_reason is not None or primary is None or not kwargs:
                tool_calls.append(
                    _skipped_input_missing_record(plan, unresolved_reason or "input_missing")
                )
                continue
            for literal in plan.argument_literals:
                kwargs[literal.schema_arg] = literal.literal_value
            tc, hits = self._call_and_extract_request(
                run_id=run_id, plan=plan, tool_args=kwargs, primary=primary,
            )
            tool_calls.append(tc)
            if hits:
                productive_tc_ids.add(tc.tool_call_id)
                all_raw_hits.extend(hits)

        merged_hits = dedup_and_sort_by_relevance(all_raw_hits)[:total_limit]
        patent_records: list[PatentRecord] = [_record_from_merged(m) for m in merged_hits]
        for tc in tool_calls:
            if tc.run_status != "success":
                continue
            if tc.tool_call_id in productive_tc_ids:
                continue
            patent_records.append(_record_from_receipt(tc))

        review_status = self._status(tool_calls, patent_records)
        table = PatentPriorArtTable(
            run_id=run_id,
            created_at=now_iso(),
            patent_review_status=review_status,  # type: ignore[arg-type]
            patent_records=patent_records,
            tool_call_records=tool_calls,
            patent_review_notes=(
                "Step 14 request-based routing (single-stage planner). Tools + "
                "schema_arg→input_ref_id mappings were LLM-planned; real values "
                "were resolved by the runtime resolver. Raw payloads remain in "
                "tool_call_records[].tool_output_ref."
            ),
            step14_request_refs=[
                {
                    "ref_id": ref.ref_id,
                    "role": ref.role,
                    "source_artifact": ref.source_artifact,
                    "source_path": ref.source_path,
                    "candidate_id": ref.candidate_id,
                    "supports_tool_args": list(ref.supports_tool_args),
                }
                for ref in request.input_refs
            ],
            step14_patent_scope=request.patent_scope.model_dump(),
            step14_llm_tool_plans=[p.model_dump() for p in planning.tool_plans],
            step14_llm_rejected_tool_plans=[
                p.model_dump() for p in planning.rejected_tool_plans
            ],
            step14_argument_mapping_audit=list(planning.argument_mapping_audit),
            step14_prompt_cache_layout_version=planning.prompt_cache_layout_version,
            step14_runtime_resolver_audit=resolver_audit,
            step14_request_source="request",
        )

        artifact_id = new_artifact_id("patent_prior_art_table")
        self.storage.write_json(
            self.storage.run_key(run_id, _ARTIFACT_KEY),
            {"artifact_id": artifact_id, **table.model_dump()},
        )
        self.registry.update_active(run_id, patent_prior_art_table_id=artifact_id)
        self.workflow_state.mark(run_id, "step_14", "completed")
        return table

    def _call_and_extract_request(
        self,
        *,
        run_id: str,
        plan: Step14ToolPlan,
        tool_args: dict[str, Any],
        primary: Step14ResolvedRef,
    ) -> tuple[ToolCallRecord, list]:
        tc_id = new_tool_call_id()
        started = now_iso()
        result = self.mcp_client.call_tool(
            agent_name=_AGENT_NAME, step_id=_STEP_ID,
            tool_name=plan.tool_name, **tool_args,
        )
        finished = now_iso()

        role = primary.role
        term = primary.value or ""
        term_source = f"request_input_ref:{primary.ref_id}"
        label = f"{role}:{term}"

        payload = result.get("payload")
        # Envelope-aware status: a wrapper / ToolUniverse envelope that carries
        # a failure `status` (e.g. upstream_error) must NOT be recorded as a
        # success even when the outer MCP call returned run_status="success".
        run_status = result.get("run_status", "pending")
        error_message = result.get("error_message")
        envelope_status = payload.get("status") if isinstance(payload, dict) else None
        if envelope_status in _ENVELOPE_FAILURE_STATUS:
            run_status = _ENVELOPE_FAILURE_STATUS[envelope_status]
            error_message = _compact_envelope_error(plan.tool_name, envelope_status, payload)

        output_ref = None
        output_artifact_id = None
        if "payload" in result:
            output_artifact_id = new_artifact_id("tool_output")
            output_key = self.storage.run_key(
                run_id, "tool_outputs", "step_14", f"{tc_id}.json"
            )
            self.storage.write_json(output_key, {
                "tool_call_id": tc_id, "tool_name": plan.tool_name,
                "label": label, "input": tool_args, "output": result["payload"],
            })
            output_ref = output_key

        summary: dict[str, Any] = {
            "label": label,
            **tool_args,
            "candidate_id": primary.candidate_id,
            "shortlist_source": "step_14_request",
            "query_role": role,
            "query_term": term,
            "query_term_source": term_source,
            "selected_by": "llm_step14",
            "selection_reason": plan.selection_reason,
            "input_ref_ids": [m.input_ref_id for m in plan.argument_mappings],
            "argument_mappings": [
                {"schema_arg": m.schema_arg, "input_ref_id": m.input_ref_id}
                for m in plan.argument_mappings
            ],
        }
        if envelope_status:
            summary["output_envelope_status"] = envelope_status
        tc = ToolCallRecord(
            tool_call_id=tc_id, tool_name=plan.tool_name,
            agent_name=_AGENT_NAME, step_id=_STEP_ID,
            run_status=run_status,
            started_at=started, finished_at=finished,
            tool_input_summary=summary,
            tool_output_artifact_id=output_artifact_id,
            tool_output_ref=output_ref,
            error_message=error_message,
        )

        hits: list = []
        if tc.run_status == "success" and "payload" in result:
            hits = extract_hits(
                result["payload"],
                source_tool=plan.tool_name,
                source_database=_TOOL_SOURCE_DB.get(plan.tool_name, "other"),
                source_ref=output_ref,
                query_role=role,
                query_term=term,
                query_term_source=term_source,
                candidate_id=primary.candidate_id or "",
            )
        return tc, hits

    # ── helpers ─────────────────────────────────────────────────────────
    def _call_and_extract(
        self,
        *,
        run_id: str,
        plan: _QueryPlan,
        tool_name: str,
        per_query_limit: int,
    ) -> tuple[ToolCallRecord, list]:
        arg_hints = _arg_hints_for_plan(plan, per_query_limit=per_query_limit)
        invocation = self._invocation_plan(
            tool_name=tool_name, plan=plan, arg_hints=arg_hints
        )
        tc_id = new_tool_call_id()
        started = now_iso()
        result = self.mcp_client.call_tool(
            agent_name=_AGENT_NAME, step_id=_STEP_ID,
            tool_name=invocation.tool_name, **invocation.arguments,
        )
        finished = now_iso()

        output_ref = None
        output_artifact_id = None
        if "payload" in result:
            output_artifact_id = new_artifact_id("tool_output")
            output_key = self.storage.run_key(
                run_id, "tool_outputs", "step_14", f"{tc_id}.json"
            )
            self.storage.write_json(output_key, {
                "tool_call_id": tc_id, "tool_name": invocation.tool_name,
                "label": _label_for_plan(plan), "input": invocation.arguments,
                "output": result["payload"],
            })
            output_ref = output_key

        summary: dict[str, Any] = {
            "label": _label_for_plan(plan),
            **invocation.arguments,
            "candidate_id": plan.candidate_id,
            "shortlist_source": plan.shortlist_source,
            "query_role": plan.role,
            "query_term": plan.term,
            "query_term_source": plan.term_source,
            **_selection_summary(invocation),
        }
        tc = ToolCallRecord(
            tool_call_id=tc_id, tool_name=invocation.tool_name,
            agent_name=_AGENT_NAME, step_id=_STEP_ID,
            run_status=result.get("run_status", "pending"),
            started_at=started, finished_at=finished,
            tool_input_summary=summary,
            tool_output_artifact_id=output_artifact_id,
            tool_output_ref=output_ref,
            error_message=result.get("error_message"),
        )

        hits: list = []
        if tc.run_status == "success" and "payload" in result:
            hits = extract_hits(
                result["payload"],
                source_tool=invocation.tool_name,
                source_database=_TOOL_SOURCE_DB.get(invocation.tool_name, "other"),
                source_ref=output_ref,
                query_role=plan.role,
                query_term=plan.term,
                query_term_source=plan.term_source,
                candidate_id=plan.candidate_id or "",
            )
        return tc, hits

    def _invocation_plan(
        self, *, tool_name: str, plan: _QueryPlan, arg_hints: dict[str, Any]
    ) -> ToolInvocationPlan:
        def fallback() -> list[ToolInvocationPlan]:
            return [
                ToolInvocationPlan(
                    tool_name=tool_name,
                    selection_reason="deterministic Step 14 hint-driven plan",
                    arguments=_patent_argument_mapping(tool_name, arg_hints),
                    argument_construction_reason="deterministic patent argument mapping",
                    selected_by="deterministic_fallback",
                )
            ]

        plans = select_and_build_invocations(
            agent_name=_AGENT_NAME,
            step_id=_STEP_ID,
            mcp_client=self.mcp_client,
            llm=self.llm,
            context=SelectionContext(
                signals={"query_role": plan.role, "compound_name": bool(arg_hints.get("compound_name"))},
                arg_hints=arg_hints,
                note=f"step_14 role={plan.role}",
            ),
            deterministic_fallback=fallback,
            deterministic_argument_mapping=_patent_argument_mapping,
        )
        for p in plans:
            if p.tool_name == tool_name:
                return p
        return fallback()[0]

    @staticmethod
    def _status(
        calls: list[ToolCallRecord], records: list[PatentRecord]
    ) -> str:
        if not calls:
            return "failed"
        any_success = any(t.run_status == "success" for t in calls)
        any_partial = any(
            t.run_status in {"failed", "dependency_unavailable", "skipped"} for t in calls
        )
        if any_success and not any_partial:
            return "completed"
        if any_success or records:
            return "completed_with_warnings"
        return "partial"


# ── request-based envelope-aware status ─────────────────────────────────────

# A wrapper / ToolUniverse envelope `status` in these values is an upstream
# failure and must override an outer run_status="success". `ToolCallRunStatus`
# has no `upstream_error` literal, so it maps onto the canonical `failed`.
_ENVELOPE_FAILURE_STATUS: dict[str, str] = {
    "upstream_error": "failed",
    "dependency_unavailable": "dependency_unavailable",
    "failed": "failed",
    "error": "failed",
}


def _compact_envelope_error(tool_name: str, envelope_status: str, payload: Any) -> str:
    """Compact, raw-payload-free error message for an envelope failure.

    Uses the envelope's own short `error_message` field when present (truncated),
    never the raw payload body — the raw envelope stays in ``tool_output_ref``.
    """
    detail = ""
    if isinstance(payload, dict):
        raw = payload.get("error_message")
        if isinstance(raw, str) and raw.strip():
            detail = ": " + " ".join(raw.split())[:200]
    return f"{tool_name} envelope status={envelope_status}{detail}"


# ── request-based skipped / uninvokable records ─────────────────────────────


def _plan_ref_ids(plan: Step14ToolPlan) -> list[str]:
    return [m.input_ref_id for m in plan.argument_mappings]


def _uninvokable_record(plan: Step14ToolPlan) -> ToolCallRecord:
    """Compact record for a plan the planner marked ``can_invoke=false`` — the
    tool is NOT called (no fake success)."""
    return ToolCallRecord(
        tool_call_id=new_tool_call_id(),
        tool_name=plan.tool_name,
        agent_name=_AGENT_NAME,
        step_id=_STEP_ID,
        run_status="skipped",
        started_at=now_iso(),
        finished_at=now_iso(),
        tool_input_summary={
            "skip_reason": "uninvokable",
            "missing_required_args": list(plan.missing_required_args),
            "input_ref_ids": _plan_ref_ids(plan),
            "selected_by": "llm_step14",
            "selection_reason": plan.selection_reason,
        },
        error_message=(
            "uninvokable: missing_required_args="
            + ",".join(plan.missing_required_args)
        ),
    )


def _skipped_input_missing_record(
    plan: Step14ToolPlan, unresolved_reason: str
) -> ToolCallRecord:
    """Compact record when a mapped input ref could not be resolved — the tool
    is NOT called (no fake success)."""
    return ToolCallRecord(
        tool_call_id=new_tool_call_id(),
        tool_name=plan.tool_name,
        agent_name=_AGENT_NAME,
        step_id=_STEP_ID,
        run_status="skipped",
        started_at=now_iso(),
        finished_at=now_iso(),
        tool_input_summary={
            "skip_reason": "input_missing",
            "unresolved_reason": unresolved_reason,
            "input_ref_ids": _plan_ref_ids(plan),
            "selected_by": "llm_step14",
        },
        error_message=f"input_missing: {unresolved_reason}",
    )


# ── query-plan construction ─────────────────────────────────────────────────


def _build_query_plans(
    *,
    downstream_hints: list[dict],
    compound_candidates: list[dict],
    scope_source: str,
    sq_entities: Optional[dict],
) -> Iterable[_QueryPlan]:
    """Build the deterministic order of patent-search plans.

    Order: hint-driven entity plans (in `_ROLE_PRIORITY`) → inferred
    expansion (`conjugation_chemistry`, `use_or_indication`) →
    candidate-bound compound flow (PubChem on `pubchem_cid`, DrugBank+OB on
    payload/linker/compound name) → structured_query payload-text fallback
    when neither hint nor candidate produced a query.
    """
    yielded_anything = False
    # Hint-driven entity plans.
    by_role: dict[str, list[dict]] = {}
    for h in downstream_hints or []:
        role = h.get("role")
        if role:
            by_role.setdefault(role, []).append(h)
    for role in _ROLE_PRIORITY:
        for hint in by_role.get(role, []):
            entity = (hint.get("entity") or "").strip()
            if not entity:
                continue
            shortlist = _hint_source_to_shortlist(hint.get("source") or "")
            yield _QueryPlan(
                role=role,
                term=entity,
                term_source=hint.get("source") or "downstream_query_hints",
                tools=_TEXT_TOOLS,
                shortlist_source=shortlist,
                matched_entity_type=_entity_type_for_role(role),
            )
            yielded_anything = True

    # Inferred expansion: conjugation_chemistry / use_or_indication.
    payload_hints = (
        by_role.get("linker_payload", []) + by_role.get("payload", []) + by_role.get("linker", [])
    )
    if payload_hints:
        entity = (payload_hints[0].get("entity") or "").strip()
        if entity:
            yield _QueryPlan(
                role="conjugation_chemistry",
                term=f"{entity} conjugation chemistry",
                term_source="inferred_expansion",
                tools=_TEXT_TOOLS,
                shortlist_source="step_05_downstream_hint",
                matched_entity_type="linker_payload",
            )
    target_hints = by_role.get("target", [])
    if target_hints:
        entity = (target_hints[0].get("entity") or "").strip()
        if entity:
            yield _QueryPlan(
                role="use_or_indication",
                term=f"{entity} use indication",
                term_source="inferred_expansion",
                tools=_TEXT_TOOLS,
                shortlist_source="step_05_downstream_hint",
                matched_entity_type="target",
            )

    # Candidate-bound compound flow.
    for cand in compound_candidates:
        cid = cand.get("candidate_id") or ""
        # PubChem on pubchem_cid identifier.
        for ident in cand.get("identifiers") or []:
            if ident.get("id_type") != "pubchem_cid":
                continue
            cid_value = ident.get("id_value")
            if not cid_value:
                continue
            yield _QueryPlan(
                role="compound",
                term=str(cid_value),
                term_source="candidate_identifier.pubchem_cid",
                tools=("PubChem_get_associated_patents_by_CID",),
                shortlist_source=scope_source,
                candidate_id=cid,
                matched_entity_type="compound",
            )
            yielded_anything = True
        # DrugBank + Orange Book on payload/linker/compound name materials.
        name = _first_material_value(cand, {"payload_name", "linker_name", "compound_name"})
        if name:
            yield _QueryPlan(
                role=_entity_role_for_compound(cand),
                term=name,
                term_source="candidate_material",
                tools=_TEXT_TOOLS,
                shortlist_source=scope_source,
                candidate_id=cid,
                matched_entity_type=_entity_type_for_role(_entity_role_for_compound(cand)),
            )
            yielded_anything = True

    # Structured-query payload-text fallback when nothing else fired.
    if not yielded_anything and sq_entities:
        sq_text = sq_entities.get("payload_text") or sq_entities.get("linker_text")
        if sq_text:
            yield _QueryPlan(
                role="payload",
                term=str(sq_text),
                term_source="mentioned_entities.payload_text",
                tools=_TEXT_TOOLS,
                shortlist_source="step_02_structured_query",
                matched_entity_type="payload",
            )


def _hint_source_to_shortlist(source: str) -> str:
    if source.startswith("mentioned_entities."):
        return "step_02_structured_query"
    return "step_05_downstream_hint"


def _entity_role_for_compound(candidate: dict) -> str:
    materials = candidate.get("materials") or []
    types = {m.get("material_type") for m in materials}
    if "payload_name" in types and "linker_name" in types:
        return "linker_payload"
    if "payload_name" in types:
        return "payload"
    if "linker_name" in types:
        return "linker"
    return "compound"


def _entity_type_for_role(role: str) -> str:
    mapping = {
        "linker_payload": "linker_payload",
        "payload": "payload",
        "linker": "linker",
        "compound": "compound",
        "target": "target",
        "complete_adc": "full_adc_construct",
        "antibody": "antibody_sequence",
        "conjugation_chemistry": "linker_payload",
        "use_or_indication": "target",
    }
    return mapping.get(role, "unknown")


def _arg_hints_for_plan(plan: _QueryPlan, *, per_query_limit: int) -> dict[str, Any]:
    if plan.role == "compound" and plan.term and plan.term.isdigit():
        return {
            "cid": plan.term,
            "pubchem_cid": plan.term,
            "query": plan.term,
            "limit": per_query_limit,
        }
    return {
        "drug_name_or_id": plan.term,
        "brand_name": plan.term,
        "compound_name": plan.term,
        "query": plan.term,
        "limit": per_query_limit,
    }


def _label_for_plan(plan: _QueryPlan) -> str:
    base = f"{plan.role}:{plan.term}"
    if plan.candidate_id:
        return f"{base}|candidate={plan.candidate_id}"
    return base


def _patent_argument_mapping(tool_name: str, arg_hints: dict) -> dict[str, Any]:
    if tool_name == "PubChem_get_associated_patents_by_CID":
        return {"cid": arg_hints.get("cid") or arg_hints.get("pubchem_cid") or ""}
    if tool_name == "drugbank_get_drug_references_by_drug_name_or_id":
        return {
            "drug_name_or_id": (
                arg_hints.get("drug_name_or_id")
                or arg_hints.get("compound_name")
                or arg_hints.get("query")
                or ""
            )
        }
    if tool_name == "FDA_OrangeBook_get_patent_info":
        return {
            "brand_name": (
                arg_hints.get("brand_name")
                or arg_hints.get("compound_name")
                or arg_hints.get("query")
                or ""
            )
        }
    return {"query": arg_hints.get("query") or arg_hints.get("compound_name") or ""}


def _resolve_scope(
    *,
    ranking: Optional[dict],
    handoff: Optional[dict],
    cct_candidate_ids: list,
) -> tuple[set[str], str]:
    """Return (allowed_candidate_ids, source) for candidate-bound queries.

    Precedence:
    1. `step_12_ranking` — when ranking_status="completed" AND ranked candidates exist.
    2. `step_10_handoff` — when scoring_handoff_package was prepared.
    3. `step_05_candidates` — final fallback over the full Step 5 list.
    """
    if ranking and ranking.get("ranking_status") == "completed":
        ranked = {rc.get("candidate_id") for rc in ranking.get("ranked_candidates") or []}
        ranked.discard(None)
        if ranked:
            return ranked, "step_12_ranking"
    if handoff:
        handoff_ids = {cid for cid in (handoff.get("candidate_ids") or []) if cid}
        if handoff_ids:
            return handoff_ids, "step_10_handoff"
    cct_set = {cid for cid in cct_candidate_ids if cid}
    return cct_set, "step_05_candidates"


def _first_material_value(candidate: dict, types: set[str]) -> Optional[str]:
    for m in candidate.get("materials") or []:
        if m.get("material_type") in types:
            v = m.get("value")
            if v:
                return str(v)
    return None


def _selection_summary(plan: ToolInvocationPlan) -> dict[str, Any]:
    return {
        "selected_by": plan.selected_by,
        "selection_reason": plan.selection_reason,
        "selection_policy_version": plan.selection_policy_version,
        "argument_construction_reason": plan.argument_construction_reason,
        "validation_status": plan.validation_status,
        "validation_warnings": plan.validation_warnings,
    }


# ── record assembly ─────────────────────────────────────────────────────────


def _record_from_merged(m: MergedHit) -> PatentRecord:
    # Orange-Book-sourced rows keep regulatory entity type for backward
    # compatibility with downstream consumers; PubChem rows are compound;
    # other sources inherit from the query role.
    if "FDA_OrangeBook" in m.sources:
        matched_entity = "drug_application_or_regulatory_reference"
    elif "PubChem" in m.sources:
        matched_entity = "compound"
    else:
        matched_entity = _entity_type_for_role(m.query_role or "unknown")

    rationale = "; ".join(m.rationale) if m.rationale else None
    return PatentRecord(
        patent_record_id=new_artifact_id("patent_record"),
        candidate_id=m.candidate_id or "",
        matched_entity_type=matched_entity,  # type: ignore[arg-type]
        matched_material_id=None,
        source_database=(
            m.source_database
            if m.source_database in {"PubChem", "DrugBank", "FDA_OrangeBook", "USPTO"}
            else "other"
        ),  # type: ignore[arg-type]
        patent_title=m.title,
        patent_number=m.patent_number,
        publication_date=m.publication_date,
        assignee=m.assignee,
        source_url=m.link,
        source_ref=m.source_refs[0] if m.source_refs else None,
        notes_limitations=(
            "Prior-art row derived from compact extraction; full claims / "
            "description / abstract remain in tool_output_ref artifacts."
        ),
        query_role=m.query_role,
        query_term=m.query_term,
        query_term_source=m.query_term_source,
        publication_year=m.publication_year,
        jurisdiction=m.jurisdiction,
        claim_focus=m.claim_focus,
        sources=list(m.sources),
        source_refs=list(m.source_refs),
        ip_relevance_score=round(m.score, 3),
        relevance_rationale=rationale,
    )


def _record_from_receipt(tc: ToolCallRecord) -> PatentRecord:
    """Synthetic record for a successful tool call whose payload had no
    extractable hits (e.g. default mock envelopes with empty `records: []`).
    Carries no patent_number / title so downstream tests can ignore it
    when checking real prior-art rows.
    """
    summary = tc.tool_input_summary or {}
    db = _TOOL_SOURCE_DB.get(tc.tool_name, "other")
    if db == "FDA_OrangeBook":
        matched_entity = "drug_application_or_regulatory_reference"
    elif db == "PubChem":
        matched_entity = "compound"
    else:
        matched_entity = _entity_type_for_role(summary.get("query_role") or "unknown")
    return PatentRecord(
        patent_record_id=new_artifact_id("patent_record"),
        candidate_id=summary.get("candidate_id") or "",
        matched_entity_type=matched_entity,  # type: ignore[arg-type]
        matched_material_id=None,
        source_database=db if db in {"PubChem", "DrugBank", "FDA_OrangeBook", "USPTO"} else "other",  # type: ignore[arg-type]
        source_ref=tc.tool_output_ref,
        notes_limitations=(
            "Search-execution receipt; raw payload contained no extractable "
            "patent hits. Raw payload remains in tool_output_ref."
        ),
        query_role=summary.get("query_role"),
        query_term=summary.get("query_term"),
        query_term_source=summary.get("query_term_source"),
        sources=[db] if db != "other" else [],
        source_refs=[tc.tool_output_ref] if tc.tool_output_ref else [],
    )
