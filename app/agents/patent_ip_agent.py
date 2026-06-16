"""PatentIPAgent — Step 14 (deterministic patent / prior-art scan).

Reads Step 2/5/9/10/12 artifacts and routes patent queries to Step 14 MCP
tools. No legal opinion is produced; the artifact carries the canonical
`legal_disclaimer` so callers cannot mistake this for an attorney review.

Scope fallback order (most-specific first):

    Step 12 ranking_table (when ranking_status="completed" and ranked_candidates non-empty)
        → Step 10 scoring_handoff_package.candidate_ids
        → Step 9 compound_screening_artifact compound hits / Step 5 compound candidates
        → Step 2 structured_query mentioned_entities.payload_text / linker_text

The chosen source is recorded on every tool call via
`tool_input_summary.shortlist_source` so audit / tests can tell which Step
provided the scope.

Tool routing:
- compound.pubchem_cid       → `PubChem_get_associated_patents_by_CID`
- payload_name / drug name   → `drugbank_get_drug_references_by_drug_name_or_id`
- payload_name (also)        → `FDA_OrangeBook_get_patent_info(brand_name=…)`
- structured_query payload   → DrugBank + Orange Book even when Step 5 has
                                no compound_component candidate (text-only
                                fallback).

Orange Book records:
- `source_database="FDA_OrangeBook"`
- `matched_entity_type="drug_application_or_regulatory_reference"`
- raw product / patent / exclusivity tables stay in
  `tool_outputs/step_14/{tool_call_id}.json` and are referenced via
  `tool_call_records[].tool_output_ref`. They are NEVER inlined into
  `patent_records[]`.
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
from ..schemas.step_14_patent_prior_art_table import (
    PatentPriorArtTable,
    PatentRecord,
)
from ..services.artifact_registry_service import ArtifactRegistryService
from ..services.storage_service import Storage
from ..services.workflow_state_service import WorkflowStateService
from ..utils.errors import WorkflowStateError
from ..utils.ids import new_artifact_id, new_tool_call_id
from ..utils.time import now_iso


_AGENT_NAME = "patent_ip_agent"
_STEP_ID = "step_14"
_ARTIFACT_KEY = "patent_prior_art_table.json"


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

    def run(self, run_id: str) -> PatentPriorArtTable:
        reg = self.registry.get(run_id)
        if not reg.active_artifacts.candidate_context_table_id:
            raise WorkflowStateError("Step 14 requires Step 5 candidate_context_table")

        cct = self.storage.read_json(
            self.storage.run_key(run_id, "candidate_context_table.json")
        )
        compound_screening = (
            self.storage.read_json(self.storage.run_key(run_id, "compound_screening_artifact.json"))
            if reg.active_artifacts.structure_variant_and_compound_screening_id
            else None
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

        compound_candidates = [
            c for c in cct.get("candidate_records") or []
            if c.get("candidate_type") == "compound_component"
        ]
        compound_hits = (compound_screening or {}).get("compound_hits") or []

        # Resolve scope: which candidate_ids to operate on, and which source
        # supplied that decision (recorded on every tool call summary).
        scope_ids, scope_source = _resolve_scope(
            ranking=ranking, handoff=handoff,
            cct_candidate_ids=[c.get("candidate_id") for c in cct.get("candidate_records") or []],
        )
        if scope_ids:
            compound_candidates = [
                c for c in compound_candidates if c.get("candidate_id") in scope_ids
            ]

        tool_calls: list[ToolCallRecord] = []
        patent_records: list[PatentRecord] = []

        # ── PubChem patents (compound hits with pubchem_cid identifier) ──
        for candidate in compound_candidates:
            for ident in candidate.get("identifiers") or []:
                if ident.get("id_type") != "pubchem_cid":
                    continue
                cid = ident.get("id_value")
                if not cid:
                    continue
                for plan in self._plans_for_patent(
                    tool_names=["PubChem_get_associated_patents_by_CID"],
                    signal="pubchem_cid",
                    arg_hints={"cid": cid, "pubchem_cid": cid},
                ):
                    tc = self._call_tool(
                        run_id=run_id, tool_name=plan.tool_name,
                        kwargs=plan.arguments,
                        label=f"pubchem:{candidate.get('candidate_id')}:{cid}",
                        extra_input_summary={
                            "shortlist_source": scope_source,
                            "candidate_id": candidate.get("candidate_id"),
                            **_selection_summary(plan),
                        },
                    )
                    tool_calls.append(tc)
                    if tc.run_status == "success":
                        patent_records.append(self._record(
                            candidate_id=candidate.get("candidate_id", ""),
                            matched_entity_type="compound",
                            matched_material_id=None,
                            source_database="PubChem",
                            source_ref=tc.tool_output_ref,
                            notes_limitations=(
                                f"Raw payload at tool_output_ref={tc.tool_output_ref}."
                            ),
                        ))

        # ── DrugBank references (per payload / linker / compound name) ───
        for candidate in compound_candidates:
            name = _first_material_value(candidate, {"payload_name", "linker_name", "compound_name"})
            if not name:
                continue
            for plan in self._plans_for_patent(
                tool_names=["drugbank_get_drug_references_by_drug_name_or_id"],
                signal="drug_name_or_id",
                arg_hints={"drug_name_or_id": name, "compound_name": name, "query": name},
            ):
                tc = self._call_tool(
                    run_id=run_id, tool_name=plan.tool_name,
                    kwargs=plan.arguments,
                    label=f"drugbank:{candidate.get('candidate_id')}:{name}",
                    extra_input_summary={
                        "shortlist_source": scope_source,
                        "candidate_id": candidate.get("candidate_id"),
                        **_selection_summary(plan),
                    },
                )
                tool_calls.append(tc)
                if tc.run_status == "success":
                    patent_records.append(self._record(
                        candidate_id=candidate.get("candidate_id", ""),
                        matched_entity_type=_entity_type_for(candidate),
                        matched_material_id=None,
                        source_database="DrugBank",
                        source_ref=tc.tool_output_ref,
                        notes_limitations=(
                            f"Raw DrugBank references at tool_output_ref={tc.tool_output_ref}."
                        ),
                    ))

        # ── FDA Orange Book (per payload / linker / compound name) ───────
        for candidate in compound_candidates:
            name = _first_material_value(candidate, {"payload_name", "linker_name", "compound_name"})
            if not name:
                continue
            for plan in self._plans_for_patent(
                tool_names=["FDA_OrangeBook_get_patent_info"],
                signal="brand_name",
                arg_hints={"brand_name": name, "compound_name": name, "query": name},
            ):
                tc = self._call_tool(
                    run_id=run_id, tool_name=plan.tool_name,
                    kwargs=plan.arguments,
                    label=f"orangebook:{candidate.get('candidate_id')}:{name}",
                    extra_input_summary={
                        "shortlist_source": scope_source,
                        "candidate_id": candidate.get("candidate_id"),
                        **_selection_summary(plan),
                    },
                )
                tool_calls.append(tc)
                if tc.run_status == "success":
                    # IMPORTANT: do NOT inline product/patent/exclusivity rows.
                    patent_records.append(self._record(
                        candidate_id=candidate.get("candidate_id", ""),
                        matched_entity_type="drug_application_or_regulatory_reference",
                        matched_material_id=None,
                        source_database="FDA_OrangeBook",
                        source_ref=tc.tool_output_ref,
                        notes_limitations=(
                            "Orange Book product-level rows are stored by reference; "
                            "this normalized row only records the search context."
                        ),
                    ))

        # ── structured_query payload/linker text fallback ────────────────
        # If Step 5 had no compound_component candidate (so the per-candidate
        # loops above issued zero compound queries), but structured_query
        # mentions a payload or linker, scan DrugBank + Orange Book on that
        # text directly. This covers user-typed payloads that never made it
        # into Step 5's candidate_records (e.g. "the warhead is MMAE" without
        # a payload_linker_text context field).
        compound_call_count = sum(
            1 for tc in tool_calls
            if tc.tool_name in {
                "drugbank_get_drug_references_by_drug_name_or_id",
                "FDA_OrangeBook_get_patent_info",
                "PubChem_get_associated_patents_by_CID",
            }
        )
        if compound_call_count == 0 and sq is not None:
            entities = sq.get("mentioned_entities") or {}
            sq_payload_text = entities.get("payload_text") or entities.get("linker_text")
            if sq_payload_text:
                for tool_name in (
                    "drugbank_get_drug_references_by_drug_name_or_id",
                    "FDA_OrangeBook_get_patent_info",
                ):
                    kwargs = (
                        {"drug_name_or_id": sq_payload_text}
                        if tool_name.startswith("drugbank")
                        else {"brand_name": sq_payload_text}
                    )
                    signal = "brand_name" if tool_name.startswith("FDA") else "drug_name_or_id"
                    for plan in self._plans_for_patent(
                        tool_names=[tool_name],
                        signal=signal,
                        arg_hints={**kwargs, "compound_name": sq_payload_text, "query": sq_payload_text},
                    ):
                        tc = self._call_tool(
                            run_id=run_id, tool_name=plan.tool_name, kwargs=plan.arguments,
                            label=f"sq_fallback:{sq_payload_text}",
                            extra_input_summary={
                                "shortlist_source": "step_02_structured_query",
                                "candidate_id": None,
                                **_selection_summary(plan),
                            },
                        )
                        tool_calls.append(tc)
                        if tc.run_status == "success":
                            patent_records.append(self._record(
                                candidate_id="",
                                matched_entity_type=(
                                    "drug_application_or_regulatory_reference"
                                    if plan.tool_name.startswith("FDA")
                                    else "payload"
                                ),
                                matched_material_id=None,
                                source_database=(
                                    "FDA_OrangeBook" if plan.tool_name.startswith("FDA")
                                    else "DrugBank"
                                ),
                                source_ref=tc.tool_output_ref,
                                notes_limitations=(
                                    f"Query derived from structured_query "
                                    f"mentioned_entities (no Step 5 compound candidate). "
                                    f"Raw payload at tool_output_ref={tc.tool_output_ref}."
                                ),
                            ))

        review_status = self._status(tool_calls, patent_records)
        table = PatentPriorArtTable(
            run_id=run_id,
            created_at=now_iso(),
            patent_review_status=review_status,  # type: ignore[arg-type]
            patent_records=patent_records,
            tool_call_records=tool_calls,
            patent_review_notes=(
                "Step 14 ran in MVP mode; tool wrappers may return mocked data "
                "(`status='mocked'`). Raw upstream payloads are referenced via "
                "tool_call_records[].tool_output_ref."
            ),
        )

        artifact_id = new_artifact_id("patent_prior_art_table")
        self.storage.write_json(
            self.storage.run_key(run_id, _ARTIFACT_KEY),
            {"artifact_id": artifact_id, **table.model_dump()},
        )
        self.registry.update_active(run_id, patent_prior_art_table_id=artifact_id)
        self.workflow_state.mark(run_id, "step_14", "completed")
        return table

    # ── helpers ─────────────────────────────────────────────────────────
    def _plans_for_patent(
        self, *, tool_names: list[str], signal: str, arg_hints: dict[str, Any]
    ) -> list[ToolInvocationPlan]:
        def fallback() -> list[ToolInvocationPlan]:
            return [
                ToolInvocationPlan(
                    tool_name=name,
                    selection_reason="deterministic Step 14 patent fallback",
                    arguments=_patent_argument_mapping(name, arg_hints),
                    argument_construction_reason="deterministic patent argument mapping",
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
                signals={signal: True, "compound_name": bool(arg_hints.get("compound_name"))},
                arg_hints=arg_hints,
                note=f"step_14 {signal}",
            ),
            deterministic_fallback=fallback,
            deterministic_argument_mapping=_patent_argument_mapping,
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
                run_id, "tool_outputs", "step_14", f"{tc_id}.json"
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
        *, candidate_id: str, matched_entity_type: str,
        matched_material_id: Optional[str], source_database: str,
        source_ref: Optional[str], notes_limitations: Optional[str],
    ) -> PatentRecord:
        return PatentRecord(
            patent_record_id=new_artifact_id("patent_record"),
            candidate_id=candidate_id,
            matched_entity_type=matched_entity_type,  # type: ignore[arg-type]
            matched_material_id=matched_material_id,
            source_database=source_database,  # type: ignore[arg-type]
            source_ref=source_ref,
            notes_limitations=notes_limitations,
        )

    @staticmethod
    def _status(calls: list[ToolCallRecord], records: list[PatentRecord]) -> str:
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


def _resolve_scope(
    *,
    ranking: Optional[dict],
    handoff: Optional[dict],
    cct_candidate_ids: list,
) -> tuple[set[str], str]:
    """Return (allowed_candidate_ids, source).

    Precedence:
    1. `step_12_ranking` — when ranking_status="completed" AND ranked candidates exist.
    2. `step_10_handoff` — when scoring_handoff_package was prepared.
    3. `step_05_candidates` — final fallback over the full Step 5 list.

    Returning `set()` means "no Step 5 candidate filter" — the agent will scan
    every compound candidate, and the structured_query payload-text fallback
    further down may still produce queries when even Step 5 is empty.
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


def _entity_type_for(candidate: dict) -> str:
    materials = candidate.get("materials") or []
    types = {m.get("material_type") for m in materials}
    if "payload_name" in types and "linker_name" in types:
        return "linker_payload"
    if "payload_name" in types:
        return "payload"
    if "linker_name" in types:
        return "linker"
    return "compound"


def _patent_argument_mapping(tool_name: str, arg_hints: dict) -> dict[str, Any]:
    if tool_name == "PubChem_get_associated_patents_by_CID":
        return {"cid": arg_hints.get("cid") or arg_hints.get("pubchem_cid") or ""}
    if tool_name == "drugbank_get_drug_references_by_drug_name_or_id":
        return {"drug_name_or_id": arg_hints.get("drug_name_or_id") or arg_hints.get("compound_name") or arg_hints.get("query") or ""}
    if tool_name == "FDA_OrangeBook_get_patent_info":
        return {"brand_name": arg_hints.get("brand_name") or arg_hints.get("compound_name") or arg_hints.get("query") or ""}
    return {"query": arg_hints.get("query") or arg_hints.get("compound_name") or ""}


def _selection_summary(plan: ToolInvocationPlan) -> dict[str, Any]:
    return {
        "selected_by": plan.selected_by,
        "selection_reason": plan.selection_reason,
        "selection_policy_version": plan.selection_policy_version,
        "argument_construction_reason": plan.argument_construction_reason,
        "validation_status": plan.validation_status,
        "validation_warnings": plan.validation_warnings,
    }
