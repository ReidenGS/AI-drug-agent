"""EvidenceAgent — Step 13 systematic literature review MVP.

Reads Step 2 / 5 / 10 / 12 artifacts (Step 2 + Step 5 required) and routes
literature search queries to the Step 13 MCP tools. Queries are built from
Step 5 `candidate_context_table.downstream_query_hints` so the search scope
follows professor guidance and is NEVER antibody-centered by default:

    linker_payload → payload → linker → compound → target → complete_adc
    → antibody (only when Step 5 explicitly captured one)

`complete_adc` sits late in the list because reference benchmarks such as
T-DM1 / T-DXd are comparators rather than the primary novel scope; ordering
them after the modular ADC pieces keeps the search payload-centric. Antibody
queries are emitted only when Step 5 wrote an antibody hint (i.e. the user
explicitly supplied an antibody candidate); the priority order does not
reintroduce antibody-centered search.

Routing is rule-based — no LLM is involved in query construction — so output
is repeatable across runs.

Shortlist fallback order for `MultiAgentLiteratureSearch` (unchanged):

    Step 12 ranking_table → Step 10 scoring_handoff_package → Step 5 candidates

Per-hit literature evidence records are built from successful tool calls by
extracting common hit fields (title / DOI / link / year), deduplicating by
DOI (preferred) or normalized title, and scoring each literature hit
deterministically. Raw upstream payloads land at
`tool_outputs/step_13/{tool_call_id}.json`; `scientific_evidence_table`
only carries the normalized record + `source_refs` so raw abstracts / full
payloads never leak into the artifact.

Step 13 only computes **evidence relevance scoring over literature hits**;
ADC candidate ranking is owned by Step 12 and `ranking_table.json` is only
read, never written here.

If wrappers raise `NotImplementedError` (or the MCP client returns
`dependency_unavailable`), the step still completes with
`review_status="partial"`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

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

# Per-call demo limit (small, fast) and a much larger ceiling for production
# usage. Both are public so the API/caller can override without touching
# code internals.
DEFAULT_TOTAL_LIMIT = 50
MAX_TOTAL_LIMIT = 1000
DEFAULT_PER_QUERY_LIMIT = 25

# Role priority for query construction. Payload/linker-payload first matches
# the professor's "Evidence/IP search should prioritize linker-payload,
# payload, linker, … compound, target/antigen …" wording; `complete_adc`
# sits late as comparator scope; `antibody` only fires when Step 5
# explicitly captured an antibody hint.
_ROLE_PRIORITY: tuple[str, ...] = (
    "linker_payload",
    "payload",
    "linker",
    "compound",
    "target",
    "complete_adc",
    "antibody",
)

# Tool routing per hint role. Antibody never gets its own tool fan-out by
# default — it shares the generic literature search backend so the surface
# stays small.
_ROLE_TO_TOOLS: dict[str, tuple[str, ...]] = {
    "target": ("EuropePMC_search_articles", "SemanticScholar_search_papers"),
    "payload": ("LiteratureSearchTool",),
    "linker_payload": ("LiteratureSearchTool",),
    "linker": ("LiteratureSearchTool",),
    "compound": ("LiteratureSearchTool",),
    "complete_adc": ("LiteratureSearchTool",),
    "antibody": ("LiteratureSearchTool",),
}

# Common keys various ToolUniverse-backed literature wrappers use for the
# hit list and per-hit fields.
_HIT_LIST_KEYS = ("results", "hits", "papers", "articles", "documents")
_TITLE_KEYS = ("title", "titleText", "name")
_DOI_KEYS = ("doi", "DOI")
_LINK_KEYS = ("url", "link", "doi_url", "html_url")
_YEAR_KEYS = ("year", "publication_year", "pub_year")
_ABSTRACT_KEYS = ("abstract", "abstractText", "summary")

# Title-normalization regex (strip punctuation, collapse whitespace).
_TITLE_PUNCT = re.compile(r"[^\w\s]")
_WS = re.compile(r"\s+")

# DOI ranking / theme heuristics.
_ADC_TOKENS = ("adc", "antibody-drug conjugate", "antibody drug conjugate")
_PAYLOAD_TOKENS = (
    "mmae", "mmaf", "dxd", "dm1", "exatecan", "calicheamicin",
    "duocarmycin", "vc-mmae", "payload",
)
_LINKER_TOKENS = ("linker", "valine-citrulline", "cleavable", "non-cleavable")


# ── Public helpers / data shapes ───────────────────────────────────────────


@dataclass
class _RawHit:
    """One literature hit before dedup/scoring; not persisted as-is."""

    title: Optional[str]
    doi: Optional[str]
    link: Optional[str]
    year: Optional[int]
    abstract: Optional[str]  # used only for scoring; never stored
    source_tool: str
    source_ref: Optional[str]
    query_role: str
    query_term: str
    candidate_id: str = ""


@dataclass
class _MergedHit:
    """Dedup-merged literature hit; basis of one EvidenceRecord."""

    title: Optional[str]
    doi: Optional[str]
    link: Optional[str]
    year: Optional[int]
    query_role: str
    query_term: str
    candidate_id: str
    sources: list[str] = field(default_factory=list)
    source_refs: list[str] = field(default_factory=list)
    has_abstract: bool = False
    score: float = 0.0


# ── Agent ──────────────────────────────────────────────────────────────────


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
        per_query_limit: int = DEFAULT_PER_QUERY_LIMIT,
        total_limit: int = DEFAULT_TOTAL_LIMIT,
    ) -> None:
        self.storage = storage
        self.registry = registry
        self.workflow_state = workflow_state
        self.mcp_client = mcp_client
        self.llm = llm or MockLLMProvider()
        self.per_query_limit = max(1, min(int(per_query_limit), MAX_TOTAL_LIMIT))
        self.total_limit = max(1, min(int(total_limit), MAX_TOTAL_LIMIT))

    def run(self, run_id: str) -> ScientificEvidenceTable:
        reg = self.registry.get(run_id)
        if not reg.active_artifacts.candidate_context_table_id:
            raise WorkflowStateError("Step 13 requires Step 5 candidate_context_table")
        if not reg.active_artifacts.structured_query_id:
            raise WorkflowStateError("Step 13 requires Step 2 structured_query")

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

        candidates = cct.get("candidate_records") or []
        hints = cct.get("downstream_query_hints") or []
        shortlist, shortlist_source = _resolve_shortlist(ranking, handoff, candidates)

        tool_calls: list[ToolCallRecord] = []
        raw_hits: list[_RawHit] = []

        # ── 1. hint-driven literature searches ──────────────────────────
        for role in _ROLE_PRIORITY:
            for hint in _hints_for_role(hints, role):
                term = (hint.get("entity") or "").strip()
                if not term:
                    continue
                for tool_name in _ROLE_TO_TOOLS.get(role, ()):  # noqa: SIM118
                    for plan in self._plans_for_query(
                        tool_names=[tool_name], query=term,
                        signal=f"{role}_literature_query",
                    ):
                        tc, payload = self._call_tool(
                            run_id=run_id, tool_name=plan.tool_name,
                            kwargs=plan.arguments,
                            label=f"{role}:{term}",
                            extra_input_summary={
                                "query_role": role,
                                "query_term": term,
                                "query_term_source": hint.get("source"),
                                **_selection_summary(plan),
                            },
                        )
                        tool_calls.append(tc)
                        if tc.run_status == "success":
                            raw_hits.extend(
                                _extract_hits(
                                    payload,
                                    source_tool=plan.tool_name,
                                    source_ref=tc.tool_output_ref,
                                    query_role=role,
                                    query_term=term,
                                    candidate_id="",
                                    limit=self.per_query_limit,
                                )
                            )

        # ── 2. per-candidate PubTator3 sweep (preserved behavior) ───────
        for candidate in candidates:
            label = candidate.get("candidate_label") or ""
            if not label:
                continue
            for plan in self._plans_for_query(
                tool_names=["PubTator3_LiteratureSearch"],
                query=label,
                signal="candidate_literature_query",
            ):
                tc, payload = self._call_tool(
                    run_id=run_id, tool_name=plan.tool_name,
                    kwargs=plan.arguments,
                    label=f"candidate:{candidate.get('candidate_id')}",
                    extra_input_summary={
                        "query_role": "candidate",
                        "query_term": label,
                        **_selection_summary(plan),
                    },
                )
                tool_calls.append(tc)
                if tc.run_status == "success":
                    raw_hits.extend(
                        _extract_hits(
                            payload,
                            source_tool=plan.tool_name,
                            source_ref=tc.tool_output_ref,
                            query_role="candidate",
                            query_term=label,
                            candidate_id=candidate.get("candidate_id", ""),
                            limit=self.per_query_limit,
                        )
                    )

        # ── 3. shortlist multi-agent search (preserved behavior) ────────
        if shortlist:
            q = ", ".join([str(cid) for cid in shortlist[:5] if cid])
            if q:
                for plan in self._plans_for_query(
                    tool_names=["MultiAgentLiteratureSearch"],
                    query=f"shortlist:{q}",
                    signal="shortlist_literature_query",
                ):
                    tc, _payload = self._call_tool(
                        run_id=run_id, tool_name=plan.tool_name,
                        kwargs=plan.arguments,
                        label=f"shortlist[{shortlist_source}]",
                        extra_input_summary={
                            "shortlist_source": shortlist_source,
                            "query_role": "shortlist",
                            "query_term": q,
                            **_selection_summary(plan),
                        },
                    )
                    tool_calls.append(tc)
                # MultiAgentLiteratureSearch hits are intentionally NOT
                # extracted into evidence records — its mock envelope
                # has total_papers=0 and we don't fabricate findings.

        # ── 4. dedup + rank ─────────────────────────────────────────────
        merged = _dedup_and_sort_by_relevance(raw_hits)
        merged = merged[: self.total_limit]

        target_text = _first_hint_entity(hints, "target")
        evidence_records: list[EvidenceRecord] = [
            _record_from_merged(h, target_text=target_text) for h in merged
        ]

        # If no hits but some tool calls succeeded, leave a minimal receipt
        # record per successful tool call so downstream consumers still see
        # which queries fired (matches prior behavior).
        if not evidence_records:
            for tc in tool_calls:
                if tc.run_status != "success":
                    continue
                role = ((tc.tool_input_summary or {}).get("query_role")) or ""
                term = ((tc.tool_input_summary or {}).get("query_term")) or ""
                evidence_records.append(
                    EvidenceRecord(
                        evidence_id=new_artifact_id("evidence"),
                        candidate_id="",
                        target=target_text,
                        evidence_type=f"{role}_literature" if role else "literature",
                        key_finding=(
                            f"Search executed via {tc.tool_name}; raw payload at "
                            f"tool_output_ref={tc.tool_output_ref}."
                        ),
                        source=tc.tool_name,
                        confidence_score=0.3,
                        query_role=role or None,
                        query_term=term or None,
                        sources=[tc.tool_name],
                        source_refs=[tc.tool_output_ref] if tc.tool_output_ref else [],
                    )
                )

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
    def _plans_for_query(
        self, *, tool_names: list[str], query: str, signal: str
    ) -> list[ToolInvocationPlan]:
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
    ) -> tuple[ToolCallRecord, Any]:
        tc_id = new_tool_call_id()
        started = now_iso()
        result = self.mcp_client.call_tool(
            agent_name=_AGENT_NAME, step_id=_STEP_ID, tool_name=tool_name, **kwargs
        )
        finished = now_iso()
        output_ref = None
        output_artifact_id = None
        payload = result.get("payload")
        if "payload" in result:
            output_artifact_id = new_artifact_id("tool_output")
            output_key = self.storage.run_key(
                run_id, "tool_outputs", "step_13", f"{tc_id}.json"
            )
            self.storage.write_json(output_key, {
                "tool_call_id": tc_id, "tool_name": tool_name,
                "label": label, "input": kwargs, "output": payload,
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
        ), payload

    @staticmethod
    def _status(
        calls: list[ToolCallRecord], records: list[EvidenceRecord]
    ) -> str:
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


# ── module-level helpers (also used by tests for unit coverage) ────────────


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


def _hints_for_role(hints: list[dict], role: str) -> list[dict]:
    return [h for h in hints if isinstance(h, dict) and h.get("role") == role]


def _first_hint_entity(hints: list[dict], role: str) -> Optional[str]:
    for h in _hints_for_role(hints, role):
        v = (h.get("entity") or "").strip()
        if v:
            return v
    return None


def _pick(d: dict, keys: Iterable[str]) -> Any:
    for k in keys:
        if k in d and d[k] not in (None, ""):
            return d[k]
    return None


def _coerce_year(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        y = int(str(value)[:4])
    except (TypeError, ValueError):
        return None
    return y if 1900 <= y <= 2100 else None


def _normalize_doi(doi: Any) -> Optional[str]:
    if not doi:
        return None
    d = str(doi).strip().lower()
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
        if d.startswith(prefix):
            d = d[len(prefix):]
    d = d.strip().strip("/")
    return d or None


def _normalize_title(title: Any) -> Optional[str]:
    if not title:
        return None
    t = str(title).lower().strip()
    t = _TITLE_PUNCT.sub(" ", t)
    t = _WS.sub(" ", t).strip()
    return t or None


def _extract_hits(
    payload: Any,
    *,
    source_tool: str,
    source_ref: Optional[str],
    query_role: str,
    query_term: str,
    candidate_id: str = "",
    limit: int = DEFAULT_PER_QUERY_LIMIT,
) -> list[_RawHit]:
    if not isinstance(payload, dict):
        return []
    items: list = []
    for key in _HIT_LIST_KEYS:
        v = payload.get(key)
        if isinstance(v, list) and v:
            items = v
            break
    out: list[_RawHit] = []
    for raw in items[:limit]:
        if not isinstance(raw, dict):
            continue
        out.append(
            _RawHit(
                title=_pick(raw, _TITLE_KEYS),
                doi=_pick(raw, _DOI_KEYS),
                link=_pick(raw, _LINK_KEYS),
                year=_coerce_year(_pick(raw, _YEAR_KEYS)),
                abstract=_pick(raw, _ABSTRACT_KEYS),
                source_tool=source_tool,
                source_ref=source_ref,
                query_role=query_role,
                query_term=query_term,
                candidate_id=candidate_id,
            )
        )
    return out


def _score_hit(hit: _RawHit) -> float:
    """Deterministic relevance score over title + abstract + DOI + year.

    Higher = more relevant for an ADC systematic review. Pure heuristic;
    no LLM, no hidden randomness.
    """
    score = 0.0
    title_l = (hit.title or "").lower()
    abs_l = (hit.abstract or "").lower()
    term_l = (hit.query_term or "").lower().strip()
    if term_l:
        if term_l in title_l:
            score += 2.0
        elif term_l in abs_l:
            score += 1.0
    if any(tok in title_l for tok in _ADC_TOKENS):
        score += 2.0
    elif any(tok in abs_l for tok in _ADC_TOKENS):
        score += 1.0
    if any(tok in title_l for tok in _PAYLOAD_TOKENS):
        score += 1.0
    if any(tok in title_l for tok in _LINKER_TOKENS):
        score += 0.5
    if hit.doi:
        score += 1.0
    if hit.year and hit.year >= 2018:
        score += 1.0
    return score


def _dedup_and_sort_by_relevance(hits: list[_RawHit]) -> list[_MergedHit]:
    """Dedup literature hits by DOI (preferred) or normalized title, then
    sort by deterministic evidence-relevance score (descending).

    Operates on literature hits only; this is NOT Step 12 candidate ranking.
    """
    by_key: dict[tuple[str, str], _MergedHit] = {}
    unkeyed: list[_MergedHit] = []
    for h in hits:
        nd = _normalize_doi(h.doi)
        nt = _normalize_title(h.title)
        if nd:
            key = ("doi", nd)
        elif nt:
            key = ("title", nt)
        else:
            unkeyed.append(_to_merged(h, score=_score_hit(h)))
            continue
        existing = by_key.get(key)
        if existing is None:
            by_key[key] = _to_merged(h, score=_score_hit(h))
        else:
            _merge_into(existing, h)
    merged_list = list(by_key.values()) + unkeyed
    # Cross-source bonus (up to +2 for hits supported by multiple sources).
    for m in merged_list:
        m.score += min(2.0, max(0, len(set(m.sources)) - 1))
    merged_list.sort(key=lambda m: (-m.score, (m.title or "").lower()))
    return merged_list


def _to_merged(hit: _RawHit, *, score: float) -> _MergedHit:
    return _MergedHit(
        title=hit.title,
        doi=hit.doi,
        link=hit.link,
        year=hit.year,
        query_role=hit.query_role,
        query_term=hit.query_term,
        candidate_id=hit.candidate_id,
        sources=[hit.source_tool],
        source_refs=[hit.source_ref] if hit.source_ref else [],
        has_abstract=bool(hit.abstract),
        score=score,
    )


def _merge_into(target: _MergedHit, hit: _RawHit) -> None:
    if hit.source_tool not in target.sources:
        target.sources.append(hit.source_tool)
    if hit.source_ref and hit.source_ref not in target.source_refs:
        target.source_refs.append(hit.source_ref)
    # Promote richer fields if the merged record was missing them.
    if not target.title and hit.title:
        target.title = hit.title
    if not target.doi and hit.doi:
        target.doi = hit.doi
    if not target.link and hit.link:
        target.link = hit.link
    if not target.year and hit.year:
        target.year = hit.year
    if hit.abstract:
        target.has_abstract = True
    # Re-score with the now-richer record so DOI/year bonuses apply.
    rescored = _score_hit(
        _RawHit(
            title=target.title, doi=target.doi, link=target.link, year=target.year,
            abstract=hit.abstract if hit.abstract else None,
            source_tool=hit.source_tool, source_ref=hit.source_ref,
            query_role=target.query_role, query_term=target.query_term,
        )
    )
    if rescored > target.score:
        target.score = rescored


def _theme_for(role: str) -> str:
    return {
        "complete_adc": "complete_adc",
        "linker_payload": "linker_payload",
        "payload": "payload",
        "linker": "linker",
        "compound": "compound",
        "target": "target_or_antigen",
        "antibody": "antibody",
        "candidate": "candidate_label",
    }.get(role, role or "literature")


def _record_from_merged(m: _MergedHit, *, target_text: Optional[str]) -> EvidenceRecord:
    primary_source = m.sources[0] if m.sources else ""
    return EvidenceRecord(
        evidence_id=new_artifact_id("evidence"),
        candidate_id=m.candidate_id,
        target=target_text,
        evidence_type=f"{m.query_role}_literature" if m.query_role else "literature",
        key_finding=(
            f"Literature hit '{(m.title or '').strip()[:120]}' matched "
            f"{m.query_role} query '{m.query_term}'. "
            f"Raw payload via source_refs."
        ).strip(),
        source=primary_source,
        confidence_score=round(min(1.0, m.score / 8.0), 3),
        title=(m.title or None),
        doi=_normalize_doi(m.doi),
        link=m.link,
        year=m.year,
        theme=_theme_for(m.query_role),
        query_role=m.query_role or None,
        query_term=m.query_term or None,
        relevance_score=round(m.score, 3),
        sources=list(dict.fromkeys(m.sources)),
        source_refs=list(dict.fromkeys(m.source_refs)),
    )
