"""Evidence wrappers (Step 13).

Thin MCP binding layer. `_live=False` (default) returns a deterministic
mock envelope. `_live=True` routes through `ToolUniverseAdapter` — there
is NO parallel manual httpx implementation in this file, by design.
Wrappers whose names are not yet supported by ToolUniverse raise
`NotImplementedError` on `_live=True`.

Audit doc: `\u9879\u76ee\u6587\u4ef6/ToolUniverse_Runtime_Integration_Audit_v0.1.md`.
"""

from __future__ import annotations

from typing import Any

from ..outcome import composite_scope_block_envelope


def _mocked(*, source: str, query: Any, **extra: Any) -> dict[str, Any]:
    return {
        "status": "mocked",
        "source": source,
        "query": query,
        "results": [],
        **extra,
    }


def _tu(name: str, args: dict[str, Any]) -> dict[str, Any]:
    from ..tooluniverse_adapter import call_tool

    return call_tool(name, args)


def LiteratureSearchTool(
    query: str = "", research_topic: str = "", *, _live: bool = False
) -> dict[str, Any]:
    """Composite literature search + AI summary (TU `ComposeTool`).

    TU `ComposeTool` whose required_tools include `MedicalLiteratureReviewer`
    (agentic LLM summarizer) plus EuropePMC / OpenAlex / PubTator3 search
    backends. Wrapper accepts either the legacy `query` arg or TU's
    canonical `research_topic`; forwards to TU as `{research_topic}`.

    Running the live path requires an LLM key recognized by ToolUniverse
    (e.g. `GEMINI_API_KEY`) present in the process environment. The key
    is consumed by TU internally — it is NEVER forwarded as a tool input,
    never logged, never embedded in normalized artifacts.
    """
    payload = query or research_topic
    if not payload:
        raise ValueError(
            "LiteratureSearchTool requires a non-empty `query` (or "
            "`research_topic`)"
        )
    if not _live:
        return _mocked(source="LiteratureSearchTool", query=payload)
    blocked = composite_scope_block_envelope("LiteratureSearchTool")
    if blocked is not None:
        return blocked
    return _tu("LiteratureSearchTool", {"research_topic": payload})


def EuropePMC_search_articles(
    query: str,
    *,
    limit: int | None = None,
    require_has_ft: bool | None = None,
    fulltext_terms: list[str] | None = None,
    enrich_missing_abstract: bool | None = None,
    extract_terms_from_fulltext: list[str] | None = None,
    page_size: int | None = None,
    _live: bool = False,
) -> dict[str, Any]:
    if limit is not None and page_size is not None:
        raise ValueError("alias_conflict:limit|page_size")
    if not query:
        raise ValueError("EuropePMC_search_articles requires a non-empty query")
    if not _live:
        return _mocked(source="EuropePMC_search_articles", query=query)
    args: dict[str, Any] = {"query": query}
    for name, value in (
        ("limit", limit),
        ("require_has_ft", require_has_ft),
        ("fulltext_terms", fulltext_terms),
        ("enrich_missing_abstract", enrich_missing_abstract),
        ("extract_terms_from_fulltext", extract_terms_from_fulltext),
        ("page_size", page_size),
    ):
        if value is not None:
            args[name] = value
    return _tu("EuropePMC_search_articles", args)


def openalex_search_works(
    query: str | None = None,
    *,
    search: str | None = None,
    filter: str | None = None,
    require_has_fulltext: bool | None = None,
    fulltext_terms: list[str] | None = None,
    per_page: int | None = None,
    limit: int | None = None,
    page: int | None = None,
    sort: str | None = None,
    mailto: str | None = None,
    _live: bool = False,
) -> dict[str, Any]:
    if query is not None and search is not None:
        raise ValueError("alias_conflict:query|search")
    if per_page is not None and limit is not None:
        raise ValueError("alias_conflict:per_page|limit")
    if not query and not search:
        raise ValueError("openalex_search_works requires a non-empty query / search")
    display_query = query or search
    if not _live:
        return _mocked(source="openalex_search_works", query=display_query)
    args: dict[str, Any] = {}
    for name, value in (
        ("query", query),
        ("search", search),
        ("filter", filter),
        ("require_has_fulltext", require_has_fulltext),
        ("fulltext_terms", fulltext_terms),
        ("per_page", per_page),
        ("limit", limit),
        ("page", page),
        ("sort", sort),
        ("mailto", mailto),
    ):
        if value is not None:
            args[name] = value
    return _tu("openalex_search_works", args)


def PubTator3_LiteratureSearch(
    query: str,
    *,
    page: int = 0,
    page_size: int = 10,
    limit: int | None = None,
    _live: bool = False,
) -> dict[str, Any]:
    if not query:
        raise ValueError("PubTator3_LiteratureSearch requires a non-empty query")
    if not _live:
        return _mocked(source="PubTator3_LiteratureSearch", query=query)
    args: dict[str, Any] = {"query": query, "page": page, "page_size": page_size}
    if limit is not None:
        args["limit"] = limit
    return _tu("PubTator3_LiteratureSearch", args)


def PubTator3_get_annotations(
    pmid: str = "",
    *,
    pmids: str = "",
    concepts: str = "gene,disease,chemical,species,mutation,cellline",
    _live: bool = False,
) -> dict[str, Any]:
    """Legacy wrapper accepts a single `pmid`; TU's official schema requires
    `pmids` (comma-separated). Both kwargs are accepted; if both are
    provided the official `pmids` value wins (after equality check)."""
    from ._arg_compat import pick

    value = pick(pmids, pmid, name="pmids") or ""
    if not value:
        raise ValueError("PubTator3_get_annotations requires pmid / pmids")
    if not _live:
        return _mocked(source="PubTator3_get_annotations", query=value)
    return _tu("PubTator3_get_annotations", {"pmids": value, "concepts": concepts})


def SemanticScholar_search_papers(
    query: str,
    *,
    limit: int = 5,
    year: str | None = None,
    sort: str | None = None,
    include_abstract: bool = False,
    _live: bool = False,
) -> dict[str, Any]:
    if not query:
        raise ValueError("SemanticScholar_search_papers requires a non-empty query")
    if not _live:
        return _mocked(source="SemanticScholar_search_papers", query=query)
    args: dict[str, Any] = {
        "query": query,
        "limit": limit,
        "include_abstract": include_abstract,
    }
    if year is not None:
        args["year"] = year
    if sort is not None:
        args["sort"] = sort
    return _tu("SemanticScholar_search_papers", args)


def MultiAgentLiteratureSearch(
    query: str,
    max_iterations: int = 1,
    quality_threshold: float = 0.7,
    *,
    _live: bool = False,
) -> dict[str, Any]:
    """Iterative multi-agent literature search (TU `ComposeTool`).

    TU `ComposeTool` with 5 LLM agents (Intent/Keyword/ResultSummarizer/
    QualityChecker/OverallSummary) + 14-way search backend fan-out.
    Running the live path requires an LLM key recognized by ToolUniverse
    (e.g. `GEMINI_API_KEY`) present in the environment.

    `max_iterations` is **restricted to exactly 1** for now to keep cost predictable
    while the agentic execution policy stabilizes — a single iteration
    still exercises the full pipeline and surfaces enough signal for
    Step 13. Relax the disclosed constraint later if/when richer iteration is
    needed.
    """
    if not query:
        raise ValueError("MultiAgentLiteratureSearch requires a non-empty query")
    if not _live:
        return _mocked(source="MultiAgentLiteratureSearch", query=query, total_papers=0)
    if isinstance(max_iterations, bool) or int(max_iterations) != 1:
        raise ValueError("MultiAgentLiteratureSearch runtime policy requires max_iterations=1")
    blocked = composite_scope_block_envelope("MultiAgentLiteratureSearch")
    if blocked is not None:
        return blocked
    return _tu(
        "MultiAgentLiteratureSearch",
        {
            "query": query,
            "max_iterations": 1,
            "quality_threshold": quality_threshold,
        },
    )


BINDINGS = [
    ("LiteratureSearchTool", LiteratureSearchTool),
    ("EuropePMC_search_articles", EuropePMC_search_articles),
    ("openalex_search_works", openalex_search_works),
    ("PubTator3_LiteratureSearch", PubTator3_LiteratureSearch),
    ("PubTator3_get_annotations", PubTator3_get_annotations),
    ("SemanticScholar_search_papers", SemanticScholar_search_papers),
    ("MultiAgentLiteratureSearch", MultiAgentLiteratureSearch),
]
