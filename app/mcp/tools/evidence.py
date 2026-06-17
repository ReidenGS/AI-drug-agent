"""Evidence wrappers (Step 13).

Thin MCP binding layer. `_live=False` (default) returns a deterministic
mock envelope. `_live=True` routes through `ToolUniverseAdapter` — there
is NO parallel manual httpx implementation in this file, by design.
Wrappers whose names are not yet supported by ToolUniverse raise
`NotImplementedError` on `_live=True`.

Audit doc: `项目文件/ToolUniverse_Runtime_Integration_Audit_v0.1.md`.
"""

from __future__ import annotations

from typing import Any


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
    return _tu("LiteratureSearchTool", {"research_topic": payload})


def EuropePMC_search_articles(
    query: str, *, page_size: int = 25, _live: bool = False
) -> dict[str, Any]:
    if not query:
        raise ValueError("EuropePMC_search_articles requires a non-empty query")
    if not _live:
        return _mocked(source="EuropePMC_search_articles", query=query)
    return _tu("EuropePMC_search_articles", {"query": query, "limit": page_size})


def openalex_search_works(query: str, *, _live: bool = False) -> dict[str, Any]:
    if not query:
        raise ValueError("openalex_search_works requires a non-empty query")
    if not _live:
        return _mocked(source="openalex_search_works", query=query)
    return _tu("openalex_search_works", {"query": query})


def PubTator3_LiteratureSearch(query: str, *, _live: bool = False) -> dict[str, Any]:
    if not query:
        raise ValueError("PubTator3_LiteratureSearch requires a non-empty query")
    if not _live:
        return _mocked(source="PubTator3_LiteratureSearch", query=query)
    return _tu("PubTator3_LiteratureSearch", {"query": query})


def PubTator3_get_annotations(pmid: str, *, _live: bool = False) -> dict[str, Any]:
    """Legacy wrapper accepts a single `pmid`; TU expects `pmids` (plural,
    comma-separated). We forward the value as `pmids` so a single id or a
    comma-separated list both work."""
    if not pmid:
        raise ValueError("PubTator3_get_annotations requires a non-empty pmid")
    if not _live:
        return _mocked(source="PubTator3_get_annotations", query=pmid)
    return _tu("PubTator3_get_annotations", {"pmids": pmid})


def SemanticScholar_search_papers(
    query: str, *, limit: int = 5, _live: bool = False
) -> dict[str, Any]:
    if not query:
        raise ValueError("SemanticScholar_search_papers requires a non-empty query")
    if not _live:
        return _mocked(source="SemanticScholar_search_papers", query=query)
    return _tu(
        "SemanticScholar_search_papers",
        {"query": query, "limit": max(1, min(int(limit), 100))},
    )


def MultiAgentLiteratureSearch(
    query: str,
    max_iterations: int = 1,
    quality_threshold: float = 0.5,
    *,
    _live: bool = False,
) -> dict[str, Any]:
    """Iterative multi-agent literature search (TU `ComposeTool`).

    TU `ComposeTool` with 5 LLM agents (Intent/Keyword/ResultSummarizer/
    QualityChecker/OverallSummary) + 14-way search backend fan-out.
    Running the live path requires an LLM key recognized by ToolUniverse
    (e.g. `GEMINI_API_KEY`) present in the environment.

    `max_iterations` is **clamped to 1** for now to keep cost predictable
    while the agentic execution policy stabilizes — a single iteration
    still exercises the full pipeline and surfaces enough signal for
    Step 13. Raise the clamp here later if/when richer iteration is
    needed.
    """
    if not query:
        raise ValueError("MultiAgentLiteratureSearch requires a non-empty query")
    if not _live:
        return _mocked(source="MultiAgentLiteratureSearch", query=query, total_papers=0)
    clamped_iters = max(1, min(int(max_iterations), 1))  # hard clamp to 1
    return _tu(
        "MultiAgentLiteratureSearch",
        {
            "query": query,
            "max_iterations": clamped_iters,
            "quality_threshold": float(quality_threshold),
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
