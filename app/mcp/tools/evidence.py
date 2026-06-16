"""Evidence wrappers (Step 13).

Mock mode (`_live=False`, default) returns a deterministic envelope so the
Step 13 graph runs without network. `_live=True` would attempt the real
upstream HTTP call but is not yet wired — it raises NotImplementedError to
keep the boundary honest.
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


def LiteratureSearchTool(query: str, *, _live: bool = False) -> dict[str, Any]:
    if not query:
        raise ValueError("LiteratureSearchTool requires a non-empty query")
    if _live:
        raise NotImplementedError("LiteratureSearchTool live mode not wired")
    return _mocked(source="LiteratureSearchTool", query=query)


def EuropePMC_search_articles(query: str, *, _live: bool = False) -> dict[str, Any]:
    if not query:
        raise ValueError("EuropePMC_search_articles requires a non-empty query")
    if _live:
        raise NotImplementedError("EuropePMC live mode not wired")
    return _mocked(source="EuropePMC_search_articles", query=query)


def openalex_search_works(query: str, *, _live: bool = False) -> dict[str, Any]:
    if not query:
        raise ValueError("openalex_search_works requires a non-empty query")
    if _live:
        raise NotImplementedError("openalex live mode not wired")
    return _mocked(source="openalex_search_works", query=query)


def PubTator3_LiteratureSearch(query: str, *, _live: bool = False) -> dict[str, Any]:
    if not query:
        raise ValueError("PubTator3_LiteratureSearch requires a non-empty query")
    if _live:
        raise NotImplementedError("PubTator3 live mode not wired")
    return _mocked(source="PubTator3_LiteratureSearch", query=query)


def PubTator3_get_annotations(pmid: str, *, _live: bool = False) -> dict[str, Any]:
    if not pmid:
        raise ValueError("PubTator3_get_annotations requires a non-empty pmid")
    if _live:
        raise NotImplementedError("PubTator3 live mode not wired")
    return _mocked(source="PubTator3_get_annotations", query=pmid)


def SemanticScholar_search_papers(query: str, *, _live: bool = False) -> dict[str, Any]:
    if not query:
        raise ValueError("SemanticScholar_search_papers requires a non-empty query")
    if _live:
        raise NotImplementedError("SemanticScholar live mode not wired")
    return _mocked(source="SemanticScholar_search_papers", query=query)


def MultiAgentLiteratureSearch(query: str, *, _live: bool = False) -> dict[str, Any]:
    if not query:
        raise ValueError("MultiAgentLiteratureSearch requires a non-empty query")
    if _live:
        # Audit note: outer wrapper exists but internal dependencies report
        # total_papers=0, so live mode is not enabled yet.
        raise NotImplementedError("MultiAgentLiteratureSearch live mode not wired")
    return _mocked(source="MultiAgentLiteratureSearch", query=query, total_papers=0)


BINDINGS = [
    ("LiteratureSearchTool", LiteratureSearchTool),
    ("EuropePMC_search_articles", EuropePMC_search_articles),
    ("openalex_search_works", openalex_search_works),
    ("PubTator3_LiteratureSearch", PubTator3_LiteratureSearch),
    ("PubTator3_get_annotations", PubTator3_get_annotations),
    ("SemanticScholar_search_papers", SemanticScholar_search_papers),
    ("MultiAgentLiteratureSearch", MultiAgentLiteratureSearch),
]
