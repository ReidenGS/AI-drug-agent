from __future__ import annotations

import pytest

from app.mcp.tools import evidence


@pytest.mark.parametrize(
    ("wrapper", "kwargs", "expected_name", "expected_args"),
    [
        (
            evidence.EuropePMC_search_articles,
            {
                "query": "HER2",
                "limit": 7,
                "require_has_ft": True,
                "fulltext_terms": ["ADC"],
                "enrich_missing_abstract": True,
                "extract_terms_from_fulltext": ["payload"],
            },
            "EuropePMC_search_articles",
            {
                "query": "HER2",
                "limit": 7,
                "require_has_ft": True,
                "fulltext_terms": ["ADC"],
                "enrich_missing_abstract": True,
                "extract_terms_from_fulltext": ["payload"],
            },
        ),
        (
            evidence.EuropePMC_search_articles,
            {"query": "HER2", "page_size": 50},
            "EuropePMC_search_articles",
            {"query": "HER2", "page_size": 50},
        ),
        (
            evidence.openalex_search_works,
            {
                "query": "HER2",
                "filter": "is_oa:true",
                "require_has_fulltext": True,
                "fulltext_terms": ["payload"],
                "per_page": 20,
                "page": 2,
                "sort": "cited_by_count:desc",
                "mailto": "science@example.org",
            },
            "openalex_search_works",
            {
                "query": "HER2",
                "filter": "is_oa:true",
                "require_has_fulltext": True,
                "fulltext_terms": ["payload"],
                "per_page": 20,
                "page": 2,
                "sort": "cited_by_count:desc",
                "mailto": "science@example.org",
            },
        ),
        (
            evidence.openalex_search_works,
            {"search": "ADC", "limit": 15},
            "openalex_search_works",
            {"search": "ADC", "limit": 15},
        ),
        (
            evidence.PubTator3_LiteratureSearch,
            {"query": "HER2", "page": 3, "page_size": 25, "limit": 40},
            "PubTator3_LiteratureSearch",
            {"query": "HER2", "page": 3, "page_size": 25, "limit": 40},
        ),
        (
            evidence.PubTator3_get_annotations,
            {"pmids": "123,456", "concepts": "gene,chemical"},
            "PubTator3_get_annotations",
            {"pmids": "123,456", "concepts": "gene,chemical"},
        ),
        (
            evidence.SemanticScholar_search_papers,
            {
                "query": "HER2",
                "limit": 12,
                "year": "2020-2026",
                "sort": "citationCount:desc",
                "include_abstract": True,
            },
            "SemanticScholar_search_papers",
            {
                "query": "HER2",
                "limit": 12,
                "year": "2020-2026",
                "sort": "citationCount:desc",
                "include_abstract": True,
            },
        ),
    ],
)
def test_all_official_parameters_are_forwarded_exactly(
    monkeypatch, wrapper, kwargs, expected_name, expected_args
):
    seen = {}

    def fake_tu(name, args):
        seen.update(name=name, args=args)
        return {"status": "ok"}

    monkeypatch.setattr(evidence, "_tu", fake_tu)
    wrapper(**kwargs, _live=True)
    assert seen == {"name": expected_name, "args": expected_args}


def test_multiagent_effective_defaults_and_runtime_constraint(monkeypatch):
    seen = {}
    monkeypatch.setattr(
        evidence,
        "_tu",
        lambda name, args: seen.update(name=name, args=args) or {"status": "ok"},
    )
    out = evidence.MultiAgentLiteratureSearch("HER2", _live=True)
    assert out["status"] == "dependency_unavailable"
    assert out["reason_code"] == "uncontained_tooluniverse_full_discovery"
    assert out["actual_execution_count"] == 0
    assert seen == {}
    with pytest.raises(ValueError, match="requires max_iterations=1"):
        evidence.MultiAgentLiteratureSearch("HER2", max_iterations=2, _live=True)


@pytest.mark.parametrize(
    ("wrapper", "kwargs", "error"),
    [
        (
            evidence.EuropePMC_search_articles,
            {"query": "HER2", "limit": 5, "page_size": 5},
            "alias_conflict:limit\\|page_size",
        ),
        (
            evidence.openalex_search_works,
            {"query": "HER2", "search": "ADC"},
            "alias_conflict:query\\|search",
        ),
        (
            evidence.openalex_search_works,
            {"query": "HER2", "per_page": 10, "limit": 10},
            "alias_conflict:per_page\\|limit",
        ),
    ],
)
def test_alias_conflicts_fail_closed_before_adapter(monkeypatch, wrapper, kwargs, error):
    called = False

    def fake_tu(_name, _args):
        nonlocal called
        called = True

    monkeypatch.setattr(evidence, "_tu", fake_tu)
    with pytest.raises(ValueError, match=error):
        wrapper(**kwargs, _live=True)
    assert called is False


@pytest.mark.parametrize(
    ("wrapper", "kwargs", "expected_forwarded", "official_defaults"),
    [
        (
            evidence.EuropePMC_search_articles,
            {"query": "HER2"},
            {"query": "HER2"},
            {
                "limit": 5,
                "require_has_ft": False,
                "enrich_missing_abstract": False,
            },
        ),
        (
            evidence.openalex_search_works,
            {"search": "ADC"},
            {"search": "ADC"},
            {"require_has_fulltext": False, "per_page": 10, "page": 1},
        ),
    ],
)
def test_unspecified_wrapper_args_defer_to_tooluniverse_official_defaults(
    monkeypatch, wrapper, kwargs, expected_forwarded, official_defaults
):
    seen = {}

    def fake_tu(_name, args):
        seen["forwarded"] = args
        seen["effective"] = {**official_defaults, **args}
        return {"status": "ok"}

    monkeypatch.setattr(evidence, "_tu", fake_tu)
    wrapper(**kwargs, _live=True)
    assert seen["forwarded"] == expected_forwarded
    assert seen["effective"] == {**official_defaults, **expected_forwarded}
