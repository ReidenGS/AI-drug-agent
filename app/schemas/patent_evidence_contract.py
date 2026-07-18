"""Shared typed authority for patent/evidence reference-to-argument mappings."""

from __future__ import annotations

from typing import Literal


PatentEvidenceInputRole = Literal[
    "linker_payload",
    "payload",
    "linker",
    "compound",
    "target",
    "complete_adc",
    "antibody",
    "pubchem_cid",
    "brand_name",
    "application_number",
    "drug_name",
    "query",
    "pmid",
    "pmids",
    "document_id",
    "title",
]

KNOWN_PATENT_EVIDENCE_ROLES = frozenset(PatentEvidenceInputRole.__args__)

# Direct literature-search backends that are both in the approved Step 13
# Patent-Evidence inventory and suitable for query-driven composite search.
# Composite tools must import this authority instead of maintaining a broader
# ToolUniverse discovery list of their own.
PATENT_EVIDENCE_MULTI_SEARCH_BACKENDS = frozenset(
    {
        "EuropePMC_search_articles",
        "openalex_search_works",
        "PubTator3_LiteratureSearch",
        "SemanticScholar_search_papers",
    }
)

_EVIDENCE_ENTITY_ROLES = frozenset(
    {
        "linker_payload",
        "payload",
        "linker",
        "compound",
        "target",
        "complete_adc",
        "antibody",
        "query",
    }
)

# Tool-specific official schema argument -> permitted typed reference roles.
# This is cataloged for the LLM and enforced again by deterministic validation.
PATENT_EVIDENCE_SCHEMA_ARG_ALLOWED_ROLES: dict[
    str, dict[str, frozenset[str]]
] = {
    "LiteratureSearchTool": {"research_topic": _EVIDENCE_ENTITY_ROLES},
    "EuropePMC_search_articles": {"query": _EVIDENCE_ENTITY_ROLES},
    "openalex_search_works": {
        "query": _EVIDENCE_ENTITY_ROLES,
        "search": _EVIDENCE_ENTITY_ROLES,
    },
    "PubTator3_LiteratureSearch": {"query": _EVIDENCE_ENTITY_ROLES},
    "PubTator3_get_annotations": {"pmids": frozenset({"pmid", "pmids"})},
    "SemanticScholar_search_papers": {"query": _EVIDENCE_ENTITY_ROLES},
    "ChEMBL_search_documents": {
        "document_id": frozenset({"document_id"}),
        "title__contains": frozenset({"title"}),
    },
    "MultiAgentLiteratureSearch": {"query": _EVIDENCE_ENTITY_ROLES},
    "PubChem_get_associated_patents_by_CID": {
        "cid": frozenset({"pubchem_cid"})
    },
    "FDA_OrangeBook_get_patent_info": {
        "brand_name": frozenset({"brand_name"}),
        "application_number": frozenset({"application_number"}),
    },
    "drugbank_get_drug_references_by_drug_name_or_id": {
        "query": frozenset({"drug_name", "query"})
    },
}

PATENT_EVIDENCE_SUPPORT_TO_SCHEMA_ARG: dict[str, dict[str, str]] = {
    "LiteratureSearchTool": {
        "research_topic": "research_topic",
        "query": "research_topic",
    },
    "EuropePMC_search_articles": {"query": "query"},
    "openalex_search_works": {"query": "query", "search": "search"},
    "PubTator3_LiteratureSearch": {"query": "query"},
    "PubTator3_get_annotations": {"pmids": "pmids", "pmid": "pmids"},
    "SemanticScholar_search_papers": {"query": "query"},
    "ChEMBL_search_documents": {
        "document_id": "document_id",
        "title__contains": "title__contains",
        "title_contains": "title__contains",
        "title": "title__contains",
    },
    "MultiAgentLiteratureSearch": {"query": "query"},
    "PubChem_get_associated_patents_by_CID": {
        "cid": "cid",
        "pubchem_cid": "cid",
    },
    "FDA_OrangeBook_get_patent_info": {
        "brand_name": "brand_name",
        "application_number": "application_number",
    },
    "drugbank_get_drug_references_by_drug_name_or_id": {
        "query": "query",
        "drug_name_or_id": "query",
    },
}

PATENT_EVIDENCE_SUPPORT_TOKEN_ALLOWED_ROLES: dict[str, frozenset[str]] = {}
for _tool_name, _support_map in PATENT_EVIDENCE_SUPPORT_TO_SCHEMA_ARG.items():
    for _support_token, _schema_arg in _support_map.items():
        _roles = PATENT_EVIDENCE_SCHEMA_ARG_ALLOWED_ROLES[_tool_name][_schema_arg]
        PATENT_EVIDENCE_SUPPORT_TOKEN_ALLOWED_ROLES[_support_token] = frozenset(
            PATENT_EVIDENCE_SUPPORT_TOKEN_ALLOWED_ROLES.get(_support_token, frozenset())
            | _roles
        )

KNOWN_PATENT_EVIDENCE_SUPPORT_TOKENS = frozenset(
    PATENT_EVIDENCE_SUPPORT_TOKEN_ALLOWED_ROLES
)
