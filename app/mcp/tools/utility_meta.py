"""Utility / meta tool wrappers — registered for metadata coverage, deferred for live.

Both groups exist in ToolUniverse 1.2.2; both have well-formed parameter
schemas exposed by `tooluniverse_adapter.get_tool_specification(...)` so
Stage 1 description + Stage 2 schema come from TU directly. We register
the BINDINGS so the MCP server's tool count covers them, but `_live=True`
raises `NotImplementedError` until each tool's readiness profile is
explicitly approved — silent success would be unsafe given the side-effects
involved.

| Wrapper | TU class | Why `_live=True` deferred |
|---|---|---|
| `dynamic_package_discovery` | `DynamicPackageDiscovery` | Composes `WebSearchTool` internally and queries PyPI metadata, which can pull in an agentic web-search backend whose key/quota profile is not in scope. No call is dispatched until that's audited. |
| `embedding_database_create` | `EmbeddingDatabase` | Requires an embedding provider (OpenAI / Azure / HuggingFace / local) — vendor key + provisioning. |
| `embedding_database_add` | `EmbeddingDatabase` | Same provider requirement; also a stateful write to a managed vector DB. |
| `embedding_database_search` | `EmbeddingDatabase` | Same provider requirement plus per-query embedding cost. |
"""

from __future__ import annotations

from typing import Any


def _ni(*_a, **_kw):
    raise NotImplementedError


def dynamic_package_discovery(
    requirements: str = "",
    *,
    functionality: str | None = None,
    constraints: dict | None = None,
    _live: bool = False,
    **_extra: Any,
) -> dict[str, Any]:
    """TU DynamicPackageDiscovery — search/evaluate PyPI packages."""
    if not requirements:
        raise ValueError(
            "dynamic_package_discovery requires a non-empty requirements string"
        )
    if not _live:
        return {
            "status": "mocked",
            "source": "dynamic_package_discovery",
            "requirements": requirements,
            "functionality": functionality,
            "constraints": dict(constraints or {}),
            "candidates": [],
        }
    raise NotImplementedError(
        "dynamic_package_discovery live mode is deferred — composes "
        "WebSearchTool with an unaudited backend; not wired."
    )


def _embedding_db_mock(
    *, source: str, database_name: str, **fields: Any,
) -> dict[str, Any]:
    return {
        "status": "mocked",
        "source": source,
        "database_name": database_name,
        **fields,
    }


def embedding_database_create(
    database_name: str = "",
    documents: list | None = None,
    *,
    metadata: list | None = None,
    provider: str | None = None,
    model: str | None = None,
    description: str | None = None,
    _live: bool = False,
    **_extra: Any,
) -> dict[str, Any]:
    """TU EmbeddingDatabase — create a named vector store from `documents`."""
    if not database_name:
        raise ValueError(
            "embedding_database_create requires a non-empty database_name"
        )
    if documents is None or not isinstance(documents, list):
        raise ValueError(
            "embedding_database_create requires a list of documents"
        )
    if not _live:
        return _embedding_db_mock(
            source="embedding_database_create",
            database_name=database_name,
            documents=list(documents),
            metadata=list(metadata or []),
            provider=provider,
            model=model,
            description=description,
            created=None,
        )
    raise NotImplementedError(
        "embedding_database_create live mode is deferred — requires an "
        "embedding provider (OpenAI / Azure / HuggingFace / local); not wired."
    )


def embedding_database_add(
    database_name: str = "",
    documents: list | None = None,
    *,
    metadata: list | None = None,
    provider: str | None = None,
    model: str | None = None,
    _live: bool = False,
    **_extra: Any,
) -> dict[str, Any]:
    """TU EmbeddingDatabase — append documents to an existing store."""
    if not database_name:
        raise ValueError(
            "embedding_database_add requires a non-empty database_name"
        )
    if documents is None or not isinstance(documents, list):
        raise ValueError("embedding_database_add requires a list of documents")
    if not _live:
        return _embedding_db_mock(
            source="embedding_database_add",
            database_name=database_name,
            documents=list(documents),
            metadata=list(metadata or []),
            provider=provider,
            model=model,
            added=None,
        )
    raise NotImplementedError(
        "embedding_database_add live mode is deferred — requires an "
        "embedding provider; not wired."
    )


def embedding_database_search(
    database_name: str = "",
    query: str = "",
    *,
    top_k: int | None = None,
    filters: dict | None = None,
    provider: str | None = None,
    model: str | None = None,
    _live: bool = False,
    **_extra: Any,
) -> dict[str, Any]:
    """TU EmbeddingDatabase — semantic search over a stored vector DB."""
    if not database_name:
        raise ValueError(
            "embedding_database_search requires a non-empty database_name"
        )
    if not query:
        raise ValueError("embedding_database_search requires a non-empty query")
    if not _live:
        return _embedding_db_mock(
            source="embedding_database_search",
            database_name=database_name,
            query=query,
            top_k=top_k,
            filters=dict(filters or {}),
            provider=provider,
            model=model,
            hits=[],
        )
    raise NotImplementedError(
        "embedding_database_search live mode is deferred — requires an "
        "embedding provider; not wired."
    )


BINDINGS = [
    ("dynamic_package_discovery", dynamic_package_discovery),
    ("embedding_database_create", embedding_database_create),
    ("embedding_database_add", embedding_database_add),
    ("embedding_database_search", embedding_database_search),
]
