"""FastAPI dependency-injection wiring."""

from __future__ import annotations

from functools import lru_cache

from .settings import get_settings
from .services.storage_service import Storage
from .services.storage_local import LocalStorage
from .services.storage_s3 import S3Storage
from .services.artifact_registry_service import ArtifactRegistryService
from .services.workflow_state_service import WorkflowStateService
from .services.tool_inventory_service import ToolInventoryService
from .llm.provider import LLMProvider, MockLLMProvider
from .llm.gemini_provider import GeminiProvider
from .llm.openai_provider import OpenAIProvider
from .mcp.client import LocalMCPClient, MCPClient


@lru_cache(maxsize=1)
def get_storage() -> Storage:
    settings = get_settings()
    if settings.storage_mode == "local":
        return LocalStorage(root=settings.local_storage_root, prefix=settings.s3_prefix)
    return S3Storage(bucket=settings.s3_bucket, prefix=settings.s3_prefix, region=settings.aws_region)


@lru_cache(maxsize=1)
def get_registry_service() -> ArtifactRegistryService:
    return ArtifactRegistryService(storage=get_storage())


@lru_cache(maxsize=1)
def get_workflow_state_service() -> WorkflowStateService:
    return WorkflowStateService(storage=get_storage())


@lru_cache(maxsize=1)
def get_tool_inventory_service() -> ToolInventoryService:
    return ToolInventoryService(get_settings().tool_inventory_xlsx)


@lru_cache(maxsize=1)
def get_mcp_client() -> MCPClient:
    """Default in-process MCP client.

    Always inventory-scoped so out-of-step / out-of-v0.2 tool calls are
    rejected uniformly in dev, test, and prod paths.
    """
    return LocalMCPClient(inventory=get_tool_inventory_service())


@lru_cache(maxsize=1)
def get_llm_provider() -> LLMProvider:
    """Return the configured LLM provider.

    Live providers (gemini / openai) are selected ONLY when
    ``LLM_PROVIDER`` is set explicitly. Having the corresponding API key
    is necessary but not sufficient, so local/test runs stay deterministic
    unless a live provider is explicitly requested.
    """
    settings = get_settings()
    if settings.llm_provider == "gemini":
        if not settings.gemini_api_key:
            raise ValueError(
                "LLM_PROVIDER=gemini but GEMINI_API_KEY is empty; "
                "either set the key or use LLM_PROVIDER=mock."
            )
        return GeminiProvider(api_key=settings.gemini_api_key, model=settings.gemini_model)
    if settings.llm_provider == "openai":
        if not settings.openai_api_key:
            raise ValueError(
                "LLM_PROVIDER=openai but OPENAI_API_KEY is empty; "
                "either set the key or use LLM_PROVIDER=mock."
            )
        return OpenAIProvider(api_key=settings.openai_api_key, model=settings.openai_model)
    return MockLLMProvider()
