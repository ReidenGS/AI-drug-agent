"""Settings loaded from environment via pydantic-settings."""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# Centralised so tests, deps, and the smoke scripts agree on the canonical
# set of accepted provider names. The settings layer is the single place that
# turns user-facing forms (Gemini / GEMINI / Mock) into the lowercase
# canonical value the rest of the codebase consumes.
SUPPORTED_LLM_PROVIDERS: tuple[str, ...] = ("mock", "gemini")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    storage_mode: Literal["local", "s3"] = "local"
    local_storage_root: str = "./.localstore"

    queue_mode: Literal["memory", "sqs"] = "memory"

    aws_region: str = "us-east-1"
    s3_bucket: str = "synagentics-adc-pilot"
    s3_prefix: str = "adc_pilot"

    ddb_table: str = "synagentics-adc-runs"

    sqs_queue_url: str = ""

    llm_provider: Literal["mock", "gemini"] = "mock"
    gemini_api_key: str = ""
    # Default updated 2026-06: Google's `gemini-1.5-pro` returns 404 unavailable
    # on the v1beta endpoint for many new keys. `gemini-3.5-flash` is the
    # smallest model that currently answers across all surfaces we test
    # against. Override via env when the account exposes a different model.
    gemini_model: str = "gemini-3.5-flash"

    api_key: str = "dev-key"

    tool_inventory_xlsx: str = "../项目文件/ToolUniversity_inventory_v0.2.xlsx"

    # Step 1 multipart upload limits (per-request, server-side; never trust the
    # frontend). Defaults are intentionally conservative — bump via env when
    # real uploads need it.
    max_upload_files_per_run: int = 10
    max_upload_bytes_per_file: int = 50 * 1024 * 1024  # 50 MiB

    @field_validator("llm_provider", mode="before")
    @classmethod
    def _normalize_llm_provider(cls, v: Any) -> Any:
        """Accept any case in env (Gemini / GEMINI / mock / MOCK).

        Normalisation lives here so the Literal stays the source of truth for
        which provider names are legal — every other consumer just reads
        `settings.llm_provider` and gets the canonical lowercase form.
        """
        if isinstance(v, str):
            stripped = v.strip()
            if not stripped:
                return "mock"
            lowered = stripped.lower()
            if lowered not in SUPPORTED_LLM_PROVIDERS:
                raise ValueError(
                    f"LLM_PROVIDER={v!r} is not supported; "
                    f"expected one of {SUPPORTED_LLM_PROVIDERS} (case-insensitive)."
                )
            return lowered
        return v


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
