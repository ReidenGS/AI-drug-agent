"""Settings loaded from environment via pydantic-settings."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


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
    gemini_model: str = "gemini-1.5-pro"

    api_key: str = "dev-key"

    tool_inventory_xlsx: str = "../项目文件/ToolUniversity_inventory_v0.2.xlsx"

    # Step 1 multipart upload limits (per-request, server-side; never trust the
    # frontend). Defaults are intentionally conservative — bump via env when
    # real uploads need it.
    max_upload_files_per_run: int = 10
    max_upload_bytes_per_file: int = 50 * 1024 * 1024  # 50 MiB


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
