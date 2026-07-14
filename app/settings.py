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
SUPPORTED_LLM_PROVIDERS: tuple[str, ...] = ("mock", "gemini", "openai", "qwen")


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

    llm_provider: Literal["mock", "gemini", "openai", "qwen"] = "mock"
    gemini_api_key: str = ""
    # Default updated 2026-06: Google's `gemini-1.5-pro` returns 404 unavailable
    # on the v1beta endpoint for many new keys. `gemini-3.5-flash` is the
    # smallest model that currently answers across all surfaces we test
    # against. Override via env when the account exposes a different model.
    gemini_model: str = "gemini-3.5-flash"

    # OpenAI provider — JSON-only LLM channel; never used to call MCP tools
    # or external biomedical APIs.
    openai_api_key: str = ""
    # `gpt-4.1-mini` is a small JSON-capable model. Override via env for
    # heavier benchmark runs.
    openai_model: str = "gpt-4.1-mini"

    # Qwen provider — DashScope OpenAI-compatible JSON-only LLM channel.
    qwen_api_key: str = ""
    qwen_model: str = "qwen-plus"
    qwen_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    # Timeout (seconds) for synchronous Qwen API calls (read + JSON parse +
    # validation envelope). Finite timeout keeps smoke/interactive runs
    # bounded when Step 6 Stage 2 schema mapping stalls.
    qwen_timeout: float = 90.0

    # NVIDIA NIM credentials used only by ToolUniverse-backed MCP wrappers
    # (for example Step 8 complex prediction tools). This is bridged into
    # os.environ as NVIDIA_API_KEY immediately before ToolUniverse is built.
    nvidia_api_key: str = ""
    # EvolutionaryScale Forge credentials used by ToolUniverse ESM wrappers
    # (for example Step 9 ESM_generate_protein_sequence / ESM_score tools).
    # This is bridged into os.environ as ESM_API_KEY before ToolUniverse runs.
    esm_api_key: str = ""

    api_key: str = "dev-key"

    # \u2500\u2500 Multi-Agent A2A worker endpoints (Turn D) \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
    # Docker-internal service URLs the Orchestrator uses to DISCOVER each worker
    # over real HTTP A2A. Discovery is never triggered at import/create_app time;
    # it only runs when WorkerDiscoveryService.discover_for_run(run_id) is called.
    step5_worker_url: str = "http://step5-worker:8005"
    step6_worker_url: str = "http://step6-worker:8006"
    structure_worker_url: str = "http://structure-worker:8009"

    # Production network timeouts (seconds) for AgentCard discovery + health
    # probes. These are explicit deployment settings (not hidden test caps) and
    # must be > 0.
    a2a_discovery_timeout_seconds: float = 5.0
    a2a_health_timeout_seconds: float = 5.0
    # Deterministic worker retry policy: attempt 0 plus attempts 1..3.
    orchestrator_max_worker_retries: int = 3

    tool_inventory_xlsx: str = "../\u9879\u76ee\u6587\u4ef6/ToolUniversity_inventory_v0.2.xlsx"

    # Step 1 multipart upload limits (per-request, server-side; never trust the
    # frontend). Defaults are intentionally conservative — bump via env when
    # real uploads need it.
    max_upload_files_per_run: int = 10
    max_upload_bytes_per_file: int = 50 * 1024 * 1024  # 50 MiB

    # MCP wrapper live-mode policy. Default OFF — tests and pipeline smokes
    # MUST NOT touch the network unless explicitly enabled.
    #
    # Policy (see `should_use_live`):
    #   - `mcp_live_tools=False`            -> never inject `_live` (default).
    #   - `mcp_live_tools=True`, allowlist NON-EMPTY -> constrained smoke/debug
    #     mode: ONLY the listed tools (comma-separated) get `_live=True`.
    #   - `mcp_live_tools=True`, allowlist EMPTY -> production all-live mode:
    #     EVERY scoped registered tool call gets `_live=True`. Wrappers that
    #     do not support live must surface dependency_unavailable /
    #     upstream_error honestly — they must never mock a fake success.
    mcp_live_tools: bool = False
    mcp_live_tool_allowlist: str = ""
    # Timeout (seconds) for outbound live HTTP requests in tool wrappers
    # that still use httpx (i.e. wrappers NOT yet migrated to the
    # ToolUniverse adapter). Migrated wrappers ignore this — TU owns its
    # own timeout policy.
    mcp_live_http_timeout: float = 15.0
    # Outer wall-clock timeout for one ToolUniverse-backed live tool call.
    # This protects the pipeline from TU tools that use requests without a
    # per-request timeout. Long-running tools such as NvidiaNIM can override
    # via TOOLUNIVERSE_LIVE_CALL_TIMEOUT in the environment.
    tooluniverse_live_call_timeout: float = 60.0
    # Separate outer timeout for ToolUniverse-backed NvidiaNIM live tools.
    # Use this for Step 8 complex prediction calls that can exceed the
    # normal ToolUniverse timeout budget.
    nvidia_nim_live_call_timeout: float = 1800.0

    def live_tool_allowlist_set(self) -> frozenset[str]:
        raw = self.mcp_live_tool_allowlist or ""
        return frozenset(name.strip() for name in raw.split(",") if name.strip())

    def should_use_live(self, tool_name: str) -> bool:
        """Decide whether `_live=True` should be injected for `tool_name`.

        - live OFF -> False.
        - live ON + non-empty allowlist -> only listed tools (smoke/debug).
        - live ON + empty allowlist -> True for every scoped tool (production
          all-live; non-live wrappers surface dependency_unavailable /
          upstream_error honestly, never a mocked success).
        """
        if not self.mcp_live_tools:
            return False
        allowlist = self.live_tool_allowlist_set()
        if not allowlist:
            return True
        return tool_name in allowlist

    @field_validator(
        "a2a_discovery_timeout_seconds",
        "a2a_health_timeout_seconds",
    )
    @classmethod
    def _positive_a2a_timeout(cls, v: float) -> float:
        """A2A discovery / health timeouts are real production network budgets
        and must be strictly positive."""
        if v is None or float(v) <= 0:
            raise ValueError("A2A discovery/health timeout seconds must be > 0")
        return float(v)

    @field_validator("orchestrator_max_worker_retries")
    @classmethod
    def _worker_retry_limit(cls, v: int) -> int:
        if int(v) != 3:
            raise ValueError("ORCHESTRATOR_MAX_WORKER_RETRIES must equal 3")
        return int(v)

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
