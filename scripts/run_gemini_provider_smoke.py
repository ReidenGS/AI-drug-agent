"""Optional live smoke test for GeminiProvider.generate_json.

This script intentionally skips cleanly unless both:
- LLM_PROVIDER=gemini
- GEMINI_API_KEY is non-empty
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.deps import get_llm_provider
from app.llm.gemini_provider import GeminiProvider
from app.llm.gemini_smoke_checks import (  # noqa: E402
    stage1_payload,
    stage2_payload,
    structured_query_payload,
    top_level_keys,
    validate_stage1,
    validate_stage2,
    validate_structured_query,
)
from app.settings import get_settings


def main() -> int:
    settings = get_settings()
    if settings.llm_provider != "gemini" or not settings.gemini_api_key:
        print(
            "SKIP: set LLM_PROVIDER=gemini and GEMINI_API_KEY to run the live "
            "Gemini provider smoke test."
        )
        return 0

    provider = get_llm_provider()
    if not isinstance(provider, GeminiProvider):
        raise RuntimeError(f"Expected GeminiProvider, got {type(provider).__name__}")

    prompt, schema = structured_query_payload()
    out = validate_structured_query(provider.generate_json(prompt, schema=schema))
    print("PASS structured_query keys:", top_level_keys(out))

    prompt, schema = stage1_payload()
    out = validate_stage1(provider.generate_json(prompt, schema=schema), schema)
    print("PASS tool_selection_stage_1 keys:", top_level_keys(out))

    prompt, schema = stage2_payload()
    out = validate_stage2(provider.generate_json(prompt, schema=schema))
    print("PASS tool_selection_stage_2 keys:", top_level_keys(out))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
