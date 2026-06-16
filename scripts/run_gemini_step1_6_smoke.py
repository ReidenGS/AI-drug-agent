"""Optional live smoke for Gemini-backed Step 1→6 pipeline.

This script intentionally skips cleanly unless both:
- LLM_PROVIDER=gemini
- GEMINI_API_KEY is non-empty

It uses the normal inventory-scoped local MCP client. It does not enable live
external wrapper APIs.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.llm.gemini_smoke_checks import validate_step1_6_artifacts  # noqa: E402
from app.settings import get_settings  # noqa: E402


def main() -> int:
    os.environ.setdefault("STORAGE_MODE", "local")
    settings = get_settings()
    if settings.llm_provider != "gemini" or not settings.gemini_api_key:
        print(
            "SKIP: set LLM_PROVIDER=gemini and GEMINI_API_KEY to run the live "
            "Gemini Step 1→6 smoke test."
        )
        return 0

    from app.deps import (  # noqa: PLC0415
        get_llm_provider,
        get_mcp_client,
        get_registry_service,
        get_storage,
        get_workflow_state_service,
    )
    from app.graph.adc_graph import build_pipeline_graph  # noqa: PLC0415
    from app.llm.gemini_provider import GeminiProvider  # noqa: PLC0415

    provider = get_llm_provider()
    if not isinstance(provider, GeminiProvider):
        raise RuntimeError(f"Expected GeminiProvider, got {type(provider).__name__}")

    storage = get_storage()
    graph = build_pipeline_graph(
        storage=storage,
        registry=get_registry_service(),
        workflow_state=get_workflow_state_service(),
        mcp_client=get_mcp_client(),
        llm=provider,
    )
    final = graph.invoke(
        {
            "intake_request": {
                "raw_user_query": (
                    "Design an ADC against HER2 using trastuzumab and vc-MMAE payload."
                ),
                "user_provided_context": {
                    "target_or_antigen_text": "HER2",
                    "candidate_text": "trastuzumab",
                    "payload_linker_text": "vc-MMAE",
                },
            }
        }
    )

    run_id = final["run_id"]
    artifacts = final["artifacts"]
    structured_query_id = artifacts.get("structured_query")
    liability_id = artifacts.get("structured_liability_summary")
    validate_step1_6_artifacts(storage=storage, run_id=run_id, artifacts=artifacts)

    print(
        json.dumps(
            {
                "status": "PASS",
                "run_id": run_id,
                "covered_steps": "step_01_to_step_06",
                "step_02_artifact": structured_query_id,
                "step_06_artifact": liability_id,
                "step_06_selection_policy_version_found": True,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
