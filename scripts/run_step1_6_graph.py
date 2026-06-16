"""Run the LangGraph Step 1→6 pipeline against local storage.

Uses the default deps wiring: LocalStorage, MockLLMProvider, and an
inventory-scoped LocalMCPClient. No outbound network calls — unwired wrappers
gracefully return `dependency_unavailable`.

For the Step 1-4 only variant see `scripts/run_minimal_graph.py`.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.deps import (  # noqa: E402
    get_llm_provider,
    get_mcp_client,
    get_registry_service,
    get_storage,
    get_workflow_state_service,
)
from app.graph.adc_graph import build_pipeline_graph  # noqa: E402


def main() -> int:
    os.environ.setdefault("STORAGE_MODE", "local")
    graph = build_pipeline_graph(
        storage=get_storage(),
        registry=get_registry_service(),
        workflow_state=get_workflow_state_service(),
        mcp_client=get_mcp_client(),
        llm=get_llm_provider(),
    )
    final = graph.invoke(
        {
            "intake_request": {
                "raw_user_query": "Design an ADC against HER2 with vc-MMAE payload, DAR 4",
                "user_provided_context": {
                    "target_or_antigen_text": "HER2",
                    "candidate_text": "Trastuzumab analog",
                    "payload_linker_text": "vc-MMAE",
                    "constraints_text": "Prefer DAR 4",
                },
            }
        }
    )
    artifacts = final["artifacts"]
    print(
        json.dumps(
            {
                "run_id": final["run_id"],
                "step_01_raw_request_record_id": artifacts.get("raw_request_record"),
                "step_02_structured_query_id": artifacts.get("structured_query"),
                "step_03_input_readiness_status_id": artifacts.get("input_readiness_status"),
                "step_04_run_step_plan_id": artifacts.get("run_step_plan"),
                "step_05_candidate_context_table_id": artifacts.get("candidate_context_table"),
                "step_06_structured_liability_summary_id": artifacts.get(
                    "structured_liability_summary"
                ),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
