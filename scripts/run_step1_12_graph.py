"""Run the LangGraph Step 1→12 pipeline against local storage.

Step 10 hands off to the external Yufei AEE module. Without a result file at
`./.localstore/adc_pilot/runs/{run_id}/inputs/external_scoring_result.json`,
Step 11 reports `awaiting_external_input` and Step 12 reports
`awaiting_external_scoring` — both honest "we don't fabricate" states.

For Step 1-9 only see `scripts/run_step1_9_graph.py`.
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
from app.graph.adc_graph import build_step1_12_graph  # noqa: E402


def main() -> int:
    os.environ.setdefault("STORAGE_MODE", "local")
    graph = build_step1_12_graph(
        storage=get_storage(),
        registry=get_registry_service(),
        workflow_state=get_workflow_state_service(),
        mcp_client=get_mcp_client(),
        llm=get_llm_provider(),
    )
    final = graph.invoke(
        {
            "intake_request": {
                "raw_user_query": (
                    "Design ADC against HER2 with vc-MMAE; reference PDB 1N8Z."
                ),
                "user_provided_context": {
                    "target_or_antigen_text": "HER2",
                    "candidate_text": "Trastuzumab analog",
                    "payload_linker_text": "vc-MMAE",
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
                "step_07_prepared_structure_input_package_id": artifacts.get(
                    "prepared_structure_input_package"
                ),
                "step_08_structure_prediction_and_interface_results_id": artifacts.get(
                    "structure_prediction_and_interface_results"
                ),
                "step_09_structure_variant_and_compound_screening_id": artifacts.get(
                    "structure_variant_and_compound_screening"
                ),
                "step_10_scoring_handoff_id": artifacts.get("scoring_handoff_package"),
                "step_11_scoring_validation_id": artifacts.get("scoring_validation"),
                "step_12_ranking_table_id": artifacts.get("ranking_table"),
                "step_11_status": final["results"]["step_11"].get("validation_status"),
                "step_12_status": final["results"]["step_12"].get("ranking_status"),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
