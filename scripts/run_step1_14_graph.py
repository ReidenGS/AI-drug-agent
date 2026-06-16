"""Run the LangGraph Step 1→14 pipeline against local storage.

Same defaults as the prior smoke scripts. Step 10 hands off to external
scoring; without a result file Step 11/12 stay in awaiting status. Step 13
(EvidenceAgent) and Step 14 (PatentIPAgent) still run — they consume the
shortlist if available, but don't require ranked scores to do work.

For Step 1-12 only see `scripts/run_step1_12_graph.py`.
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
from app.graph.adc_graph import build_step1_14_graph  # noqa: E402


def main() -> int:
    os.environ.setdefault("STORAGE_MODE", "local")
    graph = build_step1_14_graph(
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
    results = final["results"]
    print(
        json.dumps(
            {
                "run_id": final["run_id"],
                "step_01": artifacts.get("raw_request_record"),
                "step_02": artifacts.get("structured_query"),
                "step_03": artifacts.get("input_readiness_status"),
                "step_04": artifacts.get("run_step_plan"),
                "step_05": artifacts.get("candidate_context_table"),
                "step_06": artifacts.get("structured_liability_summary"),
                "step_07": artifacts.get("prepared_structure_input_package"),
                "step_08": artifacts.get("structure_prediction_and_interface_results"),
                "step_09": artifacts.get("structure_variant_and_compound_screening"),
                "step_10": artifacts.get("scoring_handoff_package"),
                "step_11": artifacts.get("scoring_validation"),
                "step_12": artifacts.get("ranking_table"),
                "step_13": artifacts.get("scientific_evidence_table"),
                "step_14": artifacts.get("patent_prior_art_table"),
                "step_11_status": results["step_11"].get("validation_status"),
                "step_12_status": results["step_12"].get("ranking_status"),
                "step_13_status": results["step_13"].get("review_status"),
                "step_14_status": results["step_14"].get("patent_review_status"),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
