"""Run the minimal LangGraph Step 1→4 pipeline against local storage.

Step 1-4 is deterministic services + the Supervisor LLM parsing; no MCP tool
calls are made here. For Step 1-6 (which adds the Step 5 / 6 agents and the
MCP transport), use `scripts/run_step1_6_graph.py`.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.deps import get_registry_service, get_storage, get_workflow_state_service  # noqa: E402
from app.graph.adc_graph import build_minimal_graph  # noqa: E402


def main() -> None:
    os.environ.setdefault("STORAGE_MODE", "local")
    graph = build_minimal_graph(
        storage=get_storage(),
        registry=get_registry_service(),
        workflow_state=get_workflow_state_service(),
    )
    final = graph.invoke({
        "intake_request": {
            "raw_user_query": "Design an ADC against HER2 with vc-MMAE payload",
            "user_provided_context": {
                "target_or_antigen_text": "HER2",
                "candidate_text": "Trastuzumab analog",
                "payload_linker_text": "vc-MMAE",
                "constraints_text": "Prefer DAR 4",
            },
        },
    })
    print(json.dumps({"run_id": final["run_id"], "artifacts": final["artifacts"]}, indent=2))


if __name__ == "__main__":
    main()
