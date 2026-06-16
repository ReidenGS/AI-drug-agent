"""Seed a local ADC run end-to-end through Step 1→4 (deterministic chain).

Run with:
    STORAGE_MODE=local QUEUE_MODE=memory python scripts/seed_local_run.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.deps import get_registry_service, get_storage, get_workflow_state_service  # noqa: E402
from app.services.intake_service import IntakeService  # noqa: E402
from app.services.input_readiness_service import InputReadinessService  # noqa: E402
from app.services.workflow_setup_service import WorkflowSetupService  # noqa: E402
from app.schemas.step_02_structured_query import (  # noqa: E402
    SourceRawRequestRef,
    StructuredQuery,
    TaskIntent,
    MentionedEntities,
)
from app.utils.ids import new_artifact_id  # noqa: E402
from app.utils.time import now_iso  # noqa: E402


def main() -> None:
    os.environ.setdefault("STORAGE_MODE", "local")
    storage = get_storage()
    registry = get_registry_service()
    workflow_state = get_workflow_state_service()

    # ── Step 1: intake
    intake = IntakeService(storage=storage, registry=registry, workflow_state=workflow_state)
    raw = intake.submit(
        raw_user_query="Design an ADC against HER2 with vc-MMAE payload",
        user_provided_context={
            "target_or_antigen_text": "HER2",
            "candidate_text": "Trastuzumab analog",
            "payload_linker_text": "vc-MMAE",
            "constraints_text": "Prefer DAR 4",
        },
        uploaded_files=[],
    )
    run_id = raw.run_id
    print(f"Step 1 → run_id={run_id}")

    # ── Step 2: structured_query (stub, no LLM)
    sq = StructuredQuery(
        run_id=run_id,
        parsed_at=now_iso(),
        source_raw_request_ref=SourceRawRequestRef(
            raw_request_record_id=registry.get(run_id).active_artifacts.raw_request_record_id
        ),
        task_intent=TaskIntent(
            task_type="adc_design",
            task_type_confidence=0.9,
            modality="ADC",
            modality_confidence=0.95,
            user_goal_summary="HER2-targeted ADC, vc-MMAE payload, DAR 4 preferred",
        ),
        mentioned_entities=MentionedEntities(
            target_or_antigen_text="HER2",
            antibody_candidate_text="Trastuzumab analog",
            payload_text="MMAE",
            linker_text="vc",
        ),
    )
    sq_artifact_id = new_artifact_id("structured_query")
    storage.write_json(
        storage.run_key(run_id, "inputs/structured_query.json"),
        {"artifact_id": sq_artifact_id, **sq.model_dump()},
    )
    registry.update_active(run_id, structured_query_id=sq_artifact_id)
    workflow_state.mark(run_id, "step_02", "completed")
    print("Step 2 → structured_query persisted")

    # ── Step 3: input_readiness
    readiness = InputReadinessService(
        storage=storage, registry=registry, workflow_state=workflow_state
    ).check(run_id)
    print(f"Step 3 → input_readiness_status={readiness.input_readiness_status}")

    # ── Step 4: run_step_plan
    plan = WorkflowSetupService(
        storage=storage, registry=registry, workflow_state=workflow_state
    ).plan(run_id)
    print(f"Step 4 → plan_status={plan.plan_status} skipped={plan.skipped_step_ids}")

    print("\nArtifact registry:")
    print(json.dumps(registry.get(run_id).model_dump(), indent=2))


if __name__ == "__main__":
    main()
