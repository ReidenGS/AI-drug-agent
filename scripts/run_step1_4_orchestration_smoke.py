"""Step 1 → 4 orchestration smoke (offline, deterministic).

Runs three scenarios end-to-end through the service layer:

  A. ready                — full ADC context → ready_to_execute plan.
  B. needs_user_input     — target present, payload missing.
  C. blocked              — no target signal at all.

For each scenario this prints a compact, professor-report-friendly summary:
scenario name, run_id, artifact ids (with relative storage paths),
readiness state, plan state, expected gate behavior. The full prompt body,
file bytes, and raw artifact JSON are NEVER printed.

Constraints honored:
- No live Gemini / no network — uses `MockLLMProvider`.
- No MCP / ToolUniverse call — Steps 1-4 are pure orchestration.
- No `.env` write / no API key printed.
- No file bytes printed.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

# Make `app.*` importable when run directly.
_HERE = Path(__file__).resolve().parent
_PROJECT = _HERE.parent
if str(_PROJECT) not in sys.path:
    sys.path.insert(0, str(_PROJECT))

from app.agents.supervisor_agent import SupervisorAgent  # noqa: E402
from app.llm.provider import MockLLMProvider  # noqa: E402
from app.services.artifact_registry_service import ArtifactRegistryService  # noqa: E402
from app.services.input_readiness_service import InputReadinessService  # noqa: E402
from app.services.intake_service import IntakeService  # noqa: E402
from app.services.storage_local import LocalStorage  # noqa: E402
from app.services.structured_query_service import StructuredQueryService  # noqa: E402
from app.services.workflow_setup_service import WorkflowSetupService  # noqa: E402
from app.services.workflow_state_service import WorkflowStateService  # noqa: E402
from app.utils.errors import WorkflowStateError  # noqa: E402


def _short(value: str | None, *, limit: int = 64) -> str:
    if not value:
        return "—"
    return value if len(value) <= limit else value[: limit - 1] + "…"


def _run_scenario(
    *,
    name: str,
    raw_user_query: str,
    user_provided_context: dict[str, Any],
    uploaded_files: list[dict] | None,
    expected_gate: str,
) -> dict[str, Any]:
    tmp = Path(tempfile.mkdtemp(prefix="adc_smoke_"))
    try:
        storage = LocalStorage(root=str(tmp), prefix="adc_pilot")
        registry = ArtifactRegistryService(storage=storage)
        workflow_state = WorkflowStateService(storage=storage)

        intake = IntakeService(
            storage=storage, registry=registry, workflow_state=workflow_state
        )
        rec = intake.submit(
            raw_user_query=raw_user_query,
            user_provided_context=user_provided_context,
            uploaded_files=uploaded_files,
        )

        StructuredQueryService(
            storage, registry, workflow_state,
            SupervisorAgent(llm=MockLLMProvider()),
        ).parse(rec.run_id)

        readiness = InputReadinessService(
            storage, registry, workflow_state
        ).check(rec.run_id)

        plan_status_str: str
        plan_artifact_id: str | None
        plan_skipped_ids: list[str] = []
        plan_error: str | None = None
        try:
            plan = WorkflowSetupService(
                storage, registry, workflow_state
            ).plan(rec.run_id)
            plan_status_str = plan.plan_status
            plan_skipped_ids = list(plan.skipped_step_ids)
            plan_artifact_id = (
                registry.get(rec.run_id).active_artifacts.run_step_plan_id
            )
        except WorkflowStateError as exc:
            plan_status_str = "not_planned"
            plan_artifact_id = None
            plan_error = exc.message

        reg_view = registry.get(rec.run_id).active_artifacts
        raw_id = reg_view.raw_request_record_id
        sq_id = reg_view.structured_query_id
        readiness_id = reg_view.input_readiness_status_id

        def _path(key: str) -> str:
            try:
                full = storage.run_key(rec.run_id, key)
                return str(Path(full).relative_to(tmp))
            except Exception:  # noqa: BLE001
                return key

        result = {
            "scenario": name,
            "run_id": rec.run_id,
            "raw_request_record": {
                "artifact_id": raw_id,
                "path": _path("inputs/raw_request_record.json"),
            },
            "structured_query": {
                "artifact_id": sq_id,
                "path": _path("inputs/structured_query.json"),
            },
            "input_readiness_status": {
                "artifact_id": readiness_id,
                "status": readiness.input_readiness_status,
                "summary": _short(readiness.readiness_summary, limit=120),
                "blocking_reasons": readiness.blocking_reasons,
            },
            "run_step_plan": {
                "artifact_id": plan_artifact_id,
                "status": plan_status_str,
                "skipped_step_ids": plan_skipped_ids,
                "error": plan_error,
            },
            "expected_gate_behavior": expected_gate,
        }
        return result
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _print_scenario(result: dict[str, Any]) -> None:
    name = result["scenario"]
    print(f"\n── scenario: {name} ──────────────────────────────────")
    print(f"  run_id                 = {result['run_id']}")
    print(
        f"  raw_request_record     = "
        f"{result['raw_request_record']['artifact_id']}  ({result['raw_request_record']['path']})"
    )
    print(
        f"  structured_query       = "
        f"{result['structured_query']['artifact_id']}  ({result['structured_query']['path']})"
    )
    rd = result["input_readiness_status"]
    print(f"  input_readiness_status = {rd['status']:<16}  artifact={rd['artifact_id']}")
    print(f"    summary              = {rd['summary']}")
    if rd["blocking_reasons"]:
        print(f"    blocking_reasons     = {rd['blocking_reasons']}")
    plan = result["run_step_plan"]
    if plan["status"] == "not_planned":
        print(f"  run_step_plan          = not_planned  (error: {plan['error']})")
    else:
        print(
            f"  run_step_plan          = {plan['status']:<16}  "
            f"artifact={plan['artifact_id']}"
        )
        if plan["skipped_step_ids"]:
            print(f"    skipped_step_ids     = {plan['skipped_step_ids']}")
    print(f"  expected_gate_behavior = {result['expected_gate_behavior']}")


_SCENARIOS = [
    {
        "name": "A_ready",
        "raw_user_query": "Design an ADC against HER2 with vc-MMAE payload",
        "user_provided_context": {
            "target_or_antigen_text": "HER2",
            "candidate_text": "Trastuzumab analog",
            "payload_linker_text": "vc-MMAE",
        },
        "uploaded_files": None,
        "expected_gate": (
            "plan_status=ready_to_execute → Step 5/6/13/14 nodes may run; "
            "Step 7-9 partial because no structure/sequence uploaded."
        ),
    },
    {
        "name": "B_needs_user_input",
        "raw_user_query": "Build an ADC against HER2",
        "user_provided_context": {
            "target_or_antigen_text": "HER2",
            "candidate_text": "Trastuzumab",
        },
        "uploaded_files": None,
        "expected_gate": (
            "plan_status=wait_for_input (payload missing) → Step 5/6 nodes / "
            "APIs gated with 409; user must supply payload/linker."
        ),
    },
    {
        "name": "C_blocked",
        "raw_user_query": "Help me get started — no specifics yet",
        "user_provided_context": {},
        "uploaded_files": None,
        "expected_gate": (
            "readiness=blocked (no target) → Step 4 refuses to plan; "
            "Step 5+ never reachable."
        ),
    },
]


def main() -> int:
    print("Step 1 → 4 orchestration smoke (offline, deterministic; "
          "MockLLMProvider; no MCP / ToolUniverse).")
    for sc in _SCENARIOS:
        result = _run_scenario(
            name=sc["name"],
            raw_user_query=sc["raw_user_query"],
            user_provided_context=sc["user_provided_context"],
            uploaded_files=sc["uploaded_files"],
            expected_gate=sc["expected_gate"],
        )
        _print_scenario(result)
    print("\nAll three scenarios completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
