from __future__ import annotations

from app.graph.adc_graph import build_minimal_graph


def test_minimal_pipeline_runs_steps_1_to_4(
    local_storage, registry_service, workflow_state_service
):
    graph = build_minimal_graph(
        storage=local_storage,
        registry=registry_service,
        workflow_state=workflow_state_service,
    )
    final = graph.invoke(
        {
            "intake_request": {
                "raw_user_query": "Design ADC against HER2 with vc-MMAE payload",
                "user_provided_context": {
                    "target_or_antigen_text": "HER2",
                    "candidate_text": "Trastuzumab analog",
                    "payload_linker_text": "vc-MMAE",
                },
            }
        }
    )

    run_id = final["run_id"]
    assert run_id.startswith("run_")
    artifacts = final["artifacts"]
    for key in ("raw_request_record", "structured_query", "input_readiness_status", "run_step_plan"):
        assert artifacts.get(key), f"missing artifact id for {key}"

    # workflow state confirms 4 steps completed
    state = workflow_state_service.get(run_id)
    for s in ("step_01", "step_02", "step_03", "step_04"):
        assert state["steps"][s] == "completed"

    # all four artifacts physically present in local storage
    for fname in (
        "inputs/raw_request_record.json",
        "inputs/structured_query.json",
        "inputs/input_readiness_status.json",
        "inputs/run_step_plan.json",
    ):
        assert local_storage.exists(local_storage.run_key(run_id, fname))
