"""Step 15-21 routers expose execute routes."""

from __future__ import annotations

from app.api import (
    step_15_ip_risk_integration_api,
    step_16_design_review_api,
    step_17_human_review_api,
    step_18_redesign_trigger_api,
    step_19_pipeline_rerun_api,
    step_20_output_package_api,
    step_21_run_tracking_api,
)


def test_step15_21_execute_routes_registered():
    routers = [
        step_15_ip_risk_integration_api.router,
        step_16_design_review_api.router,
        step_17_human_review_api.router,
        step_18_redesign_trigger_api.router,
        step_19_pipeline_rerun_api.router,
        step_20_output_package_api.router,
        step_21_run_tracking_api.router,
    ]
    for step, router in zip(range(15, 22), routers, strict=True):
        assert router.prefix == f"/runs/{{run_id}}/steps/{step}"
        assert any(
            route.path == f"/runs/{{run_id}}/steps/{step}/execute"
            for route in router.routes
        )
