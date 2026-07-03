"""FastAPI app factory.

Wires the 14 step routers + run/health endpoints. Each step is intentionally a
separate file under app/api/ per project README — do NOT merge into one router
at this stage.
"""

from __future__ import annotations

from fastapi import FastAPI

from .utils.errors import install_exception_handlers
from .api import (
    health_api,
    run_api,
    step_01_intake_api,
    step_01_intake_multipart_api,
    step_02_structured_query_api,
    step_03_input_readiness_api,
    step_04_workflow_setup_api,
    step_05_candidate_context_api,
    step_06_developability_api,
    step_07_structure_input_api,
    step_08_structure_evaluation_api,
    step_09_structure_design_api,
    step_10_scoring_handoff_api,
    step_11_scoring_validation_api,
    step_12_ranking_api,
    step_13_evidence_api,
    step_14_patent_ip_api,
    step_15_ip_risk_integration_api,
    step_16_design_review_api,
    step_17_human_review_api,
    step_18_redesign_trigger_api,
    step_19_pipeline_rerun_api,
    step_20_output_package_api,
    step_21_run_tracking_api,
)


def create_app() -> FastAPI:
    app = FastAPI(title="SynAgentics ADC Backend", version="0.1.0")
    install_exception_handlers(app)

    app.include_router(health_api.router)
    app.include_router(run_api.router)
    app.include_router(step_01_intake_multipart_api.router)
    app.include_router(step_01_intake_api.router)
    app.include_router(step_02_structured_query_api.router)
    app.include_router(step_03_input_readiness_api.router)
    app.include_router(step_04_workflow_setup_api.router)
    app.include_router(step_05_candidate_context_api.router)
    app.include_router(step_06_developability_api.router)
    app.include_router(step_07_structure_input_api.router)
    app.include_router(step_08_structure_evaluation_api.router)
    app.include_router(step_09_structure_design_api.router)
    app.include_router(step_10_scoring_handoff_api.router)
    app.include_router(step_11_scoring_validation_api.router)
    app.include_router(step_12_ranking_api.router)
    app.include_router(step_13_evidence_api.router)
    app.include_router(step_14_patent_ip_api.router)
    app.include_router(step_15_ip_risk_integration_api.router)
    app.include_router(step_16_design_review_api.router)
    app.include_router(step_17_human_review_api.router)
    app.include_router(step_18_redesign_trigger_api.router)
    app.include_router(step_19_pipeline_rerun_api.router)
    app.include_router(step_20_output_package_api.router)
    app.include_router(step_21_run_tracking_api.router)

    return app


app = create_app()
