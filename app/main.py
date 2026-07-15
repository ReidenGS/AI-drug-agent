"""FastAPI app factory.

Wires the 14 step routers + run/health endpoints. Each step is intentionally a
separate file under app/api/ per project README — do NOT merge into one router
at this stage.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI

from .deps import build_orchestrator_application_service
from .graph.orchestrator_checkpoint_runtime import (
    OrchestratorCheckpointRuntimeError,
    build_production_checkpoint_runtime,
)
from .settings import get_settings
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
)


def create_app(
    *,
    checkpoint_runtime_factory: Callable[[Any], Any] | None = None,
    orchestrator_service_factory: Callable[[Any], Any] | None = None,
) -> FastAPI:
    runtime_factory = (
        checkpoint_runtime_factory or build_production_checkpoint_runtime
    )
    service_factory = (
        orchestrator_service_factory or build_orchestrator_application_service
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        settings = get_settings()
        app.state.orchestrator_checkpoint_runtime = None
        app.state.orchestrator_service = None
        app.state.orchestrator_unavailable_code = (
            "orchestrator_checkpoint_database_url_required"
        )
        dsn = settings.langgraph_checkpoint_database_url
        if dsn is None or not dsn.get_secret_value().strip():
            # Old APIs remain available, but Step 4 A2A never gets an
            # InMemory/direct-call fallback.
            yield
            return

        runtime = runtime_factory(settings)
        try:
            await runtime.startup()
        except Exception:
            raise OrchestratorCheckpointRuntimeError(
                "checkpoint_runtime_startup_failed"
            ) from None
        try:
            service = service_factory(runtime)
            app.state.orchestrator_checkpoint_runtime = runtime
            app.state.orchestrator_service = service
            app.state.orchestrator_unavailable_code = None
            yield
        finally:
            app.state.orchestrator_service = None
            app.state.orchestrator_checkpoint_runtime = None
            await runtime.shutdown()

    app = FastAPI(
        title="SynAgentics ADC Backend",
        version="0.1.0",
        lifespan=lifespan,
    )
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

    return app


app = create_app()
