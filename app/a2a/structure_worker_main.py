"""Container entrypoint for the Structure & Design A2A worker (Turn E).

Reads the advertised AgentCard URL from Settings (``STRUCTURE_WORKER_URL``) and
the bind host/port from the environment, then hands off to the EXISTING
``run_structure_worker`` — which builds the worker from ``app.deps`` and serves it
via ``python_a2a.A2AServer`` (through ``serve_worker_http``), failing fast on a
port-in-use or advertised-URL/bind-port mismatch.

This entrypoint re-implements no worker business logic, never calls
``StructureAndDesignAgent`` directly, never bypasses ``python_a2a.A2AServer``, and
never falls back to an in-process call or a scanned/alternate port.
"""

from __future__ import annotations

import os

from ..settings import get_settings
from .structure_worker import run_structure_worker


def main() -> None:
    settings = get_settings()
    run_structure_worker(
        url=settings.structure_worker_url,
        host=os.environ.get("WORKER_BIND_HOST", "0.0.0.0"),
        port=int(os.environ.get("WORKER_BIND_PORT", "8009")),
    )


if __name__ == "__main__":  # pragma: no cover - container process entrypoint
    main()
