"""Container entrypoint for the Patent-Evidence HTTP A2A worker."""

from __future__ import annotations

import os

from ..settings import get_settings
from .patent_evidence_worker import run_patent_evidence_worker


def main() -> None:
    settings = get_settings()
    run_patent_evidence_worker(
        url=settings.patent_evidence_worker_url,
        host=os.environ.get("WORKER_BIND_HOST", "0.0.0.0"),
        port=int(os.environ.get("WORKER_BIND_PORT", "8014")),
    )


if __name__ == "__main__":  # pragma: no cover
    main()
