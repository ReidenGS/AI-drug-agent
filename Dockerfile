# Shared production image for the Multi-Agent A2A v1 services (Turn E).
#
# ONE image is built and reused by all four business services (orchestrator +
# step5/step6/structure workers); only the container `command` differs. The
# ToolUniverse inventory is NOT copied in — it is mounted read-only at runtime
# (see docker-compose.yml), so the official file is never duplicated or altered.
#
# No credentials, .env, .git, local artifacts, or raw biological data are copied
# (enforced by .dockerignore). MCP scope / ToolUniverse inventory / tool names
# are unchanged.

FROM python:3.12-slim

# HMMER is a runtime dependency of ANARCI (antibody CDR3 numbering) used only
# when a worker actually executes a Step 5 task — not for discovery/health. It is
# installed so the image is production-honest (the real dependency set is NOT
# narrowed). build-essential covers any source builds during pip install.
RUN apt-get update \
    && apt-get install -y --no-install-recommends hmmer build-essential git \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DEFAULT_TIMEOUT=300 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Install the real project dependencies from pyproject.toml (python-a2a comes
# from there — no other A2A framework is installed).
COPY pyproject.toml README.md ./
COPY app ./app
COPY build_tools ./build_tools
# The shared production worker image needs ToolUniverse + ESM and the existing
# ADMET-AI runtime used by the registered Step 6 ADMETAI tools. Install Torch
# first from PyTorch's official CPU index and constrain the only subsequent
# extras resolution so a default-index CUDA build cannot replace it. The ESM
# immutable source is declared exactly once, in the deployment extra.
RUN python -m pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cpu \
        'torch==2.13.0+cpu' \
    && printf '%s\n' 'torch==2.13.0+cpu' > /tmp/cpu-constraints.txt \
    && python -m pip install --no-cache-dir \
        --constraint /tmp/cpu-constraints.txt \
        '.[deployment,admet]' \
    && python -m pip check \
    && python build_tools/check_cpu_dependencies.py \
    && rm -f /tmp/cpu-constraints.txt

# Shared local-storage mount point + read-only inventory mount point. The
# directories are created and owned by a non-root user; the actual contents come
# from Docker volumes / bind mounts at runtime.
RUN useradd --create-home --uid 10001 adc \
    && mkdir -p /data/localstore /opt/adc/inventory \
    && chown -R adc:adc /data /opt/adc

USER adc

# No default CMD: each service in docker-compose.yml sets its own module
# entrypoint (uvicorn for the orchestrator, `python -m app.a2a.<worker>_main`
# for each worker). This avoids any implicit/wrong default service.
