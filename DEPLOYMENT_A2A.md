# Multi-Agent A2A v1 — Docker Compose deployment (Turn E)

Single-server Docker Compose skeleton for the four business services. This turn
only brings the HTTP services + internal DNS up. It does **not** wire discovery
into startup, does **not** dispatch A2A tasks, and adds **no** database / vector
store / Mem0 service.

## Services

| Service            | Command                                  | Internal port | Host port                          |
| ------------------ | ---------------------------------------- | ------------- | ---------------------------------- |
| `orchestrator`     | `uvicorn app.main:app --port 8000`       | 8000          | `${ORCHESTRATOR_HOST_PORT:-18080}` |
| `step5-worker`     | `python -m app.a2a.step5_worker_main`    | 8005 (expose) | none (internal only)               |
| `step6-worker`     | `python -m app.a2a.step6_worker_main`    | 8006 (expose) | none (internal only)               |
| `structure-worker` | `python -m app.a2a.structure_worker_main`| 8009 (expose) | none (internal only)               |

- Service names are fixed (`orchestrator`, `step5-worker`, `step6-worker`,
  `structure-worker`) because Turn D discovery defaults resolve these Docker DNS
  names: `http://step5-worker:8005`, `http://step6-worker:8006`,
  `http://structure-worker:8009`.
- Network logical key: `adc_a2a_net` (bridge). Compose creates a project-scoped
  resource such as `synagentics-a2a-v1_adc_a2a_net`; no global `name` override
  bypasses project isolation. Workers are reachable only inside it.
- One shared image (`synagentics-adc-a2a:latest`) is built once and reused by all
  four services; only the `command` differs.

## Host port

The orchestrator is the **only** service mapped to the host, via
`ORCHESTRATOR_HOST_PORT` (default `18080`, chosen because it was verified free on
this host and is clear of the Dify/RAGFlow port ranges).

> **Before deployment, explicitly verify or set `ORCHESTRATOR_HOST_PORT`.**
> Do not assume the default port is free on the target AWS host. The workers
> deliberately have no host ports; expose them only for temporary local
> debugging, never in production.

## Target architecture

The Turn E image was built and exercised locally as `linux/arm64`. This proves
that the Dockerfile builds and runs on `linux/arm64` only; it does **not** claim
that `linux/amd64`, or an existing AWS x86_64/amd64 host, has been validated.

Before deploying to an existing AWS host, inspect both the host and Docker
daemon architecture:

```bash
uname -m
docker info --format '{{.Architecture}}'
```

The target AWS architecture must match the image architecture. Verify the
built image explicitly with `docker image inspect`. If the AWS host reports
`x86_64`/`amd64`, do not run the current ARM64 image directly. Either rebuild
on that AWS target host with `docker build`, or use Docker Buildx to explicitly
produce a `linux/amd64` image. Afterward, use `docker image inspect` to verify
that the resulting image has `Architecture=amd64`.

Do not hard-code `platform` in `docker-compose.yml`: the AWS host architecture
has not yet been confirmed. An architecture mismatch must fail fast; silent
QEMU/emulation fallback is not production success. An amd64 build must still
install CPU-only PyTorch from the official CPU index and pass the GPU dependency
guard, `pip check`, and the package import/version checks before deployment.

## ToolUniverse inventory

The official inventory is **not** copied into the image and is **never
modified**. It is mounted read-only at runtime:

- Source (host): `../项目文件/ToolUniversity_inventory_v0.2.xlsx`
  (relative to the compose file; resolves to
  `程序/项目文件/ToolUniversity_inventory_v0.2.xlsx`).
- Container path (identical for all four services):
  `/opt/adc/inventory/ToolUniversity_inventory_v0.2.xlsx`
- Env for all four services: `TOOL_INVENTORY_XLSX=/opt/adc/inventory/ToolUniversity_inventory_v0.2.xlsx`

The image itself installs the production Python runtimes through
`.[deployment,admet]`: pinned `tooluniverse==1.2.2`, ESM metadata version
`3.3.0`, and the existing ADMET-AI extra. The base project continues to support
Python `>=3.11`; the Docker deployment image specifically uses Python 3.12
because the ESM 3.3.0 Forge SDK metadata requires Python `>=3.12,<3.13`.
ESM has one dependency source: the deployment extra points to the immutable
official Biohub/ESM commit
`ba4d7124864eed323a93bf3cfefcd958f573b75a`, because that version is not yet
available from the Python package index. Mounting the workbook is not a
substitute for these Python packages.

The image installs `torch==2.13.0+cpu` from PyTorch's official CPU wheel index
before resolving the extras, then constrains that resolution to the same CPU
build. A build-time dependency guard rejects NVIDIA, CUDA, cuBLAS, cuDNN, NCCL,
NVSHMEM, or Triton packages. `PIP_DEFAULT_TIMEOUT=300` is finite; pip's normal
finite retry policy is retained rather than masking dependency failures.

## Explicit LLM and MCP live-mode selection

Compose has no silent mock/offline default. Every deployment command must set:

- `LLM_PROVIDER` — explicitly choose `gemini`, `openai`, `qwen`, or a deliberate
  `mock` test/development mode.
- `MCP_LIVE_TOOLS` — explicitly choose `true` or `false`.

Do not put credentials in Compose or this document. Supply provider credentials
through the deployment environment/secret mechanism. A discovery-only smoke may
temporarily use `LLM_PROVIDER=mock MCP_LIVE_TOOLS=false`; that smoke must not send
an A2A task or execute worker business logic, LLM, MCP, or biomedical tools.

## Shared artifact / registry / workflow state

There is no database. `ArtifactRegistryService` and `WorkflowStateService` both
use the shared `Storage`. All four services therefore see the same run artifacts:

- `STORAGE_MODE=local`: the logical volume `adc_local_store` is mounted at
  `/data/localstore` in all four services, with
  `LOCAL_STORAGE_ROOT=/data/localstore` everywhere. Compose creates a
  project-scoped resource such as
  `synagentics-a2a-v1_adc_local_store`; there is no fixed global volume name.
- `STORAGE_MODE=s3`: uses the existing `S3Storage`. Inject AWS credentials via the
  deployment environment (never baked into the image or committed to the compose
  file). The S3 artifact contract is unchanged.

Artifact bodies are never placed in A2A payloads.

## Health

- Orchestrator: existing FastAPI `/healthz` (200 → healthy).
- Workers: `python -m app.a2a.container_healthcheck` validates HTTP 200, JSON
  object, `status == "ok"`, exact `agent_id`, and the exact capability set.
  Missing or extra capabilities fail health. The probe uses only Python stdlib,
  prints only compact failure codes, and never invokes agents, LLM, MCP, A2A
  tasks, or biomedical tools.

`depends_on` is intentionally **not** used to gate the orchestrator on worker
health: the backend must start even when a worker is unavailable. Worker
readiness is decided at run time by the Turn D `WorkerDiscoveryService`, not by
compose `depends_on`.

## Bring-up (isolated project name)

```bash
# Isolated project name so this stack never touches Dify/RAGFlow.
LLM_PROVIDER=<explicit-provider> MCP_LIVE_TOOLS=<true-or-false> \
  docker compose -p synagentics-a2a-v1 up -d --build
LLM_PROVIDER=<explicit-provider> MCP_LIVE_TOOLS=<true-or-false> \
  docker compose -p synagentics-a2a-v1 ps
LLM_PROVIDER=<explicit-provider> MCP_LIVE_TOOLS=<true-or-false> \
  docker compose -p synagentics-a2a-v1 logs --no-color
```

Do not run `down -v` as part of an audit unless the owner explicitly authorizes
resource deletion. Older fixed global resources from an earlier smoke may still
exist and should only be reported, not removed automatically.

## Mem0 and PostgreSQL/pgvector — NOT in Turn E

Mem0 and PostgreSQL/pgvector are **not deployed or integrated in Turn E.**
The future memory service will be added in a separate implementation turn.
Mem0 will not be the artifact store or routing authority.

Future memory-service boundary (for reference only — nothing below is created,
assumed, or depended on by this compose stack):

- future service name: `mem0`
- future internal URL example: `http://mem0:8010`
- future deployment must use PostgreSQL/pgvector
- the future database design must be reviewed separately
- the current compose stack provides and assumes **no** database
- the current orchestrator and workers do **not** depend on Mem0
- artifact / workflow state continues to use the shared `LocalStorage` or
  `S3Storage`, unchanged

## Turn E boundaries

- No Step 4 LLM routing, no A2A task dispatch, no `send_task_async`.
- Discovery is not wired into startup/lifespan and `discover_for_run` is not
  auto-called.
- No change to MCP scope, ToolUniverse inventory, registry, or tool names.
- No new database, vector store, or Mem0 service.
