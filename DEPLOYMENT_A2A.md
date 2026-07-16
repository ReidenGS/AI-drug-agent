# Multi-Agent A2A v1 — production Orchestrator deployment (Turn G)

Single-server Docker Compose topology for the FastAPI Orchestrator and three
HTTP A2A workers. Step 4 performs run-scoped discovery, validated routing,
durable Postgres checkpointing, HTTP dispatch, ingestion, retry and resume. The
stack itself still creates no database, vector store, or Mem0 service.

## Production Step 4 lifecycle and API

FastAPI lifespan owns one `OrchestratorPostgresCheckpointRuntime` per backend
process. `LANGGRAPH_CHECKPOINT_DATABASE_URL` must name an externally managed
Postgres database. Startup/migration failure aborts startup; there is no
InMemory or direct-call fallback. Shutdown closes the runtime.
Mutating execute/resume calls also take a run-scoped Postgres advisory lock;
a competing backend fails compactly instead of planning or dispatching the same
run twice. Status is read-only and may inspect the latest durable checkpoint.

The production endpoints are:

- `POST /runs/{run_id}/steps/4/execute`: plan a fresh run, or resume when a
  checkpoint already exists.
- `POST /runs/{run_id}/steps/4/resume`: explicitly resume/reconcile an existing
  checkpoint.
- `GET /runs/{run_id}/steps/4/status`: read the durable compact state without
  worker discovery or dispatch.

Responses contain compact identities, statuses/counts, next wakeup and safe
artifact refs only. They never contain endpoints, Task/request/result bodies,
completion proofs, storage paths, artifact bodies, prompts, or credentials.
Worker execution remains exclusively `A2AClient -> HTTP A2AServer -> worker
core`; the legacy serial graph is not this API's execution path.

## Services

| Service            | Command                                  | Internal port | Host port                          |
| ------------------ | ---------------------------------------- | ------------- | ---------------------------------- |
| `orchestrator`                       | `uvicorn app.main:app --port 8000`        | 8000          | `${ORCHESTRATOR_HOST_PORT:-18080}` |
| `step5-context-agent`                | `python -m app.a2a.step5_worker_main`     | 8005 (expose) | none (internal only)               |
| `step6-developability-agent`         | `python -m app.a2a.step6_worker_main`     | 8006 (expose) | none (internal only)               |
| `step7-9-structure-design-agent`     | `python -m app.a2a.structure_worker_main` | 8009 (expose) | none (internal only)               |

- Each service key is also its explicit `container_name` and Docker DNS name:
  `orchestrator`, `step5-context-agent`, `step6-developability-agent`, and
  `step7-9-structure-design-agent`. Turn D discovery defaults therefore resolve
  `http://step5-context-agent:8005`,
  `http://step6-developability-agent:8006`, and
  `http://step7-9-structure-design-agent:8009`.
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

Compose has no silent mock/offline or worker-timeout default. Every deployment
command must set:

- `LLM_PROVIDER` — explicitly choose `gemini`, `openai`, `qwen`, or a deliberate
  `mock` test/development mode.
- `MCP_LIVE_TOOLS` — explicitly choose `true` or `false`.
- `ORCHESTRATOR_WORKER_TIMEOUT_SECONDS` — a finite positive transport budget
  selected from the deployed worker SLA.
- `LANGGRAPH_CHECKPOINT_DATABASE_URL` — the external Postgres checkpoint DSN.

Do not put credentials in Compose or this document. Supply provider credentials
through the deployment environment/secret mechanism. A discovery-only smoke may
temporarily use `LLM_PROVIDER=mock MCP_LIVE_TOOLS=false`; that smoke must not send
an A2A task or execute worker business logic, LLM, MCP, or biomedical tools.

### Real OpenAI GPT-5.5 deployment

Real OpenAI deployments add the provider-specific secret overlay rather than
putting `OPENAI_API_KEY` in a service environment or copying `.env` into the
image. Configure these values explicitly in the host deployment environment:

- `LLM_PROVIDER=openai`
- `OPENAI_MODEL=gpt-5.5`
- `OPENAI_API_KEY` (host secret consumed by Docker Compose)
- `LANGGRAPH_CHECKPOINT_DATABASE_URL`
- `ORCHESTRATOR_WORKER_TIMEOUT_SECONDS`
- `MCP_LIVE_TOOLS`

Then run both Compose files:

```bash
docker compose \
  -f docker-compose.yml \
  -f docker-compose.openai.yml \
  -p synagentics-a2a-v1 \
  up -d --build
```

The overlay declares `openai_api_key` from the host `OPENAI_API_KEY` and mounts
that same secret into all four services. Containers receive only
`OPENAI_API_KEY_FILE=/run/secrets/openai_api_key`; the key value is not a
container environment variable. Application resolution keeps a non-empty
direct `OPENAI_API_KEY` first for existing host workflows, otherwise reads and
strips `OPENAI_API_KEY_FILE`, and fails closed if neither yields a key.

For the first real LLM smoke, use `MCP_LIVE_TOOLS=false`. That disables live MCP
tool calls only: the Orchestrator, Step 5, Step 6, and Step 7–9 worker LLM
providers are still real GPT-5.5 through `OpenAIProvider`, not
`MockLLMProvider`. This is not evidence of live MCP or ToolUniverse execution.

### Live NVIDIA NIM and ESM Forge credentials

Live Structure/Design MCP tools require a third, least-privilege overlay. Set
`MCP_LIVE_TOOLS=true` and the host environment variable names
`NVIDIA_API_KEY` and `ESM_API_KEY`, then combine all three Compose files:

```bash
docker compose \
  -f docker-compose.yml \
  -f docker-compose.openai.yml \
  -f docker-compose.live-tools.yml \
  -p synagentics-a2a-v1 \
  up -d --build
```

`docker-compose.live-tools.yml` converts those two host values into Docker
secrets mounted only in `step7-9-structure-design-agent`. Its container
environment contains only `NVIDIA_API_KEY_FILE=/run/secrets/nvidia_api_key` and
`ESM_API_KEY_FILE=/run/secrets/esm_api_key`, never either key value. The
Orchestrator, Step 5, and Step 6 receive neither secret. The base Compose file
and the OpenAI-only overlay remain independently usable without these live-tool
credentials.

At runtime an already-set process credential wins; otherwise Settings resolves
the corresponding direct value and then the secret file. An explicitly
configured unreadable or empty file fails closed. If neither source is
configured, the resolver returns empty so the tool retains its existing
missing-credential/dependency-unavailable result; it does not manufacture an
offline or mocked success.

## Shared artifact / registry / workflow state

The external Postgres database stores compact LangGraph checkpoints only; it is
not an artifact database. `ArtifactRegistryService` and `WorkflowStateService`
both use the shared `Storage`. All four services therefore see the same run
artifacts:

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
  ORCHESTRATOR_WORKER_TIMEOUT_SECONDS=<finite-sla-seconds> \
  LANGGRAPH_CHECKPOINT_DATABASE_URL=<external-postgres-dsn> \
  docker compose -p synagentics-a2a-v1 up -d --build
LLM_PROVIDER=<explicit-provider> MCP_LIVE_TOOLS=<true-or-false> \
  ORCHESTRATOR_WORKER_TIMEOUT_SECONDS=<finite-sla-seconds> \
  LANGGRAPH_CHECKPOINT_DATABASE_URL=<external-postgres-dsn> \
  docker compose -p synagentics-a2a-v1 ps
LLM_PROVIDER=<explicit-provider> MCP_LIVE_TOOLS=<true-or-false> \
  ORCHESTRATOR_WORKER_TIMEOUT_SECONDS=<finite-sla-seconds> \
  LANGGRAPH_CHECKPOINT_DATABASE_URL=<external-postgres-dsn> \
  docker compose -p synagentics-a2a-v1 logs --no-color
```

Do not run `down -v` as part of an audit unless the owner explicitly authorizes
resource deletion. Older fixed global resources from an earlier smoke may still
exist and should only be reported, not removed automatically.

## Mem0 and application PostgreSQL/pgvector — not added

Mem0 and an application PostgreSQL/pgvector service are not deployed here. The
external LangGraph Postgres checkpointer is integrated, but it stores compact
runtime checkpoints only. A future memory service remains a separate turn.
Mem0 will not be the artifact store or routing authority.

Future memory-service boundary (for reference only — nothing below is created,
assumed, or depended on by this compose stack):

- future service name: `mem0`
- future internal URL example: `http://mem0:8010`
- future deployment must use PostgreSQL/pgvector
- the future database design must be reviewed separately
- the current compose stack creates **no** database service
- the Orchestrator requires its explicitly configured external checkpoint DB
- the current orchestrator and workers do **not** depend on Mem0
- artifact / workflow state continues to use the shared `LocalStorage` or
  `S3Storage`, unchanged

## Turn G boundaries

- Step 4 discovery is run-scoped, not an application-startup network probe.
- Eligible business tasks use HTTP A2A dispatch; no worker domain method is
  called directly.
- No natural-language Final Response Agent is added; terminal API outcomes are
  compact state only.
- No change to MCP scope, ToolUniverse inventory, registry, or tool names.
- No Mem0, vector store, or Compose-managed database service.
