# SynAgentics ADC Backend

Backend MVP for the SynAgentics ADC agent pipeline. The current implementation focuses on a controlled, artifact-based ADC workflow from request intake through candidate context, developability, structure/design support, scoring handoff, evidence review, and patent/prior-art review.

This repository is designed as an engineering backend, not a general chatbot. Each pipeline step produces versioned artifacts, records tool calls, and preserves raw tool output separately from normalized business records.

## Current Scope

Implemented or scaffolded:

- FastAPI application with one API module per pipeline step.
- Local storage adapter for development and smoke tests.
- Artifact registry and workflow state services.
- Step 1 JSON and multipart request intake.
- Step 2 structured-query parsing with a provider abstraction for mock and Gemini-backed JSON generation.
- Step 3 deterministic input readiness checks.
- Step 4 deterministic fixed-pipeline planning and step gating.
- Step 5 candidate context construction.
- Step 6 developability and liability signal routing.
- Step 7-9 structure and design support agent.
- Step 10 external scoring handoff.
- Step 11 scoring-result validation.
- Step 12 deterministic ranking when external scores are available.
- Step 13 scientific evidence agent.
- Step 14 patent and prior-art agent.
- Inventory-scoped FastMCP registration for ToolUniverse-backed tools.
- Progressive tool selection: compact description first, selected-tool schema second.

Out of scope for the current MVP:

- Final scientific scoring formula and AEE model implementation.
- Formal legal patent conclusions.
- Full production AWS deployment.
- Full report generation, human review, redesign, rerun, and long-term memory steps.

## Architecture

```text
Client / UI / script
  -> FastAPI
    -> deterministic services
    -> local/S3 storage abstraction
    -> artifact registry
    -> LangGraph workflow
      -> step agents
        -> scoped MCP client
          -> inventory-scoped FastMCP server
            -> ToolUniverse adapter-backed wrappers
```

Important boundaries:

- Step 2 uses the LLM for structured request parsing only.
- Step 3 input readiness is deterministic.
- Step 4 fixed pipeline setup is deterministic.
- Agents do not receive the full ToolUniverse catalog.
- MCP registration is limited to the project inventory, not the full ToolUniverse registry.
- Raw tool outputs are stored under `tool_outputs/...` and referenced by `tool_output_ref`; they are not embedded in normalized artifacts.

## Requirements

- Python `>=3.11`
- FastAPI
- LangGraph
- python-a2a / FastMCP
- Pydantic v2
- google-genai, only when Gemini is enabled
- ToolUniverse, installed in the runtime environment used for live MCP-backed tools

Install for local development:

```bash
pip install -e ".[dev]"
```

## Environment

Create a local environment file from the example:

```bash
cp .env.example .env
```

`.env` is ignored by Git and must not be committed.

Default local execution uses the deterministic mock LLM provider:

```bash
LLM_PROVIDER=mock
```

Optional Gemini-backed JSON generation:

```bash
LLM_PROVIDER=gemini
GEMINI_API_KEY=your_key_here
GEMINI_MODEL=gemini-3.5-flash
```

`LLM_PROVIDER` is case-insensitive. If Google returns `404 model unavailable`, change `GEMINI_MODEL` to a model available to the account. No code change is required.

Optional live ToolUniverse execution:

```bash
MCP_LIVE_TOOLS=true
MCP_LIVE_TOOL_ALLOWLIST=EuropePMC_search_articles,ChEMBL_search_molecules
```

Live mode is opt-in and allowlist-based. Deferred tools return `dependency_unavailable` instead of silently pretending to succeed.

## Running Locally

Start the API:

```bash
uvicorn app.main:app --reload
```

Open API documentation:

```text
http://localhost:8000/docs
```

Run the minimal local pipeline smoke:

```bash
STORAGE_MODE=local QUEUE_MODE=memory python scripts/seed_local_run.py
```

Run graph smoke scripts:

```bash
python scripts/run_minimal_graph.py
python scripts/run_step1_6_graph.py
python scripts/run_step1_9_graph.py
python scripts/run_step1_12_graph.py
python scripts/run_step1_14_graph.py
```

## Request Intake

### JSON Intake

```bash
curl -X POST http://localhost:8000/runs \
  -H "Content-Type: application/json" \
  -d '{
    "raw_user_query": "Design an ADC against HER2 with vc-MMAE",
    "user_provided_context": {
      "target_or_antigen_text": "HER2",
      "candidate_text": "Trastuzumab analog",
      "payload_linker_text": "vc-MMAE"
    }
  }'
```

### Multipart Intake

Use this endpoint to simulate a frontend submission with natural language and uploaded files:

```bash
curl -X POST http://localhost:8000/runs/multipart \
  -F 'raw_user_query=Design an ADC against HER2 with vc-MMAE' \
  -F 'entry_source=ui' \
  -F 'submitted_by=tester@example.com' \
  -F 'user_provided_context={"target_or_antigen_text":"HER2","candidate_text":"Trastuzumab","payload_linker_text":"vc-MMAE"}' \
  -F 'files=@/path/to/complex.pdb;type=chemical/x-pdb' \
  -F 'files=@/path/to/heavy_chain.fasta;type=text/x-fasta'
```

Multipart safeguards:

- `raw_user_query` must be non-empty.
- `user_provided_context` must be a JSON object.
- File count is limited by `MAX_UPLOAD_FILES_PER_RUN`.
- Per-file size is limited by `MAX_UPLOAD_BYTES_PER_FILE`.
- Failed requests clean up bytes written during that request.
- Uploaded file bytes are stored separately; JSON artifacts keep metadata and storage references only.

## Step 1-4 Orchestration

The entry orchestration path produces four canonical artifacts:

```text
raw_request_record
structured_query
input_readiness_status
run_step_plan
```

Step responsibilities:

- Step 1 records the request and uploaded-file metadata.
- Step 2 parses the request into a structured ADC query using JSON-only LLM output.
- Step 3 deterministically checks input completeness.
- Step 4 deterministically creates a fixed ADC pipeline plan and gates downstream execution.

Step 2 does not call MCP tools and does not create ToolUniverse parameters. Tool-specific parameter construction happens in the step that calls the tool.

## MCP and ToolUniverse

The MCP server is built with `python-a2a` FastMCP. It registers only tools listed in the project inventory. The full ToolUniverse registry is not exposed to agents.

Tool selection uses progressive disclosure:

1. Stage 1 exposes a compact catalog with official ToolUniverse descriptions.
2. The LLM selects tool names only.
3. Stage 2 exposes the official ToolUniverse parameter schema only for selected tools.
4. The backend validates arguments before calling MCP.

Runtime checks currently verify:

- 99 MCP bindings are registered.
- 99/99 registered tools expose official ToolUniverse descriptions and parameter schemas.
- Wrapper call signatures accept ToolUniverse official parameters.
- `_live` is never exposed to the LLM.
- Deferred and intentionally disabled wrappers fail safely.

ZINC wrappers remain intentionally disabled for live mode because the upstream endpoint is unstable and captcha-gated. They must not be reported as confirmed ZINC22 support.

## Evidence and Patent Review

Step 13 performs scientific evidence search and normalization. Step 14 performs patent, prior-art, and regulatory reference scanning.

These steps:

- read upstream structured artifacts;
- route scoped MCP calls;
- write raw tool output under `tool_outputs/...`;
- keep normalized artifacts compact and auditable;
- treat `dependency_unavailable` and upstream errors as partial results, not as silent success.

Patent outputs are for demonstration and triage only. They are not legal opinions.

## External Scoring Handoff

Step 10 prepares an external scoring handoff package. Step 11 validates an external scoring result when one is present. Step 12 builds a deterministic ranking table from validated scores.

Without an external scoring result, Step 11 and Step 12 produce explicit `awaiting_external_input` / `awaiting_external_scoring` states. The system does not fabricate scores or rankings.

## Testing

Run all tests:

```bash
python -m pytest -q
```

Useful targeted suites:

```bash
python -m pytest tests/services -q
python -m pytest tests/mcp -q
python -m pytest tests/agents -q
python -m pytest tests/graph -q
```

MCP and metadata audits:

```bash
python scripts/run_mcp_smoke.py
python scripts/audit_tooluniverse_official_metadata.py
```

Gemini smoke tests are optional and skip cleanly when Gemini is not configured:

```bash
python scripts/run_gemini_provider_smoke.py
python scripts/run_gemini_step1_6_smoke.py
```

## Security and Data Handling

- Do not commit `.env`, API keys, local storage, or raw run artifacts.
- Do not print API keys, full prompts, full ToolUniverse payloads, or uploaded file bytes.
- Raw tool output is persisted by reference.
- Local development storage is ignored by Git.
- Live ToolUniverse calls are opt-in.

## Repository Hygiene

The public repository should contain English-facing documentation only. Local Chinese notes, local project-planning files, and private progress records are intentionally ignored.

