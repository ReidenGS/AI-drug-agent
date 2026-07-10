"""A2A worker execution contract schemas (Turn A).

Field-level source of truth for the ADC business payload that the Step 4
Orchestrator serialises into ``python_a2a.Task.message`` and that a worker
returns in the A2A task response. See:

- ``A2A 方案/worker_execution_contract.md``
- ``A2A 方案/orchestrator_routing_design.md``

HARD CONSTRAINTS (see multi_agent_a2a_v1_implementation_plan.md):

- These are ADC *business* schemas, NOT an A2A protocol envelope. Turn B drops
  the serialized :class:`WorkerExecutionRequest` verbatim into
  ``python_a2a.Task.message`` and the compact :class:`A2ATaskMetadata` into
  ``python_a2a.Task.metadata``. Field names are therefore stable.
- Every model uses ``extra="forbid"``. An unknown field — including any raw
  material field such as ``raw_sequence`` / ``pdb_body`` / ``api_key`` /
  ``raw_tooluniverse_payload`` / ``full_prompt`` / ``raw_llm_response`` — is
  rejected, never silently dropped.
- The request may only carry *compact* refs: input artifact refs, runtime refs,
  and safe compact inputs. Raw sequence / FASTA / PDB / CIF / A3M bodies, API
  keys, raw ToolUniverse payloads, full prompts and raw LLM responses must be
  resolved inside the worker process from DB/storage at tool-execution time,
  never embedded here.
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


# The single ``extra="forbid"`` config reused by every contract model. Unknown
# fields (including raw-looking ones) are rejected at validation time.
_FORBID = ConfigDict(extra="forbid", populate_by_name=True)


# Raw-looking dict keys that must never appear inside a compact metadata field.
# ``extra="forbid"`` only guards *model* fields; the compact dicts below are
# ``dict[str, Any]`` so raw material could otherwise be smuggled in as a nested
# key. These keys are checked case-insensitively, recursively through nested
# dicts and lists. We reject on the KEY only — we do NOT scan arbitrary string
# values, so ordinary safe metadata (length / alphabet / hash prefix) passes.
_FORBIDDEN_COMPACT_KEYS: frozenset[str] = frozenset(
    {
        "raw_sequence",
        "sequence",
        "fasta",
        "fasta_body",
        "pdb_body",
        "pdb_text",
        "cif_body",
        "cif_text",
        "a3m",
        "a3m_body",
        "api_key",
        "authorization",
        "raw_tooluniverse_payload",
        "tooluniverse_payload",
        "full_prompt",
        "raw_prompt",
        "raw_llm_response",
        "llm_response",
    }
)


def _reject_forbidden_compact_keys(value: Any, *, field_name: str) -> None:
    """Recursively reject raw-looking dict keys inside a compact metadata value.

    Walks nested dicts and lists. Raises ``ValueError`` (surfaced by pydantic as
    a validation error) when any dict key matches ``_FORBIDDEN_COMPACT_KEYS``
    (case-insensitive). Only keys are inspected; string values are left alone so
    legitimate compact summaries are not falsely rejected.
    """
    if isinstance(value, dict):
        for key, nested in value.items():
            if isinstance(key, str) and key.strip().lower() in _FORBIDDEN_COMPACT_KEYS:
                raise ValueError(
                    f"{field_name} must not contain raw-looking key "
                    f"{key!r}; raw sequence / FASTA / PDB / CIF / A3M / API key / "
                    "raw ToolUniverse payload / full prompt / raw LLM response are "
                    "forbidden in the A2A payload (use runtime refs instead)."
                )
            _reject_forbidden_compact_keys(nested, field_name=field_name)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _reject_forbidden_compact_keys(item, field_name=field_name)


def _compact_dict_privacy_validator(field_name: str):
    """Build a reusable pydantic ``field_validator`` for a compact dict field."""

    def _validate(cls, value: Any) -> Any:  # noqa: N805
        _reject_forbidden_compact_keys(value, field_name=field_name)
        return value

    return _validate


# ─────────────────────────────────────────────────────────────────────────────
# Compact refs
# ─────────────────────────────────────────────────────────────────────────────
class InputArtifactRef(BaseModel):
    """Reference to an upstream normalized artifact the worker may read.

    Carries only *where* to read (run-scoped ``artifact_id`` / ``artifact_type``)
    and *which* normalized fields to extract (``field_keys``) — never the
    artifact body. Worker runtime owns the actual DB/registry lookup.
    """

    model_config = _FORBID

    artifact_id: str
    run_id: str
    artifact_type: str
    artifact_role: Optional[str] = None
    schema_version: Optional[str] = None
    entity_type: Optional[str] = None
    selection_mode: Optional[str] = None
    field_keys: list[str] = Field(default_factory=list)
    safe_summary_ref: Optional[str] = None
    producer_task_id: Optional[str] = None
    can_read_from_db: bool = True


class RuntimeRef(BaseModel):
    """Reference to a raw / large / sensitive value resolved only at tool time.

    The LLM and the A2A payload see this ref plus an optional ``safe_summary``
    (length / alphabet / hash prefix). The raw body itself is materialised by
    the worker's deterministic runtime resolver immediately before a scoped MCP
    tool call — it never appears in this object.
    """

    model_config = _FORBID

    ref: str = Field(alias="$ref")
    ref_type: Optional[str] = None
    expected_runtime_type: Optional[str] = None
    runtime_type: Optional[str] = None
    artifact_id: Optional[str] = None
    field_path: Optional[str] = None
    material_id: Optional[str] = None
    source: Optional[str] = None
    supports_tool_args: list[str] = Field(default_factory=list)
    # Safe, non-sensitive fingerprint only (e.g. {"length": 438,
    # "alphabet": "protein"}). Never raw body.
    safe_summary: dict[str, Any] = Field(default_factory=dict)
    can_resolve_at_runtime: bool = True

    _validate_safe_summary = field_validator("safe_summary")(
        _compact_dict_privacy_validator("safe_summary")
    )


class WorkerArtifactRef(BaseModel):
    """Compact output artifact ref returned by a worker.

    Points at the worker-owned persisted artifact (source of truth stays in
    DB/registry/storage). Carries no artifact body.
    """

    model_config = _FORBID

    artifact_id: str
    artifact_type: str
    storage_key: Optional[str] = None
    run_id: Optional[str] = None
    schema_version: Optional[str] = None
    safe_summary_ref: Optional[str] = None


# ─────────────────────────────────────────────────────────────────────────────
# Request-side composites
# ─────────────────────────────────────────────────────────────────────────────
class WorkerRequestSpec(BaseModel):
    """Compact task intent the Orchestrator hands the worker."""

    model_config = _FORBID

    objective: str
    reason: Optional[str] = None
    priority: Literal["low", "normal", "high"] = "normal"


class OrchestratorRoutingDecisionRef(BaseModel):
    """The routing-decision control context a dispatched worker requires.

    Presence of this object is the ``orchestrator_routing_decision`` control
    context the worker AgentCards declare under ``required_control_context``.
    """

    model_config = _FORBID

    planned_status: str
    dispatch_mode: Literal["python_a2a"]
    deterministic_gate_status: Optional[str] = None
    routing_phase: Optional[str] = None
    expected_outputs: list[str] = Field(default_factory=list)
    reason: Optional[str] = None


class InputProjection(BaseModel):
    """The safe worker input surface.

    ``compact_inputs`` carries safe short literals / normalized summaries;
    ``input_artifact_refs`` lists upstream artifacts to read from DB;
    ``runtime_refs`` carries refs (never bodies) for raw / large / sensitive
    material resolved at tool-execution time.
    """

    model_config = _FORBID

    projection_version: str = "v1"
    compact_inputs: dict[str, Any] = Field(default_factory=dict)
    input_artifact_refs: dict[str, InputArtifactRef] = Field(default_factory=dict)
    runtime_refs: dict[str, RuntimeRef] = Field(default_factory=dict)

    _validate_compact_inputs = field_validator("compact_inputs")(
        _compact_dict_privacy_validator("compact_inputs")
    )


class PrivacyConstraints(BaseModel):
    """Explicit privacy switches carried with every request.

    Defaults enforce the project-wide rule that no raw biological material,
    API key, raw ToolUniverse payload, full prompt, or raw LLM response is ever
    placed in the A2A payload.
    """

    model_config = _FORBID

    no_raw_sequence: bool = True
    no_raw_fasta: bool = True
    no_raw_pdb_cif: bool = True
    no_raw_a3m: bool = True
    no_api_keys: bool = True
    no_raw_tooluniverse_payload: bool = True
    no_full_prompt: bool = True
    no_raw_llm_response: bool = True


class RetryContext(BaseModel):
    model_config = _FORBID

    retry_of_task_id: Optional[str] = None
    retry_attempt: int = 0
    max_retry_attempts: int = 0
    retry_reason: Optional[str] = None


class A2ATaskMetadata(BaseModel):
    """Compact routing identifiers placed in ``python_a2a.Task.metadata``.

    Contains only identity/routing fields — never an artifact body.
    """

    model_config = _FORBID

    adc_payload_type: Literal["worker_execution_request"]
    adc_payload_version: Literal["v1"]
    run_id: str
    task_id: str
    routing_plan_id: str
    routing_decision_id: str
    agent_id: str
    capability_id: str
    created_by: str


class WorkerExecutionRequest(BaseModel):
    """ADC business payload placed in ``python_a2a.Task.message.content.text``.

    Only compact refs / artifact refs / runtime refs are permitted; ``extra``
    is forbidden so raw material fields are rejected outright.
    """

    model_config = _FORBID

    payload_type: Literal["worker_execution_request"]
    payload_version: Literal["v1"]
    run_id: str
    session_id: Optional[str] = None
    task_id: str
    routing_plan_id: str
    routing_decision_id: str
    agent_id: str
    capability_id: str
    created_by: str
    worker_request: WorkerRequestSpec
    orchestrator_routing_decision: OrchestratorRoutingDecisionRef
    input_projection: InputProjection
    privacy_constraints: PrivacyConstraints
    retry_context: Optional[RetryContext] = None


# ─────────────────────────────────────────────────────────────────────────────
# Result-side composites
# ─────────────────────────────────────────────────────────────────────────────
class ToolCallSummary(BaseModel):
    """Compact roll-up of worker tool activity. No raw tool payloads."""

    model_config = _FORBID

    attempted: int = 0
    success: int = 0
    failed: int = 0
    dependency_unavailable: int = 0
    skipped: int = 0


ExecutionStatus = Literal["not_started", "running", "completed", "failed"]
ResultStatus = Literal[
    "success",
    "partial",
    "validation_failed",
    "tool_failed",
    "blocked",
    "needs_user_input",
]


class WorkerExecutionResult(BaseModel):
    """Compact result a worker returns via the A2A task response.

    Carries output artifact refs and compact summaries only. The worker
    persists the real artifact body to DB/storage; no raw artifact body,
    raw tool output, API key, full prompt, or raw LLM response is embedded.
    """

    model_config = _FORBID

    payload_type: Literal["worker_execution_result"]
    payload_version: Literal["v1"]
    run_id: str
    task_id: str
    routing_plan_id: Optional[str] = None
    routing_decision_id: Optional[str] = None
    agent_id: str
    capability_id: str
    execution_status: ExecutionStatus
    result_status: ResultStatus
    error_code: Optional[str] = None
    retry_of_task_id: Optional[str] = None
    output_artifact_refs: dict[str, WorkerArtifactRef] = Field(default_factory=dict)
    compact_summary: dict[str, Any] = Field(default_factory=dict)
    tool_call_summary: ToolCallSummary = Field(default_factory=ToolCallSummary)
    skipped_or_failed_tools: list[str] = Field(default_factory=list)
    error_summary: Optional[str] = None
    validation_errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    _validate_compact_summary = field_validator("compact_summary")(
        _compact_dict_privacy_validator("compact_summary")
    )


# ─────────────────────────────────────────────────────────────────────────────
# Worker availability / dispatch status summary (Orchestrator / transport layer)
# ─────────────────────────────────────────────────────────────────────────────
Availability = Literal[
    "available",
    "unavailable",
    "degraded",
    "unavailable_for_current_loop",
]
AgentFailureReason = Literal[
    "none",
    "discovery_timeout",
    "discovery_connection_failed",
    "card_invalid",
    "health_timeout",
    "health_failed",
    "dispatch_timeout",
    "dispatch_connection_failed",
    "dispatch_transport_error",
    "server_error",
]


class WorkerStatusSummary(BaseModel):
    """Compact availability summary for one worker.

    Belongs to the Orchestrator / A2A transport layer. Does NOT carry endpoint
    URLs (those stay in deployment config / full AgentCard cache) so it is safe
    to surface in a compact catalog.
    """

    model_config = _FORBID

    agent_id: str
    availability: Availability = "available"
    agent_failure_reason: AgentFailureReason = "none"
    discovery_error: Optional[str] = None
    routable: bool = True
    status: Literal["active", "planned", "disabled"] = "active"


__all__ = [
    "InputArtifactRef",
    "RuntimeRef",
    "WorkerArtifactRef",
    "WorkerRequestSpec",
    "OrchestratorRoutingDecisionRef",
    "InputProjection",
    "PrivacyConstraints",
    "RetryContext",
    "A2ATaskMetadata",
    "WorkerExecutionRequest",
    "ToolCallSummary",
    "WorkerExecutionResult",
    "WorkerStatusSummary",
    "ExecutionStatus",
    "ResultStatus",
    "Availability",
    "AgentFailureReason",
]
