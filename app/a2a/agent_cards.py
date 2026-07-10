"""AgentCard builders + adc_agent_contract validator + compact card catalog.

Turn A of the Multi-Agent A2A v1 plan. This module publishes worker
capabilities as ``python_a2a.AgentCard`` objects (never a custom A2A card
format) and attaches the ADC-specific routing contract under
``AgentCard.capabilities["adc_agent_contract"]``.

Design references:

- ``A2A 方案/agent_card_design.md``
- ``A2A 方案/orchestrator_routing_design.md``
- ``项目文件/multi_agent_a2a_v1_implementation_plan.md``

HARD CONSTRAINTS:

- The AgentCard input/output artifact contract mirrors the CURRENT production
  DB/storage read/write paths of Step 5 / Step 6 / structure worker — it does
  not invent idealized inputs. Every ``storage_path`` below is verified against
  the live agent code.
- Turn A adds schema + builders + validator + compact catalog only. It does not
  start an HTTP server, dispatch A2A tasks, change Step 4 routing, or touch MCP
  scope / ToolUniverse inventory / registry / tool names.
- The compact catalog is the LLM-safe view: it excludes endpoint URL, auth,
  API keys, raw artifact bodies, raw biological material, raw ToolUniverse
  payloads, full prompts, and raw LLM responses.
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, ValidationError
from python_a2a import AgentCard, AgentSkill


_FORBID = ConfigDict(extra="forbid")

# Capability id constants (stable routing identifiers used by Turn B/C/D).
# NOTE: The structure worker exposes ONE Orchestrator-facing capability
# (``structure_design_workflow``). Step 7/8/9 are the worker's INTERNAL serial
# steps, not independently dispatchable A2A capabilities — they are only ever
# referenced as ``supported_step_ids`` / ``internal_execution_order`` below.
CAP_STEP5_CANDIDATE_CONTEXT = "step_05_candidate_context"
CAP_STEP6_DEVELOPABILITY = "step_06_developability"
CAP_STRUCTURE_DESIGN_WORKFLOW = "structure_design_workflow"

# Structure worker internal step ids (NOT A2A capabilities).
STEP_07_STRUCTURE_INPUT = "step_07_structure_input"
STEP_08_STRUCTURE_EVALUATION = "step_08_structure_evaluation"
STEP_09_STRUCTURE_DESIGN = "step_09_structure_design"

# Agent id constants (match the design docs' stable snake ids).
AGENT_ID_STEP5 = "step_05_candidate_context_agent"
AGENT_ID_STEP6 = "step_06_developability_agent"
AGENT_ID_STRUCTURE = "structure_and_design_agent"

_REQUIRED_DISPATCH_MODES = ["python_a2a"]
_CONTRACT_KEY = "adc_agent_contract"


class AgentContractError(ValueError):
    """Raised when an AgentCard's adc_agent_contract is missing or malformed."""


# ─────────────────────────────────────────────────────────────────────────────
# adc_agent_contract shape (stored as a plain dict under capabilities)
# ─────────────────────────────────────────────────────────────────────────────
class ContractArtifactRef(BaseModel):
    """A named artifact and its concrete run-scoped storage path."""

    model_config = _FORBID

    artifact_name: str = Field(min_length=1)
    storage_path: str = Field(min_length=1)


class ArtifactFieldRequirement(BaseModel):
    """Orchestrator-verifiable field contract for one input artifact.

    ``required_field_keys`` are normalized schema key names that must exist in
    the artifact (their *presence*, not non-emptiness). This is NOT a JSONPath,
    NOT a raw value, and NOT a copy of the artifact body. Conditional domain
    inputs (PDB / sequence / variant / UniProt id / masked prompt) are decided
    by the worker's internal projection / LLM selection / runtime resolver, and
    must never become unconditional required fields of the whole A2A task.
    """

    model_config = _FORBID

    required_field_keys: list[str] = Field(min_length=1)
    entity_type: Optional[str] = None
    default_selection_mode: Optional[str] = None


ExecutionMode = Literal["single_step", "sequential_workflow"]


class AgentCapabilityContract(BaseModel):
    model_config = _FORBID

    capability_id: str = Field(min_length=1)
    skill_name: str = Field(min_length=1)
    capability_summary: str = Field(min_length=1)
    # ``single_step``: one Orchestrator-facing step (Step 5 / Step 6).
    # ``sequential_workflow``: one capability that runs an ordered set of
    # INTERNAL steps inside the worker (structure Step 7 -> 8 -> 9).
    execution_mode: ExecutionMode = "single_step"
    internal_execution_order: list[str] = Field(default_factory=list)
    supported_step_ids: list[str] = Field(default_factory=list)
    supported_intents: list[str] = Field(default_factory=list)
    supported_lane_flags: list[str] = Field(default_factory=list)
    required_input_artifacts: list[ContractArtifactRef] = Field(default_factory=list)
    optional_input_artifacts: list[ContractArtifactRef] = Field(default_factory=list)
    required_artifact_fields: dict[str, ArtifactFieldRequirement] = Field(default_factory=dict)
    required_control_context: list[str] = Field(default_factory=list)
    output_artifacts: list[ContractArtifactRef] = Field(min_length=1)
    uses_llm: bool
    uses_mcp: bool


class AdcAgentContract(BaseModel):
    model_config = _FORBID

    agent_id: str = Field(min_length=1)
    agent_role: Literal["orchestrator", "worker"]
    step_id: Optional[str] = None
    display_name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    capabilities: list[AgentCapabilityContract] = Field(min_length=1)
    dispatch_modes: list[str] = Field(min_length=1)
    routable: bool
    status: Literal["active", "planned", "disabled"] = "active"
    uses_llm: bool
    uses_mcp: bool
    privacy_notes: list[str] = Field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Contract data — verified against production agent code
# ─────────────────────────────────────────────────────────────────────────────
# Shared storage paths (must match the live agents byte-for-byte).
_RAW_REQUEST_RECORD = ContractArtifactRef(
    artifact_name="raw_request_record",
    storage_path="inputs/raw_request_record.json",
)
_STRUCTURED_QUERY = ContractArtifactRef(
    artifact_name="structured_query",
    storage_path="inputs/structured_query.json",
)
_CANDIDATE_CONTEXT_TABLE = ContractArtifactRef(
    artifact_name="candidate_context_table",
    storage_path="candidate_context_table.json",
)
_PREPARED_STRUCTURE_INPUT = ContractArtifactRef(
    artifact_name="prepared_structure_input_package",
    storage_path="prepared_structure_input_package.json",
)
_STRUCTURE_PREDICTION = ContractArtifactRef(
    artifact_name="structure_prediction_and_interface_results",
    storage_path="structure_prediction_and_interface_results.json",
)


def _build_agent_card(*, contract: AdcAgentContract, url: str) -> AgentCard:
    """Assemble a ``python_a2a.AgentCard`` from an ADC contract.

    One ``python_a2a.AgentSkill`` is published per capability; the ADC routing
    contract is attached under ``capabilities["adc_agent_contract"]``.
    """
    skills = [
        AgentSkill(
            id=cap.capability_id,
            name=cap.skill_name,
            description=cap.capability_summary,
            tags=sorted(set([*cap.supported_step_ids, *cap.supported_lane_flags])),
            examples=[],
        )
        for cap in contract.capabilities
    ]
    return AgentCard(
        name=contract.display_name,
        description=contract.description,
        url=url,
        version="1.0.0",
        capabilities={_CONTRACT_KEY: contract.model_dump()},
        skills=skills,
    )


def build_step5_agent_card(url: str) -> AgentCard:
    """Step 5 CandidateContextAgent card (request-based worker contract).

    Verified against ``app/agents/candidate_context_agent.py``: the request-based
    core (``run_from_artifacts``) reads ``inputs/raw_request_record.json`` and
    ``inputs/structured_query.json`` and writes ``candidate_context_table.json``.

    ``run_step_plan`` is deliberately NOT a required input: it is Step 4 /
    Orchestrator execution control, and the request-based worker's external
    control contract is ``orchestrator_routing_decision``. (Only the legacy
    ``run(run_id)`` path still gates on the registry ``run_step_plan_id``.)
    ``input_readiness_status`` is an OPTIONAL input; the Step 5 core does not
    depend on it.
    """
    contract = AdcAgentContract(
        agent_id=AGENT_ID_STEP5,
        agent_role="worker",
        step_id="step_05_candidate_context",
        display_name="Candidate Context Agent",
        description=(
            "Builds normalized candidate / material / identifier records from "
            "raw request metadata, structured query entities, referenced inputs, "
            "uploaded file metadata, and scoped context-enrichment MCP tools."
        ),
        capabilities=[
            AgentCapabilityContract(
                capability_id=CAP_STEP5_CANDIDATE_CONTEXT,
                skill_name="Candidate context build",
                capability_summary=(
                    "Assemble the run's candidate context table from request "
                    "records, structured query, and scoped enrichment tools."
                ),
                supported_step_ids=["step_05_candidate_context"],
                supported_intents=[
                    "new_adc_design",
                    "existing_adc_evaluation",
                    "developability_assessment",
                    "structure_analysis",
                    "compound_screening",
                ],
                supported_lane_flags=[
                    "target_discovery_lane",
                    "antibody_discovery_lane",
                    "antibody_lane",
                    "compound_lane",
                    "structure_lane",
                ],
                required_input_artifacts=[
                    _RAW_REQUEST_RECORD,
                    _STRUCTURED_QUERY,
                ],
                optional_input_artifacts=[
                    ContractArtifactRef(
                        artifact_name="input_readiness_status",
                        storage_path="inputs/input_readiness_status.json",
                    ),
                ],
                # Schema-key presence the Step 5 core actually reads (verified
                # against candidate_context_agent._build_candidate_context). These
                # denote key presence, not non-emptiness.
                required_artifact_fields={
                    "raw_request_record": ArtifactFieldRequirement(
                        required_field_keys=[
                            "raw_user_query",
                            "user_provided_context",
                            "uploaded_files",
                        ],
                    ),
                    "structured_query": ArtifactFieldRequirement(
                        required_field_keys=[
                            "mentioned_entities",
                            "referenced_inputs",
                            "normalized_entities",
                            "entity_decompositions",
                        ],
                    ),
                },
                required_control_context=["orchestrator_routing_decision"],
                output_artifacts=[_CANDIDATE_CONTEXT_TABLE],
                uses_llm=True,
                uses_mcp=True,
            )
        ],
        dispatch_modes=["python_a2a"],
        routable=True,
        status="active",
        uses_llm=True,
        uses_mcp=True,
        privacy_notes=[
            "Reads raw_request_record and structured_query by reference from run storage.",
            "Persisted candidate_context_table must not embed raw ToolUniverse "
            "payloads, API keys, full prompts, raw LLM responses, or raw file contents.",
        ],
    )
    return _build_agent_card(contract=contract, url=url)


def build_step6_agent_card(url: str) -> AgentCard:
    """Step 6 DevelopabilityAgent request-based worker card.

    The request-based core reads ``candidate_context_table.json`` and writes
    ``structured_liability_summary.json``. ``run_step_plan`` is deliberately
    absent: Step 4 owns the dispatch gate and supplies the validated
    ``orchestrator_routing_decision`` control context.
    """
    contract = AdcAgentContract(
        agent_id=AGENT_ID_STEP6,
        agent_role="worker",
        step_id="step_06_developability",
        display_name="Developability Agent",
        description=(
            "Runs lane-based developability and liability pre-filtering using "
            "candidate context, scoped MCP tools, progressive disclosure, and "
            "deterministic runtime resolution."
        ),
        capabilities=[
            AgentCapabilityContract(
                capability_id=CAP_STEP6_DEVELOPABILITY,
                skill_name="Developability pre-filtering",
                capability_summary=(
                    "Assess ADC candidate developability liabilities across "
                    "lanes using candidate context and scoped MCP tools."
                ),
                supported_step_ids=["step_06_developability"],
                supported_intents=[
                    "new_adc_design",
                    "existing_adc_evaluation",
                    "developability_assessment",
                    "optimization",
                ],
                supported_lane_flags=[
                    "antibody_lane",
                    "compound_lane",
                    "structure_lane",
                    "payload_linker_compound_liability",
                    "antibody_protein_sequence_liability",
                    "antigen_protein_feature_context",
                    "structure_interface_quality",
                    "compound_bioactivity_prior_context",
                ],
                required_input_artifacts=[
                    _CANDIDATE_CONTEXT_TABLE,
                ],
                # The current production core reads the normalized top-level
                # candidate_records list. Conditional lane inputs remain typed
                # fields inside each record and must not become unconditional
                # AgentCard requirements.
                required_artifact_fields={
                    "candidate_context_table": ArtifactFieldRequirement(
                        entity_type="candidate",
                        default_selection_mode="all_in_artifact",
                        required_field_keys=["candidate_records"],
                    ),
                },
                required_control_context=["orchestrator_routing_decision"],
                output_artifacts=[
                    ContractArtifactRef(
                        artifact_name="structured_liability_summary",
                        storage_path="structured_liability_summary.json",
                    )
                ],
                uses_llm=True,
                uses_mcp=True,
            )
        ],
        dispatch_modes=["python_a2a"],
        routable=True,
        status="active",
        uses_llm=True,
        uses_mcp=True,
        privacy_notes=[
            "Reads candidate_context_table by reference from run storage.",
            "Persisted structured_liability_summary must reference compact "
            "tool_call_records and must not embed raw ToolUniverse payloads, raw "
            "protein sequence, PDB/CIF/FASTA content, API keys, full prompts, or "
            "raw LLM responses.",
        ],
    )
    return _build_agent_card(contract=contract, url=url)


def build_structure_agent_card(url: str) -> AgentCard:
    """Structure and Design worker card — ONE ``structure_design_workflow`` capability.

    The Orchestrator sees a single capability and dispatches a single
    ``python_a2a.Task``. Step 7 -> Step 8 -> Step 9 are the worker's INTERNAL
    serial steps (declared via ``internal_execution_order``), not independently
    dispatchable A2A capabilities.

    External workflow entry inputs (verified against
    ``app/agents/structure_and_design_agent.py`` run_step_7 read paths):
    ``raw_request_record``, ``structured_query``, ``candidate_context_table``.
    ``prepared_structure_input_package`` and
    ``structure_prediction_and_interface_results`` are INTERNAL Step 7 / Step 8
    outputs, so they are not required inputs. ``run_step_plan`` is Step 4 /
    Orchestrator execution control; the worker's external control contract is
    ``orchestrator_routing_decision``, not a required input artifact. (The local
    ``run_step_7`` still reads ``run_step_plan`` from the registry — that is a
    worker-adapter implementation-compatibility concern, not part of this
    Orchestrator-facing card.)
    """
    contract = AdcAgentContract(
        agent_id=AGENT_ID_STRUCTURE,
        agent_role="worker",
        step_id="structure_and_design",
        display_name="Structure and Design Agent",
        description=(
            "Prepares structure-relevant inputs, evaluates structure/interface "
            "context, and runs controlled protein design and variant evaluation "
            "when inputs allow."
        ),
        capabilities=[
            AgentCapabilityContract(
                capability_id=CAP_STRUCTURE_DESIGN_WORKFLOW,
                skill_name="Structure design workflow",
                capability_summary=(
                    "Prepare structure inputs, evaluate structure/interface "
                    "context, and run protein design and variant evaluation as "
                    "one sequential worker workflow (Step 7 -> 8 -> 9)."
                ),
                execution_mode="sequential_workflow",
                internal_execution_order=[
                    STEP_07_STRUCTURE_INPUT,
                    STEP_08_STRUCTURE_EVALUATION,
                    STEP_09_STRUCTURE_DESIGN,
                ],
                supported_step_ids=[
                    STEP_07_STRUCTURE_INPUT,
                    STEP_08_STRUCTURE_EVALUATION,
                    STEP_09_STRUCTURE_DESIGN,
                ],
                supported_intents=[
                    "new_adc_design",
                    "existing_adc_evaluation",
                    "structure_analysis",
                    "optimization",
                ],
                supported_lane_flags=[
                    "structure_lane",
                    "protein_design_lane",
                    "variant_evaluation_lane",
                ],
                required_input_artifacts=[
                    _RAW_REQUEST_RECORD,
                    _STRUCTURED_QUERY,
                    _CANDIDATE_CONTEXT_TABLE,
                ],
                optional_input_artifacts=[
                    ContractArtifactRef(
                        artifact_name="structured_liability_summary",
                        storage_path="structured_liability_summary.json",
                    ),
                ],
                # Orchestrator-verifiable schema-key presence, using the CURRENT
                # real Pydantic field names (not the design doc's idealized ones).
                required_artifact_fields={
                    "raw_request_record": ArtifactFieldRequirement(
                        required_field_keys=[
                            "raw_user_query",
                            "user_provided_context",
                            "uploaded_files",
                        ],
                    ),
                    "structured_query": ArtifactFieldRequirement(
                        required_field_keys=[
                            "task_intent",
                            "referenced_inputs",
                            "requested_outputs",
                            "user_constraints",
                            "normalized_entities",
                            "canonical_query",
                        ],
                    ),
                    "candidate_context_table": ArtifactFieldRequirement(
                        entity_type="candidate",
                        default_selection_mode="all_in_artifact",
                        required_field_keys=[
                            "candidate_records",
                            "downstream_query_hints",
                        ],
                    ),
                },
                required_control_context=["orchestrator_routing_decision"],
                # Ordered Step 7 / Step 8 / Step 9 workflow outputs.
                output_artifacts=[
                    _PREPARED_STRUCTURE_INPUT,
                    _STRUCTURE_PREDICTION,
                    ContractArtifactRef(
                        artifact_name="structure_variant_and_compound_screening",
                        storage_path="compound_screening_artifact.json",
                    ),
                ],
                uses_llm=True,
                uses_mcp=True,
            ),
        ],
        dispatch_modes=["python_a2a"],
        routable=True,
        status="active",
        uses_llm=True,
        uses_mcp=True,
        privacy_notes=[
            "Reads raw_request_record, structured_query, and candidate_context_table "
            "by reference from run storage.",
            "Persisted artifacts must not embed raw sequence, FASTA, PDB/CIF, A3M, "
            "API keys, raw ToolUniverse payloads, full prompts, or raw LLM responses.",
        ],
    )
    return _build_agent_card(contract=contract, url=url)


# ─────────────────────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────────────────────
def _validate_execution_mode(cap: AgentCapabilityContract) -> None:
    """Deterministic execution-mode / internal-order checks (never silent)."""
    order = cap.internal_execution_order
    if len(order) != len(set(order)):
        raise AgentContractError(
            f"capability '{cap.capability_id}' has duplicate steps in "
            f"internal_execution_order: {order}"
        )
    unknown = [s for s in order if s not in cap.supported_step_ids]
    if unknown:
        raise AgentContractError(
            f"capability '{cap.capability_id}' internal_execution_order references "
            f"steps not in supported_step_ids: {unknown}"
        )
    if cap.execution_mode == "sequential_workflow":
        if not order:
            raise AgentContractError(
                f"capability '{cap.capability_id}' is a sequential_workflow but has "
                "an empty internal_execution_order"
            )
    else:  # single_step
        if len(order) > 1:
            raise AgentContractError(
                f"single_step capability '{cap.capability_id}' must not declare "
                f"multiple internal execution steps: {order}"
            )


def _validate_required_artifact_fields(cap: AgentCapabilityContract) -> None:
    """Field-contract checks: keys must reference declared required inputs; the
    required_field_keys must be deduped and free of empty strings."""
    required_names = {ref.artifact_name for ref in cap.required_input_artifacts}
    for artifact_name, requirement in cap.required_artifact_fields.items():
        if artifact_name not in required_names:
            raise AgentContractError(
                f"capability '{cap.capability_id}' declares required_artifact_fields "
                f"for '{artifact_name}', which is not a required_input_artifact"
            )
        keys = requirement.required_field_keys
        if any((not isinstance(k, str) or not k.strip()) for k in keys):
            raise AgentContractError(
                f"capability '{cap.capability_id}' required_artifact_fields "
                f"['{artifact_name}'] contains an empty field key"
            )
        if len(keys) != len(set(keys)):
            raise AgentContractError(
                f"capability '{cap.capability_id}' required_artifact_fields "
                f"['{artifact_name}'] has duplicate field keys: {keys}"
            )


def parse_adc_agent_contract(card: AgentCard) -> AdcAgentContract:
    """Extract and validate the adc_agent_contract embedded in ``card``.

    Raises :class:`AgentContractError` (never a silent pass) when the contract
    is missing, malformed, or inconsistent with the card's skills / url.
    """
    caps = getattr(card, "capabilities", None)
    if not isinstance(caps, dict) or _CONTRACT_KEY not in caps:
        raise AgentContractError(
            f"AgentCard is missing capabilities['{_CONTRACT_KEY}']"
        )
    raw = caps[_CONTRACT_KEY]
    if not isinstance(raw, dict):
        raise AgentContractError(
            f"capabilities['{_CONTRACT_KEY}'] must be an object, got {type(raw).__name__}"
        )

    try:
        contract = AdcAgentContract.model_validate(raw)
    except ValidationError as exc:  # explicit, not silent
        raise AgentContractError(
            f"adc_agent_contract failed schema validation: {exc}"
        ) from exc

    # agent_id / agent_role presence is already enforced by the schema (min_length
    # / Literal). Below are the cross-field checks the schema cannot express.

    # This version has exactly one production dispatch mode. Do not accept an
    # alias, secondary local mode, or duplicate entry that could later be used
    # as a silent direct-call fallback.
    if contract.dispatch_modes != _REQUIRED_DISPATCH_MODES:
        raise AgentContractError(
            f"dispatch_modes must be exactly {_REQUIRED_DISPATCH_MODES}; "
            f"got {contract.dispatch_modes}"
        )

    # Every required/output artifact ref must carry artifact_name + storage_path.
    # (min_length=1 on both fields already guarantees this; assert defensively so
    # a future edit that loosens the field constraint still fails loudly.)
    for cap in contract.capabilities:
        for ref in (*cap.required_input_artifacts, *cap.optional_input_artifacts, *cap.output_artifacts):
            if not ref.artifact_name or not ref.storage_path:
                raise AgentContractError(
                    f"capability '{cap.capability_id}' has an artifact ref missing "
                    "artifact_name or storage_path"
                )
        _validate_execution_mode(cap)
        _validate_required_artifact_fields(cap)

    # Skills published on the card must line up 1:1 with contract capabilities.
    skill_id_list = [getattr(s, "id", None) for s in (getattr(card, "skills", None) or [])]
    capability_id_list = [cap.capability_id for cap in contract.capabilities]
    if len(skill_id_list) != len(set(skill_id_list)):
        raise AgentContractError(f"AgentCard.skills contains duplicate ids: {skill_id_list}")
    if len(capability_id_list) != len(set(capability_id_list)):
        raise AgentContractError(
            "adc_agent_contract contains duplicate capability ids: "
            f"{capability_id_list}"
        )
    skill_ids = set(skill_id_list)
    capability_ids = set(capability_id_list)
    if skill_ids != capability_ids:
        raise AgentContractError(
            "AgentCard.skills do not align with adc_agent_contract capabilities: "
            f"skills={sorted(str(s) for s in skill_ids)} "
            f"capabilities={sorted(capability_ids)}"
        )

    # A routable worker must expose a usable dispatch URL.
    if contract.routable and not str(getattr(card, "url", "") or "").strip():
        raise AgentContractError(
            f"routable agent '{contract.agent_id}' must have a non-empty AgentCard url"
        )

    return contract


def validate_adc_agent_contract(card: AgentCard) -> AdcAgentContract:
    """Alias for :func:`parse_adc_agent_contract` (validate-and-return)."""
    return parse_adc_agent_contract(card)


# ─────────────────────────────────────────────────────────────────────────────
# Compact card catalog (LLM-safe view)
# ─────────────────────────────────────────────────────────────────────────────
def build_compact_card_for_agent(card: AgentCard) -> dict[str, Any]:
    """Return the LLM-safe compact summary for a single AgentCard.

    Includes only routing-relevant fields. Excludes endpoint URL, auth, API
    keys, raw artifact bodies, raw biological material, raw ToolUniverse
    payloads, full prompts, and raw LLM responses.
    """
    contract = parse_adc_agent_contract(card)
    return {
        "agent_id": contract.agent_id,
        "agent_role": contract.agent_role,
        "display_name": contract.display_name,
        "name": contract.display_name,
        "description": contract.description,
        "status": contract.status,
        "routable": contract.routable,
        "uses_llm": contract.uses_llm,
        "uses_mcp": contract.uses_mcp,
        "dispatch_modes": list(contract.dispatch_modes),
        "capabilities": [
            {
                "capability_id": cap.capability_id,
                "capability_summary": cap.capability_summary,
                "execution_mode": cap.execution_mode,
                "internal_execution_order": list(cap.internal_execution_order),
                "supported_step_ids": list(cap.supported_step_ids),
                "supported_intents": list(cap.supported_intents),
                "supported_lane_flags": list(cap.supported_lane_flags),
                "required_input_artifact_names": [
                    ref.artifact_name for ref in cap.required_input_artifacts
                ],
                "optional_input_artifact_names": [
                    ref.artifact_name for ref in cap.optional_input_artifacts
                ],
                "output_artifact_names": [
                    ref.artifact_name for ref in cap.output_artifacts
                ],
                # Field NAMES only (schema keys), never values/bodies.
                "required_artifact_field_names": {
                    artifact_name: list(requirement.required_field_keys)
                    for artifact_name, requirement in cap.required_artifact_fields.items()
                },
                "uses_llm": cap.uses_llm,
                "uses_mcp": cap.uses_mcp,
            }
            for cap in contract.capabilities
        ],
    }


def build_compact_card_catalog(cards: list[AgentCard]) -> list[dict[str, Any]]:
    """Build the LLM-safe compact catalog from a list of AgentCards."""
    return [build_compact_card_for_agent(card) for card in cards]


__all__ = [
    "AgentContractError",
    "ContractArtifactRef",
    "ArtifactFieldRequirement",
    "AgentCapabilityContract",
    "AdcAgentContract",
    "ExecutionMode",
    "build_step5_agent_card",
    "build_step6_agent_card",
    "build_structure_agent_card",
    "parse_adc_agent_contract",
    "validate_adc_agent_contract",
    "build_compact_card_for_agent",
    "build_compact_card_catalog",
    "CAP_STEP5_CANDIDATE_CONTEXT",
    "CAP_STEP6_DEVELOPABILITY",
    "CAP_STRUCTURE_DESIGN_WORKFLOW",
    "STEP_07_STRUCTURE_INPUT",
    "STEP_08_STRUCTURE_EVALUATION",
    "STEP_09_STRUCTURE_DESIGN",
    "AGENT_ID_STEP5",
    "AGENT_ID_STEP6",
    "AGENT_ID_STRUCTURE",
]
