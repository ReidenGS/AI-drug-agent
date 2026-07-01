"""Step 5 LLM-assisted tool selection (Stage 1 only).

Pipeline:

1. **Scoped tools** — the agent already calls
   ``mcp_client.list_tools(agent_name="candidate_context_agent",
   step_id="step_05")`` to get the closed set of MCP wrappers Step 5 is
   allowed to invoke. We never widen that scope here.
2. **Eligible plans** — ``step_05_enrichment_registry.plan_enrichment_for_record``
   already turns typed materials / identifiers into a metadata-driven
   list of ``EnrichmentPlan`` entries. Each plan declares the exact tool,
   schema arg name, query value, capability type, fallback group, etc.
   This step is deterministic and is unchanged.
3. **LLM relevance selection** (this module) — for each candidate that
   has eligible plans, the policy may call ``llm.generate_json(...)``
   with a compact catalog + compact candidate context and ask it to pick
   which of those tools to actually run. Anything the LLM "selects"
   that is NOT in the eligible catalog is dropped.

   Provider-path summary (NOT "every call hits an LLM"):

   - API / graph configured-provider path → ``get_llm_provider()`` →
     real Stage-1 LLM call for Step 5 selection.
   - Default offline agent path → ``MockLLMProvider`` deterministic
     ``task="tool_selection_stage_1"`` dispatch (no network).
   - Explicit ``llm=None`` policy path → NO LLM call at all; audit
     records ``llm_call_status="not_called"`` and the agent falls back
     to executing every eligible plan with
     ``tool_selection_source="deterministic_fallback"``.

4. **Deterministic argument construction** — for every plan the LLM
   keeps, the agent uses the registry's pre-built ``query`` /
   ``schema_arg_name``. The LLM never invents arguments here.
5. **Execution** — only selected (or fail-open fallback) plans run via
   MCP. Skipped plans are recorded in the audit only.

Fail-open fallback (explicit product policy):

If the LLM call raises, returns no parsable selections, OR returns only
out-of-scope names, the policy still falls back to executing every
eligible plan so Step 5 stays useful when the LLM is degraded. The
fallback path is NEVER labelled as an LLM selection — the audit records
``tool_selection_source="deterministic_fallback"`` AND an explicit
``fallback_reason`` (``llm_unavailable`` / ``llm_empty_selection`` /
``llm_out_of_scope_only``).

When the LLM returns SOME in-scope selections plus some out-of-scope
names, the policy executes only the in-scope ones — no all-eligible
fallback. Out-of-scope names appear under ``llm_dropped_out_of_scope``
and ``llm_call_status="ok_with_dropped_out_of_scope"``.

``known_live_unavailable`` plans (e.g. ZINC family) are never offered to
the LLM and never counted as a real execution. The agent still emits one
synthetic ``dependency_unavailable`` ``ToolCallRecord`` per such plan so
the gap is visible, but those records are tagged
``tool_selection_source="system_dependency_gap_record"`` and bucketed
into the audit's separate ``known_unavailable_records`` list rather than
``selected_tools``.

Hard guarantees:

- No new MCP tool, no widened scope, no widened inventory.
- No external biomedical API client.
- No raw payloads / full prompts / API keys are logged here.
- The LLM never constructs arguments and never invents identifiers.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Iterable, Optional

from ..llm.provider import LLMProvider
from .step_05_enrichment_registry import EnrichmentPlan
from .tool_selection_policy import (
    SELECTION_STAGE1_SYSTEM_PROMPT,
    SELECTION_STAGE1_USER_PROMPT,
)

logger = logging.getLogger(__name__)


SELECTION_POLICY_VERSION = "step5_llm_assisted_v1"
SELECTION_SOURCE = "step5_selection_policy"


# Reason codes — stable strings; downstream audits may grep these.
REASON_LLM_SELECTED = "llm_selected"
REASON_LLM_SKIPPED = "llm_did_not_select"
REASON_LLM_OUT_OF_SCOPE = "llm_named_tool_outside_eligible_catalog"
REASON_DETERMINISTIC_FALLBACK = "llm_unavailable_deterministic_fallback"
REASON_FALLBACK_EMPTY = "llm_empty_selection_deterministic_fallback"
REASON_FALLBACK_OUT_OF_SCOPE = "llm_out_of_scope_only_deterministic_fallback"
REASON_FALLBACK_NOT_CALLED = "llm_not_called_deterministic_fallback"
REASON_KNOWN_LIVE_UNAVAILABLE = "known_live_unavailable_recorded_as_dependency"

# llm_call_status values.
LLM_STATUS_NOT_CALLED = "not_called"
LLM_STATUS_OK = "ok"
LLM_STATUS_OK_WITH_DROPPED = "ok_with_dropped_out_of_scope"
LLM_STATUS_EMPTY = "empty"
LLM_STATUS_OUT_OF_SCOPE_ONLY = "out_of_scope_only"
LLM_STATUS_FAILED = "failed"

# fallback_reason values (only set when tool_selection_source ==
# "deterministic_fallback").
FALLBACK_REASON_NONE = ""
FALLBACK_REASON_LLM_UNAVAILABLE = "llm_unavailable"
FALLBACK_REASON_LLM_EMPTY = "llm_empty_selection"
FALLBACK_REASON_LLM_OUT_OF_SCOPE_ONLY = "llm_out_of_scope_only"
FALLBACK_REASON_LLM_NOT_CALLED = "llm_not_called"

# Source / execution_semantics for synthetic dependency records.
SOURCE_LLM_STAGE1 = "llm_stage1"
SOURCE_DETERMINISTIC_FALLBACK = "deterministic_fallback"
SOURCE_SYSTEM_DEPENDENCY_GAP = "system_dependency_gap_record"
EXEC_SEMANTICS_REAL = "real_mcp_execution"
EXEC_SEMANTICS_SYNTHETIC_DEP = "synthetic_dependency_unavailable_record"


# Step-5–specific system prompt addendum. Appended AFTER the shared
# Stage 1 prompt so the LLM still gets the catalog-only / no-args /
# no-hallucination rules, plus Step 5 scope.
STEP_05_SELECTION_SYSTEM_ADDENDUM = """You are selecting Step 5 context-enrichment tools for candidate material organization.

Role:
- Choose which already-eligible tools in `compact_catalog` should run for
  the current candidate.
- Every listed tool is already in Step 5 scope and already has enough
  typed input for deterministic argument construction.
- Your decision is relevance, not feasibility.

Rules:
- Use only exact `tool_name` values from `compact_catalog`.
- Do not construct arguments, invent identifiers, invent tool names, or
  ask for tools outside the catalog.
- Do not generate ADC candidates, rankings, pose ensembles, DAR or
  conjugation designs, hypotheses, liability verdicts, literature
  searches, patent searches, candidate-screening picks, or downstream
  conclusions.
- Prefer exact identity/context over approximate lookup when both answer
  the same question.
- Name lookup and SMILES lookup are complementary paths. If explicit
  `payload_smiles`, `linker_smiles`, or `compound_smiles` is present and
  a `query_kind=smiles` / SMILES-capable tool is in `compact_catalog`,
  select at least one SMILES lookup path even if a name lookup is also
  selected.
- Do not invent SMILES and do not use a name string as SMILES.
- Select a tool only if its expected output fills a material/context gap
  or improves downstream typed inputs for Step 6 or Step 7.
- Skip tools whose expected output is redundant with stronger existing
  candidate data, low-information queries, or another selected tool.
- Return an empty `selections` list when no eligible tool is relevant.

Candidate context may be compound, antibody, target/antigen, structure,
whole ADC reference, or a mixed case. For antibody heavy/light sequences,
runtime CDR3 extraction happens before any IEDB BCR lookup; do not assume
full VH/VL sequences are sent to IEDB. If CDR3 extraction is unavailable,
the runtime records a data gap. Step 5 does not generate new ADC design
candidates.

Compact examples:
- input: `linker_payload_name=vc-MMAE`, decomposed
  `linker_name=valine-citrulline`, `payload_name=monomethyl auristatin E`
  -> select eligible name lookup tools such as `ChEMBL_search_molecules`.
- input: `payload_smiles=CCO` or `linker_smiles=NCC(=O)O`
  -> select eligible SMILES lookup tools such as
  `ChEMBL_search_substructure`, even if name lookup is also selected.
- input: `target_antigen_name=HER2` or `uniprot_id=P04626`
  -> select eligible target/antigen context tools such as SAbDab or
  TheraSAbDab tools.
- input: `antibody_heavy_chain_sequence` +
  `antibody_light_chain_sequence` -> select eligible sequence/CDR3/BCR
  tools only when those plans are already in `compact_catalog`; do not
  invent CDR3.
- input: uploaded `structure_file` or `pdb_id`
  -> select eligible structure-context tools when present; do not
  generate poses or structure-prep jobs.

Catalog fields:
- `short_description`, `description_source`: official ToolUniverse
  description when available, otherwise fallback text.
- `expected_output_fields`, `expected_output_semantics`,
  `expected_output_source`: compact ToolUniverse-derived output
  information used to judge complementarity or redundancy.
- `project_side_hints`: local hints such as redundancy group,
  identity strength, and downstream uses; these are not ToolUniverse
  output fields.

Output:
- Return JSON matching the shared Stage-1 shape:
  `{"selections":[{"tool_name":"...","selection_reason":"...",
  "priority":1,"required_context":["..."]}],"selection_metadata":{...}}`.
- For each selected tool, `selection_reason` must briefly name the input
  signal, expected context output, and downstream use.
""".strip()


# Full Step 5 Stage-1 system prompt: the shared selection contract plus the
# Step 5 addendum, combined once so the stable text is byte-identical across
# every candidate and inspectable by cache-layout tests. This is the exact
# string passed as ``system=`` to ``llm.generate_json`` below.
STEP_05_SELECTION_SYSTEM_PROMPT = (
    SELECTION_STAGE1_SYSTEM_PROMPT
    + "\n\n"
    + STEP_05_SELECTION_SYSTEM_ADDENDUM
)


# Fixed English note that scopes the Step 5 selection to context enrichment.
# Stable across candidates/runs; kept as a module constant so the payload
# builder and cache-layout tests share one source of truth.
STEP_05_SELECTION_CONTEXT_NOTE = (
    "Step 5 material/context enrichment only. No ADC candidate "
    "generation, no pose, no ranking, no DAR/conjugation, no "
    "hypotheses. Choose only tools that fill downstream "
    "Step 6/7-needed fields."
)


@dataclass(frozen=True)
class Step5ToolDecision:
    """One per eligible / synthetic-dependency plan.

    ``tool_selection_source`` is one of:

    - ``llm_stage1`` — the LLM picked this in-scope tool.
    - ``deterministic_fallback`` — fail-open fallback because the LLM
      was unavailable / empty / out-of-scope-only / not-called. The
      decision is selected but explicitly NOT credited to the LLM.
    - ``system_dependency_gap_record`` — synthetic
      ``dependency_unavailable`` ``ToolCallRecord`` produced for
      registry-known-unavailable plans (e.g. ZINC). Not a real
      execution; bucketed separately in audit.

    ``execution_semantics`` lets downstream readers tell a real MCP
    execution apart from a synthetic dependency-gap record.
    """

    plan: EnrichmentPlan
    selected: bool
    tool_selection_source: str
    selection_reason: str
    skip_reason: str = ""
    argument_construction_source: str = "deterministic_mapping"
    execution_semantics: str = EXEC_SEMANTICS_REAL


@dataclass
class Step5SelectionAudit:
    """Compact per-candidate audit of the selection step.

    Persisted under ``CandidateContextTable.enrichment_selection_audit``
    keyed by ``candidate_id``. Holds only short, structured strings —
    never raw LLM responses, never raw tool payloads.

    Audit shape:

    - ``eligible_tools`` — every registry-eligible plan.
    - ``selected_tools`` — plans that will actually be MCP-executed
      because the LLM picked them OR because fail-open fallback
      selected the whole eligible set.
    - ``skipped_eligible_tools`` — eligible plans the LLM explicitly
      did not pick (only populated when the LLM produced ≥1 valid
      in-scope selection; fail-open fallback never populates this).
    - ``known_unavailable_records`` — synthetic dependency-gap entries
      (e.g. ZINC). NOT real MCP executions. Always separate from
      ``selected_tools`` so reviewers cannot confuse them with LLM picks.
    - ``tool_selection_source`` — overall provenance for the selected
      set: ``llm_stage1`` when the LLM produced at least one in-scope
      pick; ``deterministic_fallback`` when fail-open fallback fired.
    - ``llm_call_status`` — one of ``not_called`` / ``ok`` /
      ``ok_with_dropped_out_of_scope`` / ``empty`` /
      ``out_of_scope_only`` / ``failed``.
    - ``fallback_reason`` — populated only when
      ``tool_selection_source == "deterministic_fallback"``; one of
      ``llm_unavailable`` / ``llm_empty_selection`` /
      ``llm_out_of_scope_only`` / ``llm_not_called``.
    """

    candidate_id: str
    candidate_category: str
    policy_version: str = SELECTION_POLICY_VERSION
    eligible_tools: list[dict] = field(default_factory=list)
    selected_tools: list[dict] = field(default_factory=list)
    skipped_eligible_tools: list[dict] = field(default_factory=list)
    known_unavailable_records: list[dict] = field(default_factory=list)
    tool_selection_source: str = SOURCE_DETERMINISTIC_FALLBACK
    llm_dropped_out_of_scope: list[str] = field(default_factory=list)
    llm_call_status: str = LLM_STATUS_NOT_CALLED
    fallback_reason: str = FALLBACK_REASON_NONE

    def to_compact(self) -> dict:
        return {
            "policy_version": self.policy_version,
            "candidate_id": self.candidate_id,
            "candidate_category": self.candidate_category,
            "tool_selection_source": self.tool_selection_source,
            "llm_call_status": self.llm_call_status,
            "fallback_reason": self.fallback_reason,
            "eligible_tools": list(self.eligible_tools),
            "selected_tools": list(self.selected_tools),
            "skipped_eligible_tools": list(self.skipped_eligible_tools),
            "known_unavailable_records": list(self.known_unavailable_records),
            "llm_dropped_out_of_scope": list(self.llm_dropped_out_of_scope),
        }


def _plan_compact_entry(plan: EnrichmentPlan) -> dict:
    """Minimal, audit-safe summary of one eligible plan."""
    return {
        "tool_name": plan.tool_name,
        "capability_type": plan.capability_type,
        "query_kind": plan.query_kind,
        "query_role": plan.query_role,
        "material_type": plan.material_type,
        "fallback_group": (plan.extra_summary or {}).get("fallback_group")
        or plan.tool_name,
    }


def _coarse_input_requirements(plan: EnrichmentPlan) -> list[str]:
    """Surface the registry's slot kinds in a form the shared Stage 1
    mock matcher understands (``coarse_input_requirements`` ∩ signals).
    """
    reqs: list[str] = []
    if plan.material_type:
        reqs.append(plan.material_type)
    if plan.query_kind and plan.query_kind not in reqs:
        reqs.append(plan.query_kind)
    return reqs


def _build_signals(plans: Iterable[EnrichmentPlan]) -> dict[str, bool]:
    """All eligible plans' coarse inputs are by definition available.

    The signal dict ensures the deterministic mock-LLM (used by tests
    and offline runs without a configured provider) selects all
    eligible plans — preserving the agent's pre-LLM behavior when no
    real LLM is wired.
    """
    signals: dict[str, bool] = {}
    for plan in plans:
        for req in _coarse_input_requirements(plan):
            signals[req] = True
    return signals


DESCRIPTION_SOURCE_TU = "tooluniverse_description"
DESCRIPTION_SOURCE_FALLBACK = "fallback"

EXPECTED_OUTPUT_SOURCE_TU_SPEC = "tooluniverse_spec"
EXPECTED_OUTPUT_SOURCE_TU_DESCRIPTION = "tooluniverse_description"
EXPECTED_OUTPUT_SOURCE_TU_NO_CONTRACT = "tooluniverse_metadata_no_output_contract"
EXPECTED_OUTPUT_SOURCE_UNAVAILABLE = "unavailable"


# Keys we accept as a structured-output / return schema inside a TU spec.
# We do NOT extend or invent these — TU has historically used a couple of
# these names. Whatever a TU spec actually carries, we read; whatever it
# does not carry, we leave empty.
_TU_OUTPUT_SCHEMA_KEYS: tuple[str, ...] = (
    "return_schema",
    "output_schema",
    "response_schema",
    "returns",
    "output",
    "return",
)


# Free-text markers we treat as "the official description talks about
# returns" — used ONLY to decide whether to forward the description as
# `expected_output_semantics`. We never extract field names from text;
# field names come exclusively from a structured spec block.
_TU_DESCRIPTION_OUTPUT_MARKERS: tuple[str, ...] = (
    "returns",
    "return:",
    "returned",
    "output:",
    "output is",
    "outputs",
    "response",
    "fields:",
    "result fields",
    "fields returned",
)


def _tu_specs_for_tools(tool_names: list[str]) -> dict[str, dict | None]:
    """Bulk-fetch official TU specs for the eligible tools.

    Wraps :func:`app.mcp.tooluniverse_adapter.get_tool_specifications`
    so the catalog builder pays one metadata lookup per candidate, not
    one per (tool, query) pair. Failures or "TU not installed" degrade
    safely to ``{name: None}`` for every requested name — the catalog
    then falls back to the local registry description and
    ``expected_output_source="unavailable"``.
    """
    if not tool_names:
        return {}
    try:
        from ..mcp import tooluniverse_adapter  # noqa: PLC0415
    except Exception:  # noqa: BLE001
        return {name: None for name in tool_names}
    try:
        bulk = tooluniverse_adapter.get_tool_specifications(tool_names) or {}
    except Exception:  # noqa: BLE001
        bulk = {}
    return {name: (bulk.get(name) if isinstance(bulk, dict) else None)
            for name in tool_names}


def _compact_output_property_names(props: dict) -> list[str]:
    """Return a compact list of top-level field names from a TU output
    schema's ``properties`` block.

    Conservative — we only read the keys; we do NOT recurse into nested
    sub-schemas, do NOT expand `$ref`, and do NOT include the schema
    body itself.
    """
    out: list[str] = []
    seen: set[str] = set()
    for name in props:
        if not isinstance(name, str):
            continue
        if name.startswith("_"):
            continue
        if name in seen:
            continue
        seen.add(name)
        out.append(name)
    return out


def _short_semantics_from_text(text: str, *, limit: int = 240) -> str:
    """Truncate / single-line a free-text description for the catalog.

    Strips leading/trailing whitespace, collapses internal whitespace,
    and clips at ``limit`` chars. Never returns more than ``limit``
    characters to the LLM payload.
    """
    if not isinstance(text, str):
        return ""
    flat = " ".join(text.split()).strip()
    if not flat:
        return ""
    return flat[:limit]


def _expected_output_contract_from_spec(
    spec: dict | None,
) -> tuple[list[str], str, str, str]:
    """Derive (fields, semantics, source, notes) from one TU spec.

    Source priority:

    1. **tooluniverse_spec** — the spec carries a structured output /
       return schema. ``fields`` is the list of top-level property
       names; ``semantics`` is the schema's ``description`` line if
       present.
    2. **tooluniverse_description** — no structured output schema, but
       the spec's free-text ``description`` explicitly talks about
       returns (matches a small marker set). ``fields`` stays empty
       (we never extract names from free text); ``semantics`` is the
       compact description text. The LLM is instructed to treat this
       as low-confidence guidance.
    3. **tooluniverse_metadata_no_output_contract** — spec exists but
       carries neither a structured output schema nor any returns-
       related text. ``fields=[]``, ``semantics=""``.
    4. **unavailable** — no TU spec available at all (TU not installed,
       lookup failed, or tool not in TU inventory). ``fields=[]``,
       ``semantics=""``.
    """
    if not isinstance(spec, dict):
        return [], "", EXPECTED_OUTPUT_SOURCE_UNAVAILABLE, ""

    # 1. Structured output schema.
    for key in _TU_OUTPUT_SCHEMA_KEYS:
        block = spec.get(key)
        if not isinstance(block, dict):
            continue
        props = block.get("properties")
        fields: list[str] = []
        if isinstance(props, dict):
            fields = _compact_output_property_names(props)
        semantics = _short_semantics_from_text(block.get("description") or "")
        if fields or semantics:
            return (
                fields,
                semantics,
                EXPECTED_OUTPUT_SOURCE_TU_SPEC,
                f"derived from TU spec key '{key}'",
            )

    # 2. Description mentions returns.
    description = spec.get("description")
    if isinstance(description, str):
        low = description.lower()
        if any(marker in low for marker in _TU_DESCRIPTION_OUTPUT_MARKERS):
            semantics = _short_semantics_from_text(description)
            if semantics:
                return (
                    [],
                    semantics,
                    EXPECTED_OUTPUT_SOURCE_TU_DESCRIPTION,
                    "derived from TU description; field list not parsed from free text",
                )

    # 3. Spec exists but no output info at all.
    return [], "", EXPECTED_OUTPUT_SOURCE_TU_NO_CONTRACT, ""


def _project_side_hints_from_plan(plan: EnrichmentPlan) -> dict:
    """Project-side, non-TU hints. Kept under a distinct key so a
    reviewer cannot confuse them with TU-derived output metadata.

    - ``redundancy_group`` — Step 5 registry's ``fallback_group``
      (substructure + similarity share one group, etc.).
    - ``identity_strength`` — heuristic flag derived from
      ``capability_type``: ``exact`` for compound ID lookups,
      ``approximate`` for name / substructure / similarity searches,
      ``context`` for structure-context lookups.
    - ``downstream_uses`` — short ordered list of which downstream
      consumer this tool's compact output feeds, derived from
      ``capability_type``.
    """
    cap = plan.capability_type or ""
    if cap == "compound_id_lookup":
        identity_strength = "exact"
    elif cap in (
        "compound_name_lookup",
        "compound_substructure_lookup",
        "compound_similarity_lookup",
    ):
        identity_strength = "approximate"
    elif cap == "antibody_structure_lookup":
        identity_strength = "context"
    else:
        identity_strength = "unknown"
    downstream_uses: list[str] = []
    if cap in (
        "compound_id_lookup",
        "compound_name_lookup",
        "compound_substructure_lookup",
        "compound_similarity_lookup",
    ):
        downstream_uses.append("step_06_compound_liability_lane")
    if cap == "antibody_structure_lookup":
        downstream_uses.append("step_07_structure_context")
    return {
        "redundancy_group": (plan.extra_summary or {}).get("fallback_group")
        or plan.tool_name,
        "identity_strength": identity_strength,
        "downstream_uses": downstream_uses,
    }


def _build_compact_catalog(plans: list[EnrichmentPlan]) -> list[dict]:
    """One compact catalog entry per eligible tool.

    Entry shape (LLM-facing payload):

    - ``tool_name`` — registry-eligible MCP tool name.
    - ``short_description`` — preferred from TU official metadata
      description; falls back to a local one-line summary if TU has
      no description.
    - ``description_source`` — ``tooluniverse_description`` if the
      description came from TU; ``fallback`` if it did not.
    - ``capability_tags`` — Step 5 registry capability_type.
    - ``coarse_input_requirements`` — same Stage-1 mock signal hint as
      the shared tool_selection_policy catalog.
    - ``expected_output_fields`` — list of top-level field names from
      the TU structured output schema, or [] when none exists.
    - ``expected_output_semantics`` — compact one-line summary of what
      the tool is documented to return, drawn from the TU output
      schema's description, the TU spec's free-text description (only
      when it mentions returns), or "" otherwise.
    - ``expected_output_source`` — one of ``tooluniverse_spec`` /
      ``tooluniverse_description`` /
      ``tooluniverse_metadata_no_output_contract`` / ``unavailable``.
    - ``expected_output_notes`` — short string explaining how the
      contract was derived (e.g. "derived from TU spec key
      'return_schema'"); "" when unavailable.
    - ``project_side_hints`` — non-TU hints: redundancy_group,
      identity_strength, downstream_uses. Kept under a separate key so
      a reviewer cannot mistake them for TU-derived output metadata.
    - ``step_id`` / ``agent_name`` — scope tags.

    The catalog MUST NOT carry the full TU input schema, full raw
    output schema, raw payloads, raw example outputs, MCP arguments,
    the ``_live`` flag, or any PDB/CIF/FASTA contents.
    """
    seen: set[str] = set()
    unsorted_names: list[str] = []
    plan_by_name: dict[str, EnrichmentPlan] = {}
    for plan in plans:
        if plan.tool_name in seen:
            continue
        seen.add(plan.tool_name)
        unsorted_names.append(plan.tool_name)
        plan_by_name[plan.tool_name] = plan
    # Sort by tool_name so the catalog prefix is deterministic across
    # candidates and across runs. Provider prompt caching keys off the
    # leading byte sequence of the request, so a stable, sorted list
    # raises the odds of a cache hit on the shared system prompt +
    # Step 5 addendum + early catalog entries. No selection semantics
    # depend on this order; the policy treats `compact_catalog` as an
    # unordered set of allowed tool names.
    ordered_names = sorted(unsorted_names)

    tu_specs = _tu_specs_for_tools(ordered_names)

    out: list[dict] = []
    for name in ordered_names:
        plan = plan_by_name[name]
        spec = tu_specs.get(name)

        tu_description = spec.get("description") if isinstance(spec, dict) else None
        if isinstance(tu_description, str) and tu_description.strip():
            short_description = _short_semantics_from_text(tu_description)
            description_source = DESCRIPTION_SOURCE_TU
        else:
            short_description = (
                f"Step 5 enrichment via {plan.capability_type} "
                f"(query_kind={plan.query_kind})"
            )
            description_source = DESCRIPTION_SOURCE_FALLBACK

        fields, semantics, output_source, notes = (
            _expected_output_contract_from_spec(spec)
        )

        out.append({
            "tool_name": name,
            "short_description": short_description,
            "description_source": description_source,
            "capability_tags": [plan.capability_type],
            "coarse_input_requirements": _coarse_input_requirements(plan),
            "expected_output_fields": fields,
            "expected_output_semantics": semantics,
            "expected_output_source": output_source,
            "expected_output_notes": notes,
            "project_side_hints": _project_side_hints_from_plan(plan),
            "step_id": "step_05",
            "agent_name": "candidate_context_agent",
        })
    return out


def _compact_candidate_context(
    record,
    raw_user_query: str,
) -> dict:
    """Short, redaction-safe summary of the candidate sent to the LLM.

    No raw payloads, no PDB/CIF contents — only typed material types,
    identifier types/values, candidate role/type, and a truncated
    raw_user_query.
    """
    materials = [
        {"material_type": m.material_type, "role": m.role,
         "role_status": m.role_status,
         "value_format": m.value_format}
        for m in getattr(record, "materials", []) or []
    ]
    identifiers = [
        {"id_type": i.id_type, "id_value": i.id_value}
        for i in getattr(record, "identifiers", []) or []
    ]
    return {
        "candidate_id": getattr(record, "candidate_id", ""),
        "candidate_type": getattr(record, "candidate_type", ""),
        "candidate_role": getattr(record, "candidate_role", ""),
        "candidate_label": (getattr(record, "candidate_label", "") or "")[:120],
        "materials": materials,
        "identifiers": identifiers,
        "data_gaps": list(getattr(record, "data_gaps", []) or [])[:10],
        "raw_user_query_excerpt": (raw_user_query or "")[:280],
    }


def build_step5_stage1_payload(
    *,
    catalog: list[dict],
    signals: dict[str, bool],
    record_context: dict,
) -> dict:
    """Assemble the Step 5 Stage-1 ``schema=`` payload for ``generate_json``.

    Ordering note (cache-friendly layout): the stable tool catalog + rules
    metadata (``task`` / ``agent_name`` / ``step_id`` / ``compact_catalog``
    / ``context.note``) is what the prompt renderer places in the cacheable
    stable prefix, and the candidate/run-specific portion
    (``context.candidate`` + ``context.signals``) is what it places in the
    trailing dynamic block. The payload dict itself carries every field the
    MockLLMProvider / selection policy read from ``schema`` — no field is
    dropped, only re-ordered at render time by
    ``json_task_validation.build_json_prompt_sections``.
    """
    return {
        "task": "tool_selection_stage_1",
        "agent_name": "candidate_context_agent",
        "step_id": "step_05",
        "compact_catalog": catalog,
        "context": {
            "signals": signals,
            "note": STEP_05_SELECTION_CONTEXT_NOTE,
            "candidate": record_context,
        },
    }


def select_step5_enrichment_plans(
    *,
    record,
    eligible_plans: list[EnrichmentPlan],
    llm: Optional[LLMProvider],
    raw_user_query: str = "",
) -> tuple[list[Step5ToolDecision], Step5SelectionAudit]:
    """Return (decisions, audit).

    ``decisions`` preserves input order and contains every eligible
    plan plus any synthetic ``known_live_unavailable`` entries, each
    marked ``selected=True/False`` with a compact source + reason. The
    caller executes only ``selected=True`` decisions; synthetic
    dependency records are flagged via ``execution_semantics`` so the
    agent's tool-call writer can emit a ``dependency_unavailable``
    ``ToolCallRecord`` without crediting an LLM.
    """
    audit = Step5SelectionAudit(
        candidate_id=getattr(record, "candidate_id", ""),
        candidate_category=getattr(record, "candidate_type", ""),
    )
    audit.eligible_tools = [_plan_compact_entry(p) for p in eligible_plans]

    if not eligible_plans:
        return [], audit

    # Partition plans: real LLM-selectable plans vs. synthetic
    # dependency-gap plans. The LLM never sees the latter and never
    # gets credit for them.
    real_plans = [p for p in eligible_plans if not p.known_live_unavailable]
    dep_plans = [p for p in eligible_plans if p.known_live_unavailable]

    selected_tool_names, source, fallback_reason = _ask_llm_for_selection(
        llm=llm,
        eligible_real_plans=real_plans,
        record_context=_compact_candidate_context(record, raw_user_query),
        audit=audit,
    )
    audit.tool_selection_source = source
    audit.fallback_reason = fallback_reason

    decisions: list[Step5ToolDecision] = []

    # Synthetic dependency-gap records first (so order is stable in
    # tests). These are NOT real executions and NOT LLM selections.
    for plan in dep_plans:
        dec = Step5ToolDecision(
            plan=plan,
            selected=True,
            tool_selection_source=SOURCE_SYSTEM_DEPENDENCY_GAP,
            selection_reason=REASON_KNOWN_LIVE_UNAVAILABLE,
            execution_semantics=EXEC_SEMANTICS_SYNTHETIC_DEP,
        )
        decisions.append(dec)
        audit.known_unavailable_records.append(_decision_compact(dec))

    # Real plans next.
    for plan in real_plans:
        if plan.tool_name in selected_tool_names:
            if source == SOURCE_LLM_STAGE1:
                reason = REASON_LLM_SELECTED
            elif fallback_reason == FALLBACK_REASON_LLM_EMPTY:
                reason = REASON_FALLBACK_EMPTY
            elif fallback_reason == FALLBACK_REASON_LLM_OUT_OF_SCOPE_ONLY:
                reason = REASON_FALLBACK_OUT_OF_SCOPE
            elif fallback_reason == FALLBACK_REASON_LLM_NOT_CALLED:
                reason = REASON_FALLBACK_NOT_CALLED
            else:
                # FALLBACK_REASON_LLM_UNAVAILABLE or any other
                # deterministic-fallback path.
                reason = REASON_DETERMINISTIC_FALLBACK
            dec = Step5ToolDecision(
                plan=plan,
                selected=True,
                tool_selection_source=source,
                selection_reason=reason,
            )
            decisions.append(dec)
            audit.selected_tools.append(_decision_compact(dec))
        else:
            dec = Step5ToolDecision(
                plan=plan,
                selected=False,
                tool_selection_source=source,
                selection_reason="",
                skip_reason=REASON_LLM_SKIPPED,
            )
            decisions.append(dec)
            audit.skipped_eligible_tools.append(_decision_compact(dec))
    return decisions, audit


def _ask_llm_for_selection(
    *,
    llm: Optional[LLMProvider],
    eligible_real_plans: list[EnrichmentPlan],
    record_context: dict,
    audit: Step5SelectionAudit,
) -> tuple[set[str], str, str]:
    """Run Stage 1 selection over the eligible (non-synthetic) plans.

    Returns ``(selected_tool_names, source, fallback_reason)`` where:

    - ``source`` is ``llm_stage1`` only when the LLM produced ≥1
      in-scope selection; otherwise ``deterministic_fallback``.
    - ``fallback_reason`` is empty on the LLM path and carries one of
      ``llm_unavailable`` / ``llm_empty_selection`` /
      ``llm_out_of_scope_only`` / ``llm_not_called`` on the fail-open
      fallback path.

    Fail-open fallback policy (explicit, audited): when the LLM is
    unavailable / empty / out-of-scope-only, the selected set widens to
    every eligible real plan so Step 5 stays useful. The audit makes
    that fallback visible; the per-plan ``selection_reason`` is set by
    the caller to a deterministic-fallback reason, never
    ``llm_selected``.
    """
    allowed_tool_names = {p.tool_name for p in eligible_real_plans}
    catalog = _build_compact_catalog(eligible_real_plans)
    signals = _build_signals(eligible_real_plans)

    # No eligible real plans → nothing for the LLM to pick. Bypass the
    # call entirely; the caller's loop over real_plans will produce no
    # decisions, leaving only synthetic dependency records (if any).
    if not allowed_tool_names:
        audit.llm_call_status = LLM_STATUS_NOT_CALLED
        return set(), SOURCE_DETERMINISTIC_FALLBACK, FALLBACK_REASON_NONE

    if llm is None:
        audit.llm_call_status = LLM_STATUS_NOT_CALLED
        return (
            set(allowed_tool_names),
            SOURCE_DETERMINISTIC_FALLBACK,
            FALLBACK_REASON_LLM_NOT_CALLED,
        )

    stage1_payload: dict = build_step5_stage1_payload(
        catalog=catalog,
        signals=signals,
        record_context=record_context,
    )
    try:
        resp = llm.generate_json(
            SELECTION_STAGE1_USER_PROMPT,
            schema=stage1_payload,
            system=STEP_05_SELECTION_SYSTEM_PROMPT,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Step 5 LLM tool selection failed (%s); falling back to deterministic eligible set",
            exc,
        )
        audit.llm_call_status = LLM_STATUS_FAILED
        return (
            set(allowed_tool_names),
            SOURCE_DETERMINISTIC_FALLBACK,
            FALLBACK_REASON_LLM_UNAVAILABLE,
        )

    raw_selections = (resp or {}).get("selections") or []
    in_scope: set[str] = set()
    out_of_scope: list[str] = []
    for entry in raw_selections:
        if not isinstance(entry, dict):
            continue
        name = entry.get("tool_name")
        if not isinstance(name, str) or not name:
            continue
        if name in allowed_tool_names:
            in_scope.add(name)
        else:
            out_of_scope.append(name)
    audit.llm_dropped_out_of_scope = out_of_scope

    if in_scope:
        # Partial in-scope is still a real LLM selection — execute only
        # the in-scope picks. Out-of-scope names are dropped but
        # surfaced in the audit, never escalated to all-eligible.
        audit.llm_call_status = (
            LLM_STATUS_OK_WITH_DROPPED if out_of_scope else LLM_STATUS_OK
        )
        return in_scope, SOURCE_LLM_STAGE1, FALLBACK_REASON_NONE

    # Zero in-scope. Distinguish "LLM responded with an empty list" vs
    # "LLM responded but every name was out-of-scope". Both paths
    # fail-open (per explicit product policy) but with distinct
    # fallback_reason strings so the audit is honest.
    if out_of_scope:
        audit.llm_call_status = LLM_STATUS_OUT_OF_SCOPE_ONLY
        return (
            set(allowed_tool_names),
            SOURCE_DETERMINISTIC_FALLBACK,
            FALLBACK_REASON_LLM_OUT_OF_SCOPE_ONLY,
        )
    audit.llm_call_status = LLM_STATUS_EMPTY
    return (
        set(allowed_tool_names),
        SOURCE_DETERMINISTIC_FALLBACK,
        FALLBACK_REASON_LLM_EMPTY,
    )


def _decision_compact(decision: Step5ToolDecision) -> dict:
    plan = decision.plan
    return {
        "tool_name": plan.tool_name,
        "capability_type": plan.capability_type,
        "query_kind": plan.query_kind,
        "query_role": plan.query_role,
        "material_type": plan.material_type,
        "tool_selection_source": decision.tool_selection_source,
        "selection_reason": decision.selection_reason,
        "skip_reason": decision.skip_reason,
        "argument_construction_source": decision.argument_construction_source,
        "execution_semantics": decision.execution_semantics,
        "policy_version": SELECTION_POLICY_VERSION,
    }


def selection_provenance_for_tool_input_summary(
    decision: Step5ToolDecision,
    *,
    eligible_count: int,
    real_selected_count: int,
    skipped_count: int,
    known_unavailable_count: int,
    fallback_reason: str,
) -> dict:
    """Compact provenance dict added to ``tool_input_summary`` at execute time.

    ``real_selected_count`` is the count of REAL MCP executions
    (synthetic dependency-gap records are reported separately via
    ``known_unavailable_count``). ``fallback_reason`` is the explicit
    fail-open reason when the source is deterministic_fallback, empty
    otherwise.
    """
    return {
        "tool_selection_source": decision.tool_selection_source,
        "selection_reason": decision.selection_reason,
        "argument_construction_source": decision.argument_construction_source,
        "execution_semantics": decision.execution_semantics,
        "selection_policy_version": SELECTION_POLICY_VERSION,
        "eligible_count": eligible_count,
        "real_selected_count": real_selected_count,
        "skipped_eligible_count": skipped_count,
        "known_unavailable_count": known_unavailable_count,
        "fallback_reason": fallback_reason,
    }
