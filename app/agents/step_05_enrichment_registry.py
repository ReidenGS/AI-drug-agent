"""Step 5 metadata-driven deterministic enrichment planner.

This module is deliberately small: it maps normalized Step 5 candidate
metadata into scoped MCP tool plans. It does not call MCP, parse raw
payloads, use LLMs, or build Step 5 schema records.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional


# ── Logical capability registry ────────────────────────────────────────────


@dataclass(frozen=True)
class Step5EnrichmentCapability:
    tool_name: str
    capability_type: str
    required_input_slots: tuple[str, ...]
    accepted_input_slots: tuple[str, ...]
    schema_arg_mapping: dict[str, str]
    priority: int = 100
    fallback_group: str = ""
    max_calls_per_candidate: int = 5
    candidate_categories: tuple[str, ...] = ("compound_component",)
    output_extractor_type: str = "compound"
    provenance_policy: str = "direct_lookup"
    confidence_policy: str = "default_0_8"
    live_policy: str = "normal"
    known_unavailable_policy: str = ""
    notes: str = ""

    @property
    def known_live_unavailable(self) -> bool:
        return bool(self.known_unavailable_policy)

    def schema_arg_name_for_slot(self, slot_kind: str) -> str:
        return (
            self.schema_arg_mapping.get(slot_kind)
            or self.schema_arg_mapping.get("*")
            or "query"
        )


# Default registry — every entry MUST be a tool already inside the agent /
# step's scoped catalog. We never widen scope here; capabilities whose
# tool_name is missing from `scoped_tools` at planning time are silently
# ignored.
STEP_05_CAPABILITY_REGISTRY: tuple[Step5EnrichmentCapability, ...] = (
    # ── Antibody / target name → SAbDab structure search ─────────────
    Step5EnrichmentCapability(
        tool_name="SAbDab_search_structures",
        capability_type="antibody_structure_lookup",
        required_input_slots=("antibody_name", "target_antigen_name"),
        accepted_input_slots=("antibody_name", "target_antigen_name"),
        schema_arg_mapping={"*": "query"},
        priority=50,
        fallback_group="sabdab_structure",
        max_calls_per_candidate=1,
        candidate_categories=("antibody", "target_antigen"),
        output_extractor_type="sabdab_structure",
        provenance_policy="structure_context_lookup",
        confidence_policy="context_only",
        notes=(
            "Live SAbDab response may carry no heavy/light sequence field; "
            "no-sequence outcome is recorded as a Step 5 data_gap by the "
            "agent's SAbDab outcome annotator."
        ),
    ),
    # ── ChEMBL id → exact molecule lookup ─────────────────────────────
    Step5EnrichmentCapability(
        tool_name="ChEMBL_get_molecule",
        capability_type="compound_id_lookup",
        required_input_slots=("chembl_id",),
        accepted_input_slots=("chembl_id",),
        schema_arg_mapping={"chembl_id": "chembl_id"},
        priority=70,
        fallback_group="chembl_exact_id",
        max_calls_per_candidate=3,
        output_extractor_type="compound",
        provenance_policy="exact_identifier_lookup",
        confidence_policy="exact_id_0_9",
    ),
    # ── Compound name → ChEMBL_search_molecules ──────────────────────
    Step5EnrichmentCapability(
        tool_name="ChEMBL_search_molecules",
        capability_type="compound_name_lookup",
        required_input_slots=("payload_name", "linker_name", "compound_name", "linker_payload_name"),
        accepted_input_slots=(
            "payload_name", "linker_name", "compound_name", "linker_payload_name",
        ),
        schema_arg_mapping={"*": "query"},
        priority=100,
        fallback_group="chembl_name",
        max_calls_per_candidate=4,
        output_extractor_type="compound",
        provenance_policy="name_lookup",
        confidence_policy="name_match_0_8",
    ),
    # ── SMILES → ChEMBL_search_substructure ──────────────────────────
    Step5EnrichmentCapability(
        tool_name="ChEMBL_search_substructure",
        capability_type="compound_substructure_lookup",
        required_input_slots=("payload_smiles", "linker_smiles", "compound_smiles"),
        accepted_input_slots=("payload_smiles", "linker_smiles", "compound_smiles"),
        schema_arg_mapping={"*": "smiles"},
        priority=110,
        fallback_group="chembl_smiles",
        max_calls_per_candidate=3,
        output_extractor_type="compound",
        provenance_policy="substructure_upper_bound",
        confidence_policy="substructure_0_5",
        notes=(
            "Substructure-derived chembl_id is upper-bound identity, not "
            "confirmed exact identity for the user's compound."
        ),
    ),
    # ── SMILES → ChEMBL_search_similarity (lower priority fallback) ──
    Step5EnrichmentCapability(
        tool_name="ChEMBL_search_similarity",
        capability_type="compound_similarity_lookup",
        required_input_slots=("payload_smiles", "linker_smiles", "compound_smiles"),
        accepted_input_slots=("payload_smiles", "linker_smiles", "compound_smiles"),
        schema_arg_mapping={"*": "smiles"},
        priority=200,
        fallback_group="chembl_smiles",
        max_calls_per_candidate=2,
        output_extractor_type="compound",
        provenance_policy="similarity_upper_bound",
        confidence_policy="substructure_0_5",
        notes=(
            "Similarity-derived chembl_id; treated like substructure-derived "
            "for confidence and provenance."
        ),
    ),
    # ── ZINC family: live disabled / captcha-gated ────────────────────
    Step5EnrichmentCapability(
        tool_name="ZINC_search_by_smiles",
        capability_type="compound_substructure_lookup",
        required_input_slots=("payload_smiles", "linker_smiles", "compound_smiles"),
        accepted_input_slots=("payload_smiles", "linker_smiles", "compound_smiles"),
        schema_arg_mapping={"*": "smiles"},
        priority=300,
        fallback_group="zinc_smiles",
        max_calls_per_candidate=1,
        output_extractor_type="unsupported",
        provenance_policy="zinc_context_only",
        confidence_policy="unavailable",
        live_policy="disabled",
        known_unavailable_policy="ZINC live disabled / captcha-gated",
        notes="ZINC live disabled / captcha-gated; treat as dependency_unavailable.",
    ),
    Step5EnrichmentCapability(
        tool_name="ZINC_get_compound",
        capability_type="compound_id_lookup",
        required_input_slots=("zinc_id",),
        accepted_input_slots=("zinc_id",),
        schema_arg_mapping={"zinc_id": "zinc_id"},
        priority=300,
        fallback_group="zinc_id",
        max_calls_per_candidate=1,
        output_extractor_type="unsupported",
        provenance_policy="zinc_context_only",
        confidence_policy="unavailable",
        live_policy="disabled",
        known_unavailable_policy="ZINC live disabled / captcha-gated",
        notes="ZINC live disabled / captcha-gated.",
    ),
    Step5EnrichmentCapability(
        tool_name="ZINC_search_compounds",
        capability_type="compound_name_lookup",
        required_input_slots=("payload_name", "linker_name", "compound_name", "linker_payload_name"),
        accepted_input_slots=(
            "payload_name", "linker_name", "compound_name", "linker_payload_name",
        ),
        schema_arg_mapping={"*": "query"},
        priority=310,
        fallback_group="zinc_name",
        max_calls_per_candidate=1,
        output_extractor_type="unsupported",
        provenance_policy="zinc_context_only",
        confidence_policy="unavailable",
        live_policy="disabled",
        known_unavailable_policy="ZINC live disabled / captcha-gated",
        notes="ZINC live disabled / captcha-gated.",
    ),
)


# ── Planner output ─────────────────────────────────────────────────────────


@dataclass(frozen=True)
class EnrichmentPlan:
    tool_name: str
    query: str
    query_kind: str
    query_role: Optional[str]
    material_type: str
    schema_arg_name: str
    capability_type: str
    output_extractor_type: str
    provenance_policy: str
    confidence_policy: str
    known_live_unavailable: bool
    known_unavailable_reason: str = ""
    extra_summary: dict[str, str] = field(default_factory=dict)


# ── Candidate slot resolution ──────────────────────────────────────────────


def _slots_from_candidate(record) -> list[tuple[str, str, Optional[str]]]:
    """Yield ``(slot_kind, value, role)`` from materials AND identifiers.

    Slot kind matches ``material_type`` for materials and ``id_type`` for
    identifiers. Empty values are skipped. Role is only carried for
    materials (identifiers have no role).
    """
    out: list[tuple[str, str, Optional[str]]] = []
    for m in record.materials:
        value = (m.value or "").strip() if isinstance(m.value, str) else ""
        if value:
            out.append((m.material_type, value, m.role))
    for i in record.identifiers:
        value = (i.id_value or "").strip() if isinstance(i.id_value, str) else ""
        if value:
            out.append((i.id_type, value, None))
    return out


def _eligible_capabilities(
    scoped_tools: frozenset[str],
    candidate_category: str,
    *,
    include_known_unavailable: bool,
    registry: Iterable[Step5EnrichmentCapability],
) -> list[Step5EnrichmentCapability]:
    keep: list[Step5EnrichmentCapability] = []
    for cap in registry:
        if cap.tool_name not in scoped_tools:
            continue
        if cap.candidate_categories and candidate_category not in cap.candidate_categories:
            continue
        if cap.known_live_unavailable and not include_known_unavailable:
            continue
        keep.append(cap)
    keep.sort(key=lambda c: c.priority)
    return keep


# ── Public planner ────────────────────────────────────────────────────────


NameSanitizer = Callable[[str, Optional[str]], Optional[str]]
SmilesSanitizer = Callable[[str], bool]


_KNOWN_SHORT_CHEMBL_NAME_KEYS = {
    "dm1",
    "dm4",
    "dxd",
    "mmae",
    "mmaf",
    "sn38",
    "sn-38",
}
_CHEMBL_NAME_SLOTS = {
    "payload_name",
    "linker_name",
    "compound_name",
    "linker_payload_name",
}


def _name_key(value: str) -> str:
    return "".join(ch.lower() for ch in value.strip() if ch.isalnum() or ch == "-")


def is_low_information_chembl_name_query(
    query: str,
    *,
    material_type: str | None = None,
    role: str | None = None,
) -> bool:
    """Conservative quality gate for direct ChEMBL name lookups.

    Very short aliases such as ``vc`` are too broad for high-confidence
    molecule name enrichment. Known compact payload/drug names remain
    allowed via an explicit allowlist so this does not become a blunt
    length cutoff.
    """
    q = " ".join((query or "").strip().split())
    if not q:
        return True
    key = _name_key(q)
    if key in _KNOWN_SHORT_CHEMBL_NAME_KEYS:
        return False
    if "-" in q or " " in q:
        return False
    alnum = "".join(ch for ch in q if ch.isalnum())
    if len(alnum) <= 2:
        return True
    return False


def skipped_low_information_chembl_name_queries(
    record,
    *,
    scoped_tools: Iterable[str],
    candidate_category: str,
    name_query_sanitizer: Optional[NameSanitizer] = None,
) -> list[tuple[str, str, Optional[str]]]:
    """Return skipped ``(material_type, query, role)`` tuples for audit notes."""
    if "ChEMBL_search_molecules" not in frozenset(scoped_tools):
        return []
    if candidate_category != "compound_component":
        return []
    skipped: list[tuple[str, str, Optional[str]]] = []
    seen: set[tuple[str, str]] = set()
    for slot_kind, value, role in _slots_from_candidate(record):
        if slot_kind not in _CHEMBL_NAME_SLOTS:
            continue
        query = (
            name_query_sanitizer(value, role)
            if name_query_sanitizer is not None
            else value
        )
        if not query:
            continue
        if not is_low_information_chembl_name_query(
            query,
            material_type=slot_kind,
            role=role,
        ):
            continue
        key = (slot_kind, query.lower())
        if key in seen:
            continue
        seen.add(key)
        skipped.append((slot_kind, query, role))
    return skipped


def plan_enrichment_for_record(
    record,
    *,
    scoped_tools: Iterable[str],
    candidate_category: str,
    name_query_sanitizer: Optional[NameSanitizer] = None,
    smiles_query_sanitizer: Optional[SmilesSanitizer] = None,
    include_known_unavailable: bool = False,
    registry: Iterable[Step5EnrichmentCapability] = STEP_05_CAPABILITY_REGISTRY,
) -> list[EnrichmentPlan]:
    """Produce a deterministic list of enrichment plans for one candidate.

    ``scoped_tools`` should be the result of
    ``mcp_client.list_tools(agent_name="candidate_context_agent",
    step_id="step_05")`` — every plan we emit references a tool already in
    that scope, so the planner cannot widen MCP access.
    """
    scoped = frozenset(scoped_tools)
    slots = _slots_from_candidate(record)
    caps = _eligible_capabilities(
        scoped, candidate_category,
        include_known_unavailable=include_known_unavailable,
        registry=registry,
    )
    plans: list[EnrichmentPlan] = []
    seen: set[tuple[str, str]] = set()
    per_tool_count: dict[str, int] = {}
    used_fallback_slots: set[tuple[str, str, str]] = set()
    for cap in caps:
        for (slot_kind, value, role) in slots:
            if slot_kind not in cap.accepted_input_slots:
                continue
            query: Optional[str]
            query_kind: str
            if cap.capability_type == "compound_name_lookup":
                query = (
                    name_query_sanitizer(value, role)
                    if name_query_sanitizer is not None
                    else value
                )
                query_kind = "name"
            elif cap.capability_type in (
                "compound_substructure_lookup", "compound_similarity_lookup",
            ):
                if smiles_query_sanitizer is not None and not smiles_query_sanitizer(value):
                    query = None
                else:
                    query = value
                query_kind = "smiles"
            elif cap.capability_type == "compound_id_lookup":
                query = value
                query_kind = slot_kind  # e.g. "chembl_id"
            elif cap.capability_type == "antibody_structure_lookup":
                query = value
                query_kind = "name"
            else:
                query = value
                query_kind = slot_kind
            if not query:
                continue
            if (
                cap.tool_name == "ChEMBL_search_molecules"
                and query_kind == "name"
                and is_low_information_chembl_name_query(
                    query,
                    material_type=slot_kind,
                    role=role,
                )
            ):
                continue
            fallback_group = cap.fallback_group or cap.tool_name
            fallback_key = (fallback_group, slot_kind, query.lower())
            if fallback_key in used_fallback_slots:
                continue
            key = (cap.tool_name, query.lower())
            if key in seen:
                continue
            if per_tool_count.get(cap.tool_name, 0) >= cap.max_calls_per_candidate:
                continue
            seen.add(key)
            used_fallback_slots.add(fallback_key)
            per_tool_count[cap.tool_name] = per_tool_count.get(cap.tool_name, 0) + 1
            plans.append(EnrichmentPlan(
                tool_name=cap.tool_name,
                query=query,
                query_kind=query_kind,
                query_role=role,
                material_type=slot_kind,
                schema_arg_name=cap.schema_arg_name_for_slot(slot_kind),
                capability_type=cap.capability_type,
                output_extractor_type=cap.output_extractor_type,
                provenance_policy=cap.provenance_policy,
                confidence_policy=cap.confidence_policy,
                known_live_unavailable=cap.known_live_unavailable,
                known_unavailable_reason=cap.known_unavailable_policy,
                extra_summary={
                    "fallback_group": fallback_group,
                    "provenance_policy": cap.provenance_policy,
                    "confidence_policy": cap.confidence_policy,
                },
            ))
    return plans
