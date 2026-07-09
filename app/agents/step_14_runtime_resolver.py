"""Step 14 runtime resolver.

The Step 14 request is reference-only; the LLM selects tools from input-ref
roles / supports_tool_args without ever seeing a value. This module is the
ONLY place a real value is resolved: given a `Step14InputRef`, it reads the
declared ``source_artifact`` from run storage and pulls the value that the
ref's ``role`` (scoped to ``candidate_id`` when given) points at.

Resolution is deliberately narrow:
- it only reads artifacts the request explicitly declared in
  ``source_artifact_refs`` (so it never discovers extra Step 2/Step 5 fields
  beyond what the request handed it),
- it resolves compact, non-sensitive values only (CID, brand / drug name,
  application number, payload / linker / target names) — never raw sequences,
  FASTA, PDB/CIF, API keys, or raw tool payloads,
- a miss returns a compact ``unresolved_reason`` instead of a fake value.

Resolved values feed runtime tool arguments and the ``tool_input_summary``
only; the compact audit record deliberately omits the value.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from ..schemas.step_14_patent_request import Step14InputRef, Step14PatentRequest
from ..services.storage_service import Storage


# Declared source_artifact name → canonical run-storage key parts.
_ARTIFACT_RUN_KEYS: dict[str, tuple[str, ...]] = {
    "candidate_context_table": ("candidate_context_table.json",),
    "structured_query": ("inputs", "structured_query.json"),
}


@dataclass
class Step14ResolvedRef:
    ref_id: str
    role: str
    source_artifact: str
    source_path: str
    candidate_id: Optional[str] = None
    supports_tool_args: list[str] = field(default_factory=list)
    resolved: bool = False
    value: Optional[str] = None
    unresolved_reason: Optional[str] = None

    def audit_entry(self) -> dict[str, Any]:
        """Compact, value-free audit record (no resolved value leaks here)."""
        return {
            "ref_id": self.ref_id,
            "role": self.role,
            "source_artifact": self.source_artifact,
            "source_path": self.source_path,
            "candidate_id": self.candidate_id,
            "resolved": self.resolved,
            "unresolved_reason": self.unresolved_reason,
        }


def resolve_step14_input_ref(
    storage: Storage,
    request: Step14PatentRequest,
    input_ref: Step14InputRef,
) -> Step14ResolvedRef:
    """Resolve one Step 14 input ref to a compact value, or a miss reason."""
    result = Step14ResolvedRef(
        ref_id=input_ref.ref_id,
        role=input_ref.role,
        source_artifact=input_ref.source_artifact,
        source_path=input_ref.source_path,
        candidate_id=input_ref.candidate_id,
        supports_tool_args=list(input_ref.supports_tool_args),
    )

    # Antibody is gated by scope even at resolution time.
    if input_ref.role == "antibody" and not request.patent_scope.antibody_search_allowed:
        result.unresolved_reason = "antibody_search_not_allowed"
        return result

    # Only read artifacts the request explicitly declared.
    if input_ref.source_artifact not in request.source_artifact_refs:
        result.unresolved_reason = "source_artifact_not_declared"
        return result

    key_parts = _ARTIFACT_RUN_KEYS.get(input_ref.source_artifact)
    if not key_parts:
        result.unresolved_reason = "unsupported_source_artifact"
        return result

    key = storage.run_key(request.run_id, *key_parts)
    if not storage.exists(key):
        result.unresolved_reason = "source_artifact_missing_in_storage"
        return result
    artifact = storage.read_json(key)

    # A source_path is only honored when it is a canonical path that supports
    # this ref's role. This enforces the request-based contract: the request
    # declares WHERE the value lives, and resolution never wanders to a
    # different field just because the role would match elsewhere.
    allowed_roles = _SUPPORTED_PATHS.get(
        (input_ref.source_artifact, input_ref.source_path)
    )
    if allowed_roles is None or input_ref.role not in allowed_roles:
        result.unresolved_reason = "source_path_not_supported_for_role"
        return result

    value = _resolve_value(artifact, input_ref)
    if value is None or not str(value).strip():
        result.unresolved_reason = "value_not_found_for_role"
        return result

    result.resolved = True
    result.value = str(value).strip()
    return result


# ── canonical (source_artifact, source_path) → supported roles ──────────────
#
# Resolution is gated on these exact paths. A ref whose source_path is not
# listed here (or is listed but not for its role) is left unresolved with
# ``source_path_not_supported_for_role``.
_SUPPORTED_PATHS: dict[tuple[str, str], frozenset[str]] = {
    ("structured_query", "mentioned_entities.payload_text"): frozenset({"payload"}),
    ("structured_query", "mentioned_entities.linker_text"): frozenset({"linker"}),
    ("structured_query", "mentioned_entities.target_or_antigen_text"): frozenset({"target"}),
    ("structured_query", "mentioned_entities.antibody_candidate_text"): frozenset({"antibody"}),
    ("structured_query", "referenced_inputs[].value"): frozenset(
        {"pubchem_cid", "application_number"}
    ),
    ("structured_query", "normalized_entities[].canonical_name"): frozenset(
        {"brand_name", "drug_name", "compound", "complete_adc", "linker_payload"}
    ),
    ("structured_query", "normalized_entities[].original_text"): frozenset(
        {"brand_name", "drug_name", "compound", "complete_adc", "linker_payload"}
    ),
    ("candidate_context_table", "candidate_records[].identifiers[].id_value"): frozenset(
        {"pubchem_cid", "application_number"}
    ),
    ("candidate_context_table", "candidate_records[].materials[].value"): frozenset(
        {"brand_name", "drug_name", "compound"}
    ),
    ("candidate_context_table", "downstream_query_hints[].entity"): frozenset(
        {"payload", "linker", "linker_payload", "target", "complete_adc", "antibody"}
    ),
}


def _resolve_value(artifact: dict, ref: Step14InputRef) -> Optional[str]:
    """Resolve the value at the (already validated) canonical source_path."""
    path = ref.source_path
    if ref.source_artifact == "structured_query":
        mentioned = artifact.get("mentioned_entities") or {}
        if path == "mentioned_entities.payload_text":
            return mentioned.get("payload_text")
        if path == "mentioned_entities.linker_text":
            return mentioned.get("linker_text")
        if path == "mentioned_entities.target_or_antigen_text":
            return mentioned.get("target_or_antigen_text")
        if path == "mentioned_entities.antibody_candidate_text":
            return mentioned.get("antibody_candidate_text")
        if path == "referenced_inputs[].value":
            return _referenced_input_value(artifact, ref.role)
        if path in {
            "normalized_entities[].canonical_name",
            "normalized_entities[].original_text",
        }:
            field = "canonical_name" if path.endswith("canonical_name") else "original_text"
            return _normalized_entity_value(artifact, field)
        return None

    # candidate_context_table
    candidate = _find_candidate(artifact, ref.candidate_id)
    if path == "candidate_records[].identifiers[].id_value":
        return _identifier_value_for_role(candidate, ref.role)
    if path == "candidate_records[].materials[].value":
        return _material_value(candidate, {"payload_name", "compound_name", "linker_name"})
    if path == "downstream_query_hints[].entity":
        return _hint_entity(artifact, ref.role)
    return None


# ── candidate_context_table field resolvers ─────────────────────────────────


def _find_candidate(artifact: dict, candidate_id: Optional[str]) -> Optional[dict]:
    records = artifact.get("candidate_records") or []
    if candidate_id:
        for rec in records:
            if isinstance(rec, dict) and rec.get("candidate_id") == candidate_id:
                return rec
        return None
    # No candidate scoping: prefer a compound_component record, else the first.
    for rec in records:
        if isinstance(rec, dict) and rec.get("candidate_type") == "compound_component":
            return rec
    return records[0] if records and isinstance(records[0], dict) else None


def _identifier_value_for_role(candidate: Optional[dict], role: str) -> Optional[str]:
    if not candidate:
        return None
    wanted = {
        "pubchem_cid": {"pubchem_cid"},
        "application_number": {"application_number", "patent_application_number"},
    }.get(role)
    if not wanted:
        return None
    for ident in candidate.get("identifiers") or []:
        if isinstance(ident, dict) and ident.get("id_type") in wanted:
            v = ident.get("id_value")
            if v:
                return str(v)
    return None


def _material_value(candidate: Optional[dict], material_types: set[str]) -> Optional[str]:
    if not candidate:
        return None
    for mat in candidate.get("materials") or []:
        if isinstance(mat, dict) and mat.get("material_type") in material_types:
            v = mat.get("value")
            if v:
                return str(v)
    return None


def _hint_entity(artifact: dict, role: str) -> Optional[str]:
    for hint in artifact.get("downstream_query_hints") or []:
        if not isinstance(hint, dict):
            continue
        if hint.get("role") != role:
            continue
        entity = hint.get("entity")
        if entity:
            return str(entity)
    return None


# ── structured_query field resolvers ────────────────────────────────────────


def _referenced_input_value(artifact: dict, role: str) -> Optional[str]:
    wanted = {
        "pubchem_cid": {"pubchem_cid"},
        "application_number": {"application_number", "patent_application_number"},
    }.get(role)
    if not wanted:
        return None
    for rin in artifact.get("referenced_inputs") or []:
        if isinstance(rin, dict) and rin.get("id_type") in wanted:
            v = rin.get("value")
            if v:
                return str(v)
    return None


def _normalized_entity_value(artifact: dict, field: str) -> Optional[str]:
    for ne in artifact.get("normalized_entities") or []:
        if not isinstance(ne, dict):
            continue
        if ne.get("entity_type") in {"drug", "compound", "linker_payload"}:
            name = ne.get(field) or ne.get("canonical_name") or ne.get("original_text")
            if name:
                return str(name)
    return None
