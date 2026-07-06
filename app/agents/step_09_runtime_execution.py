"""Step 9 runtime value resolution — the bridge from planning to execution.

`step_09_runtime_planner` resolves Stage 2 mappings into a kwargs CONTRACT
(placeholders + literals only, never real values). This module takes that
contract plus `Step9InputProjection.input_fields` and resolves each
`field_ref`-sourced argument to its REAL runtime value by walking that
field's `runtime_lookup` breadcrumb back to the raw Step 5 / Step 7 / Step 8
artifact it points at.

This is, along with `step_09_input_projection`, the only module allowed to
read those raw artifacts — and only to resolve an ALREADY-projected field's
real value, never to re-derive Step 9 field semantics (that stays the sole
responsibility of `step_09_input_projection`). It never calls MCP tools
itself; `StructureAndDesignAgent.run_step_9` executes the resolved requests
through its own `_call_tool` dispatch so ToolCallRecord / storage
conventions stay centralized in one place.

Hard privacy rule: the REAL values this module resolves exist only inside
the returned `kwargs` dict (consumed once, for the MCP call, and discarded).
`kwargs_redacted_summary` — the only part of this module's output that is
safe to persist into `tool_input_summary` / the normalized artifact — never
contains a raw sequence, PDB body, storage path, or API key; only
`field_ref` / `field_type` / `value_kind` / length / sha256 prefix, or (for
official schema literals, which are fixed non-secret vocabulary) the literal
value itself.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from ..services.storage_service import Storage


def resolve_step9_execution_requests(
    *,
    kwargs_contracts: list[dict[str, Any]],
    input_fields: list[Any],
    candidate_context_table: dict[str, Any] | None,
    prepared_structure_input_package: dict[str, Any] | list[dict[str, Any]] | None,
    structure_prediction_and_interface_results: dict[str, Any] | None,
    storage: Storage,
) -> list[dict[str, Any]]:
    """Resolve every `step_09_runtime_planner` kwargs contract into either a
    real, executable kwargs dict (used only for the MCP call) or an explicit
    skip with an unresolved reason. Never invents a value: every
    `field_ref`-sourced argument is resolved strictly through that field's
    `runtime_lookup`.
    """

    candidate_context_table = candidate_context_table or {}
    prepared_inputs = _prepared_inputs_list(prepared_structure_input_package)
    step8_result = structure_prediction_and_interface_results or {}
    fields_by_ref = {
        str(field.get("field_ref") or ""): field
        for field in _model_dump_list(input_fields)
        if str(field.get("field_ref") or "")
    }

    requests: list[dict[str, Any]] = []
    for contract in kwargs_contracts:
        tool_name = str(contract.get("tool_name") or "")
        lane_type = str(contract.get("lane_type") or "")
        if contract.get("can_build_kwargs") is not True:
            requests.append(
                {
                    "tool_name": tool_name,
                    "lane_type": lane_type,
                    "can_execute": False,
                    "kwargs": None,
                    "kwargs_redacted_summary": {},
                    "skip_reason": "runtime_plan_unresolved",
                    "unresolved_reasons": list(contract.get("unresolved_reasons") or []),
                }
            )
            continue

        kwargs: dict[str, Any] = {}
        redacted: dict[str, Any] = {}
        unresolved_reasons: list[str] = []
        for item in contract.get("kwargs_plan") or []:
            if not isinstance(item, dict):
                continue
            runtime_arg = str(item.get("runtime_arg") or "")
            source = str(item.get("source") or "")
            if not runtime_arg:
                continue

            if source == "official_schema_literal":
                literal_value = item.get("literal_value")
                kwargs[runtime_arg] = literal_value
                redacted[runtime_arg] = {"source": "literal", "value": literal_value}
                continue

            if source == "field_ref":
                field_ref = str(item.get("field_ref") or "")
                field = fields_by_ref.get(field_ref)
                if field is None:
                    unresolved_reasons.append(f"{runtime_arg}:field_ref_not_in_projection")
                    continue
                value, error = resolve_step9_field_value(
                    field,
                    candidate_context_table=candidate_context_table,
                    prepared_inputs=prepared_inputs,
                    step8_result=step8_result,
                    storage=storage,
                )
                if value is None:
                    unresolved_reasons.append(f"{runtime_arg}:{error or 'value_not_resolvable'}")
                    continue
                kwargs[runtime_arg] = value
                redacted[runtime_arg] = _redacted_summary_for_field(field, value)
                continue

            unresolved_reasons.append(f"{runtime_arg}:unresolved_in_contract")

        can_execute = not unresolved_reasons
        requests.append(
            {
                "tool_name": tool_name,
                "lane_type": lane_type,
                "can_execute": can_execute,
                "kwargs": kwargs if can_execute else None,
                "kwargs_redacted_summary": redacted if can_execute else {},
                "skip_reason": "" if can_execute else "runtime_value_resolution_failed",
                "unresolved_reasons": unresolved_reasons,
            }
        )
    return requests


# ── per-field real-value resolution ─────────────────────────────────────────

def resolve_step9_field_value(
    field: dict[str, Any],
    *,
    candidate_context_table: dict[str, Any],
    prepared_inputs: list[dict[str, Any]],
    step8_result: dict[str, Any],
    storage: Storage,
) -> tuple[str | None, str | None]:
    """Resolve one `Step9InputField`'s real runtime value.

    Returns ``(value, None)`` on success or ``(None, reason)`` when the
    field cannot be resolved to a real, usable value right now. Dispatch is
    keyed on `field_ref`'s shape (the same shapes `step_09_input_projection`
    produces) plus `runtime_lookup` — never on re-inspecting Step 5/7/8
    semantics beyond following the breadcrumb.
    """

    if not field.get("can_resolve_at_runtime"):
        return None, "field_not_marked_runtime_resolvable"
    if str(field.get("status") or "") != "available":
        return None, "field_status_not_available"

    field_ref = str(field.get("field_ref") or "")

    if field_ref.startswith("identifier:"):
        return _resolve_identifier_value(field)

    if field_ref.startswith("step7_structure_ref:"):
        return _resolve_step7_structure_ref(field, prepared_inputs=prepared_inputs, storage=storage)

    if field_ref.startswith("step8_complex_ref:"):
        return _resolve_step8_complex_ref(
            field,
            step8_result=step8_result,
            prepared_inputs=prepared_inputs,
            candidate_context_table=candidate_context_table,
            storage=storage,
        )

    if field_ref.startswith("step8_validated_structure_ref:"):
        return _resolve_step8_handoff_structure_ref(
            field,
            handoff_key="validated_structure_ref",
            step8_result=step8_result,
            prepared_inputs=prepared_inputs,
            candidate_context_table=candidate_context_table,
            storage=storage,
        )

    if field_ref.startswith("step8_variant_structure_ref:"):
        return _resolve_step8_handoff_structure_ref(
            field,
            handoff_key="structure_for_variant_generation_ref",
            step8_result=step8_result,
            prepared_inputs=prepared_inputs,
            candidate_context_table=candidate_context_table,
            storage=storage,
        )

    if field_ref.startswith("step7_sequence:"):
        return _resolve_step7_sequence(
            field,
            prepared_inputs=prepared_inputs,
            candidate_context_table=candidate_context_table,
            storage=storage,
        )

    if field_ref.startswith("material:"):
        return _resolve_material_value(field, candidate_context_table=candidate_context_table)

    return None, "unsupported_field_ref_shape"


# ── identifiers embedded directly in field_ref ──────────────────────────────
#
# `identifier:{id_type}:{value}` is how `step_09_input_projection` already
# projects PDB ids / UniProt accessions / chain ids / variant-mutation
# notation for every source step (Step 5, Step 7, Step 8) — these are short
# public/domain identifiers, not raw sequence/structure payload, so the
# projection layer itself already embeds the real value in `field_ref`
# (visible even in the Stage 1/2 LLM prompts). Resolution is therefore just
# extraction, with one extra guard: a `pdb_id`-kind field must be a true
# 4-character PDB id, never an uploaded path or material id mistakenly
# routed here.

def _resolve_identifier_value(field: dict[str, Any]) -> tuple[str | None, str | None]:
    field_ref = str(field.get("field_ref") or "")
    parts = field_ref.split(":", 2)
    if len(parts) != 3 or not parts[2].strip():
        return None, "identifier_value_missing"
    value = parts[2].strip()
    if str(field.get("value_kind") or "") == "pdb_id" and not _looks_like_pdb_id(value):
        return None, "not_a_true_pdb_id"
    return value, None


# ── Step 7 structure_refs[index].storage_ref ────────────────────────────────

def _resolve_step7_structure_ref(
    field: dict[str, Any], *, prepared_inputs: list[dict[str, Any]], storage: Storage
) -> tuple[str | None, str | None]:
    runtime_lookup = field.get("runtime_lookup") or {}
    structure_input_id = str(runtime_lookup.get("structure_input_id") or "")
    index = runtime_lookup.get("index")
    if not structure_input_id or not isinstance(index, int):
        return None, "runtime_lookup_incomplete"
    for sin in prepared_inputs:
        if str(sin.get("structure_input_id") or "") != structure_input_id:
            continue
        refs = sin.get("structure_refs") or []
        if index >= len(refs):
            return None, "structure_ref_index_out_of_range"
        ref = refs[index]
        storage_ref = ref.get("storage_ref") if isinstance(ref, dict) else None
        if isinstance(storage_ref, str) and _path_exists(storage, storage_ref):
            return storage_ref, None
        return None, "structure_ref_storage_ref_invalid"
    return None, "structure_input_not_found"


# ── Step 8 complex_structure_refs[index] ────────────────────────────────────

def _resolve_step8_complex_ref(
    field: dict[str, Any],
    *,
    step8_result: dict[str, Any],
    prepared_inputs: list[dict[str, Any]],
    candidate_context_table: dict[str, Any],
    storage: Storage,
) -> tuple[str | None, str | None]:
    runtime_lookup = field.get("runtime_lookup") or {}
    candidate_id = str(runtime_lookup.get("candidate_id") or field.get("candidate_id") or "")
    index = runtime_lookup.get("index")
    if not candidate_id or not isinstance(index, int):
        return None, "runtime_lookup_incomplete"
    for result in step8_result.get("candidate_structure_results") or []:
        if not isinstance(result, dict) or str(result.get("candidate_id") or "") != candidate_id:
            continue
        refs = result.get("complex_structure_refs") or []
        if index >= len(refs):
            return None, "complex_ref_index_out_of_range"
        ref = refs[index]
        if not isinstance(ref, dict):
            return None, "complex_ref_malformed"
        storage_ref = ref.get("storage_ref")
        if isinstance(storage_ref, str) and _path_exists(storage, storage_ref):
            return storage_ref, None
        source_ref = ref.get("source_ref")
        if isinstance(source_ref, str) and source_ref.strip():
            return _resolve_structure_hint(
                source_ref.strip(),
                candidate_id=candidate_id,
                prepared_inputs=prepared_inputs,
                candidate_context_table=candidate_context_table,
                storage=storage,
            )
        return None, "complex_ref_not_resolvable"
    return None, "candidate_structure_result_not_found"


# ── Step 8 downstream_handoff.{validated_structure_ref,
#    structure_for_variant_generation_ref} ──────────────────────────────────
#
# These handoff values may be a material_id or a storage path, never a raw
# structure body (the projection layer already filters raw bodies out at
# `_project_step8_fields` time). Real resolution walks Step 7 structure_refs
# first (matching `source_ref`), then Step 5 candidate materials (matching
# `material_id`) — never passes the compact field_ref or a bare material_id
# straight through to ToolUniverse.

def _resolve_step8_handoff_structure_ref(
    field: dict[str, Any],
    *,
    handoff_key: str,
    step8_result: dict[str, Any],
    prepared_inputs: list[dict[str, Any]],
    candidate_context_table: dict[str, Any],
    storage: Storage,
) -> tuple[str | None, str | None]:
    runtime_lookup = field.get("runtime_lookup") or {}
    candidate_id = str(runtime_lookup.get("candidate_id") or field.get("candidate_id") or "")
    if not candidate_id:
        return None, "runtime_lookup_incomplete"
    for result in step8_result.get("candidate_structure_results") or []:
        if not isinstance(result, dict) or str(result.get("candidate_id") or "") != candidate_id:
            continue
        handoff = result.get("downstream_handoff")
        if not isinstance(handoff, dict):
            return None, "downstream_handoff_missing"
        raw_hint = handoff.get(handoff_key)
        if not isinstance(raw_hint, str) or not raw_hint.strip():
            return None, "handoff_value_missing"
        return _resolve_structure_hint(
            raw_hint.strip(),
            candidate_id=candidate_id,
            prepared_inputs=prepared_inputs,
            candidate_context_table=candidate_context_table,
            storage=storage,
        )
    return None, "candidate_structure_result_not_found"


def _resolve_structure_hint(
    hint: str,
    *,
    candidate_id: str,
    prepared_inputs: list[dict[str, Any]],
    candidate_context_table: dict[str, Any],
    storage: Storage,
) -> tuple[str | None, str | None]:
    """Resolve a structure "hint" — already a usable storage path, OR a
    Step 7 `structure_refs[].source_ref`, OR a Step 5 `materials[].material_id`
    — to a real, storage-verified path. Never returns a bare identifier/path
    string that hasn't been checked to actually resolve."""

    if _looks_like_raw_structure_body(hint):
        return None, "structure_hint_is_raw_body_not_path"
    if _path_exists(storage, hint):
        return hint, None

    for sin in prepared_inputs:
        if str(sin.get("candidate_id") or "") != candidate_id:
            continue
        for sref in sin.get("structure_refs") or []:
            if not isinstance(sref, dict):
                continue
            if str(sref.get("source_ref") or "") == hint:
                storage_ref = sref.get("storage_ref")
                if isinstance(storage_ref, str) and _path_exists(storage, storage_ref):
                    return storage_ref, None

    for candidate in candidate_context_table.get("candidate_records") or []:
        if not isinstance(candidate, dict) or str(candidate.get("candidate_id") or "") != candidate_id:
            continue
        for material in candidate.get("materials") or []:
            if not isinstance(material, dict):
                continue
            if str(material.get("material_id") or "") == hint:
                value = material.get("value")
                if isinstance(value, str) and _path_exists(storage, value):
                    return value, None

    return None, "structure_ref_not_resolvable"


# ── Step 7 sequence_refs_for_prediction[] (raw sequence) ────────────────────
#
# `SequenceRef.sequence` is excluded from the persisted Step 7 artifact by
# schema design (privacy), so the real amino-acid text is never present in
# `prepared_structure_input_package.json` even for an "inline" sequence.
# Real resolution always falls back to the Step 5 material the sequence
# ref's `source_ref` names (`source_kind="material_sequence"`), or reads an
# uploaded FASTA file through storage for `source_kind="uploaded_fasta"`.

def _resolve_step7_sequence(
    field: dict[str, Any],
    *,
    prepared_inputs: list[dict[str, Any]],
    candidate_context_table: dict[str, Any],
    storage: Storage,
) -> tuple[str | None, str | None]:
    runtime_lookup = field.get("runtime_lookup") or {}
    candidate_id = str(runtime_lookup.get("candidate_id") or field.get("candidate_id") or "")
    sequence_id = str(runtime_lookup.get("sequence_id") or "")
    if not candidate_id or not sequence_id:
        return None, "runtime_lookup_incomplete"

    for sin in prepared_inputs:
        if str(sin.get("candidate_id") or "") != candidate_id:
            continue
        for seq_ref in sin.get("sequence_refs_for_prediction") or []:
            if not isinstance(seq_ref, dict) or str(seq_ref.get("sequence_id") or "") != sequence_id:
                continue
            kind = seq_ref.get("prediction_input_kind")
            status = seq_ref.get("sequence_value_status")
            if kind == "amino_acid_sequence" and status == "inline":
                source_ref = seq_ref.get("source_ref")
                if isinstance(source_ref, str) and source_ref:
                    value = _material_value_by_id(candidate_context_table, candidate_id, source_ref)
                    if value:
                        return value, None
                return None, "inline_sequence_source_material_missing"
            if kind == "fasta_ref":
                path = seq_ref.get("sequence_storage_ref") or seq_ref.get("source_ref")
                if not isinstance(path, str) or not path.strip():
                    return None, "fasta_ref_path_missing"
                try:
                    content = storage.read_bytes(path).decode("utf-8")
                except Exception as exc:  # noqa: BLE001
                    return None, f"fasta_ref_read_failed:{type(exc).__name__}"
                sequences = _extract_fasta_sequences(content)
                if not sequences:
                    return None, "fasta_ref_no_sequence"
                return sequences[0], None
            return None, "identifier_only_sequence_not_runtime_resolvable"
    return None, "sequence_ref_not_found"


# ── Step 5 candidate materials (variant/contigs/inline protein sequence) ────

def _resolve_material_value(
    field: dict[str, Any], *, candidate_context_table: dict[str, Any]
) -> tuple[str | None, str | None]:
    runtime_lookup = field.get("runtime_lookup") or {}
    candidate_id = str(runtime_lookup.get("candidate_id") or field.get("candidate_id") or "")
    material_id = str(runtime_lookup.get("material_id") or "")
    if not candidate_id or not material_id:
        return None, "runtime_lookup_incomplete"
    value = _material_value_by_id(candidate_context_table, candidate_id, material_id)
    if value is None:
        return None, "material_not_found"
    return value, None


def _material_value_by_id(
    candidate_context_table: dict[str, Any], candidate_id: str, material_id: str
) -> str | None:
    for candidate in candidate_context_table.get("candidate_records") or []:
        if not isinstance(candidate, dict) or str(candidate.get("candidate_id") or "") != candidate_id:
            continue
        for material in candidate.get("materials") or []:
            if not isinstance(material, dict):
                continue
            if str(material.get("material_id") or "") != material_id:
                continue
            value = material.get("value")
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


# ── shared helpers ───────────────────────────────────────────────────────────

def _prepared_inputs_list(
    prepared_structure_input_package: dict[str, Any] | list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    if isinstance(prepared_structure_input_package, dict):
        inputs = prepared_structure_input_package.get("prepared_structure_inputs") or []
    else:
        inputs = prepared_structure_input_package or []
    return [item for item in inputs if isinstance(item, dict)]


def _path_exists(storage: Storage, value: Any) -> bool:
    if not isinstance(value, str) or not value.strip():
        return False
    try:
        if Path(value).is_file():
            return True
        return storage.exists(value)
    except Exception:  # noqa: BLE001
        return False


def _looks_like_pdb_id(value: str) -> bool:
    stripped = value.strip()
    if len(stripped) != 4 or not stripped[0].isdigit():
        return False
    return stripped.isalnum()


def _looks_like_raw_structure_body(value: str) -> bool:
    if len(value) > 500:
        return True
    upper = value[:500].upper()
    if upper.startswith(("ATOM", "HETATM", "HEADER")):
        return True
    if any(marker in upper for marker in ("\nATOM", "\nHETATM", "HEADER ", "\nHEADER")):
        return True
    lower = value[:500].lower()
    return "data_" in lower or "loop_" in lower


def _extract_fasta_sequences(content: str) -> list[str]:
    sequences: list[str] = []
    current: list[str] = []
    for line in content.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith(">"):
            if current:
                sequences.append("".join(current))
                current = []
            continue
        current.append(stripped)
    if current:
        sequences.append("".join(current))
    return sequences


def _redacted_summary_for_field(field: dict[str, Any], value: str) -> dict[str, Any]:
    """LLM-safe / audit-safe digest of a resolved field_ref value — never
    the raw value itself, only field identity + length/hash fingerprint."""
    text = str(value)
    return {
        "source": "field_ref",
        "field_ref": field.get("field_ref"),
        "field_type": field.get("field_type"),
        "value_kind": field.get("value_kind"),
        "value_length": len(text),
        "sha256_prefix": hashlib.sha256(text.encode("utf-8")).hexdigest()[:12],
    }


def _model_dump_list(items: list[Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for item in items:
        if hasattr(item, "model_dump"):
            out.append(item.model_dump())
        elif isinstance(item, dict):
            out.append(dict(item))
    return out
