"""Step 9 centralized deterministic input projection.

Step 9's inputs are not only Step 8 — per the workflow design
(`ADC_AOC_Agent_Pipeline_Workflow_v0.1.md`, Step 9), they are predicted
structures (Step 8), candidate records (Step 5), PDB backbone / structure
refs (Step 7), sequence context (Step 5/7), and mutation/variant
constraints (Step 5).

This module is the ONLY place that understands the raw Step 5
`candidate_context_table`, Step 7 `prepared_structure_input_package`, Step 8
`structure_prediction_and_interface_results`, and Step 2 `structured_query` /
raw request shapes for Step 9. `Stage1`, `Stage2`, and the runtime planner
consume only the compact `Step9InputField` list (and the accompanying
summaries) produced here — never the upstream artifacts directly.

Hard privacy constraint: no raw protein sequence, raw PDB/CIF body, raw A3M
alignment, storage path, or API key is ever placed in an `Step9InputField`,
`candidate_summaries`, `handoff_summary`, or `query_summary` value. Fields
that need to resolve to a real value at execution time carry a `runtime_lookup`
description (schema path + candidate id), never the value itself.
"""

from __future__ import annotations

import re
from typing import Any

from ..schemas.step_09_structure_variant_and_compound_screening import Step9InputField


_PDB_ID_RE = re.compile(r"[0-9][A-Za-z0-9]{3}")

_UNIPROT_IDENTIFIER_TYPES = {"uniprot_id", "uniprot"}
_VARIANT_IDENTIFIER_TYPES = {"protein_variant", "variant", "variant_sequence", "mutation"}
_CHAIN_IDENTIFIER_TYPES = {"chain_id", "chain"}
_PROTEIN_SEQUENCE_MATERIAL_TYPES = {
    "target_sequence",
    "target_antigen_sequence",
    "antibody_heavy_chain_sequence",
    "antibody_light_chain_sequence",
}
_CONTIG_MATERIAL_TYPES = {"design_contigs", "contigs"}

_STRUCTURE_TOOL_ARGS = ["input_pdb", "pdb_file", "structure", "backbone", "path"]
_COMPLEX_STRUCTURE_TOOL_ARGS = ["input_pdb", "pdb_file", "structure", "complex_structure", "backbone"]
_SEQUENCE_TOOL_ARGS = ["sequence", "prompt_sequence"]
_UNIPROT_TOOL_ARGS = ["uniprot_id", "accession", "uniprot_accession"]
_VARIANT_TOOL_ARGS = ["variant", "variants", "mutation", "mutations"]
_CHAIN_TOOL_ARGS = ["chain", "chain_id"]
_CONTIGS_TOOL_ARGS = ["contigs"]
_PDB_ID_TOOL_ARGS = ["pdb_id"]

_QUERY_SUMMARY_MAX_LEN = 300


def project_step9_inputs(
    *,
    candidate_context_table: dict[str, Any] | None,
    prepared_structure_input_package: dict[str, Any] | list[dict[str, Any]] | None,
    structure_prediction_and_interface_results: dict[str, Any] | None,
    structured_query: dict[str, Any] | None = None,
    raw_request: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Project Step 5 / Step 7 / Step 8 / query artifacts into a single,
    LLM-safe Step 9 input projection.

    Returns a dict with:
    - ``input_fields``: ``list[Step9InputField]``
    - ``candidate_summaries``: compact per-candidate overview
    - ``handoff_summary``: compact Step 8 downstream-handoff overview
    - ``missing_inputs``: compact list of gap strings
    - ``query_summary``: redacted canonical/raw query summary
    """
    candidate_context_table = candidate_context_table or {}
    structured_query = structured_query or {}
    raw_request = raw_request or {}

    if isinstance(prepared_structure_input_package, dict):
        prepared_inputs = prepared_structure_input_package.get("prepared_structure_inputs") or []
    else:
        prepared_inputs = prepared_structure_input_package or []
    if not isinstance(prepared_inputs, list):
        prepared_inputs = []

    step8_result = structure_prediction_and_interface_results or {}

    candidate_records = candidate_context_table.get("candidate_records") or []
    if not isinstance(candidate_records, list):
        candidate_records = []

    input_fields: list[Step9InputField] = []
    candidate_summaries: list[dict[str, Any]] = []
    handoff_candidates: list[dict[str, Any]] = []
    missing_inputs: list[str] = []

    for candidate in candidate_records:
        if not isinstance(candidate, dict):
            continue
        if candidate.get("candidate_type") == "compound_component":
            # Step 9 active lanes are protein_design / variant_evaluation only.
            continue
        candidate_id = str(candidate.get("candidate_id") or "")
        if not candidate_id:
            continue

        fields = _project_step5_candidate_fields(candidate, candidate_id)
        fields.extend(_project_step7_fields(prepared_inputs, candidate_id))
        fields.extend(_project_step8_fields(step8_result, candidate_id))
        input_fields.extend(fields)

        field_types_present = sorted({f.field_type for f in fields})
        candidate_summaries.append(
            {
                "candidate_id": candidate_id,
                "candidate_type": str(candidate.get("candidate_type") or "unknown"),
                "field_count": len(fields),
                "field_types_present": field_types_present,
            }
        )

        handoff = _step8_handoff_for_candidate(step8_result, candidate_id)
        handoff_candidates.append({"candidate_id": candidate_id, **handoff})

        if not ({"structure", "complex_structure", "structure_identifier", "protein_sequence"} & set(field_types_present)):
            missing_inputs.append(f"{candidate_id}:missing_structure_or_sequence_input")

    query_field, query_summary = _project_query_field(structured_query, raw_request)
    input_fields.append(query_field)

    return {
        "input_fields": _merge_duplicate_field_refs(input_fields),
        "candidate_summaries": candidate_summaries,
        "handoff_summary": {"candidates": handoff_candidates},
        "missing_inputs": list(dict.fromkeys(missing_inputs)),
        "query_summary": query_summary,
    }


# ── field_ref uniqueness contract ────────────────────────────────────────────
#
# `input_fields[].field_ref` is the projection's single global-identity key —
# Stage 2 and the runtime planner both key lookups off it. Several
# independent construction paths above can legitimately compute the same
# `field_ref` for the same real-world entity (e.g. the same UniProt accession
# surfaced by a Step 5 candidate identifier AND a Step 7 identifier-only
# sequence ref; the same PDB id shared by two candidates' Step 8 complex
# results). Rather than let a later list entry silently win a dict-keyed
# lookup downstream, this module merges every group of same-`field_ref`
# fields into one canonical field before returning, preserving every
# contributing source/candidate/metadata.

class DuplicateStep9InputFieldError(ValueError):
    """Raised when a Step9InputProjection.input_fields list contains more
    than one entry for the same `field_ref`.

    `project_step9_inputs` itself never returns duplicates (see
    `_merge_duplicate_field_refs`); this is a defense-in-depth guard for
    Stage 2 / the runtime planner in case a caller passes a hand-built or
    otherwise non-canonical `input_fields` list. Failing fast beats silently
    keeping whichever entry happens to appear last in the list.
    """


def assert_unique_input_field_refs(fields: list[dict[str, Any]]) -> None:
    seen: set[str] = set()
    duplicates: list[str] = []
    for field in fields:
        ref = str((field or {}).get("field_ref") or "")
        if not ref:
            continue
        if ref in seen and ref not in duplicates:
            duplicates.append(ref)
        seen.add(ref)
    if duplicates:
        raise DuplicateStep9InputFieldError(
            "duplicate Step9InputField field_ref(s): " + ", ".join(sorted(duplicates))
        )


def _merge_duplicate_field_refs(fields: list[Step9InputField]) -> list[Step9InputField]:
    groups: dict[str, list[Step9InputField]] = {}
    order: list[str] = []
    for field in fields:
        if field.field_ref not in groups:
            groups[field.field_ref] = []
            order.append(field.field_ref)
        groups[field.field_ref].append(field)
    return [_merge_field_group(groups[ref]) for ref in order]


def _merge_field_group(group: list[Step9InputField]) -> Step9InputField:
    """Merge one or more `Step9InputField`s that share a `field_ref` into a
    single canonical field. Always runs (even for a group of size 1) so
    every returned field has the same normalized `source_steps` /
    `candidate_ids` shape, and so provenance (`runtime_lookup.sources`) is
    populated uniformly."""

    base = group[0]
    candidate_ids = list(dict.fromkeys(f.candidate_id for f in group if f.candidate_id))
    source_steps = list(dict.fromkeys(f.source_step for f in group))
    supports_tool_args = list(dict.fromkeys(arg for f in group for arg in f.supports_tool_args))
    chain_role = next((f.chain_role for f in group if f.chain_role), None)
    semantic_role = next((f.semantic_role for f in group if f.semantic_role), None)
    can_resolve_at_runtime = any(f.can_resolve_at_runtime for f in group)
    status = "available" if any(f.status == "available" for f in group) else base.status
    missing_reason = None if status == "available" else next(
        (f.missing_reason for f in group if f.missing_reason), None
    )

    llm_safe_metadata: dict[str, Any] = {}
    for f in group:
        llm_safe_metadata.update(f.llm_safe_metadata or {})

    runtime_lookup: dict[str, Any] = {}
    for f in group:
        if isinstance(f.runtime_lookup, dict):
            runtime_lookup.update({k: v for k, v in f.runtime_lookup.items() if k != "sources"})
    runtime_lookup["sources"] = [
        {
            "source_step": f.source_step,
            "source_artifact": f.source_artifact,
            "source_path": f.source_path,
            "candidate_id": f.candidate_id,
        }
        for f in group
    ]

    return Step9InputField(
        field_ref=base.field_ref,
        candidate_id=candidate_ids[0] if candidate_ids else None,
        candidate_ids=candidate_ids,
        source_step=source_steps[0],
        source_steps=source_steps,
        source_artifact=base.source_artifact,
        source_path=base.source_path,
        field_name=base.field_name,
        field_type=base.field_type,
        value_kind=base.value_kind,
        semantic_role=semantic_role,
        chain_role=chain_role,
        supports_tool_args=supports_tool_args,
        can_resolve_at_runtime=can_resolve_at_runtime,
        llm_safe_metadata=llm_safe_metadata,
        runtime_lookup=runtime_lookup,
        status=status,
        missing_reason=missing_reason,
    )


# ── Step 5 candidate records ────────────────────────────────────────────────

def _project_step5_candidate_fields(candidate: dict[str, Any], candidate_id: str) -> list[Step9InputField]:
    out: list[Step9InputField] = []

    for ident in candidate.get("identifiers") or []:
        if not isinstance(ident, dict):
            continue
        id_type = str(ident.get("id_type") or "").lower()
        value = ident.get("id_value")
        if not isinstance(value, str) or not value.strip():
            continue
        value = value.strip()

        if id_type in _UNIPROT_IDENTIFIER_TYPES:
            out.append(
                Step9InputField(
                    field_ref=f"identifier:{id_type}:{value}",
                    candidate_id=candidate_id,
                    source_step="step_05",
                    source_artifact="candidate_context_table",
                    source_path="candidate_records[].identifiers[]",
                    field_name=id_type,
                    field_type="identifier",
                    value_kind="uniprot_id",
                    semantic_role="target_identity",
                    supports_tool_args=list(_UNIPROT_TOOL_ARGS),
                    can_resolve_at_runtime=True,
                    llm_safe_metadata={"id_type": id_type},
                    status="available",
                )
            )
        elif id_type in _VARIANT_IDENTIFIER_TYPES:
            value_kind = "mutation" if id_type == "mutation" else "variant"
            out.append(
                Step9InputField(
                    field_ref=f"identifier:{id_type}:{value}",
                    candidate_id=candidate_id,
                    source_step="step_05",
                    source_artifact="candidate_context_table",
                    source_path="candidate_records[].identifiers[]",
                    field_name=id_type,
                    field_type="variant",
                    value_kind=value_kind,
                    supports_tool_args=list(_VARIANT_TOOL_ARGS),
                    can_resolve_at_runtime=True,
                    llm_safe_metadata={"id_type": id_type},
                    status="available",
                )
            )
        elif id_type in _CHAIN_IDENTIFIER_TYPES:
            out.append(
                Step9InputField(
                    field_ref=f"identifier:{id_type}:{value}",
                    candidate_id=candidate_id,
                    source_step="step_05",
                    source_artifact="candidate_context_table",
                    source_path="candidate_records[].identifiers[]",
                    field_name=id_type,
                    field_type="chain",
                    value_kind="chain_id",
                    chain_role=None,
                    supports_tool_args=list(_CHAIN_TOOL_ARGS),
                    can_resolve_at_runtime=True,
                    llm_safe_metadata={"id_type": id_type},
                    status="available",
                )
            )
        elif id_type == "pdb_id" and _looks_like_pdb_id(value):
            out.append(
                Step9InputField(
                    field_ref=f"identifier:pdb_id:{value}",
                    candidate_id=candidate_id,
                    source_step="step_05",
                    source_artifact="candidate_context_table",
                    source_path="candidate_records[].identifiers[]",
                    field_name="pdb_id",
                    field_type="structure_identifier",
                    value_kind="pdb_id",
                    supports_tool_args=list(_PDB_ID_TOOL_ARGS),
                    can_resolve_at_runtime=True,
                    llm_safe_metadata={},
                    status="available",
                )
            )

    for material in candidate.get("materials") or []:
        if not isinstance(material, dict):
            continue
        m_type = material.get("material_type")
        mat_id = material.get("material_id")
        if not isinstance(mat_id, str) or not mat_id:
            continue
        value = material.get("value")

        if m_type in _VARIANT_IDENTIFIER_TYPES:
            if isinstance(value, str) and value.strip():
                out.append(
                    Step9InputField(
                        field_ref=f"material:{mat_id}",
                        candidate_id=candidate_id,
                        source_step="step_05",
                        source_artifact="candidate_context_table",
                        source_path="candidate_records[].materials[]",
                        field_name=str(m_type),
                        field_type="variant",
                        value_kind="mutation" if m_type == "mutation" else "variant",
                        supports_tool_args=list(_VARIANT_TOOL_ARGS),
                        can_resolve_at_runtime=True,
                        llm_safe_metadata={"material_type": str(m_type)},
                        status="available",
                    )
                )
        elif m_type in _CONTIG_MATERIAL_TYPES:
            has_value = isinstance(value, str) and bool(value.strip())
            out.append(
                Step9InputField(
                    field_ref=f"material:{mat_id}",
                    candidate_id=candidate_id,
                    source_step="step_05",
                    source_artifact="candidate_context_table",
                    source_path="candidate_records[].materials[]",
                    field_name=str(m_type),
                    field_type="design_constraint",
                    value_kind="contigs",
                    supports_tool_args=list(_CONTIGS_TOOL_ARGS),
                    can_resolve_at_runtime=has_value,
                    llm_safe_metadata={"material_type": str(m_type)},
                    status="available" if has_value else "missing",
                    missing_reason=None if has_value else "contigs_value_missing",
                )
            )
        elif m_type in _PROTEIN_SEQUENCE_MATERIAL_TYPES:
            has_value = isinstance(value, str) and bool(value.strip())
            out.append(
                Step9InputField(
                    field_ref=f"material:{mat_id}",
                    candidate_id=candidate_id,
                    source_step="step_05",
                    source_artifact="candidate_context_table",
                    source_path="candidate_records[].materials[]",
                    field_name=str(m_type),
                    field_type="protein_sequence",
                    value_kind="sequence_ref",
                    chain_role=_chain_role_from_material_type(str(m_type)),
                    supports_tool_args=list(_SEQUENCE_TOOL_ARGS) if has_value else [],
                    can_resolve_at_runtime=has_value,
                    llm_safe_metadata={
                        "material_type": str(m_type),
                        "value_length": len(value) if has_value else 0,
                    },
                    runtime_lookup=(
                        {
                            "resolution_path": ["step_05.candidate_records[].materials[].value"],
                            "candidate_id": candidate_id,
                            "material_id": mat_id,
                        }
                        if has_value
                        else {}
                    ),
                    status="available" if has_value else "missing",
                    missing_reason=None if has_value else "sequence_value_missing",
                )
            )

    return out


def _chain_role_from_material_type(material_type: str) -> str | None:
    if material_type in {"target_sequence", "target_antigen_sequence"}:
        return "antigen"
    if material_type == "antibody_heavy_chain_sequence":
        return "antibody_heavy"
    if material_type == "antibody_light_chain_sequence":
        return "antibody_light"
    return None


# ── Step 7 prepared structure inputs ────────────────────────────────────────

def _project_step7_fields(prepared_inputs: list[dict[str, Any]], candidate_id: str) -> list[Step9InputField]:
    out: list[Step9InputField] = []
    for sin in prepared_inputs:
        if not isinstance(sin, dict):
            continue
        if str(sin.get("candidate_id") or "") != candidate_id:
            continue
        structure_input_id = str(sin.get("structure_input_id") or "")

        for i, sref in enumerate(sin.get("structure_refs") or []):
            if not isinstance(sref, dict):
                continue
            storage_ref = sref.get("storage_ref")
            pdb_id = sref.get("pdb_id")
            if isinstance(storage_ref, str) and storage_ref.strip():
                value_kind = "pdb_file_ref" if str(sref.get("source_kind") or "") == "uploaded_file" else "structure_ref"
                out.append(
                    Step9InputField(
                        field_ref=f"step7_structure_ref:{structure_input_id}:{i}",
                        candidate_id=candidate_id,
                        source_step="step_07",
                        source_artifact="prepared_structure_input_package",
                        source_path="prepared_structure_inputs[].structure_refs[].storage_ref",
                        field_name="structure_ref",
                        field_type="structure",
                        value_kind=value_kind,
                        supports_tool_args=list(_STRUCTURE_TOOL_ARGS),
                        can_resolve_at_runtime=True,
                        llm_safe_metadata={"structure_format": sref.get("structure_format")},
                        runtime_lookup={
                            "resolution_path": ["step_07.prepared_structure_inputs[].structure_refs[].storage_ref"],
                            "candidate_id": candidate_id,
                            "structure_input_id": structure_input_id,
                            "index": i,
                        },
                        status="available",
                    )
                )
            if isinstance(pdb_id, str) and _looks_like_pdb_id(pdb_id):
                out.append(
                    Step9InputField(
                        field_ref=f"identifier:pdb_id:{pdb_id.strip()}",
                        candidate_id=candidate_id,
                        source_step="step_07",
                        source_artifact="prepared_structure_input_package",
                        source_path="prepared_structure_inputs[].structure_refs[].pdb_id",
                        field_name="pdb_id",
                        field_type="structure_identifier",
                        value_kind="pdb_id",
                        supports_tool_args=list(_PDB_ID_TOOL_ARGS),
                        can_resolve_at_runtime=True,
                        llm_safe_metadata={},
                        status="available",
                    )
                )

        for seq_ref in sin.get("sequence_refs_for_prediction") or []:
            if not isinstance(seq_ref, dict):
                continue
            sequence_id = str(seq_ref.get("sequence_id") or "")
            value_status = str(seq_ref.get("sequence_value_status") or "").lower()
            if value_status in {"inline", "referenced"}:
                out.append(
                    Step9InputField(
                        field_ref=f"step7_sequence:{sequence_id}",
                        candidate_id=candidate_id,
                        source_step="step_07",
                        source_artifact="prepared_structure_input_package",
                        source_path="prepared_structure_inputs[].sequence_refs_for_prediction[]",
                        field_name="sequence_ref",
                        field_type="protein_sequence",
                        value_kind="sequence_ref",
                        chain_role=seq_ref.get("chain_role"),
                        supports_tool_args=list(_SEQUENCE_TOOL_ARGS),
                        can_resolve_at_runtime=True,
                        llm_safe_metadata={"sequence_length": seq_ref.get("sequence_length")},
                        runtime_lookup={
                            "resolution_path": [
                                "step_07.prepared_structure_inputs[].sequence_refs_for_prediction[]"
                            ],
                            "candidate_id": candidate_id,
                            "sequence_id": sequence_id,
                        },
                        status="available",
                    )
                )
            elif value_status == "identifier_only" and str(seq_ref.get("prediction_input_kind") or "") == "uniprot_id":
                source_ref = seq_ref.get("source_ref")
                if isinstance(source_ref, str) and source_ref.strip():
                    out.append(
                        Step9InputField(
                            field_ref=f"identifier:uniprot_id:{source_ref.strip()}",
                            candidate_id=candidate_id,
                            source_step="step_07",
                            source_artifact="prepared_structure_input_package",
                            source_path="prepared_structure_inputs[].sequence_refs_for_prediction[].source_ref",
                            field_name="uniprot_id",
                            field_type="identifier",
                            value_kind="uniprot_id",
                            chain_role=seq_ref.get("chain_role"),
                            supports_tool_args=list(_UNIPROT_TOOL_ARGS),
                            can_resolve_at_runtime=True,
                            llm_safe_metadata={},
                            status="available",
                        )
                    )
            # MSA refs stay compact-only upstream; Step 9 tools never need
            # raw A3M, so they are intentionally not projected here.
    return out


# ── Step 8 structure prediction / interface results ─────────────────────────

def _step8_candidate_results(step8_result: dict[str, Any], candidate_id: str) -> list[dict[str, Any]]:
    return [
        item
        for item in (step8_result.get("candidate_structure_results") or [])
        if isinstance(item, dict) and str(item.get("candidate_id") or "") == candidate_id
    ]


def _project_step8_fields(step8_result: dict[str, Any], candidate_id: str) -> list[Step9InputField]:
    out: list[Step9InputField] = []
    results = _step8_candidate_results(step8_result, candidate_id)

    for result in results:
        for i, ref in enumerate(result.get("complex_structure_refs") or []):
            if not isinstance(ref, dict):
                continue
            source_kind = str(ref.get("source_kind") or "").lower()
            if source_kind not in {"existing_pdb_complex", "predicted_complex", "uploaded_local_complex"}:
                continue
            has_ref = bool(str(ref.get("storage_ref") or ref.get("pdb_id") or ref.get("source_ref") or "").strip())
            out.append(
                Step9InputField(
                    field_ref=f"step8_complex_ref:{candidate_id}:{i}",
                    candidate_id=candidate_id,
                    source_step="step_08",
                    source_artifact="structure_prediction_and_interface_results",
                    source_path="candidate_structure_results[].complex_structure_refs[]",
                    field_name="complex_structure_ref",
                    field_type="complex_structure",
                    value_kind="complex_structure_ref",
                    supports_tool_args=list(_COMPLEX_STRUCTURE_TOOL_ARGS),
                    can_resolve_at_runtime=has_ref,
                    llm_safe_metadata={"source_kind": source_kind},
                    runtime_lookup=(
                        {
                            "resolution_path": [
                                "step_08.candidate_structure_results[].complex_structure_refs[]"
                            ],
                            "candidate_id": candidate_id,
                            "index": i,
                        }
                        if has_ref
                        else {}
                    ),
                    status="available" if has_ref else "missing",
                    missing_reason=None if has_ref else "complex_structure_ref_missing",
                )
            )

            pdb_id = None
            if source_kind == "existing_pdb_complex":
                for key in ("pdb_id", "source_ref"):
                    candidate_value = ref.get(key)
                    if isinstance(candidate_value, str) and _looks_like_pdb_id(candidate_value):
                        pdb_id = candidate_value.strip()
                        break
            if pdb_id:
                out.append(
                    Step9InputField(
                        field_ref=f"identifier:pdb_id:{pdb_id}",
                        candidate_id=candidate_id,
                        source_step="step_08",
                        source_artifact="structure_prediction_and_interface_results",
                        source_path="candidate_structure_results[].complex_structure_refs[].pdb_id",
                        field_name="pdb_id",
                        field_type="structure_identifier",
                        value_kind="pdb_id",
                        supports_tool_args=list(_PDB_ID_TOOL_ARGS),
                        can_resolve_at_runtime=True,
                        llm_safe_metadata={},
                        status="available",
                    )
                )

        handoff = result.get("downstream_handoff")
        if not isinstance(handoff, dict):
            continue

        validated_ref = handoff.get("validated_structure_ref")
        if isinstance(validated_ref, str) and validated_ref.strip() and not _looks_like_raw_structure_body(validated_ref):
            out.append(
                Step9InputField(
                    field_ref=f"step8_validated_structure_ref:{candidate_id}",
                    candidate_id=candidate_id,
                    source_step="step_08",
                    source_artifact="structure_prediction_and_interface_results",
                    source_path="candidate_structure_results[].downstream_handoff.validated_structure_ref",
                    field_name="validated_structure_ref",
                    field_type="structure",
                    value_kind="validated_structure_ref",
                    supports_tool_args=list(_STRUCTURE_TOOL_ARGS),
                    can_resolve_at_runtime=True,
                    llm_safe_metadata={},
                    # `validated_structure_ref` may be a material_id or a
                    # storage path, not a usable path by itself — the actual
                    # value is NEVER copied here. Real resolution must walk
                    # back through Step 7 / Step 5 / storage at execution time.
                    runtime_lookup={
                        "resolution_path": [
                            "step_08.candidate_structure_results[].downstream_handoff.validated_structure_ref",
                            "step_07.prepared_structure_inputs[].structure_refs[].storage_ref",
                            "step_05.candidate_records[].materials[]",
                        ],
                        "candidate_id": candidate_id,
                    },
                    status="available",
                )
            )

        variant_ref = handoff.get("structure_for_variant_generation_ref")
        if isinstance(variant_ref, str) and variant_ref.strip() and not _looks_like_raw_structure_body(variant_ref):
            out.append(
                Step9InputField(
                    field_ref=f"step8_variant_structure_ref:{candidate_id}",
                    candidate_id=candidate_id,
                    source_step="step_08",
                    source_artifact="structure_prediction_and_interface_results",
                    source_path="candidate_structure_results[].downstream_handoff.structure_for_variant_generation_ref",
                    field_name="structure_for_variant_generation_ref",
                    field_type="structure",
                    value_kind="structure_ref",
                    supports_tool_args=list(_STRUCTURE_TOOL_ARGS),
                    can_resolve_at_runtime=True,
                    llm_safe_metadata={},
                    runtime_lookup={
                        "resolution_path": [
                            "step_08.candidate_structure_results[].downstream_handoff.structure_for_variant_generation_ref",
                            "step_07.prepared_structure_inputs[].structure_refs[].storage_ref",
                            "step_05.candidate_records[].materials[]",
                        ],
                        "candidate_id": candidate_id,
                    },
                    status="available",
                )
            )

    return out


def _step8_handoff_for_candidate(step8_result: dict[str, Any], candidate_id: str) -> dict[str, Any]:
    for result in _step8_candidate_results(step8_result, candidate_id):
        handoff = result.get("downstream_handoff")
        if not isinstance(handoff, dict):
            continue
        return {
            "has_complex_structure": bool(handoff.get("has_complex_structure")),
            "has_validated_structure": bool(handoff.get("has_validated_structure")),
            "has_interface_features": bool(handoff.get("has_interface_features")),
            "interface_quality_available": bool(handoff.get("interface_quality_available")),
            "prediction_confidence_available": bool(handoff.get("prediction_confidence_available")),
            "refinement_resolution_available": bool(handoff.get("refinement_resolution_available")),
            "validation_available": bool(handoff.get("validation_available")),
            "missing_for_step9": list(handoff.get("missing_for_step9") or []),
        }
    return {
        "has_complex_structure": False,
        "has_validated_structure": False,
        "has_interface_features": False,
        "interface_quality_available": False,
        "prediction_confidence_available": False,
        "refinement_resolution_available": False,
        "validation_available": False,
        "missing_for_step9": [],
    }


# ── Query (Step 2) ───────────────────────────────────────────────────────────

def _project_query_field(
    structured_query: dict[str, Any], raw_request: dict[str, Any]
) -> tuple[Step9InputField, dict[str, Any]]:
    canonical = structured_query.get("canonical_query")
    raw_query = raw_request.get("raw_user_query")
    text = canonical if isinstance(canonical, str) and canonical.strip() else raw_query
    source_path = (
        "structured_query.canonical_query"
        if isinstance(canonical, str) and canonical.strip()
        else "raw_request_record.raw_user_query"
    )
    compact = _compact_text(text) if isinstance(text, str) else ""

    field = Step9InputField(
        field_ref="query:summary",
        candidate_id=None,
        source_step="query",
        source_artifact=(
            "structured_query" if source_path.startswith("structured_query") else "raw_request_record"
        ),
        source_path=source_path,
        field_name="canonical_query" if source_path.startswith("structured_query") else "raw_user_query",
        field_type="query_context",
        value_kind="query_summary",
        semantic_role="task_intent_context",
        supports_tool_args=[],
        can_resolve_at_runtime=False,
        llm_safe_metadata={"summary": compact} if compact else {},
        status="available" if compact else "missing",
        missing_reason=None if compact else "no_query_text",
    )
    summary = {
        "canonical_query": _compact_text(canonical) if isinstance(canonical, str) else "",
        "raw_user_query": _compact_text(raw_query) if isinstance(raw_query, str) else "",
    }
    return field, summary


# ── shared safety helpers ────────────────────────────────────────────────────

def _looks_like_pdb_id(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    stripped = value.strip()
    if not stripped or len(stripped) != 4:
        return False
    return bool(_PDB_ID_RE.fullmatch(stripped))


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


def _compact_text(value: str) -> str:
    """Redact raw sequences / structure bodies / API keys from free text.

    Shared by the projection layer and the selector prompts so both stay
    byte-identical in what they consider safe to surface to an LLM.
    """
    text = " ".join(str(value or "").split())
    text = re.sub(
        r"\b[ACDEFGHIKLMNPQRSTVWY]{12,}\b",
        "[redacted_biological_sequence]",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\b(HEADER|ATOM|HETATM|MODEL|ENDMDL)\b.*", "[redacted_structure_payload]", text)
    text = re.sub(r"sk-[A-Za-z0-9_\-]{8,}", "[redacted_api_key]", text)
    return text[:_QUERY_SUMMARY_MAX_LEN]
