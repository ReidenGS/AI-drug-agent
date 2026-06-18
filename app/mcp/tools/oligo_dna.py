"""IDT / DNA / Sequence wrappers (oligo + DNA primitives).

All six tools live in ToolUniverse 1.2.2 and are migrated to the adapter:

| Wrapper | TU class | Network / deps |
|---|---|---|
| `IDT_analyze_oligo` | `IDTTool` | IDT REST API (HTTP, no key) |
| `IDT_check_self_dimer` | `IDTTool` | IDT REST API (HTTP, no key) |
| `DNA_calculate_gc_content` | `DNATool` | Pure-Python local (no network) |
| `DNA_reverse_complement` | `DNATool` | Pure-Python local (no network) |
| `Sequence_gc_content` | `SequenceAnalyzeTool` | Pure-Python local (no network) |
| `Sequence_reverse_complement` | `SequenceAnalyzeTool` | Pure-Python local (no network) |

`_live=True` routes through `tooluniverse_adapter.call_tool(...)`; wrappers
do NOT import ToolUniverse and do NOT issue HTTP themselves. `_live=False`
returns a deterministic mock envelope that mock callers can branch on but
NEVER claims a live result.
"""

from __future__ import annotations

from typing import Any

from ._arg_compat import resolve_operation


# ── IDT ──────────────────────────────────────────────────────────────────────

_IDT_OLIGO_OPTIONAL = (
    "na_concentration_mm",
    "mg_concentration_mm",
    "dntps_concentration_mm",
    "oligo_concentration_um",
    "oligo_type",
)
_IDT_DIMER_OPTIONAL = (
    "na_concentration_mm",
    "mg_concentration_mm",
    "temperature_celsius",
)


def _filter_optional(values: dict[str, Any], allowed: tuple[str, ...]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in allowed:
        if key in values and values[key] is not None:
            out[key] = values[key]
    return out


def IDT_analyze_oligo(
    sequence: str = "",
    *,
    na_concentration_mm: float | None = None,
    mg_concentration_mm: float | None = None,
    dntps_concentration_mm: float | None = None,
    oligo_concentration_um: float | None = None,
    oligo_type: str | None = None,
    _live: bool = False,
    **_extra: Any,
) -> dict[str, Any]:
    """IDT OligoAnalyzer — Tm, ΔG, secondary structure for a single oligo."""
    if not sequence:
        raise ValueError("IDT_analyze_oligo requires a non-empty sequence")
    extras = _filter_optional(
        {
            "na_concentration_mm": na_concentration_mm,
            "mg_concentration_mm": mg_concentration_mm,
            "dntps_concentration_mm": dntps_concentration_mm,
            "oligo_concentration_um": oligo_concentration_um,
            "oligo_type": oligo_type,
        },
        _IDT_OLIGO_OPTIONAL,
    )
    if not _live:
        return {
            "status": "mocked",
            "source": "IDT_analyze_oligo",
            "sequence": sequence,
            **extras,
            "properties": None,
        }
    from ..tooluniverse_adapter import call_tool

    return call_tool("IDT_analyze_oligo", {"sequence": sequence, **extras})


def IDT_check_self_dimer(
    sequence: str = "",
    *,
    na_concentration_mm: float | None = None,
    mg_concentration_mm: float | None = None,
    temperature_celsius: float | None = None,
    _live: bool = False,
    **_extra: Any,
) -> dict[str, Any]:
    """IDT self-dimer / hairpin check for a single oligo."""
    if not sequence:
        raise ValueError("IDT_check_self_dimer requires a non-empty sequence")
    extras = _filter_optional(
        {
            "na_concentration_mm": na_concentration_mm,
            "mg_concentration_mm": mg_concentration_mm,
            "temperature_celsius": temperature_celsius,
        },
        _IDT_DIMER_OPTIONAL,
    )
    if not _live:
        return {
            "status": "mocked",
            "source": "IDT_check_self_dimer",
            "sequence": sequence,
            **extras,
            "self_dimer": None,
        }
    from ..tooluniverse_adapter import call_tool

    return call_tool("IDT_check_self_dimer", {"sequence": sequence, **extras})


# ── DNA / Sequence local primitives ─────────────────────────────────────────


def _dna_call(
    expected_op: str, sequence: str, *,
    source: str, mock_field: str, _live: bool,
    operation: str | None,
) -> dict[str, Any]:
    if not sequence:
        raise ValueError(f"{source} requires a non-empty sequence")
    op = resolve_operation(operation, expected_op)
    if not _live:
        return {
            "status": "mocked",
            "source": source,
            "sequence": sequence,
            mock_field: None,
        }
    from ..tooluniverse_adapter import call_tool

    return call_tool(source, {"operation": op, "sequence": sequence})


def DNA_calculate_gc_content(
    sequence: str = "",
    *,
    operation: str | None = None,
    _live: bool = False,
    **_extra: Any,
) -> dict[str, Any]:
    """GC% over a DNA sequence (TU DNATool: operation=calculate_gc_content)."""
    return _dna_call(
        "calculate_gc_content", sequence,
        source="DNA_calculate_gc_content", mock_field="gc_content",
        _live=_live, operation=operation,
    )


def DNA_reverse_complement(
    sequence: str = "",
    *,
    operation: str | None = None,
    _live: bool = False,
    **_extra: Any,
) -> dict[str, Any]:
    """Reverse complement of a DNA sequence (TU DNATool: operation=reverse_complement)."""
    return _dna_call(
        "reverse_complement", sequence,
        source="DNA_reverse_complement", mock_field="reverse_complement",
        _live=_live, operation=operation,
    )


def Sequence_gc_content(
    sequence: str = "",
    *,
    operation: str | None = None,
    _live: bool = False,
    **_extra: Any,
) -> dict[str, Any]:
    """Generic GC% (TU SequenceAnalyzeTool: operation=gc_content)."""
    return _dna_call(
        "gc_content", sequence,
        source="Sequence_gc_content", mock_field="gc_content",
        _live=_live, operation=operation,
    )


def Sequence_reverse_complement(
    sequence: str = "",
    *,
    operation: str | None = None,
    _live: bool = False,
    **_extra: Any,
) -> dict[str, Any]:
    """Generic reverse complement (TU SequenceAnalyzeTool: operation=reverse_complement)."""
    return _dna_call(
        "reverse_complement", sequence,
        source="Sequence_reverse_complement",
        mock_field="reverse_complement",
        _live=_live, operation=operation,
    )


BINDINGS = [
    ("IDT_analyze_oligo", IDT_analyze_oligo),
    ("IDT_check_self_dimer", IDT_check_self_dimer),
    ("DNA_calculate_gc_content", DNA_calculate_gc_content),
    ("DNA_reverse_complement", DNA_reverse_complement),
    ("Sequence_gc_content", Sequence_gc_content),
    ("Sequence_reverse_complement", Sequence_reverse_complement),
]
