"""NvidiaNIM wrappers (Steps 8, 9).

These wrappers are thin ToolUniverse bindings. They do not implement a direct
NVIDIA API client and they do not provide offline mock success.
"""

from __future__ import annotations

from typing import Any


def _call_nim(tool_name: str, args: dict[str, Any], *, _live: bool = False) -> dict[str, Any]:
    if not _live:
        raise NotImplementedError(
            f"{tool_name} requires live ToolUniverse execution; enable via MCP live settings and required upstream credentials"
        )
    from ..tooluniverse_adapter import call_tool

    return call_tool(tool_name, args)


def NvidiaNIM_alphafold2_multimer(
    sequences: list[Any] | None = None,
    *,
    databases: list[Any] | None = None,
    relax_prediction: bool | None = None,
    _live: bool = False,
) -> dict[str, Any]:
    args: dict[str, Any] = {"sequences": sequences or []}
    if databases is not None:
        args["databases"] = databases
    if relax_prediction is not None:
        args["relax_prediction"] = relax_prediction
    return _call_nim("NvidiaNIM_alphafold2_multimer", args, _live=_live)


def NvidiaNIM_openfold3(
    inputs: list[Any] | None = None,
    *,
    _live: bool = False,
) -> dict[str, Any]:
    return _call_nim("NvidiaNIM_openfold3", {"inputs": inputs or []}, _live=_live)


def NvidiaNIM_boltz2(
    polymers: list[Any] | None = None,
    *,
    ligands: list[Any] | None = None,
    recycling_steps: int | None = None,
    sampling_steps: int | None = None,
    diffusion_samples: int | None = None,
    output_format: str | None = None,
    _live: bool = False,
) -> dict[str, Any]:
    args: dict[str, Any] = {"polymers": polymers or []}
    if ligands is not None:
        args["ligands"] = ligands
    if recycling_steps is not None:
        args["recycling_steps"] = recycling_steps
    if sampling_steps is not None:
        args["sampling_steps"] = sampling_steps
    if diffusion_samples is not None:
        args["diffusion_samples"] = diffusion_samples
    if output_format is not None:
        args["output_format"] = output_format
    return _call_nim("NvidiaNIM_boltz2", args, _live=_live)


def _ni(*_a, **_kw):
    raise NotImplementedError


BINDINGS = [
    ("NvidiaNIM_alphafold2_multimer", NvidiaNIM_alphafold2_multimer),
    ("NvidiaNIM_openfold3", NvidiaNIM_openfold3),
    ("NvidiaNIM_boltz2", NvidiaNIM_boltz2),
    ("NvidiaNIM_rfdiffusion", _ni),
    ("NvidiaNIM_proteinmpnn", _ni),
]
