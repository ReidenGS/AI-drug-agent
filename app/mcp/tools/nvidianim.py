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


def NvidiaNIM_msa_search(
    sequence: str | None = None,
    *,
    e_value: float | None = None,
    iterations: int | None = None,
    output_alignment_formats: list[Any] | None = None,
    databases: list[Any] | None = None,
    max_msa_sequences: int | None = None,
    _live: bool = False,
) -> dict[str, Any]:
    """GPU-accelerated MSA search (ColabFold/MMseqs2) via NVIDIA NIM.

    Official ToolUniverse schema: the only required argument is ``sequence``
    (a single protein sequence, 1-4096 aa). Optional args mirror the official
    schema exactly: ``e_value`` (default 0.0001), ``iterations`` (default 1),
    ``output_alignment_formats`` (enum ["a3m","fasta"], default ["a3m"]),
    ``databases``, ``max_msa_sequences``. We only forward args the caller set
    so ToolUniverse applies its own documented defaults otherwise. No direct
    NVIDIA client and no offline mock success — this is a thin adapter binding.
    """
    args: dict[str, Any] = {"sequence": sequence or ""}
    if e_value is not None:
        args["e_value"] = e_value
    if iterations is not None:
        args["iterations"] = iterations
    if output_alignment_formats is not None:
        args["output_alignment_formats"] = output_alignment_formats
    if databases is not None:
        args["databases"] = databases
    if max_msa_sequences is not None:
        args["max_msa_sequences"] = max_msa_sequences
    return _call_nim("NvidiaNIM_msa_search", args, _live=_live)


def NvidiaNIM_rfdiffusion(
    contigs: str = "",
    input_pdb: str = "",
    *,
    hotspot_res: list[Any] | None = None,
    diffusion_steps: int | None = None,
    random_seed: int | None = None,
    _live: bool = False,
) -> dict[str, Any]:
    """RFdiffusion backbone-conditioned structure generation (Step 9).

    Official ToolUniverse required args: ``contigs``, ``input_pdb``. Optional
    args (``hotspot_res``, ``diffusion_steps``, ``random_seed``) are only
    forwarded when the caller set them, so ToolUniverse applies its own
    documented defaults otherwise. No direct NVIDIA client and no offline
    mock success — this is a thin adapter binding.
    """
    args: dict[str, Any] = {"contigs": contigs or "", "input_pdb": input_pdb or ""}
    if hotspot_res is not None:
        args["hotspot_res"] = hotspot_res
    if diffusion_steps is not None:
        args["diffusion_steps"] = diffusion_steps
    if random_seed is not None:
        args["random_seed"] = random_seed
    return _call_nim("NvidiaNIM_rfdiffusion", args, _live=_live)


def NvidiaNIM_proteinmpnn(
    input_pdb: str = "",
    *,
    ca_only: bool | None = None,
    use_soluble_model: bool | None = None,
    sampling_temp: list[Any] | None = None,
    num_seq_per_target: int | None = None,
    _live: bool = False,
) -> dict[str, Any]:
    """ProteinMPNN backbone-conditioned sequence design (Step 9).

    Official ToolUniverse required arg: ``input_pdb``. Optional args
    (``ca_only``, ``use_soluble_model``, ``sampling_temp``,
    ``num_seq_per_target``) are only forwarded when the caller set them. No
    direct NVIDIA client and no offline mock success — this is a thin
    adapter binding.
    """
    args: dict[str, Any] = {"input_pdb": input_pdb or ""}
    if ca_only is not None:
        args["ca_only"] = ca_only
    if use_soluble_model is not None:
        args["use_soluble_model"] = use_soluble_model
    if sampling_temp is not None:
        args["sampling_temp"] = sampling_temp
    if num_seq_per_target is not None:
        args["num_seq_per_target"] = num_seq_per_target
    return _call_nim("NvidiaNIM_proteinmpnn", args, _live=_live)


BINDINGS = [
    ("NvidiaNIM_alphafold2_multimer", NvidiaNIM_alphafold2_multimer),
    ("NvidiaNIM_openfold3", NvidiaNIM_openfold3),
    ("NvidiaNIM_boltz2", NvidiaNIM_boltz2),
    ("NvidiaNIM_msa_search", NvidiaNIM_msa_search),
    ("NvidiaNIM_rfdiffusion", NvidiaNIM_rfdiffusion),
    ("NvidiaNIM_proteinmpnn", NvidiaNIM_proteinmpnn),
]
