"""DrugProps / SwissADME / ADMETAI / BindingDB wrappers (Step 6).

Live mode (`_live=True`) for migrated wrappers routes through
`ToolUniverseAdapter` — there is no per-tool gate setting; the `_live`
flag itself is the switch, and `LocalMCPClient` decides whether to inject
`_live=True` based on `MCP_LIVE_TOOLS` + `MCP_LIVE_TOOL_ALLOWLIST`.

Adapter-backed: DrugProps_calculate_qed / DrugProps_lipinski_filter /
DrugProps_pains_filter / BindingDB_get_targets_by_compound /
SwissADME_calculate_adme / SwissADME_check_druglikeness.

Deferred (`_ni` → NotImplementedError on `_live=True`): all ADMETAI_*
tools. Reason: TU implementation is `ADMETAITool`, which loads the
`admet_ai` package's `ADMETModel` (PyTorch model weights), forcing a
`torch` + `admet_ai` dependency at runtime. Heavy local model inference
is out of scope for this migration round; we do not add `torch` or
`admet_ai` as project dependencies.
"""

from __future__ import annotations

from typing import Any


_VALID_DRUGLIKENESS_RULES = {"lipinski", "ghose", "veber", "egan", "muegge"}


def _ni(*_a, **_kw):
    raise NotImplementedError


def DrugProps_calculate_qed(smiles: str = "", *, _live: bool = False) -> dict[str, Any]:
    """Quantitative Estimate of Drug-likeness.

    Mock mode returns a deterministic envelope (`qed=None`). Live mode
    routes through `tooluniverse_adapter` when policy permits; otherwise
    raises NotImplementedError so the MCPClient surfaces it as
    `dependency_unavailable`.
    """
    if not smiles:
        raise ValueError("DrugProps_calculate_qed requires a non-empty smiles string")
    if not _live:
        return {
            "status": "mocked",
            "source": "DrugProps_calculate_qed",
            "smiles": smiles,
            "qed": None,
        }
    from ..tooluniverse_adapter import call_tool

    return call_tool("DrugProps_calculate_qed", {"smiles": smiles})


def DrugProps_lipinski_filter(smiles: str = "", *, _live: bool = False) -> dict[str, Any]:
    """Lipinski Rule of Five filter.

    Mock mode returns a deterministic envelope. Live mode routes through
    `ToolUniverseAdapter`; TU requires `smiles` and (like QED) requires
    `rdkit` at runtime — missing rdkit surfaces as `upstream_error`.
    """
    if not smiles:
        raise ValueError("DrugProps_lipinski_filter requires a non-empty smiles string")
    if not _live:
        return {
            "status": "mocked",
            "source": "DrugProps_lipinski_filter",
            "smiles": smiles,
            "passes_lipinski": None,
        }
    from ..tooluniverse_adapter import call_tool

    return call_tool("DrugProps_lipinski_filter", {"smiles": smiles})


def DrugProps_pains_filter(smiles: str = "", *, _live: bool = False) -> dict[str, Any]:
    """Screen a compound for PAINS / Brenk / NIH alerts.

    TU uses RDKit's FilterCatalog. Mock returns a deterministic empty
    envelope. Live routes through `ToolUniverseAdapter`; if rdkit is not
    installed in the runtime, TU surfaces an `error` and the adapter
    normalizes it to `status="upstream_error"` (we do NOT add rdkit as a
    hard project dependency).
    """
    if not smiles:
        raise ValueError("DrugProps_pains_filter requires a non-empty smiles string")
    if not _live:
        return {
            "status": "mocked",
            "source": "DrugProps_pains_filter",
            "smiles": smiles,
            "alerts": [],
            "passes": None,
        }
    from ..tooluniverse_adapter import call_tool

    return call_tool("DrugProps_pains_filter", {"smiles": smiles})


def BindingDB_get_targets_by_compound(
    smiles: str = "",
    *,
    similarity_cutoff: float = 0.85,
    _live: bool = False,
) -> dict[str, Any]:
    """Find protein targets for a compound by SMILES (BindingDB).

    TU required: `smiles`. Optional `similarity_cutoff` (0..1, default
    0.85) — wrapper clamps to that range before forwarding.
    """
    if not smiles:
        raise ValueError(
            "BindingDB_get_targets_by_compound requires a non-empty smiles string"
        )
    if not _live:
        return {
            "status": "mocked",
            "source": "BindingDB_get_targets_by_compound",
            "smiles": smiles,
            "similarity_cutoff": float(similarity_cutoff),
            "targets": [],
        }
    cutoff = float(similarity_cutoff)
    if cutoff < 0.0:
        cutoff = 0.0
    elif cutoff > 1.0:
        cutoff = 1.0
    from ..tooluniverse_adapter import call_tool

    return call_tool(
        "BindingDB_get_targets_by_compound",
        {"smiles": smiles, "similarity_cutoff": cutoff},
    )


def SwissADME_calculate_adme(
    smiles: str = "",
    *,
    molecule_name: str | None = None,
    _live: bool = False,
) -> dict[str, Any]:
    """SwissADME bulk ADME calculation.

    TU `SwissADMETool` requires `operation="calculate_adme"` and `smiles`;
    `molecule_name` is optional. The wrapper hard-codes the operation so
    callers expose only domain-meaningful args. Mock returns a
    deterministic empty envelope; live routes through `ToolUniverseAdapter`
    against the SwissADME web service (HTTP only, no local model).
    """
    if not smiles:
        raise ValueError("SwissADME_calculate_adme requires a non-empty smiles string")
    if not _live:
        return {
            "status": "mocked",
            "source": "SwissADME_calculate_adme",
            "smiles": smiles,
            "molecule_name": molecule_name,
            "adme": None,
        }
    from ..tooluniverse_adapter import call_tool

    args: dict[str, Any] = {"operation": "calculate_adme", "smiles": smiles}
    if molecule_name:
        args["molecule_name"] = molecule_name
    return call_tool("SwissADME_calculate_adme", args)


def SwissADME_check_druglikeness(
    smiles: str = "",
    *,
    rules: list[str] | None = None,
    _live: bool = False,
) -> dict[str, Any]:
    """SwissADME drug-likeness rule check.

    TU `SwissADMETool` requires `operation="check_druglikeness"` and
    `smiles`. Optional `rules` is a subset of {lipinski, ghose, veber,
    egan, muegge}; the wrapper validates the subset and only forwards it
    when non-empty (TU evaluates all 5 rules when omitted).
    """
    if not smiles:
        raise ValueError(
            "SwissADME_check_druglikeness requires a non-empty smiles string"
        )
    normalized_rules: list[str] | None = None
    if rules is not None:
        normalized_rules = [str(r).lower() for r in rules]
        invalid = [r for r in normalized_rules if r not in _VALID_DRUGLIKENESS_RULES]
        if invalid:
            raise ValueError(
                "SwissADME_check_druglikeness received unknown rules: "
                f"{sorted(set(invalid))}"
            )
    if not _live:
        return {
            "status": "mocked",
            "source": "SwissADME_check_druglikeness",
            "smiles": smiles,
            "rules": normalized_rules,
            "results": {},
        }
    from ..tooluniverse_adapter import call_tool

    args: dict[str, Any] = {"operation": "check_druglikeness", "smiles": smiles}
    if normalized_rules:
        args["rules"] = normalized_rules
    return call_tool("SwissADME_check_druglikeness", args)


BINDINGS = [
    ("DrugProps_pains_filter", DrugProps_pains_filter),
    ("DrugProps_lipinski_filter", DrugProps_lipinski_filter),
    ("DrugProps_calculate_qed", DrugProps_calculate_qed),
    ("SwissADME_calculate_adme", SwissADME_calculate_adme),
    ("SwissADME_check_druglikeness", SwissADME_check_druglikeness),
    ("ADMETAI_predict_toxicity", _ni),
    ("ADMETAI_predict_physicochemical_properties", _ni),
    ("ADMETAI_predict_solubility_lipophilicity_hydration", _ni),
    ("ADMETAI_predict_CYP_interactions", _ni),
    ("ADMETAI_predict_bioavailability", _ni),
    ("ADMETAI_predict_clearance_distribution", _ni),
    ("ADMETAI_predict_stress_response", _ni),
    ("ADMETAI_predict_nuclear_receptor_activity", _ni),
    ("BindingDB_get_targets_by_compound", BindingDB_get_targets_by_compound),
]
