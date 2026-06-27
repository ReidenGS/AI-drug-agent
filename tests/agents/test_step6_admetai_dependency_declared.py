"""Drift fence: ADMETAI registry status must match project dependency declaration.

If the Step 6 capability registry marks any ``ADMETAI_*`` tool as
``runtime_policy="live_wired"``, ``pyproject.toml`` MUST declare the
``admet`` optional dependency so a fresh checkout can reproduce the
live wiring with ``pip install -e '.[admet]'``. Without this fence the
registry could silently flip back to "live" while ``pyproject.toml``
stayed silent, and a new operator would be unable to bring the live
path back up without trial-and-error.

The test only parses metadata; it does not import ``admet_ai``, ``torch``,
ToolUniverse, or any LLM provider — no model weights are loaded, no
network access, no environment mutation.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

from app.agents.step_06_capability_registry import STEP_06_CAPABILITY_REGISTRY


_PYPROJECT = Path(__file__).resolve().parents[2] / "pyproject.toml"


def _load_pyproject() -> dict:
    """Read ``pyproject.toml`` via stdlib ``tomllib`` (Python 3.11+).

    We deliberately avoid third-party TOML parsers so this fence runs
    in any minimal checkout.
    """
    if sys.version_info < (3, 11):  # pragma: no cover - project requires 3.11+
        pytest.skip("tomllib requires Python 3.11+")
    import tomllib

    with _PYPROJECT.open("rb") as fh:
        return tomllib.load(fh)


def _live_admetai_tool_names() -> set[str]:
    return {
        cap.tool_name
        for cap in STEP_06_CAPABILITY_REGISTRY
        if cap.tool_name.startswith("ADMETAI_")
        and cap.runtime_policy == "live_wired"
    }


def _optional_dependency_groups(pyproject: dict) -> dict[str, list[str]]:
    project = pyproject.get("project") or {}
    opt = project.get("optional-dependencies") or {}
    return {str(name): list(reqs or []) for name, reqs in opt.items()}


def _matches_admet_ai(req: str) -> bool:
    """True iff the requirement string declares the ``admet-ai`` (or
    underscore variant) package — version specifier ignored.

    PEP 508 normalizes ``-``/``_`` to the same canonical name, but we
    let the test accept either form for human readability.
    """
    head = re.split(r"[\s\[<>=!~;]", req.strip(), 1)[0].lower()
    return head in {"admet-ai", "admet_ai"}


# ── tests ───────────────────────────────────────────────────────────────────


def test_pyproject_declares_admet_optional_when_registry_marks_live_wired():
    """If any ADMETAI_* tool is registry-live, pyproject must offer an
    ``admet`` optional dependency group, and that group must list the
    ``admet-ai`` package."""
    live_admetai = _live_admetai_tool_names()
    if not live_admetai:
        pytest.skip(
            "No ADMETAI tool is registry-live; the dependency fence is "
            "vacuous in this configuration."
        )

    pyproject = _load_pyproject()
    opt = _optional_dependency_groups(pyproject)
    assert "admet" in opt, (
        "Step 6 registry marks "
        + ", ".join(sorted(live_admetai))
        + " as live_wired but pyproject.toml has no [project.optional-dependencies].admet "
        "group. Either declare the admet extra or revert the live_wired flip."
    )
    pinned = opt["admet"]
    assert any(_matches_admet_ai(req) for req in pinned), (
        f"[project.optional-dependencies].admet does not list the admet-ai "
        f"package; got {pinned!r}."
    )


def test_admet_extra_does_not_pin_admetai_into_default_dependencies():
    """The runtime ML stack (admet-ai / torch / chemprop / scikit-learn /
    lightning) is heavy. It must stay behind the ``admet`` extra so
    plain ``pip install -e .`` still resolves quickly in CI / dev
    environments that don't need the live model path."""
    pyproject = _load_pyproject()
    base = (pyproject.get("project") or {}).get("dependencies") or []
    for req in base:
        head = re.split(r"[\s\[<>=!~;]", str(req).strip(), 1)[0].lower()
        assert head not in {
            "admet-ai", "admet_ai", "torch", "chemprop", "scikit-learn",
            "pytorch-lightning", "lightning",
        }, (
            f"{head!r} leaked into [project].dependencies; keep heavy ML "
            "wheels behind the admet optional extra."
        )


def test_all_eight_admetai_tools_have_consistent_runtime_policy():
    """The 8 ADMETAI tools should rise and fall together — the
    wrappers and the dependency footprint are identical. A partial
    flip almost certainly indicates a manual-edit drift bug."""
    admetai_caps = [
        c for c in STEP_06_CAPABILITY_REGISTRY if c.tool_name.startswith("ADMETAI_")
    ]
    assert len(admetai_caps) == 8, [c.tool_name for c in admetai_caps]
    policies = {c.runtime_policy for c in admetai_caps}
    assert len(policies) == 1, (
        f"ADMETAI runtime_policy is split: {policies}. All eight tools "
        f"must share one classification."
    )
