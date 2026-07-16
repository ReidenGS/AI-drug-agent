"""ToolUniverse adapter tests (no network, no real TU load).

The real `tooluniverse` registry holds 2 000+ tools and `load_tools()` is
slow + memory-heavy. These tests inject a fake universe via the shared
`install_universe` fixture (`tests/mcp/conftest.py`) so we exercise the
envelope normalization paths offline.

After the integration audit, `_live=True` on migrated wrappers routes
through the adapter unconditionally — there is no gate setting. Tests
here cover the adapter contract; per-wrapper routing tests live in
`test_p0_live_wrappers.py` and `test_live_public_wrappers.py`.
"""

from __future__ import annotations

import time
import socket
from typing import Any

import pytest
import requests

from app.mcp import tooluniverse_adapter
from app.mcp.tools.alphafold import alphafold_get_prediction
from app.mcp.tools.developability_compounds import DrugProps_calculate_qed
from app.mcp.tools.evidence import EuropePMC_search_articles


# ── include_tools resolution (sanctioned extras must be loadable) ──────────

def test_include_tool_names_contain_sanctioned_extras():
    """`include_tools` must union the git-tracked sanctioned extras so a live
    call to NvidiaNIM_msa_search does not report 'not found even after loading
    tools'."""
    from app.mcp.scope_filter import ARCHITECTURE_SANCTIONED_EXTRA_TOOLS

    tooluniverse_adapter._reset_for_tests()
    include = tooluniverse_adapter._resolve_include_tool_names()
    extras: set[str] = set()
    for tools in ARCHITECTURE_SANCTIONED_EXTRA_TOOLS.values():
        extras |= set(tools)
    assert "NvidiaNIM_msa_search" in include
    assert extras <= include


def test_include_tool_names_do_not_widen_to_full_tooluniverse():
    """Sanctioned extras are added on top of the inventory only — not a full
    ToolUniverse load."""
    tooluniverse_adapter._reset_for_tests()
    inventory = tooluniverse_adapter._resolve_inventory_names()
    extras = tooluniverse_adapter._sanctioned_extra_tool_names()
    include = tooluniverse_adapter._resolve_include_tool_names()
    assert include == frozenset(inventory | extras)
    # The extra surface stays tiny (guards against accidental full load).
    assert len(extras) <= 5
    assert "NvidiaNIM_msa_search" in extras


def test_get_universe_passes_include_tools_with_msa_search(monkeypatch):
    """`_get_universe` forwards an `include_tools` list containing the
    sanctioned extra to `ToolUniverse.load_tools`."""
    import sys
    import types

    captured: dict[str, Any] = {}

    class _FakeTU:
        def load_tools(self, quiet: bool = True, include_tools=None) -> None:
            captured["include_tools"] = include_tools

        def get_available_tools(self, name_only: bool = True):
            return list(captured.get("include_tools") or [])

    fake_module = types.ModuleType("tooluniverse")
    fake_module.ToolUniverse = _FakeTU  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "tooluniverse", fake_module)

    tooluniverse_adapter._reset_for_tests()
    try:
        tooluniverse_adapter._get_universe()
        include = captured.get("include_tools")
        assert include is not None
        assert "NvidiaNIM_msa_search" in include
    finally:
        tooluniverse_adapter._reset_for_tests()


# ── adapter envelope normalization ─────────────────────────────────────────

def test_adapter_success_payload(install_universe):
    install_universe(tools={"EuropePMC_search_articles": lambda args: {"results": [{"id": "1"}]}})
    out = tooluniverse_adapter.call_tool("EuropePMC_search_articles", {"query": "x"})
    assert out["status"] == "ok"
    assert out["executor"] == "tooluniverse"
    assert out["arguments"] == {"query": "x"}
    assert out["payload"] == {"results": [{"id": "1"}]}


def test_adapter_redacts_long_sequence_argument_in_envelope(install_universe):
    install_universe(
        tools={
            "EuropePMC_search_articles": lambda args: {"results": [{"id": "1"}]}
        }
    )
    seq = "M" * 250
    out = tooluniverse_adapter.call_tool(
        "EuropePMC_search_articles", {"query": "x", "sequence": seq, "operation": "search"}
    )
    assert out["status"] == "ok"
    seq_arg = out["arguments"]["sequence"]
    assert isinstance(seq_arg, dict)
    assert seq_arg.get("redacted") is True
    assert seq_arg["length"] == len(seq)
    assert "operation" in out["arguments"]
    assert out["arguments"]["operation"] == "search"
    fake = tooluniverse_adapter._get_universe()
    calls = fake.calls if hasattr(fake, "calls") else []
    assert calls and calls[0]["arguments"]["sequence"] == seq


def test_adapter_redacts_raw_pdb_like_argument_in_envelope(install_universe):
    install_universe(
        tools={"EuropePMC_search_articles": lambda args: {"results": [{"id": "1"}]}}
    )
    raw = "HEADER    TEST\nATOM      1  N   ASN A   1"
    out = tooluniverse_adapter.call_tool("EuropePMC_search_articles", {"query": raw})
    arg = out["arguments"]["query"]
    assert isinstance(arg, dict)
    assert arg["redacted"] is True
    fake = tooluniverse_adapter._get_universe()
    assert fake.calls[0]["arguments"]["query"] == raw


def test_adapter_redacts_nested_openfold_style_sequence_input(install_universe):
    install_universe(
        tools={
            "NvidiaNIM_openfold3": lambda args: {"status": "ok", "model_ref": "ok"}
        }
    )
    protein_seq = "G" * 300
    args = {
        "inputs": [
            {
                "input_id": "adc_antigen_antibody_complex",
                "molecules": [
                    {"type": "protein", "sequence": protein_seq, "msa": {"main": {"a3m": {"alignment": ">query\nAAA"}}}},
                    {"type": "protein", "sequence": "ABCDE"},
                ],
                "output_format": "pdb",
            }
        ],
        "operation": "predict",
    }
    out = tooluniverse_adapter.call_tool("NvidiaNIM_openfold3", args)
    assert out["status"] == "ok"
    redacted_sequence = out["arguments"]["inputs"][0]["molecules"][0]["sequence"]
    assert isinstance(redacted_sequence, dict)
    assert redacted_sequence.get("redacted") is True
    assert redacted_sequence["length"] == len(protein_seq)
    fake = tooluniverse_adapter._get_universe()
    calls = fake.calls if hasattr(fake, "calls") else []
    assert calls
    assert calls[0]["arguments"]["inputs"][0]["molecules"][0]["sequence"] == protein_seq


def test_adapter_keeps_short_scalar_values_in_argument_envelope(install_universe):
    install_universe(
        tools={"EuropePMC_search_articles": lambda args: {"results": [{"id": "1"}]}}
    )
    out = tooluniverse_adapter.call_tool("EuropePMC_search_articles", {"query": "x", "pdb_id": "1N8Z"})
    assert out["arguments"]["query"] == "x"
    assert out["arguments"]["pdb_id"] == "1N8Z"


def test_adapter_empty_list_payload(install_universe):
    install_universe(tools={"EuropePMC_search_articles": lambda args: {"results": []}})
    out = tooluniverse_adapter.call_tool("EuropePMC_search_articles", {"query": "x"})
    assert out["status"] == "empty"


def test_adapter_upstream_error_envelope(install_universe):
    install_universe(
        tools={
            "EuropePMC_search_articles": lambda args: {
                "status": "error",
                "error": "ToolValidationError: missing query",
                "error_details": {"type": "ToolValidationError"},
            }
        }
    )
    out = tooluniverse_adapter.call_tool("EuropePMC_search_articles", {})
    assert out["status"] == "upstream_error"
    assert "missing query" in out["error_message"]
    assert out["error_details"]["type"] == "ToolValidationError"


def test_adapter_unknown_tool(install_universe):
    install_universe(tools={})
    out = tooluniverse_adapter.call_tool("NotARealTool", {})
    assert out["status"] == "upstream_error"
    assert "not found" in out["error_message"].lower()


def test_adapter_exception_is_caught(install_universe, monkeypatch):
    fake = install_universe(tools={})

    def _boom(*args, **kwargs):
        raise RuntimeError("kaboom")

    monkeypatch.setattr(fake, "run_one_function", _boom)
    out = tooluniverse_adapter.call_tool("X", {})
    assert out["status"] == "upstream_error"
    assert "kaboom" in out["error_message"]


def test_adapter_unavailable_when_tu_missing(monkeypatch):
    """If tooluniverse is not installed at import time, adapter surfaces a
    clean upstream_error envelope rather than raising."""
    tooluniverse_adapter._reset_for_tests()
    real_import = __import__

    def _fake_import(name, *args, **kwargs):
        if name == "tooluniverse":
            raise ImportError("simulated: tooluniverse not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", _fake_import)
    out = tooluniverse_adapter.call_tool("Anything", {})
    assert out["status"] == "upstream_error"
    assert "tooluniverse" in out["error_message"].lower()


# ── migrated wrappers route through TU unconditionally on _live=True ───────

def test_europepmc_mock_unchanged():
    out = EuropePMC_search_articles("any query")
    assert out["status"] == "mocked"


def test_europepmc_live_always_uses_adapter(install_universe):
    """No gate setting — `_live=True` always routes through TU."""
    fake = install_universe(
        tools={"EuropePMC_search_articles": lambda args: {"results": [{"id": "x"}]}}
    )
    out = EuropePMC_search_articles("HER2 ADC", _live=True)
    assert out["executor"] == "tooluniverse"
    assert fake.calls[0]["arguments"]["query"] == "HER2 ADC"


def test_alphafold_live_routes_through_tu(install_universe):
    fake = install_universe(
        tools={"alphafold_get_prediction": lambda args: {"predictions": [{"qualifier": args["qualifier"]}]}}
    )
    out = alphafold_get_prediction("P00533", _live=True)
    assert out["executor"] == "tooluniverse"
    assert fake.calls[0]["arguments"]["qualifier"] == "P00533"


# ── DrugProps_calculate_qed: was _ni, now wired ────────────────────────────

def test_drugprops_qed_mock_unchanged():
    out = DrugProps_calculate_qed("CC(=O)Oc1ccccc1C(=O)O")
    assert out["status"] == "mocked"
    assert out["qed"] is None


def test_drugprops_qed_requires_smiles():
    with pytest.raises(ValueError):
        DrugProps_calculate_qed("")


def test_drugprops_qed_live_routes_through_tu(install_universe):
    install_universe(
        tools={"DrugProps_calculate_qed": lambda args: {"qed": 0.55, "smiles": args["smiles"]}}
    )
    out = DrugProps_calculate_qed("CC(=O)Oc1ccccc1C(=O)O", _live=True)
    assert out["executor"] == "tooluniverse"
    assert out["status"] == "ok"
    assert out["payload"]["qed"] == 0.55


def test_drugprops_qed_live_surfaces_rdkit_missing_as_upstream_error(install_universe):
    install_universe(
        tools={
            "DrugProps_calculate_qed": lambda args: {
                "status": "error",
                "error": "RDKit is required for drug property calculations.",
            }
        }
    )
    out = DrugProps_calculate_qed("CCO", _live=True)
    assert out["status"] == "upstream_error"
    assert "rdkit" in out["error_message"].lower()


# ── env hydration ──────────────────────────────────────────────────────────

import os  # noqa: E402


def test_hydrate_env_injects_when_settings_present_and_env_missing(monkeypatch):
    """Settings object holds a key; os.environ doesn't → hydrate injects it.

    We monkeypatch the settings attribute directly (not via env reload)
    so the local `.env`'s real key cannot leak into the assertion diff.
    """
    from app.mcp.tooluniverse_adapter import _hydrate_env_from_settings
    from app.settings import get_settings

    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    get_settings.cache_clear()
    settings = get_settings()
    monkeypatch.setattr(settings, "gemini_api_key", "sentinel-key-A")
    _hydrate_env_from_settings()
    assert os.environ.get("GEMINI_API_KEY") == "sentinel-key-A"


def test_hydrate_env_does_not_overwrite_existing_env(monkeypatch):
    from app.mcp.tooluniverse_adapter import _hydrate_env_from_settings
    from app.settings import get_settings

    monkeypatch.setenv("GEMINI_API_KEY", "operator-set")
    get_settings.cache_clear()
    settings = get_settings()
    monkeypatch.setattr(settings, "gemini_api_key", "from-dotenv-different")
    _hydrate_env_from_settings()
    assert os.environ["GEMINI_API_KEY"] == "operator-set"


def test_hydrate_env_does_nothing_when_settings_empty(monkeypatch):
    from app.mcp.tooluniverse_adapter import _hydrate_env_from_settings
    from app.settings import get_settings

    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_MODEL_ID", raising=False)
    monkeypatch.delenv("TOOLUNIVERSE_LLM_MODEL_DEFAULT", raising=False)
    get_settings.cache_clear()
    settings = get_settings()
    monkeypatch.setattr(settings, "gemini_api_key", "")
    monkeypatch.setattr(settings, "gemini_model", "")
    _hydrate_env_from_settings()
    assert "GEMINI_API_KEY" not in os.environ
    assert "GEMINI_MODEL_ID" not in os.environ
    assert "TOOLUNIVERSE_LLM_MODEL_DEFAULT" not in os.environ


def test_hydrate_env_never_logs_the_key(monkeypatch, capsys):
    from app.mcp.tooluniverse_adapter import _hydrate_env_from_settings
    from app.settings import get_settings

    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    get_settings.cache_clear()
    settings = get_settings()
    monkeypatch.setattr(settings, "gemini_api_key", "super-secret-do-not-print")
    _hydrate_env_from_settings()
    captured = capsys.readouterr()
    assert "super-secret-do-not-print" not in captured.out
    assert "super-secret-do-not-print" not in captured.err


def test_hydrate_env_bridges_model_to_tu_compatible_names(monkeypatch):
    from app.mcp.tooluniverse_adapter import _hydrate_env_from_settings
    from app.settings import get_settings

    monkeypatch.delenv("GEMINI_MODEL_ID", raising=False)
    monkeypatch.delenv("TOOLUNIVERSE_LLM_MODEL_DEFAULT", raising=False)
    get_settings.cache_clear()
    settings = get_settings()
    monkeypatch.setattr(settings, "gemini_api_key", "x")
    monkeypatch.setattr(settings, "gemini_model", "gemini-3.5-flash")
    _hydrate_env_from_settings()
    assert os.environ.get("GEMINI_MODEL_ID") == "gemini-3.5-flash"
    assert os.environ.get("TOOLUNIVERSE_LLM_MODEL_DEFAULT") == "gemini-3.5-flash"


def test_hydrate_env_bridges_nvidia_key_for_nim_tools(monkeypatch):
    from app.mcp.tooluniverse_adapter import _hydrate_env_from_settings
    from app.settings import get_settings

    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    get_settings.cache_clear()
    settings = get_settings()
    monkeypatch.setattr(settings, "nvidia_api_key", "nvidia-sentinel-key")
    _hydrate_env_from_settings()
    assert os.environ.get("NVIDIA_API_KEY") == "nvidia-sentinel-key"


def test_hydrate_env_does_not_overwrite_operator_nvidia_key(monkeypatch):
    from app.mcp.tooluniverse_adapter import _hydrate_env_from_settings
    from app.settings import get_settings

    monkeypatch.setenv("NVIDIA_API_KEY", "operator-nvidia-key")
    get_settings.cache_clear()
    settings = get_settings()
    monkeypatch.setattr(settings, "nvidia_api_key", "dotenv-nvidia-key")
    _hydrate_env_from_settings()
    assert os.environ["NVIDIA_API_KEY"] == "operator-nvidia-key"


def test_hydrate_env_bridges_esm_key_for_esm_tools(monkeypatch):
    from app.mcp.tooluniverse_adapter import _hydrate_env_from_settings
    from app.settings import get_settings

    monkeypatch.delenv("ESM_API_KEY", raising=False)
    get_settings.cache_clear()
    settings = get_settings()
    monkeypatch.setattr(settings, "esm_api_key", "esm-sentinel-key")
    _hydrate_env_from_settings()
    assert os.environ.get("ESM_API_KEY") == "esm-sentinel-key"


@pytest.mark.parametrize(
    ("settings_attr", "file_attr", "env_name"),
    [
        ("nvidia_api_key", "nvidia_api_key_file", "NVIDIA_API_KEY"),
        ("esm_api_key", "esm_api_key_file", "ESM_API_KEY"),
    ],
)
def test_hydrate_env_reads_optional_secret_file(
    monkeypatch, tmp_path, settings_attr, file_attr, env_name
):
    from app.mcp.tooluniverse_adapter import _hydrate_env_from_settings
    from app.settings import get_settings

    secret_file = tmp_path / "credential"
    secret_file.write_text("  fake-file-credential-sentinel  ")
    monkeypatch.delenv(env_name, raising=False)
    get_settings.cache_clear()
    settings = get_settings()
    monkeypatch.setattr(settings, settings_attr, "")
    monkeypatch.setattr(settings, file_attr, str(secret_file))

    try:
        _hydrate_env_from_settings()
        assert os.environ.get(env_name) == "fake-file-credential-sentinel"
    finally:
        monkeypatch.delenv(env_name, raising=False)


@pytest.mark.parametrize(
    ("settings_attr", "file_attr", "env_name"),
    [
        ("nvidia_api_key", "nvidia_api_key_file", "NVIDIA_API_KEY"),
        ("esm_api_key", "esm_api_key_file", "ESM_API_KEY"),
    ],
)
def test_hydrate_env_direct_setting_wins_over_bad_file(
    monkeypatch, tmp_path, settings_attr, file_attr, env_name
):
    from app.mcp.tooluniverse_adapter import _hydrate_env_from_settings
    from app.settings import get_settings

    monkeypatch.delenv(env_name, raising=False)
    get_settings.cache_clear()
    settings = get_settings()
    monkeypatch.setattr(settings, settings_attr, "  fake-direct-credential  ")
    monkeypatch.setattr(settings, file_attr, str(tmp_path / "missing-file"))

    try:
        _hydrate_env_from_settings()
        assert os.environ.get(env_name) == "fake-direct-credential"
    finally:
        monkeypatch.delenv(env_name, raising=False)


@pytest.mark.parametrize(
    ("settings_attr", "file_attr", "env_name"),
    [
        ("nvidia_api_key", "nvidia_api_key_file", "NVIDIA_API_KEY"),
        ("esm_api_key", "esm_api_key_file", "ESM_API_KEY"),
    ],
)
def test_hydrate_env_operator_value_wins_over_bad_file(
    monkeypatch, tmp_path, settings_attr, file_attr, env_name
):
    from app.mcp.tooluniverse_adapter import _hydrate_env_from_settings
    from app.settings import get_settings

    monkeypatch.setenv(env_name, "fake-operator-credential")
    get_settings.cache_clear()
    settings = get_settings()
    monkeypatch.setattr(settings, settings_attr, "")
    monkeypatch.setattr(settings, file_attr, str(tmp_path / "missing-file"))

    _hydrate_env_from_settings()
    assert os.environ.get(env_name) == "fake-operator-credential"


@pytest.mark.parametrize(
    ("settings_attr", "file_attr", "env_name", "kind", "error_code"),
    [
        (
            "nvidia_api_key",
            "nvidia_api_key_file",
            "NVIDIA_API_KEY",
            "unreadable",
            "nvidia_api_key_file_unreadable",
        ),
        (
            "nvidia_api_key",
            "nvidia_api_key_file",
            "NVIDIA_API_KEY",
            "empty",
            "nvidia_api_key_file_empty",
        ),
        (
            "esm_api_key",
            "esm_api_key_file",
            "ESM_API_KEY",
            "unreadable",
            "esm_api_key_file_unreadable",
        ),
        (
            "esm_api_key",
            "esm_api_key_file",
            "ESM_API_KEY",
            "empty",
            "esm_api_key_file_empty",
        ),
    ],
)
def test_hydrate_env_explicit_bad_file_fails_closed_without_leak(
    monkeypatch,
    tmp_path,
    settings_attr,
    file_attr,
    env_name,
    kind,
    error_code,
):
    from app.mcp.tooluniverse_adapter import _hydrate_env_from_settings
    from app.settings import get_settings

    secret_path = tmp_path / "private-credential-path-sentinel"
    if kind == "empty":
        secret_path.write_text(" \n")
    monkeypatch.delenv(env_name, raising=False)
    get_settings.cache_clear()
    settings = get_settings()
    monkeypatch.setattr(settings, settings_attr, "")
    monkeypatch.setattr(settings, file_attr, str(secret_path))

    with pytest.raises(ValueError, match=f"^{error_code}$") as caught:
        _hydrate_env_from_settings()

    for rendered in (str(caught.value), repr(caught.value), repr(caught.value.args)):
        assert "private-credential-path-sentinel" not in rendered
        assert str(tmp_path) not in rendered
        assert "fake-" not in rendered
    assert env_name not in os.environ


@pytest.mark.parametrize(
    ("settings_attr", "file_attr", "env_name"),
    [
        ("nvidia_api_key", "nvidia_api_key_file", "NVIDIA_API_KEY"),
        ("esm_api_key", "esm_api_key_file", "ESM_API_KEY"),
    ],
)
def test_hydrate_env_unconfigured_optional_secret_stays_absent(
    monkeypatch, settings_attr, file_attr, env_name
):
    from app.mcp.tooluniverse_adapter import _hydrate_env_from_settings
    from app.settings import get_settings

    monkeypatch.delenv(env_name, raising=False)
    get_settings.cache_clear()
    settings = get_settings()
    monkeypatch.setattr(settings, settings_attr, "")
    monkeypatch.setattr(settings, file_attr, None)

    _hydrate_env_from_settings()

    assert env_name not in os.environ


def test_hydrate_env_does_not_overwrite_operator_esm_key(monkeypatch):
    from app.mcp.tooluniverse_adapter import _hydrate_env_from_settings
    from app.settings import get_settings

    monkeypatch.setenv("ESM_API_KEY", "operator-esm-key")
    get_settings.cache_clear()
    settings = get_settings()
    monkeypatch.setattr(settings, "esm_api_key", "dotenv-esm-key")
    _hydrate_env_from_settings()
    assert os.environ["ESM_API_KEY"] == "operator-esm-key"


# ── transient-failure retry policy ─────────────────────────────────────────


class _FlakyHandler:
    """Fail `fail_times` with a given error, then return `ok_payload`.

    `mode="raise"` raises an exception (network-style); `mode="error"`
    returns a TU structured `{"status": "error", ...}` envelope.
    """

    def __init__(self, *, fail_times, error_text, ok_payload, mode="raise"):
        self.fail_times = fail_times
        self.error_text = error_text
        self.ok_payload = ok_payload
        self.mode = mode
        self.calls = 0

    def __call__(self, args):
        self.calls += 1
        if self.calls <= self.fail_times:
            if self.mode == "raise":
                raise ConnectionError(self.error_text)
            return {"status": "error", "error": self.error_text}
        return self.ok_payload


@pytest.fixture(autouse=True)
def _no_backoff_sleep(monkeypatch):
    """Keep retry tests fast — no real backoff sleeping."""
    monkeypatch.setattr(tooluniverse_adapter, "_sleep_backoff", lambda attempt: None)


def test_retry_transient_then_success(install_universe):
    handler = _FlakyHandler(
        fail_times=1,
        error_text="ProxyError: Max retries exceeded (RemoteDisconnected)",
        ok_payload={"results": [{"id": "1"}]},
    )
    install_universe(tools={"EuropePMC_search_articles": handler})
    out = tooluniverse_adapter.call_tool("EuropePMC_search_articles", {"query": "x"})
    assert out["status"] == "ok"
    assert out["retry_count"] == 1
    assert out["recovered_after_transient_error"] is True
    assert out["final_error_type"] == "ConnectionError"
    assert handler.calls == 2


def test_retry_transient_exhausted_keeps_upstream_error(install_universe):
    handler = _FlakyHandler(
        fail_times=99,
        error_text="HTTPSConnectionPool: Max retries exceeded (RemoteDisconnected)",
        ok_payload={"results": []},
    )
    install_universe(tools={"EuropePMC_search_articles": handler})
    out = tooluniverse_adapter.call_tool("EuropePMC_search_articles", {"query": "x"})
    assert out["status"] == "upstream_error"
    assert out["retry_count"] == tooluniverse_adapter._MAX_LIVE_RETRIES
    assert out["retry_count"] > 0
    assert out["retryable"] is True
    assert out["final_error_type"] == "ConnectionError"
    # total attempts = 1 + max retries; never mocked a fake success.
    assert handler.calls == tooluniverse_adapter._MAX_LIVE_RETRIES + 1
    assert "payload" not in out


def test_retry_transient_5xx_structured_error(install_universe):
    handler = _FlakyHandler(
        fail_times=1,
        error_text="503 Server Error: Service Unavailable",
        ok_payload={"results": [{"id": "ok"}]},
        mode="error",
    )
    install_universe(tools={"EuropePMC_search_articles": handler})
    out = tooluniverse_adapter.call_tool("EuropePMC_search_articles", {"query": "x"})
    assert out["status"] == "ok"
    assert out["retry_count"] == 1


def test_no_retry_for_validation_error(install_universe):
    handler = _FlakyHandler(
        fail_times=99,
        error_text="ToolValidationError: missing required field 'query'",
        ok_payload={"results": []},
        mode="error",
    )
    install_universe(tools={"EuropePMC_search_articles": handler})
    out = tooluniverse_adapter.call_tool("EuropePMC_search_articles", {})
    assert out["status"] == "upstream_error"
    assert out["retry_count"] == 0
    assert out["retryable"] is False
    assert handler.calls == 1  # deterministic error → no retry


def test_no_retry_for_missing_input_error(install_universe):
    handler = _FlakyHandler(
        fail_times=99,
        error_text="missing-input: smiles is required",
        ok_payload={},
        mode="error",
    )
    install_universe(tools={"EuropePMC_search_articles": handler})
    out = tooluniverse_adapter.call_tool("EuropePMC_search_articles", {})
    assert out["status"] == "upstream_error"
    assert out["retry_count"] == 0
    assert out["retryable"] is False
    assert handler.calls == 1


def test_success_first_try_reports_zero_retries(install_universe):
    install_universe(
        tools={"EuropePMC_search_articles": lambda args: {"results": [{"id": "1"}]}}
    )
    out = tooluniverse_adapter.call_tool("EuropePMC_search_articles", {"query": "x"})
    assert out["retry_count"] == 0
    assert out["retryable"] is False
    assert "recovered_after_transient_error" not in out


def test_live_call_outer_timeout_surfaces_upstream_error(install_universe, monkeypatch):
    from app.settings import get_settings

    get_settings.cache_clear()
    settings = get_settings()
    monkeypatch.setattr(settings, "tooluniverse_live_call_timeout", 0.01)

    def _hang(_args):
        time.sleep(2)
        return {"results": [{"id": "late"}]}

    install_universe(tools={"EuropePMC_search_articles": _hang})
    out = tooluniverse_adapter.call_tool("EuropePMC_search_articles", {"query": "x"})
    assert out["status"] == "upstream_error"
    assert out["retry_count"] == tooluniverse_adapter._MAX_LIVE_RETRIES
    assert out["retryable"] is True
    assert out["final_error_type"] == "TimeoutError"
    assert "exceeded" in out["error_message"]
    assert "payload" not in out


def test_live_call_timeout_sets_and_restores_socket_default(install_universe, monkeypatch):
    from app.settings import get_settings

    get_settings.cache_clear()
    settings = get_settings()
    monkeypatch.setattr(settings, "tooluniverse_live_call_timeout", 3.5)
    monkeypatch.setattr(socket, "getdefaulttimeout", lambda: None)
    seen: list[float | None] = []

    def _capture(timeout):
        seen.append(timeout)

    monkeypatch.setattr(socket, "setdefaulttimeout", _capture)
    install_universe(
        tools={"EuropePMC_search_articles": lambda args: {"results": [{"id": "ok"}]}}
    )
    out = tooluniverse_adapter.call_tool("EuropePMC_search_articles", {"query": "x"})
    assert out["status"] == "ok"
    assert seen[0] == 3.5
    assert seen[-1] is None


def test_live_call_uses_tooluniverse_timeout_for_non_nvidia_tool(install_universe, monkeypatch):
    from app.settings import get_settings

    get_settings.cache_clear()
    settings = get_settings()
    monkeypatch.setattr(settings, "tooluniverse_live_call_timeout", 2.25)
    monkeypatch.setattr(settings, "nvidia_nim_live_call_timeout", 999.0)
    seen: list[float | None] = []

    def _capture(timeout):
        seen.append(timeout)

    monkeypatch.setattr(socket, "setdefaulttimeout", _capture)
    install_universe(
        tools={"EuropePMC_search_articles": lambda args: {"results": [{"id": "ok"}]}}
    )
    out = tooluniverse_adapter.call_tool("EuropePMC_search_articles", {"query": "x"})
    assert out["status"] == "ok"
    assert seen[0] == 2.25
    assert seen[-1] is None


def test_live_call_uses_nvidia_timeout_for_nvidia_tool(install_universe, monkeypatch):
    from app.settings import get_settings

    get_settings.cache_clear()
    settings = get_settings()
    monkeypatch.setattr(settings, "tooluniverse_live_call_timeout", 2.25)
    monkeypatch.setattr(settings, "nvidia_nim_live_call_timeout", 33.0)
    seen: list[float | None] = []

    def _capture(timeout):
        seen.append(timeout)

    monkeypatch.setattr(socket, "setdefaulttimeout", _capture)
    install_universe(
        tools={"NvidiaNIM_alphafold2_multimer": lambda args: {"predictions": []}}
    )
    out = tooluniverse_adapter.call_tool("NvidiaNIM_alphafold2_multimer", {"antigen_fasta": "x"})
    assert out["status"] == "ok"
    assert seen[0] == 33.0
    assert seen[-1] is None


def test_live_call_injects_requests_timeout_when_missing(install_universe, monkeypatch):
    captured: dict[str, Any] = {}
    timeout_during_bound_call: list[float | None] = []

    def _fake_request(self, method, url, **kwargs):
        captured["timeout"] = kwargs.get("timeout")
        timeout_during_bound_call.append(kwargs.get("timeout"))
        return type("Resp", (), {"json": lambda self: {"status": "ok"}})()

    monkeypatch.setattr(requests.sessions.Session, "request", _fake_request)

    class _RequestTool:
        def __call__(self, args):
            session = requests.Session()
            session.request("GET", "https://example.org/ping")
            return {"ok": True}

    install_universe(tools={"EuropePMC_search_articles": _RequestTool()})
    tooluniverse_adapter.call_tool("EuropePMC_search_articles", {"query": "x"})
    assert captured["timeout"] == 60.0
    assert timeout_during_bound_call == [60.0]


def test_live_call_respects_explicit_requests_timeout(install_universe, monkeypatch):
    captured: dict[str, Any] = {}
    timeout_during_bound_call: list[float | None] = []

    def _fake_request(self, method, url, **kwargs):
        captured["timeout"] = kwargs.get("timeout")
        timeout_during_bound_call.append(kwargs.get("timeout"))
        return type("Resp", (), {"json": lambda self: {"status": "ok"}})()

    monkeypatch.setattr(requests.sessions.Session, "request", _fake_request)

    class _RequestTool:
        def __call__(self, args):
            session = requests.Session()
            session.request("GET", "https://example.org/ping", timeout=7.25)
            return {"ok": True}

    install_universe(tools={"EuropePMC_search_articles": _RequestTool()})
    tooluniverse_adapter.call_tool("EuropePMC_search_articles", {"query": "x"})
    assert captured["timeout"] == 7.25
    assert timeout_during_bound_call == [7.25]


def test_live_call_restores_session_request_after_timeout_context(install_universe, monkeypatch):
    captured_request_methods: list[Any] = []

    def _fake_request(self, method, url, **kwargs):
        captured_request_methods.append(requests.sessions.Session.request)
        return type("Resp", (), {"json": lambda self: {"status": "ok"}})()

    class _RequestTool:
        def __call__(self, args):
            session = requests.Session()
            session.request("GET", "https://example.org/ping")
            return {"ok": True}

    monkeypatch.setattr(requests.sessions.Session, "request", _fake_request)
    entry_request = requests.sessions.Session.request
    install_universe(tools={"EuropePMC_search_articles": _RequestTool()})
    tooluniverse_adapter.call_tool("EuropePMC_search_articles", {"query": "x"})
    assert captured_request_methods and captured_request_methods[0] is not entry_request
    assert requests.sessions.Session.request is entry_request


def test_nvidia_tool_timeout_still_surfaces_upstream_error_on_retries(install_universe, monkeypatch):
    from app.settings import get_settings

    get_settings.cache_clear()
    settings = get_settings()
    monkeypatch.setattr(settings, "nvidia_nim_live_call_timeout", 0.01)

    def _hang(_args):
        time.sleep(0.1)
        return {"predictions": ["late"]}

    install_universe(tools={"NvidiaNIM_alphafold2_multimer": _hang})
    out = tooluniverse_adapter.call_tool("NvidiaNIM_alphafold2_multimer", {"antigen": "x"})
    assert out["status"] == "upstream_error"
    assert out["retry_count"] == tooluniverse_adapter._MAX_LIVE_RETRIES
    assert out["retryable"] is True
    assert out["final_error_type"] == "TimeoutError"
