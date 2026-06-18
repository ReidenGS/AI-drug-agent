"""Wrapper tests for the 18 newly registered ToolUniverse bindings.

Covers:

- IDT_analyze_oligo, IDT_check_self_dimer
- DNA_calculate_gc_content, DNA_reverse_complement
- Sequence_gc_content, Sequence_reverse_complement
- RNAcentral_search, RNAcentral_get_by_accession
- Rfam_search_sequence, Rfam_get_family
- LNCipedia_search_lncrna, LNCipedia_get_lncrna
- miRBase_search_mirna, miRBase_get_mirna
- dynamic_package_discovery (registered + deferred)
- embedding_database_create / _add / _search (registered + deferred)

For each migrated tool: mock unchanged, live routes through the fake
ToolUniverseAdapter (no real network), required/invalid args, upstream
error propagation. For each deferred tool: `_live=True` raises
NotImplementedError BEFORE any TU dispatch — fake universe records zero
calls.
"""

from __future__ import annotations

import pytest

from app.mcp.tools.oligo_dna import (
    DNA_calculate_gc_content,
    DNA_reverse_complement,
    IDT_analyze_oligo,
    IDT_check_self_dimer,
    Sequence_gc_content,
    Sequence_reverse_complement,
)
from app.mcp.tools.rna_databases import (
    LNCipedia_get_lncrna,
    LNCipedia_search_lncrna,
    RNAcentral_get_by_accession,
    RNAcentral_search,
    Rfam_get_family,
    Rfam_search_sequence,
    miRBase_get_mirna,
    miRBase_search_mirna,
)
from app.mcp.tools.utility_meta import (
    BINDINGS as UTILITY_META_BINDINGS,
)


# ── IDT ──────────────────────────────────────────────────────────────────────


def test_idt_oligo_mock_unchanged():
    out = IDT_analyze_oligo(sequence="ATCG")
    assert out["status"] == "mocked"
    assert out["sequence"] == "ATCG"
    assert out["properties"] is None


def test_idt_oligo_requires_sequence():
    with pytest.raises(ValueError):
        IDT_analyze_oligo()


def test_idt_oligo_live_routes_minimal(install_universe):
    fake = install_universe(
        tools={"IDT_analyze_oligo": lambda args: {"properties": {"tm_celsius": 60.0}}}
    )
    out = IDT_analyze_oligo(sequence="ATCG", _live=True)
    assert out["status"] == "ok"
    assert out["executor"] == "tooluniverse"
    # Only required arg forwarded; nothing else fabricated.
    assert fake.calls[0]["arguments"] == {"sequence": "ATCG"}


def test_idt_oligo_live_forwards_optional_only_when_set(install_universe):
    fake = install_universe(
        tools={"IDT_analyze_oligo": lambda args: {"properties": {}}}
    )
    IDT_analyze_oligo(
        sequence="ATCG",
        na_concentration_mm=50.0,
        oligo_type="DNA",
        _live=True,
    )
    forwarded = fake.calls[0]["arguments"]
    assert forwarded == {
        "sequence": "ATCG",
        "na_concentration_mm": 50.0,
        "oligo_type": "DNA",
    }


def test_idt_oligo_live_upstream_error(install_universe):
    install_universe(
        tools={
            "IDT_analyze_oligo": lambda args: {
                "status": "error",
                "error": "IDT API unavailable",
            }
        }
    )
    out = IDT_analyze_oligo(sequence="ATCG", _live=True)
    assert out["status"] == "upstream_error"


def test_idt_self_dimer_live_routes(install_universe):
    fake = install_universe(
        tools={"IDT_check_self_dimer": lambda args: {"self_dimer": {"delta_g": -2.1}}}
    )
    IDT_check_self_dimer(sequence="ATCG", temperature_celsius=37.0, _live=True)
    assert fake.calls[0]["arguments"] == {
        "sequence": "ATCG",
        "temperature_celsius": 37.0,
    }


def test_idt_self_dimer_requires_sequence():
    with pytest.raises(ValueError):
        IDT_check_self_dimer()


# ── DNA / Sequence local primitives ─────────────────────────────────────────


@pytest.mark.parametrize(
    "fn,name,operation,mock_field",
    [
        (DNA_calculate_gc_content, "DNA_calculate_gc_content", "calculate_gc_content", "gc_content"),
        (DNA_reverse_complement, "DNA_reverse_complement", "reverse_complement", "reverse_complement"),
        (Sequence_gc_content, "Sequence_gc_content", "gc_content", "gc_content"),
        (Sequence_reverse_complement, "Sequence_reverse_complement", "reverse_complement", "reverse_complement"),
    ],
)
def test_dna_and_sequence_ops(install_universe, fn, name, operation, mock_field):
    # mock unchanged
    out = fn(sequence="ATCG")
    assert out["status"] == "mocked"
    assert out["sequence"] == "ATCG"
    assert out[mock_field] is None

    # required
    with pytest.raises(ValueError):
        fn()

    # live routes with hard-coded operation
    fake = install_universe(tools={name: lambda args: {mock_field: 0.5}})
    out = fn(sequence="ATCG", _live=True)
    assert out["status"] == "ok"
    assert out["executor"] == "tooluniverse"
    assert fake.calls[0]["arguments"] == {
        "operation": operation,
        "sequence": "ATCG",
    }

    # upstream_error propagation
    install_universe(
        tools={name: lambda args: {"status": "error", "error": "bad"}}
    )
    err = fn(sequence="ATCG", _live=True)
    assert err["status"] == "upstream_error"


# ── RNAcentral ──────────────────────────────────────────────────────────────


def test_rnacentral_search_mock_and_live(install_universe):
    out = RNAcentral_search(query="EGFR")
    assert out["status"] == "mocked"
    with pytest.raises(ValueError):
        RNAcentral_search()
    fake = install_universe(
        tools={"RNAcentral_search": lambda args: {"results": [{"id": "URS1"}]}}
    )
    out = RNAcentral_search(query="EGFR", _live=True)
    assert out["status"] == "ok"
    assert fake.calls[0]["arguments"] == {"query": "EGFR"}
    # page_size only forwarded when caller sets it.
    RNAcentral_search(query="EGFR", page_size=25, _live=True)
    assert fake.calls[-1]["arguments"] == {"query": "EGFR", "page_size": 25}


def test_rnacentral_get_by_accession_mock_and_live(install_universe):
    with pytest.raises(ValueError):
        RNAcentral_get_by_accession()
    out = RNAcentral_get_by_accession(accession="URS0000ABC")
    assert out["status"] == "mocked"
    fake = install_universe(
        tools={
            "RNAcentral_get_by_accession": lambda args: {"entry": {"id": args["accession"]}}
        }
    )
    out = RNAcentral_get_by_accession(accession="URS0000ABC", _live=True)
    assert out["status"] == "ok"
    assert fake.calls[0]["arguments"] == {"accession": "URS0000ABC"}


# ── Rfam ────────────────────────────────────────────────────────────────────


def test_rfam_search_sequence_live(install_universe):
    with pytest.raises(ValueError):
        Rfam_search_sequence()
    fake = install_universe(
        tools={"Rfam_search_sequence": lambda args: {"hits": [{"family": "RF00001"}]}}
    )
    Rfam_search_sequence(sequence="AUGC", _live=True)
    assert fake.calls[0]["arguments"] == {
        "operation": "search_sequence",
        "sequence": "AUGC",
    }
    Rfam_search_sequence(sequence="AUGC", max_wait_seconds=30, _live=True)
    assert fake.calls[-1]["arguments"]["max_wait_seconds"] == 30
    # negative clamps to 0
    Rfam_search_sequence(sequence="AUGC", max_wait_seconds=-5, _live=True)
    assert fake.calls[-1]["arguments"]["max_wait_seconds"] == 0


def test_rfam_get_family_live(install_universe):
    with pytest.raises(ValueError):
        Rfam_get_family()
    with pytest.raises(ValueError):
        Rfam_get_family(family_id="RF00001", format="csv")
    fake = install_universe(
        tools={"Rfam_get_family": lambda args: {"family": {"id": args["family_id"]}}}
    )
    Rfam_get_family(family_id="RF00001", _live=True)
    assert fake.calls[0]["arguments"] == {
        "operation": "get_family",
        "family_id": "RF00001",
    }
    Rfam_get_family(family_id="RF00001", format="XML", _live=True)
    assert fake.calls[-1]["arguments"]["format"] == "xml"


# ── LNCipedia / miRBase ────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "search_fn,get_fn,search_name,get_name",
    [
        (LNCipedia_search_lncrna, LNCipedia_get_lncrna, "LNCipedia_search_lncrna", "LNCipedia_get_lncrna"),
        (miRBase_search_mirna, miRBase_get_mirna, "miRBase_search_mirna", "miRBase_get_mirna"),
    ],
)
def test_mirna_family_routes(install_universe, search_fn, get_fn, search_name, get_name):
    out = search_fn(query="malat1")
    assert out["status"] == "mocked"
    with pytest.raises(ValueError):
        search_fn()
    fake = install_universe(
        tools={
            search_name: lambda args: {"results": [{"id": "X"}]},
            get_name: lambda args: {"entry": {"id": args["rnacentral_id"]}},
        }
    )
    search_fn(query="malat1", _live=True)
    assert fake.calls[-1]["arguments"] == {"query": "malat1"}
    search_fn(query="malat1", species="human", size=20, _live=True)
    assert fake.calls[-1]["arguments"] == {
        "query": "malat1", "species": "human", "size": 20,
    }
    with pytest.raises(ValueError):
        get_fn()
    get_fn(rnacentral_id="URS1", _live=True)
    assert fake.calls[-1]["arguments"] == {"rnacentral_id": "URS1"}
    get_fn(rnacentral_id="URS1", taxid=9606, _live=True)
    assert fake.calls[-1]["arguments"] == {"rnacentral_id": "URS1", "taxid": 9606}


# ── dynamic_package_discovery + embedding_database_* — deferred ────────────


def test_utility_meta_deferred_raises_and_does_not_touch_universe(install_universe):
    fake = install_universe(
        tools={name: lambda args: {"silent": "success"} for name, _ in UTILITY_META_BINDINGS}
    )
    bindings = dict(UTILITY_META_BINDINGS)

    # dynamic_package_discovery: required arg validates; _live=True raises
    fn = bindings["dynamic_package_discovery"]
    with pytest.raises(ValueError):
        fn()
    out = fn(requirements="needs vector store")
    assert out["status"] == "mocked"
    with pytest.raises(NotImplementedError):
        fn(requirements="needs vector store", _live=True)

    # embedding_database_create
    fn = bindings["embedding_database_create"]
    with pytest.raises(ValueError):
        fn()
    with pytest.raises(ValueError):
        fn(database_name="kb")
    out = fn(database_name="kb", documents=["a", "b"])
    assert out["status"] == "mocked"
    with pytest.raises(NotImplementedError):
        fn(database_name="kb", documents=["a"], _live=True)

    # embedding_database_add
    fn = bindings["embedding_database_add"]
    with pytest.raises(ValueError):
        fn()
    with pytest.raises(ValueError):
        fn(database_name="kb")
    out = fn(database_name="kb", documents=["a"])
    assert out["status"] == "mocked"
    with pytest.raises(NotImplementedError):
        fn(database_name="kb", documents=["a"], _live=True)

    # embedding_database_search
    fn = bindings["embedding_database_search"]
    with pytest.raises(ValueError):
        fn()
    with pytest.raises(ValueError):
        fn(database_name="kb")
    out = fn(database_name="kb", query="EGFR")
    assert out["status"] == "mocked"
    with pytest.raises(NotImplementedError):
        fn(database_name="kb", query="EGFR", _live=True)

    # Across ALL deferred tools: zero dispatches to the fake universe.
    assert fake.calls == []
