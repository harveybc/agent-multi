"""Permanent regressions for DATA-SOTA-329..334 — each auditor
counterexample, frozen."""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

REPO = Path(__file__).resolve().parents[2]

from feature_branch_plugins._topology import TopologyError  # noqa: E402
from feature_branch_plugins.patchtst_branch import (  # noqa: E402
    Plugin as PatchTST)
from feature_branch_plugins.tft_branch import Plugin as TFT  # noqa: E402
from feature_branch_plugins.timesnet_branch import (  # noqa: E402
    Plugin as TimesNet)
from feature_fusion_plugins.cross_family_attention import (  # noqa: E402
    Plugin as CrossAttn)

INVENTORY = REPO / "docs/audits/evidence/DATA_INVENTORY_V1.json"


# --- 329: typed live states, no measured-fact impersonation -------------

def test_329_live_claims_are_typed_unverified():
    inv = json.loads(INVENTORY.read_text())
    fields = inv["current_contract_83"]["fields"]
    for name, f in fields.items():
        assert f["historical_status"] == "HISTORICAL_MEASURED"
        assert f["live_status"] == "LIVE_DERIVABLE_UNVERIFIED", name
        assert f["v3_eligible"] is False, name
        assert f["publication_delay"]["status"] == (
            "UNVERIFIED_UNTIL_MEASURED")
        for venue in f["live_venues"].values():
            assert venue["freshness"] == "UNKNOWN_UNTIL_COLLECTOR"


# --- 330: committed artifact carries no filesystem topology -------------

def test_330_inventory_has_no_local_topology():
    body = INVENTORY.read_text()
    for needle in ("/home/", ".local/state", ".local/share",
                   "omega", "dragon", "gamma", "harveybc"):
        assert needle not in body, f"leak: {needle}"


# --- 331: endpoint coverage — newest bar ALWAYS matters -----------------

@pytest.mark.parametrize("window,patch,stride", [
    (w, p, s)
    for w in (8, 13, 21, 32)
    for p in (2, 4, 8)
    for s in (1, 3, 5, p)
    if p <= w
])
def test_331_final_bar_always_enters_output(window, patch, stride):
    torch.manual_seed(0)
    params = dict(PatchTST.plugin_params, patch_len=patch,
                  stride=stride, d_model=16, n_heads=2, n_layers=1)
    module, _ = PatchTST.build(3, window, params)
    module.eval()
    x = torch.randn(1, window, 3)
    x2 = x.clone()
    x2[:, -1, :] += 100.0
    with torch.no_grad():
        assert not torch.equal(module(x), module(x2)), (
            f"newest bar ignored at ({window},{patch},{stride})")


def test_331_auditor_counterexample_stride5():
    params = dict(PatchTST.plugin_params, patch_len=8, stride=5)
    module, _ = PatchTST.build(4, 32, params)
    module.eval()
    x = torch.randn(1, 32, 4)
    x2 = x.clone()
    x2[:, -1, :] += 100.0
    with torch.no_grad():
        assert not torch.equal(module(x), module(x2))


# --- 332: fusion refuses count/width drift ------------------------------

def _fusion(dims, families=("a", "b", "c")):
    fam = list(families)[:len(dims)]
    params = dict(CrossAttn.plugin_params, d_model=16, n_heads=2,
                  output_dim=24, family_ids=fam)
    return CrossAttn.build(dims, params)[0], fam


def _named(fam, tensors):
    return list(zip(fam, tensors))


def test_332_extra_branch_refused():
    f, fam = _fusion([8, 8, 8])
    enc = _named(fam + ["d"], [torch.randn(2, 8) for _ in range(4)])
    with pytest.raises(ValueError, match="expected 3 named branches"):
        f(enc)


def test_332_missing_branch_refused():
    f, fam = _fusion([8, 8, 8])
    with pytest.raises(ValueError, match="expected 3 named branches"):
        f(_named(fam[:2], [torch.randn(2, 8), torch.randn(2, 8)]))


def test_332_wrong_width_refused_with_family_name():
    f, fam = _fusion([8, 16, 8])
    enc = _named(fam, [torch.randn(2, 8), torch.randn(2, 8),
                       torch.randn(2, 8)])
    with pytest.raises(ValueError, match="'b' must be \\(B, 16\\)"):
        f(enc)


def test_332_swapped_widths_refused():
    f, fam = _fusion([8, 16, 8])
    enc = _named(fam, [torch.randn(2, 16), torch.randn(2, 8),
                       torch.randn(2, 8)])
    with pytest.raises(ValueError):
        f(enc)


def test_332_family_ids_bound():
    f, _fam = _fusion([8, 8], families=("returns", "trend"))
    assert f.family_ids == ["returns", "trend"]
    assert len(f.family_digest) == 64


# --- 334: degenerate topology domains refuse ----------------------------

def test_334_timesnet_window1_refused():
    with pytest.raises(TopologyError, match="window_size must be >= 4"):
        TimesNet.build(4, 1, dict(TimesNet.plugin_params))


def test_334_timesnet_even_kernel_refused():
    with pytest.raises(TopologyError, match="ODD"):
        TimesNet.build(4, 16, dict(TimesNet.plugin_params, kernel=4))


def test_334_timesnet_topk_exceeds_bins_refused():
    with pytest.raises(TopologyError, match="spectral bins"):
        TimesNet.build(4, 8, dict(TimesNet.plugin_params, top_k=99))


@pytest.mark.parametrize("plugin,bad", [
    (PatchTST, {"d_model": 30, "n_heads": 4}),
    (TFT, {"hidden": 30, "n_heads": 4}),
    (CrossAttn, {"d_model": 30, "n_heads": 4}),
])
def test_334_head_divisibility_refused(plugin, bad):
    params = dict(plugin.plugin_params, **bad)
    with pytest.raises(TopologyError, match="divisible"):
        if plugin is CrossAttn:
            plugin.build([8, 8], params)
        else:
            plugin.build(4, 32, params)


@pytest.mark.parametrize("plugin", [PatchTST, TFT, TimesNet])
def test_334_bad_dropout_refused(plugin):
    with pytest.raises(TopologyError, match="\\[0, 1\\)"):
        plugin.build(4, 32, dict(plugin.plugin_params, dropout=1.5))


def test_334_gene_range_property_all_valid_cells_construct():
    # the DECLARED gene ranges must all construct (no dead cells)
    for d_model, heads, k in itertools.product((16, 32), (2, 4),
                                               (3, 5)):
        m, dim = TimesNet.build(4, 32, dict(
            TimesNet.plugin_params, d_model=d_model, kernel=k))
        assert dim == d_model
        m2, _ = PatchTST.build(4, 32, dict(
            PatchTST.plugin_params, d_model=d_model, n_heads=heads,
            n_layers=1))
        m3, _ = TFT.build(4, 32, dict(
            TFT.plugin_params, hidden=d_model, n_heads=heads))
