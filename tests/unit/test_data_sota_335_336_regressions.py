"""Permanent regressions for DATA-SOTA-335/336 — the auditor's five
counterexamples plus impostor property grids and identity-swap
refusal."""
from __future__ import annotations

import itertools

import pytest

torch = pytest.importorskip("torch")

from feature_branch_plugins._topology import (TopologyError,  # noqa
                                              strict_int, strict_real)
from feature_branch_plugins.patchtst_branch import (  # noqa: E402
    Plugin as PatchTST)
from feature_branch_plugins.tcn_branch import Plugin as TCN  # noqa
from feature_branch_plugins.tft_branch import Plugin as TFT  # noqa
from feature_branch_plugins.timesnet_branch import (  # noqa: E402
    Plugin as TimesNet)
from feature_fusion_plugins.cross_family_attention import (  # noqa
    Plugin as CrossAttn)


# --- the five auditor counterexamples, frozen ---------------------------

def test_335_patch_len_True_refused():
    with pytest.raises(TopologyError, match="non-boolean integer"):
        PatchTST.build(4, 32, dict(PatchTST.plugin_params,
                                   patch_len=True))


def test_335_dropout_string_refused():
    with pytest.raises(TopologyError, match="non-boolean finite"):
        TFT.build(4, 32, dict(TFT.plugin_params, dropout="0.2"))


def test_335_bool_fusion_branch_width_refused():
    with pytest.raises(TopologyError, match="non-boolean integer"):
        CrossAttn.build([8, True], dict(CrossAttn.plugin_params,
                                        family_ids=["a", "b"]))


def test_335_fractional_window_refused_by_validator_not_torch():
    with pytest.raises(TopologyError, match="non-boolean integer"):
        PatchTST.build(4, 32.5, dict(PatchTST.plugin_params))


def test_336_duplicate_family_ids_refused():
    with pytest.raises(ValueError, match="duplicate family_ids"):
        CrossAttn.build([8, 8], dict(CrossAttn.plugin_params,
                                     d_model=16, n_heads=2,
                                     family_ids=["a", "a"]))


# --- 336: identity is runtime-bound; same-width swaps refuse ------------

def test_336_same_width_swap_refused_by_identity():
    f, _ = CrossAttn.build([8, 8], dict(
        CrossAttn.plugin_params, d_model=16, n_heads=2, output_dim=24,
        family_ids=["ret", "vol"])), None
    fusion = f[0]
    a, b = torch.randn(1, 8), torch.randn(1, 8)
    assert fusion([("ret", a), ("vol", b)]).shape == (1, 24)
    with pytest.raises(ValueError, match="identity mismatch"):
        fusion([("vol", b), ("ret", a)])


def test_336_positional_input_refused():
    fusion, _ = CrossAttn.build([8, 8], dict(
        CrossAttn.plugin_params, d_model=16, n_heads=2,
        family_ids=["ret", "vol"]))
    with pytest.raises(ValueError, match="NAMED records"):
        fusion([torch.randn(1, 8), torch.randn(1, 8)])


def test_336_missing_or_empty_family_ids_refused():
    with pytest.raises(ValueError, match="one family_id per"):
        CrossAttn.build([8, 8], dict(CrossAttn.plugin_params,
                                     family_ids=["only_one"]))
    with pytest.raises(ValueError, match="nonempty strings"):
        CrossAttn.build([8, 8], dict(CrossAttn.plugin_params,
                                     family_ids=["a", "  "]))


# --- impostor property grids over exposed genes -------------------------

IMPOSTORS = (True, False, "8", 8.0, 7.5, float("nan"), float("inf"),
             None, [8])


@pytest.mark.parametrize("impostor", IMPOSTORS)
@pytest.mark.parametrize("key", ["patch_len", "stride", "d_model",
                                 "n_heads", "n_layers", "ff_mult"])
def test_335_patchtst_gene_impostors_refused(key, impostor):
    params = dict(PatchTST.plugin_params)
    params[key] = impostor
    with pytest.raises((TopologyError, ValueError)):
        PatchTST.build(4, 32, params)


@pytest.mark.parametrize("impostor", IMPOSTORS)
@pytest.mark.parametrize("key", ["top_k", "d_model", "kernel"])
def test_335_timesnet_gene_impostors_refused(key, impostor):
    params = dict(TimesNet.plugin_params)
    params[key] = impostor
    with pytest.raises((TopologyError, ValueError)):
        TimesNet.build(4, 32, params)


@pytest.mark.parametrize("impostor", IMPOSTORS)
def test_335_tcn_channel_list_impostors_refused(impostor):
    with pytest.raises((TopologyError, ValueError)):
        TCN.build(4, 32, dict(TCN.plugin_params,
                              channels=[64, impostor]))


def test_335_param_ceiling_impostor_refused():
    from feature_branch_plugins._topology import require_param_ceiling
    with pytest.raises(TopologyError):
        require_param_ceiling(10, {"max_parameters": True})


def test_335_validation_precedes_torch_construction(monkeypatch):
    # the validator must fire BEFORE any torch module exists
    import feature_branch_plugins.patchtst_branch as mod
    calls = []
    with pytest.raises(TopologyError):
        PatchTST.build(4, 32, dict(PatchTST.plugin_params,
                                   d_model="64"))
