"""Acceptance tests for the strong-route branches and cross-family
fusion (Data-First order §3): shapes, causal masks, per-step causal
functionality, gradients, tiny-fixture overfit, save/load parity."""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from feature_branch_plugins.patchtst_branch import Plugin as PatchTST
from feature_branch_plugins.tft_branch import Plugin as TFT
from feature_branch_plugins.timesnet_branch import Plugin as TimesNet
from feature_fusion_plugins.cross_family_attention import (
    Plugin as CrossAttn)

WINDOW, FEATS, BATCH = 32, 7, 4


def _build(plugin, **over):
    params = dict(plugin.plugin_params)
    params.update(over)
    module, dim = plugin.build(FEATS, WINDOW, params)
    return module, dim, params


@pytest.mark.parametrize("plugin", [PatchTST, TFT, TimesNet])
def test_branch_shape_contract(plugin):
    torch.manual_seed(0)
    module, dim, _ = _build(plugin)
    out = module(torch.randn(BATCH, WINDOW, FEATS))
    assert out.shape == (BATCH, dim)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("plugin", [PatchTST, TFT])
def test_causal_mask_is_upper_triangular_neg_inf(plugin):
    module, _dim, _ = _build(plugin)
    mask = module.causal_mask
    n = mask.shape[0]
    upper = torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)
    assert torch.isinf(mask[upper]).all() and (mask[upper] < 0).all()
    assert (mask[~upper] == 0).all()


def test_tft_per_step_causality_functional():
    # step-k internal output must be invariant to mutations at steps>k
    torch.manual_seed(1)
    module, _dim, _ = _build(TFT)
    module.eval()
    x = torch.randn(1, WINDOW, FEATS)
    x2 = x.clone()
    x2[:, WINDOW // 2 + 1:, :] += 5.0
    with torch.no_grad():
        w1 = torch.softmax(module.var_grn(x), -1)
        e1 = (w1.unsqueeze(-1) * module.feat_embed(
            x.unsqueeze(-1))).sum(2)
        s1 = module.post_select(e1)
        c1, _ = module.gru(s1)
        a1, _ = module.attn(c1, c1, c1, attn_mask=module.causal_mask,
                            need_weights=False)
        w2 = torch.softmax(module.var_grn(x2), -1)
        e2 = (w2.unsqueeze(-1) * module.feat_embed(
            x2.unsqueeze(-1))).sum(2)
        s2 = module.post_select(e2)
        c2, _ = module.gru(s2)
        a2, _ = module.attn(c2, c2, c2, attn_mask=module.causal_mask,
                            need_weights=False)
    k = WINDOW // 2
    assert torch.allclose(a1[:, :k + 1], a2[:, :k + 1], atol=1e-6)


def test_patchtst_per_patch_causality_functional():
    torch.manual_seed(2)
    module, _dim, params = _build(PatchTST)
    module.eval()
    x = torch.randn(1, WINDOW, FEATS)
    x2 = x.clone()
    x2[:, -params["patch_len"]:, :] += 5.0   # mutate ONLY final patch
    with torch.no_grad():
        def tokens(v):
            p = v.permute(0, 2, 1).unfold(-1, params["patch_len"],
                                          params["stride"])
            tok = module.embed(p) + module.pos
            return module.encoder(
                tok.reshape(-1, tok.shape[2], tok.shape[3]),
                mask=module.causal_mask)
        t1, t2 = tokens(x), tokens(x2)
    assert torch.allclose(t1[:, :-1], t2[:, :-1], atol=1e-6)
    assert not torch.allclose(t1[:, -1], t2[:, -1])


@pytest.mark.parametrize("plugin", [PatchTST, TFT, TimesNet])
def test_branch_gradients_flow(plugin):
    torch.manual_seed(3)
    module, _dim, _ = _build(plugin)
    out = module(torch.randn(BATCH, WINDOW, FEATS))
    out.sum().backward()
    grads = [p.grad.abs().sum().item() for p in module.parameters()
             if p.grad is not None]
    assert grads and sum(grads) > 0


def test_cross_family_attention_fuses_and_grads():
    torch.manual_seed(4)
    dims = [64, 64, 32]
    fusion, out_dim = CrossAttn.build(
        dims, dict(CrossAttn.plugin_params,
                   family_ids=["ret", "trend", "vol"]))
    encoded = [torch.randn(BATCH, d, requires_grad=True) for d in dims]
    named = list(zip(["ret", "trend", "vol"], encoded))
    out = fusion(named)
    assert out.shape == (BATCH, out_dim)
    out.sum().backward()
    for e in encoded:
        assert e.grad is not None and e.grad.abs().sum() > 0
    assert len(fusion.family_digest) == 64


@pytest.mark.parametrize("plugin", [PatchTST, TFT])
def test_tiny_fixture_overfit(plugin):
    torch.manual_seed(5)
    module, dim, _ = _build(plugin)
    head = torch.nn.Linear(dim, 1)
    x = torch.randn(8, WINDOW, FEATS)
    y = torch.randn(8, 1)
    opt = torch.optim.Adam(
        list(module.parameters()) + list(head.parameters()), lr=1e-3)
    first = last = None
    for step in range(200):
        opt.zero_grad()
        loss = torch.nn.functional.mse_loss(head(module(x)), y)
        if first is None:
            first = float(loss)
        loss.backward()
        opt.step()
        last = float(loss)
    assert last < first * 0.2, (first, last)


@pytest.mark.parametrize("plugin", [PatchTST, TFT, TimesNet])
def test_save_load_bitwise_output_parity(plugin, tmp_path):
    torch.manual_seed(6)
    module, _dim, params = _build(plugin)
    module.eval()
    x = torch.randn(BATCH, WINDOW, FEATS)
    with torch.no_grad():
        ref = module(x)
    torch.save(module.state_dict(), tmp_path / "m.pt")
    module2, _d2, _ = _build(plugin)
    module2.load_state_dict(torch.load(tmp_path / "m.pt",
                                       weights_only=True))
    module2.eval()
    with torch.no_grad():
        out = module2(x)
    assert torch.equal(ref, out)   # bit-level parity


def test_timesnet_deterministic_eval():
    torch.manual_seed(7)
    module, _dim, _ = _build(TimesNet)
    module.eval()
    x = torch.randn(BATCH, WINDOW, FEATS)
    with torch.no_grad():
        a, b = module(x), module(x)
    assert torch.equal(a, b)
