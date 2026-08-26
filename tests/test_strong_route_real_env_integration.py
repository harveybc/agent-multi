"""Data-First order §3 acceptance: STRONG grouped route (PatchTST +
TFT + TimesNet + TCN/GRU + cross-family attention) constructed against
the REAL GymFxEnv observation — gradients to every branch, tiny-fixture
overfit through the full extractor, save/load bit parity, param report.

Entry points for the new plugins are injected via a load_plugin patch
until the branch merges and the editable install refreshes metadata
(declared in the return packet)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from pipeline_plugins.rl_pipeline_with_validation import (  # noqa: E402
    _load_env_plugin)
import agent_plugins.grouped_features_extractor as gfe  # noqa: E402
from agent_plugins.feature_families import (  # noqa: E402
    semantic_feature_families)

DATA = Path("/home/harveybc/Documents/GitHub/predictor/examples/data/"
            "project3/ethusdt_4h_tech_stat_full_model_ready.csv")
CONFIG = REPO / ("examples/config/"
                 "project3_ethusdt_4h_sac_grouped_features_v1.json")

_LOCAL = {
    ("feature_branch.plugins", "patchtst_branch"):
        "feature_branch_plugins.patchtst_branch",
    ("feature_branch.plugins", "tft_branch"):
        "feature_branch_plugins.tft_branch",
    ("feature_branch.plugins", "timesnet_branch"):
        "feature_branch_plugins.timesnet_branch",
    ("feature_branch.plugins", "tcn_branch"):
        "feature_branch_plugins.tcn_branch",
    ("feature_branch.plugins", "gru_branch"):
        "feature_branch_plugins.gru_branch",
    ("feature_branch.plugins", "mlp_branch"):
        "feature_branch_plugins.mlp_branch",
    ("feature_fusion.plugins", "cross_family_attention"):
        "feature_fusion_plugins.cross_family_attention",
}


@pytest.fixture(autouse=True)
def _inject_local_plugins(monkeypatch):
    import importlib
    real = gfe.load_plugin

    def patched(group, name):
        key = (group, name)
        if key in _LOCAL:
            mod = importlib.import_module(_LOCAL[key])
            plugin = mod.Plugin
            required = sorted(plugin.plugin_params)
            return plugin, required
        return real(group, name)

    monkeypatch.setattr(gfe, "load_plugin", patched)


@pytest.fixture(scope="module")
def real_env_and_arch(tmp_path_factory):
    if not DATA.is_file():
        pytest.skip("real ETH csv not present")
    cfg = json.loads(CONFIG.read_text())
    sliced = tmp_path_factory.mktemp("eth") / "eth_700.csv"
    with DATA.open() as src, sliced.open("w") as dst:
        for i, line in enumerate(src):
            if i > 700:
                break
            dst.write(line)
    cfg["input_data_file"] = str(sliced)
    cfg["max_steps"] = 460
    plug = _load_env_plugin("gym_fx_env", cfg)
    env = plug.make_env(cfg)
    cols = list(cfg["feature_columns"])
    fams = semantic_feature_families(cols)
    arch = {
        "schema": "agent_multi.grouped_features.v1",
        "feature_columns": cols,
        "branches": [
            {"name": "returns_momentum", "plugin": "patchtst_branch",
             "features": fams["returns_momentum"],
             "params": {"d_model": 32, "n_heads": 4, "n_layers": 1}},
            {"name": "trend_level", "plugin": "tft_branch",
             "features": fams["trend_level"],
             "params": {"hidden": 32, "n_heads": 4}},
            {"name": "volatility_distribution",
             "plugin": "timesnet_branch",
             "features": fams["volatility_distribution"],
             "params": {"d_model": 32, "top_k": 2}},
            {"name": "oscillators", "plugin": "tcn_branch",
             "features": fams["oscillators"],
             "params": {"channels": [32, 32]}},
            {"name": "volume_flow", "plugin": "gru_branch",
             "features": fams["volume_flow"]},
        ],
        "state_keys": [k for k in env.observation_space.spaces
                       if k != "features"],
        "state_branch": {"plugin": "mlp_branch",
                         "params": {"hidden_dims": [32],
                                    "output_dim": 16}},
        "fusion": {"plugin": "cross_family_attention",
                   "params": {"d_model": 32, "n_heads": 4,
                              "output_dim": 96}},
    }
    return env, arch


def _obs_tensors(env, batch=3, seed=7):
    obs, _ = env.reset(seed=seed)
    out = {}
    for k, v in obs.items():
        arr = np.asarray(v, dtype=np.float32)
        out[k] = torch.tensor(
            np.repeat(arr[None, ...], batch, axis=0))
    return out


def test_strong_route_constructs_and_every_branch_gets_gradient(
        real_env_and_arch):
    env, arch = real_env_and_arch
    Extractor = gfe.build_grouped_extractor_class()
    torch.manual_seed(0)
    model = Extractor(env.observation_space, arch)
    obs = _obs_tensors(env)
    out = model(obs)
    assert out.shape == (3, 96)
    out.sum().backward()
    for i, branch in enumerate(model.temporal_branches):
        norms = [p.grad.abs().sum().item()
                 for p in branch.parameters() if p.grad is not None]
        assert norms and sum(norms) > 0, f"branch {i} got no gradient"
    state_norms = [p.grad.abs().sum().item()
                   for p in model.state_branch.parameters()
                   if p.grad is not None]
    assert state_norms and sum(state_norms) > 0


def test_strong_route_tiny_fixture_overfit(real_env_and_arch):
    env, arch = real_env_and_arch
    Extractor = gfe.build_grouped_extractor_class()
    torch.manual_seed(1)
    model = Extractor(env.observation_space, arch)
    head = torch.nn.Linear(96, 1)
    # SIX DISTINCT real observations (identical inputs with different
    # targets would be unfittable by construction)
    o, _ = env.reset(seed=11)
    collected = [o]
    for _ in range(5):
        o, _r, term, _tr, _i = env.step([0.0])
        collected.append(o)
        if term:
            break
    obs = {k: torch.tensor(np.stack(
        [np.asarray(c[k], dtype=np.float32) for c in collected]))
        for k in collected[0]}
    target = torch.randn(len(collected), 1)
    opt = torch.optim.Adam(list(model.parameters())
                           + list(head.parameters()), lr=1e-3)
    first = last = None
    for _ in range(120):
        opt.zero_grad()
        loss = torch.nn.functional.mse_loss(head(model(obs)), target)
        first = float(loss) if first is None else first
        loss.backward()
        opt.step()
        last = float(loss)
    assert last < first * 0.3, (first, last)


def test_strong_route_save_load_bit_parity(real_env_and_arch, tmp_path):
    env, arch = real_env_and_arch
    Extractor = gfe.build_grouped_extractor_class()
    torch.manual_seed(2)
    model = Extractor(env.observation_space, arch)
    model.eval()
    obs = _obs_tensors(env)
    with torch.no_grad():
        ref = model(obs)
    torch.save(model.state_dict(), tmp_path / "strong.pt")
    model2 = Extractor(env.observation_space, arch)
    model2.load_state_dict(torch.load(tmp_path / "strong.pt",
                                      weights_only=True))
    model2.eval()
    with torch.no_grad():
        out = model2(obs)
    assert torch.equal(ref, out)


def test_strong_route_parameter_report(real_env_and_arch):
    env, arch = real_env_and_arch
    Extractor = gfe.build_grouped_extractor_class()
    model = Extractor(env.observation_space, arch)
    total = sum(p.numel() for p in model.parameters())
    per_branch = {arch["branches"][i]["name"]:
                  sum(p.numel() for p in b.parameters())
                  for i, b in enumerate(model.temporal_branches)}
    assert total > 0 and all(v > 0 for v in per_branch.values())
    print("PARAM_REPORT", {"total": total, **per_branch})
