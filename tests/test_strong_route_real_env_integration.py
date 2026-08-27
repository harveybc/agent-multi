"""Data-First order §3 acceptance: STRONG grouped route against the
REAL GymFxEnv (DATA-SOTA-333 corrected):

- the ETH csv path is a PORTABLE contract: env AGENT_MULTI_ETH_CSV,
  falling back to the conventional sibling-checkout location;
- Tier-A mode (env TIER_A=1): data absence is a FAILURE, never a skip;
  without TIER_A the module skips explicitly as the optional variant;
- plugins resolve through REAL entry points (editable metadata) — no
  loader monkeypatch.
"""
from __future__ import annotations

import json
import os
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


def test_new_plugins_discoverable_through_real_entry_points():
    from importlib.metadata import entry_points
    names = {e.name for e in entry_points().select(
        group="feature_branch.plugins")}
    assert {"patchtst_branch", "tft_branch",
            "timesnet_branch"} <= names
    fusions = {e.name for e in entry_points().select(
        group="feature_fusion.plugins")}
    assert "cross_family_attention" in fusions
from agent_plugins.feature_families import (  # noqa: E402
    semantic_feature_families)

def _eth_csv() -> Path:
    override = os.environ.get("AGENT_MULTI_ETH_CSV")
    if override:
        return Path(override)
    return (REPO.parent.parent / "predictor/examples/data/project3/"
            "ethusdt_4h_tech_stat_full_model_ready.csv")


DATA = _eth_csv()
TIER_A = os.environ.get("TIER_A") == "1"
CONFIG = REPO / ("examples/config/"
                 "project3_ethusdt_4h_sac_grouped_features_v1.json")

@pytest.fixture(scope="module")
def real_env_and_arch(tmp_path_factory):
    if not DATA.is_file():
        if TIER_A:
            pytest.fail(f"TIER_A: real ETH csv REQUIRED at {DATA} "
                        f"(set AGENT_MULTI_ETH_CSV)")
        pytest.skip("optional variant: real ETH csv not present "
                    "(Tier-A command sets TIER_A=1)")
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


def test_pretrain_windows_bitwise_match_real_env_observations(
        real_env_and_arch):
    """DATA-SOTA-342: the WP-PRETRAIN collector and the REAL GymFxEnv
    emit the SAME feature tensor at the same bar — bitwise, per family.
    The collector consumes the runner's own fit-slice dataframe, so this
    also proves the runner's CSV parsing matches the env's loader."""
    env, arch = real_env_and_arch
    from agent_plugins.branch_pretraining import (
        collect_preprocessed_windows, load_fit_slice)

    cfg = json.loads(CONFIG.read_text())
    contract = json.loads(
        (REPO / "examples/config/"
         "pretrain_contract_eth_h4_o2022_v4.json").read_text())
    # the sliced fixture covers 2017 bars only; keep every row while
    # preserving the causal-origin validator (fit_end << score_start)
    contract["fit_end"] = "2018-06-30T00:00:00"
    sliced_csv = Path(env.config["input_data_file"])
    df, cols, _close = load_fit_slice(sliced_csv, contract)
    assert cols == list(cfg["feature_columns"])

    obs, _ = env.reset(seed=7)
    observed = [(int(env.bridge.bar_index),
                 np.asarray(obs["features"], dtype=np.float32))]
    for _ in range(4):
        obs, _r, term, _tr, _i = env.step([0.0])
        observed.append((int(env.bridge.bar_index),
                         np.asarray(obs["features"], dtype=np.float32)))
        if term:
            break
    steps = [t for t, _ in observed]
    windows = collect_preprocessed_windows(df, contract, cfg, steps)
    fams = semantic_feature_families(cols)
    for i, (t, env_features) in enumerate(observed):
        assert np.array_equal(windows[i], env_features), (
            f"bar {t}: pretraining tensor diverges from the executing "
            f"env observation")
        for family, members in fams.items():
            idx = [cols.index(m) for m in members]
            assert np.array_equal(windows[i][:, idx],
                                  env_features[:, idx]), (
                f"bar {t}, family {family}: tensors diverge")
