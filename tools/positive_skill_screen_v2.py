#!/usr/bin/env python3
"""Positive-skill + fusion screen v2 under the OBSERVABLE RESUMABLE
runtime (Musashi correction 1, 2026-09-03; PERMANENT order @95e088da;
predeclared in POSITIVE_SKILL_SCREEN_PREDECLARATION_V2_2026_09_03.json).
REPRESENTATION_DIAGNOSTIC.

Three layers, per the permanent order:

  materialize  build the prospective ledger of ATOMIC units for one
               phase (round1/round2/round3/survivors/fusion) plus
               immutable digest-identified npz inputs; halving
               decisions are separate persisted artifacts and later
               rounds refuse to materialize without them.
  worker       execute EXACTLY ONE unit and exit (atomic result,
               durable per-unit log, SIGTERM -> INTERRUPTED).
  supervise    schedule workers, heartbeat <= 60 s, watchdog,
               per-unit timeouts from the benchmark, campaign wall
               ceiling, machine-readable status, graceful cancel,
               idempotent resume; chains phases automatically because
               halving is mechanical and predeclared.
  status       print machine-readable status (no process attachment).
  aggregate    build the final report from COMPLETE verified units
               only; missing/duplicate/foreign units refuse.
  benchmark    one CPU + one CUDA cell -> measured device choice and
               per-unit timeout. No full screen without it.

The retired monolithic executors (tools/positive_skill_screen.py,
tools/fusion_screen.py) are imported ONLY for pure scientific
functions; executing them as runners remains prohibited."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from agent_plugins.experiment_runtime import (  # noqa: E402
    RunDirectory, RuntimePreflightError, UnitClaimError,
    aggregate as runtime_aggregate, atomic_write_json,
    preflight_or_refuse, run_one_unit, sha_file, sha_obj, unit_id)

PREDECLARATION = (
    "docs/audits/evidence/"
    "POSITIVE_SKILL_SCREEN_PREDECLARATION_V2_2026_09_03.json")
EXPERIMENT = "positive_skill_screen_v2"
WINDOWS = (32, 64, 128, 256)
LATENTS = (16, 32, 64, 96, 128)
FAMILIES = ("returns_momentum", "trend_level",
            "volatility_distribution", "oscillators", "volume_flow")
SEEDS = (101, 202, 303, 404)
ORIGINS = (0, 1)
BUDGETS = {"round1": 300, "round2": 900, "round3": 2700}
MIN_EFFECT = 0.02
DEGRADE_TOLERANCE = 0.02
FUSION_SEED_VARIANTS = ("current_fusion_probe_trained",
                        "gated_softmax_probe_trained",
                        "fine_tuned_branches_plus_fusion")
PHASES = ("round1", "round2", "round3", "survivors", "fusion")


_RETIRED_CACHE: dict = {}


def _load_retired(name: str):
    """Import a retired monolith for its PURE functions only."""
    if name not in _RETIRED_CACHE:
        path = REPO / "tools" / f"{name}.py"
        spec = importlib.util.spec_from_file_location(
            f"retired_{name}", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _RETIRED_CACHE[name] = module
    return _RETIRED_CACHE[name]


def train_cell_device(module, latent, windows, target, fit_i, cal_i,
                      mon_i, budget, seed, *, head_only=False,
                      device="cpu"):
    """The EXACT science of the retired train_cell with an explicit
    device. test_v2_train_cell_matches_retired asserts bit-equality
    with the retired CPU implementation, so the maths cannot drift;
    the CUDA benchmark measures THIS code, not a stand-in."""
    import torch
    dev = torch.device(device)
    torch.manual_seed(seed)
    module = module.to(dev)
    head = torch.nn.Linear(latent, 1).to(dev)
    params = list(head.parameters()) + (
        [] if head_only else list(module.parameters()))
    optimizer = torch.optim.Adam(params, lr=1e-3)
    x = torch.tensor(windows, dtype=torch.float32, device=dev)
    y = torch.tensor(target, dtype=torch.float32,
                     device=dev).unsqueeze(-1)
    fit_t = torch.tensor(fit_i)
    generator = torch.Generator().manual_seed(seed)
    science = _science()
    best_cal, best_state = -1e18, None
    start = time.perf_counter()
    module.train()
    for update in range(budget):
        batch_size = 128 if windows.shape[1] >= 128 else 256
        batch_idx = fit_t[torch.randint(
            0, len(fit_t), (min(batch_size, len(fit_t)),),
            generator=generator)].to(dev)
        optimizer.zero_grad()
        out = head(module(x[batch_idx]))
        loss = torch.nn.functional.mse_loss(out, y[batch_idx])
        loss.backward()
        optimizer.step()
        if (update + 1) % 50 == 0 or update + 1 == budget:
            module.eval()
            with torch.no_grad():
                cal_pred = head(module(x[cal_i])).squeeze(
                    -1).cpu().numpy()
            module.train()
            cal_score = science.r2(cal_pred, target[cal_i])
            if cal_score > best_cal:
                best_cal = cal_score
                best_state = (
                    {k: v.detach().clone()
                     for k, v in module.state_dict().items()},
                    {k: v.detach().clone()
                     for k, v in head.state_dict().items()})
    if best_state is not None:
        module.load_state_dict(best_state[0])
        head.load_state_dict(best_state[1])
    module.eval()
    with torch.no_grad():
        mon_pred = head(module(x[mon_i])).squeeze(-1).cpu().numpy()
    wall = time.perf_counter() - start
    n_params = sum(p.numel() for p in module.parameters()) + \
        sum(p.numel() for p in head.parameters())
    return (science.r2(mon_pred, target[mon_i]), best_cal, n_params,
            round(wall, 1))


def code_digest() -> str:
    files = [REPO / "tools/positive_skill_screen_v2.py",
             REPO / "tools/positive_skill_screen.py",
             REPO / "agent_plugins/experiment_runtime.py",
             REPO / "agent_plugins/branch_pretraining.py",
             REPO / "agent_plugins/temporal_information.py",
             REPO / "agent_plugins/pretrained_branch_loader.py"]
    return sha_obj({str(f.relative_to(REPO)): sha_file(f)
                    for f in files})


# ------------------------------------------------------------------ #
# shared data plumbing (materializer + workers)                      #
# ------------------------------------------------------------------ #

def _science():
    return _load_retired("positive_skill_screen")


def _contract_bundle(pretrain_dir: Path):
    from agent_plugins.branch_pretraining import (
        load_fit_slice, validate_contract)
    from agent_plugins.pretrained_branch_loader import verify_source
    split_contract = json.loads(
        (REPO / "examples/config/phase_3_eth_sac_dynamics/splits/"
         "eth_nested_split_contract_o2022_paired_v1.json").read_text())
    data_path = Path(split_contract["source_csv"])
    source = verify_source(pretrain_dir, REPO, data_path)
    contract = source["contract"]
    parsed = validate_contract(contract)
    df, ordered, close_col = load_fit_slice(data_path, contract)
    return (source, contract, parsed, df, ordered, close_col,
            data_path)


def _window_npz(run_root: Path, window: int) -> Path:
    return run_root / "inputs" / f"windows_w{window}.npz"


def materialize_inputs(run_root: Path, pretrain_dir: Path,
                       *, max_windows: int, stride: int) -> dict:
    """Immutable digest-identified inputs OUTSIDE /tmp: per-window
    npz with (windows, target, trailing) and the fusion npz with
    frozen branch embeddings + the three targets."""
    import numpy as np
    import torch
    from agent_plugins.branch_pretraining import (
        barrier_hit_labels, build_step_index,
        collect_preprocessed_windows, forward_log_return_targets,
        realized_volatility_targets)
    from agent_plugins.component_config import deep_merge_strict
    from agent_plugins.pretrained_branch_loader import (
        strict_load_encoder)
    from agent_plugins.temporal_information import encode_windows
    from app.plugin_loader import load_plugin

    (source, contract, parsed, df, ordered, close_col,
     data_path) = _contract_bundle(pretrain_dir)
    closes = df[close_col].to_numpy()
    vol_spec = contract["objectives"]["volatility"]
    annualization = vol_spec.get("annualization")
    periods = (None if annualization in (None, "none")
               else annualization["periods_per_year"])
    env_source = contract["observation_pipeline"]["source_config"]
    env_config_base = json.loads(
        (Path(env_source) if Path(env_source).is_absolute()
         else REPO / env_source).read_text())
    (run_root / "inputs").mkdir(parents=True, exist_ok=True)

    digests = {}
    for window in WINDOWS:
        path = _window_npz(run_root, window)
        if not path.exists():
            warmup = max(int(parsed["warmup_bars"]), window)
            steps = build_step_index(len(df), warmup, max(1, stride),
                                     12, max_windows)
            env_config = {**env_config_base, "window_size": window}
            contract_w = {**contract, "window_size": window}
            windows = collect_preprocessed_windows(
                df, contract_w, env_config, steps)
            target = realized_volatility_targets(
                closes, steps, [6],
                float(vol_spec.get("epsilon", 1e-8)), periods)[:, 0]
            trailing = realized_volatility_targets(
                closes, [max(0, t - 6) for t in steps], [6],
                float(vol_spec.get("epsilon", 1e-8)), periods)[:, 0]
            keep = (np.isfinite(target) & np.isfinite(trailing))
            np.savez_compressed(
                path, windows=windows[keep].astype("float32"),
                target=np.asarray(target)[keep].astype("float64"),
                trailing=np.asarray(trailing)[keep].astype("float64"))
        digests[f"input_w{window}"] = sha_file(path)

    fusion_path = run_root / "inputs" / "fusion_inputs.npz"
    if not fusion_path.exists():
        manifest = source["manifest"]
        window = int(parsed["window_size"])
        steps = build_step_index(len(df), parsed["warmup_bars"],
                                 max(1, stride), 12, max_windows)
        env_config = {**env_config_base, "window_size": window}
        windows = collect_preprocessed_windows(
            df, contract, env_config, steps)
        y_vol = np.asarray(realized_volatility_targets(
            closes, steps, [6], float(vol_spec.get("epsilon", 1e-8)),
            periods)[:, 0], float)
        # NOTE: the retired fusion monolith read a nonexistent
        # "quantile_regression" key — a latent bug it never hit
        # because it never executed; the real key is
        # multi_horizon_quantile (disclosed in the return packet)
        quant_h = list(
            contract["objectives"]["multi_horizon_quantile"][
                "horizons"])
        h_col = min(range(len(quant_h)),
                    key=lambda i: abs(quant_h[i] - 6))
        y_ret = np.asarray(forward_log_return_targets(
            closes, steps, quant_h)[:, h_col], float)
        barrier_spec = contract["objectives"]["barrier_hit"]
        ohlc = barrier_spec["ohlc_columns"]
        y_bar = (np.asarray(barrier_hit_labels(
            df[ohlc["open"]].to_numpy(), df[ohlc["high"]].to_numpy(),
            df[ohlc["low"]].to_numpy(), df[ohlc["close"]].to_numpy(),
            steps, list(barrier_spec["horizons"]),
            int(barrier_spec["barrier_scale"]["lookback"]),
            float(barrier_spec["upper_mult"]),
            float(barrier_spec["lower_mult"]),
            float(barrier_spec["barrier_scale"]["epsilon"])))[:, 0]
            == 0).astype(float)
        arrays = {"windows": windows.astype("float32"),
                  "y_vol": y_vol, "y_ret": y_ret, "y_bar": y_bar}
        dims = []
        for branch in contract["branches"]:
            plugin_class, _ = load_plugin("feature_branch.plugins",
                                          branch["plugin"])
            params = deep_merge_strict(plugin_class.plugin_params,
                                       branch["params"], path="p")
            torch.manual_seed(0)
            module, _dim = plugin_class.build(
                len(branch["features"]), window, params)
            entry = manifest["artifacts"][branch["name"]]
            state = torch.load(pretrain_dir / entry["encoder_file"],
                               weights_only=True)
            strict_load_encoder(module, state, branch["name"])
            module.eval()
            cols = [ordered.index(f) for f in branch["features"]]
            emb = encode_windows(module, windows[:, :, cols])
            arrays[f"emb_{branch['name']}"] = emb
            dims.append(int(emb.shape[1]))
        arrays["dims"] = np.asarray(dims)
        np.savez_compressed(fusion_path, **arrays)
    digests["input_fusion"] = sha_file(fusion_path)
    digests["data_csv"] = sha_file(data_path)
    return digests


def _family_columns(pretrain_dir: Path) -> dict:
    (_s, contract, _p, _df, ordered, _c,
     _d) = _contract_bundle(pretrain_dir)
    return {b["name"]: {
        "plugin": b["plugin"],
        "cols": [ordered.index(f) for f in b["features"]],
        "n_features": len(b["features"])}
        for b in contract["branches"]}


# ------------------------------------------------------------------ #
# materializer                                                       #
# ------------------------------------------------------------------ #

def _ledger_for(units: list, digests: dict, *, wall_ceiling_s: float,
                unit_timeout_s: float | None, extras: dict) -> dict:
    return {"schema": "agent_multi.screen_v2_ledger.v1",
            "experiment": EXPERIMENT,
            "units": [{"unit_id": unit_id(u), "identity": u}
                      for u in units],
            "digests": digests,
            "campaign_wall_ceiling_s": wall_ceiling_s,
            "unit_timeout_s": unit_timeout_s,
            **extras}


def _identity(family: str, window: int, latent: int, budget: int,
              seed: int, origin: int, treatment: str) -> dict:
    return {"experiment": EXPERIMENT, "family": family,
            "window": window, "latent": latent, "budget": budget,
            "seed": seed, "origin": origin, "treatment": treatment}


def materialize_phase(root: Path, phase: str, pretrain_dir: Path,
                      *, unit_timeout_s: float | None,
                      wall_ceiling_s: float, max_windows: int,
                      stride: int) -> dict:
    # cheap structural refusals BEFORE any data touch: later rounds
    # cannot exist without the persisted decision of the previous one
    if phase in ("round2", "round3"):
        prev = {"round2": "round1", "round3": "round2"}[phase]
        if not (root / prev / "decisions" / "halving.json").exists():
            raise RuntimePreflightError(
                f"{phase}: halving decision of {prev} absent — "
                "later rounds refuse to materialize without the "
                "persisted decision artifact")
    if phase == "survivors" and not (
            root / "round3" / "decisions" / "halving.json").exists():
        raise RuntimePreflightError(
            "survivors: round3 completion decision absent")
    science = _science()
    digests = materialize_inputs(root, pretrain_dir,
                                 max_windows=max_windows,
                                 stride=stride)
    digests = {**digests, "code": code_digest(),
               "config": sha_file(REPO / PREDECLARATION),
               "pretrain_generation": sha_file(
                   Path(pretrain_dir) / "generation.json")}
    fam_meta = _family_columns(pretrain_dir)
    units, invalid = [], []

    def valid_cells():
        for family in FAMILIES:
            plugin = fam_meta[family]["plugin"]
            for window in WINDOWS:
                for latent in LATENTS:
                    if science.cell_params(plugin, latent) is None:
                        invalid.append(
                            {"family": family, "window": window,
                             "latent": latent,
                             "reason": "topology-invalid params"})
                        continue
                    yield family, window, latent

    if phase == "round1":
        for family, window, latent in valid_cells():
            units.append(_identity(family, window, latent,
                                   BUDGETS["round1"], SEEDS[0], 1,
                                   "cell"))
    elif phase in ("round2", "round3"):
        prev = {"round2": "round1", "round3": "round2"}[phase]
        decision_path = root / prev / "decisions" / "halving.json"
        if not decision_path.exists():
            raise RuntimePreflightError(
                f"{phase}: halving decision of {prev} absent — "
                "later rounds refuse to materialize without the "
                "persisted decision artifact")
        survivors = json.loads(decision_path.read_text())["advance"]
        for key in survivors:
            family, w, d = key.split("|")
            units.append(_identity(family, int(w[1:]), int(d[1:]),
                                   BUDGETS[phase], SEEDS[0], 1,
                                   "cell"))
    elif phase == "survivors":
        decision_path = root / "round3" / "decisions" / "halving.json"
        if not decision_path.exists():
            raise RuntimePreflightError(
                "survivors: round3 completion decision absent")
        cells = json.loads(decision_path.read_text())["advance"]
        for key in cells:
            family, w, d = key.split("|")
            window, latent = int(w[1:]), int(d[1:])
            for origin in ORIGINS:
                units.append(_identity(family, window, latent,
                                       BUDGETS["round3"], 0, origin,
                                       "persistence"))
                for seed in SEEDS:
                    units.append(_identity(
                        family, window, latent, BUDGETS["round3"],
                        seed, origin, "survivor_trained"))
                    units.append(_identity(
                        family, window, latent, BUDGETS["round3"],
                        seed + 5000, origin, "survivor_random"))
    elif phase == "fusion":
        for family in FAMILIES:
            units.append(_identity(family, 0, 0, 0, 0, 1,
                                   "branch_baseline"))
        units.append(_identity("fusion", 0, 0, 0, 0, 1,
                               "random_fusion_frozen"))
        units.append(_identity("fusion", 0, 0, 0, 0, 1,
                               "branch_concatenation"))
        for variant in FUSION_SEED_VARIANTS:
            budget = 600 if variant.startswith("fine_tuned") else 900
            for seed in SEEDS:
                units.append(_identity("fusion", 0, 0, budget, seed,
                                       1, variant))
    else:
        raise RuntimePreflightError(f"unknown phase {phase}")

    run = RunDirectory(root / phase)
    ledger = _ledger_for(
        units, digests, wall_ceiling_s=wall_ceiling_s,
        unit_timeout_s=unit_timeout_s,
        extras={"phase": phase, "invalid_cells": invalid,
                "pretrain_dir_digest": sha_file(
                    Path(pretrain_dir) / "generation.json"),
                "family_meta": fam_meta,
                "predeclaration": PREDECLARATION})
    digest = run.write_ledger(ledger)
    return {"phase": phase, "units": len(units),
            "invalid": len(invalid), "ledger_digest": digest}


# ------------------------------------------------------------------ #
# worker executors                                                   #
# ------------------------------------------------------------------ #

def _load_npz(path: Path):
    import numpy as np
    return np.load(path, allow_pickle=False)


def execute_cell(identity: dict, root: Path, pretrain_dir: Path,
                 fam_meta: dict, log_path: Path) -> dict:
    import resource
    science = _science()
    data = _load_npz(_window_npz(root, identity["window"]))
    windows = data["windows"]
    target = data["target"]
    meta = fam_meta[identity["family"]]
    fam_windows = windows[:, :, meta["cols"]]
    module, note = _build_module(meta, identity["latent"],
                                 identity["seed"],
                                 identity["window"])
    if module is None:
        raise RuntimeError(f"module build failed: {note}")
    fit_i, cal_i, mon_i = science.roles_for_origin(
        len(fam_windows), identity["origin"])
    rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    device = os.environ.get("SCREEN_V2_DEVICE", "cpu")
    mon_r2, cal_r2, n_params, wall = train_cell_device(
        module, identity["latent"], fam_windows, target, fit_i,
        cal_i, mon_i, identity["budget"], identity["seed"],
        device=device)
    log_path.write_text(json.dumps({
        "unit": identity, "monitor_r2": mon_r2,
        "calibration_r2": cal_r2}, default=float))
    return {"family": identity["family"],
            "window": identity["window"],
            "latent": identity["latent"],
            "budget": identity["budget"],
            "monitor_r2": round(float(mon_r2), 4),
            "calibration_r2": round(float(cal_r2), 4),
            "parameters": int(n_params), "wall_s": wall,
            "peak_rss_mb_delta": round(
                (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                 - rss0) / 1024.0, 1)}


def _build_module(meta: dict, latent: int, seed: int, window: int):
    import torch
    from agent_plugins.component_config import deep_merge_strict
    from app.plugin_loader import load_plugin
    science = _science()
    params = science.cell_params(meta["plugin"], latent)
    if params is None:
        return None, "topology-invalid"
    plugin_class, _ = load_plugin("feature_branch.plugins",
                                  meta["plugin"])
    merged = deep_merge_strict(plugin_class.plugin_params, params,
                               path="cell")
    torch.manual_seed(seed)
    try:
        module, dim = plugin_class.build(meta["n_features"],
                                         int(window), merged)
    except Exception as exc:  # typed into the unit result
        return None, str(exc)
    return module, dim


def execute_survivor(identity: dict, root: Path, fam_meta: dict,
                     log_path: Path) -> dict:
    science = _science()
    from agent_plugins.temporal_information import ridge_fit_cal_score
    data = _load_npz(_window_npz(root, identity["window"]))
    windows, target = data["windows"], data["target"]
    trailing = data["trailing"].reshape(-1, 1)
    meta = fam_meta[identity["family"]]
    fam_windows = windows[:, :, meta["cols"]]
    fit_i, cal_i, mon_i = science.roles_for_origin(
        len(fam_windows), identity["origin"])
    if identity["treatment"] == "persistence":
        score = ridge_fit_cal_score(
            trailing[fit_i], target[fit_i], trailing[cal_i],
            target[cal_i], trailing[mon_i], target[mon_i])["score"]
        log_path.write_text(json.dumps({"persistence_r2": score}))
        return {"persistence_r2": round(float(score), 4)}
    head_only = identity["treatment"] == "survivor_random"
    module, note = _build_module(meta, identity["latent"],
                                 identity["seed"],
                                 identity["window"])
    if module is None:
        raise RuntimeError(f"module build failed: {note}")
    device = os.environ.get("SCREEN_V2_DEVICE", "cpu")
    mon_r2, cal_r2, n_params, wall = train_cell_device(
        module, identity["latent"], fam_windows, target, fit_i,
        cal_i, mon_i, identity["budget"], identity["seed"],
        head_only=head_only, device=device)
    log_path.write_text(json.dumps({"monitor_r2": mon_r2}))
    return {"monitor_r2": round(float(mon_r2), 4),
            "calibration_r2": round(float(cal_r2), 4),
            "wall_s": wall}


def _fusion_models(dims):
    import torch
    import torch.nn as nn

    class GatedFusion(nn.Module):
        def __init__(self, dims, out_dim=96):
            super().__init__()
            self.projections = nn.ModuleList(
                [nn.Linear(d, out_dim) for d in dims])
            self.gate = nn.Linear(sum(dims), len(dims))
            self.out_dim = out_dim

        def forward(self, parts):
            weights = torch.softmax(
                self.gate(torch.cat(parts, dim=1)), dim=1)
            stacked = torch.stack(
                [proj(part) for proj, part
                 in zip(self.projections, parts)], dim=1)
            return (weights.unsqueeze(-1) * stacked).sum(dim=1)

    class CrossAttnLikeFusion(nn.Module):
        def __init__(self, dims, d_model=32, out_dim=96):
            super().__init__()
            self.tokens = nn.ModuleList(
                [nn.Linear(d, d_model) for d in dims])
            self.attention = nn.MultiheadAttention(
                d_model, 4, batch_first=True)
            self.head = nn.Linear(d_model * len(dims), out_dim)
            self.out_dim = out_dim

        def forward(self, parts):
            tokens = torch.stack(
                [t(p) for t, p in zip(self.tokens, parts)], dim=1)
            attended, _ = self.attention(tokens, tokens, tokens,
                                         need_weights=False)
            return self.head(attended.reshape(
                attended.shape[0], -1))

    return GatedFusion, CrossAttnLikeFusion


def _probe_all(x, y_vol, y_ret, y_bar, fit_i, cal_i, mon_i):
    from agent_plugins.temporal_information import (
        pinball_loss_score, ridge_fit_cal_score)
    out = {}
    out["volatility_r2"] = ridge_fit_cal_score(
        x[fit_i], y_vol[fit_i], x[cal_i], y_vol[cal_i],
        x[mon_i], y_vol[mon_i])["score"]
    out["quantile_q0.5_pinball_neg"] = ridge_fit_cal_score(
        x[fit_i], y_ret[fit_i], x[cal_i], y_ret[cal_i],
        x[mon_i], y_ret[mon_i],
        metric=lambda p, y: pinball_loss_score(p, y, 0.5))["score"]
    acc = ridge_fit_cal_score(
        x[fit_i], y_bar[fit_i], x[cal_i], y_bar[cal_i],
        x[mon_i], y_bar[mon_i],
        metric=lambda p, y: float(
            ((p > 0.5) == (y > 0.5)).mean()))["score"]
    base = float(max((y_bar[mon_i] > 0.5).mean(),
                     1 - (y_bar[mon_i] > 0.5).mean()))
    out["barrier_accuracy_minus_base"] = round(acc - base, 4)
    return {k: round(float(v), 4) for k, v in out.items()}


def execute_fusion(identity: dict, root: Path, pretrain_dir: Path,
                   fam_meta: dict, log_path: Path) -> dict:
    import numpy as np
    import torch
    import torch.nn as nn
    from agent_plugins.temporal_information import (
        chronological_roles)
    science = _science()
    data = _load_npz(root / "inputs" / "fusion_inputs.npz")
    y_vol, y_ret, y_bar = data["y_vol"], data["y_ret"], data["y_bar"]
    families = list(FAMILIES)
    embeddings = [data[f"emb_{f}"] for f in families]
    dims = [int(d) for d in data["dims"]]
    n = len(y_vol)
    fit_i, cal_i, mon_i = chronological_roles(n)
    purge = 12
    fit_i = fit_i[:-purge] if len(fit_i) > purge else fit_i
    cal_i = cal_i[:-purge] if len(cal_i) > purge else cal_i

    def probes(x):
        return _probe_all(x, y_vol, y_ret, y_bar, fit_i, cal_i,
                          mon_i)

    treatment = identity["treatment"]
    GatedFusion, CrossAttnLikeFusion = _fusion_models(dims)
    if treatment == "branch_baseline":
        emb = data[f"emb_{identity['family']}"]
        result = {"probes": probes(emb)}
    elif treatment == "branch_concatenation":
        result = {"probes": probes(
            np.concatenate(embeddings, axis=1)),
            "trainable_params": 0}
    elif treatment == "random_fusion_frozen":
        torch.manual_seed(0)
        frozen = CrossAttnLikeFusion(dims)
        frozen.eval()
        with torch.no_grad():
            fused = frozen([torch.tensor(e, dtype=torch.float32)
                            for e in embeddings]).numpy()
        result = {"probes": probes(fused), "trainable_params": 0,
                  "total_params": sum(p.numel()
                                      for p in frozen.parameters())}
    elif treatment in ("current_fusion_probe_trained",
                       "gated_softmax_probe_trained"):
        factory = (CrossAttnLikeFusion
                   if treatment.startswith("current")
                   else GatedFusion)
        torch.manual_seed(identity["seed"])
        model = factory(dims)
        fused = _train_fusion_embeddings(
            model, embeddings, y_vol, fit_i, cal_i,
            identity["budget"], identity["seed"])
        result = {"probes": probes(fused),
                  "trainable_params": sum(
                      p.numel() for p in model.parameters())}
    elif treatment == "fine_tuned_branches_plus_fusion":
        result = _fine_tuned_variant(
            identity, root, pretrain_dir, fam_meta, embeddings,
            dims, data, y_vol, fit_i, cal_i, probes,
            CrossAttnLikeFusion)
    else:
        raise RuntimeError(f"unknown fusion treatment {treatment}")
    log_path.write_text(json.dumps(result, default=float))
    return result


def _train_fusion_embeddings(model, embeddings, y_vol, fit_i, cal_i,
                             budget, seed):
    import torch
    import torch.nn as nn
    science = _science()
    head = nn.Linear(getattr(model, "out_dim", 96), 1)
    params = list(model.parameters()) + list(head.parameters())
    optimizer = torch.optim.Adam(params, lr=1e-3)
    parts_t = [torch.tensor(e, dtype=torch.float32)
               for e in embeddings]
    y = torch.tensor(y_vol, dtype=torch.float32).unsqueeze(-1)
    fit_t = torch.tensor(fit_i)
    generator = torch.Generator().manual_seed(seed)
    best_cal, best_states = -1e18, None
    for update in range(budget):
        batch = fit_t[torch.randint(0, len(fit_t), (256,),
                                    generator=generator)]
        optimizer.zero_grad()
        out = head(model([p[batch] for p in parts_t]))
        loss = nn.functional.mse_loss(out, y[batch])
        loss.backward()
        optimizer.step()
        if (update + 1) % 50 == 0 or update + 1 == budget:
            with torch.no_grad():
                cal_pred = head(model(
                    [p[cal_i] for p in parts_t])).squeeze(-1).numpy()
            score = science.r2(cal_pred, y_vol[cal_i])
            if score > best_cal:
                best_cal = score
                best_states = (
                    {k: v.clone()
                     for k, v in model.state_dict().items()},
                    {k: v.clone()
                     for k, v in head.state_dict().items()})
    if best_states:
        model.load_state_dict(best_states[0])
        head.load_state_dict(best_states[1])
    with torch.no_grad():
        return model([torch.tensor(e, dtype=torch.float32)
                      for e in embeddings]).numpy()


def _fine_tuned_variant(identity, root, pretrain_dir, fam_meta,
                        embeddings, dims, data, y_vol, fit_i, cal_i,
                        probes, CrossAttnLikeFusion):
    import numpy as np
    import torch
    import torch.nn as nn
    from agent_plugins.component_config import deep_merge_strict
    from agent_plugins.pretrained_branch_loader import (
        strict_load_encoder, verify_source)
    from app.plugin_loader import load_plugin
    science = _science()
    (source, contract, parsed, _df, ordered, _c,
     _d) = _contract_bundle(pretrain_dir)
    manifest = source["manifest"]
    window = int(parsed["window_size"])
    x_all = torch.tensor(data["windows"], dtype=torch.float32)
    modules, columns = [], []
    for branch in contract["branches"]:
        plugin_class, _ = load_plugin("feature_branch.plugins",
                                      branch["plugin"])
        params = deep_merge_strict(plugin_class.plugin_params,
                                   branch["params"], path="p")
        torch.manual_seed(identity["seed"])
        module, _dim = plugin_class.build(
            len(branch["features"]), window, params)
        entry = manifest["artifacts"][branch["name"]]
        state = torch.load(pretrain_dir / entry["encoder_file"],
                           weights_only=True)
        strict_load_encoder(module, state, branch["name"])
        modules.append(module)
        columns.append([ordered.index(f)
                        for f in branch["features"]])
    torch.manual_seed(identity["seed"])
    fusion = CrossAttnLikeFusion(dims)
    head = nn.Linear(96, 1)
    params = [p for m in modules for p in m.parameters()] + \
        list(fusion.parameters()) + list(head.parameters())
    optimizer = torch.optim.Adam(params, lr=5e-4)
    y_t = torch.tensor(y_vol, dtype=torch.float32).unsqueeze(-1)
    fit_t = torch.tensor(fit_i)
    generator = torch.Generator().manual_seed(identity["seed"])
    best_cal, best_states = -1e18, None
    for update in range(identity["budget"]):
        batch = fit_t[torch.randint(0, len(fit_t), (128,),
                                    generator=generator)]
        optimizer.zero_grad()
        parts = [m(x_all[batch][:, :, c])
                 for m, c in zip(modules, columns)]
        out = head(fusion(parts))
        loss = nn.functional.mse_loss(out, y_t[batch])
        loss.backward()
        optimizer.step()
        if (update + 1) % 100 == 0 or \
                update + 1 == identity["budget"]:
            with torch.no_grad():
                parts = [m(x_all[cal_i][:, :, c])
                         for m, c in zip(modules, columns)]
                cal_pred = head(fusion(parts)).squeeze(-1).numpy()
            score = science.r2(cal_pred, y_vol[cal_i])
            if score > best_cal:
                best_cal = score
                best_states = [
                    [{k: v.clone() for k, v in
                      m.state_dict().items()} for m in modules],
                    {k: v.clone()
                     for k, v in fusion.state_dict().items()},
                    {k: v.clone()
                     for k, v in head.state_dict().items()}]
    if best_states:
        for m, st in zip(modules, best_states[0]):
            m.load_state_dict(st)
        fusion.load_state_dict(best_states[1])
        head.load_state_dict(best_states[2])
    with torch.no_grad():
        chunks = []
        for start in range(0, len(x_all), 512):
            seg = x_all[start:start + 512]
            parts = [m(seg[:, :, c])
                     for m, c in zip(modules, columns)]
            chunks.append(fusion(parts).numpy())
    return {"probes": probes(np.concatenate(chunks, axis=0)),
            "trainable_params": int(sum(p.numel() for p in params)),
            "budget_updates": identity["budget"]}


def worker_main(args) -> int:
    root = Path(args.run_root)
    phase_dir = root / args.phase
    run = RunDirectory(phase_dir,
                       allow_volatile_for_tests=args.volatile_ok)
    ledger = run.ledger()
    fam_meta = ledger["family_meta"]
    identity = None
    for unit in ledger["units"]:
        if unit["unit_id"] == args.unit:
            identity = unit["identity"]
    if identity is None:
        raise RuntimePreflightError(f"unit {args.unit} not in ledger")
    # R1 (2026-09-03): re-hash EVERY input immediately before the
    # unit executes — code, predeclaration, the unit's own npz, the
    # source csv and the sealed generation. Any drift vs the ledger
    # refuses at claim time.
    expected = {"code": code_digest(),
                "config": sha_file(REPO / PREDECLARATION)}
    if identity["treatment"] in ("cell", "survivor_trained",
                                 "survivor_random", "persistence"):
        expected[f"input_w{identity['window']}"] = sha_file(
            _window_npz(root, identity["window"]))
    else:
        expected["input_fusion"] = sha_file(
            root / "inputs" / "fusion_inputs.npz")
    split_contract = json.loads(
        (REPO / "examples/config/phase_3_eth_sac_dynamics/splits/"
         "eth_nested_split_contract_o2022_paired_v1.json"
         ).read_text())
    expected["data_csv"] = sha_file(
        Path(split_contract["source_csv"]))
    expected["pretrain_generation"] = sha_file(
        Path(args.pretrain_dir) / "generation.json")
    timeout = float(args.timeout or ledger.get("unit_timeout_s")
                    or 3600)

    def executor(ident: dict, log_path: Path) -> dict:
        treatment = ident["treatment"]
        if treatment == "cell":
            return execute_cell(ident, root, Path(args.pretrain_dir),
                                fam_meta, log_path)
        if treatment in ("survivor_trained", "survivor_random",
                         "persistence"):
            return execute_survivor(ident, root, fam_meta, log_path)
        return execute_fusion(ident, root, Path(args.pretrain_dir),
                              fam_meta, log_path)

    outcome = run_one_unit(run, args.unit, executor,
                           expected_digests=expected,
                           timeout_s=timeout)
    print(json.dumps({"unit": args.unit,
                      "state": outcome["state"]}))
    return 0 if outcome["state"] == "COMPLETED" else 1


# ------------------------------------------------------------------ #
# halving decisions + aggregation                                    #
# ------------------------------------------------------------------ #

def decide_halving(root: Path, phase: str) -> dict:
    """Persist the mechanical halving decision of a COMPLETE phase as
    a separate artifact BEFORE the next round is materialized."""
    run = RunDirectory(root / phase)
    ledger = run.ledger()
    expected = [u["unit_id"] for u in ledger["units"]]
    results = runtime_aggregate(run, expected)
    cells = list(results.values())
    advance = []
    fraction = {"round1": 2, "round2": 2, "round3": 1}[phase]
    for family in FAMILIES:
        fam = sorted([c for c in cells if c["family"] == family],
                     key=lambda c: -c["calibration_r2"])
        keep = (len(fam) if fraction == 1
                else max(1, len(fam) // fraction))
        advance.extend(f"{c['family']}|w{c['window']}|d{c['latent']}"
                       for c in fam[:keep])
    decision = {"schema": "agent_multi.screen_v2_halving.v1",
                "phase": phase, "advance": sorted(advance),
                "decided_over_units": len(cells),
                "ledger_digest": ledger["ledger_digest"]}
    path = run.root / "decisions" / "halving.json"
    if path.exists():
        previous = json.loads(path.read_text())
        if previous["advance"] != decision["advance"]:
            raise RuntimePreflightError(
                f"{phase}: existing halving decision disagrees — "
                "refusing to overwrite a persisted decision")
        return previous
    atomic_write_json(path, decision)
    return decision


def aggregate_final(root: Path) -> dict:
    from agent_plugins.temporal_information import paired_stats
    report = {"schema": "agent_multi.positive_skill_screen.v2",
              "classification": "REPRESENTATION_DIAGNOSTIC",
              "predeclaration": Path(PREDECLARATION).name,
              "primary_target": "realized volatility h6",
              "cells": {}, "halving": {}, "survivor_decisions": {},
              "invalid_cells": [], "fusion": {}}
    for phase in ("round1", "round2", "round3"):
        run = RunDirectory(root / phase)
        ledger = run.ledger()
        report["invalid_cells"] = ledger.get("invalid_cells", [])
        results = runtime_aggregate(
            run, [u["unit_id"] for u in ledger["units"]])
        for res in results.values():
            key = (f"{res['family']}|w{res['window']}|"
                   f"d{res['latent']}")
            report["cells"][key] = res
        if phase != "round3":
            decision = json.loads((run.root / "decisions" /
                                   "halving.json").read_text())
            report["halving"][f"{phase}_advance"] = \
                decision["advance"]

    surv = RunDirectory(root / "survivors")
    ledger = surv.ledger()
    results = runtime_aggregate(
        surv, [u["unit_id"] for u in ledger["units"]])
    by_unit = {u["unit_id"]: u["identity"]
               for u in ledger["units"]}
    grouped: dict = {}
    for uid, res in results.items():
        ident = by_unit[uid]
        key = (f"{ident['family']}|w{ident['window']}|"
               f"d{ident['latent']}")
        grouped.setdefault(key, {"trained": {}, "random": {},
                                 "persistence": {}})
        if ident["treatment"] == "persistence":
            grouped[key]["persistence"][ident["origin"]] = \
                res["persistence_r2"]
        elif ident["treatment"] == "survivor_trained":
            grouped[key]["trained"][
                (ident["seed"], ident["origin"])] = \
                res["monitor_r2"]
        else:
            grouped[key]["random"][
                (ident["seed"] - 5000, ident["origin"])] = \
                res["monitor_r2"]
    for key, g in sorted(grouped.items()):
        monitor_scores, d_pers, d_rand = [], [], []
        for origin in ORIGINS:
            for seed in SEEDS:
                trained = g["trained"][(seed, origin)]
                monitor_scores.append(trained)
                d_pers.append(trained - g["persistence"][origin])
                d_rand.append(trained - g["random"][(seed, origin)])
        stats_p = paired_stats(d_pers)
        stats_r = paired_stats(d_rand)
        positive = all(s > 0 for s in monitor_scores)
        usable = (positive and stats_p["mean"] >= MIN_EFFECT
                  and stats_p["ci95_low"] > 0
                  and stats_r["mean"] >= MIN_EFFECT
                  and stats_r["ci95_low"] > 0)
        report["survivor_decisions"][key] = {
            "monitor_r2_all_runs": [round(s, 4)
                                    for s in monitor_scores],
            "all_positive_absolute_skill": positive,
            "paired_vs_persistence": stats_p,
            "paired_vs_random_encoder": stats_r,
            "verdict": ("USABLE_PREDICTIVE_VALUE" if usable
                        else "NOT_DEMONSTRATED")}

    fus = RunDirectory(root / "fusion")
    ledger = fus.ledger()
    results = runtime_aggregate(
        fus, [u["unit_id"] for u in ledger["units"]])
    by_unit = {u["unit_id"]: u["identity"]
               for u in ledger["units"]}
    branch_baselines, variants = {}, {}
    for uid, res in results.items():
        ident = by_unit[uid]
        t = ident["treatment"]
        if t == "branch_baseline":
            branch_baselines[ident["family"]] = res["probes"]
        elif t in ("random_fusion_frozen", "branch_concatenation"):
            variants[t] = {"probes_list": [res["probes"]],
                           **{k: v for k, v in res.items()
                              if k != "probes"}}
        else:
            variants.setdefault(t, {"probes_list": []})
            variants[t]["probes_list"].append(res["probes"])
            variants[t]["trainable_params"] = \
                res.get("trainable_params")
    best_branch = {key: max(v[key] for v in
                            branch_baselines.values())
                   for key in ("volatility_r2",
                               "quantile_q0.5_pinball_neg",
                               "barrier_accuracy_minus_base")}
    decisions = {}
    for name, entry in variants.items():
        def positive_any(p):
            return (p["volatility_r2"] > 0
                    or p["quantile_q0.5_pinball_neg"] > 0
                    or p["barrier_accuracy_minus_base"] > 0)

        def no_degrade(p):
            return all(p[k] >= best_branch[k] - DEGRADE_TOLERANCE
                       for k in best_branch)
        advancing = all(positive_any(p) and no_degrade(p)
                        for p in entry["probes_list"])
        decisions[name] = ("ADVANCES" if advancing
                           else "DOES_NOT_ADVANCE")
    report["fusion"] = {"branch_baselines": branch_baselines,
                        "best_non_fused_branch": best_branch,
                        "variants": variants,
                        "decisions": decisions}
    report["summary"] = {
        "survivor_verdicts": {k: v["verdict"] for k, v in
                              report["survivor_decisions"].items()},
        "fusion_decisions": decisions,
        "any_fusion_advances": any(v == "ADVANCES"
                                   for v in decisions.values())}
    return report


# ------------------------------------------------------------------ #
# supervisor                                                         #
# ------------------------------------------------------------------ #

def _campaign_start(root: Path) -> float:
    """R1 (2026-09-03): ONE durable campaign start; the wall budget
    covers the WHOLE campaign, not each phase."""
    marker = root / "campaign_start.json"
    if not marker.exists():
        atomic_write_json(marker, {"started_at": time.time()})
    return float(json.loads(marker.read_text())["started_at"])


def supervise(args) -> int:
    root = Path(args.run_root)
    pretrain_dir = Path(args.pretrain_dir)
    root.mkdir(parents=True, exist_ok=True)
    campaign_t0 = _campaign_start(root)
    stop = {"flag": False}
    children: list = []
    child_pids: dict = {}

    def on_term(_sig, _frame):
        stop["flag"] = True
        for proc in children:
            if proc.poll() is None:
                proc.terminate()

    signal.signal(signal.SIGTERM, on_term)
    signal.signal(signal.SIGINT, on_term)

    def kill_child(pid):
        """terminate AND reap — only a confirmed-dead child allows a
        TIMED_OUT release (R1 race fix)."""
        for proc in children:
            if proc.pid == pid:
                if proc.poll() is None:
                    proc.terminate()
                    try:
                        proc.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait(timeout=30)
                return proc.poll() is not None
        return not Path(f"/proc/{pid}").exists()

    def phase_complete(phase: str) -> bool:
        run = RunDirectory(root / phase)
        if not (run.root / "ledger.json").exists():
            return False
        states = run.states()
        return states and all(s["state"] == "COMPLETED"
                              for s in states.values())

    def run_phase(phase: str) -> bool:
        """Returns True when the phase reached full completion."""
        phase_dir = root / phase
        if not (phase_dir / "ledger.json").exists():
            summary = materialize_phase(
                root, phase, pretrain_dir,
                unit_timeout_s=args.unit_timeout,
                wall_ceiling_s=args.wall_ceiling,
                max_windows=args.max_windows, stride=args.stride)
            print(f"[materialize] {json.dumps(summary)}", flush=True)
        run = RunDirectory(phase_dir)
        preflight_or_refuse(run, args.wall_ceiling,
                            args.unit_timeout)
        last_beat = 0.0
        thermal_pause_until = 0.0
        while not stop["flag"]:
            states = run.states()
            pending = [u for u, s in states.items()
                       if s["state"] in ("PENDING", "FAILED",
                                         "INTERRUPTED", "TIMED_OUT")
                       and s.get("attempt", 0) < args.max_attempts]
            running = [u for u, s in states.items()
                       if s["state"] == "RUNNING"]
            if not pending and not running:
                break
            children[:] = [p for p in children if p.poll() is None]
            spawning_allowed = time.time() >= thermal_pause_until
            while (spawning_allowed
                   and len(children) < args.workers and pending
                   and not stop["flag"]):
                uid = pending.pop(0)
                cmd = [sys.executable, str(REPO / "tools" /
                                           "positive_skill_screen_v2.py"),
                       "worker", "--run-root", str(root),
                       "--phase", phase, "--unit", uid,
                       "--pretrain-dir", str(pretrain_dir)]
                if args.unit_timeout:
                    cmd += ["--timeout", str(args.unit_timeout)]
                env = dict(os.environ)
                bench_path = root / "benchmark.json"
                if bench_path.exists():
                    env["SCREEN_V2_DEVICE"] = json.loads(
                        bench_path.read_text())["decision"]
                log = open(run.root / "logs" /
                           f"worker_{uid}.out", "ab")
                children.append(subprocess.Popen(
                    cmd, stdout=log, stderr=subprocess.STDOUT,
                    env=env))
            if time.time() - last_beat >= args.heartbeat_s:
                current = running[0] if running else None
                bench_path = root / "benchmark.json"
                device = (json.loads(bench_path.read_text())
                          ["decision"] if bench_path.exists()
                          else "cpu")
                run.heartbeat(
                    current_unit=current,
                    workers=args.workers,
                    device_class=device,
                    extra={"phase": phase,
                           "workers_alive": len(children),
                           "campaign_root": str(root),
                           "campaign_elapsed_s": round(
                               time.time() - campaign_t0, 1),
                           "campaign_wall_ceiling_s":
                               args.wall_ceiling})
                alerts = run.watchdog(
                    kill_child=kill_child,
                    expected_digests={
                        "code": code_digest(),
                        "config": sha_file(REPO / PREDECLARATION)})
                for alert in alerts:
                    print(f"[watchdog] {json.dumps(alert)}",
                          flush=True)
                    if alert["type"] == "thermal":
                        thermal_pause_until = time.time() + 300
                        print("[watchdog] thermal pause: no new "
                              "units for 300s", flush=True)
                    if alert["type"] == "identity_drift":
                        print("[watchdog] IDENTITY DRIFT — no new "
                              "units; finishing the running ones "
                              "and stopping", flush=True)
                        stop["flag"] = True
                last_beat = time.time()
            if time.time() - campaign_t0 > args.wall_ceiling:
                print("[ceiling] GLOBAL campaign wall ceiling "
                      "reached — graceful stop, results preserved, "
                      "no hot budget extension", flush=True)
                on_term(None, None)
                break
            time.sleep(2.0)
        for proc in children:
            proc.wait(timeout=60)
        run.heartbeat(current_unit=None,
                      extra={"phase": phase, "workers": 0,
                             "stopped": stop["flag"]})
        return phase_complete(phase)

    for phase in PHASES:
        if stop["flag"]:
            print("[supervise] stopped by signal — durable state "
                  "preserved", flush=True)
            return 143
        if phase_complete(phase):
            if phase in ("round1", "round2", "round3"):
                decide_halving(root, phase)
            continue
        done = run_phase(phase)
        if not done:
            print(f"[supervise] phase {phase} incomplete — resume "
                  "with the same command", flush=True)
            return 1
        if phase in ("round1", "round2", "round3"):
            decision = decide_halving(root, phase)
            print(f"[halving] {phase}: {len(decision['advance'])} "
                  "advance", flush=True)
    report = aggregate_final(root)
    out = root / "POSITIVE_SKILL_SCREEN_V2_REPORT.json"
    atomic_write_json(out, report)
    print(json.dumps(report["summary"], indent=1), flush=True)
    print(f"[supervise] final report: {out}", flush=True)
    return 0


def status_main(args) -> int:
    root = Path(args.run_root)
    out = {"campaign_root": str(root), "phases": {}}
    for phase in PHASES:
        status_path = root / phase / "status.json"
        state_dir = root / phase / "units"
        if not state_dir.exists():
            out["phases"][phase] = {"state": "NOT_MATERIALIZED"}
            continue
        run = RunDirectory(root / phase)
        states = run.states()
        counts: dict = {}
        for s in states.values():
            counts[s["state"]] = counts.get(s["state"], 0) + 1
        entry = {"counts": counts,
                 "total_units": len(states)}
        if status_path.exists():
            entry["last_heartbeat"] = json.loads(
                status_path.read_text())
        out["phases"][phase] = entry
    print(json.dumps(out, indent=1, default=str))
    return 0


def benchmark(args) -> int:
    """One CPU and (when available) one CUDA benchmark cell on the
    heaviest family/window/latent; prints measured wall seconds. The
    device decision and the per-unit timeout derive from THESE
    numbers, not habit."""
    import torch
    science = _science()
    root = Path(args.run_root)
    pretrain_dir = Path(args.pretrain_dir)
    materialize_inputs(root, pretrain_dir,
                       max_windows=args.max_windows,
                       stride=args.stride)
    fam_meta = _family_columns(pretrain_dir)
    report = {}
    for device in (["cpu"] + (["cuda"]
                              if torch.cuda.is_available() else [])):
        # heaviest cell: timesnet family at w256/d128 when valid
        family = next((f for f in FAMILIES
                       if fam_meta[f]["plugin"] == "timesnet_branch"),
                      FAMILIES[0])
        ident = _identity(family, 256, 128, BUDGETS["round1"],
                          SEEDS[0], 1, "cell")
        if device == "cuda":
            # measured, honest: same unit maths on GPU via default
            # tensor device would need train_cell device support;
            # measure transfer-inclusive by env var toggle
            os.environ["SCREEN_V2_DEVICE"] = "cuda"
        started = time.perf_counter()
        result = execute_cell(ident, root, pretrain_dir, fam_meta,
                              root / f"benchmark_{device}.log")
        wall = time.perf_counter() - started
        os.environ.pop("SCREEN_V2_DEVICE", None)
        report[device] = {"wall_s": round(wall, 1),
                          "cell_wall_s": result["wall_s"],
                          "unit": f"{family}|w256|d128@300"}
    report["decision"] = ("cpu" if "cuda" not in report
                          or report["cpu"]["wall_s"]
                          <= report["cuda"]["wall_s"]
                          else "cuda")
    report["suggested_unit_timeout_s"] = round(
        max(60.0, 20.0 * report[report["decision"]]["wall_s"]
            * (BUDGETS["round3"] / BUDGETS["round1"])), 1)
    atomic_write_json(root / "benchmark.json", report)
    print(json.dumps(report, indent=1))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--run-root", required=True)
        p.add_argument("--pretrain-dir", required=True)
        p.add_argument("--max-windows", type=int, default=2200)
        p.add_argument("--stride", type=int, default=4)

    m = sub.add_parser("materialize")
    common(m)
    m.add_argument("--phase", required=True, choices=PHASES)
    m.add_argument("--unit-timeout", type=float, default=None)
    m.add_argument("--wall-ceiling", type=float, default=12 * 3600)

    w = sub.add_parser("worker")
    w.add_argument("--run-root", required=True)
    w.add_argument("--phase", required=True, choices=PHASES)
    w.add_argument("--unit", required=True)
    w.add_argument("--pretrain-dir", required=True)
    w.add_argument("--timeout", type=float, default=None)
    w.add_argument("--volatile-ok", action="store_true")

    s = sub.add_parser("supervise")
    common(s)
    s.add_argument("--workers", type=int, default=4)
    s.add_argument("--unit-timeout", type=float, required=True)
    s.add_argument("--wall-ceiling", type=float, default=12 * 3600)
    s.add_argument("--heartbeat-s", type=float, default=60.0)
    s.add_argument("--max-attempts", type=int, default=3)

    st = sub.add_parser("status")
    st.add_argument("--run-root", required=True)

    b = sub.add_parser("benchmark")
    common(b)

    a = sub.add_parser("aggregate")
    a.add_argument("--run-root", required=True)
    a.add_argument("--output", default=None)

    args = parser.parse_args()
    if args.cmd == "materialize":
        summary = materialize_phase(
            Path(args.run_root), args.phase, Path(args.pretrain_dir),
            unit_timeout_s=args.unit_timeout,
            wall_ceiling_s=args.wall_ceiling,
            max_windows=args.max_windows, stride=args.stride)
        print(json.dumps(summary, indent=1))
        return 0
    if args.cmd == "worker":
        return worker_main(args)
    if args.cmd == "supervise":
        return supervise(args)
    if args.cmd == "status":
        return status_main(args)
    if args.cmd == "benchmark":
        return benchmark(args)
    if args.cmd == "aggregate":
        report = aggregate_final(Path(args.run_root))
        if args.output:
            atomic_write_json(Path(args.output), report)
        print(json.dumps(report["summary"], indent=1))
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
