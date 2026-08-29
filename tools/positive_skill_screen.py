#!/usr/bin/env python3
"""Bounded positive-skill window/bottleneck screen (order @b6b949ab
section C; predeclared in
POSITIVE_SKILL_SCREEN_PREDECLARATION_2026_08_28.json).
REPRESENTATION_DIAGNOSTIC.

Per cell (family x window x latent): supervised joint encoder+head
training on the predeclared target (realized volatility h6) over the
fit role, selection ONLY on calibration, scored ONLY on monitor.
Equal minimum budget, successive halving by calibration score
(mechanical pruning), then the survivors rerun the FULL decision
protocol: 4 encoder seeds x 2 rolling causal monitor origins with
paired CIs against the trailing-vol persistence baseline and the
matched untrained random encoder.

The gate for USABLE_PREDICTIVE_VALUE demands POSITIVE monitor R2 —
never merely less-negative performance."""
from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

WINDOWS = (32, 64, 128, 256)
LATENTS = (16, 32, 64, 96, 128)
FULL_FAMILIES = ("returns_momentum", "trend_level",
                 "volatility_distribution")
CHEAP_CONTROL_CELLS = {"oscillators": [(32, 32)],
                       "volume_flow": [(32, 32)]}
SEEDS = (101, 202, 303, 404)
MIN_BUDGET, MID_BUDGET, TOP_BUDGET = 300, 900, 2700
MIN_EFFECT = 0.02
PURGE = 12


def cell_params(plugin: str, latent: int) -> dict | None:
    """Topology-valid params for the family plugin at this latent
    width; None when the combination is structurally invalid (absent
    from the domain, never launched)."""
    heads = next((h for h in (4, 2, 1) if latent % h == 0), None)
    if plugin == "patchtst_branch":
        return {"patch_len": 8, "stride": 8, "d_model": latent,
                "n_heads": heads, "n_layers": 1, "ff_mult": 2,
                "dropout": 0.0}
    if plugin == "tft_branch":
        return {"hidden": latent, "n_heads": heads, "dropout": 0.0}
    if plugin == "timesnet_branch":
        return {"top_k": 2, "d_model": latent, "kernel": 3,
                "dropout": 0.0}
    if plugin == "tcn_branch":
        return {"channels": [latent, latent], "kernel_size": 3,
                "dilation_base": 2, "dropout": 0.0,
                "activation": "relu"}
    if plugin == "gru_branch":
        return {"hidden_size": latent, "num_layers": 1,
                "dropout": 0.0, "bidirectional": False}
    return None


def roles_for_origin(n: int, origin: int):
    import numpy as np
    if origin == 0:
        limit = int(n * 0.85)
        fit_end, cal_end = int(limit * 0.65), int(limit * 0.82)
        fit = np.arange(0, fit_end - PURGE)
        cal = np.arange(fit_end, cal_end - PURGE)
        mon = np.arange(cal_end, limit)
    else:
        fit_end, cal_end = int(n * 0.70), int(n * 0.85)
        fit = np.arange(0, fit_end - PURGE)
        cal = np.arange(fit_end, cal_end - PURGE)
        mon = np.arange(cal_end, n)
    return fit, cal, mon


def r2(pred, y):
    import numpy as np
    residual = float(((y - pred) ** 2).sum())
    total = float(((y - y.mean()) ** 2).sum()) or 1.0
    return 1.0 - residual / total


def train_cell(module, latent, windows, target, fit_i, cal_i, mon_i,
               budget, seed, *, head_only=False):
    """Supervised training on fit; best-on-calibration state; monitor
    R2. Returns (monitor_r2, calibration_r2, params, wall_s)."""
    import numpy as np
    import torch

    torch.manual_seed(seed)
    head = torch.nn.Linear(latent, 1)
    params = list(head.parameters()) + (
        [] if head_only else list(module.parameters()))
    optimizer = torch.optim.Adam(params, lr=1e-3)
    x = torch.tensor(windows, dtype=torch.float32)
    y = torch.tensor(target, dtype=torch.float32).unsqueeze(-1)
    fit_t = torch.tensor(fit_i)
    generator = torch.Generator().manual_seed(seed)
    best_cal, best_state = -1e18, None
    start = time.perf_counter()
    module.train()
    for update in range(budget):
        batch_size = 128 if windows.shape[1] >= 128 else 256
        batch_idx = fit_t[torch.randint(
            0, len(fit_t), (min(batch_size, len(fit_t)),),
            generator=generator)]
        optimizer.zero_grad()
        out = head(module(x[batch_idx]))
        loss = torch.nn.functional.mse_loss(out, y[batch_idx])
        loss.backward()
        optimizer.step()
        if (update + 1) % 50 == 0 or update + 1 == budget:
            module.eval()
            with torch.no_grad():
                cal_pred = head(module(x[cal_i])).squeeze(-1).numpy()
            module.train()
            cal_score = r2(cal_pred, target[cal_i])
            if cal_score > best_cal:
                best_cal = cal_score
                best_state = ({k: v.detach().clone()
                               for k, v in module.state_dict().items()},
                              {k: v.detach().clone()
                               for k, v in head.state_dict().items()})
    if best_state is not None:
        module.load_state_dict(best_state[0])
        head.load_state_dict(best_state[1])
    module.eval()
    with torch.no_grad():
        mon_pred = head(module(x[mon_i])).squeeze(-1).numpy()
    wall = time.perf_counter() - start
    n_params = sum(p.numel() for p in module.parameters()) + \
        sum(p.numel() for p in head.parameters())
    return (r2(mon_pred, target[mon_i]), best_cal, n_params,
            round(wall, 1))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretrain-dir", required=True)
    parser.add_argument("--max-windows", type=int, default=2200)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    import numpy as np
    import torch

    from agent_plugins.branch_pretraining import (
        build_step_index, collect_preprocessed_windows,
        load_fit_slice, realized_volatility_targets,
        validate_contract)
    from agent_plugins.component_config import deep_merge_strict
    from agent_plugins.pretrained_branch_loader import verify_source
    from agent_plugins.temporal_information import (
        order_invariant_pooled_embedding, paired_stats,
        ridge_fit_cal_score)
    from app.plugin_loader import load_plugin

    pretrain_dir = Path(args.pretrain_dir)
    split_contract = json.loads(
        (REPO / "examples/config/phase_3_eth_sac_dynamics/splits/"
         "eth_nested_split_contract_o2022_paired_v1.json"
         ).read_text())
    data_path = Path(split_contract["source_csv"])
    source = verify_source(pretrain_dir, REPO, data_path)
    contract = source["contract"]
    parsed = validate_contract(contract)
    df, ordered, close_col = load_fit_slice(data_path, contract)
    closes = df[close_col].to_numpy()
    vol_spec = contract["objectives"]["volatility"]
    annualization = vol_spec.get("annualization")
    periods = (None if annualization in (None, "none")
               else annualization["periods_per_year"])

    env_source = contract["observation_pipeline"]["source_config"]
    env_config_base = json.loads(Path(env_source).read_text()
                                 if Path(env_source).is_absolute()
                                 else (REPO / env_source).read_text())

    def data_for_window(window: int):
        warmup = max(int(parsed["warmup_bars"]), window)
        steps = build_step_index(len(df), warmup,
                                 max(1, int(args.stride)), 12,
                                 int(args.max_windows))
        env_config = {**env_config_base, "window_size": window}
        contract_w = {**contract, "window_size": window}
        windows = collect_preprocessed_windows(df, contract_w,
                                               env_config, steps)
        target = realized_volatility_targets(
            closes, steps, [6], float(vol_spec.get("epsilon", 1e-8)),
            periods)[:, 0]
        # trailing-vol persistence baseline predictor (past-only)
        trailing = realized_volatility_targets(
            closes, [max(0, t - 6) for t in steps], [6],
            float(vol_spec.get("epsilon", 1e-8)), periods)[:, 0]
        keep = np.isfinite(target) & np.isfinite(trailing)
        return (windows[keep], np.asarray(target)[keep],
                np.asarray(trailing)[keep].reshape(-1, 1))

    family_plugins = {b["name"]: b for b in contract["branches"]}
    checkpoint_path = (Path(args.output).with_suffix(".partial.json")
                       if args.output else None)
    resumed = {}
    if checkpoint_path and checkpoint_path.exists():
        resumed = json.loads(checkpoint_path.read_text())
        print(f"[resume] {len(resumed.get('cells', {}))} cells from "
              f"checkpoint", flush=True)

    def checkpoint():
        if checkpoint_path:
            checkpoint_path.write_text(
                json.dumps(report, indent=1, default=float))

    report = resumed if resumed else {
        "schema": "agent_multi.positive_skill_screen.v1",
        "classification": "REPRESENTATION_DIAGNOSTIC",
        "predeclaration":
            "POSITIVE_SKILL_SCREEN_PREDECLARATION_2026_08_28.json",
        "primary_target": "realized volatility h6",
        "cells": {}, "halving": {}, "survivor_decisions": {},
        "invalid_cells": [],
    }
    data_cache = {}
    for window in WINDOWS:
        data_cache[window] = data_for_window(window)

    def build_module(branch, latent, seed, window):
        params = cell_params(branch["plugin"], latent)
        if params is None:
            return None, None
        plugin_class, _ = load_plugin("feature_branch.plugins",
                                      branch["plugin"])
        merged = deep_merge_strict(plugin_class.plugin_params, params,
                                   path="cell")
        torch.manual_seed(seed)
        try:
            module, dim = plugin_class.build(
                len(branch["features"]), int(window), merged)
        except Exception as exc:
            return None, str(exc)
        return module, dim

    # ---- round 1: equal minimum budget over every valid cell ------
    round_one = []
    for family in FULL_FAMILIES + tuple(CHEAP_CONTROL_CELLS):
        branch = family_plugins[family]
        grid = ([(w, d) for w in WINDOWS for d in LATENTS]
                if family in FULL_FAMILIES
                else CHEAP_CONTROL_CELLS[family])
        for window, latent in grid:
            windows, target, trailing = data_cache[window]
            fam_windows = windows[:, :, [ordered.index(f)
                                         for f in branch["features"]]]
            key = f"{family}|w{window}|d{latent}"
            if key in report["cells"]:
                round_one.append(report["cells"][key])
                continue
            module, note = build_module(branch, latent, SEEDS[0],
                                        window)
            if module is None:
                report["invalid_cells"].append(
                    {"family": family, "window": window,
                     "latent": latent, "reason": note})
                continue
            fit_i, cal_i, mon_i = roles_for_origin(len(fam_windows), 1)
            rss0 = resource.getrusage(
                resource.RUSAGE_SELF).ru_maxrss
            mon_r2, cal_r2, n_params, wall = train_cell(
                module, latent, fam_windows, target, fit_i, cal_i,
                mon_i, MIN_BUDGET, SEEDS[0])
            cell = {"family": family, "window": window,
                    "latent": latent, "budget": MIN_BUDGET,
                    "monitor_r2": round(mon_r2, 4),
                    "calibration_r2": round(cal_r2, 4),
                    "parameters": n_params, "wall_s": wall,
                    "peak_rss_mb_delta": round((resource.getrusage(
                        resource.RUSAGE_SELF).ru_maxrss - rss0)
                        / 1024.0, 1)}
            round_one.append(cell)
            report["cells"][key] = cell
            print(f"[round1] {key} cal={cell['calibration_r2']} "
                  f"mon={cell['monitor_r2']} wall={cell['wall_s']}s",
                  flush=True)
            checkpoint()
    # halving by calibration score (mechanical)
    def advance(cells, budget):
        survivors = []
        for family in FULL_FAMILIES:
            fam = sorted([c for c in cells
                          if c["family"] == family],
                         key=lambda c: -c["calibration_r2"])
            survivors.extend(fam[:max(1, len(fam) // 2)])
        for cell in survivors:
            branch = family_plugins[cell["family"]]
            windows, target, trailing = data_cache[cell["window"]]
            fam_windows = windows[:, :, [
                ordered.index(f) for f in branch["features"]]]
            module, _ = build_module(branch, cell["latent"],
                                     SEEDS[0], cell["window"])
            fit_i, cal_i, mon_i = roles_for_origin(len(fam_windows), 1)
            mon_r2, cal_r2, n_params, wall = train_cell(
                module, cell["latent"], fam_windows, target, fit_i,
                cal_i, mon_i, budget, SEEDS[0])
            cell.update({"budget": budget,
                         "monitor_r2": round(mon_r2, 4),
                         "calibration_r2": round(cal_r2, 4),
                         "wall_s": cell["wall_s"] + wall})
            print(f"[halving b={budget}] {cell['family']}|"
                  f"w{cell['window']}|d{cell['latent']} "
                  f"cal={cell['calibration_r2']}", flush=True)
            checkpoint()
        return survivors

    round_two = advance(round_one, MID_BUDGET)
    report["halving"]["round2_survivors"] = [
        f"{c['family']}|w{c['window']}|d{c['latent']}"
        for c in round_two]
    round_three = advance(round_two, TOP_BUDGET)
    report["halving"]["round3_survivors"] = [
        f"{c['family']}|w{c['window']}|d{c['latent']}"
        for c in round_three]

    # ---- survivor decision protocol: 4 seeds x 2 origins ----------
    for cell in round_three:
        skey = f"{cell['family']}|w{cell['window']}|d{cell['latent']}"
        if skey in report["survivor_decisions"]:
            continue
        branch = family_plugins[cell["family"]]
        windows, target, trailing = data_cache[cell["window"]]
        fam_windows = windows[:, :, [ordered.index(f)
                                     for f in branch["features"]]]
        deltas_persistence, deltas_random, monitor_scores = [], [], []
        for origin in (0, 1):
            fit_i, cal_i, mon_i = roles_for_origin(
                len(fam_windows), origin)
            persistence = ridge_fit_cal_score(
                trailing[fit_i], target[fit_i], trailing[cal_i],
                target[cal_i], trailing[mon_i],
                target[mon_i])["score"]
            for seed in SEEDS:
                module, _ = build_module(branch, cell["latent"],
                                         seed, cell["window"])
                mon_r2, _cal, _p, _w = train_cell(
                    module, cell["latent"], fam_windows, target,
                    fit_i, cal_i, mon_i, TOP_BUDGET, seed)
                random_module, _ = build_module(
                    branch, cell["latent"], seed + 5000,
                    cell["window"])
                random_r2, _c2, _p2, _w2 = train_cell(
                    random_module, cell["latent"], fam_windows,
                    target, fit_i, cal_i, mon_i, TOP_BUDGET,
                    seed + 5000, head_only=True)
                monitor_scores.append(mon_r2)
                deltas_persistence.append(mon_r2 - persistence)
                deltas_random.append(mon_r2 - random_r2)
        stats_p = paired_stats(deltas_persistence)
        stats_r = paired_stats(deltas_random)
        positive = all(s > 0 for s in monitor_scores)
        usable = (positive
                  and stats_p["mean"] >= MIN_EFFECT
                  and stats_p["ci95_low"] > 0
                  and stats_r["mean"] >= MIN_EFFECT
                  and stats_r["ci95_low"] > 0)
        print(f"[survivor] {skey} runs done", flush=True)
        report["survivor_decisions"][skey] = {
            "monitor_r2_all_runs": [round(s, 4)
                                    for s in monitor_scores],
            "all_positive_absolute_skill": positive,
            "paired_vs_persistence": stats_p,
            "paired_vs_random_encoder": stats_r,
            "verdict": ("USABLE_PREDICTIVE_VALUE" if usable
                        else "NOT_DEMONSTRATED"),
        }
        checkpoint()

    report["summary"] = {k: v["verdict"]
                         for k, v in
                         report["survivor_decisions"].items()}
    payload = json.dumps(report, indent=1, default=float)
    print(json.dumps(report["summary"], indent=1))
    if args.output:
        Path(args.output).write_text(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
