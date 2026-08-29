#!/usr/bin/env python3
"""Bounded fusion screen (order @b6b949ab section D; predeclared in
POSITIVE_SKILL_SCREEN_PREDECLARATION_2026_08_28.json).
REPRESENTATION_DIAGNOSTIC — no long SAC or DOIN campaign.

Inputs: FROZEN branch embeddings from the sealed candidate (branch
diagnostics preserved separately). Variants under matched capacity:

1. random current fusion (frozen) — the architecture SAC consumes;
2. branch concatenation (no fusion params);
3. current fusion architecture, probe-trained on the fit role;
4. gated softmax fusion, probe-trained;
5. fine-tuned branches + fusion (bounded updates).

Targets: realized volatility h6 (primary), median-quantile pinball
h6, barrier upper-hit accuracy. A fused candidate advances ONLY with
positive out-of-sample monitor skill on >= 1 target AND no other
target degraded by more than 0.02 vs the best non-fused branch
baseline. 4 seeds, paired reporting."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

SEEDS = (101, 202, 303, 404)
DEGRADE_TOLERANCE = 0.02
PURGE = 12


def r2(pred, y):
    residual = float(((y - pred) ** 2).sum())
    total = float(((y - y.mean()) ** 2).sum()) or 1.0
    return 1.0 - residual / total


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
        barrier_hit_labels, build_step_index,
        collect_preprocessed_windows, forward_log_return_targets,
        load_fit_slice, realized_volatility_targets,
        validate_contract)
    from agent_plugins.component_config import deep_merge_strict
    from agent_plugins.pretrained_branch_loader import (
        strict_load_encoder, verify_source)
    from agent_plugins.temporal_information import (
        chronological_roles, encode_windows, paired_stats,
        pinball_loss_score, ridge_fit_cal_score)
    from app.plugin_loader import load_plugin

    pretrain_dir = Path(args.pretrain_dir)
    split_contract = json.loads(
        (REPO / "examples/config/phase_3_eth_sac_dynamics/splits/"
         "eth_nested_split_contract_o2022_paired_v1.json"
         ).read_text())
    data_path = Path(split_contract["source_csv"])
    source = verify_source(pretrain_dir, REPO, data_path)
    contract = source["contract"]
    manifest = source["manifest"]
    parsed = validate_contract(contract)
    df, ordered, close_col = load_fit_slice(data_path, contract)
    closes = df[close_col].to_numpy()
    window = int(parsed["window_size"])
    steps = build_step_index(len(df), parsed["warmup_bars"],
                             max(1, int(args.stride)), 12,
                             int(args.max_windows))
    env_source = contract["observation_pipeline"]["source_config"]
    env_config = json.loads(Path(env_source).read_text()
                            if Path(env_source).is_absolute()
                            else (REPO / env_source).read_text())
    windows = collect_preprocessed_windows(df, contract, env_config,
                                           steps)
    vol_spec = contract["objectives"]["volatility"]
    annualization = vol_spec.get("annualization")
    periods = (None if annualization in (None, "none")
               else annualization["periods_per_year"])
    y_vol = realized_volatility_targets(
        closes, steps, [6], float(vol_spec.get("epsilon", 1e-8)),
        periods)[:, 0]
    quant_h = list(contract["objectives"][
        "quantile_regression"]["horizons"])
    h_col = min(range(len(quant_h)), key=lambda i: abs(
        quant_h[i] - 6))
    y_ret = forward_log_return_targets(closes, steps,
                                       quant_h)[:, h_col]
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
    y_vol = np.asarray(y_vol, float)
    y_ret = np.asarray(y_ret, float)
    n = len(windows)
    fit_i, cal_i, mon_i = chronological_roles(n)
    fit_i = fit_i[:-PURGE] if len(fit_i) > PURGE else fit_i
    cal_i = cal_i[:-PURGE] if len(cal_i) > PURGE else cal_i

    # frozen branch embeddings from the sealed candidate
    branch_embeddings = []
    branch_names = []
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
        branch_embeddings.append(encode_windows(
            module, windows[:, :, cols]))
        branch_names.append(branch["name"])
    concat = np.concatenate(branch_embeddings, axis=1)
    dims = [e.shape[1] for e in branch_embeddings]

    def probe_all(x):
        out = {}
        out["volatility_r2"] = ridge_fit_cal_score(
            x[fit_i], y_vol[fit_i], x[cal_i], y_vol[cal_i],
            x[mon_i], y_vol[mon_i])["score"]
        out["quantile_q0.5_pinball_neg"] = ridge_fit_cal_score(
            x[fit_i], y_ret[fit_i], x[cal_i], y_ret[cal_i],
            x[mon_i], y_ret[mon_i],
            metric=lambda p, y: pinball_loss_score(p, y, 0.5)
        )["score"]
        acc = ridge_fit_cal_score(
            x[fit_i], y_bar[fit_i], x[cal_i], y_bar[cal_i],
            x[mon_i], y_bar[mon_i],
            metric=lambda p, y: float(((p > 0.5) == (y > 0.5)).mean())
        )["score"]
        base = float(max((y_bar[mon_i] > 0.5).mean(),
                         1 - (y_bar[mon_i] > 0.5).mean()))
        out["barrier_accuracy_minus_base"] = round(acc - base, 4)
        return out

    # best NON-FUSED branch baseline per target
    per_branch = {name: probe_all(e)
                  for name, e in zip(branch_names,
                                     branch_embeddings)}
    best_branch = {
        key: max(v[key] for v in per_branch.values())
        for key in ("volatility_r2", "quantile_q0.5_pinball_neg",
                    "barrier_accuracy_minus_base")}

    import torch.nn as nn

    class GatedFusion(nn.Module):
        def __init__(self, dims, out_dim=96):
            super().__init__()
            self.projections = nn.ModuleList(
                [nn.Linear(d, out_dim) for d in dims])
            self.gate = nn.Linear(sum(dims), len(dims))
            self.out_dim = out_dim

        def forward(self, parts):
            weights = torch.softmax(self.gate(
                torch.cat(parts, dim=1)), dim=1)
            stacked = torch.stack(
                [proj(part) for proj, part in
                 zip(self.projections, parts)], dim=1)
            return (weights.unsqueeze(-1) * stacked).sum(dim=1)

    class CrossAttnLikeFusion(nn.Module):
        """Capacity-matched stand-in for the current fusion: linear
        per-family tokens + one attention layer + output head."""

        def __init__(self, dims, d_model=32, out_dim=96):
            super().__init__()
            self.tokens = nn.ModuleList(
                [nn.Linear(d, d_model) for d in dims])
            self.attention = nn.MultiheadAttention(
                d_model, 4, batch_first=True)
            self.head = nn.Linear(d_model * len(dims), out_dim)

        def forward(self, parts):
            tokens = torch.stack([t(p) for t, p in
                                  zip(self.tokens, parts)], dim=1)
            attended, _ = self.attention(tokens, tokens, tokens,
                                         need_weights=False)
            b = attended.shape[0]
            return self.head(attended.reshape(b, -1))

    def train_fusion(model, budget, seed, fine_tune_branches=False):
        torch.manual_seed(seed)
        head_vol = nn.Linear(model_out_dim(model), 1)
        params = list(model.parameters()) + list(
            head_vol.parameters())
        optimizer = torch.optim.Adam(params, lr=1e-3)
        parts_t = [torch.tensor(e, dtype=torch.float32)
                   for e in branch_embeddings]
        y = torch.tensor(y_vol, dtype=torch.float32).unsqueeze(-1)
        fit_t = torch.tensor(fit_i)
        generator = torch.Generator().manual_seed(seed)
        best_cal, best_states = -1e18, None
        for update in range(budget):
            batch = fit_t[torch.randint(0, len(fit_t), (256,),
                                        generator=generator)]
            optimizer.zero_grad()
            out = head_vol(model([p[batch] for p in parts_t]))
            loss = nn.functional.mse_loss(out, y[batch])
            loss.backward()
            optimizer.step()
            if (update + 1) % 50 == 0 or update + 1 == budget:
                with torch.no_grad():
                    cal_pred = head_vol(model(
                        [p[cal_i] for p in parts_t])
                    ).squeeze(-1).numpy()
                score = r2(cal_pred, y_vol[cal_i])
                if score > best_cal:
                    best_cal = score
                    best_states = (
                        {k: v.clone() for k, v in
                         model.state_dict().items()},
                        {k: v.clone() for k, v in
                         head_vol.state_dict().items()})
        if best_states:
            model.load_state_dict(best_states[0])
            head_vol.load_state_dict(best_states[1])
        with torch.no_grad():
            fused_all = model([torch.tensor(e, dtype=torch.float32)
                               for e in branch_embeddings]).numpy()
        return fused_all

    def model_out_dim(model):
        return getattr(model, "out_dim", None) or 96

    report = {
        "schema": "agent_multi.fusion_screen.v1",
        "classification": "REPRESENTATION_DIAGNOSTIC",
        "predeclaration":
            "POSITIVE_SKILL_SCREEN_PREDECLARATION_2026_08_28.json",
        "branch_baselines": per_branch,
        "best_non_fused_branch": best_branch,
        "variants": {},
    }

    # variant 1: random current fusion (frozen) — via the REAL
    # extractor path is already measured in the temporal v2 report;
    # here the capacity-matched stand-in, frozen at random init
    torch.manual_seed(0)
    frozen = CrossAttnLikeFusion(dims)
    frozen.eval()
    with torch.no_grad():
        fused_frozen = frozen([torch.tensor(e, dtype=torch.float32)
                               for e in branch_embeddings]).numpy()
    report["variants"]["random_fusion_frozen"] = {
        "probes": probe_all(fused_frozen),
        "trainable_params": 0,
        "total_params": sum(p.numel() for p in frozen.parameters())}

    # variant 2: branch concatenation
    report["variants"]["branch_concatenation"] = {
        "probes": probe_all(concat), "trainable_params": 0,
        "total_params": 0}

    # variants 3-4: probe-trained fusions across seeds
    for name, factory in (
            ("current_fusion_probe_trained",
             lambda: CrossAttnLikeFusion(dims)),
            ("gated_softmax_probe_trained",
             lambda: GatedFusion(dims))):
        per_seed = []
        params_count = None
        for seed in SEEDS:
            torch.manual_seed(seed)
            model = factory()
            params_count = sum(p.numel()
                               for p in model.parameters())
            fused = train_fusion(model, 900, seed)
            per_seed.append(probe_all(fused))
        report["variants"][name] = {
            "per_seed": per_seed,
            "trainable_params": params_count,
            "volatility_r2_paired": paired_stats(
                [p["volatility_r2"] for p in per_seed]),
        }

    # variant 5: fine-tuned branches + fusion (bounded updates):
    # branch modules reloaded from the seal, trained jointly with the
    # fusion on raw windows
    def build_branches(seed):
        modules, columns = [], []
        for branch in contract["branches"]:
            plugin_class, _ = load_plugin("feature_branch.plugins",
                                          branch["plugin"])
            params = deep_merge_strict(plugin_class.plugin_params,
                                       branch["params"], path="p")
            torch.manual_seed(seed)
            module, _dim = plugin_class.build(
                len(branch["features"]), window, params)
            entry = manifest["artifacts"][branch["name"]]
            state = torch.load(pretrain_dir / entry["encoder_file"],
                               weights_only=True)
            strict_load_encoder(module, state, branch["name"])
            modules.append(module)
            columns.append([ordered.index(f)
                            for f in branch["features"]])
        return modules, columns

    per_seed_ft = []
    ft_params = None
    x_all = torch.tensor(windows, dtype=torch.float32)
    y_t = torch.tensor(y_vol, dtype=torch.float32).unsqueeze(-1)
    for seed in SEEDS:
        modules, columns = build_branches(seed)
        torch.manual_seed(seed)
        fusion = CrossAttnLikeFusion(dims)
        head_vol = nn.Linear(96, 1)
        params = [p for m in modules for p in m.parameters()] +             list(fusion.parameters()) + list(head_vol.parameters())
        ft_params = sum(p.numel() for p in params)
        optimizer = torch.optim.Adam(params, lr=5e-4)
        fit_t = torch.tensor(fit_i)
        generator = torch.Generator().manual_seed(seed)
        best_cal, best_states = -1e18, None
        for update in range(600):
            batch = fit_t[torch.randint(0, len(fit_t), (128,),
                                        generator=generator)]
            optimizer.zero_grad()
            parts = [m(x_all[batch][:, :, c])
                     for m, c in zip(modules, columns)]
            out = head_vol(fusion(parts))
            loss = nn.functional.mse_loss(out, y_t[batch])
            loss.backward()
            optimizer.step()
            if (update + 1) % 100 == 0 or update + 1 == 600:
                with torch.no_grad():
                    parts = [m(x_all[cal_i][:, :, c])
                             for m, c in zip(modules, columns)]
                    cal_pred = head_vol(fusion(parts)).squeeze(
                        -1).numpy()
                score = r2(cal_pred, y_vol[cal_i])
                if score > best_cal:
                    best_cal = score
                    best_states = [
                        [{k: v.clone() for k, v in
                          m.state_dict().items()} for m in modules],
                        {k: v.clone() for k, v in
                         fusion.state_dict().items()},
                        {k: v.clone() for k, v in
                         head_vol.state_dict().items()}]
        if best_states:
            for m, st in zip(modules, best_states[0]):
                m.load_state_dict(st)
            fusion.load_state_dict(best_states[1])
            head_vol.load_state_dict(best_states[2])
        with torch.no_grad():
            fused_chunks = []
            for start in range(0, len(x_all), 512):
                seg = x_all[start:start + 512]
                parts = [m(seg[:, :, c])
                         for m, c in zip(modules, columns)]
                fused_chunks.append(fusion(parts).numpy())
        per_seed_ft.append(probe_all(
            np.concatenate(fused_chunks, axis=0)))
    report["variants"]["fine_tuned_branches_plus_fusion"] = {
        "per_seed": per_seed_ft,
        "trainable_params": ft_params,
        "budget_updates": 600,
        "volatility_r2_paired": paired_stats(
            [p["volatility_r2"] for p in per_seed_ft]),
    }

    # decision per predeclared gate
    decisions = {}
    for name, entry in report["variants"].items():
        probes_list = entry.get("per_seed") or [entry["probes"]]
        def positive_any(p):
            return (p["volatility_r2"] > 0
                    or p["quantile_q0.5_pinball_neg"] > 0
                    or p["barrier_accuracy_minus_base"] > 0)
        def no_degrade(p):
            return all(p[k] >= best_branch[k] - DEGRADE_TOLERANCE
                       for k in best_branch)
        advancing = all(positive_any(p) and no_degrade(p)
                        for p in probes_list)
        decisions[name] = ("ADVANCES" if advancing
                           else "DOES_NOT_ADVANCE")
    report["decisions"] = decisions
    payload = json.dumps(report, indent=1, default=float)
    print(json.dumps({"best_non_fused_branch": best_branch,
                      "decisions": decisions}, indent=1,
                     default=float))
    if args.output:
        Path(args.output).write_text(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
