#!/usr/bin/env python3
"""Temporal-information suite v2 (scientific correction order
@c1a319c0 Priority B). REPRESENTATION_DIAGNOSTIC — never economic or
promotion authority. Decision rule, seeds and minimum effects are
PREDECLARED in TEMPORAL_SUITE_V2_PREDECLARATION_2026_08_28.json.

Corrections implemented:
* within-window permutation preserves sample/target identity
  (finding 2); the old global permutation survives only as
  `global_sample_misalignment_negative`, outside the decision rule;
* per-window phase scramble is IN the decision rule with a
  phase-dependent target (finding 1);
* 4 signal seeds x 4 random-encoder seeds, paired differences with
  t-based 95% CIs and a predeclared minimum effect (finding 3);
* lag-memory probes are chronological and out-of-sample with an
  autoregressive last-bar baseline (finding 4);
* REAL-data causal probes (quantile pinball, realized volatility,
  barrier hit) on isolated fit/calibration/monitor roles, per family
  AND fused, vs random / within-window shuffled / order-invariant
  pooled controls;
* per-feature masked next-bar reconstruction and per-band normalized
  spectral error.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

SIGNAL_SEEDS = (11, 22, 33, 44)
ENCODER_SEEDS = (101, 202, 303, 404)
MIN_EFFECT_SYNTH = 0.05
MIN_EFFECT_REAL = 0.02
PURGE = 12


def gate(deltas, minimum):
    from agent_plugins.temporal_information import paired_stats
    stats = paired_stats(list(deltas))
    stats["passes"] = bool(stats["mean"] >= minimum
                           and stats["ci95_low"] > 0.0)
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretrain-dir", required=True)
    parser.add_argument("--bars", type=int, default=1200)
    parser.add_argument("--max-windows", type=int, default=2500)
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    import numpy as np
    import torch

    from agent_plugins.branch_pretraining import (
        barrier_hit_labels, forward_log_return_targets,
        load_fit_slice, realized_volatility_targets)
    from agent_plugins.component_config import deep_merge_strict
    from agent_plugins.grouped_architecture import (
        construct_extractor, snapshot_effective_config)
    from agent_plugins.pretrained_branch_loader import (
        strict_load_encoder, verify_source)
    from agent_plugins.temporal_information import (
        chronological_roles, encode_windows, make_windows,
        order_invariant_pooled_embedding, per_window_phase_scramble,
        pinball_loss_score, probe_r2, ridge_fit_cal_score,
        synthetic_signals, within_window_permutation)
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
    snapshot = snapshot_effective_config(
        REPO / "examples/config/"
        "project3_ethusdt_4h_sac_grouped_strong_v1.json")
    window = int(snapshot["env_config"]["window_size"])

    def build_encoder(branch, load_weights: bool, seed: int = 0):
        plugin_class, _ = load_plugin("feature_branch.plugins",
                                      branch["plugin"])
        params = deep_merge_strict(plugin_class.plugin_params,
                                   branch["params"], path="p")
        torch.manual_seed(seed)
        module, _dim = plugin_class.build(
            len(branch["features"]), window, params)
        if load_weights:
            entry = manifest["artifacts"][branch["name"]]
            state = torch.load(pretrain_dir / entry["encoder_file"],
                               weights_only=True)
            strict_load_encoder(module, state, branch["name"])
        module.eval()
        return module

    report = {
        "schema": "agent_multi.temporal_information_suite.v2",
        "classification": "REPRESENTATION_DIAGNOSTIC",
        "predeclaration":
            "TEMPORAL_SUITE_V2_PREDECLARATION_2026_08_28.json",
        "candidate_seal": json.loads(
            (pretrain_dir / "generation.json").read_text())[
            "manifest_sha256"],
        "window": window,
        "synthetic": {},
        "real_data": {},
    }

    # ============ SYNTHETIC BLOCK (findings 1-3) ============
    for branch in contract["branches"]:
        family = branch["name"]
        features = len(branch["features"])
        encoder = build_encoder(branch, load_weights=True)
        randoms = [build_encoder(branch, False, seed)
                   for seed in ENCODER_SEEDS]
        per_control: dict = {"random_encoder": [], "within_window":
                             [], "phase_scramble": [], "pooled": []}
        raw_rows = []
        ceiling = False
        for signal_seed in SIGNAL_SEEDS:
            series = synthetic_signals(args.bars, features,
                                       seed=signal_seed)["series"]
            windows = make_windows(series, window)[:-1]
            target = series[window:, 0]
            n = min(len(windows), len(target))
            windows, target = windows[:n], target[:n]
            emb = encode_windows(encoder, windows)
            r_pre = probe_r2(emb, target)
            r_random = float(np.mean([
                probe_r2(encode_windows(re, windows), target)
                for re in randoms]))
            ceiling = ceiling or r_random >= 0.95
            r_within = probe_r2(encode_windows(
                encoder, within_window_permutation(
                    windows, seed=signal_seed)), target)
            r_phase = probe_r2(encode_windows(
                encoder, per_window_phase_scramble(
                    windows, seed=signal_seed)), target)
            r_pooled = probe_r2(
                order_invariant_pooled_embedding(windows), target)
            rng = np.random.default_rng(signal_seed)
            r_misaligned = probe_r2(encode_windows(
                encoder, windows[rng.permutation(len(windows))]),
                target)
            per_control["random_encoder"].append(r_pre - r_random)
            per_control["within_window"].append(r_pre - r_within)
            per_control["phase_scramble"].append(r_pre - r_phase)
            per_control["pooled"].append(r_pre - r_pooled)
            raw_rows.append({
                "signal_seed": signal_seed,
                "pretrained": r_pre, "random_mean": r_random,
                "within_window": r_within,
                "phase_scramble": r_phase, "pooled": r_pooled,
                "global_sample_misalignment_negative": r_misaligned})
        gates = {name: gate(deltas, MIN_EFFECT_SYNTH)
                 for name, deltas in per_control.items()}
        if ceiling:
            verdict = "CEILING_SATURATED_INCONCLUSIVE"
        elif all(g["passes"] for g in gates.values()):
            verdict = "PASS"
        else:
            verdict = "FAIL"
        report["synthetic"][family] = {
            "raw_per_seed": raw_rows,
            "paired_gates": gates,
            "random_ceiling_saturated": ceiling,
            "verdict": verdict,
        }

    # ============ REAL-DATA BLOCK (findings 3-4, B6-B8) ============
    # the EXACT executing chain: validate_contract -> load_fit_slice ->
    # build_step_index -> collect_preprocessed_windows (the same
    # preprocessor plugin the env calls) -> contract target builders
    from agent_plugins.branch_pretraining import (
        build_step_index, collect_preprocessed_windows,
        validate_contract)
    parsed = validate_contract(contract)
    df, ordered, close_col = load_fit_slice(data_path, contract)
    max_horizon = parsed["max_horizon_all_objectives"]
    steps = build_step_index(len(df), parsed["warmup_bars"],
                             max(1, int(args.stride)), max_horizon,
                             int(args.max_windows))
    env_config = json.loads(Path(
        contract["observation_pipeline"]["source_config"]
    ).read_text()) if Path(
        contract["observation_pipeline"]["source_config"]).is_absolute()         else json.loads((REPO / contract["observation_pipeline"][
            "source_config"]).read_text())
    windows_real = collect_preprocessed_windows(df, contract,
                                                env_config, steps)
    closes = df[close_col].to_numpy()
    quant_spec = (contract.get("objectives") or {}).get(
        "quantile_regression") or {}
    vol_spec = (contract.get("objectives") or {}).get(
        "volatility") or {}
    barrier_spec = (contract.get("objectives") or {}).get(
        "barrier_hit") or {}
    horizons = list(quant_spec.get("horizons") or [6])
    h_col = min(range(len(horizons)),
                key=lambda i: abs(horizons[i] - 6))
    y_ret = forward_log_return_targets(closes, steps,
                                       horizons)[:, h_col]
    vol_h = list(vol_spec.get("horizons") or [6])
    annualization = vol_spec.get("annualization")
    y_vol = realized_volatility_targets(
        closes, steps, vol_h, float(vol_spec.get("epsilon", 1e-8)),
        None if annualization in (None, "none")
        else annualization["periods_per_year"])[:, 0]
    y_bar = None
    if barrier_spec:
        ohlc = barrier_spec["ohlc_columns"]
        labels = barrier_hit_labels(
            df[ohlc["open"]].to_numpy(), df[ohlc["high"]].to_numpy(),
            df[ohlc["low"]].to_numpy(), df[ohlc["close"]].to_numpy(),
            steps, list(barrier_spec["horizons"]),
            int(barrier_spec["barrier_scale"]["lookback"]),
            float(barrier_spec["upper_mult"]),
            float(barrier_spec["lower_mult"]),
            float(barrier_spec["barrier_scale"]["epsilon"]))[:, 0]
        y_bar = (np.asarray(labels) == 0).astype(float)  # UPPER hit
    y_ret = np.asarray(y_ret, dtype=float)
    y_vol = np.asarray(y_vol, dtype=float)
    n = len(windows_real)
    fit_i, cal_i, mon_i = chronological_roles(n)
    fit_i = fit_i[:-PURGE] if len(fit_i) > PURGE else fit_i
    cal_i = cal_i[:-PURGE] if len(cal_i) > PURGE else cal_i

    def real_probes(embed_fn, label):
        x = embed_fn(windows_real_family)
        out = {}
        out["quantile_pinball_neg"] = {
            f"q{q}": ridge_fit_cal_score(
                x[fit_i], y_ret[fit_i], x[cal_i], y_ret[cal_i],
                x[mon_i], y_ret[mon_i],
                metric=lambda p, y, q=q: pinball_loss_score(p, y, q)
            )["score"] for q in (0.1, 0.5, 0.9)}
        out["volatility_r2"] = ridge_fit_cal_score(
            x[fit_i], y_vol[fit_i], x[cal_i], y_vol[cal_i],
            x[mon_i], y_vol[mon_i])["score"]
        if y_bar is not None:
            score = ridge_fit_cal_score(
                x[fit_i], y_bar[fit_i], x[cal_i], y_bar[cal_i],
                x[mon_i], y_bar[mon_i],
                metric=lambda p, y: float(
                    ((p > 0.5) == (y > 0.5)).mean()))
            base = float(max((y_bar[mon_i] > 0.5).mean(),
                             1 - (y_bar[mon_i] > 0.5).mean()))
            out["barrier_accuracy"] = score["score"]
            out["barrier_accuracy_minus_base"] = round(
                score["score"] - base, 4)
        return out

    def family_columns(branch):
        return [ordered.index(f) for f in branch["features"]]

    for branch in contract["branches"]:
        family = branch["name"]
        cols = family_columns(branch)
        windows_real_family = windows_real[:, :, cols]
        encoder = build_encoder(branch, load_weights=True)
        randoms = [build_encoder(branch, False, seed)
                   for seed in ENCODER_SEEDS]
        entry = {}
        entry["pretrained"] = real_probes(
            lambda w: encode_windows(encoder, w), family)
        random_scores = [real_probes(
            lambda w, re=re: encode_windows(re, w), family)
            for re in randoms]
        entry["random_encoder_mean"] = {
            "volatility_r2": round(float(np.mean(
                [r["volatility_r2"] for r in random_scores])), 4),
            "quantile_pinball_neg_q0.5": round(float(np.mean(
                [r["quantile_pinball_neg"]["q0.5"]
                 for r in random_scores])), 4),
            **({"barrier_accuracy": round(float(np.mean(
                [r["barrier_accuracy"] for r in random_scores])), 4)}
               if y_bar is not None else {})}
        shuffled_family = within_window_permutation(
            windows_real_family, seed=7)
        entry["within_window_shuffled"] = real_probes(
            lambda w: encode_windows(encoder, shuffled_family))
        entry["order_invariant_pooled"] = real_probes(
            order_invariant_pooled_embedding, family)
        margins = {
            "volatility_vs_random": round(
                entry["pretrained"]["volatility_r2"]
                - entry["random_encoder_mean"]["volatility_r2"], 4),
            "volatility_vs_shuffled": round(
                entry["pretrained"]["volatility_r2"]
                - entry["within_window_shuffled"]["volatility_r2"],
                4),
            "quantile_q0.5_vs_random": round(
                entry["pretrained"]["quantile_pinball_neg"]["q0.5"]
                - entry["random_encoder_mean"][
                    "quantile_pinball_neg_q0.5"], 4)}
        if y_bar is not None:
            margins["barrier_vs_random"] = round(
                entry["pretrained"]["barrier_accuracy"]
                - entry["random_encoder_mean"]["barrier_accuracy"], 4)
        entry["margins"] = margins
        entry["informative"] = {
            k: bool(v >= MIN_EFFECT_REAL)
            for k, v in margins.items()}
        # masked next-bar per-feature reconstruction (monitor R2)
        masked = windows_real_family.copy()
        masked[:, -1, :] = 0.0
        emb_masked = encode_windows(encoder, masked)
        truth = windows_real_family[:, -1, :]
        recon = {}
        for j, name in enumerate(branch["features"]):
            result = ridge_fit_cal_score(
                emb_masked[fit_i], truth[fit_i, j],
                emb_masked[cal_i], truth[cal_i, j],
                emb_masked[mon_i], truth[mon_i, j])
            recon[name] = result["score"]
        entry["masked_newest_bar_reconstruction_r2"] = recon
        # OOS lag memory with AR last-bar baseline + controls
        emb_full = encode_windows(encoder, windows_real_family)
        emb_shuffled = encode_windows(encoder, shuffled_family)
        emb_random0 = encode_windows(randoms[0], windows_real_family)
        newest = windows_real_family[:, -1, 0:1]
        lag_block = {}
        for lag in (2, 4, 8, 16):
            y_lag = windows_real_family[:, -1 - lag, 0]
            lag_block[f"lag_{lag}"] = {
                "pretrained": ridge_fit_cal_score(
                    emb_full[fit_i], y_lag[fit_i], emb_full[cal_i],
                    y_lag[cal_i], emb_full[mon_i],
                    y_lag[mon_i])["score"],
                "ar_last_bar_baseline": ridge_fit_cal_score(
                    newest[fit_i], y_lag[fit_i], newest[cal_i],
                    y_lag[cal_i], newest[mon_i],
                    y_lag[mon_i])["score"],
                "random_encoder": ridge_fit_cal_score(
                    emb_random0[fit_i], y_lag[fit_i],
                    emb_random0[cal_i], y_lag[cal_i],
                    emb_random0[mon_i], y_lag[mon_i])["score"],
                "within_window_shuffled": ridge_fit_cal_score(
                    emb_shuffled[fit_i], y_lag[fit_i],
                    emb_shuffled[cal_i], y_lag[cal_i],
                    emb_shuffled[mon_i], y_lag[mon_i])["score"],
            }
        entry["lag_memory_oos"] = lag_block
        # per-band normalized spectral error of the decoded feature 0
        xb = np.hstack([emb_full[np.concatenate([fit_i, cal_i])],
                        np.ones((len(fit_i) + len(cal_i), 1))])
        gram = xb.T @ xb + 1e-2 * np.eye(xb.shape[1])
        coef = np.linalg.solve(
            gram, xb.T @ truth[np.concatenate([fit_i, cal_i]), 0])
        decoded = np.hstack([emb_full[mon_i],
                             np.ones((len(mon_i), 1))]) @ coef
        f_true = np.abs(np.fft.rfft(truth[mon_i, 0]))
        f_pred = np.abs(np.fft.rfft(decoded))
        bands = np.array_split(np.arange(1, len(f_true)), 3)
        spectral = {}
        for name, band in zip(("low", "mid", "high"), bands):
            denominator = float(f_true[band].sum()) or 1.0
            spectral[name] = round(float(
                np.abs(f_true[band] - f_pred[band]).sum()
                / denominator), 4)
        entry["spectral_normalized_error_by_band"] = spectral
        report["real_data"][family] = entry

    windows_real_family = windows_real  # fused consumes all features
    # fused representation
    import gymnasium as gym
    spaces = {"features": gym.spaces.Box(
        -np.inf, np.inf, (window, len(ordered)), dtype=np.float32)}
    for key in snapshot["materialized"]["architecture"].get(
            "state_keys") or []:
        spaces[key] = gym.spaces.Box(-np.inf, np.inf, (1,),
                                     dtype=np.float32)
    obs_space = gym.spaces.Dict(spaces)

    def fused_embed(w, seed=0, load=True):
        torch.manual_seed(seed)
        extractor = construct_extractor(snapshot["materialized"],
                                        obs_space)
        if load:
            from agent_plugins.pretrained_branch_loader import (
                load_family_encoders)
            load_family_encoders(pretrain_dir, manifest, contract,
                                 extractor)
        extractor.eval()
        chunks = []
        with torch.no_grad():
            for start in range(0, len(w), 256):
                part = torch.tensor(w[start:start + 256])
                batch = {"features": part}
                for key in spaces:
                    if key != "features":
                        batch[key] = torch.zeros(len(part), 1)
                chunks.append(extractor(batch).numpy())
        return np.concatenate(chunks, axis=0)

    fused_entry = {"pretrained": real_probes(
        lambda w: fused_embed(w), "fused")}
    fused_entry["random_extractor"] = real_probes(
        lambda w: fused_embed(w, seed=101, load=False), "fused")
    shuffled_full = within_window_permutation(windows_real, seed=7)
    fused_entry["within_window_shuffled"] = real_probes(
        lambda w: fused_embed(shuffled_full), "fused")
    fused_entry["margins"] = {
        "volatility_vs_random": round(
            fused_entry["pretrained"]["volatility_r2"]
            - fused_entry["random_extractor"]["volatility_r2"], 4),
        "volatility_vs_shuffled": round(
            fused_entry["pretrained"]["volatility_r2"]
            - fused_entry["within_window_shuffled"][
                "volatility_r2"], 4)}
    report["real_data"]["fused"] = fused_entry

    report["summary"] = {
        "synthetic": {f: e["verdict"]
                      for f, e in report["synthetic"].items()},
        "real_volatility_margins_vs_random": {
            f: e.get("margins", {}).get("volatility_vs_random")
            for f, e in report["real_data"].items()},
    }
    payload = json.dumps(report, indent=1, default=float)
    print(json.dumps(report["summary"], indent=1, default=float))
    if args.output:
        Path(args.output).write_text(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
