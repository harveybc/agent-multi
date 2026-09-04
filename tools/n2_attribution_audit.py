#!/usr/bin/env python3
"""N2_ATTRIBUTION_AUDIT (order agent-multi@4c1f1532 §3 C3;
contract sealed in N2_ATTRIBUTION_AUDIT_CONTRACT_2026_09_04.json
BEFORE this result artifact).

Development-only attribution on the frozen N2 arrays: five
equal-information arms per barrier target, exact additive log-loss
decomposition (hit-vs-censored + direction-given-hit), calibration
tables, paired per-observation differences, eight-contrast Holm
family with within-window block bootstrap, collision and
block-length sensitivities (exploratory). CPU only."""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from agent_plugins.experiment_runtime import sha_file  # noqa: E402
from agent_plugins.paired_inference import holm_adjust  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "target_horizon_census_n2",
    REPO / "tools" / "target_horizon_census_n2.py")
tcn2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tcn2)

CONTRACT = ("docs/audits/evidence/"
            "N2_ATTRIBUTION_AUDIT_CONTRACT_2026_09_04.json")
BUNDLE = ("docs/audits/evidence/"
          "TARGET_HORIZON_CENSUS_N2_BUNDLE_2026_09_03.json")
NPZ_SHA = ("07c5ff085dfd8bab0dfa33d038005c8fdb2d6c2acff3961d0fe4b"
           "042ef57cca7")
BOOT_SEED = 606
BOOT_B = 2000
MARGIN_SCALE_VS_PRIOR = 0.01
MARGIN_INCREMENTAL = 0.005
CONTRASTS = (("arm2", "arm1"), ("arm3", "arm2"),
             ("arm4", "arm1"), ("arm5", "arm2"))
INCREMENTAL = {("arm3", "arm2"), ("arm5", "arm2")}
TARGETS = {"bar_h6": (0, 6), "bar_h12": (1, 12)}


def _logloss3(probs, y):
    import numpy as np
    return -np.log(np.clip(
        probs[np.arange(len(y)), y.astype(int)], 1e-12, None))


def decompose(probs, y):
    """Exact additive split: L_multi = L_hit + L_dir (L_dir only on
    hit rows)."""
    import numpy as np
    p_hit = np.clip(probs[:, 0] + probs[:, 1], 1e-12, 1 - 1e-12)
    is_hit = y < 2
    l_hit = np.where(is_hit, -np.log(p_hit), -np.log(1 - p_hit))
    l_dir = np.zeros(len(y))
    hit_idx = np.where(is_hit)[0]
    if len(hit_idx):
        p_cls = probs[hit_idx, y[hit_idx].astype(int)]
        l_dir[hit_idx] = -np.log(np.clip(
            p_cls / p_hit[hit_idx], 1e-12, None))
    return l_hit, l_dir, is_hit


def _boot_p(diff_windows, block, seed=BOOT_SEED):
    import numpy as np
    rng = np.random.default_rng(seed)
    n_low = 0
    for _ in range(BOOT_B):
        parts = []
        for d in diff_windows:
            n = len(d)
            n_blocks = math.ceil(n / block)
            starts = rng.integers(0, n, size=n_blocks)
            idx = (starts[:, None]
                   + np.arange(block)[None, :]).reshape(-1) % n
            parts.append(d[idx[:n]])
        if float(np.concatenate(parts).mean()) <= 0.0:
            n_low += 1
    return (1 + n_low) / (BOOT_B + 1)


def _fmt_p(p):
    return "<= 1/2001" if p <= 1.0 / (BOOT_B + 1) + 1e-12 \
        else round(p, 6)


def decide(contrast_stats: dict) -> str:
    """Pure verdict rule from the sealed contract. contrast_stats:
    {(target, a, b): {"pooled_skill", "all_windows_positive",
    "holm_p"}}."""
    def ok(key, margin):
        s = contrast_stats[key]
        return (s["all_windows_positive"]
                and s["pooled_skill"] >= margin
                and s["holm_p"] < 0.05)
    incremental_hit = any(
        ok((t, a, b), MARGIN_INCREMENTAL)
        for t in TARGETS for (a, b) in INCREMENTAL)
    if incremental_hit:
        return "INCREMENTAL_DEVELOPMENT_STRUCTURE_OBSERVED"
    scale_ok = all(ok((t, "arm2", "arm1"), MARGIN_SCALE_VS_PRIOR)
                   for t in TARGETS)
    if scale_ok:
        return "BARRIER_SIGNAL_EXPLAINED_BY_TARGET_DEFINITION_SCALE"
    return "ATTRIBUTION_INCONCLUSIVE"


def collision_masks(pretrain_dir: Path):
    """Same-bar-collision sensitivity mask per horizon, from the
    contract's exact validated OHLC/levels/scale (metadata only)."""
    import numpy as np
    from agent_plugins.branch_pretraining import (
        build_step_index, load_fit_slice, realized_volatility_targets,
        validate_contract, validate_ohlc)
    from agent_plugins.pretrained_branch_loader import verify_source
    split_contract = json.loads(
        (REPO / "examples/config/phase_3_eth_sac_dynamics/splits/"
         "eth_nested_split_contract_o2022_paired_v1.json").read_text())
    data_path = Path(split_contract["source_csv"])
    source = verify_source(pretrain_dir, REPO, data_path)
    contract = source["contract"]
    parsed = validate_contract(contract)
    df, ordered, close_col = load_fit_slice(data_path, contract)
    bar_spec = contract["objectives"]["barrier_hit"]
    ohlc = bar_spec["ohlc_columns"]
    scale_spec = bar_spec["barrier_scale"]
    closes = df[close_col].to_numpy()
    warmup = max(int(parsed["warmup_bars"]), tcn2.WINDOW)
    steps = build_step_index(len(df), warmup, tcn2.STRIDE, 12, 2200)
    vol_spec = contract["objectives"]["volatility"]
    eps = float(vol_spec.get("epsilon", 1e-8))
    vol_h6 = realized_volatility_targets(
        closes, steps, [6], eps, None)[:, 0]
    trail_h6 = realized_volatility_targets(
        closes, [max(0, t - 6) for t in steps], [6], eps,
        None)[:, 0]
    keep = np.isfinite(vol_h6) & np.isfinite(trail_h6)
    arrays = validate_ohlc(df[ohlc["open"]].to_numpy(),
                           df[ohlc["high"]].to_numpy(),
                           df[ohlc["low"]].to_numpy(), closes)
    log_close = np.log(arrays["close"])
    returns = np.diff(log_close)
    anchor = np.asarray(steps) - 1
    lookback = int(scale_spec["lookback"])
    trail = np.stack([returns[anchor - lookback + i]
                      for i in range(lookback)], axis=1)
    scale = np.sqrt((trail ** 2).mean(axis=1)) \
        + float(scale_spec["epsilon"])
    upper_level = arrays["close"][anchor] * (
        1.0 + float(bar_spec["upper_mult"]) * scale)
    lower_level = arrays["close"][anchor] * (
        1.0 - float(bar_spec["lower_mult"]) * scale)
    max_h = 12
    future_high = np.stack(
        [arrays["high"][anchor + i] for i in range(1, max_h + 1)],
        axis=1)
    future_low = np.stack(
        [arrays["low"][anchor + i] for i in range(1, max_h + 1)],
        axis=1)
    upper_hit = future_high >= upper_level[:, None]
    lower_hit = future_low <= lower_level[:, None]
    n = len(anchor)
    first_upper = np.where(upper_hit.any(axis=1),
                           upper_hit.argmax(axis=1), max_h + 1)
    first_lower = np.where(lower_hit.any(axis=1),
                           lower_hit.argmax(axis=1), max_h + 1)
    same_bar = first_upper == first_lower
    masks = {}
    for key, (hidx, h) in TARGETS.items():
        masks[key] = (same_bar & (first_upper < h))[keep]
    return masks


def run_audit(run_root: Path, pretrain_dir: Path,
              out_path: Path) -> dict:
    import numpy as np
    npz = run_root / "inputs" / "census_inputs.npz"
    if sha_file(npz) != NPZ_SHA:
        raise RuntimeError("census_inputs.npz digest mismatch — "
                           "refusing to audit drifted arrays")
    data = np.load(npz, allow_pickle=False)
    bundle = json.loads((REPO / BUNDLE).read_text())
    geometry = bundle["ledger"]["role_geometry"]
    window_keys = sorted(geometry["windows"])
    masks = collision_masks(pretrain_dir)
    summary, barscale = data["summary"], data["barscale"]
    report = {"schema": "agent_multi.n2_attribution_audit.v1",
              "contract": CONTRACT,
              "npz_sha256": NPZ_SHA,
              "targets": {}, "contrasts": {},
              "sensitivities_exploratory": {},
              "self_checks": {}}
    losses = {}       # (target, arm, wk) -> per-obs multiclass loss
    losses_nc = {}    # collision-excluded sums
    for key, (hidx, h) in TARGETS.items():
        y_full = data["bar"][:, hidx]
        # self-check: adverse-first labels on collision rows
        col = masks[key]
        col_rows = int(col.sum())
        col_labels_ok = bool((y_full[col] == 1).all()) \
            if col_rows else True
        trec = {"collision_rows_total": col_rows,
                "collision_labels_all_adverse": col_labels_ok,
                "windows": {}}
        for wk in window_keys:
            roles = geometry["windows"][wk]
            fit_i = np.arange(*roles["fit"])
            cal_i = np.arange(*roles["cal"])
            sc_i = np.arange(*roles["score"])
            yf, yc, ys = (y_full[fit_i], y_full[cal_i],
                          y_full[sc_i])
            fc = np.concatenate([fit_i, cal_i])
            arms = {}
            counts = np.bincount(y_full[fc].astype(int),
                                 minlength=3)
            prior = np.clip(counts / counts.sum(), 1e-12, None)
            prior = prior / prior.sum()
            arms["arm1"] = (np.tile(prior, (len(ys), 1)), None)
            for arm, x in (("arm2", barscale[:, 0:1]),
                           ("arm3", barscale),
                           ("arm4", summary),
                           ("arm5", np.concatenate(
                               [barscale, summary], axis=1))):
                probs, rec = tcn2._logistic(
                    x[fit_i], yf, x[cal_i], yc, x[sc_i])
                if probs is None:
                    raise RuntimeError(
                        f"{key} {wk} {arm}: degenerate fit — "
                        "ATTRIBUTION_INCONCLUSIVE")
                arms[arm] = (probs, rec)
            wrec = {"n_score": len(sc_i),
                    "class_support_score": {
                        str(c): int((ys == c).sum())
                        for c in (0, 1, 2)},
                    "collision_rows": int(
                        masks[key][sc_i].sum()),
                    "arms": {}}
            for arm, (probs, rec) in arms.items():
                lm = _logloss3(probs, ys)
                l_hit, l_dir, is_hit = decompose(probs, ys)
                assert np.allclose(lm, l_hit + l_dir, atol=1e-9), \
                    "decomposition not additive"
                losses[(key, arm, wk)] = lm
                keep_nc = ~masks[key][sc_i]
                losses_nc[(key, arm, wk)] = lm[keep_nc]
                onehot = np.eye(3)[ys.astype(int)]
                pred_cls = probs.argmax(axis=1)
                deciles = []
                edges = np.quantile(probs[:, 0] + probs[:, 1],
                                    np.linspace(0, 1, 11))
                p_hit = probs[:, 0] + probs[:, 1]
                for d in range(10):
                    lo, hi = edges[d], edges[d + 1]
                    sel = (p_hit >= lo) & (
                        p_hit <= hi if d == 9 else p_hit < hi)
                    if sel.sum():
                        deciles.append(
                            {"bin": d,
                             "n": int(sel.sum()),
                             "mean_predicted": round(
                                 float(p_hit[sel].mean()), 4),
                             "empirical_hit_rate": round(
                                 float(is_hit[sel].mean()), 4)})
                wrec["arms"][arm] = {
                    "multiclass_logloss": round(
                        float(lm.mean()), 6),
                    "hit_vs_censored_logloss": round(
                        float(l_hit.mean()), 6),
                    "direction_given_hit_logloss": round(
                        float(l_dir[is_hit].mean()), 6)
                    if is_hit.any() else None,
                    "brier": round(float(
                        ((probs - onehot) ** 2).sum(axis=1)
                        .mean()), 6),
                    "brier_components": {
                        str(c): round(float(
                            ((probs[:, c]
                              - (ys == c)) ** 2).mean()), 6)
                        for c in (0, 1, 2)},
                    "recall_argmax_exploratory": {
                        str(c): round(float(
                            ((pred_cls == c) & (ys == c)).sum()
                            / max(1, (ys == c).sum())), 4)
                        for c in (0, 1, 2)},
                    "fit_record": rec,
                    "calibration_deciles_hit": deciles}
            trec["windows"][wk] = wrec
        report["targets"][key] = trec
        report["self_checks"][key] = {
            "collision_labels_all_adverse": col_labels_ok,
            "decomposition_additive": True}

    # ---- contrasts, bootstrap, Holm ----
    pvals = {}
    stats = {}
    for key in TARGETS:
        for (a, b) in CONTRASTS:
            diffs = [losses[(key, b, wk)] - losses[(key, a, wk)]
                     for wk in window_keys]
            per_window_skill = {
                wk: round(1.0 - float(
                    losses[(key, a, wk)].sum()
                    / losses[(key, b, wk)].sum()), 6)
                for wk in window_keys}
            pooled = round(1.0 - float(
                sum(losses[(key, a, wk)].sum()
                    for wk in window_keys)
                / sum(losses[(key, b, wk)].sum()
                      for wk in window_keys)), 6)
            p = _boot_p(diffs, 6)
            ckey = f"{key}:{a}-vs-{b}"
            pvals[ckey] = min(1.0, p)
            stats[(key, a, b)] = {
                "pooled_skill": pooled,
                "per_window_skill": per_window_skill,
                "all_windows_positive": all(
                    v > 0 for v in per_window_skill.values()),
                "bootstrap_p": _fmt_p(p), "_p_raw": p}
    holm = holm_adjust(pvals)
    contrast_stats = {}
    for (key, a, b), s in stats.items():
        ckey = f"{key}:{a}-vs-{b}"
        s["holm_p"] = holm[ckey]
        contrast_stats[(key, a, b)] = {
            "pooled_skill": s["pooled_skill"],
            "all_windows_positive": s["all_windows_positive"],
            "holm_p": s["holm_p"]}
        out = dict(s)
        out.pop("_p_raw")
        out["holm_p"] = _fmt_p(s["holm_p"]) \
            if isinstance(s["holm_p"], float) else s["holm_p"]
        report["contrasts"][ckey] = out
    report["verdict"] = decide(contrast_stats)

    # ---- sensitivities (exploratory) ----
    sens = {}
    for key in TARGETS:
        for (a, b) in CONTRASTS:
            ckey = f"{key}:{a}-vs-{b}"
            pooled_nc = round(1.0 - float(
                sum(losses_nc[(key, a, wk)].sum()
                    for wk in window_keys)
                / sum(losses_nc[(key, b, wk)].sum()
                      for wk in window_keys)), 6)
            diffs = [losses[(key, b, wk)] - losses[(key, a, wk)]
                     for wk in window_keys]
            sens[ckey] = {
                "pooled_skill_without_collisions": pooled_nc,
                "bootstrap_p_by_block": {
                    str(blk): _fmt_p(_boot_p(diffs, blk))
                    for blk in (3, 6, 12)}}
    report["sensitivities_exploratory"] = sens

    # ---- reproduction of the order's post-hoc values ----
    # (arm3-vs-arm1 is outside the Holm family; pooled only)
    report["order_reproduction"] = {
        "note": "recomputed here, never copied as authority"}
    for key, label in (("bar_h6", "h6"), ("bar_h12", "h12")):
        pooled31 = round(1.0 - float(
            sum(losses[(key, "arm3", wk)].sum()
                for wk in window_keys)
            / sum(losses[(key, "arm1", wk)].sum()
                  for wk in window_keys)), 6)
        pooled21 = report["contrasts"][
            f"{key}:arm2-vs-arm1"]["pooled_skill"]
        report["order_reproduction"][label] = {
            "scale_plus_lags_vs_fitcal_prior": pooled31,
            "scale_only_vs_fitcal_prior": pooled21,
            "incremental_lags": round(pooled31 - pooled21, 6),
            "order_values": {
                "h6": {"scale_lags": 0.023177,
                       "scale_only": 0.022736,
                       "incremental": 0.000441},
                "h12": {"scale_lags": 0.021533,
                        "scale_only": 0.021161,
                        "incremental": 0.000372}}[label]}
    out_path.write_text(json.dumps(report, indent=1,
                                   default=float) + "\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--pretrain-dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    report = run_audit(Path(args.run_root),
                       Path(args.pretrain_dir), Path(args.out))
    print(json.dumps({"verdict": report["verdict"],
                      "reproduction": report["order_reproduction"]},
                     indent=1, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
