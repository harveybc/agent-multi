#!/usr/bin/env python3
"""TARGET_HORIZON_DATA_CENSUS_N2 (order agent-multi@8fce8da0 §5-§6;
predeclared and sealed pre-result in
TARGET_HORIZON_DATA_CENSUS_N2_PREDECLARATION_2026_09_03.json).

Development-only census over the nine target-horizon candidates fixed
by the accepted pretraining contract: forward log return h1/3/6/12,
realized volatility h3/6/12, barrier hit h6/12. CPU-only, three cheap
models per family (proper baseline / target-history lags / regularized
linear on a fixed 249-feature causal summary of the 83 inputs), four
disjoint causal score windows, uniform embargo 3 sampled rows,
dependence-aware circular block bootstrap, Holm across the nine
candidates via the R3-repaired helper, negative controls (future-leak
sentinel + causally shifted target). May SELECT at most two candidates
for a later confirmation order; makes no confirmatory claim.

Subcommands: materialize / worker / supervise / aggregate."""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from agent_plugins.experiment_runtime import (  # noqa: E402
    RunDirectory, RuntimePreflightError, atomic_write_json,
    aggregate as runtime_aggregate, run_one_unit, sha_file, sha_obj,
    unit_id)
from agent_plugins.paired_inference import holm_adjust  # noqa: E402

EXPERIMENT = "target_horizon_census_n2"
WINDOW = 64
STRIDE = 4
EMBARGO = 3  # uniform: max over candidates of ceil(h/stride)
PREDECLARATION = ("docs/audits/evidence/TARGET_HORIZON_DATA_CENSUS_"
                  "N2_PREDECLARATION_2026_09_03.json")
N1_INPUT_DIGEST = ("0f31661c339556bc96f51d9e77514ac6b5d519a7a4ba64c"
                   "00a8786d94ecfc884")
MARGIN = 0.02
BOOT_B = 2000
BOOT_SEED = 505
BLOCK = 6
MIN_EFFECTIVE_BLOCKS = 8
CLASS_SUPPORT_MIN = 30
LEAK_SKILL_MIN = 0.5
SHIFT_ROWS = 37
RIDGE_LAMBDAS = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)
LOGISTIC_CS = (0.01, 0.1, 1.0, 10.0)
EPS = 1e-8

RET_HORIZONS = (1, 3, 6, 12)
VOL_HORIZONS = (3, 6, 12)
BAR_HORIZONS = (6, 12)

# treatment key -> (family, horizon-index, mode)
CANDIDATES = {
    **{f"ret_h{h}": ("ret", i, "candidate")
       for i, h in enumerate(RET_HORIZONS)},
    **{f"vol_h{h}": ("vol", i, "candidate")
       for i, h in enumerate(VOL_HORIZONS)},
    **{f"bar_h{h}": ("bar", i, "candidate")
       for i, h in enumerate(BAR_HORIZONS)},
}
CONTROLS = {
    "ctl_leak_ret_h6": ("ret", 2, "leak"),
    "ctl_leak_vol_h6": ("vol", 1, "leak"),
    "ctl_leak_bar_h6": ("bar", 0, "leak"),
    "ctl_shift_ret_h6": ("ret", 2, "shift"),
    "ctl_shift_vol_h6": ("vol", 1, "shift"),
    "ctl_shift_bar_h6": ("bar", 0, "shift"),
}
ALL_TREATMENTS = {**CANDIDATES, **CONTROLS}
FAMILY_BASELINES = {"ret": ("zero_return", "fit_mean"),
                    "vol": ("literal_trailing", "calibrated_ar1"),
                    "bar": ("fit_class_prior",)}
HORIZON_OF = {"ret": RET_HORIZONS, "vol": VOL_HORIZONS,
              "bar": BAR_HORIZONS}


def code_digest() -> str:
    files = [REPO / "tools/target_horizon_census_n2.py",
             REPO / "agent_plugins/experiment_runtime.py",
             REPO / "agent_plugins/branch_pretraining.py",
             REPO / "agent_plugins/temporal_information.py",
             REPO / "agent_plugins/paired_inference.py"]
    return sha_obj({str(f.relative_to(REPO)): sha_file(f)
                    for f in files})


def role_geometry(n: int) -> dict:
    """Identical derivation to the frozen N1 runner, embargo 3."""
    origin0_limit = int(n * 0.85)
    frontier = int(origin0_limit * 0.82)
    cal_len = int(0.08 * n)
    length = (frontier - int(0.30 * n) - 3 * EMBARGO) // 4
    windows = {}
    end = frontier
    for k in range(4, 0, -1):
        start = end - length
        cal_lo = start - cal_len - EMBARGO
        if cal_lo <= 0 or length < 30:
            return {"sufficient": False, "n": n,
                    "reason": f"window {k} infeasible "
                              f"(cal_lo={cal_lo}, L={length})"}
        windows[f"w{k}"] = {
            "fit": [0, cal_lo], "cal": [cal_lo, start - EMBARGO],
            "score": [start, end]}
        end = start - EMBARGO
    spans = sorted(v["score"] for v in windows.values())
    for a, b in zip(spans, spans[1:]):
        if b[0] - a[1] < EMBARGO:
            return {"sufficient": False, "n": n,
                    "reason": "score windows closer than the embargo"}
    return {"sufficient": True, "n": n, "frontier": frontier,
            "embargo_rows": EMBARGO, "cal_len": cal_len,
            "score_len": length, "windows": windows}


def _identity(key: str, window_key: str) -> dict:
    fam = ALL_TREATMENTS[key][0]
    return {"experiment": EXPERIMENT, "family": fam,
            "window": WINDOW, "latent": 0, "budget": 0, "seed": 0,
            "origin": window_key, "treatment": key}


# ------------------------------------------------------------------ #
# materialization — executable builders only, no duplicated formulas #
# ------------------------------------------------------------------ #

def materialize_inputs(root: Path, pretrain_dir: Path,
                       n1_inputs: Path, *, max_windows: int,
                       stride: int) -> dict:
    import numpy as np
    from agent_plugins.branch_pretraining import (
        barrier_hit_labels, build_step_index,
        collect_preprocessed_windows, forward_log_return_targets,
        load_fit_slice, realized_volatility_targets,
        validate_contract)
    from agent_plugins.pretrained_branch_loader import verify_source
    split_contract = json.loads(
        (REPO / "examples/config/phase_3_eth_sac_dynamics/splits/"
         "eth_nested_split_contract_o2022_paired_v1.json").read_text())
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
    bar_spec = contract["objectives"]["barrier_hit"]
    ohlc = bar_spec["ohlc_columns"]
    scale_spec = bar_spec["barrier_scale"]
    env_source = contract["observation_pipeline"]["source_config"]
    env_config = json.loads(
        (Path(env_source) if Path(env_source).is_absolute()
         else REPO / env_source).read_text())
    (root / "inputs").mkdir(parents=True, exist_ok=True)
    path = root / "inputs" / "census_inputs.npz"
    if not path.exists():
        # exact N1 sampling; the frozen N1 input npz is REQUIRED as
        # the reproduction anchor (predeclared digest)
        if sha_file(n1_inputs) != N1_INPUT_DIGEST:
            raise RuntimePreflightError(
                "N1 input npz does not match the predeclared frozen "
                "digest — reproduction anchor unavailable")
        warmup = max(int(parsed["warmup_bars"]), WINDOW)
        steps = build_step_index(len(df), warmup, max(1, stride),
                                 12, max_windows)
        contract_w = {**contract, "window_size": WINDOW}
        env_w = {**env_config, "window_size": WINDOW}
        windows = collect_preprocessed_windows(df, contract_w, env_w,
                                               steps)
        eps = float(vol_spec.get("epsilon", EPS))
        vol_h6 = realized_volatility_targets(
            closes, steps, [6], eps, periods)[:, 0]
        trail_h6 = realized_volatility_targets(
            closes, [max(0, t - 6) for t in steps], [6], eps,
            periods)[:, 0]
        keep = np.isfinite(vol_h6) & np.isfinite(trail_h6)
        # prove this pipeline reproduces the N1 materialization
        n1 = np.load(n1_inputs, allow_pickle=False)
        same = (np.array_equal(
                    n1["windows"],
                    windows[keep].astype("float32"))
                and np.array_equal(
                    n1["target"],
                    np.asarray(vol_h6)[keep].astype("float64"))
                and np.array_equal(
                    n1["trailing"],
                    np.asarray(trail_h6)[keep].astype("float64")))
        if not same:
            raise RuntimePreflightError(
                "N2 materialization does NOT reproduce the frozen "
                "N1 arrays — sampling drift, refusing")
        anchor = np.asarray(steps) - 1
        if anchor.min() < WINDOW + 4 * 3 + 1:
            raise RuntimePreflightError(
                f"anchor {anchor.min()} too early for lagged "
                "trailing features")
        ret = forward_log_return_targets(
            closes, steps, list(RET_HORIZONS)).astype("float64")
        vol = realized_volatility_targets(
            closes, steps, list(VOL_HORIZONS), eps,
            periods).astype("float64")
        bar = barrier_hit_labels(
            df[ohlc["open"]].to_numpy(),
            df[ohlc["high"]].to_numpy(),
            df[ohlc["low"]].to_numpy(),
            closes, steps, list(BAR_HORIZONS),
            int(scale_spec["lookback"]),
            float(bar_spec["upper_mult"]),
            float(bar_spec["lower_mult"]),
            float(scale_spec["epsilon"]))
        returns = np.diff(np.log(closes))
        rlags = np.stack([returns[anchor - 12 + j]
                          for j in range(12)], axis=1)
        voltrail = np.zeros((len(steps), len(VOL_HORIZONS), 4))
        for hi, h in enumerate(VOL_HORIZONS):
            for k in range(4):
                shifted = [t - h - 4 * k for t in steps]
                if min(shifted) < 2:
                    raise RuntimePreflightError(
                        f"trailing lag {k} for h{h} underflows")
                voltrail[:, hi, k] = realized_volatility_targets(
                    closes, shifted, [h], eps, periods)[:, 0]
        lookback = int(scale_spec["lookback"])
        barscale = np.zeros((len(steps), 4))
        for k in range(4):
            a2 = anchor - 4 * k
            if a2.min() < lookback + 1:
                raise RuntimePreflightError(
                    f"barrier-scale lag {k} underflows")
            trail = np.stack([returns[a2 - lookback + i]
                              for i in range(lookback)], axis=1)
            barscale[:, k] = np.sqrt(
                (trail ** 2).mean(axis=1)) + float(
                    scale_spec["epsilon"])
        summary = np.concatenate(
            [windows[:, -1, :], windows.mean(axis=1),
             windows.std(axis=1)], axis=1).astype("float64")
        arrays = {"summary": summary[keep], "ret": ret[keep],
                  "vol": vol[keep], "bar": bar[keep],
                  "rlags": rlags[keep],
                  "voltrail": voltrail[keep],
                  "barscale": barscale[keep]}
        for name, arr in arrays.items():
            if name != "bar" and not np.isfinite(arr).all():
                raise RuntimePreflightError(
                    f"non-finite values in {name} after keep mask")
        np.savez_compressed(path, **arrays)
    return {"census_inputs": sha_file(path),
            "n1_inputs_reproduced": N1_INPUT_DIGEST,
            "data_csv": sha_file(data_path),
            "pretrain_generation": sha_file(
                Path(pretrain_dir) / "generation.json"),
            "pretrain_manifest": sha_file(
                Path(pretrain_dir) / "pretrain_manifest.json"),
            "code": code_digest(),
            "config": sha_file(REPO / PREDECLARATION)}


def materialize(root: Path, pretrain_dir: Path, n1_inputs: Path, *,
                max_windows: int, stride: int) -> dict:
    import numpy as np
    digests = materialize_inputs(root, pretrain_dir, n1_inputs,
                                 max_windows=max_windows,
                                 stride=stride)
    data = np.load(root / "inputs" / "census_inputs.npz",
                   allow_pickle=False)
    geometry = role_geometry(len(data["ret"]))
    if not geometry["sufficient"]:
        run = RunDirectory(root / "census")
        atomic_write_json(run.root / "INSUFFICIENT_UNITS.json", {
            "schema": "agent_multi.target_census_insufficient.v1",
            "verdict": "INCONCLUSIVE",
            "geometry": geometry})
        return {"units": 0, "verdict": "INCONCLUSIVE",
                "geometry": geometry}
    units = [_identity(key, wk)
             for wk in sorted(geometry["windows"])
             for key in ALL_TREATMENTS]
    run = RunDirectory(root / "census")
    digest = run.write_ledger({
        "schema": "agent_multi.target_census_ledger.v1",
        "experiment": EXPERIMENT,
        "units": [{"unit_id": unit_id(u), "identity": u}
                  for u in units],
        "digests": digests,
        "campaign_wall_ceiling_s": 7200.0,
        "unit_timeout_s": 600.0,
        "predeclaration": PREDECLARATION,
        "role_geometry": geometry})
    return {"units": len(units), "ledger_digest": digest,
            "geometry": geometry}


# ------------------------------------------------------------------ #
# per-family losses and estimators                                   #
# ------------------------------------------------------------------ #

def _se(pred, y):
    return (pred - y) ** 2


def _qlike(pred_log, y_log):
    import numpy as np
    f = np.maximum(np.exp(pred_log) - EPS, EPS) ** 2
    r = np.maximum(np.exp(y_log) - EPS, EPS) ** 2
    ratio = r / f
    return ratio - np.log(ratio) - 1.0


def _logloss(probs, y):
    import numpy as np
    return -np.log(np.clip(
        probs[np.arange(len(y)), y.astype(int)], 1e-12, None))


def _standardize(x_fit, *rest):
    import numpy as np
    mu = x_fit.mean(axis=0)
    sd = np.clip(x_fit.std(axis=0), 1e-12, None)
    return [(x_fit - mu) / sd] + [(x - mu) / sd for x in rest]


def _ridge(x_fit, y_fit, x_cal, y_cal, x_sc, loss_fn):
    """Grid on fit, lambda by calibration primary loss, refit on
    fit+cal (predeclared), predict score. Returns (pred, record)."""
    import numpy as np
    xf, xc, xs = _standardize(x_fit, x_cal, x_sc)

    def fit(x, y, lam):
        xb = np.hstack([x, np.ones((len(x), 1))])
        penalty = lam * np.eye(xb.shape[1])
        penalty[-1, -1] = 0.0  # intercept unpenalized
        gram = xb.T @ xb + penalty
        return np.linalg.solve(gram, xb.T @ y), float(
            np.linalg.cond(gram))

    def predict(x, coef):
        return np.hstack([x, np.ones((len(x), 1))]) @ coef

    best = None
    for lam in RIDGE_LAMBDAS:
        coef, _ = fit(xf, y_fit, lam)
        cal_loss = float(loss_fn(predict(xc, coef), y_cal).mean())
        if not math.isfinite(cal_loss):
            continue
        if best is None or cal_loss < best[0]:
            best = (cal_loss, lam)
    if best is None:
        raise RuntimeError("no finite calibration loss on the grid")
    coef, cond = fit(np.vstack([xf, xc]),
                     np.concatenate([y_fit, y_cal]), best[1])
    return predict(xs, coef), {
        "lambda": best[1], "cal_loss": round(best[0], 6),
        "cond": round(cond, 3),
        "coef_norm": round(float(np.linalg.norm(coef[:-1])), 6)}


def _logistic(x_fit, y_fit, x_cal, y_cal, x_sc):
    """Multinomial logistic; C by calibration log loss; refit on
    fit+cal. Returns (probs(n,3), record) or (None, degenerate)."""
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    if len(np.unique(y_fit)) < 2 or len(np.unique(y_cal)) < 1:
        return None, {"degenerate_fit_labels": True}
    xf, xc, xs = _standardize(x_fit, x_cal, x_sc)

    def probs3(clf, x):
        p = np.full((len(x), 3), 1e-12)
        raw = clf.predict_proba(x)
        for j, cls in enumerate(clf.classes_.astype(int)):
            p[:, cls] = raw[:, j]
        return p / p.sum(axis=1, keepdims=True)

    best = None
    for c in LOGISTIC_CS:
        clf = LogisticRegression(C=c, max_iter=2000)
        clf.fit(xf, y_fit)
        cal_loss = float(_logloss(probs3(clf, xc), y_cal).mean())
        if best is None or cal_loss < best[0]:
            best = (cal_loss, c)
    clf = LogisticRegression(C=best[1], max_iter=2000)
    clf.fit(np.vstack([xf, xc]), np.concatenate([y_fit, y_cal]))
    return probs3(clf, xs), {
        "C": best[1], "cal_loss": round(best[0], 6),
        "coef_norm": round(float(np.linalg.norm(clf.coef_)), 6)}


def execute_unit(identity: dict, root: Path,
                 log_path: Path) -> dict:
    import numpy as np
    run = RunDirectory(root / "census")
    geometry = run.ledger()["role_geometry"]
    roles = geometry["windows"][identity["origin"]]
    data = np.load(root / "inputs" / "census_inputs.npz",
                   allow_pickle=False)
    fam, hidx, mode = ALL_TREATMENTS[identity["treatment"]]
    fit_i = np.arange(*roles["fit"])
    cal_i = np.arange(*roles["cal"])
    sc_i = np.arange(*roles["score"])
    started = time.perf_counter()
    y_full = data[fam][:, hidx].copy()
    if mode == "shift":
        y_full = np.roll(y_full, SHIFT_ROWS)
    yf, yc, ys = y_full[fit_i], y_full[cal_i], y_full[sc_i]
    result: dict = {"family": fam, "mode": mode,
                    "horizon": HORIZON_OF[fam][hidx],
                    "score_rows": roles["score"],
                    "n_score": len(sc_i)}
    baselines: dict = {}
    if fam == "ret":
        primary = _se
        baselines["zero_return"] = _se(np.zeros(len(ys)), ys)
        baselines["fit_mean"] = _se(
            np.full(len(ys), float(yf.mean())), ys)
    elif fam == "vol":
        primary = _qlike
        trail = data["voltrail"][:, hidx, :]
        baselines["literal_trailing"] = _qlike(trail[sc_i, 0], ys)
        ar1_pred, ar1_rec = _ridge(
            trail[fit_i, 0:1], yf, trail[cal_i, 0:1], yc,
            trail[sc_i, 0:1], _qlike)
        baselines["calibrated_ar1"] = _qlike(ar1_pred, ys)
        result["calibrated_ar1_record"] = ar1_rec
    else:
        primary = None  # classification path below
        counts = np.bincount(yf.astype(int), minlength=3)
        prior = np.clip(counts / counts.sum(), 1e-12, None)
        prior = prior / prior.sum()
        probs = np.tile(prior, (len(ys), 1))
        baselines["fit_class_prior"] = _logloss(probs, ys)

    def regression_models():
        if fam == "ret":
            hist_x = data["rlags"]
        else:
            hist_x = data["voltrail"][:, hidx, :]
        summ_x = data["summary"]
        out = {}
        for name, x in (("target_history", hist_x),
                        ("causal_linear", summ_x)):
            pred, rec = _ridge(x[fit_i], yf, x[cal_i], yc,
                               x[sc_i], primary)
            out[name] = (primary(pred, ys), rec,
                         pred)
        return out

    def classification_models():
        out = {}
        for name, x in (("target_history", data["barscale"]),
                        ("causal_linear", data["summary"])):
            probs, rec = _logistic(x[fit_i], yf, x[cal_i], yc,
                                   x[sc_i])
            if probs is None:
                out[name] = (None, rec, None)
            else:
                out[name] = (_logloss(probs, ys), rec, probs)
        return out

    if mode == "leak":
        # future-leak sentinel: the TRUE target as the only feature
        if fam == "bar":
            onehot = np.eye(3)[y_full.astype(int)]
            probs, rec = _logistic(onehot[fit_i], yf,
                                   onehot[cal_i], yc, onehot[sc_i])
            model_losses = (_logloss(probs, ys)
                            if probs is not None else None)
        else:
            pred, rec = _ridge(
                y_full[fit_i].reshape(-1, 1), yf,
                y_full[cal_i].reshape(-1, 1), yc,
                y_full[sc_i].reshape(-1, 1), primary)
            model_losses = primary(pred, ys)
        selected, records = "leak_sentinel", {"leak_sentinel": rec}
        model_pred = None
    elif mode == "shift":
        # causally shifted target: model 3 only (predeclared)
        if fam == "bar":
            probs, rec = _logistic(
                data["summary"][fit_i], yf, data["summary"][cal_i],
                yc, data["summary"][sc_i])
            model_losses = (_logloss(probs, ys)
                            if probs is not None else None)
        else:
            pred, rec = _ridge(
                data["summary"][fit_i], yf, data["summary"][cal_i],
                yc, data["summary"][sc_i], primary)
            model_losses = primary(pred, ys)
        selected, records = "causal_linear", {"causal_linear": rec}
        model_pred = None
    else:
        models = (classification_models() if fam == "bar"
                  else regression_models())
        records = {k: v[1] for k, v in models.items()}
        finite = {k: v for k, v in models.items()
                  if v[0] is not None}
        if not finite:
            model_losses, selected, model_pred = None, None, None
        else:
            # predeclared model-choice rule: lower pooled
            # CALIBRATION primary loss
            selected = min(finite,
                           key=lambda k: finite[k][1]["cal_loss"])
            model_losses = finite[selected][0]
            model_pred = finite[selected][2]
    result["model_records"] = records
    result["selected_model"] = selected
    result["baseline_losses"] = {
        k: [round(float(v), 8) for v in arr]
        for k, arr in baselines.items()}
    result["model_losses"] = (
        None if model_losses is None
        else [round(float(v), 8) for v in model_losses])
    # licenses + secondaries
    if fam == "bar":
        result["class_support"] = {
            role: {str(c): int((y_full[idx] == c).sum())
                   for c in (0, 1, 2)}
            for role, idx in (("fit", fit_i), ("cal", cal_i),
                              ("score", sc_i))}
        if mode == "candidate" and model_pred is not None:
            pred_cls = model_pred.argmax(axis=1)
            onehot = np.eye(3)[ys.astype(int)]
            result["secondary"] = {
                "brier": round(float(
                    ((model_pred - onehot) ** 2).sum(axis=1)
                    .mean()), 6),
                "macro_recall": round(float(np.mean([
                    ((pred_cls == c) & (ys == c)).sum()
                    / max(1, (ys == c).sum())
                    for c in (0, 1, 2) if (ys == c).sum() > 0])), 6)}
    else:
        result["target_variance"] = {
            role: round(float(y_full[idx].var()), 8)
            for role, idx in (("fit", fit_i), ("cal", cal_i),
                              ("score", sc_i))}
        if mode == "candidate" and model_pred is not None:
            if fam == "ret":
                from scipy.stats import spearmanr
                hit = ((model_pred > 0) & (ys > 0)) | \
                      ((model_pred < 0) & (ys < 0))
                rho = spearmanr(model_pred, ys)
                result["secondary"] = {
                    "directional_accuracy": round(
                        float(hit.mean()), 6),
                    "spearman": round(float(
                        getattr(rho, "statistic", rho[0])), 6)}
            else:
                ss_res = float(((ys - model_pred) ** 2).sum())
                ss_tot = float(((ys - ys.mean()) ** 2).sum()) or 1.0
                result["secondary"] = {
                    "r2": round(1.0 - ss_res / ss_tot, 6)}
    result["effective_blocks"] = len(sc_i) // BLOCK
    result["wall_s"] = round(time.perf_counter() - started, 2)
    log_path.write_text(json.dumps(
        {"unit": identity,
         **{k: v for k, v in result.items()
            if k not in ("baseline_losses", "model_losses")}},
        default=float))
    return result


def worker_main(args) -> int:
    root = Path(args.run_root)
    run = RunDirectory(root / "census")
    ledger = run.ledger()
    identity = None
    for unit in ledger["units"]:
        if unit["unit_id"] == args.unit:
            identity = unit["identity"]
    if identity is None:
        raise RuntimePreflightError(f"unit {args.unit} not in ledger")
    expected = {"code": code_digest(),
                "config": sha_file(REPO / PREDECLARATION),
                "census_inputs": sha_file(
                    root / "inputs" / "census_inputs.npz")}
    outcome = run_one_unit(
        run, args.unit,
        lambda ident, log: execute_unit(ident, root, log),
        expected_digests=expected,
        timeout_s=float(args.timeout or 600))
    print(json.dumps({"unit": args.unit,
                      "state": outcome["state"]}))
    return 0 if outcome["state"] == "COMPLETED" else 1


# ------------------------------------------------------------------ #
# aggregate: recomputed from terminal unit records, never declared   #
# ------------------------------------------------------------------ #

def _block_bootstrap_p(diff_windows, rng_seed: int) -> float:
    """One-sided p for H1: mean diff > 0, circular block bootstrap
    WITHIN each score window (predeclared: block 6, B 2000)."""
    import numpy as np
    rng = np.random.default_rng(rng_seed)
    n_low = 0
    lengths = [len(d) for d in diff_windows]
    for _ in range(BOOT_B):
        parts = []
        for d, n in zip(diff_windows, lengths):
            n_blocks = math.ceil(n / BLOCK)
            starts = rng.integers(0, n, size=n_blocks)
            idx = (starts[:, None]
                   + np.arange(BLOCK)[None, :]).reshape(-1) % n
            parts.append(d[idx[:n]])
        if float(np.concatenate(parts).mean()) <= 0.0:
            n_low += 1
    return (1 + n_low) / (BOOT_B + 1)


def _candidate_assessment(key, fam, res_by_window, window_keys):
    """Skills, licenses and bootstrap p for one candidate (or a
    shifted control assessed under the same rule)."""
    import numpy as np
    assessment = {"family": fam, "windows": {}, "licensed": True,
                  "license_failures": []}
    base_names = FAMILY_BASELINES[fam]
    pooled_model, pooled_base = [], {b: [] for b in base_names}
    for wk in window_keys:
        r = res_by_window[wk]
        if r["model_losses"] is None:
            assessment["licensed"] = False
            assessment["license_failures"].append(
                f"{wk}: degenerate model (labels)")
            continue
        model = np.asarray(r["model_losses"])
        pooled_model.append(model)
        wrec = {"selected_model": r["selected_model"],
                "n": r["n_score"],
                "effective_blocks": r["effective_blocks"]}
        if r["effective_blocks"] < MIN_EFFECTIVE_BLOCKS:
            assessment["licensed"] = False
            assessment["license_failures"].append(
                f"{wk}: only {r['effective_blocks']} effective "
                "blocks")
        if fam == "bar":
            support = r["class_support"]["score"]
            wrec["class_support"] = support
            if min(int(support["0"]), int(support["1"])) < \
                    CLASS_SUPPORT_MIN:
                assessment["licensed"] = False
                assessment["license_failures"].append(
                    f"{wk}: class support {support} below "
                    f"{CLASS_SUPPORT_MIN}")
            fit_sup = r["class_support"]["fit"]
            cal_sup = r["class_support"]["cal"]
            if min(int(fit_sup["0"]), int(fit_sup["1"])) == 0 or \
                    min(int(cal_sup["0"]), int(cal_sup["1"])) == 0:
                assessment["licensed"] = False
                assessment["license_failures"].append(
                    f"{wk}: degenerate fit/cal labels")
        else:
            wrec["target_variance"] = r["target_variance"]
            if r["target_variance"]["score"] <= 0:
                assessment["licensed"] = False
                assessment["license_failures"].append(
                    f"{wk}: zero target variance")
        for b in base_names:
            base = np.asarray(r["baseline_losses"][b])
            pooled_base[b].append(base)
            wrec[f"skill_vs_{b}"] = round(
                1.0 - float(model.sum()) / float(base.sum()), 6)
        if "secondary" in r:
            wrec["secondary"] = r["secondary"]
        assessment["windows"][wk] = wrec
    if not assessment["licensed"] or len(pooled_model) < 4:
        assessment["licensed"] = False
        return assessment
    strongest = min(
        base_names,
        key=lambda b: float(np.concatenate(pooled_base[b]).sum()))
    assessment["strongest_baseline"] = strongest
    model_all = np.concatenate(pooled_model)
    base_all = np.concatenate(pooled_base[strongest])
    assessment["pooled_skill_vs_strongest"] = round(
        1.0 - float(model_all.sum()) / float(base_all.sum()), 6)
    diffs = [np.asarray(res_by_window[wk]["baseline_losses"]
                        [strongest])
             - np.asarray(res_by_window[wk]["model_losses"])
             for wk in window_keys]
    assessment["bootstrap_p_one_sided"] = round(
        _block_bootstrap_p(diffs, BOOT_SEED), 6)
    assessment["all_windows_positive_vs_every_baseline"] = all(
        assessment["windows"][wk][f"skill_vs_{b}"] > 0
        for wk in window_keys for b in base_names)
    return assessment


def aggregate_final(root: Path) -> dict:
    import numpy as np
    run = RunDirectory(root / "census")
    ledger = run.ledger()
    states = run.states()
    expected = [u["unit_id"] for u in ledger["units"]]
    by_unit = {u["unit_id"]: u["identity"] for u in ledger["units"]}
    problems = []
    for uid in expected:
        state = states.get(uid)
        if state is None:
            problems.append({"unit": uid, "why": "missing state"})
        elif state["state"] != "COMPLETED":
            problems.append({"unit": uid, "why": state["state"],
                             "treatment": by_unit[uid]["treatment"],
                             "window": by_unit[uid]["origin"]})
    trace = {"schema": "agent_multi.target_census_verdict.v1",
             "predeclaration": PREDECLARATION,
             "digests": ledger["digests"],
             "role_geometry": ledger["role_geometry"],
             "problems_preserved": problems}
    if problems:
        trace["verdict"] = "INCONCLUSIVE"
        trace["cause"] = ("failed/timed-out/missing units preserved "
                          "in the verdict — never dropped")
        return trace
    results = runtime_aggregate(run, expected)
    window_keys = sorted(ledger["role_geometry"]["windows"])
    by_key: dict = {}
    for uid, res in results.items():
        ident = by_unit[uid]
        by_key.setdefault(ident["treatment"], {})[
            ident["origin"]] = res

    # ---- negative controls first: they license the harness ----
    controls = {}
    control_failures = []
    for key, (fam, hidx, mode) in CONTROLS.items():
        res = by_key[key]
        if mode == "leak":
            assessment = _candidate_assessment(
                key, fam, res, window_keys)
            skills = [assessment["windows"][wk].get(
                f"skill_vs_{assessment.get('strongest_baseline', FAMILY_BASELINES[fam][0])}")
                for wk in window_keys] if assessment["licensed"] \
                else []
            ok = (assessment["licensed"]
                  and all(s is not None and s >= LEAK_SKILL_MIN
                          for s in skills))
            controls[key] = {"assessment": assessment,
                             "requirement": f"per-window skill >= "
                                            f"{LEAK_SKILL_MIN}",
                             "detects_leakage": ok}
            if not ok:
                control_failures.append(
                    f"{key}: harness did NOT flag leaked target as "
                    "unrealistically easy")
        else:
            assessment = _candidate_assessment(
                key, fam, res, window_keys)
            passes = (assessment["licensed"]
                      and assessment[
                          "all_windows_positive_vs_every_baseline"]
                      and assessment["pooled_skill_vs_strongest"]
                      >= MARGIN
                      and assessment["bootstrap_p_one_sided"]
                      < 0.05)
            controls[key] = {"assessment": assessment,
                             "requirement": "must NOT pass the "
                                            "advance rule "
                                            "(unadjusted p)",
                             "falsely_passes": passes}
            if passes:
                control_failures.append(
                    f"{key}: causally shifted target PASSED — "
                    "harness invalid")
    trace["negative_controls"] = controls

    # ---- nine candidates ----
    assessments = {}
    for key, (fam, hidx, mode) in CANDIDATES.items():
        assessments[key] = _candidate_assessment(
            key, fam, by_key[key], window_keys)
    licensed_p = {k: a["bootstrap_p_one_sided"]
                  for k, a in assessments.items() if a["licensed"]}
    holm = holm_adjust({k: min(1.0, max(0.0, p))
                        for k, p in licensed_p.items()})
    passers, inconclusive_candidates = [], []
    for key, a in assessments.items():
        if not a["licensed"]:
            a["outcome"] = "INCONCLUSIVE_CANDIDATE"
            inconclusive_candidates.append(key)
            continue
        a["holm_p"] = holm[key]
        passes = (a["all_windows_positive_vs_every_baseline"]
                  and a["pooled_skill_vs_strongest"] >= MARGIN
                  and holm[key] < 0.05)
        a["outcome"] = "PASSES" if passes else "FAILS"
        if passes:
            passers.append(key)
    trace["candidates"] = assessments
    trace["holm_pvalues"] = {k: round(v, 6)
                             for k, v in holm.items()}
    trace["inconclusive_candidates"] = inconclusive_candidates

    # ---- verdict ----
    if control_failures:
        trace["verdict"] = "INCONCLUSIVE"
        trace["cause"] = control_failures
    elif inconclusive_candidates and not passers:
        # unlicensed candidates block a clean negative only if no
        # candidate passed; preserved either way
        trace["verdict"] = ("INCONCLUSIVE"
                            if len(inconclusive_candidates) == 9
                            else "NO_TARGET_CANDIDATE_DEMONSTRATED")
    elif passers:
        trace["verdict"] = "TARGET_CANDIDATE_FOUND"
    else:
        trace["verdict"] = "NO_TARGET_CANDIDATE_DEMONSTRATED"
    if passers:
        # predeclared selection: minimum per-window skill vs the
        # strongest baseline (stability), then Holm p, then horizon
        def stability(k):
            a = assessments[k]
            b = a["strongest_baseline"]
            return min(a["windows"][wk][f"skill_vs_{b}"]
                       for wk in window_keys)
        ranked = sorted(
            passers,
            key=lambda k: (-stability(k), holm[k],
                           HORIZON_OF[assessments[k]["family"]][
                               CANDIDATES[k][1]]))
        trace["selection"] = {
            "ranked_passers": ranked,
            "selected": ranked[:2],
            "rule": "minimum per-window skill vs strongest baseline "
                    "(stability), ties by Holm p then horizon; "
                    "neural confirmation NOT begun automatically"}
    trace["decision_constants"] = {
        "margin": MARGIN, "block": BLOCK, "boot_b": BOOT_B,
        "boot_seed": BOOT_SEED, "leak_skill_min": LEAK_SKILL_MIN,
        "shift_rows": SHIFT_ROWS,
        "min_effective_blocks": MIN_EFFECTIVE_BLOCKS,
        "class_support_min": CLASS_SUPPORT_MIN}
    return trace


# ------------------------------------------------------------------ #
# bounded supervisor (CPU-only)                                      #
# ------------------------------------------------------------------ #

def supervise(args) -> int:
    root = Path(args.run_root)
    summary = materialize(root, Path(args.pretrain_dir),
                          Path(args.n1_inputs),
                          max_windows=args.max_windows,
                          stride=args.stride)
    print(json.dumps({"materialized": summary.get("units")}),
          flush=True)
    if summary.get("verdict") == "INCONCLUSIVE":
        print(json.dumps(summary["geometry"]), flush=True)
        return 0
    run = RunDirectory(root / "census")
    ledger = run.ledger()
    total = len(ledger["units"])
    stop_file = root / "STOP_CENSUS"
    children: dict = {}
    started = time.time()
    last_beat = 0.0
    while True:
        states = run.states()
        completed = sum(1 for s in states.values()
                        if s["state"] == "COMPLETED")
        pend = [uid for uid, st in states.items()
                if st["state"] in ("PENDING", "INTERRUPTED",
                                   "TIMED_OUT")]
        running = [u for u, s in states.items()
                   if s["state"] == "RUNNING"]
        if stop_file.exists():
            for proc in children.values():
                if proc.poll() is None:
                    proc.terminate()
                    try:
                        proc.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait()
            print("[stop-file] workers terminated and reaped; "
                  "durable states preserved", flush=True)
            break
        if not pend and not running:
            break
        children = {pid: pr for pid, pr in children.items()
                    if pr.poll() is None}
        while pend and len(children) < args.workers:
            uid = pend.pop(0)
            env = dict(os.environ)
            env["CUDA_VISIBLE_DEVICES"] = ""  # CPU-only census
            cmd = [sys.executable,
                   str(REPO / "tools/target_horizon_census_n2.py"),
                   "worker", "--run-root", str(root),
                   "--unit", uid, "--timeout", "600"]
            log = open(run.root / "logs" / f"worker_{uid}.out",
                       "ab")
            proc = subprocess.Popen(cmd, stdout=log,
                                    stderr=subprocess.STDOUT,
                                    env=env)
            children[proc.pid] = proc
            states = run.states()
        if time.time() - last_beat >= 60.0:
            def kill_child(pid):
                proc = children.get(pid)
                if proc is None:
                    return not Path(f"/proc/{pid}").exists()
                if proc.poll() is None:
                    proc.terminate()
                    try:
                        proc.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait(timeout=30)
                return proc.poll() is not None
            elapsed = time.time() - started
            run.heartbeat(
                current_unit=(running[0] if running else None),
                workers=args.workers, device_class="cpu",
                extra={"phase": "census",
                       "completed": completed, "total": total,
                       "throughput_per_min": round(
                           completed / max(elapsed / 60, 1e-9), 2),
                       "elapsed_s": round(elapsed, 1)})
            for alert in run.watchdog(kill_child=kill_child):
                print(f"[watchdog] {json.dumps(alert)}", flush=True)
            last_beat = time.time()
        if time.time() - started > 7200:
            print("[ceiling] two-hour campaign wall — stopping",
                  flush=True)
            for proc in children.values():
                if proc.poll() is None:
                    proc.terminate()
                    proc.wait(timeout=30)
            break
        time.sleep(2.0)
    for proc in children.values():
        if proc.poll() is None:
            proc.wait(timeout=120)
    trace = aggregate_final(root)
    atomic_write_json(root / "CENSUS_VERDICT_TRACE.json", trace)
    print(json.dumps({"verdict": trace["verdict"]}, indent=1),
          flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("materialize", "supervise"):
        sp = sub.add_parser(name)
        sp.add_argument("--run-root", required=True)
        sp.add_argument("--pretrain-dir", required=True)
        sp.add_argument("--n1-inputs", required=True)
        sp.add_argument("--max-windows", type=int, default=2200)
        sp.add_argument("--stride", type=int, default=4)
        if name == "supervise":
            sp.add_argument("--workers", type=int, default=4)
    w = sub.add_parser("worker")
    w.add_argument("--run-root", required=True)
    w.add_argument("--unit", required=True)
    w.add_argument("--timeout", type=float, default=None)
    a = sub.add_parser("aggregate")
    a.add_argument("--run-root", required=True)
    args = parser.parse_args()
    if args.cmd == "materialize":
        print(json.dumps(materialize(
            Path(args.run_root), Path(args.pretrain_dir),
            Path(args.n1_inputs), max_windows=args.max_windows,
            stride=args.stride), indent=1, default=str))
        return 0
    if args.cmd == "worker":
        return worker_main(args)
    if args.cmd == "supervise":
        return supervise(args)
    trace = aggregate_final(Path(args.run_root))
    print(json.dumps(trace, indent=1, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
